# Copyright 2026 KDA(Kernel Design Agents) Team
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""cute_pkdw: the m128 KDA chain kernel — persistent multi-chain CTAs.

One kernel per call: grid (G,) with G = min(nseq*H, SMs) — each
1024-thread CTA owns a host-computed LPT list of (seq, head) chains and
processes them back-to-back.  The prep instances, stage ring, and all
barrier parities run CONTINUOUSLY across chain boundaries, so the
5-stage prep pipeline for chain k+1's first chunks fills while chain
k's tail still computes (the ~10us per-chain fill/drain of the
one-CTA-per-chain scheme collapses to a ~1us state export+seed).
PR#4262-derived warp specialization with the v71 self-issue structure
(no MMA warp) and the v68 PTX-free StMatrix epilogue.  See pkdw.py.

The task adapter does not return final state, so the default specialization
compiles terminal-chain state exports out.  PKDA_NOFINAL=0 retains the
general-purpose final-state path for profiling.

V154 distinguishes split-aware zero-seeding from actual midstate handoff.
The no-repair fixed route compiles with SPLIT=1/HANDOFF=0, deleting all
producer polling, state export, repair RMW, and completion-release code.
"""
import os

import cuda.bindings.driver as cuda_driver
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from . import pkdw as _k

D = 128
C32 = 32
LOG2E = 1.4426950408889634
# dev-only: per-slot chain timestamps (nanoseconds) into a probe tensor
_TPROBE = 1 if os.environ.get("PKDA_TPROBE", "0") == "1" else 0
_NOFINAL = 1 if os.environ.get("PKDA_NOFINAL", "1") == "1" else 0

_CACHE: dict = {}


def _get_compiled(H: int, dev: torch.device, gate2: int = 0,
                  mid_bf16: bool = False, has_split: bool = False,
                  has_handoff: bool = False,
                  do_export: bool = False, have_state: bool = True,
                  beta_tma: bool = False, beta_prefetch: bool = False,
                  bf16_dv: bool = False,
                  beta_bf16: bool = False, qk_rowpair: bool = False,
                  rcp_decor: bool = True, rf_hoist: bool = True,
                  restore_tail: bool = False,
                  gcs_fp16: bool = False, gcs_packcvt: bool = False,
                  gram_w9: int = 0, reg_mode: int = 0,
                  dual: bool = False, joint_norm: bool = False,
                  norm_mode: int = 0, early_v: int = 0,
                  early_state_load: int = 0, seed_pf: int = 0,
                  seed_drop: int = 0, seed_drop4: int = 0,
                  fold: int = 0, norm_damp: int = 0, bulk_pf: int = 0):
    key = (H, gate2, mid_bf16, _NOFINAL,
           has_split, has_handoff, do_export, have_state,
           beta_tma, beta_prefetch, bf16_dv, beta_bf16,
           qk_rowpair, rcp_decor, rf_hoist, restore_tail,
           gcs_fp16, gcs_packcvt,
           gram_w9, reg_mode, dual, joint_norm, norm_mode, early_v,
           early_state_load, seed_pf, seed_drop, seed_drop4, fold, norm_damp,
           bulk_pf)
    entry = _CACHE.get(key)
    if entry is None:
        N = 2
        T = 8 * C32
        bf = torch.bfloat16
        mk = lambda t, a=16: from_dlpack(t, assumed_align=a).mark_compact_shape_dynamic(mode=0)
        q = torch.empty(T, H, D, dtype=bf, device=dev)
        k = torch.empty(T, H, D, dtype=bf, device=dev)
        v = torch.empty(T, H, D, dtype=bf, device=dev)
        g = torch.empty(T, H, D, dtype=bf, device=dev)
        beta = torch.empty(T, H, dtype=bf, device=dev)
        alog = torch.empty(H, dtype=torch.float32, device=dev)
        dtb = torch.empty(H, D, dtype=torch.float32, device=dev)
        st = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        ns = torch.empty(N * H, D, D, dtype=torch.float32, device=dev)
        cu_t = torch.zeros(N + 1, dtype=torch.int32, device=dev)
        cu_t[1] = T // 2
        cu_t[2] = T
        chains = 2 * H
        if dual:
            # One chain on each side of every dummy CTA.  Dual soff contains
            # two consecutive ranges per CTA, hence 2*G+1 entries.
            G = H
            soff = torch.arange(2 * G + 1, dtype=torch.int32, device=dev)
        else:
            G = chains
            soff = torch.arange(G + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(chains, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(chains, dtype=torch.int32, device=dev)
        sptn = torch.full((chains,), T // 2, dtype=torch.int32, device=dev)
        ssrc = torch.full((chains,), -1, dtype=torch.int32, device=dev)
        sdst = torch.full((G,), -1, dtype=torch.int32, device=dev)
        mid = torch.empty(
            2, D, D,
            dtype=torch.bfloat16 if mid_bf16 else torch.float32,
            device=dev)
        mfl = torch.zeros(2, dtype=torch.int32, device=dev)
        tpr = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        out = torch.empty(T, H, D, dtype=bf, device=dev)
        nc2 = (T // 2 + C32 - 1) // C32
        exp_ws = torch.empty(H * nc2 * 64, 1, 256, dtype=bf, device=dev)
        expt = torch.empty(H * nc2, 160, dtype=torch.float32, device=dev)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        tfn = cute.compile(
            _k._launch_pkd,
            mk(q), mk(k), mk(v), mk(g), mk(beta, 4),
            from_dlpack(alog, assumed_align=16),
            from_dlpack(dtb, assumed_align=16),
            mk(st), mk(ns), mk(out), mk(cu_t, 4),
            mk(soff, 4), mk(schain, 4),
            mk(spt0, 4), mk(sptn, 4), mk(ssrc, 4), mk(sdst, 4),
            mk(mid), mk(mfl, 4),
            mk(exp_ws), mk(expt), mk(tpr, 8),
            cutlass.Int32(1), cutlass.Int32(G),
            cutlass.Int32(0), cutlass.Int32(0), cutlass.Int32(nc2),
            cutlass.Int32(1),
            cutlass.Float32(1.0), cutlass.Float32(-1.0),
            H_=H, stream=stream, TPROBE_=_TPROBE, GATE2_=gate2,
            FINAL_=1 - _NOFINAL,
            SPLIT_=1 if has_split else 0,
            HANDOFF_=1 if has_handoff else 0,
            EXPORT_=1 if do_export else 0,
            STATE_=1 if have_state else 0,
            BETA_TMA_=1 if beta_tma else 0,
            BETA_PREFETCH_=1 if beta_prefetch else 0,
            BF16_DV_=1 if bf16_dv else 0,
            BETA_BF16_=1 if beta_bf16 else 0,
            QK_ROWPAIR_=int(qk_rowpair),
            RCP_DECOR_=1 if rcp_decor else 0,
            RF_HOIST_=1 if rf_hoist else 0,
            RESTORE_TAIL_=1 if restore_tail else 0,
            GCS_FP16_=1 if gcs_fp16 else 0,
            GCS_PACKCVT_=1 if gcs_packcvt else 0,
            GRAM_W9_=int(gram_w9),
            REG_MODE_=int(reg_mode),
            DUAL_=1 if dual else 0,
            JOINT_NORM_=int(joint_norm),
            NORM_MODE_=int(norm_mode),
            EARLY_V_=int(early_v),
            EARLY_STATE_LOAD_=int(early_state_load),
            SEED_PF_=int(seed_pf),
            SEED_DROP_=int(seed_drop),
            SEED_DROP4_=int(seed_drop4),
            FOLD_=int(fold),
            NORM_DAMP_=int(norm_damp),
            BULK_PF_=int(bulk_pf),
            options="--enable-tvm-ffi --opt-level 3")
        tfn(q, k, v, g, beta, alog, dtb, st, ns, out, cu_t, soff, schain,
            spt0, sptn, ssrc, sdst, mid, mfl,
            exp_ws, expt, tpr, 1, G, 0, 0, nc2, 1, 1.0, -1.0, stream)
        torch.cuda.synchronize()
        _CACHE[key] = tfn
        entry = tfn
    return entry


_DUMMY_EXP: dict = {}


def _dummy_exp(dev: torch.device):
    key = dev.index
    d = _DUMMY_EXP.get(key)
    if d is None:
        d = (torch.empty(64, 1, 256, dtype=torch.bfloat16, device=dev),
             torch.empty(1, 160, dtype=torch.float32, device=dev))
        _DUMMY_EXP[key] = d
    return d


_DUMMY_MID: dict = {}


def _dummy_mid(dev: torch.device):
    key = dev.index
    d = _DUMMY_MID.get(key)
    if d is None:
        d = (torch.empty(1, D, D, dtype=torch.float32, device=dev),
             torch.zeros(1, dtype=torch.int32, device=dev))
        _DUMMY_MID[key] = d
    return d


_DUMMY_TPR: dict = {}
LAST_TPROBE: list = [None]


def _tprobe_buf(G: int, dev: torch.device):
    if _TPROBE:
        t = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        LAST_TPROBE[0] = t
        return t
    key = dev.index
    t = _DUMMY_TPR.get(key)
    if t is None:
        t = torch.zeros(64, dtype=torch.int64, device=dev)
        _DUMMY_TPR[key] = t
    return t


@torch.no_grad()
def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None,
        sched=None, export=None, gate2=0, has_split=False, has_handoff=None,
        beta_tma=False, beta_prefetch=False, bf16_dv=True, beta_bf16=True,
        qk_rowpair=False,
        rcp_decor=True, rf_hoist=True, restore_tail=False, gcs_fp16=False,
        gcs_packcvt=False, gram_w9=0, reg_mode=0, dual=False, seed_pf=0,
        joint_norm=False, norm_mode=0, early_v=0, early_state_load=0,
        seed_drop=0, seed_drop4=0, fold=0, norm_damp=0, bulk_pf=0):
    dev = q.device
    B, T, H, K = q.shape
    assert K == D
    qf = q.view(T * B, H, D)
    kf = k.view(T * B, H, D)
    vf = v.view(T * B, H, D)
    gf = g.view(T * B, H, D)
    bf_ = beta.view(T * B, H)
    of = out.view(T * B, H, D)

    if cu_seqlens is None:
        cu_t = torch.tensor([0, B * T], dtype=torch.int32, device=dev)
        nseqs = 1
    else:
        cu_t = cu_seqlens.to(torch.int32)
        nseqs = int(cu_seqlens.numel()) - 1

    if final_state is None:
        final_state = torch.empty(nseqs, H, D, D, dtype=torch.float32, device=dev)
    have_state = 1 if initial_state is not None else 0
    st = initial_state if initial_state is not None else final_state
    if st.dtype != torch.float32:
        st = st.float()

    if sched is None:
        chains = nseqs * H
        soff = torch.arange(chains + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(chains, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(chains, dtype=torch.int32, device=dev)
        sptn = cu_t.diff().to(torch.int32).repeat_interleave(H)
        ssrc = torch.full((chains,), -1, dtype=torch.int32, device=dev)
        sdst = ssrc
        mid, mfl = _dummy_mid(dev)
        fepoch = 1
        G = chains
    else:
        (soff, schain, spt0, sptn, ssrc, sdst, G,
         mid, mfl, fepoch) = sched
    if export is None:
        exp_ws, expt = _dummy_exp(dev)
        do_export, export_seq, nc2 = 0, 0, 1
    else:
        exp_ws, expt, export_seq, nc2 = export
        do_export = 1

    if has_handoff is None:
        has_handoff = has_split
    tfn = _get_compiled(
        H, dev, gate2, mid.dtype == torch.bfloat16,
        has_split=has_split, has_handoff=has_handoff,
        do_export=bool(do_export),
        have_state=bool(have_state), beta_tma=beta_tma,
        beta_prefetch=beta_prefetch,
        bf16_dv=bf16_dv, beta_bf16=beta_bf16, qk_rowpair=qk_rowpair,
        rcp_decor=rcp_decor, rf_hoist=rf_hoist,
        restore_tail=restore_tail, gcs_fp16=gcs_fp16,
        gcs_packcvt=gcs_packcvt, gram_w9=gram_w9, reg_mode=reg_mode,
        dual=dual, joint_norm=joint_norm, norm_mode=norm_mode,
        early_v=early_v, early_state_load=early_state_load, seed_pf=seed_pf,
        seed_drop=seed_drop, seed_drop4=seed_drop4,
        fold=fold, norm_damp=norm_damp, bulk_pf=bulk_pf)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    tfn(qf, kf, vf, gf, bf_, A_log, dt_bias.view(H, D),
        st.view(nseqs * H, D, D), final_state.view(nseqs * H, D, D),
        of, cu_t, soff, schain, spt0, sptn, ssrc, sdst, mid, mfl,
        exp_ws, expt, _tprobe_buf(G, dev), have_state, G,
        do_export, export_seq, nc2, fepoch,
        float(scale), float(lower_bound) * LOG2E, stream)
    return out, final_state
