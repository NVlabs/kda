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

"""cute_pkdp: pair-chunk m128 KDA chain kernel host wrapper (v108).

Same launch contract as main_pkdw, but over the pkdp kernel which
processes two 32-token chunks per compute cycle.

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
"""
import os

import cuda.bindings.driver as cuda_driver
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from . import pkdp as _k

D = 128
C32 = 32
LOG2E = 1.4426950408889634
# dev-only: per-slot chain timestamps (nanoseconds) into a probe tensor
_TPROBE = 1 if os.environ.get("PKDA_TPROBE", "0") == "1" else 0
_NOFINAL = 1 if os.environ.get("PKDA_NOFINAL", "1") == "1" else 0
_LITE = 1 if os.environ.get("PKDP_LITE", "0") == "1" else 0

_CACHE: dict = {}


def _get_compiled(H: int, dev: torch.device, gate2: int = 0,
                  mid_bf16: bool = False):
    key = (H, gate2, mid_bf16, _NOFINAL)
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
        G = 2 * H
        soff = torch.arange(G + 1, dtype=torch.int32, device=dev)
        schain = torch.arange(G, dtype=torch.int32, device=dev)
        spt0 = torch.zeros(G, dtype=torch.int32, device=dev)
        sptn = torch.full((G,), T // 2, dtype=torch.int32, device=dev)
        ssrc = torch.full((G,), -1, dtype=torch.int32, device=dev)
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
            FINAL_=1 - _NOFINAL, LITE_=_LITE,
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
        sched=None, export=None, gate2=0):
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

    tfn = _get_compiled(H, dev, gate2, mid.dtype == torch.bfloat16)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    tfn(qf, kf, vf, gf, bf_, A_log, dt_bias.view(H, D),
        st.view(nseqs * H, D, D), final_state.view(nseqs * H, D, D),
        of, cu_t, soff, schain, spt0, sptn, ssrc, sdst, mid, mfl,
        exp_ws, expt, _tprobe_buf(G, dev), have_state, G,
        do_export, export_seq, nc2, fepoch,
        float(scale), float(lower_bound) * LOG2E, stream)
    return out, final_state
