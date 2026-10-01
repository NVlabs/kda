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

"""cute_pkdx: the m64 KDA chain kernel — one CTA per (seq, head, V64-half).

One kernel per call: grid (2, H, nseqs) — a 1024-thread CTA per
(seq, head, V64-half) chain.  PR#4262-derived warp specialization with
the v71 self-issue structure (no MMA warp) and the v68 PTX-free
StMatrix epilogue.  See impl/pkdx.py.
"""
import cuda.bindings.driver as cuda_driver
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from . import pkdx as _k

D = 128
C32 = 32
LOG2E = 1.4426950408889634

_CACHE: dict = {}


def _get_compiled(H: int, dev: torch.device):
    entry = _CACHE.get(H)
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
        perm = torch.arange(N, dtype=torch.int32, device=dev)
        out = torch.empty(T, H, D, dtype=bf, device=dev)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        tfn = cute.compile(
            _k._launch_pkd,
            mk(q), mk(k), mk(v), mk(g), mk(beta, 4),
            from_dlpack(alog, assumed_align=16),
            from_dlpack(dtb, assumed_align=16),
            mk(st), mk(ns), mk(out), mk(cu_t, 4), mk(perm, 4),
            cutlass.Int32(1), cutlass.Int32(N),
            cutlass.Int32(0), cutlass.Int32(H),
            cutlass.Float32(1.0), cutlass.Float32(-1.0),
            H_=H, stream=stream,
            options="--enable-tvm-ffi --opt-level 3")
        tfn(q, k, v, g, beta, alog, dtb, st, ns, out, cu_t, perm,
            1, N, 0, H, 1.0, -1.0, stream)
        torch.cuda.synchronize()
        _CACHE[H] = tfn
        entry = tfn
    return entry


_ID_PERM: dict = {}


def _identity_perm(nseqs: int, dev: torch.device) -> torch.Tensor:
    key = (nseqs, dev.index)
    p = _ID_PERM.get(key)
    if p is None:
        p = torch.arange(nseqs, dtype=torch.int32, device=dev)
        _ID_PERM[key] = p
    return p


@torch.no_grad()
def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=None, final_state=None, cu_seqlens=None,
        head_range=None, seq_perm=None):
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

    h0, h1 = head_range if head_range is not None else (0, H)
    perm = seq_perm if seq_perm is not None else _identity_perm(nseqs, dev)

    tfn = _get_compiled(H, dev)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    tfn(qf, kf, vf, gf, bf_, A_log, dt_bias.view(H, D),
        st.view(nseqs * H, D, D), final_state.view(nseqs * H, D, D),
        of, cu_t, perm, have_state, nseqs, h0, h1 - h0,
        float(scale), float(lower_bound) * LOG2E, stream)
    return out, final_state
