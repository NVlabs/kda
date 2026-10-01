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

# SPDX-FileCopyrightText: Copyright TIRx authors
# Modified here: derive the offline builder's default CTA count from hardware.

"""Native TIRx SM100a KDA backward for the packed Kimi K3 workload portfolio.

Both paths use the detected SM count. The fused path selects its schedule
tuned for 152 CTAs only for that CTA count and 768 chains.

The supported contract is the prepared-tensor contract of
``fla.ops.kda.chunk_bwd.chunk_kda_bwd`` with B=1, K=V=128, chunk_size=64,
``Hv % Hqk == 0`` (grouped value heads) and arbitrary packed sequence lengths
with masked partial trailing chunks. It covers all seventeen official
KDA-backward workloads: ``kda-bwd-packed-1024x8-h96`` (8192 tokens as eight
1024-token sequences, Hqk=Hv=96) and the sixteen packed portfolios that cross
T in {18432, 32768} and four sequence layouts with (Hqk, Hv) in
{(2,4), (2,8), (4,4), (4,8)}.

Inputs are the saved L2-normalized q/k, v, the activated update gate beta, the
saved Aqk/Akk interaction matrices, the chunk-local base-2 cumulative gate g,
the per-sequence K-first initial state, the upstream do and dht, and
cu_seqlens. The kernel writes dq, dk, dv, db, dg and dh0; dA and dbias are
absent from this contract.

The selected kernel is the ``fused-zfold`` global-best frontier member of the
2026-09-10 KDA-backward optimization run ``kda-bwd-portfolio-carry-3``. It is a
shape-aware dispatcher over two persistent tcgen05 designs:

* Fused chain kernel, used when ``Hqk == Hv``, ``Hv % 8 == 0`` and every
  sequence is a whole number of chunks. One 12-warp CTA per SM runs pass 1
  (forward chunk-state recompute into TMEM, bf16 snapshots and a bf16 2^g
  cache) for a host-scheduled chain set, publishes per-chain readiness with an
  epoch-stamped gpu-scope release flag, then runs pass 2 (backward) for its own
  chains. Pass 2 folds the W/U/Vn/dw chain into a single TMEM intermediate
  ``Z = vb - h^T kbg``. Pass 1 builds a channel-major TMEM tile with four
  tcgen05 identity MMAs per raw-K stage; the M=64 accumulators are read out
  through 16-lane half-lane readouts.
* Persistent megakernel, used for every other layout (grouped value heads,
  partial trailing chunks). Each (sequence, v-head) chain runs a forward and a
  backward recurrence stream that publish per-chunk progress flags, and every
  (chunk, qk-head) work item is then independent: it loops over the grouped
  v-heads, reduces dq/dk in TMEM, and produces dv, db and dg in one pass. Work
  units are dealt from a host-built readiness-ordered table.

Numerical notes. Gates are finite, nonpositive chunk-local cumulative base-2
log decays, monotone within each chunk. MMAs use bf16 operands and fp32
accumulation. Mild channels retain the bf16 ``2^g`` cache and reciprocal fast
path. When the cached chunk-end decay is below 1/16, the state snapshot keeps
its unscaled gradient and its consumers use bounded ``exp2(g_end - g_i)``.
A GPU guard checks amplitudes on chains with strong gates on every invocation.
For sequences of at most 4096 tokens with |v|, |do| <= 4 and |h0|, |dht| <= 1,
intra-chunk derivatives can retain tensor contractions through chunk-end logs
of -160. A common channel shift keeps each gate factor and reciprocal within
[2^-80, 2^80]; FP32 gates and bf16 high/low products limit operand rounding.
The mixed state/intra accumulators compensate that shift before their intra
additions. State recurrence and snapshots keep the bounded representation above.
Channels outside that range use adjacent bounded FP32 decays. When every
chain is mild, a separately compiled native kernel avoids paying for the
extended path's register pressure. In the megakernel, after the last MMA
reader, its helper warp scans the rounded BF16 dAqk diagonal. Items with an
absolute diagonal of at least 8 are appended to a device queue; an item-only
extended specialization then reuses the native recurrence snapshots and
overwrites only dq, dk, and dg for those items. Native dv and db remain because
the diagonal reformulation does not target them. Strong-gate inputs continue
to use the complete extended kernel. On both extended paths, query/key diagonal
contributions are omitted from dg because they cancel analytically; they are
retained in dq/dk. The fused native path instead backs up the same rounded diagonal on chip
and removes its query/key contributions inline.
Full and partial chunks use the same rounded-cache predicate for snapshot
production and consumption. Tests cover mixed channels,
strong and extreme decay, abrupt resets, and partial tails under the original
per-output tolerances.
"""

import ctypes
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from itertools import pairwise
from typing import Any
from unittest import SkipTest

import torch

import tirx_kernels.tirx_lite as txl

D = 128
CHUNK = 64
MMA_SS = "tcgen05.mma.cta_group::1.kind::f16"
TMA_LD = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1"
TMA_ST = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
TMA_PREFETCH = "cp.async.bulk.prefetch.tensor.3d.L2.global.tile"
TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_LD8 = "tcgen05.ld.sync.aligned.32x32b.x8.b32"
TC_LD4 = "tcgen05.ld.sync.aligned.32x32b.x4.b32"
TC_LD_HALF32 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TC_ST8 = "tcgen05.st.sync.aligned.32x32b.x8.b32"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
TC_FENCE_BEFORE = "tcgen05.fence::before_thread_sync"
TC_FENCE_AFTER = "tcgen05.fence::after_thread_sync"


_CUDA_MBAR_WAIT = txl.MBarrier._wait


def _ptx_mbarrier_wait(self, stage, phase):
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    barrier = txl.cuda.cvta_generic_to_shared(self.buf.ptr_to([stage]))
    target_phase = txl.cast(phase ^ self.phase_offset, "uint32")
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, barrier, target_phase, txl.uint32(10_000_000)
        )


UNITS_PER_STAGE = 512
SBO_UNITS = 64


def idesc(M, N, *, ta=0, tb=0, na=0, nb=0):
    """Dense tcgen05 instruction descriptor: bf16 x bf16 -> f32."""
    return (
        (1 << 4)
        | (1 << 7)
        | (1 << 10)
        | (na << 13)
        | (nb << 14)
        | (ta << 15)
        | (tb << 16)
        | ((N >> 3) << 17)
        | ((M >> 4) << 24)
    )


class Op:
    """A tcgen05 matrix-descriptor operand at trace-time stage `base` of the staged pool.

    The pool's base descriptor is encoded once (stage 0, 64-row column-atom stride); every
    operand is that descriptor plus a compile-time immediate (stage offset in 16 B units and,
    for 128-row tiles, the larger LBO field).
    """

    LBO_BASE = 64 * 8

    def __init__(self, base_desc, base, rows, kdim, major):
        self.rows = rows
        self.major = major
        self.n_k = kdim // 16
        ldo = rows * 8
        assert ldo >= self.LBO_BASE
        self._bd = base_desc
        self._imm = base * UNITS_PER_STAGE + ((ldo - self.LBO_BASE) << 16)

    def off(self, kp):
        if self.major == "k":
            return (kp % 4) * 2 + (kp // 4) * self.rows * 8
        return kp * 128

    def desc(self, kp, units=None):
        d = self._bd[0] + txl.uint64(self._imm + self.off(kp))
        if units is not None:
            d = d + units
        return d


def mma_chain(tm, dcol, a, b, idesc_val, accumulate, a_units=None, b_units=None):
    """One k-chain of tcgen05.mma from the calling (single, elected) thread."""
    n_k = b.n_k
    assert a.n_k == n_k, (a.n_k, n_k)
    for kp in range(n_k):
        txl.ptx[MMA_SS](
            txl.Cast("uint32", tm[0] + dcol),
            a.desc(kp, a_units),
            b.desc(kp, b_units),
            txl.uint32(idesc_val),
            txl.uint32(0),
            txl.uint32(0),
            txl.uint32(0),
            txl.uint32(0),
            txl.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )


def mma_chain_ta(tm, dcol, a_col, b, idesc_val, accumulate, b_units=None):
    """One k-chain of tcgen05.mma whose A operand is a K-major bf16 tile in Tensor Memory.

    A[M=128 lanes x K] is packed two bf16 per 32-bit column starting at column `a_col`, so each
    16-element k-step advances the A address by 8 columns (the FlashAttention-4 P-in-TMEM form).
    """
    for kp in range(b.n_k):
        txl.ptx[MMA_SS](
            txl.Cast("uint32", tm[0] + dcol),
            txl.Cast("uint32", tm[0] + a_col + 8 * kp),
            b.desc(kp, b_units),
            txl.uint32(idesc_val),
            txl.uint32(0),
            txl.uint32(0),
            txl.uint32(0),
            txl.uint32(0),
            txl.ptx.pred(1 if (accumulate or kp > 0) else 0),
        )


class AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def encode_tensor_map(tensor, dtype: str, dims, strides_bytes, box, swizzle=3, l2promo=2):
    """cuTensorMapEncodeTiled through TVM's runtime function (dims innermost first)."""
    import tvm

    desc = AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    rank = len(dims)
    assert len(strides_bytes) == rank - 1 and len(box) == rank
    encode(
        desc.ptr,
        dtype,
        rank,
        ctypes.c_void_p(int(tensor.data_ptr())),
        *[int(d) for d in dims],
        *[int(s) for s in strides_bytes],
        *[int(b) for b in box],
        *([1] * rank),
        0,
        swizzle,
        l2promo,
        0,
    )
    return desc


def token_map(tensor, T, H, inner, box_inner, box_rows=CHUNK, swizzle=3):
    """[T, H, inner] tensor viewed as dims (inner, T, H): coordinates (d0, token, head)."""
    esz = tensor.element_size()
    dtype = {2: "bfloat16", 4: "float32"}[esz]
    return encode_tensor_map(
        tensor,
        dtype,
        (inner, T, H),
        (esz * inner * H, esz * inner),
        (box_inner, box_rows, 1),
        swizzle=swizzle,
    )


def state_map(tensor, n_states):
    """[n_states, 128, 128] bf16 states viewed as dims (128 v, 128 k, n): coordinates (v0, 0, idx)."""
    return encode_tensor_map(tensor, "bfloat16", (D, D, n_states), (2 * D, 2 * D * D), (64, D, 1))


KV_BYTES = 2 * CHUNK * D * 2
A_BYTES = CHUNK * CHUNK * 2
G_BYTES = CHUNK * D * 4
IN3_BYTES = 3 * CHUNK * D * 2 + G_BYTES


def _load_bf16_f32(ptr, *, shared=False):
    word = txl.local_scalar("uint16")
    if shared:
        txl.ptx.ld.shared.b16(word, ptr)
    else:
        txl.ptx.ld.global_.nc.b16(word, ptr)
    return txl.reinterpret("float32", txl.Cast("uint32", word) << txl.uint32(16))


def _load_gate(g, token, head, channel, stride):
    value = txl.local_scalar("float32")
    txl.ptx.ld.global_.nc.f32(
        value, g.ptr_to([token * txl.int64(stride) + txl.Cast("int64", head * D + channel)])
    )
    return value


def _gate_offset(g_last, range_safe):
    return txl.Select(
        range_safe & (g_last < txl.float32(-80.0)) & (g_last >= txl.float32(-160.0)),
        txl.float32(80.0),
        txl.float32(0.0),
    )


def _gate_floor(range_safe):
    return txl.Select(range_safe, txl.float32(8.271806125530277e-25), txl.float32(0.0625))


def _needs_stable(g_last, range_safe):
    value = txl.local_scalar("float32")
    word = txl.local_scalar("uint32")
    txl.ptx.ex2.approx.ftz.f32(value, g_last + _gate_offset(g_last, range_safe))
    txl.ptx.cvt.rn.bf16x2.f32(word, value, value)
    return txl.reinterpret("float32", (word & txl.uint32(0xFFFF)) << txl.uint32(16)) < _gate_floor(
        range_safe
    )


def _state_needs_stable(g_last):
    """Match the bf16 cache consumer, including rounding at the dispatch boundary."""
    value = txl.local_scalar("float32")
    word = txl.local_scalar("uint32")
    txl.ptx.ex2.approx.ftz.f32(value, g_last)
    txl.ptx.cvt.rn.bf16x2.f32(word, value, value)
    return txl.reinterpret("float32", (word & txl.uint32(0xFFFF)) << txl.uint32(16)) < txl.float32(
        0.0625
    )


def make_range_guard(HV, PARTS):
    @txl.kernel(warps=4, arch="sm_100a", grid="num_entries")
    def guard(
        v: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        h0: txl.gptr[txl.f32],
        dht: txl.gptr[txl.f32],
        g: txl.gptr[txl.f32],
        cu: txl.gptr[txl.i64],
        flags: txl.gptr[txl.i32],
        num_entries: txl.i32,
    ):
        guard_tid = txl.thread_id()
        guard_entry = txl.Cast("int32", txl.cta_id())
        guard_chain = guard_entry // PARTS
        guard_partition = guard_entry % PARTS
        guard_seq = guard_chain // HV
        guard_hv = guard_chain % HV
        guard_bos = txl.local_scalar("int64")
        guard_eos = txl.local_scalar("int64")
        txl.ptx.ld.global_.s64(guard_bos, cu.ptr_to([guard_seq]))
        txl.ptx.ld.global_.s64(guard_eos, cu.ptr_to([guard_seq + 1]))
        guard_bad_gate = txl.local_scalar("bool", init=False)
        guard_chunk = txl.local_scalar("int64", init=txl.Cast("int64", guard_tid // 32))
        with txl.While(guard_bos + 64 * guard_chunk < guard_eos):
            # Each warp issues four independent chunk-end loads before consuming
            # them. Clamping the final group only duplicates the last real end;
            # the CTA-wide existential gate predicate is unchanged.
            guard_gates = txl.alloc_local([16], "float32")
            for batch in range(4):
                guard_token = txl.min(
                    guard_bos + 64 * (guard_chunk + 4 * batch) + 63, guard_eos - 1
                )
                guard_pos = (guard_token * HV + guard_hv) * D + (guard_tid % 32) * 4
                txl.ptx["ld.global.v4.f32"](
                    *(guard_gates[4 * batch + z] for z in range(4)), g.ptr_to([guard_pos])
                )
            for z in range(16):
                txl.assign(guard_bad_gate, guard_bad_gate | (guard_gates[z] < txl.float32(-4.0)))
            txl.assign(guard_chunk, guard_chunk + txl.int64(16))
        guard_gate_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            guard_gate_count, txl.uint32(0), txl.uint32(128), txl.ptx.pred(guard_bad_gate)
        )
        guard_bad = txl.local_scalar("bool", init=False)
        with txl.If(guard_gate_count != txl.uint32(0)), txl.Then():
            guard_token = txl.local_scalar(
                "int64", init=guard_bos + txl.Cast("int64", guard_tid // 8 + 16 * guard_partition)
            )
            with txl.While(guard_token < guard_eos):
                guard_pos = (guard_token * HV + guard_hv) * D + (guard_tid % 8) * 16
                for inputs in (v, do):
                    words = txl.alloc_local([8], "uint32")
                    txl.ptx["ld.global.v8.b32"](
                        *(words[z] for z in range(8)), inputs.ptr_to([guard_pos])
                    )
                    for z in range(8):
                        word = words[z]
                        txl.assign(
                            guard_bad,
                            guard_bad
                            | ((word & txl.uint32(0x7FFF)) > txl.uint32(0x4080))
                            | (
                                ((word >> txl.uint32(16)) & txl.uint32(0x7FFF)) > txl.uint32(0x4080)
                            ),
                        )
                txl.assign(guard_token, guard_token + txl.int64(16 * PARTS))
            guard_state_i = txl.local_scalar("int32", init=guard_tid * 4 + 512 * guard_partition)
            with txl.While(guard_state_i < txl.int32(D * D)):
                guard_pos = txl.Cast("int64", guard_chain) * (D * D) + guard_state_i
                for inputs in (h0, dht):
                    words = txl.alloc_local([4], "uint32")
                    txl.ptx["ld.global.v4.b32"](
                        *(words[z] for z in range(4)), inputs.ptr_to([guard_pos])
                    )
                    for z in range(4):
                        txl.assign(
                            guard_bad,
                            guard_bad
                            | ((words[z] & txl.uint32(0x7FFFFFFF)) > txl.uint32(0x3F800000)),
                        )
                txl.assign(guard_state_i, guard_state_i + txl.int32(512 * PARTS))
        guard_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            guard_count, txl.uint32(0), txl.uint32(128), txl.ptx.pred(guard_bad)
        )
        with txl.If(guard_tid == 0), txl.Then():
            guard_mode = txl.Select(
                guard_gate_count != txl.uint32(0), txl.uint32(1), txl.uint32(0)
            ) | txl.Select(guard_count != txl.uint32(0), txl.uint32(2), txl.uint32(0))
            txl.ptx.st.global_.u32(flags.ptr_to([guard_entry]), guard_mode)

    return guard


def make_native_mega_kernel(HQ: int, HV: int, static_grid=None):
    txl.MBarrier._wait = _ptx_mbarrier_wait
    G = HV // HQ
    HALF_DA_READOUT = HQ < 96
    HALF_XY_READOUT = HQ < 96
    HQK64 = txl.int64(HQ * D)
    G = HV // HQ
    HQK = HQ * D
    HVK = HV * D
    HVK64 = txl.int64(HVK)

    F_KV, F_AKK, F_HS, F_G, F_KG, F_KBG, F_VB = 0, 4, 6, 10, 14, 18, 22

    B_QK, B_DO, B_G, B_AQK, B_AKK, B_T2, B_DHB, B_DV2 = 0, 4, 8, 12, 13, 14, 16, 20

    MAXSEQ = 64
    TM_H, TM_W0, TM_U0, TM_KT, TM_VT = 0, 128, 256, 384, 448
    TM_DH, TM_BW, TM_DV2, TM_QT, TM_KTB, TM_T1, TM_KB = 0, 128, 192, 256, 320, 384, 448

    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8
    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2
    EG_BYTES = CHUNK * D * 2
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2

    TM_ADQ, TM_ADK = 0, 64
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    TMEM_COLS = 512

    @txl.kernel(
        warps=12,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="num_ctas" if static_grid is None else static_grid,
    )
    def kda_bwd_native_mega(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        aqk: txl.gptr[txl.bf16],
        akk: txl.gptr[txl.bf16],
        g: txl.gptr[txl.f32],
        egcache: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        dht: txl.gptr[txl.f32],
        h0: txl.gptr[txl.f32],
        hsnap: txl.gptr[txl.bf16],
        dhsnap: txl.gptr[txl.bf16],
        cu_seqlens: txl.gptr[txl.i64],
        dq: txl.gptr[txl.f32],
        dk: txl.gptr[txl.f32],
        dv: txl.gptr[txl.bf16],
        db: txl.gptr[txl.f32],
        dg: txl.gptr[txl.f32],
        dh0: txl.gptr[txl.f32],
        stream_counter: txl.gptr[txl.i32],
        flags: txl.gptr[txl.i64],
        stream_tab: txl.gptr[txl.i32],
        item_tab: txl.gptr[txl.i32],
        seq_tab: txl.gptr[txl.i32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        g_map: txl.TensorMap,
        eg_map: txl.TensorMap,
        do_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        akk_map: txl.TensorMap,
        h_map: txl.TensorMap,
        dh_map: txl.TensorMap,
        scale: txl.f32,
        num_seqs: txl.i32,
        num_items: txl.i32,
        num_ctas: txl.i32,
        epoch: txl.i32,
        range_flags: txl.gptr[txl.i32],
        range_entries: txl.i32,
    ):
        native_modes = txl.local_scalar("uint32", init=txl.uint32(0))
        native_idx = txl.local_scalar("int32", init=txl.thread_id())
        with txl.While(native_idx < range_entries):
            native_part = txl.local_scalar("uint32")
            txl.ptx.ld.global_.u32(native_part, range_flags.ptr_to([native_idx]))
            txl.assign(native_modes, native_modes | native_part)
            txl.assign(native_idx, native_idx + txl.int32(384))
        native_pred = txl.local_scalar("bool", init=(native_modes & txl.uint32(1)) != txl.uint32(0))
        native_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            native_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(native_pred)
        )
        with txl.If(native_count != txl.uint32(0)), txl.Then():
            txl.Return(txl.int32(0))
        for buf in (q, k, v, aqk, akk, g, egcache, do, hsnap, dhsnap):
            txl.keep_alive(buf.data)
        num_chains = num_seqs * txl.int32(HV)
        num_streams = num_chains * txl.int32(2)
        total_work = num_streams + num_items
        cta = txl.local_scalar("int32", init=txl.Cast("int32", txl.cta_id()))
        ITEM_RING = 4

        ep64 = txl.local_scalar("int64", init=txl.Cast("int64", epoch) * txl.int64(1 << 32))

        sp = txl.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=208)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=88)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        w10 = sp.role("w10", warps=[10], group=auxg)
        w11 = sp.role("w11", warps=[11], group=auxg)

        smem = txl.smem_pool()
        s_tmem = smem.alloc((4,), txl.i32, align=16)

        p_kv = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_akk1 = txl.Pipeline(smem, 2, full="tma", empty="tcgen05")
        p_tiles = txl.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=256)
        p_hs = txl.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = txl.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = txl.Pipeline(smem, 1, full="tma", empty="mbar", init_full=33, init_empty=256)
        b_kvT_done = txl.TCGen05Bar(smem, 1)
        b_kvT_done.init(1)
        b_kv_read = txl.MBarrier(smem, 1)
        b_kv_read.init(256)

        p_qk = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        b_qkT_done = txl.TCGen05Bar(smem, 1)
        b_qkT_done.init(1)
        b_qk_read = txl.MBarrier(smem, 1)
        b_qk_read.init(256)
        b_g_full = txl.TMABar(smem, 1)
        b_g_full.init(33)
        b_g_free = txl.MBarrier(smem, 1)
        b_g_free.init(256)
        b_bdo_full = txl.TMABar(smem, 2)
        b_bdo_full.init(1)
        b_baqk_full = txl.TMABar(smem, 1)
        b_baqk_full.init(1)
        b_baqk_masked = txl.MBarrier(smem, 1)
        b_baqk_masked.init(32)
        b_baqk_empty = txl.TCGen05Bar(smem, 1)
        b_baqk_empty.init(1)
        b_bakk_full = txl.TMABar(smem, 1)
        b_bakk_full.init(1)
        b_bakk_empty = txl.TCGen05Bar(smem, 1)
        b_bakk_empty.init(1)
        b_dhb_stored = txl.MBarrier(smem, 1)
        b_dhb_stored.init(1)
        MB = {}
        for nm in ("prep_ready", "wT_ready", "dhb_ready", "dv2T_ready"):
            MB[nm] = txl.MBarrier(smem, 1)
            MB[nm].init(256)
        TC = {}
        for nm in ("W_done", "dv2_done", "dh_done"):
            TC[nm] = txl.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        b_in_full = txl.TMABar(smem, 1)
        b_in_full.init(33)
        b_eg_full = txl.TMABar(smem, 1)
        b_eg_full.init(1)
        b_mid_free = txl.MBarrier(smem, 1)
        b_mid_free.init(256)
        b_qk_free = txl.MBarrier(smem, 1)
        b_qk_free.init(256)
        b_do_full = txl.TMABar(smem, 1)
        b_do_full.init(1)
        b_h_full = txl.TMABar(smem, 1)
        b_h_full.init(1)
        b_dhb_full = txl.TMABar(smem, 1)
        b_dhb_full.init(1)
        b_aqk_full = txl.TMABar(smem, 1)
        b_aqk_full.init(1)
        b_akk_full = txl.TMABar(smem, 2)
        b_akk_full.init(1)
        b_do_empty = txl.TCGen05Bar(smem, 1)
        b_do_empty.init(1)
        b_h_free = txl.MBarrier(smem, 1)
        b_h_free.init(256)
        b_aqk_empty = txl.TCGen05Bar(smem, 1)
        b_aqk_empty.init(1)
        b_akk_empty = txl.TCGen05Bar(smem, 2)
        b_akk_empty.init(1)
        mbg_names = [
            "t_early",
            "zT_ready",
            "vnT_ready",
            "dv2T_ready",
            "dAqk_tile_ready",
            "dAm_ready",
            "X_ready",
            "intra_ready",
            "dv_epi_done",
        ]
        MBG = {}
        for nm in mbg_names:
            MBG[nm] = txl.MBarrier(smem, 1)
            MBG[nm].init(256)
        b_dg0_ready = txl.MBarrier(smem, 1)
        b_dg0_ready.init(256)
        b_aqk_masked = txl.MBarrier(smem, 1)
        b_aqk_masked.init(64)
        b_akk_masked = txl.MBarrier(smem, 2)
        b_akk_masked.init(64)
        tcg_names = [
            "Z_done",
            "Vn_done",
            "dv2_done",
            "dAqk_done",
            "dk_done",
            "dAs_done",
            "dvb_done",
            "X_done",
            "Y_done",
            "dq2_done",
            "dkt_done",
            "chunk_done",
            "xT_done",
        ]
        TCG = {}
        for nm in tcg_names:
            TCG[nm] = txl.TCGen05Bar(smem, 1)
            TCG[nm].init(1)

        TT = smem.alloc((27, 64, 64), txl.bf16, swizzle=txl.SW128B)
        s_beta1 = smem.alloc((2, CHUNK), txl.f32, align=16)
        s_bbeta = smem.alloc((2, CHUNK), txl.f32, align=16)

        s_beta = smem.alloc((2, 64), txl.f32, align=16)
        s_dgk = smem.alloc((2, 128), txl.f32, align=16)

        s_seq = smem.alloc((MAXSEQ, 4), txl.i32, align=16)

        s_work = smem.alloc((ITEM_RING,), txl.i32, align=16)
        b_work = txl.MBarrier(smem, ITEM_RING)
        b_work.init(1)
        s_ident = smem.alloc((256,), txl.bf16, align=128)

        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_tmem[1]), txl.int32(0))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(txl.thread_id() < txl.int32(256)), txl.Then():
            tid_i = txl.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & txl.int32(15)
            txl.ptx.st.shared.u16(
                s_ident.ptr_to(
                    [
                        (n_i >> 3) * txl.int32(128)
                        + (k_i >> 3) * txl.int32(64)
                        + (n_i & txl.int32(7)) * txl.int32(8)
                        + (k_i & txl.int32(7))
                    ]
                ),
                txl.Cast("uint16", txl.Select(n_i == k_i, txl.int32(0x3F80), txl.int32(0))),
            )
            txl.ptx[FENCE_ASYNC]()
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(s_tmem[0]), txl.uint32(TMEM_COLS)
            )
        with txl.If((txl.warp_id() == 0) & (num_seqs <= txl.int32(MAXSEQ))), txl.Then():
            lane0 = txl.lane_id()
            with txl.serial((num_seqs + txl.int32(31)) >> 5) as blk:
                i = blk * txl.int32(32) + lane0
                with txl.If(i < num_seqs), txl.Then():
                    st4 = txl.alloc_local([4], "int32")
                    txl.ptx["ld.global.nc.v4.s32"](
                        st4[0], st4[1], st4[2], st4[3], seq_tab.ptr_to([i * txl.int32(4)])
                    )
                    for j in range(4):
                        txl.ptx.st.shared.s32(txl.address_of(s_seq[i, j]), st4[j])
        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_work[0]), cta)
            b_work.arrive(0)
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def tmem_preamble():
            tmv = txl.alloc_local([1], "int32")
            txl.ptx.ld.volatile.shared.s32(tmv[0], txl.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = txl.alloc_local([1], "uint32")
            txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            def phase(name):
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.range_start(name))

            def phase_end():
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def bf16_bits_to_f32(u16val):
            return txl.reinterpret("float32", txl.Cast("uint32", u16val) << txl.uint32(16))

        def lo(w):
            return txl.reinterpret("float32", w << txl.uint32(16))

        def hi(w):
            return txl.reinterpret("float32", w & txl.uint32(0xFFFF0000))

        def seq_info(seq):
            """(bos, seq_len, nch) of a sequence from the SMEM table, or from cu_seqlens when it does not fit."""
            bos = txl.local_scalar("int64", init=txl.int64(0))
            seq_len = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(num_seqs <= txl.int32(MAXSEQ)):
                with txl.Then():
                    b32 = txl.local_scalar("int32")
                    txl.ptx.ld.shared.s32(b32, txl.address_of(s_seq[seq, 0]))
                    txl.assign(bos, txl.Cast("int64", b32))
                    txl.ptx.ld.shared.s32(seq_len, txl.address_of(s_seq[seq, 1]))
                with txl.Else():
                    cs = txl.alloc_local([2], "int64")
                    txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
                    txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + txl.int32(1)]))
                    txl.assign(bos, cs[0])
                    txl.assign(seq_len, txl.Cast("int32", cs[1] - cs[0]))
            nch = txl.local_scalar("int32", init=(seq_len + txl.int32(CHUNK - 1)) >> 6)
            return bos, seq_len, nch

        def stream_coords(s):
            """Stream rank s -> (is_fwd, seq, hv, hq, bos, seq_len, nch) from the host-built stream table."""
            sv = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(sv, stream_tab.ptr_to([s]))
            is_fwd = txl.local_scalar("int32", init=sv >> txl.int32(30))
            seq = txl.local_scalar("int32", init=(sv >> txl.int32(15)) & txl.int32(0x7FFF))
            hv = txl.local_scalar("int32", init=sv & txl.int32(0x7FFF))
            hq = txl.local_scalar("int32", init=hv // txl.int32(G))
            bos, seq_len, nch = seq_info(seq)
            return is_fwd, seq, hv, hq, bos, seq_len, nch

        def chunk_base(seq):
            cb = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(num_seqs <= txl.int32(MAXSEQ)):
                with txl.Then():
                    txl.ptx.ld.shared.s32(cb, txl.address_of(s_seq[seq, 3]))
                with txl.Else():
                    with txl.serial(seq) as i:
                        cs = txl.alloc_local([2], "int64")
                        txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                        txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                        txl.assign(
                            cb,
                            cb + ((txl.Cast("int32", cs[1] - cs[0]) + txl.int32(CHUNK - 1)) >> 6),
                        )
            return cb

        def chunk_rows(seq_len, n):
            return txl.min(txl.int32(CHUNK), seq_len - n * txl.int32(CHUNK))

        def load_beta_lanes(dst_ptr_fn, bos, hv, n, rows):
            """Loader warp: every lane fetches beta for tokens lane and lane+32 of the chunk."""
            lane = txl.lane_id()
            for j in range(2):
                t = lane + txl.int32(32 * j)
                tokc = bos + txl.Cast(
                    "int64", n * txl.int32(CHUNK) + txl.min(t, rows - txl.int32(1))
                )
                u = txl.local_scalar("uint16")
                txl.ptx.ld.global_.nc.u16(
                    u, beta.ptr_to([tokc * txl.int64(HV) + txl.Cast("int64", hv)])
                )
                val = txl.Select(t < rows, bf16_bits_to_f32(u), txl.float32(0.0))
                txl.ptx.st.shared.f32(dst_ptr_fn(t), val)

        def seq_len_of(sq):
            cs = txl.alloc_local([2], "int64")
            txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([sq]))
            txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([sq + 1]))
            return cs[0], txl.Cast("int32", cs[1] - cs[0])

        def item_coords(item):
            """Item index -> (c, hq, seq, n, bos, rows, nch).

            (seq, n) comes from the host-built table, which orders chunks by their predicted
            recurrence readiness (both streams of the sequence have published the chunk); the
            qk-head index is the fastest-varying component.
            """
            pos = txl.local_scalar("int32", init=item // txl.int32(HQ))
            hq = txl.local_scalar("int32", init=item - pos * txl.int32(HQ))
            iv = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(iv, item_tab.ptr_to([pos]))
            seq = txl.local_scalar("int32", init=iv >> txl.int32(16))
            n = txl.local_scalar("int32", init=iv & txl.int32(0xFFFF))
            bos, seq_len, nch = seq_info(seq)
            cb = chunk_base(seq)
            c = txl.local_scalar("int32", init=cb + n)
            rows = txl.local_scalar(
                "int32", init=txl.min(txl.int32(CHUNK), seq_len - n * txl.int32(CHUNK))
            )
            return c, hq, seq, n, bos, rows, nch

        def load_beta_lanes_g(bos, hv, n, rows, slot):
            lane = txl.lane_id()
            for j in range(2):
                t = lane + txl.int32(32 * j)
                tokc = bos + txl.Cast(
                    "int64", n * txl.int32(CHUNK) + txl.min(t, rows - txl.int32(1))
                )
                u = txl.local_scalar("uint16")
                txl.ptx.ld.global_.nc.u16(
                    u, beta.ptr_to([tokc * txl.int64(HV) + txl.Cast("int64", hv)])
                )
                val = txl.Select(
                    t < rows,
                    txl.reinterpret("float32", txl.Cast("uint32", u) << txl.uint32(16)),
                    txl.float32(0.0),
                )
                txl.ptx.st.shared.f32(txl.address_of(s_beta[slot, t]), val)

        def work_wait(j):
            """The j-th work unit of this CTA (published by the loader); >= total_work means done."""
            slot = j % txl.int32(ITEM_RING)
            b_work.wait(slot, (j // txl.int32(ITEM_RING)) & txl.int32(1))
            v_ = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(v_, txl.address_of(s_work[slot]))
            return v_

        def claim_publish(j):
            """Loader: claim work unit j for the CTA and publish it in the ring."""
            slot = j % txl.int32(ITEM_RING)
            with txl.If(elected()), txl.Then():
                nxt = txl.local_scalar("int32")
                txl.ptx["atom.acq_rel.gpu.global.add.s32"](
                    nxt, stream_counter.ptr_to([0]), txl.int32(1)
                )
                txl.ptx.st.shared.s32(txl.address_of(s_work[slot]), nxt + num_ctas)
                b_work.arrive(slot)

        kk_ = txl.local_scalar("int32", init=txl.int32(0))
        cur = txl.local_scalar("int32", init=work_wait(kk_))

        def g_masker(MROW):
            cyc = txl.local_scalar("int32", init=txl.int32(0))
            lane = txl.lane_id()
            rowc = txl.local_scalar("int32", init=MROW * txl.int32(32) + lane)
            item = txl.local_scalar("int32", init=cur - num_streams)
            with txl.While(cur < total_work):
                txl.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                with txl.serial(G) as gi:
                    par = cyc & txl.int32(1)
                    b_aqk_full.wait(0, par)
                    diag = txl.alloc_local([4], "uint32")
                    dmat = lane >> txl.int32(3)
                    dblk = MROW * txl.int32(4) + dmat
                    dptr = TT[S_AQK].ptr_to(
                        dblk * txl.int32(8) + (lane & txl.int32(7)), dblk * txl.int32(8)
                    )
                    txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                        diag[0], diag[1], diag[2], diag[3], dptr
                    )
                    drow = lane >> txl.int32(2)
                    dcol = (lane & txl.int32(3)) * txl.int32(2)
                    dmask = txl.Select(
                        dcol > drow,
                        txl.uint32(0),
                        txl.Select(dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)),
                    )
                    for e in range(4):
                        blk_row = (MROW * txl.int32(4) + txl.int32(e)) * txl.int32(8) + drow
                        txl.assign(
                            diag[e], txl.Select(blk_row < rows, diag[e] & dmask, txl.uint32(0))
                        )
                    txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                        dptr, diag[0], diag[1], diag[2], diag[3]
                    )
                    for u in range(1, 8):
                        with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AQK].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    with txl.If(rowc >= rows), txl.Then():
                        for u in range(8):
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AQK].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    txl.ptx[FENCE_ASYNC]()
                    b_aqk_masked.arrive(0)

                    b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                    with txl.If(rowc >= rows), txl.Then():
                        for u in range(8):
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AKK + par].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    txl.ptx[FENCE_ASYNC]()
                    b_akk_masked.arrive(par)
                    txl.assign(cyc, cyc + txl.int32(1))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

        with cg:
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            xg = txl.local_scalar("int32", init=x >> 5)
            xgc = txl.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            st_te = txl.PipelineState(2, phase=0)
            st_hs = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(1, phase=0)
            st_w = txl.PipelineState(2, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            fkv = txl.local_scalar("int32", init=txl.int32(0))
            bqk = txl.local_scalar("int32", init=txl.int32(0))
            st_bg = txl.PipelineState(1, phase=0)
            bcyc = txl.local_scalar("int32", init=txl.int32(0))

            gv = txl.alloc_local([32], "float32")
            kk = txl.alloc_local([32], "float32")
            vv = txl.alloc_local([32], "float32")
            bb = txl.alloc_local([32], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            eg = txl.local_scalar("float32")
            egng = txl.local_scalar("float32")
            bu = txl.local_scalar("uint16")
            ku = txl.local_scalar("uint16")
            vu = txl.local_scalar("uint16")
            qu = txl.local_scalar("uint16")
            phase, phase_end = make_phaser()

            def fwd_body(seq, hv, hq, bos, seq_len, nch):
                """Forward state recurrence, software-pipelined: while the tensor core runs chunk n's
                Vn / state-update MMAs, the compute warps prepare chunk n+1's tiles (from K/V transposed
                into TMEM by the MMA warp) and read chunk n+2's gate."""
                hv64 = txl.Cast("int64", hv)
                gcol = txl.local_scalar("int64", init=hv64 * txl.int64(D) + x64)
                gn_c = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_1 = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_2 = txl.local_scalar("float32", init=txl.float32(0.0))
                bpair = txl.alloc_local([2], "float32")

                def f_g(m, gn_dst):
                    """g of chunk m (32 rows of this channel) into gv; its last valid row into gn_dst."""
                    rows = chunk_rows(seq_len, m)
                    phase("fw-g")
                    p_g.full.wait(0, st_g.phase)
                    phase("f-g")
                    gst = txl.local_scalar("int32", init=F_G + xg)
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(0)
                    st_g.advance()

                def f_tiles(m, gn_m, u_lo, u_hi):
                    """K/V of chunk m from their TMEM transposes (first half only), then token blocks
                    [u_lo, u_hi) of its kg / kbg / vb tiles into set m & 1 (the last half publishes)."""
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, m))
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", m * txl.int32(CHUNK))
                    )
                    sset = txl.local_scalar("int32", init=m & txl.int32(1))
                    if u_lo == 0:
                        phase("fw-kv")
                        b_kvT_done.wait(0, fkv & txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        phase("f-kv")
                        txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                        txl.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_VT + wg * 32))
                        txl.ptx[WAIT_LD]()
                        txl.ptx[TC_FENCE_BEFORE]()
                        b_kv_read.arrive(0)
                        txl.assign(fkv, fkv + txl.int32(1))
                    phase("f-tiles")
                    kg_t = txl.local_scalar("int32", init=F_KG + sset * txl.int32(2) + xs)
                    kbg_t = txl.local_scalar("int32", init=F_KBG + sset * txl.int32(2) + xs)
                    vb_t = txl.local_scalar("int32", init=F_VB + sset * txl.int32(2) + xs)
                    for u in range(u_lo, u_hi):
                        wkg = txl.alloc_local([4], "uint32")
                        wkbg = txl.alloc_local([4], "uint32")
                        wvb = txl.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            valid0 = row0 + txl.int32(i) < rows
                            valid1 = row0 + txl.int32(i + 1) < rows
                            m0 = txl.Select(valid0, txl.float32(1.0), txl.float32(0.0))
                            m1 = txl.Select(valid1, txl.float32(1.0), txl.float32(0.0))
                            eg0 = txl.local_scalar("float32")
                            eg1 = txl.local_scalar("float32")
                            en0 = txl.local_scalar("float32")
                            en1 = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            txl.ptx.ex2.approx.ftz.f32(en0, gn_m - gv[i])
                            txl.ptx.ex2.approx.ftz.f32(en1, gn_m - gv[i + 1])
                            bu0 = txl.local_scalar("uint16")
                            bu1 = txl.local_scalar("uint16")
                            txl.ptx.cvt.rn.bf16.f32(bu0, eg0)
                            txl.ptx.cvt.rn.bf16.f32(bu1, eg1)
                            egidx0 = (tok0 + txl.Cast("int64", row0 + txl.int32(i))) * HVK64 + gcol
                            egidx1 = (
                                tok0 + txl.Cast("int64", row0 + txl.int32(i + 1))
                            ) * HVK64 + gcol
                            with txl.If(valid0), txl.Then():
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    egcache.ptr_to([egidx0]), bu0
                                )
                            with txl.If(valid1), txl.Then():
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    egcache.ptr_to([egidx1]), bu1
                                )
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta1[sset, row0 + i])
                            )
                            pair0 = txl.local_scalar("uint64")
                            pair1 = txl.local_scalar("uint64")
                            pair2 = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                pair0,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(en0, en1),
                            )
                            txl.ptx["mul.rn.f32x2"](pair0, pair0, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair1,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            txl.ptx["mul.rn.f32x2"](pair1, pair1, txl.cuda.make_float2(eg0, eg1))
                            txl.ptx["mul.rn.f32x2"](
                                pair2,
                                txl.cuda.make_float2(vv[i], vv[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            pack_bf16x2(wkg[p], txl.cuda.float2_x(pair0), txl.cuda.float2_y(pair0))
                            pack_bf16x2(wkbg[p], txl.cuda.float2_x(pair1), txl.cuda.float2_y(pair1))
                            pack_bf16x2(wvb[p], txl.cuda.float2_x(pair2), txl.cuda.float2_y(pair2))
                        col = row0 + 8 * u
                        txl.ptx["st.shared.v4.b32"](
                            TT[kg_t].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[kbg_t].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[vb_t].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3]
                        )
                    if u_hi == 4:
                        txl.ptx[FENCE_ASYNC]()

                        txl.ptx["fence.proxy.async.global"]()
                        p_tiles.full.arrive(sset)

                f_g(txl.int32(0), gn_c)
                f_tiles(txl.int32(0), gn_c, 0, 4)
                with txl.If(nch > txl.int32(1)), txl.Then():
                    f_g(txl.int32(1), gn_1)
                with txl.serial(nch) as n:
                    sn = txl.local_scalar("int32", init=n & txl.int32(1))
                    txl.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("fw-hupd")

                    with txl.If(n > txl.int32(0)), txl.Then():
                        p_tiles.empty.wait(st_te.stage, st_te.phase)
                        st_te.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("fw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("f-hdecay")
                    hc0 = wg * 64
                    hsst = txl.local_scalar("int32", init=F_HS + wg * txl.int32(2) + xs)
                    with txl.If(n == txl.int32(0)):
                        with txl.Then():
                            h0base = (
                                (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                            ) * txl.int64(D) + txl.Cast("int64", hc0)
                            for m8 in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    h0.ptr_to([h0base + txl.int64(8 * m8)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[hsst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    for p in range(32):
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair,
                            txl.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                            txl.cuda.make_float2(egn, egn),
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("fw-W")
                    p_w.full.wait(st_w.stage, st_w.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("f-wT")
                    txl.ptx[TC_LD32](
                        *(acc[i] for i in range(32)), tmem_at(TM_W0 + sn * 64 + wg * 32)
                    )
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    wt_t = txl.local_scalar("int32", init=F_KBG + sn * txl.int32(2) + xs)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[wt_t].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(st_w.stage)
                    st_w.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        f_tiles(n + txl.int32(1), gn_1, 0, 2)
                    phase("fw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("f-vnT")
                    txl.ptx[TC_LD32](
                        *(acc[i] for i in range(32)), tmem_at(TM_U0 + sn * 64 + wg * 32)
                    )
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    vn_t = txl.local_scalar("int32", init=F_VB + sn * txl.int32(2) + xs)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[vn_t].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)
                    with txl.If(elected()), txl.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        f_tiles(n + txl.int32(1), gn_1, 2, 4)
                    with txl.If(n + txl.int32(2) < nch), txl.Then():
                        f_g(n + txl.int32(2), gn_2)
                    txl.assign(gn_c, gn_1)
                    txl.assign(gn_1, gn_2)
                    phase_end()

                p_tiles.empty.wait(st_te.stage, st_te.phase)
                st_te.advance()
                txl.ptx[TC_FENCE_AFTER]()

            def bwd_body(seq, hv, hq, bos, seq_len, nch):
                """Backward state-gradient recurrence, software-pipelined: chunk n-1's operand
                preparation (q^T/k^T from TMEM transposes, T1/kbg into TMEM, T2 into shared memory)
                overlaps chunk n's dv2 and dh MMAs."""
                hv64 = txl.Cast("int64", hv)
                gn_c = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_1 = txl.local_scalar("float32", init=txl.float32(0.0))
                bpair = txl.alloc_local([2], "float32")

                def b_prep(m, gn_dst):
                    """g of chunk m into gv (its last valid row into gn_dst), then q^T / k^T from TMEM."""
                    rows = chunk_rows(seq_len, m)
                    phase("bw-in")
                    b_g_full.wait(0, st_bg.phase)
                    phase("b-prep")
                    gst = txl.local_scalar("int32", init=B_G + xg)
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx[FENCE_ASYNC]()
                    b_g_free.arrive(0)
                    st_bg.advance()
                    phase("bw-qk")
                    b_qkT_done.wait(0, bqk & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("b-qk")
                    txl.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_QT + wg * 32))
                    txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KTB + wg * 32))
                    txl.ptx[WAIT_LD]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    b_qk_read.arrive(0)
                    txl.assign(bqk, bqk + txl.int32(1))

                def b_prep2(m, sset, bslot):
                    """T1 (TMEM set sset), T2 (shared) and kbg (TMEM set sset) of chunk m."""
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, m))
                    phase("b-prep2")
                    w1 = txl.alloc_local([16], "uint32")
                    w3 = txl.alloc_local([16], "uint32")
                    for u in range(4):
                        w2 = txl.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            m0 = txl.Select(
                                row0 + txl.int32(i) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            m1 = txl.Select(
                                row0 + txl.int32(i + 1) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            eg0 = txl.local_scalar("float32")
                            eg1 = txl.local_scalar("float32")
                            en0 = txl.local_scalar("float32")
                            en1 = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            txl.ptx.ex2.approx.ftz.f32(en0, txl.float32(0.0) - gv[i])
                            txl.ptx.ex2.approx.ftz.f32(en1, txl.float32(0.0) - gv[i + 1])
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_bbeta[bslot, row0 + i])
                            )
                            pair0 = txl.local_scalar("uint64")
                            pair1 = txl.local_scalar("uint64")
                            pair2 = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                pair0,
                                txl.cuda.make_float2(vv[i], vv[i + 1]),
                                txl.cuda.make_float2(eg0, eg1),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                pair0, pair0, txl.cuda.make_float2(scale, scale)
                            )
                            txl.ptx["mul.rn.f32x2"](pair0, pair0, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair1,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(en0, en1),
                            )
                            txl.ptx["mul.rn.f32x2"](pair1, pair1, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair2,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            txl.ptx["mul.rn.f32x2"](pair2, pair2, txl.cuda.make_float2(eg0, eg1))
                            pack_bf16x2(
                                w1[4 * u + p], txl.cuda.float2_x(pair0), txl.cuda.float2_y(pair0)
                            )
                            pack_bf16x2(w2[p], txl.cuda.float2_x(pair1), txl.cuda.float2_y(pair1))
                            pack_bf16x2(
                                w3[4 * u + p], txl.cuda.float2_x(pair2), txl.cuda.float2_y(pair2)
                            )
                        txl.ptx["st.shared.v4.b32"](
                            TT[B_T2 + xs].ptr_to(xr, row0 + 8 * u), w2[0], w2[1], w2[2], w2[3]
                        )
                    txl.ptx[TC_ST16](
                        tmem_at(TM_T1 + sset * 32 + wg * 16), *(w1[j] for j in range(16))
                    )
                    txl.ptx[TC_ST16](
                        tmem_at(TM_KB + sset * 32 + wg * 16), *(w3[j] for j in range(16))
                    )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["prep_ready"].arrive(0)

                b_prep(nch - txl.int32(1), gn_c)
                b_prep2(nch - txl.int32(1), bcyc & txl.int32(1), bcyc & txl.int32(1))
                with txl.serial(nch) as rn:
                    n = nch - txl.int32(1) - rn
                    par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                    txl.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("bw-dh")
                    with txl.If(rn > txl.int32(0)), txl.Then():
                        TC["dh_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("b-dhb")
                    with txl.If(rn == txl.int32(0)):
                        with txl.Then():
                            dbase = (
                                (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                            ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                            for m8 in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    dht.ptr_to([dbase + txl.int64(8 * m8)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    for p in range(32):
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair,
                            txl.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                            txl.cuda.make_float2(egn, egn),
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                    txl.ptx[TC_ST32](tmem_at(TM_DH + wg * 64), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](
                        tmem_at(TM_DH + wg * 64 + 32), *(acc[32 + i] for i in range(32))
                    )
                    phase("bw-stored")
                    with txl.If(rn > txl.int32(0)), txl.Then():
                        b_dhb_stored.wait(0, par ^ txl.int32(1))
                    phase("b-dhb2")
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    dhst = txl.local_scalar("int32", init=B_DHB + wg * txl.int32(2) + xs)
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[dhst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["dhb_ready"].arrive(0)
                    phase("bw-W")
                    TC["W_done"].wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("b-wT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_BW + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])

                    txl.ptx[TC_ST16](
                        tmem_at(TM_KB + par * 32 + wg * 16), *(wds[j] for j in range(16))
                    )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    MB["wT_ready"].arrive(0)
                    with txl.If(rn + txl.int32(1) < nch), txl.Then():
                        b_prep(n - txl.int32(1), gn_1)
                    phase("bw-dv2")
                    TC["dv2_done"].wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("b-dv2T")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DV2 + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        i = 2 * p
                        m0 = txl.Select(row0 + txl.int32(i) < rows, acc[i], txl.float32(0.0))
                        m1 = txl.Select(
                            row0 + txl.int32(i + 1) < rows, acc[i + 1], txl.float32(0.0)
                        )
                        pack_bf16x2(wds[p], m0, m1)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[B_DV2 + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["dv2T_ready"].arrive(0)
                    with txl.If(rn + txl.int32(1) < nch), txl.Then():
                        b_prep2(n - txl.int32(1), par ^ txl.int32(1), par ^ txl.int32(1))
                    txl.assign(gn_c, gn_1)
                    phase_end()
                    txl.assign(bcyc, bcyc + txl.int32(1))
                TC["dh_done"].wait(0, (bcyc & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                txl.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32))
                txl.ptx[WAIT_LD]()
                obase = (
                    (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                for m8 in range(8):
                    txl.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + txl.int64(8 * m8)]),
                        *(acc[8 * m8 + i] for i in range(8)),
                    )
                b_dhb_stored.wait(0, (bcyc & txl.int32(1)) ^ txl.int32(1))

            with txl.While(cur < num_streams):
                is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                with txl.If(is_fwd == txl.int32(1)):
                    with txl.Then():
                        fwd_body(seq, hv, hq, bos, seq_len, nch)
                    with txl.Else():
                        bwd_body(seq, hv, hq, bos, seq_len, nch)

                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

            # Converge even when this CTA starts with an item and skips streams.
            txl.ptx.bar.sync(txl.uint32(6), txl.uint32(256))

        with cg:
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            phalf = txl.local_scalar("int32", init=x & 1)
            pcol = txl.local_scalar("int32", init=x & ~1)
            prow0 = txl.local_scalar("int32", init=row0 + phalf * 16)
            ps = txl.local_scalar("int32", init=pcol >> 6)
            pr = txl.local_scalar("int32", init=pcol & 63)
            cyc = txl.local_scalar("int32", init=txl.int32(0))

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                txl.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                txl.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                txl.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st8(col, regs, base=0):
                txl.ptx[TC_ST8](tmem_at(col), *(regs[base + i] for i in range(8)))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                for u in range(nunits):
                    txl.ptx["st.shared.v4.b32"](
                        TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                        words[wbase + 4 * u],
                        words[wbase + 4 * u + 1],
                        words[wbase + 4 * u + 2],
                        words[wbase + 4 * u + 3],
                    )

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            def bar_wg():
                txl.ptx.bar.sync(txl.uint32(2) + txl.Cast("uint32", wg), txl.uint32(128))

            def twait(nm):
                TCG[nm].wait(0, cyc & txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()

                txl.ptx[FENCE_ASYNC]()
                txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))

            def marrive(nm):
                txl.ptx[TC_FENCE_BEFORE]()
                MBG[nm].arrive(0)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o],
                            frag[o + 1],
                            frag[o + 2],
                            frag[o + 3],
                            tile.m8n8x4(
                                row0 + txl.int32(16 * rb), col0 + txl.int32(16 * cb_), lane
                            ),
                        )

            def store_transpose_frag(base, frag):
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                mm = lane >> txl.int32(3)
                jj = lane & txl.int32(7)
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        ptr = tile.ptr_to(
                            col0 + txl.int32(16 * cb_) + (mm >> txl.int32(1)) * txl.int32(8) + jj,
                            row0 + txl.int32(16 * rb) + (mm & txl.int32(1)) * txl.int32(8),
                        )
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )

            egA = txl.alloc_local([16], "float32")
            egB = txl.alloc_local([16], "float32")
            enA = txl.alloc_local([16], "float32")
            enB = txl.alloc_local([16], "float32")
            egcw = txl.alloc_local([16], "uint32")
            t4 = txl.alloc_local([4], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            dgv = txl.alloc_local([32], "float32")
            oq = txl.alloc_local([8], "float32")
            ok8 = txl.alloc_local([8], "float32")
            egn = txl.local_scalar("float32")
            dgk = txl.local_scalar("float32")
            dgk_k = txl.local_scalar("float32")
            t0 = txl.local_scalar("float32")
            t1 = txl.local_scalar("float32")
            u16 = txl.local_scalar("uint16")

            def rcp(dst, val):
                txl.ptx.rcp.approx.ftz.f32(dst, val)

            def s_beta_row(c):
                b = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b, txl.address_of(s_beta[cyc & txl.int32(1), c]))
                return b

            phase, phase_end = make_phaser()
            item = txl.local_scalar("int32", init=cur - num_streams)
            with txl.While(cur < total_work):
                txl.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                hq64 = txl.Cast("int64", hq)
                tok0 = txl.local_scalar("int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK)))
                last = txl.local_scalar("int32", init=rows - txl.int32(1))
                xq_base = txl.local_scalar(
                    "int64",
                    init=(tok0 + txl.Cast("int64", row0)) * HQK64 + hq64 * txl.int64(D) + x64,
                )
                with txl.serial(G) as gi:
                    hv = txl.local_scalar("int32", init=hq * txl.int32(G) + gi)
                    hv64 = txl.Cast("int64", hv)
                    par = cyc & txl.int32(1)
                    x_base = txl.local_scalar(
                        "int64",
                        init=(tok0 + txl.Cast("int64", row0)) * HVK64 + hv64 * txl.int64(D) + x64,
                    )

                    phase("w-in")
                    b_in_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")

                    xf = txl.alloc_local([32], "float32")
                    egf = txl.alloc_local([32], "float32")
                    qw = txl.alloc_local([16], "uint32")
                    kw = txl.alloc_local([16], "uint32")
                    t3w = txl.alloc_local([16], "uint32")
                    qc = txl.alloc_local([16], "uint32")
                    kc = txl.alloc_local([16], "uint32")
                    vc = txl.alloc_local([16], "uint32")
                    prep0 = txl.local_scalar("uint64")
                    prep1 = txl.local_scalar("uint64")
                    scale_pair = txl.local_scalar("uint64", init=txl.cuda.make_float2(scale, scale))
                    bpair = txl.alloc_local([2], "float32")

                    txl.ptx[TC_LD4](t4[0], t4[1], t4[2], t4[3], tmem_at(S1 + ((last >> 2) << 2)))
                    ld32(xf, S2 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    lq = last & txl.int32(3)
                    txl.assign(
                        egn,
                        txl.Select(
                            lq == txl.int32(0),
                            t4[0],
                            txl.Select(
                                lq == txl.int32(1),
                                t4[1],
                                txl.Select(lq == txl.int32(2), t4[2], t4[3]),
                            ),
                        ),
                    )
                    for half in range(2):
                        vb32 = txl.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[par, row0 + i])
                            )
                            txl.assign(vb32[2 * p], xf[i] * bpair[0])
                            txl.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        txl.ptx[TC_ST16](
                            tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16))
                        )
                    txl.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    txl.ptx[WAIT_LD]()

                    for i in range(32):
                        txl.assign(
                            egf[i],
                            txl.Select(
                                (row0 + txl.int32(i) < rows) & (egf[i] != txl.float32(0.0)),
                                egf[i],
                                txl.float32(1.0),
                            ),
                        )
                    for i in range(16):
                        m0 = txl.Select(
                            row0 + txl.int32(2 * i) < rows, txl.float32(1.0), txl.float32(0.0)
                        )
                        m1 = txl.Select(
                            row0 + txl.int32(2 * i + 1) < rows, txl.float32(1.0), txl.float32(0.0)
                        )
                        txl.ptx["mul.rn.f32x2"](
                            prep0,
                            txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                            txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                        )
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, txl.cuda.make_float2(m0, m1))
                        pack_bf16x2(qw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        pack_bf16x2(egcw[i], egf[2 * i], egf[2 * i + 1])
                    for half in range(2):
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)),
                            tmem_at(S4 + wg * 32 + 16 * half),
                        )
                        txl.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            m0 = txl.Select(
                                row0 + txl.int32(2 * i) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            m1 = txl.Select(
                                row0 + txl.int32(2 * i + 1) < rows,
                                txl.float32(1.0),
                                txl.float32(0.0),
                            )
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[par, row0 + 2 * i])
                            )
                            rcp(t0, egf[2 * i])
                            rcp(t1, egf[2 * i + 1])
                            txl.ptx["mul.rn.f32x2"](
                                prep0,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(t0, t1),
                            )
                            txl.ptx["mul.rn.f32x2"](prep0, prep0, txl.cuda.make_float2(m0, m1))
                            pack_bf16x2(kw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                            txl.ptx["mul.rn.f32x2"](
                                prep1,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep1, prep1, txl.cuda.make_float2(bpair[0], bpair[1])
                            )
                            txl.ptx["mul.rn.f32x2"](prep1, prep1, txl.cuda.make_float2(m0, m1))
                            pack_bf16x2(t3w[i], txl.cuda.float2_x(prep1), txl.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TCG["chunk_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("t_early")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    b_dhb_full.wait(0, par)
                    phase("c2")
                    dgk2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    hst = txl.local_scalar("int32", init=wg * 2 + xs)
                    for u in range(8):
                        hw = txl.alloc_local([4], "uint32")
                        dw = txl.alloc_local([4], "uint32")
                        txl.ptx["ld.shared.v4.b32"](
                            hw[0], hw[1], hw[2], hw[3], TT[S_H + hst].ptr_to(xr, 8 * u)
                        )
                        txl.ptx["ld.shared.v4.b32"](
                            dw[0], dw[1], dw[2], dw[3], TT[DHB + hst].ptr_to(xr, 8 * u)
                        )
                        for p in range(4):
                            txl.ptx["fma.rn.f32x2"](
                                dgk2,
                                txl.cuda.make_float2(lo(hw[p]), hi(hw[p])),
                                txl.cuda.make_float2(lo(dw[p]), hi(dw[p])),
                                dgk2,
                            )
                    txl.assign(dgk, txl.cuda.float2_x(dgk2) + txl.cuda.float2_y(dgk2))

                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")

                    def readout64(slot, stage, mask, negate=False):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with txl.If(lane < txl.int32(16)), txl.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if negate:
                                        val = txl.float32(0.0) - val
                                    vv2.append(txl.Select(mask(cc, jj), val, txl.float32(0.0)))
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                txl.ptx["st.shared.v4.b32"](
                                    TT[stage].ptr_to(cc, row0 + 8 * u),
                                    wds[4 * u],
                                    wds[4 * u + 1],
                                    wds[4 * u + 2],
                                    wds[4 * u + 3],
                                )
                        txl.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        """Read one live 16-lane half of an M=64 accumulator.

                        `.16x256b.x4` maps each thread to two rows and eight
                        adjacent columns per row.  Pairwise bf16 conversion then
                        matches two non-transposed stmatrix.x4 stores exactly.
                        """
                        txl.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        txl.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> txl.int32(2))
                        cc1 = cc0 + txl.int32(8)
                        for rep in range(4):
                            jj0 = row0 + txl.int32(8 * rep) + (lane & txl.int32(3)) * txl.int32(2)
                            jj1 = jj0 + txl.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = txl.float32(0.0) - v00
                                v01 = txl.float32(0.0) - v01
                                v10 = txl.float32(0.0) - v10
                                v11 = txl.float32(0.0) - v11
                            pack_bf16x2(
                                wds[2 * rep],
                                txl.Select(mask(cc0, jj0), v00, txl.float32(0.0)),
                                txl.Select(mask(cc0, jj1), v01, txl.float32(0.0)),
                            )
                            pack_bf16x2(
                                wds[2 * rep + 1],
                                txl.Select(mask(cc1, jj0), v10, txl.float32(0.0)),
                                txl.Select(mask(cc1, jj1), v11, txl.float32(0.0)),
                            )
                        tile = TT[stage]
                        for half in range(2):
                            txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(
                                    quad * txl.int32(16), row0 + txl.int32(16 * half), lane
                                ),
                                wds[4 * half],
                                wds[4 * half + 1],
                                wds[4 * half + 2],
                                wds[4 * half + 3],
                            )
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")
                    if HALF_DA_READOUT:
                        readout64_half(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    else:
                        readout64(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")
                    if HALF_DA_READOUT:
                        readout64_half(S4 + (16 << 16), T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    else:
                        readout64(S1, T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    marrive("dAqk_tile_ready")
                    phase("w-dk")
                    twait("dk_done")
                    twait("dvb_done")
                    phase("passA")

                    pbx = txl.local_scalar("int32", init=txl.int32(PB0) + wg * txl.int32(PB1 - PB0))

                    def pass_a():
                        pa_acc = txl.local_scalar("uint64")
                        pa_v = txl.local_scalar("uint64")
                        pa_db = txl.local_scalar("uint64")
                        pa_dv = txl.local_scalar("uint64")
                        pa_word = txl.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = txl.alloc_local([4], "uint32")
                            txl.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3], TT[ST_V + xs].ptr_to(xr, row0 + 8 * b)
                            )
                            dbp = txl.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                txl.assign(
                                    pa_acc,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                )
                                txl.assign(pa_v, txl.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                txl.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                txl.assign(dbp[2 * p], txl.cuda.float2_x(pa_db))
                                txl.assign(dbp[2 * p + 1], txl.cuda.float2_y(pa_db))
                                txl.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], txl.address_of(s_beta[par, row0 + i])
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, txl.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(
                                    pa_word, txl.cuda.float2_x(pa_dv), txl.cuda.float2_y(pa_dv)
                                )
                                with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                                    txl.ptx["st.global.L1::no_allocate.b16"](
                                        dv.ptr_to([x_base + txl.int64(i * HVK)]),
                                        txl.Cast("uint16", pa_word),
                                    )
                                with txl.If(row0 + txl.int32(i + 1) < rows), txl.Then():
                                    txl.ptx["st.global.L1::no_allocate.b16"](
                                        dv.ptr_to([x_base + txl.int64((i + 1) * HVK)]),
                                        txl.Cast("uint16", pa_word >> txl.uint32(16)),
                                    )
                            for e in range(8):
                                i = 8 * b + e
                                txl.ptx.st.shared.f32(
                                    TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e]
                                )
                            dvw = txl.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            txl.ptx["st.shared.v4.b32"](
                                TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                dvw[0],
                                dvw[1],
                                dvw[2],
                                dvw[3],
                            )
                            if b < 3:
                                txl.ptx[WAIT_LD]()

                    pass_a()
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    if HALF_XY_READOUT:
                        readout64_half(S1, T6, lambda cc, jj: cc < rows)
                    else:
                        readout64(S1, T6, lambda cc, jj: cc < rows)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & txl.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & txl.int32(1)) * 32 + lane
                    dsum_v2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    dsum2 = txl.local_scalar("uint64")
                    sum_pair0 = txl.local_scalar("uint64")
                    sum_pair1 = txl.local_scalar("uint64")
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(sum_pair0, txl.cuda.make_float2(t4[0], t4[1]))
                        txl.assign(sum_pair1, txl.cuda.make_float2(t4[2], t4[3]))
                        txl.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                        txl.ptx["add.rn.f32x2"](dsum_v2, dsum_v2, sum_pair0)
                    txl.ptx[FENCE_ASYNC]()
                    b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    if HALF_XY_READOUT:
                        readout64_half(
                            S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True
                        )
                    else:
                        readout64(S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True)
                    marrive("intra_ready")
                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi-q")
                    dgk_k2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def emit_group_output(b, vals, tm_col, out, obase):
                        """dq/dk for 8 tokens of this channel: accumulate over the head group in TMEM."""
                        if G > 1:
                            with txl.If(gi > txl.int32(0)), txl.Then():
                                ld8(oq, tm_col + wg * 32 + 8 * b)
                                txl.ptx[WAIT_LD]()
                                for e in range(8):
                                    txl.assign(vals[e], vals[e] + oq[e])
                        with txl.If(gi == txl.int32(G - 1)):
                            with txl.Then():
                                for e in range(8):
                                    i = 8 * b + e
                                    with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                                        txl.ptx["st.global.L1::no_allocate.f32"](
                                            out.ptr_to([obase + txl.int64(i * HQK)]), vals[e]
                                        )
                            if G > 1:
                                with txl.Else():
                                    st8(tm_col + wg * 32 + 8 * b, vals)
                                    txl.ptx[WAIT_ST]()

                    def epilogue():
                        pair0 = txl.local_scalar("uint64")
                        pair1 = txl.local_scalar("uint64")
                        pair2 = txl.local_scalar("uint64")
                        pair3 = txl.local_scalar("uint64")
                        pair4 = txl.local_scalar("uint64")
                        pair5 = txl.local_scalar("uint64")
                        q_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p
                                rcp(enA[i >> 1], lo(egcw[i >> 1]))
                                rcp(enB[i >> 1], hi(egcw[i >> 1]))
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                    txl.cuda.make_float2(scale, scale),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair0,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                txl.assign(ok8[2 * p], txl.cuda.float2_x(pair0))
                                txl.assign(ok8[2 * p + 1], txl.cuda.float2_y(pair0))
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])),
                                    pair0,
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair1))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair1))
                            emit_group_output(b, ok8, TM_ADQ, dq, xq_base)
                            if b < 3:
                                txl.ptx[WAIT_LD]()
                        phase("w-dkt")
                        twait("dkt_done")
                        phase("epi-k")
                        dbx = 2 * wg
                        k_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                txl.assign(pair0, txl.cuda.make_float2(enA[i >> 1], enB[i >> 1]))
                                txl.assign(
                                    pair1, txl.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1]))
                                )
                                txl.ptx["add.rn.f32x2"](
                                    pair2,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    txl.cuda.make_float2(
                                        acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]
                                    ),
                                )
                                txl.ptx["mul.rn.f32x2"](pair2, pair2, pair0)
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    txl.cuda.make_float2(
                                        acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]
                                    ),
                                    txl.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                )
                                txl.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                txl.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane),
                                    txl.cuda.float2_x(pair4),
                                )
                                txl.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(
                                        4 * ((i + 1) & 15) + quad, 2 * lane
                                    ),
                                    txl.cuda.float2_y(pair4),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair4,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair0,
                                )
                                txl.ptx["mul.rn.f32x2"](pair5, pair1, pair4)
                                txl.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    pair3,
                                    txl.cuda.make_float2(
                                        s_beta_row(row0 + i), s_beta_row(row0 + i + 1)
                                    ),
                                )
                                txl.ptx["add.rn.f32x2"](pair5, pair2, pair3)
                                txl.assign(ok8[4 * (b % 2) + 2 * p], txl.cuda.float2_x(pair5))
                                txl.assign(ok8[4 * (b % 2) + 2 * p + 1], txl.cuda.float2_y(pair5))
                                txl.ptx["sub.rn.f32x2"](pair3, pair3, pair2)
                                txl.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, txl.cuda.make_float2(dgv[i], dgv[i + 1])
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair5))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair5))
                            if b % 2 == 1:
                                emit_group_output(b // 2, ok8, TM_ADK, dk, xq_base)
                            if b < 7:
                                txl.ptx[WAIT_LD]()
                        bar_wg()
                        txl.assign(dsum2, dsum_v2)
                        for u in range(8):
                            txl.ptx["ld.shared.v4.f32"](
                                t4[0],
                                t4[1],
                                t4[2],
                                t4[3],
                                TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u),
                            )
                            txl.assign(sum_pair0, txl.cuda.make_float2(t4[0], t4[1]))
                            txl.assign(sum_pair1, txl.cuda.make_float2(t4[2], t4[3]))
                            txl.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                            txl.ptx["add.rn.f32x2"](dsum2, dsum2, sum_pair0)
                        dsum = txl.local_scalar(
                            "float32", init=txl.cuda.float2_x(dsum2) + txl.cuda.float2_y(dsum2)
                        )
                        txl.ptx[FENCE_ASYNC]()
                        b_h_free.arrive(0)
                        for s in (1, 2):
                            r = txl.local_scalar("uint32")
                            txl.ptx.shfl_sync.bfly.b32(
                                r,
                                txl.reinterpret("uint32", dsum),
                                txl.uint32(s),
                                txl.uint32(0x1F),
                                txl.uint32(0xFFFFFFFF),
                            )
                            txl.assign(dsum, dsum + txl.reinterpret("float32", r))
                        with txl.If((tq == txl.int32(0)) & (row0 + ti < rows)), txl.Then():
                            txl.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to(
                                    [(tok0 + txl.Cast("int64", row0 + ti)) * txl.int64(HV) + hv64]
                                ),
                                dsum,
                            )

                    epilogue()
                    txl.assign(dgk_k, txl.cuda.float2_x(dgk_k2) + txl.cuda.float2_y(dgk_k2))
                    phase("cumsum")
                    for i in range(32):
                        txl.assign(
                            dgv[i], txl.Select(row0 + txl.int32(i) < rows, dgv[i], txl.float32(0.0))
                        )

                    tot = txl.alloc_local([16], "float32")
                    for i in range(16):
                        txl.assign(tot[i], dgv[2 * i] + dgv[2 * i + 1])
                    for w in (8, 4, 2, 1):
                        for i in range(w):
                            txl.assign(tot[i], tot[i] + tot[i + w])
                    txl.ptx.st.shared.f32(
                        txl.address_of(s_dgk[wg, x]),
                        dgk + dgk_k + txl.Select(wg == txl.int32(0), txl.float32(0.0), tot[0]),
                    )
                    b_dg0_ready.arrive(0)
                    for i in range(30, -1, -1):
                        txl.assign(dgv[i], dgv[i] + dgv[i + 1])
                    b_dg0_ready.wait(0, cyc & txl.int32(1))
                    txl.ptx.ld.shared.f32(t0, txl.address_of(s_dgk[txl.int32(1) - wg, x]))
                    txl.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                            txl.ptx["st.global.L1::no_allocate.f32"](
                                dg.ptr_to([x_base + txl.int64(i * HVK)]), dgv[i] + t1
                            )
                    phase_end()
                    txl.assign(cyc, cyc + txl.int32(1))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

        with auxg:
            with mma:
                tm = tmem_preamble()
                bd1 = txl.alloc_local([1], "uint64")
                zq1 = txl.alloc_local([1], "int32")

                op_kbg_k = Op(bd1, F_KBG, 128, 64, "k")
                op_vb_k = Op(bd1, F_VB, 128, 64, "k")
                op_kg_k = Op(bd1, F_KG, 128, 64, "k")
                op_akk1_k = Op(bd1, F_AKK, 64, 64, "k")
                op_hs_mn = Op(bd1, F_HS, 128, 128, "mn")
                op_w_mn = Op(bd1, F_KBG, 128, 128, "mn")
                op_kraw = Op(bd1, F_KV, 128, 64, "mn")
                op_vraw = Op(bd1, F_KV + 1, 128, 64, "mn")
                ID_T1 = idesc(128, 16, ta=1)
                bdI1 = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )
                st_kv1 = txl.PipelineState(1, phase=0)
                p1m = txl.local_scalar("int32", init=txl.int32(0))
                SET_UNITS = 2 * UNITS_PER_STAGE

                op_bakk_k = Op(bd1, B_AKK, 64, 64, "k")
                op_bdo_mn = Op(bd1, B_DO, 64, 64, "mn")
                DO2_UNITS = 2 * UNITS_PER_STAGE
                op_baqk_mn = Op(bd1, B_AQK, 64, 64, "mn")
                op_dhb_mn = Op(bd1, B_DHB, 128, 128, "mn")
                op_t2_mn = Op(bd1, B_T2, 128, 128, "mn")
                op_dv2_k = Op(bd1, B_DV2, 128, 64, "k")
                op_qraw = Op(bd1, B_QK, 128, 64, "mn")
                op_kraw_b = Op(bd1, B_QK + 1, 128, 64, "mn")
                st_qk1 = txl.PipelineState(1, phase=0)
                bqm = txl.local_scalar("int32", init=txl.int32(0))
                ID_M128N64 = idesc(128, 64)
                ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_HUPD = idesc(128, 128)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                st_tiles = txl.PipelineState(2, phase=0)
                st_akk = txl.PipelineState(2, phase=0)
                st_hs = txl.PipelineState(1, phase=0)
                st_w = txl.PipelineState(2, phase=0)
                st_vn = txl.PipelineState(1, phase=0)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                mphase, mphase_end = make_phaser()

                def encode_base():
                    txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                    txl.cuda.tcgen05.encode_matrix_descriptor(
                        txl.address_of(bd1[0]),
                        TT[zq1[0]].ptr_to(0, 0),
                        ldo=Op.LBO_BASE,
                        sdo=SBO_UNITS,
                        swizzle=txl.SW128B.value,
                    )

                def kv_transpose():
                    """Raw K and V chunk tiles -> channel-major fp32 K^T / V^T in TMEM (eight identity MMAs)."""
                    mphase("fmw-kv")
                    p_kv.full.wait(0, st_kv1.phase)
                    b_kv_read.wait(0, (p1m & txl.int32(1)) ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("fm-kvT")
                    with txl.If(elected()), txl.Then():
                        for src, dst in ((op_kraw, TM_KT), (op_vraw, TM_VT)):
                            for j in range(4):
                                txl.ptx[MMA_SS](
                                    txl.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    txl.uint32(ID_T1),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.ptx.pred(0),
                                )
                        b_kvT_done.arrive(0)
                        p_kv.empty.arrive(0)
                    st_kv1.advance()
                    txl.assign(p1m, p1m + txl.int32(1))

                def qk_transpose():
                    """Raw q and k chunk tiles -> channel-major fp32 q^T / k^T in TMEM (eight identity MMAs)."""
                    mphase("bmw-qk")
                    p_qk.full.wait(0, st_qk1.phase)
                    b_qk_read.wait(0, (bqm & txl.int32(1)) ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("bm-qkT")
                    with txl.If(elected()), txl.Then():
                        for src, dst in ((op_qraw, TM_QT), (op_kraw_b, TM_KTB)):
                            for j in range(4):
                                txl.ptx[MMA_SS](
                                    txl.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    txl.uint32(ID_T1),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.ptx.pred(0),
                                )
                        b_qkT_done.arrive(0)
                        p_qk.empty.arrive(0)
                    st_qk1.advance()
                    txl.assign(bqm, bqm + txl.int32(1))

                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            encode_base()
                            kv_transpose()
                            with txl.If(nch > txl.int32(1)), txl.Then():
                                kv_transpose()
                            with txl.serial(nch) as n:
                                set_u = txl.local_scalar(
                                    "uint64",
                                    init=txl.Cast("uint64", n & txl.int32(1))
                                    * txl.uint64(SET_UNITS),
                                )
                                akk_u = txl.local_scalar(
                                    "uint64",
                                    init=txl.Cast("uint64", st_akk.stage)
                                    * txl.uint64(UNITS_PER_STAGE),
                                )
                                dW = (n & txl.int32(1)) * 64
                                mphase("fmw-tiles")
                                p_tiles.full.wait(st_tiles.stage, st_tiles.phase)
                                p_akk1.full.wait(st_akk.stage, st_akk.phase)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-WU")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_W0 + dW,
                                        op_kbg_k,
                                        op_akk1_k,
                                        ID_M128N64,
                                        False,
                                        a_units=set_u,
                                        b_units=akk_u,
                                    )
                                    p_w.full.arrive(st_w.stage)
                                    mma_chain(
                                        tm,
                                        TM_U0 + dW,
                                        op_vb_k,
                                        op_akk1_k,
                                        ID_M128N64,
                                        False,
                                        a_units=set_u,
                                        b_units=akk_u,
                                    )
                                    p_akk1.empty.arrive(st_akk.stage)
                                st_akk.advance()
                                mphase("fmw-wT")
                                p_w.empty.wait(st_w.stage, st_w.phase)
                                st_w.advance()
                                mphase("fmw-hs")
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-Vn")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_U0 + dW,
                                        op_hs_mn,
                                        op_w_mn,
                                        ID_VN,
                                        True,
                                        b_units=set_u,
                                    )
                                    p_vn.full.arrive(0)
                                st_hs.advance()
                                mphase("fmw-vnT")
                                p_vn.empty.wait(0, st_vn.phase)
                                st_vn.advance()
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-hupd")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_H,
                                        op_kg_k,
                                        op_vb_k,
                                        ID_HUPD,
                                        True,
                                        a_units=set_u,
                                        b_units=set_u,
                                    )
                                    p_tiles.empty.arrive(st_tiles.stage)
                                st_tiles.advance()
                                with txl.If(n + txl.int32(2) < nch), txl.Then():
                                    kv_transpose()
                                mphase_end()
                        with txl.Else():
                            encode_base()
                            qk_transpose()
                            with txl.serial(nch) as rn:
                                par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                                t1_col = TM_T1 + par * 32
                                kb_col = TM_KB + par * 32
                                do_u = txl.local_scalar(
                                    "uint64", init=txl.Cast("uint64", par) * txl.uint64(DO2_UNITS)
                                )
                                mphase("bmw-prep")
                                MB["prep_ready"].wait(0, par)
                                b_bakk_full.wait(0, par)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-W")
                                with txl.If(elected()), txl.Then():
                                    mma_chain_ta(tm, TM_BW, kb_col, op_bakk_k, ID_M128N64, False)
                                    TC["W_done"].arrive(0)
                                    b_bakk_empty.arrive(0)
                                with txl.If(rn + txl.int32(1) < nch), txl.Then():
                                    qk_transpose()
                                mphase("bmw-dhb")
                                MB["dhb_ready"].wait(0, par)
                                b_baqk_masked.wait(0, par)
                                b_bdo_full.wait(par, (bcyc >> 1) & txl.int32(1))
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dv2")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_DV2,
                                        op_bdo_mn,
                                        op_baqk_mn,
                                        ID_128x64_TATB,
                                        False,
                                        a_units=do_u,
                                    )
                                    b_baqk_empty.arrive(0)
                                    mma_chain(tm, TM_DV2, op_dhb_mn, op_t2_mn, ID_128x64_TATB, True)
                                    TC["dv2_done"].arrive(0)
                                mphase("bmw-rd")
                                MB["wT_ready"].wait(0, par)
                                MB["dv2T_ready"].wait(0, par)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dh")
                                with txl.If(elected()), txl.Then():
                                    mma_chain_ta(
                                        tm,
                                        TM_DH,
                                        t1_col,
                                        op_bdo_mn,
                                        ID_128x128_TB,
                                        True,
                                        b_units=do_u,
                                    )
                                    mma_chain_ta(tm, TM_DH, kb_col, op_dv2_k, ID_128x128_NB, True)
                                    TC["dh_done"].arrive(0)
                                mphase_end()
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (q_map, k_map, v_map, g_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                st_kv = txl.PipelineState(1, phase=1)
                st_akk = txl.PipelineState(2, phase=1)
                st_g = txl.PipelineState(1, phase=1)
                st_qk = txl.PipelineState(1, phase=1)
                st_bg = txl.PipelineState(1, phase=1)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            with txl.serial(nch) as n:
                                tok0 = bos32 + n * txl.int32(CHUNK)
                                rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                                p_kv.empty.wait(0, st_kv.phase)
                                with txl.If(elected()), txl.Then():
                                    p_kv.full.arrive(0, tx_count=KV_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[F_KV + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                            txl.address_of(k_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                        txl.ptx[TMA_LD](
                                            TT[F_KV + txl.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0),
                                            txl.address_of(v_map),
                                            txl.int32(d0),
                                            tok0,
                                            hv,
                                            mb,
                                        )
                                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(k_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hq,
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(v_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hv,
                                            )
                                        for d0 in (0, 32, 64, 96):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(g_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hv,
                                            )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(akk_map),
                                            txl.int32(0),
                                            tok0 + txl.int32(CHUNK),
                                            hv,
                                        )
                                st_kv.advance()
                                p_g.empty.wait(0, st_g.phase)
                                with txl.If(elected()), txl.Then():
                                    p_g.full.arrive(0, tx_count=G_BYTES)
                                    mbg = txl.cuda.cvta_generic_to_shared(p_g.full.ptr_to([0]))
                                    for j in range(4):
                                        txl.ptx[TMA_LD](
                                            TT[F_G + txl.int32(j)].ptr_to(0, 0),
                                            txl.address_of(g_map),
                                            txl.int32(32 * j),
                                            tok0,
                                            hv,
                                            mbg,
                                        )

                                bslot = txl.local_scalar("int32", init=n & txl.int32(1))
                                load_beta_lanes(
                                    lambda t: txl.address_of(s_beta1[bslot, t]), bos, hv, n, rows
                                )
                                p_g.full.arrive(0)
                                st_g.advance()
                                p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                                with txl.If(elected()), txl.Then():
                                    p_akk1.full.arrive(st_akk.stage, tx_count=A_BYTES)
                                    mb2 = txl.cuda.cvta_generic_to_shared(
                                        p_akk1.full.ptr_to([st_akk.stage])
                                    )
                                    txl.ptx[TMA_LD](
                                        TT[F_AKK + st_akk.stage].ptr_to(0, 0),
                                        txl.address_of(akk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb2,
                                    )
                                st_akk.advance()
                        with txl.Else():
                            with txl.serial(nch) as rn:
                                n = nch - txl.int32(1) - rn
                                par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                                tok0 = bos32 + n * txl.int32(CHUNK)
                                rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))

                                lphase("blw-qk")
                                p_qk.empty.wait(0, st_qk.phase)
                                lphase("bl-qk")
                                with txl.If(elected()), txl.Then():
                                    p_qk.full.arrive(0, tx_count=KV_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(p_qk.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[B_QK + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                            txl.address_of(q_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                        txl.ptx[TMA_LD](
                                            TT[B_QK + txl.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0),
                                            txl.address_of(k_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                    with txl.If(n > txl.int32(0)), txl.Then():
                                        tokp = tok0 - txl.int32(CHUNK)
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(q_map), txl.int32(d0), tokp, hq
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(k_map), txl.int32(d0), tokp, hq
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(do_map), txl.int32(d0), tokp, hv
                                            )
                                        for d0 in (0, 32, 64, 96):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(g_map), txl.int32(d0), tokp, hv
                                            )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(aqk_map), txl.int32(0), tokp, hv
                                        )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(akk_map), txl.int32(0), tokp, hv
                                        )
                                st_qk.advance()

                                lphase("blw-g")
                                b_g_free.wait(0, st_bg.phase)
                                lphase("bl-g")
                                with txl.If(elected()), txl.Then():
                                    b_g_full.arrive(0, tx_count=G_BYTES)
                                    mbg = txl.cuda.cvta_generic_to_shared(b_g_full.ptr_to([0]))
                                    for j in range(4):
                                        txl.ptx[TMA_LD](
                                            TT[B_G + j].ptr_to(0, 0),
                                            txl.address_of(g_map),
                                            txl.int32(32 * j),
                                            tok0,
                                            hv,
                                            mbg,
                                        )
                                load_beta_lanes(
                                    lambda t: txl.address_of(s_bbeta[par, t]), bos, hv, n, rows
                                )
                                b_g_full.arrive(0)
                                st_bg.advance()

                                lphase("blw-do")
                                with txl.If(rn > txl.int32(1)), txl.Then():
                                    TC["dh_done"].wait(0, par)
                                lphase("bl-do")
                                with txl.If(elected()), txl.Then():
                                    b_bdo_full.arrive(par, tx_count=DO_BYTES)
                                    mbd = txl.cuda.cvta_generic_to_shared(b_bdo_full.ptr_to([par]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[
                                                B_DO + par * txl.int32(2) + txl.int32(d0 // 64)
                                            ].ptr_to(0, 0),
                                            txl.address_of(do_map),
                                            txl.int32(d0),
                                            tok0,
                                            hv,
                                            mbd,
                                        )
                                lphase("blw-a")
                                # These depth-one tiles persist across backward streams.
                                # ``rn`` resets per stream; ``bcyc`` tracks every reuse.
                                with txl.If(bcyc > txl.int32(0)), txl.Then():
                                    b_baqk_empty.wait(0, par ^ txl.int32(1))
                                with txl.If(elected()), txl.Then():
                                    b_baqk_full.arrive(0, tx_count=A_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(b_baqk_full.ptr_to([0]))
                                    txl.ptx[TMA_LD](
                                        TT[B_AQK].ptr_to(0, 0),
                                        txl.address_of(aqk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb,
                                    )
                                with txl.If(bcyc > txl.int32(0)), txl.Then():
                                    b_bakk_empty.wait(0, par ^ txl.int32(1))
                                with txl.If(elected()), txl.Then():
                                    b_bakk_full.arrive(0, tx_count=A_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(b_bakk_full.ptr_to([0]))
                                    txl.ptx[TMA_LD](
                                        TT[B_AKK].ptr_to(0, 0),
                                        txl.address_of(akk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb,
                                    )
                                lphase_end()
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    claim_publish(kk_ + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w10:
                st_hs = txl.PipelineState(1, phase=0)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    cb = chunk_base(seq)

                    fidx_s = txl.local_scalar("int32", init=seq * txl.int32(HV) + hv)
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            with txl.serial(nch) as n:
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                with txl.If(elected()), txl.Then():
                                    txl.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * txl.int32(HV) + hv
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_ST](
                                            txl.address_of(h_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            idx,
                                            TT[
                                                F_HS
                                                + st_hs.stage * txl.int32(4)
                                                + txl.int32((d0 // 64) * 2)
                                            ].ptr_to(0, 0),
                                        )
                                    txl.ptx[BULK_COMMIT]()
                                    txl.ptx[BULK_WAIT_READ](0)
                                    p_hs.empty.arrive(st_hs.stage)

                                    txl.ptx[BULK_WAIT](0)
                                    txl.ptx["fence.proxy.async.global"]()
                                    txl.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([fidx_s]),
                                        ep64 + txl.Cast("int64", n + txl.int32(1)),
                                    )
                                st_hs.advance()
                        with txl.Else():
                            with txl.serial(nch) as rn:
                                n = nch - txl.int32(1) - rn
                                par = bcyc & txl.int32(1)
                                MB["dhb_ready"].wait(0, par)
                                with txl.If(elected()), txl.Then():
                                    txl.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * txl.int32(HV) + hv
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_ST](
                                            txl.address_of(dh_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            idx,
                                            TT[B_DHB + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                        )
                                    txl.ptx[BULK_COMMIT]()
                                    txl.ptx[BULK_WAIT_READ](0)
                                    b_dhb_stored.arrive(0)
                                    txl.ptx[BULK_WAIT](0)
                                    txl.ptx["fence.proxy.async.global"]()
                                    txl.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([num_chains + fidx_s]),
                                        ep64 + txl.Cast("int64", rn + txl.int32(1)),
                                    )
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    with txl.If(elected()), txl.Then():
                        txl.ptx[BULK_WAIT](0)
                        txl.ptx["fence.proxy.async.global"]()

                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w11:
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                lane = txl.lane_id()
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with txl.If(is_fwd == txl.int32(0)), txl.Then():
                        with txl.serial(nch) as rn:
                            n = nch - txl.int32(1) - rn
                            par = bcyc & txl.int32(1)
                            rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                            b_baqk_full.wait(0, par)

                            for half in range(2):
                                diag = txl.alloc_local([4], "uint32")
                                dmat = lane >> txl.int32(3)
                                dblk = txl.int32(4 * half) + dmat
                                dptr = TT[B_AQK].ptr_to(
                                    dblk * txl.int32(8) + (lane & txl.int32(7)), dblk * txl.int32(8)
                                )
                                txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                    diag[0], diag[1], diag[2], diag[3], dptr
                                )
                                drow = lane >> txl.int32(2)
                                dcol = (lane & txl.int32(3)) * txl.int32(2)
                                dmask = txl.Select(
                                    dcol > drow,
                                    txl.uint32(0),
                                    txl.Select(
                                        dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)
                                    ),
                                )
                                for e in range(4):
                                    blk_row = txl.int32(8 * (4 * half + e)) + drow
                                    txl.assign(
                                        diag[e],
                                        txl.Select(blk_row < rows, diag[e] & dmask, txl.uint32(0)),
                                    )
                                txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                    dptr, diag[0], diag[1], diag[2], diag[3]
                                )
                            for r in range(2):
                                rowc = lane + txl.int32(32 * r)
                                for u in range(1, 8):
                                    with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                                        txl.ptx["st.shared.v4.b32"](
                                            TT[B_AQK].ptr_to(rowc, 8 * u),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                        )
                                with txl.If(rowc >= rows), txl.Then():
                                    for u in range(0, 8):
                                        txl.ptx["st.shared.v4.b32"](
                                            TT[B_AQK].ptr_to(rowc, 8 * u),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                        )
                            txl.ptx[FENCE_ASYNC]()
                            b_baqk_masked.arrive(0)
                            txl.assign(bcyc, bcyc + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            # All four auxiliary warps must synchronize before reallocating.
            txl.ptx.bar.sync(txl.uint32(7), txl.uint32(128))

        with auxg:
            with mma:
                tm = tmem_preamble()
                cyc = txl.local_scalar("int32", init=txl.int32(0))

                def mwait(nm):
                    MBG[nm].wait(0, cyc & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()
                bd = txl.alloc_local([1], "uint64")
                zq = txl.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)

                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )

                item = txl.local_scalar("int32", init=cur - num_streams)
                with txl.While(cur < total_work):
                    txl.assign(item, cur - num_streams)
                    retry_item = txl.local_scalar("bool", init=False)
                    with txl.serial(G) as gi:
                        par = cyc & txl.int32(1)
                        txl.ptx.ld.volatile.shared.s32(zq[0], txl.address_of(s_tmem[1]))
                        txl.cuda.tcgen05.encode_matrix_descriptor(
                            txl.address_of(bd[0]),
                            TT[zq[0]].ptr_to(0, 0),
                            ldo=Op.LBO_BASE,
                            sdo=SBO_UNITS,
                            swizzle=txl.SW128B.value,
                        )
                        akk_u = txl.local_scalar(
                            "uint64", init=txl.Cast("uint64", par) * txl.uint64(UNITS_PER_STAGE)
                        )
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)

                        b_h_free.wait(0, par ^ txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with txl.If(elected()), txl.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    txl.ptx[MMA_SS](
                                        txl.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        txl.uint32(ID_T),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.ptx.pred(0),
                                    )
                            TCG["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                        b_akk_masked.wait(par, (cyc >> 1) & txl.int32(1))
                        b_h_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TCG["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        b_dhb_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dv2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(
                                tm,
                                S3,
                                op_DHBmn,
                                op_T2k if False else Op(bd, T2, 128, 128, "mn"),
                                ID_128x64_TATB,
                                True,
                            )
                            TCG["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TCG["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TCG["dAs_done"].arrive(0)
                            mma_chain(
                                tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u
                            )
                            TCG["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with txl.If(elected()), txl.Then():
                            if HALF_DA_READOUT:
                                mma_chain(
                                    tm, S4 + (16 << 16), op_do_k128, op_T6mn, ID_64x64_TB, False
                                )
                            else:
                                mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TCG["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TCG["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TCG["X_done"].arrive(0)
                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            b_do_empty.arrive(0)
                            mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                        retry_bits = txl.local_scalar("uint16")
                        retry_lane = txl.lane_id()
                        retry_risk = txl.local_scalar("bool", init=False)
                        for retry_half in range(2):
                            retry_i = retry_lane + txl.int32(32 * retry_half)
                            txl.ptx.ld.shared.u16(retry_bits, TT[T5].ptr_to(retry_i, retry_i))
                            txl.assign(
                                retry_risk,
                                retry_risk
                                | ((retry_bits & txl.uint16(0x7FFF)) >= txl.uint16(0x4100)),
                            )
                        retry_mask = txl.local_scalar("uint32")
                        txl.ptx.vote_sync.ballot.b32(
                            retry_mask, txl.ptx.pred(retry_risk), txl.uint32(0xFFFFFFFF)
                        )
                        txl.assign(retry_item, retry_item | (retry_mask != txl.uint32(0)))
                        with txl.If(elected()), txl.Then():
                            TCG["dq2_done"].arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with txl.If(elected()), txl.Then():
                            mma_chain(
                                tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u
                            )
                            TCG["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dkt")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            TCG["dkt_done"].arrive(0)
                            TCG["chunk_done"].arrive(0)
                        mphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))
                    with txl.If((retry_lane == txl.int32(0)) & retry_item), txl.Then():
                        retry_slot = txl.local_scalar("int32")
                        txl.ptx["atom.global.add.s32"](
                            retry_slot, range_flags.ptr_to([range_entries]), txl.int32(1)
                        )
                        txl.ptx.st.global_.s32(
                            range_flags.ptr_to([range_entries + txl.int32(1) + retry_slot]), item
                        )
                        retry_old = txl.local_scalar("uint32")
                        txl.ptx["atom.global.or.b32"](
                            retry_old, range_flags.ptr_to([0]), txl.uint32(4)
                        )
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (q_map, k_map, v_map, eg_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                item = txl.local_scalar("int32", init=cur - num_streams)
                with txl.While(cur < total_work):
                    txl.assign(item, cur - num_streams)
                    c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    tok0 = txl.local_scalar("int32", init=bos32 + n * txl.int32(CHUNK))

                    lphase("lw-flags")
                    with txl.If(elected()), txl.Then():
                        tgt_f = txl.local_scalar(
                            "int64", init=ep64 + txl.Cast("int64", n + txl.int32(1))
                        )
                        tgt_b = txl.local_scalar("int64", init=ep64 + txl.Cast("int64", nch_i - n))
                        for gi_ in range(G):
                            fidx = seq * txl.int32(HV) + hq * txl.int32(G) + txl.int32(gi_)
                            flf = txl.local_scalar("int64", init=txl.int64(0))
                            txl.cuda.wait_until(
                                flf, flags.ptr_to([fidx]), flf >= tgt_f, scope="gpu"
                            )
                            flb = txl.local_scalar("int64", init=txl.int64(0))
                            txl.cuda.wait_until(
                                flb, flags.ptr_to([num_chains + fidx]), flb >= tgt_b, scope="gpu"
                            )

                        with txl.If(rows < txl.int32(CHUNK)), txl.Then():
                            tgt_1 = txl.local_scalar("int64", init=ep64 + txl.int64(1))
                            s2 = txl.local_scalar("int32", init=seq + txl.int32(1))
                            b2 = txl.local_scalar("int32", init=tok0 + rows)
                            with txl.While((s2 < num_seqs) & (b2 < tok0 + txl.int32(CHUNK))):
                                for gi_ in range(G):
                                    fl2 = txl.local_scalar("int64", init=txl.int64(0))
                                    txl.cuda.wait_until(
                                        fl2,
                                        flags.ptr_to(
                                            [
                                                s2 * txl.int32(HV)
                                                + hq * txl.int32(G)
                                                + txl.int32(gi_)
                                            ]
                                        ),
                                        fl2 >= tgt_1,
                                        scope="gpu",
                                    )
                                _, l2 = seq_len_of(s2)
                                txl.assign(b2, b2 + l2)
                                txl.assign(s2, s2 + txl.int32(1))
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    txl.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with txl.serial(G) as gi:
                        hv = txl.local_scalar("int32", init=hq * txl.int32(G) + gi)
                        par = cyc & txl.int32(1)
                        npar = par ^ txl.int32(1)
                        hidx = c * txl.int32(HV) + hv
                        lphase("lw-mid")
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with txl.If(elected()), txl.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_Q + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(q_map),
                                    txl.int32(d0),
                                    tok0,
                                    hq,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_K + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(k_map),
                                    txl.int32(d0),
                                    tok0,
                                    hq,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_V + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(v_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mb,
                                )
                        load_beta_lanes_g(bos, hv, n, rows, par)
                        b_in_full.arrive(0)
                        lphase("lw-chunk")
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            TCG["chunk_done"].wait(0, npar)
                        lphase("l-issue2")
                        with txl.If(elected()), txl.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = txl.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_G + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(eg_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mbe,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_do_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_DO + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(do_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mb,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_h_free.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_H + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(h_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_aqk_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_aqk_full.arrive(0, tx_count=A_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                TT[S_AQK].ptr_to(0, 0),
                                txl.address_of(aqk_map),
                                txl.int32(0),
                                tok0,
                                hv,
                                mb,
                            )
                        with txl.If(cyc > txl.int32(1)), txl.Then():
                            b_akk_empty.wait(par, ((cyc >> 1) & txl.int32(1)) ^ txl.int32(1))
                        with txl.If(elected()), txl.Then():
                            b_akk_full.arrive(par, tx_count=A_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            txl.ptx[TMA_LD](
                                TT[S_AKK + par].ptr_to(0, 0),
                                txl.address_of(akk_map),
                                txl.int32(0),
                                tok0,
                                hv,
                                mb,
                            )

                        lphase("lw-qkfree")
                        TCG["xT_done"].wait(0, par)
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            TCG["dk_done"].wait(0, npar)
                        lphase("l-dhb")
                        with txl.If(elected()), txl.Then():
                            b_dhb_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_dhb_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[DHB + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(dh_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )

                            with txl.If(gi + txl.int32(1) < txl.int32(G)):
                                with txl.Then():
                                    hvn = hv + txl.int32(1)
                                    for tmap in (v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(tmap), txl.int32(d0), tok0, hvn
                                            )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(aqk_map), txl.int32(0), tok0, hvn
                                    )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(akk_map), txl.int32(0), tok0, hvn
                                    )
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(h_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            hidx + txl.int32(1),
                                        )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(dh_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            hidx + txl.int32(1),
                                        )
                        lphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))
                    claim_publish(kk_ + txl.int32(1))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w10:
                g_masker(txl.int32(0))
            with w11:
                g_masker(txl.int32(1))
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                txl.Cast("uint32", txl.local_scalar("int32", init=tmem_preamble()[0])),
                txl.uint32(TMEM_COLS),
            )

        with txl.If(txl.thread_id() == txl.int32(0)), txl.Then():
            done = txl.local_scalar("int32")
            txl.ptx["atom.acq_rel.gpu.global.add.s32"](
                done, stream_counter.ptr_to([1]), txl.int32(1)
            )
            with txl.If(done == num_ctas - txl.int32(1)), txl.Then():
                txl.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([0]), txl.int32(0))
                txl.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([1]), txl.int32(0))

    txl.MBarrier._wait = _CUDA_MBAR_WAIT
    return kda_bwd_native_mega


AQK_BYTES = CHUNK * CHUNK * 2


def make_mega_kernel(HQ: int, HV: int, static_grid=None, item_only=False):
    txl.MBarrier._wait = _ptx_mbarrier_wait
    G = HV // HQ
    HALF_DA_READOUT = HQ < 96
    HALF_XY_READOUT = HQ < 96
    HQK64 = txl.int64(HQ * D)
    G = HV // HQ
    HQK = HQ * D
    HVK = HV * D
    HVK64 = txl.int64(HVK)

    F_KV, F_AKK, F_HS, F_G, F_KG, F_KBG, F_VB = 0, 4, 6, 10, 14, 18, 22

    B_QK, B_DO, B_G, B_AQK, B_AKK, B_T2, B_DHB, B_DV2 = 0, 4, 8, 12, 13, 14, 16, 20

    MAXSEQ = 64
    TM_H, TM_W0, TM_U0, TM_KT, TM_VT = 0, 128, 256, 384, 448
    TM_DH, TM_BW, TM_DV2, TM_QT, TM_KTB, TM_T1, TM_KB = 0, 128, 192, 256, 320, 384, 448

    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8
    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2
    EG_BYTES = CHUNK * D * 2
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2

    TM_ADQ, TM_ADK = 0, 64
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    TMEM_COLS = 512

    @txl.kernel(
        warps=12,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="num_ctas" if static_grid is None else static_grid,
    )
    def kda_bwd_mega(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        aqk: txl.gptr[txl.bf16],
        akk: txl.gptr[txl.bf16],
        g: txl.gptr[txl.f32],
        egcache: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        dht: txl.gptr[txl.f32],
        h0: txl.gptr[txl.f32],
        hsnap: txl.gptr[txl.bf16],
        dhsnap: txl.gptr[txl.bf16],
        cu_seqlens: txl.gptr[txl.i64],
        dq: txl.gptr[txl.f32],
        dk: txl.gptr[txl.f32],
        dv: txl.gptr[txl.bf16],
        db: txl.gptr[txl.f32],
        dg: txl.gptr[txl.f32],
        dh0: txl.gptr[txl.f32],
        stream_counter: txl.gptr[txl.i32],
        flags: txl.gptr[txl.i64],
        stream_tab: txl.gptr[txl.i32],
        item_tab: txl.gptr[txl.i32],
        seq_tab: txl.gptr[txl.i32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        g_map: txl.TensorMap,
        eg_map: txl.TensorMap,
        do_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        akk_map: txl.TensorMap,
        h_map: txl.TensorMap,
        dh_map: txl.TensorMap,
        scale: txl.f32,
        num_seqs: txl.i32,
        num_items: txl.i32,
        num_ctas: txl.i32,
        epoch: txl.i32,
        range_flags: txl.gptr[txl.i32],
        range_allowed: txl.i32,
        range_entries: txl.i32,
    ):
        range_bad = txl.local_scalar("uint32", init=txl.uint32(0))
        range_idx = txl.local_scalar("int32", init=txl.thread_id())
        with txl.While(range_idx < range_entries):
            range_part = txl.local_scalar("uint32")
            txl.ptx.ld.global_.u32(range_part, range_flags.ptr_to([range_idx]))
            txl.assign(range_bad, range_bad | range_part)
            txl.assign(range_idx, range_idx + txl.int32(384))
        range_count = txl.local_scalar("uint32")
        range_bad_pred = txl.local_scalar("bool", init=(range_bad & txl.uint32(2)) != txl.uint32(0))
        txl.ptx.bar.red.popc.u32(
            range_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(range_bad_pred)
        )
        range_safe = txl.local_scalar(
            "bool", init=(range_count == txl.uint32(0)) & (range_allowed != txl.int32(0))
        )
        diagonal_bit = 4 if item_only else 1
        diagonal_pred = txl.local_scalar(
            "bool", init=(range_bad & txl.uint32(diagonal_bit)) != txl.uint32(0)
        )
        diagonal_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            diagonal_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(diagonal_pred)
        )
        diagonal_needed = txl.local_scalar("bool", init=diagonal_count != txl.uint32(0))
        with txl.If(txl.Not(diagonal_needed)), txl.Then():
            txl.Return(txl.int32(0))
        for buf in (q, k, v, aqk, akk, g, egcache, do, hsnap, dhsnap):
            txl.keep_alive(buf.data)
        num_chains = num_seqs * txl.int32(HV)
        retry_count = txl.local_scalar("int32", init=num_items)
        if item_only:
            txl.ptx.ld.global_.s32(retry_count, range_flags.ptr_to([range_entries]))
        num_streams = txl.int32(0) if item_only else num_chains * txl.int32(2)
        total_work = retry_count if item_only else num_streams + num_items
        cta = txl.local_scalar("int32", init=txl.Cast("int32", txl.cta_id()))
        ITEM_RING = 4

        ep64 = txl.local_scalar("int64", init=txl.Cast("int64", epoch) * txl.int64(1 << 32))

        sp = txl.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=232)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=40)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        w10 = sp.role("w10", warps=[10], group=auxg)
        w11 = sp.role("w11", warps=[11], group=auxg)

        smem = txl.smem_pool()
        s_tmem = smem.alloc((4,), txl.i32, align=16)

        p_kv = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_akk1 = txl.Pipeline(smem, 2, full="tma", empty="tcgen05")
        p_tiles = txl.Pipeline(smem, 2, full="mbar", empty="tcgen05", init_full=256)
        p_hs = txl.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = txl.Pipeline(smem, 2, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = txl.Pipeline(smem, 1, full="tma", empty="mbar", init_full=33, init_empty=256)
        b_kvT_done = txl.TCGen05Bar(smem, 1)
        b_kvT_done.init(1)
        b_kv_read = txl.MBarrier(smem, 1)
        b_kv_read.init(256)

        p_qk = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        b_qkT_done = txl.TCGen05Bar(smem, 1)
        b_qkT_done.init(1)
        b_qk_read = txl.MBarrier(smem, 1)
        b_qk_read.init(256)
        b_g_full = txl.TMABar(smem, 1)
        b_g_full.init(33)
        b_g_free = txl.MBarrier(smem, 1)
        b_g_free.init(256)
        b_bdo_full = txl.TMABar(smem, 2)
        b_bdo_full.init(1)
        b_baqk_full = txl.TMABar(smem, 1)
        b_baqk_full.init(1)
        b_baqk_masked = txl.MBarrier(smem, 1)
        b_baqk_masked.init(32)
        b_baqk_empty = txl.TCGen05Bar(smem, 1)
        b_baqk_empty.init(1)
        b_bakk_full = txl.TMABar(smem, 1)
        b_bakk_full.init(1)
        b_bakk_empty = txl.TCGen05Bar(smem, 1)
        b_bakk_empty.init(1)
        b_dhb_stored = txl.MBarrier(smem, 1)
        b_dhb_stored.init(1)
        MB = {}
        for nm in ("prep_ready", "wT_ready", "dhb_ready", "dv2T_ready"):
            MB[nm] = txl.MBarrier(smem, 1)
            MB[nm].init(256)
        TC = {}
        for nm in ("W_done", "dv2_done", "dh_done"):
            TC[nm] = txl.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        b_in_full = txl.TMABar(smem, 1)
        b_in_full.init(33)
        b_eg_full = txl.TMABar(smem, 1)
        b_eg_full.init(1)
        b_mid_free = txl.MBarrier(smem, 1)
        b_mid_free.init(256)
        b_qk_free = txl.MBarrier(smem, 1)
        b_qk_free.init(256)
        b_do_full = txl.TMABar(smem, 1)
        b_do_full.init(1)
        b_h_full = txl.TMABar(smem, 1)
        b_h_full.init(1)
        b_dhb_full = txl.TMABar(smem, 1)
        b_dhb_full.init(1)
        b_aqk_full = txl.TMABar(smem, 1)
        b_aqk_full.init(1)
        b_akk_full = txl.TMABar(smem, 2)
        b_akk_full.init(1)
        b_do_empty = txl.TCGen05Bar(smem, 1)
        b_do_empty.init(1)
        b_h_free = txl.MBarrier(smem, 1)
        b_h_free.init(256)
        b_intra_free = txl.MBarrier(smem, 1)
        b_intra_free.init(256)
        b_stable_full = txl.MBarrier(smem, 1)
        b_stable_full.init(256)
        b_aqk_empty = txl.TCGen05Bar(smem, 1)
        b_aqk_empty.init(1)
        b_akk_empty = txl.TCGen05Bar(smem, 2)
        b_akk_empty.init(1)
        mbg_names = [
            "t_early",
            "zT_ready",
            "vnT_ready",
            "dv2T_ready",
            "dAqk_tile_ready",
            "dAm_ready",
            "X_ready",
            "intra_ready",
            "dv_epi_done",
        ]
        MBG = {}
        for nm in mbg_names:
            MBG[nm] = txl.MBarrier(smem, 1)
            MBG[nm].init(256)
        b_dg0_ready = txl.MBarrier(smem, 1)
        b_dg0_ready.init(256)
        b_aqk_masked = txl.MBarrier(smem, 1)
        b_aqk_masked.init(64)
        b_akk_masked = txl.MBarrier(smem, 2)
        b_akk_masked.init(64)
        tcg_names = [
            "Z_done",
            "Vn_done",
            "dv2_done",
            "dAqk_done",
            "dk_done",
            "dAs_done",
            "dvb_done",
            "X_done",
            "Y_done",
            "dq2_done",
            "dkt_done",
            "chunk_done",
            "xT_done",
        ]
        TCG = {}
        for nm in tcg_names:
            TCG[nm] = txl.TCGen05Bar(smem, 1)
            TCG[nm].init(1)

        TT = smem.alloc((27, 64, 64), txl.bf16, swizzle=txl.SW128B)
        s_beta1 = smem.alloc((2, CHUNK), txl.f32, align=16)
        s_bbeta = smem.alloc((2, CHUNK), txl.f32, align=16)

        s_beta = smem.alloc((2, 64), txl.f32, align=16)
        s_dgk = smem.alloc((2, 128), txl.f32, align=16)

        s_seq = smem.alloc((MAXSEQ, 4), txl.i32, align=16)

        s_work = smem.alloc((ITEM_RING,), txl.i32, align=16)
        b_work = txl.MBarrier(smem, ITEM_RING)
        b_work.init(1)
        s_ident = smem.alloc((256,), txl.bf16, align=128)

        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_tmem[1]), txl.int32(0))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(txl.thread_id() < txl.int32(256)), txl.Then():
            tid_i = txl.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & txl.int32(15)
            txl.ptx.st.shared.u16(
                s_ident.ptr_to(
                    [
                        (n_i >> 3) * txl.int32(128)
                        + (k_i >> 3) * txl.int32(64)
                        + (n_i & txl.int32(7)) * txl.int32(8)
                        + (k_i & txl.int32(7))
                    ]
                ),
                txl.Cast("uint16", txl.Select(n_i == k_i, txl.int32(0x3F80), txl.int32(0))),
            )
            txl.ptx[FENCE_ASYNC]()
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(s_tmem[0]), txl.uint32(TMEM_COLS)
            )
        with txl.If((txl.warp_id() == 0) & (num_seqs <= txl.int32(MAXSEQ))), txl.Then():
            lane0 = txl.lane_id()
            with txl.serial((num_seqs + txl.int32(31)) >> 5) as blk:
                i = blk * txl.int32(32) + lane0
                with txl.If(i < num_seqs), txl.Then():
                    st4 = txl.alloc_local([4], "int32")
                    txl.ptx["ld.global.nc.v4.s32"](
                        st4[0], st4[1], st4[2], st4[3], seq_tab.ptr_to([i * txl.int32(4)])
                    )
                    for j in range(4):
                        txl.ptx.st.shared.s32(txl.address_of(s_seq[i, j]), st4[j])
        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_work[0]), cta)
            b_work.arrive(0)
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def tmem_preamble():
            tmv = txl.alloc_local([1], "int32")
            txl.ptx.ld.volatile.shared.s32(tmv[0], txl.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = txl.alloc_local([1], "uint32")
            txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            def phase(name):
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.range_start(name))

            def phase_end():
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def bf16_bits_to_f32(u16val):
            return txl.reinterpret("float32", txl.Cast("uint32", u16val) << txl.uint32(16))

        def lo(w):
            return txl.reinterpret("float32", w << txl.uint32(16))

        def hi(w):
            return txl.reinterpret("float32", w & txl.uint32(0xFFFF0000))

        def seq_info(seq):
            """(bos, seq_len, nch) of a sequence from the SMEM table, or from cu_seqlens when it does not fit."""
            bos = txl.local_scalar("int64", init=txl.int64(0))
            seq_len = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(num_seqs <= txl.int32(MAXSEQ)):
                with txl.Then():
                    b32 = txl.local_scalar("int32")
                    txl.ptx.ld.shared.s32(b32, txl.address_of(s_seq[seq, 0]))
                    txl.assign(bos, txl.Cast("int64", b32))
                    txl.ptx.ld.shared.s32(seq_len, txl.address_of(s_seq[seq, 1]))
                with txl.Else():
                    cs = txl.alloc_local([2], "int64")
                    txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
                    txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + txl.int32(1)]))
                    txl.assign(bos, cs[0])
                    txl.assign(seq_len, txl.Cast("int32", cs[1] - cs[0]))
            nch = txl.local_scalar("int32", init=(seq_len + txl.int32(CHUNK - 1)) >> 6)
            return bos, seq_len, nch

        def stream_coords(s):
            """Stream rank s -> (is_fwd, seq, hv, hq, bos, seq_len, nch) from the host-built stream table."""
            sv = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(sv, stream_tab.ptr_to([s]))
            is_fwd = txl.local_scalar("int32", init=sv >> txl.int32(30))
            seq = txl.local_scalar("int32", init=(sv >> txl.int32(15)) & txl.int32(0x7FFF))
            hv = txl.local_scalar("int32", init=sv & txl.int32(0x7FFF))
            hq = txl.local_scalar("int32", init=hv // txl.int32(G))
            bos, seq_len, nch = seq_info(seq)
            return is_fwd, seq, hv, hq, bos, seq_len, nch

        def chunk_base(seq):
            cb = txl.local_scalar("int32", init=txl.int32(0))
            with txl.If(num_seqs <= txl.int32(MAXSEQ)):
                with txl.Then():
                    txl.ptx.ld.shared.s32(cb, txl.address_of(s_seq[seq, 3]))
                with txl.Else():
                    with txl.serial(seq) as i:
                        cs = txl.alloc_local([2], "int64")
                        txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                        txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                        txl.assign(
                            cb,
                            cb + ((txl.Cast("int32", cs[1] - cs[0]) + txl.int32(CHUNK - 1)) >> 6),
                        )
            return cb

        def chunk_rows(seq_len, n):
            return txl.min(txl.int32(CHUNK), seq_len - n * txl.int32(CHUNK))

        def load_beta_lanes(dst_ptr_fn, bos, hv, n, rows):
            """Loader warp: every lane fetches beta for tokens lane and lane+32 of the chunk."""
            lane = txl.lane_id()
            for j in range(2):
                t = lane + txl.int32(32 * j)
                tokc = bos + txl.Cast(
                    "int64", n * txl.int32(CHUNK) + txl.min(t, rows - txl.int32(1))
                )
                u = txl.local_scalar("uint16")
                txl.ptx.ld.global_.nc.u16(
                    u, beta.ptr_to([tokc * txl.int64(HV) + txl.Cast("int64", hv)])
                )
                val = txl.Select(t < rows, bf16_bits_to_f32(u), txl.float32(0.0))
                txl.ptx.st.shared.f32(dst_ptr_fn(t), val)

        def seq_len_of(sq):
            cs = txl.alloc_local([2], "int64")
            txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([sq]))
            txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([sq + 1]))
            return cs[0], txl.Cast("int32", cs[1] - cs[0])

        def item_coords(item):
            """Item index -> (c, hq, seq, n, bos, rows, nch).

            (seq, n) comes from the host-built table, which orders chunks by their predicted
            recurrence readiness (both streams of the sequence have published the chunk); the
            qk-head index is the fastest-varying component.  An item-only retry first maps its
            compact queue rank back to the native launch's global item index.
            """
            item_index = txl.local_scalar("int32", init=item)
            if item_only:
                txl.ptx.ld.global_.s32(
                    item_index, range_flags.ptr_to([range_entries + txl.int32(1) + item])
                )
            pos = txl.local_scalar("int32", init=item_index // txl.int32(HQ))
            hq = txl.local_scalar("int32", init=item_index - pos * txl.int32(HQ))
            iv = txl.local_scalar("int32")
            txl.ptx.ld.global_.nc.s32(iv, item_tab.ptr_to([pos]))
            seq = txl.local_scalar("int32", init=iv >> txl.int32(16))
            n = txl.local_scalar("int32", init=iv & txl.int32(0xFFFF))
            bos, seq_len, nch = seq_info(seq)
            cb = chunk_base(seq)
            c = txl.local_scalar("int32", init=cb + n)
            rows = txl.local_scalar(
                "int32", init=txl.min(txl.int32(CHUNK), seq_len - n * txl.int32(CHUNK))
            )
            return c, hq, seq, n, bos, rows, nch

        def load_beta_lanes_g(bos, hv, n, rows, slot):
            lane = txl.lane_id()
            for j in range(2):
                t = lane + txl.int32(32 * j)
                tokc = bos + txl.Cast(
                    "int64", n * txl.int32(CHUNK) + txl.min(t, rows - txl.int32(1))
                )
                u = txl.local_scalar("uint16")
                txl.ptx.ld.global_.nc.u16(
                    u, beta.ptr_to([tokc * txl.int64(HV) + txl.Cast("int64", hv)])
                )
                val = txl.Select(
                    t < rows,
                    txl.reinterpret("float32", txl.Cast("uint32", u) << txl.uint32(16)),
                    txl.float32(0.0),
                )
                txl.ptx.st.shared.f32(txl.address_of(s_beta[slot, t]), val)

        def work_wait(j):
            """The j-th work unit of this CTA (published by the loader); >= total_work means done."""
            slot = j % txl.int32(ITEM_RING)
            b_work.wait(slot, (j // txl.int32(ITEM_RING)) & txl.int32(1))
            v_ = txl.local_scalar("int32")
            txl.ptx.ld.shared.s32(v_, txl.address_of(s_work[slot]))
            return v_

        def claim_publish(j):
            """Loader: claim work unit j for the CTA and publish it in the ring."""
            slot = j % txl.int32(ITEM_RING)
            with txl.If(elected()), txl.Then():
                nxt = txl.local_scalar("int32")
                txl.ptx["atom.acq_rel.gpu.global.add.s32"](
                    nxt, stream_counter.ptr_to([0]), txl.int32(1)
                )
                txl.ptx.st.shared.s32(txl.address_of(s_work[slot]), nxt + num_ctas)
                b_work.arrive(slot)

        kk_ = txl.local_scalar("int32", init=txl.int32(0))
        cur = txl.local_scalar("int32", init=work_wait(kk_))

        def g_masker(MROW):
            cyc = txl.local_scalar("int32", init=txl.int32(0))
            lane = txl.lane_id()
            rowc = txl.local_scalar("int32", init=MROW * txl.int32(32) + lane)
            item = txl.local_scalar("int32", init=cur - num_streams)
            with txl.While(cur < total_work):
                txl.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                with txl.serial(G) as gi:
                    par = cyc & txl.int32(1)
                    b_aqk_full.wait(0, par)
                    diag = txl.alloc_local([4], "uint32")
                    dmat = lane >> txl.int32(3)
                    dblk = MROW * txl.int32(4) + dmat
                    dptr = TT[S_AQK].ptr_to(
                        dblk * txl.int32(8) + (lane & txl.int32(7)), dblk * txl.int32(8)
                    )
                    txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                        diag[0], diag[1], diag[2], diag[3], dptr
                    )
                    drow = lane >> txl.int32(2)
                    dcol = (lane & txl.int32(3)) * txl.int32(2)
                    dmask = txl.Select(
                        dcol > drow,
                        txl.uint32(0),
                        txl.Select(dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)),
                    )
                    for e in range(4):
                        blk_row = (MROW * txl.int32(4) + txl.int32(e)) * txl.int32(8) + drow
                        txl.assign(
                            diag[e], txl.Select(blk_row < rows, diag[e] & dmask, txl.uint32(0))
                        )
                    txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                        dptr, diag[0], diag[1], diag[2], diag[3]
                    )
                    for u in range(1, 8):
                        with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AQK].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    with txl.If(rowc >= rows), txl.Then():
                        for u in range(8):
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AQK].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    txl.ptx[FENCE_ASYNC]()
                    b_aqk_masked.arrive(0)

                    b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                    with txl.If(rowc >= rows), txl.Then():
                        for u in range(8):
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_AKK + par].ptr_to(rowc, 8 * u),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                                txl.uint32(0),
                            )
                    txl.ptx[FENCE_ASYNC]()
                    b_akk_masked.arrive(par)
                    txl.assign(cyc, cyc + txl.int32(1))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

        with cg:
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            xg = txl.local_scalar("int32", init=x >> 5)
            xgc = txl.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            st_te = txl.PipelineState(2, phase=0)
            st_hs = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(1, phase=0)
            st_w = txl.PipelineState(2, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            fkv = txl.local_scalar("int32", init=txl.int32(0))
            bqk = txl.local_scalar("int32", init=txl.int32(0))
            st_bg = txl.PipelineState(1, phase=0)
            bcyc = txl.local_scalar("int32", init=txl.int32(0))

            gv = txl.alloc_local([32], "float32")
            kk = txl.alloc_local([32], "float32")
            vv = txl.alloc_local([32], "float32")
            bb = txl.alloc_local([32], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            eg = txl.local_scalar("float32")
            egng = txl.local_scalar("float32")
            bu = txl.local_scalar("uint16")
            ku = txl.local_scalar("uint16")
            vu = txl.local_scalar("uint16")
            qu = txl.local_scalar("uint16")
            phase, phase_end = make_phaser()

            def fwd_body(seq, hv, hq, bos, seq_len, nch):
                """Forward state recurrence, software-pipelined: while the tensor core runs chunk n's
                Vn / state-update MMAs, the compute warps prepare chunk n+1's tiles (from K/V transposed
                into TMEM by the MMA warp) and read chunk n+2's gate."""
                hv64 = txl.Cast("int64", hv)
                gcol = txl.local_scalar("int64", init=hv64 * txl.int64(D) + x64)
                gn_c = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_1 = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_2 = txl.local_scalar("float32", init=txl.float32(0.0))
                bpair = txl.alloc_local([2], "float32")

                def f_g(m, gn_dst):
                    """g of chunk m (32 rows of this channel) into gv; its last valid row into gn_dst."""
                    rows = chunk_rows(seq_len, m)
                    phase("fw-g")
                    p_g.full.wait(0, st_g.phase)
                    phase("f-g")
                    gst = txl.local_scalar("int32", init=F_G + xg)
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(0)
                    st_g.advance()

                def f_tiles(m, gn_m, u_lo, u_hi):
                    """K/V of chunk m from their TMEM transposes (first half only), then token blocks
                    [u_lo, u_hi) of its kg / kbg / vb tiles into set m & 1 (the last half publishes)."""
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, m))
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", m * txl.int32(CHUNK))
                    )
                    sset = txl.local_scalar("int32", init=m & txl.int32(1))
                    if u_lo == 0:
                        phase("fw-kv")
                        b_kvT_done.wait(0, fkv & txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        phase("f-kv")
                        txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                        txl.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_VT + wg * 32))
                        txl.ptx[WAIT_LD]()
                        txl.ptx[TC_FENCE_BEFORE]()
                        b_kv_read.arrive(0)
                        txl.assign(fkv, fkv + txl.int32(1))
                    phase("f-tiles")
                    kg_t = txl.local_scalar("int32", init=F_KG + sset * txl.int32(2) + xs)
                    kbg_t = txl.local_scalar("int32", init=F_KBG + sset * txl.int32(2) + xs)
                    vb_t = txl.local_scalar("int32", init=F_VB + sset * txl.int32(2) + xs)
                    for u in range(u_lo, u_hi):
                        wkg = txl.alloc_local([4], "uint32")
                        wkbg = txl.alloc_local([4], "uint32")
                        wvb = txl.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            valid0 = row0 + txl.int32(i) < rows
                            valid1 = row0 + txl.int32(i + 1) < rows
                            m0 = txl.Select(valid0, txl.float32(1.0), txl.float32(0.0))
                            m1 = txl.Select(valid1, txl.float32(1.0), txl.float32(0.0))
                            eg0 = txl.local_scalar("float32")
                            eg1 = txl.local_scalar("float32")
                            en0 = txl.local_scalar("float32")
                            en1 = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            txl.ptx.ex2.approx.ftz.f32(en0, gn_m - gv[i])
                            txl.ptx.ex2.approx.ftz.f32(en1, gn_m - gv[i + 1])
                            bu0 = txl.local_scalar("uint16")
                            bu1 = txl.local_scalar("uint16")
                            txl.ptx.cvt.rn.bf16.f32(bu0, eg0)
                            txl.ptx.cvt.rn.bf16.f32(bu1, eg1)
                            egidx0 = (tok0 + txl.Cast("int64", row0 + txl.int32(i))) * HVK64 + gcol
                            egidx1 = (
                                tok0 + txl.Cast("int64", row0 + txl.int32(i + 1))
                            ) * HVK64 + gcol
                            with txl.If(valid0), txl.Then():
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    egcache.ptr_to([egidx0]), bu0
                                )
                            with txl.If(valid1), txl.Then():
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    egcache.ptr_to([egidx1]), bu1
                                )
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta1[sset, row0 + i])
                            )
                            pair0 = txl.local_scalar("uint64")
                            pair1 = txl.local_scalar("uint64")
                            pair2 = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                pair0,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(en0, en1),
                            )
                            txl.ptx["mul.rn.f32x2"](pair0, pair0, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair1,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            txl.ptx["mul.rn.f32x2"](pair1, pair1, txl.cuda.make_float2(eg0, eg1))
                            txl.ptx["mul.rn.f32x2"](
                                pair2,
                                txl.cuda.make_float2(vv[i], vv[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            pack_bf16x2(wkg[p], txl.cuda.float2_x(pair0), txl.cuda.float2_y(pair0))
                            pack_bf16x2(wkbg[p], txl.cuda.float2_x(pair1), txl.cuda.float2_y(pair1))
                            pack_bf16x2(wvb[p], txl.cuda.float2_x(pair2), txl.cuda.float2_y(pair2))
                        col = row0 + 8 * u
                        txl.ptx["st.shared.v4.b32"](
                            TT[kg_t].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[kbg_t].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[vb_t].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3]
                        )
                    if u_hi == 4:
                        txl.ptx[FENCE_ASYNC]()

                        txl.ptx["fence.proxy.async.global"]()
                        p_tiles.full.arrive(sset)

                f_g(txl.int32(0), gn_c)
                f_tiles(txl.int32(0), gn_c, 0, 4)
                with txl.If(nch > txl.int32(1)), txl.Then():
                    f_g(txl.int32(1), gn_1)
                with txl.serial(nch) as n:
                    sn = txl.local_scalar("int32", init=n & txl.int32(1))
                    txl.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("fw-hupd")

                    with txl.If(n > txl.int32(0)), txl.Then():
                        p_tiles.empty.wait(st_te.stage, st_te.phase)
                        st_te.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("fw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("f-hdecay")
                    hc0 = wg * 64
                    hsst = txl.local_scalar("int32", init=F_HS + wg * txl.int32(2) + xs)
                    with txl.If(n == txl.int32(0)):
                        with txl.Then():
                            h0base = (
                                (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                            ) * txl.int64(D) + txl.Cast("int64", hc0)
                            for m8 in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    h0.ptr_to([h0base + txl.int64(8 * m8)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[hsst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    for p in range(32):
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair,
                            txl.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                            txl.cuda.make_float2(egn, egn),
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("fw-W")
                    p_w.full.wait(st_w.stage, st_w.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("f-wT")
                    txl.ptx[TC_LD32](
                        *(acc[i] for i in range(32)), tmem_at(TM_W0 + sn * 64 + wg * 32)
                    )
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    wt_t = txl.local_scalar("int32", init=F_KBG + sn * txl.int32(2) + xs)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[wt_t].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(st_w.stage)
                    st_w.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        f_tiles(n + txl.int32(1), gn_1, 0, 2)
                    phase("fw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("f-vnT")
                    txl.ptx[TC_LD32](
                        *(acc[i] for i in range(32)), tmem_at(TM_U0 + sn * 64 + wg * 32)
                    )
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    vn_t = txl.local_scalar("int32", init=F_VB + sn * txl.int32(2) + xs)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[vn_t].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)
                    with txl.If(elected()), txl.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        f_tiles(n + txl.int32(1), gn_1, 2, 4)
                    with txl.If(n + txl.int32(2) < nch), txl.Then():
                        f_g(n + txl.int32(2), gn_2)
                    txl.assign(gn_c, gn_1)
                    txl.assign(gn_1, gn_2)
                    phase_end()

                p_tiles.empty.wait(st_te.stage, st_te.phase)
                st_te.advance()
                txl.ptx[TC_FENCE_AFTER]()

            def bwd_body(seq, hv, hq, bos, seq_len, nch):
                """Backward state-gradient recurrence, software-pipelined: chunk n-1's operand
                preparation (q^T/k^T from TMEM transposes, T1/kbg into TMEM, T2 into shared memory)
                overlaps chunk n's dv2 and dh MMAs."""
                hv64 = txl.Cast("int64", hv)
                gn_c = txl.local_scalar("float32", init=txl.float32(0.0))
                gn_1 = txl.local_scalar("float32", init=txl.float32(0.0))
                bpair = txl.alloc_local([2], "float32")

                def b_prep(m, gn_dst):
                    """g of chunk m into gv (its last valid row into gn_dst), then q^T / k^T from TMEM."""
                    rows = chunk_rows(seq_len, m)
                    phase("bw-in")
                    b_g_full.wait(0, st_bg.phase)
                    phase("b-prep")
                    gst = txl.local_scalar("int32", init=B_G + xg)
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn_dst, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx[FENCE_ASYNC]()
                    b_g_free.arrive(0)
                    st_bg.advance()
                    phase("bw-qk")
                    b_qkT_done.wait(0, bqk & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("b-qk")
                    txl.ptx[TC_LD32](*(vv[i] for i in range(32)), tmem_at(TM_QT + wg * 32))
                    txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KTB + wg * 32))
                    txl.ptx[WAIT_LD]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    b_qk_read.arrive(0)
                    txl.assign(bqk, bqk + txl.int32(1))

                def b_prep2(m, sset, bslot, gn):
                    """T1 (TMEM set sset), T2 (shared) and kbg (TMEM set sset) of chunk m."""
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, m))
                    phase("b-prep2")
                    offset = txl.local_scalar(
                        "float32", init=txl.Select(_state_needs_stable(gn), gn, txl.float32(0.0))
                    )
                    w1 = txl.alloc_local([16], "uint32")
                    w3 = txl.alloc_local([16], "uint32")
                    for u in range(4):
                        w2 = txl.alloc_local([4], "uint32")
                        for p in range(4):
                            i = 8 * u + 2 * p
                            m0 = txl.Select(
                                row0 + txl.int32(i) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            m1 = txl.Select(
                                row0 + txl.int32(i + 1) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            eg0 = txl.local_scalar("float32")
                            eg1 = txl.local_scalar("float32")
                            en0 = txl.local_scalar("float32")
                            en1 = txl.local_scalar("float32")
                            txl.ptx.ex2.approx.ftz.f32(eg0, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(eg1, gv[i + 1])
                            txl.ptx.ex2.approx.ftz.f32(en0, offset - gv[i])
                            txl.ptx.ex2.approx.ftz.f32(en1, offset - gv[i + 1])
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_bbeta[bslot, row0 + i])
                            )
                            pair0 = txl.local_scalar("uint64")
                            pair1 = txl.local_scalar("uint64")
                            pair2 = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                pair0,
                                txl.cuda.make_float2(vv[i], vv[i + 1]),
                                txl.cuda.make_float2(eg0, eg1),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                pair0, pair0, txl.cuda.make_float2(scale, scale)
                            )
                            txl.ptx["mul.rn.f32x2"](pair0, pair0, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair1,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(en0, en1),
                            )
                            txl.ptx["mul.rn.f32x2"](pair1, pair1, txl.cuda.make_float2(m0, m1))
                            txl.ptx["mul.rn.f32x2"](
                                pair2,
                                txl.cuda.make_float2(kk[i], kk[i + 1]),
                                txl.cuda.make_float2(bpair[0], bpair[1]),
                            )
                            txl.ptx["mul.rn.f32x2"](pair2, pair2, txl.cuda.make_float2(eg0, eg1))
                            pack_bf16x2(
                                w1[4 * u + p], txl.cuda.float2_x(pair0), txl.cuda.float2_y(pair0)
                            )
                            pack_bf16x2(w2[p], txl.cuda.float2_x(pair1), txl.cuda.float2_y(pair1))
                            pack_bf16x2(
                                w3[4 * u + p], txl.cuda.float2_x(pair2), txl.cuda.float2_y(pair2)
                            )
                        txl.ptx["st.shared.v4.b32"](
                            TT[B_T2 + xs].ptr_to(xr, row0 + 8 * u), w2[0], w2[1], w2[2], w2[3]
                        )
                    txl.ptx[TC_ST16](
                        tmem_at(TM_T1 + sset * 32 + wg * 16), *(w1[j] for j in range(16))
                    )
                    txl.ptx[TC_ST16](
                        tmem_at(TM_KB + sset * 32 + wg * 16), *(w3[j] for j in range(16))
                    )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["prep_ready"].arrive(0)

                b_prep(nch - txl.int32(1), gn_c)
                b_prep2(nch - txl.int32(1), bcyc & txl.int32(1), bcyc & txl.int32(1), gn_c)
                with txl.serial(nch) as rn:
                    n = nch - txl.int32(1) - rn
                    par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                    rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                    txl.ptx.ex2.approx.ftz.f32(egn, gn_c)
                    phase("bw-dh")
                    with txl.If(rn > txl.int32(0)), txl.Then():
                        TC["dh_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("b-dhb")
                    with txl.If(rn == txl.int32(0)):
                        with txl.Then():
                            dbase = (
                                (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                            ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                            for m8 in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m8 + i] for i in range(8)),
                                    dht.ptr_to([dbase + txl.int64(8 * m8)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    # Strong channels publish unscaled DH for the bounded
                    # K * exp2(g_end - g_i) contraction; mild channels retain
                    # the scaled snapshot shared with the reciprocal fast path.
                    snapshot_raw = txl.local_scalar("bool", init=_state_needs_stable(gn_c))
                    for p in range(32):
                        raw0 = txl.local_scalar("float32", init=acc[2 * p])
                        raw1 = txl.local_scalar("float32", init=acc[2 * p + 1])
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair, txl.cuda.make_float2(raw0, raw1), txl.cuda.make_float2(egn, egn)
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                        pack_bf16x2(
                            wds[p],
                            txl.Select(snapshot_raw, raw0, acc[2 * p]),
                            txl.Select(snapshot_raw, raw1, acc[2 * p + 1]),
                        )
                    txl.ptx[TC_ST32](tmem_at(TM_DH + wg * 64), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](
                        tmem_at(TM_DH + wg * 64 + 32), *(acc[32 + i] for i in range(32))
                    )
                    phase("bw-stored")
                    with txl.If(rn > txl.int32(0)), txl.Then():
                        b_dhb_stored.wait(0, par ^ txl.int32(1))
                    phase("b-dhb2")
                    dhst = txl.local_scalar("int32", init=B_DHB + wg * txl.int32(2) + xs)
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[dhst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["dhb_ready"].arrive(0)
                    phase("bw-W")
                    TC["W_done"].wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("b-wT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_BW + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])

                    txl.ptx[TC_ST16](
                        tmem_at(TM_KB + par * 32 + wg * 16), *(wds[j] for j in range(16))
                    )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    MB["wT_ready"].arrive(0)
                    with txl.If(rn + txl.int32(1) < nch), txl.Then():
                        b_prep(n - txl.int32(1), gn_1)
                    phase("bw-dv2")
                    TC["dv2_done"].wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("b-dv2T")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DV2 + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        i = 2 * p
                        m0 = txl.Select(row0 + txl.int32(i) < rows, acc[i], txl.float32(0.0))
                        m1 = txl.Select(
                            row0 + txl.int32(i + 1) < rows, acc[i + 1], txl.float32(0.0)
                        )
                        pack_bf16x2(wds[p], m0, m1)
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[B_DV2 + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    MB["dv2T_ready"].arrive(0)
                    with txl.If(rn + txl.int32(1) < nch), txl.Then():
                        b_prep2(n - txl.int32(1), par ^ txl.int32(1), par ^ txl.int32(1), gn_1)
                    txl.assign(gn_c, gn_1)
                    phase_end()
                    txl.assign(bcyc, bcyc + txl.int32(1))
                TC["dh_done"].wait(0, (bcyc & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_DH + wg * 64))
                txl.ptx[TC_LD32](*(acc[32 + i] for i in range(32)), tmem_at(TM_DH + wg * 64 + 32))
                txl.ptx[WAIT_LD]()
                obase = (
                    (txl.Cast("int64", seq) * txl.int64(HV) + hv64) * txl.int64(D) + x64
                ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                for m8 in range(8):
                    txl.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + txl.int64(8 * m8)]),
                        *(acc[8 * m8 + i] for i in range(8)),
                    )
                b_dhb_stored.wait(0, (bcyc & txl.int32(1)) ^ txl.int32(1))

            with txl.While(cur < num_streams):
                is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                with txl.If(is_fwd == txl.int32(1)):
                    with txl.Then():
                        fwd_body(seq, hv, hq, bos, seq_len, nch)
                    with txl.Else():
                        bwd_body(seq, hv, hq, bos, seq_len, nch)

                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

            # Converge even when this CTA starts with an item and skips streams.
            txl.ptx.bar.sync(txl.uint32(6), txl.uint32(256))

        with cg:
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            phalf = txl.local_scalar("int32", init=x & 1)
            pcol = txl.local_scalar("int32", init=x & ~1)
            prow0 = txl.local_scalar("int32", init=row0 + phalf * 16)
            ps = txl.local_scalar("int32", init=pcol >> 6)
            pr = txl.local_scalar("int32", init=pcol & 63)
            cyc = txl.local_scalar("int32", init=txl.int32(0))

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                txl.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                txl.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                txl.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st8(col, regs, base=0):
                txl.ptx[TC_ST8](tmem_at(col), *(regs[base + i] for i in range(8)))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                for u in range(nunits):
                    txl.ptx["st.shared.v4.b32"](
                        TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                        words[wbase + 4 * u],
                        words[wbase + 4 * u + 1],
                        words[wbase + 4 * u + 2],
                        words[wbase + 4 * u + 3],
                    )

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            def bar_wg():
                txl.ptx.bar.sync(txl.uint32(2) + txl.Cast("uint32", wg), txl.uint32(128))

            def twait(nm):
                TCG[nm].wait(0, cyc & txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()

                txl.ptx[FENCE_ASYNC]()
                txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))

            def marrive(nm):
                txl.ptx[TC_FENCE_BEFORE]()
                MBG[nm].arrive(0)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o],
                            frag[o + 1],
                            frag[o + 2],
                            frag[o + 3],
                            tile.m8n8x4(
                                row0 + txl.int32(16 * rb), col0 + txl.int32(16 * cb_), lane
                            ),
                        )

            def store_transpose_frag(base, frag):
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                mm = lane >> txl.int32(3)
                jj = lane & txl.int32(7)
                for rb in range(2):
                    for cb_ in range(2):
                        o = 4 * (2 * rb + cb_)
                        ptr = tile.ptr_to(
                            col0 + txl.int32(16 * cb_) + (mm >> txl.int32(1)) * txl.int32(8) + jj,
                            row0 + txl.int32(16 * rb) + (mm & txl.int32(1)) * txl.int32(8),
                        )
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )

            egA = txl.alloc_local([16], "float32")
            egB = txl.alloc_local([16], "float32")
            egcw = txl.alloc_local([16], "uint64")
            t4 = txl.alloc_local([4], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            dgv = txl.alloc_local([32], "float32")
            oq = txl.alloc_local([8], "float32")
            ok8 = txl.alloc_local([8], "float32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            dgk = txl.local_scalar("float32")
            dgk_k = txl.local_scalar("float32")
            t0 = txl.local_scalar("float32")
            t1 = txl.local_scalar("float32")
            u16 = txl.local_scalar("uint16")

            def rcp(dst, val):
                txl.ptx.rcp.approx.ftz.f32(dst, txl.Select(strong, txl.float32(1.0), val))
                txl.assign(dst, txl.Select(strong, txl.float32(0.0), dst))

            def s_beta_row(c):
                b = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b, txl.address_of(s_beta[cyc & txl.int32(1), c]))
                return b

            phase, phase_end = make_phaser()
            item = txl.local_scalar("int32", init=cur - num_streams)
            with txl.While(cur < total_work):
                txl.assign(item, cur - num_streams)
                c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                hq64 = txl.Cast("int64", hq)
                tok0 = txl.local_scalar("int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK)))
                last = txl.local_scalar("int32", init=rows - txl.int32(1))
                xq_base = txl.local_scalar(
                    "int64",
                    init=(tok0 + txl.Cast("int64", row0)) * HQK64 + hq64 * txl.int64(D) + x64,
                )
                with txl.serial(G) as gi:
                    hv = txl.local_scalar("int32", init=hq * txl.int32(G) + gi)
                    hv64 = txl.Cast("int64", hv)
                    par = cyc & txl.int32(1)
                    x_base = txl.local_scalar(
                        "int64",
                        init=(tok0 + txl.Cast("int64", row0)) * HVK64 + hv64 * txl.int64(D) + x64,
                    )

                    gn = txl.local_scalar("float32", init=txl.float32(0.0))
                    state_strong = txl.local_scalar("bool")

                    def state_decay(dst, i, cached_gate):
                        with txl.If(state_strong):
                            with txl.Then():
                                gi_value = txl.local_scalar("float32")
                                ti = txl.min(row0 + txl.int32(i), last)
                                txl.ptx.ld.global_.nc.f32(
                                    gi_value,
                                    g.ptr_to(
                                        [
                                            (tok0 + txl.Cast("int64", ti)) * HVK64
                                            + hv64 * txl.int64(D)
                                            + x64
                                        ]
                                    ),
                                )
                                txl.ptx.ex2.approx.ftz.f32(dst, gn - gi_value)
                            with txl.Else():
                                state_cached = txl.local_scalar("uint32")
                                pack_bf16x2(state_cached, cached_gate, cached_gate)
                                txl.ptx.rcp.approx.ftz.f32(dst, lo(state_cached))

                    phase("w-in")
                    b_in_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")

                    xf = txl.alloc_local([32], "float32")
                    egf = txl.alloc_local([32], "float32")
                    qw = txl.alloc_local([16], "uint32")
                    kw = txl.alloc_local([16], "uint32")
                    t3w = txl.alloc_local([16], "uint32")
                    qc = txl.alloc_local([16], "uint32")
                    kc = txl.alloc_local([16], "uint32")
                    vc = txl.alloc_local([16], "uint32")
                    prep0 = txl.local_scalar("uint64")
                    prep1 = txl.local_scalar("uint64")
                    scale_pair = txl.local_scalar("uint64", init=txl.cuda.make_float2(scale, scale))
                    bpair = txl.alloc_local([2], "float32")

                    txl.ptx[TC_LD4](t4[0], t4[1], t4[2], t4[3], tmem_at(S1 + ((last >> 2) << 2)))
                    ld32(xf, S2 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    lq = last & txl.int32(3)
                    txl.assign(
                        egn,
                        txl.Select(
                            lq == txl.int32(0),
                            t4[0],
                            txl.Select(
                                lq == txl.int32(1),
                                t4[1],
                                txl.Select(lq == txl.int32(2), t4[2], t4[3]),
                            ),
                        ),
                    )
                    txl.assign(state_strong, egn < txl.float32(0.0625))
                    with txl.If(state_strong), txl.Then():
                        txl.ptx.ld.global_.nc.f32(
                            gn,
                            g.ptr_to(
                                [
                                    (tok0 + txl.Cast("int64", last)) * HVK64
                                    + hv64 * txl.int64(D)
                                    + x64
                                ]
                            ),
                        )
                    for half in range(2):
                        vb32 = txl.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[par, row0 + i])
                            )
                            txl.assign(vb32[2 * p], xf[i] * bpair[0])
                            txl.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        txl.ptx[TC_ST16](
                            tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16))
                        )
                    txl.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    txl.ptx[WAIT_LD]()

                    for i in range(32):
                        txl.assign(
                            egf[i], txl.Select(row0 + txl.int32(i) < rows, egf[i], txl.float32(1.0))
                        )
                    for i in range(16):
                        m0 = txl.Select(
                            row0 + txl.int32(2 * i) < rows, txl.float32(1.0), txl.float32(0.0)
                        )
                        m1 = txl.Select(
                            row0 + txl.int32(2 * i + 1) < rows, txl.float32(1.0), txl.float32(0.0)
                        )
                        txl.ptx["mul.rn.f32x2"](
                            prep0,
                            txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                            txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                        )
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, txl.cuda.make_float2(m0, m1))
                        pack_bf16x2(qw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        txl.assign(egcw[i], txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))
                    for half in range(2):
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)),
                            tmem_at(S4 + wg * 32 + 16 * half),
                        )
                        txl.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            m0 = txl.Select(
                                row0 + txl.int32(2 * i) < rows, txl.float32(1.0), txl.float32(0.0)
                            )
                            m1 = txl.Select(
                                row0 + txl.int32(2 * i + 1) < rows,
                                txl.float32(1.0),
                                txl.float32(0.0),
                            )
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[par, row0 + 2 * i])
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                t0, txl.Select(state_strong, txl.float32(1.0), egf[2 * i])
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                t1, txl.Select(state_strong, txl.float32(1.0), egf[2 * i + 1])
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep0,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(t0, t1),
                            )
                            txl.ptx["mul.rn.f32x2"](prep0, prep0, txl.cuda.make_float2(m0, m1))
                            pack_bf16x2(kw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                            txl.ptx["mul.rn.f32x2"](
                                prep1,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep1, prep1, txl.cuda.make_float2(bpair[0], bpair[1])
                            )
                            txl.ptx["mul.rn.f32x2"](prep1, prep1, txl.cuda.make_float2(m0, m1))
                            pack_bf16x2(t3w[i], txl.cuda.float2_x(prep1), txl.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TCG["chunk_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    with txl.If(state_strong), txl.Then():
                        with txl.serial(16) as pi:
                            values = txl.alloc_local([2], "float32")
                            for e in range(2):
                                ti0 = row0 + 2 * pi + e
                                token = tok0 + txl.Cast("int64", txl.min(ti0, last))
                                gate = _load_gate(g, token, hv, x, HV * D)
                                kval = _load_bf16_f32(
                                    k.ptr_to(
                                        [token * txl.int64(HQ * D) + txl.Cast("int64", hq * D + x)]
                                    )
                                )
                                decay = txl.local_scalar("float32")
                                txl.ptx.ex2.approx.ftz.f32(decay, gn - gate)
                                txl.assign(
                                    values[e],
                                    txl.Select(ti0 < rows, kval * decay, txl.float32(0.0)),
                                )
                            word = txl.local_scalar("uint32")
                            pack_bf16x2(word, values[0], values[1])
                            txl.ptx.st.shared.b32(TT[T2 + xs].ptr_to(xr, row0 + 2 * pi), word)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("t_early")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    b_dhb_full.wait(0, par)
                    phase("c2")
                    dgk2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    hst = txl.local_scalar("int32", init=wg * 2 + xs)
                    for u in range(8):
                        hw = txl.alloc_local([4], "uint32")
                        dw = txl.alloc_local([4], "uint32")
                        txl.ptx["ld.shared.v4.b32"](
                            hw[0], hw[1], hw[2], hw[3], TT[S_H + hst].ptr_to(xr, 8 * u)
                        )
                        txl.ptx["ld.shared.v4.b32"](
                            dw[0], dw[1], dw[2], dw[3], TT[DHB + hst].ptr_to(xr, 8 * u)
                        )
                        for p in range(4):
                            txl.ptx["fma.rn.f32x2"](
                                dgk2,
                                txl.cuda.make_float2(lo(hw[p]), hi(hw[p])),
                                txl.cuda.make_float2(lo(dw[p]), hi(dw[p])),
                                dgk2,
                            )
                    with txl.If(state_strong), txl.Then():
                        txl.ptx.ex2.approx.ftz.f32(egn, gn)
                    txl.assign(
                        dgk,
                        (txl.cuda.float2_x(dgk2) + txl.cuda.float2_y(dgk2))
                        * txl.Select(state_strong, egn, txl.float32(1.0)),
                    )

                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    with txl.If(state_strong), txl.Then():
                        for i in range(16):
                            txl.assign(kw[i], txl.uint32(0))
                        st_row(T2, row0, kw, 0, 4)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")

                    def readout64(slot, stage, mask, negate=False):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with txl.If(lane < txl.int32(16)), txl.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if negate:
                                        val = txl.float32(0.0) - val
                                    vv2.append(txl.Select(mask(cc, jj), val, txl.float32(0.0)))
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                txl.ptx["st.shared.v4.b32"](
                                    TT[stage].ptr_to(cc, row0 + 8 * u),
                                    wds[4 * u],
                                    wds[4 * u + 1],
                                    wds[4 * u + 2],
                                    wds[4 * u + 3],
                                )
                        txl.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        """Read one live 16-lane half of an M=64 accumulator.

                        `.16x256b.x4` maps each thread to two rows and eight
                        adjacent columns per row.  Pairwise bf16 conversion then
                        matches two non-transposed stmatrix.x4 stores exactly.
                        """
                        txl.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        txl.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> txl.int32(2))
                        cc1 = cc0 + txl.int32(8)
                        for rep in range(4):
                            jj0 = row0 + txl.int32(8 * rep) + (lane & txl.int32(3)) * txl.int32(2)
                            jj1 = jj0 + txl.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = txl.float32(0.0) - v00
                                v01 = txl.float32(0.0) - v01
                                v10 = txl.float32(0.0) - v10
                                v11 = txl.float32(0.0) - v11
                            pack_bf16x2(
                                wds[2 * rep],
                                txl.Select(mask(cc0, jj0), v00, txl.float32(0.0)),
                                txl.Select(mask(cc0, jj1), v01, txl.float32(0.0)),
                            )
                            pack_bf16x2(
                                wds[2 * rep + 1],
                                txl.Select(mask(cc1, jj0), v10, txl.float32(0.0)),
                                txl.Select(mask(cc1, jj1), v11, txl.float32(0.0)),
                            )
                        tile = TT[stage]
                        for half in range(2):
                            txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(
                                    quad * txl.int32(16), row0 + txl.int32(16 * half), lane
                                ),
                                wds[4 * half],
                                wds[4 * half + 1],
                                wds[4 * half + 2],
                                wds[4 * half + 3],
                            )
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")
                    if HALF_DA_READOUT:
                        readout64_half(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    else:
                        readout64(S4, DAM, lambda cc, jj: (jj < cc) & (cc < rows))
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")
                    if HALF_DA_READOUT:
                        readout64_half(S4 + (16 << 16), T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    else:
                        readout64(S1, T5, lambda cc, jj: (jj <= cc) & (cc < rows))
                    with txl.If(diagonal_needed), txl.Then():
                        bar_all()
                        with txl.If((wg == txl.int32(0)) & (x < txl.int32(64))), txl.Then():
                            diag_bits = txl.local_scalar("uint16")
                            txl.ptx.ld.shared.u16(diag_bits, TT[T5].ptr_to(x, x))
                            txl.ptx.st.shared.f32(
                                s_beta1.ptr_to([0, x]), bf16_bits_to_f32(diag_bits) * scale
                            )
                            txl.ptx.st.shared.b16(TT[T5].ptr_to(x, x), txl.uint16(0))
                        txl.ptx[FENCE_ASYNC]()
                    marrive("dAqk_tile_ready")
                    phase("w-dk")
                    twait("dk_done")
                    twait("dvb_done")
                    phase("passA")

                    pbx = txl.local_scalar("int32", init=txl.int32(PB0) + wg * txl.int32(PB1 - PB0))

                    def pass_a():
                        pa_acc = txl.local_scalar("uint64")
                        pa_v = txl.local_scalar("uint64")
                        pa_db = txl.local_scalar("uint64")
                        pa_dv = txl.local_scalar("uint64")
                        pa_word = txl.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = txl.alloc_local([4], "uint32")
                            txl.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3], TT[ST_V + xs].ptr_to(xr, row0 + 8 * b)
                            )
                            dbp = txl.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                txl.assign(
                                    pa_acc,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                )
                                txl.assign(pa_v, txl.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                txl.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                txl.assign(dbp[2 * p], txl.cuda.float2_x(pa_db))
                                txl.assign(dbp[2 * p + 1], txl.cuda.float2_y(pa_db))
                                txl.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], txl.address_of(s_beta[par, row0 + i])
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, txl.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(
                                    pa_word, txl.cuda.float2_x(pa_dv), txl.cuda.float2_y(pa_dv)
                                )
                                if not item_only:
                                    with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                                        txl.ptx["st.global.L1::no_allocate.b16"](
                                            dv.ptr_to([x_base + txl.int64(i * HVK)]),
                                            txl.Cast("uint16", pa_word),
                                        )
                                    with txl.If(row0 + txl.int32(i + 1) < rows), txl.Then():
                                        txl.ptx["st.global.L1::no_allocate.b16"](
                                            dv.ptr_to([x_base + txl.int64((i + 1) * HVK)]),
                                            txl.Cast("uint16", pa_word >> txl.uint32(16)),
                                        )
                            for e in range(8):
                                i = 8 * b + e
                                txl.ptx.st.shared.f32(
                                    TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e]
                                )
                            dvw = txl.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            txl.ptx["st.shared.v4.b32"](
                                TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                dvw[0],
                                dvw[1],
                                dvw[2],
                                dvw[3],
                            )
                            if b < 3:
                                txl.ptx[WAIT_LD]()

                    pass_a()
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    if HALF_XY_READOUT:
                        readout64_half(S1, T6, lambda cc, jj: cc < rows)
                    else:
                        readout64(S1, T6, lambda cc, jj: cc < rows)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & txl.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & txl.int32(1)) * 32 + lane
                    dsum_v2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    dsum2 = txl.local_scalar("uint64")
                    sum_pair0 = txl.local_scalar("uint64")
                    sum_pair1 = txl.local_scalar("uint64")
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(sum_pair0, txl.cuda.make_float2(t4[0], t4[1]))
                        txl.assign(sum_pair1, txl.cuda.make_float2(t4[2], t4[3]))
                        txl.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                        txl.ptx["add.rn.f32x2"](dsum_v2, dsum_v2, sum_pair0)
                    txl.ptx[FENCE_ASYNC]()
                    with txl.If(txl.Not(state_strong)), txl.Then():
                        b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    strong = txl.local_scalar("bool", init=_needs_stable(gn, range_safe))
                    if HALF_XY_READOUT:
                        readout64_half(
                            S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True
                        )
                    else:
                        readout64(S2, T5 + 1, lambda cc, jj: (jj < cc) & (cc < rows), negate=True)
                    # Y_done proves the prior X/Vn readers of T6 have finished.
                    # Include earlier gate-cache rounding in the derivative
                    # residual, without changing the state-update operands.
                    with txl.If(range_safe & diagonal_needed & txl.Not(strong)), txl.Then():
                        precise_end = txl.local_scalar("float32")
                        txl.ptx.ld.global_.nc.f32(
                            precise_end,
                            g.ptr_to(
                                [
                                    (tok0 + txl.Cast("int64", last)) * HVK64
                                    + hv64 * txl.int64(D)
                                    + x64
                                ]
                            ),
                        )
                        for precise_pair in range(16):
                            precise_values = txl.alloc_local([2], "float32")
                            for precise_half in range(2):
                                precise_log = txl.local_scalar("float32")
                                precise_token = txl.min(
                                    row0 + txl.int32(2 * precise_pair + precise_half), last
                                )
                                txl.ptx.ld.global_.nc.f32(
                                    precise_log,
                                    g.ptr_to(
                                        [
                                            (tok0 + txl.Cast("int64", precise_token)) * HVK64
                                            + hv64 * txl.int64(D)
                                            + x64
                                        ]
                                    ),
                                )
                                txl.ptx.ex2.approx.ftz.f32(
                                    precise_values[precise_half],
                                    precise_log + _gate_offset(precise_end, range_safe),
                                )
                            txl.assign(
                                egcw[precise_pair],
                                txl.cuda.make_float2(precise_values[0], precise_values[1]),
                            )
                    with txl.If(range_safe & diagonal_needed & txl.Not(strong)), txl.Then():
                        for rebuild_pair in range(16):
                            rebuild_i = 2 * rebuild_pair
                            rebuild_beta = txl.alloc_local([2], "float32")
                            txl.ptx["ld.shared.v2.f32"](
                                rebuild_beta[0],
                                rebuild_beta[1],
                                txl.address_of(s_beta[par, row0 + rebuild_i]),
                            )
                            rebuild_gate = egcw[rebuild_pair]
                            rebuild_k = txl.cuda.make_float2(
                                lo(kc[rebuild_pair]), hi(kc[rebuild_pair])
                            )
                            rebuild_q = txl.cuda.make_float2(
                                lo(qc[rebuild_pair]), hi(qc[rebuild_pair])
                            )
                            rebuild_full = txl.local_scalar("uint64")
                            rebuild_word = txl.local_scalar("uint32")
                            rebuild_inv = txl.alloc_local([2], "float32")
                            txl.ptx.rcp.approx.ftz.f32(
                                rebuild_inv[0], txl.cuda.float2_x(rebuild_gate)
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                rebuild_inv[1], txl.cuda.float2_y(rebuild_gate)
                            )
                            for rebuild_tile in range(3):
                                if rebuild_tile == 0:
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_q, rebuild_gate)
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_full, scale_pair)
                                elif rebuild_tile == 1:
                                    txl.ptx["mul.rn.f32x2"](
                                        rebuild_full,
                                        rebuild_k,
                                        txl.cuda.make_float2(rebuild_inv[0], rebuild_inv[1]),
                                    )
                                else:
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_k, rebuild_gate)
                                    txl.ptx["mul.rn.f32x2"](
                                        rebuild_full,
                                        rebuild_full,
                                        txl.cuda.make_float2(rebuild_beta[0], rebuild_beta[1]),
                                    )
                                pack_bf16x2(
                                    rebuild_word,
                                    txl.Select(
                                        row0 + txl.int32(rebuild_i) < rows,
                                        txl.cuda.float2_x(rebuild_full),
                                        txl.float32(0.0),
                                    ),
                                    txl.Select(
                                        row0 + txl.int32(rebuild_i + 1) < rows,
                                        txl.cuda.float2_y(rebuild_full),
                                        txl.float32(0.0),
                                    ),
                                )
                                txl.ptx.st.shared.b32(
                                    TT[(T1, T2, T3)[rebuild_tile] + xs].ptr_to(
                                        xr, row0 + rebuild_i
                                    ),
                                    rebuild_word,
                                )
                    # S4 and S6 contain state contractions before the intra sums.
                    # Balance those FP32 accumulators, after their tensor producers
                    # finish at Y_done, to compensate the common intra gate shift.
                    # Execute TMEM operations with all lanes; only the factor varies.
                    with txl.If(range_safe & diagonal_needed), txl.Then():
                        shift_scale = txl.local_scalar("float32")
                        txl.ptx.ex2.approx.ftz.f32(shift_scale, -_gate_offset(gn, range_safe))
                        for shift_slot in (S4, S6):
                            for shift_group in range(4):
                                shift_regs = txl.alloc_local([8], "float32")
                                ld8(shift_regs, shift_slot + wg * 32 + 8 * shift_group)
                                txl.ptx[WAIT_LD]()
                                for shift_i in range(8):
                                    txl.assign(
                                        shift_regs[shift_i], shift_regs[shift_i] * shift_scale
                                    )
                                txl.ptx[TC_ST8](
                                    tmem_at(shift_slot + wg * 32 + 8 * shift_group),
                                    *(shift_regs[z] for z in range(8)),
                                )
                                txl.ptx[WAIT_ST]()
                    # Recover the bf16 rounding residual of beta*k*cached_gate.
                    with txl.If(range_safe & diagonal_needed), txl.Then():
                        # Y_done joins prior H tensor reads; join all scalar H readers too.
                        bar_all()
                        for residual_group in range(4):
                            residual_words = txl.alloc_local([4], "uint32")
                            residual_query_words = txl.alloc_local([4], "uint32")
                            residual_inverse_words = txl.alloc_local([4], "uint32")
                            with txl.If(txl.Not(strong)):
                                with txl.Then():
                                    for residual_pair in range(4):
                                        residual_i = 8 * residual_group + 2 * residual_pair
                                        residual_beta = txl.alloc_local([2], "float32")
                                        residual_high = txl.local_scalar("uint32")
                                        residual_full = txl.local_scalar("uint64")
                                        txl.ptx["ld.shared.v2.f32"](
                                            residual_beta[0],
                                            residual_beta[1],
                                            txl.address_of(s_beta[par, row0 + residual_i]),
                                        )
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T3 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(kc[residual_i >> 1]), hi(kc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                residual_beta[0], residual_beta[1]
                                            ),
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                        # H readers are complete and the loader waits on b_h_free.
                                        # Stable corrections reuse this tile only after chunk_done.
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T1 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(qc[residual_i >> 1]), hi(qc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full, residual_full, scale_pair
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_query_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                        residual_inv_lo = txl.local_scalar("float32")
                                        residual_inv_hi = txl.local_scalar("float32")
                                        txl.ptx.rcp.approx.ftz.f32(
                                            residual_inv_lo,
                                            txl.Select(
                                                strong,
                                                txl.float32(1.0),
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx.rcp.approx.ftz.f32(
                                            residual_inv_hi,
                                            txl.Select(
                                                strong,
                                                txl.float32(1.0),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T2 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(kc[residual_i >> 1]), hi(kc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(residual_inv_lo, residual_inv_hi),
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_inverse_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                with txl.Else():
                                    for residual_pair in range(4):
                                        txl.assign(residual_words[residual_pair], txl.uint32(0))
                                        txl.assign(
                                            residual_query_words[residual_pair], txl.uint32(0)
                                        )
                                        txl.assign(
                                            residual_inverse_words[residual_pair], txl.uint32(0)
                                        )
                            txl.ptx["st.shared.v4.b32"](
                                TT[T6 + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_words[z] for z in range(4)),
                            )
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_H + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_query_words[z] for z in range(4)),
                            )
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_H + 2 + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_inverse_words[z] for z in range(4)),
                            )
                        txl.ptx[FENCE_ASYNC]()
                    marrive("intra_ready")
                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi-q")

                    def extra_ptr(base, i):
                        return TT[base + 2 * wg + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane)

                    strong_mask = txl.local_scalar("uint32")
                    txl.ptx.vote_sync.ballot.b32(
                        strong_mask, txl.ptx.pred(strong), txl.uint32(0xFFFFFFFF)
                    )

                    def cached_ptr(base, ti):
                        return TT[base + (ti >> 4)].ptr_to(4 * (ti & 15) + quad, 2 * lane)

                    with txl.If(strong_mask != txl.uint32(0)), txl.Then():
                        twait("chunk_done")
                        with txl.If(strong), txl.Then():
                            with txl.serial(32) as i:
                                token = tok0 + txl.Cast("int64", txl.min(row0 + i, last))
                                gate = _load_gate(g, token, hv, x, HV * D)
                                previous_gate = _load_gate(
                                    g, txl.max(tok0, token - txl.int64(1)), hv, x, HV * D
                                )
                                txl.ptx.ex2.approx.ftz.f32(gate, gate - previous_gate)
                                q_value = _load_bf16_f32(
                                    q.ptr_to(
                                        [token * txl.int64(HQ * D) + txl.Cast("int64", hq * D + x)]
                                    )
                                )
                                k_value = _load_bf16_f32(
                                    k.ptr_to(
                                        [token * txl.int64(HQ * D) + txl.Cast("int64", hq * D + x)]
                                    )
                                )
                                word = txl.local_scalar("uint32")
                                pack_bf16x2(word, q_value, k_value)
                                txl.ptx.st.shared.f32(cached_ptr(12, row0 + i), gate)
                                txl.ptx.st.shared.b32(cached_ptr(16, row0 + i), word)
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    # Every cache writer publishes its stores before other
                    # warps scan the full chunk, including the other token half.
                    b_stable_full.arrive(0)
                    b_stable_full.wait(0, par)
                    with txl.If(strong_mask != txl.uint32(0)), txl.Then():
                        cooperative_count = txl.local_scalar("uint32")
                        txl.ptx.popc.b32(cooperative_count, strong_mask)
                        with txl.If(cooperative_count <= txl.uint32(12)):
                            with txl.Then():
                                cooperative_left = txl.local_scalar("uint32", init=strong_mask)
                                with txl.While(cooperative_left != txl.uint32(0)):
                                    cooperative_channel = txl.local_scalar(
                                        "int32", init=txl.int32(0)
                                    )
                                    cooperative_valid = txl.local_scalar("bool", init=False)
                                    for group in range(4):
                                        cooperative_bit = txl.local_scalar("uint32")
                                        txl.ptx.bfind.u32(cooperative_bit, cooperative_left)
                                        txl.assign(
                                            cooperative_channel,
                                            txl.Select(
                                                (lane // 8) == group,
                                                txl.Cast("int32", cooperative_bit & txl.uint32(31)),
                                                cooperative_channel,
                                            ),
                                        )
                                        txl.assign(
                                            cooperative_valid,
                                            cooperative_valid
                                            | (
                                                ((lane // 8) == group)
                                                & (cooperative_left != txl.uint32(0))
                                            ),
                                        )
                                        txl.assign(
                                            cooperative_left,
                                            cooperative_left
                                            & ~(
                                                txl.uint32(1) << (cooperative_bit & txl.uint32(31))
                                            ),
                                        )
                                    cooperative_row = row0 + (lane % 8) * 4
                                    cooperative_x = quad * 32 + cooperative_channel

                                    def cooperative_ptr(base, token_row):
                                        return TT[base + (token_row >> 4)].ptr_to(
                                            4 * (token_row & 15) + quad, 2 * cooperative_channel
                                        )

                                    with txl.If(cooperative_valid), txl.Then():
                                        cooperative_qe = txl.alloc_local([4], "float32")
                                        cooperative_kp = txl.alloc_local([4], "float32")
                                        cooperative_kf = txl.alloc_local([4], "float32")
                                        cooperative_decay = txl.alloc_local([4], "float32")
                                        for e in range(4):
                                            txl.assign(cooperative_qe[e], txl.float32(0.0))
                                            txl.assign(cooperative_kp[e], txl.float32(0.0))
                                            txl.assign(cooperative_kf[e], txl.float32(0.0))
                                            txl.assign(cooperative_decay[e], txl.float32(1.0))
                                        cooperative_j = txl.local_scalar(
                                            "int32",
                                            init=txl.min(cooperative_row + txl.int32(4 - 1), last),
                                        )
                                        with txl.While(
                                            (cooperative_j >= txl.int32(0))
                                            & (cooperative_decay[0] != txl.float32(0.0))
                                            & (cooperative_j >= cooperative_row)
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_qe[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    txl.Select(
                                                        cooperative_j == cooperative_row + e,
                                                        txl.float32(0.0),
                                                        cooperative_aq,
                                                    ),
                                                    cooperative_qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kp[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_ak,
                                                    cooperative_kp[e],
                                                )
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e]
                                                    * txl.Select(
                                                        cooperative_j <= cooperative_row + e,
                                                        cooperative_alpha,
                                                        txl.float32(1.0),
                                                    ),
                                                )
                                            txl.assign(cooperative_j, cooperative_j - txl.int32(1))
                                        with txl.While(
                                            (cooperative_j >= txl.int32(0))
                                            & (cooperative_decay[0] != txl.float32(0.0))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_qe[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_aq,
                                                    cooperative_qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kp[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_ak,
                                                    cooperative_kp[e],
                                                )
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e] * cooperative_alpha,
                                                )
                                            txl.assign(cooperative_j, cooperative_j - txl.int32(1))
                                        for e in range(4):
                                            txl.assign(cooperative_decay[e], txl.float32(1.0))
                                        txl.assign(cooperative_j, cooperative_row)
                                        with txl.While(
                                            (cooperative_j < rows)
                                            & (cooperative_decay[4 - 1] != txl.float32(0.0))
                                            & (cooperative_j < cooperative_row + txl.int32(4))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e]
                                                    * txl.Select(
                                                        cooperative_j > cooperative_row + e,
                                                        cooperative_alpha,
                                                        txl.float32(1.0),
                                                    ),
                                                )
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_term,
                                                    lo(cooperative_word) * scale,
                                                    txl.Select(
                                                        cooperative_j == cooperative_row + e,
                                                        txl.float32(0.0),
                                                        cooperative_aq,
                                                    ),
                                                    hi(cooperative_word)
                                                    * s_beta_row(cooperative_j)
                                                    * cooperative_ak,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kf[e],
                                                    cooperative_term,
                                                    cooperative_decay[e],
                                                    cooperative_kf[e],
                                                )
                                            txl.assign(cooperative_j, cooperative_j + txl.int32(1))
                                        with txl.While(
                                            (cooperative_j < rows)
                                            & (cooperative_decay[4 - 1] != txl.float32(0.0))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e] * cooperative_alpha,
                                                )
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_term,
                                                    lo(cooperative_word) * scale,
                                                    cooperative_aq,
                                                    hi(cooperative_word)
                                                    * s_beta_row(cooperative_j)
                                                    * cooperative_ak,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kf[e],
                                                    cooperative_term,
                                                    cooperative_decay[e],
                                                    cooperative_kf[e],
                                                )
                                            txl.assign(cooperative_j, cooperative_j + txl.int32(1))
                                        for e in range(4):
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(0, cooperative_row + e),
                                                cooperative_qe[e] * scale,
                                            )
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(4, cooperative_row + e),
                                                cooperative_kp[e],
                                            )
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(20, cooperative_row + e),
                                                cooperative_kf[e],
                                            )
                            with txl.Else():
                                with txl.If(strong), txl.Then():
                                    with txl.serial(8) as block:
                                        i0 = 4 * block
                                        ti0 = row0 + i0
                                        qe = txl.alloc_local([4], "float32")
                                        kp = txl.alloc_local([4], "float32")
                                        kf = txl.alloc_local([4], "float32")
                                        decay = txl.alloc_local([4], "float32")
                                        for e in range(4):
                                            txl.assign(qe[e], txl.float32(0.0))
                                            txl.assign(kp[e], txl.float32(0.0))
                                            txl.assign(kf[e], txl.float32(0.0))
                                            txl.assign(decay[e], txl.float32(1.0))
                                        j = txl.local_scalar(
                                            "int32", init=txl.min(ti0 + txl.int32(3), last)
                                        )
                                        # With alpha in [0, 1], the earliest row is last to
                                        # underflow in this direction. Only skip exact zeros.
                                        with txl.While(
                                            (j >= txl.int32(0))
                                            & (decay[0] != txl.float32(0.0))
                                            & (j >= ti0)
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(4):
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(ti0 + e, j), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(ti0 + e, j), shared=True
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    qe[e],
                                                    hi(word) * decay[e],
                                                    txl.Select(j == ti0 + e, txl.float32(0.0), aq),
                                                    qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    kp[e], hi(word) * decay[e], ak, kp[e]
                                                )
                                                txl.assign(
                                                    decay[e],
                                                    decay[e]
                                                    * txl.Select(
                                                        j <= ti0 + e, alpha, txl.float32(1.0)
                                                    ),
                                                )
                                            txl.assign(j, j - txl.int32(1))
                                        with txl.While(
                                            (j >= txl.int32(0)) & (decay[0] != txl.float32(0.0))
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(4):
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(ti0 + e, j), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(ti0 + e, j), shared=True
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    qe[e], hi(word) * decay[e], aq, qe[e]
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    kp[e], hi(word) * decay[e], ak, kp[e]
                                                )
                                                txl.assign(decay[e], decay[e] * alpha)
                                            txl.assign(j, j - txl.int32(1))
                                        for e in range(4):
                                            txl.assign(decay[e], txl.float32(1.0))
                                        txl.assign(j, ti0)
                                        # In the forward direction the latest row is last.
                                        with txl.While(
                                            (j < rows)
                                            & (decay[3] != txl.float32(0.0))
                                            & (j < ti0 + txl.int32(4))
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(4):
                                                txl.assign(
                                                    decay[e],
                                                    decay[e]
                                                    * txl.Select(
                                                        j > ti0 + e, alpha, txl.float32(1.0)
                                                    ),
                                                )
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(j, ti0 + e), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(j, ti0 + e), shared=True
                                                )
                                                term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    term,
                                                    lo(word) * scale,
                                                    txl.Select(j == ti0 + e, txl.float32(0.0), aq),
                                                    hi(word) * s_beta_row(j) * ak,
                                                )
                                                txl.ptx.fma.rn.f32(kf[e], term, decay[e], kf[e])
                                            txl.assign(j, j + txl.int32(1))
                                        with txl.While((j < rows) & (decay[3] != txl.float32(0.0))):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(4):
                                                txl.assign(decay[e], decay[e] * alpha)
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(j, ti0 + e), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(j, ti0 + e), shared=True
                                                )
                                                term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    term,
                                                    lo(word) * scale,
                                                    aq,
                                                    hi(word) * s_beta_row(j) * ak,
                                                )
                                                txl.ptx.fma.rn.f32(kf[e], term, decay[e], kf[e])
                                            txl.assign(j, j + txl.int32(1))
                                        for e in range(4):
                                            txl.ptx.st.shared.f32(
                                                extra_ptr(0, i0 + e), qe[e] * scale
                                            )
                                            txl.ptx.st.shared.f32(extra_ptr(4, i0 + e), kp[e])
                                            txl.ptx.st.shared.f32(extra_ptr(20, i0 + e), kf[e])
                        txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                        txl.ptx[FENCE_ASYNC]()
                        with txl.If(strong), txl.Then():
                            b_mid_free.arrive(0)

                    with txl.If(state_strong & txl.Not(strong)), txl.Then():
                        b_mid_free.arrive(0)

                    txl.ptx[FENCE_ASYNC]()
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    with txl.If(lane == txl.int32(0)), txl.Then():
                        b_intra_free.arrive(0, count=32)
                    dgk_k2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def emit_group_output(b, vals, tm_col, out, obase):
                        """dq/dk for 8 tokens of this channel: accumulate over the head group in TMEM."""
                        if G > 1:
                            with txl.If(gi > txl.int32(0)), txl.Then():
                                ld8(oq, tm_col + wg * 32 + 8 * b)
                                txl.ptx[WAIT_LD]()
                                for e in range(8):
                                    txl.assign(vals[e], vals[e] + oq[e])
                        with txl.If(gi == txl.int32(G - 1)):
                            with txl.Then():
                                for e in range(8):
                                    i = 8 * b + e
                                    with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                                        txl.ptx["st.global.L1::no_allocate.f32"](
                                            out.ptr_to([obase + txl.int64(i * HQK)]), vals[e]
                                        )
                            if G > 1:
                                with txl.Else():
                                    st8(tm_col + wg * 32 + 8 * b, vals)
                                    txl.ptx[WAIT_ST]()

                    def add_diagonal(dst, i, inputs):
                        with txl.If(diagonal_needed), txl.Then():
                            diagonal = txl.alloc_local([2], "float32")
                            txl.ptx["ld.shared.v2.f32"](
                                diagonal[0], diagonal[1], s_beta1.ptr_to([0, row0 + i])
                            )
                            txl.ptx["fma.rn.f32x2"](
                                dst,
                                txl.cuda.make_float2(lo(inputs), hi(inputs)),
                                txl.cuda.make_float2(diagonal[0], diagonal[1]),
                                dst,
                            )

                    def epilogue(stable):
                        pair0 = txl.local_scalar("uint64")
                        pair1 = txl.local_scalar("uint64")
                        pair2 = txl.local_scalar("uint64")
                        pair3 = txl.local_scalar("uint64")
                        pair4 = txl.local_scalar("uint64")
                        pair5 = txl.local_scalar("uint64")
                        q_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(
                                        txl.cuda.float2_x(egcw[i >> 1]),
                                        txl.cuda.float2_y(egcw[i >> 1]),
                                    ),
                                    txl.cuda.make_float2(scale, scale),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair0,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                if stable:
                                    with txl.If(strong), txl.Then():
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(0, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(0, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair0, pair0, txl.cuda.make_float2(t0, t1)
                                        )
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])),
                                    pair0,
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair1))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair1))
                                add_diagonal(pair0, i, kc[i >> 1])
                                txl.assign(ok8[2 * p], txl.cuda.float2_x(pair0))
                                txl.assign(ok8[2 * p + 1], txl.cuda.float2_y(pair0))
                            emit_group_output(b, ok8, TM_ADQ, dq, xq_base)
                            if b < 3:
                                txl.ptx[WAIT_LD]()
                        phase("w-dkt")
                        twait("dkt_done")
                        phase("epi-k")
                        dbx = 2 * wg
                        k_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                # Keep each reciprocal at its key-gradient use;
                                # retaining 32 across dq output adds local traffic.
                                intra_inverse_a = txl.local_scalar("float32")
                                intra_inverse_b = txl.local_scalar("float32")
                                if stable:
                                    rcp(intra_inverse_a, txl.cuda.float2_x(egcw[i >> 1]))
                                    rcp(intra_inverse_b, txl.cuda.float2_y(egcw[i >> 1]))
                                else:
                                    txl.ptx.rcp.approx.ftz.f32(
                                        intra_inverse_a, txl.cuda.float2_x(egcw[i >> 1])
                                    )
                                    txl.ptx.rcp.approx.ftz.f32(
                                        intra_inverse_b, txl.cuda.float2_y(egcw[i >> 1])
                                    )
                                txl.assign(
                                    pair0, txl.cuda.make_float2(intra_inverse_a, intra_inverse_b)
                                )
                                txl.assign(
                                    pair1, txl.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1]))
                                )
                                state_decay(t0, i, txl.cuda.float2_x(egcw[i >> 1]))
                                state_decay(t1, i + 1, txl.cuda.float2_y(egcw[i >> 1]))
                                state_pair = txl.local_scalar("uint64")
                                txl.ptx["mul.rn.f32x2"](
                                    state_pair,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    txl.cuda.make_float2(t0, t1),
                                )
                                txl.ptx["fma.rn.f32x2"](
                                    pair2,
                                    txl.cuda.make_float2(
                                        acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]
                                    ),
                                    pair0,
                                    state_pair,
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    txl.cuda.make_float2(
                                        acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]
                                    ),
                                    txl.cuda.make_float2(
                                        txl.cuda.float2_x(egcw[i >> 1]),
                                        txl.cuda.float2_y(egcw[i >> 1]),
                                    ),
                                )
                                if stable:
                                    with txl.If(strong), txl.Then():
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(20, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(20, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair2, pair2, txl.cuda.make_float2(t0, t1)
                                        )
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(4, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(4, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair3, pair3, txl.cuda.make_float2(t0, t1)
                                        )
                                txl.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                txl.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane),
                                    txl.cuda.float2_x(pair4),
                                )
                                txl.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(
                                        4 * ((i + 1) & 15) + quad, 2 * lane
                                    ),
                                    txl.cuda.float2_y(pair4),
                                )
                                txl.ptx["mul.rn.f32x2"](pair5, pair1, state_pair)
                                txl.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    pair3,
                                    txl.cuda.make_float2(
                                        s_beta_row(row0 + i), s_beta_row(row0 + i + 1)
                                    ),
                                )
                                txl.ptx["add.rn.f32x2"](pair5, pair2, pair3)
                                add_diagonal(pair5, i, qc[i >> 1])
                                txl.assign(ok8[4 * (b % 2) + 2 * p], txl.cuda.float2_x(pair5))
                                txl.assign(ok8[4 * (b % 2) + 2 * p + 1], txl.cuda.float2_y(pair5))
                                txl.ptx["sub.rn.f32x2"](pair3, pair3, pair2)
                                txl.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, txl.cuda.make_float2(dgv[i], dgv[i + 1])
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair5))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair5))
                            if b % 2 == 1:
                                emit_group_output(b // 2, ok8, TM_ADK, dk, xq_base)
                            if b < 7:
                                txl.ptx[WAIT_LD]()

                    with txl.If(strong_mask != txl.uint32(0)):
                        with txl.Then():
                            epilogue(True)
                        with txl.Else():
                            epilogue(False)
                    dbx = 2 * wg
                    bar_wg()
                    txl.assign(dsum2, dsum_v2)
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(sum_pair0, txl.cuda.make_float2(t4[0], t4[1]))
                        txl.assign(sum_pair1, txl.cuda.make_float2(t4[2], t4[3]))
                        txl.ptx["add.rn.f32x2"](sum_pair0, sum_pair0, sum_pair1)
                        txl.ptx["add.rn.f32x2"](dsum2, dsum2, sum_pair0)
                    dsum = txl.local_scalar(
                        "float32", init=txl.cuda.float2_x(dsum2) + txl.cuda.float2_y(dsum2)
                    )
                    txl.ptx[FENCE_ASYNC]()
                    b_h_free.arrive(0)
                    for s in (1, 2):
                        r = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.bfly.b32(
                            r,
                            txl.reinterpret("uint32", dsum),
                            txl.uint32(s),
                            txl.uint32(0x1F),
                            txl.uint32(0xFFFFFFFF),
                        )
                        txl.assign(dsum, dsum + txl.reinterpret("float32", r))
                    if not item_only:
                        with txl.If((tq == txl.int32(0)) & (row0 + ti < rows)), txl.Then():
                            txl.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to(
                                    [(tok0 + txl.Cast("int64", row0 + ti)) * txl.int64(HV) + hv64]
                                ),
                                dsum,
                            )

                    txl.assign(dgk_k, txl.cuda.float2_x(dgk_k2) + txl.cuda.float2_y(dgk_k2))
                    phase("cumsum")
                    for i in range(32):
                        txl.assign(
                            dgv[i], txl.Select(row0 + txl.int32(i) < rows, dgv[i], txl.float32(0.0))
                        )

                    tot = txl.alloc_local([16], "float32")
                    for i in range(16):
                        txl.assign(tot[i], dgv[2 * i] + dgv[2 * i + 1])
                    for w in (8, 4, 2, 1):
                        for i in range(w):
                            txl.assign(tot[i], tot[i] + tot[i + w])
                    txl.ptx.st.shared.f32(
                        txl.address_of(s_dgk[wg, x]),
                        dgk + dgk_k + txl.Select(wg == txl.int32(0), txl.float32(0.0), tot[0]),
                    )
                    b_dg0_ready.arrive(0)
                    for i in range(30, -1, -1):
                        txl.assign(dgv[i], dgv[i] + dgv[i + 1])
                    b_dg0_ready.wait(0, cyc & txl.int32(1))
                    txl.ptx.ld.shared.f32(t0, txl.address_of(s_dgk[txl.int32(1) - wg, x]))
                    txl.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        with txl.If(row0 + txl.int32(i) < rows), txl.Then():
                            txl.ptx["st.global.L1::no_allocate.f32"](
                                dg.ptr_to([x_base + txl.int64(i * HVK)]), dgv[i] + t1
                            )
                    phase_end()
                    txl.assign(cyc, cyc + txl.int32(1))
                txl.assign(kk_, kk_ + txl.int32(1))
                txl.assign(cur, work_wait(kk_))

        with auxg:
            with mma:
                tm = tmem_preamble()
                bd1 = txl.alloc_local([1], "uint64")
                zq1 = txl.alloc_local([1], "int32")

                op_kbg_k = Op(bd1, F_KBG, 128, 64, "k")
                op_vb_k = Op(bd1, F_VB, 128, 64, "k")
                op_kg_k = Op(bd1, F_KG, 128, 64, "k")
                op_akk1_k = Op(bd1, F_AKK, 64, 64, "k")
                op_hs_mn = Op(bd1, F_HS, 128, 128, "mn")
                op_w_mn = Op(bd1, F_KBG, 128, 128, "mn")
                op_kraw = Op(bd1, F_KV, 128, 64, "mn")
                op_vraw = Op(bd1, F_KV + 1, 128, 64, "mn")
                ID_T1 = idesc(128, 16, ta=1)
                bdI1 = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )
                st_kv1 = txl.PipelineState(1, phase=0)
                p1m = txl.local_scalar("int32", init=txl.int32(0))
                SET_UNITS = 2 * UNITS_PER_STAGE

                op_bakk_k = Op(bd1, B_AKK, 64, 64, "k")
                op_bdo_mn = Op(bd1, B_DO, 64, 64, "mn")
                DO2_UNITS = 2 * UNITS_PER_STAGE
                op_baqk_mn = Op(bd1, B_AQK, 64, 64, "mn")
                op_dhb_mn = Op(bd1, B_DHB, 128, 128, "mn")
                op_t2_mn = Op(bd1, B_T2, 128, 128, "mn")
                op_dv2_k = Op(bd1, B_DV2, 128, 64, "k")
                op_qraw = Op(bd1, B_QK, 128, 64, "mn")
                op_kraw_b = Op(bd1, B_QK + 1, 128, 64, "mn")
                st_qk1 = txl.PipelineState(1, phase=0)
                bqm = txl.local_scalar("int32", init=txl.int32(0))
                ID_M128N64 = idesc(128, 64)
                ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_HUPD = idesc(128, 128)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                st_tiles = txl.PipelineState(2, phase=0)
                st_akk = txl.PipelineState(2, phase=0)
                st_hs = txl.PipelineState(1, phase=0)
                st_w = txl.PipelineState(2, phase=0)
                st_vn = txl.PipelineState(1, phase=0)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                mphase, mphase_end = make_phaser()

                def encode_base():
                    txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                    txl.cuda.tcgen05.encode_matrix_descriptor(
                        txl.address_of(bd1[0]),
                        TT[zq1[0]].ptr_to(0, 0),
                        ldo=Op.LBO_BASE,
                        sdo=SBO_UNITS,
                        swizzle=txl.SW128B.value,
                    )

                def kv_transpose():
                    """Raw K and V chunk tiles -> channel-major fp32 K^T / V^T in TMEM (eight identity MMAs)."""
                    mphase("fmw-kv")
                    p_kv.full.wait(0, st_kv1.phase)
                    b_kv_read.wait(0, (p1m & txl.int32(1)) ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("fm-kvT")
                    with txl.If(elected()), txl.Then():
                        for src, dst in ((op_kraw, TM_KT), (op_vraw, TM_VT)):
                            for j in range(4):
                                txl.ptx[MMA_SS](
                                    txl.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    txl.uint32(ID_T1),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.ptx.pred(0),
                                )
                        b_kvT_done.arrive(0)
                        p_kv.empty.arrive(0)
                    st_kv1.advance()
                    txl.assign(p1m, p1m + txl.int32(1))

                def qk_transpose():
                    """Raw q and k chunk tiles -> channel-major fp32 q^T / k^T in TMEM (eight identity MMAs)."""
                    mphase("bmw-qk")
                    p_qk.full.wait(0, st_qk1.phase)
                    b_qk_read.wait(0, (bqm & txl.int32(1)) ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("bm-qkT")
                    with txl.If(elected()), txl.Then():
                        for src, dst in ((op_qraw, TM_QT), (op_kraw_b, TM_KTB)):
                            for j in range(4):
                                txl.ptx[MMA_SS](
                                    txl.Cast("uint32", tm[0] + dst + 16 * j),
                                    src.desc(j),
                                    bdI1[0],
                                    txl.uint32(ID_T1),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.ptx.pred(0),
                                )
                        b_qkT_done.arrive(0)
                        p_qk.empty.arrive(0)
                    st_qk1.advance()
                    txl.assign(bqm, bqm + txl.int32(1))

                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            encode_base()
                            kv_transpose()
                            with txl.If(nch > txl.int32(1)), txl.Then():
                                kv_transpose()
                            with txl.serial(nch) as n:
                                set_u = txl.local_scalar(
                                    "uint64",
                                    init=txl.Cast("uint64", n & txl.int32(1))
                                    * txl.uint64(SET_UNITS),
                                )
                                akk_u = txl.local_scalar(
                                    "uint64",
                                    init=txl.Cast("uint64", st_akk.stage)
                                    * txl.uint64(UNITS_PER_STAGE),
                                )
                                dW = (n & txl.int32(1)) * 64
                                mphase("fmw-tiles")
                                p_tiles.full.wait(st_tiles.stage, st_tiles.phase)
                                p_akk1.full.wait(st_akk.stage, st_akk.phase)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-WU")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_W0 + dW,
                                        op_kbg_k,
                                        op_akk1_k,
                                        ID_M128N64,
                                        False,
                                        a_units=set_u,
                                        b_units=akk_u,
                                    )
                                    p_w.full.arrive(st_w.stage)
                                    mma_chain(
                                        tm,
                                        TM_U0 + dW,
                                        op_vb_k,
                                        op_akk1_k,
                                        ID_M128N64,
                                        False,
                                        a_units=set_u,
                                        b_units=akk_u,
                                    )
                                    p_akk1.empty.arrive(st_akk.stage)
                                st_akk.advance()
                                mphase("fmw-wT")
                                p_w.empty.wait(st_w.stage, st_w.phase)
                                st_w.advance()
                                mphase("fmw-hs")
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-Vn")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_U0 + dW,
                                        op_hs_mn,
                                        op_w_mn,
                                        ID_VN,
                                        True,
                                        b_units=set_u,
                                    )
                                    p_vn.full.arrive(0)
                                st_hs.advance()
                                mphase("fmw-vnT")
                                p_vn.empty.wait(0, st_vn.phase)
                                st_vn.advance()
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("fm-hupd")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_H,
                                        op_kg_k,
                                        op_vb_k,
                                        ID_HUPD,
                                        True,
                                        a_units=set_u,
                                        b_units=set_u,
                                    )
                                    p_tiles.empty.arrive(st_tiles.stage)
                                st_tiles.advance()
                                with txl.If(n + txl.int32(2) < nch), txl.Then():
                                    kv_transpose()
                                mphase_end()
                        with txl.Else():
                            encode_base()
                            qk_transpose()
                            with txl.serial(nch) as rn:
                                par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                                t1_col = TM_T1 + par * 32
                                kb_col = TM_KB + par * 32
                                do_u = txl.local_scalar(
                                    "uint64", init=txl.Cast("uint64", par) * txl.uint64(DO2_UNITS)
                                )
                                mphase("bmw-prep")
                                MB["prep_ready"].wait(0, par)
                                b_bakk_full.wait(0, par)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-W")
                                with txl.If(elected()), txl.Then():
                                    mma_chain_ta(tm, TM_BW, kb_col, op_bakk_k, ID_M128N64, False)
                                    TC["W_done"].arrive(0)
                                    b_bakk_empty.arrive(0)
                                with txl.If(rn + txl.int32(1) < nch), txl.Then():
                                    qk_transpose()
                                mphase("bmw-dhb")
                                MB["dhb_ready"].wait(0, par)
                                b_baqk_masked.wait(0, par)
                                b_bdo_full.wait(par, (bcyc >> 1) & txl.int32(1))
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dv2")
                                with txl.If(elected()), txl.Then():
                                    mma_chain(
                                        tm,
                                        TM_DV2,
                                        op_bdo_mn,
                                        op_baqk_mn,
                                        ID_128x64_TATB,
                                        False,
                                        a_units=do_u,
                                    )
                                    b_baqk_empty.arrive(0)
                                    mma_chain(tm, TM_DV2, op_dhb_mn, op_t2_mn, ID_128x64_TATB, True)
                                    TC["dv2_done"].arrive(0)
                                mphase("bmw-rd")
                                MB["wT_ready"].wait(0, par)
                                MB["dv2T_ready"].wait(0, par)
                                txl.ptx[TC_FENCE_AFTER]()
                                mphase("bm-dh")
                                with txl.If(elected()), txl.Then():
                                    mma_chain_ta(
                                        tm,
                                        TM_DH,
                                        t1_col,
                                        op_bdo_mn,
                                        ID_128x128_TB,
                                        True,
                                        b_units=do_u,
                                    )
                                    mma_chain_ta(tm, TM_DH, kb_col, op_dv2_k, ID_128x128_NB, True)
                                    TC["dh_done"].arrive(0)
                                mphase_end()
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (q_map, k_map, v_map, g_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                st_kv = txl.PipelineState(1, phase=1)
                st_akk = txl.PipelineState(2, phase=1)
                st_g = txl.PipelineState(1, phase=1)
                st_qk = txl.PipelineState(1, phase=1)
                st_bg = txl.PipelineState(1, phase=1)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            with txl.serial(nch) as n:
                                tok0 = bos32 + n * txl.int32(CHUNK)
                                rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                                p_kv.empty.wait(0, st_kv.phase)
                                with txl.If(elected()), txl.Then():
                                    p_kv.full.arrive(0, tx_count=KV_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[F_KV + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                            txl.address_of(k_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                        txl.ptx[TMA_LD](
                                            TT[F_KV + txl.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0),
                                            txl.address_of(v_map),
                                            txl.int32(d0),
                                            tok0,
                                            hv,
                                            mb,
                                        )
                                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(k_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hq,
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(v_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hv,
                                            )
                                        for d0 in (0, 32, 64, 96):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(g_map),
                                                txl.int32(d0),
                                                tok0 + txl.int32(CHUNK),
                                                hv,
                                            )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(akk_map),
                                            txl.int32(0),
                                            tok0 + txl.int32(CHUNK),
                                            hv,
                                        )
                                st_kv.advance()
                                p_g.empty.wait(0, st_g.phase)
                                with txl.If(elected()), txl.Then():
                                    p_g.full.arrive(0, tx_count=G_BYTES)
                                    mbg = txl.cuda.cvta_generic_to_shared(p_g.full.ptr_to([0]))
                                    for j in range(4):
                                        txl.ptx[TMA_LD](
                                            TT[F_G + txl.int32(j)].ptr_to(0, 0),
                                            txl.address_of(g_map),
                                            txl.int32(32 * j),
                                            tok0,
                                            hv,
                                            mbg,
                                        )

                                bslot = txl.local_scalar("int32", init=n & txl.int32(1))
                                load_beta_lanes(
                                    lambda t: txl.address_of(s_beta1[bslot, t]), bos, hv, n, rows
                                )
                                p_g.full.arrive(0)
                                st_g.advance()
                                p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                                with txl.If(elected()), txl.Then():
                                    p_akk1.full.arrive(st_akk.stage, tx_count=A_BYTES)
                                    mb2 = txl.cuda.cvta_generic_to_shared(
                                        p_akk1.full.ptr_to([st_akk.stage])
                                    )
                                    txl.ptx[TMA_LD](
                                        TT[F_AKK + st_akk.stage].ptr_to(0, 0),
                                        txl.address_of(akk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb2,
                                    )
                                st_akk.advance()
                        with txl.Else():
                            with txl.serial(nch) as rn:
                                n = nch - txl.int32(1) - rn
                                par = txl.local_scalar("int32", init=bcyc & txl.int32(1))
                                tok0 = bos32 + n * txl.int32(CHUNK)
                                rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))

                                lphase("blw-qk")
                                p_qk.empty.wait(0, st_qk.phase)
                                lphase("bl-qk")
                                with txl.If(elected()), txl.Then():
                                    p_qk.full.arrive(0, tx_count=KV_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(p_qk.full.ptr_to([0]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[B_QK + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                            txl.address_of(q_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                        txl.ptx[TMA_LD](
                                            TT[B_QK + txl.int32((d0 // 64) * 2 + 1)].ptr_to(0, 0),
                                            txl.address_of(k_map),
                                            txl.int32(d0),
                                            tok0,
                                            hq,
                                            mb,
                                        )
                                    with txl.If(n > txl.int32(0)), txl.Then():
                                        tokp = tok0 - txl.int32(CHUNK)
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(q_map), txl.int32(d0), tokp, hq
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(k_map), txl.int32(d0), tokp, hq
                                            )
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(do_map), txl.int32(d0), tokp, hv
                                            )
                                        for d0 in (0, 32, 64, 96):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(g_map), txl.int32(d0), tokp, hv
                                            )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(aqk_map), txl.int32(0), tokp, hv
                                        )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(akk_map), txl.int32(0), tokp, hv
                                        )
                                st_qk.advance()

                                lphase("blw-g")
                                b_g_free.wait(0, st_bg.phase)
                                lphase("bl-g")
                                with txl.If(elected()), txl.Then():
                                    b_g_full.arrive(0, tx_count=G_BYTES)
                                    mbg = txl.cuda.cvta_generic_to_shared(b_g_full.ptr_to([0]))
                                    for j in range(4):
                                        txl.ptx[TMA_LD](
                                            TT[B_G + j].ptr_to(0, 0),
                                            txl.address_of(g_map),
                                            txl.int32(32 * j),
                                            tok0,
                                            hv,
                                            mbg,
                                        )
                                load_beta_lanes(
                                    lambda t: txl.address_of(s_bbeta[par, t]), bos, hv, n, rows
                                )
                                b_g_full.arrive(0)
                                st_bg.advance()

                                lphase("blw-do")
                                with txl.If(rn > txl.int32(1)), txl.Then():
                                    TC["dh_done"].wait(0, par)
                                lphase("bl-do")
                                with txl.If(elected()), txl.Then():
                                    b_bdo_full.arrive(par, tx_count=DO_BYTES)
                                    mbd = txl.cuda.cvta_generic_to_shared(b_bdo_full.ptr_to([par]))
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_LD](
                                            TT[
                                                B_DO + par * txl.int32(2) + txl.int32(d0 // 64)
                                            ].ptr_to(0, 0),
                                            txl.address_of(do_map),
                                            txl.int32(d0),
                                            tok0,
                                            hv,
                                            mbd,
                                        )
                                lphase("blw-a")
                                # These depth-one tiles persist across backward streams.
                                # ``rn`` resets per stream; ``bcyc`` tracks every reuse.
                                with txl.If(bcyc > txl.int32(0)), txl.Then():
                                    b_baqk_empty.wait(0, par ^ txl.int32(1))
                                with txl.If(elected()), txl.Then():
                                    b_baqk_full.arrive(0, tx_count=A_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(b_baqk_full.ptr_to([0]))
                                    txl.ptx[TMA_LD](
                                        TT[B_AQK].ptr_to(0, 0),
                                        txl.address_of(aqk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb,
                                    )
                                with txl.If(bcyc > txl.int32(0)), txl.Then():
                                    b_bakk_empty.wait(0, par ^ txl.int32(1))
                                with txl.If(elected()), txl.Then():
                                    b_bakk_full.arrive(0, tx_count=A_BYTES)
                                    mb = txl.cuda.cvta_generic_to_shared(b_bakk_full.ptr_to([0]))
                                    txl.ptx[TMA_LD](
                                        TT[B_AKK].ptr_to(0, 0),
                                        txl.address_of(akk_map),
                                        txl.int32(0),
                                        tok0,
                                        hv,
                                        mb,
                                    )
                                lphase_end()
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    claim_publish(kk_ + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w10:
                st_hs = txl.PipelineState(1, phase=0)
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    cb = chunk_base(seq)

                    fidx_s = txl.local_scalar("int32", init=seq * txl.int32(HV) + hv)
                    with txl.If(is_fwd == txl.int32(1)):
                        with txl.Then():
                            with txl.serial(nch) as n:
                                p_hs.full.wait(st_hs.stage, st_hs.phase)
                                with txl.If(elected()), txl.Then():
                                    txl.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * txl.int32(HV) + hv
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_ST](
                                            txl.address_of(h_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            idx,
                                            TT[
                                                F_HS
                                                + st_hs.stage * txl.int32(4)
                                                + txl.int32((d0 // 64) * 2)
                                            ].ptr_to(0, 0),
                                        )
                                    txl.ptx[BULK_COMMIT]()
                                    txl.ptx[BULK_WAIT_READ](0)
                                    p_hs.empty.arrive(st_hs.stage)

                                    txl.ptx[BULK_WAIT](0)
                                    txl.ptx["fence.proxy.async.global"]()
                                    txl.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([fidx_s]),
                                        ep64 + txl.Cast("int64", n + txl.int32(1)),
                                    )
                                st_hs.advance()
                        with txl.Else():
                            with txl.serial(nch) as rn:
                                n = nch - txl.int32(1) - rn
                                par = bcyc & txl.int32(1)
                                MB["dhb_ready"].wait(0, par)
                                with txl.If(elected()), txl.Then():
                                    txl.ptx[FENCE_ASYNC]()
                                    idx = (cb + n) * txl.int32(HV) + hv
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_ST](
                                            txl.address_of(dh_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            idx,
                                            TT[B_DHB + txl.int32((d0 // 64) * 2)].ptr_to(0, 0),
                                        )
                                    txl.ptx[BULK_COMMIT]()
                                    txl.ptx[BULK_WAIT_READ](0)
                                    b_dhb_stored.arrive(0)
                                    txl.ptx[BULK_WAIT](0)
                                    txl.ptx["fence.proxy.async.global"]()
                                    txl.ptx["st.release.gpu.global.s64"](
                                        flags.ptr_to([num_chains + fidx_s]),
                                        ep64 + txl.Cast("int64", rn + txl.int32(1)),
                                    )
                                txl.assign(bcyc, bcyc + txl.int32(1))
                    with txl.If(elected()), txl.Then():
                        txl.ptx[BULK_WAIT](0)
                        txl.ptx["fence.proxy.async.global"]()

                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w11:
                bcyc = txl.local_scalar("int32", init=txl.int32(0))
                lane = txl.lane_id()
                with txl.While(cur < num_streams):
                    is_fwd, seq, hv, hq, bos, seq_len, nch = stream_coords(cur)
                    with txl.If(is_fwd == txl.int32(0)), txl.Then():
                        with txl.serial(nch) as rn:
                            n = nch - txl.int32(1) - rn
                            par = bcyc & txl.int32(1)
                            rows = txl.local_scalar("int32", init=chunk_rows(seq_len, n))
                            b_baqk_full.wait(0, par)

                            for half in range(2):
                                diag = txl.alloc_local([4], "uint32")
                                dmat = lane >> txl.int32(3)
                                dblk = txl.int32(4 * half) + dmat
                                dptr = TT[B_AQK].ptr_to(
                                    dblk * txl.int32(8) + (lane & txl.int32(7)), dblk * txl.int32(8)
                                )
                                txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                    diag[0], diag[1], diag[2], diag[3], dptr
                                )
                                drow = lane >> txl.int32(2)
                                dcol = (lane & txl.int32(3)) * txl.int32(2)
                                dmask = txl.Select(
                                    dcol > drow,
                                    txl.uint32(0),
                                    txl.Select(
                                        dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)
                                    ),
                                )
                                for e in range(4):
                                    blk_row = txl.int32(8 * (4 * half + e)) + drow
                                    txl.assign(
                                        diag[e],
                                        txl.Select(blk_row < rows, diag[e] & dmask, txl.uint32(0)),
                                    )
                                txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                    dptr, diag[0], diag[1], diag[2], diag[3]
                                )
                            for r in range(2):
                                rowc = lane + txl.int32(32 * r)
                                for u in range(1, 8):
                                    with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                                        txl.ptx["st.shared.v4.b32"](
                                            TT[B_AQK].ptr_to(rowc, 8 * u),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                        )
                                with txl.If(rowc >= rows), txl.Then():
                                    for u in range(0, 8):
                                        txl.ptx["st.shared.v4.b32"](
                                            TT[B_AQK].ptr_to(rowc, 8 * u),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                            txl.uint32(0),
                                        )
                            txl.ptx[FENCE_ASYNC]()
                            b_baqk_masked.arrive(0)
                            txl.assign(bcyc, bcyc + txl.int32(1))
                    txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            # All four auxiliary warps must synchronize before reallocating.
            txl.ptx.bar.sync(txl.uint32(7), txl.uint32(128))

        with auxg:
            with mma:
                tm = tmem_preamble()
                cyc = txl.local_scalar("int32", init=txl.int32(0))

                def mwait(nm):
                    MBG[nm].wait(0, cyc & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()
                bd = txl.alloc_local([1], "uint64")
                zq = txl.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_residual_k = Op(bd, T6, 128, 64, "k")
                op_residual_q = Op(bd, S_H, 128, 64, "k")
                op_residual_inverse = Op(bd, S_H + 2, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)

                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )

                item = txl.local_scalar("int32", init=cur - num_streams)
                with txl.While(cur < total_work):
                    txl.assign(item, cur - num_streams)
                    with txl.serial(G) as gi:
                        par = cyc & txl.int32(1)
                        txl.ptx.ld.volatile.shared.s32(zq[0], txl.address_of(s_tmem[1]))
                        txl.cuda.tcgen05.encode_matrix_descriptor(
                            txl.address_of(bd[0]),
                            TT[zq[0]].ptr_to(0, 0),
                            ldo=Op.LBO_BASE,
                            sdo=SBO_UNITS,
                            swizzle=txl.SW128B.value,
                        )
                        akk_u = txl.local_scalar(
                            "uint64", init=txl.Cast("uint64", par) * txl.uint64(UNITS_PER_STAGE)
                        )
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)

                        b_h_free.wait(0, par ^ txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with txl.If(elected()), txl.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    txl.ptx[MMA_SS](
                                        txl.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        txl.uint32(ID_T),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.ptx.pred(0),
                                    )
                            TCG["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                        b_akk_masked.wait(par, (cyc >> 1) & txl.int32(1))
                        b_h_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TCG["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        b_dhb_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dv2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(
                                tm,
                                S3,
                                op_DHBmn,
                                op_T2k if False else Op(bd, T2, 128, 128, "mn"),
                                ID_128x64_TATB,
                                True,
                            )
                            TCG["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TCG["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TCG["dAs_done"].arrive(0)
                            mma_chain(
                                tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u
                            )
                            TCG["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with txl.If(elected()), txl.Then():
                            if HALF_DA_READOUT:
                                mma_chain(
                                    tm, S4 + (16 << 16), op_do_k128, op_T6mn, ID_64x64_TB, False
                                )
                            else:
                                mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TCG["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TCG["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TCG["X_done"].arrive(0)
                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            b_do_empty.arrive(0)
                            with txl.If(txl.Not(range_safe & diagonal_needed)), txl.Then():
                                mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            with txl.If(txl.Not(range_safe & diagonal_needed)), txl.Then():
                                TCG["dq2_done"].arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with txl.If(elected()), txl.Then():
                            mma_chain(
                                tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u
                            )
                            TCG["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dkt")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            with txl.If(range_safe & diagonal_needed), txl.Then():
                                mma_chain(tm, S3, op_residual_k, op_dAkk_mn, ID_128x64_TB, True)
                                mma_chain(tm, S3, op_residual_q, op_dAqk_mn, ID_128x64_TB, True)
                            with txl.If(range_safe & diagonal_needed), txl.Then():
                                mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                                mma_chain(tm, S4, op_residual_inverse, op_dAqk_k, ID_128x64, True)
                                mma_chain(tm, S6, op_residual_inverse, op_dAkk_k, ID_128x64, True)
                                TCG["dq2_done"].arrive(0)
                            TCG["dkt_done"].arrive(0)
                            TCG["chunk_done"].arrive(0)
                        mphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (q_map, k_map, v_map, eg_map, do_map, aqk_map, akk_map, h_map, dh_map):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                item = txl.local_scalar("int32", init=cur - num_streams)
                with txl.While(cur < total_work):
                    txl.assign(item, cur - num_streams)
                    c, hq, seq, n, bos, rows, nch_i = item_coords(item)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    tok0 = txl.local_scalar("int32", init=bos32 + n * txl.int32(CHUNK))

                    lphase("lw-flags")
                    with txl.If(elected()), txl.Then():
                        tgt_f = txl.local_scalar(
                            "int64", init=ep64 + txl.Cast("int64", n + txl.int32(1))
                        )
                        tgt_b = txl.local_scalar("int64", init=ep64 + txl.Cast("int64", nch_i - n))
                        for gi_ in range(G):
                            fidx = seq * txl.int32(HV) + hq * txl.int32(G) + txl.int32(gi_)
                            flf = txl.local_scalar("int64", init=txl.int64(0))
                            txl.cuda.wait_until(
                                flf, flags.ptr_to([fidx]), flf >= tgt_f, scope="gpu"
                            )
                            flb = txl.local_scalar("int64", init=txl.int64(0))
                            txl.cuda.wait_until(
                                flb, flags.ptr_to([num_chains + fidx]), flb >= tgt_b, scope="gpu"
                            )

                        with txl.If(rows < txl.int32(CHUNK)), txl.Then():
                            tgt_1 = txl.local_scalar("int64", init=ep64 + txl.int64(1))
                            s2 = txl.local_scalar("int32", init=seq + txl.int32(1))
                            b2 = txl.local_scalar("int32", init=tok0 + rows)
                            with txl.While((s2 < num_seqs) & (b2 < tok0 + txl.int32(CHUNK))):
                                for gi_ in range(G):
                                    fl2 = txl.local_scalar("int64", init=txl.int64(0))
                                    txl.cuda.wait_until(
                                        fl2,
                                        flags.ptr_to(
                                            [
                                                s2 * txl.int32(HV)
                                                + hq * txl.int32(G)
                                                + txl.int32(gi_)
                                            ]
                                        ),
                                        fl2 >= tgt_1,
                                        scope="gpu",
                                    )
                                _, l2 = seq_len_of(s2)
                                txl.assign(b2, b2 + l2)
                                txl.assign(s2, s2 + txl.int32(1))
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    txl.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with txl.serial(G) as gi:
                        hv = txl.local_scalar("int32", init=hq * txl.int32(G) + gi)
                        par = cyc & txl.int32(1)
                        npar = par ^ txl.int32(1)
                        hidx = c * txl.int32(HV) + hv
                        lphase("lw-mid")
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with txl.If(elected()), txl.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_Q + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(q_map),
                                    txl.int32(d0),
                                    tok0,
                                    hq,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_K + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(k_map),
                                    txl.int32(d0),
                                    tok0,
                                    hq,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_V + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(v_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mb,
                                )
                        load_beta_lanes_g(bos, hv, n, rows, par)
                        b_in_full.arrive(0)
                        lphase("lw-chunk")
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            TCG["chunk_done"].wait(0, npar)
                            # dAqk/dAkk stay live through the stable intra pass.
                            b_intra_free.wait(0, npar)
                        lphase("l-issue2")
                        with txl.If(elected()), txl.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = txl.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_G + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(eg_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mbe,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_do_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_DO + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(do_map),
                                    txl.int32(d0),
                                    tok0,
                                    hv,
                                    mb,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_h_free.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_H + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(h_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            b_aqk_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_aqk_full.arrive(0, tx_count=A_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                TT[S_AQK].ptr_to(0, 0),
                                txl.address_of(aqk_map),
                                txl.int32(0),
                                tok0,
                                hv,
                                mb,
                            )
                        with txl.If(cyc > txl.int32(1)), txl.Then():
                            b_akk_empty.wait(par, ((cyc >> 1) & txl.int32(1)) ^ txl.int32(1))
                        with txl.If(elected()), txl.Then():
                            b_akk_full.arrive(par, tx_count=A_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            txl.ptx[TMA_LD](
                                TT[S_AKK + par].ptr_to(0, 0),
                                txl.address_of(akk_map),
                                txl.int32(0),
                                tok0,
                                hv,
                                mb,
                            )

                        lphase("lw-qkfree")
                        TCG["xT_done"].wait(0, par)
                        with txl.If(cyc > txl.int32(0)), txl.Then():
                            TCG["dk_done"].wait(0, npar)
                        lphase("l-dhb")
                        with txl.If(elected()), txl.Then():
                            b_dhb_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_dhb_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[DHB + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(dh_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )

                            with txl.If(gi + txl.int32(1) < txl.int32(G)):
                                with txl.Then():
                                    hvn = hv + txl.int32(1)
                                    for tmap in (v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(tmap), txl.int32(d0), tok0, hvn
                                            )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(aqk_map), txl.int32(0), tok0, hvn
                                    )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(akk_map), txl.int32(0), tok0, hvn
                                    )
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(h_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            hidx + txl.int32(1),
                                        )
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(dh_map),
                                            txl.int32(d0),
                                            txl.int32(0),
                                            hidx + txl.int32(1),
                                        )
                        lphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))
                    claim_publish(kk_ + txl.int32(1))
                    txl.assign(kk_, kk_ + txl.int32(1))
                    txl.assign(cur, work_wait(kk_))

            with w10:
                g_masker(txl.int32(0))
            with w11:
                g_masker(txl.int32(1))
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                txl.Cast("uint32", txl.local_scalar("int32", init=tmem_preamble()[0])),
                txl.uint32(TMEM_COLS),
            )

        with txl.If(txl.thread_id() == txl.int32(0)), txl.Then():
            done = txl.local_scalar("int32")
            txl.ptx["atom.acq_rel.gpu.global.add.s32"](
                done, stream_counter.ptr_to([1]), txl.int32(1)
            )
            with txl.If(done == num_ctas - txl.int32(1)), txl.Then():
                txl.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([0]), txl.int32(0))
                txl.ptx["st.release.gpu.global.s32"](stream_counter.ptr_to([1]), txl.int32(0))
                if item_only:
                    txl.ptx["st.release.gpu.global.s32"](
                        range_flags.ptr_to([range_entries]), txl.int32(0)
                    )

    txl.MBarrier._wait = _CUDA_MBAR_WAIT
    return kda_bwd_mega


def make_retry_mega_kernel(HQ: int, HV: int, static_grid=None):
    return make_mega_kernel(HQ, HV, static_grid=static_grid, item_only=True)


AQK_BYTES = CHUNK * CHUNK * 2


def make_native_fused_kernel(H: int, sched_maxp2: int, sched_maxp1: int, static_grid=None):
    txl.MBarrier._wait = _CUDA_MBAR_WAIT
    TM_DH = 0
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2
    SCHED_MAXP2, SCHED_MAXP1 = sched_maxp2, sched_maxp1
    SCHED_STRIDE = 2 + SCHED_MAXP2 + SCHED_MAXP1
    HK = H * D
    HK64 = txl.int64(HK)
    QKVE_BYTES = 4 * CHUNK * D * 2

    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8

    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2 + CHUNK * 8 * 2
    EG_BYTES = CHUNK * D * 2

    @txl.kernel(
        warps=12,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="num_ctas" if static_grid is None else static_grid,
    )
    def kda_bwd_native_fused(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        aqk: txl.gptr[txl.bf16],
        akk: txl.gptr[txl.bf16],
        g: txl.gptr[txl.f32],
        egcache: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        dht: txl.gptr[txl.f32],
        h0: txl.gptr[txl.f32],
        hsnap: txl.gptr[txl.bf16],
        cu_seqlens: txl.gptr[txl.i64],
        flags: txl.gptr[txl.i32],
        sched: txl.gptr[txl.i32],
        dq: txl.gptr[txl.f32],
        dk: txl.gptr[txl.f32],
        dv: txl.gptr[txl.bf16],
        db: txl.gptr[txl.f32],
        dg: txl.gptr[txl.f32],
        dh0: txl.gptr[txl.f32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        g_map: txl.TensorMap,
        eg_map: txl.TensorMap,
        beta_map: txl.TensorMap,
        do_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        akk_map: txl.TensorMap,
        h_map: txl.TensorMap,
        scale: txl.f32,
        num_seqs: txl.i32,
        num_ctas: txl.i32,
        epoch: txl.i32,
        range_flags: txl.gptr[txl.i32],
        range_entries: txl.i32,
    ):
        native_modes = txl.local_scalar("uint32", init=txl.uint32(0))
        native_idx = txl.local_scalar("int32", init=txl.thread_id())
        with txl.While(native_idx < range_entries):
            native_part = txl.local_scalar("uint32")
            txl.ptx.ld.global_.u32(native_part, range_flags.ptr_to([native_idx]))
            txl.assign(native_modes, native_modes | native_part)
            txl.assign(native_idx, native_idx + txl.int32(384))
        native_pred = txl.local_scalar("bool", init=(native_modes & txl.uint32(1)) != txl.uint32(0))
        native_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            native_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(native_pred)
        )
        with txl.If(native_count != txl.uint32(0)), txl.Then():
            txl.Return(txl.int32(0))
        for buf in (q, k, v, beta, aqk, akk, g, do, hsnap, egcache):
            txl.keep_alive(buf.data)
        num_work = num_seqs * txl.int32(H)

        cta = txl.local_scalar("int32", init=txl.Cast("int32", txl.cta_id()))
        sbase = txl.local_scalar("int32", init=cta * txl.int32(SCHED_STRIDE))
        n_p2 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(n_p2, sched.ptr_to([sbase]))
        n_p1 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(n_p1, sched.ptr_to([sbase + txl.int32(1)]))

        def p2_chain(i):
            c = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(c, sched.ptr_to([sbase + txl.int32(2) + i]))
            return c

        def p1_chain(i):
            c = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(c, sched.ptr_to([sbase + txl.int32(2 + SCHED_MAXP2) + i]))
            return c

        sp = txl.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=208)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=88)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        idle = sp.role("idle", warps=[10, 11], group=auxg)

        smem = txl.smem_pool()
        s_tmem = smem.alloc((4,), txl.i32, align=16)
        b_in_full = txl.TMABar(smem, 1)
        b_in_full.init(1)
        b_eg_full = txl.TMABar(smem, 1)
        b_eg_full.init(1)
        b_mid_free = txl.MBarrier(smem, 1)
        b_mid_free.init(256)
        b_do_full = txl.TMABar(smem, 1)
        b_do_full.init(1)
        b_h_full = txl.TMABar(smem, 1)
        b_h_full.init(1)
        b_aqk_full = txl.TMABar(smem, 1)
        b_aqk_full.init(1)
        b_akk_full = txl.TMABar(smem, 2)
        b_akk_full.init(1)
        b_do_empty = txl.TCGen05Bar(smem, 1)
        b_do_empty.init(1)
        b_h_free = txl.MBarrier(smem, 1)
        b_h_free.init(256)
        b_aqk_empty = txl.TCGen05Bar(smem, 1)
        b_aqk_empty.init(1)
        b_akk_empty = txl.TCGen05Bar(smem, 2)
        b_akk_empty.init(1)
        mb_names = [
            "t_early",
            "dhb_ready",
            "zT_ready",
            "vnT_ready",
            "dv2T_ready",
            "dAqk_tile_ready",
            "dAm_ready",
            "X_ready",
            "intra_ready",
            "dv_epi_done",
        ]
        MB = {}
        for nm in mb_names:
            MB[nm] = txl.MBarrier(smem, 1)
            MB[nm].init(256)
        b_dg0_ready = txl.MBarrier(smem, 1)
        b_dg0_ready.init(256)
        b_aqk_masked = txl.MBarrier(smem, 1)
        b_aqk_masked.init(64)

        p_kv = txl.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        b_kvT_done = txl.TCGen05Bar(smem, 1)
        b_kvT_done.init(1)
        b_kv_read = txl.MBarrier(smem, 1)
        b_kv_read.init(256)
        p_akk1 = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_tiles = txl.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=256)
        p_hs = txl.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = txl.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        tc_names = [
            "Z_done",
            "Vn_done",
            "dv2_done",
            "dAqk_done",
            "dk_done",
            "dAs_done",
            "dvb_done",
            "X_done",
            "Y_done",
            "dq2_done",
            "dkt_done",
            "chunk_done",
            "xT_done",
        ]
        TC = {}
        for nm in tc_names:
            TC[nm] = txl.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        TT = smem.alloc((27, 64, 64), txl.bf16, swizzle=txl.SW128B)
        s_beta = smem.alloc((64,), txl.f32, align=16)
        s_beta_in = smem.alloc((CHUNK, 8), txl.bf16, align=128)
        s_dgk = smem.alloc((2, 128), txl.f32, align=16)
        s_cs = smem.alloc((128,), txl.f32, align=16)
        s_beta_g = smem.alloc((2, CHUNK, 8), txl.bf16, align=128)
        s_beta1 = smem.alloc((2, CHUNK), txl.f32, align=16)

        s_ident = smem.alloc((256,), txl.bf16, align=128)

        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_tmem[1]), txl.int32(0))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(txl.thread_id() < txl.int32(256)), txl.Then():
            tid_i = txl.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & txl.int32(15)
            txl.ptx.st.shared.u16(
                s_ident.ptr_to(
                    [
                        (n_i >> 3) * txl.int32(128)
                        + (k_i >> 3) * txl.int32(64)
                        + (n_i & txl.int32(7)) * txl.int32(8)
                        + (k_i & txl.int32(7))
                    ]
                ),
                txl.Cast("uint16", txl.Select(n_i == k_i, txl.int32(0x3F80), txl.int32(0))),
            )
            txl.ptx[FENCE_ASYNC]()
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(s_tmem[0]), txl.uint32(512)
            )
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = txl.alloc_local([1], "uint32")
            txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            def phase(name):
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.range_start(name))

            def phase_end():
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def tmem_preamble():
            tmv = txl.alloc_local([1], "int32")
            txl.ptx.ld.volatile.shared.s32(tmv[0], txl.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def work_coords(work):
            seq = txl.local_scalar("int32", init=work // txl.int32(H))
            head = txl.local_scalar("int32", init=work - seq * txl.int32(H))
            cs = txl.alloc_local([2], "int64")
            txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
            txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + txl.int32(1)]))
            bos = txl.local_scalar("int64", init=cs[0])
            seq_len = txl.local_scalar("int32", init=txl.Cast("int32", cs[1] - cs[0]))
            nch = txl.local_scalar("int32", init=(seq_len + txl.int32(CHUNK - 1)) >> 6)
            return seq, head, bos, seq_len, nch

        def chunk_base(seq):
            cb = txl.local_scalar("int32", init=txl.int32(0))
            with txl.serial(seq) as i:
                cs = txl.alloc_local([2], "int64")
                txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                txl.assign(
                    cb, cb + ((txl.Cast("int32", cs[1] - cs[0]) + txl.int32(CHUNK - 1)) >> 6)
                )
            return cb

        P1_KV, P1_AKK, P1_HS, P1_G, P1_KG, P1_KBG, P1_VB = 0, 8, 9, 13, 21, 23, 25
        G_BYTES = CHUNK * D * 4 + CHUNK * 8 * 2
        TM_H, TM_W, TM_U = 0, 128, 192
        TM_KT, TM_VT = 256, 320

        def bf16_bits_to_f32(u16val):
            return txl.reinterpret("float32", txl.Cast("uint32", u16val) << txl.uint32(16))

        def p1_compute():
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            xg = txl.local_scalar("int32", init=x >> 5)
            xgc = txl.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            st_kv = txl.PipelineState(2, phase=0)
            st_te = txl.PipelineState(1, phase=1)
            st_hs = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(2, phase=0)
            st_w = txl.PipelineState(1, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            p1c = txl.local_scalar("int32", init=txl.int32(0))
            gv = txl.alloc_local([32], "float32")
            kk = txl.alloc_local([32], "float32")
            vv = txl.alloc_local([32], "float32")
            bb = txl.alloc_local([32], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            eg = txl.local_scalar("float32")
            egng = txl.local_scalar("float32")
            bu = txl.local_scalar("uint16")
            ku = txl.local_scalar("uint16")
            vu = txl.local_scalar("uint16")
            phase, phase_end = make_phaser()
            tid_all = txl.local_scalar("int32", init=wr * 32 + lane)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                head64 = txl.Cast("int64", head)
                gcol = txl.local_scalar("int64", init=head64 * txl.int64(D) + x64)

                def h_c0():
                    rows = txl.int32(CHUNK)
                    p_g.full.wait(st_g.stage, st_g.phase)
                    gst = txl.local_scalar("int32", init=P1_G + st_g.stage * txl.int32(4) + xg)
                    with txl.If(lane < txl.int32(8)), txl.Then():
                        btok = wr * txl.int32(8) + lane
                        txl.ptx.ld.shared.u16(
                            bu, s_beta_g.ptr_to([st_g.stage, btok, head & txl.int32(7)])
                        )
                        txl.ptx.st.shared.f32(
                            txl.address_of(s_beta1[st_g.stage, btok]), bf16_bits_to_f32(bu)
                        )
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            bb[4 * u],
                            bb[4 * u + 1],
                            bb[4 * u + 2],
                            bb[4 * u + 3],
                            txl.address_of(s_beta1[st_g.stage, row0 + 4 * u]),
                        )
                    txl.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(st_g.stage)
                    st_g.advance()

                phase("h-c0")
                h_c0()
                with txl.serial(nch) as n:
                    rows = txl.int32(CHUNK)
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK))
                    )
                    txl.ptx.ex2.approx.ftz.f32(egn, gn)
                    phase("hw-kv")
                    b_kvT_done.wait(0, p1c & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("h-kv")
                    kst = txl.local_scalar(
                        "int32", init=P1_KV + st_kv.stage * txl.int32(4) + xs * txl.int32(2)
                    )
                    txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                    for i in range(32):
                        txl.ptx.ld.shared.u16(vu, TT[kst + txl.int32(1)].ptr_to(row0 + i, xr))
                        txl.assign(vv[i], bf16_bits_to_f32(vu))
                    txl.ptx[WAIT_LD]()
                    txl.ptx[FENCE_ASYNC]()
                    p_kv.empty.arrive(st_kv.stage)
                    st_kv.advance()
                    txl.ptx[TC_FENCE_BEFORE]()
                    b_kv_read.arrive(0)
                    txl.assign(p1c, p1c + txl.int32(1))
                    phase("hw-tiles")
                    p_tiles.empty.wait(0, st_te.phase)
                    st_te.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-tiles")
                    for u in range(4):
                        wkg = txl.alloc_local([4], "uint32")
                        wkbg = txl.alloc_local([4], "uint32")
                        wvb = txl.alloc_local([4], "uint32")
                        vals = txl.alloc_local([24], "float32")
                        for e in range(8):
                            i = 8 * u + e
                            txl.ptx.ex2.approx.ftz.f32(eg, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(egng, gn - gv[i])
                            txl.ptx.cvt.rn.bf16.f32(bu, eg)
                            egidx = (tok0 + txl.Cast("int64", row0 + txl.int32(i))) * HK64 + gcol
                            txl.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx]), bu)
                            txl.assign(vals[e], kk[i] * egng)
                            txl.assign(vals[8 + e], kk[i] * bb[i] * eg)
                            txl.assign(vals[16 + e], vv[i] * bb[i])
                        for p in range(4):
                            pack_bf16x2(wkg[p], vals[2 * p], vals[2 * p + 1])
                            pack_bf16x2(wkbg[p], vals[8 + 2 * p], vals[8 + 2 * p + 1])
                            pack_bf16x2(wvb[p], vals[16 + 2 * p], vals[16 + 2 * p + 1])
                        col = row0 + 8 * u
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KG + xs].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KBG + xs].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_VB + xs].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3]
                        )

                    txl.ptx[FENCE_ASYNC]()
                    p_tiles.full.arrive(0)
                    phase("hw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("h-decay")
                    hc0 = wg * 64
                    hsst = txl.local_scalar(
                        "int32", init=P1_HS + st_hs.stage * txl.int32(4) + wg * txl.int32(2) + xs
                    )
                    with txl.If(n == txl.int32(0)):
                        with txl.Then():
                            h0base = (
                                (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D)
                                + x64
                            ) * txl.int64(D) + txl.Cast("int64", hc0)
                            for m in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m + i] for i in range(8)),
                                    h0.ptr_to([h0base + txl.int64(8 * m)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[hsst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    for p in range(32):
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair,
                            txl.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                            txl.cuda.make_float2(egn, egn),
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("hw-W")
                    p_w.full.wait(0, st_w.phase)
                    st_w.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-wT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_W + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KBG + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(0)
                    phase("h-c0")
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        h_c0()
                    phase("hw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-vnT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_U + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_VB + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)

                    with txl.If(elected()), txl.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    phase_end()

            p_tiles.empty.wait(0, st_te.phase)
            txl.ptx[TC_FENCE_AFTER]()

        def p1_mma():
            tm = tmem_preamble()

            bd1 = txl.alloc_local([1], "uint64")
            zq1 = txl.alloc_local([1], "int32")
            op_kbg_k = Op(bd1, P1_KBG, 128, 64, "k")
            op_vb_k = Op(bd1, P1_VB, 128, 64, "k")
            op_kg_k = Op(bd1, P1_KG, 128, 64, "k")
            op_akk1_k = Op(bd1, P1_AKK, 64, 64, "k")
            op_hs_mn = Op(bd1, P1_HS, 128, 128, "mn")
            op_w_mn = Op(bd1, P1_KBG, 128, 128, "mn")
            op_kraw = Op(bd1, P1_KV, 128, 64, "mn")
            ID_T1 = idesc(128, 16, ta=1)
            bdI1 = txl.alloc_local([1], "uint64")
            txl.cuda.tcgen05.encode_matrix_descriptor(
                txl.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
            )
            st_kv1 = txl.PipelineState(2, phase=0)
            p1m = txl.local_scalar("int32", init=txl.int32(0))
            ID_M128N64 = idesc(128, 64)

            def kv_transpose():
                mphase("hmw-kv")
                p_kv.full.wait(st_kv1.stage, st_kv1.phase)
                b_kv_read.wait(0, (p1m & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                mphase("hm-kvT")
                kv_u = txl.local_scalar(
                    "uint64",
                    init=txl.Cast("uint64", st_kv1.stage) * txl.uint64(4 * UNITS_PER_STAGE),
                )
                with txl.If(elected()), txl.Then():
                    for j in range(4):
                        txl.ptx[MMA_SS](
                            txl.Cast("uint32", tm[0] + TM_KT + 16 * j),
                            op_kraw.desc(j, kv_u),
                            bdI1[0],
                            txl.uint32(ID_T1),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.ptx.pred(0),
                        )
                    b_kvT_done.arrive(0)
                st_kv1.advance()
                txl.assign(p1m, p1m + txl.int32(1))

            ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
            ID_HUPD = idesc(128, 128)
            st_tiles = txl.PipelineState(1, phase=0)
            st_akk = txl.PipelineState(1, phase=0)
            st_hs = txl.PipelineState(1, phase=0)
            st_w = txl.PipelineState(1, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            mphase, mphase_end = make_phaser()
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bd1[0]),
                    TT[zq1[0]].ptr_to(0, 0),
                    ldo=Op.LBO_BASE,
                    sdo=SBO_UNITS,
                    swizzle=txl.SW128B.value,
                )
                kv_transpose()
                with txl.serial(nch) as n:
                    txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                    txl.cuda.tcgen05.encode_matrix_descriptor(
                        txl.address_of(bd1[0]),
                        TT[zq1[0]].ptr_to(0, 0),
                        ldo=Op.LBO_BASE,
                        sdo=SBO_UNITS,
                        swizzle=txl.SW128B.value,
                    )
                    mphase("hmw-tiles")
                    p_tiles.full.wait(0, st_tiles.phase)
                    p_akk1.full.wait(st_akk.stage, st_akk.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    akk_u = txl.local_scalar(
                        "uint64",
                        init=txl.Cast("uint64", st_akk.stage) * txl.uint64(UNITS_PER_STAGE),
                    )
                    mphase("hm-WU")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_W, op_kbg_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_w.full.arrive(0)

                        mma_chain(tm, TM_U, op_vb_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_akk1.empty.arrive(st_akk.stage)
                    st_akk.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        kv_transpose()
                    mphase("hmw-wT")
                    p_w.empty.wait(0, st_w.phase)
                    st_w.advance()
                    mphase("hmw-hs")
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    hs_u = txl.local_scalar(
                        "uint64",
                        init=txl.Cast("uint64", st_hs.stage) * txl.uint64(4 * UNITS_PER_STAGE),
                    )
                    mphase("hm-Vn")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_U, op_hs_mn, op_w_mn, ID_VN, True, a_units=hs_u)
                        p_vn.full.arrive(0)
                    st_hs.advance()
                    mphase("hmw-vnT")
                    p_vn.empty.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("hm-hupd")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_H, op_kg_k, op_vb_k, ID_HUPD, True)
                        p_tiles.empty.arrive(0)
                    st_tiles.advance()
                    mphase_end()

        def p1_loader():
            st_kv = txl.PipelineState(2, phase=1)
            st_akk = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(2, phase=1)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                head8 = txl.local_scalar("int32", init=head >> txl.int32(3))
                with txl.serial(nch) as n:
                    tok0 = bos32 + n * txl.int32(CHUNK)
                    p_g.empty.wait(st_g.stage, st_g.phase)
                    with txl.If(elected()), txl.Then():
                        p_g.full.arrive(st_g.stage, tx_count=G_BYTES)
                        mbg = txl.cuda.cvta_generic_to_shared(p_g.full.ptr_to([st_g.stage]))
                        for j in range(4):
                            txl.ptx[TMA_LD](
                                TT[P1_G + st_g.stage * txl.int32(4) + txl.int32(j)].ptr_to(0, 0),
                                txl.address_of(g_map),
                                txl.int32(32 * j),
                                tok0,
                                head,
                                mbg,
                            )
                        txl.ptx[TMA_LD](
                            s_beta_g.ptr_to([st_g.stage, 0, 0]),
                            txl.address_of(beta_map),
                            txl.int32(0),
                            tok0,
                            head8,
                            mbg,
                        )
                    st_g.advance()
                    p_kv.empty.wait(st_kv.stage, st_kv.phase)
                    with txl.If(elected()), txl.Then():
                        p_kv.full.arrive(st_kv.stage, tx_count=KV_BYTES)
                        mb = txl.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([st_kv.stage]))
                        for tmap, half in ((k_map, 0), (v_map, 1)):
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[
                                        P1_KV
                                        + st_kv.stage * txl.int32(4)
                                        + txl.int32((d0 // 64) * 2 + half)
                                    ].ptr_to(0, 0),
                                    txl.address_of(tmap),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        with txl.If(n + txl.int32(1) < nch), txl.Then():
                            for tmap in (k_map, v_map):
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(tmap),
                                        txl.int32(d0),
                                        tok0 + txl.int32(CHUNK),
                                        head,
                                    )
                            for d0 in (0, 32, 64, 96):
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(g_map),
                                    txl.int32(d0),
                                    tok0 + txl.int32(CHUNK),
                                    head,
                                )
                            txl.ptx[TMA_PREFETCH](
                                txl.address_of(akk_map), txl.int32(0), tok0 + txl.int32(CHUNK), head
                            )
                    st_kv.advance()
                    p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                    with txl.If(elected()), txl.Then():
                        p_akk1.full.arrive(st_akk.stage, tx_count=AQK_BYTES)
                        mb2 = txl.cuda.cvta_generic_to_shared(p_akk1.full.ptr_to([st_akk.stage]))
                        txl.ptx[TMA_LD](
                            TT[P1_AKK + st_akk.stage].ptr_to(0, 0),
                            txl.address_of(akk_map),
                            txl.int32(0),
                            tok0,
                            head,
                            mb2,
                        )
                    st_akk.advance()

        def p1_storer():
            st_hs = txl.PipelineState(1, phase=0)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                cb = chunk_base(seq)
                with txl.serial(nch) as n:
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    with txl.If(elected()), txl.Then():
                        txl.ptx[FENCE_ASYNC]()
                        idx = (cb + n) * txl.int32(H) + head
                        for d0 in (0, 64):
                            txl.ptx[TMA_ST](
                                txl.address_of(h_map),
                                txl.int32(d0),
                                txl.int32(0),
                                idx,
                                TT[
                                    P1_HS + st_hs.stage * txl.int32(4) + txl.int32((d0 // 64) * 2)
                                ].ptr_to(0, 0),
                            )
                        txl.ptx[BULK_COMMIT]()
                        txl.ptx[BULK_WAIT_READ](0)
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()

                with txl.If(elected()), txl.Then():
                    txl.ptx[BULK_WAIT](0)
                    txl.ptx["fence.proxy.async.global"]()
                    txl.ptx.st.release.gpu.global_.s32(flags.ptr_to([chain]), epoch)

        with cg:
            p1_compute()
            txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            tid_all = txl.local_scalar("int32", init=wr * 32 + lane)
            phalf = txl.local_scalar("int32", init=x & 1)
            pcol = txl.local_scalar("int32", init=x & ~1)
            prow0 = txl.local_scalar("int32", init=row0 + phalf * 16)
            ps = txl.local_scalar("int32", init=pcol >> 6)
            pr = txl.local_scalar("int32", init=pcol & 63)
            is_odd = phalf != txl.int32(0)
            cyc = txl.local_scalar("int32", init=txl.int32(0))

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                txl.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                txl.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                txl.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                """Write this thread's row x, columns [col0, col0 + 8*nunits) of the [128][64] tile at stage0/stage0+1."""
                for u in range(nunits):
                    txl.ptx["st.shared.v4.b32"](
                        TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                        words[wbase + 4 * u],
                        words[wbase + 4 * u + 1],
                        words[wbase + 4 * u + 2],
                        words[wbase + 4 * u + 3],
                    )

            def st_pair_rows(stage0, words):
                """Pair layout: rows pcol and pcol+1, columns [prow0, prow0+16): words[0:8] row pcol, words[8:16] row pcol+1."""
                for r in range(2):
                    for u in range(2):
                        txl.ptx["st.shared.v4.b32"](
                            TT[stage0 + ps].ptr_to(pr + r, prow0 + 8 * u),
                            words[8 * r + 4 * u],
                            words[8 * r + 4 * u + 1],
                            words[8 * r + 4 * u + 2],
                            words[8 * r + 4 * u + 3],
                        )

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            def bar_wg():
                txl.ptx.bar.sync(txl.uint32(2) + txl.Cast("uint32", wg), txl.uint32(128))

            def twait(nm):
                TC[nm].wait(0, cyc & txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()

                txl.ptx[FENCE_ASYNC]()
                txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))

            def marrive(nm):
                txl.ptx[TC_FENCE_BEFORE]()
                MB[nm].arrive(0)

            def lo(w):
                return txl.reinterpret("float32", w << txl.uint32(16))

            def hi(w):
                return txl.reinterpret("float32", w & txl.uint32(0xFFFF0000))

            def shfl_xor1(val):
                r = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(
                    r,
                    txl.reinterpret("uint32", val),
                    txl.uint32(1),
                    txl.uint32(0x1F),
                    txl.uint32(0xFFFFFFFF),
                )
                return txl.reinterpret("float32", r)

            def q_ptr(c, col):
                return TT[ST_Q + (col >> 6)].ptr_to(c, col & 63)

            def k_ptr(c, col):
                return TT[ST_K + (col >> 6)].ptr_to(c, col & 63)

            def v_ptr(c, col):
                return TT[ST_V + (col >> 6)].ptr_to(c, col & 63)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                """Load this warp's 32x32 block as four groups of four 8x8 fragments."""
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o],
                            frag[o + 1],
                            frag[o + 2],
                            frag[o + 3],
                            tile.m8n8x4(row0 + txl.int32(16 * rb), col0 + txl.int32(16 * cb), lane),
                        )

            def store_transpose_frag(base, frag):
                """Transpose those fragments in place, turning [token,channel] into [channel,token]."""
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                mm = lane >> txl.int32(3)
                jj = lane & txl.int32(7)
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)

                        ptr = tile.ptr_to(
                            col0 + txl.int32(16 * cb) + (mm >> txl.int32(1)) * txl.int32(8) + jj,
                            row0 + txl.int32(16 * rb) + (mm & txl.int32(1)) * txl.int32(8),
                        )
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )

            enA = txl.alloc_local([16], "float32")
            enB = txl.alloc_local([16], "float32")
            egcw = txl.alloc_local([16], "uint32")
            t4 = txl.alloc_local([4], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            dgv = txl.alloc_local([32], "float32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            dgk = txl.local_scalar("float32")
            dgk_k = txl.local_scalar("float32")
            t0 = txl.local_scalar("float32")
            t1 = txl.local_scalar("float32")
            u16 = txl.local_scalar("uint16")
            u16b = txl.local_scalar("uint16")

            def ex2(dst, val):
                txl.ptx.ex2.approx.ftz.f32(dst, val)

            def rcp(dst, val):
                txl.ptx.rcp.approx.ftz.f32(dst, val)

            def load_u16_pair(words, i, ptr):
                if i % 2 == 0:
                    txl.ptx.ld.global_.nc.u16(u16, ptr)
                else:
                    txl.ptx.ld.global_.nc.u16(u16b, ptr)
                    txl.ptx.mov.b32(words[i >> 1], u16, u16b)

            def load_u16_pair_sh(words, i, ptr):
                if i % 2 == 0:
                    txl.ptx.ld.shared.u16(u16, ptr)
                else:
                    txl.ptx.ld.shared.u16(u16b, ptr)
                    txl.ptx.mov.b32(words[i >> 1], u16, u16b)

            def shfl_xor1_u32(val):
                r = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(
                    r, val, txl.uint32(1), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
                )
                return r

            def gcol_ptr(tensor, i):
                """Global pointer to row (row0+i) of this thread's column (clamped to the last valid row)."""
                tokc = tok0 + txl.Cast("int64", txl.min(row0 + txl.int32(i), rows - txl.int32(1)))
                return tensor.ptr_to([tokc * HK64 + gcol])

            def s_beta_row(c):
                b = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b, txl.address_of(s_beta[c]))
                return b

            def wsel(cond, a, b):
                return txl.Select(cond, a, b)

            phase, phase_end = make_phaser()

            with txl.serial(n_p2) as i2:
                work = txl.local_scalar("int32", init=p2_chain(i2))
                seq, head, bos, seq_len, nch = work_coords(work)
                head64 = txl.Cast("int64", head)
                gcol = txl.local_scalar("int64", init=head64 * txl.int64(D) + x64)
                with txl.serial(nch) as rn:
                    n = nch - txl.int32(1) - rn
                    par = cyc & txl.int32(1)

                    rows = txl.int32(CHUNK)
                    last = txl.int32(CHUNK - 1)
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK))
                    )
                    x_base = txl.local_scalar(
                        "int64", init=(tok0 + txl.Cast("int64", row0)) * HK64 + gcol
                    )

                    phase("w-in")
                    b_in_full.wait(0, par)
                    b_eg_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")
                    with txl.If(lane < txl.int32(8)), txl.Then():
                        btok = wr * txl.int32(8) + lane
                        txl.ptx.ld.shared.u16(u16, s_beta_in.ptr_to([btok, head & txl.int32(7)]))
                        txl.ptx.st.shared.f32(
                            txl.address_of(s_beta[btok]), lo(txl.Cast("uint32", u16))
                        )

                    egf = txl.alloc_local([32], "float32")
                    xf = txl.alloc_local([32], "float32")
                    qw = txl.alloc_local([16], "uint32")
                    kw = txl.alloc_local([16], "uint32")
                    t3w = txl.alloc_local([16], "uint32")
                    qc = txl.alloc_local([16], "uint32")
                    kc = txl.alloc_local([16], "uint32")
                    vc = txl.alloc_local([16], "uint32")
                    prep0 = txl.local_scalar("uint64")
                    prep1 = txl.local_scalar("uint64")
                    scale_pair = txl.local_scalar("uint64", init=txl.cuda.make_float2(scale, scale))
                    bpair = txl.alloc_local([2], "float32")

                    ld32(xf, S2 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    bar_all()
                    for half in range(2):
                        vb32 = txl.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[row0 + i])
                            )
                            txl.assign(vb32[2 * p], xf[i] * bpair[0])
                            txl.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        txl.ptx[TC_ST16](
                            tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16))
                        )
                    txl.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)

                    ld4(t4, S1 + 60)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    txl.assign(egn, t4[3])
                    for i in range(16):
                        txl.ptx["mul.rn.f32x2"](
                            prep0,
                            txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                            txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                        )
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        pack_bf16x2(qw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        pack_bf16x2(egcw[i], egf[2 * i], egf[2 * i + 1])

                    for half in range(2):
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)),
                            tmem_at(S4 + wg * 32 + 16 * half),
                        )
                        txl.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[row0 + 2 * i])
                            )
                            rcp(t0, egf[2 * i])
                            rcp(t1, egf[2 * i + 1])
                            txl.ptx["mul.rn.f32x2"](
                                prep0,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(t0, t1),
                            )
                            pack_bf16x2(kw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                            txl.ptx["mul.rn.f32x2"](
                                prep1,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep1, prep1, txl.cuda.make_float2(bpair[0], bpair[1])
                            )
                            pack_bf16x2(t3w[i], txl.cuda.float2_x(prep1), txl.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TC["chunk_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("t_early")
                    phase("c1c")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("c2")
                    dgk2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    hst = txl.local_scalar("int32", init=wg * 2 + xs)
                    dbase = (
                        (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D) + x64
                    ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                    with txl.If(rn == txl.int32(0)):
                        with txl.Then():
                            for m in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m + i] for i in range(8)),
                                    dht.ptr_to([dbase + txl.int64(8 * m)]),
                                )
                        with txl.Else():
                            ld32(acc, TM_DH + wg * 64)
                            ld32(acc, TM_DH + wg * 64 + 32, 32)
                            txl.ptx[WAIT_LD]()
                    for half in range(2):
                        hc = wg * 64 + 32 * half
                        a0 = 32 * half

                        for p in range(16):
                            dpair = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                dpair,
                                txl.cuda.make_float2(acc[a0 + 2 * p], acc[a0 + 2 * p + 1]),
                                txl.cuda.make_float2(egn, egn),
                            )
                            txl.assign(acc[a0 + 2 * p], txl.cuda.float2_x(dpair))
                            txl.assign(acc[a0 + 2 * p + 1], txl.cuda.float2_y(dpair))
                        for u in range(4):
                            txl.ptx["ld.shared.v4.b32"](
                                wds[0],
                                wds[1],
                                wds[2],
                                wds[3],
                                TT[S_H + hst].ptr_to(xr, 32 * half + 8 * u),
                            )
                            for p in range(4):
                                txl.ptx["fma.rn.f32x2"](
                                    dgk2,
                                    txl.cuda.make_float2(lo(wds[p]), hi(wds[p])),
                                    txl.cuda.make_float2(
                                        acc[a0 + 8 * u + 2 * p], acc[a0 + 8 * u + 2 * p + 1]
                                    ),
                                    dgk2,
                                )
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[a0 + 2 * p], acc[a0 + 2 * p + 1])
                        for u in range(4):
                            txl.ptx["st.shared.v4.b32"](
                                TT[DHB + hst].ptr_to(xr, 32 * half + 8 * u),
                                wds[4 * u],
                                wds[4 * u + 1],
                                wds[4 * u + 2],
                                wds[4 * u + 3],
                            )
                        txl.ptx[TC_ST32](tmem_at(TM_DH + hc), *(acc[a0 + i] for i in range(32)))
                    txl.assign(dgk, txl.cuda.float2_x(dgk2) + txl.cuda.float2_y(dgk2))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dhb_ready")

                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")

                    def readout64(slot, stage, mask, scale_by=None, negate=False):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with txl.If(lane < txl.int32(16)), txl.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if scale_by is not None:
                                        val = val * scale_by
                                    if negate:
                                        val = txl.float32(0.0) - val
                                    vv2.append(
                                        val
                                        if mask is None
                                        else txl.Select(mask(cc, jj), val, txl.float32(0.0))
                                    )
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                txl.ptx["st.shared.v4.b32"](
                                    TT[stage].ptr_to(cc, row0 + 8 * u),
                                    wds[4 * u],
                                    wds[4 * u + 1],
                                    wds[4 * u + 2],
                                    wds[4 * u + 3],
                                )
                        txl.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        txl.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        txl.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> txl.int32(2))
                        cc1 = cc0 + txl.int32(8)
                        for rep in range(4):
                            jj0 = row0 + txl.int32(8 * rep) + (lane & txl.int32(3)) * txl.int32(2)
                            jj1 = jj0 + txl.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = txl.float32(0.0) - v00
                                v01 = txl.float32(0.0) - v01
                                v10 = txl.float32(0.0) - v10
                                v11 = txl.float32(0.0) - v11
                            if mask is not None:
                                v00 = txl.Select(mask(cc0, jj0), v00, txl.float32(0.0))
                                v01 = txl.Select(mask(cc0, jj1), v01, txl.float32(0.0))
                                v10 = txl.Select(mask(cc1, jj0), v10, txl.float32(0.0))
                                v11 = txl.Select(mask(cc1, jj1), v11, txl.float32(0.0))
                            pack_bf16x2(wds[2 * rep], v00, v01)
                            pack_bf16x2(wds[2 * rep + 1], v10, v11)
                        tile = TT[stage]
                        for half in range(2):
                            txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(
                                    quad * txl.int32(16), row0 + txl.int32(16 * half), lane
                                ),
                                wds[4 * half],
                                wds[4 * half + 1],
                                wds[4 * half + 2],
                                wds[4 * half + 3],
                            )
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")

                    readout64_half(S4, DAM, lambda cc, jj: jj < cc)
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")

                    readout64_half(S1, T5, lambda cc, jj: jj <= cc)
                    marrive("dAqk_tile_ready")
                    twait("dk_done")
                    phase("w-dvb")
                    twait("dvb_done")
                    phase("passA")

                    pbx = txl.local_scalar("int32", init=txl.int32(PB0) + wg * txl.int32(PB1 - PB0))

                    def pass_a(full):
                        assert full
                        pa_acc = txl.local_scalar("uint64")
                        pa_v = txl.local_scalar("uint64")
                        pa_db = txl.local_scalar("uint64")
                        pa_dv = txl.local_scalar("uint64")
                        pa_word = txl.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = txl.alloc_local([4], "uint32")
                            txl.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3], TT[ST_V + xs].ptr_to(xr, row0 + 8 * b)
                            )
                            dbp = txl.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                txl.assign(
                                    pa_acc,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                )
                                txl.assign(pa_v, txl.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                txl.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                txl.assign(dbp[2 * p], txl.cuda.float2_x(pa_db))
                                txl.assign(dbp[2 * p + 1], txl.cuda.float2_y(pa_db))
                                txl.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], txl.address_of(s_beta[row0 + i])
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, txl.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(
                                    pa_word, txl.cuda.float2_x(pa_dv), txl.cuda.float2_y(pa_dv)
                                )
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.Cast("uint16", pa_word),
                                )
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.Cast("uint16", pa_word >> txl.uint32(16)),
                                )

                            for e in range(8):
                                i = 8 * b + e
                                txl.ptx.st.shared.f32(
                                    TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e]
                                )

                            dvw = txl.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            txl.ptx["st.shared.v4.b32"](
                                TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                dvw[0],
                                dvw[1],
                                dvw[2],
                                dvw[3],
                            )
                            if b < 3:
                                txl.ptx[WAIT_LD]()

                    pass_a(True)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    readout64_half(S1, T6, None)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & txl.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & txl.int32(1)) * 32 + lane
                    dsum_v = txl.local_scalar("float32", init=txl.float32(0.0))
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(dsum_v, dsum_v + ((t4[0] + t4[1]) + (t4[2] + t4[3])))

                    txl.ptx[FENCE_ASYNC]()
                    b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    readout64_half(S2, T5 + 1, lambda cc, jj: jj < cc, negate=True)
                    marrive("intra_ready")

                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi")
                    dgk_k2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def epilogue(full):
                        assert full
                        pair0 = txl.local_scalar("uint64")
                        pair1 = txl.local_scalar("uint64")
                        pair2 = txl.local_scalar("uint64")
                        pair3 = txl.local_scalar("uint64")
                        pair4 = txl.local_scalar("uint64")
                        pair5 = txl.local_scalar("uint64")

                        q_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p

                                # Consume raw bf16 diagonal bits from the helper copy.
                                txl.ptx.ld.shared.u16(u16, s_beta1.ptr_to([0, row0 + i]))
                                txl.assign(enA[i >> 1], bf16_bits_to_f32(u16))
                                txl.ptx.ld.shared.u16(u16, s_beta1.ptr_to([0, row0 + i + 1]))
                                txl.assign(enB[i >> 1], bf16_bits_to_f32(u16))
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                    txl.cuda.make_float2(scale, scale),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair0,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.cuda.float2_x(pair0),
                                )
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.cuda.float2_y(pair0),
                                )
                                # Match the q MMA's bf16-diagonal * bf16-(k/eg)
                                # factorization, then remove it before multiplying by q.
                                txl.ptx["mul.rn.f32x2"](
                                    pair4,
                                    txl.cuda.make_float2(enA[i >> 1], enB[i >> 1]),
                                    txl.cuda.make_float2(lo(kw[i >> 1]), hi(kw[i >> 1])),
                                )
                                txl.ptx["mul.rn.f32x2"](pair4, pair4, pair1)
                                txl.ptx["sub.rn.f32x2"](pair5, pair0, pair4)
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])),
                                    pair5,
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair1))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair1))
                            if b < 3:
                                txl.ptx[WAIT_LD]()
                        twait("dkt_done")

                        dbx = 2 * wg
                        k_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                rcp(t0, lo(egcw[i >> 1]))
                                rcp(t1, hi(egcw[i >> 1]))
                                txl.assign(pair0, txl.cuda.make_float2(t0, t1))
                                txl.assign(
                                    pair1, txl.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1]))
                                )
                                txl.ptx["add.rn.f32x2"](
                                    pair2,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    txl.cuda.make_float2(
                                        acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]
                                    ),
                                )
                                txl.ptx["mul.rn.f32x2"](pair2, pair2, pair0)
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    txl.cuda.make_float2(
                                        acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]
                                    ),
                                    txl.cuda.make_float2(lo(egcw[i >> 1]), hi(egcw[i >> 1])),
                                )
                                txl.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                txl.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane),
                                    txl.cuda.float2_x(pair4),
                                )
                                txl.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(
                                        4 * ((i + 1) & 15) + quad, 2 * lane
                                    ),
                                    txl.cuda.float2_y(pair4),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair4,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair0,
                                )
                                txl.ptx["mul.rn.f32x2"](pair5, pair1, pair4)
                                txl.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                beta_pair = txl.cuda.make_float2(
                                    s_beta_row(row0 + i), s_beta_row(row0 + i + 1)
                                )
                                txl.ptx["fma.rn.f32x2"](pair5, pair3, beta_pair, pair2)
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.cuda.float2_x(pair5),
                                )
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.cuda.float2_y(pair5),
                                )
                                txl.ptx["fma.rn.f32x2"](
                                    pair3,
                                    pair2,
                                    txl.cuda.make_float2(txl.float32(-2.0), txl.float32(-2.0)),
                                    pair5,
                                )
                                # Match and remove the k MMA's bf16-diagonal *
                                # bf16-(q*eg*scale), including its reciprocal factor.
                                txl.ptx["mul.rn.f32x2"](
                                    pair4,
                                    txl.cuda.make_float2(enA[i >> 1], enB[i >> 1]),
                                    txl.cuda.make_float2(lo(qw[i >> 1]), hi(qw[i >> 1])),
                                )
                                txl.ptx["mul.rn.f32x2"](pair4, pair4, pair0)
                                txl.ptx["add.rn.f32x2"](pair3, pair3, pair4)
                                txl.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, txl.cuda.make_float2(dgv[i], dgv[i + 1])
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair5))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair5))
                            if b < 7:
                                txl.ptx[WAIT_LD]()

                        txl.assign(dgk_k, txl.cuda.float2_x(dgk_k2) + txl.cuda.float2_y(dgk_k2))
                        bar_wg()
                        dsum = txl.local_scalar("float32", init=dsum_v)
                        for u in range(8):
                            txl.ptx["ld.shared.v4.f32"](
                                t4[0],
                                t4[1],
                                t4[2],
                                t4[3],
                                TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u),
                            )
                            txl.assign(dsum, dsum + ((t4[0] + t4[1]) + (t4[2] + t4[3])))
                        txl.ptx[FENCE_ASYNC]()
                        b_h_free.arrive(0)
                        for s in (1, 2):
                            r = txl.local_scalar("uint32")
                            txl.ptx.shfl_sync.bfly.b32(
                                r,
                                txl.reinterpret("uint32", dsum),
                                txl.uint32(s),
                                txl.uint32(0x1F),
                                txl.uint32(0xFFFFFFFF),
                            )
                            txl.assign(dsum, dsum + txl.reinterpret("float32", r))
                        with txl.If(tq == txl.int32(0)), txl.Then():
                            txl.ptx["st.global.L1::no_allocate.f32"](
                                db.ptr_to(
                                    [(tok0 + txl.Cast("int64", row0 + ti)) * txl.int64(H) + head64]
                                ),
                                dsum,
                            )

                    epilogue(True)
                    phase("cumsum")

                    for i in range(30, -1, -1):
                        txl.assign(dgv[i], dgv[i] + dgv[i + 1])
                    txl.ptx.st.shared.f32(
                        txl.address_of(s_dgk[wg, x]),
                        dgk + dgk_k + txl.Select(wg == txl.int32(0), txl.float32(0.0), dgv[0]),
                    )
                    b_dg0_ready.arrive(0)
                    b_dg0_ready.wait(0, cyc & txl.int32(1))
                    txl.ptx.ld.shared.f32(t0, txl.address_of(s_dgk[txl.int32(1) - wg, x]))
                    txl.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        txl.assign(dgv[i], dgv[i] + t1)
                    for i in range(32):
                        txl.ptx["st.global.L1::no_allocate.f32"](
                            dg.ptr_to([x_base + txl.int64(i * HK)]), dgv[i]
                        )
                    phase_end()
                    txl.assign(cyc, cyc + txl.int32(1))

                phase("dh0")
                TC["chunk_done"].wait(0, (cyc & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                ld32(acc, TM_DH + wg * 64)
                ld32(acc, TM_DH + wg * 64 + 32, 32)
                txl.ptx[WAIT_LD]()
                obase = (
                    (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D) + x64
                ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                for m in range(8):
                    txl.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + txl.int64(8 * m)]), *(acc[8 * m + i] for i in range(8))
                    )
                phase_end()

        with auxg:
            with mma:
                p1_mma()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                tm = tmem_preamble()
                cyc = txl.local_scalar("int32", init=txl.int32(0))

                def mwait(nm):
                    MB[nm].wait(0, cyc & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()

                bd = txl.alloc_local([1], "uint64")
                zq = txl.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T1mn = Op(bd, T1, 128, 128, "mn")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T2mn = Op(bd, T2, 128, 128, "mn")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_T5k = Op(bd, T5, 128, 64, "k")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBk = Op(bd, DVB, 128, 64, "k")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                ID_128x128 = idesc(128, 128)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)

                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )

                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with txl.serial(nch) as rn:
                        par = cyc & txl.int32(1)
                        txl.ptx.ld.volatile.shared.s32(zq[0], txl.address_of(s_tmem[1]))
                        txl.cuda.tcgen05.encode_matrix_descriptor(
                            txl.address_of(bd[0]),
                            TT[zq[0]].ptr_to(0, 0),
                            ldo=Op.LBO_BASE,
                            sdo=SBO_UNITS,
                            swizzle=txl.SW128B.value,
                        )
                        akk_u = txl.local_scalar(
                            "uint64", init=txl.Cast("uint64", par) * txl.uint64(UNITS_PER_STAGE)
                        )
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)

                        b_h_free.wait(0, par ^ txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with txl.If(elected()), txl.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    txl.ptx[MMA_SS](
                                        txl.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        txl.uint32(ID_T),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.ptx.pred(0),
                                    )
                            TC["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                        b_h_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TC["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        mwait("dhb_ready")
                        mphase("m-dv2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_DHBmn, op_T2mn, ID_128x64_TATB, True)
                            TC["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TC["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TC["dAs_done"].arrive(0)
                            mma_chain(
                                tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u
                            )
                            TC["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TC["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TC["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        # T5 is valid here and this helper warp has a separate
                        # register budget.  Capture two diagonal entries per lane
                        # before the aliased shared tile is released for reuse.
                        diag_bits = txl.local_scalar("uint16")
                        diag_lane = txl.lane_id()
                        for diag_half in range(2):
                            diag_i = diag_lane + txl.int32(32 * diag_half)
                            txl.ptx.ld.shared.u16(diag_bits, TT[T5].ptr_to(diag_i, diag_i))
                            txl.ptx.st.shared.b16(s_beta1.ptr_to([0, diag_i]), diag_bits)
                        txl.ptx[FENCE_ASYNC]()
                        mphase("m-X")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TC["X_done"].arrive(0)

                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            TC["dq2_done"].arrive(0)
                            mma_chain(tm, TM_DH, op_T1k, op_do_mn64, ID_128x128_TB, True)
                            b_do_empty.arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with txl.If(elected()), txl.Then():
                            mma_chain(
                                tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u
                            )
                            TC["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dk2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                        mphase("m-dkt")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            TC["dkt_done"].arrive(0)

                            mma_chain(tm, TM_DH, op_T3k, op_DVBk, ID_128x128_NB, True)
                            TC["chunk_done"].arrive(0)
                        mphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (
                        q_map,
                        k_map,
                        v_map,
                        g_map,
                        eg_map,
                        beta_map,
                        do_map,
                        aqk_map,
                        akk_map,
                        h_map,
                    ):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                p1_loader()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    cb = chunk_base(seq)
                    head8 = txl.local_scalar("int32", init=head >> txl.int32(3))

                    lphase("lw-flag")
                    with txl.If(elected()), txl.Then():
                        # The epoch flag is a declared synchronization word.
                        fl = txl.local_scalar("int32", init=txl.int32(0))
                        txl.cuda.wait_until(fl, flags.ptr_to([work]), fl == epoch, scope="gpu")
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    txl.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with txl.serial(nch) as rn:
                        n = nch - txl.int32(1) - rn
                        par = cyc & txl.int32(1)
                        npar = par ^ txl.int32(1)
                        tok0 = bos32 + n * txl.int32(CHUNK)
                        hidx = (cb + n) * txl.int32(H) + head

                        lphase("lw-mid")
                        b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with txl.If(elected()), txl.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                s_beta_in.ptr_to([0, 0]),
                                txl.address_of(beta_map),
                                txl.int32(0),
                                tok0,
                                head8,
                                mb,
                            )
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_Q + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(q_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_K + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(k_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_V + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(v_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        lphase("lw-chunk")
                        TC["chunk_done"].wait(0, npar)
                        lphase("l-issue-eg")
                        with txl.If(elected()), txl.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = txl.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_G + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(eg_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mbe,
                                )
                        b_do_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_DO + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(do_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        b_h_free.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_H + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(h_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )
                        b_aqk_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_aqk_full.arrive(0, tx_count=AQK_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                TT[S_AQK].ptr_to(0, 0),
                                txl.address_of(aqk_map),
                                txl.int32(0),
                                tok0,
                                head,
                                mb,
                            )
                        b_akk_empty.wait(par, ((cyc >> 1) & txl.int32(1)) ^ txl.int32(1))
                        with txl.If(elected()), txl.Then():
                            b_akk_full.arrive(par, tx_count=AQK_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            txl.ptx[TMA_LD](
                                TT[S_AKK + par].ptr_to(0, 0),
                                txl.address_of(akk_map),
                                txl.int32(0),
                                tok0,
                                head,
                                mb,
                            )
                            with txl.If(n == txl.int32(0)), txl.Then():
                                with txl.If(i2 + txl.int32(1) < n_p2), txl.Then():
                                    nxt = p2_chain(i2 + txl.int32(1))
                                    seq2, head2, bos2, seq_len2, nch2 = work_coords(nxt)
                                    tokn = txl.Cast("int32", bos2) + (
                                        nch2 - txl.int32(1)
                                    ) * txl.int32(CHUNK)
                                    hidn = (chunk_base(seq2) + nch2 - txl.int32(1)) * txl.int32(
                                        H
                                    ) + head2
                                    for tmap in (q_map, k_map, v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(tmap), txl.int32(d0), tokn, head2
                                            )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(aqk_map), txl.int32(0), tokn, head2
                                    )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(akk_map), txl.int32(0), tokn, head2
                                    )
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(h_map), txl.int32(d0), txl.int32(0), hidn
                                        )
                            with txl.If(n > txl.int32(0)), txl.Then():
                                tokp = tok0 - txl.int32(CHUNK)
                                for tmap in (q_map, k_map, v_map, do_map):
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(tmap), txl.int32(d0), tokp, head
                                        )
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(eg_map), txl.int32(d0), tokp, head
                                    )
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(aqk_map), txl.int32(0), tokp, head
                                )
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(akk_map), txl.int32(0), tokp, head
                                )
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(h_map),
                                        txl.int32(d0),
                                        txl.int32(0),
                                        hidx - txl.int32(H),
                                    )
                        lphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))

            with idle:
                with txl.If(txl.warp_id_in_role() == txl.int32(0)), txl.Then():
                    p1_storer()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                rowc = txl.local_scalar(
                    "int32", init=txl.warp_id_in_role() * txl.int32(32) + txl.lane_id()
                )
                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with txl.serial(nch) as rn:
                        par = cyc & txl.int32(1)
                        b_aqk_full.wait(0, par)

                        diag = txl.alloc_local([4], "uint32")
                        dmat = txl.lane_id() >> txl.int32(3)
                        dblk = txl.warp_id_in_role() * txl.int32(4) + dmat
                        dptr = TT[S_AQK].ptr_to(
                            dblk * txl.int32(8) + (txl.lane_id() & txl.int32(7)),
                            dblk * txl.int32(8),
                        )
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            diag[0], diag[1], diag[2], diag[3], dptr
                        )
                        drow = txl.lane_id() >> txl.int32(2)
                        dcol = (txl.lane_id() & txl.int32(3)) * txl.int32(2)
                        dmask = txl.Select(
                            dcol > drow,
                            txl.uint32(0),
                            txl.Select(
                                dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)
                            ),
                        )
                        for e in range(4):
                            txl.assign(diag[e], diag[e] & dmask)
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            dptr, diag[0], diag[1], diag[2], diag[3]
                        )
                        for u in range(1, 8):
                            with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                                txl.ptx["st.shared.v4.b32"](
                                    TT[S_AQK].ptr_to(rowc, 8 * u),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                )
                        txl.ptx[FENCE_ASYNC]()
                        b_aqk_masked.arrive(0)
                        txl.assign(cyc, cyc + txl.int32(1))

        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                txl.Cast("uint32", txl.local_scalar("int32", init=tmem_preamble()[0])),
                txl.uint32(512),
            )

    return kda_bwd_native_fused


def make_fused_kernel(H: int, sched_maxp2: int, sched_maxp1: int, static_grid=None):
    txl.MBarrier._wait = _CUDA_MBAR_WAIT
    TM_DH = 0
    S1, S2, S3, S4, S6, S5 = 128, 192, 256, 320, 384, 448
    DO_BYTES = CHUNK * D * 2
    H_BYTES = D * D * 2
    SCHED_MAXP2, SCHED_MAXP1 = sched_maxp2, sched_maxp1
    SCHED_STRIDE = 2 + SCHED_MAXP2 + SCHED_MAXP1
    HK = H * D
    HK64 = txl.int64(HK)
    QKVE_BYTES = 4 * CHUNK * D * 2

    T1, T2, T3, T5, T6, DHB = 0, 2, 4, 8, 10, 12
    DV2, ZT, DVB, DAM = 6, 8, 6, 9
    PB0, PB1 = 12, 14
    ST_Q, ST_K, ST_V, ST_G = 12, 14, 16, 8

    S_DO, S_H, S_AQK, S_AKK = 18, 20, 24, 25
    IN_BYTES = 3 * CHUNK * D * 2 + CHUNK * 8 * 2
    EG_BYTES = CHUNK * D * 2

    @txl.kernel(
        warps=12,
        arch="sm_100a",
        min_blocks_per_sm=1,
        grid="num_ctas" if static_grid is None else static_grid,
    )
    def kda_bwd_fused(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        v: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        aqk: txl.gptr[txl.bf16],
        akk: txl.gptr[txl.bf16],
        g: txl.gptr[txl.f32],
        egcache: txl.gptr[txl.bf16],
        do: txl.gptr[txl.bf16],
        dht: txl.gptr[txl.f32],
        h0: txl.gptr[txl.f32],
        hsnap: txl.gptr[txl.bf16],
        cu_seqlens: txl.gptr[txl.i64],
        flags: txl.gptr[txl.i32],
        sched: txl.gptr[txl.i32],
        dq: txl.gptr[txl.f32],
        dk: txl.gptr[txl.f32],
        dv: txl.gptr[txl.bf16],
        db: txl.gptr[txl.f32],
        dg: txl.gptr[txl.f32],
        dh0: txl.gptr[txl.f32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        v_map: txl.TensorMap,
        g_map: txl.TensorMap,
        eg_map: txl.TensorMap,
        beta_map: txl.TensorMap,
        do_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        akk_map: txl.TensorMap,
        h_map: txl.TensorMap,
        scale: txl.f32,
        num_seqs: txl.i32,
        num_ctas: txl.i32,
        epoch: txl.i32,
        range_flags: txl.gptr[txl.i32],
        range_allowed: txl.i32,
        range_entries: txl.i32,
    ):
        range_bad = txl.local_scalar("uint32", init=txl.uint32(0))
        range_idx = txl.local_scalar("int32", init=txl.thread_id())
        with txl.While(range_idx < range_entries):
            range_part = txl.local_scalar("uint32")
            txl.ptx.ld.global_.u32(range_part, range_flags.ptr_to([range_idx]))
            txl.assign(range_bad, range_bad | range_part)
            txl.assign(range_idx, range_idx + txl.int32(384))
        range_count = txl.local_scalar("uint32")
        range_bad_pred = txl.local_scalar("bool", init=(range_bad & txl.uint32(2)) != txl.uint32(0))
        txl.ptx.bar.red.popc.u32(
            range_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(range_bad_pred)
        )
        range_safe = txl.local_scalar(
            "bool", init=(range_count == txl.uint32(0)) & (range_allowed != txl.int32(0))
        )
        diagonal_pred = txl.local_scalar("bool", init=(range_bad & txl.uint32(1)) != txl.uint32(0))
        diagonal_count = txl.local_scalar("uint32")
        txl.ptx.bar.red.popc.u32(
            diagonal_count, txl.uint32(0), txl.uint32(384), txl.ptx.pred(diagonal_pred)
        )
        diagonal_needed = txl.local_scalar("bool", init=diagonal_count != txl.uint32(0))
        with txl.If(txl.Not(diagonal_needed)), txl.Then():
            txl.Return(txl.int32(0))
        for buf in (q, k, v, beta, aqk, akk, g, do, hsnap, egcache):
            txl.keep_alive(buf.data)
        num_work = num_seqs * txl.int32(H)

        cta = txl.local_scalar("int32", init=txl.Cast("int32", txl.cta_id()))
        sbase = txl.local_scalar("int32", init=cta * txl.int32(SCHED_STRIDE))
        n_p2 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(n_p2, sched.ptr_to([sbase]))
        n_p1 = txl.local_scalar("int32")
        txl.ptx.ld.global_.s32(n_p1, sched.ptr_to([sbase + txl.int32(1)]))

        def p2_chain(i):
            c = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(c, sched.ptr_to([sbase + txl.int32(2) + i]))
            return c

        def p1_chain(i):
            c = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(c, sched.ptr_to([sbase + txl.int32(2 + SCHED_MAXP2) + i]))
            return c

        sp = txl.specialize()
        cg = sp.role("cg", warps=list(range(8)), regs=240)
        auxg = sp.warpgroup("aux", warps=[8, 9, 10, 11], regs=24)
        loader = sp.role("loader", warps=[8], group=auxg)
        mma = sp.role("mma", warps=[9], group=auxg)
        idle = sp.role("idle", warps=[10, 11], group=auxg)

        smem = txl.smem_pool()
        s_tmem = smem.alloc((4,), txl.i32, align=16)
        b_in_full = txl.TMABar(smem, 1)
        b_in_full.init(1)
        b_eg_full = txl.TMABar(smem, 1)
        b_eg_full.init(1)
        b_mid_free = txl.MBarrier(smem, 1)
        b_mid_free.init(256)
        b_do_full = txl.TMABar(smem, 1)
        b_do_full.init(1)
        b_h_full = txl.TMABar(smem, 1)
        b_h_full.init(1)
        b_aqk_full = txl.TMABar(smem, 1)
        b_aqk_full.init(1)
        b_akk_full = txl.TMABar(smem, 2)
        b_akk_full.init(1)
        b_do_empty = txl.TCGen05Bar(smem, 1)
        b_do_empty.init(1)
        b_h_free = txl.MBarrier(smem, 1)
        b_h_free.init(256)
        b_intra_free = txl.MBarrier(smem, 1)
        b_intra_free.init(256)
        b_stable_full = txl.MBarrier(smem, 1)
        b_stable_full.init(256)
        b_aqk_empty = txl.TCGen05Bar(smem, 1)
        b_aqk_empty.init(1)
        b_akk_empty = txl.TCGen05Bar(smem, 2)
        b_akk_empty.init(1)
        mb_names = [
            "t_early",
            "dhb_ready",
            "zT_ready",
            "vnT_ready",
            "dv2T_ready",
            "dAqk_tile_ready",
            "dAm_ready",
            "X_ready",
            "intra_ready",
            "dv_epi_done",
        ]
        MB = {}
        for nm in mb_names:
            MB[nm] = txl.MBarrier(smem, 1)
            MB[nm].init(256)
        b_dg0_ready = txl.MBarrier(smem, 1)
        b_dg0_ready.init(256)
        b_aqk_masked = txl.MBarrier(smem, 1)
        b_aqk_masked.init(64)

        p_kv = txl.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        b_kvT_done = txl.TCGen05Bar(smem, 1)
        b_kvT_done.init(1)
        b_kv_read = txl.MBarrier(smem, 1)
        b_kv_read.init(256)
        p_akk1 = txl.Pipeline(smem, 1, full="tma", empty="tcgen05")
        p_tiles = txl.Pipeline(smem, 1, full="mbar", empty="tcgen05", init_full=256)
        p_hs = txl.Pipeline(smem, 1, full="mbar", empty="mbar", init_full=256, init_empty=9)
        p_w = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_vn = txl.Pipeline(smem, 1, full="tcgen05", empty="mbar", init_empty=256)
        p_g = txl.Pipeline(smem, 2, full="tma", empty="mbar", init_empty=256)
        tc_names = [
            "Z_done",
            "Vn_done",
            "dv2_done",
            "dAqk_done",
            "dk_done",
            "dAs_done",
            "dvb_done",
            "X_done",
            "Y_done",
            "dq2_done",
            "dkt_done",
            "chunk_done",
            "xT_done",
        ]
        TC = {}
        for nm in tc_names:
            TC[nm] = txl.TCGen05Bar(smem, 1)
            TC[nm].init(1)

        TT = smem.alloc((27, 64, 64), txl.bf16, swizzle=txl.SW128B)
        s_beta = smem.alloc((64,), txl.f32, align=16)
        s_beta_in = smem.alloc((CHUNK, 8), txl.bf16, align=128)
        s_dgk = smem.alloc((2, 128), txl.f32, align=16)
        s_cs = smem.alloc((128,), txl.f32, align=16)
        s_beta_g = smem.alloc((2, CHUNK, 8), txl.bf16, align=128)
        s_beta1 = smem.alloc((2, CHUNK), txl.f32, align=16)

        s_ident = smem.alloc((256,), txl.bf16, align=128)

        with txl.If(txl.thread_id() == 0), txl.Then():
            txl.ptx.st.shared.s32(txl.address_of(s_tmem[1]), txl.int32(0))
            txl.ptx.fence.mbarrier_init.release.cluster()
        with txl.If(txl.thread_id() < txl.int32(256)), txl.Then():
            tid_i = txl.thread_id()
            n_i = tid_i >> 4
            k_i = tid_i & txl.int32(15)
            txl.ptx.st.shared.u16(
                s_ident.ptr_to(
                    [
                        (n_i >> 3) * txl.int32(128)
                        + (k_i >> 3) * txl.int32(64)
                        + (n_i & txl.int32(7)) * txl.int32(8)
                        + (k_i & txl.int32(7))
                    ]
                ),
                txl.Cast("uint16", txl.Select(n_i == k_i, txl.int32(0x3F80), txl.int32(0))),
            )
            txl.ptx[FENCE_ASYNC]()
        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                txl.address_of(s_tmem[0]), txl.uint32(512)
            )
        txl.cuda.cta_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def make_phaser():
            """Sequential IKET ranges for one role: phase(name) ends the current range and starts the next."""
            tok = txl.alloc_local([1], "uint32")
            txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            def phase(name):
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.range_start(name))

            def phase_end():
                txl.cuda.iket.range_end(tok[0])
                txl.assign(tok[0], txl.cuda.iket.sentinel_token("idle"))

            return phase, phase_end

        def tmem_preamble():
            tmv = txl.alloc_local([1], "int32")
            txl.ptx.ld.volatile.shared.s32(tmv[0], txl.address_of(s_tmem[0]))
            return tmv

        def pack_bf16x2(dst, lo, hi):
            txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)

        def work_coords(work):
            seq = txl.local_scalar("int32", init=work // txl.int32(H))
            head = txl.local_scalar("int32", init=work - seq * txl.int32(H))
            cs = txl.alloc_local([2], "int64")
            txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([seq]))
            txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([seq + txl.int32(1)]))
            bos = txl.local_scalar("int64", init=cs[0])
            seq_len = txl.local_scalar("int32", init=txl.Cast("int32", cs[1] - cs[0]))
            nch = txl.local_scalar("int32", init=(seq_len + txl.int32(CHUNK - 1)) >> 6)
            return seq, head, bos, seq_len, nch

        def chunk_base(seq):
            cb = txl.local_scalar("int32", init=txl.int32(0))
            with txl.serial(seq) as i:
                cs = txl.alloc_local([2], "int64")
                txl.ptx.ld.global_.s64(cs[0], cu_seqlens.ptr_to([i]))
                txl.ptx.ld.global_.s64(cs[1], cu_seqlens.ptr_to([i + 1]))
                txl.assign(
                    cb, cb + ((txl.Cast("int32", cs[1] - cs[0]) + txl.int32(CHUNK - 1)) >> 6)
                )
            return cb

        P1_KV, P1_AKK, P1_HS, P1_G, P1_KG, P1_KBG, P1_VB = 0, 8, 9, 13, 21, 23, 25
        G_BYTES = CHUNK * D * 4 + CHUNK * 8 * 2
        TM_H, TM_W, TM_U = 0, 128, 192
        TM_KT, TM_VT = 256, 320

        def bf16_bits_to_f32(u16val):
            return txl.reinterpret("float32", txl.Cast("uint32", u16val) << txl.uint32(16))

        def p1_compute():
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            xg = txl.local_scalar("int32", init=x >> 5)
            xgc = txl.local_scalar("int32", init=(x & 31) * 2)

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            st_kv = txl.PipelineState(2, phase=0)
            st_te = txl.PipelineState(1, phase=1)
            st_hs = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(2, phase=0)
            st_w = txl.PipelineState(1, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            p1c = txl.local_scalar("int32", init=txl.int32(0))
            gv = txl.alloc_local([32], "float32")
            kk = txl.alloc_local([32], "float32")
            vv = txl.alloc_local([32], "float32")
            bb = txl.alloc_local([32], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            eg = txl.local_scalar("float32")
            egng = txl.local_scalar("float32")
            bu = txl.local_scalar("uint16")
            ku = txl.local_scalar("uint16")
            vu = txl.local_scalar("uint16")
            phase, phase_end = make_phaser()
            tid_all = txl.local_scalar("int32", init=wr * 32 + lane)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                head64 = txl.Cast("int64", head)
                gcol = txl.local_scalar("int64", init=head64 * txl.int64(D) + x64)

                def h_c0():
                    rows = txl.int32(CHUNK)
                    p_g.full.wait(st_g.stage, st_g.phase)
                    gst = txl.local_scalar("int32", init=P1_G + st_g.stage * txl.int32(4) + xg)
                    with txl.If(lane < txl.int32(8)), txl.Then():
                        btok = wr * txl.int32(8) + lane
                        txl.ptx.ld.shared.u16(
                            bu, s_beta_g.ptr_to([st_g.stage, btok, head & txl.int32(7)])
                        )
                        txl.ptx.st.shared.f32(
                            txl.address_of(s_beta1[st_g.stage, btok]), bf16_bits_to_f32(bu)
                        )
                    for i in range(32):
                        txl.ptx.ld.shared.f32(gv[i], TT[gst].ptr_to(row0 + i, xgc))
                    txl.ptx.ld.shared.f32(gn, TT[gst].ptr_to(rows - txl.int32(1), xgc))
                    txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            bb[4 * u],
                            bb[4 * u + 1],
                            bb[4 * u + 2],
                            bb[4 * u + 3],
                            txl.address_of(s_beta1[st_g.stage, row0 + 4 * u]),
                        )
                    txl.ptx[FENCE_ASYNC]()
                    p_g.empty.arrive(st_g.stage)
                    st_g.advance()

                phase("h-c0")
                h_c0()
                with txl.serial(nch) as n:
                    rows = txl.int32(CHUNK)
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK))
                    )
                    txl.ptx.ex2.approx.ftz.f32(egn, gn)
                    phase("hw-kv")
                    b_kvT_done.wait(0, p1c & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("h-kv")
                    kst = txl.local_scalar(
                        "int32", init=P1_KV + st_kv.stage * txl.int32(4) + xs * txl.int32(2)
                    )
                    txl.ptx[TC_LD32](*(kk[i] for i in range(32)), tmem_at(TM_KT + wg * 32))
                    for i in range(32):
                        txl.ptx.ld.shared.u16(vu, TT[kst + txl.int32(1)].ptr_to(row0 + i, xr))
                        txl.assign(vv[i], bf16_bits_to_f32(vu))
                    txl.ptx[WAIT_LD]()
                    txl.ptx[FENCE_ASYNC]()
                    p_kv.empty.arrive(st_kv.stage)
                    st_kv.advance()
                    txl.ptx[TC_FENCE_BEFORE]()
                    b_kv_read.arrive(0)
                    txl.assign(p1c, p1c + txl.int32(1))
                    phase("hw-tiles")
                    p_tiles.empty.wait(0, st_te.phase)
                    st_te.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-tiles")
                    for u in range(4):
                        wkg = txl.alloc_local([4], "uint32")
                        wkbg = txl.alloc_local([4], "uint32")
                        wvb = txl.alloc_local([4], "uint32")
                        vals = txl.alloc_local([24], "float32")
                        for e in range(8):
                            i = 8 * u + e
                            txl.ptx.ex2.approx.ftz.f32(eg, gv[i])
                            txl.ptx.ex2.approx.ftz.f32(egng, gn - gv[i])
                            txl.ptx.cvt.rn.bf16.f32(bu, eg)
                            egidx = (tok0 + txl.Cast("int64", row0 + txl.int32(i))) * HK64 + gcol
                            txl.ptx["st.global.L1::no_allocate.b16"](egcache.ptr_to([egidx]), bu)
                            txl.assign(vals[e], kk[i] * egng)
                            txl.assign(vals[8 + e], kk[i] * bb[i] * eg)
                            txl.assign(vals[16 + e], vv[i] * bb[i])
                        for p in range(4):
                            pack_bf16x2(wkg[p], vals[2 * p], vals[2 * p + 1])
                            pack_bf16x2(wkbg[p], vals[8 + 2 * p], vals[8 + 2 * p + 1])
                            pack_bf16x2(wvb[p], vals[16 + 2 * p], vals[16 + 2 * p + 1])
                        col = row0 + 8 * u
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KG + xs].ptr_to(xr, col), wkg[0], wkg[1], wkg[2], wkg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KBG + xs].ptr_to(xr, col), wkbg[0], wkbg[1], wkbg[2], wkbg[3]
                        )
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_VB + xs].ptr_to(xr, col), wvb[0], wvb[1], wvb[2], wvb[3]
                        )

                    txl.ptx[FENCE_ASYNC]()
                    p_tiles.full.arrive(0)
                    phase("hw-hs")
                    p_hs.empty.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("h-decay")
                    hc0 = wg * 64
                    hsst = txl.local_scalar(
                        "int32", init=P1_HS + st_hs.stage * txl.int32(4) + wg * txl.int32(2) + xs
                    )
                    with txl.If(n == txl.int32(0)):
                        with txl.Then():
                            h0base = (
                                (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D)
                                + x64
                            ) * txl.int64(D) + txl.Cast("int64", hc0)
                            for m in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m + i] for i in range(8)),
                                    h0.ptr_to([h0base + txl.int64(8 * m)]),
                                )
                        with txl.Else():
                            txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_H + hc0))
                            txl.ptx[TC_LD32](
                                *(acc[32 + i] for i in range(32)), tmem_at(TM_H + hc0 + 32)
                            )
                            txl.ptx[WAIT_LD]()
                    for p in range(32):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(8):
                        txl.ptx["st.shared.v4.b32"](
                            TT[hsst].ptr_to(xr, 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    for p in range(32):
                        dpair = txl.local_scalar("uint64")
                        txl.ptx["mul.rn.f32x2"](
                            dpair,
                            txl.cuda.make_float2(acc[2 * p], acc[2 * p + 1]),
                            txl.cuda.make_float2(egn, egn),
                        )
                        txl.assign(acc[2 * p], txl.cuda.float2_x(dpair))
                        txl.assign(acc[2 * p + 1], txl.cuda.float2_y(dpair))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0), *(acc[i] for i in range(32)))
                    txl.ptx[TC_ST32](tmem_at(TM_H + hc0 + 32), *(acc[32 + i] for i in range(32)))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_hs.full.arrive(st_hs.stage)
                    phase("hw-W")
                    p_w.full.wait(0, st_w.phase)
                    st_w.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-wT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_W + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_KBG + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_w.empty.arrive(0)
                    phase("h-c0")
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        h_c0()
                    phase("hw-Vn")
                    p_vn.full.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("h-vnT")
                    txl.ptx[TC_LD32](*(acc[i] for i in range(32)), tmem_at(TM_U + wg * 32))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                    for u in range(4):
                        txl.ptx["st.shared.v4.b32"](
                            TT[P1_VB + xs].ptr_to(xr, row0 + 8 * u),
                            wds[4 * u],
                            wds[4 * u + 1],
                            wds[4 * u + 2],
                            wds[4 * u + 3],
                        )
                    txl.ptx[TC_FENCE_BEFORE]()
                    txl.ptx[FENCE_ASYNC]()
                    p_vn.empty.arrive(0)

                    with txl.If(elected()), txl.Then():
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()
                    phase_end()

            p_tiles.empty.wait(0, st_te.phase)
            txl.ptx[TC_FENCE_AFTER]()

        def p1_mma():
            tm = tmem_preamble()

            bd1 = txl.alloc_local([1], "uint64")
            zq1 = txl.alloc_local([1], "int32")
            op_kbg_k = Op(bd1, P1_KBG, 128, 64, "k")
            op_vb_k = Op(bd1, P1_VB, 128, 64, "k")
            op_kg_k = Op(bd1, P1_KG, 128, 64, "k")
            op_akk1_k = Op(bd1, P1_AKK, 64, 64, "k")
            op_hs_mn = Op(bd1, P1_HS, 128, 128, "mn")
            op_w_mn = Op(bd1, P1_KBG, 128, 128, "mn")
            op_kraw = Op(bd1, P1_KV, 128, 64, "mn")
            ID_T1 = idesc(128, 16, ta=1)
            bdI1 = txl.alloc_local([1], "uint64")
            txl.cuda.tcgen05.encode_matrix_descriptor(
                txl.address_of(bdI1[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
            )
            st_kv1 = txl.PipelineState(2, phase=0)
            p1m = txl.local_scalar("int32", init=txl.int32(0))
            ID_M128N64 = idesc(128, 64)

            def kv_transpose():
                mphase("hmw-kv")
                p_kv.full.wait(st_kv1.stage, st_kv1.phase)
                b_kv_read.wait(0, (p1m & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                mphase("hm-kvT")
                kv_u = txl.local_scalar(
                    "uint64",
                    init=txl.Cast("uint64", st_kv1.stage) * txl.uint64(4 * UNITS_PER_STAGE),
                )
                with txl.If(elected()), txl.Then():
                    for j in range(4):
                        txl.ptx[MMA_SS](
                            txl.Cast("uint32", tm[0] + TM_KT + 16 * j),
                            op_kraw.desc(j, kv_u),
                            bdI1[0],
                            txl.uint32(ID_T1),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.uint32(0),
                            txl.ptx.pred(0),
                        )
                    b_kvT_done.arrive(0)
                st_kv1.advance()
                txl.assign(p1m, p1m + txl.int32(1))

            ID_VN = idesc(128, 64, ta=1, tb=1, nb=1)
            ID_HUPD = idesc(128, 128)
            st_tiles = txl.PipelineState(1, phase=0)
            st_akk = txl.PipelineState(1, phase=0)
            st_hs = txl.PipelineState(1, phase=0)
            st_w = txl.PipelineState(1, phase=0)
            st_vn = txl.PipelineState(1, phase=0)
            mphase, mphase_end = make_phaser()
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bd1[0]),
                    TT[zq1[0]].ptr_to(0, 0),
                    ldo=Op.LBO_BASE,
                    sdo=SBO_UNITS,
                    swizzle=txl.SW128B.value,
                )
                kv_transpose()
                with txl.serial(nch) as n:
                    txl.ptx.ld.volatile.shared.s32(zq1[0], txl.address_of(s_tmem[1]))
                    txl.cuda.tcgen05.encode_matrix_descriptor(
                        txl.address_of(bd1[0]),
                        TT[zq1[0]].ptr_to(0, 0),
                        ldo=Op.LBO_BASE,
                        sdo=SBO_UNITS,
                        swizzle=txl.SW128B.value,
                    )
                    mphase("hmw-tiles")
                    p_tiles.full.wait(0, st_tiles.phase)
                    p_akk1.full.wait(st_akk.stage, st_akk.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    akk_u = txl.local_scalar(
                        "uint64",
                        init=txl.Cast("uint64", st_akk.stage) * txl.uint64(UNITS_PER_STAGE),
                    )
                    mphase("hm-WU")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_W, op_kbg_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_w.full.arrive(0)

                        mma_chain(tm, TM_U, op_vb_k, op_akk1_k, ID_M128N64, False, b_units=akk_u)
                        p_akk1.empty.arrive(st_akk.stage)
                    st_akk.advance()
                    with txl.If(n + txl.int32(1) < nch), txl.Then():
                        kv_transpose()
                    mphase("hmw-wT")
                    p_w.empty.wait(0, st_w.phase)
                    st_w.advance()
                    mphase("hmw-hs")
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    txl.ptx[TC_FENCE_AFTER]()
                    hs_u = txl.local_scalar(
                        "uint64",
                        init=txl.Cast("uint64", st_hs.stage) * txl.uint64(4 * UNITS_PER_STAGE),
                    )
                    mphase("hm-Vn")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_U, op_hs_mn, op_w_mn, ID_VN, True, a_units=hs_u)
                        p_vn.full.arrive(0)
                    st_hs.advance()
                    mphase("hmw-vnT")
                    p_vn.empty.wait(0, st_vn.phase)
                    st_vn.advance()
                    txl.ptx[TC_FENCE_AFTER]()
                    mphase("hm-hupd")
                    with txl.If(elected()), txl.Then():
                        mma_chain(tm, TM_H, op_kg_k, op_vb_k, ID_HUPD, True)
                        p_tiles.empty.arrive(0)
                    st_tiles.advance()
                    mphase_end()

        def p1_loader():
            st_kv = txl.PipelineState(2, phase=1)
            st_akk = txl.PipelineState(1, phase=1)
            st_g = txl.PipelineState(2, phase=1)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                head8 = txl.local_scalar("int32", init=head >> txl.int32(3))
                with txl.serial(nch) as n:
                    tok0 = bos32 + n * txl.int32(CHUNK)
                    p_g.empty.wait(st_g.stage, st_g.phase)
                    with txl.If(elected()), txl.Then():
                        p_g.full.arrive(st_g.stage, tx_count=G_BYTES)
                        mbg = txl.cuda.cvta_generic_to_shared(p_g.full.ptr_to([st_g.stage]))
                        for j in range(4):
                            txl.ptx[TMA_LD](
                                TT[P1_G + st_g.stage * txl.int32(4) + txl.int32(j)].ptr_to(0, 0),
                                txl.address_of(g_map),
                                txl.int32(32 * j),
                                tok0,
                                head,
                                mbg,
                            )
                        txl.ptx[TMA_LD](
                            s_beta_g.ptr_to([st_g.stage, 0, 0]),
                            txl.address_of(beta_map),
                            txl.int32(0),
                            tok0,
                            head8,
                            mbg,
                        )
                    st_g.advance()
                    p_kv.empty.wait(st_kv.stage, st_kv.phase)
                    with txl.If(elected()), txl.Then():
                        p_kv.full.arrive(st_kv.stage, tx_count=KV_BYTES)
                        mb = txl.cuda.cvta_generic_to_shared(p_kv.full.ptr_to([st_kv.stage]))
                        for tmap, half in ((k_map, 0), (v_map, 1)):
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[
                                        P1_KV
                                        + st_kv.stage * txl.int32(4)
                                        + txl.int32((d0 // 64) * 2 + half)
                                    ].ptr_to(0, 0),
                                    txl.address_of(tmap),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        with txl.If(n + txl.int32(1) < nch), txl.Then():
                            for tmap in (k_map, v_map):
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(tmap),
                                        txl.int32(d0),
                                        tok0 + txl.int32(CHUNK),
                                        head,
                                    )
                            for d0 in (0, 32, 64, 96):
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(g_map),
                                    txl.int32(d0),
                                    tok0 + txl.int32(CHUNK),
                                    head,
                                )
                            txl.ptx[TMA_PREFETCH](
                                txl.address_of(akk_map), txl.int32(0), tok0 + txl.int32(CHUNK), head
                            )
                    st_kv.advance()
                    p_akk1.empty.wait(st_akk.stage, st_akk.phase)
                    with txl.If(elected()), txl.Then():
                        p_akk1.full.arrive(st_akk.stage, tx_count=AQK_BYTES)
                        mb2 = txl.cuda.cvta_generic_to_shared(p_akk1.full.ptr_to([st_akk.stage]))
                        txl.ptx[TMA_LD](
                            TT[P1_AKK + st_akk.stage].ptr_to(0, 0),
                            txl.address_of(akk_map),
                            txl.int32(0),
                            tok0,
                            head,
                            mb2,
                        )
                    st_akk.advance()

        def p1_storer():
            st_hs = txl.PipelineState(1, phase=0)
            with txl.serial(n_p1) as it:
                chain = txl.local_scalar("int32", init=p1_chain(it))
                seq, head, bos, seq_len, nch = work_coords(chain)
                cb = chunk_base(seq)
                with txl.serial(nch) as n:
                    p_hs.full.wait(st_hs.stage, st_hs.phase)
                    with txl.If(elected()), txl.Then():
                        txl.ptx[FENCE_ASYNC]()
                        idx = (cb + n) * txl.int32(H) + head
                        for d0 in (0, 64):
                            txl.ptx[TMA_ST](
                                txl.address_of(h_map),
                                txl.int32(d0),
                                txl.int32(0),
                                idx,
                                TT[
                                    P1_HS + st_hs.stage * txl.int32(4) + txl.int32((d0 // 64) * 2)
                                ].ptr_to(0, 0),
                            )
                        txl.ptx[BULK_COMMIT]()
                        txl.ptx[BULK_WAIT_READ](0)
                        p_hs.empty.arrive(st_hs.stage)
                    st_hs.advance()

                with txl.If(elected()), txl.Then():
                    txl.ptx[BULK_WAIT](0)
                    txl.ptx["fence.proxy.async.global"]()
                    txl.ptx.st.release.gpu.global_.s32(flags.ptr_to([chain]), epoch)

        with cg:
            p1_compute()
            txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
            tm = tmem_preamble()
            wr = txl.warp_id_in_role()
            lane = txl.lane_id()
            wg = txl.local_scalar("int32", init=wr >> 2)
            quad = txl.local_scalar("int32", init=wr & 3)
            x = txl.local_scalar("int32", init=quad * 32 + lane)
            row0 = txl.local_scalar("int32", init=wg * 32)
            x64 = txl.Cast("int64", x)
            xs = txl.local_scalar("int32", init=x >> 6)
            xr = txl.local_scalar("int32", init=x & 63)
            tid_all = txl.local_scalar("int32", init=wr * 32 + lane)
            phalf = txl.local_scalar("int32", init=x & 1)
            pcol = txl.local_scalar("int32", init=x & ~1)
            prow0 = txl.local_scalar("int32", init=row0 + phalf * 16)
            ps = txl.local_scalar("int32", init=pcol >> 6)
            pr = txl.local_scalar("int32", init=pcol & 63)
            is_odd = phalf != txl.int32(0)
            cyc = txl.local_scalar("int32", init=txl.int32(0))

            def tmem_at(col):
                return txl.Cast("uint32", tm[0] + col + (quad << 21))

            def ld32(regs, col, base=0):
                txl.ptx[TC_LD32](*(regs[base + i] for i in range(32)), tmem_at(col))

            def ld8(regs, col, base=0):
                txl.ptx[TC_LD8](*(regs[base + i] for i in range(8)), tmem_at(col))

            def ld4(regs, col, base=0):
                txl.ptx[TC_LD4](*(regs[base + i] for i in range(4)), tmem_at(col))

            def st_row(stage0, col0, words, wbase=0, nunits=4):
                """Write this thread's row x, columns [col0, col0 + 8*nunits) of the [128][64] tile at stage0/stage0+1."""
                for u in range(nunits):
                    txl.ptx["st.shared.v4.b32"](
                        TT[stage0 + xs].ptr_to(xr, col0 + 8 * u),
                        words[wbase + 4 * u],
                        words[wbase + 4 * u + 1],
                        words[wbase + 4 * u + 2],
                        words[wbase + 4 * u + 3],
                    )

            def st_pair_rows(stage0, words):
                """Pair layout: rows pcol and pcol+1, columns [prow0, prow0+16): words[0:8] row pcol, words[8:16] row pcol+1."""
                for r in range(2):
                    for u in range(2):
                        txl.ptx["st.shared.v4.b32"](
                            TT[stage0 + ps].ptr_to(pr + r, prow0 + 8 * u),
                            words[8 * r + 4 * u],
                            words[8 * r + 4 * u + 1],
                            words[8 * r + 4 * u + 2],
                            words[8 * r + 4 * u + 3],
                        )

            def bar_all():
                txl.ptx.bar.sync(txl.uint32(1), txl.uint32(256))

            def bar_wg():
                txl.ptx.bar.sync(txl.uint32(2) + txl.Cast("uint32", wg), txl.uint32(128))

            def twait(nm):
                TC[nm].wait(0, cyc & txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()

                txl.ptx[FENCE_ASYNC]()
                txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))

            def marrive(nm):
                txl.ptx[TC_FENCE_BEFORE]()
                MB[nm].arrive(0)

            def lo(w):
                return txl.reinterpret("float32", w << txl.uint32(16))

            def hi(w):
                return txl.reinterpret("float32", w & txl.uint32(0xFFFF0000))

            def shfl_xor1(val):
                r = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(
                    r,
                    txl.reinterpret("uint32", val),
                    txl.uint32(1),
                    txl.uint32(0x1F),
                    txl.uint32(0xFFFFFFFF),
                )
                return txl.reinterpret("float32", r)

            def q_ptr(c, col):
                return TT[ST_Q + (col >> 6)].ptr_to(c, col & 63)

            def k_ptr(c, col):
                return TT[ST_K + (col >> 6)].ptr_to(c, col & 63)

            def v_ptr(c, col):
                return TT[ST_V + (col >> 6)].ptr_to(c, col & 63)

            def e_ptr(c, col):
                return TT[ST_G + (col >> 6)].ptr_to(c, col & 63)

            def load_transpose_frag(base, frag):
                """Load this warp's 32x32 block as four groups of four 8x8 fragments."""
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            frag[o],
                            frag[o + 1],
                            frag[o + 2],
                            frag[o + 3],
                            tile.m8n8x4(row0 + txl.int32(16 * rb), col0 + txl.int32(16 * cb), lane),
                        )

            def store_transpose_frag(base, frag):
                """Transpose those fragments in place, turning [token,channel] into [channel,token]."""
                col0 = (quad & txl.int32(1)) * txl.int32(32)
                tile = TT[base + xs]
                mm = lane >> txl.int32(3)
                jj = lane & txl.int32(7)
                for rb in range(2):
                    for cb in range(2):
                        o = 4 * (2 * rb + cb)

                        ptr = tile.ptr_to(
                            col0 + txl.int32(16 * cb) + (mm >> txl.int32(1)) * txl.int32(8) + jj,
                            row0 + txl.int32(16 * rb) + (mm & txl.int32(1)) * txl.int32(8),
                        )
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"](
                            ptr, frag[o], frag[o + 1], frag[o + 2], frag[o + 3]
                        )

            enA = txl.alloc_local([16], "float32")
            enB = txl.alloc_local([16], "float32")
            egcw = txl.alloc_local([16], "uint64")
            t4 = txl.alloc_local([4], "float32")
            acc = txl.alloc_local([64], "float32")
            wds = txl.alloc_local([32], "uint32")
            dgv = txl.alloc_local([32], "float32")
            gn = txl.local_scalar("float32")
            egn = txl.local_scalar("float32")
            dgk = txl.local_scalar("float32")
            dgk_k = txl.local_scalar("float32")
            t0 = txl.local_scalar("float32")
            t1 = txl.local_scalar("float32")
            u16 = txl.local_scalar("uint16")
            u16b = txl.local_scalar("uint16")

            def ex2(dst, val):
                txl.ptx.ex2.approx.ftz.f32(dst, val)

            def rcp(dst, val):
                txl.ptx.rcp.approx.ftz.f32(dst, txl.Select(state_strong, txl.float32(1.0), val))
                txl.assign(dst, txl.Select(state_strong, txl.float32(0.0), dst))

            def load_u16_pair(words, i, ptr):
                if i % 2 == 0:
                    txl.ptx.ld.global_.nc.u16(u16, ptr)
                else:
                    txl.ptx.ld.global_.nc.u16(u16b, ptr)
                    txl.ptx.mov.b32(words[i >> 1], u16, u16b)

            def load_u16_pair_sh(words, i, ptr):
                if i % 2 == 0:
                    txl.ptx.ld.shared.u16(u16, ptr)
                else:
                    txl.ptx.ld.shared.u16(u16b, ptr)
                    txl.ptx.mov.b32(words[i >> 1], u16, u16b)

            def shfl_xor1_u32(val):
                r = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(
                    r, val, txl.uint32(1), txl.uint32(0x1F), txl.uint32(0xFFFFFFFF)
                )
                return r

            def gcol_ptr(tensor, i):
                """Global pointer to row (row0+i) of this thread's column (clamped to the last valid row)."""
                tokc = tok0 + txl.Cast("int64", txl.min(row0 + txl.int32(i), rows - txl.int32(1)))
                return tensor.ptr_to([tokc * HK64 + gcol])

            def s_beta_row(c):
                b = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b, txl.address_of(s_beta[c]))
                return b

            def wsel(cond, a, b):
                return txl.Select(cond, a, b)

            phase, phase_end = make_phaser()

            with txl.serial(n_p2) as i2:
                work = txl.local_scalar("int32", init=p2_chain(i2))
                seq, head, bos, seq_len, nch = work_coords(work)
                head64 = txl.Cast("int64", head)
                gcol = txl.local_scalar("int64", init=head64 * txl.int64(D) + x64)
                with txl.serial(nch) as rn:
                    n = nch - txl.int32(1) - rn
                    par = cyc & txl.int32(1)

                    rows = txl.int32(CHUNK)
                    last = txl.int32(CHUNK - 1)
                    tok0 = txl.local_scalar(
                        "int64", init=bos + txl.Cast("int64", n * txl.int32(CHUNK))
                    )
                    x_base = txl.local_scalar(
                        "int64", init=(tok0 + txl.Cast("int64", row0)) * HK64 + gcol
                    )

                    gn = txl.local_scalar("float32", init=txl.float32(0.0))
                    state_strong = txl.local_scalar("bool")

                    def state_decay(dst, i, cached_gate):
                        with txl.If(state_strong):
                            with txl.Then():
                                gi_value = txl.local_scalar("float32")
                                ti = txl.min(row0 + txl.int32(i), last)
                                txl.ptx.ld.global_.nc.f32(
                                    gi_value,
                                    g.ptr_to([(tok0 + txl.Cast("int64", ti)) * HK64 + gcol]),
                                )
                                txl.ptx.ex2.approx.ftz.f32(dst, gn - gi_value)
                            with txl.Else():
                                state_cached = txl.local_scalar("uint32")
                                pack_bf16x2(state_cached, cached_gate, cached_gate)
                                txl.ptx.rcp.approx.ftz.f32(dst, lo(state_cached))

                    phase("w-in")
                    b_in_full.wait(0, par)
                    b_eg_full.wait(0, par)
                    phase("w-xT")
                    twait("xT_done")
                    phase("c0")
                    with txl.If(lane < txl.int32(8)), txl.Then():
                        btok = wr * txl.int32(8) + lane
                        txl.ptx.ld.shared.u16(u16, s_beta_in.ptr_to([btok, head & txl.int32(7)]))
                        txl.ptx.st.shared.f32(
                            txl.address_of(s_beta[btok]), lo(txl.Cast("uint32", u16))
                        )

                    egf = txl.alloc_local([32], "float32")
                    xf = txl.alloc_local([32], "float32")
                    qw = txl.alloc_local([16], "uint32")
                    kw = txl.alloc_local([16], "uint32")
                    t3w = txl.alloc_local([16], "uint32")
                    qc = txl.alloc_local([16], "uint32")
                    kc = txl.alloc_local([16], "uint32")
                    vc = txl.alloc_local([16], "uint32")
                    prep0 = txl.local_scalar("uint64")
                    prep1 = txl.local_scalar("uint64")
                    scale_pair = txl.local_scalar("uint64", init=txl.cuda.make_float2(scale, scale))
                    bpair = txl.alloc_local([2], "float32")

                    ld32(xf, S2 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    bar_all()
                    for half in range(2):
                        vb32 = txl.alloc_local([16], "float32")
                        for p in range(8):
                            i = 16 * half + 2 * p
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[row0 + i])
                            )
                            txl.assign(vb32[2 * p], xf[i] * bpair[0])
                            txl.assign(vb32[2 * p + 1], xf[i + 1] * bpair[1])
                            pack_bf16x2(vc[i >> 1], xf[i], xf[i + 1])
                        txl.ptx[TC_ST16](
                            tmem_at(S2 + wg * 32 + 16 * half), *(vb32[j] for j in range(16))
                        )
                    txl.ptx[WAIT_ST]()
                    st_row(ST_V, row0, vc, 0, 4)

                    ld4(t4, S1 + 60)
                    ld32(egf, S1 + wg * 32)
                    ld32(xf, S3 + wg * 32)
                    txl.ptx[WAIT_LD]()
                    txl.assign(egn, t4[3])
                    txl.assign(state_strong, egn < txl.float32(0.0625))
                    txl.ptx.ld.global_.nc.f32(
                        gn, g.ptr_to([(tok0 + txl.Cast("int64", last)) * HK64 + gcol])
                    )
                    with txl.If(state_strong), txl.Then():
                        txl.ptx.ex2.approx.ftz.f32(egn, gn)
                    for i in range(16):
                        txl.ptx["mul.rn.f32x2"](
                            prep0,
                            txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                            txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                        )
                        txl.ptx["mul.rn.f32x2"](prep0, prep0, scale_pair)
                        pack_bf16x2(qw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                        pack_bf16x2(qc[i], xf[2 * i], xf[2 * i + 1])
                        txl.assign(egcw[i], txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]))

                    for half in range(2):
                        txl.ptx["tcgen05.ld.sync.aligned.32x32b.x16.b32"](
                            *(xf[16 * half + j] for j in range(16)),
                            tmem_at(S4 + wg * 32 + 16 * half),
                        )
                        txl.ptx[WAIT_LD]()
                        for pp in range(8):
                            i = 8 * half + pp
                            txl.ptx["ld.shared.v2.f32"](
                                bpair[0], bpair[1], txl.address_of(s_beta[row0 + 2 * i])
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                t0, txl.Select(state_strong, txl.float32(1.0), egf[2 * i])
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                t1, txl.Select(state_strong, txl.float32(1.0), egf[2 * i + 1])
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep0,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(t0, t1),
                            )
                            pack_bf16x2(kw[i], txl.cuda.float2_x(prep0), txl.cuda.float2_y(prep0))
                            txl.ptx["mul.rn.f32x2"](
                                prep1,
                                txl.cuda.make_float2(xf[2 * i], xf[2 * i + 1]),
                                txl.cuda.make_float2(egf[2 * i], egf[2 * i + 1]),
                            )
                            txl.ptx["mul.rn.f32x2"](
                                prep1, prep1, txl.cuda.make_float2(bpair[0], bpair[1])
                            )
                            pack_bf16x2(t3w[i], txl.cuda.float2_x(prep1), txl.cuda.float2_y(prep1))
                            pack_bf16x2(kc[i], xf[2 * i], xf[2 * i + 1])

                    phase("w-chunk")
                    TC["chunk_done"].wait(0, par ^ txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[FENCE_ASYNC]()
                    phase("c1")
                    st_row(T1, row0, qw, 0, 4)
                    st_row(T2, row0, kw, 0, 4)
                    st_row(T3, row0, t3w, 0, 4)
                    with txl.If(state_strong), txl.Then():
                        for pi in range(16):
                            values = txl.alloc_local([2], "float32")
                            for e in range(2):
                                ti0 = row0 + 2 * pi + e
                                token = tok0 + txl.Cast("int64", txl.min(ti0, last))
                                gate = _load_gate(g, token, head, x, H * D)
                                # Identity-transposed K is already live in kc for output math.
                                # Static indices avoid reloading these same token/channel pairs.
                                kval = lo(kc[pi]) if e == 0 else hi(kc[pi])
                                decay = txl.local_scalar("float32")
                                txl.ptx.ex2.approx.ftz.f32(decay, gn - gate)
                                txl.assign(
                                    values[e],
                                    txl.Select(ti0 < rows, kval * decay, txl.float32(0.0)),
                                )
                            word = txl.local_scalar("uint32")
                            pack_bf16x2(word, values[0], values[1])
                            txl.ptx.st.shared.b32(TT[T2 + xs].ptr_to(xr, row0 + 2 * pi), word)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("t_early")
                    phase("c1c")

                    phase("w-h")
                    b_h_full.wait(0, par)
                    txl.ptx[TC_FENCE_AFTER]()
                    phase("c2")
                    dgk2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )
                    hst = txl.local_scalar("int32", init=wg * 2 + xs)
                    dbase = (
                        (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D) + x64
                    ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                    with txl.If(rn == txl.int32(0)):
                        with txl.Then():
                            for m in range(8):
                                txl.ptx[
                                    "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v8.f32"
                                ](
                                    *(acc[8 * m + i] for i in range(8)),
                                    dht.ptr_to([dbase + txl.int64(8 * m)]),
                                )
                        with txl.Else():
                            ld32(acc, TM_DH + wg * 64)
                            ld32(acc, TM_DH + wg * 64 + 32, 32)
                            txl.ptx[WAIT_LD]()
                    for half in range(2):
                        hc = wg * 64 + 32 * half
                        a0 = 32 * half

                        raw_dh = txl.alloc_local([16], "uint32")
                        for p in range(16):
                            raw0 = txl.local_scalar("float32", init=acc[a0 + 2 * p])
                            raw1 = txl.local_scalar("float32", init=acc[a0 + 2 * p + 1])
                            dpair = txl.local_scalar("uint64")
                            txl.ptx["mul.rn.f32x2"](
                                dpair,
                                txl.cuda.make_float2(raw0, raw1),
                                txl.cuda.make_float2(egn, egn),
                            )
                            txl.assign(acc[a0 + 2 * p], txl.cuda.float2_x(dpair))
                            txl.assign(acc[a0 + 2 * p + 1], txl.cuda.float2_y(dpair))
                            pack_bf16x2(
                                raw_dh[p],
                                txl.Select(state_strong, raw0, acc[a0 + 2 * p]),
                                txl.Select(state_strong, raw1, acc[a0 + 2 * p + 1]),
                            )
                        for u in range(4):
                            txl.ptx["ld.shared.v4.b32"](
                                wds[0],
                                wds[1],
                                wds[2],
                                wds[3],
                                TT[S_H + hst].ptr_to(xr, 32 * half + 8 * u),
                            )
                            for p in range(4):
                                txl.ptx["fma.rn.f32x2"](
                                    dgk2,
                                    txl.cuda.make_float2(lo(wds[p]), hi(wds[p])),
                                    txl.cuda.make_float2(
                                        acc[a0 + 8 * u + 2 * p], acc[a0 + 8 * u + 2 * p + 1]
                                    ),
                                    dgk2,
                                )
                        for u in range(4):
                            txl.ptx["st.shared.v4.b32"](
                                TT[DHB + hst].ptr_to(xr, 32 * half + 8 * u),
                                raw_dh[4 * u],
                                raw_dh[4 * u + 1],
                                raw_dh[4 * u + 2],
                                raw_dh[4 * u + 3],
                            )
                        txl.ptx[TC_ST32](tmem_at(TM_DH + hc), *(acc[a0 + i] for i in range(32)))
                    txl.assign(dgk, txl.cuda.float2_x(dgk2) + txl.cuda.float2_y(dgk2))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dhb_ready")

                    def readout_to_tile(slot, stage0):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        for p in range(16):
                            pack_bf16x2(wds[p], acc[2 * p], acc[2 * p + 1])
                        st_row(stage0, row0, wds, 0, 4)
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-Z")
                    twait("Z_done")
                    phase("c3")
                    readout_to_tile(S2, ZT)
                    marrive("zT_ready")
                    phase("w-dv2")
                    twait("dv2_done")
                    phase("c5")
                    readout_to_tile(S3, DV2)
                    with txl.If(state_strong), txl.Then():
                        for i in range(16):
                            txl.assign(kw[i], txl.uint32(0))
                        st_row(T2, row0, kw, 0, 4)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv2T_ready")
                    phase("w-Vn")
                    twait("Vn_done")
                    phase("c4")
                    readout_to_tile(S2, T6)
                    marrive("vnT_ready")

                    def readout64(slot, stage, mask, scale_by=None, negate=False):
                        ld32(acc, slot + wg * 32)
                        txl.ptx[WAIT_LD]()
                        cc = quad * 16 + lane
                        with txl.If(lane < txl.int32(16)), txl.Then():
                            for p in range(16):
                                vv2 = []
                                for e in range(2):
                                    jj = row0 + 2 * p + e
                                    val = acc[2 * p + e]
                                    if scale_by is not None:
                                        val = val * scale_by
                                    if negate:
                                        val = txl.float32(0.0) - val
                                    vv2.append(
                                        val
                                        if mask is None
                                        else txl.Select(mask(cc, jj), val, txl.float32(0.0))
                                    )
                                pack_bf16x2(wds[p], vv2[0], vv2[1])
                            for u in range(4):
                                txl.ptx["st.shared.v4.b32"](
                                    TT[stage].ptr_to(cc, row0 + 8 * u),
                                    wds[4 * u],
                                    wds[4 * u + 1],
                                    wds[4 * u + 2],
                                    wds[4 * u + 3],
                                )
                        txl.ptx[FENCE_ASYNC]()

                    def readout64_half(slot, stage, mask, negate=False):
                        txl.ptx[TC_LD_HALF32](*(acc[i] for i in range(16)), tmem_at(slot + wg * 32))
                        txl.ptx[WAIT_LD]()
                        cc0 = quad * 16 + (lane >> txl.int32(2))
                        cc1 = cc0 + txl.int32(8)
                        for rep in range(4):
                            jj0 = row0 + txl.int32(8 * rep) + (lane & txl.int32(3)) * txl.int32(2)
                            jj1 = jj0 + txl.int32(1)
                            v00 = acc[4 * rep]
                            v01 = acc[4 * rep + 1]
                            v10 = acc[4 * rep + 2]
                            v11 = acc[4 * rep + 3]
                            if negate:
                                v00 = txl.float32(0.0) - v00
                                v01 = txl.float32(0.0) - v01
                                v10 = txl.float32(0.0) - v10
                                v11 = txl.float32(0.0) - v11
                            if mask is not None:
                                v00 = txl.Select(mask(cc0, jj0), v00, txl.float32(0.0))
                                v01 = txl.Select(mask(cc0, jj1), v01, txl.float32(0.0))
                                v10 = txl.Select(mask(cc1, jj0), v10, txl.float32(0.0))
                                v11 = txl.Select(mask(cc1, jj1), v11, txl.float32(0.0))
                            pack_bf16x2(wds[2 * rep], v00, v01)
                            pack_bf16x2(wds[2 * rep + 1], v10, v11)
                        tile = TT[stage]
                        for half in range(2):
                            txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                                tile.m8n8x4(
                                    quad * txl.int32(16), row0 + txl.int32(16 * half), lane
                                ),
                                wds[4 * half],
                                wds[4 * half + 1],
                                wds[4 * half + 2],
                                wds[4 * half + 3],
                            )
                        txl.ptx[FENCE_ASYNC]()

                    phase("w-dAs")
                    twait("dAs_done")
                    phase("c8")

                    readout64_half(S4, DAM, lambda cc, jj: jj < cc)
                    marrive("dAm_ready")
                    phase("w-dAqk")
                    twait("dAqk_done")
                    phase("c7")

                    readout64_half(S1, T5, lambda cc, jj: jj <= cc)
                    with txl.If(diagonal_needed), txl.Then():
                        bar_all()
                        with txl.If((wg == txl.int32(0)) & (x < txl.int32(64))), txl.Then():
                            diag_bits = txl.local_scalar("uint16")
                            txl.ptx.ld.shared.u16(diag_bits, TT[T5].ptr_to(x, x))
                            txl.ptx.st.shared.f32(
                                s_beta1.ptr_to([0, x]), bf16_bits_to_f32(diag_bits) * scale
                            )
                            txl.ptx.st.shared.b16(TT[T5].ptr_to(x, x), txl.uint16(0))
                        txl.ptx[FENCE_ASYNC]()
                    marrive("dAqk_tile_ready")
                    twait("dk_done")
                    phase("w-dvb")
                    twait("dvb_done")
                    phase("passA")

                    pbx = txl.local_scalar("int32", init=txl.int32(PB0) + wg * txl.int32(PB1 - PB0))

                    def pass_a(full):
                        assert full
                        pa_acc = txl.local_scalar("uint64")
                        pa_v = txl.local_scalar("uint64")
                        pa_db = txl.local_scalar("uint64")
                        pa_dv = txl.local_scalar("uint64")
                        pa_word = txl.local_scalar("uint32")
                        ld8(acc, S3 + wg * 32, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                ld8(acc, S3 + wg * 32 + 8 * (b + 1), 8 * ((b + 1) % 2))
                            vq = txl.alloc_local([4], "uint32")
                            txl.ptx["ld.shared.v4.b32"](
                                vq[0], vq[1], vq[2], vq[3], TT[ST_V + xs].ptr_to(xr, row0 + 8 * b)
                            )
                            dbp = txl.alloc_local([8], "float32")
                            for p in range(4):
                                i = 8 * b + 2 * p
                                txl.assign(
                                    pa_acc,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                )
                                txl.assign(pa_v, txl.cuda.make_float2(lo(vq[p]), hi(vq[p])))
                                txl.ptx["mul.rn.f32x2"](pa_db, pa_acc, pa_v)
                                txl.assign(dbp[2 * p], txl.cuda.float2_x(pa_db))
                                txl.assign(dbp[2 * p + 1], txl.cuda.float2_y(pa_db))
                                txl.ptx["ld.shared.v2.f32"](
                                    t4[0], t4[1], txl.address_of(s_beta[row0 + i])
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pa_dv, pa_acc, txl.cuda.make_float2(t4[0], t4[1])
                                )
                                pack_bf16x2(
                                    pa_word, txl.cuda.float2_x(pa_dv), txl.cuda.float2_y(pa_dv)
                                )
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.Cast("uint16", pa_word),
                                )
                                txl.ptx["st.global.L1::no_allocate.b16"](
                                    dv.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.Cast("uint16", pa_word >> txl.uint32(16)),
                                )

                            for e in range(8):
                                i = 8 * b + e
                                txl.ptx.st.shared.f32(
                                    TT[pbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane), dbp[e]
                                )

                            dvw = txl.alloc_local([4], "uint32")
                            for p in range(4):
                                pack_bf16x2(dvw[p], acc[ab + 2 * p], acc[ab + 2 * p + 1])
                            txl.ptx["st.shared.v4.b32"](
                                TT[DVB + xs].ptr_to(xr, row0 + 8 * b),
                                dvw[0],
                                dvw[1],
                                dvw[2],
                                dvw[3],
                            )
                            if b < 3:
                                txl.ptx[WAIT_LD]()

                    pass_a(True)
                    txl.ptx[FENCE_ASYNC]()
                    marrive("dv_epi_done")
                    phase("w-X")
                    twait("X_done")
                    phase("c9")
                    readout64_half(S1, T6, None)
                    marrive("X_ready")
                    phase("dbv")
                    bar_wg()
                    tq = lane & txl.int32(3)
                    ti = quad * 8 + (lane >> 2)
                    srow = (quad & txl.int32(1)) * 32 + lane
                    dsum_v = txl.local_scalar("float32", init=txl.float32(0.0))
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[pbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(dsum_v, dsum_v + ((t4[0] + t4[1]) + (t4[2] + t4[3])))

                    txl.ptx[FENCE_ASYNC]()
                    with txl.If(txl.Not(state_strong)), txl.Then():
                        b_mid_free.arrive(0)
                    phase("w-Y")
                    twait("Y_done")
                    phase("c10")
                    strong = txl.local_scalar("bool", init=_needs_stable(gn, range_safe))
                    readout64_half(S2, T5 + 1, lambda cc, jj: jj < cc, negate=True)
                    # Y_done proves the prior X/Vn readers of T6 have finished.
                    # Include earlier gate-cache rounding in the derivative
                    # residual, without changing the state-update operands.
                    with txl.If(range_safe & diagonal_needed & txl.Not(strong)), txl.Then():
                        precise_end = txl.local_scalar("float32")
                        txl.ptx.ld.global_.nc.f32(
                            precise_end, g.ptr_to([(tok0 + txl.Cast("int64", last)) * HK64 + gcol])
                        )
                        for precise_pair in range(16):
                            precise_values = txl.alloc_local([2], "float32")
                            for precise_half in range(2):
                                precise_log = txl.local_scalar("float32")
                                precise_token = txl.min(
                                    row0 + txl.int32(2 * precise_pair + precise_half), last
                                )
                                txl.ptx.ld.global_.nc.f32(
                                    precise_log,
                                    g.ptr_to(
                                        [(tok0 + txl.Cast("int64", precise_token)) * HK64 + gcol]
                                    ),
                                )
                                txl.ptx.ex2.approx.ftz.f32(
                                    precise_values[precise_half],
                                    precise_log + _gate_offset(precise_end, range_safe),
                                )
                            txl.assign(
                                egcw[precise_pair],
                                txl.cuda.make_float2(precise_values[0], precise_values[1]),
                            )
                    with txl.If(range_safe & diagonal_needed & txl.Not(strong)), txl.Then():
                        for rebuild_pair in range(16):
                            rebuild_i = 2 * rebuild_pair
                            rebuild_beta = txl.alloc_local([2], "float32")
                            txl.ptx["ld.shared.v2.f32"](
                                rebuild_beta[0],
                                rebuild_beta[1],
                                txl.address_of(s_beta[row0 + rebuild_i]),
                            )
                            rebuild_gate = egcw[rebuild_pair]
                            rebuild_k = txl.cuda.make_float2(
                                lo(kc[rebuild_pair]), hi(kc[rebuild_pair])
                            )
                            rebuild_q = txl.cuda.make_float2(
                                lo(qc[rebuild_pair]), hi(qc[rebuild_pair])
                            )
                            rebuild_full = txl.local_scalar("uint64")
                            rebuild_word = txl.local_scalar("uint32")
                            rebuild_inv = txl.alloc_local([2], "float32")
                            txl.ptx.rcp.approx.ftz.f32(
                                rebuild_inv[0], txl.cuda.float2_x(rebuild_gate)
                            )
                            txl.ptx.rcp.approx.ftz.f32(
                                rebuild_inv[1], txl.cuda.float2_y(rebuild_gate)
                            )
                            for rebuild_tile in range(3):
                                if rebuild_tile == 0:
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_q, rebuild_gate)
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_full, scale_pair)
                                elif rebuild_tile == 1:
                                    txl.ptx["mul.rn.f32x2"](
                                        rebuild_full,
                                        rebuild_k,
                                        txl.cuda.make_float2(rebuild_inv[0], rebuild_inv[1]),
                                    )
                                else:
                                    txl.ptx["mul.rn.f32x2"](rebuild_full, rebuild_k, rebuild_gate)
                                    txl.ptx["mul.rn.f32x2"](
                                        rebuild_full,
                                        rebuild_full,
                                        txl.cuda.make_float2(rebuild_beta[0], rebuild_beta[1]),
                                    )
                                pack_bf16x2(
                                    rebuild_word,
                                    txl.Select(
                                        row0 + txl.int32(rebuild_i) < rows,
                                        txl.cuda.float2_x(rebuild_full),
                                        txl.float32(0.0),
                                    ),
                                    txl.Select(
                                        row0 + txl.int32(rebuild_i + 1) < rows,
                                        txl.cuda.float2_y(rebuild_full),
                                        txl.float32(0.0),
                                    ),
                                )
                                txl.ptx.st.shared.b32(
                                    TT[(T1, T2, T3)[rebuild_tile] + xs].ptr_to(
                                        xr, row0 + rebuild_i
                                    ),
                                    rebuild_word,
                                )
                    # S4 and S6 contain state contractions before the intra sums.
                    # Balance those FP32 accumulators, after their tensor producers
                    # finish at Y_done, to compensate the common intra gate shift.
                    # Execute TMEM operations with all lanes; only the factor varies.
                    with txl.If(range_safe & diagonal_needed), txl.Then():
                        shift_scale = txl.local_scalar("float32")
                        txl.ptx.ex2.approx.ftz.f32(shift_scale, -_gate_offset(gn, range_safe))
                        for shift_slot in (S4, S6):
                            for shift_group in range(4):
                                shift_regs = txl.alloc_local([8], "float32")
                                ld8(shift_regs, shift_slot + wg * 32 + 8 * shift_group)
                                txl.ptx[WAIT_LD]()
                                for shift_i in range(8):
                                    txl.assign(
                                        shift_regs[shift_i], shift_regs[shift_i] * shift_scale
                                    )
                                txl.ptx[TC_ST8](
                                    tmem_at(shift_slot + wg * 32 + 8 * shift_group),
                                    *(shift_regs[z] for z in range(8)),
                                )
                                txl.ptx[WAIT_ST]()
                    # Recover the bf16 rounding residual of beta*k*cached_gate.
                    with txl.If(range_safe & diagonal_needed), txl.Then():
                        # Y_done joins prior H tensor reads; join all scalar H readers too.
                        bar_all()
                        for residual_group in range(4):
                            residual_words = txl.alloc_local([4], "uint32")
                            residual_query_words = txl.alloc_local([4], "uint32")
                            residual_inverse_words = txl.alloc_local([4], "uint32")
                            with txl.If(txl.Not(strong)):
                                with txl.Then():
                                    for residual_pair in range(4):
                                        residual_i = 8 * residual_group + 2 * residual_pair
                                        residual_beta = txl.alloc_local([2], "float32")
                                        residual_high = txl.local_scalar("uint32")
                                        residual_full = txl.local_scalar("uint64")
                                        txl.ptx["ld.shared.v2.f32"](
                                            residual_beta[0],
                                            residual_beta[1],
                                            txl.address_of(s_beta[row0 + residual_i]),
                                        )
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T3 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(kc[residual_i >> 1]), hi(kc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                residual_beta[0], residual_beta[1]
                                            ),
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                        # H readers are complete and the loader waits on b_h_free.
                                        # Stable corrections reuse this tile only after chunk_done.
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T1 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(qc[residual_i >> 1]), hi(qc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full, residual_full, scale_pair
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_query_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                        residual_inv_lo = txl.local_scalar("float32")
                                        residual_inv_hi = txl.local_scalar("float32")
                                        txl.ptx.rcp.approx.ftz.f32(
                                            residual_inv_lo,
                                            txl.Select(
                                                strong,
                                                txl.float32(1.0),
                                                txl.cuda.float2_x(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx.rcp.approx.ftz.f32(
                                            residual_inv_hi,
                                            txl.Select(
                                                strong,
                                                txl.float32(1.0),
                                                txl.cuda.float2_y(egcw[residual_i >> 1]),
                                            ),
                                        )
                                        txl.ptx.ld.shared.b32(
                                            residual_high, TT[T2 + xs].ptr_to(xr, row0 + residual_i)
                                        )
                                        txl.ptx["mul.rn.f32x2"](
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(kc[residual_i >> 1]), hi(kc[residual_i >> 1])
                                            ),
                                            txl.cuda.make_float2(residual_inv_lo, residual_inv_hi),
                                        )
                                        txl.ptx["sub.rn.f32x2"](
                                            residual_full,
                                            residual_full,
                                            txl.cuda.make_float2(
                                                lo(residual_high), hi(residual_high)
                                            ),
                                        )
                                        pack_bf16x2(
                                            residual_inverse_words[residual_pair],
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i) < rows),
                                                txl.cuda.float2_x(residual_full),
                                                txl.float32(0.0),
                                            ),
                                            txl.Select(
                                                txl.Not(strong)
                                                & (row0 + txl.int32(residual_i + 1) < rows),
                                                txl.cuda.float2_y(residual_full),
                                                txl.float32(0.0),
                                            ),
                                        )
                                with txl.Else():
                                    for residual_pair in range(4):
                                        txl.assign(residual_words[residual_pair], txl.uint32(0))
                                        txl.assign(
                                            residual_query_words[residual_pair], txl.uint32(0)
                                        )
                                        txl.assign(
                                            residual_inverse_words[residual_pair], txl.uint32(0)
                                        )
                            txl.ptx["st.shared.v4.b32"](
                                TT[T6 + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_words[z] for z in range(4)),
                            )
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_H + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_query_words[z] for z in range(4)),
                            )
                            txl.ptx["st.shared.v4.b32"](
                                TT[S_H + 2 + xs].ptr_to(xr, row0 + 8 * residual_group),
                                *(residual_inverse_words[z] for z in range(4)),
                            )
                        txl.ptx[FENCE_ASYNC]()
                    marrive("intra_ready")

                    phase("w-epi")
                    twait("dq2_done")
                    phase("epi")

                    def extra_ptr(base, i):
                        return TT[base + 2 * wg + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane)

                    strong_mask = txl.local_scalar("uint32")
                    txl.ptx.vote_sync.ballot.b32(
                        strong_mask, txl.ptx.pred(strong), txl.uint32(0xFFFFFFFF)
                    )

                    def cached_ptr(base, ti):
                        return TT[base + (ti >> 4)].ptr_to(4 * (ti & 15) + quad, 2 * lane)

                    with txl.If(strong_mask != txl.uint32(0)), txl.Then():
                        twait("chunk_done")
                        with txl.If(strong), txl.Then():
                            with txl.serial(32) as i:
                                token = tok0 + txl.Cast("int64", txl.min(row0 + i, last))
                                gate = _load_gate(g, token, head, x, H * D)
                                previous_gate = _load_gate(
                                    g, txl.max(tok0, token - txl.int64(1)), head, x, H * D
                                )
                                txl.ptx.ex2.approx.ftz.f32(gate, gate - previous_gate)
                                q_value = _load_bf16_f32(
                                    q.ptr_to(
                                        [token * txl.int64(H * D) + txl.Cast("int64", head * D + x)]
                                    )
                                )
                                k_value = _load_bf16_f32(
                                    k.ptr_to(
                                        [token * txl.int64(H * D) + txl.Cast("int64", head * D + x)]
                                    )
                                )
                                word = txl.local_scalar("uint32")
                                pack_bf16x2(word, q_value, k_value)
                                txl.ptx.st.shared.f32(cached_ptr(12, row0 + i), gate)
                                txl.ptx.st.shared.b32(cached_ptr(16, row0 + i), word)
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    # Every cache writer publishes its stores before other
                    # warps scan the full chunk, including the other token half.
                    b_stable_full.arrive(0)
                    b_stable_full.wait(0, par)
                    with txl.If(strong_mask != txl.uint32(0)), txl.Then():
                        cooperative_count = txl.local_scalar("uint32")
                        txl.ptx.popc.b32(cooperative_count, strong_mask)
                        with txl.If(cooperative_count <= txl.uint32(12)):
                            with txl.Then():
                                cooperative_left = txl.local_scalar("uint32", init=strong_mask)
                                with txl.While(cooperative_left != txl.uint32(0)):
                                    cooperative_channel = txl.local_scalar(
                                        "int32", init=txl.int32(0)
                                    )
                                    cooperative_valid = txl.local_scalar("bool", init=False)
                                    for group in range(4):
                                        cooperative_bit = txl.local_scalar("uint32")
                                        txl.ptx.bfind.u32(cooperative_bit, cooperative_left)
                                        txl.assign(
                                            cooperative_channel,
                                            txl.Select(
                                                (lane // 8) == group,
                                                txl.Cast("int32", cooperative_bit & txl.uint32(31)),
                                                cooperative_channel,
                                            ),
                                        )
                                        txl.assign(
                                            cooperative_valid,
                                            cooperative_valid
                                            | (
                                                ((lane // 8) == group)
                                                & (cooperative_left != txl.uint32(0))
                                            ),
                                        )
                                        txl.assign(
                                            cooperative_left,
                                            cooperative_left
                                            & ~(
                                                txl.uint32(1) << (cooperative_bit & txl.uint32(31))
                                            ),
                                        )
                                    cooperative_row = row0 + (lane % 8) * 4
                                    cooperative_x = quad * 32 + cooperative_channel

                                    def cooperative_ptr(base, token_row):
                                        return TT[base + (token_row >> 4)].ptr_to(
                                            4 * (token_row & 15) + quad, 2 * cooperative_channel
                                        )

                                    with txl.If(cooperative_valid), txl.Then():
                                        cooperative_qe = txl.alloc_local([4], "float32")
                                        cooperative_kp = txl.alloc_local([4], "float32")
                                        cooperative_kf = txl.alloc_local([4], "float32")
                                        cooperative_decay = txl.alloc_local([4], "float32")
                                        for e in range(4):
                                            txl.assign(cooperative_qe[e], txl.float32(0.0))
                                            txl.assign(cooperative_kp[e], txl.float32(0.0))
                                            txl.assign(cooperative_kf[e], txl.float32(0.0))
                                            txl.assign(cooperative_decay[e], txl.float32(1.0))
                                        cooperative_j = txl.local_scalar(
                                            "int32",
                                            init=txl.min(cooperative_row + txl.int32(4 - 1), last),
                                        )
                                        with txl.While(
                                            (cooperative_j >= txl.int32(0))
                                            & (cooperative_decay[0] != txl.float32(0.0))
                                            & (cooperative_j >= cooperative_row)
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_qe[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    txl.Select(
                                                        cooperative_j == cooperative_row + e,
                                                        txl.float32(0.0),
                                                        cooperative_aq,
                                                    ),
                                                    cooperative_qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kp[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_ak,
                                                    cooperative_kp[e],
                                                )
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e]
                                                    * txl.Select(
                                                        cooperative_j <= cooperative_row + e,
                                                        cooperative_alpha,
                                                        txl.float32(1.0),
                                                    ),
                                                )
                                            txl.assign(cooperative_j, cooperative_j - txl.int32(1))
                                        with txl.While(
                                            (cooperative_j >= txl.int32(0))
                                            & (cooperative_decay[0] != txl.float32(0.0))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_row + e, cooperative_j
                                                    ),
                                                    shared=True,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_qe[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_aq,
                                                    cooperative_qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kp[e],
                                                    hi(cooperative_word) * cooperative_decay[e],
                                                    cooperative_ak,
                                                    cooperative_kp[e],
                                                )
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e] * cooperative_alpha,
                                                )
                                            txl.assign(cooperative_j, cooperative_j - txl.int32(1))
                                        for e in range(4):
                                            txl.assign(cooperative_decay[e], txl.float32(1.0))
                                        txl.assign(cooperative_j, cooperative_row)
                                        with txl.While(
                                            (cooperative_j < rows)
                                            & (cooperative_decay[4 - 1] != txl.float32(0.0))
                                            & (cooperative_j < cooperative_row + txl.int32(4))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e]
                                                    * txl.Select(
                                                        cooperative_j > cooperative_row + e,
                                                        cooperative_alpha,
                                                        txl.float32(1.0),
                                                    ),
                                                )
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_term,
                                                    lo(cooperative_word) * scale,
                                                    txl.Select(
                                                        cooperative_j == cooperative_row + e,
                                                        txl.float32(0.0),
                                                        cooperative_aq,
                                                    ),
                                                    hi(cooperative_word)
                                                    * s_beta_row(cooperative_j)
                                                    * cooperative_ak,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kf[e],
                                                    cooperative_term,
                                                    cooperative_decay[e],
                                                    cooperative_kf[e],
                                                )
                                            txl.assign(cooperative_j, cooperative_j + txl.int32(1))
                                        with txl.While(
                                            (cooperative_j < rows)
                                            & (cooperative_decay[4 - 1] != txl.float32(0.0))
                                        ):
                                            cooperative_alpha = txl.local_scalar("float32")
                                            cooperative_word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(
                                                cooperative_alpha,
                                                cooperative_ptr(12, cooperative_j),
                                            )
                                            txl.ptx.ld.shared.b32(
                                                cooperative_word, cooperative_ptr(16, cooperative_j)
                                            )
                                            for e in range(4):
                                                txl.assign(
                                                    cooperative_decay[e],
                                                    cooperative_decay[e] * cooperative_alpha,
                                                )
                                                cooperative_aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(
                                                        cooperative_j, cooperative_row + e
                                                    ),
                                                    shared=True,
                                                )
                                                cooperative_term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_term,
                                                    lo(cooperative_word) * scale,
                                                    cooperative_aq,
                                                    hi(cooperative_word)
                                                    * s_beta_row(cooperative_j)
                                                    * cooperative_ak,
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    cooperative_kf[e],
                                                    cooperative_term,
                                                    cooperative_decay[e],
                                                    cooperative_kf[e],
                                                )
                                            txl.assign(cooperative_j, cooperative_j + txl.int32(1))
                                        for e in range(4):
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(0, cooperative_row + e),
                                                cooperative_qe[e] * scale,
                                            )
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(4, cooperative_row + e),
                                                cooperative_kp[e],
                                            )
                                            txl.ptx.st.shared.f32(
                                                cooperative_ptr(20, cooperative_row + e),
                                                cooperative_kf[e],
                                            )
                            with txl.Else():
                                with txl.If(strong), txl.Then():
                                    with txl.serial(4) as block:
                                        i0 = 8 * block
                                        ti0 = row0 + i0
                                        qe = txl.alloc_local([8], "float32")
                                        kp = txl.alloc_local([8], "float32")
                                        kf = txl.alloc_local([8], "float32")
                                        decay = txl.alloc_local([8], "float32")
                                        for e in range(8):
                                            txl.assign(qe[e], txl.float32(0.0))
                                            txl.assign(kp[e], txl.float32(0.0))
                                            txl.assign(kf[e], txl.float32(0.0))
                                            txl.assign(decay[e], txl.float32(1.0))
                                        j = txl.local_scalar(
                                            "int32", init=txl.min(ti0 + txl.int32(7), last)
                                        )
                                        # With alpha in [0, 1], the earliest row is last to
                                        # underflow in this direction. Only skip exact zeros.
                                        with txl.While(
                                            (j >= txl.int32(0))
                                            & (decay[0] != txl.float32(0.0))
                                            & (j >= ti0)
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(8):
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(ti0 + e, j), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(ti0 + e, j), shared=True
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    qe[e],
                                                    hi(word) * decay[e],
                                                    txl.Select(j == ti0 + e, txl.float32(0.0), aq),
                                                    qe[e],
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    kp[e], hi(word) * decay[e], ak, kp[e]
                                                )
                                                txl.assign(
                                                    decay[e],
                                                    decay[e]
                                                    * txl.Select(
                                                        j <= ti0 + e, alpha, txl.float32(1.0)
                                                    ),
                                                )
                                            txl.assign(j, j - txl.int32(1))
                                        with txl.While(
                                            (j >= txl.int32(0)) & (decay[0] != txl.float32(0.0))
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(8):
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(ti0 + e, j), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(ti0 + e, j), shared=True
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    qe[e], hi(word) * decay[e], aq, qe[e]
                                                )
                                                txl.ptx.fma.rn.f32(
                                                    kp[e], hi(word) * decay[e], ak, kp[e]
                                                )
                                                txl.assign(decay[e], decay[e] * alpha)
                                            txl.assign(j, j - txl.int32(1))
                                        for e in range(8):
                                            txl.assign(decay[e], txl.float32(1.0))
                                        txl.assign(j, ti0)
                                        # In the forward direction the latest row is last.
                                        with txl.While(
                                            (j < rows)
                                            & (decay[7] != txl.float32(0.0))
                                            & (j < ti0 + txl.int32(8))
                                        ):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(8):
                                                txl.assign(
                                                    decay[e],
                                                    decay[e]
                                                    * txl.Select(
                                                        j > ti0 + e, alpha, txl.float32(1.0)
                                                    ),
                                                )
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(j, ti0 + e), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(j, ti0 + e), shared=True
                                                )
                                                term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    term,
                                                    lo(word) * scale,
                                                    txl.Select(j == ti0 + e, txl.float32(0.0), aq),
                                                    hi(word) * s_beta_row(j) * ak,
                                                )
                                                txl.ptx.fma.rn.f32(kf[e], term, decay[e], kf[e])
                                            txl.assign(j, j + txl.int32(1))
                                        with txl.While((j < rows) & (decay[7] != txl.float32(0.0))):
                                            alpha = txl.local_scalar("float32")
                                            word = txl.local_scalar("uint32")
                                            txl.ptx.ld.shared.f32(alpha, cached_ptr(12, j))
                                            txl.ptx.ld.shared.b32(word, cached_ptr(16, j))
                                            for e in range(8):
                                                txl.assign(decay[e], decay[e] * alpha)
                                                aq = _load_bf16_f32(
                                                    TT[T5].ptr_to(j, ti0 + e), shared=True
                                                )
                                                ak = _load_bf16_f32(
                                                    TT[T5 + 1].ptr_to(j, ti0 + e), shared=True
                                                )
                                                term = txl.local_scalar("float32")
                                                txl.ptx.fma.rn.f32(
                                                    term,
                                                    lo(word) * scale,
                                                    aq,
                                                    hi(word) * s_beta_row(j) * ak,
                                                )
                                                txl.ptx.fma.rn.f32(kf[e], term, decay[e], kf[e])
                                            txl.assign(j, j + txl.int32(1))
                                        for e in range(8):
                                            txl.ptx.st.shared.f32(
                                                extra_ptr(0, i0 + e), qe[e] * scale
                                            )
                                            txl.ptx.st.shared.f32(extra_ptr(4, i0 + e), kp[e])
                                            txl.ptx.st.shared.f32(extra_ptr(20, i0 + e), kf[e])
                        txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                        txl.ptx[FENCE_ASYNC]()
                        with txl.If(strong), txl.Then():
                            b_mid_free.arrive(0)

                    with txl.If(state_strong & txl.Not(strong)), txl.Then():
                        b_mid_free.arrive(0)

                    txl.ptx[FENCE_ASYNC]()
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    with txl.If(lane == txl.int32(0)), txl.Then():
                        b_intra_free.arrive(0, count=32)
                    dgk_k2 = txl.local_scalar(
                        "uint64", init=txl.cuda.make_float2(txl.float32(0.0), txl.float32(0.0))
                    )

                    def q_loads(b, base):
                        ld8(acc, S4 + wg * 32 + 8 * b, base)

                    def k_loads(b, base):
                        ld4(acc, S5 + wg * 32 + 4 * b, base + 0)
                        ld4(acc, S6 + wg * 32 + 4 * b, base + 4)
                        ld4(acc, S3 + wg * 32 + 4 * b, base + 8)

                    def add_diagonal(dst, i, inputs):
                        with txl.If(diagonal_needed), txl.Then():
                            diagonal = txl.alloc_local([2], "float32")
                            txl.ptx["ld.shared.v2.f32"](
                                diagonal[0], diagonal[1], s_beta1.ptr_to([0, row0 + i])
                            )
                            txl.ptx["fma.rn.f32x2"](
                                dst,
                                txl.cuda.make_float2(lo(inputs), hi(inputs)),
                                txl.cuda.make_float2(diagonal[0], diagonal[1]),
                                dst,
                            )

                    def epilogue(stable):
                        pair0 = txl.local_scalar("uint64")
                        pair1 = txl.local_scalar("uint64")
                        pair2 = txl.local_scalar("uint64")
                        pair3 = txl.local_scalar("uint64")
                        pair4 = txl.local_scalar("uint64")
                        pair5 = txl.local_scalar("uint64")

                        q_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(4):
                            ab = 8 * (b % 2)
                            if b < 3:
                                q_loads(b + 1, 8 * ((b + 1) % 2))
                            for p in range(4):
                                i = 8 * b + 2 * p

                                txl.ptx.rcp.approx.ftz.f32(
                                    enA[i >> 1],
                                    txl.Select(
                                        strong, txl.float32(1.0), txl.cuda.float2_x(egcw[i >> 1])
                                    ),
                                )
                                txl.ptx.rcp.approx.ftz.f32(
                                    enB[i >> 1],
                                    txl.Select(
                                        strong, txl.float32(1.0), txl.cuda.float2_y(egcw[i >> 1])
                                    ),
                                )
                                txl.assign(
                                    enA[i >> 1], txl.Select(strong, txl.float32(0.0), enA[i >> 1])
                                )
                                txl.assign(
                                    enB[i >> 1], txl.Select(strong, txl.float32(0.0), enB[i >> 1])
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(
                                        txl.cuda.float2_x(egcw[i >> 1]),
                                        txl.cuda.float2_y(egcw[i >> 1]),
                                    ),
                                    txl.cuda.make_float2(scale, scale),
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair0,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    pair1,
                                )
                                if stable:
                                    with txl.If(strong), txl.Then():
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(0, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(0, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair0, pair0, txl.cuda.make_float2(t0, t1)
                                        )
                                txl.ptx["mul.rn.f32x2"](
                                    pair1,
                                    txl.cuda.make_float2(lo(qc[i >> 1]), hi(qc[i >> 1])),
                                    pair0,
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair1))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair1))
                                add_diagonal(pair0, i, kc[i >> 1])
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.cuda.float2_x(pair0),
                                )
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dq.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.cuda.float2_y(pair0),
                                )
                            if b < 3:
                                txl.ptx[WAIT_LD]()
                        twait("dkt_done")

                        dbx = 2 * wg
                        k_loads(0, 0)
                        txl.ptx[WAIT_LD]()
                        for b in range(8):
                            ab = 12 * (b % 2)
                            if b < 7:
                                k_loads(b + 1, 12 * ((b + 1) % 2))
                            for p in range(2):
                                i = 4 * b + 2 * p
                                if stable:
                                    txl.assign(
                                        pair0,
                                        txl.cuda.make_float2(
                                            txl.Select(strong, txl.float32(0.0), enA[i >> 1]),
                                            txl.Select(strong, txl.float32(0.0), enB[i >> 1]),
                                        ),
                                    )
                                else:
                                    txl.assign(
                                        pair0, txl.cuda.make_float2(enA[i >> 1], enB[i >> 1])
                                    )
                                txl.assign(
                                    pair1, txl.cuda.make_float2(lo(kc[i >> 1]), hi(kc[i >> 1]))
                                )
                                state_decay(t0, i, txl.cuda.float2_x(egcw[i >> 1]))
                                state_decay(t1, i + 1, txl.cuda.float2_y(egcw[i >> 1]))
                                state_pair = txl.local_scalar("uint64")
                                txl.ptx["mul.rn.f32x2"](
                                    state_pair,
                                    txl.cuda.make_float2(acc[ab + 2 * p], acc[ab + 2 * p + 1]),
                                    txl.cuda.make_float2(t0, t1),
                                )
                                txl.ptx["fma.rn.f32x2"](
                                    pair2,
                                    txl.cuda.make_float2(
                                        acc[ab + 8 + 2 * p], acc[ab + 8 + 2 * p + 1]
                                    ),
                                    pair0,
                                    state_pair,
                                )
                                txl.ptx["mul.rn.f32x2"](
                                    pair3,
                                    txl.cuda.make_float2(
                                        acc[ab + 4 + 2 * p], acc[ab + 4 + 2 * p + 1]
                                    ),
                                    txl.cuda.make_float2(
                                        txl.cuda.float2_x(egcw[i >> 1]),
                                        txl.cuda.float2_y(egcw[i >> 1]),
                                    ),
                                )
                                if stable:
                                    with txl.If(strong), txl.Then():
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(20, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(20, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair2, pair2, txl.cuda.make_float2(t0, t1)
                                        )
                                        txl.ptx.ld.shared.f32(t0, extra_ptr(4, i))
                                        txl.ptx.ld.shared.f32(t1, extra_ptr(4, i + 1))
                                        txl.ptx["add.rn.f32x2"](
                                            pair3, pair3, txl.cuda.make_float2(t0, t1)
                                        )
                                txl.ptx["mul.rn.f32x2"](pair4, pair1, pair3)
                                txl.ptx.st.shared.f32(
                                    TT[dbx + (i >> 4)].ptr_to(4 * (i & 15) + quad, 2 * lane),
                                    txl.cuda.float2_x(pair4),
                                )
                                txl.ptx.st.shared.f32(
                                    TT[dbx + ((i + 1) >> 4)].ptr_to(
                                        4 * ((i + 1) & 15) + quad, 2 * lane
                                    ),
                                    txl.cuda.float2_y(pair4),
                                )
                                txl.ptx["mul.rn.f32x2"](pair5, pair1, state_pair)
                                txl.ptx["add.rn.f32x2"](dgk_k2, dgk_k2, pair5)
                                beta_pair = txl.cuda.make_float2(
                                    s_beta_row(row0 + i), s_beta_row(row0 + i + 1)
                                )
                                txl.ptx["fma.rn.f32x2"](pair5, pair3, beta_pair, pair2)
                                txl.ptx["fma.rn.f32x2"](
                                    pair3,
                                    pair2,
                                    txl.cuda.make_float2(txl.float32(-2.0), txl.float32(-2.0)),
                                    pair5,
                                )
                                add_diagonal(pair5, i, qc[i >> 1])
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + txl.int64(i * HK)]),
                                    txl.cuda.float2_x(pair5),
                                )
                                txl.ptx["st.global.L1::no_allocate.f32"](
                                    dk.ptr_to([x_base + txl.int64((i + 1) * HK)]),
                                    txl.cuda.float2_y(pair5),
                                )
                                txl.ptx["fma.rn.f32x2"](
                                    pair5, pair1, pair3, txl.cuda.make_float2(dgv[i], dgv[i + 1])
                                )
                                txl.assign(dgv[i], txl.cuda.float2_x(pair5))
                                txl.assign(dgv[i + 1], txl.cuda.float2_y(pair5))
                            if b < 7:
                                txl.ptx[WAIT_LD]()

                    with txl.If(strong_mask != txl.uint32(0)):
                        with txl.Then():
                            epilogue(True)
                        with txl.Else():
                            epilogue(False)
                    dbx = 2 * wg
                    txl.assign(dgk_k, txl.cuda.float2_x(dgk_k2) + txl.cuda.float2_y(dgk_k2))
                    bar_wg()
                    dsum = txl.local_scalar("float32", init=dsum_v)
                    for u in range(8):
                        txl.ptx["ld.shared.v4.f32"](
                            t4[0], t4[1], t4[2], t4[3], TT[dbx + (quad >> 1)].ptr_to(srow, 8 * u)
                        )
                        txl.assign(dsum, dsum + ((t4[0] + t4[1]) + (t4[2] + t4[3])))
                    txl.ptx[FENCE_ASYNC]()
                    b_h_free.arrive(0)
                    for s in (1, 2):
                        r = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.bfly.b32(
                            r,
                            txl.reinterpret("uint32", dsum),
                            txl.uint32(s),
                            txl.uint32(0x1F),
                            txl.uint32(0xFFFFFFFF),
                        )
                        txl.assign(dsum, dsum + txl.reinterpret("float32", r))
                    with txl.If(tq == txl.int32(0)), txl.Then():
                        txl.ptx["st.global.L1::no_allocate.f32"](
                            db.ptr_to(
                                [(tok0 + txl.Cast("int64", row0 + ti)) * txl.int64(H) + head64]
                            ),
                            dsum,
                        )

                    phase("cumsum")

                    for i in range(30, -1, -1):
                        txl.assign(dgv[i], dgv[i] + dgv[i + 1])
                    txl.ptx.st.shared.f32(
                        txl.address_of(s_dgk[wg, x]),
                        dgk + dgk_k + txl.Select(wg == txl.int32(0), txl.float32(0.0), dgv[0]),
                    )
                    b_dg0_ready.arrive(0)
                    b_dg0_ready.wait(0, cyc & txl.int32(1))
                    txl.ptx.ld.shared.f32(t0, txl.address_of(s_dgk[txl.int32(1) - wg, x]))
                    txl.assign(t1, t0 + dgk + dgk_k)
                    for i in range(32):
                        txl.assign(dgv[i], dgv[i] + t1)
                    for i in range(32):
                        txl.ptx["st.global.L1::no_allocate.f32"](
                            dg.ptr_to([x_base + txl.int64(i * HK)]), dgv[i]
                        )
                    phase_end()
                    txl.assign(cyc, cyc + txl.int32(1))

                phase("dh0")
                TC["chunk_done"].wait(0, (cyc & txl.int32(1)) ^ txl.int32(1))
                txl.ptx[TC_FENCE_AFTER]()
                ld32(acc, TM_DH + wg * 64)
                ld32(acc, TM_DH + wg * 64 + 32, 32)
                txl.ptx[WAIT_LD]()
                obase = (
                    (txl.Cast("int64", seq) * txl.int64(H) + head64) * txl.int64(D) + x64
                ) * txl.int64(D) + txl.Cast("int64", wg * 64)
                for m in range(8):
                    txl.ptx["st.global.L1::no_allocate.v8.f32"](
                        dh0.ptr_to([obase + txl.int64(8 * m)]), *(acc[8 * m + i] for i in range(8))
                    )
                phase_end()

        with auxg:
            with mma:
                p1_mma()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                tm = tmem_preamble()
                cyc = txl.local_scalar("int32", init=txl.int32(0))

                def mwait(nm):
                    MB[nm].wait(0, cyc & txl.int32(1))
                    txl.ptx[TC_FENCE_AFTER]()

                mphase, mphase_end = make_phaser()

                bd = txl.alloc_local([1], "uint64")
                zq = txl.alloc_local([1], "int32")
                op_T1k = Op(bd, T1, 128, 64, "k")
                op_T1mn = Op(bd, T1, 128, 128, "mn")
                op_T2k = Op(bd, T2, 128, 64, "k")
                op_T2mn = Op(bd, T2, 128, 128, "mn")
                op_T3k = Op(bd, T3, 128, 64, "k")
                op_residual_k = Op(bd, T6, 128, 64, "k")
                op_residual_q = Op(bd, S_H, 128, 64, "k")
                op_residual_inverse = Op(bd, S_H + 2, 128, 64, "k")
                op_T3mn = Op(bd, T3, 128, 128, "mn")
                op_T5k = Op(bd, T5, 128, 64, "k")
                op_ZTk = Op(bd, ZT, 128, 64, "k")
                op_ZTmn = Op(bd, ZT, 128, 128, "mn")
                op_dAqk_k = Op(bd, T5, 64, 64, "k")
                op_dAkk_k = Op(bd, T5 + 1, 64, 64, "k")
                op_T6mn = Op(bd, T6, 128, 128, "mn")
                op_dAqk_mn = Op(bd, T5, 64, 64, "mn")
                op_dAm_k = Op(bd, DAM, 64, 64, "k")
                op_X_mn = Op(bd, T6, 64, 64, "mn")
                op_dAkk_mn = Op(bd, T5 + 1, 64, 64, "mn")
                op_DHBk = Op(bd, DHB, 128, 128, "k")
                op_DHBmn = Op(bd, DHB, 128, 128, "mn")
                op_DV2k = Op(bd, DV2, 128, 64, "k")
                op_DV2mn = Op(bd, DV2, 128, 128, "mn")
                op_DVBk = Op(bd, DVB, 128, 64, "k")
                op_DVBmn = Op(bd, DVB, 128, 128, "mn")
                op_do_k128 = Op(bd, S_DO, 64, 128, "k")
                op_do_mn64 = Op(bd, S_DO, 64, 64, "mn")
                op_h_k = Op(bd, S_H, 128, 128, "k")
                op_h_mn = Op(bd, S_H, 128, 128, "mn")
                op_aqk_mn = Op(bd, S_AQK, 64, 64, "mn")
                op_akk_k = Op(bd, S_AKK, 64, 64, "k")
                op_akk_mn = Op(bd, S_AKK, 64, 64, "mn")
                ID_128x64 = idesc(128, 64)
                ID_128x64_TATB_NB = idesc(128, 64, ta=1, tb=1, nb=1)
                ID_128x64_TATB = idesc(128, 64, ta=1, tb=1)
                ID_128x64_TB = idesc(128, 64, tb=1)
                ID_128x64_TB_NA = idesc(128, 64, tb=1, na=1)
                ID_128x128_TB = idesc(128, 128, tb=1)
                ID_128x128_NB = idesc(128, 128, nb=1)
                ID_128x128 = idesc(128, 128)
                ID_64x64_TB = idesc(64, 64, tb=1)
                ID_64x64_TATB = idesc(64, 64, ta=1, tb=1)
                ID_64x64 = idesc(64, 64)

                op_egT = Op(bd, ST_G, 64, 64, "mn")
                op_vT = Op(bd, ST_V, 64, 64, "mn")
                op_qT = Op(bd, ST_Q, 64, 64, "mn")
                op_kT = Op(bd, ST_K, 64, 64, "mn")
                ID_T = idesc(128, 16, ta=1)
                bdI = txl.alloc_local([1], "uint64")
                txl.cuda.tcgen05.encode_matrix_descriptor(
                    txl.address_of(bdI[0]), s_ident.ptr_to([0]), ldo=8, sdo=16, swizzle=0
                )

                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with txl.serial(nch) as rn:
                        par = cyc & txl.int32(1)
                        txl.ptx.ld.volatile.shared.s32(zq[0], txl.address_of(s_tmem[1]))
                        txl.cuda.tcgen05.encode_matrix_descriptor(
                            txl.address_of(bd[0]),
                            TT[zq[0]].ptr_to(0, 0),
                            ldo=Op.LBO_BASE,
                            sdo=SBO_UNITS,
                            swizzle=txl.SW128B.value,
                        )
                        akk_u = txl.local_scalar(
                            "uint64", init=txl.Cast("uint64", par) * txl.uint64(UNITS_PER_STAGE)
                        )
                        mphase("mw-xT")
                        b_in_full.wait(0, par)
                        b_eg_full.wait(0, par)

                        b_h_free.wait(0, par ^ txl.int32(1))
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-xT")
                        with txl.If(elected()), txl.Then():
                            for src, dst in ((op_egT, S1), (op_vT, S2), (op_qT, S3), (op_kT, S4)):
                                for j in range(4):
                                    txl.ptx[MMA_SS](
                                        txl.Cast("uint32", tm[0] + dst + 16 * j),
                                        src.desc(j),
                                        bdI[0],
                                        txl.uint32(ID_T),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.uint32(0),
                                        txl.ptx.pred(0),
                                    )
                            TC["xT_done"].arrive(0)
                        mphase("mw-early")
                        mwait("t_early")
                        b_akk_full.wait(par, (cyc >> 1) & txl.int32(1))
                        b_h_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-Z")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_h_mn, op_T3mn, ID_128x64_TATB_NB, True)
                            TC["Z_done"].arrive(0)
                        mphase("mw-aqk")
                        b_aqk_masked.wait(0, par)
                        b_do_full.wait(0, par)
                        txl.ptx[TC_FENCE_AFTER]()
                        mphase("m-dvp")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_do_mn64, op_aqk_mn, ID_128x64_TATB, False)
                            b_aqk_empty.arrive(0)
                        mphase("mw-dhb")
                        mwait("dhb_ready")
                        mphase("m-dv2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_DHBmn, op_T2mn, ID_128x64_TATB, True)
                            TC["dv2_done"].arrive(0)
                        mphase("mw-zT")
                        mwait("zT_ready")
                        mphase("m-Vn")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S2, op_ZTk, op_akk_k, ID_128x64, False, b_units=akk_u)
                            TC["Vn_done"].arrive(0)
                        mphase("mw-dv2T")
                        mwait("dv2T_ready")
                        mphase("m-dAs")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S4, op_DV2mn, op_ZTmn, ID_64x64_TATB, False)
                            TC["dAs_done"].arrive(0)
                            mma_chain(
                                tm, S3, op_DV2k, op_akk_mn, ID_128x64_TB, False, b_units=akk_u
                            )
                            TC["dvb_done"].arrive(0)
                        mphase("mw-vnT")
                        mwait("vnT_ready")
                        mphase("m-dAqk")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_do_k128, op_T6mn, ID_64x64_TB, False)
                            TC["dAqk_done"].arrive(0)
                            mma_chain(tm, S5, op_DHBk, op_T6mn, ID_128x64_TB, False)
                            TC["dk_done"].arrive(0)
                        mphase("mw-dAm")
                        mwait("dAm_ready")
                        mwait("dAqk_tile_ready")
                        mphase("m-X")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S1, op_dAm_k, op_akk_k, ID_64x64, False, b_units=akk_u)
                            TC["X_done"].arrive(0)

                            mma_chain(tm, S4, op_h_k, op_do_k128, ID_128x64, False)
                            with txl.If(txl.Not(range_safe & diagonal_needed)), txl.Then():
                                mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                            with txl.If(txl.Not(range_safe & diagonal_needed)), txl.Then():
                                TC["dq2_done"].arrive(0)
                            mma_chain(tm, TM_DH, op_T1k, op_do_mn64, ID_128x128_TB, True)
                            b_do_empty.arrive(0)
                        mphase("mw-dvepi")
                        mwait("dv_epi_done")
                        mphase("m-dwb")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_h_k, op_DVBmn, ID_128x64_TB_NA, False)
                        mphase("mw-X")
                        mwait("X_ready")
                        mphase("m-Y")
                        with txl.If(elected()), txl.Then():
                            # Finish the state use of the original T3 before
                            # Y_done lets CG rebuild it for intra derivatives.
                            mma_chain(tm, TM_DH, op_T3k, op_DVBk, ID_128x128_NB, True)
                            mma_chain(
                                tm, S2, op_akk_mn, op_X_mn, ID_64x64_TATB, False, a_units=akk_u
                            )
                            TC["Y_done"].arrive(0)
                            b_akk_empty.arrive(par)
                        mphase("mw-intra")
                        mwait("intra_ready")
                        mphase("m-dk2")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S6, op_T2k, op_dAkk_k, ID_128x64, True)
                        mphase("m-dkt")
                        with txl.If(elected()), txl.Then():
                            mma_chain(tm, S3, op_T1k, op_dAqk_mn, ID_128x64_TB, False)
                            mma_chain(tm, S3, op_T3k, op_dAkk_mn, ID_128x64_TB, True)
                            with txl.If(range_safe & diagonal_needed), txl.Then():
                                mma_chain(tm, S3, op_residual_k, op_dAkk_mn, ID_128x64_TB, True)
                                mma_chain(tm, S3, op_residual_q, op_dAqk_mn, ID_128x64_TB, True)
                            with txl.If(range_safe & diagonal_needed), txl.Then():
                                mma_chain(tm, S4, op_T2k, op_dAqk_k, ID_128x64, True)
                                mma_chain(tm, S4, op_residual_inverse, op_dAqk_k, ID_128x64, True)
                                mma_chain(tm, S6, op_residual_inverse, op_dAkk_k, ID_128x64, True)
                                TC["dq2_done"].arrive(0)
                            TC["dkt_done"].arrive(0)

                            TC["chunk_done"].arrive(0)
                        mphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))

            with loader:
                with txl.If(elected()), txl.Then():
                    for m in (
                        q_map,
                        k_map,
                        v_map,
                        g_map,
                        eg_map,
                        beta_map,
                        do_map,
                        aqk_map,
                        akk_map,
                        h_map,
                    ):
                        txl.ptx.prefetch.tensormap(txl.address_of(m))
                p1_loader()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                lphase, lphase_end = make_phaser()
                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    bos32 = txl.local_scalar("int32", init=txl.Cast("int32", bos))
                    cb = chunk_base(seq)
                    head8 = txl.local_scalar("int32", init=head >> txl.int32(3))

                    lphase("lw-flag")
                    with txl.If(elected()), txl.Then():
                        # The epoch flag is a declared synchronization word.
                        fl = txl.local_scalar("int32", init=txl.int32(0))
                        txl.cuda.wait_until(fl, flags.ptr_to([work]), fl == epoch, scope="gpu")
                    txl.ptx["bar.warp.sync"](txl.uint32(0xFFFFFFFF))
                    txl.ptx["fence.proxy.async.global"]()
                    lphase_end()
                    with txl.serial(nch) as rn:
                        n = nch - txl.int32(1) - rn
                        par = cyc & txl.int32(1)
                        npar = par ^ txl.int32(1)
                        tok0 = bos32 + n * txl.int32(CHUNK)
                        hidx = (cb + n) * txl.int32(H) + head

                        lphase("lw-mid")
                        b_mid_free.wait(0, npar)
                        lphase("l-issue")
                        with txl.If(elected()), txl.Then():
                            b_in_full.arrive(0, tx_count=IN_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_in_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                s_beta_in.ptr_to([0, 0]),
                                txl.address_of(beta_map),
                                txl.int32(0),
                                tok0,
                                head8,
                                mb,
                            )
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_Q + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(q_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_K + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(k_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                                txl.ptx[TMA_LD](
                                    TT[ST_V + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(v_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        lphase("lw-chunk")
                        TC["chunk_done"].wait(0, npar)
                        # dAqk/dAkk stay live through the stable intra pass.
                        b_intra_free.wait(0, npar)
                        lphase("l-issue-eg")
                        with txl.If(elected()), txl.Then():
                            b_eg_full.arrive(0, tx_count=EG_BYTES)
                            mbe = txl.cuda.cvta_generic_to_shared(b_eg_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[ST_G + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(eg_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mbe,
                                )
                        b_do_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_do_full.arrive(0, tx_count=DO_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_do_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_DO + d0 // 64].ptr_to(0, 0),
                                    txl.address_of(do_map),
                                    txl.int32(d0),
                                    tok0,
                                    head,
                                    mb,
                                )
                        b_h_free.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_h_full.arrive(0, tx_count=H_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_h_full.ptr_to([0]))
                            for d0 in (0, 64):
                                txl.ptx[TMA_LD](
                                    TT[S_H + (d0 // 64) * 2].ptr_to(0, 0),
                                    txl.address_of(h_map),
                                    txl.int32(d0),
                                    txl.int32(0),
                                    hidx,
                                    mb,
                                )
                        b_aqk_empty.wait(0, npar)
                        with txl.If(elected()), txl.Then():
                            b_aqk_full.arrive(0, tx_count=AQK_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_aqk_full.ptr_to([0]))
                            txl.ptx[TMA_LD](
                                TT[S_AQK].ptr_to(0, 0),
                                txl.address_of(aqk_map),
                                txl.int32(0),
                                tok0,
                                head,
                                mb,
                            )
                        b_akk_empty.wait(par, ((cyc >> 1) & txl.int32(1)) ^ txl.int32(1))
                        with txl.If(elected()), txl.Then():
                            b_akk_full.arrive(par, tx_count=AQK_BYTES)
                            mb = txl.cuda.cvta_generic_to_shared(b_akk_full.ptr_to([par]))
                            txl.ptx[TMA_LD](
                                TT[S_AKK + par].ptr_to(0, 0),
                                txl.address_of(akk_map),
                                txl.int32(0),
                                tok0,
                                head,
                                mb,
                            )
                            with txl.If(n == txl.int32(0)), txl.Then():
                                with txl.If(i2 + txl.int32(1) < n_p2), txl.Then():
                                    nxt = p2_chain(i2 + txl.int32(1))
                                    seq2, head2, bos2, seq_len2, nch2 = work_coords(nxt)
                                    tokn = txl.Cast("int32", bos2) + (
                                        nch2 - txl.int32(1)
                                    ) * txl.int32(CHUNK)
                                    hidn = (chunk_base(seq2) + nch2 - txl.int32(1)) * txl.int32(
                                        H
                                    ) + head2
                                    for tmap in (q_map, k_map, v_map, do_map, eg_map):
                                        for d0 in (0, 64):
                                            txl.ptx[TMA_PREFETCH](
                                                txl.address_of(tmap), txl.int32(d0), tokn, head2
                                            )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(aqk_map), txl.int32(0), tokn, head2
                                    )
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(akk_map), txl.int32(0), tokn, head2
                                    )
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(h_map), txl.int32(d0), txl.int32(0), hidn
                                        )
                            with txl.If(n > txl.int32(0)), txl.Then():
                                tokp = tok0 - txl.int32(CHUNK)
                                for tmap in (q_map, k_map, v_map, do_map):
                                    for d0 in (0, 64):
                                        txl.ptx[TMA_PREFETCH](
                                            txl.address_of(tmap), txl.int32(d0), tokp, head
                                        )
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(eg_map), txl.int32(d0), tokp, head
                                    )
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(aqk_map), txl.int32(0), tokp, head
                                )
                                txl.ptx[TMA_PREFETCH](
                                    txl.address_of(akk_map), txl.int32(0), tokp, head
                                )
                                for d0 in (0, 64):
                                    txl.ptx[TMA_PREFETCH](
                                        txl.address_of(h_map),
                                        txl.int32(d0),
                                        txl.int32(0),
                                        hidx - txl.int32(H),
                                    )
                        lphase_end()
                        txl.assign(cyc, cyc + txl.int32(1))

            with idle:
                with txl.If(txl.warp_id_in_role() == txl.int32(0)), txl.Then():
                    p1_storer()
                txl.ptx.bar.sync(txl.uint32(5), txl.uint32(384))
                cyc = txl.local_scalar("int32", init=txl.int32(0))
                rowc = txl.local_scalar(
                    "int32", init=txl.warp_id_in_role() * txl.int32(32) + txl.lane_id()
                )
                with txl.serial(n_p2) as i2:
                    work = txl.local_scalar("int32", init=p2_chain(i2))
                    seq, head, bos, seq_len, nch = work_coords(work)
                    with txl.serial(nch) as rn:
                        par = cyc & txl.int32(1)
                        b_aqk_full.wait(0, par)

                        diag = txl.alloc_local([4], "uint32")
                        dmat = txl.lane_id() >> txl.int32(3)
                        dblk = txl.warp_id_in_role() * txl.int32(4) + dmat
                        dptr = TT[S_AQK].ptr_to(
                            dblk * txl.int32(8) + (txl.lane_id() & txl.int32(7)),
                            dblk * txl.int32(8),
                        )
                        txl.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            diag[0], diag[1], diag[2], diag[3], dptr
                        )
                        drow = txl.lane_id() >> txl.int32(2)
                        dcol = (txl.lane_id() & txl.int32(3)) * txl.int32(2)
                        dmask = txl.Select(
                            dcol > drow,
                            txl.uint32(0),
                            txl.Select(
                                dcol == drow, txl.uint32(0x0000FFFF), txl.uint32(0xFFFFFFFF)
                            ),
                        )
                        for e in range(4):
                            txl.assign(diag[e], diag[e] & dmask)
                        txl.ptx["stmatrix.sync.aligned.m8n8.x4.shared.b16"](
                            dptr, diag[0], diag[1], diag[2], diag[3]
                        )
                        for u in range(1, 8):
                            with txl.If(txl.int32(8 * u) > rowc), txl.Then():
                                txl.ptx["st.shared.v4.b32"](
                                    TT[S_AQK].ptr_to(rowc, 8 * u),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                    txl.uint32(0),
                                )
                        txl.ptx[FENCE_ASYNC]()
                        b_aqk_masked.arrive(0)
                        txl.assign(cyc, cyc + txl.int32(1))

        txl.cuda.cta_sync()
        with txl.If(txl.warp_id() == 8), txl.Then():
            txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
            txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                txl.Cast("uint32", txl.local_scalar("int32", init=tmem_preamble()[0])),
                txl.uint32(512),
            )

    return kda_bwd_fused


SCHEDULE_CLASSES = [(40, 6, 2), (80, 5, 5), (32, 4, 9)]


def build_schedule(num_chains, num_ctas, classes):
    """Per-CTA (pass-2 chains, pass-1 chains) lists.

    classes == "legacy": base = num_chains // num_ctas; the `extra` lowest CTAs own base+1 pass-2 chains
    (strided) and run pass 1 for their first chain only; the remaining CTAs absorb the other heavy
    chains' pass-1 work first, then their own.  Otherwise classes is a list of (count, n_p2, n_p1):
    pass-2 chains are dealt in rounds; each CTA runs pass 1 for its own first n_p1 chains and, if it has
    spare pass-1 capacity, for other CTAs' later chains first (so those are published early).
    """
    if classes == "legacy":
        base = num_chains // num_ctas
        extra = num_chains - base * num_ctas
        p2 = [
            [c + r * num_ctas for r in range(base + (1 if c < extra else 0))]
            for c in range(num_ctas)
        ]
        n_light = num_ctas - extra
        extra1 = max(extra, 1)
        heavy_items = [(t % extra1) + (t // extra1 + 1) * num_ctas for t in range(extra * base)]
        p1 = []
        for c in range(num_ctas):
            if c < extra:
                p1.append([c])
            else:
                lidx = c - extra
                p1.append(
                    [heavy_items[t] for t in range(lidx, extra * base, n_light)]
                    + [c + r * num_ctas for r in range(base)]
                )
        return p2, p1
    cta_p2, cta_p1 = [], []
    for count, a, b in classes:
        cta_p2 += [a] * count
        cta_p1 += [b] * count
    assert len(cta_p2) == num_ctas and sum(cta_p2) == num_chains and sum(cta_p1) == num_chains, (
        len(cta_p2),
        sum(cta_p2),
        sum(cta_p1),
    )
    p2 = [[] for _ in range(num_ctas)]
    chain = 0
    for r in range(max(cta_p2)):
        for c in range(num_ctas):
            if cta_p2[c] > r:
                p2[c].append(chain)
                chain += 1
    leftover = []
    p1 = []
    for c in range(num_ctas):
        take = min(cta_p1[c], len(p2[c]))
        p1.append(p2[c][:take])
        leftover += p2[c][take:]
    li = 0
    for c in range(num_ctas):
        spare = cta_p1[c] - len(p1[c])
        if spare > 0:
            p1[c] = leftover[li : li + spare] + p1[c]
            li += spare
    assert li == len(leftover), (li, len(leftover))
    return p2, p1


def build_schedule_tensor(num_chains, num_ctas, classes, dev):
    """Per-CTA [n_p2, n_p1, p2 chains..., p1 chains...] table plus its two list capacities."""
    p2, p1 = build_schedule(num_chains, num_ctas, classes)
    assert sorted(c for l in p2 for c in l) == list(range(num_chains))
    assert sorted(c for l in p1 for c in l) == list(range(num_chains))
    maxp2 = max(1, max(len(l) for l in p2))
    maxp1 = max(1, max(len(l) for l in p1))
    stride = 2 + maxp2 + maxp1
    table = torch.full((num_ctas, stride), -1, dtype=torch.int32)
    for c in range(num_ctas):
        table[c, 0] = len(p2[c])
        table[c, 1] = len(p1[c])
        for i, ch in enumerate(p2[c]):
            table[c, 2 + i] = ch
        for i, ch in enumerate(p1[c]):
            table[c, 2 + maxp2 + i] = ch
    return table.reshape(-1).to(dev), maxp2, maxp1


_TUNED_NUM_CTAS = 152
_TUNED_NUM_CHAINS = 768


def build_mega_tables(lens, HQ, HV, num_ctas):
    """Host-side schedule tables for the mega kernel.

    stream_tab: work-unit order of the recurrence streams (backward streams first, longest
    sequences first, value heads ascending; then the forward streams in the same order), packed as
    (is_fwd << 30) | (seq << 15) | hv.
    item_tab: (seq << 16) | chunk for every chunk, sorted by the predicted step at which both of its
    streams have published it.  Streams are assumed to start when a CTA of the initial wave frees
    up (list-scheduled in stream order) and to take one step per chunk; a chunk is published by the
    forward stream after n + 1 steps and by the backward stream after nch - n steps.  A partial
    trailing chunk also waits for the first chunk of the following sequence(s) that share its tile.
    seq_tab: [N, 4] int32 rows (bos, len, nch, chunk_base).
    """
    import heapq

    N = len(lens)
    nchs = [(l + CHUNK - 1) // CHUNK for l in lens]
    cbs, boss = [], []
    cb = bo = 0
    for s in range(N):
        cbs.append(cb)
        boss.append(bo)
        cb += nchs[s]
        bo += lens[s]
    order = sorted(range(N), key=lambda s: (-lens[s], s))
    rank = {s: r for r, s in enumerate(order)}
    streams = [(0, s, hv) for s in order for hv in range(HV)] + [
        (1, s, hv) for s in order for hv in range(HV)
    ]
    free = [0] * max(1, min(num_ctas, len(streams)))
    heapq.heapify(free)
    start = {}
    for key in streams:
        t0 = heapq.heappop(free)
        start[key] = t0
        heapq.heappush(free, t0 + nchs[key[1]])
    fwd_start = [max(start[(1, s, hv)] for hv in range(HV)) for s in range(N)]
    bwd_start = [max(start[(0, s, hv)] for hv in range(HV)) for s in range(N)]
    items = []
    for s in range(N):
        for n in range(nchs[s]):
            r = max(fwd_start[s] + n + 1, bwd_start[s] + nchs[s] - n)
            if n == nchs[s] - 1 and lens[s] % CHUNK != 0:
                tok0 = boss[s] + n * CHUNK
                b2 = boss[s] + lens[s]
                s2 = s + 1
                while s2 < N and b2 < tok0 + CHUNK:
                    r = max(r, fwd_start[s2] + 1)
                    b2 += lens[s2]
                    s2 += 1
            items.append((r, rank[s], n, s))
    items.sort()
    assert N < (1 << 15) and max(nchs) < (1 << 15) and HV < (1 << 15)
    stream_tab = torch.tensor(
        [(d << 30) | (s << 15) | hv for (d, s, hv) in streams], dtype=torch.int32
    )
    item_tab = torch.tensor([(s << 16) | n for (_, _, n, s) in items], dtype=torch.int32)
    seq_tab = torch.tensor(
        [[boss[s], lens[s], nchs[s], cbs[s]] for s in range(N)], dtype=torch.int32
    ).reshape(-1)
    return stream_tab, item_tab, seq_tab


_KERNEL_CACHE = {}
_DEBUG = {}


def _target():
    """Compile target without waking CUDA: prepare arch env, then a live device, else sm_100a."""
    import tvm
    from tirx_kernels.runner import PREPARE_CUDA_ARCH_ENV, cuda_target

    if os.environ.get(PREPARE_CUDA_ARCH_ENV) is not None:
        return cuda_target()
    if torch.cuda.is_initialized():
        cap = torch.cuda.get_device_capability()
        return tvm.target.Target({"kind": "cuda", "arch": f"sm_{cap[0]}{cap[1]}a"})
    return tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})


def _range_parts(num_chains):
    return min(16, max(1, (1024 + num_chains - 1) // num_chains))


def _compile(kind, *key_args):
    key = (kind, *key_args)
    if key not in _KERNEL_CACHE:
        import tvm

        kernel = {
            "mega": make_mega_kernel,
            "retry_mega": make_retry_mega_kernel,
            "fused": make_fused_kernel,
            "guard": make_range_guard,
            "native_mega": make_native_mega_kernel,
            "native_fused": make_native_fused_kernel,
        }[kind](*key_args)
        target = _target()
        previous = os.environ.get("TVM_CUDA_PTXAS_REG_LEVEL")
        # Ordinary specializations keep the original compile setting. The
        # extended mega body has a separately measured register-usage level.
        os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = "5" if kind in ("mega", "retry_mega") else "10"
        try:
            with target:
                _KERNEL_CACHE[key] = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
        finally:
            if previous is None:
                os.environ.pop("TVM_CUDA_PTXAS_REG_LEVEL", None)
            else:
                os.environ["TVM_CUDA_PTXAS_REG_LEVEL"] = previous
    return _KERNEL_CACHE[key]


def build_kernels_for_shape(H, num_chains=768, num_ctas=None, HV=None):
    """Trace-only entry; default to the same hardware profile as ``setup``."""
    if num_ctas is None:
        from tirx_kernels.runner import hardware_num_sms

        num_ctas = min(hardware_num_sms(), num_chains)
    HV = H if HV is None else HV
    out = {
        "kda_bwd_guard": make_range_guard(HV, _range_parts(num_chains)),
        "kda_bwd_native_mega": make_native_mega_kernel(H, HV),
        "kda_bwd_mega": make_mega_kernel(H, HV),
    }
    if H == HV and H % 8 == 0:
        classes = (
            SCHEDULE_CLASSES
            if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS)
            else "legacy"
        )
        p2, p1 = build_schedule(num_chains, num_ctas, classes)
        key = (H, max(1, max(len(l) for l in p2)), max(1, max(len(l) for l in p1)))
        out["kda_bwd_native_fused"] = make_native_fused_kernel(*key)
        out["kda_bwd_fused"] = make_fused_kernel(*key)
    return out


def _count_chunks(cu_seqlens):
    """Total 64-token chunks over the packed sequences (a launch-extent fact, like FLA's chunk_indices)."""
    lens = torch.diff(cu_seqlens.cpu())
    return int(((lens + CHUNK - 1) // CHUNK).sum()), bool(((lens % CHUNK) == 0).all())


def setup(data, B, T, H):
    from tirx_kernels.runner import hardware_num_sms

    q, k, v, beta = data["q"], data["k"], data["v"], data["beta"]
    Aqk, Akk, g = data["Aqk"], data["Akk"], data["g"]
    h0, do, dht = data["initial_state"], data["do"], data["dht"]
    scale = float(data["scale"])
    chunk_size = int(data["chunk_size"])
    cu_seqlens = data["cu_seqlens"]
    dq, dk, dv, db, dg, dh0 = (data[n] for n in ("dq", "dk", "dv", "db", "dg", "dh0"))
    device = q.device
    HQ = H
    HV = v.shape[2]
    N = h0.shape[0]
    if B != 1 or chunk_size != CHUNK or q.shape[-1] != D or v.shape[-1] != D or HV % HQ != 0:
        raise ValueError("unsupported KDA backward shape")
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=device)
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int64).contiguous()
    for name in (
        "q",
        "k",
        "v",
        "beta",
        "Aqk",
        "Akk",
        "g",
        "initial_state",
        "do",
        "dht",
        "dq",
        "dk",
        "dv",
        "db",
        "dg",
        "dh0",
    ):
        if not data[name].is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    total_chunks, full_chunks = _count_chunks(cu_seqlens)
    # Same oracle as ``prepare_bench``: the prepare-stage override wins, then the
    # live device, so the cache key primed before READY is the one used here.
    num_sms = hardware_num_sms()
    num_chunks_max = (T + CHUNK - 1) // CHUNK + N
    hsnap = torch.empty((num_chunks_max, HV, D, D), dtype=torch.bfloat16, device=device)
    egcache = torch.empty((1, T, HV, D), dtype=torch.bfloat16, device=device)
    stream_counter = torch.zeros((4,), dtype=torch.int32, device=device)
    maps = {
        "q": token_map(q, T, HQ, D, 64),
        "k": token_map(k, T, HQ, D, 64),
        "v": token_map(v, T, HV, D, 64),
        "g": token_map(g, T, HV, D, 32),
        "do": token_map(do, T, HV, D, 64),
        "eg": token_map(egcache, T, HV, D, 64),
        "aqk": token_map(Aqk, T, HV, CHUNK, CHUNK),
        "akk": token_map(Akk, T, HV, CHUNK, CHUNK),
        "h": state_map(hsnap, num_chunks_max * HV),
    }
    _DEBUG.clear()
    range_parts = _range_parts(N * HV)
    range_entries = N * HV * range_parts
    # Mega's tail stores a retry count and at most one global item index
    # per native item.  Fused dispatch does not allocate this unused workspace;
    # the guard writes only the leading range_entries words in either family.
    mega_dispatch = not (HQ == HV and HV % 8 == 0 and full_chunks)
    retry_capacity = total_chunks * HQ if mega_dispatch else 0
    retry_words = 1 + retry_capacity if mega_dispatch else 0
    range_flags = torch.empty((range_entries + retry_words,), dtype=torch.int32, device=device)
    if retry_words:
        range_flags[range_entries:].zero_()
    range_guard = _compile("guard", HV, range_parts)
    range_allowed = int(int(torch.diff(cu_seqlens.cpu()).max()) <= 4096)

    def check_range():
        range_guard(
            v.view(-1),
            do.view(-1),
            h0.view(-1),
            dht.view(-1),
            g.view(-1),
            cu_seqlens,
            range_flags,
            range_entries,
        )

    if HQ == HV and HV % 8 == 0 and full_chunks:
        num_chains = N * HV
        num_ctas = min(num_sms, num_chains)
        classes = (
            SCHEDULE_CLASSES
            if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS)
            else "legacy"
        )
        sched, maxp2, maxp1 = build_schedule_tensor(num_chains, num_ctas, classes, device)
        flags = torch.zeros((num_chains,), dtype=torch.int32, device=device)
        maps["beta"] = token_map(beta, T, HV // 8, 8, 8, swizzle=0)
        fused = _compile("fused", HV, maxp2, maxp1)
        native_fused = _compile("native_fused", HV, maxp2, maxp1)
        args = (
            q.view(-1),
            k.view(-1),
            v.view(-1),
            beta.view(-1),
            Aqk.view(-1),
            Akk.view(-1),
            g.view(-1),
            egcache.view(-1),
            do.view(-1),
            dht.view(-1),
            h0.view(-1),
            hsnap.view(-1),
            cu_seqlens,
            flags,
            sched,
            dq.view(-1),
            dk.view(-1),
            dv.view(-1),
            db.view(-1),
            dg.view(-1),
            dh0.view(-1),
            maps["q"].ptr,
            maps["k"].ptr,
            maps["v"].ptr,
            maps["g"].ptr,
            maps["eg"].ptr,
            maps["beta"].ptr,
            maps["do"].ptr,
            maps["aqk"].ptr,
            maps["akk"].ptr,
            maps["h"].ptr,
            scale,
            N,
            num_ctas,
        )
        state = {"epoch": 0}

        def run():
            state["epoch"] += 1
            # CUDA Graph replay reuses the epoch value captured from this
            # Python closure.  Capture a ready-flag clear as the first graph
            # node so waits cannot accept snapshots from the prior replay.
            if torch.cuda.is_current_stream_capturing():
                flags.zero_()
            check_range()
            native_fused(*args, state["epoch"], range_flags, range_entries)
            fused(*args, state["epoch"], range_flags, range_allowed, range_entries)

        run._keep_alive = (args, maps, hsnap, egcache, flags, sched, cu_seqlens)
        _DEBUG.update(family="fused", h=hsnap)
    else:
        dhsnap = torch.empty((num_chunks_max, HV, D, D), dtype=torch.bfloat16, device=device)
        maps["dh"] = state_map(dhsnap, num_chunks_max * HV)
        num_chains = N * HV
        flags = torch.zeros((2 * num_chains,), dtype=torch.int64, device=device)
        num_items = total_chunks * HQ
        num_ctas = min(num_sms, 2 * num_chains + num_items)
        lens = torch.diff(cu_seqlens.cpu()).tolist()
        stream_tab, item_tab, seq_tab = build_mega_tables(lens, HQ, HV, num_ctas)
        assert item_tab.numel() == total_chunks
        stream_tab = stream_tab.to(device)
        item_tab = item_tab.to(device)
        seq_tab = seq_tab.to(device)
        mega = _compile("mega", HQ, HV)
        retry_mega = _compile("retry_mega", HQ, HV)
        native_mega = _compile("native_mega", HQ, HV)
        args = (
            q.view(-1),
            k.view(-1),
            v.view(-1),
            beta.view(-1),
            Aqk.view(-1),
            Akk.view(-1),
            g.view(-1),
            egcache.view(-1),
            do.view(-1),
            dht.view(-1),
            h0.view(-1),
            hsnap.view(-1),
            dhsnap.view(-1),
            cu_seqlens,
            dq.view(-1),
            dk.view(-1),
            dv.view(-1),
            db.view(-1),
            dg.view(-1),
            dh0.view(-1),
            stream_counter,
            flags,
            stream_tab,
            item_tab,
            seq_tab,
            maps["q"].ptr,
            maps["k"].ptr,
            maps["v"].ptr,
            maps["g"].ptr,
            maps["eg"].ptr,
            maps["do"].ptr,
            maps["aqk"].ptr,
            maps["akk"].ptr,
            maps["h"].ptr,
            maps["dh"].ptr,
            scale,
            N,
            num_items,
            num_ctas,
        )
        state = {"epoch": 0}

        def run():
            state["epoch"] += 1
            # CUDA Graph replay reuses the epoch value captured from this
            # Python closure.  Capture a ready-flag clear as the first graph
            # node so waits cannot accept snapshots from the prior replay.
            if torch.cuda.is_current_stream_capturing():
                flags.zero_()
            check_range()
            native_mega(*args, state["epoch"], range_flags, range_entries)
            # Bit 4 is published only after native recurrence snapshots and item
            # outputs are complete.  Reuse those epoch-stamped snapshots and
            # overwrite only dq/dk/dg with the exact diagonal formulation.
            retry_mega(*args, state["epoch"], range_flags, range_allowed, range_entries)
            mega(*args, state["epoch"], range_flags, range_allowed, range_entries)

        run._keep_alive = (
            args,
            maps,
            hsnap,
            dhsnap,
            egcache,
            cu_seqlens,
            stream_counter,
            flags,
            stream_tab,
            item_tab,
            seq_tab,
        )
        _DEBUG.update(family="mega", h=hsnap, dhb=dhsnap)

    _DEBUG["range_flags"] = range_flags[:range_entries]
    _DEBUG["retry_queue"] = range_flags[range_entries:]
    _DEBUG["guard_only"] = check_range
    run()
    torch.cuda.synchronize(device)
    return run


def debug_state():
    """Kernel-owned intermediates for the remote test harness."""
    return {k: v for k, v in _DEBUG.items() if k in ("h", "dhb")}


# ----------------------------------------------------------------------------
# Public kernel identity and supported configurations
# ----------------------------------------------------------------------------

KERNEL_META = {
    "name": "kda_backward_packed",
    "category": "kda",
    "runtime_cuda_archs": ["sm_100a"],
    "reference_requirements": (
        {
            "package": "flash-linear-attention",
            "git": {
                "url": "https://github.com/fla-org/flash-linear-attention.git",
                "commit": "9c8e42e762fce087c27b673af4922795d9edb85e",
            },
            "import": "fla",
        },
    ),
    "provenance": {
        "generator": "hmz",
        "run": "kda-bwd-portfolio-carry-3",
        "selected_version": "fused-zfold",
    },
}

# The seventeen official workloads: (label, total tokens, Hqk, Hv, sequence lengths).
_OFFICIAL_WORKLOADS = (
    ("packed_1024x8_h96", 8192, 96, 96, (1024,) * 8),
    ("p01_hq4_hv8_t32768", 32768, 4, 8, (3200,) * 9 + (3968,)),
    ("p02_hq2_hv8_t18432", 18432, 2, 8, (2000,) * 8 + (2432,)),
    ("p03_hq4_hv4_t32768", 32768, 4, 4, (2656,) * 11 + (3552,)),
    ("p04_hq2_hv4_t18432", 18432, 2, 4, (1648,) * 10 + (1952,)),
    ("p05_hq4_hv8_t18432", 18432, 4, 8, (2000,) * 8 + (2432,)),
    ("p06_hq2_hv8_t32768", 32768, 2, 8, (3200,) * 9 + (3968,)),
    ("p07_hq2_hv4_t18432", 18432, 2, 4, (2000,) * 8 + (2432,)),
    ("p08_hq4_hv4_t18432", 18432, 4, 4, (1648,) * 10 + (1952,)),
    ("p09_hq2_hv8_t32768", 32768, 2, 8, (2656,) * 11 + (3552,)),
    ("p10_hq4_hv8_t18432", 18432, 4, 8, (1648,) * 10 + (1952,)),
    ("p11_hq4_hv4_t32768", 32768, 4, 4, (3200,) * 9 + (3968,)),
    ("p12_hq2_hv4_t32768", 32768, 2, 4, (2656,) * 11 + (3552,)),
    ("p13_hq4_hv4_t18432", 18432, 4, 4, (2000,) * 8 + (2432,)),
    ("p14_hq2_hv4_t32768", 32768, 2, 4, (3200,) * 9 + (3968,)),
    ("p15_hq4_hv8_t32768", 32768, 4, 8, (2656,) * 11 + (3552,)),
    ("p16_hq2_hv8_t18432", 18432, 2, 8, (1648,) * 10 + (1952,)),
)

_SUPPORTED = {label: (total, hq, hv, lens) for label, total, hq, hv, lens in _OFFICIAL_WORKLOADS}


@dataclass(frozen=True, slots=True)
class KDABackwardConfig:
    label: str
    num_qk_heads: int
    num_v_heads: int
    seq_lens: tuple[int, ...]
    seed: int = 0
    scale: float = 1.0 / math.sqrt(D)
    gate_profile: str = "mild"

    def validate(self) -> None:
        if self.gate_profile not in (
            "mild",
            "strong",
            "mixed",
            "reset",
            "extreme",
            "init",
            "init_large_grad",
            "aligned",
            "aligned_shift",
        ):
            raise ValueError(f"unsupported gate profile {self.gate_profile!r}")
        if self.num_v_heads % self.num_qk_heads != 0:
            raise ValueError(
                f"Hv must be a multiple of Hqk, got {self.num_v_heads} % {self.num_qk_heads}"
            )
        if not self.seq_lens or any(length <= 0 for length in self.seq_lens):
            raise ValueError(f"invalid packed sequence layout {self.seq_lens}")
        if not math.isclose(self.scale, 1.0 / math.sqrt(D), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"scale must be 1/sqrt({D}), got {self.scale}")

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens)

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)

    @property
    def uses_fused_path(self) -> bool:
        """Mirror ``setup``'s dispatch rule without touching CUDA."""
        whole_chunks = all(length % CHUNK == 0 for length in self.seq_lens)
        return self.num_qk_heads == self.num_v_heads and self.num_v_heads % 8 == 0 and whole_chunks


CONFIGS = [
    {
        "label": label,
        "num_qk_heads": hq,
        "num_v_heads": hv,
        "seq_lens": lens,
        "seed": 2858210400 + index,
    }
    for index, (label, _total, hq, hv, lens) in enumerate(_OFFICIAL_WORKLOADS)
]

# Keep the established performance portfolio while exercising the numerical
# range in correctness, including GVA and masked partial chunks.
BENCH_CONFIGS = list(CONFIGS)
CONFIGS += [
    {
        "label": f"{profile}_{family}",
        "num_qk_heads": 8 if family == "fused" else 2,
        "num_v_heads": 8 if family == "fused" else 4,
        "seq_lens": (128, 128) if family == "fused" else (129, 79),
        "seed": 23,
        "gate_profile": profile,
    }
    for family in ("fused", "mega")
    for profile in (
        "strong",
        "mixed",
        "reset",
        "extreme",
        "init",
        "init_large_grad",
        "aligned",
        "aligned_shift",
    )
]
CONFIGS += [
    {
        "label": f"{profile}_gva4_tails",
        "num_qk_heads": 2,
        "num_v_heads": 8,
        "seq_lens": (1, 63, 64, 65, 127, 128, 129),
        "seed": 31,
        "gate_profile": profile,
    }
    for profile in ("mixed", "strong")
]
CONFIGS += [
    {**config, "label": f"strong_{config['label']}", "gate_profile": "strong"}
    for config in BENCH_CONFIGS
    if config["label"] in ("packed_1024x8_h96", "p04_hq2_hv4_t18432")
]


# Exercise both sides of the sequence-length range guard without expanding
# the established performance portfolio.
CONFIGS += [
    {
        "label": "init_fused_t4096",
        "num_qk_heads": 8,
        "num_v_heads": 8,
        "seq_lens": (4096,),
        "seed": 23,
        "gate_profile": "init",
    },
    {
        "label": "init_mega_t4097",
        "num_qk_heads": 2,
        "num_v_heads": 4,
        "seq_lens": (4097,),
        "seed": 23,
        "gate_profile": "init",
    },
]


def _cfg(**kwargs: Any) -> KDABackwardConfig:
    names = {field.name for field in fields(KDABackwardConfig)}
    values = {name: value for name, value in kwargs.items() if name in names}
    if "seq_lens" in values:
        values["seq_lens"] = tuple(int(length) for length in values["seq_lens"])
    values.setdefault("label", "custom")
    cfg = KDABackwardConfig(**values)
    cfg.validate()
    return cfg


def _launch_cache_key(cfg: KDABackwardConfig) -> tuple:
    """The ``_compile`` key ``setup`` derives for this configuration, from config alone."""
    if cfg.uses_fused_path:
        from tirx_kernels.runner import hardware_num_sms

        num_chains = cfg.num_seqs * cfg.num_v_heads
        num_ctas = min(hardware_num_sms(), num_chains)
        classes = (
            SCHEDULE_CLASSES
            if (num_ctas == _TUNED_NUM_CTAS and num_chains == _TUNED_NUM_CHAINS)
            else "legacy"
        )
        p2, p1 = build_schedule(num_chains, num_ctas, classes)
        maxp2 = max(1, max(len(entries) for entries in p2))
        maxp1 = max(1, max(len(entries) for entries in p1))
        return ("fused", cfg.num_v_heads, maxp2, maxp1)
    return ("mega", cfg.num_qk_heads, cfg.num_v_heads)


def _launch_cache_keys(cfg: KDABackwardConfig) -> tuple[tuple, ...]:
    main = _launch_cache_key(cfg)
    return (
        ("guard", cfg.num_v_heads, _range_parts(cfg.num_seqs * cfg.num_v_heads)),
        ("native_" + main[0], *main[1:]),
        main,
    )


def get_kernel(**kwargs: Any):
    """Return every PrimFunc in the GPU-guarded launch sequence."""
    builders = {
        "guard": make_range_guard,
        "native_mega": make_native_mega_kernel,
        "native_fused": make_native_fused_kernel,
        "mega": make_mega_kernel,
        "fused": make_fused_kernel,
    }
    return [builders[kind](*args).func for kind, *args in _launch_cache_keys(_cfg(**kwargs))]


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------


def _randn(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    device: torch.device,
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32, device=device, generator=generator) * scale).to(
        dtype
    )


def _l2_normalize_bf16(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float()
    return (values * torch.rsqrt(values.square().sum(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)


@contextmanager
def _native_fla_backend():
    values = {"FLA_FLASH_KDA": "0", "FLA_TILELANG": "0"}
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _saved_interaction_matrices(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Aqk/Akk exactly as FLA's varlen intra path saves them for the backward."""
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra

        outputs = chunk_kda_fwd_intra(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            gk=case["g"],
            beta=case["beta"],
            scale=case["scale"],
            cu_seqlens=case["cu_seqlens"],
            chunk_size=CHUNK,
        )
    return outputs[4], outputs[5]


def prepare_data(**kwargs: Any) -> dict[str, Any]:
    cfg = _cfg(**kwargs)
    device = torch.device(kwargs.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SkipTest("CUDA is required for native TIRx KDA backward")

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)
    total, hq, hv = cfg.total_tokens, cfg.num_qk_heads, cfg.num_v_heads
    qk_shape = (1, total, hq, D)
    v_shape = (1, total, hv, D)
    state_shape = (cfg.num_seqs, hv, D, D)

    q = _l2_normalize_bf16(
        _randn(qk_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    k = _l2_normalize_bf16(
        _randn(qk_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    )
    v = _randn(v_shape, torch.bfloat16, device=device, generator=generator, scale=0.5)
    beta = torch.sigmoid(
        _randn((1, total, hv), torch.float32, device=device, generator=generator, scale=0.5)
    ).to(torch.bfloat16)
    gate_increments = -(
        0.01 + 0.04 * torch.rand(v_shape, dtype=torch.float32, device=device, generator=generator)
    )
    offsets = [0]
    for length in cfg.seq_lens:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int64, device=device)

    if cfg.gate_profile in ("init", "init_large_grad"):
        # Synthetic FLA initialization with unit-variance projection noise;
        # this is regression input, not a measured training distribution.
        init_rng = torch.Generator(device=device).manual_seed(701)
        decay_rate = 1 + 15 * torch.rand((1, 1, hv, 1), device=device, generator=init_rng)
        dt = torch.exp(
            torch.rand((1, 1, hv, D), device=device, generator=init_rng) * math.log(100)
            + math.log(0.001)
        )
        bias = dt + torch.log(-torch.expm1(-dt))
        projection = torch.randn(v_shape, device=device, generator=init_rng)
        gate_increments = (
            -decay_rate * torch.nn.functional.softplus(projection + bias) * math.log2(math.e)
        )
    elif cfg.gate_profile in ("aligned", "aligned_shift"):
        gate_increments.fill_(-2.0 if cfg.gate_profile == "aligned_shift" else -0.75)
    elif cfg.gate_profile == "strong":
        gate_increments.mul_(256.0)
    elif cfg.gate_profile == "mixed":
        # Mix the fast path, its dispatch boundary, both floating-point cliffs,
        # and gates far beyond either cliff within each head and warp.
        levels = torch.tensor(
            [0.0, 0.03, 0.0625, 0.06253, 0.06255, 0.49, 0.51, 1.5, 1.9375, 1.96875, 2.0, 3.6, 8.0],
            device=device,
        )
        channels = torch.arange(D, device=device) % levels.numel()
        gate_increments.copy_(-levels[channels].view(1, 1, 1, D).expand(v_shape))
    elif cfg.gate_profile == "reset":
        # A large first-token decay must not erase interactions among the
        # following tokens: their pairwise exponent differences are still zero.
        gate_increments.zero_()
        for bos, eos in pairwise(offsets):
            gate_increments[:, bos:eos:CHUNK].fill_(-512.0)
    elif cfg.gate_profile == "extreme":
        gate_increments.fill_(-512.0)

    with _native_fla_backend():
        from fla.ops.utils import chunk_local_cumsum

        g = chunk_local_cumsum(gate_increments, chunk_size=CHUNK, cu_seqlens=cu_seqlens)

    case: dict[str, Any] = {
        "config": cfg,
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "g": g,
        "scale": cfg.scale,
        "chunk_size": CHUNK,
        "cu_seqlens": cu_seqlens,
        "initial_state": _randn(
            state_shape, torch.float32, device=device, generator=generator, scale=0.01
        ),
        "do": _randn(v_shape, torch.bfloat16, device=device, generator=generator, scale=0.25),
        "dht": _randn(state_shape, torch.float32, device=device, generator=generator, scale=0.01),
    }
    if cfg.gate_profile == "init_large_grad":
        case["do"].mul_(16)
        case["dht"].mul_(16)
    if cfg.gate_profile in ("aligned", "aligned_shift"):
        case["q"].fill_(1 / math.sqrt(D))
        case["k"].fill_(1 / math.sqrt(D))
        case["beta"].fill_(0.75 if cfg.gate_profile == "aligned_shift" else 0.5)
        case["v"].fill_(4)
        case["do"].fill_(4)
        case["initial_state"].fill_(1)
        case["dht"].fill_(1)
    case["Aqk"], case["Akk"] = _saved_interaction_matrices(case)
    case["dq"] = torch.empty(qk_shape, dtype=torch.float32, device=device)
    case["dk"] = torch.empty(qk_shape, dtype=torch.float32, device=device)
    case["dv"] = torch.empty_like(v)
    case["db"] = torch.empty((1, total, hv), dtype=torch.float32, device=device)
    case["dg"] = torch.empty(v_shape, dtype=torch.float32, device=device)
    case["dh0"] = torch.empty(state_shape, dtype=torch.float32, device=device)
    return case


def _launcher(case: dict[str, Any]):
    """Bind the kernel-owned scratch, schedule tables and tensor maps."""
    cfg: KDABackwardConfig = case["config"]
    return setup(case, cfg.batch_size, cfg.total_tokens, cfg.num_qk_heads)


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------

_OUTPUT_NAMES = ("dq", "dk", "dv", "db", "dg", "dh0")
# FLA's test_kda.py normalized RMS error-ratio limits, in return order.
_RMS_LIMITS = (8e-3, 8e-3, 8e-3, 2e-2, 2e-2, 8e-3)


def _assert_supported_arch() -> None:
    if not torch.cuda.is_available():
        raise SkipTest("CUDA is required for native TIRx KDA backward")
    capability = torch.cuda.get_device_capability()
    runtime_arch = f"sm_{capability[0]}{capability[1]}a"
    if runtime_arch not in KERNEL_META["runtime_cuda_archs"]:
        raise SkipTest(
            "native TIRx KDA backward requires one of "
            f"{KERNEL_META['runtime_cuda_archs']}, got {runtime_arch}"
        )


def _run_fla_reference(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    with _native_fla_backend(), torch.no_grad():
        from fla.ops.kda.chunk_bwd import chunk_kda_bwd

        dq, dk, dv, db, dg, dh0, dA, dbias = chunk_kda_bwd(
            q=case["q"],
            k=case["k"],
            v=case["v"],
            beta=case["beta"],
            Aqk=case["Aqk"],
            Akk=case["Akk"],
            g=case["g"],
            initial_state=case["initial_state"],
            do=case["do"],
            dht=case["dht"],
            scale=case["scale"],
            chunk_size=CHUNK,
            cu_seqlens=case["cu_seqlens"],
        )
    if dA is not None or dbias is not None:
        raise AssertionError("prepared-gate chunk_kda_bwd unexpectedly returned dA/dbias")
    return dq, dk, dv, db, dg, dh0


def check_correctness(outputs: dict[str, Any], **kwargs: Any) -> None:
    _cfg(**kwargs)
    first, actual, reference = outputs["first"], outputs["actual"], outputs["reference"]
    for name, tensor in zip(_OUTPUT_NAMES, actual):
        if not torch.isfinite(tensor).all():
            raise AssertionError(f"{name} contains non-finite values")
    for name, one, two in zip(_OUTPUT_NAMES, first, actual):
        if not torch.equal(one, two):
            max_abs = float((one.float() - two.float()).abs().max())
            raise AssertionError(
                f"identical launches are not exactly repeatable for {name}; max abs diff={max_abs}"
            )
    for name, got, want, limit in zip(_OUTPUT_NAMES, actual, reference, _RMS_LIMITS):
        torch.testing.assert_close(
            got, want, atol=1e-1, rtol=1e-1, msg=lambda message, n=name: f"{n}: {message}"
        )
        # Finite FP32 gradients can overflow when squared or reduced in FP32.
        got64, want64 = got.double(), want.double()
        diff_rms = torch.sqrt(torch.mean((got64 - want64).square()))
        reference_rms = torch.sqrt(torch.mean(want64.square()))
        rms_ratio = float(diff_rms / (reference_rms + 1e-8))
        if not math.isfinite(rms_ratio) or rms_ratio >= limit:
            raise AssertionError(
                f"{name} normalized RMS error ratio {rms_ratio:.6e} must be below {limit:.0e}"
            )


def _clone_outputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(case[name].clone() for name in _OUTPUT_NAMES)


def _poison_outputs(case: dict[str, Any], value: float) -> None:
    for name in _OUTPUT_NAMES:
        case[name].fill_(value)


def run_test(**kwargs: Any) -> None:
    _assert_supported_arch()
    case = prepare_data(**kwargs)
    launch = _launcher(case)
    _poison_outputs(case, float("nan"))
    launch()
    torch.cuda.synchronize()
    first = _clone_outputs(case)
    _poison_outputs(case, 42.0)
    launch()
    torch.cuda.synchronize()
    actual = _clone_outputs(case)
    reference = _run_fla_reference(case)
    torch.cuda.synchronize()
    check_correctness({"first": first, "actual": actual, "reference": reference}, **kwargs)


# ----------------------------------------------------------------------------
# Benchmarking
# ----------------------------------------------------------------------------


def prepare_bench(**kwargs: Any):
    """Trace and compile before bench-suite assigns a GPU.

    ``setup`` compiles through this module's own ``_KERNEL_CACHE``, so priming
    that cache under the same key here is what keeps ``run_gpu`` free of
    ``tvm.compile``.
    """
    from tirx_kernels.runner import prepared_gpu_benchmark

    for key in _launch_cache_keys(_cfg(**kwargs)):
        _compile(*key)
    return prepared_gpu_benchmark(run_gpu, {"config": dict(kwargs)})


def run_gpu(
    prepared,
    *,
    warmup: int | None = None,
    repeat: int | None = None,
    timer: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    _assert_supported_arch()
    config = dict(prepared["config"])
    config.update(kwargs)
    rounds = config.pop("rounds", 5)
    cooldown_s = config.pop("cooldown_s", 1.0)
    case = prepare_data(**config)
    launch = _launcher(case)
    torch.cuda.synchronize()

    def _fla_builder():
        return lambda: _run_fla_reference(case)

    from tirx_kernels.runner import bench

    return bench(
        {"tirx": launch},
        references={"fla_chunk_kda_bwd": _fla_builder},
        warmup=warmup,
        repeat=repeat,
        timer=timer,
        rounds=rounds,
        cooldown_s=cooldown_s,
    )


def run_bench(
    *, warmup: int | None = None, repeat: int | None = None, timer: str | None = None, **kwargs: Any
) -> dict[str, Any]:
    config = dict(kwargs)
    protocol = {name: config.pop(name) for name in ("rounds", "cooldown_s") if name in config}
    prepared = prepare_bench(**config)
    return prepared.run_gpu(warmup=warmup, repeat=repeat, timer=timer, **protocol)


__all__ = [
    "BENCH_CONFIGS",
    "CONFIGS",
    "KERNEL_META",
    "check_correctness",
    "get_kernel",
    "prepare_bench",
    "prepare_data",
    "run_bench",
    "run_test",
]
