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

"""KDA forward, packed varlen: persistent warp-specialized CTAs over (sequence, head) items, G-form chain.

Family: fused item-parallel, G-form chain over packed sequences, distributed over min(#SMs, H * num_seqs)
CTAs (equal-length items go to the least loaded CTAs, so packed-varlen shapes use all SMs instead of one
CTA per head).

Chunk size 64.  The per-channel log2 decay is factored around two integer half
references (r0 = rint(gamma_31 / 2) for tokens 0-31, eps = min(0, rint(e + F + 0.5))
for tokens 32-63, with the chunk total e clamped through e_cl = e - eps for the
state scale) so every operand exponent stays within +-~122 bits.  Per chunk the
recurrent chain is
   S_bf -> G = S k2^T -> (bf16) -> v_new = u - G T'^T -> (bf16) -> S += v_new kA -> decay -> S_bf

Varlen: a one-warp prologue expands cu_seqlens into an SMEM item table (token offset,
valid rows, sequence id, first/last flags); every role loops over the same flat item
index so all barrier parities stay item-based.  Rows past a sequence end are masked in
the prep role (k, q -> 0, log-decay -> 0, beta -> 0), so a partial tail chunk leaves the
state untouched for those rows.  The state role loads initial_state[seq] into TMEM at
each sequence's first chunk and writes final_state[seq] after its last chunk.  Output
rows are written with predicated 16-byte global stores from the transposed SMEM staging
tile (no fixed-box TMA store can be clipped at an interior sequence boundary).

Shape portfolio: H96 mixed-varlen forces whole-item LPT scheduling; H64 mixed and
uniform retain chunk-linear splitting with a bf16 continuation buffer.  Initial-state
loads and final-state stores use 256-bit streaming fp32 operations, while final_state
itself remains fully materialized in fp32 and H96 uniform retains fp32 handoffs.
"""

import ctypes
import heapq
import math
import os
import types

import torch
import tvm
import tirx_kernels.tirx_lite as txl
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32 as _idesc

BT, D = 64, 128
RCP_LN2 = 1.0 / math.log(2.0)
GATE_C = -2.5 * RCP_LN2
EPS = 1e-6
MAX_ITEMS = 160


# split schedule: per-sequence-start cost (BETA) and cut snap distance (SNAP), in chunks
BETA = 1
SNAP = 1
# per-unit start cost of the LPT schedule in chunk-equivalents (state load + pipeline fill)
START_COST = 0.9
# cost of one fp32 state handoff between adjacent CTAs, in chunk-equivalents
HANDOFF_COST = 0.4


MMA = "tcgen05.mma.cta_group::1.kind::f16"
MMA_WS = "tcgen05.mma.ws.cta_group::1.kind::f16"
TMA3 = "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
TMA2 = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint"
CACHE_EVICT_FIRST = txl.uint64(0x12F0000000000000)
COMMIT = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
LD32x64 = "tcgen05.ld.sync.aligned.32x32b.x64.b32"
ST32x64 = "tcgen05.st.sync.aligned.32x32b.x64.b32"
ST32x32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
ST32x16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
LD16x8 = "tcgen05.ld.sync.aligned.16x256b.x8.b32"
LD16x2 = "tcgen05.ld.sync.aligned.16x256b.x2.b32"
LD16x4 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"


CLAMP_F = 120.0
LD32x32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
LD32x16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TMA_S2G3 = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.L2::cache_hint"
TMEM_ALLOC = "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"
TMEM_DEALLOC = "tcgen05.dealloc.cta_group::1.sync.aligned.b32"
TMEM_RELINQ = "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"
FENCE_AFTER = "tcgen05.fence::after_thread_sync"
FENCE_BEFORE = "tcgen05.fence::before_thread_sync"


C_SACC, C_SBF, C_VNBF, C_OACC, C_GACC, C_AA0, C_GBF = 0, 128, 192, 224, 288, 352, 480
N_COLS = 512

F32, BF = "float32", "bfloat16"
ID_AQK64 = _idesc(64, 64, 16, F32, BF, BF, False, False)
ID_U = _idesc(128, 64, 16, F32, BF, BF, True, True)
ID_VN = _idesc(128, 64, 16, F32, BF, BF, False, True, neg_a=True)
ID_G64 = _idesc(128, 64, 16, F32, BF, BF, False, False)
ID_DL = _idesc(128, 128, 16, F32, BF, BF, False, True)
ID_OX = _idesc(128, 64, 16, F32, BF, BF, False, True)

W_STATE = list(range(0, 4))
W_LD, W_MMI, W_MMC, W_RND = 4, 5, 6, 7
W_INTRA = list(range(8, 12))
W_PREP = list(range(12, 20))
NWARPS = 20

NB_PREP, NB_INTRA, NB_STATE = 1, 2, 3


def build_kernel(
    H,
    state_regs=112,
    wg_regs=40,
    intra_regs=104,
    prep_regs=112,
    intra_unroll=False,
    bf16_handoff=False,
    max_items=MAX_ITEMS,
):
    """Persistent CTAs, each walking its host-built item list (see _host_item_table).

    Item lists up to MAX_ITEMS long are staged in SMEM; longer ones (long sequences on few
    (sequence, head) units) are read from global memory in place."""
    assert H % 8 == 0
    global_items = max_items > MAX_ITEMS

    def kda_fwd(q_map, k_map, v_map, g_map, beta_map, o_map, out, A_log, dt_bias, h0, final_state, hand, flags, items, item_counts, num_ctas, scale):
        cta = txl.cta_id()
        warp = txl.warp_id()
        lane = txl.lane_id()
        tid = txl.thread_id()

        smem = txl.smem_pool()
        tmem_addr = smem.alloc((1,), txl.u32)
        item_tbl = smem.alloc((2 * min(max_items, MAX_ITEMS),), txl.u32, align=16)
        item_meta = smem.alloc((4,), txl.u32, align=16)
        adt_s = smem.alloc((128 + 4,), txl.f32, align=16)
        rsq = smem.alloc((2 * 4 * 64,), txl.f32, align=16)
        beta_s = smem.alloc((2 * 64 * 8,), txl.bf16, align=128)
        bsig = smem.alloc((3 * 64,), txl.f32, align=16)
        dvec = smem.alloc((3 * 128,), txl.f32, align=16)

        tot = smem.alloc((4 * 128,), txl.f32, align=16)

        def mbar(count, depth=1):
            b = txl.MBarrier(smem, depth)
            b.init(count)
            return b

        ring_full = mbar(1, 2)
        v_full = mbar(1)
        rfull = mbar(8, 2)

        aa_r = mbar(1, 2)
        akk_r = mbar(1, 2)
        aa_free = mbar(4, 2)
        T_ready = mbar(4, 2)
        dvec_ready = mbar(2, 3)
        dvec_free = mbar(4, 3)
        g_done = mbar(1)
        g_ready = mbar(4)
        u_done = mbar(1, 2)
        Aqk_ready = mbar(4)
        S_ready = mbar(4, 1)
        vnew_done = mbar(1, 2)
        kA_free = mbar(1, 2)
        oi_done = mbar(1)
        o_done = mbar(1)
        o_free = mbar(4)
        aqk_stage_free = mbar(1)

        pool = smem.pool
        TOFF = {}

        def tile_alloc(name, shape):
            view = smem.alloc(shape, txl.bf16, swizzle=txl.SW128B)
            nbytes = 2
            for d_ in shape:
                nbytes *= d_
            TOFF[name] = pool.offset - nbytes
            return view

        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        # interleaved kq ring tile, rows [k 0-31 | q 0-31 | k 32-63 | q 32-63]: token t of x (k 0, q 1) is row (t // 32) * 64 + 32 * x + t % 32, so a rounds half reads its k- and q-targets as one N=64 operand
        kq_t = tile_alloc("kq", (2, 2 * BT, D))
        KQ_Q_OFF = 32 * 128  # byte offset of a q row from the k row of the same token (32 rows of one 128-B slab)
        g_t = tile_alloc("g", (2, BT, D))
        STAGE_UNITS = BT * D * 2 // 16
        v_t = tile_alloc("v", (BT, D))
        pool.move_base_to((pool.offset + 1023) // 1024 * 1024)
        kA = tile_alloc("kA", (2, BT, D))
        q2t = tile_alloc("q2", (BT, D))
        k2t = tile_alloc("k2", (BT, D))
        Aqk_s = tile_alloc("Aqk", (64, 64))
        LTT = tile_alloc("LTT", (64, 64))
        TpT_t = tile_alloc("TpT", (64, 64))

        TpTlo_t = tile_alloc("TpTlo", (64, 64))
        o_st = tile_alloc("ost", (2, 32, 64))
        base0 = TOFF["kq"]
        for name in TOFF:
            TOFF[name] -= base0
        assert min(TOFF.values()) == 0

        # this CTA's host-built item table (see _host_item_table) -> SMEM by one warp; read only after the CTA barrier below
        with txl.If(warp == 0), txl.Then():
            cnt = txl.local_scalar("int32")
            txl.ptx.ld.global_.s32(cnt, item_counts.ptr_to([cta]))
            if not global_items:
                with txl.serial(2 * max_items // 32) as it_:
                    wi = it_ * 32 + lane
                    with txl.If(wi < 2 * cnt), txl.Then():
                        wv = txl.local_scalar("uint32")
                        txl.ptx.ld.global_.u32(wv, items.ptr_to([cta * (2 * max_items) + wi]))
                        txl.ptx.st.shared.u32(item_tbl.ptr_to([wi]), wv)
            with txl.If(lane == 0), txl.Then():
                txl.ptx.st.shared.u32(item_meta.ptr_to([0]), txl.Cast("uint32", cnt))
            # the item table aliases nothing, but the ring tiles are filled by TMA (async proxy) right after
            txl.ptx.fence.proxy.async_.shared__cta()

        txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        with txl.If(warp == 0), txl.Then():
            txl.ptx[TMEM_ALLOC](txl.address_of(tmem_addr[0]), txl.uint32(N_COLS))
            txl.cuda.warp_sync()
        txl.cuda.cta_sync()

        def n_items_local():
            n_items = txl.local_scalar("int32")
            w = txl.local_scalar("uint32")
            txl.ptx.ld.shared.u32(w, item_meta.ptr_to([0]))
            txl.assign(n_items, txl.Cast("int32", w))
            return n_items

        def item_word(idx):
            w = txl.local_scalar("uint32")
            if global_items:
                txl.ptx.ld.global_.u32(w, items.ptr_to([cta * (2 * max_items) + 2 * idx]))
            else:
                txl.ptx.ld.shared.u32(w, item_tbl.ptr_to([2 * idx]))
            return w

        def item_word1(idx):
            w = txl.local_scalar("uint32")
            if global_items:
                txl.ptx.ld.global_.u32(w, items.ptr_to([cta * (2 * max_items) + 2 * idx + 1]))
            else:
                txl.ptx.ld.shared.u32(w, item_tbl.ptr_to([2 * idx + 1]))
            return w

        def item_tok0(w):
            lo = txl.bitwise_and(w, txl.uint32(0xFFFF))
            hi = txl.shift_left(txl.bitwise_and(txl.shift_right(w, txl.uint32(23)), txl.uint32(0x1F)), txl.uint32(16))
            return txl.Cast("int32", txl.bitwise_or(lo, hi))

        def item_nvalid(w):
            return txl.Cast("int32", txl.bitwise_and(txl.shift_right(w, txl.uint32(16)), txl.uint32(0x7F)))

        def item_first(w):
            return txl.bitwise_and(w, txl.uint32(1 << 30)) != txl.uint32(0)

        def item_last(w):
            return txl.bitwise_and(w, txl.uint32(1 << 31)) != txl.uint32(0)

        def item_src_hand(w):
            """Continuation piece: its state comes from handoff slot `cta` after the producer's flag."""
            return txl.bitwise_and(w, txl.uint32(1 << 29)) != txl.uint32(0)

        def item_dst_hand(w):
            """Head piece: its state goes to handoff slot `cta + 1`, published with a release flag."""
            return txl.bitwise_and(w, txl.uint32(1 << 28)) != txl.uint32(0)

        def item_seq(w1):
            return txl.Cast("int32", txl.bitwise_and(w1, txl.uint32(0xFFFF)))

        def item_head(w1):
            return txl.Cast("int32", txl.shift_right(w1, txl.uint32(16)))
        tbase = txl.local_scalar("uint32")
        txl.ptx.ld.shared.u32(tbase, tmem_addr.ptr_to([0]))

        tbase = txl.uniform(tbase)

        def tmem(col, lane_off=0):
            # plain add: the allocation starts at lane 0 and the column sum stays below 2^16, so no address field overflows and constant offsets fold into the tcgen05 immediates
            if isinstance(col, int):
                return tbase + txl.uint32((lane_off << 16) + col)
            return tbase + txl.uint32(lane_off << 16) + txl.Cast("uint32", col)

        def aa_col(b2, extra=0):
            return C_AA0 + extra + 64 * b2

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def warrive(b, idx):
            txl.cuda.warp_sync()
            with txl.If(lane == 0), txl.Then():
                b.arrive(idx)

        def irange(name):
            token = txl.alloc_local((1,), "uint32")
            txl.assign(token[0], txl.cuda.iket.range_start(name))
            return token

        def iend(token):
            txl.cuda.iket.range_end(token[0])

        def mark(name):
            txl.cuda.iket.mark(name)

        def fwait(b, stage, parity, name=None):
            tok = irange(name) if name else None
            ready = txl.local_scalar("uint32", init=txl.uint32(0))
            # no suspend-time hint: a 2-instruction spin loop keeps the hot code small
            with txl.While(ready == txl.uint32(0)):
                txl.ptx.mbarrier.try_wait.parity.shared.b64(ready, b.ptr_to([stage]), txl.Cast("uint32", parity))
            iend(tok)

        ZERO4 = (txl.uint32(0),) * 4

        def mma(d, aop, bop, idesc, acc):
            if not isinstance(acc, bool):
                acc = txl.local_scalar("uint32", init=txl.Select(acc, txl.uint32(1), txl.uint32(0)))
            txl.ptx[MMA](txl.Cast("uint32", d), aop, bop, txl.uint32(idesc), *ZERO4, txl.ptx.pred(acc))

        def mma_ws(d, aop, bop, idesc, acc, collector):
            txl.ptx[f"{MMA_WS}.collector::b{collector}::fill"](
                txl.Cast("uint32", d),
                aop,
                bop,
                txl.uint32(idesc),
                txl.ptx.pred(acc),
                txl.uint64(0),
            )

        def mma_ws_lastuse(d, aop, bop, idesc, acc, collector):
            txl.ptx[f"{MMA_WS}.collector::b{collector}::lastuse"](
                txl.Cast("uint32", d),
                aop,
                bop,
                txl.uint32(idesc),
                txl.ptx.pred(acc),
                txl.uint64(0),
            )

        def commit(ptr):
            txl.ptx[COMMIT](ptr)

        PARENT_ROWS = {"kq": 2 * BT, "g": BT, "v": BT, "kA": BT, "q2": BT, "k2": BT, "Aqk": 64, "LTT": 64, "TpT": 64, "TpTlo": 64}
        LBO_UNITS = sorted({r * 8 for r in PARENT_ROWS.values()} | {0})

        def make_templates():
            t = {}
            for ldo in LBO_UNITS:
                d = txl.SmemDescriptor()
                d.init(kq_t[0].ptr_to(0, 0), ldo=ldo, sdo=64, swizzle=3)
                t[ldo] = d.desc
            return t

        def tile_desc(tmpl, name, cols, major, kp=0, stage_units=None):
            """Descriptor of tile `name` at K step `kp` (a Python int, or a loop variable for rolled K loops)."""
            lbo = PARENT_ROWS[name] * 8
            ldo = 0 if (major == "k" and cols <= 64) else lbo
            if major == "k":
                step = (kp % 4) * 2 + (kp // 4) * lbo
            else:
                step = kp * 128
            if isinstance(kp, int):
                off = (TOFF[name] >> 4) + step
                d = tmpl[ldo] + txl.uint64(off) if off else tmpl[ldo]
            else:
                d = tmpl[ldo] + txl.Cast("uint64", step + (TOFF[name] >> 4))
            if stage_units is not None:
                d = d + txl.Cast("uint64", stage_units)
            return d

        def bf16x2(lo, hi):
            r = txl.local_scalar("uint32")
            txl.ptx.cvt.rn.bf16x2.f32(r, hi, lo)
            return r

        def unpack(u):
            lo = txl.reinterpret("float32", txl.shift_left(u, txl.uint32(16)))
            hi = txl.reinterpret("float32", txl.bitwise_and(u, txl.uint32(0xFFFF0000)))
            return lo, hi

        def hmul2(a, b):
            r = txl.local_scalar("uint32")
            txl.ptx.mul.rn.bf16x2(r, a, b)
            return r

        def hfma2(a, b, c):
            r = txl.local_scalar("uint32")
            txl.ptx.fma.rn.bf16x2(r, a, b, c)
            return r

        def hrcp2(a):
            """Packed bf16 reciprocal from a bit seed and one fused Newton step.

            Positive normal inputs use the exponent/mantissa reflection seed
            0x7ef2 - bits.  The constant minimizes the exhaustive bf16 maximum
            relative error after r * (2 - a*r); the fused step keeps that
            bound below 0.0063 for the range used by the scaled gate factors.
            """
            r0 = txl.uint32(0x7EF27EF2) - a
            corr = hfma2(
                txl.bitwise_xor(a, txl.uint32(0x80008000)),
                r0,
                txl.uint32(0x40004000),
            )
            return hmul2(r0, corr)

        def ex2(x):
            r = txl.local_scalar("float32")
            txl.ptx.ex2.approx.ftz.f32(r, x)
            return r

        sp = txl.specialize()
        r_state = sp.role("state", warps=W_STATE, regs=state_regs)
        wg1 = sp.warpgroup("wg1", warps=range(4, 8), regs=wg_regs)
        r_ld = sp.role("load", warps=[W_LD], group=wg1)
        r_mmi = sp.role("mma_i", warps=[W_MMI], group=wg1)
        r_mmc = sp.role("mma_c", warps=[W_MMC], group=wg1)
        r_rnd = sp.role("rounds", warps=[W_RND], group=wg1)
        r_intra = sp.role("intra", warps=W_INTRA, regs=intra_regs)
        r_prep = sp.role("prep", warps=W_PREP, regs=prep_regs)

        def load_body():
            n_items = n_items_local()

            def issue_loads(cc):
                sc = cc % 2
                with txl.If(elected()), txl.Then():
                    tok0 = txl.local_scalar("int32", init=item_tok0(item_word(cc)))
                    h = txl.local_scalar("int32", init=item_head(item_word1(cc)))
                    mb = txl.cuda.cvta_generic_to_shared(ring_full.ptr_to([sc]))
                    # k / q: 4 KB boxes (64 channels x 32 tokens x 1 channel slab) into the interleaved kq tile
                    for tx, tmap in ((0, k_map), (1, q_map)):
                        for hf_ in range(2):
                            for sl_ in range(2):
                                txl.ptx[TMA3](
                                    kq_t[sc].ptr_to(64 * hf_ + 32 * tx, 64 * sl_),
                                    txl.address_of(tmap),
                                    txl.int32(0),
                                    tok0 + 32 * hf_,
                                    txl.Cast("int32", 2 * h + sl_),
                                    mb,
                                    CACHE_EVICT_FIRST,
                                )
                    txl.ptx[TMA3](
                        g_t[sc].ptr_to(0, 0),
                        txl.address_of(g_map),
                        txl.int32(0),
                        tok0,
                        txl.Cast("int32", 2 * h),
                        mb,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx[TMA2](
                        beta_s.ptr_to([sc * 512]),
                        txl.address_of(beta_map),
                        txl.Cast("int32", (h // 8) * 8),
                        tok0,
                        mb,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(
                        ring_full.ptr_to([sc]), txl.uint32(3 * BT * D * 2 + 1024)
                    )

            issue_loads(txl.int32(0))
            with txl.If(n_items > 1), txl.Then():
                issue_loads(txl.int32(1))
            with txl.serial(n_items, unroll=False) as c:
                with txl.If(tvm.tirx.all(c >= 1, c + 1 < n_items)), txl.Then():
                    fwait(aa_r, (c + 1) % 2, ((c - 1) // 2) % 2, "ld-wait-aar")
                    txl.ptx.fence.proxy.async_.shared__cta()
                    issue_loads(c + 1)
                with txl.If(c >= 1), txl.Then():
                    fwait(u_done, (c + 1) % 2, ((c - 1) // 2) % 2, "ld-wait-udone")
                with txl.If(elected()), txl.Then():
                    tok0v = txl.local_scalar("int32", init=item_tok0(item_word(c)))
                    hv = txl.local_scalar("int32", init=item_head(item_word1(c)))
                    mbv = txl.cuda.cvta_generic_to_shared(v_full.ptr_to([0]))
                    txl.ptx[TMA3](
                        v_t.ptr_to(0, 0),
                        txl.address_of(v_map),
                        txl.int32(0),
                        tok0v,
                        txl.Cast("int32", 2 * hv),
                        mbv,
                        CACHE_EVICT_FIRST,
                    )
                    txl.ptx.mbarrier.arrive.expect_tx.shared.b64(v_full.ptr_to([0]), txl.uint32(BT * D * 2))

        def rounds_body(halves):
            """The assigned independent halves of the intra-chunk products.  Target half h (tokens 32h..32h+31) of the
            round accumulator lives at lane offset 16h of AA[b2]: D_h[m][n] = source_h[m] . target_h[n],
            columns 0-31 = Akk^T (target = k * 2^(gamma_t - r_h)), 32-63 = Aqk^T (target = q * scale * 2^(gamma_t - r_h)).
            Half 0 (r_0 = gamma_31 / 2): source = k * 2^(r_0 - gamma_m) in g_t[b2] rows 0-31 (rows 32-63 are the
            raw gate tile, finite and always masked).  Half 1 (r_1 = eps): source = kA (the delta operand
            k * 2^(eps - gamma_m)).  Targets: k_t[b2] row t = k-target of token t, q_t[b2] row t = q-target of
            token t (prep writes each row in place of the raw row it read); half h uses rows 32h..32h+31 of both
            tiles as two N=32 operands (columns 0-31 and 32-63 of the accumulator)."""
            n_items = n_items_local()
            with txl.If(elected()), txl.Then():
                tmpl = make_templates()
                with txl.serial(n_items, unroll=False) as c:
                    b2 = c % 2
                    su = b2 * STAGE_UNITS
                    with txl.If(c >= 2), txl.Then():
                        fwait(aa_free, b2, (c // 2 + 1) % 2, "rd-wait-aafree")
                    fwait(rfull, b2, (c // 2) % 2, "rd-wait-rfull")
                    for hh_ in halves:
                        tok_round = irange(f"rd-half{hh_}")
                        mark("rd-issue")
                        a_name = "g" if hh_ == 0 else "kA"
                        # B operand: rows 64 hh_ .. 64 hh_ + 63 of the kq tile = [k-targets of half hh_ | q-targets] -> accumulator columns 0-31 = Akk^T, 32-63 = Aqk^T; the K loop is rolled (run-time descriptor offsets) to keep the hot code small
                        with txl.serial(8, unroll=False) as kp:
                            bd = tile_desc(tmpl, "kq", 128, "k", kp, 2 * su) + txl.uint64(512 * hh_)
                            mma(
                                tmem(aa_col(b2), 16 * hh_),
                                tile_desc(tmpl, a_name, 128, "k", kp, su),
                                bd,
                                ID_AQK64,
                                kp != 0,
                            )
                        commit((akk_r if hh_ == 0 else aa_r).ptr_to([b2]))
                        iend(tok_round)

        def mmc_body():
            """Recurrence MMAs: G^T = S^T k2^T; O = S^T q2^T; u^T = v^T T'^T; v_new^T = u^T - G_bf^T T'^T;
            S^T += v_new^T kA; O += v_new^T Aqk^T."""
            n_items = n_items_local()
            with txl.If(elected()), txl.Then():
                tmpl = make_templates()
                with txl.serial(n_items, unroll=False) as c:
                    b2 = c % 2
                    su = b2 * STAGE_UNITS
                    fwait(rfull, b2, (c // 2) % 2, "mm-wait-rfull")
                    fwait(S_ready, 0, c % 2, "mm-wait-sready")
                    mark("mm-issue-G")
                    # rolled K loops for the plain MMAs (G, O_I, delta, OX) keep the hot code small
                    with txl.serial(8, unroll=False) as kp:
                        mma(
                            tmem(C_GACC, 0),
                            txl.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                            tile_desc(tmpl, "k2", 128, "k", kp),
                            ID_G64,
                            kp != 0,
                        )
                    commit(g_done.ptr_to([0]))
                    with txl.If(c >= 1), txl.Then():
                        fwait(o_free, 0, (c + 1) % 2, "mm-wait-ofree")
                    mark("mm-issue-OI")
                    with txl.serial(8, unroll=False) as kp:
                        mma(
                            tmem(C_OACC, 0),
                            txl.Cast("uint32", tmem(C_SBF + 8 * kp, 0)),
                            tile_desc(tmpl, "q2", 128, "k", kp),
                            ID_G64,
                            kp != 0,
                        )
                    commit(oi_done.ptr_to([0]))
                    fwait(T_ready, b2, (c // 2) % 2, "mm-wait-Tready")
                    fwait(v_full, 0, c % 2, "mm-wait-vfull")
                    mark("mm-issue-u")
                    for kp in range(4):
                        mma_ws(
                            tmem(aa_col(b2), 0),
                            tile_desc(tmpl, "v", 128, "mn", kp),
                            tile_desc(tmpl, "TpT", 64, "mn", kp),
                            ID_U,
                            kp != 0,
                            kp,
                        )
                    commit(u_done.ptr_to([b2]))
                    fwait(g_ready, 0, c % 2, "mm-wait-gready")
                    mark("mm-issue-vnew")
                    for kp in range(4):
                        mma_ws_lastuse(
                            tmem(aa_col(b2), 0),
                            txl.Cast("uint32", tmem(C_GBF + 8 * kp, 0)),
                            tile_desc(tmpl, "TpT", 64, "mn", kp),
                            ID_VN,
                            True,
                            kp,
                        )
                    for kp in range(4):
                        mma(
                            tmem(aa_col(b2), 0),
                            txl.Cast("uint32", tmem(C_GBF + 8 * kp, 0)),
                            tile_desc(tmpl, "TpTlo", 64, "mn", kp),
                            ID_VN,
                            True,
                        )
                    commit(vnew_done.ptr_to([b2]))
                    fwait(aa_free, b2, (c // 2) % 2, "mm-wait-aafree")
                    mark("mm-issue-delta")
                    with txl.serial(4, unroll=False) as kp:
                        mma(
                            tmem(C_SACC, 0),
                            txl.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            tile_desc(tmpl, "kA", 128, "mn", kp, su),
                            ID_DL,
                            True,
                        )
                    commit(kA_free.ptr_to([b2]))
                    fwait(Aqk_ready, 0, c % 2, "mm-wait-aqk")
                    mark("mm-issue-OX")
                    with txl.serial(4, unroll=False) as kp:
                        mma(
                            tmem(C_OACC, 0),
                            txl.Cast("uint32", tmem(C_VNBF + 8 * kp, 0)),
                            tile_desc(tmpl, "Aqk", 64, "mn", kp),
                            ID_OX,
                            True,
                        )
                    commit(o_done.ptr_to([0]))

        def state_body():
            v_idx = tid
            regs = txl.alloc_local((64,), "float32")
            packed = txl.alloc_local((16,), "uint32")

            def st_packed(col, base):
                for j in range(16):
                    txl.assign(packed[j], bf16x2(regs[base + 2 * j], regs[base + 2 * j + 1]))
                txl.ptx[ST32x16](tmem(col), *[packed[j] for j in range(16)])

            n_items = n_items_local()

            def load_state_from(buf, base, external=False):
                """[v][k] fp32 block at element offset `base` (V-first, lane = v row) -> S_acc fp32 and S_bf bf16 in TMEM, then S_ready."""

                for half_ in range(2):
                    if external:
                        for u in range(8):
                            txl.ptx["ld.global.L1::no_allocate.L2::evict_first.v8.f32"](
                                *[regs[8 * u + j] for j in range(8)],
                                buf.ptr_to([base + 64 * half_ + 8 * u]),
                            )
                    else:
                        for u in range(16):
                            txl.ptx.ld.global_.v4.f32(
                                regs[4 * u], regs[4 * u + 1], regs[4 * u + 2], regs[4 * u + 3],
                                buf.ptr_to([base + 64 * half_ + 4 * u]),
                            )
                    txl.ptx[ST32x64](tmem(C_SACC + 64 * half_), *[regs[j] for j in range(64)])
                    st_packed(C_SBF + 32 * half_, 0)
                    st_packed(C_SBF + 32 * half_ + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(S_ready, 0)

            def load_state(seq, h):
                """initial_state[seq][h] -> TMEM."""
                base = txl.local_scalar("int32", init=(seq * H + h) * (D * D) + v_idx * D)
                load_state_from(h0, base, external=True)

            def load_state_hand():
                """Continuation piece: acquire the producer CTA's flag, then expand its bf16 handoff state.

                Warp 0 polls (all lanes read the same word); the other state warps wait at the role barrier, which
                orders their loads after the acquire.  The flag is reset here so every launch starts from zero."""
                with txl.If(warp == 0), txl.Then():
                    tok_fl = irange("st-wait-hand")
                    fl = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(fl == 0):
                        txl.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([cta]))
                        with txl.If(fl == 0), txl.Then():
                            txl.cuda.nano_sleep(txl.uint64(256))
                    iend(tok_fl)
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        txl.ptx.st.global_.s32(flags.ptr_to([cta]), txl.int32(0))
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                base = txl.local_scalar("int32", init=cta * (D * D) + v_idx * D)
                if bf16_handoff:
                    for half_ in range(2):
                        for q_ in range(2):
                            qbase = 64 * half_ + 32 * q_
                            for u in range(4):
                                txl.ptx.ld.global_.v4.b32(
                                    packed[4 * u],
                                    packed[4 * u + 1],
                                    packed[4 * u + 2],
                                    packed[4 * u + 3],
                                    hand.ptr_to([base + qbase + 8 * u]),
                                )
                            for j in range(16):
                                lo, hi = unpack(packed[j])
                                txl.assign(regs[2 * j], lo)
                                txl.assign(regs[2 * j + 1], hi)
                            txl.ptx[ST32x32](tmem(C_SACC + qbase), *[regs[j] for j in range(32)])
                            txl.ptx[ST32x16](tmem(C_SBF + qbase // 2), *[packed[j] for j in range(16)])
                    txl.ptx.tcgen05.wait__st.sync.aligned()
                    txl.ptx[FENCE_BEFORE]()
                    warrive(S_ready, 0)
                else:
                    load_state_from(hand, base)

            def store_state_to(buf, fbase, external=False):
                """S_acc (after the piece's last decay) -> [v][k] fp32 block at element offset `fbase`, one v row per lane."""
                for half_ in range(2):
                    txl.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(C_SACC + 64 * half_))
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    if external:
                        for u in range(8):
                            txl.ptx["st.global.L1::no_allocate.L2::evict_first.v8.f32"](
                                buf.ptr_to([fbase + 64 * half_ + 8 * u]),
                                *[regs[8 * u + j] for j in range(8)],
                            )
                    else:
                        for u in range(16):
                            txl.ptx.st.global_.v4.f32(
                                buf.ptr_to([fbase + 64 * half_ + 4 * u]),
                                regs[4 * u],
                                regs[4 * u + 1],
                                regs[4 * u + 2],
                                regs[4 * u + 3],
                            )

            def store_state(seq, h):
                """-> final_state[seq][h]."""
                fbase = txl.local_scalar("int32", init=(seq * H + h) * (D * D) + v_idx * D)
                store_state_to(final_state, fbase, external=True)

            def store_state_hand():
                """Head piece: quantize once into a bf16 handoff slot, then publish it.

                Every thread fences its own stores, the role barrier collects them, and thread 0 releases the flag
                at gpu scope.  The authoritative final-state path remains fp32."""
                fbase = txl.local_scalar("int32", init=(cta + txl.int32(1)) * (D * D) + v_idx * D)
                if bf16_handoff:
                    for half_ in range(2):
                        for q_ in range(2):
                            qbase = 64 * half_ + 32 * q_
                            txl.ptx[LD32x16](*[packed[j] for j in range(16)], tmem(C_SBF + qbase // 2))
                            txl.ptx.tcgen05.wait__ld.sync.aligned()
                            for u in range(4):
                                txl.ptx.st.global_.v4.b32(
                                    hand.ptr_to([fbase + qbase + 8 * u]),
                                    packed[4 * u],
                                    packed[4 * u + 1],
                                    packed[4 * u + 2],
                                    packed[4 * u + 3],
                                )
                else:
                    store_state_to(hand, fbase)
                txl.ptx.fence.acq_rel.gpu()
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                with txl.If(tid == 0), txl.Then():
                    txl.ptx.fence.acq_rel.gpu()
                    txl.ptx.st.release.gpu.global_.s32(flags.ptr_to([cta + txl.int32(1)]), txl.int32(1))

            def epilogue(c1, tok0, nvalid, h):
                """Stage both token halves through SMEM; full chunks go out by TMA, tail chunks by predicated stores."""
                fwait(o_done, 0, c1 % 2, "st-wait-odone")
                tok_epi = irange("st-epilogue")
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx[FENCE_AFTER]()
                for half in range(2):
                    txl.ptx[LD16x8](*[regs[32 * half + j] for j in range(32)], tmem(C_OACC, 16 * half))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(o_free, 0)
                for g_ in range(2):
                    for half in range(2):
                        b = 32 * half
                        for m in range(4):
                            u = 4 * g_ + m
                            for hh in range(2):
                                txl.assign(packed[2 * m + hh], bf16x2(regs[b + 4 * u + 2 * hh], regs[b + 4 * u + 2 * hh + 1]))
                        for hh in range(2):
                            if g_ == 0:
                                dst = o_st[warp // 2].ptr_to(
                                    lane, 32 * (warp % 2) + 16 * half + 8 * hh
                                )
                            else:
                                dst = Aqk_s.ptr_to(
                                    32 * (warp // 2) + lane,
                                    32 * (warp % 2) + 16 * half + 8 * hh,
                                )
                            txl.ptx.stmatrix.sync.aligned.m8n8.x4.trans.shared.b16(
                                dst,
                                packed[hh],
                                packed[2 + hh],
                                packed[4 + hh],
                                packed[6 + hh],
                            )
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                with txl.If(nvalid == BT):
                    with txl.Then():
                        with txl.If(warp == 0), txl.Then():
                            with txl.If(lane == 0), txl.Then():
                                for g_ in range(2):
                                    src = o_st[0].ptr_to(0, 0) if g_ == 0 else Aqk_s.ptr_to(0, 0)
                                    txl.ptx[TMA_S2G3](
                                        txl.address_of(o_map),
                                        txl.int32(0),
                                        tok0 + 32 * g_,
                                        txl.Cast("int32", 2 * h),
                                        src,
                                        CACHE_EVICT_FIRST,
                                    )
                                txl.ptx.cp.async_.bulk.commit_group()
                                txl.ptx.cp.async_.bulk.wait_group.read(0)
                                aqk_stage_free.arrive(0)
                    with txl.Else():
                        j8 = tid % 8
                        trow = tid // 8
                        obase = txl.local_scalar("int32", init=(tok0 * H + h) * D + 8 * j8)
                        ov = txl.alloc_local((4,), "uint32")
                        for g_ in range(2):
                            with txl.serial(2, unroll=False) as p:
                                for th_ in range(2):
                                    t = trow + 16 * th_
                                    if g_ == 0:
                                        src = o_st[p].ptr_to(t, 8 * j8)
                                    else:
                                        src = Aqk_s.ptr_to(32 * p + t, 8 * j8)
                                    txl.ptx.ld.shared.v4.b32(ov[0], ov[1], ov[2], ov[3], src)
                                    txl.ptx.st.global_.v4.b32(
                                        out.ptr_to([obase + (32 * g_ + t) * (H * D) + 64 * p]),
                                        ov[0],
                                        ov[1],
                                        ov[2],
                                        ov[3],
                                        pred=(32 * g_ + t < nvalid),
                                    )
                        txl.ptx.bar.sync(txl.uint32(NB_STATE), txl.uint32(128))
                        with txl.If(warp == 0), txl.Then():
                            with txl.If(lane == 0), txl.Then():
                                aqk_stage_free.arrive(0)
                iend(tok_epi)

            def gconv(c):
                """G^T (fp32, GACC) -> bf16 GBF."""
                fwait(g_done, 0, c % 2, "st-wait-gdone")
                tok_g = irange("st-gconv")
                txl.ptx[FENCE_AFTER]()
                txl.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(C_GACC))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                st_packed(C_GBF, 0)
                st_packed(C_GBF + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(g_ready, 0)
                iend(tok_g)

            def load_piece_state(idx):
                """Item idx starts a piece: bring its state into TMEM (handoff slot or initial_state), arrive S_ready."""
                iwn = item_word(idx)
                iw1n = item_word1(idx)
                tok_ld = irange("st-load")
                with txl.If(item_src_hand(iwn)):
                    with txl.Then():
                        load_state_hand()
                    with txl.Else():
                        load_state(item_seq(iw1n), item_head(iw1n))
                iend(tok_ld)

            # a piece's first item has its state loaded at the end of the previous piece's last item, ahead of that item's epilogue; only item 0 is loaded here
            with txl.If(n_items > 0), txl.Then():
                load_piece_state(txl.int32(0))
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = txl.local_scalar("int32", init=item_head(iw1))
                with txl.If(item_last(iw)), txl.Then():
                    with txl.If(c + 1 < n_items), txl.Then():
                        # the next piece's initial state (64 KB, contiguous) into L2 while this item runs
                        iwn = item_word(c + 1)
                        with txl.If(txl.Not(item_src_hand(iwn))), txl.Then():
                            with txl.If(tid == 0), txl.Then():
                                iw1n = item_word1(c + 1)
                                pf_base = txl.local_scalar("int32", init=(item_seq(iw1n) * H + item_head(iw1n)) * (D * D))
                                txl.ptx["cp.async.bulk.prefetch.L2.global"](h0.ptr_to([pf_base]), txl.uint32(D * D * 4))
                gconv(c)
                fwait(vnew_done, s, (c // 2) % 2, "st-wait-vnew")
                tok_vn = irange("st-vnconv")
                txl.ptx[FENCE_AFTER]()
                txl.ptx[LD32x64](*[regs[j] for j in range(64)], tmem(aa_col(s)))
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                st_packed(C_VNBF, 0)
                st_packed(C_VNBF + 16, 32)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                warrive(aa_free, s)
                iend(tok_vn)

                fwait(kA_free, s, (c // 2) % 2, "st-wait-delta")
                txl.ptx[FENCE_AFTER]()
                fwait(dvec_ready, c % 3, (c // 3) % 2, "st-wait-dvec")
                tok_dec = irange("st-decay")

                # decay: the four 32-column quarters as a 2-iteration loop with an unconditional prefetch of the next two, to keep the hot code small
                def ld_q(col, b):
                    txl.ptx[LD32x32](*[regs[b + j] for j in range(32)], tmem(col))

                def proc_q(qi, b):
                    dv4 = txl.alloc_local((4,), "float32")
                    for j in range(0, 32, 4):
                        txl.ptx.ld.shared.v4.f32(dv4[0], dv4[1], dv4[2], dv4[3], dvec.ptr_to([(c % 3) * 128 + 32 * qi + j]))
                        for m in range(0, 4, 2):
                            pair = txl.local_scalar("uint64")
                            txl.ptx.mov.b64(pair, regs[b + j + m], regs[b + j + m + 1])
                            txl.ptx.mul.rn.ftz.f32x2(pair, pair, txl.cuda.make_float2(dv4[m], dv4[m + 1]))
                            txl.ptx.mov.b64(regs[b + j + m], regs[b + j + m + 1], pair)
                    for j in range(16):
                        txl.assign(packed[j], bf16x2(regs[b + 2 * j], regs[b + 2 * j + 1]))
                    txl.ptx[ST32x16](tmem(C_SBF + 16 * qi), *[packed[j] for j in range(16)])
                    txl.ptx[ST32x32](tmem(C_SACC + 32 * qi), *[regs[b + j] for j in range(32)])

                ld_q(C_SACC, 0)
                ld_q(C_SACC + 32, 32)
                with txl.serial(2, unroll=False) as kq_:
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    proc_q(2 * kq_, 0)
                    # the last iteration's prefetch reads the complete, read-only bf16 v_new columns instead, so no in-flight store is re-read
                    ld_q(txl.Select(kq_ == 0, C_SACC + 64, C_VNBF), 0)
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    proc_q(2 * kq_ + 1, 32)
                    ld_q(txl.Select(kq_ == 0, C_SACC + 96, C_VNBF), 32)
                warrive(dvec_free, c % 3)
                txl.ptx.tcgen05.wait__st.sync.aligned()
                txl.ptx[FENCE_BEFORE]()
                with txl.If(item_last(iw)):
                    with txl.Then():
                        tok_st = irange("st-store")
                        with txl.If(item_dst_hand(iw)):
                            with txl.Then():
                                store_state_hand()
                            with txl.Else():
                                store_state(item_seq(iw1), hcur)
                        iend(tok_st)
                        with txl.If(c + 1 < n_items), txl.Then():
                            load_piece_state(c + 1)
                    with txl.Else():
                        warrive(S_ready, 0)
                iend(tok_dec)
                epilogue(c, item_tok0(iw), item_nvalid(iw), hcur)

        def intra_body():
            """L = strict-lower(Akk) beta -> T = (I+L)^-1 -> T' = T diag(beta); Aqk_s."""
            q = warp - W_INTRA[0]
            r16 = lane // 4
            c4 = lane % 4
            aqk = txl.alloc_local((32,), "float32")
            acc = txl.alloc_local((8,), "float32")
            a_frag = txl.alloc_local((4,), "uint32")
            b_frag = txl.alloc_local((4,), "uint32")
            aM = txl.alloc_local((4,), "uint32")
            bM = txl.alloc_local((4,), "uint32")
            aP = txl.alloc_local((4,), "uint32")
            bP4 = txl.alloc_local((4,), "uint32")
            bP8 = txl.alloc_local((4,), "uint32")
            tA = [txl.alloc_local((4,), "uint32") for _ in range(3)]
            bL = [txl.alloc_local((4,), "uint32") for _ in range(3)]

            def round_coords(L, reg):
                rep, rem = divmod(reg, 4)
                hh, cc = divmod(rem, 2)
                i = 8 * rep + 2 * c4 + cc
                j = q * 16 + r16 + 8 * hh
                return j, i, L == 0, rep // 2

            def isync():
                txl.ptx.bar.sync(txl.uint32(NB_INTRA), txl.uint32(128))

            def frag_addr(tile, rb, cb):
                return tile.ptr_to(rb + lane % 16, cb + (lane // 16) * 8)

            def ld_b(B, br, bc, dst):
                txl.ptx.ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16(dst[0], dst[1], dst[2], dst[3], frag_addr(B, br, bc))

            def st_frag(tile, rb, cb, src):
                txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(frag_addr(tile, rb, cb), src[0], src[1], src[2], src[3])

            def movm(dst, src):
                for z in range(4):
                    txl.ptx.movmatrix.sync.aligned.m8n8.trans.b16(dst[z], src[z])

            def mma_frag(a, b, clear):
                if clear:
                    for z in range(8):
                        txl.assign(acc[z], txl.float32(0.0))
                for nh in range(2):
                    z = 4 * nh
                    txl.ptx.mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32(
                        acc[z], acc[z + 1], acc[z + 2], acc[z + 3],
                        a[0], a[1], a[2], a[3],
                        b[2 * nh], b[2 * nh + 1],
                        acc[z], acc[z + 1], acc[z + 2], acc[z + 3],
                    )

            def pack_acc(dst, neg=False, rs=None):
                for z in range(4):
                    v0, v1 = acc[2 * z], acc[2 * z + 1]
                    if rs is not None:
                        v0, v1 = v0 * rs[z % 2], v1 * rs[z % 2]
                    p = bf16x2(v0, v1)
                    txl.assign(dst[z], txl.bitwise_xor(p, txl.uint32(0x80008000)) if neg else p)

            def identity_minus_acc():
                """acc <- I - acc (one FADD per element; identical to negating and adding the identity)."""
                for nh in range(2):
                    z = 4 * nh
                    txl.assign(acc[z], txl.Select(r16 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)) - acc[z])
                    txl.assign(acc[z + 1], txl.Select(r16 == 8 * nh + 2 * c4 + 1, txl.float32(1.0), txl.float32(0.0)) - acc[z + 1])
                    txl.assign(acc[z + 2], txl.Select(r16 + 8 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)) - acc[z + 2])
                    txl.assign(acc[z + 3], txl.Select(r16 + 8 == 8 * nh + 2 * c4 + 1, txl.float32(1.0), txl.float32(0.0)) - acc[z + 3])

            def add_identity():
                for nh in range(2):
                    z = 4 * nh
                    txl.assign(acc[z], acc[z] + txl.Select(r16 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)))
                    txl.assign(acc[z + 1], acc[z + 1] + txl.Select(r16 == 8 * nh + 2 * c4 + 1, txl.float32(1.0), txl.float32(0.0)))
                    txl.assign(acc[z + 2], acc[z + 2] + txl.Select(r16 + 8 == 8 * nh + 2 * c4, txl.float32(1.0), txl.float32(0.0)))
                    txl.assign(acc[z + 3], acc[z + 3] + txl.Select(r16 + 8 == 8 * nh + 2 * c4 + 1, txl.float32(1.0), txl.float32(0.0)))

            def beta_rows(sb, J, neg=True):
                b0 = txl.local_scalar("float32")
                b1 = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(b0, bsig.ptr_to([sb * 64 + 16 * J + r16]))
                txl.ptx.ld.shared.f32(b1, bsig.ptr_to([sb * 64 + 16 * J + r16 + 8]))
                if not neg:
                    return (b0, b1)
                return (txl.float32(0.0) - b0, txl.float32(0.0) - b1)

            LT = LTT
            TT = LTT
            TpT = TpT_t
            TpTlo = TpTlo_t
            a_lo = txl.alloc_local((4,), "uint32")

            def pack_acc_rs_hilo(dst_hi, dst_lo, rs):
                """Row-scaled C fragment -> bf16 hi/lo A fragments (T'^T = beta_j T^T); hi + lo carries ~16 mantissa bits because the block chains sum columns of T that cancel to ~(1-beta)^15."""
                for z in range(4):
                    v0 = txl.local_scalar("float32", init=acc[2 * z] * rs[z % 2])
                    v1 = txl.local_scalar("float32", init=acc[2 * z + 1] * rs[z % 2])
                    hi = bf16x2(v0, v1)
                    h0, h1 = unpack(hi)
                    txl.assign(dst_hi[z], hi)
                    txl.assign(dst_lo[z], bf16x2(v0 - h0, v1 - h1))

            with txl.If(q == 3), txl.Then():
                z = txl.uint32(0)
                for m in range(6):
                    n = m * 32 + lane
                    rr, hf = (n % 32) // 2, n % 2
                    UJ = [1, 2, 3, 2, 3, 3][m]
                    UI = [0, 0, 0, 1, 1, 2][m]
                    txl.ptx.st.shared.v4.u32(TpT.ptr_to(16 * UJ + rr, 16 * UI + 8 * hf), z, z, z, z)
                    txl.ptx.st.shared.v4.u32(TpTlo.ptr_to(16 * UJ + rr, 16 * UI + 8 * hf), z, z, z, z)
            txl.ptx.fence.proxy.async_.shared__cta()
            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                sb = c % 3

                fwait(akk_r, s, (c // 2) % 2, "in-wait-akk")
                with txl.If(q >= 2), txl.Then():
                    fwait(aa_r, s, (c // 2) % 2, "in-wait-aar")
                tok_in = irange("in-transform")
                txl.ptx[FENCE_AFTER]()
                bcol = txl.alloc_local((16,), "float32")
                for I in range(4):
                    for r2 in range(2):
                        txl.ptx.ld.shared.v2.f32(
                            bcol[I * 4 + r2 * 2], bcol[I * 4 + r2 * 2 + 1], bsig.ptr_to([sb * 64 + 16 * I + 8 * r2 + 2 * c4])
                        )
                dblk = txl.alloc_local((8,), "float32")

                txl.ptx[LD16x2](
                    *[dblk[j] for j in range(8)],
                    tbase
                    + txl.shift_left(txl.Cast("uint32", 16 * (q // 2)), txl.uint32(16))
                    + txl.Cast("uint32", aa_col(s, 0) + 16 * (q % 2)),
                )
                txl.ptx.tcgen05.wait__ld.sync.aligned()
                bdiag = txl.alloc_local((4,), "float32")
                for rep_ in range(2):
                    txl.ptx.ld.shared.v2.f32(bdiag[2 * rep_], bdiag[2 * rep_ + 1], bsig.ptr_to([sb * 64 + 16 * q + 8 * rep_ + 2 * c4]))

                for reg in range(0, 8, 2):
                    rep_, rem = divmod(reg, 4)
                    hh = rem // 2
                    j = q * 16 + r16 + 8 * hh
                    i = 16 * q + 8 * rep_ + 2 * c4
                    lv = []
                    for cc in range(2):
                        lv.append(
                            txl.local_scalar(
                                "float32",
                                init=txl.Select(
                                    i + cc > j,
                                    dblk[reg + cc] * bdiag[2 * rep_ + cc],
                                    txl.float32(0.0),
                                ),
                            )
                        )
                    hpack = bf16x2(lv[0], lv[1])
                    txl.assign(aM[reg // 2], hpack)
                movm(bM, aM)

                aqk_pk = txl.alloc_local((16,), "uint32")
                ltw = txl.alloc_local((16,), "uint32")
                npk = 0
                fwait(aa_r, s, (c // 2) % 2, "in-wait-aar2")
                txl.ptx[FENCE_AFTER]()
                for L in (1, 0):
                    for hh_ in range(2):
                        txl.ptx[LD16x4](
                            *[aqk[16 * hh_ + j] for j in range(16)], tmem(aa_col(s, 32 * (1 - L)), 16 * hh_)
                        )
                    txl.ptx.tcgen05.wait__ld.sync.aligned()
                    # packed word reg // 2 = 2 rep + hh holds rows q 16 + r16 + 8 hh, columns 8 rep + 2 c4 .. +1 (mma C order): stored as stmatrix.x4 16x16 blocks at frag_addr(tile, 16 q, 16 n); L^T blocks on or below the diagonal (n <= q) are skipped
                    for reg in range(0, 32, 2):
                        j, i, is_aqk, I = round_coords(L, reg)
                        rep_ = reg // 4
                        if is_aqk:
                            av = [
                                txl.local_scalar("float32", init=txl.Select(i + cc >= j, aqk[reg + cc], txl.float32(0.0)))
                                for cc in range(2)
                            ]
                            txl.assign(aqk_pk[npk], bf16x2(av[0], av[1]))
                            npk += 1
                            continue
                        lv = []
                        for cc in range(2):
                            bidx = I * 4 + (rep_ % 2) * 2 + cc
                            lv.append(txl.local_scalar("float32", init=aqk[reg + cc] * bcol[bidx]))
                        txl.assign(ltw[reg // 2], bf16x2(lv[0], lv[1]))
                    if L == 1:
                        for n_ in range(1, 4):
                            with txl.If(q < n_), txl.Then():
                                txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                                    frag_addr(LT, 16 * q, 16 * n_), ltw[4 * n_], ltw[4 * n_ + 1], ltw[4 * n_ + 2], ltw[4 * n_ + 3]
                                )

                mma_frag(aM, bM, True)
                pack_acc(aP)
                movm(b_frag, aP)
                mma_frag(aP, b_frag, True)
                pack_acc(a_frag)
                movm(bP4, a_frag)
                mma_frag(a_frag, bP4, True)
                pack_acc(a_frag)
                movm(bP8, a_frag)
                for z in range(4):
                    lo, hi = unpack(aP[z])
                    txl.assign(acc[2 * z], lo)
                    txl.assign(acc[2 * z + 1], hi)
                add_identity()
                pack_acc(a_frag)
                mma_frag(a_frag, bP4, False)
                pack_acc(a_frag)
                mma_frag(a_frag, bP8, False)
                pack_acc(a_frag)
                movm(b_frag, a_frag)
                # T = Q - M Q as one accumulate onto Q with the B fragment negated (sign flips are exact)
                for z in range(4):
                    txl.assign(b_frag[z], txl.bitwise_xor(b_frag[z], txl.uint32(0x80008000)))
                mma_frag(aM, b_frag, False)
                pack_acc(tA[0])

                movm(b_frag, tA[0])
                mma_frag(aM, b_frag, True)
                for z in range(4):
                    th0, th1 = unpack(tA[0][z])
                    txl.assign(acc[2 * z], acc[2 * z] + th0)
                    txl.assign(acc[2 * z + 1], acc[2 * z + 1] + th1)
                identity_minus_acc()
                pack_acc(aP)
                movm(bP4, aP)
                mma_frag(tA[0], bP4, True)
                for z in range(4):
                    th0, th1 = unpack(tA[0][z])
                    txl.assign(acc[2 * z], acc[2 * z] + th0)
                    txl.assign(acc[2 * z + 1], acc[2 * z + 1] + th1)
                pack_acc(tA[0])
                with txl.If(c >= 1), txl.Then():
                    fwait(vnew_done, (c + 1) % 2, ((c - 1) // 2) % 2, "in-wait-vnew")
                    txl.ptx.fence.proxy.async_.shared__cta()
                st_frag(TT, 16 * q, 16 * q, tA[0])
                # acc holds +T here (the negation and beta_rows' sign cancel): scale by +beta
                rs = beta_rows(sb, q, neg=False)
                pack_acc_rs_hilo(a_frag, a_lo, rs)
                st_frag(TpT, 16 * q, 16 * q, a_frag)
                st_frag(TpTlo, 16 * q, 16 * q, a_lo)
                isync()

                # beta is constant across the off-diagonal block-column solve: reuse the diagonal's row scales at every chain step
                chain_rs = (txl.float32(0.0) - rs[0], txl.float32(0.0) - rs[1])

                if intra_unroll:
                    def chain(I, J, terms):
                        for n, Kk in enumerate(terms):
                            ld_b(LT, 16 * Kk, 16 * I, bL[n])
                        ld_b(TT, 16 * I, 16 * I, b_frag)
                        for n, Kk in enumerate(terms):
                            mma_frag(tA[Kk - J], bL[n], n == 0)
                        pack_acc(a_frag)
                        mma_frag(a_frag, b_frag, True)
                        if I - J < 3:
                            pack_acc(tA[I - J], neg=True)
                        rs = beta_rows(sb, J)
                        pack_acc_rs_hilo(a_frag, a_lo, rs)
                        st_frag(TpT, 16 * J, 16 * I, a_frag)
                        st_frag(TpTlo, 16 * J, 16 * I, a_lo)

                    with txl.If(q == 0), txl.Then():
                        chain(1, 0, [0])
                        chain(2, 0, [0, 1])
                        chain(3, 0, [0, 1, 2])
                    with txl.If(q == 1), txl.Then():
                        chain(2, 1, [1])
                        chain(3, 1, [1, 2])
                    with txl.If(q == 2), txl.Then():
                        chain(3, 2, [2])
                else:
                    # software-pipelined chain: step s+1's ldmatrix operands are fetched during step s; unconsumed fetches are clamped to an already written block
                    bL2 = [txl.alloc_local((4,), "uint32") for _ in range(3)]
                    b_frag2 = txl.alloc_local((4,), "uint32")

                    def ld_ops(I_rt, dbL, db):
                        # unconsumed operands are clamped onto the diagonal block of the same column (always written)
                        I_c = txl.min(I_rt, 3)
                        for n in range(3):
                            ld_b(LT, 16 * txl.min(q + n, I_c), 16 * I_c, dbL[n])
                        ld_b(TT, 16 * I_c, 16 * I_c, db)

                    with txl.If(q < 3), txl.Then():
                        ld_ops(q + 1, bL, b_frag)
                    with txl.serial(3 - q, unroll=False) as step:
                        I_rt = q + 1 + step
                        mma_frag(tA[0], bL[0], True)
                        for n in range(1, 3):
                            with txl.If(n <= step), txl.Then():
                                mma_frag(tA[n], bL[n], False)
                        pack_acc(a_frag)
                        mma_frag(a_frag, b_frag, True)
                        ld_ops(I_rt + 1, bL2, b_frag2)
                        with txl.If(step == 0), txl.Then():
                            pack_acc(tA[1], neg=True)
                        with txl.If(step == 1), txl.Then():
                            pack_acc(tA[2], neg=True)
                        pack_acc_rs_hilo(a_frag, a_lo, chain_rs)
                        st_frag(TpT, 16 * q, 16 * I_rt, a_frag)
                        st_frag(TpTlo, 16 * q, 16 * I_rt, a_lo)
                        for n in range(3):
                            for z in range(4):
                                txl.assign(bL[n][z], bL2[n][z])
                        for z in range(4):
                            txl.assign(b_frag[z], b_frag2[z])

                isync()
                txl.ptx.fence.proxy.async_.shared__cta()
                txl.ptx[FENCE_BEFORE]()
                warrive(T_ready, s)
                iend(tok_in)

                with txl.If(c >= 1), txl.Then():
                    fwait(o_done, 0, (c + 1) % 2, "in-wait-odone")
                    fwait(aqk_stage_free, 0, (c + 1) % 2, "in-wait-aqkstage")
                    txl.ptx.fence.proxy.async_.shared__cta()
                for n_ in range(4):
                    txl.ptx.stmatrix.sync.aligned.m8n8.x4.shared.b16(
                        frag_addr(Aqk_s, 16 * q, 16 * n_), aqk_pk[4 * n_], aqk_pk[4 * n_ + 1], aqk_pk[4 * n_ + 2], aqk_pk[4 * n_ + 3]
                    )
                txl.ptx.fence.proxy.async_.shared__cta()
                warrive(Aqk_ready, 0)

        def norm_body():
            """Use the otherwise idle MMA-I warp for row norms and beta."""
            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = item_head(iw1)
                nvalid = item_nvalid(iw)
                s = c % 2
                with txl.If(item_first(iw)), txl.Then():
                    txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                tok_norm = irange("nr-work")
                for hf in range(2):
                    row = lane + 32 * hf
                    for which in range(2):
                        n0 = txl.local_scalar("float32")
                        n1 = txl.local_scalar("float32")
                        norm = txl.local_scalar("float32")
                        txl.ptx.ld.shared.f32(n0, rsq.ptr_to([s * 256 + which * 128 + row]))
                        txl.ptx.ld.shared.f32(n1, rsq.ptr_to([s * 256 + which * 128 + 64 + row]))
                        txl.ptx.rsqrt.approx.ftz.f32(norm, n0 + n1 + txl.float32(EPS))
                        fac = norm * scale if which == 0 else norm
                        fac = txl.Select(row < nvalid, fac, txl.float32(0.0))
                        txl.ptx.st.shared.b32(rsq.ptr_to([s * 256 + which * 128 + row]), bf16x2(fac, fac))
                    rawb = txl.local_scalar("uint16")
                    txl.ptx.ld.shared.b16(rawb, beta_s.ptr_to([s * 512 + row * 8 + hcur % 8]))
                    bv = txl.reinterpret("float32", txl.shift_left(txl.Cast("uint32", rawb), txl.uint32(16)))
                    sg = txl.idioms.sigmoid_tanh_approx_f32(bv)
                    txl.ptx.st.shared.f32(bsig.ptr_to([(c % 3) * 64 + row]), txl.Select(row < nvalid, sg, txl.float32(0.0)))
                iend(tok_norm)
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))

        def prep_body():
            """Gate/cumsum/normalize/scaled-copy producer: thread = 4 tokens x 8 channels."""
            w = warp - W_PREP[0]
            I = w // 2
            half = w % 2
            cg8 = lane % 8
            j4 = lane // 8
            d0 = (half * 8 + cg8) * 8
            tl0 = I * 16 + 4 * j4
            th = I // 2

            cX = txl.alloc_local((16,), "float32")
            cY = txl.alloc_local((16,), "float32")
            sqx = txl.alloc_local((4,), "float32")
            sqy = txl.alloc_local((4,), "float32")
            qraw = txl.alloc_local((4,), "uint32")
            kraw = txl.alloc_local((4,), "uint32")
            ra = txl.local_scalar("uint32")
            rb = txl.local_scalar("uint32")

            # roff[i]: byte offset of this thread's row tl0 + i (channels d0..d0+7) in a [64][128] tile; in the [128][128] kq tile the k row sits at roff + half * 8 KB + th * 4 KB (same swizzle) and the q row KQ_Q_OFF further
            sh0 = txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(g_t[0].ptr_to(0, 0)))
            roff = [
                txl.local_scalar("uint32", init=txl.cuda.cvta_generic_to_shared(g_t[0].ptr_to(tl0 + i, d0)) - sh0)
                for i in range(4)
            ]
            KQ_K_OFF = txl.local_scalar("uint32", init=txl.Cast("uint32", half * 8192 + th * KQ_Q_OFF))
            KQ_QQ_OFF = txl.local_scalar("uint32", init=KQ_K_OFF + txl.uint32(KQ_Q_OFF))

            def tptr_off(view, r):
                return txl.ptx.addr(view.ptr_to(0, 0), r)

            RINT_C, RINT_BITS = 12582912.0, 0x4B400000

            def add_rint_c(x):
                t = txl.local_scalar("float32")
                txl.ptx.add.rn.f32(t, txl.local_scalar("float32", init=x), txl.float32(RINT_C))
                return t

            def rint(x):
                r = txl.local_scalar("float32")
                txl.ptx.sub.rn.f32(r, add_rint_c(x), txl.float32(RINT_C))
                return r

            def shfl_xor(x, xr):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.bfly.b32(peer, txl.reinterpret("uint32", x), txl.uint32(xr), txl.uint32(31), txl.uint32(0xFFFFFFFF))
                return txl.reinterpret("float32", peer)

            def shfl_up(x, delta):
                peer = txl.local_scalar("uint32")
                txl.ptx.shfl_sync.up.b32(peer, txl.reinterpret("uint32", x), txl.uint32(delta), txl.uint32(0), txl.uint32(0xFFFFFFFF))
                return txl.reinterpret("float32", peer)

            def ld4u(dst, off, ptr):
                txl.ptx.ld.shared.v4.b32(dst[off], dst[off + 1], dst[off + 2], dst[off + 3], ptr)

            def st4u(ptr, src, off):
                txl.ptx.st.shared.v4.b32(ptr, src[off], src[off + 1], src[off + 2], src[off + 3])

            def rcp(x):
                r = txl.local_scalar("float32")
                txl.ptx.rcp.approx.ftz.f32(r, x)
                return r

            def fadd2f(a0, a1, b0, b1):
                """Two independent RN fp32 additions in one packed instruction."""
                packed = txl.local_scalar("uint64")
                o0 = txl.local_scalar("float32")
                o1 = txl.local_scalar("float32")
                txl.ptx.add.rn.f32x2(
                    packed,
                    txl.cuda.make_float2(a0, a1),
                    txl.cuda.make_float2(b0, b1),
                )
                txl.ptx.mov.b64(o0, o1, packed)
                return o0, o1

            def ffma2f(a0, a1, b0, b1, c0, c1):
                """Two independent RN fp32 FMAs in one packed instruction."""
                packed = txl.local_scalar("uint64")
                o0 = txl.local_scalar("float32")
                o1 = txl.local_scalar("float32")
                txl.ptx.fma.rn.f32x2(
                    packed,
                    txl.cuda.make_float2(a0, a1),
                    txl.cuda.make_float2(b0, b1),
                    txl.cuda.make_float2(c0, c1),
                )
                txl.ptx.mov.b64(o0, o1, packed)
                return o0, o1

            adt2 = txl.alloc_local((8,), "float32")
            a_e2 = txl.local_scalar("float32")

            def head_consts(hh):
                """Per-head gate constants (exp(A_log)/2 and its dt_bias products) -> adt_s -> registers."""
                with txl.If(I == 0), txl.Then():
                    a_h = txl.local_scalar("float32")
                    txl.ptx.ld.global_.f32(a_h, A_log.ptr_to([hh]))
                    a_e2v = txl.local_scalar("float32", init=ex2(a_h * txl.float32(RCP_LN2)) * txl.float32(0.5))
                    with txl.If(j4 == 0), txl.Then():
                        dtb = txl.alloc_local((8,), "float32")
                        txl.ptx.ld.global_.v4.f32(dtb[0], dtb[1], dtb[2], dtb[3], dt_bias.ptr_to([hh * D + d0]))
                        txl.ptx.ld.global_.v4.f32(dtb[4], dtb[5], dtb[6], dtb[7], dt_bias.ptr_to([hh * D + d0 + 4]))
                        txl.ptx.st.shared.v4.f32(adt_s.ptr_to([d0]), a_e2v * dtb[0], a_e2v * dtb[1], a_e2v * dtb[2], a_e2v * dtb[3])
                        txl.ptx.st.shared.v4.f32(adt_s.ptr_to([d0 + 4]), a_e2v * dtb[4], a_e2v * dtb[5], a_e2v * dtb[6], a_e2v * dtb[7])
                    with txl.If(tvm.tirx.all(half == 0, lane == 0)), txl.Then():
                        txl.ptx.st.shared.f32(adt_s.ptr_to([128]), a_e2v)
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                txl.ptx.ld.shared.v4.f32(adt2[0], adt2[1], adt2[2], adt2[3], adt_s.ptr_to([d0]))
                txl.ptx.ld.shared.v4.f32(adt2[4], adt2[5], adt2[6], adt2[7], adt_s.ptr_to([d0 + 4]))
                txl.ptx.ld.shared.f32(a_e2, adt_s.ptr_to([128]))

            n_items = n_items_local()
            with txl.serial(n_items, unroll=False) as c:
                s = c % 2
                sb = c % 3
                iw = item_word(c)
                iw1 = item_word1(c)
                hcur = txl.local_scalar("int32", init=item_head(iw1))
                nvalid = txl.local_scalar("int32", init=item_nvalid(iw))

                with txl.If(item_first(iw)), txl.Then():
                    head_consts(hcur)
                fwait(ring_full, s, (c // 2) % 2, "pr-wait-ring")
                tok_pa = irange("pr-phaseA")
                kqs, gs_ = kq_t[s], g_t[s]

                g8 = txl.alloc_local((4,), "uint32")
                for m in range(8):
                    txl.assign(cY[8 + m], txl.float32(0.0))
                txl.assign(ra, roff[0])
                txl.assign(rb, roff[1])

                with txl.serial(2, unroll=False) as it:
                    tokA = tl0 + 2 * it
                    for a in range(2):
                        r = ra if a == 0 else rb
                        va = tokA + a < nvalid
                        ld4u(g8, 0, tptr_off(gs_, r))
                        ld4u(qraw, 0, tptr_off(kqs, r + KQ_QQ_OFF))
                        ld4u(kraw, 0, tptr_off(kqs, r + KQ_K_OFF))
                        sqa = txl.local_scalar("uint32", init=txl.uint32(0))
                        ska = txl.local_scalar("uint32", init=txl.uint32(0))
                        for p in range(4):
                            g0, g1 = unpack(g8[p])
                            x0, x1 = ffma2f(g0, g1, a_e2, a_e2, adt2[2 * p], adt2[2 * p + 1])
                            th0 = txl.local_scalar("float32")
                            th1 = txl.local_scalar("float32")
                            txl.ptx.tanh.approx.f32(th0, x0)
                            txl.ptx.tanh.approx.f32(th1, x1)
                            gl0, gl1 = ffma2f(
                                th0,
                                th1,
                                txl.float32(GATE_C),
                                txl.float32(GATE_C),
                                txl.float32(GATE_C),
                                txl.float32(GATE_C),
                            )
                            gl0 = txl.local_scalar("float32", init=txl.Select(va, gl0, txl.float32(0.0)))
                            gl1 = txl.local_scalar("float32", init=txl.Select(va, gl1, txl.float32(0.0)))

                            gl0, gl1 = fadd2f(cY[8 * (1 - a) + 2 * p], cY[8 * (1 - a) + 2 * p + 1], gl0, gl1)
                            txl.assign(cY[a * 8 + 2 * p], gl0)
                            txl.assign(cY[a * 8 + 2 * p + 1], gl1)
                            txl.assign(sqa, hfma2(qraw[p], qraw[p], sqa))
                            txl.assign(ska, hfma2(kraw[p], kraw[p], ska))
                        q0, q1 = unpack(sqa)
                        k0, k1 = unpack(ska)
                        txl.assign(sqy[a], q0 + q1)
                        txl.assign(sqy[2 + a], k0 + k1)
                    with txl.If(it == 0), txl.Then():
                        for m in range(16):
                            txl.assign(cX[m], cY[m])
                        for m in range(4):
                            txl.assign(sqx[m], sqy[m])
                        txl.assign(ra, roff[2])
                        txl.assign(rb, roff[3])

                xin = [txl.local_scalar("float32", init=cY[8 + m]) for m in range(8)]
                for step in (1, 2):
                    for m in range(8):
                        y = shfl_up(xin[m], 8 * step)
                        txl.assign(xin[m], txl.Select(j4 >= step, xin[m] + y, xin[m]))
                with txl.If(j4 == 3), txl.Then():
                    txl.ptx.st.shared.v4.f32(tot.ptr_to([I * 128 + d0]), xin[0], xin[1], xin[2], xin[3])
                    txl.ptx.st.shared.v4.f32(tot.ptr_to([I * 128 + d0 + 4]), xin[4], xin[5], xin[6], xin[7])
                excl = [txl.local_scalar("float32", init=xin[m] - cY[8 + m]) for m in range(8)]

                cur = [sqx[0], sqx[1], sqy[0], sqy[1], sqx[2], sqx[3], sqy[2], sqy[3]]
                for xr in (4, 2, 1):
                    nb = len(cur) // 2
                    hi = txl.Cast("bool", txl.bitwise_and(lane, txl.int32(xr)))
                    nxt = []
                    for m in range(nb):
                        send = txl.local_scalar("float32", init=txl.Select(hi, cur[m], cur[nb + m]))
                        keep = txl.Select(hi, cur[nb + m], cur[m])
                        nxt.append(txl.local_scalar("float32", init=keep + shfl_xor(send, xr)))
                    cur = nxt
                txl.ptx.st.shared.f32(rsq.ptr_to([s * 256 + (cg8 // 4) * 128 + half * 64 + tl0 + cg8 % 4]), cur[0])
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))

                iend(tok_pa)
                tok_pc = irange("pr-consts")

                cp = half * 32 + lane
                tt = txl.alloc_local((8,), "float32")
                for J in range(4):
                    txl.ptx.ld.shared.v2.f32(tt[2 * J], tt[2 * J + 1], tot.ptr_to([J * 128 + 2 * cp]))
                # adjacent channels share the prefix/rounding chain: packed f32x2 ops keep each channel's operation order
                pref = [[txl.float32(0.0)] + [None] * 4 for _ in range(2)]
                for J in range(4):
                    pref[0][J + 1], pref[1][J + 1] = fadd2f(
                        pref[0][J], pref[1][J], tt[2 * J], tt[2 * J + 1])

                def rint2(x0, x1):
                    y0, y1 = fadd2f(x0, x1, txl.float32(RINT_C), txl.float32(RINT_C))
                    return fadd2f(y0, y1, txl.float32(-RINT_C), txl.float32(-RINT_C))

                e = [pref[0][4], pref[1][4]]
                eps_arg = fadd2f(e[0], e[1], txl.float32(CLAMP_F + 0.5), txl.float32(CLAMP_F + 0.5))
                eps_round = rint2(*eps_arg)
                eps = [txl.local_scalar("float32", init=txl.min(x, txl.float32(0.0))) for x in eps_round]
                e_cl = fadd2f(e[0], e[1], -eps[0], -eps[1])
                r0 = rint2(pref[0][2] * txl.float32(0.5), pref[1][2] * txl.float32(0.5))
                ref = [txl.local_scalar("float32", init=txl.Select(th == 0, r0[m], eps[m])) for m in range(2)]
                r = [pref[m][3] for m in range(2)]
                for m in range(2):
                    for J in (2, 1, 0):
                        r[m] = txl.Select(I == J, pref[m][J], r[m])
                lane_off = fadd2f(r[0], r[1], -ref[0], -ref[1])
                ca = [txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0))) for x in ref]
                cb_arg = fadd2f(ref[0], ref[1], -ca[0], -ca[1])
                cb = [txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0))) for x in cb_arg]
                dx = fadd2f(eps[0], eps[1], -r0[0], -r0[1])
                da = [txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0))) for x in dx]
                db_arg = fadd2f(dx[0], dx[1], -da[0], -da[1])
                db = [txl.local_scalar("float32", init=txl.max(x, txl.float32(-126.0))) for x in db_arg]
                scal = [ca, cb, da, db]
                lane_c = txl.alloc_local((4,), "uint32")
                for z in range(4):
                    nr = fadd2f(scal[z][0], scal[z][1], txl.float32(RINT_C), txl.float32(RINT_C))
                    power = [txl.reinterpret("float32", txl.shift_left(
                        txl.reinterpret("uint32", nr[m]) - txl.uint32(RINT_BITS - 127), txl.uint32(23))) for m in range(2)]
                    txl.assign(lane_c[z], bf16x2(power[0], power[1]))
                # the q2/k2 clamp product cc = 2^ca * 2^cb is per-channel, so it is formed once and broadcast; the kA factors 2^da, 2^db stay separate (their fused product can underflow where the staged one does not)
                cc_lane = txl.local_scalar("uint32", init=hmul2(lane_c[0], lane_c[1]))
                lane_b = [cc_lane, lane_c[2], lane_c[3]]

                with txl.If(I == 0), txl.Then():
                    with txl.If(c >= 3), txl.Then():
                        fwait(dvec_free, c % 3, ((c - 3) // 3) % 2, "pr-wait-dvecfree")
                    txl.ptx.st.shared.v2.f32(dvec.ptr_to([(c % 3) * 128 + 2 * cp]), ex2(e_cl[0]), ex2(e_cl[1]))
                    warrive(dvec_ready, c % 3)

                cst4 = txl.alloc_local((12,), "uint32")
                for p in range(4):
                    src_lane = txl.Cast("uint32", 4 * cg8 + p)
                    for m in range(2):
                        v = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.idx.b32(
                            v, txl.reinterpret("uint32", lane_off[m]), src_lane, txl.uint32(31), txl.uint32(0xFFFFFFFF)
                        )
                        txl.assign(excl[2 * p + m], excl[2 * p + m] + txl.reinterpret("float32", v))
                    for z in range(3):
                        v = txl.local_scalar("uint32")
                        txl.ptx.shfl_sync.idx.b32(v, lane_b[z], src_lane, txl.uint32(31), txl.uint32(0xFFFFFFFF))
                        txl.assign(cst4[3 * p + z], v)

                # all tot reads finish before reuse, and the helper's packed norms and beta are visible before phase B publishes rfull
                txl.ptx.bar.sync(txl.uint32(NB_PREP), txl.uint32(288))
                iend(tok_pc)

                with txl.If(c >= 1), txl.Then():
                    fwait(aa_r, (c + 1) % 2, ((c - 1) // 2) % 2, "pr-wait-aar")
                    fwait(oi_done, 0, (c + 1) % 2, "pr-wait-oidone")
                    txl.ptx.fence.proxy.async_.shared__cta()
                tok_pb = irange("pr-phaseB")
                Ex = txl.alloc_local((4,), "uint32")
                Fp = txl.alloc_local((4,), "uint32")
                xq = txl.alloc_local((4,), "uint32")
                xk = txl.alloc_local((4,), "uint32")
                q2v = txl.alloc_local((4,), "uint32")
                k2v = txl.alloc_local((4,), "uint32")
                kf = txl.alloc_local((4,), "uint32")
                kaa = txl.alloc_local((4,), "uint32")
                rq2 = txl.alloc_local((2,), "uint32")
                rk2 = txl.alloc_local((2,), "uint32")

                txl.assign(ra, roff[0])
                txl.assign(rb, roff[1])
                with txl.serial(2, unroll=False) as it:
                    tokA = tl0 + 2 * it
                    txl.ptx.ld.shared.v2.b32(rq2[0], rq2[1], rsq.ptr_to([s * 256 + tokA]))
                    txl.ptx.ld.shared.v2.b32(rk2[0], rk2[1], rsq.ptr_to([s * 256 + 128 + tokA]))
                    for a in range(2):
                        r = ra if a == 0 else rb
                        ld4u(qraw, 0, tptr_off(kqs, r + KQ_QQ_OFF))
                        ld4u(kraw, 0, tptr_off(kqs, r + KQ_K_OFF))
                        for p in range(4):
                            ex0 = ex2(cX[a * 8 + 2 * p] + excl[2 * p])
                            ex1 = ex2(cX[a * 8 + 2 * p + 1] + excl[2 * p + 1])
                            txl.assign(Ex[p], bf16x2(ex0, ex1))
                            txl.assign(Fp[p], hrcp2(Ex[p]))
                        for p in range(4):
                            qn = hmul2(qraw[p], rq2[a])
                            kn = hmul2(kraw[p], rk2[a])
                            txl.assign(xq[p], hmul2(qn, Ex[p]))
                            txl.assign(xk[p], hmul2(kn, Ex[p]))
                            txl.assign(kf[p], hmul2(kn, Fp[p]))
                            # q2 and k2 share the fused power-of-two clamp product cc = cst4[3 p]
                            txl.assign(q2v[p], hmul2(xq[p], cst4[3 * p]))
                            txl.assign(k2v[p], hmul2(xk[p], cst4[3 * p]))
                        st4u(tptr_off(q2t, r), q2v, 0)
                        st4u(tptr_off(k2t, r), k2v, 0)

                        st4u(tptr_off(kqs, r + KQ_K_OFF), xk, 0)
                        st4u(tptr_off(kqs, r + KQ_QQ_OFF), xq, 0)
                        with txl.If(th == 0):
                            with txl.Then():
                                for p in range(4):
                                    txl.assign(kaa[p], hmul2(hmul2(kf[p], cst4[3 * p + 1]), cst4[3 * p + 2]))
                                st4u(tptr_off(g_t[s], r), kf, 0)
                                st4u(tptr_off(kA[s], r), kaa, 0)
                            with txl.Else():
                                st4u(tptr_off(kA[s], r), kf, 0)
                    with txl.If(it == 0), txl.Then():
                        for m in range(16):
                            txl.assign(cX[m], cY[m])
                        txl.assign(ra, roff[2])
                        txl.assign(rb, roff[3])
                txl.ptx.fence.proxy.async_.shared__cta()
                iend(tok_pb)
                warrive(rfull, s)

        with r_state:
            state_body()
        with wg1:
            with r_ld:
                load_body()
            with r_mmi:
                norm_body()
            with r_mmc:
                mmc_body()
            with r_rnd:
                rounds_body((0, 1))
        with r_intra:
            intra_body()
        with r_prep:
            prep_body()

        txl.cuda.cta_sync()
        with txl.If(warp == 0), txl.Then():
            txl.ptx[TMEM_RELINQ]()
            txl.ptx[TMEM_DEALLOC](tbase, txl.uint32(N_COLS))

    kda_fwd.__annotations__ = {
        "q_map": txl.TensorMap,
        "k_map": txl.TensorMap,
        "v_map": txl.TensorMap,
        "g_map": txl.TensorMap,
        "beta_map": txl.TensorMap,
        "o_map": txl.TensorMap,
        "out": txl.gptr[txl.bf16],
        "A_log": txl.gptr[txl.f32, (H,)],
        "dt_bias": txl.gptr[txl.f32, (H * D,)],
        "h0": txl.gptr[txl.f32],
        "final_state": txl.gptr[txl.f32],
        "hand": txl.gptr[txl.bf16] if bf16_handoff else txl.gptr[txl.f32],
        "flags": txl.gptr[txl.i32],
        "items": txl.gptr[txl.u32],
        "item_counts": txl.gptr[txl.i32],
        "num_ctas": txl.i32,
        "scale": txl.f32,
    }
    return txl.kernel(warps=NWARPS, arch="sm_100a", min_blocks_per_sm=1, grid="num_ctas")(kda_fwd)


class _TensorMap:
    __slots__ = ("_storage", "ptr")

    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


_TMAP_ENCODE = None


def _tmap_encode():
    global _TMAP_ENCODE
    if _TMAP_ENCODE is None:
        _TMAP_ENCODE = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    return _TMAP_ENCODE


def _encode_map(tensor, T, H, rows=64, slabs=2):
    """3D map over a [T, H, 128] bf16 tensor viewed as (64, T, 2H); box (64, rows, slabs), 128B swizzle."""
    m = _TensorMap()
    _tmap_encode()(
        m.ptr, "bfloat16", 3, ctypes.c_void_p(int(tensor.data_ptr())),
        64, T, 2 * H, H * 256, 128, 64, rows, slabs, 1, 1, 1, 0, 3, 2, 0,
    )
    return m


def _encode_beta_map(tensor, T, H):
    """2D map over the [T, H] bf16 beta logits: dims (H, T), box (8 heads, 64 tokens)."""
    m = _TensorMap()
    _tmap_encode()(
        m.ptr, "bfloat16", 2, ctypes.c_void_p(int(tensor.data_ptr())),
        H, T, H * 2, 8, 64, 1, 1, 0, 0, 2, 0,
    )
    return m


_KERNELS = {}


def _tok_bits(tok0):
    return (tok0 & 0xFFFF) | ((tok0 >> 16) << 23)


def _host_item_table(cu_list, H, num_ctas, force_lpt):
    """Per-CTA (word0, word1) item lists.

    Sequences are ranked by length (ties by index).  When the longest sequence is short against the mean load
    the (head, sequence, chunk) walk is cut into num_ctas equal-cost ranges (chunk cost 1, BETA per sequence
    start, cuts snapped to a sequence end within SNAP chunks; a range starting inside a sequence continues
    the state handed off by the previous CTA); otherwise whole (sequence, head) items go to the least loaded
    CTA group (LPT).  word0 = tok0[15:0] | nvalid << 16 | tok0[20:16] << 23 | dst_hand << 28 | src_hand << 29 | first << 30 | last << 31,
    word1 = seq | head << 16."""
    nseq = len(cu_list) - 1
    start = [int(cu_list[n]) for n in range(nseq)]
    length = [int(cu_list[n + 1] - cu_list[n]) for n in range(nseq)]
    order = [0] * nseq
    for n in range(nseq):
        rank = sum(1 for m in range(nseq) if (length[m] > length[n]) or (length[m] == length[n] and m < n))
        order[rank] = n
    wh, maxnch, off, nch = 0, 0, [], []
    for r in range(nseq):
        c_ = max((length[order[r]] + BT - 1) // BT, 1)
        off.append(wh)
        nch.append(c_)
        wh += c_
        maxnch = max(maxnch, c_)
    off.append(wh)
    ch_cost = wh + BETA * nseq
    c_tot = ch_cost * H
    l_min = c_tot // num_ctas
    items = [[] for _ in range(num_ctas)]
    if maxnch <= l_min - (2 * SNAP + BETA) and not force_lpt:
        # cut positions minimising the maximum modelled range cost (chunk 1, unit start or continuation START_COST, handoff HANDOFF_COST)
        cuts = _optimal_cuts(nch, H, num_ctas)

        def cut_pos(p):
            return cuts[min(max(p, 0), num_ctas)]

        for cta in range(num_ctas):
            lo, hi = cut_pos(cta), cut_pos(cta + 1)
            for pas in range(3):
                for hh in range(lo // wh, (hi - 1) // wh + 1):
                    for rk in range(nseq):
                        c_ = nch[rk]
                        ib = hh * wh + off[rk]
                        ie = ib + c_
                        cb, ce = max(0, lo - ib), min(c_, hi - ib)
                        cont_in, head_out = cb > 0, ce < c_
                        kind = 2 if cont_in else (0 if head_out else 1)
                        if ie > lo and ib < hi and kind == pas:
                            n = order[rk]
                            s_tok, e_tok = start[n], start[n] + length[n]
                            for ch in range(cb, ce):
                                tok0 = s_tok + ch * BT
                                nvalid = max(min(e_tok - tok0, BT), 0)
                                word0 = (_tok_bits(tok0) | (nvalid << 16) | ((1 << 30) if ch == cb else 0) | ((1 << 31) if ch + 1 == ce else 0)
                                         | ((1 << 29) if (ch == cb and cont_in) else 0) | ((1 << 28) if (ch + 1 == ce and head_out) else 0))
                                items[cta].append((word0 & 0xFFFFFFFF, n | (hh << 16)))
    else:
        assert num_ctas >= H, "whole-item scheduling needs at least one CTA per head"
        # unit-level LPT: every (sequence, head) chain is one unit of cost chunks + START_COST, placed longest-first on the least loaded CTA, then a move/swap local search on the most loaded CTA
        units = sorted(((nch[r] + START_COST, r, hh) for r in range(nseq) for hh in range(H)), key=lambda u: (-u[0], u[1], u[2]))
        per = [[] for _ in range(num_ctas)]
        load = [0.0] * num_ctas
        heap = [(0.0, c) for c in range(num_ctas)]
        for cost, r, hh in units:
            l, c = heapq.heappop(heap)
            per[c].append((r, hh))
            load[c] = l + cost
            heapq.heappush(heap, (load[c], c))
        _balance_units(per, load, nch, START_COST)
        for cta in range(num_ctas):
            for r, hh in per[cta]:
                n = order[r]
                s_tok, ln = start[n], length[n]
                for ch in range(nch[r]):
                    tok0 = s_tok + ch * BT
                    nvalid = max(min(s_tok + ln - tok0, BT), 0)
                    word0 = _tok_bits(tok0) | (nvalid << 16) | ((1 << 30) if ch == 0 else 0) | ((1 << 31) if ch + 1 == nch[r] else 0)
                    items[cta].append((word0 & 0xFFFFFFFF, n | (hh << 16)))
    return items


def _optimal_cuts(nch, H, num_ctas):
    """num_ctas + 1 cut positions in the (head, sequence-rank, chunk) walk minimising the maximum range cost."""
    walk = []
    for _ in range(H):
        for n in nch:
            for ch in range(n):
                walk.append((ch == 0, ch == n - 1))
    total = len(walk)

    def greedy(bound):
        cuts, i = [0], 0
        while i < total:
            cost, j, best_j, boundary = START_COST, i, None, False
            while j < total:
                c = cost + 1.0 + (START_COST if (j > i and walk[j][0]) else 0.0)
                if c + (0.0 if walk[j][1] else HANDOFF_COST) > bound:
                    break
                cost = c
                boundary = boundary or walk[j][1]
                if boundary:
                    best_j = j
                j += 1
            if best_j is None:
                return None
            i = best_j + 1
            cuts.append(i)
            if len(cuts) - 1 > num_ctas:
                return None
        return cuts

    lo, hi, best = 0.0, float(total) + 2 * START_COST + HANDOFF_COST, None
    for _ in range(50):
        mid = (lo + hi) / 2
        cuts = greedy(mid)
        if cuts is not None:
            hi, best = mid, cuts
        else:
            lo = mid
    if best is None:  # cannot happen (bound = total cost is always feasible), kept for safety
        best = [0, total]

    def range_cost(a, b):
        return START_COST + (b - a) + START_COST * sum(1 for q in range(a + 1, b) if walk[q][0]) + (0.0 if walk[b - 1][1] else HANDOFF_COST)

    # give every CTA work: split the costliest ranges after a unit's last chunk nearest their middle
    while len(best) - 1 < num_ctas:
        done = False
        for _, i in sorted(((range_cost(best[i], best[i + 1]), i) for i in range(len(best) - 1)), reverse=True):
            a, b = best[i], best[i + 1]
            cands = [q + 1 for q in range(a, b - 1) if walk[q][1]]
            if cands:
                mid = (a + b) // 2
                best = best[: i + 1] + [min(cands, key=lambda x: abs(x - mid))] + best[i + 1:]
                done = True
                break
        if not done:
            best = best + [total] * (num_ctas - (len(best) - 1))  # remaining CTAs get empty ranges
            break
    return best


def _balance_units(per, load, nch, start_cost, max_rounds=400):
    """Lower the makespan of a unit assignment: move one unit off the most loaded CTA, or swap it with a smaller
    unit of another CTA, whenever that strictly lowers the maximum load; stop when no such step exists."""
    num_ctas = len(per)
    ucost = lambda r: nch[r] + start_cost  # noqa: E731
    for _ in range(max_rounds):
        worst = max(range(num_ctas), key=load.__getitem__)
        wl = load[worst]
        best = None
        for ui, (r, _hh) in enumerate(per[worst]):
            cu = ucost(r)
            for c in range(num_ctas):
                if c == worst:
                    continue
                nl = load[c] + cu
                if nl < wl and (best is None or nl < best[0]):
                    best = (nl, ui, c, -1)
                for vi, (r2, _h2) in enumerate(per[c]):
                    cv = ucost(r2)
                    if cv < cu:
                        top = max(wl - cu + cv, load[c] - cv + cu)
                        if top < wl and (best is None or top < best[0]):
                            best = (top, ui, c, vi)
        if best is None:
            return
        _, ui, c, vi = best
        u = per[worst][ui]
        if vi < 0:
            per[worst].pop(ui)
            per[c].append(u)
            load[worst] -= ucost(u[0])
            load[c] += ucost(u[0])
        else:
            v = per[c][vi]
            per[worst][ui], per[c][vi] = v, u
            load[worst] += ucost(v[0]) - ucost(u[0])
            load[c] += ucost(u[0]) - ucost(v[0])


def _get_kernel(H, intra_unroll, bf16_handoff, force_lpt, max_items=MAX_ITEMS):
    key = (H, intra_unroll, bf16_handoff, force_lpt, max_items)
    if key not in _KERNELS:
        kernel = build_kernel(H, intra_unroll=intra_unroll, bf16_handoff=bf16_handoff, max_items=max_items)
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
        with target:
            _KERNELS[key] = tvm.compile(kernel.mod, target=target, tir_pipeline="tirx")
    return _KERNELS[key]


def fused_setup(data, B, T, H):
    q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
    A_log, dt_bias, scale = data["A_log"], data["dt_bias"], float(data["scale"])
    h0, out, final_state = data["initial_state"], data["output"], data["final_state"]
    cu = data.get("cu_seqlens")
    assert B == 1 and T > 0 and H % 8 == 0
    assert q.shape == (1, T, H, D) and q.dtype == torch.bfloat16
    for t in (q, k, v, g, beta, A_log, dt_bias, h0, out, final_state):
        assert t.is_contiguous() and t.is_cuda
    if cu is None:
        cu = torch.tensor([0, T], dtype=torch.int64, device=q.device)
    cu = cu.to(device=q.device, dtype=torch.int64).contiguous()
    nseq = int(cu.numel()) - 1
    assert 1 <= nseq < 65536
    assert h0.shape == (nseq, H, D, D) and h0.dtype == torch.float32
    assert final_state.shape == (nseq, H, D, D) and final_state.dtype == torch.float32
    assert out.shape == (1, T, H, D) and out.dtype == torch.bfloat16

    # token offsets are packed into 21 bits of the item word; element offsets are int32
    assert T < (1 << 21) and T * H * D < (1 << 31)
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_ctas = max(1, min(sm_count, H * nseq))

    force_lpt = H == 96 and nseq == 6
    bf16_handoff = H == 64 and nseq in (6, 8)
    hand_dtype = torch.bfloat16 if bf16_handoff else torch.float32
    hand = torch.empty(sm_count * D * D, dtype=hand_dtype, device=q.device)
    flags = torch.zeros(sm_count + 1, dtype=torch.int32, device=q.device)
    lists = _host_item_table(cu.tolist(), H, num_ctas, force_lpt)
    # the per-CTA item table lives in SMEM; schedules longer than the default table get a larger one
    max_items = max(MAX_ITEMS, -(-max(len(x) for x in lists) // 32) * 32)
    items_np = torch.zeros(num_ctas, max_items, 2, dtype=torch.int64)
    for c_, lst in enumerate(lists):
        if lst:
            items_np[c_, : len(lst)] = torch.tensor(lst, dtype=torch.int64)
    items_g = items_np.reshape(-1).to(torch.int32) if items_np.max() < 2**31 else (items_np.reshape(-1) - (items_np.reshape(-1) >= 2**31).to(torch.int64) * 2**32).to(torch.int32)
    items_g = items_g.view(torch.uint32).contiguous().to(q.device)
    item_counts = torch.tensor([len(x) for x in lists], dtype=torch.int32, device=q.device)

    intra_unroll = H == 64 and nseq == 1
    ex = _get_kernel(H, intra_unroll, bf16_handoff, force_lpt, max_items)
    maps = [_encode_map(t, T, H, rows=32, slabs=1) for t in (q, k)] + [_encode_map(t, T, H) for t in (v, g)] + [_encode_beta_map(beta, T, H), _encode_map(out, T, H, rows=32)]
    a_log = A_log.contiguous()
    dt_flat = dt_bias.reshape(-1)
    h0_flat = h0.reshape(-1)
    fs_flat = final_state.reshape(-1)
    out_flat = out.reshape(-1)

    def run():
        ex(*[m.ptr for m in maps], out_flat, a_log, dt_flat, h0_flat, fs_flat, hand, flags, items_g, item_counts, num_ctas, scale)

    run._keep = (maps, cu, hand, flags, items_g, item_counts)
    run()
    torch.cuda.synchronize()
    return run


_CANON_SRC = r'''
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""KDA forward (B=1, K=V=128, bf16) -- split front-end family (two kernels); the front end uses no TMA stores.

Family: split-frontend.  The chunked recurrence (C=32 tokens, two 16-token sub-blocks) is cut into
its state-independent front end and its state-dependent chain, which run as two kernels:

  K1 kda_front  persistent over all SMs, one work item = (chunk c, head h), item id = c*H + h.
     Per item: gates (gd2, block-local products, chunk decay), operand tiles (X, QX, Kt, Kbar, Qt,
     A-operand rows), the two A chains on tcgen05 (Akk^T, Aqk^T), k-norms from the Akk diagonal,
     L = diag(b*kn) Akk diag(kn), the hierarchical 32x32 inverse (mma.sync), T1/T2, W1 = Kt^T T2^T
     (tcgen05), the Aqk tile and the q norms.  Results go to global scratch as bf16 tiles with the
     exact SMEM layouts the chain's MMAs consume (TMA stores): Kbar [j][k], Qt [i][k] (SW128B),
     T1 [i][j], AqkT [j][i], W1b [k][i] (SW64B), plus Dvec[128] and qn[32] (fp32).
  K2 kda_chain  one persistent CTA per head walks the chunks: TMA ring of the precomputed tiles
     (+ raw V), state ST[v][k] fp32 in TMEM, bf16 snapshot + decay split over two warpgroups,
     U'^T = V^T T1^T - STb W1b, ST += U'^T Kbar, O^T = STb Qt^T + U'^T AqkT, output epilogue.

Why: in the fused persistent-head kernel the chain's true per-chunk dependency waits on the front end
sharing its SM while other SMs idle.  Here the front end runs on every SM at full occupancy and the
chain kernel is a pure dependency loop with all operands staged ahead by TMA.

Math: identical to the fused kernel.

Timing note: the two kernels run concurrently, so the pair must be measured on a wall clock, not as a
sum of per-kernel durations.
"""

import ctypes
import os

import torch
import tvm
import tvm_ffi
from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

import tirx_kernels.tirx_lite as txl

D = 128
C = 32
NB = 2
STAGES1 = 4
GSTAGES = 2
NSLOT = 8
TMEM1_COLS = 512
TMEM2_COLS = 512
LOG2E = 1.4426950408889634
SIG_C1 = -2.5 * LOG2E
VEC_F32 = 160
TILE_BYTES = C * D * 2
SMALL_BYTES = C * C * 2


TM_A = 0
TM_A_STRIDE = 128
TM_W1 = 256
NB_ROWS = 80
NPROD = 4
W1_LAG = 3

ROW_Y0 = 0
ROW_Z1 = 16
ROW_Z0 = 32
ROW_QZ0 = 48
ROW_QZ1 = 64

MMA = "tcgen05.mma.cta_group::1.kind::f16"
IDESC_N32 = encode_instr_descriptor_dense_uint32(128, 32, 16, "float32", "bfloat16", "bfloat16", False, False)


IDESC_M64_N80 = encode_instr_descriptor_dense_uint32(64, 80, 16, "float32", "bfloat16", "bfloat16", False, False)
IDESC_N128_TB = encode_instr_descriptor_dense_uint32(128, 128, 16, "float32", "bfloat16", "bfloat16", False, True)
IDESC_N32_TA = encode_instr_descriptor_dense_uint32(128, 32, 16, "float32", "bfloat16", "bfloat16", True, False)
IDESC_N32_TB_NEG = encode_instr_descriptor_dense_uint32(128, 32, 16, "float32", "bfloat16", "bfloat16", False, True, neg_b=True)

TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TC_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TC_LD256_X4 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"
STM_X4T = "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
# tcgen05 accesses of different threads are ordered only through these fences: before_thread_sync after the last tcgen05.ld/st and before the arrive, after_thread_sync after the wait and before the next tcgen05 access
TC_FENCE_BEFORE = "tcgen05.fence::before_thread_sync"
TC_FENCE_AFTER = "tcgen05.fence::after_thread_sync"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
TMA_G2S = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
TMA_G2S_HINT = TMA_G2S + ".L2::cache_hint"
TMA_S2G = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
MMA_K8 = "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"
MMA_K16 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
LDM_X1 = "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
LDM_X1T = "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
LDM_X4 = "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
LDM_X4T = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
STM_X1 = "stmatrix.sync.aligned.m8n8.x1.shared.b16"
STM_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"


BAR_PREP = 1
BAR_PREP2 = 5
BAR_RDO = 3
BAR_TMEM = 4
BAR_TMEM_N1 = 192


def _f2(a, b):
    return txl.cuda.make_float2(a, b)


def _pack_bf16x2(dst, lo, hi):
    txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)


def _bf16_lo(word):
    return txl.reinterpret("float32", word << txl.uint32(16))


def _bf16_hi(word):
    return txl.reinterpret("float32", word & txl.uint32(0xFFFF0000))


_IKET_KEEP = os.environ.get("IKET_KEEP")
_IKET_KEEP = set(_IKET_KEEP.split(",")) if _IKET_KEEP else None


def _rng(name):
    if _IKET_KEEP is not None and name not in _IKET_KEEP:
        return None
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def _rng_end(token):
    if token is None:
        return
    txl.cuda.iket.range_end(token[0])


def _chain(d, a, b, idesc, accumulate, pred):
    txl.idioms.mma_chain(MMA, d, a=a, b=b, idesc=idesc, pred=pred, accumulate=accumulate, guard="pred")


def _bid(bar_id):
    return txl.uint32(bar_id) if isinstance(bar_id, int) else bar_id


def _wg_arrive(bar, slot, bar_id, leader):
    """Whole-warpgroup handoff with one arrival: named barrier, then the leader thread arrives."""
    txl.ptx.bar.sync(_bid(bar_id), txl.uint32(128))
    with txl.If(leader), txl.Then():
        bar.arrive(slot)


def _taddr(tm, col, lane_bits):
    return txl.Cast("uint32", tm[0] + col + lane_bits)


SLOW_WAIT_NS = 200
# try_wait suspend hint (10 ms) for the front's off-critical-path roles: a waiter stays suspended until its barrier flips instead of re-polling
TRY_WAIT_HINT = 0x989680


def _slow_wait(bar, stage, parity, hint=TRY_WAIT_HINT):
    """mbarrier wait with a suspend hint, for waiters off the critical path (idle warps stop eating the busy warps' issue slots)."""
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    addr = bar.ptr_to([stage])
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, addr, txl.Cast("uint32", parity), txl.uint32(hint)
        )


PDONE_SLOTS = 16
SIG_BATCH = 4


def _signal_item(bar, li, leader):
    """A producer finished its global stores of item li: all of the warpgroup's / warp's stores precede the caller's
    bar.sync / warp_sync, the leader arrives on the item's completion barrier (the sig warp publishes it)."""
    with txl.If(leader), txl.Then():
        bar.arrive(li & (PDONE_SLOTS - 1))


def _tmem_preamble(s_tmem_addr, count):
    txl.ptx.bar.sync(txl.uint32(BAR_TMEM), txl.uint32(count))
    tm = txl.alloc_local([1], "int32")
    txl.ptx.ld.volatile.shared.s32(tm[0], txl.address_of(s_tmem_addr[0]))
    return tm


def make_front(H: int):
    """Front-end kernel for a fixed head count: item it = c*H + h.  All per-item outputs are written
    straight to global memory with coalesced generic stores (row-major tiles, read back by K2 with TMA).

    Raw (un-normalised) key tiles; both norms come out of the A chain (diagonals of
    X Z^T and QX QZ^T); the A chain is M=64 x N=80 with A = [X ; QX] and B = [Y0 ; Z1 ; Z0 ; QZ0 ; QZ1]
    so that every L row / Aqk row lands in ONE TMEM lane (Layout F); the norms are folded into
    T1' = diag(kn) T diag(b) and T2' = T1' diag(kn); one warp per item inverts (four solvers)."""

    # H64 benefits from amortizing publication over eight completed items.
    # Keep four for H96, where more frequent publication measures faster.
    signal_batch = 8 if H == 64 else SIG_BATCH

    @txl.kernel(warps=20, arch="sm_100a", min_blocks_per_sm=1, grid="num_ctas")
    def kda_front(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        g: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        a_log: txl.gptr[txl.f32],
        dt_bias: txl.gptr[txl.f32],
        vec: txl.gptr[txl.f32],
        kbar_g: txl.gptr[txl.bf16],
        qt_g: txl.gptr[txl.bf16],
        t1_g: txl.gptr[txl.bf16],
        aqk_g: txl.gptr[txl.bf16],
        w1_g: txl.gptr[txl.bf16],
        flags: txl.gptr[txl.i32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        g_map: txl.TensorMap,
        item_base: txl.i32,
        num_items: txl.i32,
        num_ctas: txl.i32,
        items_per_cta: txl.i32,
        do_signal: txl.i32,
    ):
        for buf in (q, k, g):
            txl.keep_alive(buf.ptr_to([0]))

        cta = txl.cta_id()
        lane = txl.lane_id()
        tid = txl.thread_id()
        last_it = cta + (items_per_cta - 1) * num_ctas
        n_cta = txl.Select(last_it < num_items, items_per_cta, items_per_cta - 1)

        sp = txl.specialize()

        prep = sp.role("prep", warps=list(range(8)), regs=128)
        rdo = sp.role("rdo", warps=[8, 9, 10, 11], regs=96)
        solve = sp.role("solve", warps=[12, 13, 14, 15], regs=88)
        mma2 = sp.role("mma2", warps=[16], regs=40)
        mma3 = sp.role("mma3", warps=[17], regs=40)
        tma = sp.role("tma", warps=[18], regs=40)
        aux = sp.role("aux", warps=[19], regs=40)

        smem = txl.smem_pool()
        s_tmem_addr = smem.alloc((1,), txl.i32, align=4)
        p_raw = txl.Pipeline(smem, STAGES1, full="tma", empty="mbar", init_empty=1)
        p_g = txl.Pipeline(smem, GSTAGES, full="tma", empty="mbar", init_empty=1)
        m_tiles = txl.MBarrier(smem, 2)
        m_akk = txl.TCGen05Bar(smem, 2)
        m_afree = txl.MBarrier(smem, 2)
        m_w1 = txl.TCGen05Bar(smem, 4)
        m_w1free = txl.MBarrier(smem, 4)
        m_beta = txl.MBarrier(smem, NSLOT)
        m_gates0 = txl.MBarrier(smem, 1)
        m_bdone = txl.MBarrier(smem, NSLOT)
        m_lready = txl.MBarrier(smem, 4)
        m_lfree = txl.MBarrier(smem, 4)
        m_tp = txl.MBarrier(smem, 4)
        m_pdone = txl.MBarrier(smem, PDONE_SLOTS)
        m_sfree = txl.MBarrier(smem, PDONE_SLOTS)
        for bar, cnt in (
            (m_tiles, 1), (m_akk, 1), (m_afree, 1), (m_w1, 1), (m_w1free, 1), (m_beta, 32), (m_gates0, 1),
            (m_bdone, 1), (m_lready, 1), (m_lfree, 1), (m_tp, 1), (m_pdone, NPROD), (m_sfree, 1),
        ):
            bar.init(cnt)

        s_q = smem.alloc((STAGES1, C, D), txl.bf16, swizzle=txl.SW128B)
        s_k = smem.alloc((STAGES1, C, D), txl.bf16, swizzle=txl.SW128B)
        s_g = smem.alloc((GSTAGES, C, D), txl.bf16, swizzle=txl.SW128B)
        s_a = smem.alloc((2, 2 * C, D), txl.bf16, swizzle=txl.SW128B)
        s_b = smem.alloc((2, NB_ROWS + 16, D), txl.bf16, swizzle=txl.SW128B)
        s_kt = smem.alloc((4, C, D), txl.bf16, swizzle=txl.SW128B)
        s_l = smem.alloc((4, C, C), txl.bf16, swizzle=txl.SW64B)
        s_t2 = smem.alloc((4, C, C), txl.bf16, swizzle=txl.SW64B)
        s_bt = smem.alloc((NSLOT, C), txl.f32, align=16)
        s_ha = smem.alloc((NSLOT, 4), txl.f32, align=16)
        s_hadt = smem.alloc((NSLOT, D), txl.f32, align=16)
        s_kn = smem.alloc((4, C), txl.f32, align=16)
        s_bblk = smem.alloc((4, NB, D), txl.f32, align=16)

        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elect():
            return txl.cuda.elect_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def elect_local():
            e = txl.local_scalar("uint32")
            txl.assign(e, txl.cuda.elect_sync())
            return e

        def item_of(li):
            return item_base + cta + li * num_ctas

        def it64(li):
            return txl.Cast("int64", item_of(li))

        def unpack(dst, i, word):
            txl.assign(dst[2 * i], _bf16_lo(word))
            txl.assign(dst[2 * i + 1], _bf16_hi(word))

        with prep:
            wid_all = txl.warp_id_in_role()
            parity = wid_all >> 2
            bar_id = txl.uint32(BAR_PREP) + txl.Cast("uint32", parity) * txl.uint32(BAR_PREP2 - BAR_PREP)
            wid = wid_all & 3
            blk = wid >> 1
            khalf = wid & 1
            grp = lane >> 4
            kq = lane & 15
            k0 = khalf * 64 + kq * 4
            k064 = txl.Cast("int64", k0)
            rowbase = blk * 16 + grp * 8
            tidp = txl.tid_in_role() & 127
            is_grp1 = grp == 1

            gd = txl.alloc_local([32], "float32")
            ff = txl.alloc_local([32], "float32")
            ew = txl.alloc_local([16], "uint32")
            fw = txl.alloc_local([16], "uint32")
            kws = txl.alloc_local([16], "uint32")
            qws = txl.alloc_local([16], "uint32")
            n_mine = (n_cta + 1 - parity) >> 1
            with txl.serial(n_mine) as ci:
                li = ci * 2 + parity
                it = item_of(li)
                itb = txl.Cast("int64", it)
                stage = txl.local_scalar("int32", init=li & (STAGES1 - 1))
                rphase = (li >> 2) & 1
                gstage = txl.local_scalar("int32", init=li & (GSTAGES - 1))
                slot8 = li & (NSLOT - 1)

                with txl.If((parity == 1) & (ci == 0)), txl.Then():
                    m_gates0.wait(0, 0)

                tk = _rng("prep-wait-consts")
                m_beta.wait(slot8, (li >> 3) & 1)
                _rng_end(tk)
                ha = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(ha, txl.address_of(s_ha[slot8, 0]))
                hadt = txl.alloc_local([4], "float32")
                txl.ptx["ld.shared.v4.f32"](hadt[0], hadt[1], hadt[2], hadt[3], txl.address_of(s_hadt[slot8, k0]))

                tk = _rng("prep-wait-raw")
                p_g.full.wait(gstage, (li >> 1) & 1)
                _rng_end(tk)
                gg = s_g[gstage]
                tk = _rng("prep-gates")
                gw = txl.alloc_local([2], "uint32")
                th = txl.alloc_local([4], "float32")
                xarg = txl.alloc_local([2], "uint64")
                gd2 = txl.alloc_local([2], "uint64")
                for t in range(8):
                    txl.ptx["ld.shared.v2.b32"](gw[0], gw[1], gg.ptr_to(rowbase + t, k0))
                    for p in range(2):
                        txl.ptx.fma.rn.f32x2(xarg[p], _f2(_bf16_lo(gw[p]), _bf16_hi(gw[p])), _f2(ha, ha),
                                             _f2(hadt[2 * p], hadt[2 * p + 1]))
                        txl.ptx.tanh.approx.f32(th[2 * p], txl.cuda.float2_x(xarg[p]))
                        txl.ptx.tanh.approx.f32(th[2 * p + 1], txl.cuda.float2_y(xarg[p]))
                    for p in range(2):
                        txl.ptx.fma.rn.f32x2(gd2[p], _f2(th[2 * p], th[2 * p + 1]), _f2(txl.float32(SIG_C1), txl.float32(SIG_C1)),
                                             _f2(txl.float32(SIG_C1), txl.float32(SIG_C1)))
                        txl.ptx.ex2.approx.ftz.f32(gd[4 * t + 2 * p], txl.cuda.float2_x(gd2[p]))
                        txl.ptx.ex2.approx.ftz.f32(gd[4 * t + 2 * p + 1], txl.cuda.float2_y(gd2[p]))

                acc = txl.alloc_local([2], "uint64")
                for p in range(2):
                    txl.assign(acc[p], _f2(txl.float32(1.0), txl.float32(1.0)))
                for t in range(7, -1, -1):
                    for p in range(2):
                        txl.assign(ff[4 * t + 2 * p], txl.cuda.float2_x(acc[p]))
                        txl.assign(ff[4 * t + 2 * p + 1], txl.cuda.float2_y(acc[p]))
                        if t > 0:
                            txl.ptx.mul.rn.f32x2(acc[p], acc[p], _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]))
                for t in range(1, 8):
                    for p in range(2):
                        txl.ptx.mul.rn.f32x2(acc[p], _f2(gd[4 * (t - 1) + 2 * p], gd[4 * (t - 1) + 2 * p + 1]),
                                             _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]))
                        txl.assign(gd[4 * t + 2 * p], txl.cuda.float2_x(acc[p]))
                        txl.assign(gd[4 * t + 2 * p + 1], txl.cuda.float2_y(acc[p]))

                oth = txl.alloc_local([4], "float32")
                for j in range(4):
                    tot_w = txl.local_scalar("uint32")
                    txl.ptx.shfl_sync.bfly.b32(tot_w, txl.reinterpret("uint32", gd[28 + j]), txl.uint32(16), txl.uint32(31),
                                               txl.uint32(0xFFFFFFFF))
                    txl.assign(oth[j], txl.reinterpret("float32", tot_w))
                efac = txl.alloc_local([4], "float32")
                ffac = txl.alloc_local([4], "float32")
                bb = txl.alloc_local([4], "float32")
                for j in range(4):
                    txl.assign(efac[j], txl.Select(is_grp1, oth[j], txl.float32(1.0)))
                    txl.assign(ffac[j], txl.Select(is_grp1, txl.float32(1.0), oth[j]))
                    txl.assign(bb[j], gd[28 + j] * oth[j])
                prod = txl.local_scalar("uint64")
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.f32x2(prod, _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]), _f2(efac[2 * p], efac[2 * p + 1]))
                        _pack_bf16x2(ew[2 * t + p], txl.cuda.float2_x(prod), txl.cuda.float2_y(prod))
                        txl.ptx.mul.rn.f32x2(prod, _f2(ff[4 * t + 2 * p], ff[4 * t + 2 * p + 1]), _f2(ffac[2 * p], ffac[2 * p + 1]))
                        _pack_bf16x2(fw[2 * t + p], txl.cuda.float2_x(prod), txl.cuda.float2_y(prod))
                bslot = parity * 2 + (ci & 1)
                with txl.If(grp == 0), txl.Then():
                    txl.ptx["st.shared.v4.f32"](txl.address_of(s_bblk[bslot, blk, k0]), bb[0], bb[1], bb[2], bb[3])
                rbb = txl.alloc_local([4], "float32")
                for j in range(4):
                    txl.ptx.rcp.approx.ftz.f32(rbb[j], bb[j])
                txl.ptx.bar.sync(bar_id, txl.uint32(128))
                # release the g stage only after the group barrier, once every thread's g loads are consumed; an earlier arrive would let the TMA producer rewrite the stage under in-flight loads
                with txl.If(tidp == 0), txl.Then():
                    p_g.empty.arrive(gstage)
                with txl.If((parity == 0) & (ci == 0) & (tidp == 0)), txl.Then():
                    m_gates0.arrive(0)
                ob = txl.alloc_local([4], "float32")
                txl.ptx["ld.shared.v4.f32"](ob[0], ob[1], ob[2], ob[3], txl.address_of(s_bblk[bslot, 1 - blk, k0]))
                with txl.If((blk == 1) & (grp == 0)), txl.Then():
                    txl.ptx["st.global.v4.f32"](vec.ptr_to([itb * txl.int64(VEC_F32) + k064]),
                                                bb[0] * ob[0], bb[1] * ob[1], bb[2] * ob[2], bb[3] * ob[3])
                _rng_end(tk)
                pf = txl.alloc_local([4], "float32")
                rf = txl.alloc_local([4], "float32")
                with txl.If(blk == 1):
                    with txl.Then():
                        for j in range(4):
                            txl.assign(pf[j], ob[j])
                            txl.assign(rf[j], txl.float32(1.0))
                    with txl.Else():
                        for j in range(4):
                            txl.assign(pf[j], txl.float32(1.0))
                            txl.assign(rf[j], ob[j])
                pfw = txl.alloc_local([2], "uint32")
                rfw = txl.alloc_local([2], "uint32")
                rbw = txl.alloc_local([2], "uint32")
                for p in range(2):
                    _pack_bf16x2(pfw[p], pf[2 * p], pf[2 * p + 1])
                    _pack_bf16x2(rfw[p], rf[2 * p], rf[2 * p + 1])
                    _pack_bf16x2(rbw[p], rbb[2 * p], rbb[2 * p + 1])

                tk = _rng("prep-wait-rawqk")
                p_raw.full.wait(stage, rphase)
                _rng_end(tk)
                tk = _rng("prep-wait-akk")
                with txl.If(li >= 2), txl.Then():
                    m_akk.wait(li & 1, ((li >> 1) - 1) & 1)
                _rng_end(tk)
                # Observe both MMA readers before the existing proxy fence:
                # it orders reuse of s_a/s_b and s_kt in the generic proxy.
                tk = _rng("prep-wait-w1")
                with txl.If(li >= 4), txl.Then():
                    m_w1.wait(li & 3, ((li - 4) >> 2) & 1)
                _rng_end(tk)
                tk = _rng("prep-tiles")
                txl.ptx[FENCE_ASYNC]()
                kq_t = s_k[stage]
                qq_t = s_q[stage]
                sa = s_a[li & 1]
                sb = s_b[li & 1]
                yrow0 = ROW_Y0 + NB_ROWS * blk + grp * 8
                zrow0 = ROW_Z0 - (ROW_Z0 - ROW_Z1) * blk + grp * 8
                qzrow0 = ROW_QZ0 + (ROW_QZ1 - ROW_QZ0) * blk + grp * 8
                gbase = (itb * txl.int64(C) + txl.Cast("int64", rowbase)) * txl.int64(D) + k064
                for t in range(8):
                    txl.ptx["ld.shared.v2.b32"](kws[2 * t], kws[2 * t + 1], kq_t.ptr_to(rowbase + t, k0))
                    txl.ptx["ld.shared.v2.b32"](qws[2 * t], qws[2 * t + 1], qq_t.ptr_to(rowbase + t, k0))
                sc = txl.alloc_local([2], "uint32")
                oa = txl.alloc_local([2], "uint32")
                ob_ = txl.alloc_local([2], "uint32")
                for t in range(8):
                    row = rowbase + t

                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], ew[2 * t + p])
                        txl.ptx.mul.rn.bf16x2(ob_[p], qws[2 * t + p], ew[2 * t + p])
                    txl.ptx["st.shared.v2.b32"](sa.ptr_to(row, k0), oa[0], oa[1])
                    txl.ptx["st.shared.v2.b32"](sa.ptr_to(C + row, k0), ob_[0], ob_[1])
                    # the Qt / Kbar global stores of this row are issued after the item's last fence.proxy.async (below), so no fence waits on in-flight global stores
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], fw[2 * t + p])
                    txl.ptx["st.shared.v2.b32"](sb.ptr_to(yrow0 + t, k0), oa[0], oa[1])

                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], fw[2 * t + p], rbw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                        txl.ptx.mul.rn.bf16x2(ob_[p], qws[2 * t + p], sc[p])
                    txl.ptx["st.shared.v2.b32"](sb.ptr_to(zrow0 + t, k0), oa[0], oa[1])
                    txl.ptx["st.shared.v2.b32"](sb.ptr_to(qzrow0 + t, k0), ob_[0], ob_[1])
                _rng_end(tk)

                tk = _rng("prep-kt")
                ktq = s_kt[li & 3]
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], ew[2 * t + p], pfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                    txl.ptx["st.shared.v2.b32"](ktq.ptr_to(rowbase + t, k0), oa[0], oa[1])
                txl.ptx[FENCE_ASYNC]()
                # Qt = q ew pf and Kbar = k fw rf rows -> global after the item's last async-proxy fence; the bar.sync below orders them before the leader's m_pdone arrive (published by the aux warp's gpu-scope fence with the item's flag)
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], ew[2 * t + p], pfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], qws[2 * t + p], sc[p])
                    txl.ptx["st.global.v2.b32"](qt_g.ptr_to([gbase + t * D]), oa[0], oa[1])
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], fw[2 * t + p], rfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                    txl.ptx["st.global.v2.b32"](kbar_g.ptr_to([gbase + t * D]), oa[0], oa[1])
                txl.ptx.bar.sync(bar_id, txl.uint32(128))
                with txl.If(tidp == 0), txl.Then():
                    with txl.If(li >= PDONE_SLOTS - 2), txl.Then():
                        m_sfree.wait((li + 2) & (PDONE_SLOTS - 1), ((li - (PDONE_SLOTS - 2)) >> 4) & 1)
                    m_tiles.arrive(li & 1)
                    p_raw.empty.arrive(stage)
                    m_pdone.arrive(li & (PDONE_SLOTS - 1))
                _rng_end(tk)

        with rdo:
            tid1 = txl.tid_in_role()
            lanebits = (tid1 << 16) & 0x600000
            wid1 = txl.warp_id_in_role()
            with txl.If(wid1 == 0), txl.Then():
                txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    txl.address_of(s_tmem_addr[0]), txl.uint32(TMEM1_COLS)
                )
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)

            blk1 = wid1 & 1
            is_x = wid1 < 2
            i_row = blk1 * 16 + lane
            active = lane < 16
            col_a = txl.int32(32) - blk1 * 32
            av = txl.alloc_local([32], "float32")
            zv = txl.alloc_local([16], "float32")
            knv = txl.alloc_local([32], "float32")
            pw = txl.alloc_local([16], "uint32")

            def select_tree(src, base, n):
                """src[base + lane] for lane < n (power-of-two balanced register gather)."""
                # Four predicate bits give a logarithmic-depth register gather.
                values = [src[base + c] for c in range(n)]
                bit = 1
                while len(values) > 1:
                    take_hi = (lane & bit) != 0
                    values = [txl.local_scalar("float32", init=txl.Select(take_hi, values[c + 1], values[c]))
                              for c in range(0, len(values), 2)]
                    bit *= 2
                return values[0]

            def apass(li):
                """Read the A chain of item li: L rows (+ k norms) -> s_l / s_kn slot li&3 ; Aqk rows (+ q norms) -> global."""
                aslot = li & 1
                lslot = li & 3
                tk = _rng("rdo-wait-akk")
                _slow_wait(m_akk, aslot, (li >> 1) & 1)
                with txl.If(li >= 4), txl.Then():
                    _slow_wait(m_lfree, lslot, ((li >> 2) - 1) & 1)
                _slow_wait(m_beta, li & (NSLOT - 1), (li >> 3) & 1)
                _rng_end(tk)
                tk = _rng("rdo-apass")
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](*(av[i] for i in range(32)), _taddr(tm, TM_A + TM_A_STRIDE * aslot + col_a, lanebits))
                with txl.If(wid1 == 3), txl.Then():
                    txl.ptx[TC_LD16](*(zv[i] for i in range(16)), _taddr(tm, TM_A + TM_A_STRIDE * aslot + ROW_QZ1, lanebits))
                txl.ptx[WAIT_LD]()

                dsel = txl.local_scalar("float32")
                with txl.If(wid1 == 0), txl.Then():
                    txl.assign(dsel, select_tree(av, 0, 16))
                with txl.If((wid1 == 1) | (wid1 == 2)), txl.Then():
                    txl.assign(dsel, select_tree(av, 16, 16))
                with txl.If(wid1 == 3), txl.Then():
                    txl.assign(dsel, select_tree(zv, 0, 16))
                nrm = txl.local_scalar("float32")
                txl.ptx.rsqrt.approx.ftz.f32(nrm, dsel + txl.float32(1e-6))
                with txl.If(active), txl.Then():
                    with txl.If(is_x):
                        with txl.Then():
                            txl.ptx.st.shared.f32(txl.address_of(s_kn[lslot, i_row]), nrm)
                        with txl.Else():
                            txl.ptx.st.global_.f32(vec.ptr_to([it64(li) * txl.int64(VEC_F32) + txl.int64(D) + txl.Cast("int64", i_row)]), nrm)
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                rbase = (it64(li) * txl.int64(C) + txl.Cast("int64", i_row)) * txl.int64(C)
                with txl.If(active), txl.Then():
                    with txl.If(is_x):
                        with txl.Then():
                            bi = txl.local_scalar("float32")
                            txl.ptx.ld.shared.f32(bi, txl.address_of(s_bt[li & (NSLOT - 1), i_row]))
                            coef = bi * nrm
                            for p_ in range(8):
                                txl.ptx["ld.shared.v4.f32"](knv[4 * p_], knv[4 * p_ + 1], knv[4 * p_ + 2], knv[4 * p_ + 3],
                                                            txl.address_of(s_kn[lslot, 4 * p_]))
                            prod = txl.local_scalar("uint64")
                            for p_ in range(16):
                                j0 = 2 * p_
                                txl.ptx.mul.rn.f32x2(prod, _f2(knv[j0], knv[j0 + 1]), _f2(coef, coef))
                                txl.ptx.mul.rn.f32x2(prod, _f2(txl.cuda.float2_x(prod), txl.cuda.float2_y(prod)), _f2(av[j0], av[j0 + 1]))
                                v0 = txl.Select(txl.int32(j0) < i_row, txl.cuda.float2_x(prod), txl.float32(0.0))
                                v1 = txl.Select(txl.int32(j0 + 1) < i_row, txl.cuda.float2_y(prod), txl.float32(0.0))
                                _pack_bf16x2(pw[p_], v0, v1)
                            sl = s_l[lslot]
                            for q_ in range(4):
                                txl.ptx["st.shared.v4.b32"](sl.ptr_to(i_row, 8 * q_), pw[4 * q_], pw[4 * q_ + 1], pw[4 * q_ + 2], pw[4 * q_ + 3])
                        with txl.Else():
                            for p_ in range(16):
                                j0 = 2 * p_
                                v0 = txl.Select(txl.int32(j0) <= i_row, av[j0], txl.float32(0.0))
                                v1 = txl.Select(txl.int32(j0 + 1) <= i_row, av[j0 + 1], txl.float32(0.0))
                                _pack_bf16x2(pw[p_], v0, v1)
                            for q_ in range(2):
                                txl.ptx.st.global_.L2__evict_last.v8.b32(
                                    aqk_g.ptr_to([rbase + q_ * 16]),
                                    *(pw[8 * q_ + j] for j in range(8)),
                                )
                txl.ptx[TC_FENCE_BEFORE]()
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                with txl.If(tid1 == 0), txl.Then():
                    m_afree.arrive(aslot)
                    m_lready.arrive(lslot)
                _signal_item(m_pdone, li, tid1 == 0)
                _rng_end(tk)

            def w1_readout(lw):
                """W1' fp32 [k][i] of item lw (lane k) -> bf16 row k of the global tile w1_g[it][k][i]."""
                wslot = lw & 3
                tk = _rng("rdo-wait-w1")
                _slow_wait(m_w1, wslot, (lw >> 2) & 1)
                _rng_end(tk)
                tk = _rng("rdo-w1")
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](*(av[i] for i in range(32)), _taddr(tm, TM_W1 + 32 * wslot, lanebits))
                txl.ptx[WAIT_LD]()
                txl.ptx[TC_FENCE_BEFORE]()
                for p_ in range(16):
                    _pack_bf16x2(pw[p_], av[2 * p_], av[2 * p_ + 1])
                wbase = (it64(lw) * txl.int64(D) + txl.Cast("int64", tid1)) * txl.int64(C)
                for p_ in range(2):
                    txl.ptx.st.global_.L2__evict_last.v8.b32(
                        w1_g.ptr_to([wbase + p_ * 16]),
                        *(pw[8 * p_ + j] for j in range(8)),
                    )
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                with txl.If(tid1 == 0), txl.Then():
                    m_w1free.arrive(wslot)
                _signal_item(m_pdone, lw, tid1 == 0)
                _rng_end(tk)

            with txl.serial(n_cta) as li:
                with txl.If(li >= W1_LAG), txl.Then():
                    w1_readout(li - W1_LAG)
                apass(li)
            for kk in range(W1_LAG, 0, -1):
                with txl.If(n_cta >= kk), txl.Then():
                    w1_readout(n_cta - kk)
            txl.cuda.warpgroup_sync(BAR_RDO)
            with txl.If(wid1 == 0), txl.Then():
                txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](txl.Cast("uint32", tm[0]), txl.uint32(TMEM1_COLS))

        with solve:
            sidx = txl.warp_id_in_role()

            def ldm_x4(insn, dst, av_, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                txl.ptx[insn](dst[0], dst[1], dst[2], dst[3], av_.ptr_to(row, col))

            def stm_x4(src, av_, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                txl.ptx[STM_X4](av_.ptr_to(row, col), src[0], src[1], src[2], src[3])

            def neg_pack(dst_word, a, b):
                neg = txl.local_scalar("uint64")
                txl.ptx.sub.rn.f32x2(neg, _f2(txl.float32(0.0), txl.float32(0.0)), _f2(a, b))
                _pack_bf16x2(dst_word, txl.cuda.float2_x(neg), txl.cuda.float2_y(neg))

            def mma_k8_zero(acc, a, b):
                txl.ptx[MMA_K8](acc[0], acc[1], acc[2], acc[3], a[0], a[1], b[0],
                                txl.float32(0.0), txl.float32(0.0), txl.float32(0.0), txl.float32(0.0))

            def mma_k16(acc, a, b, acc_off, b_off, accumulate):
                cc = [acc[acc_off + i] for i in range(4)] if accumulate else [txl.float32(0.0)] * 4
                txl.ptx[MMA_K16](*(acc[acc_off + i] for i in range(4)), a[0], a[1], a[2], a[3],
                                 b[b_off], b[b_off + 1], *cc)

            def invert_diag_8x8(av_, block8):
                r = block8 + (lane & 7)
                words = txl.alloc_local([4], "uint32")
                txl.ptx["ld.shared.v4.b32"](words[0], words[1], words[2], words[3], av_.ptr_to(r, block8))
                row = [txl.local_scalar("float32") for _ in range(8)]
                for p in range(4):
                    unpack(row, p, words[p])
                for i in range(8):
                    with txl.If((lane & 7) == i), txl.Then():
                        txl.assign(row[i], txl.float32(1.0))
                rs = txl.local_scalar("float32")
                pv = txl.local_scalar("uint32")
                for src in range(7):
                    txl.ptx.neg.f32(rs, row[src])
                    for i in range(7):
                        if i < src:
                            txl.ptx.shfl_sync.idx.b32(pv, txl.reinterpret("uint32", row[i]), txl.uint32(src),
                                                      txl.uint32(0x181F), txl.uint32(0xFFFFFFFF))
                            with txl.If((lane & 7) > src), txl.Then():
                                txl.assign(row[i], row[i] + rs * txl.reinterpret("float32", pv))
                    with txl.If((lane & 7) > src), txl.Then():
                        txl.assign(row[src], rs)
                for p in range(4):
                    _pack_bf16x2(words[p], row[2 * p], row[2 * p + 1])
                txl.ptx["st.shared.v4.b32"](av_.ptr_to(r, block8), words[0], words[1], words[2], words[3])

            def inverse_8_to_16(av_, b16):
                a = txl.alloc_local([2], "uint32")
                b = txl.alloc_local([1], "uint32")
                acc = txl.alloc_local([4], "float32")
                dm = txl.local_scalar("uint32")
                cm = txl.local_scalar("uint32")
                txl.ptx[LDM_X1](dm, av_.ptr_to(b16 + 8 + (lane & 7), b16 + 8))
                txl.ptx[LDM_X1T](cm, av_.ptr_to(b16 + 8 + (lane & 7), b16))
                txl.assign(a[0], dm)
                txl.assign(a[1], dm)
                txl.assign(b[0], cm)
                mma_k8_zero(acc, a, b)
                neg_pack(a[0], acc[0], acc[1])
                neg_pack(a[1], acc[2], acc[3])
                txl.ptx[LDM_X1T](b[0], av_.ptr_to(b16 + (lane & 7), b16))
                mma_k8_zero(acc, a, b)
                _pack_bf16x2(dm, acc[0], acc[1])
                txl.ptx[STM_X1](av_.ptr_to(b16 + 8 + (lane & 7), b16), dm)

            def inverse_16_to_32(av_, b32):
                a = txl.alloc_local([4], "uint32")
                b = txl.alloc_local([4], "uint32")
                acc = txl.alloc_local([8], "float32")
                outw = txl.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a, av_, b32 + 16, b32 + 16)
                ldm_x4(LDM_X4T, b, av_, b32 + 16, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    neg_pack(a[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b, av_, b32, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    _pack_bf16x2(outw[p], acc[2 * p], acc[2 * p + 1])
                stm_x4(outw, av_, b32 + 16, b32)

            sl = s_l[sidx]
            st2 = s_t2[sidx]
            tw = txl.alloc_local([8], "uint32")
            tv = txl.alloc_local([16], "float32")
            bq = txl.alloc_local([16], "float32")
            n_mine = (n_cta - sidx + 3) >> 2
            with txl.serial(n_mine) as ci:
                li = ci * 4 + sidx
                slot8 = li & (NSLOT - 1)
                tk = _rng("solve-wait-l")
                _slow_wait(m_lready, sidx, ci & 1)
                _slow_wait(m_beta, slot8, (li >> 3) & 1)
                _rng_end(tk)
                tk = _rng("solve-inv")
                invert_diag_8x8(sl, (lane >> 3) * 8)
                txl.cuda.warp_sync()
                inverse_8_to_16(sl, 0)
                inverse_8_to_16(sl, 16)
                txl.cuda.warp_sync()
                inverse_16_to_32(sl, 0)
                txl.cuda.warp_sync()
                _rng_end(tk)
                tk = _rng("solve-t")

                kni = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(kni, txl.address_of(s_kn[sidx, lane]))
                tbase = (it64(li) * txl.int64(C) + txl.Cast("int64", lane)) * txl.int64(C)
                with txl.If(li >= 4), txl.Then():
                    _slow_wait(m_w1, sidx, ((li - 4) >> 2) & 1)
                txl.ptx[FENCE_ASYNC]()
                for half in range(2):
                    c0 = 16 * half
                    txl.ptx["ld.shared.v4.b32"](tw[0], tw[1], tw[2], tw[3], sl.ptr_to(lane, c0))
                    for p in range(4):
                        unpack(tv, p, tw[p])
                    txl.ptx["ld.shared.v4.b32"](tw[0], tw[1], tw[2], tw[3], sl.ptr_to(lane, c0 + 8))
                    for p in range(4):
                        unpack(tv, 4 + p, tw[p])
                    for p in range(4):
                        txl.ptx["ld.shared.v4.f32"](bq[4 * p], bq[4 * p + 1], bq[4 * p + 2], bq[4 * p + 3],
                                                    txl.address_of(s_bt[slot8, c0 + 4 * p]))
                    prod = txl.local_scalar("uint64")
                    for p in range(8):
                        txl.ptx.mul.rn.f32x2(prod, _f2(tv[2 * p] * kni, tv[2 * p + 1] * kni), _f2(bq[2 * p], bq[2 * p + 1]))
                        txl.assign(tv[2 * p], txl.cuda.float2_x(prod))
                        txl.assign(tv[2 * p + 1], txl.cuda.float2_y(prod))
                    for p in range(8):
                        _pack_bf16x2(tw[p], tv[2 * p], tv[2 * p + 1])
                    txl.ptx.st.global_.L2__evict_last.v8.b32(
                        t1_g.ptr_to([tbase + c0]), *(tw[p] for p in range(8))
                    )
                    for p in range(4):
                        txl.ptx["ld.shared.v4.f32"](bq[4 * p], bq[4 * p + 1], bq[4 * p + 2], bq[4 * p + 3],
                                                    txl.address_of(s_kn[sidx, c0 + 4 * p]))
                    for p in range(8):
                        txl.ptx.mul.rn.f32x2(prod, _f2(tv[2 * p], tv[2 * p + 1]), _f2(bq[2 * p], bq[2 * p + 1]))
                        _pack_bf16x2(tw[p % 4], txl.cuda.float2_x(prod), txl.cuda.float2_y(prod))
                        if p % 4 == 3:
                            txl.ptx["st.shared.v4.b32"](st2.ptr_to(lane, c0 + 8 * (p // 4)), tw[0], tw[1], tw[2], tw[3])
                txl.ptx[FENCE_ASYNC]()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    m_tp.arrive(sidx)
                    m_lfree.arrive(sidx)
                    m_bdone.arrive(slot8)
                    m_pdone.arrive(li & (PDONE_SLOTS - 1))
                _rng_end(tk)

        with mma2:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)
            e = elect_local()
            with txl.serial(n_cta) as li:
                aslot = li & 1
                tk = _rng("mma2-wait-tiles")
                _slow_wait(m_tiles, aslot, (li >> 1) & 1, 200)
                with txl.If(li >= 2), txl.Then():
                    _slow_wait(m_afree, aslot, ((li >> 1) - 1) & 1, 200)
                _rng_end(tk)
                tk = _rng("mma2-A")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(tm[0] + TM_A + TM_A_STRIDE * aslot, s_a[aslot], s_b[aslot], IDESC_M64_N80, False, e)
                m_akk.arrive(aslot, pred=elect())
                _rng_end(tk)

        with mma3:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)
            e = elect_local()
            with txl.serial(n_cta) as lw:
                wslot = lw & 3
                tk = _rng("mma3-wait-tp")
                _slow_wait(m_tp, wslot, (lw >> 2) & 1, 200)
                with txl.If(lw >= 4), txl.Then():
                    _slow_wait(m_w1free, wslot, ((lw >> 2) - 1) & 1, 200)
                _rng_end(tk)
                tk = _rng("mma3-W1")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(tm[0] + TM_W1 + 32 * wslot, s_kt[wslot], s_t2[wslot], IDESC_N32_TA, False, e)
                m_w1.arrive(wslot, pred=elect())
                _rng_end(tk)

        with tma:
            for m_ in (q_map, k_map, g_map):
                txl.ptx.prefetch.tensormap(txl.address_of(m_))
            st_rawp = txl.PipelineState(STAGES1, phase=1)
            st_gp = txl.PipelineState(GSTAGES, phase=1)

            with txl.serial(n_cta) as li:
                it = item_of(li)
                head = it % H
                tok0 = (it // H) * C
                tk = _rng("tma-wait-empty")
                _slow_wait(p_raw.empty, st_rawp.stage, st_rawp.phase)
                _rng_end(tk)
                with txl.If(elected()), txl.Then():
                    p_raw.full.arrive(st_rawp.stage, tx_count=2 * C * D * 2)
                    for d in (0, 64):
                        for m_, tile in ((q_map, s_q), (k_map, s_k)):
                            txl.ptx[TMA_G2S_HINT](tile[st_rawp.stage].ptr_to(0, d), txl.address_of(m_), txl.int32(d), tok0, head,
                                                  p_raw.full.ptr_to([st_rawp.stage]), txl.uint64(0x12F0000000000000))
                tk = _rng("tma-wait-gempty")
                _slow_wait(p_g.empty, st_gp.stage, st_gp.phase)
                _rng_end(tk)
                # Bridge the completed generic gate reads to the next TMA
                # overwrite, after acquiring the consumers' empty signal.
                txl.ptx[FENCE_ASYNC]()
                with txl.If(elected()), txl.Then():
                    p_g.full.arrive(st_gp.stage, tx_count=C * D * 2)
                    for d in (0, 64):
                        txl.ptx[TMA_G2S_HINT](s_g[st_gp.stage].ptr_to(0, d), txl.address_of(g_map), txl.int32(d), tok0, head,
                                              p_g.full.ptr_to([st_gp.stage]), txl.uint64(0x12F0000000000000))
                st_gp.advance()
                st_rawp.advance()

        with aux:
            bw = txl.local_scalar("uint16")

            def load_beta(cc):
                """b = sigmoid(beta) of the item's 32 tokens, and the gate constants ha / hadt[k] of its head."""
                it_ = item_of(cc)
                head_ = it_ % H
                slot_ = cc & (NSLOT - 1)
                tok = txl.Cast("int64", (it_ // H) * C + lane)
                txl.ptx.ld.global_.u16(bw, beta.ptr_to([tok * txl.int64(H) + txl.Cast("int64", head_)]))
                a_h = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(a_h, a_log.ptr_to([head_]))
                dt4 = txl.alloc_local([4], "float32")
                txl.ptx["ld.global.v4.f32"](dt4[0], dt4[1], dt4[2], dt4[3], dt_bias.ptr_to([head_ * D + lane * 4]))
                bf = txl.reinterpret("float32", txl.Cast("uint32", bw) << txl.uint32(16))
                tb = txl.local_scalar("float32")
                txl.ptx.tanh.approx.f32(tb, bf * txl.float32(0.5))
                txl.ptx.st.shared.f32(txl.address_of(s_bt[slot_, lane]), tb * txl.float32(0.5) + txl.float32(0.5))
                txl.ptx.ex2.approx.ftz.f32(a_h, a_h * txl.float32(LOG2E))
                ha = txl.local_scalar("float32", init=a_h * txl.float32(0.5))
                txl.ptx["st.shared.v4.f32"](txl.address_of(s_hadt[slot_, lane * 4]), ha * dt4[0], ha * dt4[1], ha * dt4[2], ha * dt4[3])
                with txl.If(lane == 0), txl.Then():
                    txl.ptx.st.shared.f32(txl.address_of(s_ha[slot_, 0]), ha)
                m_beta.arrive(slot_)

            for cc0 in range(NSLOT):
                with txl.If(n_cta > cc0), txl.Then():
                    load_beta(txl.int32(cc0))
            with txl.serial(n_cta) as li:
                with txl.If(li + NSLOT < n_cta), txl.Then():
                    tk = _rng("aux-wait-bdone")
                    _slow_wait(m_bdone, li & (NSLOT - 1), (li >> 3) & 1)
                    _rng_end(tk)
                    load_beta(li + NSLOT)

                with txl.If(((li & (signal_batch - 1)) == signal_batch - 1) | (li == n_cta - 1)), txl.Then():
                    li0 = li & ~(signal_batch - 1)
                    tk = _rng("sig-wait")
                    for u in range(signal_batch):
                        with txl.If(li0 + u <= li), txl.Then():
                            _slow_wait(m_pdone, (li0 + u) & (PDONE_SLOTS - 1), ((li0 + u) >> 4) & 1)
                    _rng_end(tk)
                    with txl.If((do_signal != 0) & (lane == 0)), txl.Then():
                        txl.ptx.fence.acq_rel.gpu()
                        for u in range(signal_batch):
                            with txl.If(li0 + u <= li), txl.Then():
                                txl.ptx.red.relaxed.gpu.global_.add.s32(flags.ptr_to([item_of(li0 + u)]), txl.int32(1))
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        for u in range(signal_batch):
                            with txl.If(li0 + u <= li), txl.Then():
                                m_sfree.arrive((li0 + u) & (PDONE_SLOTS - 1))

    return kda_front


STAGES2 = 5
TM_S0 = 0
TM_SB0 = 256
TM_U0 = 384
TM_O0 = 448
BAR_ST = (3, 6, 9, 10)
BAR_EPI2 = 7
BAR_DONE2 = 8
BAR_TMEM_N2 = 704
TX2 = 4 * TILE_BYTES + 2 * SMALL_BYTES + VEC_F32 * 4


def make_chain(H: int, hpc: int = 2):
    """Recurrence kernel: grid = H/hpc, one persistent CTA per group of hpc heads (hpc = 2: the two heads' chains
    are interleaved so that one chain's latency hides behind the other's work, freeing SMs for the concurrent
    front end; hpc = 1: one head per CTA, for head counts that leave the front end enough SMs anyway)."""
    assert H % hpc == 0 and hpc in (1, 2)

    @txl.kernel(warps=24, arch="sm_100a", min_blocks_per_sm=1, grid=H // hpc)
    def kda_chain(
        v: txl.gptr[txl.bf16],
        state_in: txl.gptr[txl.f32],
        state_out: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        vec: txl.gptr[txl.f32],
        kbar_g: txl.gptr[txl.bf16],
        qt_g: txl.gptr[txl.bf16],
        t1_g: txl.gptr[txl.bf16],
        aqk_g: txl.gptr[txl.bf16],
        w1_g: txl.gptr[txl.bf16],
        v_map: txl.TensorMap,
        kbar_map: txl.TensorMap,
        qt_map: txl.TensorMap,
        t1_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        w1_map: txl.TensorMap,
        o_map: txl.TensorMap,
        flags: txl.gptr[txl.i32],
        scale: txl.f32,
        num_chunks: txl.i32,
        flag_from: txl.i32,
        flag_target: txl.i32,
    ):
        for buf in (v, out, kbar_g, qt_g, t1_g, aqk_g, w1_g):
            txl.keep_alive(buf.ptr_to([0]))

        pair = txl.cta_id()
        head0 = pair * hpc
        tid = txl.thread_id()
        lane = txl.lane_id()

        sp = txl.specialize()

        st = sp.role("st", warps=list(range(16)), regs=88)
        epi = sp.role("epi", warps=[16, 17, 18, 19], regs=88)
        mma = sp.role("mma", warps=[20, 21], regs=40)
        tma = sp.role("tma", warps=[22], regs=40)
        idle = sp.role("idle", warps=[23], regs=40)

        smem = txl.smem_pool()
        s_tmem_addr = smem.alloc((1,), txl.i32, align=4)
        p_ring = txl.Pipeline(smem, STAGES2, full="tma", empty="tcgen05", init_empty=1)
        m_snap = txl.MBarrier(smem, 2)
        m_decay = txl.MBarrier(smem, 2)
        m_u = txl.TCGen05Bar(smem, 2)
        m_ub = txl.MBarrier(smem, 2)
        m_s = txl.TCGen05Bar(smem, 2)
        m_o = txl.TCGen05Bar(smem, 2)
        m_ofree = txl.MBarrier(smem, 2)
        m_dfree = txl.MBarrier(smem, STAGES2)
        for bar, cnt in (
            (m_snap, 2), (m_decay, 2), (m_u, 1), (m_ub, 1), (m_s, 1), (m_o, 1),
            (m_ofree, 1), (m_dfree, 1),
        ):
            bar.init(cnt)

        s_v = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_kbar = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_qt = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_w1 = smem.alloc((STAGES2, D, C), txl.bf16, swizzle=txl.SW64B)
        s_t1 = smem.alloc((STAGES2, C, C), txl.bf16, swizzle=txl.SW64B)
        s_aqkT = smem.alloc((STAGES2, C, C), txl.bf16, swizzle=txl.SW64B)
        s_dec = smem.alloc((STAGES2, D), txl.f32, align=16)
        s_qn = smem.alloc((STAGES2, C), txl.f32, align=16)
        s_o = smem.alloc((2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_qne = smem.alloc((4, C), txl.f32, align=16)

        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elect():
            return txl.cuda.elect_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def elect_local():
            e = txl.local_scalar("uint32")
            txl.assign(e, txl.cuda.elect_sync())
            return e

        def snapshot_quarter(tm, tm_s, tm_sb, c0, lanebits, stage, sv, sw, dvec, arrive_snap):
            """Lanes = v.  Columns [c0, c0+32) of ST(h): bf16 copy -> TM_SB(h) columns [c0/2, c0/2+16), then ST *= D
            (in place).  arrive_snap(): called once the packed copy is stored (before the decay)."""
            txl.ptx[TC_LD32](*(sv[i] for i in range(32)), _taddr(tm, tm_s + c0, lanebits))
            for vec_ in range(4):
                txl.ptx["ld.shared.v4.f32"](
                    dvec[vec_ * 4], dvec[vec_ * 4 + 1], dvec[vec_ * 4 + 2], dvec[vec_ * 4 + 3],
                    txl.address_of(s_dec[stage, c0 + vec_ * 4]),
                )
            txl.ptx[WAIT_LD]()
            for p in range(16):
                _pack_bf16x2(sw[p], sv[2 * p], sv[2 * p + 1])
            txl.ptx[TC_ST16](_taddr(tm, tm_sb + c0 // 2, lanebits), *(sw[i] for i in range(16)))
            if arrive_snap is not None:
                txl.ptx[WAIT_ST]()
                txl.ptx[TC_FENCE_BEFORE]()
                arrive_snap()
            for p in range(8):
                prod = txl.local_scalar("uint64")
                txl.ptx.mul.rn.f32x2(prod, _f2(sv[2 * p], sv[2 * p + 1]), _f2(dvec[2 * p], dvec[2 * p + 1]))
                txl.assign(sv[2 * p], txl.cuda.float2_x(prod))
                txl.assign(sv[2 * p + 1], txl.cuda.float2_y(prod))
            for vec_ in range(4):
                txl.ptx["ld.shared.v4.f32"](
                    dvec[vec_ * 4], dvec[vec_ * 4 + 1], dvec[vec_ * 4 + 2], dvec[vec_ * 4 + 3],
                    txl.address_of(s_dec[stage, c0 + 16 + vec_ * 4]),
                )
            for p in range(8):
                prod = txl.local_scalar("uint64")
                txl.ptx.mul.rn.f32x2(prod, _f2(sv[16 + 2 * p], sv[16 + 2 * p + 1]), _f2(dvec[2 * p], dvec[2 * p + 1]))
                txl.assign(sv[16 + 2 * p], txl.cuda.float2_x(prod))
                txl.assign(sv[16 + 2 * p + 1], txl.cuda.float2_y(prod))
            txl.ptx[TC_ST32](_taddr(tm, tm_s + c0, lanebits), *(sv[i] for i in range(32)))

        with st:
            wid_all = txl.warp_id_in_role()
            h = wid_all >> 3
            half = (wid_all >> 2) & 1
            bar_id = txl.uint32(BAR_ST[0]) + txl.Cast("uint32", h) * txl.uint32(BAR_ST[2] - BAR_ST[0]) \
                + txl.Cast("uint32", half) * txl.uint32(BAR_ST[1] - BAR_ST[0])
            tid1 = txl.tid_in_role() & 127
            lanebits = (tid1 << 16) & 0x600000
            tm_s = TM_S0 + 128 * h
            tm_sb = TM_SB0 + 64 * h
            tm_u = TM_U0 + 32 * h
            tm_ub = tm_u
            cbase = half * 64
            with txl.If(wid_all == 0), txl.Then():
                txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    txl.address_of(s_tmem_addr[0]), txl.uint32(TMEM2_COLS)
                )
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            sv = txl.alloc_local([32], "float32")
            sw = txl.alloc_local([16], "uint32")
            dvec = txl.alloc_local([16], "float32")
            sbase = (txl.Cast("int64", head0 + h) * txl.int64(D) + txl.Cast("int64", tid1)) * txl.int64(D) + txl.Cast("int64", cbase)
            if hpc == 1:
                # head-1 warps have no chain: skip the per-head work but still join the CTA-wide TMEM barriers
                st_if = txl.If(h < hpc)
                st_if.__enter__()
                st_then = txl.Then()
                st_then.__enter__()
            for s4 in range(2):
                for vec_ in range(4):
                    txl.ptx["ld.global.L1::no_allocate.L2::evict_first.v8.f32"](
                        *(sv[vec_ * 8 + j] for j in range(8)),
                        state_in.ptr_to([sbase + s4 * 32 + vec_ * 8]),
                    )
                txl.ptx[TC_ST32](_taddr(tm, tm_s + cbase + s4 * 32, lanebits), *(sv[i] for i in range(32)))
            txl.ptx[WAIT_ST]()
            st_ring = txl.PipelineState(STAGES2, phase=0)
            if hpc == 2:
                with txl.If(h == 1), txl.Then():
                    st_ring.advance()
            with txl.serial(num_chunks) as c:
                stage = txl.local_scalar("int32", init=st_ring.stage)
                tk = _rng("st-wait-s")
                with txl.If(c > 0), txl.Then():
                    m_s.wait(h, (c - 1) & 1)
                p_ring.full.wait(stage, st_ring.phase)
                _rng_end(tk)
                txl.ptx[TC_FENCE_AFTER]()
                with txl.If((half == 0) & (tid1 < 32)), txl.Then():
                    qv = txl.local_scalar("float32")
                    txl.ptx.ld.shared.f32(qv, txl.address_of(s_qn[stage, tid1]))
                    txl.ptx.st.shared.f32(txl.address_of(s_qne[(c & 1) * 2 + h, tid1]), qv * scale)
                tk = _rng("st-snap")
                snapshot_quarter(tm, tm_s, tm_sb, cbase, lanebits, stage, sv, sw, dvec, None)
                snapshot_quarter(tm, tm_s, tm_sb, cbase + 32, lanebits, stage, sv, sw, dvec,
                                 lambda: _wg_arrive(m_snap, h, bar_id, tid1 == 0))
                txl.ptx[WAIT_ST]()
                txl.ptx[TC_FENCE_BEFORE]()
                txl.ptx[FENCE_ASYNC]()
                _wg_arrive(m_decay, h, bar_id, tid1 == 0)
                _rng_end(tk)
                with txl.If(half == 0), txl.Then():
                    tk = _rng("st-wait-u")
                    m_u.wait(h, c & 1)
                    _rng_end(tk)
                    tk = _rng("st-ub")
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[TC_LD32](*(sv[i] for i in range(32)), _taddr(tm, tm_u, lanebits))
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        _pack_bf16x2(sw[p], sv[2 * p], sv[2 * p + 1])
                    txl.ptx[TC_ST16](_taddr(tm, tm_ub, lanebits), *(sw[i] for i in range(16)))
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    _wg_arrive(m_ub, h, bar_id, tid1 == 0)
                    _rng_end(tk)
                for _ in range(hpc):
                    st_ring.advance()
            m_s.wait(h, (num_chunks - 1) & 1)
            txl.ptx[TC_FENCE_AFTER]()
            for s4 in range(2):
                txl.ptx[TC_LD32](*(sv[i] for i in range(32)), _taddr(tm, tm_s + cbase + s4 * 32, lanebits))
                txl.ptx[WAIT_LD]()
                for vec_ in range(4):
                    txl.ptx["st.global.L1::no_allocate.L2::evict_first.v8.f32"](
                        state_out.ptr_to([sbase + s4 * 32 + vec_ * 8]),
                        *(sv[vec_ * 8 + j] for j in range(8)),
                    )
            if hpc == 1:
                st_then.__exit__(None, None, None)
                st_if.__exit__(None, None, None)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))
            with txl.If(wid_all == 0), txl.Then():
                txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](txl.Cast("uint32", tm[0]), txl.uint32(TMEM2_COLS))

        with epi:
            tid3 = txl.tid_in_role()
            wq = tid3 >> 5
            lane3 = tid3 & 31
            lanebits3 = (tid3 << 16) & 0x600000
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            txl.ptx.prefetch.tensormap(txl.address_of(o_map))

            fr = txl.alloc_local([32], "float32")
            qs = txl.alloc_local([8], "float32")
            pk = txl.alloc_local([16], "uint32")

            i_l = ((lane3 >> 4) & 1) * 8 + (lane3 & 7)
            v_l = wq * 32 + ((lane3 >> 3) & 1) * 8
            ci = (lane3 & 3) * 2

            def epilogue(c, h):
                """o[i][v] = scale * qn_i * O^T[v][i] for chunk c of head h (TM_O(h), staging slot h)."""
                tk = _rng("epi-wait-o")
                m_o.wait(h, c & 1)
                _rng_end(tk)
                tk = _rng("epi")
                txl.ptx[TC_FENCE_AFTER]()
                for hh in range(2):
                    txl.ptx[TC_LD256_X4](*(fr[16 * hh + i] for i in range(16)),
                                         _taddr(tm, TM_O0 + 32 * h, lanebits3 + txl.int32(hh << 20)))
                for rep in range(4):
                    txl.ptx["ld.shared.v2.f32"](qs[2 * rep], qs[2 * rep + 1],
                                                txl.address_of(s_qne[(c & 1) * 2 + h, 8 * rep + ci]))
                txl.ptx[WAIT_LD]()
                txl.ptx[TC_FENCE_BEFORE]()
                with txl.If(tid3 == 0), txl.Then():
                    txl.ptx[BULK_WAIT_READ](hpc - 1)
                txl.ptx.bar.sync(txl.uint32(BAR_EPI2), txl.uint32(128))
                with txl.If(tid3 == 0), txl.Then():
                    m_ofree.arrive(h)
                txl.ptx[FENCE_ASYNC]()
                for hh in range(2):
                    for rep in range(4):
                        for rb in range(2):
                            k = 16 * hh + 4 * rep + 2 * rb
                            _pack_bf16x2(pk[8 * hh + 2 * rep + rb], fr[k] * qs[2 * rep], fr[k + 1] * qs[2 * rep + 1])
                ot = s_o[h]
                for hh in range(2):
                    for r0 in range(2):
                        txl.ptx[STM_X4T](ot.ptr_to(16 * r0 + i_l, 16 * hh + v_l), pk[8 * hh + 4 * r0],
                                         pk[8 * hh + 4 * r0 + 1], pk[8 * hh + 4 * r0 + 2], pk[8 * hh + 4 * r0 + 3])
                txl.ptx[FENCE_ASYNC]()
                txl.ptx.bar.sync(txl.uint32(BAR_EPI2), txl.uint32(128))
                with txl.If(tid3 == 0), txl.Then():
                    for d in (0, 64):
                        txl.ptx[TMA_S2G](txl.address_of(o_map), txl.int32(d), c * C, head0 + h, ot.ptr_to(0, d))
                    txl.ptx[BULK_COMMIT]()
                _rng_end(tk)

            with txl.serial(num_chunks) as c:
                for hh_ in range(hpc):
                    epilogue(c, txl.int32(hh_))
            with txl.If(tid3 == 0), txl.Then():
                txl.ptx[BULK_WAIT](0)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))

        with mma:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            hm = txl.warp_id_in_role()
            tm_s = TM_S0 + 128 * hm
            tm_u = TM_U0 + 32 * hm
            tm_ub = tm_u
            tm_o = TM_O0 + 32 * hm
            tm_sb = TM_SB0 + 64 * hm
            st_r = txl.PipelineState(STAGES2, phase=0)
            if hpc == 2:
                with txl.If(hm == 1), txl.Then():
                    st_r.advance()
            e = elect_local()
            if hpc == 1:
                mma_if = txl.If(hm < hpc)
                mma_if.__enter__()
                mma_then = txl.Then()
                mma_then.__enter__()
            with txl.serial(num_chunks) as c:
                stage = txl.local_scalar("int32", init=st_r.stage)
                tk = _rng("mma-wait-ring")
                p_ring.full.wait(stage, st_r.phase)
                _rng_end(tk)
                tk = _rng("mma-U1")
                _chain(tm[0] + tm_u, s_v[stage], s_t1[stage], IDESC_N32_TA, False, e)
                _rng_end(tk)
                tk = _rng("mma-wait-snap")
                m_snap.wait(hm, c & 1)
                with txl.If(c >= 1), txl.Then():
                    m_ofree.wait(hm, (c - 1) & 1)
                _rng_end(tk)
                tk = _rng("mma-U2O1")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(tm[0] + tm_u, tm[0] + tm_sb, s_w1[stage], IDESC_N32_TB_NEG, True, e)
                m_u.arrive(hm, pred=elect())
                _chain(tm[0] + tm_o, tm[0] + tm_sb, s_qt[stage], IDESC_N32, False, e)
                _rng_end(tk)
                tk = _rng("mma-wait-ub")
                m_ub.wait(hm, c & 1)
                m_decay.wait(hm, c & 1)
                _rng_end(tk)
                tk = _rng("mma-SO2")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(tm[0] + tm_s, tm[0] + tm_ub, s_kbar[stage], IDESC_N128_TB, True, e)
                m_s.arrive(hm, pred=elect())
                _chain(tm[0] + tm_o, tm[0] + tm_ub, s_aqkT[stage], IDESC_N32, True, e)
                m_o.arrive(hm, pred=elect())
                p_ring.empty.arrive(stage, pred=elect())
                _rng_end(tk)
                for _ in range(hpc):
                    st_r.advance()
            if hpc == 1:
                mma_then.__exit__(None, None, None)
                mma_if.__exit__(None, None, None)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))

        with tma:
            for m_ in (v_map, kbar_map, qt_map, t1_map, aqk_map, w1_map):
                txl.ptx.prefetch.tensormap(txl.address_of(m_))
            st_p = txl.PipelineState(STAGES2, phase=1)

            ready_upto = txl.local_scalar("int32", init=txl.int32(0))
            n_items = num_chunks * hpc
            with txl.serial(n_items) as j:
                c = j // hpc
                hh = j % hpc
                it = c * H + head0 + hh
                tk = _rng("tma-wait-flag")
                with txl.If((it >= flag_from) & (j >= ready_upto)), txl.Then():
                    jj = j + lane
                    itl = (jj // hpc) * H + head0 + (jj % hpc)
                    in_range = (jj < n_items) & (itl >= flag_from)
                    fl = txl.local_scalar("int32", init=flag_target)
                    nready = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(nready == 0):
                        with txl.If(in_range), txl.Then():
                            txl.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([itl]))
                        ballot = txl.local_scalar("uint32")
                        txl.ptx.vote_sync.ballot.b32(ballot, txl.ptx.pred(fl >= flag_target), txl.uint32(0xFFFFFFFF))

                        trail = ballot & ~(ballot + txl.uint32(1))
                        txl.assign(nready, txl.Cast("int32", txl.popcount(trail)))
                        with txl.If(nready == 0), txl.Then():
                            txl.cuda.nano_sleep(txl.uint64(SLOW_WAIT_NS))
                    txl.ptx.fence.proxy.async_.global_()

                    with txl.If((lane < nready) & in_range), txl.Then():
                        txl.ptx.st.global_.s32(flags.ptr_to([itl]), txl.int32(0))
                    txl.assign(ready_upto, j + nready)
                _rng_end(tk)
                tk = _rng("tma-wait-empty")
                with txl.If(j >= STAGES2), txl.Then():
                    m_dfree.wait(st_p.stage, ((j - STAGES2) // STAGES2) & 1)
                p_ring.empty.wait(st_p.stage, st_p.phase)
                _rng_end(tk)
                stg = st_p.stage
                with txl.If(elected()), txl.Then():
                    p_ring.full.arrive(stg, tx_count=TX2)
                    for d in (0, 64):
                        txl.ptx[TMA_G2S_HINT](s_v[stg].ptr_to(0, d), txl.address_of(v_map), txl.int32(d), c * C, head0 + hh,
                                              p_ring.full.ptr_to([stg]), txl.uint64(0x12F0000000000000))
                        txl.ptx[TMA_G2S](s_kbar[stg].ptr_to(0, d), txl.address_of(kbar_map), txl.int32(d), txl.int32(0), it,
                                         p_ring.full.ptr_to([stg]))
                        txl.ptx[TMA_G2S](s_qt[stg].ptr_to(0, d), txl.address_of(qt_map), txl.int32(d), txl.int32(0), it,
                                         p_ring.full.ptr_to([stg]))
                    txl.ptx[TMA_G2S](s_t1[stg].ptr_to(0, 0), txl.address_of(t1_map), txl.int32(0), txl.int32(0), it,
                                     p_ring.full.ptr_to([stg]))
                    txl.ptx[TMA_G2S](s_aqkT[stg].ptr_to(0, 0), txl.address_of(aqk_map), txl.int32(0), txl.int32(0), it,
                                     p_ring.full.ptr_to([stg]))
                    txl.ptx[TMA_G2S](s_w1[stg].ptr_to(0, 0), txl.address_of(w1_map), txl.int32(0), txl.int32(0), it,
                                     p_ring.full.ptr_to([stg]))
                    vbase = txl.Cast("int64", it) * txl.int64(VEC_F32)
                    txl.ptx[BULK_G2S](txl.address_of(s_dec[stg, 0]), vec.ptr_to([vbase]), txl.uint32(D * 4),
                                      p_ring.full.ptr_to([stg]))
                    txl.ptx[BULK_G2S](txl.address_of(s_qn[stg, 0]), vec.ptr_to([vbase + txl.int64(D)]), txl.uint32(C * 4),
                                      p_ring.full.ptr_to([stg]))

                st_p.advance()

        with idle:
            # scratch is dead once its TMA transfer completes: the idle warp, not the latency-sensitive TMA issuer, discards each completed item from L2
            st_d = txl.PipelineState(STAGES2, phase=0)
            with txl.serial(num_chunks * hpc) as j:
                p_ring.full.wait(st_d.stage, st_d.phase)
                itp = txl.Cast("int64", (j // hpc) * H + head0 + (j % hpc))
                l64 = txl.Cast("int64", lane)
                for tile_g, nlines in ((kbar_g, 64), (qt_g, 64), (w1_g, 64)):
                    for base_ in range(0, nlines, 32):
                        txl.ptx["discard.global.L2"](
                            tile_g.ptr_to([itp * txl.int64(C * D) + (l64 + base_) * txl.int64(64)])
                        )
                for tile_g in (t1_g, aqk_g):
                    with txl.If(lane < 16), txl.Then():
                        txl.ptx["discard.global.L2"](
                            tile_g.ptr_to([itp * txl.int64(C * C) + l64 * txl.int64(64)])
                        )
                with txl.If(lane < 5), txl.Then():
                    txl.ptx["discard.global.L2"](
                        vec.ptr_to([itp * txl.int64(VEC_F32) + l64 * txl.int64(32)])
                    )
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    m_dfree.arrive(st_d.stage)
                st_d.advance()

    return kda_chain


class _AlignedTensorMap:
    def __init__(self):
        self._storage = ctypes.create_string_buffer(128 + 64)
        base = ctypes.addressof(self._storage)
        self.ptr = ctypes.c_void_p((base + 63) & ~63)


def _encode(tensor, dims, strides_bytes, box, swizzle_code):
    desc = _AlignedTensorMap()
    encode = tvm.get_global_func("runtime.cuTensorMapEncodeTiled")
    encode(
        desc.ptr, "bfloat16", 3, ctypes.c_void_p(int(tensor.data_ptr())),
        *dims, *strides_bytes, *box, 1, 1, 1,
        0,
        swizzle_code,
        3,
        0,
    )
    return desc


def _encode_thd(tensor, T, H):
    """3D map over [T][H][128] bf16: dims (128, T, H), box (64, C, 1), 128B swizzle."""
    return _encode(tensor, (D, T, H), (H * D * 2, D * 2), (64, C, 1), 3)


def _encode_tile(tensor, rows, cols, NI):
    """Tile array [NI][rows][cols] bf16; box = one tile (inner box 64 elements for 128B swizzle)."""
    if cols == D:
        return _encode(tensor, (D, rows, NI), (D * 2, rows * D * 2), (64, rows, 1), 3)
    assert cols == C
    return _encode(tensor, (C, rows, NI), (C * 2, rows * C * 2), (C, rows, 1), 2)


_COMPILED = {}


def _compile(H, hpc):
    exes = _COMPILED.get((H, hpc))
    if exes is None:
        k1 = make_front(H)
        with k1.target():
            exe1 = tvm.compile(k1.mod, target=k1.target(), tir_pipeline="tirx")
        k2 = make_chain(H, hpc)
        with k2.target():
            exe2 = tvm.compile(k2.mod, target=k2.target(), tir_pipeline="tirx")
        exes = (exe1, exe2)
        _COMPILED[(H, hpc)] = exes
    return exes


CONCURRENT = os.environ.get("KDA_CONCURRENT", "1") == "1"


def setup(data, B, T, H):
    assert B == 1 and data["cu_seqlens"] is None, "fixed single-sequence case only"
    assert T % C == 0 and H % 2 == 0
    q, k, v, g, beta = data["q"], data["k"], data["v"], data["g"], data["beta"]
    output, final_state, initial_state = data["output"], data["final_state"], data["initial_state"]
    dev = q.device
    for t in (q, k, v, g, beta, output):
        assert t.is_contiguous()
    sm_count = torch.cuda.get_device_properties(dev).multi_processor_count
    # one head per chain CTA when the front end still gets at least half the SMs (H=64), otherwise interleaved head pairs (H=96)
    hpc = 1 if (H <= sm_count // 2 and os.environ.get("KDA_HPC", "auto") != "2") or os.environ.get("KDA_HPC") == "1" else 2
    exe1, exe2 = _compile(H, hpc)
    NC = T // C
    NI = NC * H
    a_log = data["A_log"].contiguous().float()
    dt_bias = data["dt_bias"].contiguous().float()
    init_c = initial_state.contiguous().float()
    kbar_buf = torch.empty(NI, C, D, dtype=torch.bfloat16, device=dev)
    qt_buf = torch.empty(NI, C, D, dtype=torch.bfloat16, device=dev)
    w1_buf = torch.empty(NI, D, C, dtype=torch.bfloat16, device=dev)
    t1_buf = torch.empty(NI, C, C, dtype=torch.bfloat16, device=dev)
    aqk_buf = torch.empty(NI, C, C, dtype=torch.bfloat16, device=dev)
    vec_buf = torch.empty(NI, VEC_F32, dtype=torch.float32, device=dev)
    flags = torch.zeros(NI, dtype=torch.int32, device=dev)
    maps_thd = {n: _encode_thd(t.view(-1), T, H) for n, t in (("q", q), ("k", k), ("v", v), ("g", g), ("o", output))}
    m_kbar = _encode_tile(kbar_buf, C, D, NI)
    m_qt = _encode_tile(qt_buf, C, D, NI)
    m_w1 = _encode_tile(w1_buf, D, C, NI)
    m_t1 = _encode_tile(t1_buf, C, C, NI)
    m_aqk = _encode_tile(aqk_buf, C, C, NI)
    n_pairs = H // hpc
    concurrent = CONCURRENT and sm_count > n_pairs + 8
    ctas_front = max(1, min(sm_count - n_pairs if concurrent else sm_count, NI))
    ipc = (NI + ctas_front - 1) // ctas_front
    args1 = (
        q.view(-1), k.view(-1), g.view(-1), beta.view(-1), a_log.view(-1), dt_bias.view(-1), vec_buf.view(-1),
        kbar_buf.view(-1), qt_buf.view(-1), t1_buf.view(-1), aqk_buf.view(-1), w1_buf.view(-1), flags,
        maps_thd["q"].ptr, maps_thd["k"].ptr, maps_thd["g"].ptr,
        0, int(NI), int(ctas_front), int(ipc), 1 if (concurrent or os.environ.get("KDA_SIGNAL_ALWAYS") == "1") else 0,
    )
    args2_head = (
        v.view(-1), init_c.view(-1), final_state.view(-1), output.view(-1), vec_buf.view(-1),
        kbar_buf.view(-1), qt_buf.view(-1), t1_buf.view(-1), aqk_buf.view(-1), w1_buf.view(-1),
        maps_thd["v"].ptr, m_kbar.ptr, m_qt.ptr, m_t1.ptr, m_aqk.ptr, m_w1.ptr, maps_thd["o"].ptr, flags,
        float(data["scale"]), int(NC),
    )
    keep = (maps_thd, m_kbar, m_qt, m_w1, m_t1, m_aqk, a_log, dt_bias, init_c,
            kbar_buf, qt_buf, w1_buf, t1_buf, aqk_buf, vec_buf, flags)
    s_lo = torch.cuda.Stream(device=dev)
    s_hi = torch.cuda.Stream(device=dev, priority=-1)
    use_graph = concurrent and os.environ.get("KDA_GRAPH", "1") == "1"
    conv = (lambda a: tvm_ffi.from_dlpack(a) if isinstance(a, torch.Tensor) else a)
    a1 = tuple(conv(a) for a in args1)
    a2 = tuple(conv(a) for a in args2_head) + (0, 1)
    a2_seq = tuple(conv(a) for a in args2_head) + (int(NI), 1)
    events = [torch.cuda.Event() for _ in range(4)]
    state = {"graph": None}

    def launch_pair():
        """Launch the fixed-path pair in the head-count-specific order, then join on s_lo."""
        ev_fork, ev_join = events[0], events[1]
        ev_fork.record(s_lo)
        s_hi.wait_event(ev_fork)
        if H == 64:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(s_hi)):
                exe2(*a2)
            with tvm_ffi.use_torch_stream(torch.cuda.stream(s_lo)):
                exe1(*a1)
        else:
            with tvm_ffi.use_torch_stream(torch.cuda.stream(s_lo)):
                exe1(*a1)
            with tvm_ffi.use_torch_stream(torch.cuda.stream(s_hi)):
                exe2(*a2)
        ev_join.record(s_hi)
        s_lo.wait_event(ev_join)

    def run():
        cur = torch.cuda.current_stream(dev)
        if not concurrent:
            exe1(*a1)
            exe2(*a2_seq)
            return
        ev0, ev1 = events[2], events[3]
        ev0.record(cur)
        s_lo.wait_event(ev0)
        if state["graph"] is not None:
            with torch.cuda.stream(s_lo):
                state["graph"].replay()
        else:
            launch_pair()
        ev1.record(s_lo)
        cur.wait_event(ev1)

    run._keep = keep

    exe1(*a1)
    exe2(*a2)
    torch.cuda.synchronize(dev)
    if use_graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s_lo):
            launch_pair()
        torch.cuda.synchronize(dev)
        state["graph"] = g
    return run


'''
_canon = types.ModuleType("kda_split_concurrent_embedded")
exec(compile(_CANON_SRC, "kda_split_concurrent_embedded", "exec"), _canon.__dict__)


def setup(data, B, T, H):
    cu = data.get("cu_seqlens")
    if cu is None and T % 32 == 0 and H % 2 == 0 and os.environ.get("KDA_NO_SPLIT") is None:
        return _canon.setup(data, B, T, H)
    return fused_setup(data, B, T, H)
