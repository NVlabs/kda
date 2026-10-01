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

"""pkdc: m64 KDA chain kernel with CLUSTER-ALTERNATED PREP.

Derived from pkdx (the m64 one-CTA-per-V64-half kernel).  pkdx loses to
pkdw because both half-CTAs duplicate the whole prep pipeline: prep
delivery (~2.0us/chunk) binds the chain cadence while the halved m64
compute (~1.4us) idles.  pkdc launches the two V64-half CTAs of a chain
as a (2,1,1) CLUSTER and ALTERNATES prep between them:

  - CTA rank r preps only chunks with (c & 1) == r, in-place in its own
    arena slot (c//2) % 4 (slots 0-3, recurrence 8 chain-chunks so the
    ~10us prep serial fits the slot-free window).
  - After qk_full it PUSHES the consumer product block — arena bytes
    [0, 32768) = qd|kd|ft(+gT)|L|INV — plus the gt/bt table columns to
    the PEER's slot 4 via cp.async.bulk shared::cluster copies
    (33,408 B) that complete_tx on the peer's MB_QK+4 (the canonical
    publication mechanism: tx completion on the waited barrier gives
    the receiver cross-proxy visibility).
  - Slot 4 is the pure RECEIVE slot: no prep instance runs there; its
    V region (disjoint from the push range) is still filled by the
    local V loader every peer-parity chunk.
  - A FORWARDER warp (warp 8) recycles slot 4 and echoes delivery:
    per peer-parity chunk it waits the local MMA4b commit
    (MB_SFREE+4), re-arms MB_QK+4 with expect_tx, remote-arms the
    producer's MB_PSHFREE+(m%4) (slot free -> next push may launch),
    then waits MB_QK+4 and remote-arms MB_PDELIV+(m%4): the producer
    blocks on that echo before leaving its lap, which certifies its
    source slot is no longer read by the bulk engine
    (mbarrier-completion copies are invisible to bulk async-groups).
    All next-lap writes are ordered behind that wait via plw0 program
    order (TMAs), the unconditional GRAW gate, and the instance
    barrier (scan/decorations/tables/beta store).
  - Consumer/epilogue/loader walk ALL chunks with slot index
    own ? (c//2)%4 : 4 and per-recurrence barrier parities.

Everything else (TMEM map, MMA legs, epilogue, numerics) is pkdx
byte-for-byte; smem layout and total are unchanged (slot 4 and table
column 4 are repurposed, not added).  Registers: compute 168 /
others 48.
  grid (2, H, nseqs), cluster (2,1,1): one CTA per (seq, head,
  V64-half) chain; 1024 threads = 32 warps:
    warps 0-3   COMPUTE   state seed, per-chunk [master T2R -> bf16 INP
                          R2T -> x gt -> master R2T], MMA1 issue,
                          residual FMA + MMA3 issue, v* requantize +
                          MMA4 issue, final-state export
    warps 4-7   EPILOGUE  OUT T2R -> bf16 -> stmatrix (DSL StMatrix
                          atom, no inline PTX) -> smem SW128 tile ->
                          TMA store (2x4KB ping-pong); partial chunks
                          take a guarded scalar path
    warp 10     LOAD      V TMA ring
    warp 8      FORWARDER slot-4 recycle + PSHFREE remote arms
    warps 9,11, 28-31     donors
    warps 12-27 PREP      4 round-robin instances x 4 warps; instance i
                          owns OWN-parity chunks 8k+2i+rank and smem
                          stage i: raw TMAs,
                          gate activation + f32 cumsum, l2norm,
                          anchored decorations qd/kd/ki, warp-MMA grams,
                          (I+L)^-1 via hierarchical shuffle solve,
                          restore pass, ft.
  C = 32, 5 smem stages (stage == instance), raw mbarrier choreography
  (10 barrier groups + out_empty), TMEM 256 cols:
    INP bf16 [64,128] @0-63 | resid/v* bf16 [64,32] @0-15 (INP reuse)
    v* f32 @32-63 | master f32 @64-191 | OUT f32 @192-223 | U f32 @224
  MMA4 is the PR's signature merged accumulate: D@64 spans
  [master || OUT || 32 junk pad cols] with N=192 (junk lands on the
  dead U region; pad D columns are never read).

Numerics = the PR's split-anchor scheme at A = lb*log2e*16:
  decay = exp2(gcs2 - A); qd = q^*decay*scale, kd = k^*decay,
  ki = k^*exp2(A - gcs2)  (gram products telescope exactly);
  restore: qd *= 2^A, kd *= 2^A (absolute), kr = ki*rf[k] = ft;
  gt = exp2(gcs2[31]); q^/k^ rounded to bf16 post-l2norm (fla chain).
State master stays f32 in TMEM end-to-end (harness f32 contract).

Stage arena aliasing contract (per PR): regions written by prep after
the smem_free (MMA4 of chunk-5) wait, in this order per chunk:
  gate walker writes ALL gcs rows -> decorations read gcs, write
  qd/kd/ki (in place over raws) -> tables (beta/gt/rf) overwrite gcs
  rows 16-18 -> grams write L (gcs rows 8-11) + G^T (ft block2 = gcs
  rows 0-7) -> solve writes INV (gcs rows 12-15) -> restore rewrites
  qd/kd/ki(->kr) -> qk_full.  V TMA (gcs rows 20-27) is gated on
  qk_full; gcs row 31 stays live for gt/rf until qk_full.
"""
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.nvgpu import cpasync
from cutlass.address_space import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir as _ir
from cutlass._mlir.dialects import llvm as _llvm
from cutlass._mlir.dialects import nvvm as _nvvm
import cuda.bindings.driver as cuda_driver


@dsl_user_op
def _bulk_push_dsmem(dst_ptr, src_ptr, nbytes, mbar_ptr, peer_rank,
                     mbar_rank=None, *, loc=None, ip=None):
    """One cp.async.bulk.shared::cluster push of ``nbytes`` from local
    smem to the peer CTA's symmetric address, completing (tx bytes) on
    the mbarrier at ``mbar_ptr`` in CTA ``mbar_rank`` (default: the
    peer).  ``dst_ptr``/``mbar_ptr`` are this CTA's local addresses;
    both are mapa-mapped (the DSL's CopyBulkS2SOp maps only the
    destination, which mis-addresses the barrier — same emission
    pattern as cute.arch's store_async_dsmem).  Completing on the OWN
    rank's barrier doubles as the source-read-done signal: these
    mbarrier-completion copies are NOT tracked by bulk async-groups.
    """
    dsmem_ty = _llvm.PointerType.get(AddressSpace.dsmem)
    rank_ir = cutlass.Int32(peer_rank).ir_value(loc=loc, ip=ip)
    mrank = peer_rank if mbar_rank is None else mbar_rank
    mrank_ir = cutlass.Int32(mrank).ir_value(loc=loc, ip=ip)
    d_ptr = _nvvm.mapa(dsmem_ty, dst_ptr.to_llvm_ptr(loc=loc, ip=ip),
                       rank_ir, loc=loc, ip=ip)
    m_ptr = _nvvm.mapa(dsmem_ty, mbar_ptr.to_llvm_ptr(loc=loc, ip=ip),
                       mrank_ir, loc=loc, ip=ip)
    d32 = _llvm.ptrtoint(T.i32(), d_ptr, loc=loc, ip=ip)
    m32 = _llvm.ptrtoint(T.i32(), m_ptr, loc=loc, ip=ip)
    s32 = _llvm.ptrtoint(T.i32(), src_ptr.to_llvm_ptr(loc=loc, ip=ip),
                         loc=loc, ip=ip)
    n32 = cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip)
    _llvm.inline_asm(
        None, [d32, s32, n32, m32],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx"
        "::bytes [$0], [$1], $2, [$3];",
        "r,r,r,r", has_side_effects=True, is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

D = 128
C = 32
BV = 64
NFT = 192
STAGES = 5
THREADS = 1024
TMEM_COLS = 256
STAGE_BYTES = 40960
STAGE_ELTS = STAGE_BYTES // 2
STAGE_F32 = STAGE_BYTES // 4
OFF_QD = 0
OFF_KD = 8192
OFF_FT = 16384
OFF_GCS = 24576
OFF_LW = OFF_GCS + 4096
OFF_INV = OFF_GCS + 6144
OFF_GT = OFF_GCS + 8192
OFF_RF = OFF_GCS + 8704
OFF_BT = OFF_GCS + 9344
OFF_V = OFF_GCS + 10240

MB_QK = 0
MB_GRAW = 5
MB_QKRAW = 10
MB_VFULL = 15
MB_VFREE = 20
MB_SFREE = 25
MB_RFREE = 30
MB_OOUT = 40
MB_U2ACC = 50
MB_OFIN = 55
MB_TAB = 45
MB_FIN = 60
MB_OEMPTY = 65
MB_PSHFREE = 66
MB_PDELIV = 70

N_OWN = 4
RECV = 4
PUSH_ARENA = 32768
TX_BYTES = PUSH_ARENA + 512 + 128


def _exp2f(x):
    return cute.math.exp2(x, fastmath=True)


def _tanhf(x):
    return cute.math.tanh(x, fastmath=True)


@cute.struct
class PkdStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 74]
    tmem_holding: cutlass.Int32


@cute.kernel
def _pkd(
    mma_1: cute.TiledMma, mma_3: cute.TiledMma, mma_4a: cute.TiledMma,
    mma_4b: cute.TiledMma,
    tma_q: cute.CopyAtom, mQ: cute.Tensor,
    tma_k: cute.CopyAtom, mK: cute.Tensor,
    tma_g: cute.CopyAtom, mG: cute.Tensor,
    tma_v: cute.CopyAtom, mV: cute.Tensor,
    tma_o: cute.CopyAtom, mO: cute.Tensor,
    q: cute.Tensor, k: cute.Tensor, g: cute.Tensor, beta: cute.Tensor,
    a_log: cute.Tensor, dt_bias: cute.Tensor,
    out_raw: cute.Tensor,
    state0: cute.Tensor, stateT: cute.Tensor,
    cu: cute.Tensor, perm: cute.Tensor,
    have_state: cutlass.Int32, h0: cutlass.Int32,
    scale: cutlass.Float32, lb2: cutlass.Float32,
    lay_qd: cute.ComposedLayout, lay_inv: cute.ComposedLayout,
    lay_ft: cute.ComposedLayout, lay_ov: cute.ComposedLayout,
    H_: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    vs_idx, hraw, seq_raw = cute.arch.block_idx()
    # head-range base (hybrid tail routing) + LPT sequence dispatch order
    hidx = hraw + h0
    seq_idx = cute.arch.make_warp_uniform(perm[seq_raw])
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx & 31
    v_off = vs_idx * BV

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(PkdStorage)
    arena = smem.allocate_tensor(
        cutlass.Int8, cute.make_layout((STAGES * STAGE_BYTES,)), 1024)
    s_out = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((2 * C * BV,)), 1024)
    s_dtb = smem.allocate_tensor(cutlass.Float32, cute.make_layout((D,)), 16)
    s_gt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((D, STAGES), stride=(1, D)), 16)
    # rf row stride padded D+1 -> 136 (16B-aligned rows/stages so the
    # restore's vec8 rf slices autovectorize to LDS.128 pairs).  Shape
    # padded to the full 136 stride so the 544B per-column push into
    # column 4 stays inside the allocation.
    s_rf5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((136, STAGES), stride=(1, 136)), 16)
    s_bt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((C, STAGES), stride=(1, C)), 16)

    ar0 = arena.iterator

    # ---- multi-stage canonical operand views (stage stride = arena) ----
    p_qd = cute.recast_ptr(ar0 + OFF_QD, lay_qd.inner, dtype=cutlass.BFloat16)
    t_bqd = cute.make_tensor(p_qd, cute.make_layout(
        ((32, 16), 1, (4, 2), STAGES),
        stride=((64, 1), 0, (16, 2048), STAGE_ELTS)))
    p_kd = cute.recast_ptr(ar0 + OFF_KD, lay_qd.inner, dtype=cutlass.BFloat16)
    t_bkd = cute.make_tensor(p_kd, cute.make_layout(
        ((32, 16), 1, (4, 2), STAGES),
        stride=((64, 1), 0, (16, 2048), STAGE_ELTS)))
    p_inv = cute.recast_ptr(ar0 + OFF_INV, lay_inv.inner, dtype=cutlass.BFloat16)
    t_binv = cute.make_tensor(p_inv, cute.make_layout(
        ((32, 16), 1, 2, STAGES),
        stride=((32, 1), 0, 16, STAGE_ELTS)))
    p_ft = cute.recast_ptr(ar0 + OFF_FT, lay_ft.inner, dtype=cutlass.BFloat16)
    t_bfta = cute.make_tensor(p_ft, cute.make_layout(
        (((64, 2), 16), 1, 2, STAGES),
        stride=(((1, 2048), 64), 0, 1024, STAGE_ELTS)))
    t_bftb = cute.make_tensor(p_ft + 4096, cute.make_layout(
        ((32, 16), 1, 2, STAGES),
        stride=((1, 64), 0, 1024, STAGE_ELTS)))

    b_qd = mma_1.make_fragment_B(t_bqd)
    b_kd = mma_1.make_fragment_B(t_bkd)
    b_inv = mma_3.make_fragment_B(t_binv)
    b_fta = mma_4a.make_fragment_B(t_bfta)
    b_ftb = mma_4b.make_fragment_B(t_bftb)

    # logical fill views (pre-swizzle affine over swizzled pointers);
    # vec8-sliceable forms: (row, 8elt, (seg8, blk), stage)
    v_qd8 = cute.make_tensor(p_qd, cute.make_layout(
        (C, 8, 8, 2, STAGES), stride=(64, 1, 8, 2048, STAGE_ELTS)))
    v_kd8 = cute.make_tensor(p_kd, cute.make_layout(
        (C, 8, 8, 2, STAGES), stride=(64, 1, 8, 2048, STAGE_ELTS)))
    v_ki8 = cute.make_tensor(p_ft, cute.make_layout(
        (C, 8, 8, 2, STAGES), stride=(64, 1, 8, 2048, STAGE_ELTS)))
    v_gT = cute.make_tensor(p_ft + 4096, cute.make_layout(
        (C, C, STAGES), stride=(1, 64, STAGE_ELTS)))
    v_inv = cute.make_tensor(p_inv, cute.make_layout(
        (C, C, STAGES), stride=(32, 1, STAGE_ELTS)))
    v_graw = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_QD, dtype=cutlass.BFloat16),
        cute.make_layout((C, D, STAGES), stride=(D, 1, STAGE_ELTS)))
    v_graw8 = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_QD, dtype=cutlass.BFloat16),
        cute.make_layout((C, 8, 16, STAGES), stride=(D, 1, 8, STAGE_ELTS)))
    s_dtb8 = cute.make_tensor(
        s_dtb.iterator, cute.make_layout((16, 8), stride=(8, 1)))
    p_gcs = cute.recast_ptr(ar0 + OFF_GCS, dtype=cutlass.Float32)
    v_gcs = cute.make_tensor(p_gcs, cute.make_layout(
        (C, D, STAGES), stride=(D, 1, STAGE_F32)))
    v_gcs8 = cute.make_tensor(p_gcs, cute.make_layout(
        (C, 8, 16, STAGES), stride=(D, 1, 8, STAGE_F32)))
    v_lw = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_LW, dtype=cutlass.BFloat16),
        cute.make_layout((C, C, STAGES), stride=(32, 1, STAGE_ELTS)))
    v_gt = s_gt5
    v_rf = s_rf5
    v_rf8 = cute.make_tensor(s_rf5.iterator, cute.make_layout(
        (16, 8, STAGES), stride=(8, 1, 136)))
    v_bt = s_bt5
    # pair views for LDS.64 table reads (fragment quads share col pairs)
    v_gt2 = cute.make_tensor(s_gt5.iterator, cute.make_layout(
        (D // 2, 2, STAGES), stride=(2, 1, D)))
    v_bt2 = cute.make_tensor(s_bt5.iterator, cute.make_layout(
        (C // 2, 2, STAGES), stride=(2, 1, C)))
    # V is read only by scalar per-token loads (no MMA): raw row-major
    # staging keeps per-element address math static (no swizzle XORs)
    p_v = cute.recast_ptr(ar0 + OFF_V, dtype=cutlass.BFloat16)
    v_v = cute.make_tensor(p_v, cute.make_layout(
        (C, BV, STAGES), stride=(64, 1, STAGE_ELTS)))
    p_o = cute.recast_ptr(s_out.iterator, lay_ov.inner, dtype=cutlass.BFloat16)
    v_o = cute.make_tensor(p_o, cute.make_layout(
        (C, BV, 2), stride=(64, 1, C * BV)))

    # ---- mbarrier init ----
    mb = storage.mbar.data_ptr()
    if warp == 0:
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(STAGES):
                cute.arch.mbarrier_init(mb + MB_QK + i, 1)
                cute.arch.mbarrier_init(mb + MB_GRAW + i, 1)
                cute.arch.mbarrier_init(mb + MB_QKRAW + i, 1)
                cute.arch.mbarrier_init(mb + MB_VFULL + i, 1)
                cute.arch.mbarrier_init(mb + MB_VFREE + i, 4)
                cute.arch.mbarrier_init(mb + MB_SFREE + i, 1)
                cute.arch.mbarrier_init(mb + MB_RFREE + i, 1)
                cute.arch.mbarrier_init(mb + MB_OOUT + i, 1)
                cute.arch.mbarrier_init(mb + MB_U2ACC + i, 1)
                cute.arch.mbarrier_init(mb + MB_OFIN + i, 1)
                cute.arch.mbarrier_init(mb + MB_TAB + i, 1)
                cute.arch.mbarrier_init(mb + MB_FIN + i, 1)
            cute.arch.mbarrier_init(mb + MB_OEMPTY, 4)
            for i in cutlass.range_constexpr(N_OWN):
                cute.arch.mbarrier_init(mb + MB_PSHFREE + i, 1)
                cute.arch.mbarrier_init(mb + MB_PDELIV + i, 1)
        cute.arch.mbarrier_init_fence()
    if (warp >= 12) & (warp < 16):
        tl0 = tidx - 384
        if tl0 < D:
            s_dtb[tl0] = dt_bias[hidx, tl0]
    cute.arch.sync_threads()
    # peer barriers must be initialized before any remote arrive or push
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
    rank_i = vs_idx
    peer_i = vs_idx ^ 1

    alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem_holding.ptr, barrier_for_retrieve=alloc_bar)
    tmem.allocate(TMEM_COLS)
    tmem.wait_for_alloc()
    tmem.relinquish_alloc_permit()
    tmem_ptr = tmem.retrieve_ptr(cutlass.Int32)

    # ---- TMEM tensors ----
    thr1 = mma_1.get_slice(0)
    thr3 = mma_3.get_slice(0)
    thr4a = mma_4a.get_slice(0)
    thr4b = mma_4b.get_slice(0)
    inp_base = thr1.make_fragment_A(mma_1.partition_shape_A((BV, D)))
    t_inp = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), inp_base.layout)
    ri3_base = thr3.make_fragment_A(mma_3.partition_shape_A((BV, C)))
    # dv resid parks in the U window (f32 cols 224-239): U is dead after
    # the dv leg's T2R, and MMA4's junk pad overwrites it only after MMA3
    # consumed it (in-order pipe).  Keeping it out of INP cols 0-63 lets
    # the U-first MMA1 split commit MB_OOUT while the OUT group still
    # reads INP (the dv store no longer races that read).
    t_ri3 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.BFloat16), ri3_base.layout)
    t_riv = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), ri3_base.layout)
    ri4_base = thr4a.make_fragment_A(mma_4a.partition_shape_A((BV, C)))
    t_ri4 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), ri4_base.layout)
    acc32_base = thr1.make_fragment_C(mma_1.partition_shape_C((BV, C)))
    t_vst = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 32, dtype=cutlass.Float32), acc32_base.layout)
    t_out = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), acc32_base.layout)
    t_u = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.Float32), acc32_base.layout)
    d4a_base = thr4a.make_fragment_C(mma_4a.partition_shape_C((BV, D)))
    t_d4a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), d4a_base.layout)
    d4b_base = thr4b.make_fragment_C(mma_4b.partition_shape_C((BV, C)))
    t_d4b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), d4b_base.layout)
    mst_base = thr1.make_fragment_C(mma_1.partition_shape_C((BV, 64)))
    mst0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), mst_base.layout)
    inp_half_base = thr1.make_fragment_A(mma_1.partition_shape_A((BV, 64)))
    t_inp0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), inp_half_base.layout)

    ld64_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32)
    st64_atom = cute.make_copy_atom(
        tcgen05.St16x256bOp(tcgen05.Repetition.x8), cutlass.Float32)
    mst_ld = tcgen05.make_tmem_copy(ld64_atom, mst0)
    mst_st = tcgen05.make_tmem_copy(st64_atom, mst0)
    ld32_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32)
    u_ld = tcgen05.make_tmem_copy(ld32_atom, t_u)
    st_inp_atom = cute.make_copy_atom(
        tcgen05.St16x128bOp(tcgen05.Repetition.x8), cutlass.BFloat16)
    inp_st = tcgen05.make_tmem_copy(st_inp_atom, t_inp0)
    st_ri_atom = cute.make_copy_atom(
        tcgen05.St16x128bOp(tcgen05.Repetition.x4), cutlass.BFloat16)
    ri_st = tcgen05.make_tmem_copy(st_ri_atom, t_ri3)

    bos = cutlass.Int32(cu[seq_idx])
    eos = cutlass.Int32(cu[seq_idx + 1])
    seq_len = eos - bos
    t_tiles = (seq_len + C - 1) // C
    chain = seq_idx * H_ + hidx

    gQ = cute.flat_divide(cute.domain_offset((bos, 0, 0), mQ), (C, 1, D))
    gK = cute.flat_divide(cute.domain_offset((bos, 0, 0), mK), (C, 1, D))
    gG = cute.flat_divide(cute.domain_offset((bos, 0, 0), mG), (C, 1, D))
    gV = cute.flat_divide(cute.domain_offset((bos, 0, 0), mV), (C, 1, BV))
    gO = cute.flat_divide(cute.domain_offset((bos, 0, 0), mO), (C, 1, BV))

    # =====================================================================
    # COMPUTE warpgroup (warps 0-3)
    # =====================================================================
    if warp < 4:
        cute.arch.warpgroup_reg_alloc(168)
        mst_thr = mst_ld.get_slice(tidx)
        mst_sthr = mst_st.get_slice(tidx)
        mst_id = mst_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BV, 64))))
        r_mst = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        inp_thr = inp_st.get_slice(tidx)
        r_inp = cute.make_rmem_tensor(
            inp_thr.partition_S(
                thr1.partition_A(cute.make_identity_tensor((BV, 64)))).shape,
            cutlass.BFloat16)
        r_mstB = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        r_inpB = cute.make_rmem_tensor(r_inp.layout, cutlass.BFloat16)
        g2t = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
        u_thr = u_ld.get_slice(tidx)
        u_id = u_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BV, C))))
        r_u = cute.make_rmem_tensor(u_id.shape, cutlass.Float32)
        r_gt = cute.make_rmem_tensor(cute.make_layout((16,)), cutlass.Float32)
        r_bte = cute.make_rmem_tensor(u_id.shape, cutlass.Float32)
        r_vf = cute.make_rmem_tensor(u_id.shape, cutlass.BFloat16)
        ri_thr = ri_st.get_slice(tidx)
        r_ri = cute.make_rmem_tensor(
            ri_thr.partition_S(
                thr3.partition_A(cute.make_identity_tensor((BV, C)))).shape,
            cutlass.BFloat16)

        for half in cutlass.range_constexpr(2):
            msth = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 64 + half * 64, dtype=cutlass.Float32),
                mst_base.layout)
            if have_state == 1:
                # quad fragment (v59 law): elements [4g,4g+2)/[4g+2,4g+4)
                # are col-adjacent pairs at two rows -> LDG.64 slices
                for gq in cutlass.range_constexpr(8):
                    cq0 = half * 64 + mst_id[4 * gq][1]
                    for pp in cutlass.range_constexpr(2):
                        rr0 = mst_id[4 * gq + 2 * pp][0]
                        r2i = cute.make_tensor(
                            r_mst.iterator + 4 * gq + 2 * pp,
                            cute.make_layout((2,)))
                        g2i = cute.local_tile(
                            state0, (1, 1, 2),
                            (chain, v_off + rr0, cq0 >> 1))
                        cute.autovec_copy(g2i, r2i)
            else:
                for e in cutlass.range_constexpr(cute.size(r_mst)):
                    r_mst[e] = cutlass.Float32(0.0)
            cute.copy(mst_st, r_mst, mst_sthr.partition_D(msth))
        cute.arch.fence_view_async_tmem_store()

        cbar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        p_oe = cutlass.Int32(1)
        cso = cutlass.Int32(0)
        p_qk_o = cutlass.Int32(0)
        p_vf_o = cutlass.Int32(0)
        p_oo_o = cutlass.Int32(0)
        p_u2a_o = cutlass.Int32(0)
        p_fin_o = cutlass.Int32(0)
        p_qk_r = cutlass.Int32(0)
        p_vf_r = cutlass.Int32(0)
        p_oo_r = cutlass.Int32(0)
        p_u2a_r = cutlass.Int32(0)
        p_fin_r = cutlass.Int32(0)
        for t in cutlass.range(t_tiles):
            isr = (t ^ rank_i) & 1
            iso = 1 - isr
            csc = cso * iso + RECV * isr
            p_qk = p_qk_o * iso + p_qk_r * isr
            p_vf = p_vf_o * iso + p_vf_r * isr
            p_oo = p_oo_o * iso + p_oo_r * isr
            p_u2a = p_u2a_o * iso + p_u2a_r * isr
            p_fin = p_fin_o * iso + p_fin_r * isr
            cute.arch.mbarrier_wait(mb + MB_QK + csc, p_qk)
            # both master halves pipeline through ONE load fence / ONE
            # store fence (wide-rep x16 atoms fail in libNVVM; two x8
            # copies back-to-back pipeline in the TMEM unit).  INP (the
            # pre-decay bf16 render) stores immediately so MMA1's operand
            # is in flight while the decay multiply runs.
            msth0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32),
                mst_base.layout)
            msth1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32),
                mst_base.layout)
            inph0 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16),
                inp_half_base.layout)
            inph1 = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 32, dtype=cutlass.BFloat16),
                inp_half_base.layout)
            cute.copy(mst_ld, mst_thr.partition_S(msth0), r_mst)
            cute.copy(mst_ld, mst_thr.partition_S(msth1), r_mstB)
            cute.arch.fence_view_async_tmem_load()
            r_inp.store(r_mst.load().to(cutlass.BFloat16))
            r_inpB.store(r_mstB.load().to(cutlass.BFloat16))
            cute.copy(inp_st, r_inp, inp_thr.partition_D(inph0))
            cute.copy(inp_st, r_inpB, inp_thr.partition_D(inph1))
            for gq in cutlass.range_constexpr(8):
                cq = mst_id[4 * gq][1]
                cute.autovec_copy(v_gt2[(cq >> 1, None, csc)], g2t)
                for e2 in cutlass.range_constexpr(4):
                    r_mst[4 * gq + e2] = r_mst[4 * gq + e2] * g2t[e2 & 1]
                cute.autovec_copy(v_gt2[((cq + 64) >> 1, None, csc)], g2t)
                for e2 in cutlass.range_constexpr(4):
                    r_mstB[4 * gq + e2] = r_mstB[4 * gq + e2] * g2t[e2 & 1]
            cute.copy(mst_st, r_mst, mst_sthr.partition_D(msth0))
            cute.copy(mst_st, r_mstB, mst_sthr.partition_D(msth1))
            cute.arch.fence_view_async_tmem_store()
            # self-issue: warpgroup rendezvous replaces the SINP
            # wake-hop; warp 0 issues MMA1 (U first, v66 order)
            cbar.arrive_and_wait()
            if warp == 0:
                cute.arch.mbarrier_wait(mb + MB_OEMPTY, p_oe)
                mma_1.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(D // 16):
                    cute.gemm(mma_1, t_u, t_inp[None, None, kb],
                              b_kd[None, None, kb, csc], t_u)
                    mma_1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mb + MB_OOUT + csc)
                mma_1.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(D // 16):
                    cute.gemm(mma_1, t_out, t_inp[None, None, kb],
                              b_qd[None, None, kb, csc], t_out)
                    mma_1.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mb + MB_RFREE + csc)
            p_oe ^= 1

            cute.arch.mbarrier_wait(mb + MB_VFULL + csc, p_vf)
            cute.arch.mbarrier_wait(mb + MB_OOUT + csc, p_oo)
            cute.copy(u_ld, u_thr.partition_S(t_u), r_u)
            cute.arch.fence_view_async_tmem_load()
            # beta expanded at load time (quad law: each pair lands
            # twice at [4a,4a+2) and [4a+2,4a+4)) + v staged into a
            # fragment so the dv math runs packed (f32x2/F2FP)
            for gq2 in cutlass.range_constexpr(4):
                tq = u_id[4 * gq2][1]
                b2a = cute.make_tensor(
                    r_bte.iterator + 4 * gq2, cute.make_layout((2,)))
                b2b = cute.make_tensor(
                    r_bte.iterator + 4 * gq2 + 2, cute.make_layout((2,)))
                cute.autovec_copy(v_bt2[(tq >> 1, None, csc)], b2a)
                cute.autovec_copy(v_bt2[(tq >> 1, None, csc)], b2b)
            for e in cutlass.range_constexpr(cute.size(r_u)):
                r_vf[e] = v_v[u_id[e][1], u_id[e][0], csc]
            dvv = (r_vf.load().to(cutlass.Float32) - r_u.load()) \
                * r_bte.load()
            r_ri.store(dvv.to(cutlass.BFloat16))
            cute.copy(ri_st, r_ri, ri_thr.partition_D(t_ri3))
            cute.arch.fence_view_async_tmem_store()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mb + MB_VFREE + csc)
            cbar.arrive_and_wait()
            if warp == 0:
                mma_3.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(C // 16):
                    cute.gemm(mma_3, t_vst, t_ri3[None, None, kb],
                              b_inv[None, None, kb, csc], t_vst)
                    mma_3.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(mb + MB_U2ACC + csc)

            cute.arch.mbarrier_wait(mb + MB_U2ACC + csc, p_u2a)
            cute.copy(u_ld, u_thr.partition_S(t_vst), r_u)
            cute.arch.fence_view_async_tmem_load()
            r_ri.store(r_u.load().to(cutlass.BFloat16))
            cute.copy(ri_st, r_ri, ri_thr.partition_D(t_riv))
            cute.arch.fence_view_async_tmem_store()
            cbar.arrive_and_wait()
            if warp == 0:
                mma_4a.set(tcgen05.Field.ACCUMULATE, True)
                mma_4b.set(tcgen05.Field.ACCUMULATE, True)
                for kb in cutlass.range_constexpr(C // 16):
                    cute.gemm(mma_4a, t_d4a, t_ri4[None, None, kb],
                              b_fta[None, None, kb, csc], t_d4a)
                with cute.arch.elect_one():
                    tcgen05.commit(mb + MB_FIN + csc)
                for kb in cutlass.range_constexpr(C // 16):
                    cute.gemm(mma_4b, t_d4b, t_ri4[None, None, kb],
                              b_ftb[None, None, kb, csc], t_d4b)
                with cute.arch.elect_one():
                    tcgen05.commit(mb + MB_OFIN + csc)
                    tcgen05.commit(mb + MB_SFREE + csc)

            cute.arch.mbarrier_wait(mb + MB_FIN + csc, p_fin)
            if isr == 0:
                cso += 1
                if cso == N_OWN:
                    cso = 0
                    p_qk_o ^= 1
                    p_vf_o ^= 1
                    p_oo_o ^= 1
                    p_u2a_o ^= 1
                    p_fin_o ^= 1
            else:
                p_qk_r ^= 1
                p_vf_r ^= 1
                p_oo_r ^= 1
                p_u2a_r ^= 1
                p_fin_r ^= 1

        for half in cutlass.range_constexpr(2):
            msth = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 64 + half * 64, dtype=cutlass.Float32),
                mst_base.layout)
            cute.copy(mst_ld, mst_thr.partition_S(msth), r_mst)
            cute.arch.fence_view_async_tmem_load()
            for gq in cutlass.range_constexpr(8):
                cq2 = half * 64 + mst_id[4 * gq][1]
                for pp in cutlass.range_constexpr(2):
                    rr2 = mst_id[4 * gq + 2 * pp][0]
                    r2o = cute.make_tensor(
                        r_mst.iterator + 4 * gq + 2 * pp,
                        cute.make_layout((2,)))
                    g2o = cute.local_tile(
                        stateT, (1, 1, 2),
                        (chain, v_off + rr2, cq2 >> 1))
                    cute.autovec_copy(r2o, g2o)

    # =====================================================================
    # EPILOGUE warpgroup (warps 4-7)
    # =====================================================================
    if (warp >= 4) & (warp < 8):
        cute.arch.warpgroup_reg_dealloc(48)
        etx = tidx - 128
        out_thr = u_ld.get_slice(etx)
        o_id = out_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BV, C))))
        r_o = cute.make_rmem_tensor(o_id.shape, cutlass.Float32)
        r_ob = cute.make_rmem_tensor(o_id.shape, cutlass.BFloat16)
        stm_np = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4),
            cutlass.BFloat16)
        ep_bar = pipeline.NamedBarrier(barrier_id=6, num_threads=128)
        # stmatrix.x4.trans lane addressing (PR m64 formula; XOR =
        # Sw(3,4,3) in element space, matching the OUT TMA box)
        elw = etx >> 5
        lne = etx & 31
        mtx = lne >> 3
        row8 = lne & 7
        so_base = cutlass.Int32(s_out.iterator.toint())

        ping = cutlass.Int32(0)
        cse_o = cutlass.Int32(0)
        pe_fin_o = cutlass.Int32(0)
        pe_fin_r = cutlass.Int32(0)
        for t in cutlass.range(t_tiles):
            cc_t = seq_len - t * C
            isr_e = (t ^ rank_i) & 1
            cse = cse_o * (1 - isr_e) + RECV * isr_e
            pe_fin = pe_fin_o * (1 - isr_e) + pe_fin_r * isr_e
            cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)
            cute.copy(u_ld, out_thr.partition_S(t_out), r_o)
            cute.arch.fence_view_async_tmem_load()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mb + MB_OEMPTY)
            if cc_t >= C:
                if warp == 4:
                    if t >= 2:
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                ep_bar.arrive_and_wait()
                # PTX-free stmatrix: DSL StMatrix atom over the same
                # lane addresses (probe scripts/stm_probe.py = byte-
                # identical to the inline-PTX block)
                r_ob.store(r_o.load().to(cutlass.BFloat16))
                for tg in cutlass.range_constexpr(2):
                    dim_base = elw * 16 + (mtx & 1) * 8
                    ta = tg * 16 + (mtx >> 1) * 8 + row8
                    tp = ta >> 1
                    par = ta & 1
                    raw_col = ((dim_base & 63)
                               ^ ((tp & 3) << 4) ^ (par << 3)) \
                        + par * 64
                    addr = so_base + ping * 4096 \
                        + (tp * 128 + raw_col) * 2
                    e0 = tg * 8
                    s8 = cute.make_tensor(
                        r_ob.iterator + e0, cute.make_layout((8,)))
                    dst = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16, addr,
                            cute.AddressSpace.smem,
                            assumed_align=16),
                        cute.make_layout((8,)))
                    cute.copy(stm_np, s8, dst)
                cute.arch.fence_proxy("async.shared", space="cta")
                ep_bar.arrive_and_wait()
                if warp == 4:
                    f_o = cute.make_tensor(
                        p_o + ping * (C * BV),
                        cute.make_layout((C, 1, BV), stride=(64, 0, 1)))
                    o_s, o_g = cpasync.tma_partition(
                        tma_o, 0, cute.make_layout(1),
                        cute.group_modes(f_o, 0, 3), cute.group_modes(gO, 0, 3))
                    cute.copy(tma_o, o_s, o_g[(None, t, hidx, vs_idx)])
                    cute.arch.cp_async_bulk_commit_group()
                ping ^= 1
            else:
                r_ob.store(r_o.load().to(cutlass.BFloat16))
                for e in cutlass.range_constexpr(cute.size(r_o)):
                    tt2 = o_id[e][1]
                    if tt2 < cc_t:
                        out_raw[bos + t * C + tt2, hidx, v_off + o_id[e][0]] = \
                            r_ob[e]
            if isr_e == 0:
                cse_o += 1
                if cse_o == N_OWN:
                    cse_o = 0
                    pe_fin_o ^= 1
            else:
                pe_fin_r ^= 1
        if warp == 4:
            cute.arch.cp_async_bulk_wait_group(0)

    # =====================================================================
    # WG2: MMA (warp 9), LOAD (warp 10), donors (8, 11)
    # =====================================================================
    if (warp >= 8) & (warp < 12):
        cute.arch.warpgroup_reg_dealloc(48)
        if warp == 8:
            # FORWARDER: recycle the depth-1 receive slot and echo
            # delivery.  Per peer-parity chunk m: (1) after the local
            # MMA4b commit (slot 4 fully consumed), re-arm MB_QK+4 with
            # the push transaction count and tell the producer instance
            # (m%4 on the peer) the slot is free; (2) once MB_QK+4
            # completes (push m delivered), echo MB_PDELIV so the
            # producer can recycle its source slot.
            n_recv = (t_tiles + rank_i) >> 1
            p_sf4 = cutlass.Int32(1)
            p_qk4 = cutlass.Int32(0)
            mI = cutlass.Int32(0)
            for i in cutlass.range(n_recv):
                cute.arch.mbarrier_wait(mb + MB_SFREE + RECV, p_sf4)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mb + MB_QK + RECV, TX_BYTES)
                    cute.arch.mbarrier_arrive(
                        mb + MB_PSHFREE + (mI & 3),
                        peer_cta_rank_in_cluster=peer_i)
                cute.arch.mbarrier_wait(mb + MB_QK + RECV, p_qk4)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(
                        mb + MB_PDELIV + (mI & 3),
                        peer_cta_rank_in_cluster=peer_i)
                p_sf4 ^= 1
                p_qk4 ^= 1
                mI += 1
        if warp == 10:
            csl_o = cutlass.Int32(0)
            pl_vfree_o = cutlass.Int32(1)
            pl_vfree_r = cutlass.Int32(1)
            pl_qk = cutlass.Int32(0)
            for t in cutlass.range(t_tiles):
                isr_l = (t ^ rank_i) & 1
                csl = csl_o * (1 - isr_l) + RECV * isr_l
                pl_vfree = pl_vfree_o * (1 - isr_l) + pl_vfree_r * isr_l
                cute.arch.mbarrier_wait(mb + MB_VFREE + csl, pl_vfree)
                if isr_l == 0:
                    # own slots alias V over prep's gate scratch: wait
                    # local prep done.  Slot 4 V is disjoint from the
                    # push range: no QK gate needed.
                    cute.arch.mbarrier_wait(mb + MB_QK + csl, pl_qk)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mb + MB_VFULL + csl, C * BV * 2)
                f_v = cute.make_tensor(
                    p_v + csl * STAGE_ELTS,
                    cute.make_layout((C, 1, BV), stride=(64, 0, 1)))
                v_d, v_s = cpasync.tma_partition(
                    tma_v, 0, cute.make_layout(1),
                    cute.group_modes(f_v, 0, 3), cute.group_modes(gV, 0, 3))
                cute.copy(tma_v, v_s[(None, t, hidx, vs_idx)], v_d,
                          tma_bar_ptr=mb + MB_VFULL + csl)
                if isr_l == 0:
                    csl_o += 1
                    if csl_o == N_OWN:
                        csl_o = 0
                        pl_vfree_o ^= 1
                        pl_qk ^= 1
                else:
                    pl_vfree_r ^= 1

    # =====================================================================
    # PREP (warps 12-31): 5 instances x 4 warps; instance == stage
    # =====================================================================
    if warp >= 12:
        cute.arch.warpgroup_reg_dealloc(48)
        inst = (warp - 12) >> 2
        plw = (warp - 12) & 3
        ptl = plw * 32 + lane
        # instance i preps own-parity chunks 8k + 2i + rank in slot i
        n_own_pc = (t_tiles + 1 - rank_i) >> 1
        nn_i = n_own_pc - inst
        n_iters = (nn_i + N_OWN - 1) >> 2
        if nn_i <= 0:
            n_iters = cutlass.Int32(0)
        if inst >= N_OWN:
            n_iters = cutlass.Int32(0)
        peer32 = cutlass.Int32(peer_i)
        pp_pshf = cutlass.Int32(0)
        pp_pdone = cutlass.Int32(0)

        mma_a = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(
                cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
            cute.make_layout((1, 1, 1)))
        thr_a = mma_a.get_slice(lane)
        tIdA = thr_a.partition_C(cute.make_identity_tensor((16, 16)))
        tIdA16 = thr_a.partition_A(cute.make_identity_tensor((16, 16)))
        tIdB16 = thr_a.partition_B(cute.make_identity_tensor((16, 16)))
        # gram operands via native ldmatrix (probe-validated over the
        # swizzled canonical tiles; static base pointer + block/stage as
        # tensor modes sliced after partitioning = alignment provenance)
        atom_n = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.BFloat16)
        cpA_g = cute.make_tiled_copy_A(atom_n, mma_a)
        cpB_g = cute.make_tiled_copy_B(atom_n, mma_a)
        thrA_g = cpA_g.get_slice(lane)
        thrB_g = cpB_g.get_slice(lane)
        lay_blk = cute.make_layout(
            (16, (16, 4, 2), 2, STAGES),
            stride=(64, (1, 16, 2048), 1024, STAGE_ELTS))
        tKD_all = thrA_g.partition_S(cute.make_tensor(p_kd, lay_blk))
        tQD_all = thrA_g.partition_S(cute.make_tensor(p_qd, lay_blk))
        tKI_all = thrB_g.partition_S(cute.make_tensor(p_ft, lay_blk))

        ea = cute.math.exp(a_log[hidx], fastmath=True)
        lb2h = lb2 * 0.5
        anch = lb2 * 16.0
        konst2 = _exp2f(anch)

        w8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        qr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        kr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        g8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
        rf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)

        pp_rfree = cutlass.Int32(1)
        pp_graw = cutlass.Int32(0)
        pp_sfree = cutlass.Int32(1)
        pp_qkraw = cutlass.Int32(0)
        ibar = pipeline.NamedBarrier(barrier_id=7 + inst, num_threads=128)

        for it in cutlass.range(n_iters):
            ci = it * 8 + 2 * inst + rank_i
            r0 = bos + ci * C
            cc = cutlass.min(cutlass.Int32(C), seq_len - ci * C)
            full = seq_len >= (ci + 1) * C

            # -- phase 0: raw g/k TMA (qd/kd slots free after MMA2(ci-8);
            # the previous lap's PDONE wait already certified that its
            # push finished reading this slot) --
            if full:
                cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)
                if plw == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mb + MB_GRAW + inst, C * D * 2)
                    f_g = cute.make_tensor(
                        cute.recast_ptr(ar0 + (OFF_QD + inst * STAGE_BYTES),
                                        dtype=cutlass.BFloat16),
                        cute.make_layout((C, 1, D), stride=(D, 0, 1)))
                    g_d, g_s = cpasync.tma_partition(
                        tma_g, 0, cute.make_layout(1),
                        cute.group_modes(f_g, 0, 3), cute.group_modes(gG, 0, 3))
                    cute.copy(tma_g, g_s[(None, ci, hidx, 0)], g_d,
                              tma_bar_ptr=mb + MB_GRAW + inst)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mb + MB_QKRAW + inst, 2 * C * D * 2)
                    f_k = cute.make_tensor(
                        cute.recast_ptr(ar0 + (OFF_KD + inst * STAGE_BYTES),
                                        lay_qd.inner, dtype=cutlass.BFloat16),
                        cute.make_layout((C, 1, (16, 4, 2)),
                                         stride=(64, 0, (1, 16, 2048))))
                    k_d, k_s = cpasync.tma_partition(
                        tma_k, 0, cute.make_layout(1),
                        cute.group_modes(f_k, 0, 3), cute.group_modes(gK, 0, 3))
                    cute.copy(tma_k, k_s[(None, ci, hidx, 0)], k_d,
                              tma_bar_ptr=mb + MB_QKRAW + inst)
            else:
                # partial chunk: no g TMA; arm GRAW from plw0 so the
                # unconditional phase-2a wait stays ordered behind this
                # warp's previous-lap PDONE tail
                if plw == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mb + MB_GRAW + inst)

            # -- phase 1: everything else waits MMA4(ci-5) --
            cute.arch.mbarrier_wait(mb + MB_SFREE + inst, pp_sfree)
            if full:
                if plw == 0:
                    f_qr = cute.make_tensor(
                        cute.recast_ptr(ar0 + (OFF_FT + inst * STAGE_BYTES),
                                        lay_qd.inner, dtype=cutlass.BFloat16),
                        cute.make_layout((C, 1, (16, 4, 2)),
                                         stride=(64, 0, (1, 16, 2048))))
                    q_d, q_s = cpasync.tma_partition(
                        tma_q, 0, cute.make_layout(1),
                        cute.group_modes(f_qr, 0, 3), cute.group_modes(gQ, 0, 3))
                    cute.copy(tma_q, q_s[(None, ci, hidx, 0)], q_d,
                              tma_bar_ptr=mb + MB_QKRAW + inst)

            # beta: load + activate early (gmem latency hides behind the
            # gate phase), but the SMEM store is deferred past the first
            # ibar — that orders it behind plw0's PDONE wait so it cannot
            # overwrite the previous lap's pending table push.
            btv = cutlass.Float32(0.0)
            if plw == 2:
                if lane < cc:
                    bx = cutlass.Float32(beta[r0 + lane, hidx])
                    btv = 0.5 + 0.5 * _tanhf(0.5 * bx)

            # -- phase 2a: gate values, row-parallel (vec8 in/out; the
            # tanh chain has no serial dependency here).  The GRAW wait
            # is unconditional: partial chunks arm it from plw0's phase 0
            # (post-tail program order), so the gate's gcs writes can
            # never overwrite the previous lap's in-flight push source.
            cute.arch.mbarrier_wait(mb + MB_GRAW + inst, pp_graw)
            for wpg in cutlass.range(4):
                rwg = wpg * 8 + (ptl >> 4)
                sgg = ptl & 15
                if rwg < cc:
                    if full:
                        gg8 = v_graw8[(rwg, None, sgg, inst)]
                        cute.autovec_copy(gg8, qr)
                    else:
                        gg8m = cute.local_tile(g, (1, 1, 8),
                                               (r0 + rwg, hidx, sgg))
                        cute.autovec_copy(gg8m, qr)
                    dt8 = s_dtb8[(sgg, None)]
                    cute.autovec_copy(dt8, g8)
                    for u in cutlass.range_constexpr(8):
                        ga = ea * (cutlass.Float32(qr[u]) + g8[u])
                        g8[u] = lb2h * _tanhf(0.5 * ga) + lb2h
                else:
                    for u in cutlass.range_constexpr(8):
                        g8[u] = 0.0
                gd8 = v_gcs8[(rwg, None, sgg, inst)]
                cute.autovec_copy(g8, gd8)
            ibar.arrive_and_wait()
            if plw == 2:
                if lane < C:
                    v_bt[lane, inst] = btv
            # -- phase 2b: per-channel scan (independent column loads
            # feeding a pure FADD chain) --
            if ptl < D:
                acc = cutlass.Float32(0.0)
                for rw in cutlass.range_constexpr(C):
                    acc += v_gcs[rw, ptl, inst]
                    v_gcs[rw, ptl, inst] = acc
            ibar.arrive_and_wait()

            # -- phase 3: q/k load + l2norm + anchored decorations --
            if full:
                cute.arch.mbarrier_wait(mb + MB_QKRAW + inst, pp_qkraw)
            for wp in cutlass.range(4):
                rw2 = wp * 8 + (ptl >> 4)
                sg2 = ptl & 15
                if full:
                    q8s = v_ki8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                    cute.autovec_copy(q8s, qr)
                    k8s = v_kd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                    cute.autovec_copy(k8s, kr)
                else:
                    if rw2 < cc:
                        gq8 = cute.local_tile(q, (1, 1, 8), (r0 + rw2, hidx, sg2))
                        cute.autovec_copy(gq8, qr)
                        gk8 = cute.local_tile(k, (1, 1, 8), (r0 + rw2, hidx, sg2))
                        cute.autovec_copy(gk8, kr)
                    else:
                        for u in cutlass.range_constexpr(8):
                            qr[u] = cutlass.BFloat16(0.0)
                            kr[u] = cutlass.BFloat16(0.0)
                sq = cutlass.Float32(0.0)
                sk = cutlass.Float32(0.0)
                for u in cutlass.range_constexpr(8):
                    fq = cutlass.Float32(qr[u])
                    fk = cutlass.Float32(kr[u])
                    sq += fq * fq
                    sk += fk * fk
                for off in [1, 2, 4, 8]:
                    sq += cute.arch.shuffle_sync_bfly(sq, off)
                    sk += cute.arch.shuffle_sync_bfly(sk, off)
                rqn = cute.math.rsqrt(sq + 1e-6, fastmath=True)
                rkn = cute.math.rsqrt(sk + 1e-6, fastmath=True)
                g8s = v_gcs8[(rw2, None, sg2, inst)]
                cute.autovec_copy(g8s, g8)
                qhv = qr.load().to(cutlass.Float32) * rqn
                khv = kr.load().to(cutlass.Float32) * rkn
                gv8 = g8.load()
                decv = cute.math.exp2(gv8 - anch, fastmath=True)

                qr.store((qhv * decv * scale).to(cutlass.BFloat16))
                kr.store((khv * decv).to(cutlass.BFloat16))
                idecv = cute.math.exp2(anch - gv8, fastmath=True)
                w8.store((khv * idecv).to(cutlass.BFloat16))
                dq8 = v_qd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                cute.autovec_copy(qr, dq8)
                dk8 = v_kd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                cute.autovec_copy(kr, dk8)
                di8 = v_ki8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                cute.autovec_copy(w8, di8)
            # gt/rf (dedicated buffers; gcs row 31 stable since the
            # post-walker barrier -> rides the decoration phase)
            if ptl < D:
                gl_t = v_gcs[C - 1, ptl, inst]
                rfv = _exp2f(gl_t - anch)
                v_rf[ptl, inst] = rfv
                v_gt[ptl, inst] = rfv * konst2
            if ptl == 0:
                v_rf[D, inst] = _exp2f(anch)
            ibar.arrive_and_wait()

            # -- phase 5: grams (3 lower block-pairs on warps 0, 2, 3) --
            for WL in cutlass.range_constexpr(4):
                if cutlass.const_expr(WL != 1):
                    if plw == WL:
                        bi = WL >> 1
                        bj = WL & 1
                        Ck = mma_a.make_fragment_C(
                            mma_a.partition_shape_C((16, 16)))
                        Cq = mma_a.make_fragment_C(
                            mma_a.partition_shape_C((16, 16)))
                        Ck.fill(0.0)
                        Cq.fill(0.0)
                        fa = mma_a.make_fragment_A(
                            tKD_all[None, None, 0, 0, 0])
                        fb = mma_a.make_fragment_B(
                            tKI_all[None, None, 0, 0, 0])
                        fqa = mma_a.make_fragment_A(
                            tQD_all[None, None, 0, 0, 0])
                        for kb in cutlass.range_constexpr(D // 16):
                            cute.copy(atom_n,
                                      tKD_all[None, None, kb, bi, inst], fa)
                            cute.copy(atom_n,
                                      tKI_all[None, None, kb, bj, inst], fb)
                            cute.copy(atom_n,
                                      tQD_all[None, None, kb, bi, inst], fqa)
                            cute.gemm(mma_a, Ck, fa, fb, Ck)
                            cute.gemm(mma_a, Cq, fqa, fb, Cq)
                        for e in cutlass.range_constexpr(cute.size(Ck)):
                            crd = tIdA[e]
                            gi = bi * 16 + crd[0]
                            gj = bj * 16 + crd[1]
                            lv = cutlass.Float32(0.0)
                            if gj < gi:
                                lv = Ck[e] * v_bt[gi, inst]
                            v_lw[gi, gj, inst] = cutlass.BFloat16(lv)
                            mv = cutlass.Float32(0.0)
                            if gj <= gi:
                                mv = Cq[e]
                            v_gT[gi, gj, inst] = cutlass.BFloat16(mv)
                if cutlass.const_expr(WL == 1):
                    if plw == WL:
                        zr = lane >> 1
                        zc = 16 + (lane & 1) * 8
                        for u in cutlass.range_constexpr(8):
                            v_lw[zr, zc + u, inst] = cutlass.BFloat16(0.0)
                            v_gT[zr, zc + u, inst] = cutlass.BFloat16(0.0)
            ibar.arrive_and_wait()

            # -- phase 6: solve, PR-style hierarchy.  Level 1: four
            # 8x8 unit-lower inverses by in-register shuffle
            # elimination (zero LDS on the elimination chain);
            # level 2: both 16-blocks' -B^-1 C A^-1 combines batched
            # as block-diagonal [16,16] warp-MMAs. --
            if plw == 0:
                dgb = lane >> 3
                lid = lane & 7
                b8 = dgb * 8
                for c8 in cutlass.range_constexpr(8):
                    lv8 = cutlass.Float32(0.0)
                    if c8 < lid:
                        lv8 = cutlass.Float32(v_lw[b8 + lid, b8 + c8, inst])
                    if c8 == lid:
                        lv8 = cutlass.Float32(1.0)
                    g8[c8] = lv8
                for sr in cutlass.range_constexpr(7):
                    rs8 = cutlass.Float32(0.0) - g8[sr]
                    for pc in cutlass.range_constexpr(7):
                        if cutlass.const_expr(pc < sr):
                            pv8 = cute.arch.shuffle_sync(
                                g8[pc], b8 + sr)
                            if lid > sr:
                                g8[pc] = rs8 * pv8 + g8[pc]
                    if lid > sr:
                        g8[sr] = rs8
                for c8 in cutlass.range_constexpr(8):
                    v_inv[b8 + lid, b8 + c8, inst] = cutlass.BFloat16(g8[c8])
                # zero the upper-right 8x8 of each 16-block and the
                # 32-level upper-right [0:16,16:32)
                if dgb == 0:
                    for c8 in cutlass.range_constexpr(8):
                        v_inv[lid, 8 + c8, inst] = cutlass.BFloat16(0.0)
                if dgb == 2:
                    for c8 in cutlass.range_constexpr(8):
                        v_inv[16 + lid, 24 + c8, inst] = cutlass.BFloat16(0.0)
                halfl = lane >> 4
                coll = lane & 15
                if halfl == 0:
                    for i in cutlass.range_constexpr(16):
                        v_inv[i, 16 + coll, inst] = cutlass.BFloat16(0.0)
                cute.arch.sync_warp()
                # level 2: Y = diag(A1inv,B1inv) @ diag(C_A,C_B)
                Cy2 = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                Cy2.fill(0.0)
                fa2 = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                fb2 = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                for e in cutlass.range_constexpr(cute.size(fa2)):
                    ac = tIdA16[e]
                    va2 = cutlass.BFloat16(0.0)
                    if (ac[0] >> 3) == (ac[1] >> 3):
                        va2 = v_inv[8 + 16 * (ac[0] >> 3) + (ac[0] & 7),
                                    8 + 16 * (ac[0] >> 3) + (ac[1] & 7), inst]
                    fa2[e] = va2
                for e in cutlass.range_constexpr(cute.size(fb2)):
                    bc = tIdB16[e]
                    vb2 = cutlass.BFloat16(0.0)
                    if (bc[1] >> 3) == (bc[0] >> 3):
                        vb2 = v_lw[8 + 16 * (bc[1] >> 3) + (bc[1] & 7),
                                   16 * (bc[1] >> 3) + (bc[0] & 7), inst]
                    fb2[e] = vb2
                cute.gemm(mma_a, Cy2, fa2, fb2, Cy2)
                yA2 = thr_a.make_fragment_A(mma_a.partition_shape_A((16, 16)))
                for e in cutlass.range_constexpr(cute.size(Cy2)):
                    yA2[e] = cutlass.BFloat16(Cy2[e])
                fbt2 = thr_a.make_fragment_B(mma_a.partition_shape_B((16, 16)))
                for e in cutlass.range_constexpr(cute.size(fbt2)):
                    bc2 = tIdB16[e]
                    vt2 = cutlass.BFloat16(0.0)
                    if (bc2[1] >> 3) == (bc2[0] >> 3):
                        vt2 = v_inv[16 * (bc2[1] >> 3) + (bc2[1] & 7),
                                    16 * (bc2[1] >> 3) + (bc2[0] & 7), inst]
                    fbt2[e] = vt2
                Cc2 = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                Cc2.fill(0.0)
                cute.gemm(mma_a, Cc2, yA2, fbt2, Cc2)
                for e in cutlass.range_constexpr(cute.size(Cc2)):
                    crd = tIdA[e]
                    if (crd[0] >> 3) == (crd[1] >> 3):
                        v_inv[8 + 16 * (crd[0] >> 3) + (crd[0] & 7),
                              16 * (crd[0] >> 3) + (crd[1] & 7), inst] = \
                            cutlass.BFloat16(0.0 - Cc2[e])
                cute.arch.sync_warp()
            # (no barrier: the combine is warp 0's own program order; the
            #  restore reads only pre-gram-barrier data)

            # -- phase 7: off-diag combine (warp 0) || restore (warps 1-3) --
            # INV[16:,:16] = -(Tb @ Lc) @ Ta; fragments loaded elementwise
            # by identity coords (transposed dynamic-offset views reject
            # autovec: provenance law).
            if plw == 0:
                Cy = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                Cy.fill(0.0)
                fat = thr_a.make_fragment_A(
                    mma_a.partition_shape_A((16, 16)))
                fbl = thr_a.make_fragment_B(
                    mma_a.partition_shape_B((16, 16)))
                for e in cutlass.range_constexpr(cute.size(fat)):
                    ac = tIdA16[e]
                    fat[e] = v_inv[16 + ac[0], 16 + ac[1], inst]
                for e in cutlass.range_constexpr(cute.size(fbl)):
                    bc = tIdB16[e]
                    fbl[e] = v_lw[16 + bc[1], bc[0], inst]
                cute.gemm(mma_a, Cy, fat, fbl, Cy)
                yA = thr_a.make_fragment_A(
                    mma_a.partition_shape_A((16, 16)))
                for e in cutlass.range_constexpr(cute.size(Cy)):
                    yA[e] = cutlass.BFloat16(Cy[e])
                fbt = thr_a.make_fragment_B(
                    mma_a.partition_shape_B((16, 16)))
                for e in cutlass.range_constexpr(cute.size(fbt)):
                    bc2 = tIdB16[e]
                    fbt[e] = v_inv[bc2[1], bc2[0], inst]
                Cc = mma_a.make_fragment_C(mma_a.partition_shape_C((16, 16)))
                Cc.fill(0.0)
                cute.gemm(mma_a, Cc, yA, fbt, Cc)
                for e in cutlass.range_constexpr(cute.size(Cc)):
                    crd = tIdA[e]
                    v_inv[16 + crd[0], crd[1], inst] = \
                        cutlass.BFloat16(0.0 - Cc[e])
            else:
                rf128 = v_rf[D, inst]
                for wp in cutlass.range(6):
                    item = wp * 96 + (ptl - 32)
                    if item < C * 16:
                        rw3 = item >> 4
                        sg3 = item & 15
                        sq8 = v_qd8[(rw3, None, sg3 & 7, sg3 >> 3, inst)]
                        cute.autovec_copy(sq8, qr)
                        sk8 = v_kd8[(rw3, None, sg3 & 7, sg3 >> 3, inst)]
                        cute.autovec_copy(sk8, kr)
                        si8 = v_ki8[(rw3, None, sg3 & 7, sg3 >> 3, inst)]
                        cute.autovec_copy(si8, w8)
                        cute.autovec_copy(v_rf8[(sg3, None, inst)], rf8)
                        qr.store((qr.load().to(cutlass.Float32)
                                  * rf128).to(cutlass.BFloat16))
                        kr.store((kr.load().to(cutlass.Float32)
                                  * rf128).to(cutlass.BFloat16))
                        w8.store((w8.load().to(cutlass.Float32)
                                  * rf8.load()).to(cutlass.BFloat16))
                        cute.autovec_copy(qr, sq8)
                        cute.autovec_copy(kr, sk8)
                        cute.autovec_copy(w8, si8)
            cute.arch.fence_proxy("async.shared", space="cta")
            ibar.arrive_and_wait()
            if plw == 0:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_QK + inst)
                # PUSH: peer's slot 4 is free (forwarder-armed), ship the
                # product block + table columns; completion fires the
                # peer's MB_QK+RECV transaction count.
                cute.arch.mbarrier_wait(mb + MB_PSHFREE + inst, pp_pshf)
                with cute.arch.elect_one():
                    # tx completion on the PEER's MB_QK+4: the canonical
                    # publication mechanism (cross-proxy visibility for
                    # the receiving consumer's MMA and LDS reads)
                    _bulk_push_dsmem(
                        cute.recast_ptr(ar0 + RECV * STAGE_BYTES,
                                        dtype=cutlass.Int8),
                        cute.recast_ptr(ar0 + inst * STAGE_BYTES,
                                        dtype=cutlass.Int8),
                        PUSH_ARENA, mb + MB_QK + RECV, peer32)
                    _bulk_push_dsmem(
                        s_gt5.iterator + D * RECV,
                        s_gt5.iterator + D * inst,
                        D * 4, mb + MB_QK + RECV, peer32)
                    _bulk_push_dsmem(
                        s_bt5.iterator + C * RECV,
                        s_bt5.iterator + C * inst,
                        C * 4, mb + MB_QK + RECV, peer32)
                # PDELIV: the peer's forwarder echoes its QK+4 completion
                # back here.  Blocking on it certifies the pushes were
                # DELIVERED, hence the source slot is no longer being
                # read (mbarrier-completion bulk copies are invisible to
                # bulk async-groups, so this echo is the only
                # source-read-done signal).  All next-lap writes to this
                # slot are ordered behind this wait via plw0's program
                # order (TMAs) and GRAW/ibar (other warps).
                cute.arch.mbarrier_wait(mb + MB_PDELIV + inst, pp_pdone)
                pp_pdone ^= 1
                pp_pshf ^= 1
            pp_rfree ^= 1
            pp_graw ^= 1
            pp_sfree ^= 1
            pp_qkraw ^= 1

    cute.arch.sync_threads()
    # neither CTA may exit while the peer could still push into its smem
    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
    tmem.free(tmem_ptr)


@cute.jit
def _launch_pkd(
    q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, g: cute.Tensor,
    beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
    state0: cute.Tensor, stateT: cute.Tensor, out: cute.Tensor,
    cu: cute.Tensor, perm: cute.Tensor,
    have_state: cutlass.Int32,
    nseqs: cutlass.Int32,
    h0: cutlass.Int32, hcnt: cutlass.Int32,
    scale: cutlass.Float32, lb2: cutlass.Float32,
    H_: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
):
    mma_1 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BV, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ))
    mma_3 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BV, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ))
    mma_4 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BV, NFT, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    mma_4a = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BV, D, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    mma_4b = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BV, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    lay_qd = sm100_utils.make_smem_layout_b(mma_1, (BV, C, D), cutlass.BFloat16, 1)
    lay_inv = sm100_utils.make_smem_layout_b(mma_3, (BV, C, C), cutlass.BFloat16, 1)
    lay_ft = sm100_utils.make_smem_layout_b(mma_4, (BV, NFT, C), cutlass.BFloat16, 1)
    lay_ov = cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3), 0,
        cute.make_layout((C, 1, BV), stride=(64, 0, 1)))
    l3k = cute.make_composed_layout(
        lay_qd.inner, 0,
        cute.make_layout((C, 1, (16, 4, 2)), stride=(64, 0, (1, 16, 2048))))
    lg_raw = cute.make_layout((C, 1, D), stride=(D, 0, 1))
    tma_q, mQ = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), q, l3k, (C, 1, D))
    tma_k, mK = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), k, l3k, (C, 1, D))
    tma_g, mG = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), g, lg_raw, (C, 1, D))
    lv_raw = cute.make_layout((C, 1, BV), stride=(BV, 0, 1))
    tma_v, mV = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), v, lv_raw, (C, 1, BV))
    tma_o, mO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), out, lay_ov, (C, 1, BV))
    _pkd(mma_1, mma_3, mma_4a, mma_4b,
         tma_q, mQ, tma_k, mK, tma_g, mG, tma_v, mV, tma_o, mO,
         q, k, g, beta, a_log, dt_bias,
         out, state0, stateT, cu, perm, have_state, h0, scale, lb2,
         lay_qd, lay_inv, lay_ft, lay_ov, H_).launch(
        grid=(2, hcnt, nseqs), block=(THREADS, 1, 1),
        cluster=(2, 1, 1), stream=stream)
