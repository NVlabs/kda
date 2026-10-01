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

"""pkdp: PAIR-CHUNK m128 KDA persistent multi-chain kernel (v108).

Fork of pkdw that processes TWO 32-token chunks per compute cycle:
one master decay per pair (per-channel gt product), batched MMA1
U/O legs against the pair-start state using prep-side gamma-folded
decorations (qd_B/kd_B *= gt_A in the restore; ft_A *= gt_B in
place), and cross-chunk grams GBA = kd_B @ kr_A^T (u_B fix) and
QBA = qd_B @ kr_A^T (o_B fix) applied as mma_3-shaped legs of the
A-side MMA4 group.  Exact algebra (pcdev/proto_pair.py): pair relRMS
vs f64 == sequential relRMS (~0.005), zero 5e-2 tolerance violations.

TMEM (512 cols): INP bf16[128,128]@0-63 (ri4A@0-15, ri4B@16-31 reuse),
master f32@64-191, OUT-A@192-223, OUT-B@224-255, U-A@256-287,
U-B@288-319, ri3 bf16@320-335 (shared A/B), vst f32@336-367 (must not
alias INP: MMA3-A executes pipe-before the deferred O legs).

Cross grams live in the dead-L smem regions (OFF_LW) of the pair's
two stages: GBA in stage-B (own), QBA in stage-A, both through the
lay_inv swizzle so they feed mma_3 B-fragments directly.  Barrier
traffic stays one-arm-per-chunk-per-stage (new MB_UFIX group: the
A-group's post-fix commit that gates dv_B).  Odd chunk counts take a
solo tail identical to the pkdw per-chunk body.

--- original pkdw header ---
pkdw: m128 KDA persistent multi-chain kernel.  Based on pkdm: the
m128 variant of pkd — CuTe DSL imitation of FlashInfer PR#4262's
`flashkda_bf16_fused_m128` (cute_pkdm).  No inline PTX.

B300 v101 fixed-chain wide residual: the fixed-shape specialization
stages V through an MN-major, 128b-swizzled A-operand layout and views
the existing 32x32b TMEM round trips as (feature, token) tiles.  This
lets CuTe vectorize the V residual fetch while preserving the exact
TMEM address map and arithmetic order.  The raw feature-major path is
kept for varlen chains, where the extra transformed-copy setup loses at
frequent chain boundaries.  NCU: 585.8 -> 571.8 us on fixed H96,
eligible warps 1.075 -> 1.103, with unchanged registers/shared memory.

B300 v94 persistence: grid (G,) with G = min(nseq*H, SMs); each CTA
walks a host-computed LPT list of chains (soff/schain).  The 5-stage
prep ring, TMEM state windows, and every mbarrier parity run
CONTINUOUSLY across chain boundaries — prep for chain k+1's first
chunks fills during chain k's tail, collapsing the per-chain
fill/drain (~10us) to a ~1us state export+seed.  Per-chain state is
re-derived per role (dt_bias reads moved smem->gmem for per-chain
hidx); the final-state export splits by master ownership (compute
windows 0-1, epilogue 2-3).

B300 v93 decay-split: the per-chunk master decay ([128,128] f32 TMEM
round trip + bf16 INP production, formerly ~700ns serial in the
compute warpgroup) is split by K-halves: compute owns windows 0-1
(unchanged full-SSA 2-deep ping-pong), the EPILOGUE warpgroup owns
windows 2-3 as four [128,16] half-window copies (x16/x8 atoms) run
right after its OFIN(t) wait — i.e. during chunk t+1's decay window,
before it exports OUT(t) (safe: OUT stays in TMEM until MMA1(t+1)'s
OEMPTY-gated leg).  Rendezvous before MMA1 issue is the arrive-only
MB_INP mbarrier (count 8 = compute 4 + epilogue 4); only warp 0
blocks on it.  OEMPTY is waited between the two MMA1 legs (it only
guards the t_out overwrite).  Registers: compute 152 / epilogue 88 /
wg2 32 / prep 5x48.

One CTA per (seq, head) owns the WHOLE chain: all four MMAs at M=128,
the full [128,128] f32 state resident in TMEM (4 x [128,32] column
windows, uniform 32x32b fragments: each thread owns one v-row with
col-sequential elements -> static vec8 table reads), V/OUT tiles full
[32,128] width.  The prep pipeline, solve, and barrier choreography are
byte-identical to pkd (v-independent); the m64 grid's x2 prep
duplication disappears and the CTA count halves (h96: 96 CTAs = one
wave).  V moves into gcs rows 16-31 (row 31 dies after the decoration
phase reads it; V is qk_full-gated).

Structure derived from the PR (docs/pr4262_flashkda_analysis.md) with
one measured departure (v71 SELF-ISSUE, +3%): there is NO dedicated MMA
warp — the compute warpgroup issues its own UTCHMMA groups after each
leg's R2T fence (intra-warpgroup NamedBarrier rendezvous, warp 0
issues + commits; SINP/UINP/U2INP hops deleted).  Registers: compute
192 / epilogue 48 / wg2 32 / prep 48 (v75 re-sweep of the full
512-unit budget after warp 9 became a donor).
  grid (H, nseqs): one CTA per (seq, head) chain;
  1024 threads = 32 warps:
    warps 0-3   COMPUTE   state seed, per-chunk [master T2R -> bf16 INP
                          R2T -> x gt -> master R2T], MMA1 issue,
                          residual FMA + MMA3 issue, v* requantize +
                          MMA4 issue, final-state export
    warps 4-7   EPILOGUE  OUT T2R -> bf16 -> stmatrix (DSL StMatrix
                          atom, no inline PTX) -> smem SW128 tile ->
                          TMA store (2x8KB ping-pong); partial chunks
                          take a guarded scalar path
    warp 10     LOAD      V TMA ring
    warps 8,9,11          donors
    warps 12-31 PREP      5 round-robin instances x 4 warps; instance i
                          owns chunks 5k+i and smem stage i: raw TMAs,
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
State master stays f32 in TMEM end-to-end (harness f32 contract).  Split
chains may use a bf16-only internal transit image between producer and
consumer CTAs; the consumer converts it back while seeding its f32 master.

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
import cuda.bindings.driver as cuda_driver

D = 128
C = 32
BM = 128
NFT = 192
STAGES = 5
THREADS = 1024
TMEM_COLS = 512
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
OFF_V = OFF_GCS + 8192

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
MB_INP = 35
MB_FIN = 60
MB_OEMPTY = 65
MB_UFIX = 66
MB_OEMPB = 71
MB_RFK = 72
MB_XG = 77
MB_XGM = 78


def _exp2f(x):
    return cute.math.exp2(x, fastmath=True)


def _tanhf(x):
    return cute.math.tanh(x, fastmath=True)


@cute.struct
class PkdStorage:
    mbar: cute.struct.MemRange[cutlass.Int64, 79]
    tmem_holding: cutlass.Int32


@cute.kernel
def _pkd(
    mma_1: cute.TiledMma, mma_3: cute.TiledMma, mma_4a: cute.TiledMma,
    mma_4b: cute.TiledMma, mma_xg: cute.TiledMma,
    tma_q: cute.CopyAtom, mQ: cute.Tensor,
    tma_k: cute.CopyAtom, mK: cute.Tensor,
    tma_g: cute.CopyAtom, mG: cute.Tensor,
    tma_v: cute.CopyAtom, mV: cute.Tensor,
    tma_o: cute.CopyAtom, mO: cute.Tensor,
    tma_e: cute.CopyAtom, mE: cute.Tensor,
    q: cute.Tensor, k: cute.Tensor, g: cute.Tensor, beta: cute.Tensor,
    a_log: cute.Tensor, dt_bias: cute.Tensor,
    out_raw: cute.Tensor,
    state0: cute.Tensor, stateT: cute.Tensor,
    cu: cute.Tensor, soff: cute.Tensor, schain: cute.Tensor,
    spt0: cute.Tensor, sptn: cute.Tensor,
    ssrc: cute.Tensor, sdst: cute.Tensor,
    midstate: cute.Tensor, mflags: cute.Tensor,
    expt: cute.Tensor, tprobe: cute.Tensor,
    have_state: cutlass.Int32,
    do_export: cutlass.Int32, export_seq: cutlass.Int32,
    nc2: cutlass.Int32, fepoch: cutlass.Int32,
    scale: cutlass.Float32, lb2: cutlass.Float32,
    lay_qd: cute.ComposedLayout, lay_inv: cute.ComposedLayout,
    lay_ft: cute.ComposedLayout, lay_v: cute.ComposedLayout,
    H_: cutlass.Constexpr[int],
    TPROBE_: cutlass.Constexpr[int] = 0,
    GATE2_: cutlass.Constexpr[int] = 0,
    FINAL_: cutlass.Constexpr[int] = 1,
    LITE_: cutlass.Constexpr[int] = 0,
):
    tidx, _, _ = cute.arch.thread_idx()
    slot, _, _ = cute.arch.block_idx()
    # persistent multi-chain: this CTA owns chains schain[k0:k1] (LPT)
    k0 = cute.arch.make_warp_uniform(soff[slot])
    k1 = cute.arch.make_warp_uniform(soff[slot + 1])
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx & 31

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(PkdStorage)
    arena = smem.allocate_tensor(
        cutlass.Int8, cute.make_layout((STAGES * STAGE_BYTES,)), 1024)
    s_out = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((2 * C * D,)), 1024)
    s_gt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((D, STAGES), stride=(1, D)), 16)
    # rf row stride padded D+1 -> 136 (16B-aligned rows/stages so the
    # restore's vec8 rf slices autovectorize to LDS.128 pairs)
    s_rf5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((D + 1, STAGES), stride=(1, 136)), 16)
    s_bt5 = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((C, STAGES), stride=(1, C)), 16)

    ar0 = arena.iterator

    # ---- multi-stage canonical operand views (stage stride = arena) ----
    p_qd = cute.recast_ptr(ar0 + OFF_QD, lay_qd.inner, dtype=cutlass.BFloat16)

    p_kd = cute.recast_ptr(ar0 + OFF_KD, lay_qd.inner, dtype=cutlass.BFloat16)

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

    t_bqd = cute.make_tensor(p_qd, cute.make_layout(
        ((32, 16), 1, (4, 2), STAGES),
        stride=((64, 1), 0, (16, 2048), STAGE_ELTS)))
    t_bkd = cute.make_tensor(p_kd, cute.make_layout(
        ((32, 16), 1, (4, 2), STAGES),
        stride=((64, 1), 0, (16, 2048), STAGE_ELTS)))
    b_qd = mma_1.make_fragment_B(t_bqd)
    b_kd = mma_1.make_fragment_B(t_bkd)
    b_inv = mma_3.make_fragment_B(t_binv)
    b_fta = mma_4a.make_fragment_B(t_bfta)
    b_ftb = mma_4b.make_fragment_B(t_bftb)
    # pair-mode cross grams on the tensor pipe: ONE SS MMA with a
    # compound-M A view over the adjacent qd||kd(csB) slots (rows 64-127
    # read ft/garbage and are never consumed) against B = kr(csA), giving
    # QBA in rows 0-31 and GBA in rows 32-63 of a [128,32] accumulator.
    # The epilogue requantizes both into the dead-L regions below.
    p_xg = cute.recast_ptr(ar0 + OFF_LW, lay_inv.inner, dtype=cutlass.BFloat16)
    t_bxg = cute.make_tensor(p_xg, cute.make_layout(
        ((32, 16), 1, 2, STAGES),
        stride=((32, 1), 0, 16, STAGE_ELTS)))
    b_xg = mma_3.make_fragment_B(t_bxg)
    v_xg = cute.make_tensor(p_xg, cute.make_layout(
        (C, C, STAGES), stride=(32, 1, STAGE_ELTS)))
    t_axg = cute.make_tensor(p_qd, cute.make_layout(
        (((32, 4), 16), 1, (4, 2), STAGES),
        stride=(((64, 4096), 1), 0, (16, 2048), STAGE_ELTS)))
    t_bkr = cute.make_tensor(p_ft, cute.make_layout(
        ((32, 16), 1, (4, 2), STAGES),
        stride=((64, 1), 0, (16, 2048), STAGE_ELTS)))
    a_xgm = mma_xg.make_fragment_A(t_axg)
    b_xgm = mma_xg.make_fragment_B(t_bkr)

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
        cute.recast_ptr(ar0 + OFF_KD, dtype=cutlass.BFloat16),
        cute.make_layout((C, D, STAGES), stride=(D, 1, STAGE_ELTS)))
    v_graw8 = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_KD, dtype=cutlass.BFloat16),
        cute.make_layout((C, 8, 16, STAGES), stride=(D, 1, 8, STAGE_ELTS)))
    v_graw2 = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_KD, dtype=cutlass.BFloat16),
        cute.make_layout((C, 2, 64, STAGES), stride=(D, 1, 2, STAGE_ELTS)))
    p_gcs = cute.recast_ptr(ar0 + OFF_GCS, dtype=cutlass.Float32)
    v_gcs = cute.make_tensor(p_gcs, cute.make_layout(
        (C, D, STAGES), stride=(D, 1, STAGE_F32)))
    v_gcs8 = cute.make_tensor(p_gcs, cute.make_layout(
        (C, 8, 16, STAGES), stride=(D, 1, 8, STAGE_F32)))
    v_gcs2 = cute.make_tensor(p_gcs, cute.make_layout(
        (C, 2, 64, STAGES), stride=(D, 1, 2, STAGE_F32)))
    v_lw = cute.make_tensor(
        cute.recast_ptr(ar0 + OFF_LW, dtype=cutlass.BFloat16),
        cute.make_layout((C, C, STAGES), stride=(32, 1, STAGE_ELTS)))
    v_gt = s_gt5
    v_rf = s_rf5
    v_rf8 = cute.make_tensor(s_rf5.iterator, cute.make_layout(
        (16, 8, STAGES), stride=(8, 1, 136)))
    v_bt = s_bt5
    # pair views for LDS.64 table reads (fragment quads share col pairs)
    v_gt8 = cute.make_tensor(s_gt5.iterator, cute.make_layout(
        (D // 8, 8, STAGES), stride=(8, 1, D)))
    v_bt8 = cute.make_tensor(s_bt5.iterator, cute.make_layout(
        (C // 8, 8, STAGES), stride=(8, 1, C)))
    # Fixed chains use an MN-major A-operand staging layout for wide
    # residual loads.  Varlen chains retain the measured-faster raw layout.
    p_v_wide = cute.recast_ptr(
        ar0 + OFF_V, lay_v.inner, dtype=cutlass.BFloat16)
    p_v_raw = cute.recast_ptr(ar0 + OFF_V, dtype=cutlass.BFloat16)
    v_v = cute.make_tensor(p_v_raw, cute.make_layout(
        (C, D, STAGES), stride=(D, 1, STAGE_ELTS)))
    p_o = cute.recast_ptr(s_out.iterator, lay_qd.inner, dtype=cutlass.BFloat16)
    v_o = cute.make_tensor(p_o, cute.make_layout(
        (C, 64, 2, 2), stride=(64, 1, 2048, C * D)))

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
                cute.arch.mbarrier_init(mb + MB_UFIX + i, 1)
                cute.arch.mbarrier_init(mb + MB_RFK + i, 1)
            cute.arch.mbarrier_init(mb + MB_INP, 8)
            cute.arch.mbarrier_init(mb + MB_OEMPTY, 4)
            cute.arch.mbarrier_init(mb + MB_OEMPB, 4)
            cute.arch.mbarrier_init(mb + MB_XG, 4)
            cute.arch.mbarrier_init(mb + MB_XGM, 1)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

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
    inp_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, D)))
    t_inp = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), inp_base.layout)
    # pair-frame image bf16(gt_A (.) h) for the B-side MMA1 legs @368-431
    t_inp_b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 368, dtype=cutlass.BFloat16), inp_base.layout)
    ri3_base = thr3.make_fragment_A(mma_3.partition_shape_A((BM, C)))
    # pair map: dv resid parks at fresh cols 320-335 (bf16), shared by
    # the A and B sub-chunks (MMA3-A is pipe-complete before ri3-B's
    # store, which waits the A-group's UFIX commit).
    t_ri3 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 320, dtype=cutlass.BFloat16), ri3_base.layout)
    ri4_base = thr4a.make_fragment_A(mma_4a.partition_shape_A((BM, C)))
    t_ri4 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), ri4_base.layout)
    t_ri4b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 16, dtype=cutlass.BFloat16), ri4_base.layout)
    # ri4-A viewed with mma_3's A-fragment layout for the cross-fix legs
    t_ri4m3 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), ri3_base.layout)
    acc32_base = thr1.make_fragment_C(mma_1.partition_shape_C((BM, C)))
    # vst must NOT alias INP (cols 32-63): MMA3-A executes pipe-BEFORE
    # the deferred O legs, which still read the full INP width
    t_vst = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 336, dtype=cutlass.Float32), acc32_base.layout)
    t_out = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), acc32_base.layout)
    t_out_b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.Float32), acc32_base.layout)
    t_u = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 256, dtype=cutlass.Float32), acc32_base.layout)
    t_u_b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 288, dtype=cutlass.Float32), acc32_base.layout)
    t_xgacc = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 432, dtype=cutlass.Float32), acc32_base.layout)

    d4a_base = thr4a.make_fragment_C(mma_4a.partition_shape_C((BM, D)))
    t_d4a = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), d4a_base.layout)
    d4b_base = thr4b.make_fragment_C(mma_4b.partition_shape_C((BM, C)))
    t_d4b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 192, dtype=cutlass.Float32), d4b_base.layout)
    t_d4b_b = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 224, dtype=cutlass.Float32), d4b_base.layout)
    mst_base = acc32_base  # [128,32] column-window template
    # bf16 window template for INP stores ([128,32] bf16 = 16 words)
    w32b_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, 32)))

    ld32_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    st32_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x32), cutlass.Float32)
    stwb_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.BFloat16)
    mst0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32), mst_base.layout)
    mst_ld = tcgen05.make_tmem_copy(ld32_atom, mst0)
    mst_st = tcgen05.make_tmem_copy(st32_atom, mst0)
    u_ld = mst_ld
    twb0 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), w32b_base.layout)
    w16b_st = tcgen05.make_tmem_copy(stwb_atom, twb0)
    u2f = cute.make_tensor(
        t_u.iterator,
        cute.make_layout(((BM, 1), (C, 1)),
                         stride=((65536, 0), (1, 0))))
    u2b = cute.make_tensor(
        twb0.iterator,
        cute.make_layout(((BM, 1), (16, 2)),
                         stride=((131072, 0), (1, 16))))
    u_wide_ld = tcgen05.make_tmem_copy(ld32_atom, u2f)
    u_wide_st = tcgen05.make_tmem_copy(stwb_atom, u2b)
    v_vt = cute.make_tensor(p_v_wide, lay_v.outer)
    # half-window ([128,16] channels) bf16 INP store template for wg2
    w8b_base = thr1.make_fragment_A(mma_1.partition_shape_A((BM, 16)))
    stwb8_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x8), cutlass.BFloat16)
    twb8 = cute.make_tensor(
        cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16), w8b_base.layout)
    w8b_st = tcgen05.make_tmem_copy(stwb8_atom, twb8)

    # =====================================================================
    # COMPUTE warpgroup (warps 0-3)
    # =====================================================================
    if warp < 4:
        cute.arch.warpgroup_reg_alloc(152)
        mst_thr = mst_ld.get_slice(tidx)
        mst_sthr = mst_st.get_slice(tidx)
        mst_id = mst_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BM, C))))
        r_mst = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        uw_ldthr = u_wide_ld.get_slice(tidx)
        uw_uid = uw_ldthr.partition_D(cute.make_identity_tensor((BM, C)))
        uw_v = uw_ldthr.partition_D(v_vt[None, 0, None, None])
        r_u_wide = cute.make_tensor(
            r_mst.iterator, cute.make_layout(uw_uid.shape))
        r_u_old = r_mst
        r_tab = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        r_gt = r_tab
        r_bt_wide = cute.make_tensor(
            r_tab.iterator, cute.make_layout(uw_uid.shape))
        r_bt_old = r_tab
        # bf16 INP staging fragment stored via bf16-typed St32x32b.x16
        # (no rmem pointer recast: O3 miscompiles aliased register views)
        w16b_thr = w16b_st.get_slice(tidx)
        w16b_id = w16b_thr.partition_S(
            thr1.partition_A(cute.make_identity_tensor((BM, 32))))
        rb32f = cute.make_rmem_tensor(w16b_id.shape, cutlass.BFloat16)
        r_mst2 = cute.make_rmem_tensor(mst_id.shape, cutlass.Float32)
        uw_sthr = u_wide_st.get_slice(tidx)
        uw_sid = uw_sthr.partition_S(cute.make_identity_tensor((BM, C)))
        rb_wide = cute.make_tensor(
            rb32f.iterator, cute.make_layout(uw_sid.shape))
        r_vv_old = cute.make_rmem_tensor(mst_id.shape, cutlass.BFloat16)
        r_vv_wide = cute.make_tensor(
            r_vv_old.iterator, cute.make_layout(uw_sid.shape))

        r_gtb = cute.make_rmem_tensor(
            cute.make_layout((8,)), cutlass.Float32)

        cbar = pipeline.NamedBarrier(barrier_id=2, num_threads=128)
        p_oe = cutlass.Int32(1)
        p_oeb = cutlass.Int32(1)
        p_inp = cutlass.Int32(0)
        csc = cutlass.Int32(0)
        p_qk = cutlass.Int32(0)
        p_vf = cutlass.Int32(0)
        p_oo = cutlass.Int32(0)
        p_u2a = cutlass.Int32(0)
        p_fin = cutlass.Int32(0)
        p_tab = cutlass.Int32(0)
        p_ufix = cutlass.Int32(0)
        p_rfr = cutlass.Int32(0)
        p_xgw = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            # piece-of-chain descriptor (v96 split): token window + state
            # routing.  Whole chains: pt0=0, ptn=seq len, src=dst=-1.
            bos = cutlass.Int32(cu[seq_idx]) \
                + cute.arch.make_warp_uniform(spt0[k0 + kx])
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            simp = cute.arch.make_warp_uniform(ssrc[k0 + kx])
            sexp = cute.arch.make_warp_uniform(sdst[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            if cutlass.const_expr(TPROBE_ == 1):
                if warp == 0:
                    with cute.arch.elect_one():
                        tprobe[slot * 64 + kx * 2] = cute.arch.globaltimer()
            # mid-chain seed: wait for the producer piece's full-count
            # release (256 per-thread arrivals x fepoch; B300 hazard memo:
            # EVERY consuming thread acquire-polls, per-thread releases)
            if simp >= 0:
                ftgt = fepoch * 256
                fcur = cute.arch.atomic_add(
                    mflags.iterator + simp, cutlass.Int32(0),
                    sem="acquire", scope="gpu")
                while fcur < ftgt:
                    fcur = cute.arch.atomic_add(
                        mflags.iterator + simp, cutlass.Int32(0),
                        sem="acquire", scope="gpu")
            # epilogue WG seeds + decays master windows 2-3
            for win in cutlass.range_constexpr(2):
                msth = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64 + win * 32, dtype=cutlass.Float32),
                    mst_base.layout)
                if simp >= 0:
                    vv0m = mst_id[0][0]
                    for jj in cutlass.range_constexpr(4):
                        r8i = cute.make_tensor(
                            r_mst.iterator + 8 * jj, cute.make_layout((8,)))
                        g8i = cute.local_tile(
                            midstate, (1, 1, 8), (simp, vv0m, win * 4 + jj))
                        if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16):
                            for ee in cutlass.range_constexpr(8):
                                r8i[ee] = cutlass.Float32(g8i[ee])
                        else:
                            cute.autovec_copy(g8i, r8i)
                else:
                    if have_state == 1:
                        # fragment = one row/thread x 32 sequential cols; the
                        # 32 scalar LDGs vectorize to 8-wide row slices
                        vv0 = mst_id[0][0]
                        for jj in cutlass.range_constexpr(4):
                            r8i = cute.make_tensor(
                                r_mst.iterator + 8 * jj, cute.make_layout((8,)))
                            g8i = cute.local_tile(
                                state0, (1, 1, 8), (chain, vv0, win * 4 + jj))
                            cute.autovec_copy(g8i, r8i)
                    else:
                        for e in cutlass.range_constexpr(cute.size(r_mst)):
                            r_mst[e] = cutlass.Float32(0.0)
                cute.copy(mst_st, r_mst, mst_sthr.partition_D(msth))
            cute.arch.fence_view_async_tmem_store()
            t_pairs = t_tiles // 2
            for tp in cutlass.range(t_pairs):
                csA = csc
                csB = csA + 1
                fB = cutlass.Int32(0)
                if csB == STAGES:
                    csB = cutlass.Int32(0)
                    fB = cutlass.Int32(1)
                pB_qk = p_qk ^ fB
                pB_tab = p_tab ^ fB
                pB_vf = p_vf ^ fB
                pB_u2a = p_u2a ^ fB
                pB_fin = p_fin ^ fB
                pB_ufix = p_ufix ^ fB
                pB_rfr = p_rfr ^ fB
                # single decay per pair: gt = gt_A * gt_B, gated on the
                # two table phases only (prep phase 3), not full MB_QK
                cute.arch.mbarrier_wait(mb + MB_TAB + csA, p_tab)
                cute.arch.mbarrier_wait(mb + MB_TAB + csB, pB_tab)
                m0 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32),
                    mst_base.layout)
                cute.copy(mst_ld, mst_thr.partition_S(m0), r_mst)
                for win in cutlass.range_constexpr(2):
                    rcur = r_mst if win % 2 == 0 else r_mst2
                    rnxt = r_mst2 if win % 2 == 0 else r_mst
                    msth = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 64 + win * 32,
                                        dtype=cutlass.Float32),
                        mst_base.layout)
                    twin = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + win * 16,
                                        dtype=cutlass.BFloat16),
                        w32b_base.layout)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(win < 1):
                        mnxt = cute.make_tensor(
                            cute.recast_ptr(tmem_ptr + 64 + (win + 1) * 32,
                                            dtype=cutlass.Float32),
                            mst_base.layout)
                        cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)
                    twinB = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 368 + win * 16,
                                        dtype=cutlass.BFloat16),
                        w32b_base.layout)
                    for gv in cutlass.range_constexpr(4):
                        g8d = cute.make_tensor(
                            r_gt.iterator + 8 * gv, cute.make_layout((8,)))
                        cute.autovec_copy(
                            v_gt8[(win * 4 + gv, None, csA)], g8d)
                    mv = rcur.load()
                    rb32f.store(mv.to(cutlass.BFloat16))
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))
                    mv2 = mv * r_gt.load()
                    rb32f.store(mv2.to(cutlass.BFloat16))
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twinB))
                    for gv in cutlass.range_constexpr(4):
                        g8d = cute.make_tensor(
                            r_gt.iterator + 8 * gv, cute.make_layout((8,)))
                        cute.autovec_copy(
                            v_gt8[(win * 4 + gv, None, csB)], g8d)
                    rcur.store(mv2 * r_gt.load())
                    cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))
                cute.arch.fence_view_async_tmem_store()
                # INP rendezvous (once per pair); warp 0 issues ONLY the
                # two U legs here — the O legs move behind MMA3-A so the
                # dv chain never queues behind the OEMPTY-gated overwrite
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_INP)
                if warp == 0:
                    cute.arch.mbarrier_wait(mb + MB_INP, p_inp)
                    cute.arch.mbarrier_wait(mb + MB_QK + csA, p_qk)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_u, t_inp[None, None, kb],
                                  b_kd[None, None, kb, csA], t_u)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OOUT + csA)
                        tcgen05.commit(mb + MB_RFK + csA)
                    cute.arch.mbarrier_wait(mb + MB_QK + csB, pB_qk)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_u_b, t_inp_b[None, None, kb],
                                  b_kd[None, None, kb, csB], t_u_b)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OOUT + csB)
                    mma_xg.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_xg, t_xgacc,
                                  a_xgm[None, None, kb, csB],
                                  b_xgm[None, None, kb, csA], t_xgacc)
                        mma_xg.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_XGM)
                p_inp ^= 1

                twri = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16),
                    w32b_base.layout)
                twri_b2 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 16, dtype=cutlass.BFloat16),
                    w32b_base.layout)
                twri_dv = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 320, dtype=cutlass.BFloat16),
                    w32b_base.layout)

                # ---- sub-chunk A ----
                cute.arch.mbarrier_wait(mb + MB_VFULL + csA, p_vf)
                cute.arch.mbarrier_wait(mb + MB_OOUT + csA, p_oo)
                if cutlass.const_expr(GATE2_ == 1):
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(u2f), r_u_wide)
                    cute.autovec_copy(
                        uw_v[None, None, None, csA], r_vv_wide)
                    for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                        r_bt_wide[ee] = v_bt[uw_uid[ee][1], csA]
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_u), r_u_old)
                    vv1 = mst_id[0][0]
                    for tt in cutlass.range_constexpr(C):
                        r_vv_old[tt] = v_v[tt, vv1, csA]
                    for gv2 in cutlass.range_constexpr(4):
                        b8d = cute.make_tensor(
                            r_bt_old.iterator + 8 * gv2,
                            cute.make_layout((8,)))
                        cute.autovec_copy(v_bt8[(gv2, None, csA)], b8d)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    dvv = (r_vv_wide.load().to(cutlass.Float32)
                           - r_u_wide.load()) * r_bt_wide.load()
                    rb_wide.store(dvv.to(cutlass.BFloat16))
                    twri_dv_w = cute.make_tensor(
                        twri_dv.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide,
                        uw_sthr.partition_D(twri_dv_w))
                else:
                    dvv = (r_vv_old.load().to(cutlass.Float32)
                           - r_u_old.load()) * r_bt_old.load()
                    rb32f.store(dvv.to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri_dv))
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_VFREE + csA)
                cbar.arrive_and_wait()
                if warp == 0:
                    mma_3.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_3, t_vst, t_ri3[None, None, kb],
                                  b_inv[None, None, kb, csA], t_vst)
                        mma_3.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_U2ACC + csA)
                    # deferred O legs: pipe-ordered after MMA3-A so the
                    # dv chain is not hostage to the epilogue's OUT reads;
                    # per-half OEMPTY barriers gate the overwrites
                    cute.arch.mbarrier_wait(mb + MB_OEMPTY, p_oe)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_out, t_inp[None, None, kb],
                                  b_qd[None, None, kb, csA], t_out)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_RFREE + csA)
                    cute.arch.mbarrier_wait(mb + MB_OEMPB, p_oeb)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_out_b, t_inp_b[None, None, kb],
                                  b_qd[None, None, kb, csB], t_out_b)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                p_oe ^= 1
                p_oeb ^= 1

                cute.arch.mbarrier_wait(mb + MB_U2ACC + csA, p_u2a)
                # ri4-A overwrites INP_A cols 0-15: the deferred O-A leg
                # must finish reading INP_A first (O-B reads INP_B)
                cute.arch.mbarrier_wait(mb + MB_RFREE + csA, p_rfr)
                if cutlass.const_expr(GATE2_ == 1):
                    t_vst_w = cute.make_tensor(t_vst.iterator, u2f.layout)
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(t_vst_w), r_u_wide)
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_vst), r_u_old)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    rb_wide.store(r_u_wide.load().to(cutlass.BFloat16))
                    twri_w = cute.make_tensor(twri.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide, uw_sthr.partition_D(twri_w))
                else:
                    rb32f.store(r_u_old.load().to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri))
                cute.arch.fence_view_async_tmem_store()
                cbar.arrive_and_wait()
                if warp == 0:
                    # cross grams + ft(A) fold are produced by the epilogue
                    # WG in its OFIN-A slack window; gate the fix legs.
                    # kd/qd(csB) slots release only now (gram operands).
                    cute.arch.mbarrier_wait(mb + MB_XG, p_xgw)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_RFK + csB)
                        tcgen05.commit(mb + MB_RFREE + csB)
                    # A-side MMA4 group: U-B fix first (gates dv-B), then
                    # O-B fix, OUT-A intra (early OFIN-A), master leg last
                    mma_3.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_3, t_u_b, t_ri4m3[None, None, kb],
                                  b_xg[None, None, kb, csB], t_u_b)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_UFIX + csB)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_3, t_out_b, t_ri4m3[None, None, kb],
                                  b_xg[None, None, kb, csA], t_out_b)
                    mma_4b.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_4b, t_d4b, t_ri4[None, None, kb],
                                  b_ftb[None, None, kb, csA], t_d4b)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OFIN + csA)
                    mma_4a.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_4a, t_d4a, t_ri4[None, None, kb],
                                  b_fta[None, None, kb, csA], t_d4a)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_FIN + csA)
                        tcgen05.commit(mb + MB_SFREE + csA)

                # ---- sub-chunk B ----
                cute.arch.mbarrier_wait(mb + MB_VFULL + csB, pB_vf)
                cute.arch.mbarrier_wait(mb + MB_UFIX + csB, pB_ufix)
                if cutlass.const_expr(GATE2_ == 1):
                    u2f_b = cute.make_tensor(t_u_b.iterator, u2f.layout)
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(u2f_b), r_u_wide)
                    cute.autovec_copy(
                        uw_v[None, None, None, csB], r_vv_wide)
                    for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                        r_bt_wide[ee] = v_bt[uw_uid[ee][1], csB]
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_u_b), r_u_old)
                    vv1b = mst_id[0][0]
                    for tt in cutlass.range_constexpr(C):
                        r_vv_old[tt] = v_v[tt, vv1b, csB]
                    for gv2 in cutlass.range_constexpr(4):
                        b8d = cute.make_tensor(
                            r_bt_old.iterator + 8 * gv2,
                            cute.make_layout((8,)))
                        cute.autovec_copy(v_bt8[(gv2, None, csB)], b8d)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    dvvb = (r_vv_wide.load().to(cutlass.Float32)
                            - r_u_wide.load()) * r_bt_wide.load()
                    rb_wide.store(dvvb.to(cutlass.BFloat16))
                    twri_dv_w2 = cute.make_tensor(
                        twri_dv.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide,
                        uw_sthr.partition_D(twri_dv_w2))
                else:
                    dvvb = (r_vv_old.load().to(cutlass.Float32)
                            - r_u_old.load()) * r_bt_old.load()
                    rb32f.store(dvvb.to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri_dv))
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_VFREE + csB)
                cbar.arrive_and_wait()
                if warp == 0:
                    mma_3.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_3, t_vst, t_ri3[None, None, kb],
                                  b_inv[None, None, kb, csB], t_vst)
                        mma_3.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_U2ACC + csB)

                cute.arch.mbarrier_wait(mb + MB_U2ACC + csB, pB_u2a)
                if cutlass.const_expr(GATE2_ == 1):
                    t_vst_w2 = cute.make_tensor(t_vst.iterator, u2f.layout)
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(t_vst_w2), r_u_wide)
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_vst), r_u_old)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    rb_wide.store(r_u_wide.load().to(cutlass.BFloat16))
                    twri_w2 = cute.make_tensor(twri_b2.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide, uw_sthr.partition_D(twri_w2))
                else:
                    rb32f.store(r_u_old.load().to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri_b2))
                cute.arch.fence_view_async_tmem_store()
                cbar.arrive_and_wait()
                if warp == 0:
                    # B-side MMA4 group: master leg first (early FIN),
                    # then OUT-B intra; UFIX+csA is a parity-only arm
                    mma_4a.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_4a, t_d4a, t_ri4b[None, None, kb],
                                  b_fta[None, None, kb, csB], t_d4a)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_FIN + csB)
                        tcgen05.commit(mb + MB_UFIX + csA)
                    mma_4b.set(tcgen05.Field.ACCUMULATE, True)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_4b, t_d4b_b, t_ri4b[None, None, kb],
                                  b_ftb[None, None, kb, csB], t_d4b_b)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OFIN + csB)
                        tcgen05.commit(mb + MB_SFREE + csB)

                cute.arch.mbarrier_wait(mb + MB_FIN + csA, p_fin)
                cute.arch.mbarrier_wait(mb + MB_FIN + csB, pB_fin)
                p_xgw ^= 1
                csc += 2
                if csc >= STAGES:
                    csc -= STAGES
                    p_qk ^= 1
                    p_vf ^= 1
                    p_oo ^= 1
                    p_u2a ^= 1
                    p_fin ^= 1
                    p_tab ^= 1
                    p_ufix ^= 1
                    p_rfr ^= 1

            # ---- solo tail (odd chunk count): pkdw per-chunk body ----
            if t_tiles > 2 * t_pairs:
                cute.arch.mbarrier_wait(mb + MB_TAB + csc, p_tab)
                m0s = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64, dtype=cutlass.Float32),
                    mst_base.layout)
                cute.copy(mst_ld, mst_thr.partition_S(m0s), r_mst)
                for win in cutlass.range_constexpr(2):
                    rcur = r_mst if win % 2 == 0 else r_mst2
                    rnxt = r_mst2 if win % 2 == 0 else r_mst
                    msth = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 64 + win * 32,
                                        dtype=cutlass.Float32),
                        mst_base.layout)
                    twin = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + win * 16,
                                        dtype=cutlass.BFloat16),
                        w32b_base.layout)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(win < 1):
                        mnxt = cute.make_tensor(
                            cute.recast_ptr(tmem_ptr + 64 + (win + 1) * 32,
                                            dtype=cutlass.Float32),
                            mst_base.layout)
                        cute.copy(mst_ld, mst_thr.partition_S(mnxt), rnxt)
                    for gv in cutlass.range_constexpr(4):
                        g8d = cute.make_tensor(
                            r_gt.iterator + 8 * gv, cute.make_layout((8,)))
                        cute.autovec_copy(
                            v_gt8[(win * 4 + gv, None, csc)], g8d)
                    mv = rcur.load()
                    rb32f.store(mv.to(cutlass.BFloat16))
                    rcur.store(mv * r_gt.load())
                    cute.copy(w16b_st, rb32f, w16b_thr.partition_D(twin))
                    cute.copy(mst_st, rcur, mst_sthr.partition_D(msth))
                cute.arch.fence_view_async_tmem_store()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_INP)
                if warp == 0:
                    cute.arch.mbarrier_wait(mb + MB_INP, p_inp)
                    cute.arch.mbarrier_wait(mb + MB_QK + csc, p_qk)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_u, t_inp[None, None, kb],
                                  b_kd[None, None, kb, csc], t_u)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OOUT + csc)
                        tcgen05.commit(mb + MB_RFK + csc)
                    cute.arch.mbarrier_wait(mb + MB_OEMPTY, p_oe)
                    mma_1.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(D // 16):
                        cute.gemm(mma_1, t_out, t_inp[None, None, kb],
                                  b_qd[None, None, kb, csc], t_out)
                        mma_1.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_RFREE + csc)
                p_oe ^= 1
                p_inp ^= 1

                cute.arch.mbarrier_wait(mb + MB_VFULL + csc, p_vf)
                cute.arch.mbarrier_wait(mb + MB_OOUT + csc, p_oo)
                twri_s = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr, dtype=cutlass.BFloat16),
                    w32b_base.layout)
                twri_dv_s = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 320, dtype=cutlass.BFloat16),
                    w32b_base.layout)
                if cutlass.const_expr(GATE2_ == 1):
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(u2f), r_u_wide)
                    cute.autovec_copy(
                        uw_v[None, None, None, csc], r_vv_wide)
                    for ee in cutlass.range_constexpr(cute.size(uw_uid)):
                        r_bt_wide[ee] = v_bt[uw_uid[ee][1], csc]
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_u), r_u_old)
                    vv1s = mst_id[0][0]
                    for tt in cutlass.range_constexpr(C):
                        r_vv_old[tt] = v_v[tt, vv1s, csc]
                    for gv2 in cutlass.range_constexpr(4):
                        b8d = cute.make_tensor(
                            r_bt_old.iterator + 8 * gv2,
                            cute.make_layout((8,)))
                        cute.autovec_copy(v_bt8[(gv2, None, csc)], b8d)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    dvvs = (r_vv_wide.load().to(cutlass.Float32)
                            - r_u_wide.load()) * r_bt_wide.load()
                    rb_wide.store(dvvs.to(cutlass.BFloat16))
                    twri_dv_ws = cute.make_tensor(
                        twri_dv_s.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide,
                        uw_sthr.partition_D(twri_dv_ws))
                else:
                    dvvs = (r_vv_old.load().to(cutlass.Float32)
                            - r_u_old.load()) * r_bt_old.load()
                    rb32f.store(dvvs.to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri_dv_s))
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
                if cutlass.const_expr(GATE2_ == 1):
                    t_vst_ws = cute.make_tensor(t_vst.iterator, u2f.layout)
                    cute.copy(
                        u_wide_ld, uw_ldthr.partition_S(t_vst_ws), r_u_wide)
                else:
                    cute.copy(
                        mst_ld, mst_thr.partition_S(t_vst), r_u_old)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(GATE2_ == 1):
                    rb_wide.store(r_u_wide.load().to(cutlass.BFloat16))
                    twri_ws = cute.make_tensor(twri_s.iterator, u2b.layout)
                    cute.copy(
                        u_wide_st, rb_wide, uw_sthr.partition_D(twri_ws))
                else:
                    rb32f.store(r_u_old.load().to(cutlass.BFloat16))
                    cute.copy(
                        w16b_st, rb32f, w16b_thr.partition_D(twri_s))
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
                        tcgen05.commit(mb + MB_UFIX + csc)
                    for kb in cutlass.range_constexpr(C // 16):
                        cute.gemm(mma_4b, t_d4b, t_ri4[None, None, kb],
                                  b_ftb[None, None, kb, csc], t_d4b)
                    with cute.arch.elect_one():
                        tcgen05.commit(mb + MB_OFIN + csc)
                        tcgen05.commit(mb + MB_SFREE + csc)

                cute.arch.mbarrier_wait(mb + MB_FIN + csc, p_fin)
                csc += 1
                if csc == STAGES:
                    csc = cutlass.Int32(0)
                    p_qk ^= 1
                    p_vf ^= 1
                    p_oo ^= 1
                    p_u2a ^= 1
                    p_fin ^= 1
                    p_tab ^= 1
                    p_ufix ^= 1
                    p_rfr ^= 1

            # State export: split producers always write the midstate ring;
            # terminal stateT writes compile out for this no-final task.
            for win in cutlass.range_constexpr(2):
                msth = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 64 + win * 32, dtype=cutlass.Float32),
                    mst_base.layout)
                cute.copy(mst_ld, mst_thr.partition_S(msth), r_mst)
                cute.arch.fence_view_async_tmem_load()
                vv2 = mst_id[0][0]
                for jj in cutlass.range_constexpr(4):
                    r8o = cute.make_tensor(
                        r_mst.iterator + 8 * jj, cute.make_layout((8,)))
                    if sexp >= 0:
                        g8m = cute.local_tile(
                            midstate, (1, 1, 8), (sexp, vv2, win * 4 + jj))
                        if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16):
                            for ee in cutlass.range_constexpr(8):
                                g8m[ee] = r8o[ee].to(cutlass.BFloat16)
                        else:
                            cute.autovec_copy(r8o, g8m)
                    else:
                        if cutlass.const_expr(FINAL_ == 1):
                            g8o = cute.local_tile(
                                stateT, (1, 1, 8),
                                (chain, vv2, win * 4 + jj))
                            cute.autovec_copy(r8o, g8o)
            if sexp >= 0:
                cute.arch.atomic_add(
                    mflags.iterator + sexp, cutlass.Int32(1),
                    sem="release", scope="gpu")
            if cutlass.const_expr(TPROBE_ == 1):
                if warp == 0:
                    with cute.arch.elect_one():
                        tprobe[slot * 64 + kx * 2 + 1] = \
                            cute.arch.globaltimer()

    # =====================================================================
    # EPILOGUE warpgroup (warps 4-7)
    # =====================================================================
    if (warp >= 4) & (warp < 8):
        cute.arch.warpgroup_reg_alloc(88)
        etx = tidx - 128
        o16_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32)
        o16_cp = tcgen05.make_tmem_copy(o16_atom, t_out)
        out_thr = o16_cp.get_slice(etx)
        o16_cp_b = tcgen05.make_tmem_copy(o16_atom, t_out_b)
        outb_thr = o16_cp_b.get_slice(etx)
        r_gtb2 = cute.make_rmem_tensor(
            cute.make_layout((8,)), cutlass.Float32)
        o_id = out_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BM, C))))
        r_o = cute.make_rmem_tensor(o_id.shape, cutlass.Float32)
        r_ob = cute.make_rmem_tensor(o_id.shape, cutlass.BFloat16)
        stm_np = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(True, 4),
            cutlass.BFloat16)
        ep_bar = pipeline.NamedBarrier(barrier_id=6, num_threads=128)
        # stmatrix.x4.trans lane addressing (PR m128 formula; the XOR
        # is Sw(3,4,3) in element space, matching the l3k TMA box)
        elw = etx >> 5
        lne = etx & 31
        mtx = lne >> 3
        row8 = lne & 7
        so_base = cutlass.Int32(s_out.iterator.toint())

        # ---- decay-help machinery: [128,16] half-window TMEM copies
        # (mma_h exists only to mint the [BM,16] f32 accumulator layout)
        mma_h = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                cutlass.BFloat16, cutlass.Float32, (BM, 16, 16),
                tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
                tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K))
        thr_h = mma_h.get_slice(0)
        h16_base = thr_h.make_fragment_C(mma_h.partition_shape_C((BM, 16)))
        ld16_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
        st16_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32)
        h0t = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32),
            h16_base.layout)
        h_ld = tcgen05.make_tmem_copy(ld16_atom, h0t)
        h_st = tcgen05.make_tmem_copy(st16_atom, h0t)
        h_thr = h_ld.get_slice(etx)
        h_sthr = h_st.get_slice(etx)
        h_id = h_thr.partition_D(
            thr_h.partition_C(cute.make_identity_tensor((BM, 16))))
        r_e1 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        r_e2 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        gt16 = cute.make_rmem_tensor(h_id.shape, cutlass.Float32)
        w8b_thr = w8b_st.get_slice(etx)
        w8b_id = w8b_thr.partition_S(
            thr1.partition_A(cute.make_identity_tensor((BM, 16))))
        rbh = cute.make_rmem_tensor(w8b_id.shape, cutlass.BFloat16)

        # cross-gram requant: the tensor pipe produced QBA (rows 0-31)
        # and GBA (rows 32-63) in t_xgacc; this WG converts both to bf16
        # in the dead-L regions and folds ft(A) *= gt(B), all in the
        # OFIN-A slack window.
        exg_thr = mst_ld.get_slice(etx)
        e_xid = exg_thr.partition_D(
            thr1.partition_C(cute.make_identity_tensor((BM, C))))
        r_exg = cute.make_rmem_tensor(e_xid.shape, cutlass.Float32)
        ew8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        erf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)

        ping = cutlass.Int32(0)
        cse = cutlass.Int32(0)
        pe_fin = cutlass.Int32(0)
        p_xgm = cutlass.Int32(0)
        p_eqk = cutlass.Int32(0)
        csn = cutlass.Int32(0)
        pqn = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            bos = cutlass.Int32(cu[seq_idx]) \
                + cute.arch.make_warp_uniform(spt0[k0 + kx])
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            simp = cute.arch.make_warp_uniform(ssrc[k0 + kx])
            sexp = cute.arch.make_warp_uniform(sdst[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            gO = cute.flat_divide(
                cute.domain_offset((bos, 0, 0), mO), (C, 1, D))
            if simp >= 0:
                ftgt = fepoch * 256
                fcur = cute.arch.atomic_add(
                    mflags.iterator + simp, cutlass.Int32(0),
                    sem="acquire", scope="gpu")
                while fcur < ftgt:
                    fcur = cute.arch.atomic_add(
                        mflags.iterator + simp, cutlass.Int32(0),
                        sem="acquire", scope="gpu")
            # seed master windows 2-3 (four 16-ch halves; compute seeds 0-1)
            for hh in cutlass.range_constexpr(4):
                hten = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 128 + hh * 16,
                                    dtype=cutlass.Float32),
                    h16_base.layout)
                if simp >= 0:
                    vv0m = h_id[0][0]
                    for jj in cutlass.range_constexpr(2):
                        r8i = cute.make_tensor(
                            r_e1.iterator + 8 * jj, cute.make_layout((8,)))
                        g8i = cute.local_tile(
                            midstate, (1, 1, 8), (simp, vv0m, 8 + hh * 2 + jj))
                        if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16):
                            for ee in cutlass.range_constexpr(8):
                                r8i[ee] = cutlass.Float32(g8i[ee])
                        else:
                            cute.autovec_copy(g8i, r8i)
                else:
                    if have_state == 1:
                        vv0 = h_id[0][0]
                        for jj in cutlass.range_constexpr(2):
                            r8i = cute.make_tensor(
                                r_e1.iterator + 8 * jj, cute.make_layout((8,)))
                            g8i = cute.local_tile(
                                state0, (1, 1, 8), (chain, vv0, 8 + hh * 2 + jj))
                            cute.autovec_copy(g8i, r8i)
                    else:
                        for e in cutlass.range_constexpr(cute.size(r_e1)):
                            r_e1[e] = cutlass.Float32(0.0)
                cute.copy(h_st, r_e1, h_sthr.partition_D(hten))
            cute.arch.fence_view_async_tmem_store()

            # unit-0 decay help (running stage slot); pairs use gt products
            u0pair = cutlass.Int32(0)
            if t_tiles >= 2:
                u0pair = cutlass.Int32(1)
            cute.arch.mbarrier_wait(mb + MB_TAB + csn, pqn)
            csn2 = csn + 1
            pq2 = pqn
            if csn2 == STAGES:
                csn2 = cutlass.Int32(0)
                pq2 = pqn ^ 1
            if u0pair == 1:
                cute.arch.mbarrier_wait(mb + MB_TAB + csn2, pq2)
            h0p = cute.make_tensor(
                cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32),
                h16_base.layout)
            cute.copy(h_ld, h_thr.partition_S(h0p), r_e1)
            for hh in cutlass.range_constexpr(4):
                rcur = r_e1 if hh % 2 == 0 else r_e2
                rnxt = r_e2 if hh % 2 == 0 else r_e1
                hmst = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 128 + hh * 16,
                                    dtype=cutlass.Float32),
                    h16_base.layout)
                cute.arch.fence_view_async_tmem_load()
                if cutlass.const_expr(hh < 3):
                    hnxt = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 128 + (hh + 1) * 16,
                                        dtype=cutlass.Float32),
                        h16_base.layout)
                    cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)
                for gv in cutlass.range_constexpr(2):
                    g8d = cute.make_tensor(
                        gt16.iterator + 8 * gv, cute.make_layout((8,)))
                    cute.autovec_copy(
                        v_gt8[(8 + hh * 2 + gv, None, csn)], g8d)
                mv = rcur.load()
                rbh.store(mv.to(cutlass.BFloat16))
                tw8 = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 32 + hh * 8,
                                    dtype=cutlass.BFloat16),
                    w8b_base.layout)
                cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))
                mv2 = mv * gt16.load()
                if u0pair == 1:
                    rbh.store(mv2.to(cutlass.BFloat16))
                    tw8b = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 368 + 32 + hh * 8,
                                        dtype=cutlass.BFloat16),
                        w8b_base.layout)
                    cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8b))
                    for gv in cutlass.range_constexpr(2):
                        g8d = cute.make_tensor(
                            gt16.iterator + 8 * gv, cute.make_layout((8,)))
                        cute.autovec_copy(
                            v_gt8[(8 + hh * 2 + gv, None, csn2)], g8d)
                    rcur.store(mv2 * gt16.load())
                else:
                    rcur.store(mv2)
                cute.copy(h_st, rcur, h_sthr.partition_D(hmst))
            cute.arch.fence_view_async_tmem_store()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(mb + MB_INP)
            csn += 1
            if csn == STAGES:
                csn = cutlass.Int32(0)
                pqn ^= 1
            if u0pair == 1:
                csn += 1
                if csn == STAGES:
                    csn = cutlass.Int32(0)
                    pqn ^= 1

            t_pairs_e = t_tiles // 2
            for tp in cutlass.range(t_pairs_e):
                ceA = cse
                ceB = ceA + 1
                feB = cutlass.Int32(0)
                if ceB == STAGES:
                    ceB = cutlass.Int32(0)
                    feB = cutlass.Int32(1)
                peB_fin = pe_fin ^ feB
                tA = tp * 2
                # ---- cross-gram requant + ft(A) fold (compute's shadow) ----
                cute.arch.mbarrier_wait(mb + MB_XGM, p_xgm)
                cute.copy(mst_ld, exg_thr.partition_S(t_xgacc), r_exg)
                cute.arch.fence_view_async_tmem_load()
                exrow = e_xid[0][0]
                for e in cutlass.range_constexpr(cute.size(r_exg)):
                    exc = e_xid[e][1]
                    bvx = cutlass.BFloat16(r_exg[e])
                    if exrow < 32:
                        v_xg[exrow, exc, ceA] = bvx
                    else:
                        if exrow < 64:
                            v_xg[exrow - 32, exc, ceB] = bvx
                for wp4 in cutlass.range(4):
                    it4 = wp4 * 128 + etx
                    rw4 = it4 >> 4
                    sg4 = it4 & 15
                    sf4 = v_ki8[(rw4, None, sg4 & 7, sg4 >> 3, ceA)]
                    cute.autovec_copy(sf4, ew8)
                    cute.autovec_copy(v_gt8[(sg4, None, ceB)], erf8)
                    ew8.store((ew8.load().to(cutlass.Float32)
                               * erf8.load()).to(cutlass.BFloat16))
                    cute.autovec_copy(ew8, sf4)
                cute.arch.fence_proxy("async.shared", space="cta")
                ep_bar.arrive_and_wait()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_XG)
                p_xgm ^= 1
                cute.arch.mbarrier_wait(mb + MB_OFIN + ceA, pe_fin)
                # OUT-A is final at OFIN-A (mid-pair): export it while the
                # compute WG still runs the B sub-chunk
                cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_OEMPTY)
                if warp == 4:
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                ep_bar.arrive_and_wait()
                r_ob.store(r_o.load().to(cutlass.BFloat16))
                for dh in cutlass.range_constexpr(2):
                    for tg in cutlass.range_constexpr(2):
                        dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                        ta = tg * 16 + (mtx >> 1) * 8 + row8
                        tp2 = ta >> 1
                        par = ta & 1
                        raw_row = tp2 + (dim_base >> 6) * 16
                        raw_col = ((dim_base & 63)
                                   ^ ((tp2 & 3) << 4) ^ (par << 3)) \
                            + par * 64
                        addr = so_base + ping * 8192 \
                            + (raw_row * 128 + raw_col) * 2
                        e0 = dh * 16 + tg * 8
                        s8 = cute.make_tensor(
                            r_ob.iterator + e0,
                            cute.make_layout((8,)))
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
                    f_oa = cute.make_tensor(
                        p_o + ping * (C * D),
                        cute.make_layout((C, 1, (64, 2)),
                                         stride=(64, 0, (1, 2048))))
                    oa_s, oa_g = cpasync.tma_partition(
                        tma_o, 0, cute.make_layout(1),
                        cute.group_modes(f_oa, 0, 3),
                        cute.group_modes(gO, 0, 3))
                    cute.copy(tma_o, oa_s, oa_g[(None, tA, hidx, 0)])
                    cute.arch.cp_async_bulk_commit_group()
                ping ^= 1

                cute.arch.mbarrier_wait(mb + MB_OFIN + ceB, peB_fin)
                # decay help for the next unit (pair -> gt product)
                if tA + 2 < t_tiles:
                    nx2 = cutlass.Int32(0)
                    if tA + 3 < t_tiles:
                        nx2 = cutlass.Int32(1)
                    cute.arch.mbarrier_wait(mb + MB_TAB + csn, pqn)
                    csn2b = csn + 1
                    pq2b = pqn
                    if csn2b == STAGES:
                        csn2b = cutlass.Int32(0)
                        pq2b = pqn ^ 1
                    if nx2 == 1:
                        cute.arch.mbarrier_wait(mb + MB_TAB + csn2b, pq2b)
                    h0q = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + 128, dtype=cutlass.Float32),
                        h16_base.layout)
                    cute.copy(h_ld, h_thr.partition_S(h0q), r_e1)
                    for hh in cutlass.range_constexpr(4):
                        rcur = r_e1 if hh % 2 == 0 else r_e2
                        rnxt = r_e2 if hh % 2 == 0 else r_e1
                        hmst = cute.make_tensor(
                            cute.recast_ptr(tmem_ptr + 128 + hh * 16,
                                            dtype=cutlass.Float32),
                            h16_base.layout)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(hh < 3):
                            hnxt = cute.make_tensor(
                                cute.recast_ptr(tmem_ptr + 128 + (hh + 1) * 16,
                                                dtype=cutlass.Float32),
                                h16_base.layout)
                            cute.copy(h_ld, h_thr.partition_S(hnxt), rnxt)
                        for gv in cutlass.range_constexpr(2):
                            g8d = cute.make_tensor(
                                gt16.iterator + 8 * gv, cute.make_layout((8,)))
                            cute.autovec_copy(
                                v_gt8[(8 + hh * 2 + gv, None, csn)], g8d)
                        mv = rcur.load()
                        rbh.store(mv.to(cutlass.BFloat16))
                        tw8 = cute.make_tensor(
                            cute.recast_ptr(tmem_ptr + 32 + hh * 8,
                                            dtype=cutlass.BFloat16),
                            w8b_base.layout)
                        cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8))
                        mv2 = mv * gt16.load()
                        if nx2 == 1:
                            rbh.store(mv2.to(cutlass.BFloat16))
                            tw8b = cute.make_tensor(
                                cute.recast_ptr(tmem_ptr + 368 + 32 + hh * 8,
                                                dtype=cutlass.BFloat16),
                                w8b_base.layout)
                            cute.copy(w8b_st, rbh, w8b_thr.partition_D(tw8b))
                            for gv in cutlass.range_constexpr(2):
                                g8d = cute.make_tensor(
                                    gt16.iterator + 8 * gv,
                                    cute.make_layout((8,)))
                                cute.autovec_copy(
                                    v_gt8[(8 + hh * 2 + gv, None, csn2b)],
                                    g8d)
                            rcur.store(mv2 * gt16.load())
                        else:
                            rcur.store(mv2)
                        cute.copy(h_st, rcur, h_sthr.partition_D(hmst))
                    cute.arch.fence_view_async_tmem_store()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mb + MB_INP)
                    csn += 1
                    if csn == STAGES:
                        csn = cutlass.Int32(0)
                        pqn ^= 1
                    if nx2 == 1:
                        csn += 1
                        if csn == STAGES:
                            csn = cutlass.Int32(0)
                            pqn ^= 1
                # OUT-B export
                cute.copy(o16_cp_b, outb_thr.partition_S(t_out_b), r_o)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_OEMPB)
                cc_b = seq_len - (tA + 1) * C
                if cc_b >= C:
                    if warp == 4:
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                    ep_bar.arrive_and_wait()
                    r_ob.store(r_o.load().to(cutlass.BFloat16))
                    for dh in cutlass.range_constexpr(2):
                        for tg in cutlass.range_constexpr(2):
                            dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                            ta = tg * 16 + (mtx >> 1) * 8 + row8
                            tp3 = ta >> 1
                            par = ta & 1
                            raw_row = tp3 + (dim_base >> 6) * 16
                            raw_col = ((dim_base & 63)
                                       ^ ((tp3 & 3) << 4) ^ (par << 3)) \
                                + par * 64
                            addr = so_base + ping * 8192 \
                                + (raw_row * 128 + raw_col) * 2
                            e0 = dh * 16 + tg * 8
                            s8 = cute.make_tensor(
                                r_ob.iterator + e0,
                                cute.make_layout((8,)))
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
                        f_ob = cute.make_tensor(
                            p_o + ping * (C * D),
                            cute.make_layout((C, 1, (64, 2)),
                                             stride=(64, 0, (1, 2048))))
                        ob_s, ob_g = cpasync.tma_partition(
                            tma_o, 0, cute.make_layout(1),
                            cute.group_modes(f_ob, 0, 3),
                            cute.group_modes(gO, 0, 3))
                        cute.copy(tma_o, ob_s, ob_g[(None, tA + 1, hidx, 0)])
                        cute.arch.cp_async_bulk_commit_group()
                    ping ^= 1
                else:
                    r_ob.store(r_o.load().to(cutlass.BFloat16))
                    for e in cutlass.range_constexpr(cute.size(r_o)):
                        tt2 = o_id[e][1]
                        if tt2 < cc_b:
                            out_raw[bos + (tA + 1) * C + tt2, hidx,
                                    o_id[e][0]] = r_ob[e]
                cse += 2
                if cse >= STAGES:
                    cse -= STAGES
                    pe_fin ^= 1
                    p_eqk ^= 1

            # ---- solo tail export ----
            if t_tiles > 2 * t_pairs_e:
                t = 2 * t_pairs_e
                cc_t = seq_len - t * C
                cute.arch.mbarrier_wait(mb + MB_OFIN + cse, pe_fin)
                cute.copy(o16_cp, out_thr.partition_S(t_out), r_o)
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(mb + MB_OEMPTY)
                if cc_t >= C:
                    if warp == 4:
                        # unconditional: stores may be in flight across
                        # chain boundaries; no-op when nothing pending
                        cute.arch.cp_async_bulk_wait_group(1, read=True)
                    ep_bar.arrive_and_wait()
                    # PTX-free stmatrix: DSL StMatrix atom over the
                    # same lane addresses (probe scripts/stm_probe.py
                    # = byte-identical to the inline-PTX block)
                    r_ob.store(r_o.load().to(cutlass.BFloat16))
                    for dh in cutlass.range_constexpr(2):
                        for tg in cutlass.range_constexpr(2):
                            dim_base = elw * 32 + dh * 16 + (mtx & 1) * 8
                            ta = tg * 16 + (mtx >> 1) * 8 + row8
                            tp = ta >> 1
                            par = ta & 1
                            raw_row = tp + (dim_base >> 6) * 16
                            raw_col = ((dim_base & 63)
                                       ^ ((tp & 3) << 4) ^ (par << 3)) \
                                + par * 64
                            addr = so_base + ping * 8192 \
                                + (raw_row * 128 + raw_col) * 2
                            e0 = dh * 16 + tg * 8
                            s8 = cute.make_tensor(
                                r_ob.iterator + e0,
                                cute.make_layout((8,)))
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
                            p_o + ping * (C * D),
                            cute.make_layout((C, 1, (64, 2)),
                                             stride=(64, 0, (1, 2048))))
                        o_s, o_g = cpasync.tma_partition(
                            tma_o, 0, cute.make_layout(1),
                            cute.group_modes(f_o, 0, 3),
                            cute.group_modes(gO, 0, 3))
                        cute.copy(tma_o, o_s, o_g[(None, t, hidx, 0)])
                        cute.arch.cp_async_bulk_commit_group()
                    ping ^= 1
                else:
                    r_ob.store(r_o.load().to(cutlass.BFloat16))
                    for e in cutlass.range_constexpr(cute.size(r_o)):
                        tt2 = o_id[e][1]
                        if tt2 < cc_t:
                            out_raw[bos + t * C + tt2, hidx, o_id[e][0]] = \
                                r_ob[e]
                cse += 1
                if cse == STAGES:
                    cse = 0
                    pe_fin ^= 1
                    p_eqk ^= 1
            # Split-state windows 2-3; terminal stateT writes compile out.
            for hh in cutlass.range_constexpr(4):
                hten = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + 128 + hh * 16,
                                    dtype=cutlass.Float32),
                    h16_base.layout)
                cute.copy(h_ld, h_thr.partition_S(hten), r_e1)
                cute.arch.fence_view_async_tmem_load()
                vv2e = h_id[0][0]
                for jj in cutlass.range_constexpr(2):
                    r8o = cute.make_tensor(
                        r_e1.iterator + 8 * jj, cute.make_layout((8,)))
                    if sexp >= 0:
                        g8m = cute.local_tile(
                            midstate, (1, 1, 8), (sexp, vv2e, 8 + hh * 2 + jj))
                        if cutlass.const_expr(
                                midstate.element_type is cutlass.BFloat16):
                            for ee in cutlass.range_constexpr(8):
                                g8m[ee] = r8o[ee].to(cutlass.BFloat16)
                        else:
                            cute.autovec_copy(r8o, g8m)
                    else:
                        if cutlass.const_expr(FINAL_ == 1):
                            g8o = cute.local_tile(
                                stateT, (1, 1, 8),
                                (chain, vv2e, 8 + hh * 2 + jj))
                            cute.autovec_copy(r8o, g8o)
            if sexp >= 0:
                cute.arch.atomic_add(
                    mflags.iterator + sexp, cutlass.Int32(1),
                    sem="release", scope="gpu")
        if warp == 4:
            cute.arch.cp_async_bulk_wait_group(0)

    # =====================================================================
    # WG2: MMA (warp 9), LOAD (warp 10), donors (8, 11)
    # =====================================================================
    if (warp >= 8) & (warp < 12):
        cute.arch.warpgroup_reg_dealloc(32)
        if warp == 10:
            csl = cutlass.Int32(0)
            pl_vfree = cutlass.Int32(1)
            pl_qk = cutlass.Int32(0)
            gE = cute.flat_divide(mE, (64, 1, 256))
            for kx in cutlass.range(k1 - k0):
                chain = cute.arch.make_warp_uniform(schain[k0 + kx])
                seq_idx = chain // H_
                hidx = chain % H_
                bos = cutlass.Int32(cu[seq_idx]) \
                    + cute.arch.make_warp_uniform(spt0[k0 + kx])
                seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
                t_tiles = (seq_len + C - 1) // C
                if cutlass.const_expr(GATE2_ == 1):
                    gV = cute.flat_divide(
                        cute.domain_offset((0, 0, bos), mV), (D, 1, C))
                else:
                    gV = cute.flat_divide(
                        cute.domain_offset((bos, 0, 0), mV), (C, 1, D))
                for t in cutlass.range(t_tiles):
                    cute.arch.mbarrier_wait(mb + MB_VFREE + csl, pl_vfree)
                    cute.arch.mbarrier_wait(mb + MB_QK + csl, pl_qk)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mb + MB_VFULL + csl, C * D * 2)
                    if cutlass.const_expr(GATE2_ == 1):
                        f_v = v_vt[None, None, None, csl]
                        v_d, v_s = cpasync.tma_partition(
                            tma_v, 0, cute.make_layout(1),
                            cute.group_modes(f_v, 0, 3),
                            cute.group_modes(gV, 0, 3))
                        cute.copy(
                            tma_v, v_s[(None, 0, hidx, t)], v_d,
                            tma_bar_ptr=mb + MB_VFULL + csl)
                    else:
                        f_v = cute.make_tensor(
                            p_v_raw + csl * STAGE_ELTS,
                            cute.make_layout(
                                (C, 1, D), stride=(D, 0, 1)))
                        v_d, v_s = cpasync.tma_partition(
                            tma_v, 0, cute.make_layout(1),
                            cute.group_modes(f_v, 0, 3),
                            cute.group_modes(gV, 0, 3))
                        cute.copy(
                            tma_v, v_s[(None, t, hidx, 0)], v_d,
                            tma_bar_ptr=mb + MB_VFULL + csl)
                    if do_export == 1:
                        if seq_idx == export_seq:
                            # flat byte-image dump of the whole prep stage
                            # (v36 law: reload with the same flat box + the
                            # canonical views reproduces the operands); the
                            # 5-deep ring gives ~4 chunk periods of slack
                            # before prep(t+5) overwrites this stage
                            f_e = cute.make_tensor(
                                cute.recast_ptr(ar0 + csl * STAGE_BYTES,
                                                dtype=cutlass.BFloat16),
                                cute.make_layout((64, 1, 256),
                                                 stride=(256, 0, 1)))
                            e_s, e_g = cpasync.tma_partition(
                                tma_e, 0, cute.make_layout(1),
                                cute.group_modes(f_e, 0, 3),
                                cute.group_modes(gE, 0, 3))
                            cute.copy(tma_e, e_s,
                                      e_g[(None, hidx * nc2 + t, 0, 0)])
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(2, read=True)
                            # gt/beta tables (dedicated buffers, scalar STG)
                            for kt5 in cutlass.range_constexpr(4):
                                expt[hidx * nc2 + t, kt5 * 32 + lane] = \
                                    v_gt[kt5 * 32 + lane, csl]
                            expt[hidx * nc2 + t, 128 + lane] = v_bt[lane, csl]
                    csl += 1
                    if csl == STAGES:
                        csl = 0
                        pl_vfree ^= 1
                        pl_qk ^= 1

    # =====================================================================
    # PREP (warps 12-31): 5 instances x 4 warps; instance == stage
    # =====================================================================
    if warp >= 12:
        cute.arch.warpgroup_reg_dealloc(48)
        inst = (warp - 12) >> 2
        plw = (warp - 12) & 3
        ptl = plw * 32 + lane

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

        lb2h = lb2 * 0.5
        anch = lb2 * 16.0
        konst2 = _exp2f(anch)

        w8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        qr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        kr = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.BFloat16)
        g8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
        rf8 = cute.make_rmem_tensor(cute.make_layout((8,)), cutlass.Float32)
        c16 = cute.make_rmem_tensor(cute.make_layout((16,)), cutlass.Float32)
        w2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.BFloat16)
        g2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)
        dtb2 = cute.make_rmem_tensor(cute.make_layout((2,)), cutlass.Float32)

        pp_rfree = cutlass.Int32(1)
        pp_rfk = cutlass.Int32(1)
        pp_graw = cutlass.Int32(0)
        pp_sfree = cutlass.Int32(1)
        pp_qkraw = cutlass.Int32(0)
        ibar = pipeline.NamedBarrier(barrier_id=7 + inst, num_threads=128)

        ph0 = cutlass.Int32(0)
        for kx in cutlass.range(k1 - k0):
            chain = cute.arch.make_warp_uniform(schain[k0 + kx])
            seq_idx = chain // H_
            hidx = chain % H_
            bos = cutlass.Int32(cu[seq_idx]) \
                + cute.arch.make_warp_uniform(spt0[k0 + kx])
            seq_len = cute.arch.make_warp_uniform(sptn[k0 + kx])
            t_tiles = (seq_len + C - 1) // C
            gQ = cute.flat_divide(
                cute.domain_offset((bos, 0, 0), mQ), (C, 1, D))
            gK = cute.flat_divide(
                cute.domain_offset((bos, 0, 0), mK), (C, 1, D))
            gG = cute.flat_divide(
                cute.domain_offset((bos, 0, 0), mG), (C, 1, D))
            ea = cute.math.exp(a_log[hidx], fastmath=True)
            ea2c = 0.5 * ea
            if cutlass.const_expr(GATE2_ == 1):
                if ptl < 64:
                    gdt2 = cute.local_tile(dt_bias, (1, 2), (hidx, ptl))
                    cute.autovec_copy(gdt2, dtb2)
                    dtb2[0] = ea2c * dtb2[0]
                    dtb2[1] = ea2c * dtb2[1]
            # this instance owns local chunks li0, li0+5, ... (< t_tiles):
            # li0 aligns inst with the CTA-global chunk stream position
            li0 = inst - ph0
            if li0 < 0:
                li0 += STAGES
            n_iters = (t_tiles - li0 + STAGES - 1) // STAGES
            ph0 += t_tiles
            ph0 = ph0 % STAGES
            for it in cutlass.range(n_iters):
                ci = li0 + it * STAGES
                r0 = bos + ci * C
                cc = cutlass.min(cutlass.Int32(C), seq_len - ci * C)
                full = seq_len >= (ci + 1) * C
                # pair role: odd local chunk = B side of pair (ci-1, ci)
                is_b = cutlass.Int32(0)
                if (ci & 1) == 1:
                    is_b = cutlass.Int32(1)

                # -- phase 0: raw g TMA -> KD slot (free after the U legs,
                # MB_RFK) and raw k TMA -> QD slot (free after the deferred
                # O legs, MB_RFREE); decorated outputs keep their homes --
                if full:
                    cute.arch.mbarrier_wait(mb + MB_RFK + inst, pp_rfk)
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mb + MB_GRAW + inst, C * D * 2)
                        f_g = cute.make_tensor(
                            cute.recast_ptr(ar0 + (OFF_KD + inst * STAGE_BYTES),
                                            dtype=cutlass.BFloat16),
                            cute.make_layout((C, 1, D), stride=(D, 0, 1)))
                        g_d, g_s = cpasync.tma_partition(
                            tma_g, 0, cute.make_layout(1),
                            cute.group_modes(f_g, 0, 3), cute.group_modes(gG, 0, 3))
                        cute.copy(tma_g, g_s[(None, ci, hidx, 0)], g_d,
                                  tma_bar_ptr=mb + MB_GRAW + inst)
                    cute.arch.mbarrier_wait(mb + MB_RFREE + inst, pp_rfree)
                    if plw == 0:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mb + MB_QKRAW + inst, 2 * C * D * 2)
                        f_k = cute.make_tensor(
                            cute.recast_ptr(ar0 + (OFF_QD + inst * STAGE_BYTES),
                                            lay_qd.inner, dtype=cutlass.BFloat16),
                            cute.make_layout((C, 1, (16, 4, 2)),
                                             stride=(64, 0, (1, 16, 2048))))
                        k_d, k_s = cpasync.tma_partition(
                            tma_k, 0, cute.make_layout(1),
                            cute.group_modes(f_k, 0, 3), cute.group_modes(gK, 0, 3))
                        cute.copy(tma_k, k_s[(None, ci, hidx, 0)], k_d,
                                  tma_bar_ptr=mb + MB_QKRAW + inst)

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

                # beta (dedicated buffer -> written pre-walker; read by the
                # grams two barriers later and by compute after qk_full)
                if plw == 2:
                    if lane < C:
                        btv = cutlass.Float32(0.0)
                        if lane < cc:
                            bx = cutlass.Float32(beta[r0 + lane, hidx])
                            btv = 0.5 + 0.5 * _tanhf(0.5 * bx)
                        v_bt[lane, inst] = btv

                if cutlass.const_expr(GATE2_ == 1):
                    # v98 fused gate+cumsum (fixed-shape kernels only):
                    # channel pair per thread (warps 0-1), running sum in
                    # registers, one STS.64 per row — replaces the
                    # store-gate / reload / walker phase pair (~256 fewer
                    # L1 wavefronts per chunk on the ~86%-busy LSU pipe).
                    # v99 note kept out: TMAs stay full-chunk-gated here.
                    if full:
                        cute.arch.mbarrier_wait(mb + MB_GRAW + inst, pp_graw)
                    if ptl < 64:
                        acc0 = cutlass.Float32(0.0)
                        acc1 = cutlass.Float32(0.0)
                        if full:
                            for rw in cutlass.range_constexpr(C):
                                g2s = v_graw2[(rw, None, ptl, inst)]
                                cute.autovec_copy(g2s, w2)
                                acc0 += lb2h * _tanhf(
                                    ea2c * cutlass.Float32(w2[0])
                                    + dtb2[0]) + lb2h
                                acc1 += lb2h * _tanhf(
                                    ea2c * cutlass.Float32(w2[1])
                                    + dtb2[1]) + lb2h
                                g2[0] = acc0
                                g2[1] = acc1
                                gd2 = v_gcs2[(rw, None, ptl, inst)]
                                cute.autovec_copy(g2, gd2)
                        else:
                            for rw in cutlass.range_constexpr(C):
                                if rw < cc:
                                    g2m = cute.local_tile(
                                        g, (1, 1, 2), (r0 + rw, hidx, ptl))
                                    cute.autovec_copy(g2m, w2)
                                    acc0 += lb2h * _tanhf(
                                        ea2c * cutlass.Float32(w2[0])
                                        + dtb2[0]) + lb2h
                                    acc1 += lb2h * _tanhf(
                                        ea2c * cutlass.Float32(w2[1])
                                        + dtb2[1]) + lb2h
                                g2[0] = acc0
                                g2[1] = acc1
                                gd2 = v_gcs2[(rw, None, ptl, inst)]
                                cute.autovec_copy(g2, gd2)
                    ibar.arrive_and_wait()
                if cutlass.const_expr(GATE2_ == 0):
                    # -- phase 2a: gate values, row-parallel (vec8 in/out; the
                    # tanh chain has no serial dependency here).  dt_bias row
                    # slice is wpg-invariant: load once per chunk into rf8
                    # (dead until the restore) and pre-fold ea/2 so the inner
                    # element chain is FFMA+TANH+FFMA --
                    if full:
                        cute.arch.mbarrier_wait(mb + MB_GRAW + inst, pp_graw)
                    sgg = ptl & 15
                    gdt8 = cute.local_tile(dt_bias, (1, 8), (hidx, sgg))
                    cute.autovec_copy(gdt8, rf8)
                    ea2 = 0.5 * ea
                    for u in cutlass.range_constexpr(8):
                        rf8[u] = ea2 * rf8[u]
                    for wpg in cutlass.range(4):
                        rwg = wpg * 8 + (ptl >> 4)
                        if rwg < cc:
                            if full:
                                gg8 = v_graw8[(rwg, None, sgg, inst)]
                                cute.autovec_copy(gg8, qr)
                            else:
                                gg8m = cute.local_tile(g, (1, 1, 8),
                                                       (r0 + rwg, hidx, sgg))
                                cute.autovec_copy(gg8m, qr)
                            for u in cutlass.range_constexpr(8):
                                ga2 = ea2 * cutlass.Float32(qr[u]) + rf8[u]
                                if cutlass.const_expr(LITE_ == 1):
                                    g8[u] = lb2h * 0.5 + lb2h
                                else:
                                    g8[u] = lb2h * _tanhf(ga2) + lb2h
                        else:
                            for u in cutlass.range_constexpr(8):
                                g8[u] = 0.0
                        gd8 = v_gcs8[(rwg, None, sgg, inst)]
                        cute.autovec_copy(g8, gd8)
                    ibar.arrive_and_wait()
                    # -- phase 2b: per-channel scan (independent column loads
                    # feeding a pure FADD chain; depth-4 load prefetch
                    # measured-closed: IKET pp_gate unchanged — the compiler
                    # already hides the walker's LDS latency) --
                    if ptl < D:
                        acc = cutlass.Float32(0.0)
                        for hb in cutlass.range_constexpr(2):
                            for rw in cutlass.range_constexpr(16):
                                c16[rw] = v_gcs[hb * 16 + rw, ptl, inst]
                            for rw in cutlass.range_constexpr(16):
                                acc += c16[rw]
                                v_gcs[hb * 16 + rw, ptl, inst] = acc
                    ibar.arrive_and_wait()
                # -- phase 3: q/k load + l2norm + anchored decorations --
                # (unroll=2 measured-closed: 1.569x -> 1.445x fixed_h96 —
                # the 48-reg prep diet spills, reconfirming the v60 re-roll)
                if full:
                    cute.arch.mbarrier_wait(mb + MB_QKRAW + inst, pp_qkraw)
                for wp in cutlass.range(4):
                    rw2 = wp * 8 + (ptl >> 4)
                    sg2 = ptl & 15
                    if full:
                        q8s = v_ki8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
                        cute.autovec_copy(q8s, qr)
                        k8s = v_qd8[(rw2, None, sg2 & 7, sg2 >> 3, inst)]
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
                    # scale folded into rqn: one FMUL replaces 8 per slice
                    rqn = cute.math.rsqrt(sq + 1e-6, fastmath=True) * scale
                    rkn = cute.math.rsqrt(sk + 1e-6, fastmath=True)
                    g8s = v_gcs8[(rw2, None, sg2, inst)]
                    cute.autovec_copy(g8s, g8)
                    qhv = qr.load().to(cutlass.Float32) * rqn
                    khv = kr.load().to(cutlass.Float32) * rkn
                    gv8 = g8.load()
                    decv = cute.math.exp2(gv8 - anch, fastmath=True)

                    qr.store((qhv * decv).to(cutlass.BFloat16))
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
                if plw == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mb + MB_TAB + inst)

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
                pp_rfree ^= 1
                pp_rfk ^= 1
                pp_sfree ^= 1
                # GRAW/QKRAW are armed+waited only on full chunks; their
                # parities must advance 1:1 with actual completions or the
                # NEXT chain in this persistent slot waits a stale phase
                # (RFREE/SFREE are compute-armed every chunk -> uncond.)
                if full:
                    pp_graw ^= 1
                    pp_qkraw ^= 1

    cute.arch.sync_threads()
    tmem.free(tmem_ptr)


@cute.jit
def _launch_pkd(
    q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, g: cute.Tensor,
    beta: cute.Tensor, a_log: cute.Tensor, dt_bias: cute.Tensor,
    state0: cute.Tensor, stateT: cute.Tensor, out: cute.Tensor,
    cu: cute.Tensor, soff: cute.Tensor, schain: cute.Tensor,
    spt0: cute.Tensor, sptn: cute.Tensor,
    ssrc: cute.Tensor, sdst: cute.Tensor,
    midstate: cute.Tensor, mflags: cute.Tensor,
    exp_ws: cute.Tensor, expt: cute.Tensor, tprobe: cute.Tensor,
    have_state: cutlass.Int32,
    gcnt: cutlass.Int32,
    do_export: cutlass.Int32, export_seq: cutlass.Int32,
    nc2: cutlass.Int32, fepoch: cutlass.Int32,
    scale: cutlass.Float32, lb2: cutlass.Float32,
    H_: cutlass.Constexpr[int],
    stream: cuda_driver.CUstream,
    TPROBE_: cutlass.Constexpr[int] = 0,
    GATE2_: cutlass.Constexpr[int] = 0,
    FINAL_: cutlass.Constexpr[int] = 1,
    LITE_: cutlass.Constexpr[int] = 0,
):
    mma_1 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ))
    mma_3 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ))
    mma_4 = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, NFT, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    # v91 split-MMA4: master leg (N=128) commits FIN for the next decay
    # before the OUT leg (N=32, commits OFIN + SFREE); junk pad gone.
    mma_4a = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, D, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    mma_4b = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.TMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN,
        ))
    mma_xg = cute.make_tiled_mma(
        tcgen05.MmaF16BF16Op(
            cutlass.BFloat16, cutlass.Float32, (BM, C, 16),
            tcgen05.CtaGroup.ONE, tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
        ))
    lay_qd = sm100_utils.make_smem_layout_b(mma_1, (BM, C, D), cutlass.BFloat16, 1)
    lay_inv = sm100_utils.make_smem_layout_b(mma_3, (BM, C, C), cutlass.BFloat16, 1)
    lay_ft = sm100_utils.make_smem_layout_b(mma_4, (BM, NFT, C), cutlass.BFloat16, 1)
    lay_v0 = sm100_utils.make_smem_layout_a(
        mma_1, (BM, C, C), cutlass.BFloat16, STAGES, is_k_major=False)
    lvo = lay_v0.outer
    lay_v = cute.make_composed_layout(
        lay_v0.inner, 0,
        cute.make_layout(
            (lvo.shape[0][0], lvo.shape[1],
             (lvo.shape[0][1], lvo.shape[2]), lvo.shape[3]),
            stride=(lvo.stride[0][0], lvo.stride[1],
                    (lvo.stride[0][1], lvo.stride[2]), STAGE_ELTS)))
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
    if cutlass.const_expr(GATE2_ == 1):
        vT = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (D, H_, v.shape[0]), stride=(1, D, H_ * D)))
        lv = cute.select(lay_v, mode=[0, 1, 2])
        tma_v, mV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), vT, lv, (D, 1, C))
    else:
        tma_v, mV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), v, lg_raw, (C, 1, D))
    tma_o, mO = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), out, l3k, (C, 1, D))
    le_flat = cute.make_layout((64, 1, 256), stride=(256, 0, 1))
    tma_e, mE = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), exp_ws, le_flat, (64, 1, 256))
    _pkd(mma_1, mma_3, mma_4a, mma_4b, mma_xg,
         tma_q, mQ, tma_k, mK, tma_g, mG, tma_v, mV, tma_o, mO,
         tma_e, mE,
         q, k, g, beta, a_log, dt_bias,
         out, state0, stateT, cu, soff, schain,
         spt0, sptn, ssrc, sdst, midstate, mflags, expt, tprobe,
         have_state, do_export, export_seq, nc2, fepoch, scale, lb2,
         lay_qd, lay_inv, lay_ft, lay_v,
         H_, TPROBE_, GATE2_, FINAL_, LITE_).launch(
        grid=(gcnt, 1, 1), block=(THREADS, 1, 1), stream=stream)
