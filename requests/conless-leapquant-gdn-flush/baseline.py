# SPDX-License-Identifier: Apache-2.0
"""Best-known LeapQuant flush kernel (window-boundary re-quantization of the recurrent state; see README.md).

Warp-specialised TileLang kernel for sm_100 (B200): one persistent CTA per SM, a TMA producer warp and three
128-thread consumer groups, products on tensor cores (mma.sync m16n8k16, bf16 hi/lo operands), the rebuilt state
resident in shared memory as fp16 with the old rank-4 part carried algebraically in fp32, CholeskyQR2 with a relative
pivot floor for the new Compensator Tokens, the due programs balanced across CTAs by a rank scan.

  run(...)             functional form with the input order / outputs of definition.json (packs the inputs into a slot
                       pool, calls the kernel, returns the new checkpoint as fresh tensors) -- used for correctness
  Pool, flush_inplace  deployment form: caller-owned strided slot pool updated in place -- what benchmark.py times

Requires: torch (CUDA), tilelang 0.1.12, CUDA toolkit 13 (nvcc), NVIDIA B200.
"""
import torch, tilelang, tilelang.language as T
from tilelang.layout import make_swizzled_layout

F32, BF16, I32, I8 = "float32", "bfloat16", "int32", "int8"
QMAX = 127.0
EPS = 1e-12
HV, K, V, L, R = 32, 128, 128, 16, 4          # heads per sequence, key dim, value dim, window length, rank
SS, OFF = 33 * V * K, 12288                   # pool geometry in fp32 words: slot stride (one spare head per slot), base offset


@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_flush_kernel(NS, HV, K, V, L, SS, R=4, ITERS=1, SMS=148, NCG=3, MAXIT=128, PROF=False, PAD=8, KDA=False, GB16=True, PROFIT=False, DROPLO=False, MMT="bf16", ABL="", PNMIN=1e-20, GRAM16=False, NSTAGE=0, NMAX=0, **_ignored):
    """TileLang generator of the flush kernel (the KDA=True branches serve a per-channel-gate variant that this request does not use).

    v2 keeps S in the mma fragments (128 fp32 registers per thread), which caps the CTA at two 4-warp programs in flight;
    every phase of the flush is a dependent chain (mma -> smem -> barrier -> reduction), so the flush ran at ~22k cycles
    per program per warpgroup with 8 warps resident.  v4 makes shared memory the home of the state: the rebuild produces
    B = pn deq(S0) + buffer in two 64-column halves (64 accumulator registers) and stores it as fp16 with stmatrix; the
    old rank-R part A = pn Q_old U_old^T is carried algebraically (see below); the residual E = B + A - Q U^T is written
    back over B (ldmatrix / stmatrix), and the quantisation streams E row-wise (8 lanes per row, 16 columns per lane, one
    16 B global store per lane -- no staging tile, no copy-out).  Registers drop to ~128 per thread, which fits NCG = 3
    programs in flight (12 consumer warps); the stage (int8 tile, buffers, old factors) is released after the residual.
    Every per-program smem object beyond the stage lives in two per-group buffers: St_s (V x (K + 8) fp16: B / E in the
    columns < K, the fp16 Q tile of the iteration in the 8 padding columns) and UQn_s (fp32 Q for the CholeskyQR rows,
    the fp16 P tile, then the new -U^T for the residual mma, then the |E| column partials).

    Numerics: v2's fp16 St_s folds the fp16 rounding of S (2^-11 |S|) into the residual, which for a row captured by the
    factors (hot rows: |E| << |S|) is a large error relative to E.  v4 keeps only B in fp16 and carries A exactly: the
    iteration's products get the R x R corrections P += pn U_old (Q_old^T Q) and Q += pn Q_old (U_old^T P), with the
    R x R matrices accumulated on the tensor cores alongside the products (every warp holds the full matrix, no
    reduction), and the residual has both rank-R terms on the tensor cores in fp32.  The fp16 rounding is then 2^-11 |B|,
    the size of the checkpoint residual itself; E is rounded to fp16 once more before the quantisation.
    """
    # Gbuf holds fp16 factors; it is declared bf16 so the TMA target (Kbl_s, later the lo key tile) matches -- the bits are reinterpreted on read
    assert 1 <= R <= 4 and K == 128 and V == 128 and (not KDA or GB16) and PAD == 8
    F16 = "float16"
    N = T.symbolic("N")
    KW = K // 4
    NPROD = 32
    NCONS = 128 * NCG
    NT = NPROD + NCONS
    NWARPS = NT // 32
    NDUE = min(NS - 8, NMAX) if NMAX else NS - 8      # rows that can be due at once: one call's rows (NMAX = the engine's batch cap) or the whole table
    MAXIT = max(MAXIT, (NDUE * HV + SMS - 1) // SMS + 1)   # dlist capacity: a CTA's share of all programs when every such row is due
                                                       # (with NMAX the list is ~350 B instead of 5.5 KB on an 8155-slot engine: on sm_120 that is the difference between a 2-stage flush fitting in 99 KB or not)
    KMAX = min((NS + NWARPS - 1) // NWARPS, 40)  # n-steps per warp kept in registers by the compaction (3 x 40 registers; N <= 40 * NWARPS =
                                                 # 520 sequences at NCG = 3, vLLM's max_num_seqs 512); larger batches take a serial tail
                                                 # with the flags recomputed in the rank pass (64 steps spilled: +3 us none-due)
    NSTAGE = NSTAGE or NCG                        # NSTAGE > NCG: each group double-buffers its stage (the TMA warp prefetches the group's
    assert NSTAGE % NCG == 0                     # next due program while it works) -- worth it where one group is all that fits (sm_120)
    SPG = NSTAGE // NCG
    KP = K + PAD                                 # St_s row stride: conflict-free fragment stores, ldmatrix rows and 16 B row reads; the pad holds the fp16 Q tile
    KV8 = K + V + 8                              # UQn_s row stride (halfwords): 528 B -> conflict-free ldmatrix rows
    P16OFF = 4 * KV8                             # fp16 P tile (K x 8) at halfword offset 4 rows into UQn_s (after the fp32 Q plane)
    assert 4 * KV8 * 2 >= (V + 4) * 4 * 4 and 4 * KV8 >= K * 8

    @T.prim_func
    def flush(Sq: T.StridedTensor((NS, HV, V, KW), (SS, V * K, KW, 1), I32),
              Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
              Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
              Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), F32),
              Kbuf: T.Tensor((NS, HV, L, K), BF16), Ubuf: T.Tensor((NS, HV, L, V), BF16),
              Wbuf: T.Tensor((NS, HV, L), F32), Pbuf: T.Tensor((NS, HV, K if KDA else 1), F32), Gbuf: T.Tensor((NS, HV, L, K) if KDA else (1, 1, 1, 1), BF16), hcnt: T.Tensor((NS, HV), I32),
              Sq16: T.StridedTensor((NS, HV, V, K // 2), (2 * SS, 2 * V * K, K // 2, 1), "int16"),   # unused (kept for the v2 call signature)
              UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),           # fp16 factors U^T | Q^T after the scales
              Q0: T.Tensor((V, R), F32),                                                             # deterministic start of the subspace iteration
              idx: T.Tensor((N,), I32)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGE, V, KW), I32)
            skv_s = T.alloc_shared((NSTAGE, K + V), F32)
            Kb_s = T.alloc_shared((NSTAGE, 2, L, K // 2), BF16)      # two 128 B-wide halves, 128 B swizzle (conflict-free ldmatrix)
            Ub_s = T.alloc_shared((NSTAGE, 2, L, V // 2), BF16)
            Wb_s = T.alloc_shared((NSTAGE, 32), F32)                 # 128 B per stage (TMA destination alignment)
            UQ_s = T.alloc_shared((NSTAGE, 4, 8, 64), F16)     # old factors U^T | Q^T in four 128 B-wide column blocks (swizzled: conflict-free ldmatrix rows), k-rows padded to 8 (rows R.. zero)
            St_s = T.alloc_shared((NCG, V, KP), F16)           # B (fp16) for the iteration, then the residual E in place; columns K.. = the fp16 Q tile (V x 8, columns R.. zero)
            UQn_s = T.alloc_shared((NCG, 8, KV8), F16)         # new -U^T (rows R.. zero) for the residual mma; aliases: PQ_s (fp32 Q rows), P16 (fp16 P tile), cpart (column partials)
            PQ_s = T.view(UQn_s, (NCG, 2, KV8 // 2, 4), F32)   # plane 0 = Q (V x R) fp32 for the CholeskyQR rows
            UQn1 = T.view(UQn_s, (NCG, 8 * KV8), F16)          # flat view: the fp16 P tile P16[c][r] at P16OFF + 8 c + r
            cpart = T.view(UQn_s, (NCG, 8, KV8 // 2), F32)     # per-warp column partials of |E| (rows 0..3)
            rsk = T.view(UQn_s, (NCG, 8, KV8 // 2), F32)       # reciprocal column scales: row 6 of the cpart view (free once the partials are read)
            dlist = T.alloc_shared((MAXIT,), I32)          # this CTA's due programs p (global rank == bid mod SMS), in rank order
            ndue_s = T.alloc_shared((1,), I32)
            wsum_s = T.alloc_shared((NWARPS,), I32)        # per-warp due counts of the block-wide compaction
            if KDA:
                Kbl_s = T.alloc_shared((NSTAGE, 2, L, K // 2), BF16)     # per-entry decay factors (TMA target, fp16 bits) -> lo part of the decayed keys
                pb_s = T.alloc_shared((NSTAGE, K), F32)
                pc_s = T.alloc_shared((NCG, K), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            consumed = T.alloc_barrier([128] * NSTAGE)
            if KDA:
                T.annotate_layout({Qs: make_swizzled_layout(Qs), Kb_s: make_swizzled_layout(Kb_s), Ub_s: make_swizzled_layout(Ub_s), Kbl_s: make_swizzled_layout(Kbl_s), UQ_s: make_swizzled_layout(UQ_s)})
            else:
                T.annotate_layout({Qs: make_swizzled_layout(Qs), Kb_s: make_swizzled_layout(Kb_s), Ub_s: make_swizzled_layout(Ub_s), UQ_s: make_swizzled_layout(UQ_s)})
            tx = T.get_thread_binding()
            # ---- block-wide balanced compaction (no atomics): every CTA scans all N * HV due flags -- warp w takes the sequences
            #      n = w, w + NWARPS, ..., lane hv the head (one broadcast idx load + one coalesced 128 B hcnt load per n; an uncoalesced
            #      gather costs 32 L1 lookups per warp-load and was 8x slower) -- ranks the due programs with ballot/popc in the fixed order
            #      (warp, n-step, hv) and keeps the ranks == bid (mod SMS).  The flush is latency-bound at 1/16 duty (1-3 programs per
            #      group): the static assignment's binomial imbalance (max 12 of mean 3.6 per CTA at bs 256) cost ~1.4x; this gives +-1.
            #      dlist holds q = slot * HV + hv. ----
            tc = T.alloc_local((4,), "int64")
            if PROF:
                tc[0] = T.call_extern("int64", "clock64")
            wlane = tx % 32
            wid = tx // 32
            kn = T.alloc_local((1,), I32)
            kn[0] = T.ceildiv(N, NWARPS)                      # n-steps per warp
            cnt = T.alloc_local((1,), I32)
            cnt[0] = 0
            slk = T.alloc_local((KMAX,), I32)
            hck = T.alloc_local((KMAX,), I32)
            mk = T.alloc_local((KMAX,), "uint32")
            for k in T.unroll(KMAX):                            # all loads in flight at once (explicit clamps: no guard branches)
                slk[k] = idx[T.max(T.min(wid + NWARPS * k, N - 1), 0)]
            for k in T.unroll(KMAX):
                hck[k] = hcnt[T.min(T.max(slk[k], 0), NS - 1), wlane]
            for k in T.unroll(KMAX):
                f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N) & (slk[k] > 0) & (hck[k] == L), 1, 0)
                mk[k] = T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f)
                cnt[0] += T.call_extern(I32, "__popc", mk[k])
            sl8 = T.alloc_local((8,), I32)
            hc8 = T.alloc_local((8,), I32)
            for ck in T.serial(T.ceildiv(kn[0] - KMAX, 8)):    # tail (N > KMAX * NWARPS only): serial chunks, flags recomputed in the rank pass
                for j in T.unroll(8):
                    sl8[j] = idx[T.max(T.min(wid + NWARPS * (KMAX + ck * 8 + j), N - 1), 0)]
                for j in T.unroll(8):
                    hc8[j] = hcnt[T.min(T.max(sl8[j], 0), NS - 1), wlane]
                for j in T.unroll(8):
                    k = KMAX + ck * 8 + j
                    f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N) & (sl8[j] > 0) & (hc8[j] == L), 1, 0)
                    cnt[0] += T.call_extern(I32, "__popc", T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f))
            if PROF:
                tc[1] = T.call_extern("int64", "clock64")
            if wlane == 0:
                wsum_s[wid] = cnt[0]
            T.sync_threads(14, NPROD + NCONS)
            base = T.alloc_local((1,), I32)
            base[0] = 0
            for ww in T.serial(NWARPS):
                base[0] += T.if_then_else(ww < wid, wsum_s[ww], 0)
            tot = T.alloc_local((1,), I32)
            tot[0] = 0
            for ww in T.serial(NWARPS):
                tot[0] += wsum_s[ww]
            T.sync_threads(14, NPROD + NCONS)                 # wsum_s reads done before dlist (which may share its smem) is written
            lmask = T.cast((T.int64(1) << T.cast(wlane, "int64")) - T.int64(1), "uint32")
            if cnt[0] > 0:                                    # warps without due programs skip the pass (masks past kn are zero)
                rst = T.alloc_local((2,), I32)                    # next rank == bid (mod SMS) at or above base, and its dlist index (one division per thread)
                rst[0] = base[0] + (bid - base[0] % SMS + SMS) % SMS
                rst[1] = rst[0] // SMS
                for k in T.unroll(KMAX):                        # static indexing of the mask registers; one predicate per step, no division
                    m = mk[k]
                    rank = base[0] + T.call_extern(I32, "__popc", m & lmask)
                    if (((m >> T.cast(wlane, "uint32")) & T.uint32(1)) == T.uint32(1)) & (rank == rst[0]):
                        dlist[rst[1]] = idx[wid + NWARPS * k] * HV + wlane
                    base[0] += T.call_extern(I32, "__popc", m)
                    if rst[0] < base[0]:                        # the warp's rank sequence passed the target (warp-uniform; at most one target per step)
                        rst[0] += SMS
                        rst[1] += 1
                for ck in T.serial(T.ceildiv(kn[0] - KMAX, 8)):
                    for j in T.unroll(8):
                        sl8[j] = idx[T.max(T.min(wid + NWARPS * (KMAX + ck * 8 + j), N - 1), 0)]
                    for j in T.unroll(8):
                        hc8[j] = hcnt[T.min(T.max(sl8[j], 0), NS - 1), wlane]
                    for j in T.unroll(8):
                        k = KMAX + ck * 8 + j
                        f = T.if_then_else((k < kn[0]) & (wid + NWARPS * k < N) & (sl8[j] > 0) & (hc8[j] == L), 1, 0)
                        m = T.call_extern("uint32", "__ballot_sync", T.uint32(0xFFFFFFFF), f)
                        if f == 1:
                            rank = base[0] + T.call_extern(I32, "__popc", m & lmask)
                            if rank % SMS == bid:
                                dlist[rank // SMS] = sl8[j] * HV + wlane
                        base[0] += T.call_extern(I32, "__popc", m)
            if tx == 0:
                ndue_s[0] = T.max(T.ceildiv(tot[0] - bid, SMS), 0)
            if PROF:
                tc[2] = T.call_extern("int64", "clock64")
            if tx < NPROD:
                for i in T.serial(NSTAGE * 4 * (8 - R) * 64 // NPROD):     # zero the padding k-rows of UQ_s once (only rows < R are ever written)
                    e = tx + NPROD * i
                    UQ_s[e // (4 * (8 - R) * 64), (e // ((8 - R) * 64)) % 4, R + (e % ((8 - R) * 64)) // 64, e % 64] = T.cast(0.0, F16)
            T.sync_threads(14, NPROD + NCONS)
            if tx < NPROD:
                T.fence_proxy_async()
                for k in T.serial(ndue_s[0]):
                    q = dlist[k]
                    gc = k % NCG
                    ig = k // NCG
                    s = gc * SPG + ig % SPG
                    hv = q % HV
                    slot = q // HV
                    T.mbarrier_wait_parity(consumed[s], ((ig // SPG) & 1) ^ 1)
                    T.tma_copy(Sq[slot, hv, :, :], Qs[s, :, :], barrier=loaded[s])
                    T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                    T.tma_copy(Kbuf[slot, hv, :, 0:K // 2], Kb_s[s, 0, :, :], barrier=loaded[s])
                    T.tma_copy(Kbuf[slot, hv, :, K // 2:K], Kb_s[s, 1, :, :], barrier=loaded[s])
                    T.tma_copy(Ubuf[slot, hv, :, 0:V // 2], Ub_s[s, 0, :, :], barrier=loaded[s])
                    T.tma_copy(Ubuf[slot, hv, :, V // 2:V], Ub_s[s, 1, :, :], barrier=loaded[s])
                    T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                    for cb in T.unroll(4):
                        T.tma_copy(UQ[slot, hv, :, cb * 64:(cb + 1) * 64], UQ_s[s, cb, 0:R, :], barrier=loaded[s])
                    if KDA:
                        T.tma_copy(Gbuf[slot, hv, :, 0:K // 2], Kbl_s[s, 0, :, :], barrier=loaded[s])
                        T.tma_copy(Gbuf[slot, hv, :, K // 2:K], Kbl_s[s, 1, :, :], barrier=loaded[s])
                        T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                    T.mbarrier_arrive(loaded[s])
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                w = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                bar = 8 + g
                qw = T.alloc_local((4,), I32)
                svr_l = T.alloc_local((4,), F32)
                rmax = T.alloc_local((4,), F32)
                afrag = T.alloc_local((32 if not KDA else 16,), BF16)
                araw = T.alloc_local((8,), BF16)
                wv = T.alloc_local((4,), F32)
                bfrag = T.alloc_local((32 if KDA else 16,), BF16)  # 2 x (ring-buffer B tile, KDA: hi + lo) -- alternating per unrolled step
                afrag2 = T.alloc_local((8,), F16)           # rank-R A fragments (m16n8k8) per m-tile: old Q^T (residual)
                afrag3 = T.alloc_local((8,), F16)           # rank-R A fragments of the new factors (residual)
                cfrag = T.alloc_local((64,), F32)           # half of B / E in fragment layout: index (ntl * 4 + (mt * 2 + r)) * 2 + c1, ntl = local n-tile 0..7
                efrag = T.alloc_local((8,), F16)            # ldmatrix.x4 result: B values of (ntl, top), (ntl, bottom), (ntl+1, top), (ntl+1, bottom)
                colp = T.alloc_local((32,), F32)            # column partials of |E| (16 n-tiles x 2 columns of this lane)
                skl = T.alloc_local((16,), F32)             # this lane's 16 column scales of the current half (dequant)
                ah = T.alloc_local((32,), F16)              # m16n8k16 A fragments (ldmatrix.x4 from the fp16 St_s): 2 k-parities x 2 m-tiles
                bh2 = T.alloc_local((8,), F16)              # B fragments (Q or P column pairs, fp16, ldmatrix.x2.trans): 2 k-parities
                ao = T.alloc_local((16,), F16)              # A fragments of the old factor (rows 0..7 of a 16-row m-tile; rows 8..15 zero): 2 k-parities
                bfr = T.alloc_local((8,), F16)              # 2 x rank-R B fragment (residual)
                bhi = T.alloc_local((4,), I32)              # |E| B fragments (sign bits cleared) of one n-tile, 2 k-steps
                aone = T.alloc_local((8,), F16)             # A fragment of ones (column sums on the tensor cores)
                c32 = T.alloc_local((16,), F32)             # 2 m-tiles x 4, two accumulator chains (even/odd k-steps)
                mc = T.alloc_local((4,), F32)               # R x R correction matrix accumulator (M = Q_old^T Q, M2 = U_old^T P), full in every warp
                mm = T.alloc_local((8,), F32)               # M[a][2 q4 + i] for a = 0..3 (shuffled from lanes 4 a + q4)
                uo = T.alloc_local((16,), F32)              # old factor entries U_old[a][c] / Q_old[a][v] of this lane's 4 rows
                pr = T.alloc_local((4,), F32)               # this thread's row of Q
                gg = T.alloc_local((10,), F32)              # Gram entries (0,0) (1,0) (1,1) (2,0) (2,1) (2,2) (3,0) (3,1) (3,2) (3,3)
                ll = T.alloc_local((10,), F32)              # Cholesky factor, same packing
                uh = T.alloc_local((4,), F16)
                qh = T.alloc_local((4,), F16)
                el = T.alloc_local((L,), F32)               # KDA: this column's per-entry decay factors
                dcur = T.alloc_local((1,), F32)
                eh = T.alloc_local((32,), F16)              # quantisation: 2 x 2 x 8 halfwords of E (alternating per row pass)
                ph = T.alloc_local((8,), F16)               # fp16 pairs for stmatrix (vectorised cast -> one F2FP.PACK per pair)
                ex = T.alloc_local((16,), F32)              # quantisation: this lane's 16 scaled residual values of the row
                rk = T.alloc_local((16,), F32)              # quantisation: this lane's 16 reciprocal column scales
                tk = T.alloc_local((10,), "int64")
                tk2 = T.alloc_local((6,), "int64")
                tacc = T.alloc_local((10,), F32)
                for i in T.unroll(10):
                    tacc[i] = 0.0
                if PROF:
                    tacc[7] = T.cast(tc[1] - tc[0], F32)      # compaction: loads + ballots
                    tacc[9] = T.cast(tc[2] - tc[1], F32)      # compaction: scan + rank pass
                for i in T.unroll(8):
                    aone[i] = T.cast(1.0, F16)
                for i in T.unroll(2):                        # the old-factor A fragments: register pairs 1 and 3 (m-rows 8..15) stay zero
                    ao[2 + i] = T.cast(0.0, F16)
                    ao[6 + i] = T.cast(0.0, F16)
                    ao[10 + i] = T.cast(0.0, F16)
                    ao[14 + i] = T.cast(0.0, F16)
                for r in T.unroll(4):                        # the fp16 Q tile's padding columns R..7 stay zero (this thread's row)
                    St_s[g, t, K + 4 + r] = T.cast(0.0, F16)
                for r in T.unroll(4 - R):
                    St_s[g, t, K + R + r] = T.cast(0.0, F16)
                for it0 in T.serial(T.ceildiv(ndue_s[0] - g, NCG)):
                    k = g + it0 * NCG
                    q = dlist[k]
                    hv = q % HV
                    s = g * SPG + it0 % SPG
                    if PROF:
                        tk[0] = T.call_extern("int64", "clock64")
                    T.mbarrier_wait_parity(loaded[s], (it0 // SPG) & 1)
                    if PROF:
                        tk[1] = T.call_extern("int64", "clock64")
                    slot = q // HV
                    if KDA:
                        # pre-pass (thread per column c = t): pc = exp(G); decayed keys w_j k_j D_j as bf16 hi (in place) + lo (over the factors);
                        # U^T pre-multiplied by pc (the checkpoint's column decay)
                        pcc = T.exp(pb_s[s, t])
                        pc_s[g, t] = pcc
                        for jj in T.unroll(L):
                            el[jj] = T.cast(T.reinterpret(F16, Kbl_s[s, t // (K // 2), jj, t % (K // 2)]), F32)
                        T.sync_threads(bar, 128)                  # every lane holds its factors before the lo tile overwrites them
                        dcur[0] = 1.0
                        for jj in T.unroll(L):                    # backward: D_{L-1} = 1, D_j = D_{j+1} e_{j+1}
                            j = L - 1 - jj
                            kd = T.cast(Kb_s[s, t // (K // 2), j, t % (K // 2)], F32) * (Wb_s[s, j] * dcur[0])
                            khi = T.cast(kd, BF16)
                            Kb_s[s, t // (K // 2), j, t % (K // 2)] = khi
                            Kbl_s[s, t // (K // 2), j, t % (K // 2)] = T.cast(kd - T.cast(khi, F32), BF16)
                            dcur[0] = dcur[0] * el[j]
                        for r in T.unroll(R):                     # pc U^T in place (fp16: 2^-11 of the rank-R term, measured irrelevant next to the int8 residual, 10.46)
                            UQ_s[s, t // 64, r, t % 64] = T.cast(T.cast(UQ_s[s, t // 64, r, t % 64], F32) * pcc, F16)
                        T.sync_threads(bar, 128)
                        for i in T.unroll(4):
                            svr_l[i] = skv_s[s, K + w * 32 + (i // 2) * 16 + (i % 2) * 8 + lane // 4] * (1.0 / QMAX)
                    else:
                        pn = T.max(Pbuf[slot, hv, 0], PNMIN)     # floor: a fully decayed head has a denormal product (2e-41 seen on 9B layer 11) and 1/pn = inf
                        rpn = 1.0 / pn                           # poisoned the residual (Sk = inf -> NaN outputs, 10.49); the floor changes E by <= 1e-20 |Q_old U_old^T|
                    if KDA:
                        pn = 1.0
                    if not KDA:
                        for i in T.unroll(4):
                            svr_l[i] = skv_s[s, K + w * 32 + (i // 2) * 16 + (i % 2) * 8 + lane // 4] * (pn / QMAX)
                    # (1) A fragments of the ring buffer u_j (GDN: w_j-scaled, bf16 hi + lo; KDA: exact bf16)
                    if KDA:
                        for mt in T.unroll(2):
                            T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Ub_s[s, (w * 32 + mt * 16) // (V // 2), (lane % 8) + 8 * (lane // 16), (w * 32 + mt * 16) % (V // 2) + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(afrag[mt * 8], "w", extent=8))
                    else:
                        for i in T.unroll(4):
                            wv[i] = Wb_s[s, 2 * q4 + (i % 2) + 8 * (i // 2)]
                        for mt in T.unroll(2):
                            T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Ub_s[s, (w * 32 + mt * 16) // (V // 2), (lane % 8) + 8 * (lane // 16), (w * 32 + mt * 16) % (V // 2) + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(araw[0], "w", extent=8))
                            for i in T.unroll(8):
                                wu = wv[(i % 2) + 2 * (i // 4)] * T.cast(araw[i], F32)
                                afrag[mt * 16 + i] = T.cast(wu, BF16)
                                afrag[mt * 16 + 8 + i] = T.cast(wu - T.cast(afrag[mt * 16 + i], F32), BF16)
                    # (2) two 64-column halves: B = ring buffer (mma) + pn * deq(S0), out to St_s (fp16, stmatrix); the old rank-R part stays algebraic
                    for h in T.unroll(2):
                        # dequantise first (independent FFMA chains, no accumulator dependency), then the ring-buffer mma accumulates on top
                        for ntl in T.unroll(8):
                            for c1 in T.vectorized(2):
                                if KDA:
                                    skl[ntl * 2 + c1] = skv_s[s, (h * 8 + ntl) * 8 + q4 * 2 + c1] * pc_s[g, (h * 8 + ntl) * 8 + q4 * 2 + c1]
                                else:
                                    skl[ntl * 2 + c1] = skv_s[s, (h * 8 + ntl) * 8 + q4 * 2 + c1]
                        for rr in (T.unroll(0) if "nodeq" in ABL else T.unroll(4)):
                            v0 = w * 32 + (rr // 2) * 16 + (rr % 2) * 8 + r0
                            for jcl in T.unroll(4):                # 16-column chunk = n-tiles 2jc, 2jc+1; lane q4 needs word q4 // 2 of each n-tile
                                jc = h * 4 + jcl
                                for nn in T.unroll(2):
                                    qw[nn] = Qs[s, v0, jc * 4 + nn * 2 + q4 // 2]
                                for nn in T.unroll(2):
                                    ntl = jcl * 2 + nn
                                    pkw = qw[nn] ^ T.int32(-2139062144)
                                    for c1 in T.unroll(2):        # the magic-number subtraction must stay a separate exact fp32 op: folding -K*skl into an FFMA
                                        x = T.reinterpret(F32, T.call_extern(I32, "__byte_perm", pkw, T.int32(0x4B000000), T.int32(0x7650) + (q4 % 2) * 2 + c1)) - 8388736.0   # costs ulp(2^23 skl) = half a code
                                        cfrag[(ntl * 4 + rr) * 2 + c1] = x * svr_l[rr] * skl[ntl * 2 + c1]
                        # stmatrix.x4: matrices (ntl, rows 0-7), (ntl, rows 8-15), (ntl+1, rows 0-7), (ntl+1, rows 8-15) of m-tile mt
                        for npl in T.unroll(4):
                            n0 = (h * 4 + npl) * 16
                            bo = (npl % 2) * (16 if KDA else 8)
                            T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kb_s[s, n0 // (K // 2), (lane % 8) + 8 * ((lane // 8) % 2), n0 % (K // 2) + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo], "w", extent=8))
                            if KDA:
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(Kbl_s[s, n0 // (K // 2), (lane % 8) + 8 * ((lane // 8) % 2), n0 % (K // 2) + 8 * (lane // 16)], "r", extent=8), T.access_ptr(bfrag[bo + 8], "w", extent=8))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    if KDA:
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", afrag.data, mt * 8, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", afrag.data, mt * 8, bfrag.data, bo + 8 + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                    else:
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, mt * 16, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                                        if not DROPLO:
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", MMT, MMT, "fp32", afrag.data, mt * 16 + 8, bfrag.data, bo + nn * 4, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        for mt in (T.unroll(0) if "nostb" in ABL else T.unroll(2)):
                            for npl in T.unroll(4):
                                for jj in T.unroll(4):                     # 4 fp16 pairs: (ntl, rows 0-7), (ntl, rows 8-15), (ntl+1, ...) -- one F2FP.PACK each
                                    for c1 in T.vectorized(2):
                                        ph[jj * 2 + c1] = T.cast(cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1], F16)
                                T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x4",
                                              T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * 8 + npl * 2 + lane // 16) * 8], "w", extent=8),
                                              T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]), T.reinterpret(I32, ph[4:6]), T.reinterpret(I32, ph[6:8]))
                    # the iteration's start Q0 (fp16 tile in the St_s pad columns; this thread's row) -- before the barrier that publishes B
                    for r in T.unroll(R):
                        St_s[g, t, K + r] = T.cast(Q0[t, r], F16)
                    T.sync_threads(bar, 128)
                    if PROF:
                        tk[2] = T.call_extern("int64", "clock64")
                    # ================= subspace iteration on B (fp16 mma) with the algebraic rank-R corrections =================
                    for itr in T.serial(ITERS + 1):
                        if PROFIT:
                            tk2[0] = T.call_extern("int64", "clock64")
                        # ---- P = B^T Q  (K x R): A[m=c][k=v] = St_s[v][c] via ldmatrix.x4.trans; B[k=v][n=r] = Q16[v][r] via ldmatrix.x2.trans;
                        #      alongside M = Q_old^T Q (R x R): A[m=r][k=v] = Q_old^T (rows 8..15 zero), same B fragments ----
                        for i in T.unroll(16):
                            c32[i] = 0.0
                        for i in T.unroll(4):
                            mc[i] = 0.0
                        for k0 in T.unroll(V // 16):
                            kp = k0 % 2                                    # alternating fragment registers: the next step's ldmatrix can overlap this step's mma
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(St_s[g, k0 * 16 + 8 * ((lane // 8) % 2) + lane % 8, K], "r", extent=4), T.access_ptr(bh2[kp * 4], "w", extent=4))
                            for mt in T.unroll(2):
                                m0 = w * 32 + mt * 16
                                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(St_s[g, k0 * 16 + 8 * (lane // 16) + lane % 8, m0 + 8 * ((lane // 8) % 2)], "r", extent=8), T.access_ptr(ah[(kp * 2 + mt) * 8], "w", extent=8))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ah.data, (kp * 2 + mt) * 8, bh2.data, kp * 4, c32.data, (k0 % 2) * 8 + mt * 4, T.bool(False))
                            T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (K + k0 * 16) // 64, lane % 8, (K + k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                            T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (K + k0 * 16 + 8) // 64, lane % 8, (K + k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ao.data, kp * 8, bh2.data, kp * 4, mc.data, 0, T.bool(False))
                        for i in T.unroll(8):
                            c32[i] += c32[8 + i]
                        if PROFIT:
                            tk2[1] = T.call_extern("int64", "clock64")
                        # ---- P += pn U_old M in the fragments: rows c = m0 + r0 (+8), columns n = 2 q4 + i; M[a][n] sits in lane 4 a + q4 ----
                        for a in T.unroll(4):
                            mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                            mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                        for mt in T.unroll(2):
                            for hh in T.unroll(2):
                                cc = w * 32 + mt * 16 + hh * 8 + r0
                                for a in T.unroll(4):
                                    uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, cc // 64, a, cc % 64], F32), 0.0)
                                for i in T.unroll(2):
                                    c32[mt * 4 + hh * 2 + i] += pn * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                        # ---- the fp16 P tile P16[c][r] (columns >= R zero: lanes q4 >= R/2 write zeros) via stmatrix.x2 (rows 0-7, rows 8-15 of each m-tile) ----
                        for mt in T.unroll(2):
                            for c1 in T.vectorized(2):
                                ph[c1] = T.cast(T.if_then_else(2 * q4 + c1 < R, c32[mt * 4 + c1], 0.0), F16)
                                ph[2 + c1] = T.cast(T.if_then_else(2 * q4 + c1 < R, c32[mt * 4 + 2 + c1], 0.0), F16)
                            T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x2",
                                          T.access_ptr(UQn1[g, P16OFF + (w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8) * 8], "w", extent=8),
                                          T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]))
                        T.sync_threads(bar, 128)
                        if PROFIT:
                            tk2[2] = T.call_extern("int64", "clock64")
                            tacc[0] += T.cast(tk2[1] - tk2[0], F32)
                            tacc[1] += T.cast(tk2[2] - tk2[1], F32)
                        if itr < ITERS:
                            # ---- Q = B P  (V x R): A[m=v][k=c] = St_s[v][c] via ldmatrix.x4; B[k=c][n=r] = P16[c][r] via ldmatrix.x2.trans; alongside M2 = U_old^T P ----
                            for i in T.unroll(16):
                                c32[i] = 0.0
                            for i in T.unroll(4):
                                mc[i] = 0.0
                            for k0 in T.unroll(K // 16):
                                kp = k0 % 2
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQn1[g, P16OFF + (k0 * 16 + 8 * ((lane // 8) % 2) + lane % 8) * 8], "r", extent=4), T.access_ptr(bh2[kp * 4], "w", extent=4))
                                for mt in T.unroll(2):
                                    m0 = w * 32 + mt * 16
                                    T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(St_s[g, m0 + 8 * ((lane // 8) % 2) + lane % 8, k0 * 16 + 8 * (lane // 16)], "r", extent=8), T.access_ptr(ah[(kp * 2 + mt) * 8], "w", extent=8))
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ah.data, (kp * 2 + mt) * 8, bh2.data, kp * 4, c32.data, (k0 % 2) * 8 + mt * 4, T.bool(False))
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (k0 * 16) // 64, lane % 8, (k0 * 16) % 64], "r", extent=2), T.access_ptr(ao[kp * 8], "w", extent=2))
                                T.ptx_ldmatrix(T.bool(False), 1, T.access_ptr(UQ_s[s, (k0 * 16 + 8) // 64, lane % 8, (k0 * 16 + 8) % 64], "r", extent=2), T.access_ptr(ao[kp * 8 + 4], "w", extent=2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ao.data, kp * 8, bh2.data, kp * 4, mc.data, 0, T.bool(False))
                            for i in T.unroll(8):
                                c32[i] += c32[8 + i]
                            # ---- Q += pn Q_old M2 in the fragments, then the fp32 rows to PQ_s plane 0 for the CholeskyQR ----
                            for a in T.unroll(4):
                                mm[a * 2] = T.shfl_sync(mc[0], a * 4 + q4)
                                mm[a * 2 + 1] = T.shfl_sync(mc[1], a * 4 + q4)
                            for mt in T.unroll(2):
                                for hh in T.unroll(2):
                                    vv = w * 32 + mt * 16 + hh * 8 + r0
                                    for a in T.unroll(4):
                                        uo[(mt * 2 + hh) * 4 + a] = T.if_then_else(a < R, T.cast(UQ_s[s, (K + vv) // 64, a, (K + vv) % 64], F32), 0.0)
                                    for i in T.unroll(2):
                                        c32[mt * 4 + hh * 2 + i] += pn * (uo[(mt * 2 + hh) * 4] * mm[i] + uo[(mt * 2 + hh) * 4 + 1] * mm[2 + i] + uo[(mt * 2 + hh) * 4 + 2] * mm[4 + i] + uo[(mt * 2 + hh) * 4 + 3] * mm[6 + i])
                            if q4 * 2 < R:
                                for mt in T.unroll(2):
                                    PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * q4] = c32[mt * 4]
                                    PQ_s[g, 0, w * 32 + mt * 16 + r0, 2 * q4 + 1] = c32[mt * 4 + 1]
                                    PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * q4] = c32[mt * 4 + 2]
                                    PQ_s[g, 0, w * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = c32[mt * 4 + 3]
                            T.sync_threads(bar, 128)
                            if PROFIT:
                                tk2[3] = T.call_extern("int64", "clock64")
                                tacc[2] += T.cast(tk2[3] - tk2[2], F32)
                            # ---- CholeskyQR2 of Q (V rows = 128 threads), relative pivot floor for rank-deficient heads.  Pass 1 in fp32 (shuffle tree);
                            #      pass 2's Gram on the tensor cores in fp16 (8 mma; the factors are stored in fp16 anyway and the residual uses the same
                            #      fp16 factors, so a 2^-11 non-orthonormality costs nothing).  A 3xTF32 Gram for pass 1 was slower (48 dependent mma). ----
                            for r in T.unroll(4):
                                pr[r] = T.if_then_else(r < R, PQ_s[g, 0, t, r], 0.0)
                            for cq in range(2):
                                if cq == 0 or not GRAM16:         # pass 1 (kappa up to 1e3): fp32 Gram by the warp-shuffle tree + smem partials.  GRAM16=False: pass 2 too --
                                                                  # after a floored pass-1 pivot (rank-deficient head, e.g. svals 4.8 / 0.2 / 8e-3 / 1e-3) the 4th column has norm 1e-3..1e-5,
                                                                  # its fp16 Gram entry is subnormal/zero, rsqrt blows the column up by 1e3..1e15 and U_4 = B^T q_4 with it (10.49)
                                    for r in T.unroll(4):
                                        for c in T.unroll(r + 1):
                                            gg[r * (r + 1) // 2 + c] = pr[r] * pr[c]
                                    for off in (T.unroll(0) if "nogram" in ABL else T.unroll(5)):
                                        for i in T.unroll(10):
                                            gg[i] += T.shfl_xor(gg[i], 16 >> off)
                                    if lane == 0:
                                        for i in T.unroll(10):
                                            cpart[g, 4, w * 16 + i] = gg[i]           # per-warp Gram partials (cpart row 4: free during the iteration)
                                    T.sync_threads(bar, 128)
                                    for i in T.unroll(10):
                                        gg[i] = cpart[g, 4, i] + cpart[g, 4, 16 + i] + cpart[g, 4, 32 + i] + cpart[g, 4, 48 + i]
                                else:                             # pass 2 (Q nearly orthonormal): Gram on the tensor cores from the fp16 Q tile,
                                    for i in T.unroll(4):          # A = Q^T (rows R.. zero) and B = Q are the same ldmatrix.x2.trans fragment
                                        mc[i] = 0.0
                                    for k0 in T.unroll(V // 16):
                                        kp = k0 % 2
                                        T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(St_s[g, k0 * 16 + 8 * ((lane // 8) % 2) + lane % 8, K], "r", extent=4), T.access_ptr(bh2[kp * 4], "w", extent=4))
                                        for i in T.unroll(2):
                                            ao[kp * 8 + i] = bh2[kp * 4 + i]
                                            ao[kp * 8 + 4 + i] = bh2[kp * 4 + 2 + i]
                                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", ao.data, kp * 8, bh2.data, kp * 4, mc.data, 0, T.bool(False))
                                    gg[0] = T.shfl_sync(mc[0], 0)                             # G[a][c] sits in lane 4 a + c // 2, register c % 2
                                    gg[1] = T.shfl_sync(mc[0], 4)
                                    gg[2] = T.shfl_sync(mc[1], 4)
                                    gg[3] = T.shfl_sync(mc[0], 8)
                                    gg[4] = T.shfl_sync(mc[1], 8)
                                    gg[5] = T.shfl_sync(mc[0], 9)
                                    gg[6] = T.shfl_sync(mc[0], 12)
                                    gg[7] = T.shfl_sync(mc[1], 12)
                                    gg[8] = T.shfl_sync(mc[0], 13)
                                    gg[9] = T.shfl_sync(mc[1], 13)
                                ll[0] = T.rsqrt(T.max(gg[0], 1e-30))                       # reciprocals of the diagonal (1/L_jj)
                                ll[1] = gg[1] * ll[0]
                                ll[2] = T.rsqrt(T.max(T.max(gg[2] - ll[1] * ll[1], 1e-6 * gg[2]), 1e-30))
                                ll[3] = gg[3] * ll[0]
                                ll[4] = (gg[4] - ll[3] * ll[1]) * ll[2]
                                ll[5] = T.rsqrt(T.max(T.max(gg[5] - ll[3] * ll[3] - ll[4] * ll[4], 1e-6 * gg[5]), 1e-30))
                                ll[6] = gg[6] * ll[0]
                                ll[7] = (gg[7] - ll[6] * ll[1]) * ll[2]
                                ll[8] = (gg[8] - ll[6] * ll[3] - ll[7] * ll[4]) * ll[5]
                                ll[9] = T.rsqrt(T.max(T.max(gg[9] - ll[6] * ll[6] - ll[7] * ll[7] - ll[8] * ll[8], 1e-6 * gg[9]), 1e-30))
                                pr[0] = pr[0] * ll[0]
                                pr[1] = (pr[1] - pr[0] * ll[1]) * ll[2]
                                pr[2] = (pr[2] - pr[0] * ll[3] - pr[1] * ll[4]) * ll[5]
                                pr[3] = (pr[3] - pr[0] * ll[6] - pr[1] * ll[7] - pr[2] * ll[8]) * ll[9]
                                for r in T.unroll(R):
                                    St_s[g, t, K + r] = T.cast(pr[r], F16)                       # the fp16 Q tile (pass 2's Gram operand, then the next product)
                                T.sync_threads(bar, 128)
                            if PROFIT:
                                tk2[4] = T.call_extern("int64", "clock64")
                                tacc[3] += T.cast(tk2[4] - tk2[3], F32)
                    if PROF:
                        tk[3] = T.call_extern("int64", "clock64")
                    # ---- the fp16 factors (U^T[r][c] = P16[c][r], Q^T[r][v] = Q16[v][r]) to global and, after a barrier, -U^T over the P/Q planes for the residual mma ----
                    for r in T.unroll(R):
                        uh[r] = UQn1[g, P16OFF + t * 8 + r]
                        qh[r] = St_s[g, t, K + r]
                        UQ[slot, hv, r, t] = uh[r]
                        UQ[slot, hv, r, K + t] = qh[r]
                    T.sync_threads(bar, 128)
                    for r in T.unroll(R):
                        UQn_s[g, r, t] = -uh[r]                   # -U^T: the residual mma accumulates the subtraction
                    for r in T.unroll(8 - R):
                        UQn_s[g, R + r, t] = T.cast(0.0, F16)
                    T.sync_threads(bar, 128)
                    # ---- residual E = fp16(B) + pn Q_old U_old^T - Q U^T in two halves (fp16 mma, fp32 accumulate), |E| column partials, E over B in St_s ----
                    for mt in T.unroll(2):        # A[m=v][k=r] = Q_old^T[r][v]: x2.trans -> (m 0-7), (m 8-15) of k-rows 0-7
                        T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, (K + w * 32 + mt * 16 + 8 * ((lane // 8) % 2)) // 64, lane % 8, (K + w * 32 + mt * 16 + 8 * ((lane // 8) % 2)) % 64], "r", extent=4), T.access_ptr(afrag2[mt * 4], "w", extent=4))
                        # A[m=v][k=r] = Q16[v][r] (row-major [m][k], k 0-7): x2 (non-trans) -> rows 0-7, rows 8-15
                        T.ptx_ldmatrix(T.bool(False), 2, T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, K], "r", extent=4), T.access_ptr(afrag3[mt * 4], "w", extent=4))
                    for h in T.unroll(2):
                        # E = pn (B / pn + Q_old U_old^T) - Q U^T: the accumulator starts from B (scaled by 1 / pn on GDN, loaded with ldmatrix.x4 in the
                        # fragment layout), so both rank-R mma chains accumulate straight into it and there is no elementwise add afterwards
                        for mt in T.unroll(2):
                            for npl in T.unroll(4):
                                T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * 8 + npl * 2 + lane // 16) * 8], "r", extent=8), T.access_ptr(efrag[0], "w", extent=8))
                                for jj in T.unroll(4):
                                    for c1 in T.unroll(2):
                                        if KDA:
                                            cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1] = T.cast(efrag[jj * 2 + c1], F32)
                                        else:
                                            cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1] = T.cast(efrag[jj * 2 + c1], F32) * rpn
                        for npl in T.unroll(4):
                            n0 = (h * 4 + npl) * 16
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQ_s[s, n0 // 64, lane % 8, n0 % 64 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k8", "row", "col", "fp16", "fp16", "fp32", afrag2.data, mt * 4, bfr.data, (npl % 2) * 4 + nn * 2, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        if not KDA:
                            for i in T.unroll(64):
                                cfrag[i] = cfrag[i] * pn
                        for npl in T.unroll(4):
                            n0 = (h * 4 + npl) * 16
                            T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(UQn_s[g, lane % 8, n0 + 8 * ((lane // 8) % 2)], "r", extent=4), T.access_ptr(bfr[(npl % 2) * 4], "w", extent=4))
                            for mt in T.unroll(2):
                                for nn in T.unroll(2):
                                    T.ptx_mma("float32", "m16n8k8", "row", "col", "fp16", "fp16", "fp32", afrag3.data, mt * 4, bfr.data, (npl % 2) * 4 + nn * 2, cfrag.data, ((npl * 2 + nn) * 4 + mt * 2) * 2, T.bool(False))
                        for mt in T.unroll(2):
                            for npl in T.unroll(4):
                                if "noste" not in ABL:
                                    for jj in T.unroll(4):
                                        for c1 in T.vectorized(2):
                                            ph[jj * 2 + c1] = T.cast(cfrag[((npl * 2 + jj // 2) * 4 + mt * 2 + jj % 2) * 2 + c1], F16)
                                    T.call_extern("handle", "tl::ptx_stmatrix_m8n8_x4",
                                                  T.access_ptr(St_s[g, w * 32 + mt * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * 8 + npl * 2 + lane // 16) * 8], "w", extent=8),
                                                  T.reinterpret(I32, ph[0:2]), T.reinterpret(I32, ph[2:4]), T.reinterpret(I32, ph[4:6]), T.reinterpret(I32, ph[6:8]))
                        # ---- |E| column partials of this warp's 32 rows on the tensor cores (this half's 8 n-tiles): C[m][c] = sum_v 1 * |E[v][c]| with
                        #      A = ones and B = |E| read back from the fp16 E tile (ldmatrix.x2.trans + sign-bit clear), accumulated into the dead cfrag;
                        #      every m-row of C holds the sums -- lane rows r0 == 0 are used.  No shuffle reduction. ----
                        T.sync_warp()
                        for i in T.unroll(32):
                            cfrag[i] = 0.0
                        for ntl in T.unroll(8):
                            for ks in T.unroll(2):
                                T.ptx_ldmatrix(T.bool(True), 2, T.access_ptr(St_s[g, w * 32 + ks * 16 + 8 * ((lane // 8) % 2) + lane % 8, (h * 8 + ntl) * 8], "r", extent=4), T.access_ptr(bh2[ks * 4], "w", extent=4))
                            for i in T.unroll(4):
                                bhi[i] = T.reinterpret(I32, bh2[i * 2:i * 2 + 2]) & T.int32(0x7FFF7FFF)
                            for ks in T.unroll(2):
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", aone.data, 0, bhi.data, ks * 2, cfrag.data, ntl * 4, T.bool(False))
                        for ntl in T.unroll(8):
                            for c1 in T.unroll(2):
                                colp[(h * 8 + ntl) * 2 + c1] = cfrag[ntl * 4 + c1]
                    T.sync_threads(bar, 128)                  # the residual mma operands (UQn_s, UQ_s) are consumed: the partials may take the buffer, the stage is free
                    T.fence_proxy_async()
                    T.mbarrier_arrive(consumed[s])
                    if r0 == 0:
                        for nt in T.unroll(16):
                            for c1 in T.vectorized(2):
                                cpart[g, w, nt * 8 + q4 * 2 + c1] = colp[nt * 2 + c1]
                    T.sync_threads(bar, 128)
                    if PROF:
                        tk[4] = T.call_extern("int64", "clock64")
                    # (4) column scale s_k (thread per column) from the 4 warp partials, reciprocal to smem
                    msum = (cpart[g, 0, t] + cpart[g, 1, t] + cpart[g, 2, t] + cpart[g, 3, t]) * (1.0 / V)
                    rskn = T.min(T.rsqrt(msum), 1.0 / EPS)                      # 1/max(sqrt(m), EPS) without the IEEE division slow path
                    Sk[slot, hv, t] = T.max(T.sqrt(msum), EPS)
                    rsk[g, 6, t] = rskn
                    T.sync_threads(bar, 128)
                    if PROF:
                        tk[5] = T.call_extern("int64", "clock64")
                    # (5) row-wise quantisation: warp w rows 32w.., 8 lanes per row (16 columns each), 4 rows per pass;
                    #     row max over the 8 lanes, int8 codes packed in registers, one coalesced 16 B store per lane and row
                    for i in T.unroll(4):
                        for c1 in T.vectorized(4):
                            rk[i * 4 + c1] = rsk[g, 6, (lane % 8) * 16 + i * 4 + c1]
                    for i in T.unroll(8):
                        v0 = w * 32 + i * 4 + lane // 8
                        rmax[0] = 0.0
                        for hh in T.unroll(2):
                            for c1 in T.vectorized(8):
                                eh[((i % 2) * 2 + hh) * 8 + c1] = St_s[g, v0, (lane % 8) * 16 + hh * 8 + c1]
                        for hh in T.unroll(2):
                            for c1 in T.unroll(8):
                                ex[hh * 8 + c1] = T.cast(eh[((i % 2) * 2 + hh) * 8 + c1], F32) * rk[hh * 8 + c1]
                                rmax[0] = T.max(rmax[0], T.abs(ex[hh * 8 + c1]))
                        rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 1))
                        rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 2))
                        rmax[0] = T.max(rmax[0], T.shfl_xor(rmax[0], 4))
                        rmax[0] = T.max(rmax[0], EPS)
                        rmax[1] = QMAX * T.call_extern(F32, "__frcp_rn", rmax[0])
                        if lane % 8 == 0:
                            Sv[slot, hv, v0] = rmax[0]
                        if "nocode" not in ABL:
                            for c1 in T.unroll(4):
                                qw[c1] = T.call_extern(I32, "__byte_perm",
                                                       T.call_extern(I32, "__byte_perm", T.reinterpret(I32, ex[c1 * 4] * rmax[1] + 12582912.0), T.reinterpret(I32, ex[c1 * 4 + 1] * rmax[1] + 12582912.0), T.int32(0x0040)),
                                                       T.call_extern(I32, "__byte_perm", T.reinterpret(I32, ex[c1 * 4 + 2] * rmax[1] + 12582912.0), T.reinterpret(I32, ex[c1 * 4 + 3] * rmax[1] + 12582912.0), T.int32(0x0040)),
                                                       T.int32(0x5410))
                            for c1 in T.vectorized(4):
                                Sq[slot, hv, v0, (lane % 8) * 4 + c1] = qw[c1]
                    if PROF:
                        tk[6] = T.call_extern("int64", "clock64")
                    if t < L:
                        Wbuf[slot, hv, t] = 0.0
                    if KDA:
                        Pbuf[slot, hv, t] = 0.0
                        for jj in T.unroll(L):
                            Gbuf[slot, hv, jj, t] = T.reinterpret(BF16, T.cast(1.0, F16))          # entry factors -> fp16 1.0
                    else:
                        if t == 0:
                            Pbuf[slot, hv, 0] = 1.0
                    # hcnt stays at L: 'flushed, ring empty' for the step kernels (a reset here would change the due flags other CTAs are still scanning)
                    T.sync_threads(bar, 128)          # St_s / UQn_s reuse by the next program of this group
                    if PROF:
                        tk[7] = T.call_extern("int64", "clock64")
                        if not PROFIT:
                            for i in T.unroll(7):
                                tacc[i] += T.cast(tk[i + 1] - tk[i], F32)
                        tacc[8] += 1.0
                if PROF and g == 0 and t == 0 and bid < HV:
                    for i in T.unroll(10):
                        Wbuf[0, bid, i] = tacc[i]
    return flush



class Pool:
    """Caller-owned state pool, the layout the serving engine uses.  Every (slot, head) owns a V x K fp32-sized region
    (64 KB) of one flat buffer; the checkpoint lives in its first 19 KB:
        bytes [0, 16384)            int8 codes, row-major [V, K]
        bytes [16384, 16896)        fp32 column scales s_k [K]
        bytes [16896, 17408)        fp32 row scales    s_v [V]
        bytes [17408, 19456)        fp16 Compensator Tokens [R, K + V]: row r = (u[r, :K] | q[r, :V])
    Slots are SS fp32 words apart; slot 0 is reserved (idx 0 = "no sequence").  The update buffer (kbuf, ubuf, w, p) and the
    per-(slot, head) fill counter hcnt are separate dense tensors.  A program (slot, head) is due when hcnt == L."""

    def __init__(self, num_slots, device="cuda"):
        NS = self.NS = num_slots
        self.raw = torch.zeros(NS * SS + OFF, dtype=torch.float32, device=device); st = self.raw.untyped_storage()
        def view(dtype, shape, strides, off):
            return torch.empty(0, dtype=dtype, device=device).set_(st).as_strided(shape, strides, off)
        self.codes = view(torch.int8, (NS, HV, V, K), (4 * SS, 4 * V * K, K, 1), 4 * OFF)
        self.codes32 = view(torch.int32, (NS, HV, V, K // 4), (SS, V * K, K // 4, 1), OFF)
        self.codes16 = view(torch.int16, (NS, HV, V, K // 2), (2 * SS, 2 * V * K, K // 2, 1), 2 * OFF)
        self.s_k = view(torch.float32, (NS, HV, K), (SS, V * K, 1), OFF + V * K // 4)
        self.s_v = view(torch.float32, (NS, HV, V), (SS, V * K, 1), OFF + V * K // 4 + K)
        self.s_kv = view(torch.float32, (NS, HV, K + V), (SS, V * K, 1), OFF + V * K // 4)
        self.uq = view(torch.float16, (NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), 2 * (OFF + V * K // 4 + K + V))
        self.kbuf = torch.zeros(NS, HV, L, K, dtype=torch.bfloat16, device=device)
        self.ubuf = torch.zeros(NS, HV, L, V, dtype=torch.bfloat16, device=device)
        self.w = torch.zeros(NS, HV, L, device=device)
        self.p = torch.ones(NS, HV, 1, device=device)
        self.hcnt = torch.zeros(NS, HV, dtype=torch.int32, device=device)
        self._gate = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=device)      # placeholder argument (KDA variant only)
        self.kernel = make_flush_kernel(NS, HV, K, V, L, SS, R=R, ITERS=1, NCG=3,
                                        SMS=torch.cuda.get_device_properties(self.raw.device).multi_processor_count)


def flush_inplace(pool, idx, q0):
    """Deployment form.  idx: int32 [N] slot per sequence (0 = skip).  For every (idx[n], head) with hcnt == L: rebuild the
    state from the checkpoint and the update buffer, write the new checkpoint over the old one, reset w = 0, p = 1, hcnt = 0."""
    pool.kernel(pool.codes32, pool.s_k, pool.s_v, pool.s_kv, pool.kbuf, pool.ubuf, pool.w, pool.p, pool._gate, pool.hcnt,
                pool.codes16, pool.uq, q0, idx)


_POOLS = {}


def run(codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0):
    B = codes.shape[0]
    pool = _POOLS.get((B, codes.device))
    if pool is None:
        pool = _POOLS[(B, codes.device)] = Pool(B + 8, codes.device)
    sl = slice(1, B + 1)
    pool.codes[sl] = codes; pool.s_k[sl] = s_k; pool.s_v[sl] = s_v; pool.uq[sl, :, :, :K] = u; pool.uq[sl, :, :, K:] = q
    pool.kbuf[sl] = kbuf; pool.ubuf[sl] = ubuf; pool.w[sl] = w; pool.p[sl, :, 0] = p
    pool.hcnt.zero_(); pool.hcnt[sl] = L
    flush_inplace(pool, torch.arange(1, B + 1, dtype=torch.int32, device=codes.device), q0.contiguous())
    return (pool.codes[sl].clone(), pool.s_k[sl].clone(), pool.s_v[sl].clone(),
            pool.uq[sl, :, :, :K].clone(), pool.uq[sl, :, :, K:].clone())
