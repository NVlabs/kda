# SPDX-License-Identifier: Apache-2.0
"""Best-known LeapQuant decode-step kernel for Kimi Delta Attention (per-key-channel gate; see README.md).

TileLang kernel for sm_100 (B200): one persistent CTA per SM, a TMA producer warp, three norm warps, two consumer groups of
4 warps; the int8 checkpoint tile goes through the tensor cores (int8 -> fp16 by PRMT, exact; mma.sync m16n8k16 against an
fp16 hi/lo split of the scaled and gate-decayed key / query), the rest in fp32 on CUDA cores.

  run(...)              functional form with the input order / outputs of definition.json (packs the inputs into a slot
                        pool, calls the kernel, returns the outputs as fresh tensors) -- used for correctness
  Pool, step_inplace    deployment form: caller-owned strided slot pool, the buffered update appended in place -- timed

Requires: torch (CUDA), tilelang 0.1.12, CUDA toolkit 13 (nvcc), NVIDIA B200.
"""
import torch, tilelang, tilelang.language as T
from tilelang.layout import make_swizzled_layout

F32, BF16, I32 = "float32", "bfloat16", "int32"
QMAX = 127.0
SOFTPLUS_THRESHOLD = 20.0
HV, HK, K, V, L, R = 32, 32, 128, 128, 16, 4  # heads (no GQA), key dim, value dim, window, Compensator Tokens
SS, OFF = 33 * V * K, 12288                   # pool geometry in fp32 words: slot stride (one spare head per slot), base offset


@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_step_kernel(NS, HV, H, K, V, L, SS, R=4, SMS=148, NSTAGE=6, NSTAGEQ=4, NCG=2, NNW=2, scale=128 ** -0.5, MAXIT=256, PROF=False, GB16=True, DBG=0, TMAPG=False, FASTG=True, DFOLD=True, VECD=True, MMADOT=False, DNORM=False, WFOLD=True, WEXACT=False, GLB=0.0, DLO=True, **_ignored):
    """TileLang generator of the KDA decode-step kernel (see the module docstring).
      warp 0           : TMA ring (checkpoint tile on its own, shallower ring; the rest per stage)
      norm warps       : l2 norms, the per-channel gate (cumulative log-gate, this step's decay factor, the checkpoint's decay
                         pc), the fp16 mma B operand [kf qf] / kfmax with pc folded in, the per-program scalars; they also append
                         the new cumulative log-gate and decay factor
      consumers        : the decayed buffered keys k_j * D_j (D_j = this step's factor times the later entries' factors) as a
                         bf16 hi (in place) + bf16 lo (per-group tile) pair, int8 tile -> fp16 (exact) -> mma.m16n8k16 against
                         [kf qf], the buffer and factor dots, one named barrier, contribution, output, global appends.
    Numerics: kf / qf are fp16 hi / lo after scaling by 1 / max|kf, qf|; the int8 -> fp16 conversion is exact; accumulation is fp32;
    the gate uses __expf / __logf (FASTG); the decayed keys carry 16 significant bits (hi + lo).  The other keyword switches are
    ablations (defaults are the deployed configuration)."""
    # FASTG: the gate is 4 transcendentals per key channel x 128 channels per program, all inside the norm warp,
    # which arrives normed[s] -- i.e. straight on every consumer's critical path.  TileLang lowers T.exp / T.log to
    # the ACCURATE expf / logf (range reduction + polynomial, ~15-20 instructions); __expf / __logf are one MUFU
    # each and ~2 ulp, far below the bf16 ring's own error.
    # FASTG is load-bearing, and not only for instruction count: with the accurate expf/logf the norm warp's
    # register demand raises the whole kernel's budget, and the extra locals VECD/DFOLD introduce then spill --
    # FASTG=0 with the rest on measures 0.0803 ms at bs 256 against 0.0617 with it, i.e. worse than not having
    # made any of these changes.  The four only pay off together.
    FEXP = (lambda x: T.call_extern(F32, "__expf", x)) if FASTG else T.exp
    FLOG = (lambda x: T.call_extern(F32, "__logf", x)) if FASTG else T.log
    # GLB != 0 (another model's gate): bounded gate, log-decay GLB * sigmoid(exp(A_log) (a + dt_bias)) in (GLB, 0) instead of
    # Kimi's -exp(A_log) softplus(a + dt_bias).  GLB == 0 builds exactly the old expressions.
    KGATE_SP = (lambda xg, ea: 1.0 / (1.0 + FEXP(-ea * xg))) if GLB != 0.0 else (lambda xg, ea: T.if_then_else(xg <= SOFTPLUS_THRESHOLD, FLOG(1.0 + FEXP(xg)), xg))
    KGATE_GN = (lambda pb, ea, spg: pb + GLB * spg) if GLB != 0.0 else (lambda pb, ea, spg: pb - ea * spg)
    KGATE_EN = (lambda ea, spg: FEXP(GLB * spg)) if GLB != 0.0 else (lambda ea, spg: FEXP(-ea * spg))
    GBT = "float16" if GB16 else F32                # dtype of the per-entry gate history (fp16: half the bytes; |G| x 2^-11 abs error in the exponent)
    GQA = HV // H
    N = T.symbolic("N")
    KW = K // 4
    NCONS = 128 * NCG
    assert NSTAGE % NCG == 0
    SPG = NSTAGE // NCG
    # TMAPG (one TMA warp per consumer group) was a win when it was introduced, but with the fast-math gate and
    # the decay folded into the ring tile it is a 4-5% LOSS at every batch size (0.0659 -> 0.0630 at bs 256), so
    # the default is back to a single TMA warp.  It is also the only thing that pins the kernel to NCG == 2.
    assert NSTAGEQ % NCG == 0
    SPGQ = NSTAGEQ // NCG                           # the int8 tile has its own (shallower) ring: it is dead after the checkpoint stream
    assert NSTAGE % NNW == 0, "every stage must be normed by ONE warp in fill order (program it -> warp it % NNW): otherwise a stage's fills alternate between warps, and a warp >= 2 fills ahead passes its mbarrier parity wait early (phase aliasing) -> stale data / deadlock"
    assert not (DLO and MMADOT), "DLO is written for the CUDA-core buffer dots"
    NTW = NCG if TMAPG else 1                       # TMA warps: one per consumer group (TMAPG) or a single one
    assert NCG == 2 or not TMAPG, "per-group TMA warps are written out for NCG == 2"
    NPROD = 32 * (NTW + NNW)
    MIX = 2 * H * K + HV * V
    F16 = "float16"
    assert K == 128 and V == 128 and L <= 16
    MAGICX = T.int32(-2139062144)                  # 0x80808080: int8 -> offset byte
    H2MAG = T.int32(0x64646464)                    # fp16 exponent byte: 0x64xx = 1024 + xx

    @T.prim_func
    def step(Sq: T.StridedTensor((NS, HV, V, KW), (SS, V * K, KW, 1), I32),
             Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
             Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
             Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), F32),
             Kbuf: T.Tensor((NS, HV, L, K), BF16), Ubuf: T.Tensor((NS, HV, L, V), BF16),
             Wbuf: T.Tensor((NS, HV, L), F32), Pbuf: T.Tensor((NS, HV, K), F32), Gbuf: T.Tensor((NS, HV, L, K), GBT), hcnt: T.Tensor((NS, HV), I32),
             mixed: T.Tensor((N, MIX), BF16), a: T.Tensor((N, HV, K), BF16), b: T.Tensor((N, HV), BF16),
             A_log: T.Tensor((HV,), F32), dt_bias: T.Tensor((HV, K), F32),
             UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),
             idx: T.Tensor((N,), I32), o: T.Tensor((N, HV, V), BF16)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGEQ, V, KW), I32)
            UQ_s = T.alloc_shared((NSTAGE, R, K + V), F16)
            skv_s = T.alloc_shared((NSTAGE, K + V), F32)
            qx_s = T.alloc_shared((NSTAGE, K), BF16)
            kx_s = T.alloc_shared((NSTAGE, K), BF16)
            vx_s = T.alloc_shared((NSTAGE, V), BF16)
            Kb_s = T.alloc_shared((NSTAGE, L, K), BF16)
            Ub_s = T.alloc_shared((NSTAGE, L, V), BF16)
            T.annotate_layout({Kb_s: make_swizzled_layout(Kb_s)})     # the buffer dots read 32 B per lane over 4 rows per warp: 4-way bank conflicts unswizzled
                                                                     # (UQ_s cannot be swizzled: its 512 B rows are not the TMA swizzle granularity)
            Wb_s = T.alloc_shared((NSTAGE, 32), F32)
            Gb_s = T.alloc_shared((NSTAGE, L, K), GBT)         # per-entry cumulative log-gate at insertion (KDA)
            pb_s = T.alloc_shared((NSTAGE, K), F32)            # slot's cumulative log-gate since the checkpoint
            ax_s = T.alloc_shared((NSTAGE, K), BF16)           # this token's raw gate input a[n, hv, :]
            dtb_s = T.alloc_shared((NSTAGE, K), F32)           # dt_bias[hv, :] (TMA'd per program: no global latency in the norm warp)
            alog_s = T.alloc_shared((HV,), F32)
            klo_s = T.alloc_shared((NCG, 2, L, K) if DLO else (1, 1, 1, 8), BF16)   # DLO: lo part of the decayed keys, per group and program parity
            if DLO:
                T.annotate_layout({klo_s: make_swizzled_layout(klo_s)})
            # ---- per-stage data produced by the norm warp ----
            bf_s = T.alloc_shared((NSTAGE, 8, K + 16), F16)   # B operand: rows 0/1 = hi(kf, qf)/kfmax, 2/3 = lo parts, 4..7 = 0
                                                             # (+16: row stride 288 B -> the 8 r0 rows of a fragment load land on different banks; K = 8-way conflicts)
            kn_s = T.alloc_shared((NSTAGE, (K // 16) * 20), F32)   # chunk-padded: column c at (c // 16) * 20 + c % 16 (conflict-free 64 B chunk reads)
            qn_s = T.alloc_shared((NSTAGE, (K // 16) * 20), F32)
            en_s = T.alloc_shared((NSTAGE, K), F32)            # this step's decay factor exp(g_now)
            pc_s = T.alloc_shared((NSTAGE, K), F32)            # pc = exp(cumulative log-gate): applied to the factor dots in fp32 (U^T stays the stored fp16)
            bn_s = T.alloc_shared((NSTAGE, 8 if MMADOT else 1, K if MMADOT else 1), BF16)   # ring-dot B operand
            # DFOLD: the per-entry decay is folded into the staged ring tile in place, so there is no D_s array:
            # the dot then reads only Kb_s (as the GDN kernel does), and the 20 KiB this used to cost is what
            # made NCG=2 the only affordable width.
            D_s = T.alloc_shared((1 if DFOLD else NCG, 1 if DFOLD else 2, 1 if DFOLD else L, 1 if DFOLD else (K // 16) * 20), F16)
            w_s = T.alloc_shared((NSTAGE, L), F32)
            sc_s = T.alloc_shared((NSTAGE, 8), F32)           # 0 kfmax, 1 kq, 2 pn, 3 beta
            # ---- per-group scratch, double-buffered by program parity (no end-of-program barrier) ----
            mqy_s = T.alloc_shared((NCG, 2, V, 4), F32)        # [mq_hi, yq_hi, mq_lo, yq_lo] per row
            dk_s = T.alloc_shared((NCG, 2, L + R), F32)
            dq_s = T.alloc_shared((NCG, 2, L + R), F32)
            slot_s = T.alloc_shared((MAXIT,), I32)
            h_s = T.alloc_shared((MAXIT,), I32)
            b_s = T.alloc_shared((MAXIT,), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            normed = T.alloc_barrier([32] * NSTAGE)
            consumed = T.alloc_barrier([128] * NSTAGE)
            loadedQ = T.alloc_barrier([32] * NSTAGEQ)
            consumedQ = T.alloc_barrier([128] * NSTAGEQ)
            T.annotate_layout({Qs: make_swizzled_layout(Qs)})
            tx = T.get_thread_binding()
            nprog = N * HV
            nit = T.ceildiv(nprog - bid, SMS)
            if tx < NPROD:
                for j in T.serial(T.ceildiv(nit, NPROD)):
                    it = tx + NPROD * j
                    if it < nit:
                        p = bid + it * SMS
                        n = p // HV
                        hv = p % HV
                        slot = idx[n]
                        slot_s[it] = slot
                        hh = hcnt[T.max(slot, 0), hv]
                        h_s[it] = T.if_then_else(hh >= L, 0, hh)   # == L: the flush emptied the ring (it leaves hcnt at L so its due-scan is stable across CTAs)
                        b_s[it] = T.sigmoid(T.cast(b[n, hv], F32)) if WEXACT else T.cast(T.cast(T.sigmoid(T.cast(b[n, hv], F32)), BF16), F32)
                for j in T.serial(T.ceildiv(HV, NPROD)):
                    if tx + NPROD * j < HV:
                        alog_s[tx + NPROD * j] = T.exp(A_log[tx + NPROD * j])
                T.sync_threads(15, NPROD)
                # TMA warps as literal 32-thread regions: a Python loop becomes a TIR loop here and the election extent would be 128
                if TMAPG:
                    if tx >= 0 and tx < 32:        # ---- TMA warp 0: the stages of consumer group 0 (TMAPG) / all stages in fill order ----
                      tp = T.alloc_local((2,), "int64")
                      tpacc = T.alloc_local((2,), F32)
                      tpacc[0] = 0.0
                      tpacc[1] = 0.0
                      for ig in T.serial(T.ceildiv(nit - 0, NCG) if TMAPG else nit):
                          it = (0 + ig * NCG) if TMAPG else ig
                          p = bid + it * SMS
                          gc = 0 if TMAPG else it % NCG
                          ig2 = ig if TMAPG else it // NCG
                          s = gc * SPG + ig2 % SPG
                          sq = gc * SPGQ + ig2 % SPGQ
                          n = p // HV
                          hv = p % HV
                          hk = hv // GQA
                          slot = slot_s[it]
                          if PROF:
                              tp[0] = T.call_extern("int64", "clock64")
                          T.mbarrier_wait_parity(consumedQ[sq], ((ig2 // SPGQ) & 1) ^ 1)
                          if slot > 0:
                              T.tma_copy(Sq[slot, hv, :, :], Qs[sq, :, :], barrier=loadedQ[sq])
                          T.mbarrier_arrive(loadedQ[sq])
                          T.mbarrier_wait_parity(consumed[s], ((ig2 // SPG) & 1) ^ 1)
                          if PROF:
                              tpacc[0] += T.cast(T.call_extern("int64", "clock64") - tp[0], F32)
                              tpacc[1] += 1.0
                          if slot > 0:
                              T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                              T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                              T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Gbuf[slot, hv, :, :], Gb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                              T.tma_copy(a[n, hv, :], ax_s[s, :], barrier=loaded[s])
                              T.tma_copy(dt_bias[hv, :], dtb_s[s, :], barrier=loaded[s])
                          T.mbarrier_arrive(loaded[s])
                      if PROF and tx == 0 and bid < HV and 0 == 0:
                          Wbuf[0, bid, 8] = tpacc[0]
                          Wbuf[0, bid, 9] = tpacc[1]
                    if tx >= 32 and tx < 64:        # ---- TMA warp 1: the stages of consumer group 1 (TMAPG) / all stages in fill order ----
                      tp = T.alloc_local((2,), "int64")
                      tpacc = T.alloc_local((2,), F32)
                      tpacc[0] = 0.0
                      tpacc[1] = 0.0
                      for ig in T.serial(T.ceildiv(nit - 1, NCG) if TMAPG else nit):
                          it = (1 + ig * NCG) if TMAPG else ig
                          p = bid + it * SMS
                          gc = 1 if TMAPG else it % NCG
                          ig2 = ig if TMAPG else it // NCG
                          s = gc * SPG + ig2 % SPG
                          sq = gc * SPGQ + ig2 % SPGQ
                          n = p // HV
                          hv = p % HV
                          hk = hv // GQA
                          slot = slot_s[it]
                          if PROF:
                              tp[0] = T.call_extern("int64", "clock64")
                          T.mbarrier_wait_parity(consumedQ[sq], ((ig2 // SPGQ) & 1) ^ 1)
                          if slot > 0:
                              T.tma_copy(Sq[slot, hv, :, :], Qs[sq, :, :], barrier=loadedQ[sq])
                          T.mbarrier_arrive(loadedQ[sq])
                          T.mbarrier_wait_parity(consumed[s], ((ig2 // SPG) & 1) ^ 1)
                          if PROF:
                              tpacc[0] += T.cast(T.call_extern("int64", "clock64") - tp[0], F32)
                              tpacc[1] += 1.0
                          if slot > 0:
                              T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                              T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                              T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Gbuf[slot, hv, :, :], Gb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                              T.tma_copy(a[n, hv, :], ax_s[s, :], barrier=loaded[s])
                              T.tma_copy(dt_bias[hv, :], dtb_s[s, :], barrier=loaded[s])
                          T.mbarrier_arrive(loaded[s])
                      if PROF and tx == 0 and bid < HV and 1 == 0:
                          Wbuf[0, bid, 8] = tpacc[0]
                          Wbuf[0, bid, 9] = tpacc[1]
                else:
                    if tx >= 0 and tx < 32:        # ---- TMA warp 0: the stages of consumer group 0 (TMAPG) / all stages in fill order ----
                      tp = T.alloc_local((2,), "int64")
                      tpacc = T.alloc_local((2,), F32)
                      tpacc[0] = 0.0
                      tpacc[1] = 0.0
                      for ig in T.serial(T.ceildiv(nit - 0, NCG) if TMAPG else nit):
                          it = (0 + ig * NCG) if TMAPG else ig
                          p = bid + it * SMS
                          gc = 0 if TMAPG else it % NCG
                          ig2 = ig if TMAPG else it // NCG
                          s = gc * SPG + ig2 % SPG
                          sq = gc * SPGQ + ig2 % SPGQ
                          n = p // HV
                          hv = p % HV
                          hk = hv // GQA
                          slot = slot_s[it]
                          if PROF:
                              tp[0] = T.call_extern("int64", "clock64")
                          T.mbarrier_wait_parity(consumedQ[sq], ((ig2 // SPGQ) & 1) ^ 1)
                          if slot > 0:
                              T.tma_copy(Sq[slot, hv, :, :], Qs[sq, :, :], barrier=loadedQ[sq])
                          T.mbarrier_arrive(loadedQ[sq])
                          T.mbarrier_wait_parity(consumed[s], ((ig2 // SPG) & 1) ^ 1)
                          if PROF:
                              tpacc[0] += T.cast(T.call_extern("int64", "clock64") - tp[0], F32)
                              tpacc[1] += 1.0
                          if slot > 0:
                              T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=loaded[s])
                              T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                              T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=loaded[s])
                              T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Gbuf[slot, hv, :, :], Gb_s[s, :, :], barrier=loaded[s])
                              T.tma_copy(Pbuf[slot, hv, :], pb_s[s, :], barrier=loaded[s])
                              T.tma_copy(a[n, hv, :], ax_s[s, :], barrier=loaded[s])
                              T.tma_copy(dt_bias[hv, :], dtb_s[s, :], barrier=loaded[s])
                          T.mbarrier_arrive(loaded[s])
                      if PROF and tx == 0 and bid < HV and 0 == 0:
                          Wbuf[0, bid, 8] = tpacc[0]
                          Wbuf[0, bid, 9] = tpacc[1]
                for nw in range(NNW):
                  if tx >= 32 * (NTW + nw) and tx < 32 * (NTW + 1 + nw):       # ---- norm warp nw: programs it = nw, nw + NNW, ... ----
                    lane = tx - 32 * (NTW + nw)
                    ql = T.alloc_local((4,), F32)
                    kl = T.alloc_local((4,), F32)
                    kfl = T.alloc_local((4,), F32)
                    qfl = T.alloc_local((4,), F32)
                    red = T.alloc_local((4,), F32)
                    tn = T.alloc_local((2,), "int64")
                    tnacc = T.alloc_local((2,), F32)
                    dn_l = T.alloc_local((1,), F32)
                    tnacc[0] = 0.0
                    tnacc[1] = 0.0
                    if nw == 0 and MMADOT:
                        for i in T.serial(NSTAGE * 4 * K // 32):        # rows 4..7 of every stage's B operand are zero
                            e = lane + 32 * i
                            bn_s[e // (4 * K), 4 + (e % (4 * K)) // K, e % K] = T.cast(0.0, BF16)
                    if nw == 0:
                        for i in T.serial(NSTAGE * 4 * K // 32):          # zero rows 4..7 of every stage's B operand once
                            e = lane + 32 * i
                            bf_s[e // (4 * K), 4 + (e % (4 * K)) // K, e % K] = T.cast(0.0, F16)
                    for itn in T.serial(T.ceildiv(nit - nw, NNW)):
                        it = nw + itn * NNW
                        p = bid + it * SMS
                        gc = it % NCG
                        ig = it // NCG
                        s = gc * SPG + ig % SPG
                        hv = p % HV
                        slot = slot_s[it]
                        if PROF:
                            tn[0] = T.call_extern("int64", "clock64")
                        T.mbarrier_wait_parity(loaded[s], (ig // SPG) & 1)
                        if PROF:
                            tn[1] = T.call_extern("int64", "clock64")
                            tnacc[0] += T.cast(tn[1] - tn[0], F32)
                        if slot > 0:
                            red[0] = 0.0
                            red[1] = 0.0
                            red[2] = 0.0
                            for i in T.unroll(4):
                                ql[i] = T.cast(qx_s[s, lane + 32 * i], F32)
                                kl[i] = T.cast(kx_s[s, lane + 32 * i], F32)
                                red[0] += ql[i] * ql[i]
                                red[1] += kl[i] * kl[i]
                                red[2] += kl[i] * ql[i]
                            nq = T.warp_reduce_sum(red[0])
                            nk = T.warp_reduce_sum(red[1])
                            nkq = T.warp_reduce_sum(red[2])
                            inq = T.rsqrt(nq + 1e-6) * scale
                            ink = T.rsqrt(nk + 1e-6)
                            red[3] = 0.0
                            ea = alog_s[hv]
                            for i in T.unroll(4):
                                c = lane + 32 * i
                                kn = kl[i] * ink
                                qn = ql[i] * inq
                                xg = T.cast(ax_s[s, c], F32) + dtb_s[s, c]
                                spg = KGATE_SP(xg, ea)
                                gn = KGATE_GN(pb_s[s, c], ea, spg)
                                pc = FEXP(gn)
                                # exp(gn - pb) algebraically, NOT as a difference: pb reaches -1365 in real models, and
                                # subtracting two numbers of that size to recover a ~0.3 exponent loses most of it
                                en = KGATE_EN(ea, spg)
                                en_s[s, c] = en
                                Pbuf[slot, hv, c] = gn                                    # the norm warp owns the channel vectors: appends go here
                                Gbuf[slot, hv, h_s[it], c] = T.cast(en, GBT)
                                kn_s[s, (c // 16) * 20 + c % 16] = kn
                                qn_s[s, (c // 16) * 20 + c % 16] = qn
                                if MMADOT:                     # bf16 hi + lo keeps ~16 mantissa bits of kn / qn
                                    knh = T.cast(kn, BF16)
                                    qnh = T.cast(qn, BF16)
                                    bn_s[s, 0, c] = knh
                                    bn_s[s, 1, c] = T.cast(kn - T.cast(knh, F32), BF16)
                                    bn_s[s, 2, c] = qnh
                                    bn_s[s, 3, c] = T.cast(qn - T.cast(qnh, F32), BF16)
                                pc_s[s, c] = pc                                           # (scaling U^T in place in fp16 would cost 2^-11 on the rank-R term)
                                kfl[i] = kn * skv_s[s, c] * (1.0 / QMAX) * pc
                                qfl[i] = qn * skv_s[s, c] * (1.0 / QMAX) * pc
                                red[3] = T.max(red[3], T.max(T.abs(kfl[i]), T.abs(qfl[i])))
                            kfmax = T.max(T.warp_reduce_max(red[3]), 1e-30)
                            rk = T.call_extern(F32, "__frcp_rn", kfmax)
                            for i in T.unroll(4):
                                c = lane + 32 * i
                                khi = T.cast(kfl[i] * rk, F16)
                                qhi = T.cast(qfl[i] * rk, F16)
                                bf_s[s, 0, c] = khi
                                bf_s[s, 1, c] = qhi
                                bf_s[s, 2, c] = T.cast(kfl[i] * rk - T.cast(khi, F32), F16)
                                bf_s[s, 3, c] = T.cast(qfl[i] * rk - T.cast(qhi, F32), F16)
                            if DNORM:                      # D_j[c] = en[c] * prod_{i>j} e_i[c], then k_j[c] * D_j[c]
                                for i in T.unroll(4):
                                    c = lane + 32 * i
                                    dn_l[0] = en_s[s, c]
                                    for jj in T.unroll(L):
                                        j_ = L - 1 - jj
                                        Kb_s[s, j_, c] = T.cast(
                                            T.cast(Kb_s[s, j_, c], F32) * T.if_then_else(j_ < h_s[it], dn_l[0], 0.0), BF16)
                                        dn_l[0] = dn_l[0] * T.cast(Gb_s[s, j_, c], F32)
                            if lane < L:
                                w_s[s, lane] = Wb_s[s, lane]
                            if lane == 0:
                                sc_s[s, 0] = kfmax
                                sc_s[s, 1] = nkq * inq * ink
                                sc_s[s, 2] = 0.0
                                sc_s[s, 3] = b_s[it]
                        if PROF:
                            tnacc[1] += T.cast(T.call_extern("int64", "clock64") - tn[1], F32)
                        T.mbarrier_arrive(normed[s])
                    if PROF and lane == 0 and bid < HV and nw == 0:
                        Wbuf[0, bid, 6] = tnacc[0]
                        Wbuf[0, bid, 7] = tnacc[1]
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                warp = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                bar = 8 + g
                afrag = T.alloc_local((4,), I32)             # A fragment (4 x fp16x2)
                bh = T.alloc_local((4,), F16)                # B fragment (2 x fp16x2)
                cfrag = T.alloc_local((8,), F32)             # 2 m-tiles x 4
                acc = T.alloc_local((8,), F32)
                knl = T.alloc_local((16 if VECD else 1,), F32)      # batched dot operands
                qnl = T.alloc_local((16 if VECD else 1,), F32)
                kbl = T.alloc_local((16 if VECD else 1,), BF16)
                klol = T.alloc_local((16 if (VECD and DLO) else 1,), BF16)
                afr2 = T.alloc_local((8 if MMADOT else 1,), BF16)   # ring-dot A fragment (ldmatrix.x4)
                bfr2 = T.alloc_local((4 if MMADOT else 1,), BF16)   # ring-dot B fragment
                dfrag = T.alloc_local((4 if MMADOT else 1,), F32)
                tk = T.alloc_local((8,), "int64")
                tacc = T.alloc_local((8,), F32)
                for i in T.unroll(8):
                    tacc[i] = 0.0
                ncons = T.ceildiv(nit - g, NCG)
                for it0 in T.serial(ncons):
                    it = g + it0 * NCG
                    p = bid + it * SMS
                    s = g * SPG + it0 % SPG
                    sq = g * SPGQ + it0 % SPGQ
                    pb = it0 % 2
                    n = p // HV
                    hv = p % HV
                    if PROF:
                        tk[0] = T.call_extern("int64", "clock64")
                    T.mbarrier_wait_parity(loaded[s], (it0 // SPG) & 1)
                    T.mbarrier_wait_parity(loadedQ[sq], (it0 // SPGQ) & 1)
                    T.mbarrier_wait_parity(normed[s], (it0 // SPG) & 1)
                    if PROF:
                        tk[1] = T.call_extern("int64", "clock64")
                    slot = slot_s[it]
                    h = h_s[it]
                    if slot <= 0:
                        o[n, hv, t] = T.cast(0.0, BF16)
                        T.mbarrier_arrive(consumedQ[sq])
                    else:
                        v0 = T.cast(vx_s[s, t], F32)
                        kfmax = sc_s[s, 0]
                        kq = sc_s[s, 1]
                        beta = sc_s[s, 3]
                        kn = kn_s[s, (t // 16) * 20 + t % 16]
                        # ---- per-entry decay D_j[c] = en[c] * prod_{i>j, i<h} e_i[c] (thread per column c = t), backward product ----
                        # (entries j >= h hold e_j = 1: the flush resets Gbuf to 1, so the product needs no predicate)
                        dcur = T.alloc_local((1,), F32)
                        kdl = T.alloc_local((1,), F32)
                        kdh = T.alloc_local((1,), BF16)
                        e_l = T.alloc_local((L if not DNORM else 1,), F32)
                        kb_l = T.alloc_local((L if (DFOLD and not DNORM) else 1,), F32)
                        if not DNORM:
                          for jj in T.unroll(L):                           # all factor loads first (smem loads cannot be hoisted above the stores)
                            e_l[jj] = T.cast(Gb_s[s, jj, t], F32)
                        if DFOLD and not DNORM:
                            for jj in T.unroll(L):
                                kb_l[jj] = T.cast(Kb_s[s, jj, t], F32)
                        dcur[0] = en_s[s, t]
                        for jj in T.unroll(0 if DNORM else L):
                            if DFOLD:                                      # k_j[c] * D_j[c], zeroed past hcnt so the dot needs no predicate
                                if DLO:
                                    kdl[0] = kb_l[L - 1 - jj] * T.if_then_else(L - 1 - jj < h, dcur[0], 0.0)
                                    kdh[0] = T.cast(kdl[0], BF16)
                                    Kb_s[s, L - 1 - jj, t] = kdh[0]
                                    klo_s[g, pb, L - 1 - jj, t] = T.cast(kdl[0] - T.cast(kdh[0], F32), BF16)
                                else:
                                    Kb_s[s, L - 1 - jj, t] = T.cast(
                                        kb_l[L - 1 - jj] * T.if_then_else(L - 1 - jj < h, dcur[0], 0.0), BF16)
                            else:
                                D_s[g, pb, L - 1 - jj, (t // 16) * 20 + t % 16] = T.cast(dcur[0], F16)
                            dcur[0] = dcur[0] * e_l[L - 1 - jj]
                        if PROF:
                            tk[2] = T.call_extern("int64", "clock64")
                        # ---- checkpoint stream on tensor cores: C[v, 0:2] = sum_c q8[v, c] [kf qf][c] / kfmax ----
                        for i in T.unroll(8):
                            cfrag[i] = 0.0
                        for kk in T.unroll(KW // 4):                       # 16-column k-steps
                            for i in T.vectorized(4):
                                bh[i] = bf_s[s, r0, kk * 16 + q4 * 4 + i]
                            for mt in T.unroll(2):
                                rr = warp * 32 + mt * 16 + r0
                                wa = Qs[sq, rr, kk * 4 + q4] ^ MAGICX
                                wb = Qs[sq, rr + 8, kk * 4 + q4] ^ MAGICX
                                afrag[0] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[1] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[2] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[3] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", afrag.data, 0, bh.data, 0, cfrag.data, mt * 4, T.bool(False))
                        if q4 < 2:                                         # q4 = 0: columns n = 0, 1 (hi parts); q4 = 1: n = 2, 3 (lo parts)
                            for mt in T.unroll(2):
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4] = cfrag[mt * 4]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4 + 1] = cfrag[mt * 4 + 1]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4] = cfrag[mt * 4 + 2]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = cfrag[mt * 4 + 3]
                        T.fence_proxy_async()                              # the int8 tile is dead for this program: release its ring slot early
                        T.mbarrier_arrive(consumedQ[sq])
                        if PROF:
                            tk[3] = T.call_extern("int64", "clock64")
                        T.sync_threads(bar, 128)                           # D_s (and nothing else) must be complete before the dots
                        # ---- buffer dots: dk[j] = K[j].kn, dq[j] = K[j].qn (8 lanes per row j) ----
                        jr = t // 8
                        part = t % 8
                        acc[2] = 0.0
                        acc[3] = 0.0
                        if MMADOT:
                            # A = Kb_s (already k_j * D_j, zeroed past hcnt by DFOLD) as one m16 tile;
                            # B = [kn_hi, kn_lo, qn_hi, qn_lo].  One warp covers all L entries.
                            if warp == 0:
                                for i in T.unroll(4):
                                    dfrag[i] = 0.0
                                for kk in T.unroll(K // 16):
                                    T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(Kb_s[s, lane % 16, kk * 16 + 8 * (lane // 16)], "r", extent=8), T.access_ptr(afr2[0], "w", extent=8))
                                    T.ptx_ldmatrix(T.bool(False), 2, T.access_ptr(bn_s[s, (lane % 8), kk * 16 + 8 * (lane // 8 % 2)], "r", extent=4), T.access_ptr(bfr2[0], "w", extent=4))
                                    T.ptx_mma("float32", "m16n8k16", "row", "col", "bf16", "bf16", "fp32", afr2.data, 0, bfr2.data, 0, dfrag.data, 0, T.bool(False))
                                if lane % 4 == 0:
                                    dk_s[g, pb, lane // 4] = (dfrag[0] + dfrag[1]) * w_s[s, lane // 4]
                                    dk_s[g, pb, lane // 4 + 8] = (dfrag[2] + dfrag[3]) * w_s[s, lane // 4 + 8]
                                if lane % 4 == 1:
                                    dq_s[g, pb, lane // 4] = (dfrag[0] + dfrag[1]) * w_s[s, lane // 4]
                                    dq_s[g, pb, lane // 4 + 8] = (dfrag[2] + dfrag[3]) * w_s[s, lane // 4 + 8]
                        elif VECD and DFOLD:
                            # the dot is 48 scalar smem loads per thread on an issue-bound kernel; the chunk
                            # padding (part * 20) makes both 16-float runs 16 B aligned, so they are 4 vector
                            # loads each, and the 16 ring values are two 16 B loads
                            for v4 in T.unroll(4):
                                for c1 in T.vectorized(4):
                                    knl[v4 * 4 + c1] = kn_s[s, part * 20 + v4 * 4 + c1]
                            for v4 in T.unroll(4):
                                for c1 in T.vectorized(4):
                                    qnl[v4 * 4 + c1] = qn_s[s, part * 20 + v4 * 4 + c1]
                            for v8 in T.unroll(2):
                                for c1 in T.vectorized(8):
                                    kbl[v8 * 8 + c1] = Kb_s[s, jr, part * 16 + v8 * 8 + c1]
                            if DLO:
                                for v8 in T.unroll(2):
                                    for c1 in T.vectorized(8):
                                        klol[v8 * 8 + c1] = klo_s[g, pb, jr, part * 16 + v8 * 8 + c1]
                            for c1 in T.unroll(16):
                                kb = T.cast(kbl[c1], F32) + (T.cast(klol[c1], F32) if DLO else 0.0)
                                acc[2] += kb * knl[c1]
                                acc[3] += kb * qnl[c1]
                        else:
                            for c1 in T.unroll(16):
                                if DFOLD or (DBG & 1):                                     # decay already in Kb_s (or ablated away)
                                    kb = T.cast(Kb_s[s, jr, part * 16 + c1], F32) + (T.cast(klo_s[g, pb, jr, part * 16 + c1], F32) if (DLO and DFOLD) else 0.0)
                                else:
                                    kb = T.cast(Kb_s[s, jr, part * 16 + c1], F32) * T.if_then_else(jr < h, T.cast(D_s[g, pb, jr, part * 20 + c1], F32), 0.0)
                                acc[2] += kb * kn_s[s, part * 20 + c1]
                                acc[3] += kb * qn_s[s, part * 20 + c1]
                        if not MMADOT:
                          acc[2] += T.shfl_xor(acc[2], 1)
                          acc[3] += T.shfl_xor(acc[3], 1)
                          acc[2] += T.shfl_xor(acc[2], 2)
                          acc[3] += T.shfl_xor(acc[3], 2)
                          acc[2] += T.shfl_xor(acc[2], 4)
                          acc[3] += T.shfl_xor(acc[3], 4)
                          if part == 0:
                              # scale by w_j HERE, once: the contribution loop below is thread-per-row, so
                              # leaving it there costs 128 x 16 x 2 redundant multiplies per program.
                              dk_s[g, pb, jr] = acc[2] * (w_s[s, jr] if WFOLD else 1.0)
                              dq_s[g, pb, jr] = acc[3] * (w_s[s, jr] if WFOLD else 1.0)
                        if warp < R:                                       # factor dots: dU[r] = U^T[r].kn, U^T[r].qn (one warp per r)
                            acc[6] = 0.0
                            acc[7] = 0.0
                            for i in T.unroll(4):
                                uf = T.cast(UQ_s[s, warp, lane + 32 * i], F32) * pc_s[s, lane + 32 * i]
                                acc[6] += uf * kn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                                acc[7] += uf * qn_s[s, ((lane + 32 * i) // 16) * 20 + lane % 16]
                            acc[6] = T.warp_reduce_sum(acc[6])
                            acc[7] = T.warp_reduce_sum(acc[7])
                            if lane == 0:
                                dk_s[g, pb, L + warp] = acc[6]
                                dq_s[g, pb, L + warp] = acc[7]
                        T.sync_threads(bar, 128)
                        if PROF:
                            tk[4] = T.call_extern("int64", "clock64")
                        # ---- contribution, correction, output, append (thread-per-row v = t) ----
                        sv = skv_s[s, K + t]
                        mq = (mqy_s[g, pb, t, 0] + mqy_s[g, pb, t, 2]) * kfmax * sv
                        yq = (mqy_s[g, pb, t, 1] + mqy_s[g, pb, t, 3]) * kfmax * sv
                        acc[4] = 0.0
                        acc[5] = 0.0
                        for j in T.unroll(L):
                            ub = T.cast(Ub_s[s, j, t], F32)                 # dk_s / dq_s already carry w_j
                            acc[4] += ub * dk_s[g, pb, j] * (1.0 if WFOLD else w_s[s, j])
                            acc[5] += ub * dq_s[g, pb, j] * (1.0 if WFOLD else w_s[s, j])
                        acc[6] = 0.0                                        # rank-R part of the checkpoint: Q^T[r][v] (U^T[r].kn)
                        acc[7] = 0.0
                        for r in T.unroll(R):
                            qf = T.cast(UQ_s[s, r, K + t], F32)
                            acc[6] += qf * dk_s[g, pb, L + r]
                            acc[7] += qf * dq_s[g, pb, L + r]
                        u = beta * (v0 - (mq + acc[6] + acc[4]))
                        y = yq + acc[7] + acc[5] + kq * u
                        o[n, hv, t] = T.cast(y, BF16)
                        Ubuf[slot, hv, h, t] = T.cast(u, BF16)
                        Kbuf[slot, hv, h, t] = T.cast(kn, BF16)
                        if t == 0:
                            Wbuf[slot, hv, h] = 1.0
                            hcnt[slot, hv] = h + 1
                        if PROF:
                            tk[5] = T.call_extern("int64", "clock64")
                            for i in T.unroll(5):
                                tacc[i] += T.cast(tk[i + 1] - tk[i], F32)
                            tacc[5] += 1.0
                    T.fence_proxy_async()
                    T.mbarrier_arrive(consumed[s])
                if PROF and g == 0 and t == 0 and bid < HV:
                    for i in T.unroll(6):
                        Wbuf[0, bid, i] = tacc[i]
    return step


class Pool:
    """Caller-owned state pool, the layout the serving engine uses (the same as the KDA flush request).  Every (slot, head)
    owns a V x K fp32-sized region (64 KB) of one flat buffer; the checkpoint lives in its first 19 KB:
        bytes [0, 16384)            int8 codes, row-major [V, K]
        bytes [16384, 16896)        fp32 column scales s_k [K]
        bytes [16896, 17408)        fp32 row scales    s_v [V]
        bytes [17408, 19456)        fp16 Compensator Tokens [R, K + V]: row r = (u[r, :K] | q[r, :V])
    Slots are SS fp32 words apart; slot 0 is reserved (idx 0 = "no sequence").  The buffered updates (kbuf, ubuf, gbuf, w), the
    cumulative log-gate pcum and the fill counter hcnt are separate dense tensors; hcnt == L means "just flushed, empty"."""

    def __init__(self, num_slots, device="cuda"):
        NS = self.NS = num_slots
        self.raw = torch.zeros(NS * SS + OFF, dtype=torch.float32, device=device); st = self.raw.untyped_storage()
        def view(dtype, shape, strides, off):
            return torch.empty(0, dtype=dtype, device=device).set_(st).as_strided(shape, strides, off)
        self.codes = view(torch.int8, (NS, HV, V, K), (4 * SS, 4 * V * K, K, 1), 4 * OFF)
        self.codes32 = view(torch.int32, (NS, HV, V, K // 4), (SS, V * K, K // 4, 1), OFF)
        self.s_k = view(torch.float32, (NS, HV, K), (SS, V * K, 1), OFF + V * K // 4)
        self.s_v = view(torch.float32, (NS, HV, V), (SS, V * K, 1), OFF + V * K // 4 + K)
        self.s_kv = view(torch.float32, (NS, HV, K + V), (SS, V * K, 1), OFF + V * K // 4)
        self.uq = view(torch.float16, (NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), 2 * (OFF + V * K // 4 + K + V))
        self.kbuf = torch.zeros(NS, HV, L, K, dtype=torch.bfloat16, device=device)
        self.ubuf = torch.zeros(NS, HV, L, V, dtype=torch.bfloat16, device=device)
        self.gbuf = torch.ones(NS, HV, L, K, dtype=torch.float16, device=device)
        self.w = torch.zeros(NS, HV, L, device=device)
        self.pcum = torch.zeros(NS, HV, K, device=device)
        self.hcnt = torch.zeros(NS, HV, dtype=torch.int32, device=device)
        self.kernel = make_step_kernel(NS, HV, HK, K, V, L, SS, R=R, NSTAGE=6, NSTAGEQ=4, NCG=2, NNW=3, GB16=True,
                                       SMS=torch.cuda.get_device_properties(self.raw.device).multi_processor_count)


def step_inplace(pool, idx, mixed, a, b, A_log, dt_bias, o):
    """Deployment form, the engine's call.  idx int32 [N]: slot per sequence (0 = skip, output zeros); mixed bf16
    [N, 2 HK K + HV V] = (q | k | v) as the projection produces it; a bf16 [N, HV, K] (per-channel gate input), b bf16 [N, HV];
    A_log f32 [HV], dt_bias f32 [HV, K]; o bf16 [N, HV, V] (written).  Appends the new update, its decay factor and weight at
    entry hcnt (hcnt == L counts as 0), stores the new cumulative log-gate, increments hcnt."""
    pool.kernel(pool.codes32, pool.s_k, pool.s_v, pool.s_kv, pool.kbuf, pool.ubuf, pool.w, pool.pcum, pool.gbuf, pool.hcnt,
                mixed, a, b, A_log, dt_bias, pool.uq, idx, o)


_POOLS = {}


def run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias):
    B = codes.shape[0]
    dev = codes.device
    pool = _POOLS.get((B, dev))
    if pool is None:
        pool = _POOLS[(B, dev)] = Pool(B + 8, dev)
    sl = slice(1, B + 1)
    pool.codes[sl] = codes; pool.s_k[sl] = s_k; pool.s_v[sl] = s_v; pool.uq[sl, :, :, :K] = u; pool.uq[sl, :, :, K:] = q
    pool.kbuf[sl] = kbuf; pool.ubuf[sl] = ubuf; pool.gbuf[sl] = gbuf; pool.w[sl] = w; pool.pcum[sl] = pcum; pool.hcnt[sl] = h
    mixed = torch.cat([qx.reshape(B, -1), kx.reshape(B, -1), vx.reshape(B, -1)], 1).contiguous()
    o = torch.empty(B, HV, V, dtype=torch.bfloat16, device=dev)
    step_inplace(pool, torch.arange(1, B + 1, dtype=torch.int32, device=dev), mixed, a.contiguous(), b.contiguous(),
                 A_log.contiguous(), dt_bias.contiguous(), o)
    hh = h.long(); ar = torch.arange(B, device=dev)[:, None]; hv = torch.arange(HV, device=dev)[None, :]
    return (o, pool.kbuf[sl][ar, hv, hh].clone(), pool.ubuf[sl][ar, hv, hh].clone(), pool.gbuf[sl][ar, hv, hh].clone(),
            pool.w[sl].clone(), pool.pcum[sl].clone())
