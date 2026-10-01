# SPDX-License-Identifier: Apache-2.0
"""Best-known LeapQuant decode-step kernel (gated delta rule with a quantized recurrent state; see README.md).

TileLang kernel for sm_100 (B200): one persistent CTA per SM, a TMA producer warp, three consumer groups of 4 warps with
one norm warp each; the int8 checkpoint tile goes through the tensor cores (int8 -> fp16 by PRMT, exact; mma.sync
m16n8k16 against an fp16 hi/lo split of the scaled key / query), everything else in fp32 on CUDA cores.

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
HV, HK, K, V, L, R = 32, 16, 128, 128, 16, 4  # value heads, key heads (GQA), key dim, value dim, window, Compensator Tokens
SS, OFF = 33 * V * K, 12288                   # pool geometry in fp32 words: slot stride (one spare head per slot), base offset
_GTIMER_SRC = "__device__ __forceinline__ long long tl_gtimer() { long long t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(t)); return t; }\n"


@tilelang.jit(pass_configs={"tl.disable_thread_storage_sync": True})
def make_step_kernel(NS, HV, H, K, V, L, SS, R=4, SMS=148, NSTAGE=6, NCG=3, scale=128 ** -0.5, MAXIT=256, PROF=False, WEXACT=False, DBG=0, STAMP=False, **_ignored):
    # STAMP: per CTA, cycles after its entry into Sq[0] (reserved slot) word bid * 16 + k: k 0 TMA loop done, 1..NNW norm warps done,
    #        4 + g consumer group g first data, 8 + g consumer group g done, 12 CTA entry clock low 31 bits
    """TileLang generator of the decode-step kernel (see the module docstring).
      warp 0           : TMA; the norm inputs (scales, q, k, weights, buffered keys, Compensator Tokens) on sloaded,
                         the int8 tile / buffered values / v on loaded
      warps 1..NCG     : norm warp g serves consumer group g: l2 norms, the fp16 mma B operand, the 2 (L + R) dot products of
                         kn / qn with the buffered keys and the Compensator Tokens (lane = 4 contiguous columns, reduce-scatter /
                         butterfly), the decayed weights
      consumers        : thread t = row v; tensor-core stream over the warp's 32 rows of the int8 tile, contribution of the
                         buffered updates and the Compensator Tokens, output, release the stage, append the new update.
    DBG / STAMP / PROF are ablation and instrumentation switches (off by default)."""
    GQA = HV // H
    N = T.symbolic("N")
    KW = K // 4
    NCONS = 128 * NCG
    assert NSTAGE % NCG == 0
    SPG = NSTAGE // NCG
    NNW = NCG
    NPROD = 32 * (1 + NNW)
    MJ = (MAXIT + 31) // 32                        # prologue loads per TMA-warp lane
    MJN = (MAXIT + 32 * NNW - 1) // (32 * NNW)     # per norm-warp lane
    MIX = 2 * H * K + HV * V
    F16 = "float16"
    assert K == 128 and V == 128 and L == 16 and R <= 4
    MAGICX = T.int32(-2139062144)                  # 0x80808080: int8 -> offset byte
    H2MAG = T.int32(0x64646464)                    # fp16 exponent byte: 0x64xx = 1024 + xx

    @T.prim_func
    def step(Sq: T.StridedTensor((NS, HV, V, KW), (SS, V * K, KW, 1), I32),
             Sk: T.StridedTensor((NS, HV, K), (SS, V * K, 1), F32),
             Sv: T.StridedTensor((NS, HV, V), (SS, V * K, 1), F32),
             Skv: T.StridedTensor((NS, HV, K + V), (SS, V * K, 1), F32),
             Kbuf: T.Tensor((NS, HV, L, K), BF16), Ubuf: T.Tensor((NS, HV, L, V), BF16),
             Wbuf: T.Tensor((NS, HV, L), F32), Pbuf: T.Tensor((NS, HV, 1), F32), hcnt: T.Tensor((NS, HV), I32),
             mixed: T.Tensor((N, MIX), BF16), a: T.Tensor((N, HV, 1), BF16), b: T.Tensor((N, HV), BF16),
             A_log: T.Tensor((HV,), F32), dt_bias: T.Tensor((HV, 1), F32),
             UQ: T.StridedTensor((NS, HV, R, K + V), (2 * SS, 2 * V * K, K + V, 1), F16),
             idx: T.Tensor((N,), I32), o: T.Tensor((N, HV, V), BF16)):
        with T.Kernel(SMS, threads=NPROD + NCONS) as bid:
            Qs = T.alloc_shared((NSTAGE, V, KW), I32)
            UQ_s = T.alloc_shared((NSTAGE, R, K + V), F16)
            skv_s = T.alloc_shared((NSTAGE, K + V), F32)
            qx_s = T.alloc_shared((NSTAGE, K), BF16)
            kx_s = T.alloc_shared((NSTAGE, K), BF16)
            vx_s = T.alloc_shared((NSTAGE, V), BF16)
            Kb_s = T.alloc_shared((NSTAGE, L, K), BF16)
            Ub_s = T.alloc_shared((NSTAGE, L, V), BF16)
            Wb_s = T.alloc_shared((NSTAGE, 32), F32)
            # ---- per-stage norm outputs ----
            bf_s = T.alloc_shared((NSTAGE, 8, K + 16), F16)   # mma B operand: rows 0/1 = hi(kf, qf)/kfmax, 2/3 = lo parts, 4..7 = 0
            kn_s = T.alloc_shared((NSTAGE, K), F32)
            dd_s = T.alloc_shared((NSTAGE, 2 * L + 2 * R), F32)   # [w_j dk_j (L) | w_j dq_j (L) | dU_r . kn (R) | dU_r . qn (R)]
            sc_s = T.alloc_shared((NSTAGE, 8), F32)           # 0 kfmax, 1 kq, 2 pn, 3 beta
            mqy_s = T.alloc_shared((NCG, 2, V, 4), F32)       # per warp rows; parity double buffer
            slot_s = T.alloc_shared((MAXIT,), I32)
            h_s = T.alloc_shared((MAXIT,), I32)
            P_s = T.alloc_shared((MAXIT,), F32)
            a_s = T.alloc_shared((MAXIT,), F32)
            b_s = T.alloc_shared((MAXIT,), F32)
            loaded = T.alloc_barrier([32] * NSTAGE)
            sloaded = T.alloc_barrier([32] * NSTAGE)
            normed = T.alloc_barrier([32] * NSTAGE)          # dots, weights, pn / beta ready
            normb = T.alloc_barrier([32] * NSTAGE)           # mma B operand, kn, kfmax / kq ready (the consumers' mma can start)
            consumed = T.alloc_barrier([128] * NSTAGE)
            T.annotate_layout({Qs: make_swizzled_layout(Qs)})
            if STAMP:
                T.import_source(_GTIMER_SRC)
            tx = T.get_thread_binding()
            t_entry = T.alloc_local((1,), "int64")
            t_entry[0] = T.call_extern("int64", "clock64")
            if STAMP:
                if tx == 0:
                    gt = T.call_extern("int64", "tl_gtimer")
                    Sq[0, (bid * 32 + 12) // (V * KW), ((bid * 32 + 12) % (V * KW)) // KW, (bid * 32 + 12) % KW] = T.cast(gt & T.int64(0x7fffffff), I32)
            nprog = N * HV
            nit = T.ceildiv(nprog - bid, SMS)
            if tx < NPROD:
                pro_i = T.alloc_local((MJ,), I32)
                pro_s = T.alloc_local((MJN,), I32)
                pro_h = T.alloc_local((MJN,), I32)
                pro_p = T.alloc_local((MJN,), F32)
                pro_a = T.alloc_local((MJN,), F32)
                pro_b = T.alloc_local((MJN,), F32)
                pro_g = T.alloc_local((MJN,), F32)
                if tx < 32:
                    # ================= TMA warp =================
                    tq = T.alloc_local((2,), "int64")
                    tq[1] = T.int64(0)
                    for j in T.unroll(MJ):                          # all loads in flight at once, then the stores
                        it = tx + 32 * j
                        pro_i[j] = T.if_then_else(it < nit, idx[T.min((bid + it * SMS) // HV, N - 1)], 0)
                    for j in T.unroll(MJ):
                        if tx + 32 * j < nit:
                            slot_s[tx + 32 * j] = pro_i[j]
                    T.sync_warp()
                    if STAMP:
                        if tx == 0:
                            Sq[0, (bid * 32 + 19) // (V * KW), ((bid * 32 + 19) % (V * KW)) // KW, (bid * 32 + 19) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                    for it in T.serial(nit):
                        p = bid + it * SMS
                        gc = it % NCG
                        ig = it // NCG
                        s = gc * SPG + ig % SPG
                        n = p // HV
                        hv = p % HV
                        hk = hv // GQA
                        slot = slot_s[it]
                        T.mbarrier_wait_parity(consumed[s], ((ig // SPG) & 1) ^ 1)
                        if STAMP:
                            tq[0] = T.call_extern("int64", "clock64")
                        if slot > 0:
                            T.tma_copy(Skv[slot, hv, :], skv_s[s, :], barrier=sloaded[s])
                            T.tma_copy(mixed[n, hk * K:(hk + 1) * K], qx_s[s, :], barrier=sloaded[s])
                            T.tma_copy(mixed[n, H * K + hk * K:H * K + (hk + 1) * K], kx_s[s, :], barrier=sloaded[s])
                            T.tma_copy(Wbuf[slot, hv, :], Wb_s[s, 0:L], barrier=sloaded[s])
                            T.tma_copy(Kbuf[slot, hv, :, :], Kb_s[s, :, :], barrier=sloaded[s])
                            T.tma_copy(UQ[slot, hv, :, :], UQ_s[s, :, :], barrier=sloaded[s])
                        T.mbarrier_arrive(sloaded[s])
                        if slot > 0:
                            T.tma_copy(Sq[slot, hv, :, :], Qs[s, :, :], barrier=loaded[s])
                            T.tma_copy(Ubuf[slot, hv, :, :], Ub_s[s, :, :], barrier=loaded[s])
                            T.tma_copy(mixed[n, 2 * H * K + hv * V:2 * H * K + (hv + 1) * V], vx_s[s, :], barrier=loaded[s])
                        T.mbarrier_arrive(loaded[s])
                        if STAMP:
                            tq[1] = tq[1] + (T.call_extern("int64", "clock64") - tq[0])
                        if STAMP:
                            if tx == 0 and it == 0:
                                Sq[0, (bid * 32 + 16) // (V * KW), ((bid * 32 + 16) % (V * KW)) // KW, (bid * 32 + 16) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                    if STAMP:
                        if tx == 0:
                            Sq[0, (bid * 32 + 20) // (V * KW), ((bid * 32 + 20) % (V * KW)) // KW, (bid * 32 + 20) % KW] = T.cast(tq[1], I32)
                    if STAMP:
                        if tx == 0:
                            Sq[0, (bid * 32 + 0) // (V * KW), ((bid * 32 + 0) % (V * KW)) // KW, (bid * 32 + 0) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                else:
                    # ================= norm warps: per-program scalars, then each serves its group =================
                    for j in T.unroll(MJN):                         # round 1: idx, a, b for all of this thread's programs
                        it = (tx - 32) + 32 * NNW * j
                        p = bid + T.min(it, nit - 1) * SMS
                        n = p // HV
                        hv = p % HV
                        pro_s[j] = idx[n]
                        pro_a[j] = T.cast(a[n, hv, 0], F32) + dt_bias[hv, 0]
                        pro_b[j] = T.cast(b[n, hv], F32)
                        pro_g[j] = A_log[hv]
                    for j in T.unroll(MJN):                         # round 2: the loads that need the slot
                        it = (tx - 32) + 32 * NNW * j
                        p = bid + T.min(it, nit - 1) * SMS
                        hv = p % HV
                        pro_h[j] = hcnt[T.max(pro_s[j], 0), hv]
                        pro_p[j] = Pbuf[T.max(pro_s[j], 0), hv, 0]
                    for j in T.unroll(MJN):
                        it = (tx - 32) + 32 * NNW * j
                        if it < nit:
                            xg = pro_a[j]
                            spg = T.if_then_else(xg <= SOFTPLUS_THRESHOLD, T.log(1.0 + T.exp(xg)), xg)
                            h_s[it] = T.if_then_else(pro_h[j] >= L, 0, pro_h[j])   # == L: the flush emptied the ring (it leaves hcnt at L so its due-scan is stable across CTAs)
                            P_s[it] = pro_p[j]
                            a_s[it] = T.exp(-T.exp(pro_g[j]) * spg)
                            b_s[it] = T.sigmoid(pro_b[j]) if WEXACT else T.cast(T.cast(T.sigmoid(pro_b[j]), BF16), F32)
                    T.sync_threads(15, 32 * NNW)
                    if STAMP:
                        if tx == 32:
                            Sq[0, (bid * 32 + 17) // (V * KW), ((bid * 32 + 17) % (V * KW)) // KW, (bid * 32 + 17) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                    nw = tx // 32 - 1
                    lane = tx % 32
                    ql = T.alloc_local((4,), F32)
                    kl = T.alloc_local((4,), F32)
                    kfl = T.alloc_local((4,), F32)
                    qfl = T.alloc_local((4,), F32)
                    red = T.alloc_local((8,), F32)
                    kb = T.alloc_local((4,), BF16)
                    uf = T.alloc_local((4,), F16)
                    dv = T.alloc_local((2 * L,), F32)                   # ring-dot partials: [dk_0..dk_15 | dq_0..dq_15]
                    fv = T.alloc_local((2 * R,), F32)                   # factor-dot partials
                    for i in T.serial(SPG * 4 * K // 32):               # zero rows 4..7 of this group's B operands once
                        e = lane + 32 * i
                        bf_s[nw * SPG + e // (4 * K), 4 + (e % (4 * K)) // K, e % K] = T.cast(0.0, F16)
                    for itn in T.serial(T.ceildiv(nit - nw, NCG)):
                        it = nw + itn * NCG
                        p = bid + it * SMS
                        s = nw * SPG + itn % SPG
                        hv = p % HV
                        T.mbarrier_wait_parity(sloaded[s], (itn // SPG) & 1)
                        slot = slot_s[it]
                        if slot > 0 and DBG & 2 == 0:
                            # ---- l2 norms: lane owns columns 4 lane .. 4 lane + 3 ----
                            for i in T.unroll(8):
                                red[i] = 0.0
                            for i in T.unroll(4):
                                ql[i] = T.cast(qx_s[s, 4 * lane + i], F32)
                                kl[i] = T.cast(kx_s[s, 4 * lane + i], F32)
                                red[0] += ql[i] * ql[i]
                                red[1] += kl[i] * kl[i]
                                red[2] += kl[i] * ql[i]
                            nq = T.warp_reduce_sum(red[0])
                            nk = T.warp_reduce_sum(red[1])
                            nkq = T.warp_reduce_sum(red[2])
                            inq = T.rsqrt(nq + 1e-6) * scale
                            ink = T.rsqrt(nk + 1e-6)
                            for i in T.unroll(4):
                                kl[i] = kl[i] * ink                    # kn
                                ql[i] = ql[i] * inq                    # qn
                                kn_s[s, 4 * lane + i] = kl[i]
                                kfl[i] = kl[i] * skv_s[s, 4 * lane + i] * (1.0 / QMAX)
                                qfl[i] = ql[i] * skv_s[s, 4 * lane + i] * (1.0 / QMAX)
                                red[3] = T.max(red[3], T.max(T.abs(kfl[i]), T.abs(qfl[i])))
                            kfmax = T.max(T.warp_reduce_max(red[3]), 1e-30)
                            rk = T.call_extern(F32, "__frcp_rn", kfmax)
                            for i in T.unroll(4):
                                c = 4 * lane + i
                                khi = T.cast(kfl[i] * rk, F16)
                                qhi = T.cast(qfl[i] * rk, F16)
                                bf_s[s, 0, c] = khi
                                bf_s[s, 1, c] = qhi
                                bf_s[s, 2, c] = T.cast(kfl[i] * rk - T.cast(khi, F32), F16)
                                bf_s[s, 3, c] = T.cast(qfl[i] * rk - T.cast(qhi, F32), F16)
                            if lane == 0:
                                sc_s[s, 0] = kfmax
                                sc_s[s, 1] = nkq * inq * ink
                        T.mbarrier_arrive(normb[s])                     # the mma operands are ready; the dots follow
                        if slot > 0 and DBG & 2 == 0:
                            # ---- ring dots K_j . kn, K_j . qn: partials over this lane's 4 columns ----
                            for j in T.unroll(L):
                                for i in T.vectorized(4):
                                    kb[i] = Kb_s[s, j, 4 * lane + i]
                                dv[j] = 0.0
                                dv[L + j] = 0.0
                                for i in T.unroll(4):
                                    dv[j] += T.cast(kb[i], F32) * kl[i]
                                    dv[L + j] += T.cast(kb[i], F32) * ql[i]
                            for r in T.unroll(R):
                                for i in T.vectorized(4):
                                    uf[i] = UQ_s[s, r, 4 * lane + i]
                                fv[r] = 0.0
                                fv[R + r] = 0.0
                                for i in T.unroll(4):
                                    fv[r] += T.cast(uf[i], F32) * kl[i]
                                    fv[R + r] += T.cast(uf[i], F32) * ql[i]
                            # reduce-scatter of the 32 ring partials: afterwards lane l holds the full dot number l
                            for st in T.unroll(5):
                                m = 16 >> st
                                for k0 in T.unroll(16 >> st):
                                    lo = dv[k0]
                                    hi = dv[k0 + (16 >> st)]
                                    up = (lane & m) != 0
                                    dv[k0] = T.if_then_else(up, hi, lo) + T.shfl_xor(T.if_then_else(up, lo, hi), m)
                            for r in T.unroll(2 * R):                   # butterfly all-reduce of the factor partials
                                for st in T.unroll(5):
                                    fv[r] += T.shfl_xor(fv[r], 1 << st)
                            h = h_s[it]
                            at = a_s[it]
                            jl = lane % L                               # lanes 0..15: dk_j, 16..31: dq_j with j = lane % 16
                            wj = Wb_s[s, jl] * T.if_then_else(jl < h, at, 1.0)
                            dd_s[s, lane] = wj * dv[0]
                            if lane < h:
                                Wbuf[slot, hv, lane] = wj
                            if lane == 0:
                                for r in T.unroll(2 * R):
                                    dd_s[s, 2 * L + r] = fv[r]
                                sc_s[s, 2] = P_s[it] * at
                                sc_s[s, 3] = b_s[it]
                        T.mbarrier_arrive(normed[s])
                        if STAMP:
                            if tx == 32 and itn == 0:
                                Sq[0, (bid * 32 + 18) // (V * KW), ((bid * 32 + 18) % (V * KW)) // KW, (bid * 32 + 18) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                    if STAMP:
                        if lane == 0:
                            Sq[0, (bid * 32 + 1 + nw) // (V * KW), ((bid * 32 + 1 + nw) % (V * KW)) // KW, (bid * 32 + 1 + nw) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
            else:
                g = (tx - NPROD) // 128
                t = (tx - NPROD) % 128
                warp = t // 32
                lane = t % 32
                q4 = lane % 4
                r0 = lane // 4
                afrag = T.alloc_local((4,), I32)
                bh = T.alloc_local((4,), F16)
                cfrag = T.alloc_local((8,), F32)
                acc = T.alloc_local((8,), F32)
                ubl = T.alloc_local((L,), F32)
                qfl = T.alloc_local((R,), F32)
                tk = T.alloc_local((8,), "int64")
                tacc = T.alloc_local((8,), F32)
                for i in T.unroll(8):
                    tacc[i] = 0.0
                ncons = T.ceildiv(nit - g, NCG)
                for it0 in T.serial(ncons):
                    it = g + it0 * NCG
                    p = bid + it * SMS
                    s = g * SPG + it0 % SPG
                    pb = it0 % 2
                    n = p // HV
                    hv = p % HV
                    if PROF:
                        tk[0] = T.call_extern("int64", "clock64")
                    T.mbarrier_wait_parity(sloaded[s], (it0 // SPG) & 1)
                    T.mbarrier_wait_parity(loaded[s], (it0 // SPG) & 1)
                    T.mbarrier_wait_parity(normb[s], (it0 // SPG) & 1)
                    if PROF:
                        tk[1] = T.call_extern("int64", "clock64")
                    if STAMP:
                        if t == 0 and it0 == 0:
                            Sq[0, (bid * 32 + 4 + g) // (V * KW), ((bid * 32 + 4 + g) % (V * KW)) // KW, (bid * 32 + 4 + g) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                    slot = slot_s[it]
                    h = h_s[it]
                    if slot <= 0 or DBG & 1 == 1:
                        T.mbarrier_wait_parity(normed[s], (it0 // SPG) & 1)
                        if slot <= 0:
                            o[n, hv, t] = T.cast(0.0, BF16)
                        T.fence_proxy_async()
                        T.mbarrier_arrive(consumed[s])
                    else:
                        v0 = T.cast(vx_s[s, t], F32)
                        kfmax = sc_s[s, 0]
                        kq = sc_s[s, 1]
                        kn = kn_s[s, t]
                        sv = skv_s[s, K + t]
                        for j in T.unroll(L):
                            ubl[j] = T.cast(Ub_s[s, j, t], F32)
                        for r in T.unroll(R):
                            qfl[r] = T.cast(UQ_s[s, r, K + t], F32)
                        if PROF:
                            tk[2] = T.call_extern("int64", "clock64")
                        # ---- checkpoint stream on tensor cores, this warp's 32 rows ----
                        for i in T.unroll(8):
                            cfrag[i] = 0.0
                        for kk in T.unroll(KW // 4):
                            for i in T.vectorized(4):
                                bh[i] = bf_s[s, r0, kk * 16 + q4 * 4 + i]
                            for mt in T.unroll(2):
                                rr = warp * 32 + mt * 16 + r0
                                wa = Qs[s, rr, kk * 4 + q4] ^ MAGICX
                                wb = Qs[s, rr + 8, kk * 4 + q4] ^ MAGICX
                                afrag[0] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[1] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4140))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[2] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wa, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                afrag[3] = T.reinterpret(I32, T.reinterpret("float16x2", T.call_extern(I32, "__byte_perm", wb, H2MAG, T.int32(0x4342))) - T.Broadcast(T.float16(1152.0), 2))
                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", afrag.data, 0, bh.data, 0, cfrag.data, mt * 4, T.bool(False))
                        T.mbarrier_wait_parity(normed[s], (it0 // SPG) & 1)
                        pn = sc_s[s, 2]
                        beta = sc_s[s, 3]
                        acc[4] = 0.0                                   # ring and factor terms from the norm warp's dots
                        acc[5] = 0.0
                        for j in T.unroll(L):
                            acc[4] += ubl[j] * dd_s[s, j]
                            acc[5] += ubl[j] * dd_s[s, L + j]
                        acc[6] = 0.0
                        acc[7] = 0.0
                        for r in T.unroll(R):
                            acc[6] += qfl[r] * dd_s[s, 2 * L + r]
                            acc[7] += qfl[r] * dd_s[s, 2 * L + R + r]
                        T.fence_proxy_async()                          # every stage read of this thread is done: release
                        T.mbarrier_arrive(consumed[s])
                        if q4 < 2:
                            for mt in T.unroll(2):
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4] = cfrag[mt * 4]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0, 2 * q4 + 1] = cfrag[mt * 4 + 1]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4] = cfrag[mt * 4 + 2]
                                mqy_s[g, pb, warp * 32 + mt * 16 + r0 + 8, 2 * q4 + 1] = cfrag[mt * 4 + 3]
                        T.sync_warp()                                   # this warp's rows only
                        if PROF:
                            tk[3] = T.call_extern("int64", "clock64")
                            tk[4] = tk[3]
                        mq = (mqy_s[g, pb, t, 0] + mqy_s[g, pb, t, 2]) * kfmax * sv
                        yq = (mqy_s[g, pb, t, 1] + mqy_s[g, pb, t, 3]) * kfmax * sv
                        u = beta * (v0 - (pn * (mq + acc[6]) + acc[4]))
                        y = pn * (yq + acc[7]) + acc[5] + kq * u
                        if DBG & 4:
                            mqy_s[g, pb, t, 0] = y + u
                        else:
                            o[n, hv, t] = T.cast(y, BF16)
                            Ubuf[slot, hv, h, t] = T.cast(u, BF16)
                            Kbuf[slot, hv, h, t] = T.cast(kn, BF16)
                        if t == 0 and DBG & 4 == 0:
                            Wbuf[slot, hv, h] = 1.0
                            Pbuf[slot, hv, 0] = pn
                            hcnt[slot, hv] = h + 1
                        if PROF:
                            tk[5] = T.call_extern("int64", "clock64")
                            for i in T.unroll(5):
                                tacc[i] += T.cast(tk[i + 1] - tk[i], F32)
                            tacc[5] += 1.0
                if STAMP:
                    if t == 0:
                        Sq[0, (bid * 32 + 8 + g) // (V * KW), ((bid * 32 + 8 + g) % (V * KW)) // KW, (bid * 32 + 8 + g) % KW] = T.cast(T.call_extern("int64", "clock64") - t_entry[0], I32)
                if STAMP:
                    if t == 0:
                        gt2 = T.call_extern("int64", "tl_gtimer")
                        Sq[0, (bid * 32 + 13 + g) // (V * KW), ((bid * 32 + 13 + g) % (V * KW)) // KW, (bid * 32 + 13 + g) % KW] = T.cast(gt2 & T.int64(0x7fffffff), I32)
                if PROF and g == 0 and t == 0 and bid < HV:
                    for i in T.unroll(6):
                        Wbuf[0, bid, i] = tacc[i]
    return step


class Pool:
    """Caller-owned state pool, the layout the serving engine uses (the same as the LeapQuant flush request).  Every
    (slot, head) owns a V x K fp32-sized region (64 KB) of one flat buffer; the checkpoint lives in its first 19 KB:
        bytes [0, 16384)            int8 codes, row-major [V, K]
        bytes [16384, 16896)        fp32 column scales s_k [K]
        bytes [16896, 17408)        fp32 row scales    s_v [V]
        bytes [17408, 19456)        fp16 Compensator Tokens [R, K + V]: row r = (u[r, :K] | q[r, :V])
    Slots are SS fp32 words apart; slot 0 is reserved (idx 0 = "no sequence").  The buffered updates (kbuf, ubuf, w, p) and
    the per-(slot, head) fill counter hcnt are separate dense tensors; hcnt == L means "just flushed, empty"."""

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
        self.w = torch.zeros(NS, HV, L, device=device)
        self.p = torch.ones(NS, HV, 1, device=device)
        self.hcnt = torch.zeros(NS, HV, dtype=torch.int32, device=device)
        self.kernel = make_step_kernel(NS, HV, HK, K, V, L, SS, R=R,
                                       SMS=torch.cuda.get_device_properties(self.raw.device).multi_processor_count)


def step_inplace(pool, idx, mixed, a, b, A_log, dt_bias, o):
    """Deployment form, the engine's call.  idx int32 [N]: slot per sequence (0 = skip, output zeros); mixed bf16
    [N, 2 HK K + HV V] = (q | k | v) as the projection produces it; a bf16 [N, HV, 1], b bf16 [N, HV]; A_log f32 [HV],
    dt_bias f32 [HV, 1]; o bf16 [N, HV, V] (written).  Appends the new update at entry hcnt (hcnt == L counts as 0), stores
    the decayed weights and gate product, increments hcnt."""
    pool.kernel(pool.codes32, pool.s_k, pool.s_v, pool.s_kv, pool.kbuf, pool.ubuf, pool.w, pool.p, pool.hcnt,
                mixed, a, b, A_log, dt_bias, pool.uq, idx, o)


_POOLS = {}


def run(codes, s_k, s_v, u, q, kbuf, ubuf, w, p, h, qx, kx, vx, a, b, A_log, dt_bias):
    B = codes.shape[0]
    dev = codes.device
    pool = _POOLS.get((B, dev))
    if pool is None:
        pool = _POOLS[(B, dev)] = Pool(B + 8, dev)
    sl = slice(1, B + 1)
    pool.codes[sl] = codes; pool.s_k[sl] = s_k; pool.s_v[sl] = s_v; pool.uq[sl, :, :, :K] = u; pool.uq[sl, :, :, K:] = q
    pool.kbuf[sl] = kbuf; pool.ubuf[sl] = ubuf; pool.w[sl] = w; pool.p[sl, :, 0] = p; pool.hcnt[sl] = h
    mixed = torch.cat([qx.reshape(B, -1), kx.reshape(B, -1), vx.reshape(B, -1)], 1).contiguous()
    o = torch.empty(B, HV, V, dtype=torch.bfloat16, device=dev)
    step_inplace(pool, torch.arange(1, B + 1, dtype=torch.int32, device=dev), mixed, a.reshape(B, HV, 1).contiguous(), b.contiguous(),
                 A_log.contiguous(), dt_bias.reshape(HV, 1).contiguous(), o)
    hh = h.long(); ar = torch.arange(B, device=dev)[:, None]; hv = torch.arange(HV, device=dev)[None, :]
    return (o, pool.kbuf[sl][ar, hv, hh].clone(), pool.ubuf[sl][ar, hv, hh].clone(), pool.w[sl].clone(), pool.p[sl, :, 0].clone())
