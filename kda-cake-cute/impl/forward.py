"""Direct CuTe DSL KDA forward host path and exact scheduler.

One host-side scheduling lever on top of the v75 champion kernels (the
kernels themselves only gained an h0 head-base + seq-dispatch perm in the
prologue):

LPT dispatch (mixed varlen): CTAs dispatch roughly in grid launch order,
so the longest chains must launch first or they become stragglers (the
harness's mixed cu puts the 96-chunk sequence LAST).  A tiny per-cu
permutation tensor reorders sequence dispatch by descending length
(identity for uniform/fixed shapes).  Measured mixed_h64 +24% /
mixed_h96 +14%, landing exactly on the greedy-list-scheduling
simulation's LPT bound; exhaustive block-order search says LPT is
optimal within the reachable (per-seq block) order space.

Routing per call:
  - pure m128 persistent with LPT slot lists for fixed shapes.
  - final-state mixed H64/H96 and uniform H64 use the retained exact single-launch
    head/tail schedules with native FP32 state handoffs.
  - final-state uniform H96 uses the faster CuTe whole-chain/piece route; its
    retained exact-inline schedule was measured and rejected.
  - final-state fixed H64 uses the M128 persistent kernel.
"""

import heapq
import math

import cuda.bindings.driver as cuda_driver
import cutlass
import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from . import pkdw as _k
from .planner import _one_chunk_warm_tables, pack_bins, plan_pieces

# Retained workload-specific schedules. Keep their predicates here rather than
# using sequence count as an implicit workload label in kernel routing.
_MIXED_LENS = [1300, 547, 2048, 963, 271, 3063]


def _is_mixed(lens) -> bool:
    return lens == _MIXED_LENS


def _is_uniform_1024(lens) -> bool:
    return len(lens) == 8 and all(length == 1024 for length in lens)


D = 128
C32 = 32
LOG2E = 1.4426950408889634

# Device compilation controls and caches.
_TPROBE = 0
_COMPILED: dict = {}
_DUMMY_EXP: dict = {}
_DUMMY_MID: dict = {}
_DUMMY_TPR: dict = {}
_FAILURE_HOST: dict = {}
_DONE_EVENT: dict = {}
LAST_TPROBE: list = [None]

_PLAN_CACHE: dict = {}
_STATE_SAFE: dict = {}
_SMS: dict = {}


def _sm_count(dev: torch.device) -> int:
    n = _SMS.get(dev.index)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _SMS[dev.index] = n
    return n


def _stream_state_is_safe(initial_state) -> bool:
    if initial_state is None:
        return True
    key = (initial_state.data_ptr(), initial_state._version)
    cached = _STATE_SAFE.get(key)
    if cached is None:
        state_max = float(initial_state.detach().abs().max())
        safe = math.isfinite(state_max) and state_max <= 2**12
        cached = (safe, initial_state.untyped_storage())
        _STATE_SAFE[key] = cached
    return cached[0]


def _get_compiled(
    H: int,
    dev: torch.device,
    gate2: int = 0,
    mid_bf16: bool = False,
    has_split: bool = False,
    do_export: bool = False,
    have_state: bool = True,
    beta_tma: bool = False,
    bf16_dv: bool = False,
    beta_bf16: bool = False,
    qk_rowpair: bool = False,
    rcp_decor: bool = True,
    rf_hoist: bool = True,
    restore_tail: bool = False,
    gram_w9: int = 0,
    reg_mode: int = 0,
    store_final_state: bool = False,
    full_chunks: bool = False,
    lower_bound_m5: bool = False,
    approximate_split: bool = False,
    prep_peel: bool = False,
    gate_roll: bool = False,
    cluster_size: int = 1,
):
    key = (
        H,
        gate2,
        mid_bf16,
        store_final_state,
        has_split,
        do_export,
        have_state,
        beta_tma,
        bf16_dv,
        beta_bf16,
        qk_rowpair,
        rcp_decor,
        rf_hoist,
        restore_tail,
        gram_w9,
        reg_mode,
        full_chunks,
        lower_bound_m5,
        approximate_split,
        prep_peel,
        cluster_size,
        gate_roll,
    )
    entry = _COMPILED.get(key)
    if entry is None:
        N = 2
        T = 8 * C32
        bf = torch.bfloat16
        mk = lambda t, a=16: from_dlpack(t, assumed_align=a).mark_compact_shape_dynamic(
            mode=0
        )
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
            2, D, D, dtype=torch.bfloat16 if mid_bf16 else torch.float32, device=dev
        )
        mfl = torch.zeros(2, dtype=torch.int32, device=dev)
        tpr = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        out = torch.empty(T, H, D, dtype=bf, device=dev)
        nc2 = (T // 2 + C32 - 1) // C32
        exp_ws = torch.empty(H * nc2 * 64, 1, 256, dtype=bf, device=dev)
        expt = torch.empty(H * nc2, 160, dtype=torch.float32, device=dev)
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
        tfn = cute.compile(
            _k._launch_pkd,
            mk(q),
            mk(k),
            mk(v),
            mk(g),
            mk(beta, 4),
            from_dlpack(alog, assumed_align=16),
            from_dlpack(dtb, assumed_align=16),
            mk(st),
            mk(ns),
            mk(out),
            mk(cu_t, 4),
            mk(soff, 4),
            mk(schain, 4),
            mk(spt0, 4),
            mk(sptn, 4),
            mk(ssrc, 4),
            mk(sdst, 4),
            mk(mid),
            mk(mfl, 4),
            mk(exp_ws),
            mk(expt),
            mk(tpr, 8),
            cutlass.Int64(_failure_marker(dev).data_ptr()),
            cutlass.Int32(1),
            cutlass.Int32(G),
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Int32(nc2),
            cutlass.Int32(1),
            cutlass.Float32(1.0),
            cutlass.Float32(-1.0),
            H_=H,
            stream=stream,
            TPROBE_=_TPROBE,
            GATE2_=gate2,
            FINAL_=1 if store_final_state else 0,
            SPLIT_=1 if has_split and not approximate_split else 0,
            WARM_SPLIT_=1 if approximate_split else 0,
            EXPORT_=1 if do_export else 0,
            STATE_=1 if have_state else 0,
            BETA_TMA_=1 if beta_tma else 0,
            BF16_DV_=1 if bf16_dv else 0,
            BETA_BF16_=1 if beta_bf16 else 0,
            QK_ROWPAIR_=int(qk_rowpair),
            RCP_DECOR_=1 if rcp_decor else 0,
            RF_HOIST_=1 if rf_hoist else 0,
            RESTORE_TAIL_=1 if restore_tail else 0,
            GRAM_W9_=int(gram_w9),
            REG_MODE_=int(reg_mode),
            FULL_CHUNKS_=1 if full_chunks else 0,
            LOWER_BOUND_M5_=1 if lower_bound_m5 else 0,
            CLUSTER_=int(cluster_size),
            PREP_PEEL_=1 if prep_peel else 0,
            GATE_ROLL_=1 if gate_roll else 0,
            options="--enable-tvm-ffi --opt-level 3",
        )
        tfn(
            q,
            k,
            v,
            g,
            beta,
            alog,
            dtb,
            st,
            ns,
            out,
            cu_t,
            soff,
            schain,
            spt0,
            sptn,
            ssrc,
            sdst,
            mid,
            mfl,
            exp_ws,
            expt,
            tpr,
            _failure_marker(dev).data_ptr(),
            1,
            G,
            0,
            0,
            nc2,
            1,
            1.0,
            -1.0,
            stream,
        )
        torch.cuda.synchronize()
        _COMPILED[key] = tfn
        entry = tfn
    return entry


def _dummy_exp(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_EXP.get(key)
    if tensors is None:
        tensors = (
            torch.empty(64, 1, 256, dtype=torch.bfloat16, device=dev),
            torch.empty(1, 160, dtype=torch.float32, device=dev),
        )
        _DUMMY_EXP[key] = tensors
    return tensors


def _dummy_mid(dev: torch.device):
    key = dev.index
    tensors = _DUMMY_MID.get(key)
    if tensors is None:
        tensors = (
            torch.empty(1, D, D, dtype=torch.float32, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        )
        _DUMMY_MID[key] = tensors
    return tensors


def _tprobe_buf(G: int, dev: torch.device):
    if _TPROBE:
        tensor = torch.zeros(G * 64, dtype=torch.int64, device=dev)
        LAST_TPROBE[0] = tensor
        return tensor
    key = dev.index
    tensor = _DUMMY_TPR.get(key)
    if tensor is None:
        tensor = torch.zeros(64, dtype=torch.int64, device=dev)
        _DUMMY_TPR[key] = tensor
    return tensor


def _failure_marker(dev: torch.device):
    key = dev.index
    marker = _FAILURE_HOST.get(key)
    if marker is None:
        marker = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        _FAILURE_HOST[key] = marker
        _DONE_EVENT[key] = torch.cuda.Event()
    return marker


@torch.no_grad()
def _launch_forward(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    sched=None,
    export=None,
    gate2=0,
    has_split=False,
    beta_tma=False,
    bf16_dv=True,
    beta_bf16=True,
    qk_rowpair=False,
    rcp_decor=True,
    rf_hoist=True,
    restore_tail=False,
    gram_w9=0,
    reg_mode=0,
    full_chunks=False,
    lower_bound_m5=False,
    approximate_split=False,
    prep_peel=False,
    cluster_size=1,
    gate_roll=False,
):
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

    store_final_state = final_state is not None
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
        (soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, fepoch) = sched
    if export is None:
        exp_ws, expt = _dummy_exp(dev)
        do_export, export_seq, nc2 = 0, 0, 1
    else:
        exp_ws, expt, export_seq, nc2 = export
        do_export = 1

    tfn = _get_compiled(
        H,
        dev,
        gate2,
        mid.dtype == torch.bfloat16,
        has_split=has_split,
        do_export=bool(do_export),
        have_state=bool(have_state),
        beta_tma=beta_tma,
        bf16_dv=bf16_dv,
        beta_bf16=beta_bf16,
        qk_rowpair=qk_rowpair,
        rcp_decor=rcp_decor,
        rf_hoist=rf_hoist,
        restore_tail=restore_tail,
        gram_w9=gram_w9,
        reg_mode=reg_mode,
        store_final_state=store_final_state,
        full_chunks=full_chunks,
        lower_bound_m5=lower_bound_m5,
        approximate_split=approximate_split,
        prep_peel=prep_peel,
        cluster_size=cluster_size,
        gate_roll=gate_roll,
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    tfn(
        qf,
        kf,
        vf,
        gf,
        bf_,
        A_log,
        dt_bias.view(H, D),
        st.view(nseqs * H, D, D),
        final_state.view(nseqs * H, D, D),
        of,
        cu_t,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        mid,
        mfl,
        exp_ws,
        expt,
        _tprobe_buf(G, dev),
        _failure_marker(dev).data_ptr(),
        have_state,
        G,
        do_export,
        export_seq,
        nc2,
        fepoch,
        float(scale),
        float(lower_bound) * LOG2E,
        stream,
    )
    return out, final_state


C_TOK = 32  # chunk tokens; split points are chunk-aligned
INLINE_SPLIT_MARGIN = 0.98
HANDOFF_SPLIT_SLOPE = 0.04


def _exact_inline_schedule(lens, H: int, nslots: int):
    """Build the retained exact head/tail handoff schedule for INT21 varlen shapes."""
    if nslots != 148 or H not in (64, 96):
        return None
    if not (
        (_is_mixed(lens) and H in (64, 96)) or (H == 64 and _is_uniform_1024(lens))
    ):
        return None

    chunks = [(length + C_TOK - 1) // C_TOK for length in lens]
    full = [length // C_TOK for length in lens]
    chain_chunks = [chunks[chain // H] for chain in range(len(lens) * H)]
    chain_full = [full[chain // H] for chain in range(len(lens) * H)]
    order_desc = sorted(
        range(len(chain_chunks)), key=lambda chain: -chain_chunks[chain]
    )

    def lpt_makespan(costs):
        heap = [0.0] * nslots
        heapq.heapify(heap)
        for cost in sorted(costs, reverse=True):
            load = heapq.heappop(heap)
            heapq.heappush(heap, load + cost)
        return max(heap)

    base_span = lpt_makespan([count + 1 for count in chain_chunks])
    area_floor = -(-(sum(chain_chunks) + len(chain_chunks)) // nslots)
    longest = max(chain_chunks)
    if base_span * INLINE_SPLIT_MARGIN <= area_floor + 1:
        return None
    if longest + 1 >= base_span:
        return None

    def build_jobs(n_split, prefix):
        jobs = []
        for pos, chain in enumerate(order_desc):
            count = chain_chunks[chain]
            if pos < n_split:
                if prefix >= count or prefix > chain_full[chain]:
                    return None
                jobs.append((prefix + 1, 1, chain))
                jobs.append((count - prefix + 1, 2, chain))
            else:
                jobs.append((count + 1, 0, chain))
        return jobs

    def pack_and_simulate(jobs):
        heap = [(0, slot) for slot in range(nslots)]
        heapq.heapify(heap)
        bins = [[] for _ in range(nslots)]
        for job in sorted(
            range(len(jobs)), key=lambda idx: (-jobs[idx][0], jobs[idx][1])
        ):
            load, slot = heapq.heappop(heap)
            bins[slot].append(job)
            heapq.heappush(heap, (load + jobs[job][0], slot))
        rank = {1: 0, 0: 1, 2: 2}
        ready = {}
        for slot_jobs in bins:
            slot_jobs.sort(key=lambda idx: (rank[jobs[idx][1]], -jobs[idx][0]))
            elapsed = 0
            for job in slot_jobs:
                cost, kind, chain = jobs[job]
                if kind == 2:
                    continue
                elapsed += cost
                if kind == 1:
                    ready[chain] = elapsed
        span = 0
        for slot_jobs in bins:
            elapsed = 0
            for job in slot_jobs:
                cost, kind, chain = jobs[job]
                if kind == 2:
                    elapsed = max(elapsed, ready[chain])
                elapsed += cost
            span = max(span, elapsed)
        return span, bins

    best = None
    for n_split in range(8, len(chain_chunks) + 1, 8):
        for prefix in range(1, longest):
            jobs = build_jobs(n_split, prefix)
            if jobs is None:
                continue
            span, bins = pack_and_simulate(jobs)
            key = (span + HANDOFF_SPLIT_SLOPE * n_split, n_split, prefix)
            if best is None or key < best[0]:
                best = (key, n_split, prefix, jobs, bins)
    if best is None or best[0][0] > base_span - 1.0:
        return None

    if H == 96 and _is_mixed(lens):
        n_split, prefix = 56, 4
        jobs = build_jobs(n_split, prefix)
        _, bins = pack_and_simulate(jobs)
    else:
        _, n_split, prefix, jobs, bins = best
    buffer_for = {chain: index for index, chain in enumerate(order_desc[:n_split])}
    items = []
    for slot_jobs in bins:
        slot = []
        for job in slot_jobs:
            _cost, kind, chain = jobs[job]
            length = lens[chain // H]
            if kind == 0:
                slot.append((chain, 0, length, -1, -1))
            elif kind == 1:
                slot.append((chain, 0, prefix * C_TOK, -1, buffer_for[chain]))
            else:
                start = prefix * C_TOK
                slot.append((chain, start, length - start, buffer_for[chain], -1))
        items.append(slot)
    return items, n_split


def _piece_schedule(
    lens, H: int, nslots: int, dev: torch.device, store_final_state: bool
):
    """LPT of the nseq*H chains onto nslots persistent CTAs, plus one
    round of overflow-chain halving (v96 split) when the whole-chain
    makespan is quantization-bound (uniform varlen: 768 chains -> 28
    slots x 6 binds at 192 chunk-times vs the 166 ideal).

    Pieces are (chain, t0, tn, src, dst).  H64 uses two halves: the producer
    runs just before a low-load slot's final whole chain and the consumer is
    appended elsewhere.  H96 uses five 6/6/6/7/7-chunk pieces staggered after
    one through five whole chains.  Each dependency therefore has a full-chain
    lead.  The piece DAG is acyclic and every intermediate state has one
    producer and one consumer.
    """
    inline = _exact_inline_schedule(lens, H, nslots) if store_final_state else None
    if inline is not None:
        items, nbuf = inline
        fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
        for slot in items:
            for chain, t0, tn, src, dst in slot:
                fc.append(chain)
                ft0.append(t0)
                ftn.append(tn)
                fsrc.append(src)
                fdst.append(dst)
            off.append(len(fc))
        i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=dev)
        return (
            i32(off),
            i32(fc),
            i32(ft0),
            i32(ftn),
            i32(fsrc),
            i32(fdst),
            len(items),
            nbuf,
        )

    nseq = len(lens)
    chains = nseq * H
    G = min(chains, nslots)
    heap = [(0, s) for s in range(G)]
    heapq.heapify(heap)
    slots = [[] for _ in range(G)]
    loads = [0] * G
    order = sorted(range(nseq), key=lambda i: -lens[i])
    for s_i in order:
        for h in range(H):
            load, sl = heapq.heappop(heap)
            slots[sl].append(s_i * H + h)
            loads[sl] = load + lens[s_i]
            heapq.heappush(heap, (loads[sl], sl))
    items = [[(c, 0, lens[c // H], -1, -1) for c in sl] for sl in slots]
    ideal = sum(lens) * H / G
    M0 = max(loads)
    nbuf = 0
    if H == 96 and G == 148 and _is_uniform_1024(lens):
        # Remove the 28 sixth chains, then split each 32-chunk chain as
        # 6+6+6+7+7.  The 140 pieces occupy distinct slots after one
        # through five whole chains, giving each dependency a full-chain
        # lead and reaching the 167-chunk integer lower bound.
        peaks = [(s, slots[s][-1]) for s in range(G) if loads[s] == M0]
        for s, c in peaks:
            idx = slots[s].index(c)
            slots[s].pop(idx)
            items[s].pop(idx)
        np = len(peaks)
        cuts = (0, 6, 12, 18, 25, 32)
        for i, (_, c) in enumerate(peaks):
            bufs = tuple(range(nbuf, nbuf + 4))
            nbuf += 4
            for p in range(5):
                src = -1 if p == 0 else bufs[p - 1]
                dst = -1 if p == 4 else bufs[p]
                t0 = cuts[p] * C_TOK
                tn = (cuts[p + 1] - cuts[p]) * C_TOK
                sl = p * np + i
                items[sl].insert(1 + p, (c, t0, tn, src, dst))
    elif (
        H == 96
        and G == nslots == 148
        and sorted((L + C_TOK - 1) // C_TOK for L in lens) == [9, 18, 31, 41, 64, 96]
    ):
        # Whole-chain 169-chunk packing for the official mixed H96 shape.
        # Raw-token LPT leaves 32 slots at 173 chunks.  Each exchange below
        # repartitions three complete slots at equal total work and removes
        # one such tail.  The resulting max is 169 chunks (the 168 exact
        # average is infeasible with whole chains), with no state handoff.
        from collections import Counter, defaultdict
        from itertools import permutations

        def nch(c):
            return (lens[c // H] + C_TOK - 1) // C_TOK

        def slot_shape(slot):
            return tuple(sorted((nch(c) for c in slot), reverse=True))

        def repartition(slot_ids, target_shapes):
            old = [slots[s] for s in slot_ids]
            old_counts = [Counter(map(nch, slot)) for slot in old]
            best_overlap = -1
            targets = None
            for candidate in permutations(target_shapes):
                overlap = sum(
                    sum(
                        min(old_counts[i][n], Counter(candidate[i])[n])
                        for n in old_counts[i]
                    )
                    for i in range(3)
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    targets = candidate

            result = [[] for _ in range(3)]
            remaining = defaultdict(list)
            for i, slot in enumerate(old):
                need = Counter(targets[i])
                for c in slot:
                    n = nch(c)
                    if need[n]:
                        result[i].append(c)
                        need[n] -= 1
                    else:
                        remaining[n].append(c)
            for i in range(3):
                need = Counter(targets[i]) - Counter(map(nch, result[i]))
                for n, count in need.items():
                    for _ in range(count):
                        result[i].append(remaining[n].pop())
                result[i].sort(key=lambda c: (-nch(c), c))
            assert not any(remaining.values())
            for s, slot in zip(slot_ids, result):
                slots[s] = slot

        transforms = (
            (
                4,
                ((96, 41, 18, 9, 9), (64, 41, 31, 31), (96, 41, 18, 9)),
                ((96, 41, 31), (96, 41, 31), (64, 41, 18, 18, 9, 9, 9)),
            ),
            (
                4,
                ((96, 41, 18, 18), (64, 41, 31, 31), (96, 41, 18, 9)),
                ((96, 41, 31), (96, 41, 31), (64, 41, 18, 18, 18, 9)),
            ),
            (
                8,
                ((96, 41, 18, 18), (96, 31, 31, 9), (96, 41, 18, 9)),
                ((96, 41, 31), (96, 41, 31), (96, 18, 18, 18, 9, 9)),
            ),
            (
                16,
                ((96, 41, 18, 18), (64, 64, 31, 9), (96, 41, 18, 9)),
                ((64, 64, 41), (96, 18, 18, 18, 9, 9), (96, 41, 31)),
            ),
        )
        for count, old_shapes, new_shapes in transforms:
            for _ in range(count):
                used = set()
                slot_ids = []
                for wanted in old_shapes:
                    s = next(
                        s
                        for s in range(G)
                        if s not in used and slot_shape(slots[s]) == wanted
                    )
                    used.add(s)
                    slot_ids.append(s)
                repartition(slot_ids, new_shapes)
        items = [[(c, 0, lens[c // H], -1, -1) for c in slot] for slot in slots]
        chunk_loads = [sum(map(nch, slot)) for slot in slots]
        assert max(chunk_loads) == 169
        assert Counter(chunk_loads) == Counter({168: 128, 169: 16, 164: 4})
    elif (
        H == 64
        and G == nslots == 148
        and sorted((L + C_TOK - 1) // C_TOK for L in lens) == [9, 18, 31, 41, 64, 96]
    ):
        # v109 minimal-split schedule for the official mixed H64 shape:
        # 64x[96+18] + 64x[64+41+9] + 12x[31,31,31] fill 140 slots at
        # <=114 chunks; the four leftover 31-chains split (10, 21) with
        # the producer first on one [31,31,31] slot and the consumer last
        # on another, giving an 83-chunk dependency lead.  Makespan 114
        # versus 125 for LPT, with only four bf16 mid-state handoffs.
        by_n = {}
        for i2, L in enumerate(lens):
            by_n.setdefault((L + C_TOK - 1) // C_TOK, i2)
        sq = {n: [s_i * H + h for h in range(H)] for n, s_i in by_n.items()}
        items = []
        for j in range(64):
            items.append(
                [
                    (sq[96][j], 0, lens[sq[96][j] // H], -1, -1),
                    (sq[18][j], 0, lens[sq[18][j] // H], -1, -1),
                ]
            )
        for j in range(64):
            items.append(
                [
                    (sq[64][j], 0, lens[sq[64][j] // H], -1, -1),
                    (sq[41][j], 0, lens[sq[41][j] // H], -1, -1),
                    (sq[9][j], 0, lens[sq[9][j] // H], -1, -1),
                ]
            )
        c31 = sq[31]
        L31 = lens[c31[0] // H]
        for j in range(12):
            items.append([(c31[3 * j + k], 0, L31, -1, -1) for k in range(3)])
        for j in range(4):
            base = 36 + 6 * j
            c_sp = c31[60 + j]
            prod_slot = [(c_sp, 0, 10 * C_TOK, -1, nbuf)]
            prod_slot += [(c31[base + k], 0, L31, -1, -1) for k in range(3)]
            cons_slot = [(c31[base + 3 + k], 0, L31, -1, -1) for k in range(3)]
            cons_slot += [(c_sp, 10 * C_TOK, L31 - 10 * C_TOK, nbuf, -1)]
            nbuf += 1
            items.append(prod_slot)
            items.append(cons_slot)
        G = len(items)
        slots = [[it[0] for it in sl] for sl in items]
    elif G == nslots and M0 - ideal > 0.04 * ideal:
        # halve the longest (>=16-chunk) chain of every peak slot;
        # simulate placement, mutate only if the makespan improves
        rm = {}
        for s in range(G):
            if loads[s] == M0:
                cands = [c for c in slots[s] if lens[c // H] >= 16 * C_TOK]
                if cands:
                    rm[s] = max(cands, key=lambda c2: lens[c2 // H])
        if rm:
            tload = loads[:]
            for s, c in rm.items():
                tload[s] -= lens[c // H]
            heap2 = [(tload[s], s) for s in range(G)]
            heapq.heapify(heap2)
            place = []
            for s, c in sorted(rm.items()):
                L = lens[c // H]
                tiles = (L + C_TOK - 1) // C_TOK
                t1 = ((tiles + 1) // 2) * C_TOK
                l1, s1 = heapq.heappop(heap2)
                l2, s2 = heapq.heappop(heap2)
                heapq.heappush(heap2, (l1 + t1, s1))
                heapq.heappush(heap2, (l2 + (L - t1), s2))
                tload[s1] += t1
                tload[s2] += L - t1
                place.append((s, c, s1, s2, t1, L))
            if max(tload) <= M0 - max(C_TOK, 0.02 * ideal):
                # pop all split chains first: piece inserts would shift
                # items[] out of sync with slots[] index lookups
                for s, c, s1, s2, t1, L in place:
                    idx = slots[s].index(c)
                    slots[s].pop(idx)
                    items[s].pop(idx)
                for s, c, s1, s2, t1, L in place:
                    # Keep the first four uniform waves phase-aligned.
                    # Placing the producer just before the slot's final
                    # whole chain gives it one full-chain lead over the
                    # appended consumer without injecting a short chain
                    # into the device-wide t=0 wave.
                    ppos = max(0, len(items[s1]) - 1)
                    items[s1].insert(ppos, (c, 0, t1, -1, nbuf))
                    items[s2].append((c, t1, L - t1, nbuf, -1))
                    nbuf += 1
    # B300 maps persistent block IDs to SM/topology positions repeatably.
    # Rotate only heterogeneous/full-device official schedules so their
    # critical slot band lands on the faster block-ID region.  Slot contents,
    # within-slot order, dependencies, and all output ownership stay intact.
    # Donor-Gram routes were re-swept after their issue timing changed
    # (mixed H64 in v138, uniform H96 in v139). Uniform H96 uses rotation 106,
    # tuned while that route still stored its gate table as FP16.
    mixed = _is_mixed(lens)
    uniform = _is_uniform_1024(lens)
    rotation = 0
    if G == 148:
        if H == 96 and mixed:
            rotation = 18
        elif H == 64 and mixed:
            rotation = 134
        elif H == 96 and uniform:
            rotation = 106
        elif H == 64 and uniform:
            rotation = 53
    if rotation:
        items = items[rotation:] + items[:rotation]
    fc, ft0, ftn, fsrc, fdst, off = [], [], [], [], [], [0]
    for s in range(G):
        for c, t0, tn, src, dst in items[s]:
            fc.append(c)
            ft0.append(t0)
            ftn.append(tn)
            fsrc.append(src)
            fdst.append(dst)
        off.append(len(fc))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return (i32(off), i32(fc), i32(ft0), i32(ftn), i32(fsrc), i32(fdst), G, nbuf)


def _plan(
    cu_seqlens,
    H: int,
    T: int,
    dev: torch.device,
    approximate: bool,
    store_final_state: bool,
):
    """Cached per-cu/H piece plan + midstate ring + epoch counter."""
    key = (
        0 if cu_seqlens is None else cu_seqlens.data_ptr(),
        # Inference tensors carry no version counter: key on their values.
        0
        if cu_seqlens is None
        else tuple(cu_seqlens.tolist())
        if cu_seqlens.is_inference()
        else cu_seqlens._version,
        1 if cu_seqlens is None else int(cu_seqlens.numel()),
        H,
        T,
        dev.index,
        approximate,
        store_final_state,
    )
    hit = _PLAN_CACHE.get(key)
    if hit is not None:
        return hit
    if cu_seqlens is None:
        lens = [T]
        cu32 = torch.tensor([0, T], dtype=torch.int32, device=dev)
    else:
        lens = torch.diff(cu_seqlens).tolist()  # one host sync per distinct cu
        cu32 = cu_seqlens.to(torch.int32)
    nslots = _sm_count(dev)
    if approximate:
        offsets = [0]
        for length in lens:
            offsets.append(offsets[-1] + length)
        rows, has_split, _ = plan_pieces(
            offsets,
            H,
            nslots,
            _one_chunk_warm_tables(offsets),
            1,
            int(store_final_state),
        )
        order, cta_vals = pack_bins(rows, nslots)
        rows = [rows[index] for index in order]
        starts = cta_vals[::3]
        counts = cta_vals[1::3]
        values = (
            starts + [len(rows)],
            [row[3] * H + row[6] for row in rows],
            [row[0] - offsets[row[3]] for row in rows],
            [row[1] - row[0] for row in rows],
            [-1 - (row[2] - row[0]) // C_TOK for row in rows],
            [-1 if row[5] else -2 for row in rows],
        )
        i32 = lambda value: torch.tensor(value, dtype=torch.int32, device=dev)
        soff, schain, spt0, sptn, ssrc, sdst = map(i32, values)
        G = len(counts)
        mid, mfl = _dummy_mid(dev)
        nbuf = 0
    else:
        has_split = False
    if not approximate:
        (soff, schain, spt0, sptn, ssrc, sdst, G, nbuf) = _piece_schedule(
            lens, H, nslots, dev, store_final_state
        )
    if nbuf > 0:
        native_handoff = H in (64, 96) and (_is_mixed(lens) or _is_uniform_1024(lens))
        mid = torch.empty(
            nbuf,
            D,
            D,
            dtype=torch.float32 if native_handoff else torch.bfloat16,
            device=dev,
        )
        mfl = torch.zeros(nbuf, dtype=torch.int32, device=dev)
    elif not approximate:
        mid, mfl = _dummy_mid(dev)
    # Pin the source allocation with its cached plan.  For int64 inputs, cu32
    # is a separate allocation; without this storage reference the CUDA
    # allocator can reuse the source data_ptr for a different layout and
    # stale-hit the cache key.  ep[0] remains the launch epoch consumed by
    # fwd(); ep[1] is never read on the launch path.
    cu_storage = None if cu_seqlens is None else cu_seqlens.untyped_storage()
    ep = [0, cu_storage]
    plan = (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        mfl,
        ep,
        has_split if approximate else nbuf > 0,
        all(length % C_TOK == 0 for length in lens),
    )
    _PLAN_CACHE[key] = plan
    return plan


@torch.no_grad()
def fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    allow_approximate_split=False,
):
    dev = q.device
    H = q.shape[2]
    T = q.shape[0] * q.shape[1]
    nseq = 1 if cu_seqlens is None else int(cu_seqlens.numel()) - 1
    fixed = nseq == 1
    store_final_state = final_state is not None
    allow_approximate_split = allow_approximate_split and _stream_state_is_safe(initial_state)

    (
        cu32,
        soff,
        schain,
        spt0,
        sptn,
        ssrc,
        sdst,
        G,
        mid,
        mfl,
        ep,
        has_split,
        full_chunks,
    ) = _plan(cu_seqlens, H, T, dev, allow_approximate_split, store_final_state)
    ep[0] += 1
    # Split handoff flags count the producer's 256 releases.  Clear them on the
    # launch stream before every launch (a CUDA graph records the clear with the
    # kernel) and wait for one launch's worth, so eager calls, graph captures and
    # replays in any order all wait for this launch's producer.
    fepoch = ep[0]
    if has_split and not allow_approximate_split:
        mfl.zero_()
        fepoch = 1
    # v98 fused-gate kernel variant: routed to single-sequence (fixed)
    # shapes only — it wins ~1-3% there (long chains, steady-state L1
    # relief) but costs ~2% on short varlen chains (prep-latency exposure
    # at chain starts; profiles/NOTES.md session 2).
    common = {
        "initial_state": initial_state,
        "final_state": final_state,
        "cu_seqlens": cu32,
        "sched": (soff, schain, spt0, sptn, ssrc, sdst, G, mid, mfl, fepoch),
        "gate2": 1 if fixed else 0,
    }
    # NCU v119 source counters identify the prep warpgroup's raw-Q/K indexing
    # as a compiler-spill trigger.  Mode 4 is algebraically just a four-row
    # half-warp pairing through the combined prep-thread coordinate.  Mode 8
    # keeps that row map and enables the fused per-channel gate scan while
    # retaining varlen's measured-faster raw V/residual layout.
    qk_rowpair = 4 if fixed else 8
    # Central, preaddressed Gram issue relieves generic-tail prep scheduling.
    # Full-chunk varlen routes retain per-instance self-issue.
    generic = not fixed and not full_chunks
    gram_w9 = 2 if generic else 0
    # Full-chunk varlen favors 160/88/24/48. Packed normalization lets generic
    # mixed use 152/96/24/48 without H96's old 56-reg prep. Both fixed shapes
    # use 160/88/24/48 after the strict final-state retune.
    reg_mode = 1 if fixed else 0
    if not fixed:
        reg_mode = 1 if full_chunks or (H == 96 and store_final_state) else 2
    # Cluster-2 also wins on uniform H64 after the dedicated MMA issuer;
    # matched 15x100 bookends improve by 0.208% over cluster-1.
    cluster_size = 4 if fixed else (2 if nseq in (6, 8) else 1)
    result = _launch_forward(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        has_split=has_split,
        beta_tma=True,
        qk_rowpair=qk_rowpair,
        gram_w9=gram_w9,
        restore_tail=False,
        reg_mode=reg_mode,
        full_chunks=full_chunks,
        # H64 generic also benefits from deleting the runtime lower-bound
        # scalar; the same specialization regresses generic H96.
        lower_bound_m5=(full_chunks or H == 64) and lower_bound == -5.0,
        approximate_split=allow_approximate_split and has_split,
        prep_peel=H == 96 and generic,
        cluster_size=cluster_size,
        # Rolling eight-row gate batches also shorten generic H64 by about
        # 0.17% in matched 5x100 bookends; full-chunk H64 still prefers the
        # scalar schedule.
        gate_roll=generic,
        **common,
    )
    if allow_approximate_split and has_split:
        done = _DONE_EVENT[dev.index]
        done.record()
        done.synchronize()
        if int(_failure_marker(dev)[0]) == ep[0]:
            return fwd(
                q,
                k,
                v,
                g,
                beta,
                scale,
                out,
                A_log,
                dt_bias,
                lower_bound,
                initial_state=initial_state,
                final_state=final_state,
                cu_seqlens=cu_seqlens,
                allow_approximate_split=False,
            )
    return result


@torch.no_grad()
def run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens=None):
    """Task interface used by the judge."""
    out = torch.empty_like(v)
    final_state = torch.empty_like(initial_state, dtype=torch.float32)
    fwd(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        out,
        A_log,
        dt_bias.reshape(q.shape[2], D),
        -5.0,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
    return out, final_state
