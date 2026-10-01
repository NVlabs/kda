"""Direct CuTe DSL piece-stream planner for B300.

Every case routes through the piece-table M128 kernel
(impl/kda_fwd_pieces_m128.py): each CTA walks one [walk_start, end) token
range of one (sequence, head) chain and stores outputs from out_start on.
Long chains are cut into parallel pieces; a non-first piece rebuilds the
recurrent state from a zero-state warmup walk. Pieces are LPT-packed onto at
most one CTA per SM, and each CTA streams its pieces through a single
continuous pipeline, so per-piece launch, TMEM-alloc, and fill/drain costs are
paid once per CTA instead of once per piece. Cut locations are shape-only. The
stream kernel itself proves every warmup window against the current `g`: the
prep gate walk's window total is compared to the decay threshold in-kernel,
and a shallow window publishes the launch generation to a host-pinned scalar
(UVA sysmem store).  `g` may therefore change on every invocation without a
separate detector kernel or D2H copy; on the rare proof failure the host
reruns the complete unsplit fallback over the whole batch.

The provided FP32 initial state is loaded by the stream kernel and final states
are produced natively in FP32. The loom M64 schedule (with BF16 state staging)
is also the complete fallback when the dynamic decay proof fails.
"""

from __future__ import annotations

import heapq
from itertools import pairwise

K2_CHUNK = 32
HEAD_DIM = 128
PIECE_OVERHEAD_CHUNKS = 3
MAX_WARM = 8
WARM_EST = 3
# Align the split proof with the benchmark's 5e-2 numeric contract.  This
# threshold bounds the discarded boundary contribution; the independent state
# guard below constrains the stream kernel's intermediate dynamic range.
PROOF_THRESHOLD_LOG2 = -48.0
# Leave two log2 units between the approximate in-kernel detector and the
# established proof threshold.
DETECT_THRESHOLD_LOG2 = PROOF_THRESHOLD_LOG2 - 2.0
SPLIT_FRACTIONS = (
    0.03125,
    0.0625,
    0.09375,
    0.125,
    0.15625,
    0.1875,
    0.21875,
    0.25,
    0.3125,
    0.375,
    0.5,
    0.75,
    1.0,
)
JITTERS = (-2, 0, 2)
# Measured LPT-model bias: split plans run ~4% slower than modeled, so a
# split plan must beat the trivial plan by more than this margin to dispatch.
SPLIT_MARGIN = 0.96
# A B300 sweep of dense random FP32 state remained finite through scale 2^12
# (observed max above 1.6e4) and became nonfinite at scale 2^16.  Keeping the
# absolute-value limit at 2^12 is conservative for dense inputs while retaining
# the required sparse striped-state split case whose maximum is exactly 2^12.
MAX_STREAM_INITIAL_STATE_ABS = float(2**12)


def _lpt_makespan(job_lens, machines):
    """Longest-processing-time greedy makespan.

    :param job_lens: Per-job costs in chunk slots.
    :type job_lens: list
    :param machines: Machine count.
    :type machines: int
    :return: Modeled makespan in chunk slots.
    :rtype: float
    """
    heap = [0.0] * machines
    heapq.heapify(heap)
    for cost in sorted(job_lens, reverse=True):
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + cost)
    return max(heap)


def _one_chunk_warm_tables(offsets):
    """Shape-only candidate tables whose cuts all use one warmup chunk."""

    tables = []
    for bos, eos in pairwise(offsets):
        nfull = (eos - bos) // K2_CHUNK
        tables.append(None if nfull < 2 else [None] + [1] * nfull)
    return tables


def _balanced_cuts(num_chunks, pieces, jitter, cut_max):
    """Balanced cut positions for a pieces-way chain split.

    :param num_chunks: Chain length in chunks (ceil, incl. partial tail).
    :type num_chunks: int
    :param pieces: Piece count.
    :type pieces: int
    :param jitter: Offset applied to the first cut.
    :type jitter: int
    :param cut_max: Largest chunk-aligned cut position.
    :type cut_max: int
    :return: Cut positions, or None when infeasible.
    :rtype: tuple
    """
    tail = max(1, round((num_chunks - WARM_EST) / pieces))
    first = num_chunks - (pieces - 1) * tail + jitter
    if first < 1:
        return None
    cuts = tuple(first + i * tail for i in range(pieces - 1))
    if cuts[-1] > cut_max or num_chunks - cuts[-1] < 1:
        return None
    return cuts


def _msweep_cuts(num_chunks, target, warms_tab, cut_max, piece_overhead):
    """Greedy unequal cuts so every piece cost stays within ``target``.

    Walks the chain head-to-tail emitting the largest provable piece each
    step, so long chains become several ``target``-cost pieces plus one
    small remainder that LPT can pack onto otherwise idle SMs.

    :param num_chunks: Chain length in chunks (ceil, incl. partial tail).
    :type num_chunks: int
    :param target: Modeled per-piece cost bound in chunk slots.
    :type target: int
    :param warms_tab: Per-cut proven warm chunks (None when unprovable).
    :type warms_tab: list
    :param cut_max: Largest chunk-aligned cut position.
    :type cut_max: int
    :return: (cuts, warms) or None when no valid split exists.
    :rtype: tuple
    """
    if num_chunks + piece_overhead <= target:
        return None
    cuts = []
    warms = []
    start = 0
    warm_in = 0
    while num_chunks - start + warm_in + piece_overhead > target:
        limit = start + target - piece_overhead - warm_in
        cut = min(limit, cut_max)
        found = None
        while cut > start and limit - cut <= MAX_WARM + 4:
            if warms_tab[cut] is not None:
                found = cut
                break
            cut -= 1
        if found is None:
            return None
        cuts.append(found)
        warms.append(warms_tab[found])
        warm_in = warms_tab[found]
        start = found
    if not cuts:
        return None
    return tuple(cuts), warms


def pack_bins(rows, num_sms):
    """LPT-pack piece rows onto at most one stream CTA per SM.

    :param rows: Piece-table rows from plan_pieces().
    :type rows: list
    :param num_sms: SM count bounding the CTA count.
    :type num_sms: int
    :return: (order, cta_vals) where order reindexes rows grouped by CTA and
        cta_vals is the flattened [row_start, row_count, total_chunks] table.
    :rtype: tuple

    .. code-block:: python

        order, cta_vals = pack_bins(rows, 148)
        rows = [rows[i] for i in order]
    """
    costs = [(r[1] - r[0] + K2_CHUNK - 1) // K2_CHUNK for r in rows]
    bins = min(num_sms, len(rows))
    heap = [(0, b) for b in range(bins)]
    heapq.heapify(heap)
    assign = [[] for _ in range(bins)]
    for i in sorted(range(len(rows)), key=lambda i: -costs[i]):
        load, b = heapq.heappop(heap)
        assign[b].append(i)
        heapq.heappush(heap, (load + costs[i], b))
    order = []
    cta_vals = []
    for b in range(bins):
        cta_vals.extend([len(order), len(assign[b]), sum(costs[i] for i in assign[b])])
        order.extend(assign[b])
    return order, cta_vals


def plan_pieces(offsets, num_heads, num_sms, warm_tables, use_init, store_final):
    """Choose per-chain piece splits that minimize modeled LPT makespan.

    :param offsets: Packed cu_seqlens boundaries as Python ints.
    :type offsets: list
    :param num_heads: Head count.
    :type num_heads: int
    :param num_sms: SM count used as the machine count.
    :type num_sms: int
    :param warm_tables: Per-sequence per-cut proven warm chunks.
    :type warm_tables: list
    :param use_init: Initial-state load flag.
    :type use_init: int
    :param store_final: Final-state store flag.
    :type store_final: int
    :return: (rows, has_split, windows) where rows are piece-table entries
        sorted by modeled cost and windows are (walk_start, out_start) proof
        windows.
    :rtype: tuple
    """
    seqs = []
    for seq_idx in range(len(offsets) - 1):
        bos, eos = offsets[seq_idx], offsets[seq_idx + 1]
        num_chunks = (eos - bos + K2_CHUNK - 1) // K2_CHUNK
        nfull = (eos - bos) // K2_CHUNK
        cut_max = min(nfull, num_chunks - 1)
        seqs.append((num_chunks, seq_idx, bos, eos, cut_max))
    seqs.sort(key=lambda item: -item[0])
    total_chains = len(seqs) * num_heads
    piece_overhead = (
        1
        if total_chains <= num_sms // 2 or total_chains >= 5 * num_sms
        else PIECE_OVERHEAD_CHUNKS
    )

    fill_counts = []
    for waves in range(1, (total_chains - 1) // num_sms + 1):
        fill = total_chains - waves * num_sms
        if 0 < fill < total_chains:
            fill_counts.append(fill)
    split_counts = sorted(
        {max(1, round(total_chains * f)) for f in SPLIT_FRACTIONS} | set(fill_counts)
    )

    candidates = [{}]
    for pieces in (2, 3, 4):
        for n_split in split_counts:
            for jitter in JITTERS:
                plan = {}
                remaining = n_split
                valid = True
                for num_chunks, seq_idx, _bos, _eos, cut_max in seqs:
                    if remaining <= 0:
                        break
                    k_here = min(remaining, num_heads)
                    warms_tab = warm_tables[seq_idx]
                    cuts = None
                    if warms_tab is not None and num_chunks >= 2 * pieces:
                        cuts = _balanced_cuts(num_chunks, pieces, jitter, cut_max)
                    if cuts is None:
                        valid = False
                        break
                    warms = [warms_tab[c] for c in cuts]
                    if any(w is None for w in warms):
                        valid = False
                        break
                    plan[seq_idx] = (k_here, cuts, warms)
                    remaining -= k_here
                if valid and plan:
                    candidates.append(plan)

    lower_area = sum(nc + piece_overhead for nc, *_ in seqs) * num_heads
    lower_bound_span = max(1, -(-lower_area // num_sms))
    for target in range(lower_bound_span, lower_bound_span + lower_bound_span // 4 + 8):
        plan = {}
        for num_chunks, seq_idx, _bos, _eos, cut_max in seqs:
            warms_tab = warm_tables[seq_idx]
            if warms_tab is None:
                continue
            result = _msweep_cuts(
                num_chunks, target, warms_tab, cut_max, piece_overhead
            )
            if result is not None:
                plan[seq_idx] = (num_heads, result[0], list(result[1]))
        if plan:
            candidates.append(plan)

    best = None
    trivial_makespan = None
    for plan in candidates:
        jobs = []
        area = 0
        for num_chunks, seq_idx, _bos, _eos, _cut_max in seqs:
            k_here, cuts, warms = plan.get(seq_idx, (0, (), ()))
            whole = num_chunks + piece_overhead
            if k_here:
                piece_jobs = []
                prev = 0
                for i, cut in enumerate(tuple(cuts) + (num_chunks,)):
                    warm = 0 if i == 0 else warms[i - 1]
                    piece_jobs.append(cut - prev + warm + piece_overhead)
                    prev = cut
                jobs.extend(piece_jobs * k_here)
                area += sum(piece_jobs) * k_here
            jobs.extend([whole] * (num_heads - k_here))
            area += whole * (num_heads - k_here)
        makespan = _lpt_makespan(jobs, num_sms)
        if trivial_makespan is None:
            trivial_makespan = makespan
        key = (makespan, area, len(jobs))
        if best is None or key < best[0]:
            best = (key, plan)
    plan = best[1]
    if plan and best[0][0] >= trivial_makespan * SPLIT_MARGIN:
        plan = {}

    rows = []
    windows = []
    for num_chunks, seq_idx, bos, eos, _cut_max in seqs:
        k_here, cuts, warms = plan.get(seq_idx, (0, (), ()))
        for head in range(num_heads):
            if head >= k_here:
                cost = num_chunks + piece_overhead
                rows.append(
                    (cost, [bos, eos, bos, seq_idx, use_init, store_final, head, 0])
                )
                continue
            prev = 0
            for i, cut in enumerate(tuple(cuts) + (num_chunks,)):
                start_tok = bos + prev * K2_CHUNK
                end_tok = eos if cut == num_chunks else bos + cut * K2_CHUNK
                warm = 0 if i == 0 else warms[i - 1]
                walk_tok = start_tok - warm * K2_CHUNK
                cost = (end_tok - walk_tok + K2_CHUNK - 1) // K2_CHUNK
                cost += piece_overhead
                rows.append(
                    (
                        cost,
                        [
                            walk_tok,
                            end_tok,
                            start_tok,
                            seq_idx,
                            use_init if i == 0 else 0,
                            store_final if cut == num_chunks else 0,
                            head,
                            0,
                        ],
                    )
                )
                if warm and head == 0:
                    windows.append((walk_tok, start_tok))
                prev = cut
    rows.sort(key=lambda item: -item[0])
    has_split = any(r[1][0] != r[1][2] for r in rows)
    return [r[1] for r in rows], has_split, windows
