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

"""Host planner for pkdz — dual-chain interleave (uniform varlen routes).

Design (session 49, profiles/iket_v145_chain_prep_maps_uniform_h64_20260815):
each CTA runs TWO independent chain-walks ("sides"); the chunk stream
alternates sides (stream position p -> side p & 1).  The 5-stage prep ring
and every per-stage barrier follow the stream unchanged; each side's
recurrence spans two stream slots, so the ~2.0-2.2us single-chain
dependency-latency loop overlaps across sides.  Compute pipelines two
stages (iteration i = chunk i front half + chunk i-1 requant/MMA4 tail);
the epilogue lags one slot with a single OFIN(stage(i-1)) wait.

This module only builds the dual schedule arrays.  Kernel plumbing lands
with pkdz.py; nothing routes here yet.

Schedule contract (v1, whole-chain pairing):
- sides 2g (A) and 2g+1 (B) of CTA g must carry EQUAL total chunk counts,
  so strict A,B,A,B alternation always lands on a live chunk;
- pieces are whole chains (no midstate handoffs) in v1; v2 will add
  16/8-chunk pieces to reach the ~55.4-chunk integer bound per side.
Arrays mirror main_pkdh's piece format with a doubled offset table:
  soff2[2G+1], schain[np], spt0[np], sptn[np], ssrc[np], sdst[np].
"""
import torch

C_TOK = 32


def dual_schedule(lens, H: int, nslots: int, dev: torch.device):
    """v1 whole-chain pairing for equal-length chains (uniform routes).

    Returns (soff2, schain, spt0, sptn, ssrc, sdst, G, nbuf) or None when
    the shape is not a supported uniform route (caller falls back to pkdw).
    """
    nseq = len(lens)
    if nseq < 2 or len(set(lens)) != 1:
        return None
    L = lens[0]
    chains = nseq * H
    # Whole chains per CTA must be even so the two sides split equally.
    G = min(nslots, chains // 2)
    if G <= 0:
        return None
    base = chains // G
    extra = chains - base * G
    # Counts per CTA in {base, base+1}; round every CTA to an even count
    # by pairing the odd ones (base odd or extra spill).  Constructive:
    # hand out chains two at a time.
    counts = [base + (1 if g < extra else 0) for g in range(G)]
    # make every count even, moving one chain from odd CTA i to odd CTA j
    odd = [g for g in range(G) if counts[g] & 1]
    for a, b in zip(odd[0::2], odd[1::2]):
        counts[a] += 1
        counts[b] -= 1
    if sum(counts) != chains or any(c & 1 for c in counts):
        return None
    # Chain ids: seq-major then head, matching the single-chain LPT order
    # closely enough for uniform (all chains identical).
    ids = [s * H + h for s in range(nseq) for h in range(H)]
    items_a, items_b = [], []
    pos = 0
    for g in range(G):
        c = counts[g]
        half = c // 2
        items_a.append(ids[pos:pos + half])
        items_b.append(ids[pos + half:pos + c])
        pos += c
    fc, ft0, ftn, fsrc, fdst, off2 = [], [], [], [], [], [0]
    for g in range(G):
        for side in (items_a[g], items_b[g]):
            for ch in side:
                fc.append(ch)
                ft0.append(0)
                ftn.append(L)
                fsrc.append(-1)
                fdst.append(-1)
            off2.append(len(fc))
    i32 = lambda x: torch.tensor(x, dtype=torch.int32, device=dev)
    return (i32(off2), i32(fc), i32(ft0), i32(ftn), i32(fsrc), i32(fdst),
            G, 0)


def validate_schedule(sched, lens, H):
    """CPU checks: coverage, equal sides, no overlaps."""
    soff2, schain, spt0, sptn, ssrc, sdst, G, nbuf = sched
    soff2 = soff2.tolist()
    schain = schain.tolist()
    sptn = sptn.tolist()
    L = lens[0]
    seen = {}
    for g in range(G):
        a0, a1, b1 = soff2[2 * g], soff2[2 * g + 1], soff2[2 * g + 2]
        ca = sum(sptn[i] // C_TOK for i in range(a0, a1))
        cb = sum(sptn[i] // C_TOK for i in range(a1, b1))
        assert ca == cb, f"CTA {g}: side chunks differ {ca} vs {cb}"
        for i in range(a0, b1):
            seen[schain[i]] = seen.get(schain[i], 0) + sptn[i]
    chains = len(lens) * H
    assert len(seen) == chains, f"coverage {len(seen)} != {chains}"
    assert all(v == L for v in seen.values()), "piece lengths broken"
    side_chunks = [sum(sptn[i] // C_TOK
                       for i in range(soff2[2 * g], soff2[2 * g + 1]))
                   for g in range(G)]
    return max(side_chunks), sum(side_chunks) / len(side_chunks)


if __name__ == "__main__":
    for nseq, H in ((8, 64), (8, 96)):
        lens = [1024] * nseq
        sched = dual_schedule(lens, H, 148, torch.device("cpu"))
        mx, avg = validate_schedule(sched, lens, H)
        ideal = nseq * H * (lens[0] // C_TOK) / (2 * sched[6])
        print(f"uniform H{H}: G={sched[6]} side-makespan={mx} "
              f"avg={avg:.1f} ideal={ideal:.1f} eff={ideal / mx:.2%}")
