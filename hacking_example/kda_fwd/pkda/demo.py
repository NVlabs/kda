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

"""Standalone launch demo for pkda.fwd — no harness required.

    python -m pkda.demo                      # fixed_h64 + varlen_mixed_h64
    python -m pkda.demo --case fixed_h96     # one harness case
    python -m pkda.demo --case all           # all 6 harness cases

Builds random inputs, runs the kernel, reports CUDA-event timing.
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

# the 6 harness cases: name -> (seq_lens, H)
CASES = {
    "fixed_h96": ((8192,), 96),
    "varlen_mixed_h96": ((1300, 547, 2048, 963, 271, 3063), 96),
    "varlen_uniform_h96": ((1024,) * 8, 96),
    "fixed_h64": ((8192,), 64),
    "varlen_mixed_h64": ((1300, 547, 2048, 963, 271, 3063), 64),
    "varlen_uniform_h64": ((1024,) * 8, 64),
}


def _inputs(seq_lens, H, dev, seed=0):
    D = 128
    torch.manual_seed(seed)
    total = sum(seq_lens)
    nseq = len(seq_lens)
    f = torch.nn.functional
    q = f.normalize(torch.randn(1, total, H, D, device=dev), p=2, dim=-1).to(torch.bfloat16)
    k = f.normalize(torch.randn(1, total, H, D, device=dev), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(1, total, H, D, dtype=torch.bfloat16, device=dev)
    g = torch.randn(1, total, H, D, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(1, total, H, dtype=torch.bfloat16, device=dev)
    a_log = torch.rand(H, dtype=torch.float32, device=dev)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=dev)
    state0 = torch.randn(nseq, H, D, D, dtype=torch.float32, device=dev)
    cu = None
    if nseq > 1:
        cu = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), 0)),
            dtype=torch.int64, device=dev)
    return q, k, v, g, beta, 1.0 / math.sqrt(D), a_log, dt_bias, state0, cu


def _run(name, seq_lens, H, iters=30):
    import pkda

    dev = torch.device("cuda")
    q, k, v, g, beta, scale, a_log, dt_bias, state0, cu = _inputs(seq_lens, H, dev)
    nseq = state0.shape[0]

    def call():
        out = torch.empty_like(q)
        fs = torch.empty(nseq, H, 128, 128, dtype=torch.float32, device=dev)
        pkda.fwd(q, k, v, g, beta, scale, out, a_log, dt_bias, -5.0,
                 initial_state=state0, final_state=fs, cu_seqlens=cu)
        return out, fs

    out, fs = call()  # first call JIT-compiles
    torch.cuda.synchronize()
    assert out.isfinite().all() and fs.isfinite().all()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s.record()
        call()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    mean = sum(times) / len(times)
    print(f"[{name}] T_total={sum(seq_lens)} H={H} nseq={len(seq_lens)}: "
          f"gpu_mean {mean:.4f} ms (min {min(times):.4f})  "
          f"out[0,0,0,:4]={out[0, 0, 0, :4].float().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(CASES) + ["all"], default=None,
                        help="harness case to run (default: fixed_h64 + varlen_mixed_h64)")
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    # CUDA must initialize before this package's JIT paths run
    torch.cuda.init()
    torch.cuda.synchronize()
    sys.argv = sys.argv[:1]  # cutlass's DSL parses argv on import
    if args.case is None:
        names = ["fixed_h64", "varlen_mixed_h64"]
    elif args.case == "all":
        names = list(CASES)
    else:
        names = [args.case]
    for name in names:
        seq_lens, H = CASES[name]
        _run(name, seq_lens, H, iters=args.iters)
    print("pkda standalone launch OK")


if __name__ == "__main__":
    main()
