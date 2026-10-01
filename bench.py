"""Speedup of a KDA forward kernel over FlashKDA on the Int21 workloads.

    uv run python bench.py cute
    uv run python bench.py tirx
    uv run python bench.py ptx
    uv run python bench.py path/to/kernel.py

The six timed workloads of the KDA-internal ``kda_forward`` task (Int21-AI/KDA-B200
``compare_cutlass_gb200.py``): H = 96 and 64, 8192 tokens each as one sequence, six
mixed-length sequences and eight 1024-token sequences. Inputs follow the task's
recipe (K3-realistic ``A_log`` / ``dt_bias``, Gaussian q/k/v/g/beta, fp32 initial
state), with a fixed seed per workload.

The baseline is FlashKDA's fused CUTLASS forward through FLA's ``chunk_kda``, with
the final state requested so it pays for the same state writeback. Correctness is
checked against FLA's Triton ``chunk_kda`` with the task's gate: an element fails
when its error exceeds both half of the reference tensor's RMS and 5% of its own
reference magnitude, and a tensor fails above relative L2 0.03.

Timing follows the task's protocol: FlashInfer's CUPTI timer, cold L2, the call
captured in a CUDA graph (a ``prepare``d launch that cannot be captured, because it
replays its own graph, is timed as issued), median of iterations, median of trials.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import statistics
import sys
from pathlib import Path

import torch

D = 128
ATOL_RMS, RTOL, MAX_REL_L2 = 0.5, 0.05, 0.03
MIXED = [1300, 547, 2048, 963, 271, 3063]
WORKLOADS = {
    "h96-fixed": (96, [8192]),
    "h96-mixed_varlen": (96, MIXED),
    "h96-uniform_varlen": (96, [1024] * 8),
    "h64-fixed": (64, [8192]),
    "h64-mixed_varlen": (64, MIXED),
    "h64-uniform_varlen": (64, [1024] * 8),
}
KERNELS = {"cute": "kda-cake-cute/kernel.py", "tirx": "kda-tirx/kernel.py", "ptx": "kda-cake-ptx/kernel.py"}


def make_inputs(heads, seq_lens, seed, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    T = sum(seq_lens)

    def randn(shape, scale, dtype=torch.bfloat16):
        return (torch.randn(shape, generator=gen, device=device) * scale).to(dtype)

    A_log = torch.log(
        torch.empty(heads, device=device).uniform_(1.0, 16.0, generator=gen)
    )
    dt = torch.exp(
        torch.rand(heads * D, generator=gen, device=device)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    ).clamp_(min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    q, k, v, g = (randn((1, T, heads, D), 0.5) for _ in range(4))
    beta = randn((1, T, heads), 0.5)
    initial_state = randn((len(seq_lens), heads, D, D), 0.25, torch.float32)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device
        )
    return [
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        1.0 / math.sqrt(D),
        initial_state,
        cu_seqlens,
    ]


def fla_chunk_kda(
    flash_kda, q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens
):
    os.environ["FLA_FLASH_KDA"] = "1" if flash_kda else "0"
    os.environ["FLA_TILELANG"] = "0"
    from fla.ops.kda import chunk_kda

    return chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        safe_gate=True,
        lower_bound=-5.0,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )


def load_kernel(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("kda_candidate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["kda_candidate"] = module
    spec.loader.exec_module(module)
    return module


def check(got, want):
    """The task's correctness gate on one tensor; returns (ok, worst ratio, rel L2)."""
    x, y = got.float(), want.float()
    err = (x - y).abs()
    ratio = torch.minimum(
        err / (ATOL_RMS * y.pow(2).mean().sqrt()), err / (RTOL * y.abs() + 1e-30)
    )
    rel_l2 = float(torch.linalg.vector_norm(x - y) / torch.linalg.vector_norm(y))
    worst = float(torch.nan_to_num(ratio, nan=float("inf")).max())
    return (
        worst <= 1.0 and rel_l2 <= MAX_REL_L2 and bool(torch.isfinite(x).all()),
        worst,
        rel_l2,
    )


def cupti_ms(fn, warmup, iters, use_cuda_graph):
    from flashinfer.testing import bench_gpu_time_with_cupti

    times = bench_gpu_time_with_cupti(
        fn=fn,
        dry_run_iters=warmup,
        repeat_iters=iters,
        cold_l2_cache=True,
        use_cuda_graph=use_cuda_graph,
    )
    return statistics.median(times)


def time_launch(launch, warmup, iters):
    try:
        return cupti_ms(launch, warmup, iters, use_cuda_graph=True)
    except Exception:  # noqa: BLE001 - replays its own CUDA graph: time it as issued
        torch.cuda.synchronize()
        return cupti_ms(launch, warmup, iters, use_cuda_graph=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "kernel",
        help="cute, tirx, ptx, or a path to a kernel.py exposing run (and optionally prepare)",
    )
    ap.add_argument(
        "--workloads", nargs="*", default=list(WORKLOADS), choices=list(WORKLOADS)
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    module = load_kernel(
        here / KERNELS[args.kernel] if args.kernel in KERNELS else args.kernel
    )
    prepare = getattr(module, "prepare", None)
    print(
        f"{torch.cuda.get_device_name()} | kernel {args.kernel} | "
        f"{'prepare/launch' if prepare else 'run'} | {args.trials} x {args.iters} iters"
    )
    print(
        f"{'workload':20s} {'FlashKDA ms':>11s} {'kernel ms':>9s} {'speedup':>8s}  correctness vs FLA"
    )

    speedups, all_ok = [], True
    for index, name in enumerate(args.workloads):
        heads, seq_lens = WORKLOADS[name]
        inputs = make_inputs(heads, seq_lens, args.seed * 1000 + index)
        with torch.no_grad():
            reference = fla_chunk_kda(False, *inputs)
            if prepare is not None:
                launch = prepare(*inputs)
            else:

                def launch(inputs=inputs):
                    return module.run(*inputs)

            result = launch()
            torch.cuda.synchronize()
            if torch.is_tensor(
                result
            ):  # returned the output alone: final_state missing
                result = (result,)
            verdicts = [check(got, want) for got, want in zip(result, reference)]
            ok = len(result) == 2 and all(v[0] for v in verdicts)
            all_ok &= ok

            def baseline(inputs=inputs):
                return fla_chunk_kda(True, *inputs)

            base_ms, cand_ms = [], []
            for _ in range(args.trials):
                base_ms.append(
                    cupti_ms(baseline, args.warmup, args.iters, use_cuda_graph=True)
                )
                cand_ms.append(time_launch(launch, args.warmup, args.iters))
        base, cand = statistics.median(base_ms), statistics.median(cand_ms)
        speedups.append(base / cand)
        detail = ", ".join(
            f"{t} worst {w:.2f} relL2 {r:.4f}"
            for t, (_, w, r) in zip(("out", "state"), verdicts)
        )
        if len(result) < 2:
            detail += ", no final_state"
        print(
            f"{name:20s} {base:11.4f} {cand:9.4f} {base / cand:7.3f}x  "
            f"{'PASS' if ok else 'FAIL'} ({detail})",
            flush=True,
        )
    geomean = math.exp(sum(map(math.log, speedups)) / len(speedups))
    print(
        f"{'geomean':20s} {'':11s} {'':9s} {geomean:7.3f}x  {'all pass' if all_ok else 'FAILURES'}"
    )


if __name__ == "__main__":
    main()
