"""FlashKDA vs our CuTe, TIRx and PTX kernels on a real Kimi-Linear prefill workload.

    uv run --with matplotlib --with safetensors --with huggingface_hub python scripts/real_workload_error_plots.py
    uv run --with matplotlib python scripts/real_workload_error_plots.py --plot-only

The workload is a real prefill of Kimi-Linear-48B-A3B-Instruct on a MATH-500 prompt
(8183 tokens), captured at the input of the model's KDA layers and published as
humanfia-lab/kda-datasets on Hugging Face (dataset folder ``kda-forward/``). The three
layers used here (about 1 GB) are downloaded to ``--data`` on first use. The heads of
layers 00, 14 and 25 (32 each) are stacked into one H = 96 call.

Every kernel is scored against an fp64 token-by-token recurrence of the operator with
FlashKDA's test metric (tests/test_fwd.py: relative RMS error ``rms(gold - x) / rms(gold)``).
figures/real_workload_accuracy.png shows the output error against context length (per
64-token window, smoothed over 8 windows; the output is causal, so this is the error at
that context length) and the final-state error. The numbers are saved to
figures/real_workload_error.json; ``--plot-only`` redraws the figure from it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT))

HF_REPO = "humanfia-lab/kda-datasets"
SAMPLE = "math500-multi-8192"
LAYERS = (0, 14, 25)
D = 128
EPS = 1e-6
WINDOW, SMOOTH = 64, 8

KERNELS = [("FlashKDA", "#2a78d6"), ("Ours (CuTe)", "#e8702a"), ("Ours (TIRx)", "#1baf7a"), ("Ours (PTX)", "#eda100")]


def load_kernel(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def download(data_dir):
    """Fetch the sample's layers from Hugging Face (skipped when already present)."""
    from huggingface_hub import snapshot_download

    files = [f"kda-forward/{SAMPLE}/layer{i:02d}.safetensors" for i in LAYERS]
    snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=files, local_dir=data_dir)
    return Path(data_dir) / "kda-forward" / SAMPLE


def task_inputs(sample_dir):
    """The captured layers in the kda_forward ABI (see the dataset's README): heads of the
    layers stacked, A_log = dt_bias = 0, g = logit(-decay / 5), beta logits = logit(beta)."""
    import torch
    from safetensors.torch import load_file

    layers = [load_file(str(sample_dir / f"layer{i:02d}.safetensors"), device="cuda") for i in LAYERS]

    def cat(name):
        return torch.cat([layer[name] for layer in layers], dim=2)

    q, k, v = cat("q"), cat("k"), cat("v")
    decay, beta = cat("g").float(), cat("beta").float()
    H = q.shape[2]
    g = torch.logit((-decay / 5.0).clamp(EPS, 1 - EPS)).to(torch.bfloat16)
    beta_logit = torch.logit(beta.clamp(EPS, 1 - EPS)).to(torch.bfloat16)
    A_log = torch.zeros(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(H * D, dtype=torch.float32, device="cuda")
    initial_state = torch.zeros(1, H, D, D, dtype=torch.float32, device="cuda")
    return [q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(), beta_logit.contiguous(),
            A_log, dt_bias, 1.0 / math.sqrt(D), initial_state, None]


def recurrence_forward(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens):
    """The operator evaluated token by token in fp64 (the reference)."""
    import torch

    H = A_log.shape[0]
    q, k = q[0].double(), k[0].double()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    x = A_log.double().exp()[None, :, None] * (g[0].double() + dt_bias.double().view(1, H, D))
    decay = torch.exp(-5.0 * torch.sigmoid(x))
    weight = beta[0].double().sigmoid()
    output = torch.empty(v.shape[1:], dtype=torch.float64, device=v.device)
    final_state = torch.empty(initial_state.shape, dtype=torch.float64, device=v.device)
    bounds = [0, v.shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    for seq, (start, end) in enumerate(pairwise(bounds)):
        state = initial_state[seq].double()  # [H, V, K]
        for t in range(start, end):
            state = state * decay[t, :, None, :]
            delta = (v[0, t].double() - (state * k[t, :, None, :]).sum(-1)) * weight[t, :, None]
            state = state + delta[:, :, None] * k[t, :, None, :]
            output[t] = (state * q[t, :, None, :]).sum(-1) * scale
        final_state[seq] = state
    return output[None], final_state


def err_ratio(gold, x):
    """FlashKDA's metric: rms(gold - x) / rms(gold)."""
    return float((gold.double() - x.double()).square().mean().sqrt() / (gold.double().square().mean().sqrt() + 1e-8))


def windowed(gold, x):
    """err_ratio per WINDOW-token window along the sequence."""
    n = gold.shape[1] // WINDOW
    g = gold[:, : n * WINDOW].double().reshape(n, -1)
    d = (gold[:, : n * WINDOW].double() - x[:, : n * WINDOW].double()).reshape(n, -1)
    return (d.square().mean(1).sqrt() / (g.square().mean(1).sqrt() + 1e-8)).tolist()


def compute(data_dir):
    import torch
    from bench import fla_chunk_kda

    args = task_inputs(download(data_dir))
    with torch.no_grad():
        gold, gold_ht = recurrence_forward(*args)

    def fresh():
        return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]

    cute = load_kernel("cute_kernel", ROOT / "kda-cake-cute/kernel.py")
    tirx = load_kernel("tirx_kernel", ROOT / "kda-tirx/kernel.py")
    ptx = load_kernel("ptx_kernel", ROOT / "kda-cake-ptx/kernel.py")
    fns = {"FlashKDA": lambda *a: fla_chunk_kda(True, *a), "Ours (CuTe)": cute.run, "Ours (TIRx)": tirx.run,
           "Ours (PTX)": ptx.run}
    T, H = args[0].shape[1], args[0].shape[2]
    row = {"case": f"MATH-500, 1 seq x {T}", "sample": SAMPLE, "layers": list(LAYERS), "T": T, "H": H, "kernels": {}}
    for label, fn in fns.items():
        with torch.no_grad():
            o, ht = fn(*fresh())
        torch.cuda.synchronize()
        row["kernels"][label] = {"output": err_ratio(gold, o), "final_state": err_ratio(gold_ht, ht),
                                 "windowed_output": windowed(gold, o)}
        print(f"{label:12s} output {row['kernels'][label]['output']:.3e} "
              f"final_state {row['kernels'][label]['final_state']:.3e}", flush=True)
    (FIGURES / "real_workload_error.json").write_text(json.dumps(row, indent=1))
    return row


def smooth(xs, w):
    half = w // 2
    return [sum(xs[max(0, i - half): i + half + 1]) / len(xs[max(0, i - half): i + half + 1]) for i in range(len(xs))]


def plot(row):
    """One figure: output error vs context length and final-state error."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 15})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 5.8), width_ratios=[2, 1], layout="constrained")

    for label, color in KERNELS:
        ys = smooth(row["kernels"][label]["windowed_output"], SMOOTH)
        xs = [(i + 1) * WINDOW for i in range(len(ys))]
        ax.plot(xs, [100 * y for y in ys], color=color, lw=3, label=label)
    ax.set_xlabel("context length (tokens)")
    ax.set_ylabel("output relative RMSE (%)")
    ax.set_title("Output: relative RMSE vs context length")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=16, frameon=False, loc="upper left")

    values = [100 * row["kernels"][label]["final_state"] for label, _ in KERNELS]
    bars = bx.bar([label.replace(" (", "\n(") for label, _ in KERNELS], values,
                  color=[c for _, c in KERNELS], width=0.6)
    bx.bar_label(bars, fmt="%.2f%%", padding=3)
    bx.set_ylabel("final-state relative RMSE (%)")
    bx.set_title(f"Final state after {row['T']} tokens")
    bx.set_ylim(0, max(values) * 1.15)

    for a in (ax, bx):
        a.grid(axis="y", alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Kimi-Linear-48B prefill of a MATH-500 prompt ({row['T']} tokens, {row['H']} heads)\n"
                 "relative RMSE = rms(x - x_fp64) / rms(x_fp64), FlashKDA's test metric; lower is better",
                 fontsize=16)
    fig.savefig(FIGURES / "real_workload_accuracy.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true", help="redraw from figures/real_workload_error.json")
    ap.add_argument("--data", default=str(ROOT / "data"), help="download directory for the dataset")
    args = ap.parse_args()
    row = json.loads((FIGURES / "real_workload_error.json").read_text()) if args.plot_only else compute(args.data)
    plot(row)


if __name__ == "__main__":
    main()
