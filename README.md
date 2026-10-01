# KDA forward kernels for NVIDIA Blackwell

Three implementations of the Kimi Delta Attention (KDA) forward pass
(chunked, per-channel-gated delta rule; K3 gate `-5 * sigmoid(exp(A_log) * (g + dt_bias))`,
in-kernel q/k L2 normalization, `beta = sigmoid(beta)`, fp32 V-first recurrent state),
all following the KDA-internal `kda_forward` task ABI:

    run(q, k, v, g, beta, A_log, dt_bias, scale, initial_state, cu_seqlens) -> (output, final_state)

| directory | language | entry point | source |
|---|---|---|---|
| `cute/` | CuTe DSL (Python) | `cute/kernel.py` (`run`) | humanfia/kda-for-kda `yahui-2.88x-cute` @ `dde00d0`, with three fixes (see below) |
| `tirx/` | TIRx (TVM) | `tirx/kernel.py` (`prepare` / `run`) | humanfia/kda-tirx `20260922-b300-tune` @ `def3dfc` (judge submission `2b64c874`), with its size limits lifted (see below) |
| `ptx/` | static PTX (sm_103a) + TVM FFI host shims | `ptx/kernel.py` (`prepare` / `run`) | humanfia/kda-for-kda `yahui-2.89x-ptx` @ `82a6a79`, with three fixes (see below) |

Inputs: bf16 `q/k/v/g [1, T, H, 128]`, bf16 beta logits `[1, T, H]`, fp32 `A_log [H]`,
fp32 `dt_bias [H*128]`, fp32 `initial_state [N, H, 128, 128]`, int64 `cu_seqlens [N+1]` or `None`.
Outputs: bf16 `output [1, T, H, 128]`, fp32 `final_state [N, H, 128, 128]`.

## Results (NVIDIA B300; CuTe and TIRx 2026-09-24, PTX 2026-09-28)

Measured with the KDA-internal judge (`bench_kda_forward_standalone.py`): speedup over
FlashKDA 7afb9f4's fused CUTLASS forward, executed live on every workload; 8192 total tokens.

| workload | CuTe | TIRx | PTX |
|---|---:|---:|---:|
| H96 fixed (1 x 8192) | 2.760x | 3.102x | 2.770x |
| H96 mixed varlen (6 seqs) | 3.000x | 3.038x | 3.204x |
| H96 uniform varlen (8 x 1024) | 2.583x | 2.463x | 2.657x |
| H64 fixed (1 x 8192) | 2.513x | 3.601x | 2.800x |
| H64 mixed varlen (6 seqs) | 3.421x | 3.265x | 3.495x |
| H64 uniform varlen (8 x 1024) | 2.542x | 2.412x | 2.653x |
| **geomean** | **2.786x** | **2.950x** | **2.914x** |
| judge correctness (workloads, stress / exact probes, holdout, 5 real probes) | 24/24 | 24/24 | 24/24 |

PTX was judged on 2026-09-28. The judge's upload whitelist has no `.ptx`, so that submission
embeds each PTX file as a string in a `.py` file; the code is otherwise the same as `ptx/`.

### Accuracy on a real long prefill

![Output and final-state relative RMSE of FlashKDA, CuTe, TIRx and PTX on a real Kimi-Linear prefill](figures/real_workload_accuracy.png)

A MATH-500 prompt prefilled through Kimi-Linear-48B-A3B-Instruct (8183 tokens; the 32 heads
of KDA layers 00, 14 and 25 stacked into 96), scored against an fp64 token-by-token
recurrence with FlashKDA's own test metric, relative RMSE. FlashKDA's error grows with
context length and its final state ends at 3.98% off; the three kernels here stay flat and
keep the final state at 0.23% (CuTe), 0.31% (TIRx) and 0.19% (PTX).

To reproduce it, run (after [Install](#install)):

```bash
uv run --with matplotlib --with safetensors --with huggingface_hub \
    python scripts/real_workload_error_plots.py
```

The script downloads the three captured layers (about 1 GB) of the `math500-multi-8192`
sample from the public Hugging Face dataset
[`humanfia-lab/kda-datasets`](https://huggingface.co/datasets/humanfia-lab/kda-datasets)
(folder `kda-forward/`, which documents how the captures were made) into `data/`, runs the
fp64 reference and the four kernels, and writes `figures/real_workload_error.json` and
`figures/real_workload_accuracy.png`. `--data DIR` changes the download directory, and
`--plot-only` redraws the figure from the JSON without a GPU. To fetch the data by hand:

```bash
hf download humanfia-lab/kda-datasets --repo-type dataset --local-dir data \
    kda-forward/math500-multi-8192/layer00.safetensors \
    kda-forward/math500-multi-8192/layer14.safetensors \
    kda-forward/math500-multi-8192/layer25.safetensors
```

## CuTe: changes from the source branch

- The per-chunk cumulative log-decay table is always kept in FP32 (the FP16 storage path
  `GCS_FP16_` / `GCS_PACKCVT_` and the unused M64 kernel `pkdx.py` are removed).
- The q/k normalization and decay decoration are computed in FP32 and rounded to BF16 once
  on every route (the packed-BF16 variant used when a sequence length is not a multiple of
  32 is removed).
- The split-sequence state handoff clears its flags on the launch stream before every launch,
  so eager calls, CUDA graph captures and replays can be mixed in any order.

## TIRx: provenance

jinhongyii's TIRx gist 4c50fafe (a fused persistent packed-varlen kernel plus an embedded
two-kernel split-concurrent route for single sequences, derived from the tirx-kernels
`agent_evolved` KDA forward kernels), tuned in place for B300 by AI agents (untouched gist:
2.544x). `kernel.py` wraps the gist's `setup(data, B, T, H) -> run` protocol as
`prepare(...) -> launch`; `prepare` compiles, builds the host work list from `cu_seqlens`,
and records a private CUDA graph, and `launch` replays it on the current buffer contents.

Changes from the source branch (work-list handling only; the numerics and the timed-workload
speed are unchanged):

- The packed-varlen route no longer refuses large batches. Per-CTA work lists longer than
  the 160-entry SMEM table are read from global memory, the sequence-count cap is removed,
  and token offsets are packed into 21 bits instead of 16.

### TIRx backward

[`kda-tirx-bwd/kda_backward_packed.py`](kda-tirx-bwd/kda_backward_packed.py) is imported from
the [TIRx-kernels packed KDA backward kernel](https://github.com/mlc-ai/TIRx-kernels/blob/8b6ed130a330e522a22b62eed2fa15ed487acec0/tirx_kernels/kda/kda_backward_packed.py).
It targets B200 (`sm_100a`) and supports B=1, K=V=128, chunk size 64, grouped value heads
(`Hv % Hqk == 0`) and packed sequences with partial trailing chunks.

The offline builder's default CTA count now follows the detected SM count, matching
runtime setup and benchmark preparation. On a 148-SM B200, the fused path builds its
schedule for up to 148 CTAs. The upstream 152-CTA / 768-chain tuned schedule is retained
and selected only when both counts match.

The entry point is `setup(data, B, T, Hqk) -> launch`. It takes FLA's saved backward inputs:
L2-normalized `q/k`, `v`, activated `beta`, saved `Aqk/Akk`, chunk-local cumulative base-2
log gates `g`, a **K-first** `initial_state`, upstream `do/dht`, `scale`, `chunk_size`, and
`cu_seqlens`. The caller supplies contiguous output buffers `dq`, `dk`, `dv`, `db`, `dg`
and `dh0` in `data`; `launch()` writes those buffers. `setup` compiles, allocates scratch,
and runs once before returning. This uses a different input/state contract from the
forward `tirx/kernel.py`; `bench.py` measures the forward kernels only.

For a small correctness check against FLA, using the existing project dependencies:

```bash
PYTHONPATH=kda-tirx-bwd uv run python - <<'PY'
import kda_backward_packed as bwd

bwd.run_test(num_qk_heads=8, num_v_heads=8, seq_lens=(128, 128))
bwd.run_test(num_qk_heads=2, num_v_heads=4, seq_lens=(129, 79))
PY
```

The module also retains upstream's `prepare_data`, `CONFIGS`, and `run_bench` helpers.
Copyright (c) 2026 TIRx authors. This file is licensed under
[Apache-2.0](kda-tirx-bwd/LICENSE-Apache-2.0).

## PTX: provenance and changes

The static PTX forward implementation of kda-for-kda `yahui-2.89x-ptx`. Its latest commit
`82a6a79` fixes a flush-to-zero precision loss in slow-decay channels of the first release
`d3a08ef`.
- Four retained PTX kernels (`kda_ptx/shims/*.ptx`, `.target sm_103a`) are launched
  through TVM FFI host shims (`kda_ptx/shims/*.cc`).
- A Python host scheduler (`kda_ptx/scheduler.py`) packs every (sequence, head) recurrence
  onto the SMs. It splits long ones across CTAs with an in-kernel FP32 state handoff.
- The scheduler has hand-picked plans for the six Int21 layouts. Shifting the mixed layout
  by one token gives the same speed (3.22x / 3.49x vs 3.21x / 3.50x on `bench.py`'s timer).
- Two cases run an exact token-by-token Triton recurrence (`kda_ptx/recurrent_kernel.py`)
  whose initial and final states pass through BF16 staging:
  - head counts other than 64 and 96;
  - initial states above 2^12 in magnitude.

Changes from the source branch:

- The PTX is `.version 9.2`, which drivers older than CUDA 13.2 (such as 580) cannot JIT.
  `kda_ptx/static_runtime.py` instead assembles each file once and embeds the cubin:
  - it uses ptxas >= 13.2, from the `nvidia-cuda-nvcc` wheel or `KDA_PTXAS`;
  - the cubins are cached in `~/.cache/kda-ptx` (set `KDA_PTX_CACHE_DIR` to change it).
- The host shims kept one process-wide cache of TMA descriptors and uploaded a new one for
  every tensor address they saw. This caused two failures:
  - it never freed slots, so a process ran out of its 4096 slots after about 700 calls with
    varying shapes;
  - how much host-to-device work a call did depended on allocator address reuse, which the
    judge's per-call CUPTI check rejects.

  Each prepared launch now owns a descriptor table, which `prepare` encodes and uploads once
  (outside any capture); a launch only passes pointers into it. The kernels acquire each
  descriptor with `fence.proxy.tensormap` before use, so the PTX is unchanged.
- `kernel.py` exports `prepare` as well as `run`.
  - The source entry exports only `run`, which plans on every call and cannot be captured in
    a CUDA graph. The judge rejects it, and timed as issued it runs at 0.25x.
- Packaging only:
  - the source's `impl/` package is `kda_ptx/` here, so it loads in the same process as
    CuTe's `impl/`;
  - its compatibility modules (`impl/kernel.py`, `impl/forward.py`) and benchmark script are
    left out.

## Environment

All three kernels, the FlashKDA baseline and the benchmark share one uv environment, pinned in
`pyproject.toml` / `uv.lock`:

| component | version | used by |
|---|---|---|
| Python | 3.12 | all |
| torch | 2.12.1+cu130 | all |
| nvidia-cutlass-dsl (CuTe DSL) | 4.7.0 | CuTe kernel |
| apache-tvm (with TIRx), apache-tvm-ffi, tirx-kernels | git ed5e2fed3, be35ec1, 65d9a075 | TIRx kernel |
| apache-tvm-ffi, nvidia-cuda-nvcc (ptxas), triton (with torch) | git be35ec1, 13.2.86, 3.7.1 | PTX kernel |
| flash-kda (FlashKDA) | git 7afb9f4 | baseline |
| fla-core | 0.5.2 | baseline wrapper, correctness reference |
| flashinfer-python, cupti-python | 0.6.13, 13.0.1 | CUPTI timer |

TVM and FlashKDA are compiled from source during `uv sync`; everything else installs as
wheels.

### Prerequisites

- An NVIDIA Blackwell GPU (sm_100a / sm_103a) with a driver for CUDA 13. The PTX kernel
  runs on B300 (sm_103a) only.
- The CUDA 13 toolkit (`nvcc`); FlashKDA's CUTLASS extension is built with it.
- A C++17 compiler, CMake >= 3.18, and the LLVM 18 development files TVM needs for host
  code generation.
- [uv](https://docs.astral.sh/uv/) >= 0.8.

On Ubuntu 24.04:

```bash
sudo apt-get install build-essential cmake llvm-18-dev libxml2-dev zlib1g-dev libzstd-dev
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
```

### Install

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
uv sync
```

The first `uv sync` compiles TVM and FlashKDA from source (30-60 minutes); later syncs reuse uv's
cache. Notes:

- FlashKDA builds for the GPU it sees. On a build host without a GPU, set
  `FLASH_KDA_CUDA_ARCHS` (for example `103a` for B300, `100a` for B200) before `uv sync`.
- TVM finds LLVM through `llvm-config-18` (set in `[tool.uv.config-settings-package]` in
  `pyproject.toml`). If your LLVM 18 `llvm-config` has another name or path, change it
  there and run `uv sync --reinstall-package apache-tvm`.

### Check

```bash
uv run python -c "import cutlass, tvm, tirx_kernels.tirx_lite, flash_kda, fla; print('ok', cutlass.__version__, tvm.__version__)"
```

## Benchmark

```bash
uv run python bench.py cute    # or: tirx, ptx, or a path to another kernel.py
```

`bench.py` runs the six timed workloads of the KDA-internal `kda_forward` task (the Int21
set above), checks both outputs against FLA's Triton `chunk_kda` with the task's
tolerance, and reports each workload's FlashKDA and kernel times and the geomean speedup.
It follows the task's timing protocol (CUPTI, cold L2, CUDA graph, median of 30 iterations
x 3 trials), with the task's input distributions and fixed seeds. On the B300 above it
reports 2.799x (CuTe), 2.963x (TIRx) and 2.927x (PTX), within 0.5% of the judge's geomeans.

A kernel passed by path must expose `run(...)` with the signature above and may expose
`prepare(...) -> launch`, which is then planned once per workload and only `launch()` is
timed.

## Supported inputs

Head size 128, bf16 activations and an fp32 state; tested with H = 32, 64 and 96, up to
170001 tokens in one sequence and 300 packed sequences. TIRx requires `H % 8 == 0` and
`T * H * 128 < 2^31`.
PTX needs a B300 (sm_103a) and was tested up to 16384 tokens in one sequence and 300 packed
sequences. It runs H = 64 and 96 on its PTX kernels; other head counts
(such as H = 32) use the exact token-by-token recurrence, which is 12-20x slower than
FlashKDA.

## Hacking example

[`hacking_example/`](hacking_example/) keeps a disqualified forward kernel that a
multi-agent optimization run produced against a loose verifier: it claimed 3.74x over
FlashKDA by replacing the q/k norms with a distribution constant, dropping initial-state
channels, skipping the cross-CTA state handoff and the intra-chunk solve. Its README
documents each cheat and what a verifier needs to catch them. Do not use it.

## License

MIT, see [LICENSE](LICENSE), except `kda-tirx-bwd/kda_backward_packed.py`, which is
licensed under [Apache-2.0](kda-tirx-bwd/LICENSE-Apache-2.0).
