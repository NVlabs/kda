# Request: 3D real-to-complex FFT with packed Hermitian half-spectrum output (fp64)

Forward half of a forward/backward 3D FFT pair used at every iteration of
spectral scientific-computing applications: pseudo-spectral PDE solvers
(fluid dynamics, wave propagation, phase-field modeling), FFT-based
convolution/denoising pipelines, and tomographic reconstruction. These codes
transform one real-space grid per time step from real space to a **packed
Hermitian half-spectrum**, operate in Fourier space, and transform back. The
forward transform (`fft3d_r2c_halfspec_f64`) is the request here; the inverse
(`fft3d_c2r_halfspec_f64`) is submitted alongside as a companion request.

## Why this shape of FFT is worth optimizing

- **Grid sizes are modest and latency-bound.** Production spectral grids are
  typically 24..256 points per axis, batch size 1, and the transform is called
  twice per time step on the critical path. At these sizes, per-kernel launch
  overhead, plan-internal staging copies, and generic-stride kernels leave
  measurable headroom that large-batch deep-learning FFT benchmarks never see.
- **Anisotropic grids are the norm.** Boxes like 80×80×112 or 96×96×144 (long
  axis in the transform-fastest dimension) are common; the half-spectrum
  layout makes the fastest dimension the one that is halved, and it is the
  strided edge/edge-plane cases where generic implementations lose efficiency.
- **A tight output layout is required by the consumer.** Downstream
  Fourier-space kernels index the packed half-spectrum directly with an x-row
  stride of exactly `nx//2+1` (interleaved re/im, **no even-row padding**).
  Any implementation must produce/consume this exact layout in place; a
  library that pads odd-length rows or requires its own workspace layout adds
  a repack pass that can dominate at these sizes.

## Contract and Baseline

- Input: contiguous `float64` tensor `input` with shape `[nz, ny, nx]`; `nx`
  is the fastest-varying dimension (C-contiguous). Inputs are not modified.
- Output: new contiguous `float64` tensor `[nz, ny, nf, 2]` with
  `nf = nx//2 + 1`, interleaved `(re, im)`, x-row stride exactly `nf` with no
  padding — i.e. bit-compatible with `torch.view_as_real(torch.fft.rfftn(x))`
  on CUDA (cuFFT D2Z out-of-place).
- Normalization: **unnormalized forward** (`norm="backward"`); the round trip
  forward→inverse scales by `N = nz*ny*nx`. Applications fold physical
  scaling into their own Fourier-space kernels.
- Reference: float64 internal arithmetic in `definition.json`, cast to
  `float64` on output, used for correctness.
- Baseline: cuFFT reached through `torch.fft.rfftn` (PyTorch ≥ 2.1 with CUDA
  12.x, which dispatches to cuFFT `CUFFT_D2Z` out-of-place). Source and
  version recorded below; see `baseline.py`.
- Hardware: NVIDIA B200 or B300. Correctness tolerances: `rtol = 1e-9`,
  `atol = 1e-6` relative to the float64 reference. Both reference and
  baseline evaluate in float64, and the baseline (cuFFT D2Z) is **bit-exact**
  against the reference on every workload (measured max-abs 0.0). The
  tolerances exist for implementations using different summation orders,
  which typically deviate from cuFFT by ~1e-13 relative — roughly 4 orders
  of margin — while still rejecting single-precision shortcuts disguised as
  float64 output.
- This is the double-precision twin of
  [`jasvixban-fft3d-r2c`](../jasvixban-fft3d-r2c/README.md): identical
  contract, workloads, and deployment requirements; only the dtype pair
  differs.

## Workloads

Eight workloads in `workloads.jsonl`: cubic grids 32³, 64³, 96³, 128³, 192³,
256³ plus anisotropic 80×80×112 and 96×96×144 (nz×ny×nx, x fastest). These
mirror the grid-size buckets spectral applications actually run; axes are
FFT-friendly composites of small factors (2/3/5/7), which is the regime where
a hand-tuned implementation can beat a general plan chooser.

## Input data realism (optional)

Evaluation-protocol inputs are uniform random (white spectrum). Real
structured-grid application inputs are not: energy concentrates in specific
wavenumber bands and the high-k tail sits near the representation floor,
which stresses cancellation and dynamic range differently. Implementers who
want realistic input statistics can use `gen_surrogate.py` in this directory
with the two reference profiles in `spectra/`, measured from production runs
of a structured-grid spectral application — radially averaged |X(k)|² only,
one number per wavenumber shell; no spatial, phase, or time information is
published. Generate from this request directory (requires only numpy; keep
the generated `.npy` files out of the repository):

```bash
# --verify reports per-shell agreement of the generated field
python3 gen_surrogate.py spectra/r2c_a.csv \
    --shape 64 64 64  --std 0.604 --seed 1 --out r2c_in_64.npy --verify --verify
python3 gen_surrogate.py spectra/r2c_b.csv \
    --shape 144 96 96 --std 0.600 --seed 2 --out r2c_in_144.npy --verify
```

Surrogate fields reproduce the reference spectrum to ~8% median relative
error per shell (single-realization sampling noise) at matched amplitude
scale, and have cross-correlation < 0.002 with any source-data frame.
The generator emits float32; load with `np.load(...).astype(np.float64)`
for this request.

## Evaluate with FlashInfer Bench

Use Python 3.12, a CUDA-enabled PyTorch installation, and a B200/B300-compatible
NVIDIA driver and CUDA toolkit:

```bash
python -m pip install torch==2.11.0 flashinfer-python==0.6.18.post1 flashinfer-bench==0.1.2
```

From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 REQUEST_DIR=requests/jasvixban-fft3d-r2c python -c '
import os
from pathlib import Path
import torch
from flashinfer_bench import Benchmark, BenchmarkConfig
from flashinfer_bench.data import BuildSpec, Definition, Solution, SourceFile, Trace, TraceSet

hardware = torch.cuda.get_device_name(0)
assert any(model in hardware.split() for model in ("B200", "B300")), hardware
request = Path(os.environ["REQUEST_DIR"])
definition = Definition.model_validate_json((request / "definition.json").read_text())
workloads = [Trace.model_validate_json(line) for line in
             (request / "workloads.jsonl").read_text().splitlines() if line.strip()]
assert workloads and all(t.definition == definition.name for t in workloads)
baseline = Solution(
    name=f"{definition.name}_baseline", definition=definition.name, author="baseline",
    spec=BuildSpec(language="python", target_hardware=["B200", "B300"],
                   entry_point="baseline.py::run", destination_passing_style=False),
    sources=[SourceFile(path="baseline.py", content=(request / "baseline.py").read_text())],
)
dataset = TraceSet(definitions={definition.name: definition},
                   workloads={definition.name: workloads},
                   solutions={definition.name: [baseline]})
benchmark = Benchmark(dataset, BenchmarkConfig(
    warmup_runs=10, iterations=100, num_trials=3, rtol=1e-9, atol=1e-6))
try:
    results = benchmark.run_all(dump_traces=False)
finally:
    benchmark.close()
traces = results.traces.get(definition.name, [])
for trace in traces:
    print(trace.model_dump_json())
assert len(traces) == len(workloads), "Missing evaluation results"
assert all(t.is_successful() for t in traces), "Correctness or execution failed"
'
```

## Deployment contract (beyond the benchmark)

The consuming application reuses fixed device buffers and schedules on CUDA
streams; a delivered kernel must honor all of the following to be pluggable
(within a small adapter layer):

1. Read/write **caller-owned** device buffers in place; no internal copies to
   a private layout (a repack defeats the purpose of the tight layout).
2. Pure stream semantics: enqueue work on a supplied stream and return; no
   internal device-wide synchronization, and any autotuning/JIT/plan
   selection must happen once at plan-creation time, never mid-call.
3. Handle the non-power-of-two and anisotropic sizes in the workload set as
   first-class cases, not fallbacks.

## Validation results

Baseline (cuFFT via `torch.fft.rfftn`, torch 2.14.0+cu130), random uniform
inputs in [-1, 1], CUDA-event medians over 200 iterations, correctness vs.
the float64 reference at `rtol=1e-3 / atol=1e-2`:

| grid (nz×ny×nx) | baseline latency (ms) | correctness |
| --- | --- | --- |
| 32×32×32 | 0.0318 | PASS (rel-L2 2.5e-07) |
| 64×64×64 | 0.0308 | PASS (rel-L2 2.3e-07) |
| 96×96×96 | 0.0315 | PASS (2.4e-07) |
| 128×128×128 | 0.0342 | PASS (2.4e-07) |
| 192×192×192 | 0.1003 | PASS (2.8e-07) |
| 256×256×256 | 0.3303 | PASS (2.8e-07) |
| 80×80×112 | 0.0320 | PASS (2.1e-07) |
| 96×96×144 | 0.0325 | PASS (2.4e-07) |

_Measured on NVIDIA GeForce RTX 5090 (sm_120) as a pre-submission smoke
test; grids up to 128³ are dispatch-overhead-dominated (~30 µs floor), so
kernel-level comparisons should focus on 192³/256³. Results on the target
B200/B300 will be appended here before PR review._

## License

This wrapper and the embedded reference are Apache-2.0 code; this
documentation is Creative Commons Attribution 4.0 under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The
baseline calls cuFFT shipped with the CUDA toolkit (NVIDIA proprietary
library, used via PyTorch's BSD-licensed `torch.fft` interface); no cuFFT
source is bundled.
