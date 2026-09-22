# Request: Inverse 3D FFT from packed Hermitian half-spectrum (fp64)

Backward half of the forward/backward 3D FFT pair requested in
[`jasvixban-fft3d-r2c-f64`](../jasvixban-fft3d-r2c-f64/README.md); submit and evaluate
the two together (same applications: pseudo-spectral PDE solvers, FFT-based
convolution, tomographic reconstruction — one inverse transform per
iteration, on the critical path).

## Contract and Baseline

- Input: contiguous `float64` tensor `input` with shape `[nz, ny, nf, 2]`,
  `nf = nx//2 + 1`; interleaved `(re, im)` packed Hermitian half-spectrum,
  x-row stride exactly `nf`, no even-row padding. Inputs are not modified.
- Transform convention: identical to cuFFT Z2D / `torch.fft.irfftn` — the
  array is treated as the `k_x >= 0` half of a Hermitian spectrum and
  implicitly completed; imaginary parts on the `k_x = 0` plane (and the
  `k_x = nx/2` plane where applicable) are ignored.
- Output: new contiguous `float64` tensor `[nz, ny, nx]` with
  `nx = 2*(nf-1)`, real space, **unnormalized** (forward followed by inverse
  scales by `N = nz*ny*nx`; applications fold physical scaling into their
  own Fourier-space kernels).
- Reference: float64 internal arithmetic in `definition.json`, cast to
  `float64` on output, used for correctness.
- Baseline: cuFFT reached through `torch.fft.irfftn` (PyTorch ≥ 2.1 with
  CUDA 12.x, dispatching to cuFFT `CUFFT_Z2D` out-of-place). See
  `baseline.py`.
- Hardware: NVIDIA B200 or B300. Correctness tolerances: `rtol = 1e-9`,
  `atol = 1e-6` versus the float64 reference. Both reference and baseline
  evaluate in float64, and the baseline (cuFFT Z2D) is **bit-exact** against
  the reference on every workload (measured max-abs 0.0). The tolerances
  leave ~4 orders of margin for implementations using different summation
  orders (typical relative deviation from cuFFT ~1e-13) while still
  rejecting single-precision shortcuts disguised as float64 output.
- This is the double-precision twin of
  [`jasvixban-fft3d-c2r`](../jasvixban-fft3d-c2r/README.md): identical
  contract, workloads, and deployment requirements; only the dtype pair
  differs.

## Workloads

Same eight grids as the forward request (32³ … 256³ plus anisotropic
80×80×112, 96×96×144; `nx` fastest-varying), with `nf = nx//2+1`. Inputs are
generated as random arrays; no Hermitian-consistency preprocessing is needed
because both the reference and any conforming candidate apply the same
implicit-completion convention to arbitrary packed input.

## Input data realism (optional)

Uniform-random protocol inputs have a white spectrum; real application
spectra concentrate energy in specific wavenumber bands, which stresses
cancellation and dynamic range differently. Implementers who want realistic
input statistics can use `gen_surrogate.py --complex` in this directory with
the two reference profiles in `spectra/c2r_*.csv`, measured from production
half-spectra of a structured-grid spectral application — radially averaged
|X(k)|² only, one number per wavenumber shell; no spatial, phase, or time
information is published. Generate from this request directory (requires only
numpy; keep the generated `.npy` files out of the repository):

```bash
# --verify reports per-shell agreement of the generated field
python3 gen_surrogate.py spectra/c2r_a.csv \
    --shape 64 64 64  --complex --scale 9.269e-4 --seed 1 --out c2r_in_64.npy --verify
python3 gen_surrogate.py spectra/c2r_b.csv \
    --shape 144 96 96 --complex --scale 4.066e-4 --seed 2 --out c2r_in_144.npy --verify
```

Surrogate spectra reproduce the reference to ~15% median relative error per
shell (single-realization sampling noise) at matched rms magnitude, and have
cross-correlation < 0.002 with any source-data frame. The
generator emits complex64; load with `np.load(...).astype(np.complex128)`
for this request.

## Evaluate with FlashInfer Bench

Identical to the forward request, with `REQUEST_DIR` changed:

```bash
python -m pip install torch==2.11.0 flashinfer-python==0.6.18.post1 flashinfer-bench==0.1.2
CUDA_VISIBLE_DEVICES=0 REQUEST_DIR=requests/jasvixban-fft3d-c2r python -c '
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

An additional round-trip check worth running when evaluating any candidate:
`inverse(forward(x)) == N * x` to float64 tolerance (the unnormalized
convention is part of the contract).

## Deployment contract (beyond the benchmark)

Same three requirements as the forward request: caller-owned in-place
buffers, pure stream semantics (no internal device-wide sync; autotune only
at plan-creation time), and first-class handling of non-power-of-two and
anisotropic sizes.

## Validation results

Baseline (cuFFT Z2D via `torch.fft.irfftn`, torch 2.14.0+cu130), random
uniform inputs in [-1, 1], CUDA-event medians over 100 iterations,
correctness vs. the float64 reference at `rtol=1e-9 / atol=1e-6`
(baseline is bit-exact against the reference — identical code path):

| grid (nz×ny×nx) | baseline latency (ms) | correctness |
| --- | --- | --- |
| 32×32×32 | 0.0276 | PASS (bit-exact) |
| 64×64×64 | 0.0303 | PASS (bit-exact) |
| 96×96×96 | 0.0727 | PASS (bit-exact) |
| 128×128×128 | 0.1837 | PASS (bit-exact) |
| 192×192×192 | 0.5790 | PASS (bit-exact) |
| 256×256×256 | 1.4384 | PASS (bit-exact) |
| 80×80×112 | 0.0636 | PASS (bit-exact) |
| 96×96×144 | 0.1116 | PASS (bit-exact) |

_Measured on NVIDIA GeForce RTX 5090 (sm_120) as a pre-submission smoke
test only. Consumer Blackwell cuts FP64 throughput hard (1/64 of FP32), so
these numbers are unrepresentative of B200/B300 in both absolute and
relative terms — they validate the pipeline, not the hardware. Results on
the target B200/B300 will be appended here before PR review._

## License

Apache-2.0 code / CC-BY-4.0 documentation under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). Baseline
calls cuFFT (NVIDIA proprietary, CUDA toolkit) via PyTorch's BSD-licensed
`torch.fft`; no cuFFT source is bundled.
