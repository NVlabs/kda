# Request: Inverse 3D FFT from packed Hermitian half-spectrum (fp32)

Backward half of the forward/backward 3D FFT pair requested in
[`jasvixban-fft3d-r2c`](../jasvixban-fft3d-r2c/README.md); submit and evaluate
the two together (same applications: pseudo-spectral PDE solvers, FFT-based
convolution, tomographic reconstruction — one inverse transform per
iteration, on the critical path).

## Contract and Baseline

- Input: contiguous `float32` tensor `input` with shape `[nz, ny, nf, 2]`,
  `nf = nx//2 + 1`; interleaved `(re, im)` packed Hermitian half-spectrum,
  x-row stride exactly `nf`, no even-row padding. Inputs are not modified.
- Transform convention: identical to cuFFT C2R / `torch.fft.irfftn` — the
  array is treated as the `k_x >= 0` half of a Hermitian spectrum and
  implicitly completed; imaginary parts on the `k_x = 0` plane (and the
  `k_x = nx/2` plane where applicable) are ignored.
- Output: new contiguous `float32` tensor `[nz, ny, nx]` with
  `nx = 2*(nf-1)`, real space, **unnormalized** (forward followed by inverse
  scales by `N = nz*ny*nx`; applications fold physical scaling into their
  own Fourier-space kernels).
- Reference: float64 internal arithmetic in `definition.json`, cast to
  `float32` on output, used for correctness.
- Baseline: cuFFT reached through `torch.fft.irfftn` (PyTorch ≥ 2.1 with
  CUDA 12.x, dispatching to cuFFT `CUFFT_C2R` out-of-place). See
  `baseline.py`.
- Hardware: NVIDIA B200 or B300. Correctness tolerances: `rtol = 1e-3`,
  `atol = 1e-2` versus the float64 reference. The unnormalized inverse's
  output magnitude grows with N, and cuFFT itself was measured to need at
  most `2.5e-3` atol at 256³; `1e-2` keeps ~4x margin. Relative L2 error is
  ~2e-7 on every workload.

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
cross-correlation < 0.002 with any source-data frame.

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
    warmup_runs=10, iterations=100, num_trials=3, rtol=1e-3, atol=1e-2))
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
`inverse(forward(x)) == N * x` to float32 tolerance (the unnormalized
convention is part of the contract).

## Deployment contract (beyond the benchmark)

Same three requirements as the forward request: caller-owned in-place
buffers, pure stream semantics (no internal device-wide sync; autotune only
at plan-creation time), and first-class handling of non-power-of-two and
anisotropic sizes.

## Validation results

Baseline (cuFFT via `torch.fft.irfftn` with `norm="forward"`, torch
2.14.0+cu130), random interleaved half-spectrum inputs, CUDA-event medians
over 200 iterations, correctness vs. the float64 reference at
`rtol=1e-3 / atol=1e-2`:

| grid (nz×ny×nx) | baseline latency (ms) | correctness |
| --- | --- | --- |
| 32×32×32 | 0.0236 | PASS (rel-L2 2.9e-07) |
| 64×64×64 | 0.0231 | PASS (2.3e-07) |
| 96×96×96 | 0.0232 | PASS (2.2e-07) |
| 128×128×128 | 0.0425 | PASS (2.3e-07) |
| 192×192×192 | 0.0893 | PASS (2.7e-07) |
| 256×256×256 | 0.2896 | PASS (2.5e-07) |
| 80×80×112 | 0.0232 | PASS (2.0e-07) |
| 96×96×144 | 0.0237 | PASS (2.3e-07) |

_Measured on NVIDIA GeForce RTX 5090 (sm_120) as a pre-submission smoke
test; grids up to 96³ are dispatch-overhead-dominated (~23 µs floor), so
kernel-level comparisons should focus on 192³/256³. Results on the target
B200/B300 will be appended here before PR review._

## License

Apache-2.0 code / CC-BY-4.0 documentation under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). Baseline
calls cuFFT (NVIDIA proprietary, CUDA toolkit) via PyTorch's BSD-licensed
`torch.fft`; no cuFFT source is bundled.
