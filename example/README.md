# Example: RMSNorm

This example shows the submission format. Copy it to `requests/<github-username>-<kernel-name>/` and replace this introduction with your affected project, observed bottleneck, and desired improvement. Submit a pull request directly to `wishlist`; an issue is optional. Keep the full request details in this README and link any existing discussion if useful.

RMSNorm (root mean square normalization) scales each input row by the reciprocal square root of its mean squared value plus epsilon, then applies a weight vector.

## Contract and Baseline

- Input: contiguous bfloat16 tensor `input` with shape `[batch_size, 4096]`.
- Weight: contiguous bfloat16 tensor `weight` with shape `[4096]`.
- Epsilon: Python float, `1e-6` in the provided workloads.
- Output: a new contiguous bfloat16 tensor of the same shape as `input`; inputs are not modified.
- Reference: explicit float32 arithmetic in `definition.json`, used for correctness.
- Baseline: the optimized `flashinfer.norm.rmsnorm` implementation called by `baseline.py`, used for performance comparison.
- Hardware: NVIDIA B200 or B300. Relative and absolute correctness tolerances are both `1e-2`.

The workloads use batch sizes 1, 32, and 256 to illustrate small and larger inputs. They are examples, not a measured production workload distribution. Each line in `workloads.jsonl` is a FlashInfer Trace workload record; `solution` and `evaluation` are `null` until evaluation.

The comparison implementation comes from [FlashInfer 0.6.18.post1](https://pypi.org/project/flashinfer-python/0.6.18.post1/), using the documented [RMSNorm API](https://docs.flashinfer.ai/generated/flashinfer.norm.rmsnorm.html). For a real request, supply the best-known implementation for your workloads and identify its source, version, and evidence in this README.

## Evaluate with FlashInfer Bench

This example has no `benchmark.py`, so evaluation uses `flashinfer-bench`. Use Python 3.12, a CUDA-enabled PyTorch installation, and a B200/B300-compatible NVIDIA driver and CUDA toolkit. Install the pinned packages in your chosen environment:

```bash
python -m pip install torch==2.11.0 flashinfer-python==0.6.18.post1 flashinfer-bench==0.1.2
```

From the repository root, select an idle supported GPU and run the following command. Change `REQUEST_DIR` to evaluate another request. `baseline.py::run` is registered as a Python solution; this keeps the submission layout simple while using FlashInfer Bench's native evaluator.

```bash
CUDA_VISIBLE_DEVICES=0 REQUEST_DIR=example python -c '
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
    warmup_runs=10, iterations=100, num_trials=3, rtol=1e-2, atol=1e-2))
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

The command validates the baseline against the mathematical reference and measures it with 10 warm-up runs, 100 timed iterations, and 3 trials. It prints evaluation records, including environment and latency, without writing benchmark output into the repository. FlashInfer Bench manages its build cache separately.

The reported `speedup_factor` compares the registered implementation with the mathematical reference. When evaluating a new candidate, compare its latency with the latency of `baseline.py`; speedup over the reference alone does not establish improvement over the best-known implementation. Set tolerances and measurement settings appropriate to your own request.

If your request needs a custom measurement method, include an optional `benchmark.py` and document its command, environment, warm-up, timing, and correctness checks in the request README.

## License

This wrapper and the embedded reference are Apache-2.0 code; this documentation is Creative Commons Attribution 4.0 under the [project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline calls the separately installed [FlashInfer library](https://github.com/flashinfer-ai/flashinfer), which uses Apache-2.0. No external implementation source or dataset is bundled here.
