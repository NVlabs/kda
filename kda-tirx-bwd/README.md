# Packed TIRx KDA backward

[`kda_backward_packed.py`](kda_backward_packed.py) implements the Kimi Delta Attention
(KDA) backward pass for NVIDIA B200 (`sm_100a`). Its upstream source is the
[TIRx-kernels packed KDA backward kernel](https://github.com/mlc-ai/TIRx-kernels/blob/8b6ed130a330e522a22b62eed2fa15ed487acec0/tirx_kernels/kda/kda_backward_packed.py).

## Supported inputs and API

The entry point is `setup(data, B, T, H) -> launch`, where `H` is the query/key head
count (`Hqk`), `T` is the total packed token count, and `B` must be 1. The kernel
supports K=V=128, chunk size 64, grouped value heads (`Hv % Hqk == 0`), and packed
sequences with partial trailing chunks.

`data` uses the prepared-tensor contract of FLA's `chunk_kda_bwd`:

| Keys | Meaning |
|---|---|
| `q`, `k` | Saved L2-normalized queries and keys |
| `v` | Values |
| `beta` | Activated update gate |
| `Aqk`, `Akk` | Saved interaction matrices from the forward pass |
| `g` | Chunk-local cumulative base-2 log gates |
| `initial_state` | Per-sequence **K-first** initial state |
| `do`, `dht` | Upstream output and final-state gradients |
| `scale`, `chunk_size`, `cu_seqlens` | Attention scale, chunk size (64), and packed sequence offsets |
| `dq`, `dk`, `dv`, `db`, `dg`, `dh0` | Caller-allocated output buffers |

All input and output tensors except `cu_seqlens` must be contiguous. `cu_seqlens`
may be `None` for a single sequence; otherwise it is converted to contiguous int64 offsets
on the input device. `setup` compiles the kernels, allocates scratch space, and runs
once before returning. Each `launch()` writes the output buffers in `data`.
This contract does not return `dA` or `dbias`.

The forward [`kda-tirx/kernel.py`](../kda-tirx/kernel.py) accepts raw inputs and a
V-first state. Its `prepare` / `run` interface is separate from this backward API.

## Scheduling

The offline builder's default CTA count follows the detected SM count, matching
runtime setup and benchmark preparation. On a 148-SM B200, the fused path builds
its schedule for up to 148 CTAs. The upstream 152-CTA / 768-chain tuned schedule
is retained and selected only when both counts match.

The fused path handles equal query/key and value head counts divisible by eight
when every sequence consists of full 64-token chunks. Grouped value heads and
partial trailing chunks use the persistent megakernel path.

## Setup example

Use the dependencies from the root [installation instructions](../README.md#install).
Run the following commands from the repository root on a B200. This example uses
`prepare_data` to create saved inputs and output buffers for two packed sequences:

```bash
PYTHONPATH=kda-tirx-bwd uv run python - <<'PY'
import kda_backward_packed as bwd

seq_lens = (129, 79)
data = bwd.prepare_data(num_qk_heads=2, num_v_heads=4, seq_lens=seq_lens)
launch = bwd.setup(data, B=1, T=sum(seq_lens), H=2)
launch()
# Gradients are written to data["dq"], data["dk"], data["dv"],
# data["db"], data["dg"], and data["dh0"].
PY
```

## Correctness checks

The checks below compare the fused and grouped-head/partial-chunk paths against
FLA and check repeatability across launches:

```bash
PYTHONPATH=kda-tirx-bwd uv run python - <<'PY'
import kda_backward_packed as bwd

bwd.run_test(num_qk_heads=8, num_v_heads=8, seq_lens=(128, 128))
bwd.run_test(num_qk_heads=2, num_v_heads=4, seq_lens=(129, 79))
PY
```

## Benchmark helpers

The module retains `prepare_data`, `CONFIGS`, `BENCH_CONFIGS`, and `run_bench`.
For example, benchmark the first official workload against FLA:

```bash
PYTHONPATH=kda-tirx-bwd uv run python - <<'PY'
import kda_backward_packed as bwd

print(bwd.run_bench(**bwd.BENCH_CONFIGS[0]))
PY
```

The root [`bench.py`](../bench.py) measures the forward implementations only.
