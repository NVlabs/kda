# Request: LeapQuant flush for Kimi Delta Attention — window-boundary re-quantization with a per-channel gate

Kimi Delta Attention (the linear-attention layers of Kimi-Linear-48B) keeps one recurrent state matrix `S`
(128 × 128 fp32, 64 KB) per sequence and head and decays it per key channel on every token. The stock vLLM decode
kernel reads and writes that matrix on every token, so a decode step moves 129 KB per (sequence, head) and is bound by
memory bandwidth, not arithmetic.

[LeapQuant](https://arxiv.org/abs/2609.38166) removes most of that traffic with per-window quantization. The state is
stored at 1.19 bytes per element — four fp16 Compensator Tokens (a rank-4 part) plus a smoothed int8 residual — and the
last 16 rank-1 updates are buffered in bf16 together with the per-channel decay of the step that produced them. At the
end of each 16-token window a **flush** materialises the state and re-quantizes it for the next window. This is the
Kimi variant of `requests/conless-leapquant-gdn-flush/` (same checkpoint, same algorithm, a per-channel instead of a scalar
gate). At batch 256 the baseline takes 0.320 ms per layer with every program due, against about 0.073 ms of memory
traffic; the Kimi decode step is in `requests/conless-leapquant-kda-step/`.

## Contract and Baseline

One *program* is one (sequence, head). All shapes below have leading `[batch_size, 32]`; `V = K = 128`, window
`L = 16`, `R = 4` Compensator Tokens.

- Inputs (old checkpoint): `codes` int8 `[V, K]`, column scales `s_k` fp32 `[K]`, row scales `s_v` fp32 `[V]`,
  Compensator Tokens `u` fp16 `[R, K]` and `q` fp16 `[R, V]`. The checkpoint decodes to
  `S_old[v, k] = Σ_r q[r, v] u[r, k] + codes[v, k] · s_k[k] / 127 · s_v[v]`.
- Inputs (the window): `kbuf` bf16 `[L, K]`, `ubuf` bf16 `[L, V]`, weights `w` fp32 `[L]` in `[0, 1]`, the
  per-channel decay factor of the step that appended each entry `gbuf` fp16 `[L, K]` in `(0, 1]`, and the log-gate
  summed over the window `pcum` fp32 `[K]` (≤ 0; values below −1000 occur in real models).
- Input `q0` fp32 `[V, R]`: fixed start matrix of the subspace iteration, shared by all programs.
- Math (`definition.json`, plain fp32 torch): `S = S_old · diag(exp(pcum)) + Σ_j w[j] · ubuf[j] ⊗ (kbuf[j] ⊙ D_j)`
  with `D_j = Π_{i>j} gbuf[i]`, the decay entry `j` has seen since it was appended; then exactly the GDN flush: new
  Compensator Tokens from **one** round of subspace iteration started at `q0` (Gram-Schmidt with the projections
  applied twice), rounded to fp16; residual `E = S − qᵀu`; smoothing scales `s_k[k] = sqrt(mean_v |E[v, k]|)`,
  `s_v[v] = max_k |E[v, k]| / s_k[k]`; `codes = round(127 · E / (s_k s_v))`.
- Outputs: the new `codes`, `s_k`, `s_v`, `u`, `q` with the input dtypes and shapes.
- Domain: scales positive, `|S|` below 1e4, any batch up to 512 sequences per call.
- Fixed: the checkpoint format and the algorithm. Free: the order of operations, the precision of intermediates,
  tensor cores or not, thread and memory layout — anything that keeps the checkpoint within 1 % of the reference's
  accuracy on every head, see the criterion below.
- Baseline (`baseline.py`): our TileLang kernel for sm_100 (the generator of the GDN flush request built with
  `KDA=True`), original work of this request's authors, first published here. Same structure as the GDN flush; the
  decayed keys enter the rebuild as a bf16 hi + lo pair; the subspace iteration runs on an fp16 copy of the rebuilt
  state, while the residual is accumulated in fp32 from the checkpoint, the buffered updates and both rank-4 parts
  (a second dequantisation pass) and staged for the quantisation as fp16 with a power-of-two scale per 32 × 64 block.
- Hardware: NVIDIA B200.

## Correctness criterion

The same criterion as the GDN flush request: the decoded state is judged, not the codes. For every head, with `S` the
exactly rebuilt (fp64) state and `err = mean |decode(checkpoint) − S|` over the head's 128 × 128 elements:

- `err_candidate ≤ 1.01 · err_reference` if `err_reference ≥ 1e-4 · mean |S|`;
- `err_candidate ≤ 1e-3 · mean |S|` otherwise (a head the format captures almost exactly);
- all outputs finite and codes in `[−127, 127]`.

`benchmark.py` applies the criterion to three inputs per batch size: a synthetic checkpoint with edge-case heads
(rank 1, rank 2, all zero, tiny, huge, a hot row, a checkpoint decayed to nothing, channels that never decay), the
checkpoint the reference produced from it (a second window), and unstructured in-domain random tensors. Measured
worst ratios over batch sizes 64, 256 and 512:

| implementation | structured / second window / random | verdict |
| --- | --- | --- |
| the reference computed in fp64 | 1.0000 / 1.0000 / 1.0000 | pass |
| the baseline | 1.0016 / 1.0017 / 1.0026 | pass |
| our previous kernel (decayed old Compensator Tokens in fp16, residual from the fp16 state) | 1.075 / 1.005 / 1.013 | fail |
| the decayed old Compensator Tokens rounded to fp16 | 1.044 / 1.002 / 1.002 | fail |
| the state rounded to fp16 before compression | 1.045 / 1.004 / 1.012 | fail |
| the residual rounded to fp16 | 1.017 / 1.002 / 1.002 | fail |
| the residual rounded to bf16 | 1.039 / 1.037 / 1.034 | fail |
| the decayed buffered keys rounded to bf16 | over the absolute cap / 1.109 / 1.138 | fail |
| the state rounded to bf16 | over the absolute cap / 1.161 / 1.399 | fail |
| codes rounded toward zero instead of to nearest | 2.0 / 2.0 / 2.0 | fail |

The fp16 variants fail here although they pass the GDN request: the per-channel log-gate (down to −20 in the random
inputs) leaves some key channels of a head many orders of magnitude below the rest, and on the tiny head the whole state
sits near fp16's subnormal range.

## Workloads

Six workloads in `workloads.jsonl`: batch sizes 16, 32, 64, 128, 256, 512 with 32 heads, every program due. The trace
inputs are declared `random` for the schema; `benchmark.py` generates in-domain inputs itself.

## Evaluate

Environment used for the results below: Python 3.12, `torch==2.13.0+cu130`, `tilelang==0.1.12`, CUDA toolkit 13.0
(`nvcc` on `PATH`, needed by TileLang's JIT), driver 580.126.20. Use an otherwise idle GPU.

```bash
python -m pip install torch==2.13.0 tilelang==0.1.12
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-kda-flush/benchmark.py                     # baseline
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-kda-flush/benchmark.py --impl my_flush.py  # a candidate
```

Correctness runs the functional form `run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, q0)` on every batch size.
Timing runs the deployment form (`Pool` and `flush_inplace`): CUDA kernel time from the torch profiler, 4 warm-up and
16 timed calls, **L2-cold** — eight disjoint slot sets are flushed in rotation.

## Deployment contract (beyond the benchmark)

As for the GDN flush: update a caller-owned pool in place (each (slot, head) owns a 64 KB region whose first 19 KB are
the checkpoint; exact offsets in `baseline.py::Pool`), addressed through an `idx` tensor; select the due programs on the
GPU (fill counter equal to `L`, about 1/16 of them at steady state) and keep a call with nothing due almost free, since
it runs every decode step inside a captured CUDA graph; reset each flushed program's window (`w = 0`, `pcum = 0`,
`gbuf = 1`; the fill counter stays at `L`, which the decode step reads as "flushed, empty"); no device-wide
synchronisation, JIT or autotuning inside the call.

## Validation results

`benchmark.py` on NVIDIA B200 (sm_100, 148 SMs), baseline from this directory:

| batch_size | programs | worst-head error ratio vs reference (structured / second window / random) | correctness | all-due latency (ms) | memory bound (ms) |
| --- | --- | --- | --- | --- | --- |
| 16 | 512 | 1.0023 / 1.0020 / 1.0030 | PASS | 0.0318 | 0.005 |
| 32 | 1024 | 1.0019 / 1.0019 / 1.0028 | PASS | 0.0518 | 0.009 |
| 64 | 2048 | 1.0016 / 1.0017 / 1.0026 | PASS | 0.0955 | 0.018 |
| 128 | 4096 | 1.0017 / 1.0017 / 1.0024 | PASS | 0.1750 | 0.037 |
| 256 | 8192 | 1.0015 / 1.0015 / 1.0022 | PASS | 0.3198 | 0.073 |
| 512 | 16384 | 1.0016 / 1.0016 / 1.0022 | PASS | 0.6069 | 0.147 |

The memory bound is 55.1 KB per program (checkpoint, window and gate history read; checkpoint written and the window
reset) at 6.3 TB/s, the read + write bandwidth the stock vLLM fp32 decode kernel reaches on this GPU at batch 256.

## License

`baseline.py`, `benchmark.py` and the embedded reference are Apache-2.0 code by the request authors; this
documentation is Creative Commons Attribution 4.0 under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline is compiled with the separately
installed [TileLang](https://github.com/tile-ai/tilelang) (MIT) and runs on PyTorch (BSD-3-Clause); neither is bundled.
