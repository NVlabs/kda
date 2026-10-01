# Request: LeapQuant flush — window-boundary re-quantization of the recurrent state (linear-attention decode)

Linear-attention and hybrid LLMs (the Gated DeltaNet layers of Qwen3.5-9B / Qwen3.5-35B-A3B, Kimi Delta Attention in
Kimi-Linear-48B) keep one recurrent state matrix `S` (128 × 128 fp32, 64 KB) per sequence and head. The stock vLLM
decode kernel reads and writes that matrix on every token, so a decode step moves 129 KB per (sequence, head) and is
bound by memory bandwidth, not arithmetic.

[LeapQuant](https://arxiv.org/abs/2609.38166) removes most of that traffic with per-window quantization. The state is
stored at 1.19 bytes per element — four fp16 Compensator Tokens (a rank-4 part) plus a smoothed int8 residual — and the
last 16 rank-1 updates are buffered in bf16. A decode step then reads 27.7 KB and writes only its buffered update.
At the end of each 16-token window a **flush** materialises the state and re-quantizes it for the next window. The
flush is the kernel requested here: at batch 256 it takes 0.247 ms per layer with every program due, against about
0.06 ms of memory traffic, and it is what caps the amortised decode speed-up over the stock fp32 kernel at 2.65×
(3.0× with a flush at its memory bound).

## Contract and Baseline

One *program* is one (sequence, head). All shapes below have leading `[batch_size, 32]`; `V = K = 128`, window
`L = 16`, `R = 4` Compensator Tokens.

- Inputs (old checkpoint): `codes` int8 `[V, K]`, column scales `s_k` fp32 `[K]`, row scales `s_v` fp32 `[V]`,
  Compensator Tokens `u` fp16 `[R, K]` and `q` fp16 `[R, V]`. The checkpoint decodes to
  `S_old[v, k] = Σ_r q[r, v] u[r, k] + codes[v, k] · s_k[k] / 127 · s_v[v]`.
- Inputs (buffered updates of the window): `kbuf` bf16 `[L, K]`, `ubuf` bf16 `[L, V]`, weights `w` fp32 `[L]` in
  `[0, 1]`, gate product `p` fp32 scalar in `(0, 1]`.
- Input `q0` fp32 `[V, R]`: fixed start matrix of the subspace iteration, shared by all programs.
- Math (`definition.json`, plain fp32 torch, about 40 lines): `S = p · S_old + Σ_j w[j] · ubuf[j] ⊗ kbuf[j]`; new
  Compensator Tokens from **one** round of subspace iteration started at `q0` (orthonormalise `Sᵀ q0`, then `S ·`
  that, then `u = Sᵀ q`; Gram-Schmidt with the projections applied twice so rank-deficient heads stay orthogonal),
  rounded to fp16; residual `E = S − qᵀu`; smoothing scales `s_k[k] = sqrt(mean_v |E[v, k]|)`,
  `s_v[v] = max_k |E[v, k]| / s_k[k]`; `codes = round(127 · E / (s_k s_v))`.
- Outputs: the new `codes`, `s_k`, `s_v`, `u`, `q` with the input dtypes and shapes.
- Domain: scales positive, `|S|` below 1e4 (the deployment range), any batch up to 512 sequences per call.
- Fixed: the checkpoint format and the algorithm (`q0`, one round, no warm start from the previous Compensator
  Tokens, which measurably hurts model perplexity). Free: the order of operations, the precision of every
  intermediate, tensor cores or not, thread and memory layout — anything that keeps the checkpoint as accurate as the
  reference's, see the criterion below.
- Baseline (`baseline.py`): our TileLang kernel for sm_100, original work of this request's authors, first published
  here. One persistent CTA per SM, a TMA producer warp, three 128-thread consumer groups, products on tensor cores,
  the rebuilt state in shared memory as fp16 with the old rank-4 part carried algebraically in fp32, CholeskyQR2 with
  a relative pivot floor for the orthonormalisation, the due programs balanced across CTAs by a rank scan.
- Hardware: NVIDIA B200.

## Correctness criterion

The outputs cannot be compared elementwise. Round-to-nearest codes flip whenever two implementations differ in the
last bits of `E`, and the fp16 rounding of the Compensator Tokens amplifies any earlier difference: two
implementations of the reference that differ only in summation order agree on 92–99.8 % of the codes and on 4–6 %
of the scales bitwise, yet their decoded states differ by 5e-4 of `|S|`, a ninth of the quantisation error. We
therefore judge the **decoded state**. For every head, with `S` the exactly rebuilt (fp64) state and
`err = mean |decode(checkpoint) − S|` over the head's 128 × 128 elements:

- `err_candidate ≤ 1.01 · err_reference` if `err_reference ≥ 1e-4 · mean |S|` (every head with a real residual);
- `err_candidate ≤ 1e-3 · mean |S|` otherwise (a head the format captures almost exactly, where the reference's
  error is rounding noise);
- all outputs finite and codes in `[−127, 127]`.

`benchmark.py` applies this to three inputs per batch size: a synthetic checkpoint with edge-case heads (rank 1,
rank 2, all zero, tiny, huge, a hot row), the checkpoint the reference produced from it (a second window), and
unstructured in-domain random tensors. What the bound does with implementations we have measured:

| implementation | worst-head error ratio | verdict |
| --- | --- | --- |
| the reference computed in fp64 (other rounding, other order) | 1.0000 | pass |
| TF32 tensor cores with hi/lo splits, CholeskyQR2 | 1.0001–1.0019 | pass |
| the baseline: fp16 shared-memory state, old rank-4 part carried in fp32 | 1.0022–1.0032 | pass |
| the state rounded to fp16 before compression | 1.0020–1.0031 | pass |
| the state and the residual both rounded to fp16 | 1.0028–1.0039 | pass |
| the scales rounded to fp16 | 1.0013–1.0018 | pass |
| the residual rounded to bf16 | 1.035–1.049 | fail |
| the state rounded to bf16 | 1.067–1.127 | fail |
| codes rounded toward zero instead of to nearest | 2.0–2.3 | fail |

So reorganising the arithmetic is free, fp16 intermediates are within the bound, bf16 ones are not, and any
shortcut in the quantisation itself is far outside it: precision can be traded for speed only within 1 % of the
reference's reconstruction error on every head. We also have two bit-exact variants of this request (an IEEE fp32 specification with fixed
reduction orders, and one with a bit-accurate model of the TF32 `mma.sync` instruction, at 0.596 ms and 0.410 ms
against the baseline's 0.247 ms) and can provide them if an exact reference is preferred.

## Workloads

Six workloads in `workloads.jsonl`: batch sizes 16, 32, 64, 128, 256, 512 with 32 heads, every program due. These
are the concurrency levels of our vLLM deployments (`max_num_seqs` up to 512); batch 256 is the headline point.
The trace inputs are declared `random` for the schema; `benchmark.py` generates in-domain inputs itself.

## Evaluate

This request ships `benchmark.py`, so use it instead of flashinfer-bench (whose elementwise check cannot pass any
implementation, see above). Environment used for the results below: Python 3.12, `torch==2.13.0+cu130`,
`tilelang==0.1.12`, CUDA toolkit 13.0 (`nvcc` on `PATH`, needed by TileLang's JIT), driver 580.126.20.

```bash
python -m pip install torch==2.13.0 tilelang==0.1.12
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-gdn-flush/benchmark.py                     # baseline
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-gdn-flush/benchmark.py --impl my_flush.py  # a candidate
```

Correctness runs the functional form `run(codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0)` on every batch size.
Timing runs the deployment form (`Pool` and `flush_inplace`, see below): CUDA kernel time from the torch profiler,
4 warm-up and 16 timed calls, **L2-cold** — eight disjoint slot sets are flushed in rotation, as in a real model
where 24–30 layers each own their state and no checkpoint is still in L2 when its layer comes round again.

## Deployment contract (beyond the benchmark)

A delivered kernel must be pluggable into the serving engine:

1. Update a **caller-owned pool in place**. Each (slot, head) owns a 64 KB region of one flat buffer and the
   checkpoint occupies its first 19 KB (codes, then scales, then Compensator Tokens; exact offsets in
   `baseline.py::Pool`). Slots are addressed through an `idx` tensor, not contiguously.
2. Select the work on the GPU. Only programs whose fill counter equals `L` are due. At steady state that is 1/16 of
   them, unevenly spread; the kernel must find and balance them itself (the baseline scans the counters and assigns
   programs to CTAs by rank). The call runs every decode step inside a captured CUDA graph, so there is no host-side
   branching on the due set: nothing due must cost almost nothing (baseline 0.005 ms, 1/16 due 0.031 ms at batch 256).
3. Reset the update buffer of each flushed program: `w = 0`, `p = 1`. The fill counter is left at `L`: the decode
   step kernel treats `L` as "flushed, empty", and clearing it in the flush would race with CTAs still scanning the
   flags.
4. No device-wide synchronisation and no JIT or autotuning inside the call.

## Validation results

`benchmark.py` on NVIDIA B200 (sm_100, 148 SMs), baseline from this directory:

| batch_size | programs | worst-head error ratio vs reference (structured / second window / random) | correctness | all-due latency (ms) | memory bound (ms) |
| --- | --- | --- | --- | --- | --- |
| 16 | 512 | 1.0026 / 1.0027 / 1.0025 | PASS | 0.0255 | 0.004 |
| 32 | 1024 | 1.0024 / 1.0027 / 1.0027 | PASS | 0.0409 | 0.008 |
| 64 | 2048 | 1.0025 / 1.0030 / 1.0023 | PASS | 0.0725 | 0.015 |
| 128 | 4096 | 1.0025 / 1.0032 / 1.0023 | PASS | 0.1353 | 0.031 |
| 256 | 8192 | 1.0025 / 1.0030 / 1.0022 | PASS | 0.2465 | 0.061 |
| 512 | 16384 | 1.0025 / 1.0029 / 1.0022 | PASS | 0.4812 | 0.123 |

The memory bound is 46.1 KB per program at 6.3 TB/s, the read + write bandwidth the stock vLLM fp32 decode kernel
reaches on this GPU at batch 256; small batches cannot reach that bandwidth, so their bound is optimistic.

## License

`baseline.py`, `benchmark.py` and the embedded reference are Apache-2.0 code by the request authors; this
documentation is Creative Commons Attribution 4.0 under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline is compiled with the separately
installed [TileLang](https://github.com/tile-ai/tilelang) (MIT) and runs on PyTorch (BSD-3-Clause); neither is bundled.
