# Request: LeapQuant decode step — gated delta rule on a quantized recurrent state (linear-attention decode)

Linear-attention and hybrid LLMs (the Gated DeltaNet layers of Qwen3.5-9B / Qwen3.5-35B-A3B, Kimi Delta Attention in
Kimi-Linear-48B) keep one recurrent state matrix `S` (128 × 128 fp32, 64 KB) per sequence and head. The stock vLLM
decode kernel reads and writes that matrix on every token, so a decode step moves 129 KB per (sequence, head) and is
bound by memory bandwidth, not arithmetic.

[LeapQuant](https://arxiv.org/abs/2609.38166) removes most of that traffic with per-window quantization. The state is
stored at 1.19 bytes per element — four fp16 Compensator Tokens (a rank-4 part) plus a smoothed int8 residual — and the
last 16 rank-1 updates are buffered in bf16. The **decode step** requested here computes each token's output and its
new rank-1 update from that representation, reading 27.8 KB per (sequence, head) and writing only the new update; it
never materialises or writes back the state (the window-boundary flush is a separate request,
`requests/conless-leapquant-gdn-flush/`). At batch 256 the baseline takes 0.0449 ms per layer against 0.0367 ms for its
memory traffic at the bandwidth a large read stream reaches on the same GPU.

## Contract and Baseline

One *program* is one (sequence, head). Shapes below have leading `[batch_size, 32]` unless noted; `V = K = 128`,
window `L = 16`, `R = 4` Compensator Tokens, 16 key heads shared by the 32 value heads (GQA).

- Inputs (the window's checkpoint): `codes` int8 `[V, K]`, column scales `s_k` fp32 `[K]`, row scales `s_v` fp32
  `[V]`, Compensator Tokens `u` fp16 `[R, K]` and `q` fp16 `[R, V]`. The checkpoint decodes to
  `S0[v, k] = Σ_r q[r, v] u[r, k] + codes[v, k] · s_k[k] / 127 · s_v[v]`.
- Inputs (the window's buffered updates so far): `kbuf` bf16 `[L, K]`, `ubuf` bf16 `[L, V]`, weights `w` fp32 `[L]`,
  gate product `p` fp32, fill count `h` int32 with `0 ≤ h < L`; entries `j ≥ h` are empty (`w_j = 0`).
- Inputs (this token): `qx`, `kx` bf16 `[batch_size, 16, K]`, `vx` bf16 `[V]`, gate and beta pre-activations `a`, `b`
  bf16, and the per-head constants `A_log`, `dt_bias` fp32 `[32]`.
- Math (`definition.json`, plain fp32 torch, about 30 lines): `kn`, `qn` = l2-normalised key / query (`qn` scaled by
  `K^-1/2`); decay `at = exp(-exp(A_log) · softplus(a + dt_bias))`; `beta = sigmoid(b)` rounded to bf16 (as the model
  carries it); `pn = p · at`; decayed weights `wd_j = w_j · at` for live entries;
  `S = pn · S0 + Σ_j wd_j · ubuf_j ⊗ kbuf_j`; then the gated delta rule
  `u_new = beta · (v − S · kn)`, `o = S · qn + (kn · qn) · u_new`.
- Outputs: `o` bf16 `[V]`, the new buffered update `k_row = kn` bf16 `[K]` and `u_row = u_new` bf16 `[V]` (entry `h`),
  `w_new` = the decayed weights with `w_new[h] = 1`, and `p_new = pn`.
- Free: the order of operations, the precision of every intermediate, tensor cores or not, thread and memory layout —
  anything that meets the criterion below. The state must not be materialised in global memory.
- Baseline (`baseline.py`): our TileLang kernel for sm_100, original work of this request's authors, first published
  here. One persistent CTA per SM, a TMA producer warp, three consumer groups of four warps with one helper warp each
  (l2 norms, the dot products with the buffered keys and the Compensator Tokens); the int8 tile goes through the tensor
  cores (int8 → fp16 exactly, `mma.sync m16n8k16` against an fp16 hi/lo split of the scaled key and query), the
  consumer warps never wait for each other.
- Hardware: NVIDIA B200.

## Correctness criterion

The outputs are continuous, so they are compared with the reference directly, per (sequence, head):

- `o`, `k_row`, `u_row`: relative L2 error ≤ 3e-3 and max absolute error ≤ 1 % of the head's max |reference|, or every
  element within one bf16 step of the reference (a head whose only differences are output rounding: on heads with
  large values a correct implementation can flip a few bf16 roundings, which alone exceeds the relative L2 bound);
  an all-zero reference head must come out all zero;
- `w_new`, `p_new`: relative error ≤ 1e-5;
- all outputs finite.

`benchmark.py` applies this to three inputs per batch size: a synthetic state with edge-case heads (empty buffer,
all-zero state, full buffer, tiny and huge scales, rank-1 checkpoint), the following step (the first step's update
appended to the buffer), and unstructured in-domain random tensors. What the criterion does with variants we have
measured (worst error divided by the tolerance; ≤ 1 passes):

| implementation | worst error / tolerance | verdict |
| --- | --- | --- |
| the baseline | 0.07–1.00 | pass |
| the decoded checkpoint rounded to fp16 | 1.66 | fail |
| beta not rounded to bf16 (the model carries it in bf16) | 1.60 | fail |
| the state rounded to bf16 | 19.7 | fail |

## Workloads

Six workloads in `workloads.jsonl`: batch sizes 16, 32, 64, 128, 256, 512 with 32 heads. These are the concurrency
levels of our vLLM deployments (`max_num_seqs` up to 512); batch 256 is the headline point. The trace inputs are
declared `random` for the schema; `benchmark.py` generates in-domain inputs itself.

## Evaluate

Environment used for the results below: Python 3.12, `torch==2.13.0+cu130`, `tilelang==0.1.12`, CUDA toolkit 13.0
(`nvcc` on `PATH`, needed by TileLang's JIT), driver 580.126.20.

```bash
python -m pip install torch==2.13.0 tilelang==0.1.12
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-gdn-step/benchmark.py                    # baseline
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-gdn-step/benchmark.py --impl my_step.py  # a candidate
```

Correctness runs the functional form
`run(codes, s_k, s_v, u, q, kbuf, ubuf, w, p, h, qx, kx, vx, a, b, A_log, dt_bias)` on every batch size. Timing runs
the deployment form (`Pool` and `step_inplace`, see below): CUDA kernel time from the torch profiler, 6 warm-up and 32
timed calls, **L2-cold** — eight disjoint slot sets are used in rotation, as in a real model where 24–30 layers each own
their state and no checkpoint is still in L2 when its layer comes round again. Use an otherwise idle GPU: a second
process on the same GPU moved the batch-512 time from 0.084 to 0.095–0.157 ms in our runs.

## Deployment contract (beyond the benchmark)

A delivered kernel must be pluggable into the serving engine:

1. Read the checkpoint from a **caller-owned pool**, the same layout as the flush request: each (slot, head) owns a
   64 KB region of one flat buffer and the checkpoint occupies its first 19 KB (codes, then scales, then Compensator
   Tokens; exact offsets in `baseline.py::Pool`). Slots are addressed through an `idx` tensor, not contiguously; slot
   0 means "no sequence" and must produce a zero output.
2. Take q / k / v as the projection produces them: one bf16 row per sequence, `(q for 16 key heads | k for 16 key
   heads | v for 32 value heads)`.
3. Append the new update in place: `kbuf` / `ubuf` entry `h`, the decayed weights, `p`, and increment the fill count.
   A fill count of `L` means "just flushed, empty" and counts as 0.
4. The call runs every decode step inside a captured CUDA graph: no device-wide synchronisation, no host round trips,
   no JIT or autotuning inside the call.

## Validation results

`benchmark.py` on NVIDIA B200 (sm_100, 148 SMs), baseline from this directory:

| batch_size | programs | worst error / tolerance (structured / next step / random) | correctness | latency (ms) | memory bound (ms) |
| --- | --- | --- | --- | --- | --- |
| 16 | 512 | 0.07 / 0.34 / 0.26 | PASS | 0.0067 | 0.0023 |
| 32 | 1024 | 0.21 / 0.18 / 0.14 | PASS | 0.0099 | 0.0046 |
| 64 | 2048 | 0.37 / 0.40 / 0.56 | PASS | 0.0153 | 0.0092 |
| 128 | 4096 | 0.48 / 0.55 / 0.34 | PASS | 0.0251 | 0.0183 |
| 256 | 8192 | 0.48 / 0.33 / 0.50 | PASS | 0.0449 | 0.0367 |
| 512 | 16384 | 1.00 / 0.50 / 0.60 | PASS | 0.0840 | 0.0733 |

The memory bound is 28.6 KB per program (27.8 KB read, 0.8 KB written) at 6.54 TB/s, the bandwidth a large read-only
stream reaches on this GPU; small batches cannot reach that bandwidth, so their bound is optimistic.

## License

`baseline.py`, `benchmark.py` and the embedded reference are Apache-2.0 code by the request authors; this
documentation is Creative Commons Attribution 4.0 under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline is compiled with the separately
installed [TileLang](https://github.com/tile-ai/tilelang) (MIT) and runs on PyTorch (BSD-3-Clause); neither is bundled.
