# Request: LeapQuant decode step for Kimi Delta Attention — per-channel-gated delta rule on a quantized state

Kimi Delta Attention (the linear-attention layers of Kimi-Linear-48B) keeps one recurrent state matrix `S`
(128 × 128 fp32, 64 KB) per sequence and head and decays it per key channel on every token. The stock vLLM decode
kernel reads and writes that matrix on every token, so a decode step moves 129 KB per (sequence, head) and is bound by
memory bandwidth, not arithmetic.

[LeapQuant](https://arxiv.org/abs/2609.38166) removes most of that traffic with per-window quantization. The state is
stored at 1.19 bytes per element — four fp16 Compensator Tokens (a rank-4 part) plus a smoothed int8 residual — and the
last 16 rank-1 updates are buffered in bf16 together with the per-channel decay of the step that produced them. The
**decode step** requested here computes each token's output and its new rank-1 update from that representation,
reading 32.6 KB per (sequence, head) and writing only the new update; it never materialises or writes back the state.
This is the Kimi variant of `requests/conless-leapquant-gdn-step/`; the window-boundary flush is
`requests/conless-leapquant-kda-flush/`. At batch 256 the baseline takes 0.068 ms per layer against 0.044 ms for its
memory traffic.

## Contract and Baseline

One *program* is one (sequence, head). Shapes below have leading `[batch_size, 32]` unless noted; `V = K = 128`,
window `L = 16`, `R = 4` Compensator Tokens, 32 key heads (no GQA).

- Inputs (the window's checkpoint): `codes` int8 `[V, K]`, column scales `s_k` fp32 `[K]`, row scales `s_v` fp32
  `[V]`, Compensator Tokens `u` fp16 `[R, K]` and `q` fp16 `[R, V]`. The checkpoint decodes to
  `S0[v, k] = Σ_r q[r, v] u[r, k] + codes[v, k] · s_k[k] / 127 · s_v[v]`.
- Inputs (the window so far): `kbuf` bf16 `[L, K]`, `ubuf` bf16 `[L, V]`, the decay factor of the step that appended
  each entry `gbuf` fp16 `[L, K]` (1 past the fill count), weights `w` fp32 `[L]` (1 for live entries), the log-gate
  summed since the checkpoint `pcum` fp32 `[K]`, and the fill count `h` int32 with `0 ≤ h < L`.
- Inputs (this token): `qx`, `kx` bf16 `[K]`, `vx` bf16 `[V]`, the per-channel gate pre-activation `a` bf16 `[K]`, the
  beta pre-activation `b` bf16, and the per-head constants `A_log` fp32 `[32]`, `dt_bias` fp32 `[32, K]`.
- Math (`definition.json`, plain fp32 torch, about 35 lines): `kn`, `qn` = l2-normalised key / query (`qn` scaled by
  `K^-1/2`); log-decay `g = −exp(A_log) · softplus(a + dt_bias)` per channel, `en = exp(g)`, `pcum_new = pcum + g`;
  `beta = sigmoid(b)` rounded to bf16 (as the model carries it); the state before this token's update
  `S = S0 · diag(exp(pcum_new)) + Σ_{j<h} w_j · ubuf_j ⊗ (kbuf_j ⊙ D_j)` with `D_j = en · Π_{j<i<h} gbuf_i`; then
  `u_new = beta · (v − S · kn)`, `o = S · qn + (kn · qn) · u_new`.
- Outputs: `o` bf16 `[V]`; the new buffered update for entry `h`: `k_row = kn` bf16 `[K]`, `u_row = u_new` bf16 `[V]`,
  `g_row = en` fp16 `[K]`; `w_new` (= `w` with `w_new[h] = 1`) and `pcum_new` fp32 `[K]`.
- Free: the order of operations, tensor cores or not, thread and memory layout — anything that meets the criterion
  below, which leaves no room for reduced-precision intermediates (every intermediate needs about fp32 precision; hi/lo
  splits on the tensor cores are fine). The state must not be materialised in global memory.
- Baseline (`baseline.py`): our TileLang kernel for sm_100, original work of this request's authors, first published
  here. One persistent CTA per SM, a TMA producer warp, three warps for the norms and the per-channel gate, two consumer
  groups of four warps; the int8 tile goes through the tensor cores (int8 → fp16 exactly, `mma.sync m16n8k16` against
  an fp16 hi/lo split of the scaled, gate-decayed key and query); the decayed buffered keys are kept in shared memory
  as a bf16 hi + lo pair; the gate uses `__expf` / `__logf`.
- Hardware: NVIDIA B200.

## Correctness criterion

The outputs are compared element by element with the definition's formula evaluated in fp64. `o` and `u_new` are sums
whose terms can cancel strongly (in our random inputs an output 33× smaller than its terms is common), so the
allowance is measured against the scale of the terms rather than the output itself — the usual forward-error bound.
With `M` the same expression evaluated on the absolute values of every term (state, checkpoint and dot products
included):

- `o`, `k_row`, `u_row`: `|x − exact| ≤ ½ bf16 step + 1e-5 · M` (half a step is the rounding of the output itself);
  an element whose terms are all zero must come out zero;
- `g_row`: `|x − exact| ≤ ½ fp16 step + 1e-5 · |exact|`; `w_new` exact; `pcum_new`: `|x − exact| ≤ 1e-5 · (|exact| + 1)`;
- all outputs finite.

1e-5 · M is 200× what fp32 arithmetic in any order costs (the fp32 reference: 5e-8 · M), so reordering, other
reductions and hi/lo tensor-core splits pass, and any intermediate rounded to bf16, fp16 or TF32 fails.
`benchmark.py` applies this to three inputs per batch size: a synthetic state with edge-case heads (empty buffer,
all-zero state, full buffer, tiny and huge scales, a checkpoint decayed to nothing), the following step (the first
step's update appended to the window), and unstructured in-domain random tensors. Worst error / tolerance over batch
sizes 64, 256 and 512 (≤ 1 passes):

| implementation | structured / next step / random | verdict |
| --- | --- | --- |
| the fp32 reference | 0.033 / 0.040 / 0.028 | pass |
| the baseline | 0.068 / 0.079 / 0.111 | pass |
| the decayed buffered keys rounded to bf16 (our previous kernel) | 60 / 74 / 84 | fail |
| the decoded checkpoint rounded to fp16 | 8.3 / 7.5 / 18.5 | fail |
| the state rounded to fp16 | 7.9 / 8.6 / 17.0 | fail |
| the state rounded to bf16 | 77 / 80 / 166 | fail |
| state × key / query products in TF32 | 49 / 49 / 49 | fail |
| the normalised key and query rounded to bf16 | 113 / 114 / 173 | fail |
| beta not rounded to bf16 (the model carries it in bf16) | 377 / 376 / 326 | fail |

## Workloads

Six workloads in `workloads.jsonl`: batch sizes 16, 32, 64, 128, 256, 512 with 32 heads. The trace inputs are declared
`random` for the schema; `benchmark.py` generates in-domain inputs itself.

## Evaluate

Environment used for the results below: Python 3.12, `torch==2.13.0+cu130`, `tilelang==0.1.12`, CUDA toolkit 13.0
(`nvcc` on `PATH`, needed by TileLang's JIT), driver 580.126.20. Use an otherwise idle GPU.

```bash
python -m pip install torch==2.13.0 tilelang==0.1.12
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-kda-step/benchmark.py                    # baseline
CUDA_VISIBLE_DEVICES=0 python requests/conless-leapquant-kda-step/benchmark.py --impl my_step.py  # a candidate
```

Correctness runs the functional form
`run(codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias)` on every batch size.
Timing runs the deployment form (`Pool` and `step_inplace`): CUDA kernel time from the torch profiler, 6 warm-up and 32
timed calls, **L2-cold** — eight disjoint slot sets are used in rotation.

## Deployment contract (beyond the benchmark)

As for the GDN step: read the checkpoint from the caller-owned pool shared with the flush (exact offsets in
`baseline.py::Pool`), addressed through an `idx` tensor (slot 0 = no sequence, zero output); take q / k / v as one bf16
row per sequence `(q | k | v)`; append in place (`kbuf` / `ubuf` / `gbuf` entry `h`, `w[h] = 1`, `pcum`, fill count
+ 1; a fill count of `L` means "just flushed, empty" and counts as 0); no device-wide synchronisation, host round trips,
JIT or autotuning inside the call, since it runs every decode step inside a captured CUDA graph.

## Validation results

`benchmark.py` on NVIDIA B200 (sm_100, 148 SMs), baseline from this directory:

| batch_size | programs | worst error / tolerance (structured / next step / random) | correctness | latency (ms) | memory bound (ms) |
| --- | --- | --- | --- | --- | --- |
| 16 | 512 | 0.030 / 0.032 / 0.076 | PASS | 0.0107 | 0.0027 |
| 32 | 1024 | 0.023 / 0.028 / 0.067 | PASS | 0.0143 | 0.0055 |
| 64 | 2048 | 0.037 / 0.041 / 0.081 | PASS | 0.0221 | 0.0109 |
| 128 | 4096 | 0.045 / 0.057 / 0.083 | PASS | 0.0381 | 0.0219 |
| 256 | 8192 | 0.041 / 0.079 / 0.080 | PASS | 0.0683 | 0.0437 |
| 512 | 16384 | 0.068 / 0.070 / 0.111 | PASS | 0.1278 | 0.0874 |

The memory bound is 34.1 KB per program (32.6 KB read, 1.5 KB written) at 6.54 TB/s, the bandwidth a large read-only
stream reaches on this GPU; small batches cannot reach that bandwidth, so their bound is optimistic.

## License

`baseline.py`, `benchmark.py` and the embedded reference are Apache-2.0 code by the request authors; this
documentation is Creative Commons Attribution 4.0 under the
[project license](https://github.com/NVlabs/kda/blob/main/LICENSE). The baseline is compiled with the separately
installed [TileLang](https://github.com/tile-ai/tilelang) (MIT) and runs on PyTorch (BSD-3-Clause); neither is bundled.
