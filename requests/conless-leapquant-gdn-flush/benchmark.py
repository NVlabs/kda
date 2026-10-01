# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing of a flush implementation against definition.json (see README.md for the criterion).

  python benchmark.py                      # the baseline in this directory
  python benchmark.py --impl my_flush.py   # a candidate exposing run(...) and, for timing, Pool / flush_inplace

Correctness (functional form `run`, every batch size in workloads.jsonl, three inputs each: a synthetic checkpoint with
edge-case heads, the checkpoint the reference produced from it, and unstructured in-domain random tensors):
the new checkpoint is decoded back to a V x K state and compared with the exactly rebuilt fp64 state, head by head:
    err = mean |decode(checkpoint) - S| over the head's V x K elements, for the candidate and for the reference.
    PASS  <=>  all outputs finite, codes in [-127, 127], and for every head
               err_candidate <= (1 + REL_TOL) * err_reference           if err_reference >= FLOOR * mean |S|
               err_candidate <= CAP * mean |S|                          otherwise (a head the format captures almost exactly)
The order of operations and the precision of intermediates are free; what is bounded is the accuracy of the
checkpoint.  Timing (deployment form `flush_inplace` on a caller-owned pool, every program due): CUDA kernel time
from the torch profiler, L2-cold -- NSETS disjoint slot sets are flushed in rotation so no call finds its data in L2.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

HERE = Path(__file__).resolve().parent
HV, K, V, L, R, QMAX = 32, 128, 128, 16, 4, 127.0
REL_TOL, FLOOR, CAP = 0.01, 1e-4, 1e-3       # 1 % more error than the reference; heads whose reference error is below 1e-4 |S| are held to 1e-3 |S|
NSETS, WARMUP, ITERS = 8, 4, 16
KB_PER_PROGRAM = 19 + 8 + 19 + 0.1          # checkpoint read + update buffer read + checkpoint write + resets
REF_TBPS = 6.3                               # read + write bandwidth of the stock vLLM fp32 decode kernel on B200 at batch 256 (the memory-bound yardstick)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def decode(codes, s_k, s_v, u, q):
    return (torch.einsum("bhrv,bhrk->bhvk", q.double(), u.double())
            + codes.double() * (s_k.double()[..., None, :] / QMAX) * s_v.double()[..., :, None])


def update_buffer(bs, dev, g):
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    kbuf = (rn(bs, HV, L, K) * 0.3).to(torch.bfloat16); ubuf = (rn(bs, HV, L, V) * 0.3).to(torch.bfloat16)
    w = torch.rand(bs, HV, L, device=dev, generator=g); p = torch.rand(bs, HV, device=dev, generator=g) * 0.5 + 0.5
    return kbuf, ubuf, w, p


def structured_inputs(bs, dev, seed=0):
    """A synthetic checkpoint (full-rank codes with positive scales, a low-rank part, one hot row) and a random update
    buffer.  Heads 1..6 are edge cases: rank 1, all zero, rank 2, tiny (1e-4), huge (3e2), rank 1 plus a noise floor."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, HV, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, HV, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, HV, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, HV, R, K) * 0.5).to(torch.float16); q = (rn(bs, HV, R, V) * 0.5).to(torch.float16)
    kbuf, ubuf, w, p = update_buffer(bs, dev, g)
    for h in (1, 2, 3, 6): codes[:, h] = 0; u[:, h] = 0; q[:, h] = 0
    w[:, 1, 1:] = 0; w[:, 2] = 0; w[:, 3, 2:] = 0; w[:, 6, 1:] = 0
    codes[:, 6] = (rn(bs, V, K) * 40).round().clamp(-127, 127).to(torch.int8); s_k[:, 6] *= 1e-6
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    q0 = torch.randn(V, R, generator=torch.Generator().manual_seed(0)).to(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0


def random_inputs(bs, dev, seed=1):
    """Unstructured tensors inside the deployment domain: positive scales, 0 < p <= 1, 0 <= w <= 1."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    lu = lambda *s: torch.exp(rn(*s) * 0.7)      # positive scales spread over about two decades; |S| stays in the deployment range (below 1e4)
    return (torch.randint(-127, 128, (bs, HV, V, K), device=dev, generator=g, dtype=torch.int8), lu(bs, HV, K) * 0.05, lu(bs, HV, V) * 0.3,
            rn(bs, HV, R, K, dt=torch.float16), rn(bs, HV, R, V, dt=torch.float16), rn(bs, HV, L, K, dt=torch.bfloat16),
            rn(bs, HV, L, V, dt=torch.bfloat16), torch.rand(bs, HV, L, device=dev, generator=g), torch.rand(bs, HV, device=dev, generator=g) * 0.999 + 0.001, rn(V, R))


def exact_state(inp):
    codes, s_k, s_v, u, q, kbuf, ubuf, w, p, q0 = inp
    return p.double()[..., None, None] * decode(codes, s_k, s_v, u, q) + torch.einsum("bhj,bhjv,bhjk->bhvk", w.double(), ubuf.double(), kbuf.double())


def judge(out, ref, S):
    finite = all(bool(torch.isfinite(t.float()).all()) for t in out[1:]); inrange = bool((out[0].int().abs() <= 127).all())
    e_out = (decode(*out) - S).abs().mean((0, 2, 3)); e_ref = (decode(*ref) - S).abs().mean((0, 2, 3)); mag = S.abs().mean((0, 2, 3))
    tight = (e_ref >= FLOOR * mag) & (e_ref > 0)                       # an all-zero head has e_ref = 0 and falls under the cap (which is then 0)
    ok = finite and inrange and bool((e_out[tight] <= (1 + REL_TOL) * e_ref[tight]).all()) and bool((e_out[~tight] <= CAP * mag[~tight]).all())
    ratio = (e_out[tight] / e_ref[tight]).max().item()                   # worst head among those compared relatively
    return ok, ratio


def check(bs, impl, ref_run, dev):
    first = structured_inputs(bs, dev, seed=bs)
    ref1 = ref_run(*[t.clone() for t in first])
    second = (*ref1, *update_buffer(bs, dev, torch.Generator(device=dev).manual_seed(bs + 1)), first[9])     # the next window, from the reference's checkpoint
    res = []
    for inp, ref in ((first, ref1), (second, None), (random_inputs(bs, dev, seed=bs + 2), None)):
        ref = ref_run(*[t.clone() for t in inp]) if ref is None else ref
        out = impl.run(*[t.clone() for t in inp]); torch.cuda.synchronize()
        res.append(judge(out, ref, exact_state(inp)))
    return res


def ktime(fn):
    for i in range(WARMUP): fn(i % NSETS)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(ITERS): fn(i % NSETS)
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type.name == "CUDA" and "fill" not in e.name.lower() and "mem" not in e.name.lower()]
    return sum(e.device_time for e in ev) / ITERS / 1000


def timing(bs, impl, dev):
    pool = impl.Pool(NSETS * bs + 8, dev)
    pool.raw.normal_(0, 0.1); pool.uq.normal_(0, 0.1)          # arbitrary checkpoint contents: timing only
    pool.kbuf.copy_(torch.randn(pool.kbuf.shape, device=dev)); pool.ubuf.copy_(torch.randn(pool.ubuf.shape, device=dev)); pool.w.uniform_(0, 1)
    q0 = torch.randn(V, R, generator=torch.Generator().manual_seed(0)).to(dev)
    idxs = [torch.arange(1 + j * bs, 1 + (j + 1) * bs, dtype=torch.int32, device=dev) for j in range(NSETS)]
    def call(j):
        pool.hcnt.index_fill_(0, idxs[j].long(), L)             # every program of the set is due (the kernel resets the counter)
        impl.flush_inplace(pool, idxs[j], q0)
    return ktime(call)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", default=str(HERE / "baseline.py")); ap.add_argument("--skip-timing", action="store_true")
    a = ap.parse_args(); dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0); assert any(m in name.split() for m in ("B200", "B300")), name
    d = json.loads((HERE / "definition.json").read_text()); ns = {}; exec(d["reference"], ns); ref_run = ns["run"]
    sizes = [json.loads(l)["workload"]["axes"]["batch_size"] for l in (HERE / "workloads.jsonl").read_text().splitlines() if line_ok(l)]
    impl = load(a.impl, "flush_impl")
    print(f"{name}, torch {torch.__version__}, impl {Path(a.impl).name}\n")
    print("| batch_size | programs | worst-head error ratio vs reference (structured / second window / random) | correctness | all-due latency (ms) | memory bound (ms) |")
    print("| --- | --- | --- | --- | --- | --- |")
    allok = True
    for bs in sizes:
        res = check(bs, impl, ref_run, dev); ok = all(r[0] for r in res); allok &= ok
        t_ms = float("nan") if a.skip_timing or not hasattr(impl, "flush_inplace") else timing(bs, impl, dev)
        bound = KB_PER_PROGRAM * 1024 * bs * HV / REF_TBPS / 1e12 * 1e3
        print(f"| {bs} | {bs * HV} | {res[0][1]:.4f} / {res[1][1]:.4f} / {res[2][1]:.4f} | {'PASS' if ok else 'FAIL'} | {t_ms:.4f} | {bound:.3f} |", flush=True)
    assert allok, f"correctness failed (limit: {1 + REL_TOL:.2f}x the reference's error per head, or {CAP:g} |S| where the reference error is below {FLOOR:g} |S|)"


def line_ok(l):
    return bool(l.strip())


if __name__ == "__main__":
    main()
