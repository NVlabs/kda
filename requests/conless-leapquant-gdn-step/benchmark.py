# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing of a decode-step implementation against definition.json (see README.md for the criterion).

  python benchmark.py                      # the baseline in this directory
  python benchmark.py --impl my_step.py    # a candidate exposing run(...) and, for timing, Pool / step_inplace

Correctness (functional form `run`, every batch size in workloads.jsonl, three inputs each: a synthetic state with edge-case
heads, the step after it (the first step's update appended to the buffer), and unstructured in-domain random tensors):
    o, k_row, u_row : per (sequence, head) relative L2 error <= REL_L2 and max abs error <= REL_MAX * max |reference|,
                      or every element within one bf16 step of the reference (the head differs only by output rounding)
    w_new, p_new    : relative error <= 1e-5
The order of operations and the precision of intermediates are free.  Timing (deployment form `step_inplace` on a
caller-owned pool, every sequence decoding, ring filled to entry 1): CUDA kernel time from the torch profiler, L2-cold --
NSETS disjoint slot sets are used in rotation so no call finds its checkpoints in L2.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

HERE = Path(__file__).resolve().parent
HV, HK, K, V, L, R, QMAX = 32, 16, 128, 128, 16, 4, 127.0
REL_L2, REL_MAX, REL_SCALAR = 3e-3, 1e-2, 1e-5
NSETS, WARMUP, ITERS = 8, 6, 32
READ_B = V * K + (K + V) * 4 + R * (K + V) * 2 + L * (K + V) * 2 + L * 4 + (2 * K + V) * 2   # checkpoint + buffer + q/k/v per (sequence, head)
WRITE_B = 3 * 2 * V + 16                                                                      # output, appended key / value rows, scalars
REF_TBPS = 6.54                                                                               # read bandwidth the stock decode traffic reaches on B200 (large read-only stream)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def structured_inputs(bs, dev, seed=0):
    """A synthetic window state: full-range codes with positive scales, a hot row, Compensator Tokens, a partly filled buffer
    (h in [0, L)).  Heads 1..6 are edge cases: empty buffer and no residual, all-zero state, h = L - 1, tiny (1e-4), huge (3e2),
    rank-1 checkpoint."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, HV, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, HV, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, HV, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, HV, R, K) * 0.5).to(torch.float16); q = (rn(bs, HV, R, V) * 0.5).to(torch.float16)
    h = torch.randint(0, L, (bs, HV), device=dev, generator=g, dtype=torch.int32)
    kbuf = (rn(bs, HV, L, K) * 0.1).to(torch.bfloat16); ubuf = (rn(bs, HV, L, V) * 0.3).to(torch.bfloat16)
    w = torch.rand(bs, HV, L, device=dev, generator=g); p = torch.rand(bs, HV, device=dev, generator=g) * 0.5 + 0.5
    codes[:, 1] = 0; h[:, 1] = 0
    codes[:, 2] = 0; u[:, 2] = 0; q[:, 2] = 0; h[:, 2] = 0
    h[:, 3] = L - 1
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    codes[:, 6] = 0; q[:, 6, 1:] = 0
    w = w * (torch.arange(L, device=dev) < h[..., None])          # entries j >= h are empty
    qx = rn(bs, HK, K).to(torch.bfloat16); kx = rn(bs, HK, K).to(torch.bfloat16); vx = rn(bs, HV, V).to(torch.bfloat16)
    a = (rn(bs, HV) * 2).to(torch.bfloat16); b = (rn(bs, HV) * 2).to(torch.bfloat16)
    gg = torch.Generator().manual_seed(1)
    A_log = (torch.randn(HV, generator=gg) * 0.5 - 1.0).to(dev); dt_bias = torch.randn(HV, generator=gg).to(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, w, p, h, qx, kx, vx, a, b, A_log, dt_bias


def next_step(inp, ref_out, dev, seed):
    """The following decode step: the reference's update appended to the buffer, fresh q / k / v and gates."""
    codes, s_k, s_v, u, q, kbuf, ubuf, w, p, h, qx, kx, vx, a, b, A_log, dt_bias = inp
    _, k_row, u_row, w_new, p_new = ref_out
    kbuf, ubuf = kbuf.clone(), ubuf.clone(); hh = h.long(); ar = torch.arange(h.shape[0], device=dev)[:, None]; hv = torch.arange(HV, device=dev)[None, :]
    kbuf[ar, hv, hh] = k_row; ubuf[ar, hv, hh] = u_row
    h2 = (h + 1).clamp_max(L - 1)                                   # heads already at L - 1 overwrite their last entry (still a valid state)
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    return (codes, s_k, s_v, u, q, kbuf, ubuf, w_new * (torch.arange(L, device=dev) < h2[..., None]), p_new, h2.to(torch.int32),
            rn(*qx.shape).to(torch.bfloat16), rn(*kx.shape).to(torch.bfloat16), rn(*vx.shape).to(torch.bfloat16),
            (rn(*a.shape) * 2).to(torch.bfloat16), (rn(*b.shape) * 2).to(torch.bfloat16), A_log, dt_bias)


def random_inputs(bs, dev, seed=1):
    """Unstructured tensors inside the deployment domain: positive scales, 0 < p <= 1, 0 <= w <= 1 with w_j = 0 for j >= h."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    h = torch.randint(0, L, (bs, HV), device=dev, generator=g, dtype=torch.int32)
    w = torch.rand(bs, HV, L, device=dev, generator=g) * (torch.arange(L, device=dev) < h[..., None])
    return (torch.randint(-127, 128, (bs, HV, V, K), device=dev, generator=g, dtype=torch.int8), torch.exp(rn(bs, HV, K) * 0.7) * 0.05,
            torch.exp(rn(bs, HV, V) * 0.7) * 0.3, rn(bs, HV, R, K, dt=torch.float16), rn(bs, HV, R, V, dt=torch.float16),
            rn(bs, HV, L, K, dt=torch.bfloat16), rn(bs, HV, L, V, dt=torch.bfloat16), w, torch.rand(bs, HV, device=dev, generator=g) * 0.999 + 0.001, h,
            rn(bs, HK, K, dt=torch.bfloat16), rn(bs, HK, K, dt=torch.bfloat16), rn(bs, HV, V, dt=torch.bfloat16),
            (rn(bs, HV) * 2).to(torch.bfloat16), (rn(bs, HV) * 2).to(torch.bfloat16), rn(HV) * 0.5 - 1.0, rn(HV))


def judge(out, ref):
    worst = 0.0; ok = True
    for x, y in zip(out[:3], ref[:3]):
        x, y = x.float(), y.float(); d = x - y
        rl2 = d.flatten(2).norm(dim=-1) / y.flatten(2).norm(dim=-1).clamp_min(1e-30)
        rmx = d.abs().flatten(2).amax(-1) / y.abs().flatten(2).amax(-1).clamp_min(1e-30)
        zero = y.flatten(2).abs().amax(-1) == 0                     # an all-zero reference head must come out all zero
        rl2 = torch.where(zero, (x.flatten(2).abs().amax(-1) > 0).float() * 1e9, rl2); rmx = torch.where(zero, rl2, rmx)
        ulp = torch.exp2(torch.floor(torch.log2(y.abs().clamp_min(1e-30))) - 7)       # one bf16 step at the reference value
        in1ulp = (d.abs() <= ulp).flatten(2).all(-1)                                  # the head differs only by output rounding
        r = torch.maximum(rl2 / REL_L2, rmx / REL_MAX); r = torch.where(in1ulp, torch.minimum(r, torch.ones_like(r)), r)
        ok &= bool((r <= 1).all()) and bool(torch.isfinite(x).all())
        worst = max(worst, r.max().item())
    for x, y in zip(out[3:], ref[3:]):
        x, y = x.float(), y.float()
        e = ((x - y).abs() / y.abs().clamp_min(1e-30)).max().item(); ok &= e <= REL_SCALAR; worst = max(worst, e / REL_SCALAR)
    return ok, worst


def check(bs, impl, ref_run, dev):
    first = structured_inputs(bs, dev, seed=bs); r1 = ref_run(*[t.clone() for t in first])
    res = []
    for inp, ref in ((first, r1), (next_step(first, r1, dev, bs + 1), None), (random_inputs(bs, dev, seed=bs + 2), None)):
        ref = ref_run(*[t.clone() for t in inp]) if ref is None else ref
        out = impl.run(*[t.clone() for t in inp]); torch.cuda.synchronize()
        res.append(judge(out, ref))
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
    pool.raw.normal_(0, 0.1); pool.uq.normal_(0, 0.1)
    pool.kbuf.copy_(torch.randn(pool.kbuf.shape, device=dev)); pool.ubuf.copy_(torch.randn(pool.ubuf.shape, device=dev)); pool.w.uniform_(0, 1)
    idxs = [torch.arange(1 + j * bs, 1 + (j + 1) * bs, dtype=torch.int32, device=dev) for j in range(NSETS)]
    mixed = torch.randn(bs, 2 * HK * K + HV * V, device=dev).to(torch.bfloat16)
    a = (torch.randn(bs, HV, 1, device=dev) * 2).to(torch.bfloat16); b = (torch.randn(bs, HV, device=dev) * 2).to(torch.bfloat16)
    A_log = torch.randn(HV, device=dev) * 0.5 - 1.0; dt_bias = torch.randn(HV, 1, device=dev); o = torch.empty(bs, HV, V, dtype=torch.bfloat16, device=dev)
    def call(j):
        pool.hcnt.fill_(1)
        impl.step_inplace(pool, idxs[j], mixed, a, b, A_log, dt_bias, o)
    return ktime(call)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--impl", default=str(HERE / "baseline.py")); ap.add_argument("--skip-timing", action="store_true"); ap.add_argument("--bs", default="")
    a = ap.parse_args(); dev = torch.device("cuda")
    name = torch.cuda.get_device_name(0); assert any(m in name.split() for m in ("B200", "B300")), name
    d = json.loads((HERE / "definition.json").read_text()); ns = {}; exec(d["reference"], ns); ref_run = ns["run"]
    sizes = [json.loads(l)["workload"]["axes"]["batch_size"] for l in (HERE / "workloads.jsonl").read_text().splitlines() if l.strip()]
    if a.bs: sizes = [int(x) for x in a.bs.split(",")]
    impl = load(a.impl, "step_impl")
    print(f"{name}, torch {torch.__version__}, impl {Path(a.impl).name}\n")
    print("| batch_size | programs | worst error / tolerance (structured / next step / random) | correctness | latency (ms) | memory bound (ms) |")
    print("| --- | --- | --- | --- | --- | --- |")
    allok = True
    for bs in sizes:
        res = check(bs, impl, ref_run, dev); ok = all(r[0] for r in res); allok &= ok
        t_ms = float("nan") if a.skip_timing or not hasattr(impl, "step_inplace") else timing(bs, impl, dev)
        bound = bs * HV * (READ_B + WRITE_B) / REF_TBPS / 1e12 * 1e3
        print(f"| {bs} | {bs * HV} | {res[0][1]:.2f} / {res[1][1]:.2f} / {res[2][1]:.2f} | {'PASS' if ok else 'FAIL'} | {t_ms:.4f} | {bound:.4f} |", flush=True)
    assert allok, "correctness failed"


if __name__ == "__main__":
    main()
