# SPDX-License-Identifier: Apache-2.0
"""Correctness and timing of a KDA decode-step implementation against definition.json (see README.md for the criterion).

  python benchmark.py                      # the baseline in this directory
  python benchmark.py --impl my_step.py    # a candidate exposing run(...) and, for timing, Pool / step_inplace

Correctness (functional form `run`, every batch size in workloads.jsonl, three inputs each: a synthetic state with edge-case
heads, the step after it (the first step's update appended to the buffer), and unstructured in-domain random tensors), against
the definition's formula evaluated in fp64 (`exact`), element by element:
    o, k_row, u_row : |x - exact| <= half a bf16 step + TAU * M, with M = the same expression evaluated on the absolute values of
                      every term (the forward-error scale: o and u are sums that can cancel, so their own size is not a usable
                      denominator); an element whose terms are all zero must come out zero
    g_row           : |x - exact| <= half an fp16 step + TAU * |exact|
    w_new           : equal;  pcum_new : |x - exact| <= REL_SCALAR * (|exact| + 1)
Half a step is the rounding of the output itself; TAU is 200x what fp32 arithmetic in any order costs (the fp32 reference:
5e-8 M), so every intermediate has to carry about fp32 precision -- one rounded to bf16, fp16 or tf32 fails.
The order of operations is free.  Timing (deployment form `step_inplace` on a
caller-owned pool, every sequence decoding, buffer filled to entry 1): CUDA kernel time from the torch profiler, L2-cold --
NSETS disjoint slot sets are used in rotation so no call finds its checkpoints in L2.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch
from torch.profiler import profile, ProfilerActivity

HERE = Path(__file__).resolve().parent
H, K, V, L, R = 32, 128, 128, 16, 4
TAU, REL_SCALAR = 1e-5, 1e-5
QMAX = 127.0
NSETS, WARMUP, ITERS = 8, 6, 32
READ_B = V * K + (K + V) * 4 + R * (K + V) * 2 + L * (K + V) * 2 + L * K * 2 + L * 4 + K * 4 + (2 * K + V) * 2 + K * 2   # + gate history, cumulative gate, gate input
WRITE_B = 3 * 2 * V + K * 2 + K * 4 + 8                                                                                   # output, appended rows, decay factor, cumulative gate
REF_TBPS = 6.54


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


def gate_consts(dev):
    gg = torch.Generator().manual_seed(1)
    return (torch.log(torch.rand(H, generator=gg) * 1.5 + 0.5)).to(dev), (torch.randn(H, K, generator=gg) * 0.5 - 1.0).to(dev)


def structured_inputs(bs, dev, seed=0):
    """A synthetic window state: full-range codes with positive scales, a hot row, Compensator Tokens, a partly filled buffer
    (h in [0, L)) with per-channel decay histories in [0.5, 1] and a cumulative log-gate consistent with h.  Heads 1..6 are edge
    cases: empty buffer and no residual, all-zero state, h = L - 1, tiny (1e-4), huge (3e2), almost fully decayed checkpoint."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    codes = (rn(bs, H, V, K) * 40).round().clamp(-127, 127).to(torch.int8)
    s_k = torch.rand(bs, H, K, device=dev, generator=g) * 0.2 + 0.05; s_v = torch.rand(bs, H, V, device=dev, generator=g) * 0.5 + 0.1
    s_v[..., 5] *= 30
    u = (rn(bs, H, R, K) * 0.5).to(torch.float16); q = (rn(bs, H, R, V) * 0.5).to(torch.float16)
    h = torch.randint(0, L, (bs, H), device=dev, generator=g, dtype=torch.int32)
    kbuf = (rn(bs, H, L, K) * 0.1).to(torch.bfloat16); ubuf = (rn(bs, H, L, V) * 0.3).to(torch.bfloat16)
    codes[:, 1] = 0; h[:, 1] = 0
    codes[:, 2] = 0; u[:, 2] = 0; q[:, 2] = 0; h[:, 2] = 0
    h[:, 3] = L - 1
    s_k[:, 4] *= 1e-4; u[:, 4] *= 1e-2; ubuf[:, 4] *= 1e-4
    s_k[:, 5] *= 3e2; u[:, 5] *= 17; ubuf[:, 5] *= 3e2
    live = torch.arange(L, device=dev) < h[..., None]
    gbuf = torch.where(live[..., None], torch.rand(bs, H, L, K, device=dev, generator=g) * 0.5 + 0.5, torch.ones(bs, H, L, K, device=dev)).half()
    w = live.float()
    pcum = gbuf.float().log().sum(-2) - torch.rand(bs, H, K, device=dev, generator=g)       # log-gate since the window start
    pcum[:, 6] -= 200.0                                                                         # a checkpoint decayed to nothing
    qx = rn(bs, H, K).to(torch.bfloat16); kx = rn(bs, H, K).to(torch.bfloat16); vx = rn(bs, H, V).to(torch.bfloat16)
    a = rn(bs, H, K).to(torch.bfloat16); b = (rn(bs, H) * 2).to(torch.bfloat16)
    A_log, dt_bias = gate_consts(dev)
    return codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias


def next_step(inp, ref_out, dev, seed):
    """The following decode step: the reference's update appended to the buffer, fresh q / k / v and gates."""
    codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias = inp
    _, k_row, u_row, g_row, w_new, pcum_new = ref_out
    kbuf, ubuf, gbuf = kbuf.clone(), ubuf.clone(), gbuf.clone(); hh = h.long(); ar = torch.arange(h.shape[0], device=dev)[:, None]; hv = torch.arange(H, device=dev)[None, :]
    kbuf[ar, hv, hh] = k_row; ubuf[ar, hv, hh] = u_row; gbuf[ar, hv, hh] = g_row
    full = h >= L - 1                                               # heads already at L - 1 keep h (they overwrite their last entry)
    h2 = torch.where(full, h, h + 1)
    gbuf = torch.where((torch.arange(L, device=dev) < h2[..., None])[..., None], gbuf, torch.ones_like(gbuf))
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s: torch.randn(*s, device=dev, generator=g)
    return (codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w_new * (torch.arange(L, device=dev) < h2[..., None]), pcum_new, h2.to(torch.int32),
            rn(*qx.shape).to(torch.bfloat16), rn(*kx.shape).to(torch.bfloat16), rn(*vx.shape).to(torch.bfloat16),
            rn(*a.shape).to(torch.bfloat16), (rn(*b.shape) * 2).to(torch.bfloat16), A_log, dt_bias)


def random_inputs(bs, dev, seed=1):
    """Unstructured tensors inside the deployment domain: positive scales, decay factors in (0, 1] (1 past h), w = 1 below h."""
    g = torch.Generator(device=dev).manual_seed(seed)
    rn = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    h = torch.randint(0, L, (bs, H), device=dev, generator=g, dtype=torch.int32)
    live = torch.arange(L, device=dev) < h[..., None]
    gbuf = torch.where(live[..., None], torch.rand(bs, H, L, K, device=dev, generator=g) * 0.999 + 0.001, torch.ones(bs, H, L, K, device=dev)).half()
    A_log, dt_bias = gate_consts(dev)
    return (torch.randint(-127, 128, (bs, H, V, K), device=dev, generator=g, dtype=torch.int8), torch.exp(rn(bs, H, K) * 0.7) * 0.05,
            torch.exp(rn(bs, H, V) * 0.7) * 0.3, rn(bs, H, R, K, dt=torch.float16), rn(bs, H, R, V, dt=torch.float16),
            rn(bs, H, L, K, dt=torch.bfloat16), rn(bs, H, L, V, dt=torch.bfloat16), gbuf, live.float(), -torch.rand(bs, H, K, device=dev, generator=g) * 20, h,
            rn(bs, H, K, dt=torch.bfloat16), rn(bs, H, K, dt=torch.bfloat16), rn(bs, H, V, dt=torch.bfloat16),
            rn(bs, H, K, dt=torch.bfloat16), (rn(bs, H) * 2).to(torch.bfloat16), A_log, dt_bias)


def exact(inp, absval=False):
    """The definition's formula in fp64 (beta rounded to bf16 as the model carries it).  absval=True: the same expression with every
    term replaced by its absolute value (the scale M of o, k_row and u_row)."""
    codes, s_k, s_v, u, q, kbuf, ubuf, gbuf, w, pcum, h, qx, kx, vx, a, b, A_log, dt_bias = inp
    d = torch.float64; f = torch.abs if absval else (lambda t: t)
    Kd = codes.shape[-1]; Ln = kbuf.shape[-2]
    kf = kx.to(d); qf = qx.to(d)
    kn = kf * torch.rsqrt((kf * kf).sum(-1, keepdim=True) + 1e-6); qn = qf * torch.rsqrt((qf * qf).sum(-1, keepdim=True) + 1e-6) * Kd ** -0.5
    x = a.to(d) + dt_bias.to(d); g = -torch.exp(A_log.to(d))[:, None] * torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
    en = torch.exp(g); gn = pcum.to(d) + g; pc = torch.exp(gn); beta = torch.sigmoid(b.float()).to(torch.bfloat16).to(d)
    j = torch.arange(Ln, device=w.device)
    e = torch.where((j[:, None] < h[..., None, None]), gbuf.to(d), torch.ones_like(gbuf, dtype=d))
    after = torch.flip(torch.cumprod(torch.flip(e, [-2]), -2), [-2]); after = torch.cat([after[..., 1:, :], torch.ones_like(after[..., :1, :])], -2)
    D = en[..., None, :] * after * (j[:, None] < h[..., None, None])
    s0 = f(torch.einsum("bhrv,bhrk->bhvk", q.to(d), u.to(d))) if not absval else torch.einsum("bhrv,bhrk->bhvk", q.to(d).abs(), u.to(d).abs())
    s0 = s0 + f(codes.to(d) * (s_k.to(d) / QMAX)[..., None, :] * s_v.to(d)[..., :, None])
    s = s0 * pc[..., None, :] + torch.einsum("bhj,bhjv,bhjk->bhvk", f(w.to(d)), f(ubuf.to(d)), f(kbuf.to(d) * D))
    kk, qq = f(kn), f(qn)
    upd = beta[..., None] * (f(vx.to(d)) + (1 if absval else -1) * torch.einsum("bhvk,bhk->bhv", s, kk))
    y = torch.einsum("bhvk,bhk->bhv", s, qq) + (kk * qq).sum(-1, keepdim=True) * upd
    return y, kk, upd, en, gn


def half_step(t, mant, emin):
    """Half the spacing of a float format with `mant` stored mantissa bits (smallest exponent emin) at |t|."""
    _, ex = torch.frexp(t.abs())
    return torch.ldexp(torch.full_like(t, 0.5), torch.clamp(ex.int() - 1 - mant, min=emin - mant))


def judge(out, inp):
    """(ok, worst error / tolerance): error beyond the output's own rounding, divided by the allowance, maximised over elements."""
    ex = exact(inp); mags = exact(inp, absval=True); worst = 0.0; ok = all(bool(torch.isfinite(t.float()).all()) for t in out)
    for x, e, m in zip(out[:3], ex[:3], mags[:3]):
        x = x.double()
        r = ((x - e).abs() - half_step(torch.maximum(x.abs(), e.abs()), 7, -126)) / (TAU * m)
        r = torch.where(m == 0, (x != 0).double() * 1e9, r.clamp_min(0))       # all terms zero: the output must be zero
        worst = max(worst, r.max().item())
    x = out[3].double(); e = ex[3]
    worst = max(worst, (((x - e).abs() - half_step(torch.maximum(x.abs(), e.abs()), 10, -14)) / (TAU * e.abs())).clamp_min(0).max().item())
    if not torch.equal(out[4].float(), torch.where(torch.arange(out[4].shape[-1], device=out[4].device) == inp[10][..., None].long(), 1.0, inp[8].float())):
        worst = max(worst, 1e9)
    worst = max(worst, ((out[5].double() - ex[4]).abs() / (REL_SCALAR * (ex[4].abs() + 1))).max().item())
    return ok and worst <= 1, worst


def check(bs, impl, ref_run, dev):
    first = structured_inputs(bs, dev, seed=bs); r1 = ref_run(*[t.clone() for t in first])     # the reference's update seeds the next step
    res = []
    for inp in (first, next_step(first, r1, dev, bs + 1), random_inputs(bs, dev, seed=bs + 2)):
        out = impl.run(*[t.clone() for t in inp]); torch.cuda.synchronize()
        res.append(judge(out, inp))
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
    mixed = torch.randn(bs, 2 * H * K + H * V, device=dev).to(torch.bfloat16)
    a = torch.randn(bs, H, K, device=dev).to(torch.bfloat16); b = (torch.randn(bs, H, device=dev) * 2).to(torch.bfloat16)
    A_log, dt_bias = gate_consts(dev); o = torch.empty(bs, H, V, dtype=torch.bfloat16, device=dev)
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
        bound = bs * H * (READ_B + WRITE_B) / REF_TBPS / 1e12 * 1e3
        print(f"| {bs} | {bs * H} | {res[0][1]:.3f} / {res[1][1]:.3f} / {res[2][1]:.3f} | {'PASS' if ok else 'FAIL'} | {t_ms:.4f} | {bound:.4f} |", flush=True)
    assert allok, "correctness failed"


if __name__ == "__main__":
    main()
