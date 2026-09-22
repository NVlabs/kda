#!/usr/bin/env python3
"""Generate surrogate 3D FFT inputs whose radial power spectrum matches a
reference profile measured from real application grid data.

Why: the evaluation protocol feeds uniform random grids, whose spectrum is
white. Real workloads on structured grids concentrate energy in specific
wavenumber bands; implementations can behave differently on such inputs
(cancellation, dynamic range, near-zero high-k bins). The spectra/ profiles next to
this script capture that realism while containing **only** radially
averaged |X(k)|^2 (one number per integer wavenumber shell, normalized to
mean 1 over k >= 1, DC excluded). They carry no spatial, phase, or
time information; generated fields have random phase, so no structure from
the original data can be recovered from them.

Modes
-----
real (default)  real-space input grid for the r2c request
                output: float32 [nz, ny, nx], x innermost (saved via np.save)
--complex       packed Hermitian half-spectrum input for the c2r request
                output: complex64 [nz, ny, nx//2+1], x-half-spectrum layout
                (note: the imaginary part of the kx = 0 and kx = nx/2 planes
                is arbitrary; transforms ignore it)

Examples
--------
  # 64^3 real-space grid, amplitude matched to the reference workload:
  python3 gen_surrogate.py spectra/r2c_a.csv \
      --shape 64 64 64 --std 0.604 --seed 1 --out r2c_in_64.npy --verify

  # 144x96x96 half-spectrum input for c2r:
  python3 gen_surrogate.py spectra/c2r_b.csv \
      --shape 144 96 96 --complex --scale 4.07e-4 --seed 2 \
      --out c2r_in_144x96x96.npy --verify
"""
import argparse
import numpy as np


def load_profile(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 1]  # P_norm[k], k = 0..kmax


def radial_index(nx, ny, nz):
    """Integer wavenumber magnitude on the rfftn / packed half-spectrum grid.
    Axes are ordered (z, y, x); x is the transformed (halved) axis."""
    fx = np.arange(nx // 2 + 1)
    fy = np.minimum(np.arange(ny), ny - np.arange(ny))
    fz = np.minimum(np.arange(nz), nz - np.arange(nz))
    kz, ky, kx = np.meshgrid(fz, fy, fx, indexing="ij")
    return np.sqrt(kx**2 + ky**2 + kz**2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", help="CSV spectrum profile (k,P_norm)")
    ap.add_argument("--shape", type=int, nargs=3, required=True,
                    metavar=("nx", "ny", "nz"), help="grid size, x innermost")
    ap.add_argument("--std", type=float, default=1.0,
                    help="target grid standard deviation (real mode)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="target rms element magnitude (--complex mode)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--complex", action="store_true",
                    help="emit packed half-spectrum instead of real grid")
    ap.add_argument("--verify", action="store_true",
                    help="report spectrum agreement of the generated field")
    ap.add_argument("--out", required=True, help="output .npy path")
    a = ap.parse_args()

    nx, ny, nz = a.shape
    P = load_profile(a.profile)
    k = radial_index(nx, ny, nz)
    idx = np.clip(np.rint(k).astype(int), 0, len(P) - 1)
    amp = np.sqrt(P[idx])
    amp[0, 0, 0] = 0.0  # no DC (no net monopole)

    rng = np.random.default_rng(a.seed)
    half = (nz, ny, nx // 2 + 1)
    X = amp * (rng.standard_normal(half) + 1j * rng.standard_normal(half)) / np.sqrt(2)

    if a.complex:
        out = X
        out *= a.scale / np.sqrt(np.mean(np.abs(out) ** 2))
        out = out.astype(np.complex64)
    else:
        x = np.fft.irfftn(X, s=(nz, ny, nx), axes=(0, 1, 2))
        x -= x.mean()
        x *= a.std / x.std()
        out = x.astype(np.float32)
    np.save(a.out, out)

    if a.verify:
        if a.complex:
            Pm = np.abs(out.astype(np.complex128)) ** 2
        else:
            Pm = np.abs(np.fft.rfftn(out.astype(np.float64))) ** 2
        Pm = Pm.copy(); Pm[0, 0, 0] = 0.0
        kk = np.minimum(k.astype(int), len(P) - 1).ravel()
        w = np.bincount(kk, weights=Pm.ravel(), minlength=len(P))
        c = np.bincount(kk, minlength=len(P))
        ok = (c >= 8) & (np.arange(len(P)) > 0)
        meas = w[ok] / c[ok]
        scale = meas.sum() / P[ok].sum()  # fields match shape, not absolute gain
        rel = np.abs(meas / (P[ok] * scale) - 1.0)
        kind = "grid" if not a.complex else "spectrum"
        print(f"{a.out}: {kind}, rms-matched radial-spectrum median "
              f"rel-err over {ok.sum()} bins: {np.median(rel):.1%}, "
              f"90th pct: {np.percentile(rel, 90):.1%}")


if __name__ == "__main__":
    main()
