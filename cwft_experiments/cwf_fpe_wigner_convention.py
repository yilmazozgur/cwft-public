#!/usr/bin/env python3
"""UNIT TESTS PINNING THE WIGNER/FOURIER CONVENTION (manuscript sec:wigner).

Convention (one choice, used everywhere): the decoder is
    psi_v(x) = (1/N) sum_j v_j e^{-i x th_j}                (eq:decode)
and the frequency content of a profile is read with
    psihat(p) = (2 pi)^{-1/2} int psi(x) e^{+i p x} dx ,
so a decoded atom of codebook frequency th0 sits at p = +th0: the momentum
axis IS the codebook frequency axis, as the phase-space dictionary requires.
The matching Wigner--Ville distribution is
    W(x,p) = (1/pi) int psi*(x+s) psi(x-s) e^{-2 i p s} ds .
Consequences of the pairing with e^{-i x th}: a boost that RAISES p by p0 is
the modulation psi -> e^{-i p0 x} psi, and the shear that tilts p by +gamma*x
is psi -> e^{-i gamma x^2/2} psi.

Numerical mapping: with the discrete WV below (np.fft.fft over the correlation
r_m = psi(n+m) psi*(n-m)), the FFT frequency axis f = 2 pi fftfreq(M, dx)
corresponds to p = -f/2.  Three unit tests pin all signs:
 (1) PURE TONE: an atom at codebook carrier +6 -> WV band centre at p = +6.
 (2) MARGINALS: for a windowed Gaussian state, sum_p W = |psi|^2 exactly and
     sum_x W matches |psihat|^2 on the p grid.
 (3) SHEAR: e^{-i gamma x^2/2} tilts <p>(x0) with slope +gamma.
Seeded, CPU.  Writes cwf_fpe_wigner_convention_results.json.
"""
import json

import numpy as np


def wigner_ville(psi):
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n)
        m = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex)
        r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)


def wv_with_axis(psi, dx):
    """Discrete WV plus its TRUE p axis (p = theta convention): p = -f/2."""
    M = len(psi)
    W = np.fft.fftshift(wigner_ville(psi), axes=1)
    f = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(M, dx))
    p = -0.5 * f
    order = np.argsort(p)
    return W[:, order], p[order]


def psihat(psi, xs, ps):
    """psihat(p) = int psi(x) e^{+i p x} dx  (the convention's + sign)."""
    dx = xs[1] - xs[0]
    return (np.exp(1j * np.outer(ps, xs)) @ psi) * dx


def main():
    out = {}
    rng = np.random.default_rng(0)
    print("WIGNER/FOURIER CONVENTION UNIT TESTS (p = codebook frequency theta)")

    # ---- (1) pure tone: atom at carrier +6 -> WV band at p = +6 -------------
    th = 6.0 + 0.5 * rng.standard_normal(3000)
    xs = np.linspace(-10, 10, 400)
    dx = xs[1] - xs[0]
    psi = np.exp(-1j * np.outer(xs, th)).mean(1)          # decoded atom at x0=0
    W, p = wv_with_axis(psi, dx)
    pc = float(p[np.argmax(np.abs(W).sum(0))])
    ok1 = abs(pc - 6.0) < 0.2
    print(f"  (1) pure tone at carrier +6: WV band centre p = {pc:+.3f}  "
          f"{'PASS' if ok1 else 'FAIL'}")
    out["tone"] = dict(band_centre=pc, ok=bool(ok1))

    # ---- (2) marginals on a windowed Gaussian state -------------------------
    x0, th0, sx = 1.5, 6.0, 1.2
    psi_g = np.exp(-((xs - x0) ** 2) / (4 * sx ** 2)) * np.exp(-1j * th0 * xs)
    W, p = wv_with_axis(psi_g, dx)
    dp = p[1] - p[0]
    margx = W.sum(1) * dp
    Px = np.abs(psi_g) ** 2
    err_x = float(np.linalg.norm(margx / margx.max() - Px / Px.max())
                  / np.linalg.norm(Px / Px.max()))
    margp = W.sum(0) * dx
    Pp = np.abs(psihat(psi_g, xs, p)) ** 2
    err_p = float(np.linalg.norm(margp / margp.max() - Pp / Pp.max())
                  / np.linalg.norm(Pp / Pp.max()))
    mp = float((p * margp).sum() / margp.sum())
    ok2 = err_x < 0.05 and err_p < 0.05 and abs(mp - th0) < 0.1
    print(f"  (2) marginals: |x-marg - |psi|^2| = {err_x:.4f}, "
          f"|p-marg - |psihat|^2| = {err_p:.4f}, <p> = {mp:+.3f} (want +6)  "
          f"{'PASS' if ok2 else 'FAIL'}")
    out["marginals"] = dict(err_x=err_x, err_p=err_p, mean_p=mp, ok=bool(ok2))

    # ---- (3) shear: e^{-i gamma x^2/2} tilts <p> with slope +gamma ----------
    gamma = 0.5
    slopes = []
    x0s = np.linspace(-3, 3, 7)
    pbars = []
    for x0 in x0s:
        psi_g = np.exp(-((xs - x0) ** 2) / (4 * 1.0 ** 2)) * np.exp(-1j * 6.0 * xs)
        psi_c = psi_g * np.exp(-1j * gamma * xs ** 2 / 2)
        Pp = np.abs(psihat(psi_c, xs, np.linspace(0, 12, 600))) ** 2
        pg = np.linspace(0, 12, 600)
        pbars.append(float((pg * Pp).sum() / Pp.sum()))
    slope = float(np.polyfit(x0s, pbars, 1)[0])
    ok3 = abs(slope - gamma) < 0.05
    print(f"  (3) shear e^{{-i gamma x^2/2}}, gamma={gamma}: fitted d<p>/dx0 = "
          f"{slope:+.3f}  {'PASS' if ok3 else 'FAIL'}")
    out["shear"] = dict(gamma=gamma, slope=slope, ok=bool(ok3))

    ok = ok1 and ok2 and ok3
    print(f"  -> convention {'CONSISTENT' if ok else 'INCONSISTENT'}: "
          "decoded atoms sit at p = +theta; boost = e^{-i p0 x}; "
          "shear = e^{-i gamma x^2/2}.")
    out["all_ok"] = bool(ok)
    with open("cwf_fpe_wigner_convention_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_wigner_convention_results.json")


if __name__ == "__main__":
    main()
