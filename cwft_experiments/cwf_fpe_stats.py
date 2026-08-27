#!/usr/bin/env python3
"""STATISTICS for the Wigner-witness headline numbers.

Two questions the single-seed run (cwf_fpe_wigner.py) leaves open:
  (a) seed variation: floor / coherent / mixture / %removed across independent
      codebook draws (same construction: N=3000, omega0=4, sigma=1, cat at {2,8},
      grid 220 pts on [-6,16], 300-draw phase-randomized mixture);
  (b) convergence: the theoretical mixture removes 100% of the genuine negativity
      (cross terms vanish identically in expectation); the measured ~88% is a
      finite-ensemble residual.  Does %removed -> 100% as the ensemble grows?

Seeded, CPU.  Writes cwf_fpe_stats_results.json.
"""
import json

import numpy as np

N = 3000
OMEGA0, SIGMA = 4.0, 1.0
XS = np.linspace(-6, 16, 220)
CENTERS = [2.0, 8.0]
NT = 300


def make_freqs(rng):
    th = OMEGA0 + SIGMA * rng.standard_normal(N)
    return th[th > 0]


def profile(v, theta, xs):
    return (np.exp(-1j * np.outer(xs, theta)) @ v) / len(theta)


def bundle_profile(centers, theta, xs, phases=None):
    if phases is None:
        phases = np.zeros(len(centers))
    v = sum(np.exp(1j * p) * np.exp(1j * c * theta) for c, p in zip(centers, phases))
    return profile(v, theta, xs)


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


def negativity(W):
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def one_seed(seed, nt=NT):
    rng = np.random.default_rng(seed)
    th = make_freqs(rng)
    floor = negativity(wigner_ville(bundle_profile([5.0], th, XS)))
    coh = negativity(wigner_ville(bundle_profile(CENTERS, th, XS)))
    Wmix = np.zeros((len(XS), len(XS)))
    for _ in range(nt):
        ph = rng.uniform(0, 2 * np.pi, len(CENTERS))
        Wmix += wigner_ville(bundle_profile(CENTERS, th, XS, phases=ph))
    mix = negativity(Wmix / nt)
    removed = (coh - mix) / (coh - floor + 1e-12)
    # matched baseline: the EXACT incoherent mixture of this configuration --
    # cross terms drop analytically, so W_incoh = W(psi_1) + W(psi_2).  Measuring
    # removal against it (instead of the single-value floor) makes numerator and
    # denominator use the same estimator, capping removal at ~100%.
    Winc = (wigner_ville(bundle_profile([CENTERS[0]], th, XS)) +
            wigner_ville(bundle_profile([CENTERS[1]], th, XS)))
    incoh = negativity(Winc)
    removed_matched = (coh - mix) / (coh - incoh + 1e-12)
    return floor, coh, mix, removed, incoh, removed_matched


def main():
    out = {"params": dict(N=N, omega0=OMEGA0, sigma=SIGMA, centers=CENTERS,
                          grid=[float(XS[0]), float(XS[-1]), len(XS)], NT=NT)}
    # (a) seed variation
    res = np.array([one_seed(s) for s in range(8)])
    lab = ["floor", "coherent", "mixture", "frac_removed", "incoherent_baseline",
           "frac_removed_matched"]
    out["seed_variation"] = {k: dict(mean=float(res[:, i].mean()),
                                     std=float(res[:, i].std()))
                             for i, k in enumerate(lab)}
    print("seed variation (8 codebook draws; +/- is SD across seeds):")
    for k in lab:
        v = out["seed_variation"][k]
        print(f"  {k:>21}: {v['mean']:.4f} +/- {v['std']:.4f}")
    # (b) convergence of %removed with ensemble size (seed 0)
    conv = []
    for nt in [30, 100, 300, 1000]:
        r = one_seed(0, nt=nt)
        conv.append(dict(NT=nt, frac_removed=float(r[3]),
                         frac_removed_matched=float(r[5])))
        print(f"  NT={nt:>5}: frac_removed = {r[3]:.3f}  matched = {r[5]:.3f}")
    out["convergence"] = conv
    with open("cwf_fpe_stats_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_stats_results.json")


if __name__ == "__main__":
    main()
