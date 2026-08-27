#!/usr/bin/env python3
"""PHASE QUANTIZATION: where the coherence (and its uses) die on the way to
binary VSAs and hardware phases.

FHRR stores continuous phases; bipolar/binary VSAs and hardware phasor
implementations quantize them.  Sweep q = phase levels (component phases snapped to
the nearest multiple of 2pi/q; q=2 is the bipolar limit) and measure, vs q:

  (a) the genuine (excess-above-floor) Wigner negativity of a two-value bundle
      (the coherence order parameter of the manuscript's sec. 5);
  (b) the period-finding peak ratio (coherent peak at the planted lag / decohered
      value there) of sec. 6.

This turns the witness into a design tool: how many phase levels does a pipeline
need before the interference-borne capabilities are FHRR-grade?  4 codebook seeds
per point.  Writes cwf_fpe_quantize_results.json.
"""
import json

import numpy as np

N = 3000
OMEGA0, SIGMA = 4.0, 1.0
XS = np.linspace(-6, 16, 220)
CENTERS = [2.0, 8.0]
QS = [2, 3, 4, 8, 16, 64, 0]     # 0 = continuous (no quantization)
SEEDS = 40

# period-finding setup (sec. 6 parameters)
PF_SIGMA, PF_N, PF_DELTA, PF_PAIRS = 2.0, 4000, 3.0, 5
PF_DS = np.linspace(2, 12, 500)


def quantize(v, q):
    """Snap the phase of each unit component to the nearest multiple of 2pi/q."""
    if q == 0:
        return v
    ang = np.round(np.angle(v) * q / (2 * np.pi)) * (2 * np.pi / q)
    return np.exp(1j * ang)


def wigner_ville(psi):
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n)
        mm = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex)
        r[mm % M] = psi[n + mm] * np.conj(psi[n - mm])
        W[n] = np.fft.fft(r)
    return np.real(W)


def negativity(W):
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def one_seed(seed, q):
    rng = np.random.default_rng(seed)
    th = OMEGA0 + SIGMA * rng.standard_normal(N)
    th = th[th > 0]
    # (a) Wigner excess of the quantized two-value bundle
    atom = lambda c: quantize(np.exp(1j * c * th), q)
    dec = lambda v: (np.exp(-1j * np.outer(XS, th)) @ v) / len(th)
    floor = negativity(wigner_ville(dec(atom(5.0))))
    coh = negativity(wigner_ville(dec(atom(CENTERS[0]) + atom(CENTERS[1]))))
    excess = coh - floor
    # (b) period-finding peak ratio with atom phases quantized BEFORE bundling
    # (the hardware-relevant scheme: each stored atom carries q-level phases;
    # the bundle's component magnitudes |b_k| -- which carry the relational
    # signal -- are then formed from the quantized atoms).  Quantizing the
    # bundle itself to unit phasors instead destroys |b_k|^2 and with it the
    # relational readout at ANY q -- the signal lives in the magnitudes.
    rng2 = np.random.default_rng(1000 + seed)
    th2 = PF_SIGMA * rng2.standard_normal(PF_N)
    base = np.sort(rng2.uniform(0, 25, PF_PAIRS))
    vals = np.concatenate([base, base + PF_DELTA])
    b = sum(quantize(np.exp(1j * v * th2), q) for v in vals)
    score = (np.cos(np.outer(PF_DS, th2)) @ (np.abs(b) ** 2)) / PF_N
    i0 = np.argmin(np.abs(PF_DS - PF_DELTA))
    peak = float(score[i0 - 12:i0 + 13].max())
    bg = float(np.median(np.abs(score)))
    found = bool(abs(PF_DS[np.argmax(score)] - PF_DELTA) < 0.3)
    return excess, peak / (bg + 1e-9), found


def main():
    out = {"params": dict(N=N, omega0=OMEGA0, sigma=SIGMA, centers=CENTERS,
                          seeds=SEEDS, pf=dict(sigma=PF_SIGMA, N=PF_N,
                                               Delta=PF_DELTA, pairs=PF_PAIRS)),
           "rows": []}
    print("phase-quantization sweep (q = phase levels; q=2 bipolar, 0 = continuous)")
    for q in QS:
        res = np.array([one_seed(s, q) for s in range(SEEDS)])
        row = dict(q=q, excess_mean=float(res[:, 0].mean()),
                   excess_std=float(res[:, 0].std()),
                   peak_ratio=float(res[:, 1].mean()),
                   period_found=float(res[:, 2].mean()))
        out["rows"].append(row)
        print(f"  q={q if q else 'cont':>4}: Wigner excess = "
              f"{row['excess_mean']:+.3f} +/- {row['excess_std']:.3f}   "
              f"period peak/bg = {row['peak_ratio']:6.1f}  "
              f"found = {row['period_found']:.2f}")
    with open("cwf_fpe_quantize_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_quantize_results.json")


if __name__ == "__main__":
    main()
