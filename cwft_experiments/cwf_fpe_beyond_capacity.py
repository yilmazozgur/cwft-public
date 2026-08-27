#!/usr/bin/env python3
"""RELATIONAL READOUT BEYOND ITEM-RECOVERY CAPACITY (future-work route iii, tested).

Can the difference histogram of a bundle be read at loads where the items
themselves can no longer be recovered?  Design mirrors the holding-capacity
experiment (cwf_fpe_uncertainty.py part3): L = 4N well-separated grid cells
(sigma=4, spacing=3, K(spacing)~0), k of them stored, item readout by thresholded
similarity (sim > 0.5).  The planted relation: the k stored cells form k/2 pairs
separated by DLAG = 3 cells (Delta = 9.0).  Three curves vs load k:

  1. item recall    -- fraction of stored cells with sim > 0.5 (plus the FP rate
                       among the L-k non-stored cells);
  2. decode-route   -- threshold-decode the cell list (with its false positives),
                       then difference it: does lag DLAG win among lag multiples?
  3. direct spectral -- the |b|^2 autocorrelation readout (no decoding): does the
                       score at DLAG win among lag multiples 1..8 (excluding 0)?

Item capacity is ~0.07N; the question is whether curve 3 outlives curve 1.

TWO SIGNAL REGIMES ARE SWEPT, and the distinction is the whole scope of the result.
The error law Var[score] ~ k^2/N means detecting s coincident pairs at z standard
errors needs N > z^2 (k/s)^2.  So:

  * PROPORTIONAL (s = k/2): the requirement is independent of the load, and survival
    to arbitrary k is what the law PREDICTS -- this is the headline curve;
  * FIXED (s = S_FIXED regardless of k): the requirement grows as k^2, so detection
    must fail around k ~ s sqrt(N) / z.  Measuring this is what turns the headline
    from an unconditional claim into a conditional one.

Seeded, CPU.  Writes cwf_fpe_beyond_capacity_results.json.
"""
import json

import numpy as np

N = 1000
SIGMA = 4.0
SPACING = 3.0
DLAG = 3                    # planted pair displacement, in cells (Delta = 9.0)
LAG_CANDS = np.arange(1, 9)  # candidate lags (cells) for detection
TRIALS = 20
KS = [20, 40, 70, 140, 300, 600, 1000, 1600, 2400]
S_FIXED = 35                # planted pairs in the fixed-s arm (= k/2 at k=70)


L_LIB = 4 * N               # candidate library size
RANGE = L_LIB * SPACING     # value range, unchanged from the lattice design
DELTA = DLAG * SPACING      # planted lag in VALUE units (9.0)
MIN_GAP = 1.0               # ~3.4 kernel half-widths at SIGMA=4: values stay resolvable
HWHM = float(np.sqrt(2 * np.log(2)) / SIGMA)
LAG_GRID = np.arange(3.0, 27.0, HWHM / 2)   # continuous lag search
CHANCE = float(2 * HWHM / (LAG_GRID[-1] - LAG_GRID[0]))


def _draw_separated(rng, n, lo, hi, forbidden, min_gap):
    """n values in [lo,hi], each at least min_gap from every other and from forbidden."""
    out = []
    have = list(forbidden)
    tries = 0
    while len(out) < n and tries < 200 * n:
        c = rng.uniform(lo, hi)
        if all(abs(c - z) >= min_gap for z in have):
            out.append(c)
            have.append(c)
        tries += 1
    return np.array(out)


def one_trial(k, rng, n_pair=None):
    """One load point, on a CONTINUUM value support.

    The earlier design put every value on a lattice of step SPACING, so every
    pairwise difference was an exact multiple of it and the lag search was an
    8-way choice among commensurate candidates.  That is maximal difference
    degeneracy -- precisely the regime hypothesis (iii) of the sketch proposition
    EXCLUDES, and the regime the AGMS-parity Remark says inflates the phase
    sketch's variance.  Here the values are drawn from a continuum (with only a
    minimum-separation constraint, so items stay individually resolvable), the
    planted lag is a real number, and the lag search runs over a continuous grid.
    """
    th = SIGMA * rng.standard_normal(N)
    if n_pair is None:
        n_pair = k // 2
    n_pair = min(n_pair, k // 2)

    # planted pairs at the exact lag DELTA, on a continuum
    base = _draw_separated(rng, n_pair, 0.0, RANGE - DELTA - MIN_GAP, [], 2 * MIN_GAP)
    stored = np.concatenate([base, base + DELTA])
    # unpaired filler so the LOAD is k in both arms and only the SIGNAL differs
    n_fill = max(0, k - len(stored))
    if n_fill:
        filler = _draw_separated(rng, n_fill, 0.0, RANGE, list(stored), MIN_GAP)
        stored = np.concatenate([stored, filler]) if len(filler) else stored
    stored = np.sort(stored)

    # library = the stored values plus continuum distractors
    n_dis = max(0, L_LIB - len(stored))
    dis = _draw_separated(rng, n_dis, 0.0, RANGE, list(stored), MIN_GAP) if n_dis else np.array([])
    lib = np.sort(np.concatenate([stored, dis])) if len(dis) else np.sort(stored)
    is_stored = np.isin(lib, stored)

    b = np.exp(1j * np.outer(stored, th)).sum(0)

    # 1) item recall / false positives at the sim > 0.5 threshold
    sim = np.real(np.exp(1j * np.outer(lib, th)) @ b.conj()) / N
    pred = sim > 0.5
    recall = float((pred & is_stored).sum() / max(1, is_stored.sum()))
    fp = float((pred & ~is_stored).sum() / max(1, (~is_stored).sum()))

    # 2) decode route: difference histogram of the decoded (thresholded) list,
    #    scored on the same continuous lag grid with a one-HWHM tolerance
    dec_vals = lib[pred]
    ok_decode = False
    if len(dec_vals) >= 2:
        d = np.abs(dec_vals[None, :] - dec_vals[:, None])
        d = d[d > 0]
        counts = [np.sum(np.abs(d - lag) < HWHM) for lag in LAG_GRID]
        ok_decode = bool(abs(LAG_GRID[int(np.argmax(counts))] - DELTA) < HWHM)

    # 3) direct spectral readout over the same continuous lag grid (no decoding)
    score = (np.cos(np.outer(LAG_GRID, th)) @ (np.abs(b) ** 2)) / N
    ok_direct = bool(abs(LAG_GRID[int(np.argmax(score))] - DELTA) < HWHM)

    return recall, fp, ok_decode, ok_direct


def sweep(n_pair_of_k, label, seed_off):
    rows = []
    for k in KS:
        res = [one_trial(k, np.random.default_rng(seed_off + 31 * k + t),
                         n_pair=n_pair_of_k(k))
               for t in range(TRIALS)]
        rows.append(dict(k=k,
                         s=min(n_pair_of_k(k), k // 2),
                         recall=float(np.mean([r[0] for r in res])),
                         fp=float(np.mean([r[1] for r in res])),
                         decode=float(np.mean([r[2] for r in res])),
                         direct=float(np.mean([r[3] for r in res]))))
        r = rows[-1]
        print(f"  [{label}] k={k:>4} (s={r['s']:>4}): recall={r['recall']:.2f} "
              f"(fp={r['fp']:.3f})  decode={r['decode']:.2f}  direct={r['direct']:.2f}")
    return rows


def main():
    out = {"params": dict(N=N, sigma=SIGMA, spacing=SPACING, dlag=DLAG,
                          L=4 * N, trials=TRIALS, capacity_007N=0.07 * N,
                          s_fixed=S_FIXED, chance=1.0 / len(LAG_CANDS))}
    print(f"N={N} (item capacity ~ {0.07*N:.0f}); chance = 1/{len(LAG_CANDS)} "
          f"= {1/len(LAG_CANDS):.3f}")
    print("proportional signal (s = k/2): the regime the error law says should survive")
    rows = sweep(lambda k: k // 2, "prop", 1000)
    print(f"fixed signal (s = {S_FIXED}): the same readout, signal held at its value "
          f"at the operating point")
    rows_fixed = sweep(lambda k: S_FIXED, "fix ", 7000)
    out["rows"] = rows
    out["rows_fixed_s"] = rows_fixed
    with open("cwf_fpe_beyond_capacity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_beyond_capacity_results.json")


if __name__ == "__main__":
    main()
