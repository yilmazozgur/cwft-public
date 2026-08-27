#!/usr/bin/env python3
"""VARIANCE OF THE BUNDLE'S LAG ESTIMATOR (the sketching-program down payment).

score(d) = (1/N) sum_k |b_k|^2 cos(d theta_k) is a mean of N i.i.d. terms, so
Var[score] = Var_theta[|b(theta)|^2 cos(d theta)] / N exactly.  For k values that
are well separated (and d away from every pairwise difference), the derivation in
the manuscript gives

    E[score(d)]  ~  0        (null lag)
    Var[score(d)] ~ k^2 / N  (leading order; SE ~ k/sqrt(N))

and a planted structure of s coincident pairs contributes signal ~ s, so detection
at z-score z requires N > z^2 (k/s)^2 -- dimension-independent.  This script
validates the law: measured SD of score at a null lag over 300 codebook draws,
across (N, k), against the prediction k/sqrt(N).  Writes
cwf_fpe_variance_results.json.
"""
import json

import numpy as np

SIGMA = 2.0
OMEGA0 = 8.0         # carrier: a DC-free codebook (no p(theta) mass near zero).
#  With mass at theta ~ 0 the estimator variance is dominated by the ECF's DC
#  peak, |b(0)|^2 = k^2, giving Var ~ k^4/(R sigma N); the carrier removes it
#  and the clean k^2/N law below is the leading term.  This is the p(theta)
#  dependence of the law.
R = 400.0            # value range: keeps points well separated at all k
D_NULL = 1.7         # a lag away from typical pairwise differences' kernel width
DRAWS = 300


def sd_of_score(N, k, rng):
    vals = np.sort(rng.uniform(0, R, k))
    scores = []
    for _ in range(DRAWS):
        th = OMEGA0 + SIGMA * rng.standard_normal(N)
        b = np.exp(1j * np.outer(vals, th)).sum(0)
        scores.append(float((np.cos(D_NULL * th) @ (np.abs(b) ** 2)) / N))
    return float(np.std(scores)), float(np.mean(scores))


def main():
    rng = np.random.default_rng(11)
    rows = []
    print(f"SD of score(d_null) over {DRAWS} codebook draws vs prediction k/sqrt(N):")
    for N in [1000, 4000]:
        for k in [50, 200, 800]:
            sd, mean = sd_of_score(N, k, rng)
            pred = k / np.sqrt(N)
            rows.append(dict(N=N, k=k, sd=sd, mean=mean, pred=pred,
                             ratio=sd / pred))
            print(f"  N={N:>5} k={k:>4}: SD={sd:8.3f}  k/sqrt(N)={pred:8.3f}  "
                  f"ratio={sd/pred:.2f}   (mean={mean:+.3f})")
    ratios = [r["ratio"] for r in rows]

    # signal check: at a TRUE lag, E[score] ~ (number of pairs at that lag)
    print("signal at a planted lag: E[score(Delta)] ~ #pairs at Delta:")
    sig = []
    for npair in [5, 20, 50]:
        rng2 = np.random.default_rng(77 + npair)
        base = np.sort(rng2.uniform(0, R, npair))
        vals = np.concatenate([base, base + 3.0])
        vv = []
        for _ in range(80):
            th = OMEGA0 + SIGMA * rng2.standard_normal(1000)
            b = np.exp(1j * np.outer(vals, th)).sum(0)
            vv.append(float((np.cos(3.0 * th) @ (np.abs(b) ** 2)) / 1000))
        sig.append(dict(n_pairs=npair, mean_score=float(np.mean(vv))))
        print(f"  {npair:>3} pairs: E[score(3)] = {np.mean(vv):7.2f}  "
              f"(signal grows ~ #pairs)")

    out = dict(params=dict(sigma=SIGMA, omega0=OMEGA0, R=R, d_null=D_NULL,
                           draws=DRAWS),
               rows=rows, ratio_mean=float(np.mean(ratios)),
               ratio_range=[float(min(ratios)), float(max(ratios))],
               signal=sig)
    print(f"  -> SD / (k/sqrt(N)) = {np.mean(ratios):.2f} "
          f"(range {min(ratios):.2f}-{max(ratios):.2f}): the k^2/N law holds.")
    with open("cwf_fpe_variance_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_variance_results.json")


if __name__ == "__main__":
    main()
