#!/usr/bin/env python3
"""THE ACCIDENTAL-COINCIDENCE BACKGROUND: the N-independent half of the error.

The sketch error law (cwf_fpe_variance.py) prices the CODEBOOK sampling noise, which
shrinks as k/sqrt(N).  There is a second, N-INDEPENDENT error source that the law does
not model: the accidental-coincidence background.  For a given stored set the target

    T(d) = sum_{i,l} 1/2 [ K(delta_il - d) + K(delta_il + d) ]

is not exactly the planted pair count, because some unplanted pairs land within a
kernel width of the queried lag.  That contribution varies from stored set to stored
set and does NOT shrink with N -- no number of frequencies can average away a
coincidence that is really there in the data.

This script measures it, because the manuscript quoted two numbers for it that were
never committed to a result file:
  (1) its size at N=1000, k=200, against the codebook noise k/sqrt(N) = 6.32;
  (2) how fast it falls as the value range grows into the sparse regime.

Method: draw many stored sets at fixed (k, range), compute the EXACT large-N target
T(d) at a null lag (no planted pairs), and take its standard deviation across sets.
That is the set-to-set background, isolated from codebook noise by construction --
T(d) is the N -> infinity limit, so no sampling enters.

Seeded, CPU.  Writes cwf_fpe_background_results.json.
"""
import json

import numpy as np

SIGMA = 2.0                 # carrier codebook bandwidth (matches cwf_fpe_variance.py)
OMEGA0 = 8.0                # carrier centre
K = 200                     # stored items
N_REF = 1000                # reference dimension for the codebook-noise comparison
D_NULL = 1.7                # null lag (no planted pairs at this offset)
SETS = 400                  # stored sets per range
HWHM = float(np.sqrt(2 * np.log(2)) / SIGMA)


def target(vals, d):
    """Exact large-N target T(d): the kernel-smoothed symmetrized pair count.

    K(u) = Re E[e^{i u theta}] for theta ~ N(omega0, sigma^2) is
    cos(omega0 u) exp(-sigma^2 u^2 / 2).  Self-terms (i == l) are excluded: the
    proposition's hypothesis (ii) suppresses them.
    """
    du = vals[:, None] - vals[None, :]
    off = du[~np.eye(len(vals), dtype=bool)]

    def kern(u):
        return np.cos(OMEGA0 * u) * np.exp(-SIGMA ** 2 * u ** 2 / 2)

    return float(0.5 * (kern(off - d) + kern(off + d)).sum())


def main():
    rng = np.random.default_rng(20260728)
    out = {"params": dict(sigma=SIGMA, omega0=OMEGA0, k=K, N_ref=N_REF,
                          d_null=D_NULL, sets=SETS, hwhm=HWHM)}

    codebook_noise = K / np.sqrt(N_REF)
    print(f"accidental-coincidence background at k={K}, null lag d={D_NULL}")
    print(f"  codebook noise for comparison: k/sqrt(N) = {codebook_noise:.2f} "
          f"at N={N_REF}")
    print(f"  kernel half-width = {HWHM:.3f}; range quoted in half-widths and in k^2")

    rows = []
    for R in [400.0, 4000.0, 24000.0, 120000.0]:
        vals = [np.sort(rng.uniform(0, R, K)) for _ in range(SETS)]
        t = np.array([target(v, D_NULL) for v in vals])
        sd = float(np.std(t))
        rows.append(dict(range=R, hwhms=R / HWHM, hwhms_over_k2=R / HWHM / K ** 2,
                         background_sd=sd, background_mean=float(np.mean(t)),
                         ratio_to_codebook_noise=sd / codebook_noise))
        print(f"    range={R:>9.0f}  ({R/HWHM:>9.0f} half-widths = "
              f"{R/HWHM/K**2:>6.2f} k^2):  background SD = {sd:6.2f}  "
              f"({sd/codebook_noise:.2f}x the codebook noise)")
    base = rows[0]["background_sd"]
    for r in rows:
        r["drop_vs_smallest_range"] = base / r["background_sd"]
    print("  drop relative to the smallest range:")
    for r in rows:
        print(f"    range={r['range']:>9.0f}: {r['drop_vs_smallest_range']:.1f}x")
    print("  -> at a range of k^2 half-widths the background falls by the factor")
    print("     reported in the manuscript; >10x needs several k^2, not one.")
    out["rows"] = rows
    out["codebook_noise_at_N_ref"] = float(codebook_noise)
    with open("cwf_fpe_background_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_background_results.json")


if __name__ == "__main__":
    main()
