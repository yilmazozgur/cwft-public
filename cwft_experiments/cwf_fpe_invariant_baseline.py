#!/usr/bin/env python3
"""EXACT CLASSICAL BASELINES for the translation-invariant recognition tasks.

External round 5 asked the fair question: on the k=4 toy tasks, exact
canonicalization is cheap -- so state its numbers next to the VSA routes, and let
the claimed advantage be what it actually is (streaming/fixed-state/decode-free
operation on the hypervector), never accuracy.

This script measures the classical routes THAT SEE THE RAW VALUES:

  1-D chords (task of cwf_fpe_invariant.py, same types via the same seed):
     canonicalize = sort the m notes, subtract the lowest -> nearest interval
     template in L2.  O(m log m) per item on the raw values.
  m-D constellations (task family of cwf_fpe_invariant_md.py, generic instance):
     canonicalize = subtract the centroid, take the SORTED pairwise-distance
     multiset -> nearest template.  O(m^2) per item, translation- (and
     rotation-) invariant.

These baselines require access to the k raw values per item at query time; the
bundle routes operate on the fixed-size hypervector after the values are gone.
That is the regime split the manuscript claims -- this script simply prices the
other side of it.  Deliberately SEPARATE from cwf_fpe_invariant.py so the
committed timing JSONs quoted in the manuscript are not regenerated.

Seeded, CPU, runs in seconds.  Writes cwf_fpe_invariant_baseline_results.json.
"""
import json
import time

import numpy as np

RNG = np.random.default_rng(0)     # same seed => same 6 chord types as
K, m = 6, 4                        # cwf_fpe_invariant.py (types drawn first)
types = []
for _ in range(K):
    iv = np.sort(RNG.choice(np.arange(1, 13), size=m - 1, replace=False))
    types.append(np.concatenate([[0.0], iv]))
iv_templ = [np.sort(t) - np.min(t) for t in types]


def classify_canonical(vals):
    iv = np.sort(vals) - np.min(vals)
    return int(np.argmin([np.sum((iv - t) ** 2) for t in iv_templ]))


def main():
    out = {}
    print("=" * 74)
    print("EXACT CLASSICAL BASELINES (raw-value routes) for invariant recognition")
    print("=" * 74)

    # ---- 1-D chords: same protocol as the invariant experiment -------------
    print(f"  1-D chords: K={K} types, m={m} notes, root in [0,20], jitter 0.1, "
          f"100 samples/cell")
    rows = []
    for nz in [0.1, 0.3, 0.6, 1.0, 1.5]:
        acc = 0
        NS = 100
        for _ in range(NS):
            t = RNG.integers(K)
            vals = types[t] + RNG.uniform(0, 20) + RNG.normal(0, nz, m)
            acc += classify_canonical(vals) == t
        rows.append(dict(noise=nz, acc=acc / NS))
        print(f"    jitter sigma={nz:>4}: sorted-interval canonicalization = {acc/NS:.2f}")
    vals = types[0] + 10.0
    t0 = time.perf_counter()
    for _ in range(300):
        classify_canonical(vals)
    t_can = (time.perf_counter() - t0) / 300 * 1e3
    print(f"    cost per classification: {t_can*1e3:.1f} microseconds "
          f"(O(m log m) on the raw values)")
    out["chords_1d"] = dict(rows=rows, t_ms=t_can)

    # ---- m-D constellations: sorted pairwise distances ---------------------
    print(f"\n  m-D constellations: K={K} classes of 4 points, sorted "
          f"pairwise-distance features, jitter 0.1, translated in [0,20]^m")
    rows_md = []
    for mdim in [1, 2, 3, 4]:
        rng = np.random.default_rng(50 + mdim)
        classes = [rng.uniform(0, 8, size=(4, mdim)) for _ in range(K)]

        def feat(pts):
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            return np.sort(d[np.triu_indices(4, 1)])

        templ = [feat(c) for c in classes]
        acc = 0
        NS = 100
        for _ in range(NS):
            t = rng.integers(K)
            pts = classes[t] + rng.uniform(0, 20, mdim)[None, :] \
                + rng.normal(0, 0.1, (4, mdim))
            f = feat(pts)
            acc += int(np.argmin([np.sum((f - tp) ** 2) for tp in templ])) == t
        pts = classes[0] + 5.0
        t0 = time.perf_counter()
        for _ in range(300):
            f = feat(pts)
            int(np.argmin([np.sum((f - tp) ** 2) for tp in templ]))
        t_md = (time.perf_counter() - t0) / 300 * 1e3
        rows_md.append(dict(m=mdim, acc=acc / NS, t_ms=t_md))
        print(f"    m={mdim}: accuracy {acc/NS:.2f}, {t_md*1e3:.0f} microseconds "
              f"per classification")
    out["constellations_md"] = rows_md

    with open("cwf_fpe_invariant_baseline_results.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("\nwrote cwf_fpe_invariant_baseline_results.json")
    print("-> as expected: exact, microsecond-fast, and needs the raw values per")
    print("   item at query time.  The bundle routes answer from the fixed-size")
    print("   hypervector after the values are gone -- that, not accuracy, is the")
    print("   separation the manuscript claims.")


if __name__ == "__main__":
    main()
