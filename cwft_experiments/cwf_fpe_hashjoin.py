#!/usr/bin/env python3
"""THE MISSING SPARSE BASELINE: exact hash self-join vs the bundle readout.

External round 5 pointed out, correctly, that O(k^2 m) pair enumeration is NOT the
natural exact sparse comparator for a FIXED lag on quantized keys.  A hash table of
value frequencies f(u) is built in expected O(k) and answers the ordered self-join

    J_+(d) = sum_u f(u) f(u - d)

by one scan of the support: expected O(s), s = # distinct stored values.  For a
TOLERANCE query on continuous m-D values, the analogue is a spatial hash: bucket
points into cells of side ~ the kernel width and inspect the neighbor cells of each
translated query point -- O(k * 3^m) per lag, not O(k^2 m).  This script measures
both baselines against the bundle on the SAME tasks the manuscript already uses:

  PART 1 (exact, 1-D quantized): the planted-pair universe of cwf_fpe_ams.py
    (B = 4e6 grid, k/2 pairs at lag D0).  Methods: dict hash join, sort-merge join
    (the vectorized exact-sparse route), pair enumeration, FPE bundle query
    (N = 4000, cached key).  Sweep k from 40 to 400,000.
  PART 2 (tolerance, m-D continuous): the planted-displacement detection task of
    cwf_fpe_highdim.py (k = 20, m = 1..6; and a k-sweep at m = 3).  Methods:
    spatial-hash kernel join (cell side 2*HWHM, +/-1 neighbor cells, truncation
    checked against the exact smoothed score), exact pairwise smoothed score,
    FPE bundle readout.

Honest expectations, stated up front: the hash join should WIN or TIE on speed for
exact quantized lags at every k.  What the bundle keeps is (i) a state whose size
is independent of load, support and dimension (16N bytes vs 8(m+1) bytes per
distinct support point -- crossover at s ~ 2N/(m+1)); (ii) smoothed continuous-lag
queries with no quantization commitment at build time; (iii) mergeability;
(iv) lag-oblivious reuse.  Whatever the numbers say below goes in the paper.

In-script correctness asserts: hash join == sort join == brute force on random
multisets WITH duplicates; spatial-hash score within truncation tolerance of the
exact smoothed score.  Seeded, CPU.  Writes cwf_fpe_hashjoin_results.json.
"""
import json
import time
from collections import Counter

import numpy as np

# ---- Part 1 constants: match cwf_fpe_ams.py -------------------------------
B_UNIVERSE = 4_000_000
D0 = 7                       # planted lag
SIGMA_1D, OMEGA0 = 2.0, 8.0
N_BUNDLE = 4000
K_SWEEP = [40, 400, 4_000, 40_000, 400_000]
PAIR_CAP = 4_000             # pair enumeration beyond this: report as skipped
#                              (k^2 int64 differences: 128 MB at 4e3, 12.8 GB at 4e4)
REPS = 7                     # timing repetitions (median)

# ---- Part 2 constants: match cwf_fpe_highdim.py ---------------------------
SIGMA_MD = 2.0               # per-axis frequency bandwidth
HWHM = float(np.sqrt(2 * np.log(2)) / SIGMA_MD)      # 0.589
CELL = 2.0 * HWHM            # spatial-hash cell side; +/-1 cells => cutoff >= CELL
R_MD = 10.0                  # value range per axis
K_MD = 20
K_PAIRS = 10
TRIALS = 40
M_SWEEP = [1, 2, 3, 4, 6]
K_SWEEP_MD = [20, 200, 2000]


def make_set_1d(k, rng):
    """k distinct grid positions with EXACTLY k/2 ordered pairs at lag D0.

    Collision-free by construction (rejection sampling hangs at large k: with
    k/2 = 2e4 base points on a 4e6 grid, base and base+D0 overlap ~100 times
    per draw).  Bases are distinct multiples of 16 and partners sit at +7, so
    base values are 0 mod 16, partners 7 mod 16: no collisions, and the only
    pairwise difference equal to D0=7 is the planted one."""
    assert D0 == 7
    base = rng.choice(B_UNIVERSE // 16 - 1, size=k // 2, replace=False) * 16
    pos = np.concatenate([base, base + D0])
    return pos.astype(np.int64), k // 2


def hash_join(pos, d):
    """Ordered exact self-join J_+(d) via a frequency dict: expected O(s)."""
    counts = Counter(pos.tolist())
    return sum(c * counts.get(u - d, 0) for u, c in counts.items())


def sort_join(pos, d):
    """Same J_+(d) via sort + merge (the vectorized exact-sparse route), O(s log s)."""
    u, c = np.unique(pos, return_counts=True)
    idx = np.searchsorted(u, u - d)
    idx = np.clip(idx, 0, len(u) - 1)
    hit = u[idx] == (u - d)
    return int((c[hit] * c[idx[hit]]).sum())


def pair_enum(pos, d):
    """Brute-force ordered pair enumeration, O(k^2)."""
    diff = pos[:, None] - pos[None, :]
    return int((diff == d).sum())


def timeit(fn, reps=REPS):
    fn()                                  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3     # ms


def part1(out, rng):
    print("=" * 74)
    print("PART 1 -- exact quantized 1-D self-join J_+(d): hash join vs the rest")
    print("=" * 74)
    # correctness first: random multisets WITH duplicates
    for trial in range(20):
        r = np.random.default_rng(1000 + trial)
        vals = r.integers(0, 50, size=r.integers(5, 60)).astype(np.int64)
        d = int(r.integers(-10, 10))
        j_hash, j_sort, j_brut = hash_join(vals, d), sort_join(vals, d), pair_enum(vals, d)
        assert j_hash == j_sort == j_brut, (trial, d, j_hash, j_sort, j_brut)
    print("  correctness: hash == sort == brute force on 20 random multisets "
          "with duplicates  [OK]")

    th = OMEGA0 + SIGMA_1D * rng.standard_normal(N_BUNDLE)
    cos_key = np.cos(D0 * th)                       # cached query key, built once
    rows = []
    print(f"  {'k':>7} | {'hash O(s)':>10} {'sort':>8} {'pairs O(k^2)':>13} "
          f"{'FPE query':>10} {'FPE build':>10} | {'J_+(D0)':>8} | state bytes h/b")
    for k in K_SWEEP:
        pos, J_true = make_set_1d(k, np.random.default_rng(10 + k))
        t_hash = timeit(lambda: hash_join(pos, D0))
        t_sort = timeit(lambda: sort_join(pos, D0))
        t_pair = timeit(lambda: pair_enum(pos, D0)) if k <= PAIR_CAP else None
        # bundle: build once (timed, chunked to bound memory), cached-key query
        vals = pos.astype(float)
        t0 = time.perf_counter()
        b = np.zeros(N_BUNDLE, dtype=complex)
        for lo in range(0, len(vals), 2000):
            b += np.exp(1j * np.outer(vals[lo:lo + 2000], th)).sum(0)
        t_build = (time.perf_counter() - t0) * 1e3
        p = np.abs(b) ** 2
        t_fpe = timeit(lambda: float(cos_key @ p) / N_BUNDLE)
        j = hash_join(pos, D0)
        assert j == 2 * J_true or j == J_true, (j, J_true)   # ordered: both orientations? D0>0: J_+ = J_true
        s = len(np.unique(pos))
        state_hash = 16 * s                          # 8B key + 8B count per entry
        state_bundle = 16 * N_BUNDLE                 # N complex128
        rows.append(dict(k=k, s=s, J_true=int(J_true), J_hash=int(j),
                         t_hash_ms=t_hash, t_sort_ms=t_sort,
                         t_pair_ms=t_pair, t_fpe_query_ms=t_fpe,
                         t_fpe_build_ms=t_build,
                         state_hash_bytes=state_hash,
                         state_bundle_bytes=state_bundle))
        tp = f"{t_pair:>13.3f}" if t_pair is not None else f"{'skipped':>13}"
        print(f"  {k:>7} | {t_hash:>10.3f} {t_sort:>8.3f} {tp} "
              f"{t_fpe:>10.4f} {t_build:>10.1f} | {j:>8d} | {state_hash}/{state_bundle}")
    out["part1"] = dict(params=dict(B=B_UNIVERSE, D0=D0, N=N_BUNDLE,
                                    sigma=SIGMA_1D, omega0=OMEGA0, reps=REPS),
                        rows=rows,
                        state_crossover_s=N_BUNDLE)  # 16s > 16N  <=>  s > N
    print(f"  -> state crossover: dict bigger than the 64 KB bundle once the "
          f"support exceeds s = N = {N_BUNDLE} distinct values.")


# ---------------------------------------------------------------------------
def make_task_md(m, rng):
    """Planted vs null point sets for the displacement-detection task (highdim)."""
    delta = rng.uniform(2.0, 4.0, size=m)
    base = rng.uniform(0, R_MD, size=(K_PAIRS, m))
    planted = np.vstack([base, base + delta])
    null = rng.uniform(0, R_MD + np.mean(delta), size=(2 * K_PAIRS, m))
    return planted, null, delta


def exact_smoothed(pts, delta):
    """Exact kernel-smoothed shifted pair score: sum_{i,l} K(x_i - x_l - delta)."""
    diff = pts[:, None, :] - pts[None, :, :] - delta[None, None, :]
    return float(np.exp(-(SIGMA_MD ** 2) * (diff ** 2).sum(-1) / 2.0).sum())


def spatial_hash_build(pts):
    cells = {}
    keys = np.floor(pts / CELL).astype(np.int64)
    for i, key in enumerate(map(tuple, keys)):
        cells.setdefault(key, []).append(i)
    return cells


def spatial_hash_score(pts, cells, delta, m):
    """Kernel-weighted join by +/-1-cell neighbor scan around each x_i + delta."""
    from itertools import product as iproduct
    offsets = list(iproduct((-1, 0, 1), repeat=m))
    score = 0.0
    targets = pts + delta[None, :]
    tkeys = np.floor(targets / CELL).astype(np.int64)
    for i in range(len(pts)):
        t = targets[i]
        base = tkeys[i]
        for off in offsets:
            key = tuple(base + np.array(off))
            for jdx in cells.get(key, ()):
                u = t - pts[jdx]
                score += np.exp(-(SIGMA_MD ** 2) * float(u @ u) / 2.0)
    return score


def bundle_score(pts, delta, th_md):
    b = np.exp(1j * (pts @ th_md.T)).sum(0)          # th_md: (N, m)
    return float(np.cos(th_md @ delta) @ (np.abs(b) ** 2)) / len(th_md)


def detect(score_fn, rng_seed, m):
    """Accuracy of planted-vs-null detection at the midpoint threshold (TRIALS)."""
    rng = np.random.default_rng(rng_seed)
    sp, sn = [], []
    t_total = 0.0
    for _ in range(TRIALS):
        planted, null, delta = make_task_md(m, rng)
        t0 = time.perf_counter()
        sp.append(score_fn(planted, delta))
        sn.append(score_fn(null, delta))
        t_total += time.perf_counter() - t0
    thr = 0.5 * (np.median(sp) + np.median(sn))
    acc = 0.5 * (np.mean(np.array(sp) > thr) + np.mean(np.array(sn) <= thr))
    return float(acc), t_total / (2 * TRIALS) * 1e3  # per-score ms


def part2(out):
    print("=" * 74)
    print("PART 2 -- tolerance (smoothed) m-D join: spatial hash vs bundle")
    print("=" * 74)
    # truncation check: +/-1-cell scan vs exact smoothed score
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(30):
        m = int(rng.integers(1, 5))
        planted, _, delta = make_task_md(m, rng)
        cells = spatial_hash_build(planted)
        s_hash = spatial_hash_score(planted, cells, delta, m)
        s_exact = exact_smoothed(planted, delta)
        worst = max(worst, abs(s_hash - s_exact) / max(s_exact, 1e-9))
    print(f"  truncation check (+/-1 cells, side 2*HWHM): worst relative "
          f"deviation from the exact smoothed score = {worst:.3%}  [OK]")
    assert worst < 0.05

    rows_m = []
    print(f"  {'m':>3} | {'acc hash':>9} {'t/score ms':>10} | {'acc pairs':>9} "
          f"{'t ms':>7} | {'acc bundle':>10} {'t ms':>7}")
    for m in M_SWEEP:
        th_md = np.random.default_rng(100 + m).normal(0, SIGMA_MD, (N_BUNDLE, m))
        def hash_fn(pts, delta):
            return spatial_hash_score(pts, spatial_hash_build(pts), delta, len(delta))
        a_h, t_h = detect(hash_fn, 200 + m, m)
        a_p, t_p = detect(lambda p, d: exact_smoothed(p, d), 200 + m, m)
        a_b, t_b = detect(lambda p, d: bundle_score(p, d, th_md), 200 + m, m)
        rows_m.append(dict(m=m, acc_hash=a_h, t_hash_ms=t_h, acc_pairs=a_p,
                           t_pairs_ms=t_p, acc_bundle=a_b, t_bundle_ms=t_b,
                           neighbor_cells=3 ** m))
        print(f"  {m:>3} | {a_h:>9.3f} {t_h:>10.3f} | {a_p:>9.3f} {t_p:>7.3f} "
              f"| {a_b:>10.3f} {t_b:>7.3f}")

    rows_k = []
    m = 3
    th_md = np.random.default_rng(103).normal(0, SIGMA_MD, (N_BUNDLE, m))
    print(f"  k-sweep at m=3 (build+query per score):")
    print(f"  {'k':>6} | {'hash ms':>9} | {'pairs ms':>9} | {'bundle query ms':>15}")
    for k in K_SWEEP_MD:
        rng = np.random.default_rng(300 + k)
        delta = rng.uniform(2.0, 4.0, size=m)
        base = rng.uniform(0, R_MD * (k / 20) ** (1 / 3), size=(k // 2, m))
        pts = np.vstack([base, base + delta])       # constant density scaling
        cells = spatial_hash_build(pts)
        t_h = timeit(lambda: spatial_hash_score(pts, cells, delta, m), reps=3)
        t_p = timeit(lambda: exact_smoothed(pts, delta), reps=3) if k <= 2000 else None
        b = np.exp(1j * (pts @ th_md.T)).sum(0)
        p2 = np.abs(b) ** 2
        key = np.cos(th_md @ delta)
        t_b = timeit(lambda: float(key @ p2) / N_BUNDLE)
        rows_k.append(dict(k=k, t_hash_ms=t_h, t_pairs_ms=t_p, t_bundle_query_ms=t_b))
        tp = f"{t_p:>9.3f}" if t_p is not None else f"{'skipped':>9}"
        print(f"  {k:>6} | {t_h:>9.3f} | {tp} | {t_b:>15.4f}")
    out["part2"] = dict(params=dict(N=N_BUNDLE, sigma=SIGMA_MD, cell=CELL,
                                    k=K_MD, pairs=K_PAIRS, trials=TRIALS,
                                    truncation_worst=worst),
                        rows_m=rows_m, rows_k=rows_k)


def main():
    out = {}
    rng = np.random.default_rng(0)
    part1(out, rng)
    part2(out)
    with open("cwf_fpe_hashjoin_results.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("wrote cwf_fpe_hashjoin_results.json")
    print()
    print("VERDICT (to be reported as measured): for exact quantized fixed-lag")
    print("queries the hash join is the right sparse baseline and is expected to")
    print("win or tie at every k; the bundle's remaining edges are the fixed-size")
    print("state (crossover at support s ~ N), the smoothed continuous-lag query")
    print("model, mergeability, and lag-oblivious reuse.")


if __name__ == "__main__":
    main()
