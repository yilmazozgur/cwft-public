#!/usr/bin/env python3
"""HIGHER-ORDER INTERFERENCE: READING TRIPLE RELATIONS FROM ONE BUNDLE.

Second-order interference (|b|^2) reads pairwise differences.  This experiment
goes one order up.  With one extra NATIVE operation -- squaring each atom
before bundling (binding an atom with itself), which yields
b2_j = sum_i exp(i 2 th_j x_i) -- the third-order statistic

    score = (1/N) sum_j Re[ conj(b2_j) * b_j^2 ]
          -> sum_{i,l,n} K(x_i + x_l - 2 x_n)

is a kernel-smoothed AGGREGATE over 3-term arithmetic progressions
(x, x+u, x+2u) among the stored values -- one scalar summarizing an O(k^3)
family, read in O(N) once the two accumulators are built (build cost O(kNm)).
All reported scores are fully CENTERED: every diagonal family (i=l=n, i=l!=n,
i=n!=l, l=n!=i) is removed by inclusion-exclusion from the SAME two
accumulators (see ap_scores).

Findings (the split mirrors the second order, sharpened):
 (1) KNOWN CASE: one planted AP among separated decoys gives score - k ~ 2
     (each unordered AP contributes 2); decoherence kills it -- third-order
     interference is coherence-borne too.
 (2) IN 1-D THE STATISTIC IS BACKGROUND-DOMINATED (honest negative).  The
     kernel smoothing counts APPROXIMATE progressions, and a random 1-D set
     at practical density is full of them: measured centered background ~1700
     at k=70 versus ~20 for ten planted exact APs.  The needle is two orders
     of magnitude under the haystack (and in 1-D a histogram+FFT computes AP
     counts classically anyway).
 (3) IN 3-D THE DETECTION IS CLEAN.  Collinear coincidences are rare in
     higher dimension, the background collapses to ~0, and the SAME O(N)
     readout detects planted collinear triples (signal ~ 2s as predicted);
     decoherence kills it.  The third order pays exactly where the
     second-order analysis said the advantage lives: the dimension axis.
 (4) THE THIRD-ORDER ERROR LAW (empirical, codebook noise on fixed sets):
     null sd ~ N^(-1/2) (fitted), growing ~ k^(1.2-1.5) (fitted; the naive
     random-phase argument predicts 3/2).
 (5) COST SPLIT: dense grids pay B^m memory; sparse pair-midpoint hashing
     pays O(k^2 m); the bundle keeps a fixed 2N-component state (O(N) post-
     build readout; the O(kNm) build and O(Nm) codebook are the other costs).

Seeded, CPU.  Writes cwf_fpe_bispectrum_results.json.
"""
import json
import time

import numpy as np

N = 10000
SIGMA = 2.0
K_BG = 40
RANGE = 100.0
MIN_SEP = 1.0
TRIALS = 40
SEED = 0


def sep_sample(rng, k, lo, hi, min_sep):
    pts = []
    while len(pts) < k:
        c = rng.uniform(lo, hi)
        if all(abs(c - p) > min_sep for p in pts):
            pts.append(c)
    return np.array(pts)


def ap_scores(xs, th, rng=None, decohere=False):
    """(raw - k, fully CENTERED) third-order scores; xs (k,) or (k,m).

    raw = (1/N) sum_j Re[conj(b2_j) b_j^2] -> sum_{i,l,n} K(x_i + x_l - 2 x_n).
    Diagonal families and their exact removal (inclusion-exclusion, from the
    SAME two accumulators -- no third sketch needed):
      i=l=n  (k terms, K(0)=1)               ->  k
      i=l!=n: sum_{i!=n} K(2(x_i - x_n))     ->  S2 - k,  S2 = (1/N)sum|b2_j|^2
      i=n!=l and l=n!=i (K even)             ->  2(S1 - k), S1 = (1/N)sum|b_j|^2
    centered = raw - S2 - 2*S1 + 2k counts only distinct-index triples.  The
    corrections share the codebook with the raw score (correlated noise, priced
    into the measured error law).  In the decoherence control the per-atom
    phase alpha_i enters b as e^{i alpha} and b2 as e^{2 i alpha} (alpha is
    added to the phase matrix BEFORE doubling) -- the coherent transformation
    of a self-bound atom, not independent randomization."""
    ph = xs @ th.T if xs.ndim == 2 else np.outer(xs, th)
    if decohere:
        alpha = (rng.uniform(0, 2 * np.pi, ph.shape[0]))[:, None]
        ph = ph + alpha
    k = ph.shape[0]
    b = np.exp(1j * ph).sum(axis=0)
    b2 = np.exp(1j * 2 * ph).sum(axis=0)
    raw = float(np.mean(np.real(np.conj(b2) * b * b)))
    S1 = float(np.mean(np.abs(b) ** 2))
    S2 = float(np.mean(np.abs(b2) ** 2))
    return raw - k, raw - S2 - 2.0 * S1 + 2.0 * k


def ap_score(xs, th, rng=None, decohere=False):
    """The centered third-order score (all diagonal families removed)."""
    return ap_scores(xs, th, rng, decohere)[1]


def main():
    out = {"params": dict(N=N, sigma=SIGMA, k_bg=K_BG, range=RANGE,
                          min_sep=MIN_SEP, trials=TRIALS)}
    rng = np.random.default_rng(SEED)
    th = SIGMA * rng.standard_normal(N)

    # ---------------- (1) known case ----------------------------------------
    print("(1) known case: one planted AP among separated decoys")
    xs = np.array([10.0, 17.3, 24.6, 41.7, 56.3])   # AP: 10, 17.3, 24.6
    sc_raw, sc = ap_scores(xs, th)
    sc_d = np.mean([ap_score(xs, th, np.random.default_rng(100 + i), True)
                    for i in range(20)])
    print(f"    coherent centered = {sc:.2f} (expect ~2; raw-k = {sc_raw:.2f}); "
          f"decohered = {sc_d:.2f} (expect ~0)")
    out["known_case"] = dict(coherent=sc, coherent_raw=sc_raw,
                             decohered=float(sc_d))

    # ---------------- (2) 1-D: background-dominated (honest negative) --------
    print("(2) 1-D at practical density: approximate-AP background dominates")
    s = 10
    k_tot = K_BG + 3 * s
    null = [ap_score(sep_sample(np.random.default_rng(1000 + t), k_tot, 0,
                                RANGE, MIN_SEP), th) for t in range(20)]

    def planted_1d(r, s):
        xs = sep_sample(r, K_BG, 0, RANGE, MIN_SEP)
        for _ in range(s):
            a = r.uniform(5, 80)
            u = r.uniform(4, 9)
            xs = np.concatenate([xs, [a, a + u, a + 2 * u]])
        return xs

    null_raw = [ap_scores(sep_sample(np.random.default_rng(1000 + t), k_tot, 0,
                                     RANGE, MIN_SEP), th)[0] for t in range(20)]
    pl = [ap_score(planted_1d(np.random.default_rng(2000 + t), s), th)
          for t in range(20)]
    print(f"    matched count k={k_tot}: centered background = "
          f"{np.mean(null):.0f} +- {np.std(null):.0f}"
          f"  (raw-k background {np.mean(null_raw):.0f})")
    print(f"    with s={s} planted exact APs (signal ~ {2*s}): "
          f"{np.mean(pl):.0f} +- {np.std(pl):.0f}")
    print("    -> even fully centered (all diagonal families removed), the")
    print("       exact-AP signal (~20) sits far below the approximate-AP")
    print("       background among DISTINCT triples; 1-D detection of exact")
    print("       structure is not viable here (and histogram+FFT ties anyway)")
    out["oned_background"] = dict(k=k_tot, s=s,
                                  null_mean=float(np.mean(null)),
                                  null_sd=float(np.std(null)),
                                  null_mean_raw=float(np.mean(null_raw)),
                                  planted_mean=float(np.mean(pl)),
                                  planted_sd=float(np.std(pl)))

    # ---------------- (3) 3-D: clean detection ------------------------------
    print("(3) 3-D: the same O(N) readout, clean detection")
    th3 = SIGMA * rng.standard_normal((N, 3))

    def make_set_3d(r, s):
        pts = []
        while len(pts) < K_BG:
            c = r.uniform(0, 30, 3)
            if all(np.linalg.norm(c - p) > 2.0 for p in pts):
                pts.append(c)
        xs = np.array(pts)
        for _ in range(s):
            a = r.uniform(5, 20, 3)
            u = r.uniform(-4, 4, 3)
            u = u / np.linalg.norm(u) * r.uniform(5, 8)
            xs = np.vstack([xs, a, a + u, a + 2 * u])
        return xs

    # COUNT-MATCHED NULL.  make_set_3d(r, s) returns K_BG + 3s points, so a single
    # s=0 null holds FEWER items than every planted arm.  With the third-order noise
    # growing as k^1.4 (see the error law below), a threshold calibrated at k=K_BG is
    # too loose for the arms it is applied to.  We therefore build one null PER s,
    # holding the same k as that arm but with no planted progression, and calibrate
    # the 3-sigma threshold and the false-positive rate separately at each load.
    # (The 1-D arm above is already count-matched; this makes the two consistent.)
    def null_set_3d(r, s):
        """K_BG + 3s well-separated points, no planted progression."""
        pts = []
        while len(pts) < K_BG + 3 * s:
            c = r.uniform(0, 30, 3)
            if all(np.linalg.norm(c - p) > 2.0 for p in pts):
                pts.append(c)
        return np.array(pts)

    r3, thr_by_s = [], {}
    for s in [0, 2, 5, 10]:
        nulls = [ap_score(null_set_3d(np.random.default_rng(5000 + 71 * t + s), s),
                          th3) for t in range(TRIALS)]
        mu, sd = float(np.mean(nulls)), float(np.std(nulls))
        thr_by_s[s] = mu + 3 * sd
        if s == 0:
            mu3, sd3, th3sh = mu, sd, mu + 3 * sd
            r3.append(dict(s=0, mean=mu, sd=sd, detect=0.0, k=K_BG,
                           null_mean=mu, null_sd=sd, threshold=thr_by_s[0]))
            print(f"    s= 0: score-k = {mu:+.2f} +- {sd:.2f}  "
                  f"(k={K_BG}, 3-sigma threshold {thr_by_s[0]:.1f})")
            continue
        scs = [ap_score(make_set_3d(np.random.default_rng(6000 + 31 * t + s), s),
                        th3) for t in range(TRIALS)]
        det = float(np.mean([sc > thr_by_s[s] for sc in scs]))
        r3.append(dict(s=s, mean=float(np.mean(scs)), sd=float(np.std(scs)),
                       detect=det, k=K_BG + 3 * s, null_mean=mu, null_sd=sd,
                       threshold=thr_by_s[s]))
        print(f"    s={s:>2}: score-k = {np.mean(scs):+.2f} +- {np.std(scs):.2f}"
              f"  (k={K_BG + 3*s}, matched null {mu:+.2f} +- {sd:.2f}, "
              f"thr {thr_by_s[s]:.1f})  detection {det:.2f}  (signal ~ {2*s})")
    dec3 = [ap_score(make_set_3d(np.random.default_rng(7000 + t), 10), th3,
                     np.random.default_rng(7500 + t), True)
            for t in range(TRIALS)]
    det_dec = float(np.mean([sc > thr_by_s[10] for sc in dec3]))
    print(f"    s=10 DECOHERED: score-k = {np.mean(dec3):+.2f} +- "
          f"{np.std(dec3):.2f}  detection rate {det_dec:.2f}  "
          f"(the MEAN is removed; the SD is not -- it is larger than any coherent "
          f"arm's, which is why some decohered runs still cross)")
    # false-positive calibration at the LOAD THE ARM ACTUALLY HAS (k = K_BG + 30),
    # against that arm's own threshold -- not at k = K_BG against a looser one.
    null_fresh = [ap_score(null_set_3d(np.random.default_rng(8000 + t), 10), th3)
                  for t in range(100)]
    fp_rate = float(np.mean([sc > thr_by_s[10] for sc in null_fresh]))
    print(f"    null calibration: {len(null_fresh)} fresh nulls -> empirical "
          f"false-positive rate {fp_rate:.2f} at the 3-sigma threshold")
    out["detect_3d"] = dict(rows=r3, threshold=th3sh,
                            thresholds_by_s={str(k): v for k, v in thr_by_s.items()},
                            count_matched_nulls=True,
                            decohered_mean=float(np.mean(dec3)),
                            decohered_sd=float(np.std(dec3)),
                            decohered_detect=det_dec,
                            null_fp_rate=fp_rate, null_fp_n=len(null_fresh))

    # ---------------- (4) error law: codebook noise on fixed sets -----------
    print("(4) third-order error law (codebook noise, fixed 3-D sets):")

    def rand3(r, k, side):
        pts = []
        while len(pts) < k:
            c = r.uniform(0, side, 3)
            if all(np.linalg.norm(c - p) > 2.0 for p in pts):
                pts.append(c)
        return np.array(pts)

    def boot_slope(xvals, score_lists, rng, nboot=300):
        """slope of log(sd) vs log(x) + bootstrap 95% CI over codebook draws."""
        sds = [float(np.std(s)) for s in score_lists]
        slope = float(np.polyfit(np.log(xvals), np.log(sds), 1)[0])
        bs = []
        for _ in range(nboot):
            sdb = [np.std(rng.choice(s, size=len(s), replace=True))
                   for s in score_lists]
            bs.append(np.polyfit(np.log(xvals), np.log(sdb), 1)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return slope, float(lo), float(hi), sds

    ks = [20, 40, 80, 160]
    scores_k = []
    for k in ks:
        xs = rand3(np.random.default_rng(0), k, 30 * (k / 40) ** (1 / 3))
        scores_k.append(np.array([ap_score(xs, SIGMA * np.random
                        .default_rng(100 + i).standard_normal((N, 3)))
                        for i in range(20)]))
    slope_k, k_lo, k_hi, sds_k = boot_slope(ks, scores_k,
                                            np.random.default_rng(42))
    print("    sd vs k  (N=%d): " % N
          + ", ".join(f"k={k}: {s:.2f}" for k, s in zip(ks, sds_k))
          + f"  -> fitted exponent {slope_k:.2f} "
          f"[95% CI {k_lo:.2f}, {k_hi:.2f}] (naive prediction 1.5)")
    Ns = [2500, 10000, 40000]
    xs = rand3(np.random.default_rng(0), K_BG, 30)
    scores_n = []
    for n in Ns:
        scores_n.append(np.array([ap_score(xs, SIGMA * np.random
                        .default_rng(200 + i).standard_normal((n, 3)))
                        for i in range(20)]))
    slope_n, n_lo, n_hi, sds_n = boot_slope(Ns, scores_n,
                                            np.random.default_rng(43))
    print("    sd vs N  (k=%d): " % K_BG
          + ", ".join(f"N={n}: {s:.2f}" for n, s in zip(Ns, sds_n))
          + f"  -> fitted exponent {slope_n:.2f} "
          f"[95% CI {n_lo:.2f}, {n_hi:.2f}] (prediction -0.5)")
    out["error_law"] = dict(ks=ks, sd_k=sds_k, exp_k=slope_k,
                            exp_k_ci=[k_lo, k_hi],
                            Ns=Ns, sd_N=sds_n, exp_N=slope_n,
                            exp_N_ci=[n_lo, n_hi])

    # ---------------- (5) cost split ----------------------------------------
    print("(5) cost split (same shape as second order):")
    B = 600
    grid = {}
    for m in [1, 3, 4]:
        mem = 8 * B ** m
        grid[m] = mem
        print(f"    dense grid, m={m}: {mem/1e9:.3g} GB" if mem > 1e9 else
              f"    dense grid, m={m}: {mem/1e3:.3g} KB")
    sparse = {}
    for k in [200, 2000]:
        xs = np.sort(np.random.default_rng(9).uniform(0, 10 * k, k))
        t0 = time.perf_counter()
        cells = set(np.round(xs / 0.1).astype(int))
        cnt = sum(1 for i in range(k) for j in range(i + 1, k)
                  if round((xs[i] + xs[j]) / 2 / 0.1) in cells)
        dt = time.perf_counter() - t0
        sparse[k] = dt
        print(f"    sparse pair-midpoint hash, k={k}: {dt*1e3:.1f} ms "
              f"(O(k^2); found {cnt})")
    print(f"    bundle: two accumulators (b, b2) = 32N = {32*N/1e3:.0f} KB state;")
    print(f"    O(N) post-build readout at any m; build cost O(kNm)")
    out["costs"] = dict(grid_bytes={str(m): v for m, v in grid.items()},
                        sparse_seconds={str(k): v for k, v in sparse.items()})

    print("VERDICT: one native op (atom self-binding) opens the third order.")
    print("         1-D: background-dominated (and FFT ties) -- no.  3-D: the")
    print("         dimension-blind O(N) readout detects collinear structure")
    print("         cleanly, and decoherence kills it.  The split mirrors the")
    print("         second order, on the same two axes.")
    with open("cwf_fpe_bispectrum_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_bispectrum_results.json")


if __name__ == "__main__":
    main()
