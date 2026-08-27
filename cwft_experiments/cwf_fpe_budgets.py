"""
cwf_fpe_budgets.py -- WHICH BUDGET BINDS: the geometric vs statistical capacity
crossover of FPE bundling.  (Follows cwf_fpe_uncertainty.py part 3; the paper's
claim is k_max = min(statistical ~ 0.07 N, geometric ~ R sigma / hbar_c) with
hbar_c = sqrt(2 ln 2).)

Fix range R = 40.  For N in {512, 1024}, sweep the Gaussian codebook bandwidth
sigma over 12 log-spaced points in [0.03, 8] and measure the bundling capacity
k_max with the SAME criterion as cwf_fpe_uncertainty.py part 3 (recall >= 95%
of stored items read above threshold 0.5, false positives < 1% of non-stored
candidates, 8 trials per k, sequential scan + bisection refinement).

TWO ARMS, because the library's granularity turns out to be the whole question:

 ARM A (the literal transplant): library = 4N candidates on a uniform grid over
   [0, R] (spacing ~0.02).  MEASURED RESULT: degenerate -- k_max <= 1 at EVERY
   sigma in [0.03, 8] for both N.  Every candidate within the kernel half-width
   hbar_c/sigma of a stored item reads above threshold DETERMINISTICALLY, and at
   the largest sigma in the sweep that half-width (0.147) still covers ~15 grid
   spacings, so the 1% per-candidate false-positive budget is unmeetable.  An
   attribution-window repair (forgive candidates within ~1.75 half-widths of a
   stored item) was tried and is WORSE than degenerate: once k windows tile the
   range there are no far candidates left and the criterion goes vacuous.  A
   dense library and a <1% per-candidate FP budget are jointly incompatible with
   any finite-resolution code; this is reported, not hidden.

 ARM B (the paper's own design, generalized): library candidates at the CODE'S
   CELL GRANULARITY -- min-separation 2 * HWHM = 2 hbar_c / sigma (kernel
   similarity 0.0625 between neighbors, i.e. the first spacing that clears the
   0.5 threshold with the plateau's crosstalk margin), never finer than the 4N
   grid.  Part 3 was exactly this protocol at sigma = 4 (spacing 3.0 >>
   half-width).  The candidates are an RSA (random-sequential-adsorption)
   packing of [0, R], NOT a lattice: a periodic library makes every candidate
   pair share the same spacing, so one quenched near-commensurate lag of the
   empirical kernel poisons all pairs at once (measured: non-stored
   similarities up to 1.18 and bimodal capacity scans on the lattice; up to
   ~0.8 and stable scans under RSA).  RSA packs to ~the Renyi parking density
   (measured 0.75-0.78), so the geometric cell count is n_lib ~ 0.76 R sigma /
   (2 hbar_c).  Stored items are drawn uniformly from the library.  On the
   geometric branch the library EXHAUSTS (pigeonhole); the measurement there is
   the FILL FRACTION -- whether the code really delivers 95/1 readout at full
   cell occupancy.  k_max = median over 5 codebook+library seeds.

RESULTS (measured, this run; 16 trials/k, median over 5 seeds):
  - Small sigma: k_max rides the geometric line k = c * R * sigma with c =
    0.313 at BOTH N (= 0.37 of the ideal packing slope 1/hbar_c = 0.849;
    the RSA protocol predicts packing/(2 hbar_c) = 0.32-0.33 from the measured
    packing 0.75-0.78 -- the branch is exactly the cell budget at the tested
    granularity).  Fill fraction 1.00/0.99: the code delivers 95/1 readout at
    FULL cell occupancy right up to the crossover.  Below sigma ~ 0.06
    (cell wider than R) capacity is 0-1 and untestable.
  - Large sigma: statistical plateau -- N=512: 47.3 (0.092 N), N=1024: 78.3
    (0.076 N) over the non-exhausted points; part 3's 0.07 N band, with a mild
    decline in sigma consistent with the N/(2 ln n_lib) log-dependence as the
    library grows.  Per-seed scatter on the plateau remains large (quenched
    codebook effect, e.g. 29-92 at one point); the median is the estimator.
  - The kink: measured sigma* = 3.78 (N=512) and 6.26 (N=1024) vs the protocol
    prediction 2 hbar_c k_plateau / (packing R) = 3.55 / 6.17 -- within 7%,
    and the crossover scales up with N in proportion to the plateau exactly as
    min(a N, b R sigma) demands (ratio 1.66 both sides).
  min(statistical, geometric) structure: CONFIRMED (plateau + linear branch +
  N-scaled crossover); the geometric PREFACTOR is protocol-set (cell spacing
  tested at 2*HWHM, random packing 0.76), the ideal R sigma / hbar_c is its
  upper bound; and the dense-library literal transplant is degenerate.

CPU, numpy, seeded.  Writes cwf_fpe_budgets_results.json.
"""

import json
import numpy as np

R = 40.0
HBARC = float(np.sqrt(2 * np.log(2)))        # the phase-space cell, sqrt(2 ln 2)
NS = [512, 1024]
# 12 log-spaced points over the prescribed [0.03, 8], plus three above: under a
# fixed range R the cell-granular library only outgrows the statistical budget
# (i.e. the plateau only becomes measurable) at sigma >~ 2 sigma*, which for
# N=1024 is ~8.5 -- just past the prescribed sweep end.
SIGMAS = [float(s) for s in np.geomspace(0.03, 8.0, 12)] + [11.31, 16.0, 22.63]
TRIALS = 16                                  # 2x part-3's 8, to tame scan noise
THR = 0.5                                    # part-3 threshold, unchanged
SPACING_CELLS = 2.0                          # arm-B library spacing, in HWHM units
CODE_SEEDS = 5                               # codebook draws per (N, sigma);
# k_max is the MEDIAN across them.  Necessary and diagnosed, not cosmetic: on
# the statistical branch the FP rate sits knife-edge at the 1% criterion and is
# dominated by the QUENCHED codebook draw -- the lattice library admits rare
# near-resonant lags (empirical-kernel revivals with non-stored similarities
# measured up to 0.94), so single-seed sequential scans scatter wildly
# (k_max 17 vs 91 at adjacent sigmas in a pilot run).


def evaluate_k(Phi, k, N, rng):
    """Part-3 criterion literally: mean (recall, fp) over TRIALS bundles."""
    L = Phi.shape[0]
    rec, fp = [], []
    for _ in range(TRIALS):
        S = rng.choice(L, size=k, replace=False)
        b = Phi[S].sum(0)
        sim = np.real(Phi @ b.conj()) / N
        pred = sim > THR
        true = np.zeros(L, bool)
        true[S] = True
        rec.append((pred & true).sum() / k)
        fp.append((pred & ~true).sum() / max(1, (~true).sum()))
    return float(np.mean(rec)), float(np.mean(fp))


def kmax_scan(Phi, N, kcap, rng):
    """Sequential scan (break on first failure, as in part 3), bisection-refined.
    Returns (k_max, exhausted): exhausted = the scan passed every k up to kcap."""
    ks = list(range(1, 13)) + list(range(14, 25, 2))
    step = max(4, N // 64)
    ks += list(range(24 + step, int(0.14 * N) + 1, step))
    ks = [k for k in ks if k <= kcap]
    if not ks or ks[-1] < kcap:
        ks.append(kcap)
    kmax, fail = 0, None
    for k in ks:
        recm, fpm = evaluate_k(Phi, k, N, rng)
        if recm >= 0.95 and fpm < 0.01:
            kmax = k
        else:
            fail = k
            break
    if fail is not None and fail - kmax > 1:
        lo, hi = kmax, fail
        while hi - lo > 1:
            mid = (lo + hi) // 2
            recm, fpm = evaluate_k(Phi, mid, N, rng)
            if recm >= 0.95 and fpm < 0.01:
                lo = mid
            else:
                hi = mid
        kmax = lo
    return kmax, fail is None


def arm_A_dense(N, sigma):
    """Literal transplant: 4N-candidate uniform grid over [0,R]."""
    rng = np.random.default_rng([0, N, int(sigma * 1e6)])
    xs = np.linspace(0.0, R, 4 * N)
    theta = rng.normal(0, sigma, N)
    Phi = np.exp(1j * np.outer(xs, theta))
    kmax, _ = kmax_scan(Phi, N, kcap=int(0.14 * N), rng=rng)
    return kmax


def rsa_library(s, rng, proposals=4000):
    """Random sequential adsorption: candidates uniform in [0,R], accepted if
    >= s from every accepted one (near-jammed at ~the Renyi parking density,
    0.75 of the lattice count).  A PERIODIC (lattice) library is unusable here:
    all candidate pairs then share the same spacing, so one quenched near-
    commensurate lag of the empirical kernel poisons every pair at once --
    measured non-stored similarities up to 1.18 and bimodal k_max scans.
    Aperiodic packing kills that mechanism (max non-stored sim ~0.8)."""
    pts = []
    for x in rng.uniform(0, R, proposals):
        if all(abs(x - p) >= s for p in pts):
            pts.append(x)
    return np.sort(np.array(pts))


def arm_B_cell(N, sigma):
    """Cell-granular library: RSA packing at min-separation
    max(2*HWHM, R/(4N)) over [0,R].  Median k_max across CODE_SEEDS
    codebook+library draws (see header note)."""
    d_lib = max(SPACING_CELLS * HBARC / sigma, R / (4 * N))
    kms, exhs, nls = [], [], []
    for rep in range(CODE_SEEDS):
        rng = np.random.default_rng([1 + rep, N, int(sigma * 1e6)])
        xs = rsa_library(d_lib, rng)
        n_lib = len(xs)
        nls.append(n_lib)
        if n_lib < 2:
            kms.append(0)
            exhs.append(False)
            continue
        theta = rng.normal(0, sigma, N)
        Phi = np.exp(1j * np.outer(xs, theta))
        km, exh = kmax_scan(Phi, N, kcap=n_lib, rng=rng)
        kms.append(km)
        exhs.append(exh)
    kmax = int(np.median(kms))
    n_lib = int(np.median(nls))
    exhausted = bool(np.median(exhs) > 0.5)
    return kmax, kms, n_lib, exhausted


if __name__ == "__main__":
    print("=" * 74)
    print("WHICH BUDGET BINDS: geometric (R sigma / hbar_c) vs statistical (0.07 N)")
    print("=" * 74)
    print(f"R={R}, hbar_c={HBARC:.4f}, criterion: recall>=0.95 AND FP<0.01 at "
          f"threshold {THR}, {TRIALS} trials/k")
    print(f"arm A: dense library 4N over [0,R]; arm B: cell-granular RSA library, "
          f"min-sep {SPACING_CELLS}*HWHM (floored at the 4N grid); "
          f"{CODE_SEEDS}-seed median")

    out = {"params": dict(R=R, hbar_c=HBARC, Ns=NS, sigmas=SIGMAS, trials=TRIALS,
                          code_seeds=CODE_SEEDS,
                          kmax_estimator="median over codebook seeds",
                          threshold=THR, spacing_cells=SPACING_CELLS,
                          criterion="recall>=0.95 and FP<0.01 (part-3 criterion)",
                          arm_A="library = 4N uniform grid over [0,R] (literal)",
                          arm_B=("library = RSA (random sequential adsorption) "
                                 "packing of [0,R] at min-separation "
                                 "max(2*hbar_c/sigma, R/4N); stored items drawn "
                                 "uniformly from it; PERIODIC libraries are "
                                 "unusable (quenched kernel revivals, see "
                                 "rsa_library docstring)"),
                          note=("arm A is DEGENERATE (k_max<=1 everywhere): every "
                                "candidate within the kernel half-width of a stored "
                                "item fires deterministically, so the 1% per-"
                                "candidate FP budget is unmeetable on a dense grid; "
                                "an attribution-window repair goes vacuous at high "
                                "load (no far candidates left).  The geometric "
                                "budget is only measurable at cell granularity.")),
           "per_N": {}}

    for N in NS:
        print(f"\nN={N} (statistical prediction 0.07N = {0.07*N:.0f}, "
              f"N/(2 ln 4N) = {N/(2*np.log(4*N)):.0f})")
        print(f"  {'sigma':>7} | {'k_max':>5} | {'n_lib':>5} | {'exh':>3} | "
              f"{'geo ideal':>9} | {'ratio':>6} | denseA")
        rows = []
        for sigma in SIGMAS:
            kB, kms, n_lib, exh = arm_B_cell(N, sigma)
            kA = arm_A_dense(N, sigma)
            geo = R * sigma / HBARC
            d_lib = max(SPACING_CELLS * HBARC / sigma, R / (4 * N))
            rows.append(dict(sigma=sigma, k_max=kB, k_max_per_seed=kms,
                             n_lib=n_lib,
                             packing_fraction=n_lib * d_lib / R,
                             library_exhausted=bool(exh),
                             fill_fraction=kB / n_lib if n_lib else 0.0,
                             k_max_dense_armA=kA,
                             geometric_ideal=geo,
                             ratio_to_geometric=kB / geo if geo else 0.0,
                             hwhm=HBARC / sigma))
            print(f"  {sigma:>7.3f} | {kB:>5} | {n_lib:>5} | {str(exh)[0]:>3} | "
                  f"{geo:>9.1f} | {kB/geo:>6.2f} | {kA}   seeds {kms}")

        # statistical plateau: only NON-exhausted points count (an exhausted
        # point is pigeonhole-capped, i.e. still on the geometric branch)
        non_exh = [r for r in rows if r["k_max"] >= 1
                   and not r["library_exhausted"]]
        plateau = (float(np.mean([r["k_max"] for r in non_exh[-3:]]))
                   if len(non_exh) >= 2 else float("nan"))
        # geometric branch: exhausted points only; LS slope through the origin
        pts = [(r["sigma"], r["k_max"]) for r in rows
               if r["k_max"] >= 1 and r["library_exhausted"]]
        if pts:
            ss = np.array([p[0] for p in pts])
            kk = np.array([p[1] for p in pts])
            c = float((ss * kk).sum() / (R * (ss ** 2).sum()))
            sigma_star_meas = plateau / (c * R)
        else:
            c, sigma_star_meas = float("nan"), float("nan")
        fill = [r["fill_fraction"] for r in rows
                if r["k_max"] >= 1 and r["library_exhausted"]]
        packing = float(np.mean([r["packing_fraction"] for r in rows
                                 if r["n_lib"] >= 4]))
        summ = dict(plateau_k=plateau, plateau_over_N=plateau / N,
                    stat_prediction_007N=0.07 * N,
                    stat_prediction_capacity=N / (2 * np.log(4 * N)),
                    geometric_slope_c=c,
                    slope_ratio_to_ideal=c * HBARC,      # ideal packing = 1/hbar_c
                    lattice_bound_slope=1 / (SPACING_CELLS * HBARC),
                    mean_packing_fraction=packing,       # RSA ~ Renyi 0.75
                    expected_rsa_slope=packing / (SPACING_CELLS * HBARC),
                    mean_fill_fraction=float(np.mean(fill)) if fill else float("nan"),
                    n_branch_points=len(pts),
                    sigma_star_measured=sigma_star_meas,
                    sigma_star_predicted=(SPACING_CELLS * HBARC * plateau
                                          / (packing * R)),
                    sigma_star_ideal_line=HBARC * plateau / R,
                    dense_armA_kmax_max=max(r["k_max_dense_armA"] for r in rows))
        out["per_N"][N] = dict(rows=rows, summary=summ)
        print(f"  -> plateau k = {plateau:.1f} ({plateau/N:.3f} N); geometric branch "
              f"k = c*R*sigma, c = {c:.3f} ({c*HBARC:.2f} of ideal 1/hbar_c; "
              f"expected RSA slope {summ['expected_rsa_slope']:.3f}, "
              f"packing {packing:.2f}), fill = {summ['mean_fill_fraction']:.2f}")
        print(f"     kink: measured sigma* = {sigma_star_meas:.2f} vs predicted "
              f"{summ['sigma_star_predicted']:.2f} "
              f"(ideal-line {summ['sigma_star_ideal_line']:.2f}); "
              f"dense arm A: max k_max = {summ['dense_armA_kmax_max']} (degenerate)")

    s512 = out["per_N"][512]["summary"]
    s1024 = out["per_N"][1024]["summary"]
    print(f"\nVERDICT: min(statistical, geometric) structure holds -- plateau "
          f"{s512['plateau_over_N']:.3f}N/{s1024['plateau_over_N']:.3f}N at large "
          f"sigma, linear branch c*R*sigma at small sigma")
    print(f"         (c = {s512['geometric_slope_c']:.2f}/"
          f"{s1024['geometric_slope_c']:.2f}, protocol-set by 2*HWHM spacing), "
          f"crossover sigma* {s512['sigma_star_measured']:.2f} -> "
          f"{s1024['sigma_star_measured']:.2f} scales with N as predicted.")

    with open("cwf_fpe_budgets_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_budgets_results.json")
