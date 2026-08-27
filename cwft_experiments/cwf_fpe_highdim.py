#!/usr/bin/env python3
"""HIGH-DIMENSIONAL relational queries: where the grid dies and the bundle does not.

The 1-D complexity verdict (cwf_fpe_complexity.py) was a tie: a histogram->FFT
computes the difference histogram as fast as the bundle readout.  But the FPE bundle
is the empirical characteristic function (ECF) of the stored multiset sampled at N
random frequencies, while the histogram is the ECF on a REGULAR GRID -- and a grid
over an m-dimensional value domain costs B^m bins.  Monte-Carlo frequency sampling
(random features) does not.  This script measures the split.

Task (same relational query in every dimension m): a PLANTED bundle stores k = 20
points of which 10 pairs share a displacement vector Delta; a NULL bundle stores 20
unpaired points at the same density.  Detection: is the difference-histogram value
at Delta larger for the planted set than for the null set?  (Well-posed at every m;
in m = 1 the accidental-pair background is dense, so both methods degrade there
alike.)  Methods: (a) the FPE bundle |b|^2 readout (N = 4000 complex components for
EVERY m), (b) SPARSE pairwise bin-coincidence from the raw points (the honest
small-k classical baseline: O(k^2 m) time, O(km) memory, any m), (c) a DENSE
B-bins-per-axis histogram via shifted dot products (exact, non-circular; cost and
memory ~ B^m).  The B^m wall applies to (c) only; at small k, (b) wins everything --
the m-separation over dense grids is real, but the bundle-vs-best-classical contest
at small k belongs to the sparse route (see the manuscript's rescoped claim).

Seeded, CPU.  Writes cwf_fpe_highdim_results.json.
"""
import json
import time

import numpy as np
from scipy.ndimage import correlate1d

N = 4000          # bundle dimension, fixed for all m
SIGMA = 2.0       # frequency bandwidth per axis (kernel HWHM ~ 0.59)
R = 10.0          # value range per axis
B = 48            # histogram bins per axis (width 0.21 << kernel HWHM)
K_PAIRS = 10      # planted pairs sharing the displacement Delta
TRIALS = 40
MEM_CAP = 2e9     # bytes; grids beyond this are reported as estimated-only
# Grid resolutions to sweep.  B=48 (the headline setting) resolves 2.8x finer per
# axis than the bundle's own kernel half-width (sqrt(2 ln 2)/SIGMA = 0.589, i.e.
# R/0.589 ~ 17 bins), and that factor enters the memory as B^m.  Charging the grid
# for resolution the bundle never delivers would manufacture the wall, so we sweep
# B and report the SMALLEST grid that still meets the accuracy bar.
B_SWEEP = [12, 17, 24, 34, 48]
ACC_BAR = 0.95


def _shifted_dot2(Ha, Hb, sh, nb, m):
    """Non-circular shifted dot product of Ha with Hb at integer shift vector sh."""
    sl_a = tuple(slice(0, nb - s) for s in sh)
    sl_b = tuple(slice(s, nb) for s in sh)
    return float((Ha[sl_a] * Hb[sl_b]).sum())


def hist_score(pts, delta, m, nb=None):
    """Difference-histogram value at displacement delta from an nb^m grid.

    Two things here are needed to make this a FAIR baseline rather than a
    handicapped one, because the bundle's readout gets both for free:

    (1) KERNEL SMOOTHING.  The bundle computes cos(theta.delta) @ |b|^2, which is
        the pair count smoothed by the FPE kernel (Gaussian, std 1/sigma in value
        units).  A raw bin-coincidence count is not the same estimator: fine bins
        scatter coincidences that the bundle's kernel would still collect.  We
        therefore convolve H with the SAME Gaussian, separably -- O(B^m * m * w)
        time and no extra memory.
    (2) FRACTIONAL SHIFT.  The bundle is evaluated at the exact real delta; rounding
        the grid's shift to whole bins costs up to half a bin per axis (0.42 at
        B=12, against a kernel half-width of 0.59) and compounds across m axes.
        We shift by the exact real amount instead.

    Both are done in ONE pass by folding the fractional shift into the smoothing
    kernel: correlating the second copy with a Gaussian centred at the sub-bin
    offset is exactly a fractional translation, so no 2^m corner interpolation is
    needed and the cost stays O(B^m * m * w).
    """
    nb = B if nb is None else nb
    H = np.zeros((nb,) * m)
    idx = np.clip((pts / R * nb).astype(int), 0, nb - 1)
    np.add.at(H, tuple(idx.T), 1.0)

    raw = delta / R * nb
    lo = np.clip(np.floor(raw).astype(int), 0, nb - 1)
    frac = np.clip(raw - lo, 0.0, 1.0)
    sig_bins = max((1.0 / SIGMA) / (R / nb), 0.35)   # the bundle's own kernel width

    half = max(1, int(np.ceil(4 * sig_bins)))
    t = np.arange(-half, half + 1)

    def _gk(shift):
        g = np.exp(-0.5 * ((t - shift) / sig_bins) ** 2)
        return g / g.sum()

    Ha = H
    for ax in range(m):                      # copy A: centred kernel
        Ha = correlate1d(Ha, _gk(0.0), axis=ax, mode="constant")
    Hb = H
    for ax in range(m):                      # copy B: kernel offset by the sub-bin part
        Hb = correlate1d(Hb, _gk(frac[ax]), axis=ax, mode="constant")
    return _shifted_dot2(Ha, Hb, lo, nb, m)


def one_trial(m, rng):
    theta = rng.normal(0.0, SIGMA, (N, m))
    delta = rng.uniform(1.0, 3.0, m)                       # planted displacement
    base = rng.uniform(0.5, R - 3.5, (K_PAIRS, m))
    pts = np.vstack([base, base + delta])                  # planted: 2*K_PAIRS pts
    # DENSITY-MATCHED NULL.  The planted set is two clusters of K_PAIRS points, each
    # filling a box of side R-4, one displaced by delta.  Drawing the null uniformly
    # over the FULL box [0.5, R-0.5] instead gives it a (R-4)/(R-1) linear density
    # ratio -- 0.088 of the volume by m=6, so the planted set would carry ~5.7x more
    # ACCIDENTAL coincidences at every lag, delta included, and "detection" could be
    # won by density alone.  We therefore build the null from the same two boxes,
    # drawing both blocks independently so the support and density match exactly and
    # only the PAIRING is absent.
    null = np.vstack([rng.uniform(0.5, R - 3.5, (K_PAIRS, m)),
                      rng.uniform(0.5, R - 3.5, (K_PAIRS, m)) + delta])

    # ---- (a) bundle readout ------------------------------------------------
    t0 = time.perf_counter()
    b = np.exp(1j * pts @ theta.T).sum(0)                  # bundle, O(kN)
    t_build = time.perf_counter() - t0
    b0 = np.exp(1j * null @ theta.T).sum(0)
    cosd = np.cos(theta @ delta)
    t0 = time.perf_counter()
    s_pl = float(cosd @ (np.abs(b) ** 2)) / N              # ECF autocorr at delta
    t_bundle = time.perf_counter() - t0
    s_nu = float(cosd @ (np.abs(b0) ** 2)) / N
    ok_b = bool(s_pl > s_nu)

    # ---- (b) SPARSE pairwise route (the honest small-k classical baseline):
    #      exact continuous pairwise comparison from the k raw points -- no
    #      binning at all.  O(k^2 m) time, O(k m) memory, any m. -------------
    def sparse_score(P):
        diff = P[:, None, :] - P[None, :, :]               # (k,k,m)
        return int((np.abs(diff - delta) < 0.3).all(-1).sum())
    t0 = time.perf_counter()
    s_sp = sparse_score(pts)
    t_sparse = time.perf_counter() - t0
    ok_s = bool(s_sp > sparse_score(null))
    sparse_mem = pts.nbytes

    # ---- (c) DENSE grid (histogram) route ----------------------------------
    mem = (B ** m) * 8.0                                   # float64 grid
    if mem > MEM_CAP:
        sweep = {}
        for nb in B_SWEEP:
            mm = (nb ** m) * 8.0
            if mm > MEM_CAP:
                sweep[nb] = dict(ok=None, t=None, mem=mm)
                continue
            t0 = time.perf_counter()
            p = hist_score(pts, delta, m, nb)
            t = time.perf_counter() - t0
            sweep[nb] = dict(ok=bool(p > hist_score(null, delta, m, nb)),
                             t=t, mem=mm)
        return dict(ok_b=ok_b, t_bundle=t_bundle, t_build=t_build,
                    ok_s=ok_s, t_sparse=t_sparse, sparse_mem=sparse_mem,
                    ok_g=None, t_grid=None, mem=mem, sweep=sweep)
    t0 = time.perf_counter()
    g_pl = hist_score(pts, delta, m)
    t_grid = time.perf_counter() - t0
    g_nu = hist_score(null, delta, m)
    ok_g = bool(g_pl > g_nu)

    # ---- (d) the SAME grid at coarser resolutions, so the wall can be priced
    #      at matched accuracy rather than at a fixed bin count. --------------
    sweep = {}
    for nb in B_SWEEP:
        mm = (nb ** m) * 8.0
        if mm > MEM_CAP:
            sweep[nb] = dict(ok=None, t=None, mem=mm)
            continue
        t0 = time.perf_counter()
        p = hist_score(pts, delta, m, nb)
        t = time.perf_counter() - t0
        sweep[nb] = dict(ok=bool(p > hist_score(null, delta, m, nb)),
                         t=t, mem=mm)
    return dict(ok_b=ok_b, t_bundle=t_bundle, t_build=t_build,
                ok_s=ok_s, t_sparse=t_sparse, sparse_mem=sparse_mem,
                ok_g=ok_g, t_grid=t_grid, mem=mem, sweep=sweep)


def main():
    out = {"params": dict(N=N, sigma=SIGMA, R=R, B=B, k=2 * K_PAIRS,
                          trials=TRIALS)}
    rows = []
    print(f"k={2*K_PAIRS} points, {K_PAIRS} pairs sharing Delta; N={N} for all m; "
          f"grid B={B} bins/axis")
    for m in [1, 2, 3, 4, 5, 6]:
        res = [one_trial(m, np.random.default_rng(100 * m + t))
               for t in range(TRIALS)]
        acc_b = float(np.mean([r["ok_b"] for r in res]))
        tb = float(np.median([r["t_bundle"] for r in res]))
        acc_s = float(np.mean([r["ok_s"] for r in res]))
        ts = float(np.median([r["t_sparse"] for r in res]))
        smem = res[0]["sparse_mem"]
        grid_ok = [r["ok_g"] for r in res if r["ok_g"] is not None]
        acc_g = float(np.mean(grid_ok)) if grid_ok else None
        tg = (float(np.median([r["t_grid"] for r in res if r["t_grid"]]))
              if grid_ok else None)
        mem = res[0]["mem"]
        # B-sweep: smallest grid meeting the accuracy bar, and its cost
        sw_rows, best = [], None
        for nb in B_SWEEP:
            oks = [r["sweep"][nb]["ok"] for r in res if r["sweep"][nb]["ok"] is not None]
            a = float(np.mean(oks)) if oks else None
            t_ = ([r["sweep"][nb]["t"] for r in res if r["sweep"][nb]["t"] is not None])
            sw_rows.append(dict(B=nb, acc=a,
                                t_ms=(float(np.median(t_)) * 1e3 if t_ else None),
                                mem_bytes=res[0]["sweep"][nb]["mem"],
                                feasible=bool(oks)))
            if best is None and a is not None and a >= ACC_BAR:
                best = sw_rows[-1]
        rows.append(dict(m=m, acc_bundle=acc_b, t_bundle_ms=tb * 1e3,
                         acc_sparse=acc_s, t_sparse_ms=ts * 1e3,
                         sparse_mem_bytes=smem,
                         acc_grid=acc_g, t_grid_ms=(tg * 1e3 if tg else None),
                         grid_mem_bytes=mem, grid_feasible=bool(grid_ok),
                         b_sweep=sw_rows, grid_min_B_at_bar=best))
        gm = f"{mem/1e6:.1f} MB" if mem < 1e9 else f"{mem/1e9:.1f} GB"
        gs = (f"dense grid acc={acc_g:.2f} t={tg*1e3:.2f} ms" if grid_ok
              else "dense grid > mem cap")
        if best:
            bm = best["mem_bytes"]
            bms = f"{bm/1e6:.1f} MB" if bm < 1e9 else f"{bm/1e9:.1f} GB"
            print(f"    B-sweep: smallest grid at acc>={ACC_BAR}: B={best['B']} "
                  f"({bms}, {best['t_ms']:.3f} ms)")
        else:
            feas = [r for r in sw_rows if r["feasible"]]
            best_acc = max((r["acc"] for r in feas), default=None)
            print(f"    B-sweep: NO feasible grid reaches acc>={ACC_BAR} "
                  f"(best feasible acc={best_acc})")
        print(f"  m={m}: bundle acc={acc_b:.2f} t={tb*1e3:.2f} ms  |  "
              f"sparse acc={acc_s:.2f} t={ts*1e3:.3f} ms mem={smem} B  |  "
              f"{gs}  mem={gm}")
    out["rows"] = rows
    with open("cwf_fpe_highdim_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_highdim_results.json")


if __name__ == "__main__":
    main()
