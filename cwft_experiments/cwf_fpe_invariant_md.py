"""
cwf_fpe_invariant_md.py -- translation-invariant recognition in m DIMENSIONS:
does the raw-|b|^2 route's advantage over decode-based routes grow with m?
(Extends cwf_fpe_invariant.py's chord task to m-D constellations.)

TASK: C=6 classes, each a fixed constellation of 4 points in [0,10]^m (fixed
seed, distinct pairwise-distance multisets, min intra-class point distance 2.0).
A test item is the class constellation translated by a random vector in
[0, 20]^m (absolute positions range over [0,30]^m) plus per-point jitter
N(0, 0.1) per axis.  Encoding: m-D FPE phi(x)_j = exp(i sum_a x_a theta_ja),
theta_ja ~ N(0, sigma^2), sigma=2, N=4000.  Three routes, m = 1,2,3,4:

 (a) RAW-|b|^2 templates: mean-centered cosine on the N-vector |b|^2 (exactly
     translation-invariant: a shift multiplies each b_j by a unit phasor).
     Memory: C x N floats.  No grid at all.
 (b) DECODE-THEN-COMPARE: matched-filter decode on a dense grid over [0,30]^m
     (B bins/axis: B=60 at m<=2; at m=3,4 the largest B keeping the grid under
     ~2e6 points -> B=125 (m=3), B=37 (m=4)), peak-pick 4 maxima with
     suppression (radius max(1.0, 1.5*bin)) and per-axis quadratic sub-bin
     refinement, canonicalize (subtract the min corner), nearest class by
     optimal point-set assignment (Hungarian).  Memory: the grid.
 (c) REFERENCE-UNBINDING: global argmax on the SAME grid as (b) -> unbind that
     anchor -> cosine against the C x 4 anchor-conditioned class templates
     (each class template anchored at each of its 4 points, since the argmax
     may land on any member).

The grid decode is computed ONCE per test item and shared by (b) and (c); its
time is included in both routes' reported time.

RESULTS (measured, this run; trials per m: 100/100/60/50):
  - Route (a) stays at accuracy 1.00 for ALL m = 1..4, at a flat 0.03-0.05 ms
    per classification and 0.19 MB of templates -- no dependence on m on any
    axis (the |b|^2 spectrum lives in the N-dim code space, never in x-space).
  - Route (b): accuracy also 1.00 at every m, but the grid explodes:
    60 -> 3.6e3 -> 1.95e6 -> 1.87e6 points and 0.45 -> 1.6 -> 325 -> 534 ms
    per classification (~1200x growth m=1->4; ~11,000x route (a) at m=4), with
    51-86 MB of decode-side buffers vs (a)'s 0.19 MB.  The bin-coarseness cost
    shows up in localization, not classification: mean decoded-point error
    0.020-0.031 at m<=2 (bin 0.51), 0.023 at m=3 (bin 0.24), 0.177 at m=4
    (bin 0.83) -- an 8x localization degradation at m=4 that these
    well-separated shapes (class sep 0.65-2.5) can absorb.
  - Route (c): rides the same grid (325-534 ms at m>=3 including the shared
    decode); accuracy 1.00 at every m -- the anchor misplacement at m=4
    (~0.18) stays within the sigma=2 kernel's tolerance.
  HONEST NOTE: (b)/(c) lose NO accuracy on this task -- the separation between
  class shapes is large relative to even the coarsest bins, so the honest
  headline is cost, not accuracy: the grid routes pay an exponential-in-m
  time/memory price for the same answer, while (a) is m-independent.  On a
  task with finer class differences (~0.2) the m=4 localization error (0.177)
  would start flipping decisions; that regime was not probed here.

CPU, numpy+scipy, seeded.  Writes cwf_fpe_invariant_md_results.json.
"""

import json
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

N = 4000
SIGMA = 2.0
C, P = 6, 4                       # classes, points per constellation
BOX = 10.0                        # class shapes live in [0,10]^m
R_SHIFT = 20.0                    # translations in [0,20]^m
JITTER = 0.1
GRID_LO, GRID_HI = 0.0, 30.0
GRID_CAP = 2_000_000              # max grid points at m>=3
TRIALS = {1: 100, 2: 100, 3: 60, 4: 50}
MIN_PT_DIST = 2.0                 # intra-class point separation (resolvable peaks)
CLASS_MARGIN = 0.5                # min distance between class distance-multisets


def make_classes(m, rng):
    """C constellations of P points in [0,10]^m with distinct difference
    structures (sorted pairwise-distance vectors separated by CLASS_MARGIN)."""
    classes, dvecs = [], []
    guard = 0
    while len(classes) < C:
        guard += 1
        if guard > 20000:
            raise RuntimeError("class generation stalled")
        pts = rng.uniform(0, BOX, (P, m))
        d = np.sort([np.linalg.norm(pts[i] - pts[j])
                     for i in range(P) for j in range(i + 1, P)])
        if d[0] < MIN_PT_DIST:
            continue
        if any(np.linalg.norm(d - dv) < CLASS_MARGIN for dv in dvecs):
            continue
        classes.append(pts)
        dvecs.append(d)
    sep = min(np.linalg.norm(a - b) for i, a in enumerate(dvecs)
              for b in dvecs[i + 1:])
    return classes, float(sep)


def bundle(pts, Theta):
    """sum_p exp(i Theta @ x_p); pts (P,m), Theta (N,m)."""
    return np.exp(1j * (pts @ Theta.T)).sum(axis=0)


def rawspec(b):
    Pw = np.abs(b) ** 2
    Pw = Pw - Pw.mean()
    return Pw / (np.linalg.norm(Pw) + 1e-12)


def decode_grid(b, E, n):
    """Similarity tensor Re[(1/n) sum_j b_j prod_a exp(-i g_a theta_ja)] on the
    per-axis grids encoded in E (list of (B,N) complex64 phase matrices)."""
    m, B = len(E), E[0].shape[0]
    b64 = b.astype(np.complex64)
    if m == 1:
        return (np.real(E[0] @ b64) / n).astype(np.float32)
    if m == 2:
        return (np.real((E[0] * b64) @ E[1].T) / n).astype(np.float32)
    if m == 3:
        S = np.empty((B, B, B), np.float32)
        for a in range(B):
            S[a] = np.real((E[1] * (b64 * E[0][a])) @ E[2].T) / n
        return S
    if m == 4:
        S = np.empty((B, B, B, B), np.float32)
        for a in range(B):
            w = b64 * E[0][a]
            Q = ((E[1] * w)[:, None, :] * E[2][None, :, :]).reshape(B * B, -1)
            S[a] = (np.real(Q @ E[3].T) / n).reshape(B, B, B)
        return S
    raise ValueError(m)


def refine(S, idx, grid):
    """Per-axis quadratic sub-bin refinement of a peak at multi-index idx."""
    h = grid[1] - grid[0]
    y = np.array([grid[i] for i in idx], float)
    for a in range(len(idx)):
        i = idx[a]
        if i == 0 or i == len(grid) - 1:
            continue
        lo = list(idx); lo[a] = i - 1
        hi = list(idx); hi[a] = i + 1
        sm, s0, sp = float(S[tuple(lo)]), float(S[tuple(idx)]), float(S[tuple(hi)])
        den = sm - 2 * s0 + sp
        if den < 0:
            y[a] += float(np.clip(0.5 * (sm - sp) / den, -0.5, 0.5)) * h
    return y


def peak_pick(S, coords, grid, npk, r_sup):
    """npk maxima with suppression radius r_sup, sub-bin refined."""
    flat = S.reshape(-1).copy()
    pts = []
    for _ in range(npk):
        i = int(np.argmax(flat))
        idx = np.unravel_index(i, S.shape)
        pts.append(refine(S, idx, grid))
        d2 = ((coords - coords[i]) ** 2).sum(axis=1)
        flat[d2 < r_sup ** 2] = -np.inf
    return np.array(pts)


def canon(pts):
    return pts - pts.min(axis=0)


def setmatch_cost(A, Bp):
    """Optimal-assignment total squared distance between two point sets."""
    cost = ((A[:, None, :] - Bp[None, :, :]) ** 2).sum(axis=2)
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].sum())


def loc_error(dec, true_pts):
    """Mean per-point distance between decoded and true points (canonicalized),
    under the optimal assignment -- the bin-coarseness diagnostic."""
    A, Bp = canon(dec), canon(true_pts)
    cost = np.sqrt(((A[:, None, :] - Bp[None, :, :]) ** 2).sum(axis=2))
    ri, ci = linear_sum_assignment(cost)
    return float(cost[ri, ci].mean())


def run_m(m, out):
    rng_shape = np.random.default_rng(7)        # fixed: class shapes
    rng_code = np.random.default_rng(1)         # fixed: codebook
    rng_trial = np.random.default_rng(2 + m)    # trial nuisances

    classes, sep = make_classes(m, rng_shape)
    Theta = rng_code.normal(0, SIGMA, (N, m))
    ntr = TRIALS[m]

    # grid for routes (b),(c)
    B = 60 if m <= 2 else int(GRID_CAP ** (1.0 / m))
    grid = np.linspace(GRID_LO, GRID_HI, B)
    binw = float(grid[1] - grid[0])
    r_sup = max(1.0, 1.5 * binw)
    E = [np.exp(-1j * np.outer(grid, Theta[:, a])).astype(np.complex64)
         for a in range(m)]
    G = B ** m
    mesh = np.meshgrid(*([grid.astype(np.float32)] * m), indexing="ij")
    coords = np.stack([g.reshape(-1) for g in mesh], axis=1)   # (G, m)

    # memory accounting (bytes)
    mem_a = C * N * 8                                          # spectra templates
    q_bytes = (B * B * N * 8) if m == 4 else (B * N * 8 * (2 if m >= 2 else 1))
    mem_b = G * 4 + m * B * N * 8 + coords.nbytes + q_bytes    # sim + phase mats
    mem_c = mem_b + C * P * N * 16                             # + anchored templates

    # ---- templates ----
    templ_a = [rawspec(bundle(cl, Theta)) for cl in classes]           # route a
    templ_bc = [canon(cl) for cl in classes]                           # route b
    templ_cu = []                                                      # route c
    for cl in classes:
        for p in range(P):
            t = bundle(cl - cl[p], Theta)
            templ_cu.append((t / np.linalg.norm(t), len(templ_cu) // P))
    templ_cu = [(t, ci) for t, ci in templ_cu]

    acc = dict(a=0, b=0, c=0)
    locerrs = []
    t_enc = t_a = t_dec = t_b = t_c = 0.0
    for _ in range(ntr):
        tcls = rng_trial.integers(C)
        shift = rng_trial.uniform(0, R_SHIFT, m)
        pts = classes[tcls] + shift + rng_trial.normal(0, JITTER, (P, m))

        t0 = time.perf_counter()
        b = bundle(pts, Theta)
        t_enc += time.perf_counter() - t0

        # (a) raw-|b|^2 template matching
        t0 = time.perf_counter()
        rs = rawspec(b)
        pred_a = int(np.argmax([rs @ t for t in templ_a]))
        t_a += time.perf_counter() - t0

        # shared grid decode
        t0 = time.perf_counter()
        S = decode_grid(b, E, N)
        t_dec += time.perf_counter() - t0

        # (b) peak-pick 4 -> canonicalize -> assignment matching
        t0 = time.perf_counter()
        dec_pts = peak_pick(S, coords, grid, P, r_sup)
        cd = canon(dec_pts)
        pred_b = int(np.argmin([setmatch_cost(cd, t) for t in templ_bc]))
        t_b += time.perf_counter() - t0
        locerrs.append(loc_error(dec_pts, pts))

        # (c) global-argmax anchor -> unbind -> cosine vs anchored templates
        t0 = time.perf_counter()
        i = int(np.argmax(S.reshape(-1)))
        y_star = refine(S, np.unravel_index(i, S.shape), grid)
        b_sh = b * np.exp(-1j * (Theta @ y_star))
        b_sh = b_sh / (np.linalg.norm(b_sh) + 1e-12)
        sims = [np.real(np.vdot(t, b_sh)) for t, _ in templ_cu]
        pred_c = templ_cu[int(np.argmax(sims))][1]
        t_c += time.perf_counter() - t0

        acc["a"] += pred_a == tcls
        acc["b"] += pred_b == tcls
        acc["c"] += pred_c == tcls

    row = dict(m=m, trials=ntr, B=B, grid_points=G, bin_width=binw,
               class_sep=sep, suppression_radius=r_sup,
               encode_ms=t_enc / ntr * 1e3,
               a=dict(acc=acc["a"] / ntr, ms=t_a / ntr * 1e3, mem_MB=mem_a / 1e6),
               b=dict(acc=acc["b"] / ntr, ms=(t_dec + t_b) / ntr * 1e3,
                      decode_ms=t_dec / ntr * 1e3, mem_MB=mem_b / 1e6,
                      loc_err_mean=float(np.mean(locerrs))),
               c=dict(acc=acc["c"] / ntr, ms=(t_dec + t_c) / ntr * 1e3,
                      mem_MB=mem_c / 1e6))
    out["per_m"].append(row)
    print(f"  m={m} (B={B}, grid {G:.0f} pts, bin {binw:.3f}, {ntr} trials, "
          f"class sep {sep:.2f}):")
    print(f"    (a) raw-|b|^2 : acc={row['a']['acc']:.2f}  {row['a']['ms']:8.2f} ms"
          f"  {row['a']['mem_MB']:9.2f} MB")
    print(f"    (b) decode    : acc={row['b']['acc']:.2f}  {row['b']['ms']:8.2f} ms"
          f"  {row['b']['mem_MB']:9.2f} MB  (decode {row['b']['decode_ms']:.2f} ms;"
          f" loc err {row['b']['loc_err_mean']:.3f})")
    print(f"    (c) ref-unbind: acc={row['c']['acc']:.2f}  {row['c']['ms']:8.2f} ms"
          f"  {row['c']['mem_MB']:9.2f} MB")


if __name__ == "__main__":
    print("=" * 74)
    print("TRANSLATION-INVARIANT RECOGNITION IN m DIMENSIONS: spectrum vs grid")
    print("=" * 74)
    print(f"C={C} classes x {P} points in [0,{BOX:.0f}]^m; shift U[0,{R_SHIFT:.0f}]^m;"
          f" jitter {JITTER}; N={N}, sigma={SIGMA}")
    out = {"params": dict(N=N, sigma=SIGMA, C=C, P=P, box=BOX, r_shift=R_SHIFT,
                          jitter=JITTER, grid=[GRID_LO, GRID_HI], grid_cap=GRID_CAP,
                          trials=TRIALS, min_point_dist=MIN_PT_DIST,
                          class_margin=CLASS_MARGIN,
                          notes=("grid decode shared by routes b and c and included"
                                 " in both times; sub-bin quadratic peak refinement;"
                                 " route-c templates anchored at each of the 4"
                                 " constellation points")),
           "per_m": []}
    for m in [1, 2, 3, 4]:
        run_m(m, out)

    a_acc = [float(r["a"]["acc"]) for r in out["per_m"]]
    b_ms = [r["b"]["ms"] for r in out["per_m"]]
    a_ms = [r["a"]["ms"] for r in out["per_m"]]
    print(f"\nVERDICT: route (a) accuracy {a_acc} -- flat in m at ~{np.mean(a_ms):.2f}"
          f" ms; grid routes blow up {b_ms[0]:.1f} -> {b_ms[-1]:.0f} ms "
          f"({b_ms[-1]/max(a_ms[-1],1e-9):.0f}x route (a) at m=4).")
    with open("cwf_fpe_invariant_md_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_invariant_md_results.json")
