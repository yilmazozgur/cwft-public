#!/usr/bin/env python3
"""GAP-HAMMING CALIBRATION CHECK: what the SMOOTHED readout actually equals on
the reduction instance, and how to correct it.

Companion to cwf_gaphamming_embedding.py, which verified the *combinatorics*
(J(D) = |A cap B| exactly, non-signal multiplicity <= 2).  This script checks the
*cost accounting*, which the manuscript's Remark got wrong.

The bundle does not compute the exact self-join J(D).  It computes the smoothed

    T(d) = sum_{i,l} (1/2) [ K(delta_il - d) + K(delta_il + d) ],   delta_il = x_i - x_l.

On the reduction instance (Alice streams S_a*M, Bob streams S_b*M + D, query lag
d = D) this decomposes EXACTLY into three pieces:

    T(D) = (1 + K(2D)) * |A cap B|          <- signal, with a MULTIPLICATIVE distortion
         + k * K(D)                          <- the k diagonal pairs i = l, at delta = 0
         + R,   |R| <= k^2 * sup_{|u| >= M-2D} |K(u)|      <- remote non-signal diffs

Only R depends on M.  The manuscript's condition k^2 sup_{|u|>=M-1}|K(u)| << eps*k
kills R and NOTHING ELSE.  At the paper's own codebook (sigma=2, omega0=8, kernel
half-width 0.59) and the paper's shift D=1, the self term is k*K(1) = -0.0197*k,
which exceeds the whole eps*k budget once eps < 0.02 -- the regime the reduction
runs in.  So the reduction as written is broken, and the fix is to choose the
shift D in KERNEL UNITS, not to set it to 1.

Checks performed:
  (A) the embedding still works for a general shift D (J(D) = |A cap B|, values
      distinct, non-signal multiplicity <= 2);
  (B) the exact three-term decomposition above, to machine precision;
  (C) naive (T) vs corrected estimators, over many random instances, for D = 1..4;
  (D) a finite-N Monte-Carlo run of the ACTUAL FPE bundle, confirming that
      E[score(D)] = T(D) and that the corrected estimator recovers |A cap B|
      inside the sampling noise k/sqrt(N).

Seeded, CPU.  Writes cwf_gaphamming_calibration_results.json.
"""
import json
from collections import Counter

import numpy as np

# ---- the paper's carrier codebook (cwf_fpe_variance.py / cwf_fpe_ams.py) ----
SIGMA = 2.0
OMEGA0 = 8.0
DELTA_X = np.sqrt(2 * np.log(2)) / SIGMA          # envelope HWHM = 0.5887 (paper: 0.59)

N_ELEM = 100          # universe [n]
M_MULT = 64           # spacing between Sidon slots (>> 2D, so R is exactly 0 here)
SHIFTS = (1, 2, 3, 4)  # shift D: 1 is the manuscript's; >=3 is the repair
INSTANCES = 500
SEED = 20260728

# finite-N Monte Carlo
MC_N = 100_000        # features
MC_DRAWS = 200        # codebook draws
MC_CHUNK = 5_000


def K(u):
    """Similarity kernel of the carrier-Gaussian codebook: Re E[exp(i u theta)]."""
    u = np.asarray(u, dtype=float)
    env = np.zeros_like(u)
    small = np.abs(u) < 40.0 / SIGMA               # else the envelope underflows to 0
    env[small] = np.exp(-0.5 * SIGMA ** 2 * u[small] ** 2)
    out = np.zeros_like(u)
    out[small] = np.cos(OMEGA0 * u[small]) * env[small]
    return out


def sidon_set(n):
    """n integers with all pairwise differences distinct (Erdos-Turan form)."""
    def is_prime(v):
        return v > 1 and all(v % f for f in range(2, int(v ** 0.5) + 1))

    p = n
    while not is_prime(p):
        p += 1
    S = np.array([2 * p * i + (i * i) % p for i in range(n)], dtype=np.int64)
    d = S[:, None] - S[None, :]
    off = d[~np.eye(n, dtype=bool)]
    assert len(set(off.tolist())) == len(off), "Sidon property violated"
    return S


def instance(S, rng, D, M=M_MULT):
    """Build one reduction instance; return streamed values and the truth."""
    while True:
        A = np.where(rng.random(len(S)) < 0.5)[0]
        B = np.where(rng.random(len(S)) < 0.5)[0]
        if len(A) and len(B):
            break
    vals = np.concatenate([S[A] * M, S[B] * M + D]).astype(np.int64)
    truth = int(len(np.intersect1d(A, B)))
    return vals, truth, A, B


def T_exact(vals, d):
    """The population smoothed readout T(d), by the defining double sum."""
    delta = (vals[:, None] - vals[None, :]).astype(float)
    return float(0.5 * (K(delta - d) + K(delta + d)).sum())


def remote_sup(M, D):
    """sup_{|u| >= M-2D} |K(u)| -- the only M-controlled piece."""
    g = M - 2 * D
    return float(np.exp(-0.5 * SIGMA ** 2 * g ** 2)) if g < 40.0 / SIGMA else 0.0


# ------------------------------------------------------------------ (A) + (B) + (C)
def combinatorics_and_decomposition(S, rng):
    rows = {}
    for D in SHIFTS:
        fail_J = fail_mult = 0
        max_mult = 0
        dec_res = []          # |T - [(1+K(2D))*I + k*K(D)]| , should be <= k^2 sup
        naive_err = []        # |T - I|
        corr_err = []         # |(T - k K(D))/(1+K(2D)) - I|
        ks = []
        for _ in range(INSTANCES):
            vals, truth, A, B = instance(S, rng, D)
            assert len(set(vals.tolist())) == len(vals), "streamed values not distinct"
            k = len(vals)
            ks.append(k)

            dd = vals[:, None] - vals[None, :]
            off = dd[~np.eye(k, dtype=bool)]
            cnt = Counter(off.tolist())
            if cnt.get(D, 0) != truth:
                fail_J += 1
            nonsig = [c for u, c in cnt.items() if abs(u) != D]
            m = max(nonsig) if nonsig else 0
            max_mult = max(max_mult, m)
            if m > 2:
                fail_mult += 1

            T = T_exact(vals, D)
            kd, k2d = float(K(D)), float(K(2 * D))
            model = (1.0 + k2d) * truth + k * kd
            dec_res.append(abs(T - model))
            naive_err.append(abs(T - truth))
            corr_err.append(abs((T - k * kd) / (1.0 + k2d) - truth))

        kbar = float(np.mean(ks))
        rows[D] = dict(
            D=D, D_in_halfwidths=float(D / DELTA_X),
            K_D=float(K(D)), K_2D=float(K(2 * D)),
            mean_k=kbar,
            remote_bound=remote_sup(M_MULT, D) * kbar ** 2,
            failures_J=fail_J, failures_mult=fail_mult, max_nonsignal_mult=int(max_mult),
            decomposition_max_residual=float(np.max(dec_res)),
            naive_err_mean=float(np.mean(naive_err)),
            naive_err_max=float(np.max(naive_err)),
            naive_err_mean_over_k=float(np.mean(naive_err) / kbar),
            corrected_err_mean=float(np.mean(corr_err)),
            corrected_err_max=float(np.max(corr_err)),
            corrected_err_max_over_k=float(np.max(corr_err) / kbar),
        )
    return rows


# ------------------------------------------------------------------------- (D)
def monte_carlo(S, rng, D):
    """Run the ACTUAL FPE bundle at finite N on one instance."""
    vals, truth, _, _ = instance(S, rng, D)
    k = len(vals)
    T = T_exact(vals, D)
    x = vals.astype(np.float64)
    scores = np.empty(MC_DRAWS)
    for r in range(MC_DRAWS):
        acc = 0.0
        for start in range(0, MC_N, MC_CHUNK):
            nb = min(MC_CHUNK, MC_N - start)
            th = OMEGA0 + SIGMA * rng.standard_normal(nb)
            b = np.exp(1j * np.outer(x, th)).sum(axis=0)      # (nb,) complex bundle
            acc += float(np.dot(np.abs(b) ** 2, np.cos(D * th)))
        scores[r] = acc / MC_N
    kd, k2d = float(K(D)), float(K(2 * D))
    corrected = (scores - k * kd) / (1.0 + k2d)
    return dict(D=D, k=k, truth=truth, T_exact=T,
                score_mean=float(scores.mean()), score_sd=float(scores.std(ddof=1)),
                predicted_sd=float(k / np.sqrt(MC_N)),
                naive_bias=float(scores.mean() - truth),
                corrected_mean=float(corrected.mean()),
                corrected_bias=float(corrected.mean() - truth),
                corrected_rmse=float(np.sqrt(np.mean((corrected - truth) ** 2))))


def main():
    rng = np.random.default_rng(SEED)
    S = sidon_set(N_ELEM)
    print(f"carrier codebook sigma={SIGMA}, omega0={OMEGA0}, kernel half-width "
          f"Delta_x={DELTA_X:.4f}")
    print(f"Sidon set: {N_ELEM} elements in [0,{S.max()}], spacing M={M_MULT}, "
          f"{INSTANCES} random instances each\n")

    rows = combinatorics_and_decomposition(S, rng)

    print("D    D/dx   K(D)         K(2D)        |  embed  |  decomp resid  "
          "|  NAIVE err (mean/k)   |  CORRECTED err (max/k)")
    for D in SHIFTS:
        r = rows[D]
        ok = "ok" if (r["failures_J"] == 0 and r["failures_mult"] == 0) else "FAIL"
        print(f"{D}  {r['D_in_halfwidths']:6.2f}  {r['K_D']:+.4e}  {r['K_2D']:+.4e}"
              f"  |  {ok:4s}   |  {r['decomposition_max_residual']:.3e}    "
              f"|  {r['naive_err_mean']:9.4f} ({r['naive_err_mean_over_k']:.5f})  "
              f"|  {r['corrected_err_max']:.3e} ({r['corrected_err_max_over_k']:.2e})")
    print(f"\nmean k = {rows[SHIFTS[0]]['mean_k']:.1f};  remote bound k^2 "
          f"sup_|u|>=M-2D |K| = {rows[SHIFTS[0]]['remote_bound']:.3e} (exactly 0 "
          f"at M={M_MULT})")
    print("-> the naive residual at D=1 equals |k*K(1)| = "
          f"{abs(rows[1]['mean_k'] * float(K(1))):.4f}, i.e. {abs(float(K(1))):.4f}*k, "
          "and is M-INDEPENDENT.")

    # minimum integer shift meeting |K(D)| <= eps for a few eps
    dmin = {}
    for eps in (1e-2, 1e-3, 1e-4, 1e-6):
        D = 1
        while np.exp(-0.5 * SIGMA ** 2 * D ** 2) > eps:
            D += 1
        dmin[f"{eps:g}"] = dict(D_min=D, D_min_over_dx=float(D / DELTA_X),
                                envelope=float(np.exp(-0.5 * SIGMA ** 2 * D ** 2)))
    print("\nsmallest integer shift D with envelope exp(-sigma^2 D^2/2) <= eps:")
    for e, v in dmin.items():
        print(f"  eps={e:>6}: D={v['D_min']}  ({v['D_min_over_dx']:.2f} half-widths, "
              f"envelope {v['envelope']:.3e})")

    print(f"\nfinite-N Monte Carlo (actual FPE bundle, N={MC_N}, {MC_DRAWS} codebook draws):")
    mc = {}
    for D in (1, 4):
        m = monte_carlo(S, rng, D)
        mc[D] = m
        print(f"  D={D}: k={m['k']}, |A cap B|={m['truth']}, T_exact={m['T_exact']:.4f}, "
              f"E[score]={m['score_mean']:.4f} (SD {m['score_sd']:.4f}, "
              f"predicted k/sqrt(N)={m['predicted_sd']:.4f})")
        print(f"        naive bias {m['naive_bias']:+.4f} | corrected bias "
              f"{m['corrected_bias']:+.4f}, RMSE {m['corrected_rmse']:.4f}")

    out = dict(params=dict(sigma=SIGMA, omega0=OMEGA0, delta_x=DELTA_X, n=N_ELEM,
                           M=M_MULT, instances=INSTANCES, seed=SEED,
                           mc_N=MC_N, mc_draws=MC_DRAWS),
               by_shift={str(D): rows[D] for D in SHIFTS},
               min_shift_for_eps=dmin,
               monte_carlo={str(D): mc[D] for D in mc})
    with open("cwf_gaphamming_calibration_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_gaphamming_calibration_results.json")


if __name__ == "__main__":
    main()
