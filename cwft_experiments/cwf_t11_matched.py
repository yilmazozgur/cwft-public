#!/usr/bin/env python3
"""
cwf_t11_matched.py  --  the within-automata trade-off law on a matched protocol.

Why this script exists.  In cwf_t11_ca_faithful.py the closure residual N_rich of each
elementary cellular automaton comes from 16-cell rings (6 burn-in steps, 14 measured
steps, 220 trajectories, seeds 0 and 1; key T11_scale_extensive), while the faithful
entropy rate h_t comes from 160-cell rings (160 burn-in steps, 260 measured steps,
40 trajectories; cwf_t11_ca_entropy.ca_symbolic_entropy defaults).  The two numbers are
paired by rule label, so they are not measured on the same finite substrate.  A review
(2026-09-06) flagged this; for example, on a 16-cell ring rule 90 reaches the all-zero
state by step 8 from every initial state.

This script recomputes h_t with the same estimator on two protocols that match the ring
size of the closure runs, and recomputes the within-automata Spearman correlation with the
recorded N_rich:
  (a) matched:   n = 16, burn = 6, T = 14, 220 trajectories, seeds 0 and 1 (averaged),
                 block lengths capped at 12 so that blocks fit the 14-step window,
                 i.e. the closure runs' own ring size and time window;
  (b) ring-16:   n = 16, burn = 160, T = 260, 40 trajectories, seed 0,
                 i.e. the closure runs' ring size in the estimator's long-run regime.
Bootstrap intervals use the same resampling as cwf_t11_ca_faithful.  The recorded
key T11_ca_faithful is not modified.  Writes t11_matched_results.json.

Ranks.  cwf_t11_ca_faithful ranks with argsort, which breaks ties in index order (six
rules have h_t exactly 0 on the recorded protocol, and five more sit at 1e-9).  Each
protocol is therefore reported three ways: argsort ranks (the record's convention, for
comparison), average ranks for ties, and average ranks after setting numerically zero
entropies (below FLOOR = 1e-8 bits) to exactly 0.  The last is the headline; its
bootstrap interval and permutation p-value use the same rank rule.  A floor sweep
(1e-8 to 1e-2 bits) shows the sensitivity.

What 'matched' measures.  On the 14-step window the estimator returns a block-entropy
difference over a short, non-stationary stretch (block lengths up to 12 in 14 steps),
a finite-window conditional entropy, not an entropy rate.  It is the quantity measured
on the same finite substrate as N_rich, which is the point of the check.
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

from cwf_t11_ca_entropy import ca_symbolic_entropy
from cwf_t11_ca_faithful import bootstrap, perm_p, ADDITIVE
from cwf_b2_bridge import spearman

HERE = os.path.dirname(os.path.abspath(__file__))
FLOOR = 1e-8                   # entropies below this (bits) are numerically zero
FLOOR_SWEEP = (1e-8, 1e-6, 1e-4, 1e-2)


def spearman_avg(x, y):
    """Spearman correlation with average ranks for ties."""
    rx = rankdata(x) - (len(x) + 1) / 2.0
    ry = rankdata(y) - (len(y) + 1) / 2.0
    den = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float(rx @ ry / den) if den > 0 else 0.0


def bootstrap_avg(x, y, B=4000, seed=7):
    rng = np.random.default_rng(seed)
    n = len(x)
    b = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        b[i] = spearman_avg(x[idx], y[idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def perm_p_avg(x, y, P=20000, seed=11):
    rng = np.random.default_rng(seed)
    obs = abs(spearman_avg(x, y))
    yc = y.copy()
    ge = 0
    for _ in range(P):
        rng.shuffle(yc)
        if abs(spearman_avg(x, yc)) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (P + 1)


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    per = R["T11_ca_faithful"]["per_rule"]
    names = sorted(per, key=lambda k: per[k]["rule"])
    N = np.array([per[k]["N_rich"] for k in names])
    h_rec = np.array([per[k]["h_t"] for k in names])

    h_a, h_b = [], []
    for k in names:
        rule = per[k]["rule"]
        ha = np.mean([ca_symbolic_entropy(rule, n=16, T=14, burn=6, n_traj=220, seed=s, Lmax=12)
                      for s in (0, 1)])
        hb = ca_symbolic_entropy(rule, n=16, T=260, burn=160, n_traj=40, seed=0)
        h_a.append(float(ha))
        h_b.append(float(hb))
        print(f"  rule {rule:3d}  N_rich={per[k]['N_rich']:.3f}  h_t(160-ring)={per[k]['h_t']:.3f}"
              f"  h_t(matched)={ha:.3f}  h_t(16-ring long)={hb:.3f}"
              f"{'  [additive]' if rule in ADDITIVE else ''}")
    h_a, h_b = np.array(h_a), np.array(h_b)

    out = {"protocols": {
        "recorded": "h_t on n=160, burn=160, T=260, 40 trajectories (cwf_t11_ca_entropy defaults)",
        "matched": "h_t on n=16, burn=6, T=14, 220 trajectories, seeds 0 and 1 averaged",
        "ring16_long": "h_t on n=16, burn=160, T=260, 40 trajectories, seed 0"},
        "closure_residual": "N_rich from results.json T11_scale_extensive (n=16, burn=6, T=14)",
        "rules": {k: dict(rule=per[k]["rule"], N_rich=per[k]["N_rich"], h_t_recorded=per[k]["h_t"],
                          h_t_matched=float(h_a[i]), h_t_ring16_long=float(h_b[i]),
                          additive=bool(per[k]["rule"] in ADDITIVE))
                  for i, k in enumerate(names)}}
    out["rank_rule"] = ("headline: average ranks for ties, entropies below %g bits set to 0; "
                        "also reported: argsort ranks (record convention) and average ranks on raw "
                        "values" % FLOOR)
    for key, h in (("recorded", h_rec), ("matched", h_a), ("ring16_long", h_b)):
        rho = spearman(N, h)
        lo, hi = bootstrap(N, h)
        pv = perm_p(N, h)
        hf = np.where(h < FLOOR, 0.0, h)
        rho_f = spearman_avg(N, hf)
        lo_f, hi_f = bootstrap_avg(N, hf)
        pv_f = perm_p_avg(N, hf)
        sweep = {f"{t:g}": spearman_avg(N, np.where(h < t, 0.0, h)) for t in FLOOR_SWEEP}
        out[f"spearman_{key}"] = dict(rho=float(rho_f), ci95=[lo_f, hi_f], perm_p=float(pv_f),
                                      rho_avg_ranks_raw=spearman_avg(N, h),
                                      rho_argsort=float(rho), ci95_argsort=[lo, hi],
                                      perm_p_argsort=float(pv), floor_sweep=sweep)
        print(f"\n  within-CA Spearman(N_rich, h_t [{key}]) = {rho_f:+.3f}  95% CI [{lo_f:+.3f}, {hi_f:+.3f}]"
              f"  perm p = {pv_f:.4f}   (argsort {rho:+.3f} [{lo:+.3f}, {hi:+.3f}];"
              f" raw average ranks {spearman_avg(N, h):+.3f}; floor sweep "
              f"{min(sweep.values()):+.3f} to {max(sweep.values()):+.3f})")
    json.dump(out, open(os.path.join(HERE, "t11_matched_results.json"), "w"), indent=1)
    print("\nWrote t11_matched_results.json")


if __name__ == "__main__":
    main()
