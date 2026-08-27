"""
DIAGNOSTIC: why does the observability weight beta keep fitting to 0?

The conserved combination is kappa = N + alpha*(1-L) + beta*(1-O), fit by
minimising the coefficient of variation CV = std(kappa)/mean(kappa) along the
representation ladder. Across B1 and B2 the fit keeps returning beta ~ 0.

Hypothesis. Conservation works by CANCELLATION: as the dictionary enriches, the
closure error N FALLS. For kappa to stay flat a partner term must RISE to
compensate. The nonlocality (1-L) rises with dictionary richness, so it cancels
N -> alpha>0. If the inverse-observability (1-O) instead FALLS with richness --
because a richer dictionary improves prediction (O up) at the same time it
improves closure (N down) -- then (1-O) co-moves with N, reinforcing the trend
rather than cancelling it, and the CV-minimiser zeroes its weight.

If true, two things should hold:
  (1) along the ladder, corr(N, 1-L) < 0 (anti-correlated, cancels) but
      corr(N, 1-O) > 0 (co-moves, reinforces) -- at least at large observer
      capacity c;
  (2) shrinking the observer capacity c (a BINDING capacity bound) should flip
      O into a competing axis: corr(N, 1-O_c) -> negative, and then beta turns on.

This script measures both, per system, as a function of c. Reuses the B1
ladder and functionals.
"""
import numpy as np
from cwf_b1_tradeoff import (collect_ca, collect_cml, master_subsets,
                             ladder_dict, functionals)


def build_ladder(n, maxdeg=3, radii=(1, 2, 3, 4, 6, 8), degs=(1, 2, 3)):
    master = master_subsets(n, maxdeg)
    lad, seen = [], set()
    for d in degs:
        for r in radii:
            subs = ladder_dict(master, r, d)
            if (d, len(subs)) in seen:
                continue
            seen.add((d, len(subs))); lad.append((r, d, subs))
    lad.sort(key=lambda t: len(t[2]))
    return lad

def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / d) if d > 1e-12 else 0.0

def fit_ab(N, Lc, Oc, grid):
    """grid-fit (alpha,beta) minimising CV over the given symmetric/positive grid.
    NB: CV=std/mean is gameable by mean inflation -- a large bounded-variance term
    cuts CV regardless of conservation -- so weights hitting the grid edge are the
    artifact's signature. Use pca loadings for the scale-free answer."""
    best = (np.inf, 0.0, 0.0)
    for a in grid:
        for b in grid:
            k = N + a * Lc + b * Oc
            cv = abs(k.std() / (abs(k.mean()) + 1e-12))
            if cv < best[0]:
                best = (cv, a, b)
    return best   # (cv,a,b)

def pca_loadings(N, Lc, Oc):
    """Scale-free conserved combination: the minimum-variance direction of the
    z-standardised [N, 1-L, 1-O]. Returns the UNIT eigenvector loadings
    (v_N, v_L, v_O) -- directly comparable because standardised, and stable
    (no ratio blow-up) -- plus quality = lambda_min/sum(lambda) (small => a
    near-constant combination exists). |v_O| small vs |v_N|,|v_L| => O does not
    participate; sign of v_O vs v_N => the direction it enters."""
    F = np.column_stack([N, Lc, Oc])
    sd = F.std(0) + 1e-12
    Fz = (F - F.mean(0)) / sd
    w, V = np.linalg.eigh(np.cov(Fz.T))      # ascending
    v = V[:, 0]
    if v[0] < 0:                              # fix a sign convention (v_N >= 0)
        v = -v
    quality = float(w[0] / (w.sum() + 1e-12))
    return v, quality                         # v=(v_N,v_L,v_O) unit-norm


def main():
    n = 16; seeds = [0, 1]; n_traj, T, burn = 220, 14, 6
    caps = [n, n // 2, n // 4, 4, 2]
    systems = [
        ("CA-184", "ca", dict(rule=184)),
        ("CA-110", "ca", dict(rule=110)),
        ("CA-30",  "ca", dict(rule=30)),
        ("CML-3.8", "cml", dict(a=3.8, eps=0.10)),
        ("CML-4.0", "cml", dict(a=4.0, eps=0.30)),
    ]
    ladder = build_ladder(n)

    print(f"ladder D = {[len(s) for _,_,s in ladder]}\n")
    print("Per-system, along the ladder: how N, (1-L), (1-O_c) co-vary, and the")
    print("beta the CV-fit selects at each observer capacity c.\n")

    for (nm, typ, p) in systems:
        agg = []
        for li, (r, d, subs) in enumerate(ladder):
            vals = []
            for sd in seeds:
                X, Y = (collect_ca(p["rule"], n, n_traj, T, burn, 50 + sd) if typ == "ca"
                        else collect_cml(p["a"], p["eps"], n, n_traj, T, burn, 50 + sd))
                mu = X.mean(0); sdv = X.std(0) + 1e-9
                vals.append(functionals((X - mu) / sdv, (Y - mu) / sdv, subs, n, caps))
            agg.append({k: float(np.mean([v[k] for v in vals])) for k in vals[0]})

        N = np.array([a["N"] for a in agg])
        Lc = np.array([1 - a["L_deg"] for a in agg])
        rNL = pearson(N, Lc)
        gpos = np.linspace(0, 3, 31); gsym = np.linspace(-3, 3, 61)
        print(f"=== {nm} ===   corr(N, 1-L) = {rNL:+.2f}   "
              f"(N: {N[0]:.2f}->{N[-1]:.2f}, 1-L: {Lc[0]:.2f}->{Lc[-1]:.2f})")
        print(f"     {'c':>4s} {'corr(N,1-O)':>12s} | {'b(>=0 grid)':>11s} "
              f"{'b(+/- grid)':>11s} | {'PCA loadings (vN,vL,vO)':>26s} {'qual':>6s}")
        for c in caps:
            O = np.array([a[f"O{c}"] for a in agg])
            Oc = 1 - O
            rNO = pearson(N, Oc)
            _, _, bpos = fit_ab(N, Lc, Oc, gpos)
            _, _, bsym = fit_ab(N, Lc, Oc, gsym)
            v, q = pca_loadings(N, Lc, Oc)
            tag = "<-pinned@0" if abs(bpos) < 1e-9 and bsym < -0.1 else ""
            print(f"     {c:>4d} {rNO:>+12.2f} | {bpos:>11.1f} {bsym:>11.1f} | "
                  f"({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})        {q:>6.3f} {tag}")
        print()

    print("READ (why beta kept fitting to 0):")
    print(" (1) The CV grid was constrained to beta>=0. The genuinely conserved")
    print("     combination loads O with the OPPOSITE sign (O RISES as N falls, so")
    print("     it must be ADDED, i.e. negative coef on (1-O)). Compare the b(>=0)")
    print("     and b(+/-) columns: where b(>=0)=0 but b(+/-)<0, beta=0 was a")
    print("     BOUNDARY artifact, not 'O irrelevant'.")
    print(" (2) CV=std/mean is mean-inflation-gameable, so where weights move they")
    print("     pin to the grid edge -- not real conservation either.")
    print(" PCA loadings (scale-free, stable): |vO| comparable to |vN|,|vL| => O")
    print(" DOES participate, with sign opposite to vN. The fix: drop the beta>=0")
    print(" constraint (O is a co-improving, not competing, axis) OR redefine the")
    print(" observability term so high-O is the resource, not (1-O).")


if __name__ == "__main__":
    main()
