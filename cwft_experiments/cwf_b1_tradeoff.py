"""
B1 -- the dimensional trade-off phase diagram (Program I; Chapter 6, Sec 6.4).

Publication-direction build. Tests the Dimensional Trade-off Inequality (no
representation jointly achieves low N, high L, high O) and its conservation
form (a representation-independent kappa) across TWO substrate classes:
  - elementary cellular automata (discrete, Wolfram classes II/III/IV);
  - coupled logistic-map lattices (continuous, periodic -> chaotic).
For each system it sweeps a translation-invariant local-window representation
ladder parameterised by (locality radius r, nonlinearity degree d).

Functionals (each computable on standardized features so CA and CML share the
pipeline):
  D     = dictionary size.
  N     = EDMD Koopman closure error, ||Psi_Y - Psi_X A||_F / ||Psi_Y||_F
          (ridge-regularised, column-standardized features).
  L_deg = degree-weighted eigenmode locality (interpretable proxy).
  L_pos = position-basis eigenmode locality (spatial IPR of the eigenmodes'
          site-occupation profile).
  O_c   = capacity-bounded predictability of the next state from the top-c
          principal components (ridge R^2); reported at c=n and c=n/2.

Rigour over the previous pass:
  - second substrate class (continuous CML) -> the result is not CA-specific;
  - SHUFFLE CONTROL: with (x_t,x_{t+1}) pairing destroyed, N saturates and the
    conserved kappa must vanish -- shows the effect is dynamics-driven, not a
    regression artifact;
  - O at two observer capacities (robustness);
  - larger n, more seeds (error bars), finer (r,d) ladder;
  - per-system AND universal (alpha,beta) weight fits.

Remaining for full publication: mutual-information O (vs R^2); RNN / real-data
substrates; the B2 bridge test (does kappa track hbar_c across the catalog?).
"""
import json, os, itertools, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Substrates
# =========================================================================

def ca_table(rule):
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)

def ca_step(state, table):
    l = np.roll(state, 1); r = np.roll(state, -1)
    return table[(4 * l + 2 * state + r).astype(int)]

def collect_ca(rule, n, n_traj, T, burn, seed):
    rng = np.random.default_rng(seed); table = ca_table(rule)
    X, Y = [], []
    for _ in range(n_traj):
        s = rng.integers(0, 2, n).astype(np.int8)
        for _ in range(burn):
            s = ca_step(s, table)
        for _ in range(T):
            s2 = ca_step(s, table); X.append(s.copy()); Y.append(s2.copy()); s = s2
    return np.array(X, float), np.array(Y, float)

def cml_step(x, a, eps):
    f = a * x * (1 - x)
    return (1 - eps) * f + 0.5 * eps * (np.roll(f, 1) + np.roll(f, -1))

def collect_cml(a, eps, n, n_traj, T, burn, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_traj):
        s = rng.random(n)
        for _ in range(burn):
            s = cml_step(s, a, eps)
        for _ in range(T):
            s2 = cml_step(s, a, eps); X.append(s.copy()); Y.append(s2.copy()); s = s2
    return np.array(X, float), np.array(Y, float)


# =========================================================================
# Representation ladder: translation-invariant local-window monomials
# =========================================================================

def ring_diameter(subset, n):
    if len(subset) <= 1:
        return 0
    p = sorted(subset)
    gaps = [(p[(i + 1) % len(p)] - p[i]) % n for i in range(len(p))]
    return n - max(gaps)

def master_subsets(n, maxdeg):
    out = []
    for d in range(1, maxdeg + 1):                  # drop constant (degree 0)
        for S in itertools.combinations(range(n), d):
            out.append((S, len(S), ring_diameter(S, n)))
    return out

def ladder_dict(master, r, d):
    return [S for (S, deg, diam) in master if deg <= d and diam <= 2 * r]

def monomials(Z, subsets):
    Psi = np.empty((Z.shape[0], len(subsets)))
    for j, S in enumerate(subsets):
        Psi[:, j] = np.prod(Z[:, S], axis=1)
    return Psi


# =========================================================================
# Functionals
# =========================================================================

def functionals(Zx, Zy, subsets, n, capacities):
    Psi_X = monomials(Zx, subsets); Psi_Y = monomials(Zy, subsets)
    D = len(subsets)
    # column-standardize by Psi_X stats (conditioning; unifies CA/CML)
    mu = Psi_X.mean(0); sd = Psi_X.std(0) + 1e-9
    Psi_X = (Psi_X - mu) / sd; Psi_Y = (Psi_Y - mu) / sd

    # N: ridge EDMD closure
    lam = 1e-2 * Psi_X.shape[0]
    G = Psi_X.T @ Psi_X + lam * np.eye(D)
    A = np.linalg.solve(G, Psi_X.T @ Psi_Y)
    N = float(np.linalg.norm(Psi_Y - Psi_X @ A) / (np.linalg.norm(Psi_Y) + 1e-12))

    # eigenmodes
    w, V = np.linalg.eig(A)
    P = np.abs(V) ** 2; P = P / (P.sum(0, keepdims=True) + 1e-12)
    wt = np.abs(w); wt = wt / (wt.sum() + 1e-12)

    degs = np.array([len(S) for S in subsets], float); maxdeg = max(degs.max(), 1.0)
    meandeg = (P * degs[:, None]).sum(0)
    loc_deg = np.ones_like(meandeg) if maxdeg <= 1 else 1 - (meandeg - 1) / (maxdeg - 1)
    L_deg = float(np.clip((wt * loc_deg).sum(), 0, 1))

    M = np.zeros((D, n))
    for j, S in enumerate(subsets):
        for i in S:
            M[j, i] = 1.0
    occ = M.T @ P; occ = occ / (occ.sum(0, keepdims=True) + 1e-12)
    ipr = (occ ** 2).sum(0)
    loc_pos = (ipr - 1.0 / n) / (1 - 1.0 / n + 1e-12)
    L_pos = float(np.clip((wt * loc_pos).sum(), 0, 1))

    # O at each capacity via top-c PCA scores
    Phic = Psi_X - Psi_X.mean(0)
    U, Sv, _ = np.linalg.svd(Phic, full_matrices=False)
    Ytgt = Zy - Zy.mean(0)
    ss_tot = float((Ytgt ** 2).sum()) + 1e-12
    Os = {}
    for c in capacities:
        cc = min(c, D, U.shape[1])
        sc = U[:, :cc] * Sv[:cc]
        Wt = np.linalg.solve(sc.T @ sc + 1e-3 * np.eye(cc), sc.T @ Ytgt)
        Os[c] = float(np.clip(1 - ((Ytgt - sc @ Wt) ** 2).sum() / ss_tot, 0, 1))
    return dict(D=D, N=N, L_deg=L_deg, L_pos=L_pos,
                **{f"O{c}": Os[c] for c in capacities})


def fit_kappa(N, Lc, Oc):
    """DEPRECATED (first-pass; superseded by conserved_direction). Minimises the
    coefficient of variation of kappa = N + a*Lc + b*Oc over a NONNEGATIVE (a,b)
    grid. cwf_b1_beta_diag.py showed this is ill-posed: CV=std/mean is offset-
    gameable (weights pin to the grid edge by mean inflation) and the beta>=0 grid
    cannot represent O's true (opposite) sign. Kept only so the historical
    cwf_b1_tradeoff first pass still runs; do not use for new fits."""
    grid = np.linspace(0, 3, 31); best = (np.inf, 0.0, 0.0)
    for a in grid:
        for b in grid:
            k = N + a * Lc + b * Oc
            cv = k.std() / (abs(k.mean()) + 1e-12)
            if cv < best[0]:
                best = (cv, a, b)
    return best

def conserved_direction(F):
    """Corrected conserved-combination analysis (Program I), replacing fit_kappa.

    F: (m, k) array of raw functionals along the representation ladder, columns
    e.g. [N, 1-L, 1-O] with O at a BINDING capacity. Returns the MINIMUM-VARIANCE
    DIRECTION of the z-standardised columns -- the optimal SIGNED-weight conserved
    combination, offset-invariant (not gameable by mean inflation) and stable:
        v   unit loadings (sign fixed so v[0] >= 0);
        q   conservation quality lambda_min/sum(lambda) in [0, 1/k]; q -> 0
            certifies a near-constant combination, q = 1/k means the functionals
            are independent (no conserved law);
        C   k x k correlation matrix (to pool into a universal direction);
        sd  per-column std (to translate loadings to raw-functional weights).
    """
    F = np.asarray(F, float)
    sd = F.std(0) + 1e-12
    Fz = (F - F.mean(0)) / sd
    C = np.cov(Fz.T)
    w, V = np.linalg.eigh(C)
    v = V[:, 0]
    if v[0] < 0:
        v = -v
    return v, float(w[0] / (w.sum() + 1e-12)), C, sd


# =========================================================================
# Sweep
# =========================================================================

def main():
    n = 16; maxdeg = 3
    radii = [1, 2, 3, 4, 6, 8]; degs = [1, 2, 3]
    seeds = [0, 1, 2]
    n_traj, T, burn = 250, 14, 6
    caps = [n, n // 2]; Lkey = "L_deg"
    systems = [
        ("CA-110", "ca", dict(rule=110), "IV"),
        ("CA-30",  "ca", dict(rule=30),  "III"),
        ("CA-90",  "ca", dict(rule=90),  "III"),
        ("CA-184", "ca", dict(rule=184), "II"),
        ("CML-3.6", "cml", dict(a=3.6, eps=0.10), "periodic"),
        ("CML-3.8", "cml", dict(a=3.8, eps=0.10), "chaotic"),
        ("CML-4.0", "cml", dict(a=4.0, eps=0.30), "chaotic"),
    ]

    master = master_subsets(n, maxdeg)
    ladder, seen = [], set()
    for d in degs:
        for r in radii:
            subs = ladder_dict(master, r, d)
            if (d, len(subs)) in seen:
                continue
            seen.add((d, len(subs))); ladder.append((r, d, subs))
    ladder.sort(key=lambda t: len(t[2]))
    print(f"B1 (publication-direction): n={n}, {len(ladder)} representations, "
          f"{len(systems)} systems, {len(seeds)} seeds\n")

    def get_data(typ, params, seed):
        if typ == "ca":
            return collect_ca(params["rule"], n, n_traj, T, burn, seed)
        return collect_cml(params["a"], params["eps"], n, n_traj, T, burn, seed)

    def encode(X, mu, sd):
        return (X - mu) / sd

    t0 = time.time(); results = {}; shuffle_report = {}
    for si, (name, typ, params, cls) in enumerate(systems):
        base = 1000 * (si + 1)
        per_seed = []
        for sd_i in seeds:
            X, Y = get_data(typ, params, base + sd_i)
            mu = X.mean(0); sdv = X.std(0) + 1e-9
            Zx, Zy = encode(X, mu, sdv), encode(Y, mu, sdv)
            per_seed.append([functionals(Zx, Zy, subs, n, caps) for (r, d, subs) in ladder])
        agg = []
        for li, (r, d, subs) in enumerate(ladder):
            pts = [per_seed[s][li] for s in range(len(seeds))]
            a = {k: float(np.mean([p[k] for p in pts])) for k in pts[0]}
            a.update({k + "_sd": float(np.std([p[k] for p in pts])) for k in pts[0]})
            a["r"] = r; a["d"] = d
            agg.append(a)
        Nv = np.array([a["N"] for a in agg])
        Lc = np.array([1 - a[Lkey] for a in agg])
        Oc = np.array([1 - a[f"O{n}"] for a in agg])
        cv, alpha, beta = fit_kappa(Nv, Lc, Oc)
        kappa = Nv + alpha * Lc + beta * Oc
        results[name] = dict(name=name, typ=typ, wclass=cls, ladder=agg,
                             alpha=alpha, beta=beta, kappa_mean=float(kappa.mean()),
                             kappa_cv=cv, kappa=[float(x) for x in kappa])
        # shuffle control: destroy x_t->x_{t+1} pairing (one seed)
        X, Y = get_data(typ, params, base)
        mu = X.mean(0); sdv = X.std(0) + 1e-9
        rng = np.random.default_rng(123); perm = rng.permutation(len(Y))
        Zx, Zy = encode(X, mu, sdv), encode(Y[perm], mu, sdv)
        sh = [functionals(Zx, Zy, subs, n, caps) for (r, d, subs) in ladder]
        Ns = np.array([x["N"] for x in sh])
        Ls = np.array([1 - x[Lkey] for x in sh]); Osh = np.array([1 - x[f"O{n}"] for x in sh])
        scv, _, _ = fit_kappa(Ns, Ls, Osh)
        shuffle_report[name] = dict(N_mean=float(Ns.mean()), O_mean=float(1 - Osh.mean()),
                                    kappa_cv=float(scv))
        print(f"  {name:9s}[{cls:>8}]: a={alpha:.1f} b={beta:.1f}  "
              f"kappa={kappa.mean():.3f} CV={cv:.3f}  | shuffle: N={Ns.mean():.2f} "
              f"O={1-Osh.mean():.2f} CV={scv:.3f}")

    # universal weights
    grid = np.linspace(0, 3, 31); best = (np.inf, 0.0, 0.0)
    for a in grid:
        for b in grid:
            cvs = []
            for name in results:
                ag = results[name]["ladder"]
                Nv = np.array([x["N"] for x in ag]); Lc = np.array([1 - x[Lkey] for x in ag])
                Oc = np.array([1 - x[f"O{n}"] for x in ag])
                k = Nv + a * Lc + b * Oc; cvs.append(k.std() / (abs(k.mean()) + 1e-12))
            if np.mean(cvs) < best[0]:
                best = (float(np.mean(cvs)), a, b)
    uni = best
    print(f"\n  universal weights: a={uni[1]:.1f} b={uni[2]:.1f}  mean CV={uni[0]:.3f}")
    print(f"  runtime {time.time()-t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["B1_tradeoff"] = dict(n=n, n_ladder=len(ladder), Lkey=Lkey,
                                systems=[s[0] for s in systems], by_system=results,
                                shuffle_control=shuffle_report,
                                universal=dict(alpha=uni[1], beta=uni[2], mean_cv=uni[0]))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(results, shuffle_report, n, Lkey, uni, caps)
    print("Wrote results.json key: B1_tradeoff")


def plot_results(results, shuffle, n, Lkey, uni, caps):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    names = list(results); cmap = plt.cm.turbo
    cols = {nm: cmap(i / max(len(names) - 1, 1)) for i, nm in enumerate(names)}
    mk = {"ca": "o", "cml": "^"}

    ax = axes[0, 0]
    for nm in names:
        r = results[nm]; ag = r["ladder"]
        x = [1 - a[Lkey] for a in ag]; y = [a["N"] for a in ag]
        s = [20 + 160 * a[f"O{n}"] for a in ag]
        ax.scatter(x, y, s=s, color=cols[nm], marker=mk[r["typ"]], alpha=0.65,
                   edgecolors="k", linewidths=0.3, label=f"{nm} ({r['wclass']})")
    # shuffle-control cloud (gray) -- should sit at high N
    for nm in names:
        ax.scatter([0.5], [shuffle[nm]["N_mean"]], marker="x", color="gray", s=40,
                   alpha=0.5)
    ax.set_xlabel(r"nonlocality $1-L$"); ax.set_ylabel(r"nonlinearity residual $N$")
    ax.set_title("(a) trade-off frontier: CA (o), CML ($\\triangle$); "
                 "shuffle ($\\times$, grey)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for nm in names:
        r = results[nm]; D = [a["D"] for a in r["ladder"]]
        ax.plot(D, r["kappa"], mk[r["typ"]] + "-", color=cols[nm], lw=1.2, ms=4,
                label=f"{nm}: CV={r['kappa_cv']:.2f}")
    ax.set_xscale("log"); ax.set_xlabel(r"dimension $D$")
    ax.set_ylabel(r"$\kappa=N+\alpha(1{-}L)+\beta(1{-}O)$")
    ax.set_title("(b) per-system $\\kappa$ conservation"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for nm in names:
        ag = results[nm]["ladder"]
        ax.scatter([a["L_deg"] for a in ag], [a["L_pos"] for a in ag],
                   color=cols[nm], marker=mk[results[nm]["typ"]], alpha=0.6, s=28)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel(r"$L_{\rm deg}$"); ax.set_ylabel(r"$L_{\rm pos}$")
    ax.set_title("(c) locality-measure robustness"); ax.grid(alpha=0.3)

    ax = axes[1, 1]; ax.axis("off")
    txt = "B1 phase diagram (publication-direction):\n\n"
    for nm in names:
        r = results[nm]; sh = shuffle[nm]
        txt += (f"  {nm:8s}[{r['wclass']:>8}] $\\kappa$CV={r['kappa_cv']:.2f}"
                f"  (shuffle N={sh['N_mean']:.2f}, O={sh['O_mean']:.2f})\n")
    cvs = np.array([results[nm]["kappa_cv"] for nm in names])
    txt += (f"\n  per-system CV: mean {cvs.mean():.2f} [{cvs.min():.2f},{cvs.max():.2f}]\n"
            f"  universal ($\\alpha$={uni[1]:.1f},$\\beta$={uni[2]:.1f}) mean CV={uni[0]:.2f}\n\n"
            "  Reads:\n"
            "   - empty low-$N$/high-$L$ corner => inequality holds;\n"
            "   - low per-system CV => conserved $\\kappa$ (per system),\n"
            "     clearest for CA classes III/IV;\n"
            "   - shuffle: N->1, O->0 (real systems N<0.9, O>0) =>\n"
            "     the functionals track real dynamics, not artifact;\n"
            "   - CML (local-monomial lift): N stays high/flat =>\n"
            "     trade-off SHARPNESS is representation-dependent;\n"
            "   - universal CV > per-system mean => weights are\n"
            "     somewhat system-dependent (no single clean law).\n\n"
            f"  n={n}, {len(results[names[0]]['ladder'])} reps, 3 seeds; O at "
            f"c={caps}; primary L={Lkey}.")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, family="monospace", va="top")

    fig.suptitle("B1: dimensional trade-off phase diagram (Program I) --- "
                 "cellular automata + coupled-map lattices", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_B1_tradeoff.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
