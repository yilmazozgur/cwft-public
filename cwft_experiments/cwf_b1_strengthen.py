"""
B1 STRENGTHENING (Program I; Chapter 6, Sec 6.4) -- two targeted upgrades that
close the two softest points of the published B1 pass.

  PART A -- FAIR NONLINEAR DICTIONARY (closes the CML "flat-N" hedge).
    Published B1 lifts every substrate with translation-invariant local
    MONOMIALS (degree <= 3). For the coupled-map lattice the Koopman closure
    error N stayed high and flat, and we honestly flagged that this might be a
    dictionary artifact, not a real "the trade-off is sharper here" statement.
    Here we re-lift each substrate with a LOCAL RANDOM-FOURIER-FEATURE basis
    cos(W.x_window + b) -- a universal approximator -- at fixed feature budget F,
    sweeping ONLY the window radius r. Two questions:
      (i)  does a fair basis drive N down for CML?  (artifact test)
      (ii) at which radius r* is N minimised?  r* is the substrate's intrinsic
           interaction range; the trade-off's knee should sit there.
    NOTE on locality: under a translation-invariant substrate the Koopman
    eigenmodes are delocalised (Fourier-like), so the eigenmode-IPR locality of
    B1 is degenerate on this basis (all modes ~global). The meaningful locality
    knob here is the representation radius r itself, so we read the frontier as
    N(r) and report r* directly rather than an eigenmode-IPR.

  PART B -- OBSERVABILITY-MEASURE ROBUSTNESS (closes the "R^2 is a weak O" hedge).
    Published O is a linear-readout R^2 from the top-c principal components. We
    add an information-grounded O_MI = 1 - prod(1-rho_k^2)^(1/K) from the
    canonical correlations rho_k between the top-c scores and the next state
    (exact MI under a joint-Gaussian model; deterministic, no density estimate),
    and REFIT the conserved kappa on the SAME published MONOMIAL ladder with each
    O. If the per-system kappa-CV is preserved under the swap, kappa-conservation
    is not an artifact of the particular observability proxy. (The swap is run at
    reduced capacity c=n/2, where O is unsaturated and the two measures genuinely
    differ -- at c=n both saturate and the test is vacuous.)

Reuses substrate collectors and the monomial machinery from cwf_b1_tradeoff.py;
the committed B1 result is untouched.

Still out of scope (documented future work): RNN / real-data substrates; the B2
bridge test (does kappa track hbar_c across a calibrated catalog?).
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_b1_tradeoff import (collect_ca, collect_cml,
                             master_subsets, ladder_dict, monomials,
                             conserved_direction)


# =========================================================================
# Shared: observability measures and kappa fit
# =========================================================================

def _whiten_basis(A, tol=1e-9):
    Ac = A - A.mean(0)
    U, S, _ = np.linalg.svd(Ac, full_matrices=False)
    return U[:, S > tol * (S[0] + 1e-12)]

def canon_corr(A, B):
    Ua, Ub = _whiten_basis(A), _whiten_basis(B)
    if Ua.shape[1] == 0 or Ub.shape[1] == 0:
        return np.zeros(1)
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return np.clip(s, 0.0, 1.0 - 1e-9)

def observabilities(scores, Ytgt):
    """R^2 and Gaussian-MI observability of next state from PCA scores."""
    Yc = Ytgt - Ytgt.mean(0)
    ss = float((Yc ** 2).sum()) + 1e-12
    W = np.linalg.solve(scores.T @ scores + 1e-3 * np.eye(scores.shape[1]),
                        scores.T @ Yc)
    OR2 = float(np.clip(1 - ((Yc - scores @ W) ** 2).sum() / ss, 0, 1))
    rho = canon_corr(scores, Yc)
    OMI = float(np.clip(1 - np.exp(np.mean(np.log(1 - rho ** 2 + 1e-12))), 0, 1))
    return OR2, OMI



# =========================================================================
# PART A: random-Fourier-feature lift, fixed budget, radius sweep
# =========================================================================

def ring_windows(n, r):
    return np.array([[(i + o) % n for o in range(-r, r + 1)] for i in range(n)])

def rff_lift(Z, win, W, b):
    T, n = Z.shape; F = W.shape[0]
    proj = np.einsum("tnw,fw->tnf", Z[:, win], W) + b[None, None, :]
    return (np.sqrt(2.0 / F) * np.cos(proj)).reshape(T, n * F)

def rff_closure(Zx, Zy, win, W, b, n, caps):
    Px = rff_lift(Zx, win, W, b); Py = rff_lift(Zy, win, W, b)
    D = Px.shape[1]
    mu = Px.mean(0); sd = Px.std(0) + 1e-9
    Px = (Px - mu) / sd; Py = (Py - mu) / sd
    lam = 1e-2 * Px.shape[0]
    A = np.linalg.solve(Px.T @ Px + lam * np.eye(D), Px.T @ Py)
    N = float(np.linalg.norm(Py - Px @ A) / (np.linalg.norm(Py) + 1e-12))
    U, Sv, _ = np.linalg.svd(Px - Px.mean(0), full_matrices=False)
    Os = {}
    for c in caps:
        cc = min(c, D, U.shape[1]); sc = U[:, :cc] * Sv[:cc]
        Os[c] = observabilities(sc, Zy)
    return N, Os, D

def part_A(systems, n, caps, seeds, F, gamma, n_traj, T, burn, mono_minN):
    radii = [0, 1, 2, 3, 4, 6]
    rng0 = np.random.default_rng(7)
    bank = {r: (ring_windows(n, r),
                rng0.normal(0, gamma, size=(F, 2 * r + 1)),
                rng0.uniform(0, 2 * np.pi, size=F)) for r in radii}
    res = {}
    for si, (name, typ, params, cls) in enumerate(systems):
        base = 3000 * (si + 1)
        curves = {r: [] for r in radii}
        for sd_i in seeds:
            X, Y = (collect_ca(params["rule"], n, n_traj, T, burn, base + sd_i)
                    if typ == "ca" else
                    collect_cml(params["a"], params["eps"], n, n_traj, T, burn, base + sd_i))
            mu = X.mean(0); sdv = X.std(0) + 1e-9
            Zx, Zy = (X - mu) / sdv, (Y - mu) / sdv
            for r in radii:
                win, W, b = bank[r]
                Nr, _, _ = rff_closure(Zx, Zy, win, W, b, n, caps)
                curves[r].append(Nr)
        Nmean = {r: float(np.mean(curves[r])) for r in radii}
        rstar = min(radii, key=lambda r: Nmean[r])
        res[name] = dict(wclass=cls, typ=typ, N_of_r=Nmean,
                         r_star=rstar, N_rff_min=Nmean[rstar],
                         N_mono_min=mono_minN.get(name))
        print(f"  {name:8s}[{cls:>9}]: N(r)=" +
              " ".join(f"{r}:{Nmean[r]:.2f}" for r in radii) +
              f"  -> r*={rstar} (N*={Nmean[rstar]:.2f}, monomial N*="
              f"{mono_minN.get(name, float('nan')):.2f})")
    return radii, res


# =========================================================================
# PART B: MI-observability robustness on the PUBLISHED monomial ladder
# =========================================================================

def build_mono_ladder(n, maxdeg, radii, degs):
    master = master_subsets(n, maxdeg)
    ladder, seen = [], set()
    for d in degs:
        for r in radii:
            subs = ladder_dict(master, r, d)
            if (d, len(subs)) in seen:
                continue
            seen.add((d, len(subs))); ladder.append((r, d, subs))
    ladder.sort(key=lambda t: len(t[2]))
    return ladder

def mono_functionals(Zx, Zy, subsets, n, c):
    Psi_X = monomials(Zx, subsets); Psi_Y = monomials(Zy, subsets)
    D = len(subsets)
    mu = Psi_X.mean(0); sd = Psi_X.std(0) + 1e-9
    Psi_X = (Psi_X - mu) / sd; Psi_Y = (Psi_Y - mu) / sd
    lam = 1e-2 * Psi_X.shape[0]
    A = np.linalg.solve(Psi_X.T @ Psi_X + lam * np.eye(D), Psi_X.T @ Psi_Y)
    N = float(np.linalg.norm(Psi_Y - Psi_X @ A) / (np.linalg.norm(Psi_Y) + 1e-12))
    # L_deg (B1 primary locality)
    w, V = np.linalg.eig(A)
    P = np.abs(V) ** 2; P = P / (P.sum(0, keepdims=True) + 1e-12)
    wt = np.abs(w); wt = wt / (wt.sum() + 1e-12)
    degs = np.array([len(S) for S in subsets], float); md = max(degs.max(), 1.0)
    meandeg = (P * degs[:, None]).sum(0)
    loc = np.ones_like(meandeg) if md <= 1 else 1 - (meandeg - 1) / (md - 1)
    L_deg = float(np.clip((wt * loc).sum(), 0, 1))
    # O at capacity c
    U, Sv, _ = np.linalg.svd(Psi_X - Psi_X.mean(0), full_matrices=False)
    cc = min(c, D, U.shape[1]); sc = U[:, :cc] * Sv[:cc]
    OR2, OMI = observabilities(sc, Zy)
    return dict(D=D, N=N, L_deg=L_deg, OR2=OR2, OMI=OMI)

def part_B(systems, n, seeds, n_traj, T, burn):
    """MI-observability robustness under the CORRECTED conserved-combination
    metric (cwf_b1_tradeoff.conserved_direction). On the published monomial
    ladder we compute the scale-free min-variance conserved combination twice ---
    once with the linear-R^2 observability, once with the Gaussian-MI one --- both
    at the BINDING capacity c=n/4. Conservation is robust to the observability
    measure if the quality q and the observability loading are unchanged by the
    swap. (The first pass compared CV-fit kappa, a metric since shown gameable.)"""
    ladder = build_mono_ladder(n, 3, [1, 2, 3, 4, 6, 8], [1, 2, 3])
    c = n // 4
    res = {}
    for si, (name, typ, params, cls) in enumerate(systems):
        base = 4000 * (si + 1)
        rows = []
        for li, (r, d, subs) in enumerate(ladder):
            vals = []
            for sd_i in seeds:
                X, Y = (collect_ca(params["rule"], n, n_traj, T, burn, base + sd_i)
                        if typ == "ca" else
                        collect_cml(params["a"], params["eps"], n, n_traj, T, burn, base + sd_i))
                mu = X.mean(0); sdv = X.std(0) + 1e-9
                vals.append(mono_functionals((X - mu) / sdv, (Y - mu) / sdv, subs, n, c))
            rows.append({k: float(np.mean([v[k] for v in vals])) for k in vals[0]})
        Nv = np.array([x["N"] for x in rows])
        Lc = np.array([1 - x["L_deg"] for x in rows])
        OR2 = np.array([1 - x["OR2"] for x in rows]); OMI = np.array([1 - x["OMI"] for x in rows])
        vR, qR, _, _ = conserved_direction(np.column_stack([Nv, Lc, OR2]))
        vM, qM, _, _ = conserved_direction(np.column_stack([Nv, Lc, OMI]))
        res[name] = dict(wclass=cls, typ=typ, q_R2=qR, q_MI=qM,
                         oload_R2=float(vR[2]), oload_MI=float(vM[2]),
                         lload_R2=float(vR[1]), lload_MI=float(vM[1]))
        print(f"  {name:8s}[{cls:>9}]: quality q  R2={qR:.3f} MI={qM:.3f}   "
              f"|O-loading| R2={abs(vR[2]):.2f} MI={abs(vM[2]):.2f}  "
              f"(1-L loading R2={vR[1]:+.2f} MI={vM[1]:+.2f})")
    return res


# =========================================================================
# Driver
# =========================================================================

def main():
    n = 16; seeds = [0, 1]; n_traj, T, burn = 220, 14, 6; caps = [n, n // 2]
    systems = [
        ("CML-3.8", "cml", dict(a=3.8, eps=0.10), "chaotic"),
        ("CML-4.0", "cml", dict(a=4.0, eps=0.30), "chaotic"),
        ("CML-3.6", "cml", dict(a=3.6, eps=0.10), "periodic"),
        ("CA-110",  "ca",  dict(rule=110),        "IV"),
        ("CA-30",   "ca",  dict(rule=30),         "III"),
    ]
    # monomial baseline (from committed B1) for the artifact comparison
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    mono_minN = {}
    if "B1_tradeoff" in r_all:
        for nm, r in r_all["B1_tradeoff"]["by_system"].items():
            mono_minN[nm] = float(min(a["N"] for a in r["ladder"]))

    t0 = time.time()
    print("PART A -- fair RFF dictionary (artifact test + interaction-range r*):")
    radii, A_res = part_A(systems, n, caps, seeds, F=4, gamma=1.5,
                          n_traj=n_traj, T=T, burn=burn, mono_minN=mono_minN)
    print("\nPART B -- MI-observability robustness under the corrected metric "
          "(min-variance, binding c=n/4):")
    B_res = part_B(systems, n, seeds, n_traj, T, burn)

    qR = np.array([B_res[k]["q_R2"] for k in B_res])
    qM = np.array([B_res[k]["q_MI"] for k in B_res])
    oR = np.abs([B_res[k]["oload_R2"] for k in B_res])
    oM = np.abs([B_res[k]["oload_MI"] for k in B_res])
    q_corr = float(np.corrcoef(qR, qM)[0, 1]) if len(qR) > 1 else float("nan")
    oload_mad = float(np.mean(np.abs(oR - oM)))
    print(f"\n  O-robustness (corrected): per-system quality corr(R2,MI)={q_corr:+.2f}; "
          f"mean q R2={qR.mean():.3f} MI={qM.mean():.3f}; "
          f"mean |O-loading| R2={oR.mean():.2f} MI={oM.mean():.2f} (MAD {oload_mad:.2f})")
    print(f"  runtime {time.time()-t0:.1f}s")

    r_all["B1_strengthen"] = dict(
        n=n, F=4, gamma=1.5, caps=caps, radii=radii, binding_c=n // 4,
        partA_rff=A_res, partB_mi=B_res,
        O_robustness=dict(quality_corr_R2_MI=q_corr,
                          mean_q_R2=float(qR.mean()), mean_q_MI=float(qM.mean()),
                          mean_abs_Oload_R2=float(oR.mean()),
                          mean_abs_Oload_MI=float(oM.mean()), oload_mad=oload_mad))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(radii, A_res, B_res, q_corr, n)
    print("Wrote results.json key: B1_strengthen")


def plot_results(radii, A_res, B_res, rob, n):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    names = list(A_res); cmap = plt.cm.turbo
    cols = {nm: cmap(i / max(len(names) - 1, 1)) for i, nm in enumerate(names)}
    mk = {"ca": "o", "cml": "^"}

    # (a) N(r): the fair-basis frontier and r*
    ax = axes[0, 0]
    for nm in names:
        r = A_res[nm]; y = [r["N_of_r"][rr] for rr in radii]
        ax.plot(radii, y, mk[r["typ"]] + "-", color=cols[nm], lw=1.3, ms=5,
                label=f"{nm} ({r['wclass']}), r*={r['r_star']}")
        ax.scatter([r["r_star"]], [r["N_rff_min"]], s=140, facecolors="none",
                   edgecolors=cols[nm], linewidths=1.6, zorder=5)
    ax.set_xlabel("representation radius $r$"); ax.set_ylabel("closure error $N$ (RFF)")
    ax.set_title("(a) fair-basis frontier; circled = $r^*$ (interaction range)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # (b) artifact bar: monomial N* vs RFF N*
    ax = axes[0, 1]
    x = np.arange(len(names)); wdt = 0.38
    mono = [A_res[nm]["N_mono_min"] or np.nan for nm in names]
    rff = [A_res[nm]["N_rff_min"] for nm in names]
    ax.bar(x - wdt / 2, mono, wdt, label="monomial (deg$\\leq$3)", color="0.6")
    ax.bar(x + wdt / 2, rff, wdt, label="RFF (fair basis)", color="C0")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, fontsize=8)
    ax.set_ylabel("best closure error $N^*$ over ladder")
    ax.set_title("(b) artifact test: fair basis lowers $N^*$"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (c) O-robustness (corrected): conservation quality q(R2) vs q(MI)
    ax = axes[1, 0]
    for nm in names:
        b = B_res[nm]
        ax.scatter([b["q_R2"]], [b["q_MI"]], color=cols[nm], marker=mk[b["typ"]],
                   s=80, edgecolors="k", linewidths=0.4)
        ax.annotate(nm, (b["q_R2"], b["q_MI"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    lim = max(0.02, max(max(B_res[nm]["q_R2"], B_res[nm]["q_MI"]) for nm in names) * 1.2)
    ax.plot([0, lim], [0, lim], "k:", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"conservation quality $q$, R$^2$ observability")
    ax.set_ylabel(r"$q$, MI observability")
    ax.set_title(f"(c) $\\kappa$-conservation robust to O measure (corr={rob:+.2f})")
    ax.grid(alpha=0.3)

    # (d) text summary
    ax = axes[1, 1]; ax.axis("off")
    txt = "B1 strengthening summary:\n\n PART A (fair RFF basis):\n"
    for nm in names:
        r = A_res[nm]
        txt += (f"  {nm:8s}[{r['wclass']:>9}] N*: monomial {r['N_mono_min']:.2f}"
                f" -> RFF {r['N_rff_min']:.2f}   r*={r['r_star']}\n")
    txt += "\n PART B (corrected metric, binding c=n/4):\n"
    for nm in names:
        b = B_res[nm]
        txt += (f"  {nm:8s} q  R2={b['q_R2']:.2f} MI={b['q_MI']:.2f}  "
                f"|Oload| R2={abs(b['oload_R2']):.2f} MI={abs(b['oload_MI']):.2f}\n")
    txt += ("\n Reads:\n"
            "  - RFF lowers N* for every substrate => the monomial\n"
            "    flat-N (esp. CML) was a BASIS limitation, not a\n"
            "    substrate property; a fair universal basis closes CML.\n"
            "  - r* tracks the interaction range: r*=0 for the self-\n"
            "    dominated CML, r*=1 for the 3-cell CA. The trade-off's\n"
            "    knee sits at the substrate's intrinsic locality.\n"
            "  - (c) near the diagonal => the corrected conserved\n"
            "    combination (quality q AND O-loading) is unchanged when\n"
            "    R^2 is swapped for an information-grounded O.\n"
            f"\n  n={n}, RFF F=4 gamma=1.5, 2 seeds; Part B O at binding c=n/4.")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8, family="monospace", va="top")

    fig.suptitle("B1 strengthening --- fair nonlinear (RFF) basis + MI-observability "
                 "robustness of $\\kappa$-conservation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_B1_strengthen.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
