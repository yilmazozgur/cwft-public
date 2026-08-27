"""
T1.1 -- the trade-off law at scale (FUTURE_WORK Tier 1, item 1).

Hardens the one surviving gauge-invariant empirical regularity of Program I:
the association between the best-closure floor N_rich and the dynamics-intrinsic
action a_c (information production, bits/step). The B2 bridge found
  Spearman(N_rich, a_c) = +0.64   at n = 7 substrates.
That is the most load-bearing claim after hbar_c was shown to be gauge -- and the
weakest-supported (n=7). This script re-measures it at n ~ 36 substrates spanning
Wolfram classes I-IV plus a coupled-logistic-map grid, with:

  * the SAME pipeline and hyperparameters as cwf_b2_bridge.main() (apples-to-apples
    with the n=7 baseline -- nothing changed but the catalog);
  * a BOOTSTRAP CI on the Spearman correlation (resample substrates w/ replacement);
  * a PERMUTATION NULL (shuffle a_c against N_rich) -> a p-value for "is +0.64 real";
  * the universal conserved-direction quality Q at the larger n (does it stabilise?).

Clean split either way: the correlation holds/strengthens with a CI excluding 0
(-> a trade-off LAW), or it washes out (-> per-substrate regularity only, an honest
negative). Non-destructive: writes results.json key T11_scale; leaves B2_bridge.

Reuses ALL machinery from cwf_b1_tradeoff / cwf_b2_bridge -- no reimplementation.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_b1_tradeoff import conserved_direction, functionals, ca_table, ca_step
from cwf_b2_bridge import (build_ladder, system_curves, ca_lyapunov,
                           cml_ks_entropy, spearman)
from cwf_t11_reservoir import make_ring_C, collect_rcl, rcl_ks_entropy

LN2 = float(np.log(2))

# CA a_c estimator: "extensive" = QR Boolean-Jacobian Pesin sum of positive exponents
# (consistent with the CML/RCL extensive Pesin path, faithful for ballistic rules);
# "defect" = the legacy single-vector Bagnoli-Rechtman-Ruffo MAX exponent (intensive,
# and conflates defect-spreading VELOCITY with entropy for ballistic class-II rules --
# see ../cwft_manuscripts/P5_tradeoff_law/THEOREM_ac_bound.md). Default to the consistent extensive estimator.
CA_AC_MODE = "extensive"


def ca_ks_spectrum(rule, n, T=400, burn=80, seed=0):
    """Extensive CA KS entropy (bits/step) = sum of POSITIVE Lyapunov exponents via QR
    of the Boolean Jacobian (XOR-derivatives), matching the CML/RCL Pesin convention.
    Ballistic (translation-like) rules -> permutation Jacobian -> exponents ~0; additive
    (rule 90/150) and chaotic rules -> positive, extensive (O(n))."""
    tbl = ca_table(rule); rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n).astype(np.int8)
    for _ in range(burn):
        s = ca_step(s, tbl)
    Q = np.eye(n); acc = np.zeros(n); idx = np.arange(n)
    for _ in range(T):
        l = np.roll(s, 1); c = s; r = np.roll(s, -1)
        base = tbl[(4 * l + 2 * c + r).astype(int)]
        dl = (tbl[(4 * (l ^ 1) + 2 * c + r).astype(int)] ^ base).astype(float)
        dc = (tbl[(4 * l + 2 * (c ^ 1) + r).astype(int)] ^ base).astype(float)
        dr = (tbl[(4 * l + 2 * c + (r ^ 1)).astype(int)] ^ base).astype(float)
        J = np.zeros((n, n))
        J[idx, (idx - 1) % n] = dl; J[idx, idx] = dc; J[idx, (idx + 1) % n] = dr
        Qn, R = np.linalg.qr(J @ Q)
        acc += np.log(np.abs(np.diag(R)) + 1e-300); Q = Qn
        s = ca_step(s, tbl)
    lam = acc / T / LN2
    return float(np.sum(lam[lam > 0]))


def action_ac(typ, p, n):
    """Dynamics-intrinsic action a_c (bits/step), NON-NEGATIVE = information PRODUCTION,
    measured as the extensive Pesin sum of positive Lyapunov exponents on a common
    footing across all three classes. Returns (a_c, was_clamped)."""
    if typ == "ca":
        if CA_AC_MODE == "extensive":
            return (ca_ks_spectrum(p["rule"], n), False)
        lam = ca_lyapunov(p["rule"], n)          # legacy intensive defect-max exponent
        return (max(0.0, lam), lam < 0.0)
    if typ == "cml":
        return (cml_ks_entropy(p["a"], p["eps"], n), False)
    if typ == "rcl":
        return (rcl_ks_entropy(p["a"], p["eps"], p["C"], n), False)
    raise ValueError(typ)


def curves_for(typ, p, n, ladder, caps, seeds, n_traj, T, burn):
    """Functional curves along the representation ladder. CA/CML delegate to the
    B2 pipeline unchanged; the RCL class mirrors it (same standardisation +
    functionals) but draws snapshots from collect_rcl."""
    if typ in ("ca", "cml"):
        return system_curves(typ, p, n, ladder, caps, seeds, n_traj, T, burn)
    base = (p["seed"] * 131) % 1000
    per = []
    for sd in seeds:
        X, Y = collect_rcl(n, p["a"], p["eps"], p["C"], n_traj, T, burn, base + sd)
        mu = X.mean(0); sdv = X.std(0) + 1e-9
        Zx, Zy = (X - mu) / sdv, (Y - mu) / sdv
        per.append([functionals(Zx, Zy, subs, n, caps) for (_, _, subs) in ladder])
    agg = []
    for li in range(len(ladder)):
        pts = [per[s][li] for s in range(len(seeds))]
        agg.append({k: float(np.mean([pt[k] for pt in pts])) for k in pts[0]})
    return agg


# Randomly-coupled recurrent-lattice specs (built once n is known): logistic chaos
# engine + random asymmetric ring coupling. The logistic param a (and coupling eps,
# radius R, coupling seed) sweep ordered->chaotic, spanning the a_c axis like the CMLs
# but through random asymmetric connectivity rather than symmetric diffusion.
RCL_SPECS = [
    ("RCL-a3.5",  3.5, 0.10, 2, 1, "rcl-ord"),
    ("RCL-a3.6",  3.6, 0.10, 2, 1, "rcl-ord"),
    ("RCL-a3.7",  3.7, 0.10, 2, 1, "rcl-edge"),
    ("RCL-a3.8",  3.8, 0.10, 2, 1, "rcl-edge"),
    ("RCL-a3.9",  3.9, 0.10, 2, 1, "rcl-chaos"),
    ("RCL-a4.0",  4.0, 0.10, 2, 1, "rcl-chaos"),
    ("RCL-a3.9b", 3.9, 0.30, 2, 2, "rcl-chaos"),
    ("RCL-a4.0b", 4.0, 0.30, 1, 2, "rcl-chaos"),
    ("RCL-a3.8c", 3.8, 0.20, 3, 3, "rcl-edge"),
]

def build_rcl_entries(n):
    out = []
    for (nm, a, eps, R, seed, cls) in RCL_SPECS:
        C = make_ring_C(n, R, seed)
        out.append((nm, "rcl", dict(a=a, eps=eps, C=C, R=R, seed=seed), cls))
    return out


# =========================================================================
# The scaled catalog: ~28 ECAs spanning Wolfram classes I-IV + 8 CMLs.
# Class labels follow Wolfram's standard classification. The point of the
# spread is to populate the a_c axis from ~0 (class I/II) to high (class III).
# =========================================================================
CATALOG = [
    # ---- Class I (homogeneous / quiescent): low a_c anchors ----
    ("CA-8",   "ca", dict(rule=8),   "I"),
    ("CA-128", "ca", dict(rule=128), "I"),
    ("CA-136", "ca", dict(rule=136), "I"),
    ("CA-160", "ca", dict(rule=160), "I"),
    # ---- Class II (periodic / low-moderate a_c) ----
    ("CA-1",   "ca", dict(rule=1),   "II"),
    ("CA-2",   "ca", dict(rule=2),   "II"),
    ("CA-4",   "ca", dict(rule=4),   "II"),
    ("CA-12",  "ca", dict(rule=12),  "II"),
    ("CA-56",  "ca", dict(rule=56),  "II"),
    ("CA-94",  "ca", dict(rule=94),  "II"),
    ("CA-108", "ca", dict(rule=108), "II"),
    ("CA-178", "ca", dict(rule=178), "II"),
    ("CA-184", "ca", dict(rule=184), "II"),
    ("CA-232", "ca", dict(rule=232), "II"),
    # ---- Class III (chaotic / high a_c) ----
    ("CA-18",  "ca", dict(rule=18),  "III"),
    ("CA-22",  "ca", dict(rule=22),  "III"),
    ("CA-30",  "ca", dict(rule=30),  "III"),
    ("CA-45",  "ca", dict(rule=45),  "III"),
    ("CA-60",  "ca", dict(rule=60),  "III"),
    ("CA-90",  "ca", dict(rule=90),  "III"),
    ("CA-105", "ca", dict(rule=105), "III"),
    ("CA-126", "ca", dict(rule=126), "III"),
    ("CA-146", "ca", dict(rule=146), "III"),
    ("CA-150", "ca", dict(rule=150), "III"),
    # ---- Class IV (complex / moderate a_c) ----
    ("CA-54",  "ca", dict(rule=54),  "IV"),
    ("CA-110", "ca", dict(rule=110), "IV"),
    ("CA-137", "ca", dict(rule=137), "IV"),
    ("CA-147", "ca", dict(rule=147), "IV"),
    # ---- Coupled logistic-map lattices (continuous): periodic -> chaotic ----
    ("CML-3.5", "cml", dict(a=3.5, eps=0.10), "periodic"),
    ("CML-3.6", "cml", dict(a=3.6, eps=0.10), "periodic"),
    ("CML-3.7", "cml", dict(a=3.7, eps=0.10), "chaotic"),
    ("CML-3.8", "cml", dict(a=3.8, eps=0.10), "chaotic"),
    ("CML-3.9", "cml", dict(a=3.9, eps=0.10), "chaotic"),
    ("CML-4.0", "cml", dict(a=4.0, eps=0.10), "chaotic"),
    ("CML-3.9c", "cml", dict(a=3.9, eps=0.30), "chaotic"),
    ("CML-4.0c", "cml", dict(a=4.0, eps=0.30), "chaotic"),
]


# =========================================================================
# Correlation inference: bootstrap CI + permutation null
# =========================================================================
def bootstrap_spearman(x, y, B=4000, seed=12345):
    """Percentile bootstrap CI for Spearman rho (resample pairs w/ replacement)."""
    rng = np.random.default_rng(seed)
    n = len(x); boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = spearman(x[idx], y[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(boot)), float(np.std(boot))

def permutation_pvalue(x, y, P=20000, seed=999):
    """Two-sided permutation p-value: shuffle y against x, |rho| >= observed."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(x, y)); n = len(x); ge = 0
    yc = y.copy()
    for _ in range(P):
        rng.shuffle(yc)
        if abs(spearman(x, yc)) >= obs - 1e-12:
            ge += 1
    return float((ge + 1) / (P + 1))


# =========================================================================
# Driver -- mirrors cwf_b2_bridge.main() settings exactly, only the catalog grows
# =========================================================================
def main():
    # IDENTICAL hyperparameters to the n=7 baseline (cwf_b2_bridge.main):
    n = 16; seeds = [0, 1]; n_traj, T, burn = 220, 14, 6
    c_bind = n // 4; caps = [n, c_bind]
    t0 = time.time()
    ladder = build_ladder(n)
    catalog = CATALOG + build_rcl_entries(n)
    print(f"T1.1 scale: {len(catalog)} substrates "
          f"({sum(t=='ca' for _,t,_,_ in catalog)} CA / "
          f"{sum(t=='cml' for _,t,_,_ in catalog)} CML / "
          f"{sum(t=='rcl' for _,t,_,_ in catalog)} RCL), "
          f"n={n}, ladder rungs={len(ladder)}, binding c={c_bind}\n")

    rows = {}; Fmats = {}; Cs = {}; degenerate = []; n_clamped = 0
    for (nm, typ, p, cls) in catalog:
        ag = curves_for(typ, p, n, ladder, caps, seeds, n_traj, T, burn)
        N = np.array([x["N"] for x in ag])
        L = np.array([1 - x["L_deg"] for x in ag])
        O = np.array([1 - x[f"O{c_bind}"] for x in ag])
        if not np.all(np.isfinite(N)) or not np.all(np.isfinite(O)):
            degenerate.append(nm)                       # quiescent substrate -> NaN guard
            continue
        F = np.column_stack([N, L, O]); Fmats[nm] = F
        v, q, C, _ = conserved_direction(F); Cs[nm] = C
        ac, clamped = action_ac(typ, p, n); n_clamped += int(clamped)
        rows[nm] = dict(wclass=cls, typ=typ, quality=float(q),
                        loadings=[float(x) for x in v],
                        N_rich=float(N[-1]), N_min=float(np.min(N)),
                        a_c=float(ac), a_c_clamped=bool(clamped),
                        a_c_src="CA-Lyap" if typ == "ca" else "Pesin-KS")
        print(f"  {nm:9s}{cls:>9s}  q={q:5.3f}  N_rich={N[-1]:6.3f}  "
              f"N_min={np.min(N):6.3f}  a_c={ac:6.3f}{'  [clamped<-neg]' if clamped else ''}")
    if degenerate:
        print(f"\n  [dropped {len(degenerate)} quiescent/degenerate: {degenerate}]")
    if n_clamped:
        print(f"  [{n_clamped} CA a_c clamped to 0 (contracting tangent; info prod = 0)]")

    nms = list(rows)
    # universal conserved direction at the larger n (does it stabilise vs n=7?)
    Cbar = np.mean([Cs[m] for m in nms], axis=0)
    wU, VU = np.linalg.eigh(Cbar); wdir = VU[:, 0]
    if wdir[0] < 0:
        wdir = -wdir
    Quniv = float(wU[0] / (wU.sum() + 1e-12))
    allF = np.vstack([Fmats[m] for m in nms])
    pmu = allF.mean(0); psd = allF.std(0) + 1e-12
    for m in nms:
        rows[m]["kappa_proj"] = float((((Fmats[m] - pmu) / psd) @ wdir).mean())

    # --- the bridge correlations, with inference ---
    Nr = np.array([rows[m]["N_rich"] for m in nms])
    Nm = np.array([rows[m]["N_min"] for m in nms])
    Kp = np.array([rows[m]["kappa_proj"] for m in nms])
    A = np.array([rows[m]["a_c"] for m in nms])

    rho_n = spearman(Nr, A); rho_nm = spearman(Nm, A); rho_k = spearman(Kp, A)
    lo, hi, bmean, bstd = bootstrap_spearman(Nr, A)
    pval = permutation_pvalue(Nr, A)
    lo_m, hi_m, _, _ = bootstrap_spearman(Nm, A)

    # robustness: graded law or just an ordered-vs-chaotic cluster jump?
    # re-measure on the genuinely ACTIVE subset (a_c above a small floor).
    active = A >= 0.05
    rho_act = spearman(Nr[active], A[active]) if active.sum() >= 4 else float("nan")
    lo_a, hi_a, _, _ = (bootstrap_spearman(Nr[active], A[active])
                        if active.sum() >= 6 else (float("nan"),) * 4)

    # robustness leg (Phase 1b): does the law hold WITHIN each substrate class,
    # not just pooled? (the "not CA/CML-specific" claim)
    typv = np.array([rows[m]["typ"] for m in nms])
    per_class = {}
    for cl in ["ca", "cml", "rcl"]:
        mask = typv == cl
        if mask.sum() >= 4:
            per_class[cl] = dict(n=int(mask.sum()),
                                 rho=spearman(Nr[mask], A[mask]))
            if mask.sum() >= 6:
                lc, hc, _, _ = bootstrap_spearman(Nr[mask], A[mask])
                per_class[cl]["ci95"] = [lc, hc]

    print(f"\n  universal conserved direction "
          f"({wdir[0]:+.2f},{wdir[1]:+.2f},{wdir[2]:+.2f}), Q={Quniv:.3f}  "
          f"(n=7 baseline Q=0.03-0.15)")
    print(f"\n(T2) bridge at n={len(nms)}:")
    print(f"   Spearman(N_rich, a_c) = {rho_n:+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]   perm p = {pval:.4f}")
    print(f"   Spearman(N_min,  a_c) = {rho_nm:+.3f}  95% CI [{lo_m:+.3f}, {hi_m:+.3f}]")
    print(f"   Spearman(kappa_proj, a_c) = {rho_k:+.3f}   (n=7 baseline: -0.54)")
    print(f"   [robustness] active subset (a_c>=0.05, n={int(active.sum())}): "
          f"Spearman(N_rich, a_c) = {rho_act:+.3f}  95% CI [{lo_a:+.3f}, {hi_a:+.3f}]")
    print(f"\n   per-class (not-CA/CML-specific?):")
    for cl in ["ca", "cml", "rcl"]:
        if cl in per_class:
            pc = per_class[cl]
            ci = (f"  95% CI [{pc['ci95'][0]:+.3f}, {pc['ci95'][1]:+.3f}]"
                  if "ci95" in pc else "")
            print(f"     {cl.upper():4s} (n={pc['n']:2d}): "
                  f"Spearman(N_rich, a_c) = {pc['rho']:+.3f}{ci}")
    print(f"\n   baseline (n=7): Spearman(N_rich, a_c) = +0.64")

    # --- persist (NEW key; B2_bridge untouched) ---
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    key = "T11_scale" if CA_AC_MODE == "defect" else f"T11_scale_{CA_AC_MODE}"
    r_all[key] = dict(
        n_lattice=n, binding_c=c_bind, n_substrates=len(nms),
        ca_ac_mode=CA_AC_MODE, dropped_degenerate=degenerate,
        hyperparams=dict(seeds=seeds, n_traj=n_traj, T=T, burn=burn,
                         note="identical to cwf_b2_bridge.main for apples-to-apples"),
        universal=dict(direction=[float(x) for x in wdir], quality=Quniv),
        by_system=rows,
        n_ca_clamped=n_clamped,
        bridge=dict(
            spearman_Nrich_ac=rho_n, spearman_Nmin_ac=rho_nm, spearman_kappaproj_ac=rho_k,
            Nrich_ci95=[lo, hi], Nrich_boot_mean=bmean, Nrich_boot_std=bstd,
            Nrich_perm_pvalue=pval, Nmin_ci95=[lo_m, hi_m],
            spearman_Nrich_ac_active=rho_act, Nrich_ci95_active=[lo_a, hi_a],
            n_active=int(active.sum()), per_class=per_class,
            baseline_n7_Nrich_ac=0.64, baseline_n7_kappaproj_ac=-0.54,
            note="N_rich = closure error at richest rung (matches B2); N_min = min over "
                 "ladder; CI = percentile bootstrap over substrates; p = two-sided "
                 "permutation null shuffling a_c against N_rich; active subset = a_c>=0.05 "
                 "(graded-law vs ordered/chaotic-cluster-jump check)"))
    json.dump(r_all, open(out, "w"), indent=2)

    plot_results(rows, rho_n, lo, hi, pval, Quniv)
    print(f"\nruntime {time.time()-t0:.1f}s\nWrote results.json key: {key} (CA a_c = {CA_AC_MODE})")


def plot_results(rows, rho_n, lo, hi, pval, Quniv):
    nms = list(rows)
    clscol = {"I": "#3b6", "II": "#39c", "III": "#e33", "IV": "#a3e",
              "periodic": "#fa0", "chaotic": "#f60",
              "rcl-ord": "#0aa", "rcl-edge": "#077", "rcl-chaos": "#044"}
    mkr = {"ca": "o", "cml": "^", "rcl": "s"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    a = ax[0]
    for m in nms:
        r = rows[m]
        a.scatter(r["a_c"], r["N_rich"], color=clscol.get(r["wclass"], "#888"),
                  marker=mkr.get(r["typ"], "o"), s=70,
                  edgecolors="k", linewidths=0.4)
    a.set_xlabel(r"intrinsic action $a_c$ (bits/step)")
    a.set_ylabel(r"best-closure floor $N_{\mathrm{rich}}$")
    a.set_title(f"trade-off bridge at n={len(nms)}:  "
                f"Spearman$(N_{{\\mathrm{{rich}}}},a_c)$={rho_n:+.2f}\n"
                f"95% CI [{lo:+.2f}, {hi:+.2f}],  perm p={pval:.4f}")
    a.grid(alpha=0.3)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markeredgecolor="k", label=k, markersize=8)
               for k, c in clscol.items()]
    a.legend(handles=handles, fontsize=7, title="Wolfram class", loc="best")

    a = ax[1]
    ac_all = np.array([rows[m]["a_c"] for m in nms])
    order = np.argsort(ac_all)
    a.bar(range(len(nms)), ac_all[order],
          color=[clscol.get(rows[nms[i]]["wclass"], "#888") for i in order])
    a.set_xticks(range(len(nms)))
    a.set_xticklabels([nms[i] for i in order], rotation=90, fontsize=6)
    a.set_ylabel(r"$a_c$ (bits/step)")
    a.set_title(f"a_c spectrum (sanity: I/II $\\approx$0 < IV < III);  "
                f"conserved-axis Q={Quniv:.3f}")
    a.grid(alpha=0.3, axis="y")

    fig.suptitle("T1.1 -- the trade-off law at scale (N_rich vs a_c bridge)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_T11_scale.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
