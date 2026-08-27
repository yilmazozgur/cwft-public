"""
B2 (SCAFFOLD) -- the bridge test: does the Program-I trade-off invariant kappa
track the substrate Planck constant hbar_c?  (Chapter 6, Sec 6.4 / Conj kappa-hbarc;
Chapter 2 calibration catalog; Chapter 5 eta_c=1 anchor.)

This scaffold confronts a conceptual problem the bridge conjecture has to survive
before any correlation is meaningful:

  hbar_c = (k_B T ln2) * tau_min  is set by the substrate's PHYSICAL HOST
  (operating temperature T, minimal event time tau_min). It spans the catalog's
  twelve orders of magnitude precisely because T and tau_min vary across hosts.
  It does NOT depend on the COMPUTATION: the same Rule 110 has one hbar_c on a
  CMOS chip and another, 1e7x larger, run per-layer on a GPU.

  kappa (the B1 invariant kappa = N + a(1-L) + b(1-O)) is computed from the
  DYNAMICS and is host-independent in our pipeline.

So the naive bridge kappa = hbar_c is GAUGE-DEPENDENT (ill-posed): you can move
hbar_c over orders of magnitude at fixed kappa by re-hosting the same computation.
And the dimensionless form hbar_c/(k_B T tau_min) = ln2 is a CONSTANT for every
Landauer-calibrated substrate -- so the dimensionless bridge predicts kappa to be
universal, which B1 already refuted (no universal (a,b)).

The well-posed remnant the scaffold therefore tests is a bridge between two
DYNAMICS-INTRINSIC, dimensionless quantities:

  kappa            -- the Program-I trade-off invariant (host-independent), and
  a_c              -- information production per step (entropy rate, bits/step):
                      the implementation-stripped core of hbar_c. hbar_c packages
                      (k_B T)(tau_min)(bits-per-event); a_c is the bits-per-event
                      content alone, read directly from the dynamics.

Deliverables of this first pass:
  (T1) GAUGE DEMONSTRATION -- hold the computation fixed (Rule 110), sweep the
       host across the catalog: hbar_c moves 12 orders while kappa and a_c are
       invariant. Then hold the host fixed, vary the computation: hbar_c is flat
       while kappa and a_c vary. => hbar_c _|_ (kappa, a_c). Naive bridge ill-posed.
  (T2) WELL-POSED BRIDGE -- Spearman rank correlation of kappa vs a_c across the
       simulable substrate set (CA classes II/III/IV + coupled-map lattices).
       Reported honestly whatever the sign/strength.
  (T3) ANCHOR -- Chapter 5's perfect-tensor eta_c=1 fixes G_c=1/(4 hbar_c): one
       calibration point that lives on the gravitational face of the bridge.

Validation: for the continuous lattices the entropy-rate a_c is cross-checked
against the Pesin KS-entropy (sum of positive Lyapunov exponents via Jacobian QR).

Reuses substrate collectors / monomial machinery from cwf_b1_tradeoff.py.
Out of scope (hooks left): the physical hosts (GPU NN, brain, internet) are not
simulable in this pipeline -- only their hbar_c is catalogued; a real kappa for
them needs the substrate's own state to be observed (future work, with C-cluster).
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_b1_tradeoff import (ca_table, ca_step, cml_step,
                             collect_ca, collect_cml,
                             master_subsets, ladder_dict, functionals,
                             conserved_direction)


# =========================================================================
# Physical constants and the Chapter-2 host catalog
# =========================================================================
kB = 1.380649e-23; HBAR = 1.054571817e-34; LN2 = float(np.log(2))

HOSTS = {                      # name: (T[K], tau_min[s])  -- book Sec 2.6
    "CMOS-R110":    (350, 1e-9),
    "FFNN-H100":    (350, 1e-10),
    "Transf-cyc":   (350, 5e-10),
    "Transf-layer": (350, 1e-4),
    "Brain":        (310, 1e-3),
    "Internet":     (300, 5e-2),
}

def hbar_c(T, tau):
    return kB * T * LN2 * tau


# =========================================================================
# Dynamics-intrinsic action a_c: information production per step (bits/step)
# =========================================================================

def ca_lyapunov(rule, n, T=600, burn=60, seed=0, n_rep=4):
    """Maximal CA Lyapunov exponent (bits/step), Bagnoli-Rechtman-Ruffo defect
    method: a nonnegative tangent field is propagated by the local Boolean
    Jacobian and renormalised each step; lambda = <ln(growth)>/ln2. Robust where
    the plug-in block-entropy estimator is undersampling-biased. Sanity: the
    additive rule 90 (e'_i = e_{i-1}+e_{i+1}) gives exactly 1 bit/step."""
    tbl = ca_table(rule); accs = []
    for rep in range(n_rep):
        rng = np.random.default_rng(900 + 17 * seed + rep)
        s = rng.integers(0, 2, n).astype(np.int8)
        for _ in range(burn):
            s = ca_step(s, tbl)
        e = rng.random(n) + 1e-3; e /= e.sum(); acc = 0.0
        for _ in range(T):
            l = np.roll(s, 1); c = s; r = np.roll(s, -1)
            base = tbl[(4 * l + 2 * c + r).astype(int)]
            dl = (tbl[(4 * (l ^ 1) + 2 * c + r).astype(int)] ^ base).astype(float)
            dc = (tbl[(4 * l + 2 * (c ^ 1) + r).astype(int)] ^ base).astype(float)
            dr = (tbl[(4 * l + 2 * c + (r ^ 1)).astype(int)] ^ base).astype(float)
            e = dl * np.roll(e, 1) + dc * e + dr * np.roll(e, -1)
            S = e.sum()
            if S <= 1e-300:
                acc += np.log(1e-12); e = rng.random(n) + 1e-3; e /= e.sum()
            else:
                acc += np.log(S); e /= S
            s = ca_step(s, tbl)
        accs.append(acc / T / LN2)
    return float(np.mean(accs))

def cml_ks_entropy(a, eps, n, T=600, burn=80, seed=0):
    """Pesin KS entropy = sum of positive Lyapunov exponents (bits/step), via
    Jacobian QR. Cross-check for the CML entropy-rate ordering."""
    rng = np.random.default_rng(seed); x = rng.random(n)
    for _ in range(burn):
        x = cml_step(x, a, eps)
    Q = np.eye(n); acc = np.zeros(n); idx = np.arange(n)
    for _ in range(T):
        fp = a * (1 - 2 * x)
        J = np.zeros((n, n))
        J[idx, idx] = (1 - eps) * fp
        J[idx, (idx - 1) % n] = 0.5 * eps * fp[(idx - 1) % n]
        J[idx, (idx + 1) % n] = 0.5 * eps * fp[(idx + 1) % n]
        Qn, R = np.linalg.qr(J @ Q)
        acc += np.log(np.abs(np.diag(R)) + 1e-300); Q = Qn
        x = cml_step(x, a, eps)
    lam = acc / T                       # nats/step
    return float(np.sum(lam[lam > 0]) / LN2)   # bits/step


# =========================================================================
# kappa with UNIVERSAL (host-independent) weights, comparable across substrates
# =========================================================================

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

def system_curves(typ, params, n, ladder, caps, seeds, n_traj, T, burn):
    base = hash(str(params)) % 1000
    per = []
    for sd in seeds:
        X, Y = (collect_ca(params["rule"], n, n_traj, T, burn, base + sd) if typ == "ca"
                else collect_cml(params["a"], params["eps"], n, n_traj, T, burn, base + sd))
        mu = X.mean(0); sdv = X.std(0) + 1e-9
        Zx, Zy = (X - mu) / sdv, (Y - mu) / sdv
        per.append([functionals(Zx, Zy, subs, n, caps) for (_, _, subs) in ladder])
    agg = []
    for li in range(len(ladder)):
        pts = [per[s][li] for s in range(len(seeds))]
        agg.append({key: float(np.mean([p[key] for p in pts])) for key in pts[0]})
    return agg

# =========================================================================
# Driver
# =========================================================================

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))

def main():
    n = 16; seeds = [0, 1]; n_traj, T, burn = 220, 14, 6; caps = [n, n // 2]
    systems = [
        ("CA-184", "ca", dict(rule=184), "II"),
        ("CA-110", "ca", dict(rule=110), "IV"),
        ("CA-90",  "ca", dict(rule=90),  "III"),
        ("CA-30",  "ca", dict(rule=30),  "III"),
        ("CML-3.6", "cml", dict(a=3.6, eps=0.10), "periodic"),
        ("CML-3.8", "cml", dict(a=3.8, eps=0.10), "chaotic"),
        ("CML-4.0", "cml", dict(a=4.0, eps=0.30), "chaotic"),
    ]
    t0 = time.time()
    ladder = build_ladder(n)
    c_bind = n // 4; caps = [n, c_bind]    # O at a binding capacity (corrected B1)

    # --- per-system functional curves ---
    curves = {nm: system_curves(typ, p, n, ladder, caps, seeds, n_traj, T, burn)
              for (nm, typ, p, _) in systems}

    # --- corrected conserved combination per system (min-variance, binding O) ---
    # Under the corrected B1 metric kappa is a DIRECTION, not a scalar; we keep its
    # quality q and loadings, and for the cross-substrate bridge use two well-defined
    # scalars: N_rich (the fit-free best-closure floor) and kappa_proj (position
    # along the universal conserved axis, pooled-standardised -> comparable).
    rows = {}; Fmats = {}; Cs = {}
    for (nm, typ, p, cls) in systems:
        ag = curves[nm]
        N = np.array([x["N"] for x in ag]); L = np.array([1 - x["L_deg"] for x in ag])
        O = np.array([1 - x[f"O{c_bind}"] for x in ag])
        F = np.column_stack([N, L, O]); Fmats[nm] = F
        v, q, C, _ = conserved_direction(F); Cs[nm] = C
        ac = ca_lyapunov(p["rule"], n) if typ == "ca" else cml_ks_entropy(p["a"], p["eps"], n)
        rows[nm] = dict(wclass=cls, typ=typ, quality=q,
                        loadings=[float(x) for x in v], N_rich=float(N[-1]),
                        a_c=ac, a_c_src="CA-Lyap" if typ == "ca" else "Pesin-KS")

    nms = list(rows)
    Cbar = np.mean([Cs[m] for m in nms], axis=0)
    wU, VU = np.linalg.eigh(Cbar); wdir = VU[:, 0]
    if wdir[0] < 0:
        wdir = -wdir
    Quniv = float(wU[0] / (wU.sum() + 1e-12))
    allF = np.vstack([Fmats[m] for m in nms])
    pmu = allF.mean(0); psd = allF.std(0) + 1e-12
    for m in nms:
        rows[m]["kappa_proj"] = float((((Fmats[m] - pmu) / psd) @ wdir).mean())

    print(f"corrected B2: binding c={c_bind}; universal conserved direction "
          f"({wdir[0]:+.2f},{wdir[1]:+.2f},{wdir[2]:+.2f}), Q={Quniv:.3f}\n")
    print(f"{'system':9s}{'cls':>9s}{'q':>7s}{'kappa_proj':>11s}{'N_rich':>8s}"
          f"{'a_c':>7s}  source")
    for m in nms:
        r = rows[m]
        print(f"{m:9s}{r['wclass']:>9s}{r['quality']:7.3f}{r['kappa_proj']:11.3f}"
              f"{r['N_rich']:8.3f}{r['a_c']:7.3f}  {r['a_c_src']}")

    # --- T2 bridge: corrected kappa scalar (and fit-free N_rich) vs a_c ---
    Kp = np.array([rows[m]["kappa_proj"] for m in nms])
    Nr = np.array([rows[m]["N_rich"] for m in nms])
    A = np.array([rows[m]["a_c"] for m in nms])
    rho_k = spearman(Kp, A); rho_n = spearman(Nr, A)
    print(f"\n(T2) bridge:  Spearman(kappa_proj, a_c) = {rho_k:+.2f}   "
          f"Spearman(N_rich, a_c) = {rho_n:+.2f}")

    # --- T1: gauge demonstration (metric-independent) ---
    r110 = rows["CA-110"]
    host_hbar = {h: hbar_c(*HOSTS[h]) for h in HOSTS}
    print("\n(T1) gauge demonstration:")
    print("  fixed computation (Rule 110), sweep host -> hbar_c moves, kappa_proj/a_c fixed:")
    for h in HOSTS:
        print(f"     host {h:13s}: hbar_c={host_hbar[h]:.2e} J.s   "
              f"(kappa_proj={r110['kappa_proj']:+.3f}, a_c={r110['a_c']:.3f} fixed)")
    span = max(host_hbar.values()) / min(host_hbar.values())
    print(f"     -> hbar_c spans {span:.1e} at fixed (kappa_proj,a_c).")
    print("  fixed host (CMOS), vary computation -> hbar_c flat, kappa_proj/a_c vary:")
    hb_fixed = hbar_c(*HOSTS["CMOS-R110"])
    for m in nms:
        print(f"     {m:9s}: hbar_c={hb_fixed:.2e} (flat)  "
              f"kappa_proj={rows[m]['kappa_proj']:+.3f} a_c={rows[m]['a_c']:.3f}")

    # --- persist + plot ---
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["B2_bridge"] = dict(
        n=n, binding_c=c_bind,
        universal=dict(direction=[float(x) for x in wdir], quality=Quniv),
        by_system=rows,
        bridge=dict(spearman_kappaproj_ac=rho_k, spearman_Nrich_ac=rho_n,
                    note="corrected metric: kappa is a conserved DIRECTION; "
                         "kappa_proj = mean position along the universal direction "
                         "(pooled-standardised); N_rich = fit-free best-closure floor"),
        gauge=dict(host_hbar_c={h: host_hbar[h] for h in HOSTS},
                   hbar_c_span_fixed_computation=float(span),
                   note="hbar_c set by host (T,tau_min); orthogonal to the dynamics scalars"),
        anchor=dict(eta_c=1.0, G_c="1/(4 hbar_c)", source="Chapter 5 perfect tensor"))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(rows, host_hbar, rho_k, rho_n, span)
    print(f"\nruntime {time.time()-t0:.1f}s\nWrote results.json key: B2_bridge")


def plot_results(rows, host_hbar, rho_k, rho_n, span):
    nms = list(rows); cmap = plt.cm.turbo
    cols = {m: cmap(i / max(len(nms) - 1, 1)) for i, m in enumerate(nms)}
    mk = {"ca": "o", "cml": "^"}
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) the well-posed bridge: kappa_proj vs a_c
    a = ax[0, 0]
    for m in nms:
        r = rows[m]
        a.scatter(r["a_c"], r["kappa_proj"], color=cols[m], marker=mk[r["typ"]],
                  s=90, edgecolors="k", linewidths=0.4)
        a.annotate(f"{m}", (r["a_c"], r["kappa_proj"]), fontsize=7,
                   textcoords="offset points", xytext=(5, 3))
    a.set_xlabel(r"intrinsic action $a_c$ (bits/step)")
    a.set_ylabel(r"$\kappa_{\mathrm{proj}}$ (universal conserved axis)")
    a.set_title(f"(a) bridge: Spearman$(\\kappa_{{\\mathrm{{proj}}}},a_c)$={rho_k:+.2f}")
    a.grid(alpha=0.3)

    # (b) a_c by substrate (ordering sanity: class II ~ 0, III/IV/chaotic > 0)
    a = ax[0, 1]
    a.bar(range(len(nms)), [rows[m]["a_c"] for m in nms],
          color=[cols[m] for m in nms])
    a.set_xticks(range(len(nms)))
    a.set_xticklabels([f"{m}\n({rows[m]['wclass']})" for m in nms], rotation=35,
                      fontsize=6.5, ha="right")
    a.set_ylabel(r"$a_c$ (bits/step): CA-Lyapunov / Pesin-KS")
    a.set_title("(b) intrinsic information production by substrate"); a.grid(alpha=0.3, axis="y")

    # (c) gauge demo: hbar_c by host (fixed computation) -- spans orders
    a = ax[1, 0]
    hs = list(host_hbar); vals = [host_hbar[h] for h in hs]
    a.bar(range(len(hs)), vals, color="0.6")
    a.set_yscale("log"); a.set_xticks(range(len(hs)))
    a.set_xticklabels(hs, rotation=35, fontsize=7, ha="right")
    a.set_ylabel(r"$\hbar_c$ (J$\cdot$s), Rule 110 fixed")
    a.set_title(f"(c) gauge: $\\hbar_c$ spans {span:.0e} at fixed $\\kappa,a_c$")
    a.grid(alpha=0.3, axis="y")

    # (d) summary
    a = ax[1, 1]; a.axis("off")
    txt = "B2 bridge scaffold (corrected B1 metric):\n\n"
    txt += f"{'system':9s}{'cls':>5s}{'k_proj':>8s}{'N_rich':>7s}{'a_c':>6s}\n"
    for m in nms:
        r = rows[m]
        txt += (f"{m:9s}{r['wclass']:>5s}{r['kappa_proj']:8.3f}"
                f"{r['N_rich']:7.3f}{r['a_c']:6.2f}\n")
    txt += ("\n Reads:\n"
            "  (T1) hbar_c = (k_B T ln2) tau_min is HOST-set: it moves\n"
            f"     {span:.0e}x for fixed Rule 110 across the catalog,\n"
            "     while the dynamics scalars (kappa_proj, a_c) stay put.\n"
            "     Same host, different computation: hbar_c flat, they vary.\n"
            "     So the naive bridge kappa=hbar_c is gauge-dependent, and\n"
            "     hbar_c/(k_B T tau)=ln2 is constant => would force kappa\n"
            "     universal, which corrected B1 refutes.\n"
            "  (T2) well-posed remnant: corrected kappa vs the dynamics-\n"
            f"     intrinsic action a_c. Spearman(kappa_proj,a_c)={rho_k:+.2f},\n"
            f"     Spearman(N_rich,a_c)={rho_n:+.2f} -- partial. Rule 90 the\n"
            "     outlier (max a_c, low cost: XOR linearly closable).\n"
            "  (T3) anchor: perfect-tensor eta_c=1 => G_c=1/(4 hbar_c).\n\n"
            " NB kappa is a conserved DIRECTION (corrected metric); kappa_proj\n"
            " is its position along the universal axis. Hosts GPU/brain/\n"
            " internet are catalogued in hbar_c but not simulable for kappa.")
    a.text(0.02, 0.98, txt, transform=a.transAxes, fontsize=8, family="monospace", va="top")

    fig.suptitle("B2 (scaffold) --- the $\\kappa$-$\\hbar_c$ bridge: gauge separation "
                 "+ the well-posed dynamics-intrinsic remnant", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_B2_bridge.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
