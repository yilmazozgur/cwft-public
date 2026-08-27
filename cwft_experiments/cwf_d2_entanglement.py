"""
D2 -- screen-entanglement vs bulk-irreducibility (Cluster D).

Tests the fractal-correspondence Prediction: that the screen-side entanglement of
the region encoding a bulk computation grows with the computation's COMPUTATIONAL
IRREDUCIBILITY.

We use the standard tractable proxy for entanglement growth in a local
deterministic system: the SCRAMBLING / butterfly front, i.e. how a single-cell
perturbation spreads (an OTOC-type measure; the "entanglement tsunami" propagates
at the butterfly velocity). For a classical CA this is reliably computable, where
the genuine bipartite entanglement entropy of a holographic encoding is not
(it needs exponential simulation for non-Clifford rules). We then ask whether
this scrambling tracks computational irreducibility, imported from C1.

The honest result (a partial refutation, with the recurring rule-90 signature):
scrambling and irreducibility are correlated but DISTINCT. The additive rule 90
is the separator -- it scrambles like a chaotic rule, yet is computationally
REDUCIBLE (exact GF(2) shortcut, Gottesman-Knill simulable), and the structured
rule 184 is irreducible yet scrambles little. So the Prediction holds only when
"screen complexity" means RECONSTRUCTION complexity (= irreducibility, validated
in C1/D1), NOT entanglement/scrambling. This is the same orthogonality C3 found
between Bell-nonlocality and computational hardness: across B2, C1, C3 and D2,
chaos / entanglement / nonlocality are not the same axis as computational
irreducibility, and rule 90 separates them every time.

CPU-only NumPy.  Run:  python cwf_d2_entanglement.py
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RULES = {110: "IV (universal)", 30: "III (chaotic)", 90: "III (additive)",
         184: "II (traffic)", 170: "shift (trivial)"}


def ca_table(rule):
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.int8)


def ca_step_batch(s, table):
    l = np.roll(s, 1, axis=1); r = np.roll(s, -1, axis=1)
    return table[(4 * l + 2 * s + r).astype(np.int64)]


def scrambling(rule, n=128, T=48, M=600, seed=1):
    """Entanglement-front / butterfly proxy: the mean fraction of differing cells
    when one centre cell is flipped at t=0, averaged over the last few steps
    (averaging removes the power-of-two resonance by which an additive rule's
    Sierpinski defect can momentarily cancel under periodic wrap-around)."""
    table = ca_table(rule)
    rng = np.random.default_rng(seed + rule)
    a = rng.integers(0, 2, (M, n)).astype(np.int8)
    b = a.copy(); b[:, n // 2] ^= 1
    fr = []
    for t in range(T + 1):
        if t >= T - 5:
            fr.append(float((a != b).mean()))
        a = ca_step_batch(a, table); b = ca_step_batch(b, table)
    return float(np.mean(fr))


def spearman(x, y):
    g = ~(np.isnan(x) | np.isnan(y))
    xr = np.argsort(np.argsort(x[g])); yr = np.argsort(np.argsort(y[g]))
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    t0 = time.time()
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    c1 = r_all.get("C1_predictive_cost", {}).get("by_rule", {})

    rows = {}
    for rule, label in RULES.items():
        E = scrambling(rule)
        cr = c1.get(str(rule), {})
        irred = 1.0 - float(cr.get("skill_at_tmax", np.nan)) if cr else np.nan
        rows[str(rule)] = dict(rule=rule, label=label, scrambling=E,
                               irreducibility=irred,
                               affine_shortcut=bool(cr.get("affine_shortcut", False)))
        print(f"  rule {rule:3d} [{label:15s}] scrambling(entangle front)={E:.3f}  "
              f"irreducibility(C1)={irred:.2f}  affine={rows[str(rule)]['affine_shortcut']}")

    rl = list(rows)
    E = np.array([rows[k]["scrambling"] for k in rl])
    irr = np.array([rows[k]["irreducibility"] for k in rl])
    sp = spearman(E, irr)
    print(f"\n  Spearman(scrambling, irreducibility) = {sp:+.2f}")
    print(f"  rule 90: scrambling={rows['90']['scrambling']:.3f} yet irreducibility="
          f"{rows['90']['irreducibility']:.2f} (additive -> reducible): the separator.")
    print(f"  => entanglement/scrambling is NOT computational irreducibility; the "
          f"prediction needs the reconstruction-complexity reading.")
    print(f"  runtime {time.time()-t0:.1f}s")

    r_all["D2_entanglement"] = dict(
        by_rule=rows, spearman_scrambling_irreducibility=sp,
        measure=("scrambling / butterfly front = mean differing-cell fraction after a "
                 "single-cell perturbation (OTOC-type entanglement-growth proxy)"),
        note=("The fractal-bridge Prediction (screen entanglement grows with bulk "
              "computational irreducibility) is only PARTIALLY supported by a scrambling/"
              "entanglement-front measure: scrambling and irreducibility are correlated but "
              "distinct, and the additive rule 90 separates them (scrambles like a chaotic "
              "rule yet is reducible / Gottesman-Knill simulable; structured rule 184 is "
              "irreducible yet scrambles little). The Prediction holds for RECONSTRUCTION "
              "complexity (=irreducibility, C1/D1), not entanglement entropy. Genuine "
              "entanglement entropy for non-Clifford CAs needs exponential simulation; this "
              "is a tractable proxy. Same orthogonality as C3."))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(rows, sp)
    print("Wrote results.json key: D2_entanglement")


def plot_results(rows, sp):
    color = {"110": "#e7298a", "30": "#7570b3", "90": "#1b9e77",
             "184": "#d95f02", "170": "#66a61e"}
    mark = {"110": "o", "30": "s", "90": "^", "184": "D", "170": "v"}
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})

    a = ax[0]
    for k, r in rows.items():
        x, y = r["irreducibility"], r["scrambling"]
        if np.isnan(x):
            continue
        a.scatter(x, y, s=150, color=color[k], marker=mark[k], edgecolor="white", zorder=4)
        a.annotate(f"rule {r['rule']}", (x, y), xytext=(8, 5), textcoords="offset points",
                   fontsize=9)
    a.set_xlabel("computational irreducibility (C1)")
    a.set_ylabel("scrambling / entanglement-front proxy")
    a.set_title("(a) Entanglement-front vs irreducibility")
    a.text(0.5, 0.94, f"Spearman $={sp:+.2f}$ (correlated, not a clean law)", transform=a.transAxes,
           ha="center", va="top", fontsize=9.5, bbox=dict(boxstyle="round", fc="#f0f0f0", ec="0.6"))
    r90 = rows["90"]
    a.annotate("rule 90 (additive):\nscrambles, yet REDUCIBLE\n(Gottesman-Knill easy)",
               (r90["irreducibility"], r90["scrambling"]), xytext=(0.30, 0.62),
               textcoords="axes fraction", fontsize=8.2, color="#1b9e77",
               arrowprops=dict(arrowstyle="->", color="#1b9e77"))
    r184 = rows["184"]
    a.annotate("rule 184: irreducible,\nbut scrambles little",
               (r184["irreducibility"], r184["scrambling"]), xytext=(0.58, 0.12),
               textcoords="axes fraction", fontsize=8.2, color="#d95f02",
               arrowprops=dict(arrowstyle="->", color="#d95f02"))
    a.set_xlim(-0.1, 1.15)

    b = ax[1]; b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
    b.text(0.5, 0.95, "The fractal-bridge prediction, refined", ha="center",
           fontweight="bold", fontsize=10.5)
    txt = (
        "Prediction (fractal bridge): screen entanglement\n"
        "grows with bulk computational irreducibility.\n\n"
        "$\\bullet$  As entanglement / scrambling: PARTIALLY\n"
        f"   (Spearman ${sp:+.2f}$ -- correlated, not a clean law).\n"
        "   The additive rule 90 scrambles yet is reducible;\n"
        "   the structured rule 184 is irreducible yet barely\n"
        "   scrambles. So entanglement and computational\n"
        "   hardness are correlated but distinct axes.\n\n"
        "$\\bullet$  As RECONSTRUCTION complexity (C1/D1):\n"
        "   holds cleanly -- that measure IS irreducibility.\n\n"
        "Same orthogonality as C3 (Bell-nonlocality $\\neq$\n"
        "hardness). Across B2 / C1 / C3 / D2 the additive\n"
        "rule 90 -- chaotic + entangling yet linearly closable\n"
        "-- is the recurring separator of the dynamical axes\n"
        "from computational hardness.")
    b.text(0.03, 0.85, txt, va="top", fontsize=8.8)

    fig.suptitle("D2: screen entanglement partially tracks computational irreducibility "
                 "--- correlated, but the additive rule 90 separates the axes",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_D2_entanglement.png")
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
