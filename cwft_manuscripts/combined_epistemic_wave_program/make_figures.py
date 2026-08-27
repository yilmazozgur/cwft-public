#!/usr/bin/env python3
"""Generate the four figures for the epistemic-wave / irreducibility-axis paper (paper B).

Reads the committed experiment result JSONs from the CWF experiments directory and emits
publication PNGs (300 dpi) into ./figures/. No experiments are re-run: the figures are
plotted from the seeded reference values, so they are bit-reproducible.

Figures:
  fig_tradeoff.png        Program I: closure floor N* vs information production h_t (28 CAs),
                          additive/linear rules flagged as the genuine bound-violators.
  fig_compression.png     Amplitude decomposition (compression clause): depth-unfolded C_mu vs
                          C_q on genuine-memory rules (<=55%); chaotic rules flagged as the
                          finite-sample C_mu artifact the q-machine denoises (not compression).
  fig_contextuality.png   Amplitude decomposition (phase clause): contextual fraction --
                          epistemic restrictions ~0 vs a genuine qubit 0.414, PR box 1.0.
  fig_costwall.png        Program II shadow: cheapest-predictor parameter count vs horizon;
                          irreducible rules wall, reducible rules grow linearly.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.environ.get(
    "CWF_EXP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "cwft_experiments"),
)
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 300


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Figure 1 -- Program I trade-off: closure floor N* vs information production h_t (CAs).
# --------------------------------------------------------------------------------------
def fig_tradeoff():
    d = load("results.json")["T11_ca_faithful"]
    per = d["per_rule"]
    rho = d["within_ca_spearman_faithful"]
    xs_n, ys_a, add_n, add_a, lab_a = [], [], [], [], []
    for k, v in per.items():
        if v["additive"]:
            add_n.append(v["N_rich"])
            add_a.append(v["h_t"])
            lab_a.append(v["rule"])
        else:
            xs_n.append(v["N_rich"])
            ys_a.append(v["h_t"])
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.scatter(xs_n, ys_a, s=42, c="#2c6fbb", edgecolors="k", linewidths=0.4,
               label="non-additive rules", zorder=3)
    ax.scatter(add_n, add_a, s=72, marker="D", c="#d1492f", edgecolors="k",
               linewidths=0.5, label="additive / linear rules\n(genuine violators: 60, 90, 105, 150)",
               zorder=4)
    ax.set_xlabel(r"closure floor $N^\star$ (richest-dictionary EDMD residual)")
    ax.set_ylabel(r"information production $h_t$ (bits/step)")
    ax.set_title("Program I: closure floor vs information production")
    ax.text(0.04, 0.06, rf"$\rho_{{\rm Spearman}}=+{rho:.2f}$"
            "\n(28 elementary CAs)", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_xlim(-0.04, 0.86)
    ax.set_ylim(-0.06, 1.12)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_tradeoff.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------------------
# Figure 2 -- compression clause: C_mu vs C_q per rule (q-machine).
# --------------------------------------------------------------------------------------
def fig_compression():
    # Depth-unfolded q-machine (psi_scaleup). The GENUINE compression advantage is on the
    # structured (genuine-memory) rules 110/54/184 (<=55%). The chaotic rules 30/90 show much
    # larger APPARENT savings, but those are the q-machine denoising a finite-sample inflation
    # of C_mu (a fair coin reconstructs to ~5 bits), not genuine structure -- so they are shown
    # hatched and flagged, not counted. C_q is the noise-robust estimator.
    d = load("psi_scaleup_results.json")["substrates"]
    byrule = {v["rule"]: (v["C_mu_bits"], v["Cq_depth_bits"]) for v in d.values()}
    genuine = [(110, "Rule 110\n(universal)"), (54, "Rule 54\n(class IV)"),
               (184, "Rule 184\n(class II)")]
    artifact = [(30, "Rule 30\n(chaotic)"), (90, "Rule 90\n(additive)")]
    rules = genuine + artifact
    n_gen = len(genuine)
    names = [nm for _, nm in rules]
    cmu = [byrule[r][0] for r, _ in rules]
    cq = [byrule[r][1] for r, _ in rules]
    w = 0.38
    top = max(cmu) + 1.6
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i in range(len(rules)):
        hatch = None if i < n_gen else "//"
        a = 1.0 if i < n_gen else 0.5
        ax.bar(i - w / 2, cmu[i], w, color="#9aa7b3", edgecolor="k", linewidth=0.4,
               hatch=hatch, alpha=a,
               label=(r"$C_\mu$ (classical $\epsilon$-machine)" if i == 0 else None))
        ax.bar(i + w / 2, cq[i], w, color="#2c6fbb", edgecolor="k", linewidth=0.4,
               hatch=hatch, alpha=a,
               label=(r"$C_q$ ($q$-machine, real amplitudes)" if i == 0 else None))
        save = 100 * (1 - cq[i] / cmu[i])
        col = "#7a2417" if i < n_gen else "#8a8a8a"
        ax.annotate(f"-{save:.0f}%", (i, max(cmu[i], cq[i]) + 0.12), ha="center",
                    fontsize=8.5, color=col)
    ax.axvline(n_gen - 0.5, ls=":", lw=1.0, color="0.5")
    ax.annotate("apparent only:\n$C_\\mu$ finite-sample artifact\n(not genuine compression)",
                ((n_gen + len(rules) - 1) / 2, top - 0.85), ha="center", fontsize=8,
                color="#8a8a8a")
    ax.set_xticks(range(len(rules)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("statistical complexity (bits)")
    ax.set_title("Compression advantage (depth-unfolded $q$-machine)")
    ax.set_ylim(0, top)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_compression.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------------------
# Figure 3 -- phase clause: contextual fraction across epistemic restrictions vs quantum.
# --------------------------------------------------------------------------------------
def fig_contextuality():
    pc = load("phase_contextuality_results.json")["substrates"]
    sp = load("spekkens_restriction_results.json")
    bars = []  # (label, value, color)
    # epistemic restrictions over a definite ground -- all ~0 (sampling floor)
    for k in ["Rule110", "Rule30", "Rule90"]:
        bars.append((f"{k}\n(coarse-grain)", pc[k]["contextual_fraction"], "#9aa7b3"))
    bars.append(("Spekkens\ntoy theory", sp["spekkens_check"].get("CF", 0.0)
                 if isinstance(sp["spekkens_check"].get("CF", 0.0), float) else 0.0, "#9aa7b3"))
    bars.append(("Rule110\nknow.-balance", sp["rule110_kb"]["contextual_fraction"], "#9aa7b3"))
    # quantum references
    bars.append(("genuine qubit\n(Tsirelson)", sp["quantum_CF"], "#2c6fbb"))
    bars.append(("PR box\n(super-quantum)", 1.0, "#d1492f"))
    labels = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(range(len(bars)), vals, color=cols, edgecolor="k", linewidth=0.4)
    ax.axhline(sp["quantum_CF"], ls="--", lw=0.8, color="#2c6fbb", alpha=0.7)
    ax.annotate(r"qubit $0.414$", (0.1, sp["quantum_CF"] + 0.02), fontsize=8, color="#2c6fbb")
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v + 0.015), ha="center", fontsize=7.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("contextual fraction CF")
    ax.set_title("No phase from epistemic restriction over a definite ground")
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_contextuality.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------------------
# Figure 4 -- Program II shadow: cheapest-predictor cost vs horizon (the cost wall).
# --------------------------------------------------------------------------------------
def fig_costwall():
    d = load("results.json")["C1_cost_scaling"]
    by = d["by_rule"]
    # Rule 30 coincides with Rule 110 (wall at t=8); Rule 170 with Rule 90 (linear).
    # Collapse the coincident pairs into representative behaviour classes.
    reps = [
        ("110", "Rules 110 / 30 (irreducible)", "#d1492f", "o", "-"),
        ("184", "Rule 184 (intermediate)", "#e08a2f", "^", "-"),
        ("90", "Rules 90 / 170 (reducible)", "#2c6fbb", "v", "--"),
    ]
    fig, ax = plt.subplots(figsize=(5.7, 4.1))
    for rule, label, color, mk, ls in reps:
        if rule not in by:
            continue
        ts, ps = [], []
        last_ok = None
        wall = None
        for row in by[rule]["rows"]:
            rp = row.get("req_params")
            if rp is not None:
                ts.append(row["t"])
                ps.append(rp)
                last_ok = row["t"]
            else:
                wall = row["t"]
                break
        ax.plot(ts, ps, marker=mk, ls=ls, color=color, label=label, ms=5, lw=1.5)
        if wall is not None:
            ax.scatter([wall], [ps[-1] * 2.0], marker="x", s=80, color=color, zorder=5)
            ax.annotate("wall", (wall, ps[-1] * 2.4), ha="center", fontsize=8.5, color=color)
    ax.set_yscale("log")
    ax.set_xlabel("forecast horizon $t$ (steps)")
    ax.set_ylabel("cheapest predictor: parameter count to reach skill $\\geq 0.75$")
    ax.set_title("Predictive-cost wall (finite shadow of irreducibility)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_costwall.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_tradeoff()
    fig_compression()
    fig_contextuality()
    fig_costwall()
