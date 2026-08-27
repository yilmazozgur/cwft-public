#!/usr/bin/env python3
"""Figures for the P1 two-knob section, plotted from the committed results JSONs
(no experiments re-run; bit-reproducible from the seeded reference values).

  fig_twoknob_mipt.png : I3 four-quarter crossing at p_c ~ 0.16 + S(N/2) volume->area
                         (from ap_phaseF_twoknob_results.json)
  fig_twoknob_rt.png   : S(L) vs conformal chord; only the critical curve is straight
                         (from ap_phaseG_rt_results.json)

Usage: python3 make_figures_twoknob.py   (writes into MDPI_template_APA/figures/)
"""
import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "..", "cwft_experiments")
OUT = os.path.join(HERE, "MDPI_template_APA", "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9.5})

# ---------------------------------------------------------------- fig 1: MIPT
d = json.load(open(os.path.join(EXP, "ap_phaseF_twoknob_results.json")))
p = np.array(d["setup"]["p_grid"], float)
Ns = d["setup"]["N_list"]
p_c = d["analysis"]["p_c_crossing"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), constrained_layout=True)
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(Ns)))
for c, N in zip(cmap, Ns):
    ax1.plot(p, d["I3"][str(N)], "o-", ms=3, lw=1.2, color=c, label=f"$N={N}$")
ax1.axvline(p_c, color="crimson", ls="--", lw=1.0)
ax1.axvline(0.5, color="gray", ls=":", lw=1.4)
ax1.text(p_c + 0.008, ax1.get_ylim()[0] * 0.97, r"$p_c\approx0.16$", color="crimson",
         fontsize=8, va="bottom")
ax1.text(0.5 + 0.008, -8.8, "one-knob lock\n$p=1/2$", color="gray", fontsize=8)
ax1.set_xlabel(r"monitoring rate $p=\hbar_c/(\hbar_c+\theta_c)$")
ax1.set_ylabel(r"tripartite mutual information $I_3$")
ax1.set_title(r"(a) $I_3$ $N$-curves cross at $p_c$")
ax1.legend(fontsize=7, frameon=False)

for c, N in zip(cmap, Ns):
    ax2.plot(p, d["S_half"][str(N)], "s-", ms=3, lw=1.2, color=c, label=f"$N={N}$")
ax2.axvline(p_c, color="crimson", ls="--", lw=1.0)
ax2.axvline(0.5, color="gray", ls=":", lw=1.4)
ax2.set_xlabel(r"monitoring rate $p$")
ax2.set_ylabel(r"half-chain entropy $S(N/2)$ (bits)")
ax2.set_title(r"(b) volume law below $p_c$, area law above")
ax2.legend(fontsize=7, frameon=False)

fig.savefig(os.path.join(OUT, "fig_twoknob_mipt.png"), dpi=300)
plt.close(fig)
print("wrote fig_twoknob_mipt.png  (p_c crossing:", p_c,
      "| slopes low/high p:", d["analysis"]["slope_lowp"], d["analysis"]["slope_highp"], ")")

# ----------------------------------------------------------- fig 2: RT critical line
g = json.load(open(os.path.join(EXP, "ap_phaseG_rt_results.json")))
N = g["setup"]["N"]
L = np.array(g["L"], float)
X = np.log((N / math.pi) * np.sin(math.pi * L / N))
crit = g["critical_p_rt"]

fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
# official fit protocol (cwf_ap_phaseG_rt.py): drop lattice-scale L=1,2; values from
# the committed JSON's own `fits` block -- nothing is re-fit here.
mask = (L >= 3) & (L <= N // 2)
for pv, S in sorted(g["S_of_L"].items(), key=lambda kv: float(kv[0])):
    pv_f = float(pv)
    S = np.array(S, float)
    iscrit = abs(pv_f - crit) < 1e-9
    f = g["fits"][pv]
    lbl = f"$p={pv_f:g}$" + (
        f"  ($R^2_{{\\rm conf}}={f['r2_conformal']:.3f}$, $c_{{\\rm eff}}={f['c_eff']:.2f}$)"
        if iscrit else "")
    ax.plot(X, S, "o-" if iscrit else "o", ms=3.5 if iscrit else 2.5,
            lw=1.6 if iscrit else 0, color="crimson" if iscrit else None,
            alpha=1.0 if iscrit else 0.45, label=lbl, zorder=5 if iscrit else 2)
    if iscrit:
        # overlay the official conformal fit (slope = c_eff/3, intercept from masked lstsq)
        slope = f["c_eff"] / 3.0
        b = (S[mask] - slope * X[mask]).mean()
        ax.plot(X[mask], slope * X[mask] + b, "-", color="crimson", lw=0.8, alpha=0.6, zorder=4)
    print(f"  p={pv_f:g}: official R2_conf = {f['r2_conformal']:.4f}, c_eff = {f['c_eff']:.3f}")
ax.set_xlabel(r"conformal chord $X=\log[(N/\pi)\sin(\pi L/N)]$")
ax.set_ylabel(r"interval entropy $S(L)$ (bits)")
ax.set_title(rf"$S(L)$ vs conformal chord ($N={N}$): only $p={crit}$ is straight")
ax.legend(fontsize=7, frameon=False)
fig.savefig(os.path.join(OUT, "fig_twoknob_rt.png"), dpi=300)
plt.close(fig)
print("wrote fig_twoknob_rt.png  (critical p:", crit, ")")
