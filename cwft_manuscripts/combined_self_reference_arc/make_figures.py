#!/usr/bin/env python3
"""Figures for the self-reference arc paper (paper C: P3 phase + P4 electromagnetism + P7).

Reads the committed self-reference experiment values (SR* keys in the CWF experiments
results.json) and emits publication PNGs (300 dpi) into ./figures/. No experiments are
re-run; the figures are plotted from the seeded reference values, so they are reproducible.

Figures:
  fig_trilemma.png    The self-reference trilemma (schematic): linear / faithful / real,
                      pick two -- the three faces.
  fig_renou.png       Face 2 anchor: Renou real-vs-complex separation realised from
                      self-referential sources (SR7) -- T=6sqrt2 > real bound 7.66, and the
                      collapse to 4sqrt2 < 7.66 when the self-referential sigma_Y is removed.
  fig_dispersion.png  The gauge consequence (SR12): the self-reference U(1) sector's photon
                      dispersion omega(k), linear at long wavelength, Lorentz-violating at the
                      lattice scale.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

EXP = os.environ.get(
    "CWF_EXP_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "cwft_experiments"),
)
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 300


def results():
    with open(os.path.join(EXP, "results.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Figure 1 -- the self-reference trilemma (schematic, no data).
# --------------------------------------------------------------------------------------
def fig_trilemma():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    # equilateral-ish triangle
    V = {"linear": (0.5, 0.92), "faithful": (0.08, 0.12), "real": (0.92, 0.12)}
    tri = Polygon([V["linear"], V["faithful"], V["real"]], closed=True,
                  fill=False, edgecolor="#333333", lw=1.6)
    ax.add_patch(tri)
    for name, (x, y) in V.items():
        dy = 0.045 if name == "linear" else -0.06
        ax.text(x, y + dy, name.upper(), ha="center", va="center",
                fontsize=13, fontweight="bold", color="#1a1a1a")
    # edge midpoints -> the face kept on each edge (give up the opposite vertex)
    def mid(a, b):
        return ((V[a][0] + V[b][0]) / 2, (V[a][1] + V[b][1]) / 2)
    # give up REAL (keep linear+faithful): Face 2 -- the wave branch (highlight)
    mx, my = mid("linear", "faithful")
    ax.text(mx - 0.12, my + 0.02, "Face 2:\nthe imaginary unit $i$\n(the wave branch)",
            ha="center", va="center", fontsize=9.5, color="#b5341f", fontweight="bold")
    # give up FAITHFUL (keep linear+real): coarse/approximate
    mx, my = mid("linear", "real")
    ax.text(mx + 0.12, my + 0.02, "give up faithfulness:\napprox./coarse\nintrospection\n(not pursued)",
            ha="center", va="center", fontsize=9.5, color="#555555")
    # give up LINEAR (keep faithful+real): Face 1 -- no-broadcasting (rigorous)
    mx, my = mid("faithful", "real")
    ax.text(mx, my - 0.10, "Face 1:\nno-broadcasting obstruction (proved);\nnonlinearity one response",
            ha="center", va="center", fontsize=9.5, color="#1f5fb5", fontweight="bold")
    ax.text(0.5, 1.02, "A self-description cannot be simultaneously:",
            ha="center", va="center", fontsize=11, color="#333333")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.10, 1.10)
    ax.axis("off")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_trilemma.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------------------
# Figure 2 -- Renou separation from self-referential sources (SR7).
# --------------------------------------------------------------------------------------
def fig_renou():
    d = results()["SR7_renou_realize"]
    t_self = d["T_selfref_sources"]
    t_real = d["real_bound"]
    t_noY = d["T_without_Y"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    labels = ["self-referential $i$\n($\\sigma_Y$ present)", "$\\sigma_Y$ removed\n(no $i$)"]
    vals = [t_self, t_noY]
    cols = ["#b5341f", "#9aa7b3"]
    bars = ax.bar(labels, vals, color=cols, edgecolor="k", linewidth=0.5, width=0.55)
    ax.axhline(t_real, ls="--", lw=1.4, color="#1f5fb5")
    ax.text(0.985, t_real / (t_self + 1.1) + 0.012,
            f"real-QM bound (Renou SDP) $= {t_real}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#1f5fb5", fontweight="bold")
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v + 0.12),
                    ha="center", fontsize=10, fontweight="bold")
    ax.annotate(r"$6\sqrt{2}$", (0, t_self - 0.5), ha="center", fontsize=9, color="white")
    ax.annotate(r"$4\sqrt{2}$", (1, t_noY - 0.5), ha="center", fontsize=9, color="#444444")
    ax.set_ylabel("bilocal separation statistic $T$")
    ax.set_title("Real-vs-complex separation from self-referential sources")
    ax.set_ylim(0, t_self + 1.1)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(OUT, "fig_renou.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------------------
# Figure 3 -- photon dispersion of the self-reference U(1) gauge sector (SR12).
# --------------------------------------------------------------------------------------
def fig_dispersion():
    d = results()["SR12_lattice_maxwell"]
    disp = d["dispersion"]  # list of [k, omega] (see cwf_sr12_lattice_maxwell.py)
    c = d["light_speed"]
    lv = d["lorentz_violation_at_kmax"]
    ks = [row[0] for row in disp]
    omega = [row[1] for row in disp]
    lin = [c * k for k in ks]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(ks, lin, ls="--", color="#1f5fb5", lw=1.4, label=r"linear $\omega=c\,|k|$ (Lorentz)")
    ax.plot(ks, omega, marker="o", ms=4, color="#b5341f", lw=1.5,
            label=r"self-reference U(1) sector $\omega(k)$")
    ax.set_xlabel("wavenumber $k$")
    ax.set_ylabel(r"frequency $\omega$")
    ax.set_title("Photons from the self-reference gauge sector")
    ax.annotate(f"Lorentz-violating\nat band edge ($\\sim{100*lv:.0f}\\%$)",
                (ks[-1], omega[-1]), textcoords="offset points", xytext=(-104, -30),
                ha="left", va="top", fontsize=8.5, color="#b5341f",
                arrowprops=dict(arrowstyle="->", color="#b5341f", lw=0.8))
    ax.text(0.04, 0.92, f"Gauss's law conserved\n(drift $< 10^{{-14}}$); $c={c:.2f}$",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_dispersion.png")
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_trilemma()
    fig_renou()
    fig_dispersion()
