#!/usr/bin/env python3
"""Figures for "Contextuality without Complexity: how far a vector-symbolic substrate
reaches into the quantum tiers".

Reads the committed result files in ../../cwft_experiments and emits 300 dpi PNGs into
./figures/.  Reproducible (all numbers from the seeded experiments).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = os.path.join(os.path.dirname(__file__), "..", "..", "cwft_experiments")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "cm",
})
C = {"coh": "#1f4e79", "dec": "#c0392b", "alt": "#7f8c8d",
     "acc": "#2e7d32", "warn": "#e67e22", "pur": "#6c3483"}


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


def fig_ring():
    """Stage 1: CF vs ring size (odd rings contextual, growing); the four-condition
    scorecard (frustrated-coherent vs the three CF=0 controls)."""
    j = load("cwf_selfref_ring_results.json")
    odd = [r for r in j["ring"] if r.get("cf") is not None]
    ns = [r["n"] for r in odd]
    cfs = [r["cf"] for r in odd]
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4))
    # (A) CF vs odd ring size
    ax[0].plot(ns, cfs, "o-", color=C["coh"], lw=2, ms=8)
    for n, cf in zip(ns, cfs):
        ax[0].annotate(f"{cf:.2f}", (n, cf), textcoords="offset points",
                       xytext=(0, 8), ha="center", fontsize=8, color=C["coh"])
    ax[0].axhline(0, color="#bbbbbb", lw=0.8)
    ax[0].set_xlabel("frustrated ring size $n$ (odd)")
    ax[0].set_ylabel("contextual fraction CF")
    ax[0].set_title("(A) odd rings are contextual (CF grows with $n$)")
    ax[0].set_xticks(ns)
    ax[0].set_ylim(-0.05, 0.85)
    # (B) scorecard
    labels = ["definite\nground\n(FPE)", "even\nring", "odd ring\ndecohered",
              "odd ring\ncoherent"]
    vals = [0.0, 0.0, 0.0, j["scorecard"]["frustrated_ring_odd"]]
    cols = [C["alt"], C["alt"], C["dec"], C["coh"]]
    ax[1].bar(range(4), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("contextual fraction CF")
    ax[1].set_title("(B) CF $>$ 0 iff no definite ground AND coherent")
    ax[1].set_ylim(0, 0.58)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ring.png")); plt.close(fig)
    print("wrote fig_ring")


def fig_product():
    """Stage 3: Renou T monotone in source entanglement; VSA-native sources separable."""
    j = load("cwf_selfref_product_results.json")
    rows = j["T_vs_concurrence"]
    Cc = [r["concurrence"] for r in rows]; T = [r["T"] for r in rows]
    real_bound = j["benchmarks"]["real_bound"]; cmax = j["benchmarks"]["complex_max"]
    cross = j["concurrence_at_real_bound"]; T_vsa = j["T_vsa_native"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(Cc, T, "o-", color=C["coh"], lw=2, ms=6,
            label="Renou $T$ vs source entanglement")
    ax.axhline(real_bound, color=C["acc"], ls="--", lw=1.3,
               label=f"real-QM bound ({real_bound})")
    ax.axhline(cmax, color=C["dec"], ls=":", lw=1.2,
               label=r"complex max $6\sqrt{2}=8.49$")
    ax.plot([0.0], [T_vsa], "s", color=C["pur"], ms=12,
            label=f"VSA-native ($C{{=}}0$): $T={T_vsa:.2f}$")
    ax.plot([cross], [real_bound], "v", color=C["acc"], ms=9)
    ax.annotate(f"$C\\approx{cross:.2f}$", (cross, real_bound),
                textcoords="offset points", xytext=(-4, 9), fontsize=8, color=C["acc"])
    ax.set_xlabel("source concurrence $C$ (entanglement)")
    ax.set_ylabel(r"Renou bilocal functional $T$")
    ax.set_title("The complex tier needs entanglement VSA lacks\n"
                 "(dimension-preserving binding yields only separable sources)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_product.png")); plt.close(fig)
    print("wrote fig_product")


if __name__ == "__main__":
    fig_ring()
    fig_product()
    print("all figures in", OUT)
