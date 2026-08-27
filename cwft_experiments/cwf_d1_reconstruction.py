"""
D1 -- reconstruction-complexity ladder on a 2D universal CA (Cluster D).

The holographic experiment: a 2D cellular automaton is the SCREEN; a bulk
observable at radial depth d (a cell, or a Radon-type line integral) is to be
RECONSTRUCTED from screen access. We give the reconstructor the full initial
light cone of the bulk observable (so the information is present on the screen --
this is a COMPUTATIONAL, not informational, question) and ask whether a predictor
cheaper than simulation can reconstruct it. The 2D analogue of C1, with the
distinctly holographic additions: the radial bulk-depth axis, the Radon /
integral-geometry observable, and the DARK FRACTION (the bulk fraction
unconstructable at fixed observer capacity), whose growth marks the
reconstruction horizon -- the surface where reconstruction crosses from
constructable to (effectively) uncomputable.

Split, mirroring C1 in 2D:
  - Game of Life (universal; Berlekamp-Conway-Guy, Rendell): reconstruction skill
    collapses with depth, the dark fraction grows toward 1 -- a reconstruction
    horizon at finite depth.
  - a 2D additive (von-Neumann XOR) CA (the reducible control, the 2D rule-90):
    the GF(2)-affine reconstructor is exact at all depths, dark fraction ~ 0.

Honesty (identical to C1): this is computational IRREDUCIBILITY of reconstruction
(finite, decidable by simulation), the finite shadow of unconstructability -- not
literal uncomputability; "no cheaper-than-simulation reconstruction" is over the
tested predictor families. The dark fraction is measured at a fixed observer
capacity and is budget-relative; the robust content is the SPLIT and the growth.

Reuses gf2_affine_fit / train_mlp from cwf_c1_predictive_cost. CPU NumPy.
Run:  python cwf_d1_reconstruction.py
"""
import json, os, time
import numpy as np
from cwf_c1_predictive_cost import gf2_affine_fit, train_mlp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# 2D cellular automata (periodic), vectorised over a batch of grids
# =========================================================================

def neighbor_sum(g):
    return (np.roll(g, 1, 1) + np.roll(g, -1, 1) + np.roll(g, 1, 2) + np.roll(g, -1, 2)
            + np.roll(np.roll(g, 1, 1), 1, 2) + np.roll(np.roll(g, 1, 1), -1, 2)
            + np.roll(np.roll(g, -1, 1), 1, 2) + np.roll(np.roll(g, -1, 1), -1, 2))


def life_step(g):
    s = neighbor_sum(g)
    return ((s == 3) | ((g == 1) & (s == 2))).astype(np.int8)


def add2d_step(g):
    """2D additive (von-Neumann XOR) CA -- the reducible '2D rule 90'."""
    return (g ^ np.roll(g, 1, 1) ^ np.roll(g, -1, 1)
            ^ np.roll(g, 1, 2) ^ np.roll(g, -1, 2)).astype(np.int8)


STEP = {"life": life_step, "add2d": add2d_step}


def gen(substrate, d, n_samples, seed, density=0.5):
    """Bulk observable (the centre cell at radial depth d) with the full initial
    light cone as the screen access (radius d, so the information is present --
    a COMPUTATIONAL, not informational, reconstruction question). n = 4d+11
    keeps the periodic boundary out of the cone."""
    n = 4 * d + 11
    c = n // 2
    rng = np.random.default_rng(seed)
    g = (rng.random((n_samples, n, n)) < density).astype(np.int8)
    X0 = g.copy()
    step = STEP[substrate]
    for _ in range(d):
        g = step(g)
    y = g[:, c, c].astype(np.int8)                              # bulk cell at depth d
    Xwin = X0[:, c - d:c + d + 1, c - d:c + d + 1].reshape(n_samples, -1).astype(np.float64)
    return Xwin, y


# =========================================================================
# Reconstruction skill = best sub-simulation reconstructor, above baseline
# =========================================================================

def recon_skill(substrate, d, n_samples=6000, H=64, seed=0):
    Xw, y = gen(substrate, d, n_samples, seed=seed + d)
    ntr = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = Xw[:ntr], Xw[ntr:], y[:ntr], y[ntr:]
    base = max(yte.mean(), 1 - yte.mean())
    accs = [gf2_affine_fit(Xtr, ytr, Xte, yte),
            train_mlp(Xtr, ytr, Xte, yte, H=0, seed=1, epochs=250),
            train_mlp(Xtr, ytr, Xte, yte, H=H, seed=1, epochs=250)]
    best = max(accs)
    return float(np.clip((best - base) / (1 - base + 1e-12), 0, 1)), float(base)


def main():
    depths = [1, 2, 3, 4, 6, 8, 10]
    eps = 0.75
    t0 = time.time()
    res = {}
    for sub in ("add2d", "life"):
        sk = []
        for d in depths:
            s, base = recon_skill(sub, d)
            sk.append(s)
            print(f"  {sub:6s} d={d:2d}  recon-skill={s:.2f} (base {base:.2f})")
        res[f"{sub}_point"] = sk
    # dark fraction at fixed capacity: 1 - reconstruction skill (point observable)
    dark = {sub: [1 - s for s in res[f"{sub}_point"]] for sub in ("add2d", "life")}
    # reconstruction horizon = first depth where the dark fraction crosses 0.5
    horizon = {}
    for sub in ("add2d", "life"):
        h = next((depths[i] for i in range(len(depths)) if dark[sub][i] > 0.5), None)
        horizon[sub] = h
    print(f"\n  reconstruction horizon (dark>0.5): life d={horizon['life']}, "
          f"add2d d={horizon['add2d']}")
    print(f"  runtime {time.time()-t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["D1_reconstruction"] = dict(
        depths=depths, skill=res, dark_fraction=dark, horizon=horizon, eps=eps,
        substrates=dict(life="Game of Life (2D universal)",
                        add2d="2D additive von-Neumann XOR (reducible control)"),
        note=("2D bulk-from-screen reconstruction with full-cone access and a bounded "
              "(GF2-affine or MLP) reconstructor. Universal GoL: skill collapses, dark "
              "fraction grows -> reconstruction horizon. Additive 2D CA: GF(2)-affine "
              "reconstructs exactly, no dark fraction. Computational irreducibility of "
              "reconstruction (finite shadow of unconstructability), not literal "
              "uncomputability; dark fraction is at fixed capacity, budget-relative."))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(depths, res, dark, horizon, eps)
    print("Wrote results.json key: D1_reconstruction")


def plot_results(depths, res, dark, horizon, eps):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    col = {"life": "#e7298a", "add2d": "#1b9e77"}
    lab = {"life": "Game of Life (universal)", "add2d": "2D additive (reducible)"}

    # (a) reconstruction skill vs bulk depth
    a = ax[0]
    for sub in ("add2d", "life"):
        a.plot(depths, res[f"{sub}_point"], "o-", color=col[sub], lw=1.9, ms=6,
                label=lab[sub])
    a.axhline(eps, ls=":", c="0.5", lw=1)
    a.text(depths[-1], eps + 0.02, "skill $\\epsilon$", ha="right", fontsize=8, color="0.4")
    a.set_xlabel("bulk radial depth $d$"); a.set_ylabel("reconstruction skill (bounded predictor)")
    a.set_title("(a) Bulk reconstruction vs depth")
    a.set_ylim(-0.03, 1.05); a.legend(fontsize=8, loc="center right")
    a.text(0.5, 0.30, "2D cone $\\sim(2d{+}1)^2$ cells:\narea-law complexity\nexplosion",
           transform=a.transAxes, fontsize=7.6, style="italic", color="0.4", ha="center")

    # (b) dark fraction vs depth -> the reconstruction horizon
    b = ax[1]
    for sub in ("add2d", "life"):
        b.plot(depths, dark[sub], "o-", color=col[sub], lw=1.9, ms=6, label=lab[sub])
    b.axhline(0.5, ls="--", c="0.5", lw=1)
    if horizon["life"] is not None:
        b.axvline(horizon["life"], ls=":", c=col["life"], lw=1.4)
        b.text(horizon["life"], 0.06, f" horizon\n $d^*={horizon['life']}$", color=col["life"],
               fontsize=8.5)
    b.set_xlabel("bulk radial depth $d$"); b.set_ylabel("dark fraction (unconstructable)")
    b.set_title("(b) The dark fraction grows $\\Rightarrow$ horizon")
    b.set_ylim(-0.03, 1.05); b.legend(fontsize=8, loc="center right")

    # (c) schematic: bulk depth as radial direction, horizon as a shell
    c = ax[2]; c.axis("off"); c.set_xlim(-1.2, 1.2); c.set_ylim(-1.2, 1.2)
    th = np.linspace(0, 2 * np.pi, 200)
    dmax = depths[-1]
    c.plot(np.cos(th), np.sin(th), color="0.4", lw=1.2)            # screen (boundary)
    c.text(0, 1.08, "2D CA screen", ha="center", fontsize=9)
    if horizon["life"]:
        rh = 1 - horizon["life"] / dmax
        c.add_patch(plt.Circle((0, 0), rh, fc="#f3d9e6", ec=col["life"], lw=1.4, ls=":"))
        c.add_patch(plt.Circle((0, 0), max(rh - 0.0, 0), fc="#c9c9c9", ec="none"))
        c.text(0, 0, "unconstructable\ncore\n(dark)", ha="center", va="center", fontsize=8,
               color="0.25")
        c.annotate("reconstruction\nhorizon $d^*$", xy=(rh * 0.7, rh * 0.7), xytext=(0.75, 0.75),
                   fontsize=8.5, color=col["life"],
                   arrowprops=dict(arrowstyle="->", color=col["life"]))
    c.annotate("", xy=(0.96, -0.2), xytext=(0.0, 0.0),
               arrowprops=dict(arrowstyle="->", color="0.4"))
    c.text(0.5, -0.32, "depth $d$ (radial)", fontsize=8.5, color="0.4", rotation=-12)
    c.text(0, -1.13, "reconstructable wedge (light) $\\to$ dark core",
           ha="center", fontsize=8.3, color="0.4")
    c.set_title("(c) Bulk depth as a radial horizon")

    fig.suptitle("D1: reconstruction-complexity ladder on a 2D universal CA --- "
                 "the dark fraction and the reconstruction horizon",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_D1_reconstruction.png")
    fig.savefig(p, dpi=118, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
