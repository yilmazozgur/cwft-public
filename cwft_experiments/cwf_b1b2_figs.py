"""
Clean, publication-quality figures for the Program-I (B1) and bridge (B2)
results, regenerated directly from results.json.

These supersede the working-notes figures (fig_B1_*.png, fig_B2_bridge.png),
which carry embedded monospace text panels unsuitable for the chapter.

Outputs:
    fig_B1_program1.png   -- trade-off frontier, conservation quality, shared-axis alignment
    fig_B2_bridge_clean.png -- closure floor vs info-production, conserved-axis vs info-production, host gauge span

Run:  python cwf_b1b2_figs.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
})

R = json.load(open("results.json"))

# Canonical substrate order + display style -------------------------------
ORDER = ["CA-184", "CA-90", "CA-30", "CA-110", "CML-3.6", "CML-3.8", "CML-4.0"]
WCLASS = {"CA-184": "II", "CA-90": "III", "CA-30": "III", "CA-110": "IV",
          "CML-3.6": "periodic", "CML-3.8": "chaotic", "CML-4.0": "chaotic"}
COLOR = {
    "CA-184": "#1b9e77", "CA-90": "#d95f02", "CA-30": "#7570b3",
    "CA-110": "#e7298a", "CML-3.6": "#66a61e", "CML-3.8": "#e6ab02",
    "CML-4.0": "#a6761d",
}
MARK = {s: ("^" if s.startswith("CA") else "o") for s in ORDER}
LABEL = {s: f"{s} ({WCLASS[s]})" for s in ORDER}


# =========================================================================
# Figure 1 -- Program I (B1)
# =========================================================================
def fig_b1():
    tr = R["B1_tradeoff"]["by_system"]
    sh = R["B1_tradeoff"]["shuffle_control"]
    kc = R["B1_kappa_corrected"]
    uni = kc["universal"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # --- (a) trade-off frontier: N vs nonlocality (1 - L) --------------
    a = ax[0]
    handles = []
    for s in ORDER:
        pts = tr[s]["ladder"]
        x = [1.0 - p["L_deg"] for p in pts]
        y = [p["N"] for p in pts]
        h = a.scatter(x, y, c=COLOR[s], marker=MARK[s], s=42, alpha=0.85,
                      edgecolor="white", linewidth=0.4, label=LABEL[s])
        handles.append(h)
    # shuffle control: time-pairing destroyed -> N -> 1 (and O -> 0); it has
    # no meaningful nonlocality, so show it as a band, not a located point.
    sh_mean = np.mean([sh[s]["N_mean"] for s in ORDER])
    hsh = a.axhline(sh_mean, ls="--", c="0.5", lw=1.4,
                    label="shuffle control ($\\tilde N\\!\\to\\!1,\\ \\tilde O\\!\\to\\!0$)")
    handles.append(hsh)
    # empty corner: low N + high locality (small 1-L) is unoccupied
    a.add_patch(plt.Rectangle((0, 0), 0.22, 0.28, facecolor="0.86",
                              edgecolor="none", zorder=0))
    a.text(0.02, 0.04, "empty corner\n(low $N$, high locality)", fontsize=8.3,
           style="italic", color="0.35")
    a.set_xlabel("degree load  $1-\\tilde L$  ($L_{\\rm deg}$)")
    a.set_ylabel("nonlinearity residual  $\\tilde N$")
    a.set_title("(a) Trade-off frontier")
    a.set_xlim(-0.02, 1.02)
    a.set_ylim(-0.02, 1.05)

    # --- (b) conservation quality q (binding capacity c = n/4) ---------
    b = ax[1]
    q = [kc["by_system"][s]["c4"]["quality"] for s in ORDER]
    cols = [COLOR[s] for s in ORDER]
    xs = np.arange(len(ORDER))
    b.bar(xs, q, color=cols, edgecolor="white")
    b.bar(len(ORDER) + 0.4, uni["quality"], color="0.35", edgecolor="white")
    b.axhline(1 / 3, ls="--", c="firebrick", lw=1.3)
    b.text(len(ORDER) + 0.6, 1 / 3 - 0.018, "$q=1/3$: no conservation",
           ha="right", va="top", color="firebrick", fontsize=8.5)
    b.axhline(uni["mean_per_system_quality"], ls=":", c="navy", lw=1.2)
    b.text(-0.3, uni["mean_per_system_quality"] + 0.006,
           f"$\\bar q={uni['mean_per_system_quality']:.3f}$", color="navy", fontsize=8.5)
    b.set_xticks(list(xs) + [len(ORDER) + 0.4])
    b.set_xticklabels([s.replace("CML-", "CML\n") for s in ORDER] + ["shared\naxis"],
                      fontsize=7.6, rotation=0)
    b.set_ylabel("conservation quality  $q=\\lambda_{\\min}/\\sum\\lambda$")
    b.set_title("(b) A conserved combination? (7-substrate pilot;\nretired at scale, T1.1)")
    b.set_ylim(0, 0.36)

    # --- (c) alignment with the single shared direction ----------------
    c = ax[2]
    al = [uni["alignment"][s] for s in ORDER]
    c.barh(xs, al, color=cols, edgecolor="white")
    c.axvline(0.77, ls="--", c="0.3", lw=1.2)
    c.text(0.77, len(ORDER) - 0.3, " 6/7 align $\\geq 0.77$", fontsize=8.5,
           color="0.25", rotation=90, va="top")
    c.set_yticks(xs)
    c.set_yticklabels([LABEL[s] for s in ORDER], fontsize=8)
    c.invert_yaxis()
    c.set_xlim(0, 1.0)
    c.set_xlabel("$|\\cos\\angle|$ with shared axis $(0.15,\\,0.69,\\,-0.71)$")
    c.set_title("(c) One axis, system-dependent tilt")

    fig.suptitle("Program I (B1): the dimensional trade-off and the pilot's conserved direction",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig("fig_B1_program1.png", bbox_inches="tight")
    print("wrote fig_B1_program1.png")


# =========================================================================
# Figure 2 -- the bridge (B2)
# =========================================================================
def fig_b2():
    bs = R["B2_bridge"]["by_system"]
    br = R["B2_bridge"]["bridge"]
    gauge = R["B2_bridge"]["gauge"]["host_hbar_c"]

    ac = np.array([bs[s]["a_c"] for s in ORDER])
    Nrich = np.array([bs[s]["N_rich"] for s in ORDER])
    kproj = np.array([bs[s]["kappa_proj"] for s in ORDER])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    def scatter_labeled(a, x, y):
        for i, s in enumerate(ORDER):
            a.scatter(x[i], y[i], c=COLOR[s], marker=MARK[s], s=80,
                      edgecolor="white", linewidth=0.6, zorder=3)
            dy = 0.02 * (a.get_ylim()[1] - a.get_ylim()[0])
            a.annotate(s, (x[i], y[i]), fontsize=7.4, ha="center",
                       va="bottom", xytext=(0, 5), textcoords="offset points")

    # --- (a) closure floor N* vs info production : the robust association
    a = ax[0]
    scatter_labeled(a, ac, Nrich)
    a.set_xlabel("information production  $a_c$  (bits/step)")
    a.set_ylabel("closure floor  $N^\\star$")
    a.set_title("(a) Closure cost tracks information production")
    a.text(0.96, 0.06, f"Spearman $={br['spearman_Nrich_ac']:+.2f}$",
           transform=a.transAxes, ha="right", fontsize=10,
           bbox=dict(boxstyle="round", fc="#eef6ee", ec="0.6"))
    a.annotate("rule 90:\nmax $a_c$, mid $N^\\star$\n(XOR linearly closable)",
               xy=(bs["CA-90"]["a_c"], bs["CA-90"]["N_rich"]),
               xytext=(0.45, 0.30), textcoords="axes fraction", fontsize=7.8,
               color="#d95f02",
               arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.1))

    # --- (b) conserved-axis position vs info production : the sign flip --
    b = ax[1]
    scatter_labeled(b, ac, kproj)
    b.axhline(0, c="0.7", lw=0.8)
    b.set_xlabel("information production  $a_c$  (bits/step)")
    b.set_ylabel("conserved-axis position  $\\kappa_{\\mathrm{proj}}$")
    b.set_title("(b) The other scalar proxy disagrees in sign")
    b.text(0.96, 0.92, f"Spearman $={br['spearman_kappaproj_ac']:+.2f}$",
           transform=b.transAxes, ha="right", va="top", fontsize=10,
           bbox=dict(boxstyle="round", fc="#fdeeee", ec="0.6"))

    # --- (c) the gauge: hbar_c set by host, spans ~1e8 -------------------
    c = ax[2]
    hosts = list(gauge.keys())
    vals = [gauge[h] for h in hosts]
    yy = np.arange(len(hosts))
    c.barh(yy, vals, color="#4575b4", edgecolor="white")
    c.set_xscale("log")
    c.set_yticks(yy)
    c.set_yticklabels(hosts, fontsize=8)
    c.invert_yaxis()
    c.set_xlabel("$\\hbar_c$  (host-set, computation fixed)")
    c.set_title("(c) $\\hbar_c$ is a host gauge, not dynamics")
    span = R["B2_bridge"]["gauge"]["hbar_c_span_fixed_computation"]
    c.text(0.97, 0.30, f"span $\\approx{span:.2g}\\times$\nat fixed computation",
           transform=c.transAxes, ha="right", va="center", fontsize=8.5, color="0.3",
           bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))

    fig.suptitle("The bridge (B2): gauge separation, and the one host-independent association",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig("fig_B2_bridge_clean.png", bbox_inches="tight")
    print("wrote fig_B2_bridge_clean.png")


if __name__ == "__main__":
    fig_b1()
    fig_b2()
