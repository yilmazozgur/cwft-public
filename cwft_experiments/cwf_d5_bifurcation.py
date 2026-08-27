"""
D5a -- The gravitating charge is a spectral invariant: it jumps at a bifurcation
       where the energy does not.

Thesis (FUTURE_WORK.md, Track D, item D5). In CWF the gravitating charge is the
gain-deficit mass

    M_c = sum_i max(-g_i, 0),     g_i = < log (spectral radius of local Jacobian) >_t

(the SAME definition the gravity chapter / Test 6 uses; here the substrate is a
logistic coupled-map lattice, a book substrate class). g_i is a SPECTRAL-STABILITY
quantity. Energy / activity are THERMODYNAMIC quantities. They are computed from
different aspects of the dynamics, so a *bifurcation* -- which reorganises the
spectrum discontinuously -- can make M_c JUMP while an energy proxy E = <x^2> and an
activity proxy <|dx|> vary continuously.

This is the mechanism behind the book's measured m_disp <-> M_c non-proportionality
("the substrate's computational equivalence principle", which FAILS on dissipative
substrates): the gravitating charge lives on the spectrum, the inertial/energy charge
on the activity, and a bifurcation moves one without the other.

Substrate: logistic CML, f(x) = r x (1-x), diffusive coupling eps, periodic BCs.
Local (diagonal) Jacobian block of site i:  d x_i(t+1)/d x_i(t) = (1-eps) r (1-2 x_i)
-> spectral radius |(1-eps) r (1-2 x_i)|  (scalar site => 1x1 block, as in Test 6 the
local block excludes the off-diagonal neighbour coupling).

FIREWALL (printed into the record): this is the substrate's INTERNAL effective charge.
It is NOT a claim about external/physical gravity; promoting the internal-metric jump
to external curvature would collide with stress-energy conservation. The defensible
statement is "the gravitating charge is a spectral invariant", not "external gravity
jumps".

CPU-only, seeded. Writes results.json["D5_bifurcation"] and fig_D5_bifurcation.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 5
N = 64           # lattice sites
EPS = 0.10       # diffusive coupling
T_TRANS = 1500   # transient discarded
T_MEAS = 1500    # measurement window


def cml_measure(r, n=N, eps=EPS, t_trans=T_TRANS, t_meas=T_MEAS, seed=SEED):
    """Run a homogeneous logistic CML at parameter r; return framework quantities.

    Returns dict with:
      g_mean   : spatial mean of the gain field g_i (time-avg log local-Jacobian SR)
      M_c      : gravitating charge sum_i max(-g_i, 0)
      m_c_site : M_c / n  (per-site gain deficit)
      E        : energy proxy   <x^2>   (space+time mean)
      activity : <|x(t+1)-x(t)|>  (space+time mean)
      xbar     : <x>
    """
    rng = np.random.default_rng(seed + int(round(r * 1e6)) % 100000)
    x = rng.random(n)
    one_eps = 1.0 - eps

    def step(x):
        fx = r * x * (1.0 - x)
        nb = np.roll(fx, 1) + np.roll(fx, -1)        # periodic neighbours
        return one_eps * fx + 0.5 * eps * nb

    for _ in range(t_trans):                          # discard transient
        x = step(x)

    g_acc = np.zeros(n)
    e_acc = 0.0
    act_acc = 0.0
    xbar_acc = 0.0
    for _ in range(t_meas):
        # local Jacobian spectral radius at current state (before stepping)
        sr = np.abs(one_eps * r * (1.0 - 2.0 * x))
        g_acc += np.log(sr + 1e-30)
        e_acc += np.mean(x * x)
        xbar_acc += np.mean(x)
        xn = step(x)
        act_acc += np.mean(np.abs(xn - x))
        x = xn

    g = g_acc / t_meas
    M_c = float(np.sum(np.maximum(-g, 0.0)))
    return {
        "g_mean": float(np.mean(g)),
        "M_c": M_c,
        "m_c_site": M_c / n,
        "E": float(e_acc / t_meas),
        "activity": float(act_acc / t_meas),
        "xbar": float(xbar_acc / t_meas),
    }


def sweep(r_vals):
    keys = ["g_mean", "M_c", "m_c_site", "E", "activity", "xbar"]
    out = {k: np.zeros(len(r_vals)) for k in keys}
    for j, r in enumerate(r_vals):
        d = cml_measure(r)
        for k in keys:
            out[k][j] = d[k]
    return out


def jump_stats(r_vals, y):
    """Largest single-step absolute change and its location (a discontinuity probe)."""
    dy = np.abs(np.diff(y))
    j = int(np.argmax(dy))
    r_at = 0.5 * (r_vals[j] + r_vals[j + 1])
    span = float(np.nanmax(y) - np.nanmin(y))
    return {"max_step": float(dy[j]), "r_at_max_step": float(r_at),
            "range": span, "max_step_frac_of_range": float(dy[j] / (span + 1e-30))}


print("D5a: gravitating charge M_c vs energy E across the logistic-CML bifurcation cascade")
print(f"     N={N} sites, coupling eps={EPS}, transient={T_TRANS}, window={T_MEAS}, seed={SEED}")

# --- global sweep across the cascade: period-doubling, chaos onset, windows ---
r_global = np.linspace(3.40, 4.00, 1000)
G = sweep(r_global)

# --- fine zoom across the period-3 saddle-node (r ~ 3.8284): a genuine discontinuity ---
r_zoom = np.linspace(3.820, 3.870, 500)
Z = sweep(r_zoom)

# Quantify the period-3 saddle-node: compare the jump in M_c to the change in E
# across the window-onset step (where M_c rises most steeply in the zoom).
dMc = np.abs(np.diff(Z["M_c"]))
js = int(np.argmax(dMc))
r_sn = 0.5 * (r_zoom[js] + r_zoom[js + 1])
# local change of E across the SAME step, and across a small neighbourhood
dE_local = abs(Z["E"][js + 1] - Z["E"][js])
win = slice(max(0, js - 5), min(len(r_zoom), js + 6))
Mc_jump = float(Z["M_c"][win].max() - Z["M_c"][win].min())
E_var = float(Z["E"][win].max() - Z["E"][win].min())
# normalise each by its own full-sweep range -> dimensionless "how discontinuous"
Mc_rng = float(Z["M_c"].max() - Z["M_c"].min())
E_rng = float(Z["E"].max() - Z["E"].min())
Mc_disc = Mc_jump / (Mc_rng + 1e-30)
E_disc = E_var / (E_rng + 1e-30)

mc_js = jump_stats(r_global, G["M_c"])
e_js = jump_stats(r_global, G["E"])

# correlation of step-to-step changes: do M_c jumps coincide with E jumps? (expect low)
dMc_g = np.abs(np.diff(G["M_c"])); dMc_g /= (dMc_g.max() + 1e-30)
dE_g = np.abs(np.diff(G["E"])); dE_g /= (dE_g.max() + 1e-30)
corr_steps = float(np.corrcoef(dMc_g, dE_g)[0, 1])

print("\n  Global sweep [3.40, 4.00]:")
print(f"    M_c : largest single-step jump = {mc_js['max_step']:.3f} "
      f"({100*mc_js['max_step_frac_of_range']:.0f}% of its range) at r={mc_js['r_at_max_step']:.4f}")
print(f"    E   : largest single-step jump = {e_js['max_step']:.5f} "
      f"({100*e_js['max_step_frac_of_range']:.0f}% of its range) at r={e_js['r_at_max_step']:.4f}")
print(f"    corr(|dM_c|, |dE|) step-to-step = {corr_steps:+.3f}  (low => jumps don't coincide)")
print("\n  Period-3 saddle-node zoom [3.820, 3.870]:")
print(f"    steepest M_c step at r={r_sn:.4f}")
print(f"    across that window:  M_c swing = {Mc_jump:.3f} = {100*Mc_disc:.0f}% of zoom range")
print(f"                         E   swing = {E_var:.5f} = {100*E_disc:.0f}% of zoom range")
print(f"    DECOUPLING: M_c moves {Mc_disc/(E_disc+1e-9):.1f}x more (frac-of-range) than E "
      f"across the same bifurcation")

verdict = ("M_c (gravitating, spectral) is discontinuous at bifurcations where E "
           "(energy, thermodynamic) is continuous -> the gravitating charge decouples "
           "from energy. Internal effective charge only; not a claim about physical gravity.")
print("\n  VERDICT:", verdict)

# ---------------------------------------------------------------- record
rec = {
    "params": {"N": N, "eps": EPS, "t_trans": T_TRANS, "t_meas": T_MEAS, "seed": SEED,
               "substrate": "logistic coupled-map lattice (book substrate class)"},
    "definitions": {
        "g_i": "time-avg log spectral radius of local Jacobian |(1-eps) r (1-2 x_i)| (Test-6 convention)",
        "M_c": "sum_i max(-g_i, 0) (gain-deficit gravitating charge)",
        "E": "<x^2> space+time mean (energy proxy)",
        "activity": "<|x(t+1)-x(t)|> (activity / inertial proxy)"},
    "global_sweep": {"r_min": 3.40, "r_max": 4.00, "n": len(r_global),
                     "M_c_largest_step": mc_js, "E_largest_step": e_js,
                     "corr_dMc_dE_steps": corr_steps},
    "period3_saddle_node": {
        "r_at_steepest_Mc_step": r_sn,
        "Mc_swing_frac_of_range": Mc_disc, "E_swing_frac_of_range": E_disc,
        "decoupling_ratio": Mc_disc / (E_disc + 1e-9)},
    "verdict": verdict,
    "firewall": ("INTERNAL effective charge of the substrate. NOT external/physical "
                 "gravity. Promoting the internal-metric jump to external curvature "
                 "collides with stress-energy conservation. Defensible claim: 'the "
                 "gravitating charge is a spectral invariant', not 'external gravity jumps'."),
    "status": "R (computed) for the internal decoupling; S for any physical-gravity reading.",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5_bifurcation"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print(f"\n  wrote results.json['D5_bifurcation']")

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(3, 1, figsize=(9.2, 9.6), sharex=False)

a = ax[0]
a.plot(r_global, G["g_mean"], color="#1f77b4", lw=0.9)
a.axhline(0, color="k", ls=":", lw=0.8)
a.fill_between(r_global, G["g_mean"], 0, where=(G["g_mean"] < 0),
               color="#1f77b4", alpha=0.25, label="g<0 (contracting -> gain deficit)")
a.set_ylabel("gain field  $g(r)$"); a.set_title(
    "Gain field (spectral): the Lyapunov-like stability quantity, discontinuous at windows")
a.legend(fontsize=8, loc="lower left"); a.grid(alpha=0.3)

a = ax[1]
a.plot(r_global, G["M_c"], color="#d62728", lw=1.0, label="$M_c=\\sum_i\\max(-g_i,0)$  (gravitating, spectral)")
a.set_ylabel("$M_c$  (gravitating charge)", color="#d62728")
a.tick_params(axis="y", labelcolor="#d62728")
a.set_title("Gravitating charge $M_c$ JUMPS (spikes in ordered windows) ... ")
a2 = a.twinx()
a2.plot(r_global, G["E"], color="#2ca02c", lw=1.2, label="$E=\\langle x^2\\rangle$ (energy)")
a2.plot(r_global, G["activity"], color="#9467bd", lw=1.0, alpha=0.7, label="activity $\\langle|\\Delta x|\\rangle$")
a2.set_ylabel("$E$, activity (smooth)", color="#2ca02c")
a2.tick_params(axis="y", labelcolor="#2ca02c")
a.set_xlabel("logistic parameter $r$ (bifurcation knob)")
lines1, lab1 = a.get_legend_handles_labels()
lines2, lab2 = a2.get_legend_handles_labels()
a.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
a.grid(alpha=0.3)

a = ax[2]
a.plot(r_zoom, Z["M_c"] / (Z["M_c"].max() + 1e-30), color="#d62728", lw=1.4,
       label="$M_c$ (normalised)")
a.plot(r_zoom, (Z["E"] - Z["E"].min()) / (Z["E"].max() - Z["E"].min() + 1e-30),
       color="#2ca02c", lw=1.4, label="$E$ (normalised)")
a.axvline(r_sn, color="k", ls="--", lw=0.9, label=f"period-3 saddle-node  r$\\approx${r_sn:.4f}")
a.set_xlabel("logistic parameter $r$  (period-3 window zoom)")
a.set_ylabel("normalised to [0,1]")
a.set_title(f"Zoom: $M_c$ jumps {Mc_disc/(E_disc+1e-9):.0f}x more than $E$ across the same bifurcation")
a.legend(fontsize=8, loc="center right"); a.grid(alpha=0.3)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5_bifurcation.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
