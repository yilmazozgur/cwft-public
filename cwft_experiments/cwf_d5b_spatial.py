"""
D5b -- A gravitating REGION created by a bifurcation: localization, bifurcation
       control, and fragility to environmental coupling.

D5a (homogeneous) showed the gravitating charge M_c jumps at a bifurcation where the
energy does not (clean 15x decoupling). D5b makes it SPATIAL: a "well" in the
bifurcation parameter puts a small CORE of the lattice in a contracting (periodic,
g<0 -> gain deficit) regime inside a chaotic SEA (g>0, no deficit). Three findings,
each computed with the framework's own gain field:

  (1) LOCALIZATION. M_c localizes into the core -- a gravitating region bounded by a
      g=0 surface (the computational horizon of Ch5 / Test 6, the black-hole-interior
      analog). ~100% of M_c sits inside the engineered core, and it is ROBUST: a
      deep-periodic core survives strong environmental coupling.

  (2) BIFURCATION CONTROL. Sweeping r_core, the gravitating region is switched on/off by
      the bifurcation knob: present below the chaos onset, gone in chaos, REAPPEARING
      discontinuously at the period-3 saddle-node (~3.8284). The region (M_c ~ several
      units) switches on while the lattice mean energy density moves only a fraction of
      a percent -- the spatial echo of D5a's energy decoupling (the clean quantitative
      decoupling is D5a's; here the small core keeps the total energy nearly flat).

  (3) COUPLING-FRAGILITY (new). A fine-tuned (narrow-window, period-3) gravitating core
      is destroyed by environmental coupling above eps ~ 0.01; a robust (deep-periodic)
      core persists to eps >~ 0.05. Whether a fine-tuned gravitating region exists at all
      is bounded by its stability against its environment.

Substrate: spatial-r logistic coupled-map lattice (book class),
r_i = r_sea + (r_core - r_sea) * plateau(i). Local Jacobian block (Test-6 convention):
g_i = < log|(1-eps) r_i (1-2 x_i)| >_t , M_c = sum_i max(-g_i,0).

FIREWALL: internal effective charge. The g=0 boundary is the computational horizon, not
a spacetime horizon. NOT external/physical gravity; externalising the jump collides with
stress-energy conservation. Defensible claim: "a localized gain-deficit region appears at
a bifurcation", not "spacetime curves".

CPU-only, seeded. Writes results.json["D5b_spatial"] and fig_D5b_spatial.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 7
N = 256
CORE_HW = 10          # core plateau half-width (core ~ 8% of lattice -> energy stays flat)
EDGE = 4.0
R_SEA = 3.93
EPS_ROBUST = 0.05     # strong coupling: robust deep-periodic core + horizon
EPS_WEAK = 0.005      # weak coupling: lets a fine-tuned (period-3) core survive
T_TRANS = 2000
T_MEAS = 2000


def plateau(n=N, c=None, hw=CORE_HW, s=EDGE):
    if c is None:
        c = n // 2
    i = np.arange(n)
    return 0.5 * (np.tanh((i - (c - hw)) / s) - np.tanh((i - (c + hw)) / s))


def cml_spatial(r_core, eps, n=N, r_sea=R_SEA, t_trans=T_TRANS, t_meas=T_MEAS, seed=SEED):
    W = plateau(n)
    r = r_sea + (r_core - r_sea) * W
    rng = np.random.default_rng(seed + int(round(r_core * 1e6)) % 100000 + int(eps * 1e5))
    x = rng.random(n)
    one_eps = 1.0 - eps

    def step(x):
        fx = r * x * (1.0 - x)
        nb = np.roll(fx, 1) + np.roll(fx, -1)
        return one_eps * fx + 0.5 * eps * nb

    for _ in range(t_trans):
        x = step(x)
    g_acc = np.zeros(n); e_acc = np.zeros(n)
    for _ in range(t_meas):
        g_acc += np.log(np.abs(one_eps * r * (1.0 - 2.0 * x)) + 1e-30)
        e_acc += x * x
        x = step(x)
    g = g_acc / t_meas; E = e_acc / t_meas
    deficit = np.maximum(-g, 0.0)
    core = W > 0.5
    return {"r": r, "g": g, "E": E, "deficit": deficit, "core_mask": core,
            "M_c": float(deficit.sum()), "core_size": int(np.sum(g < 0.0)),
            "M_c_in_core_frac": float(deficit[core].sum() / (deficit.sum() + 1e-30)),
            "E_total": float(E.mean())}


print("D5b: a gravitating region created by a bifurcation (spatial logistic CML)")
print(f"     N={N}, core hw={CORE_HW} (~{2*CORE_HW/N:.0%} of lattice), sea r={R_SEA}, "
      f"eps_robust={EPS_ROBUST}, eps_weak={EPS_WEAK}, seed={SEED}")

# (1) LOCALIZATION -- robust deep-periodic core at strong coupling
snap_chaos = cml_spatial(3.90, EPS_ROBUST)
snap_robust = cml_spatial(3.45, EPS_ROBUST)
snap_p3 = cml_spatial(3.845, EPS_WEAK)
print(f"\n  (1) LOCALIZATION (eps={EPS_ROBUST}):")
print(f"      chaotic core r=3.90: M_c={snap_chaos['M_c']:.2f}, contracting sites={snap_chaos['core_size']}")
print(f"      deep-periodic core r=3.45: M_c={snap_robust['M_c']:.2f}, "
      f"contracting sites={snap_robust['core_size']}, "
      f"{100*snap_robust['M_c_in_core_frac']:.0f}% of M_c inside the core plateau")
print(f"      period-3 core r=3.845 (eps={EPS_WEAK}): M_c={snap_p3['M_c']:.2f}, "
      f"{100*snap_p3['M_c_in_core_frac']:.0f}% in core")

# (2) BIFURCATION CONTROL -- sweep r_core at weak coupling
r_vals = np.linspace(3.40, 3.95, 320)
Mc = np.zeros_like(r_vals); E_tot = np.zeros_like(r_vals); core_sz = np.zeros_like(r_vals)
for j, rc in enumerate(r_vals):
    d = cml_spatial(rc, EPS_WEAK, t_trans=1500, t_meas=1500)
    Mc[j] = d["M_c"]; E_tot[j] = d["E_total"]; core_sz[j] = d["core_size"]

# period-3 saddle-node: single-step jump in M_c vs the energy change at the SAME step
win = (r_vals >= 3.80) & (r_vals <= 3.87); idx = np.where(win)[0]
js = idx[int(np.argmax(np.abs(np.diff(Mc[idx]))))]
r_sn = 0.5 * (r_vals[js] + r_vals[js + 1])
dMc_step = abs(Mc[js + 1] - Mc[js])
dE_step = abs(E_tot[js + 1] - E_tot[js])
E_here = E_tot[js]
dE_step_pct = 100.0 * dE_step / (E_here + 1e-30)
Mc_window = float(Mc[idx].max() - Mc[idx].min())
E_window_pct = 100.0 * (E_tot[idx].max() - E_tot[idx].min()) / (np.mean(E_tot[idx]) + 1e-30)
print(f"\n  (2) BIFURCATION CONTROL (sweep r_core, eps={EPS_WEAK}):")
print(f"      period-3 saddle-node at r_core~{r_sn:.4f}:")
print(f"        M_c single-step jump = {dMc_step:.2f} (region switches on; window swing {Mc_window:.1f})")
print(f"        mean energy density change at that step = {dE_step:.5f} = {dE_step_pct:.2f}% "
      f"(window swing {E_window_pct:.1f}%)")
print(f"        -> the gravitating region appears with a near-flat ({E_window_pct:.1f}%) energy density")

# (3) COUPLING-FRAGILITY
eps_vals = np.linspace(0.002, 0.060, 30)
Mc_fragile = np.array([cml_spatial(3.845, e, t_trans=1500, t_meas=1500)["M_c"] for e in eps_vals])
Mc_robust = np.array([cml_spatial(3.45, e, t_trans=1500, t_meas=1500)["M_c"] for e in eps_vals])
peak_f = Mc_fragile.max()
held = eps_vals[Mc_fragile > 0.1 * peak_f]
eps_crit = float(held.max()) if held.size else float(eps_vals[0])
robust_at_005 = float(Mc_robust[np.argmin(np.abs(eps_vals - 0.05))])
print(f"\n  (3) COUPLING-FRAGILITY:")
print(f"      fine-tuned (period-3) core collapses above eps~{eps_crit:.3f} (peak M_c {peak_f:.1f} -> ~0)")
print(f"      robust (deep-periodic) core persists: M_c={robust_at_005:.1f} at eps=0.05")

verdict = ("Spatial: M_c localizes into an engineered core (~100%) bounded by a g=0 "
           "horizon; the gravitating region is switched on/off by the bifurcation knob "
           "(reappearing discontinuously at the period-3 saddle-node) while the lattice "
           "energy density stays nearly flat; and a fine-tuned region is destroyed by "
           "environmental coupling above a threshold while a robust region persists. A "
           "gravitating region created by a bifurcation, not by adding energy. Internal "
           "effective charge; not a claim about physical spacetime.")
print("\n  VERDICT:", verdict)

# ---------------------------------------------------------------- record
rec = {
    "params": {"N": N, "core_halfwidth": CORE_HW, "core_frac": 2 * CORE_HW / N,
               "edge": EDGE, "r_sea": R_SEA, "eps_robust": EPS_ROBUST, "eps_weak": EPS_WEAK,
               "t_trans": T_TRANS, "t_meas": T_MEAS, "seed": SEED,
               "substrate": "spatial-r logistic coupled-map lattice"},
    "localization": {
        "chaotic_core_r3.90_eps0.05": {"M_c": snap_chaos["M_c"], "contracting_sites": snap_chaos["core_size"]},
        "deep_periodic_core_r3.45_eps0.05": {"M_c": snap_robust["M_c"],
            "contracting_sites": snap_robust["core_size"], "M_c_in_core_frac": snap_robust["M_c_in_core_frac"]}},
    "bifurcation_control": {
        "eps": EPS_WEAK, "r_at_period3_saddle_node": r_sn,
        "Mc_single_step_jump": dMc_step, "Mc_window_swing": Mc_window,
        "energy_density_step_change_abs": dE_step, "energy_density_step_change_pct": dE_step_pct,
        "energy_window_swing_pct": E_window_pct,
        "note": "fraction-of-range decoupling is D5a's (homogeneous, 15x); here the small "
                "core keeps total energy near-flat, so the honest metric is the % energy change"},
    "coupling_fragility": {
        "fragile_core_r3.845_eps_crit": eps_crit, "fragile_peak_Mc": float(peak_f),
        "robust_core_r3.45_Mc_at_eps0.05": robust_at_005,
        "reading": ("fine-tuned (narrow-window) gravitating cores are destroyed by "
                    "environmental coupling above a threshold; deep-periodic cores persist")},
    "verdict": verdict,
    "firewall": ("INTERNAL effective charge. The g=0 boundary is the computational horizon "
                 "(Ch5/Test 6), not a spacetime horizon. NOT external/physical gravity; "
                 "externalising the jump collides with stress-energy conservation."),
    "status": "R (computed) for localization/bifurcation-control/fragility; S for any physical reading.",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5b_spatial"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print("\n  wrote results.json['D5b_spatial']")

# ---------------------------------------------------------------- figure
xs = np.arange(N)
fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.6))

a = ax[0, 0]
a.plot(xs, snap_chaos["g"], color="#ff7f0e", lw=1.0, label="chaotic core ($r_c=3.90$): no deficit")
a.plot(xs, snap_robust["g"], color="#1f77b4", lw=1.4, label="deep-periodic core ($r_c=3.45$)")
a.axhline(0, color="k", ls=":", lw=0.9)
a.fill_between(xs, snap_robust["g"], 0, where=(snap_robust["g"] < 0),
               color="#1f77b4", alpha=0.25, label="$g<0$: gain-deficit core")
a.set_xlabel("site"); a.set_ylabel("gain field $g_i$")
a.set_title("(a) Contracting core appears, bounded by a $g=0$ horizon (robust, eps=0.05)")
a.legend(fontsize=8, loc="lower center"); a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(xs, snap_robust["deficit"], color="#d62728", lw=1.4, label="$\\max(-g_i,0)$ ($M_c$ density)")
a.plot(xs, snap_chaos["deficit"], color="#ff7f0e", lw=1.0, alpha=0.8, label="chaotic core (~0)")
a.axvspan(N // 2 - CORE_HW, N // 2 + CORE_HW, color="grey", alpha=0.18, label="engineered core")
a.set_xlabel("site"); a.set_ylabel("$M_c$ density")
a.set_title(f"(b) Charge localizes: {100*snap_robust['M_c_in_core_frac']:.0f}% of $M_c$ inside the core")
a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

a = ax[1, 0]
a.plot(r_vals, Mc, color="#d62728", lw=1.1, label="$M_c$ (gravitating region)")
a.set_xlabel("core parameter $r_{core}$ (eps=0.005)"); a.set_ylabel("$M_c$", color="#d62728")
a.tick_params(axis="y", labelcolor="#d62728")
a.axvline(r_sn, color="k", ls="--", lw=0.8, label=f"period-3 saddle-node ~{r_sn:.3f}")
a.axvline(3.5699, color="grey", ls=":", lw=0.8, label="chaos onset ~3.570")
a2 = a.twinx()
a2.plot(r_vals, E_tot, color="#2ca02c", lw=1.4, label="$E_{total}=\\langle x^2\\rangle$ (abs scale)")
a2.set_ylim(0.0, max(0.6, E_tot.max() * 1.2))
a2.set_ylabel("$E_{total}$ (near-flat, abs)", color="#2ca02c"); a2.tick_params(axis="y", labelcolor="#2ca02c")
l1, lb1 = a.get_legend_handles_labels(); l2, lb2 = a2.get_legend_handles_labels()
a.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="upper center")
a.set_title("(c) Region switched by bifurcation; energy density near-flat (abs scale)")
a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(eps_vals, Mc_fragile, "o-", color="#d62728", ms=3, lw=1.1, label="fine-tuned core ($r_c=3.845$)")
a.plot(eps_vals, Mc_robust, "s-", color="#1f77b4", ms=3, lw=1.1, label="robust core ($r_c=3.45$)")
a.axvline(eps_crit, color="#d62728", ls="--", lw=0.8, label=f"fragile collapse ~{eps_crit:.3f}")
a.set_xlabel("environmental coupling $\\varepsilon$"); a.set_ylabel("$M_c$ of the region")
a.set_title("(d) Coupling-fragility: fine-tuned region dies, robust region persists")
a.legend(fontsize=8, loc="upper right"); a.grid(alpha=0.3)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5b_spatial.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
