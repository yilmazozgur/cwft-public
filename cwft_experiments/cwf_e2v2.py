"""
Decisive trapping test (E2 v2).

The first E2 confounded trapping with a dissipative (sub-critical) periphery
that absorbs everything. Here the periphery is SUPER-critical (rho=1.7) so a
perturbation genuinely propagates. We inject at the centre and ask whether the
influence front reaches a 'detector' placed in the far periphery.

Configurations:
  uniform        : rho=1.7 everywhere (no band) -- front SHOULD reach detector.
  high_rho_band  : central band rho=2.6 (a source/amplifier).
  damped_band    : central band rho=1.7 + leak=2.0 (an absorber/wall).

Trapping = front fails to reach the detector despite a propagating medium.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, front_positions, RNG

THETA = 1e-4

def run_case(n=261, m=24, coupling=0.18, rho_bg=1.7, band=30,
             rho_band=1.7, leak_band=0.0, T=400, seed=11, seeds=8):
    center = n // 2
    detector = center + band + 30          # well outside the band, inside lattice
    rho = np.full(n, rho_bg)
    rho[center - band: center + band + 1] = rho_band
    leak = np.zeros(n)
    if leak_band:
        leak[center - band: center + band + 1] = leak_band

    reach_times, max_radii = [], []
    Dlast = None
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=seed + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, center)
        Dlast = D
        r = front_positions(D, center, theta=THETA)
        rmax = np.nanmax(r) if np.any(~np.isnan(r)) else 0.0
        max_radii.append(rmax)
        # did divergence at the detector ever exceed threshold?
        hit = np.where(D[:, detector] > THETA)[0]
        reach_times.append(int(hit[0]) if hit.size else -1)  # -1 = never
    reached = np.array(reach_times)
    frac_reached = float(np.mean(reached >= 0))
    med_reach = float(np.median(reached[reached >= 0])) if np.any(reached >= 0) else None
    return {"detector_dist": detector - center,
            "frac_runs_front_reached_detector": frac_reached,
            "median_reach_time": med_reach,
            "median_max_radius": float(np.median(max_radii)),
            "_D": Dlast, "_center": center, "_band": band, "_detector": detector}

res = json.load(open("results.json"))
print("E2 v2 -- propagating periphery (rho_bg=1.7); trapping = front fails to "
      "reach detector:")
configs = {
    "uniform":       dict(rho_band=1.7, leak_band=0.0),
    "high_rho_band": dict(rho_band=2.6, leak_band=0.0),
    "damped_band":   dict(rho_band=1.7, leak_band=2.0),
}
out, fields = {}, {}
for name, kw in configs.items():
    r = run_case(**kw)
    fields[name] = r
    summary = {k: v for k, v in r.items() if not k.startswith("_")}
    out[name] = summary
    tag = ("front reached detector" if summary["frac_runs_front_reached_detector"] > 0.5
           else "TRAPPED (front blocked)")
    print(f"    {name:14s} reached={summary['frac_runs_front_reached_detector']:.2f} "
          f"med_reach_t={summary['median_reach_time']} "
          f"med_rmax={summary['median_max_radius']:.0f} "
          f"(detector at {summary['detector_dist']}) -> {tag}")
res["E2_v2"] = out

# figure: divergence fields for the three configs with detector marked
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, name in zip(axes, configs):
    r = fields[name]
    Z = np.log10(r["_D"] + 1e-16)
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="inferno", vmin=-12, vmax=0)
    c, band, det = r["_center"], r["_band"], r["_detector"]
    ax.axvline(c - band, color="cyan", lw=0.8, ls="--")
    ax.axvline(c + band, color="cyan", lw=0.8, ls="--")
    ax.axvline(det, color="lime", lw=1.2, ls=":")
    ax.set_xlabel("site"); ax.set_ylabel("time step"); ax.set_title(name)
fig.colorbar(im, ax=axes, label=r"$\log_{10}$ divergence", shrink=0.8)
fig.suptitle("E2 v2: propagating periphery (cyan=band edge, green=detector)")
plt.savefig("fig_E2_trapping.png", dpi=130); plt.close()

json.dump(res, open("results.json", "w"), indent=2)
print("\nUpdated results.json and regenerated E2 figure (v2).")
