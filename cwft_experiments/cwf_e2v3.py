"""
E2 v3 -- the wall test (the correct horizon test).

Fast-propagating background (rho=2.6, robust v_B). Inject a perturbation to the
LEFT of a central band and place the detector to the RIGHT of it, so the front
must CROSS the band to be detected. A genuine trapping surface = a band that
blocks transmission a uniform medium would allow.

  uniform        : rho=2.6 everywhere               -> front should cross (control)
  high_rho_band  : central band rho=3.6 (amplifier) -> cross or block?
  damped_band    : central band leak=2.5 (absorber) -> the candidate wall
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, RNG

THETA = 1e-4

def wall_case(n=301, m=24, coupling=0.18, rho_bg=2.6, band=15,
              rho_band=2.6, leak_band=0.0, inject_off=-55, detect_off=+55,
              T=1200, seed=21, seeds=8):
    center = n // 2
    inj = center + inject_off
    det = center + detect_off
    rho = np.full(n, rho_bg)
    rho[center - band: center + band + 1] = rho_band
    leak = np.zeros(n)
    if leak_band:
        leak[center - band: center + band + 1] = leak_band

    reach, Dlast = [], None
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=seed + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)   # inject at inj (left of band)
        Dlast = D
        hit = np.where(D[:, det] > THETA)[0]
        reach.append(int(hit[0]) if hit.size else -1)
    reach = np.array(reach)
    return {"frac_crossed": float(np.mean(reach >= 0)),
            "median_cross_time": (float(np.median(reach[reach >= 0]))
                                  if np.any(reach >= 0) else None),
            "_D": Dlast, "_c": center, "_band": band, "_inj": inj, "_det": det}

res = json.load(open("results.json"))
print("E2 v3 -- WALL TEST (inject left, detect right, must cross band):")
configs = {"uniform":       dict(rho_band=2.6, leak_band=0.0),
           "high_rho_band": dict(rho_band=3.6, leak_band=0.0),
           "damped_band":   dict(rho_band=2.6, leak_band=2.5)}
out, fields = {}, {}
for name, kw in configs.items():
    r = wall_case(**kw)
    fields[name] = r
    s = {k: v for k, v in r.items() if not k.startswith("_")}
    out[name] = s
    tag = ("TRANSMITS" if s["frac_crossed"] > 0.5 else "BLOCKS (wall/horizon)")
    print(f"    {name:14s} crossed={s['frac_crossed']:.2f} "
          f"cross_t={s['median_cross_time']} -> {tag}")
res["E2_wall"] = out

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
for ax, name in zip(axes, configs):
    r = fields[name]
    Z = np.log10(r["_D"] + 1e-16)
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="inferno", vmin=-12, vmax=0)
    ax.axvline(r["_c"] - r["_band"], color="cyan", lw=0.8, ls="--")
    ax.axvline(r["_c"] + r["_band"], color="cyan", lw=0.8, ls="--")
    ax.axvline(r["_inj"], color="white", lw=1.0, ls="-")
    ax.axvline(r["_det"], color="lime", lw=1.2, ls=":")
    ax.set_xlabel("site"); ax.set_ylabel("time step"); ax.set_title(name)
fig.colorbar(im, ax=axes, label=r"$\log_{10}$ divergence", shrink=0.8)
fig.suptitle("E2 wall test (white=injection, cyan=band, green=detector)")
plt.savefig("fig_E2_wall.png", dpi=130); plt.close()

json.dump(res, open("results.json", "w"), indent=2)
print("\nUpdated results.json; wrote fig_E2_wall.png.")
