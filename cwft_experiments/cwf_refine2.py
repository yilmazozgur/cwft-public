"""
Second refinement.
  E2  proper trapping metric: max front radius (in band-halfwidth units) and
      whether/when the front crosses the band edge. >1 means it escaped.
  E3  more seeds, target front radius set safely inside the boundary, honest
      ballistic-vs-log comparison with monotonicity check.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import (divergence_field, front_positions, E2_trapping, RNG)

res = json.load(open("results.json"))

# ---------------- E2 front-based trapping metric ----------------
print("E2 trapping (max front radius / band-halfwidth; >1 = escaped):")
cases = {"control_lowrho": dict(rho_hi=0.6, leak_band=0.0),
         "high_rho_band":  dict(rho_hi=2.4, leak_band=0.0),
         "damped_band":    dict(rho_hi=1.2, leak_band=1.5)}
e2 = {}
for name, kw in cases.items():
    D, r, c, band = E2_trapping(**kw)
    rmax = np.nanmax(r) if np.any(~np.isnan(r)) else 0.0
    ratio = rmax / band
    cross = np.where(r > band)[0]
    cross_t = int(cross[0]) if cross.size else None
    e2[name] = {"max_radius_over_band": float(ratio),
                "escaped": bool(ratio > 1.0),
                "crossing_time": cross_t}
    tag = "ESCAPED" if ratio > 1 else "TRAPPED/confined"
    print(f"    {name:16s}  rmax/band={ratio:5.2f}  cross_t={cross_t}  -> {tag}")
res["E2"]["trapping_metric"] = e2

# ---------------- E3 robust ballistic scrambling ----------------
def scrambling_robust(sizes, rho=2.0, m=24, coupling=0.18, seeds=12):
    tm, ts = [], []
    for n in sizes:
        center = n // 2
        T = 10 * n
        vals = []
        for s in range(seeds):
            lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=500 + s)
            H0 = 0.1 * RNG.standard_normal((n, m))
            D = divergence_field(lat, H0, T, center)
            r = front_positions(D, center)
            target = 0.7 * (n // 2)            # stay clear of open-boundary edge
            hit = np.where(r >= target)[0]
            if hit.size:
                vals.append(hit[0])
        vals = np.array(vals)
        tm.append(float(np.median(vals)) if vals.size else np.nan)  # median: robust
        ts.append(float(np.std(vals)) if vals.size else np.nan)
    return np.array(tm), np.array(ts)

sizes = np.array([41, 61, 81, 101, 141, 181, 221, 281, 341])
print("E3 (rho=2.0, 12 seeds, median): scrambling time vs N")
t3, t3s = scrambling_robust(sizes)
for nn, tt in zip(sizes, t3):
    print(f"    N={nn:4d}  t*={tt:.1f}")
good = ~np.isnan(t3)
mono = bool(np.all(np.diff(t3[good]) >= 0))
lin = np.polyfit(sizes[good], t3[good], 1)
logf = np.polyfit(np.log(sizes[good]), t3[good], 1)
lin_res = float(np.sum((np.polyval(lin, sizes[good]) - t3[good]) ** 2))
log_res = float(np.sum((np.polyval(logf, np.log(sizes[good])) - t3[good]) ** 2))
# also report R^2 for linear
ss_tot = np.sum((t3[good] - t3[good].mean()) ** 2)
r2_lin = 1 - lin_res / ss_tot
verdict = "ballistic (t* ~ N)" if lin_res < log_res else "log (t* ~ log N)"
print(f"    monotone={mono}  linear resid={lin_res:.0f} (R^2={r2_lin:.3f})  "
      f"log resid={log_res:.0f}  -> {verdict}")
print(f"    implied v_B from ballistic slope = {1.0/lin[0]:.3f} sites/step "
      f"(compare E1: ~0.11)")
res["E3"] = {"rho": 2.0, "seeds": 12, "N": sizes.tolist(),
             "tstar_median": t3.tolist(), "tstar_std": t3s.tolist(),
             "monotone": mono, "linear_fit_resid": lin_res, "linear_R2": float(r2_lin),
             "log_fit_resid": log_res, "verdict": verdict,
             "implied_vB": float(1.0 / lin[0]) if lin[0] > 0 else None}

plt.figure(figsize=(7, 4.4))
plt.errorbar(sizes, t3, yerr=t3s, marker="s", capsize=3, label="measured $t_*$ (median)")
xs = np.linspace(sizes.min(), sizes.max(), 100)
plt.plot(xs, np.polyval(lin, xs), "r--", label=f"linear fit ($R^2$={r2_lin:.3f})")
plt.plot(xs, np.polyval(logf, np.log(xs)), "g--", label="log fit")
plt.xlabel("system size  N"); plt.ylabel(r"scrambling time  $t_*$")
plt.title(f"E3: scrambling-time scaling ($\\rho=2.0$, local coupling)  ->  {verdict}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig_E3_scrambling.png", dpi=130); plt.close()

json.dump(res, open("results.json", "w"), indent=2)
print("\nUpdated results.json and regenerated E3 figure.")
