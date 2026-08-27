"""
Refinement pass addressing three issues found in the first run:
  E2  add a quantitative *escape fraction* (how much divergence crosses the
      band boundary) so 'trapping' is measured, not just eyeballed.
  E3  rerun scrambling at a robustly super-critical rho=2.0 (the first run
      used rho=1.4, marginal -> NaNs); fit ballistic vs log.
  E4  measure LEFT and RIGHT Rule-110 cone speeds separately (the cone is
      asymmetric, so a single slope is meaningless).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice, rule110_run
from cwf_experiments import (divergence_field, front_positions, fit_velocity,
                             E2_trapping, RNG)

res = json.load(open("results.json"))

# ---------------- E2 quantitative escape fraction ----------------
def escape_fraction(D, center, band):
    """Fraction of total divergence 'energy' (summed over the second half of
    the run, to use the settled regime) located OUTSIDE the band."""
    T1, n = D.shape
    dist = np.abs(np.arange(n) - center)
    outside = dist > band
    late = D[T1 // 2:]              # settled regime
    total = late.sum() + 1e-30
    return float(late[:, outside].sum() / total)

print("E2 quantitative trapping (escape fraction; lower = more trapped):")
cases = {
    "control_lowrho":  dict(rho_hi=0.6, leak_band=0.0),
    "high_rho_band":   dict(rho_hi=2.4, leak_band=0.0),
    "damped_band":     dict(rho_hi=1.2, leak_band=1.5),
}
e2 = {}
for name, kw in cases.items():
    D, r, c, band = E2_trapping(**kw)
    ef = escape_fraction(D, c, band)
    e2[name] = ef
    print(f"    {name:16s}  escape_fraction = {ef:.3f}")
res["E2"]["escape_fraction"] = e2

# ---------------- E3 clean scrambling at rho=2.0 ----------------
def scrambling_clean(sizes, rho=2.0, m=24, coupling=0.18, seeds=6):
    tm, ts = [], []
    for n in sizes:
        center = n // 2
        T = 8 * n
        vals = []
        for s in range(seeds):
            lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=300 + s)
            H0 = 0.1 * RNG.standard_normal((n, m))
            D = divergence_field(lat, H0, T, center)
            r = front_positions(D, center)
            target = 0.9 * (n // 2)
            hit = np.where(r >= target)[0]
            if hit.size:
                vals.append(hit[0])
        tm.append(np.mean(vals) if vals else np.nan)
        ts.append(np.std(vals) if vals else np.nan)
    return np.array(tm), np.array(ts)

sizes = np.array([41, 61, 81, 101, 141, 181, 221, 281])
print("E3 (rho=2.0): scrambling time vs N")
t3, t3s = scrambling_clean(sizes)
for nn, tt in zip(sizes, t3):
    print(f"    N={nn:4d}  t*={tt:.1f}")
good = ~np.isnan(t3)
lin = np.polyfit(sizes[good], t3[good], 1)
logf = np.polyfit(np.log(sizes[good]), t3[good], 1)
lin_res = float(np.sum((np.polyval(lin, sizes[good]) - t3[good]) ** 2))
log_res = float(np.sum((np.polyval(logf, np.log(sizes[good])) - t3[good]) ** 2))
verdict = "ballistic (t* ~ N)" if lin_res < log_res else "log/fast-scrambler (t* ~ log N)"
print(f"    linear resid={lin_res:.1f}  log resid={log_res:.1f}  -> {verdict}")
res["E3"] = {"rho": 2.0, "N": sizes.tolist(), "tstar": t3.tolist(),
             "tstar_std": t3s.tolist(), "linear_fit_resid": lin_res,
             "log_fit_resid": log_res, "verdict": verdict,
             "implied_vB": float(1.0 / lin[0]) if lin[0] > 0 else None}

# ---------------- E4 asymmetric Rule-110 cone ----------------
def rule110_cone_lr(n=401, T=200, seed=1):
    rng = np.random.default_rng(seed)
    row0 = rng.integers(0, 2, n).astype(np.uint8)
    row1 = row0.copy(); row1[n // 2] ^= 1
    a = rule110_run(row0, T); b = rule110_run(row1, T)
    diff = (a ^ b).astype(np.uint8)
    center = n // 2
    left_edge = np.full(T + 1, center, float)
    right_edge = np.full(T + 1, center, float)
    for t in range(T + 1):
        cols = np.where(diff[t] > 0)[0]
        if cols.size:
            left_edge[t] = cols.min()
            right_edge[t] = cols.max()
    tt = np.arange(T + 1)
    vL = -np.polyfit(tt, left_edge, 1)[0]    # leftward speed (cells/step)
    vR = np.polyfit(tt, right_edge, 1)[0]    # rightward speed
    return diff, left_edge, right_edge, center, vL, vR

diff110, le, re_, c4, vL, vR = rule110_cone_lr()
print(f"E4 Rule 110 cone: left speed={vL:.3f}, right speed={vR:.3f} cells/step "
      f"(LR bound = 1.0)")
res["E4"] = {"left_cone_speed": float(vL), "right_cone_speed": float(vR),
             "LR_bound": 1.0}

# ---------------- regenerate E3 and E4 figures ----------------
plt.figure(figsize=(7, 4.4))
plt.errorbar(sizes, t3, yerr=t3s, marker="s", capsize=3, label="measured $t_*$")
xs = np.linspace(sizes.min(), sizes.max(), 100)
plt.plot(xs, np.polyval(lin, xs), "r--", label="linear (ballistic) fit")
plt.plot(xs, np.polyval(logf, np.log(xs)), "g--", label="log (fast-scrambler) fit")
plt.xlabel("system size  N"); plt.ylabel(r"scrambling time  $t_*$")
plt.title(f"E3: scrambling-time scaling ($\\rho=2.0$)  ->  {verdict}")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig_E3_scrambling.png", dpi=130); plt.close()

plt.figure(figsize=(6.4, 5.2))
plt.imshow(diff110, aspect="auto", origin="lower", cmap="binary",
           interpolation="nearest")
tt = np.arange(len(le))
plt.plot(le, tt, "r-", lw=1.4, label=f"left edge ~ {vL:.2f} c/step")
plt.plot(re_, tt, "b-", lw=1.4, label=f"right edge ~ {vR:.2f} c/step")
plt.xlabel("cell"); plt.ylabel("time step")
plt.title("E4: Rule 110 asymmetric perturbation cone")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig("fig_E4_rule110_cone.png", dpi=130); plt.close()

json.dump(res, open("results.json", "w"), indent=2)
print("\nUpdated results.json and regenerated E3, E4 figures.")
