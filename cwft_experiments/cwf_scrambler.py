"""
Scrambling comparison: does locality decide ballistic vs fast (log) scrambling?

  local    : 1D nearest-neighbour reservoir lattice. Perturb centre; t* = time
             for the divergence front to reach the far edge. Expect t* ~ N.
  all2all  : fully-connected reservoir (every node couples to every node).
             Perturb one node; t* = time for ALL nodes to exceed threshold.
             Expect t* ~ log N  (fast-scrambler signature).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, front_positions

RNG = np.random.default_rng(123)
THETA = 1e-4
EPS = 1e-7

def t_local(n, rho=2.6, m=24, coupling=0.18, seeds=10):
    center = n // 2
    T = 12 * n
    vals = []
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=1000 + s)
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, center)
        r = front_positions(D, center, theta=THETA)
        target = 0.7 * (n // 2)
        hit = np.where(r >= target)[0]
        if hit.size:
            vals.append(hit[0])
    return np.median(vals) if vals else np.nan, np.std(vals) if vals else np.nan

def t_all2all(n, rho=2.0, seeds=10):
    """Fully-connected scalar reservoir. t* = time for all nodes to scramble."""
    T = 200
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(2000 + s)
        J = rng.standard_normal((n, n))
        J *= rho / max(abs(np.linalg.eigvals(J)))     # spectral radius rho
        h = 0.1 * rng.standard_normal(n)
        hp = h.copy(); hp[0] += EPS
        ref, per = h.copy(), hp.copy()
        tstar = np.nan
        for t in range(1, T + 1):
            ref = np.tanh(J @ ref)
            per = np.tanh(J @ per)
            D = np.abs(per - ref)
            if np.all(D > THETA):
                tstar = t
                break
        if not np.isnan(tstar):
            vals.append(tstar)
    return np.median(vals) if vals else np.nan, np.std(vals) if vals else np.nan

print("Scrambling: local (nearest-neighbour) vs all-to-all")
sizes = np.array([21, 41, 61, 81, 121, 161, 221, 301])
tl, tls, ta, tas = [], [], [], []
for n in sizes:
    a, b = t_local(n); tl.append(a); tls.append(b)
    c, d = t_all2all(n); ta.append(c); tas.append(d)
    print(f"    N={n:4d}   local t*={a:7.1f}    all2all t*={c:6.1f}")
tl, ta = np.array(tl, float), np.array(ta, float)

# fits
gl = ~np.isnan(tl); ga = ~np.isnan(ta)
lin_local = np.polyfit(sizes[gl], tl[gl], 1)
log_local = np.polyfit(np.log(sizes[gl]), tl[gl], 1)
lin_a2a = np.polyfit(sizes[ga], ta[ga], 1)
log_a2a = np.polyfit(np.log(sizes[ga]), ta[ga], 1)

def resid(y, yhat): return float(np.sum((y - yhat) ** 2))
rl_lin = resid(tl[gl], np.polyval(lin_local, sizes[gl]))
rl_log = resid(tl[gl], np.polyval(log_local, np.log(sizes[gl])))
ra_lin = resid(ta[ga], np.polyval(lin_a2a, sizes[ga]))
ra_log = resid(ta[ga], np.polyval(log_a2a, np.log(sizes[ga])))
v_local = "ballistic (~N)" if rl_lin < rl_log else "log (~log N)"
v_a2a = "log (~log N)" if ra_log < ra_lin else "ballistic (~N)"
print(f"  local   -> {v_local}   (lin resid {rl_lin:.0f} vs log {rl_log:.0f})")
print(f"  all2all -> {v_a2a}   (lin resid {ra_lin:.0f} vs log {ra_log:.0f})")

res = json.load(open("results.json"))
res["Scrambling"] = {"N": sizes.tolist(),
                     "local_tstar": tl.tolist(), "all2all_tstar": ta.tolist(),
                     "local_verdict": v_local, "all2all_verdict": v_a2a}
json.dump(res, open("results.json", "w"), indent=2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
a1.errorbar(sizes, tl, yerr=tls, marker="o", capsize=3)
xs = np.linspace(sizes.min(), sizes.max(), 100)
a1.plot(xs, np.polyval(lin_local, xs), "r--", label="linear")
a1.plot(xs, np.polyval(log_local, np.log(xs)), "g--", label="log")
a1.set_title(f"local (nearest-neighbour): {v_local}")
a1.set_xlabel("N"); a1.set_ylabel(r"$t_*$"); a1.legend(); a1.grid(alpha=0.3)

a2.errorbar(sizes, ta, yerr=tas, marker="s", capsize=3, color="purple")
a2.plot(xs, np.polyval(lin_a2a, xs), "r--", label="linear")
a2.plot(xs, np.polyval(log_a2a, np.log(xs)), "g--", label="log")
a2.set_title(f"all-to-all: {v_a2a}")
a2.set_xlabel("N"); a2.set_ylabel(r"$t_*$"); a2.legend(); a2.grid(alpha=0.3)
fig.suptitle("Scrambling time: locality decides ballistic vs fast (log) scrambling")
plt.tight_layout(); plt.savefig("fig_scrambling_compare.png", dpi=130); plt.close()
print("Wrote fig_scrambling_compare.png")
