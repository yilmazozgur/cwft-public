"""
Test 6 -- the decisive analogy-vs-mechanism test.

Claim (Section 7): a computational horizon (where information transport -> 0)
coincides with a vanishing effective lapse (where the local computational
clock rate -> 0). If a single local field controls BOTH transport and clock
rate, and the horizon sits where it vanishes, that is evidence for an
effective-metric description rather than mere transport-blocking.

Setup: a fast-propagating background (rho=2.6) with a SMOOTH damping well
leak(x) = Lmax * exp(-((x-c)/w)^2). We measure two independent spatial
profiles and test their coincidence:

  clock(x)  -- local Lyapunov / information-production rate (lapse proxy):
               perturb locally at x, measure short-time local growth of the
               divergence at x before it spreads.
  transport -- inject far to the left, measure how deep into the well the
               front penetrates (rightmost site reached). The transport
               horizon is where the front stalls.

Coincidence of (clock -> 0) with (transport stalls) => effective-metric-like.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, front_positions, RNG

THETA = 1e-4

def make_well(n, c, w, Lmax):
    x = np.arange(n)
    return Lmax * np.exp(-((x - c) / w) ** 2)

def local_clock_profile(n=301, m=24, coupling=0.18, rho=2.6, Lmax=3.0, w=18,
                        tau=10, seeds=6):
    """Local Lyapunov rate at each site: perturb at x, measure mean log-growth
    of divergence AT x over the first tau steps (local expansion = clock rate)."""
    c = n // 2
    leak = make_well(n, c, w, Lmax)
    sites = np.arange(8, n - 8, 3)
    clock = np.zeros(len(sites))
    for k, xs in enumerate(sites):
        rates = []
        for s in range(seeds):
            lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=900 + s)
            lat.leak = leak
            H0 = 0.1 * RNG.standard_normal((n, m))
            D = divergence_field(lat, H0, tau + 2, xs)
            d = D[:, xs]
            d = d[d > 0]
            if d.size > 3:
                lg = np.diff(np.log(d[: tau]))
                rates.append(np.mean(lg[np.isfinite(lg)]))
        clock[k] = np.mean(rates) if rates else np.nan
    return sites, clock, leak

def transport_penetration(n=301, m=24, coupling=0.18, rho=2.6, Lmax=3.0, w=18,
                          inject_off=-70, T=1500, seeds=8):
    """Inject far left; rightmost site the front reaches (penetration depth)."""
    c = n // 2
    leak = make_well(n, c, w, Lmax)
    inj = c + inject_off
    reach = np.zeros(n)  # fraction of runs each site is reached
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=950 + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)
        ever = (D > THETA).any(axis=0)
        reach += ever
    reach /= seeds
    return reach, leak, c, inj

print("Test 6: measuring local clock profile and transport penetration ...")
sites, clock, leak = local_clock_profile()
reach, leak2, c, inj = transport_penetration()

# normalise clock to a lapse in [0,1] (1 = full speed far away)
cl = np.array(clock)
far = np.nanmedian(cl[(sites < c - 60) | (sites > c + 60)])
lapse = np.clip(cl / far, 0, 1.5)

# transport horizon (coming from the left): deepest site reached >=50% of runs
reached_sites = np.where(reach >= 0.5)[0]
# the front comes from the left (inj < c); horizon = rightmost reached before the gap
left_of_c = reached_sites[reached_sites <= c]
x_h = int(left_of_c.max()) if left_of_c.size else inj
# lapse at horizon
lap_at_h = float(np.interp(x_h, sites, lapse))
# lapse minimum location
x_lapse_min = int(sites[np.nanargmin(lapse)])

print(f"    transport horizon (front stalls) at site x_h = {x_h}  (center={c})")
print(f"    lapse minimum at site = {x_lapse_min}")
print(f"    normalized lapse at the transport horizon = {lap_at_h:.3f}")
print(f"    => horizon sits where the effective clock is suppressed to "
      f"{lap_at_h:.0%} of its far-field rate")

# correlation between lapse profile and (smoothed) reach profile over the well
reach_at_sites = reach[sites]
mask = np.isfinite(lapse)
corr = float(np.corrcoef(lapse[mask], reach_at_sites[mask])[0, 1])
print(f"    Pearson corr(lapse, transport-reach) across the well = {corr:.3f}")

res = json.load(open("results.json"))
res["Test6"] = {"center": int(c), "transport_horizon": x_h,
                "lapse_min_site": x_lapse_min,
                "normalized_lapse_at_horizon": lap_at_h,
                "corr_lapse_transport": corr,
                "Lmax": 3.0, "well_width": 18}
json.dump(res, open("results.json", "w"), indent=2)

# figure
fig, ax1 = plt.subplots(figsize=(8, 4.6))
ax1.plot(sites, lapse, "b-o", ms=3, label="effective lapse  (clock rate, normalised)")
ax1.plot(np.arange(len(reach)), reach, "g-", alpha=0.7,
         label="transport reach (frac of runs)")
ax1.axvline(x_h, color="r", ls="--", label=f"transport horizon x={x_h}")
ax1.axvline(c, color="k", ls=":", lw=0.7, label="well centre")
# overlay damping well (scaled)
ax1.plot(np.arange(len(leak)), leak / leak.max(), color="orange", alpha=0.5,
         label="damping well (scaled)")
ax1.set_xlabel("site"); ax1.set_ylabel("normalised quantity")
ax1.set_xlim(c - 90, c + 90)
ax1.set_title(f"Test 6: transport horizon vs effective lapse  "
              f"(corr={corr:.2f}, lapse@horizon={lap_at_h:.2f})")
ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_Test6_metric.png", dpi=130); plt.close()
print("Wrote fig_Test6_metric.png")
