"""
cwf_kappa.py -- surface gravity kappa_c and the Hawking-analog temperature T_c
from the Test 6 effective-metric gain field.

Restores the script behind results.json["EinsteinAnchor"], which had become an
orphaned key (the producing script was missing). No new physics: this reads kappa
off the SAME local-Jacobian gain field g(x) that cwf_test6b.py constructs for the
effective-metric / clock-freeze-horizon test, using the book's definitions
(Chapter "Computational Gravity", sec:grav-test6):

    kappa_c       = (1/2) |d g / d x|  at the g=0 clock-freeze horizon
                    (lattice units a = dt = 1; v_LR absorbed into the convention,
                     book Eq. kappa-c and the Units remark),
    k_B T_c       = hbar_c * kappa_c / (2 pi)   (Gibbons-Hawking analog),
    => T_c/hbar_c = kappa_c / (2 pi).

The gain field g(i) is the time-averaged log spectral radius of each site's local
Jacobian block (identical construction to cwf_test6b.jacobian_gain_profile); g>0 is
the information-producing exterior, g<0 the frozen core, g=0 the horizon. CPU; numpy.
Writes results.json["EinsteinAnchor"] = {horizon_site, dg_dx_at_horizon, kappa_c,
T_c_over_hbar_c}.
"""
import json
import os

import numpy as np

from cwf_substrate import ReservoirLattice
from cwf_experiments import RNG

HERE = os.path.dirname(os.path.abspath(__file__))


def make_well(n, c, w, Lmax):
    x = np.arange(n)
    return Lmax * np.exp(-((x - c) / w) ** 2)


def jacobian_gain_profile(lat, H0, T=400):
    """Time-averaged log spectral radius of each site's local Jacobian block
    (identical to cwf_test6b.jacobian_gain_profile -- the effective-metric gain)."""
    n, m = lat.n, lat.m
    H = H0.copy()
    acc = np.zeros(n)
    cnt = 0
    for t in range(T):
        left = np.zeros_like(H); right = np.zeros_like(H)
        left[1:] = H[:-1]; right[:-1] = H[1:]
        neigh = (left + right) @ lat.P.T
        leak = lat.leak
        leak_arr = (leak if not np.isscalar(leak) else np.full(n, leak))
        pre = lat.rho[:, None] * (H @ lat.W0.T) + lat.c * neigh - leak_arr[:, None] * H
        sech2 = 1.0 - np.tanh(pre) ** 2
        if t > 50:                                # discard transient
            for i in range(n):
                Jii = (sech2[i][:, None] *
                       (lat.rho[i] * lat.W0 - leak_arr[i] * np.eye(m)))
                acc[i] += np.log(np.max(np.abs(np.linalg.eigvals(Jii))) + 1e-30)
            cnt += 1
        H = np.tanh(pre)
    return acc / max(cnt, 1)


def main():
    # identical setup to cwf_test6b.py (the effective-metric test)
    n, c, w, Lmax, rho = 301, 150, 18, 3.0, 2.6
    leak = make_well(n, c, w, Lmax)
    lat = ReservoirLattice(n, m=24, coupling=0.18, rho=rho, seed=970)
    lat.leak = leak
    H0 = 0.1 * RNG.standard_normal((n, 24))
    g = jacobian_gain_profile(lat, H0, T=400)

    # g=0 clock-freeze horizon: last upward-to-downward 0-crossing left of the well centre
    cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    horizon = int(cross[cross < c][-1]) if np.any(cross < c) else c

    # surface gravity: half the |gradient| of g at the horizon (windowed linear fit).
    # The discrete gain field is jagged at the horizon, so kappa_c carries ~10%
    # window-dependence (kappa_c ~ 0.055-0.066 for windows of +-3..+-5 sites); we use
    # the stable +-4-site (9-point) fit.
    win = 4
    xs = np.arange(horizon - win, horizon + win + 1)
    slope = float(np.polyfit(xs, g[xs], 1)[0])
    kappa_c = 0.5 * abs(slope)
    T_over_hbar = kappa_c / (2 * np.pi)

    anchor = {"horizon_site": horizon,
              "dg_dx_at_horizon": slope,
              "kappa_c": float(kappa_c),
              "T_c_over_hbar_c": float(T_over_hbar)}
    print(f"g=0 horizon @ site {horizon};  dg/dx = {slope:+.4f};  "
          f"kappa_c = {kappa_c:.4f};  T_c/hbar_c = {T_over_hbar:.4f}")

    out = os.path.join(HERE, "results.json")
    res = json.load(open(out))
    res["EinsteinAnchor"] = anchor
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote results.json['EinsteinAnchor']")
    return anchor


if __name__ == "__main__":
    main()
