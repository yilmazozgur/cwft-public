#!/usr/bin/env python3
"""SQUEEZED FPE, the conjugate side: local bandwidth broadening and the local cell.

The squeezing result (cwf_fpe_squeezing.py) measured the resolution side only.
Here we measure the CONJUGATE side directly from the encoded signal: the local
frequency content of the warped atom phi(w(x))_k has phase slope d/dx[w(x) theta_k]
= w'(x) theta_k, so the local bandwidth is w'(x)*sigma.  We estimate the slopes by
finite differences of the component phases (a direct signal measurement, no access
to w'), form the local cell

    res(x) * bw(x)  (kernel-HWHM x std convention),

and test pointwise conservation: it should equal hbar_c = sqrt(2 ln 2) everywhere,
fovea and periphery alike.  Also quantifies the equivariance trade: a nominal bind
shift a displaces a stored value by w^{-1}(w(x)+a) - x, which is 1/w'(x) * a to
first order -- unequal at fovea vs periphery.

Seeded, CPU.  Writes cwf_fpe_localcell_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
A, B, XF = 0.0, 10.0, 5.0
HBARC = np.sqrt(2 * np.log(2))


def warp():
    xg = np.linspace(A, B, 4000)
    dens = 0.2 + np.exp(-0.5 * ((xg - XF) / 1.0) ** 2)
    cdf = np.cumsum(dens)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return xg, A + (B - A) * cdf


def res_at(xg, wg, theta, x0, dmax=4.0, n=1200):
    """Two-sided local resolution: average of the +d and -d first crossings of
    K(w(x0+/-d)-w(x0)) below 1/2 (a one-sided probe biases res where w'' is
    large).  Each direction is probed only within the domain; a direction that
    reaches the boundary without crossing is dropped."""
    w0 = np.interp(x0, xg, wg)
    res = []
    for sgn in (+1, -1):
        lim = min(dmax, (B - x0) if sgn > 0 else (x0 - A))
        d = np.linspace(0, lim, n)
        wd = np.interp(x0 + sgn * d, xg, wg)
        K = np.cos(np.outer(wd - w0, theta)).mean(1)
        below = np.where(K < 0.5)[0]
        if len(below):
            res.append(float(d[below[0]]))
    return float(np.mean(res)) if res else float(dmax)


def local_bw(xg, wg, theta, x0, eps=1e-3):
    """Local frequency std from finite-difference phase slopes of the encoded
    components (no analytic access to w')."""
    w0 = np.interp(x0, xg, wg)
    w1 = np.interp(x0 + eps, xg, wg)
    slopes = (w1 - w0) / eps * theta        # phase slope of component k at x0
    return float(np.std(slopes))


def main():
    rng = np.random.default_rng(0)
    theta = SIGMA * rng.standard_normal(N)
    xg, wg = warp()
    xpts = np.linspace(0.5, 8.5, 17)
    rows = []
    for x0 in xpts:
        r = res_at(xg, wg, theta, x0)
        bw = local_bw(xg, wg, theta, x0)
        rows.append(dict(x=float(x0), res=r, bw=bw, cell=r * bw))
    cells = np.array([r["cell"] for r in rows])
    dev = float(np.max(np.abs(cells - HBARC)) / HBARC)
    # equivariance trade: displacement under a nominal bind shift a=1
    a = 1.0
    disp = {}
    for label, x0 in [("fovea", XF), ("periphery", 1.5)]:
        w0 = np.interp(x0, xg, wg)
        x1 = float(np.interp(w0 + a, wg, xg))     # w^{-1}(w(x0)+a)
        disp[label] = x1 - x0
    out = dict(params=dict(N=N, sigma=SIGMA, fovea=XF, hbar_c=float(HBARC)),
               rows=rows, max_rel_dev_from_hbarc=dev,
               bind_shift_a=a, displacement=disp)
    print(f"local cell res(x)*bw(x): mean={cells.mean():.4f} "
          f"(hbar_c={HBARC:.4f}); max rel. deviation = {dev*100:.1f}%")
    print(f"bind shift a={a}: displacement at fovea = {disp['fovea']:.2f}, "
          f"at periphery = {disp['periphery']:.2f}")
    with open("cwf_fpe_localcell_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_localcell_results.json")


if __name__ == "__main__":
    main()
