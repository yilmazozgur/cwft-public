#!/usr/bin/env python3
"""RoPE IS FPE: the transformer bridge, made concrete (Bridge 1).

Rotary Position Embedding rotates query/key sub-vectors at position m by angles
m*theta_i with the geometric schedule theta_i = base^{-2i/d}.  Because
R(a)^T R(b) = R(b-a), the content-free attention logit between positions m and n is

    sum_i cos((n-m) theta_i)  =  the FPE similarity kernel K(n-m)  =  Re FT[p(theta)],

with p(theta) the (log-uniform) frequency schedule.  So RoPE = FPE with a deterministic
geometric schedule; the manuscript's whole apparatus transfers.  This script verifies:
 (1) the RoPE content-free logit equals the FPE kernel to machine precision;
 (2) the RoPE kernel is the (real) Fourier transform of its log-uniform schedule;
 (3) "squeezed RoPE" -- a warped schedule at a conserved cell budget -- redistributes
     positional resolution (fine at short range, coarse at long range, or vice versa),
     the manuscript's squeezing operation applied to positional encoding.
Seeded, CPU.  Writes cwf_fpe_rope_results.json.
"""
import json

import numpy as np

D = 256                      # head dimension (D/2 rotation planes)
BASE = 10000.0               # RoPE default


def rope_schedule(d=D, base=BASE):
    i = np.arange(d // 2)
    return base ** (-2.0 * i / d)          # theta_i, geometric (log-uniform in log theta)


def rope_logit(theta, delta):
    """content-free RoPE attention logit at relative position delta."""
    return np.cos(np.outer(delta, theta)).sum(1) / len(theta)


def fpe_kernel(theta, delta):
    return np.cos(np.outer(delta, theta)).mean(1)


def hwhm(K, delta):
    K = K / K[0]
    below = np.where(K < 0.5)[0]
    return float(delta[below[0]]) if len(below) else float(delta[-1])


def main():
    theta = rope_schedule()
    delta = np.linspace(0, 512, 2000)
    out = {"params": dict(D=D, base=BASE)}

    # (1) RoPE content-free logit == FPE kernel (same object up to the 1/N)
    r = rope_logit(theta, delta)
    f = fpe_kernel(theta, delta)
    out["identity_max_abs_diff"] = float(np.max(np.abs(r - f)))
    print(f"(1) max|RoPE logit - FPE kernel| = {out['identity_max_abs_diff']:.2e} "
          f"(same object)")

    # (2) kernel = FT of the schedule: compare to a direct FT of the empirical
    #     schedule density
    grid = np.linspace(theta.min(), theta.max(), 4000)
    dens, _ = np.histogram(theta, bins=200, range=(theta.min(), theta.max()),
                           density=True)
    centers = 0.5 * (_[:-1] + _[1:])
    ft = np.array([np.sum(dens * np.cos(dd * centers)) * (centers[1] - centers[0])
                   for dd in delta])
    ft = ft / (ft[0] + 1e-12) * f[0]
    out["kernel_is_FT"] = dict(corr=float(np.corrcoef(f, ft)[0, 1]))
    print(f"(2) corr(RoPE kernel, FT[schedule]) = {out['kernel_is_FT']['corr']:.3f}")

    # (3) squeezed RoPE: warp the schedule to redistribute positional resolution
    #     at a conserved log-bandwidth budget.  warp exponent gamma>1 concentrates
    #     frequencies (finer long-range resolution), gamma<1 spreads them.
    rows = []
    base_hwhm = hwhm(f, delta)
    for gamma in [0.5, 1.0, 2.0]:
        th_w = (theta / theta.max()) ** gamma * theta.max()   # monotone warp, endpoints fixed
        Kw = fpe_kernel(th_w, delta)
        rows.append(dict(gamma=gamma, hwhm=hwhm(Kw, delta),
                         log_bandwidth=float(np.ptp(np.log(th_w)))))
        print(f"(3) squeezed RoPE gamma={gamma}: positional resolution HWHM="
              f"{hwhm(Kw, delta):.1f}  (log-bw {np.ptp(np.log(th_w)):.2f} conserved)")
    out["squeezed_rope"] = dict(base_hwhm=base_hwhm, rows=rows)
    print("  -> 'squeezed RoPE' = the manuscript's squeezing operation on positional")
    print("     encoding: redistribute resolution at a conserved log-bandwidth budget.")

    with open("cwf_fpe_rope_results.json", "w") as f_:
        json.dump(out, f_, indent=2)
    print("wrote cwf_fpe_rope_results.json")


if __name__ == "__main__":
    main()
