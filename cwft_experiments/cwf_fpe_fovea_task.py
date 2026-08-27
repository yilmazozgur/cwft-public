#!/usr/bin/env python3
"""A DOWNSTREAM TASK for squeezed FPE: value decoding under channel noise.

Encode a value x with flat vs MULTI-SCALE vs squeezed (foveated) FPE at matched
dimension and matched total spectral second moment, corrupt the hypervector with
complex Gaussian channel noise, decode by matched-filter argmax over a fine grid,
and report RMSE(x_hat) separately for values inside the fovea (|x-5| < 1) and in
the periphery (x in [1,3] u [7,9]).

The multi-scale arm is the baseline the novelty claim actually hangs on: multi-scale
SSP (Dumont & Eliasmith 2020) is the nearest VSA precedent for "spend resolution
non-uniformly".  It splits the N frequencies into G groups with different bandwidths
sigma_g, chosen so that sum_g sigma_g^2 / G == SIGMA^2 -- i.e. the same total
spectral second moment as the flat codebook, hence the same phase-space budget.

Prediction: multi-scale CANNOT foveate, because its kernel is stationary -- it
depends only on the gap between two values, never on where they sit -- so its
resolution is position-independent by construction.  Only the warp, which is
non-stationary, can move resolution to a region of interest.  Squeezing should
therefore buy fovea precision at the price of peripheral precision, while
multi-scale should be flat across regions.

SCOPE (important, and measured in part 3 below).  The matched-second-moment budget
above is a MODELLING CHOICE, not a physical constraint: in this task nothing charges
for bandwidth, so a flat codebook at larger sigma buys resolution for free and
dominates squeezed FPE everywhere (measured: sweep_uncapped).  Squeezing is the
right move only where the bandwidth budget actually BINDS -- i.e. under an aperture.
Part 3 imposes one and re-runs the comparison there.  Note why the comparison is
exactly fair under an aperture: squeezed FPE reuses the SAME codebook (it warps the
INPUT, not theta), so both arms sit at identical maximum frequency, and the flat arm
cannot buy its way out by widening.

PART 4 (added after external review) RE-EXAMINES THAT LAST SENTENCE.  Two facts make
it wrong as written.  (i) `sigma_cap` in part 3 is the STANDARD DEVIATION of a Gaussian
draw, not a cap: ~31.7% of |theta| exceed it and the largest exceeds it by 3.6-4.8x.
(ii) More importantly, the squeezed atom is exp(i theta_j w(x)), whose phase turns at
d/dx = theta_j w'(x).  The EFFECTIVE local spatial frequency is therefore theta_j w'(x),
and the warp has max_x w'(x) = 2.663.  So the two arms do NOT sit at identical maximum
spatial frequency: the warped arm runs 2.663x faster at the fovea.  If the device
constrains the largest representable spatial frequency -- which is what an aperture
physically is -- the warped arm violates it.

Part 4 therefore re-runs the aperture comparison under a genuinely HARD-BOUNDED codebook
(truncated normal, |theta| <= 3 sigma exactly) and three explicit matched resources:
  (a) same base-frequency law   -- what part 3 / the manuscript does (th_sq = th_flat)
  (b) same RMS effective slope  -- sqrt(E_x[w'(x)^2]) * sigma_sq == sigma_flat
  (c) same MAX effective slope  -- max_x w'(x) * max|theta_sq| == max|theta_flat|
(c) is the honest reading of "aperture" as a bound on representable spatial frequency.

200 trials/cell, 3 codebook seeds (parts 1-3); 400 trials x 8 seeds (part 4).
Writes cwf_fpe_fovea_task_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
A, B, XF = 0.0, 10.0, 5.0
NOISE = 1.0                 # channel noise std per complex component
TRIALS = 200
SEEDS = 3
GRID = np.linspace(A, B, 2001)

# Part 3: the aperture sweep.  Each value is a hard cap on the codebook bandwidth
# (a device aperture: the largest spatial frequency the substrate can represent).
CAPS = (1.0, 2.0, 4.0, 8.0)
# Part 2: the uncapped control -- what the flat arm does when bandwidth is free.
UNCAPPED = (2.0, 3.0, 5.0, 8.0, 20.0)

# Multi-scale bandwidths: sum of squares == 3 * SIGMA^2, so the total spectral
# second moment (and hence the phase-space budget) matches the flat codebook.
MS_SIGMAS = (1.0, 2.0, np.sqrt(3.0 * SIGMA ** 2 - 1.0 ** 2 - 2.0 ** 2))


def warp():
    xg = np.linspace(A, B, 4000)
    dens = 0.2 + np.exp(-0.5 * ((xg - XF) / 1.0) ** 2)
    cdf = np.cumsum(dens)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return xg, A + (B - A) * cdf


def multiscale_freqs(rng):
    """N frequencies split evenly across MS_SIGMAS, matched total second moment."""
    g = len(MS_SIGMAS)
    parts = [MS_SIGMAS[i] * rng.standard_normal(N // g + (1 if i < N % g else 0))
             for i in range(g)]
    return np.concatenate(parts)


def run(seed):
    rng = np.random.default_rng(seed)
    th = SIGMA * rng.standard_normal(N)
    th_ms = multiscale_freqs(rng)
    xg, wg = warp()
    wgrid = np.interp(GRID, xg, wg)
    enc = {"flat": lambda x: np.exp(1j * x * th),
           "multiscale": lambda x: np.exp(1j * x * th_ms),
           "squeezed": lambda x: np.exp(1j * np.interp(x, xg, wg) * th)}
    dec_book = {"flat": np.exp(-1j * np.outer(GRID, th)),
                "multiscale": np.exp(-1j * np.outer(GRID, th_ms)),
                "squeezed": np.exp(-1j * np.outer(wgrid, th))}
    kinds = ("flat", "multiscale", "squeezed")
    out = {}
    # Periphery is the union of both flanks, as the docstring says; sampling only
    # the left flank halved the peripheral sample in earlier runs.
    for region, bands in [("fovea", [(XF - 1, XF + 1)]),
                          ("periphery", [(1.0, 3.0), (7.0, 9.0)])]:
        halves = [rng.uniform(lo, hi, TRIALS // len(bands)) for lo, hi in bands]
        xs_true = np.concatenate(halves)
        for kind in kinds:
            err = []
            for x in xs_true:
                v = enc[kind](x) + NOISE * (rng.standard_normal(N)
                                            + 1j * rng.standard_normal(N)) / np.sqrt(2)
                sim = np.real(dec_book[kind] @ v)
                err.append(GRID[int(np.argmax(sim))] - x)
            out[(region, kind)] = float(np.sqrt(np.mean(np.square(err))))
    return out


def _rmse_regions(th, encode, decbook, rng):
    """RMSE in fovea and periphery for one (codebook, encoder, decoder) triple."""
    out, gross = {}, {}
    for region, bands in [("fovea", [(XF - 1, XF + 1)]),
                          ("periphery", [(1.0, 3.0), (7.0, 9.0)])]:
        halves = [rng.uniform(lo, hi, TRIALS // len(bands)) for lo, hi in bands]
        xs_true = np.concatenate(halves)
        err = []
        for x in xs_true:
            v = encode(x) + NOISE * (rng.standard_normal(N)
                                     + 1j * rng.standard_normal(N)) / np.sqrt(2)
            sim = np.real(decbook @ v)
            i = int(np.argmax(sim))
            # Three-point parabolic refinement of the peak.  Without it the RMSE
            # floors at the grid quantization (dx/sqrt(12) = 1.4e-3), which is
            # exactly where the widest apertures land -- the gain would then look
            # like it saturates when it is only the ruler running out.
            if 0 < i < len(GRID) - 1:
                y0, y1, y2 = sim[i - 1], sim[i], sim[i + 1]
                den = y0 - 2 * y1 + y2
                d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
                d = float(np.clip(d, -1.0, 1.0))
            else:
                d = 0.0
            err.append(GRID[i] + d * (GRID[1] - GRID[0]) - x)
        err = np.asarray(err)
        out[region] = float(np.sqrt(np.mean(err ** 2)))
        # a "gross" error is a decode landing outside one fovea half-width: it
        # signals the range-ambiguity limit, i.e. that the geometric budget binds.
        gross[region] = float(np.mean(np.abs(err) > 0.5))
    return out, gross


def run_uncapped(seed, sigma):
    """PART 2 CONTROL: flat FPE at a free choice of bandwidth.

    Nothing in this task charges for sigma, so this sweep measures how much
    resolution a flat codebook can buy simply by widening -- the reason the
    matched-second-moment comparison of part 1 cannot support an engineering claim.
    """
    rng = np.random.default_rng(1000 + seed)
    th = sigma * rng.standard_normal(N)
    book = np.exp(-1j * np.outer(GRID, th))
    return _rmse_regions(th, lambda x: np.exp(1j * x * th), book, rng)


def run_capped(seed, sigma_cap):
    """PART 3: flat vs squeezed under a hard bandwidth aperture.

    The aperture caps the codebook bandwidth at sigma_cap.  Squeezed FPE warps the
    INPUT and reuses the same theta, so both arms sit at exactly the same aperture:
    the flat arm cannot widen its way out, and redistribution is the only move left.
    This is the regime where the geometric budget binds and squeezing is a real
    engineering option rather than a bookkeeping artifact.
    """
    rng = np.random.default_rng(2000 + seed)
    th = sigma_cap * rng.standard_normal(N)
    xg, wg = warp()
    wgrid = np.interp(GRID, xg, wg)
    flat, gf = _rmse_regions(th, lambda x: np.exp(1j * x * th),
                             np.exp(-1j * np.outer(GRID, th)), rng)
    sq, gs = _rmse_regions(th, lambda x: np.exp(1j * np.interp(x, xg, wg) * th),
                           np.exp(-1j * np.outer(wgrid, th)), rng)
    return flat, sq, gf, gs


# ======================================================================
# PART 4: matched-resource audit of the aperture claim (added after review)
# ======================================================================
TRIALS4 = 400          # trials per region per cell
SEEDS4 = 8             # codebook seeds
TRUNC = 3.0            # hard bound on |theta|, in units of sigma
REGIONS = [("fovea", [(XF - 1, XF + 1)]),
           ("periphery", [(1.0, 3.0), (7.0, 9.0)])]


def warp_slope():
    """(max_x w'(x), rms_x w'(x), min_x w'(x)) for the committed warp."""
    xg, wg = warp()
    wp = np.gradient(wg, xg)
    return float(wp.max()), float(np.sqrt(np.mean(wp ** 2))), float(wp.min()), xg, wp


def truncated_freqs(rng, sigma, n=N, trunc=TRUNC):
    """Gaussian frequencies with a GENUINE hard bound |theta| <= trunc*sigma.

    Resampling (not clipping) keeps the shape of the bulk; the bound is then a real
    aperture, seed-independent, so the max-slope matching below is exact rather than
    hostage to the tail of a particular draw.
    """
    th = sigma * rng.standard_normal(n)
    bad = np.abs(th) > trunc * sigma
    while bad.any():
        th[bad] = sigma * rng.standard_normal(int(bad.sum()))
        bad = np.abs(th) > trunc * sigma
    return th


def _rmse_batched(th, u_of_x, decbook, rng, trials=TRIALS4):
    """Vectorized twin of _rmse_regions: encode exp(i*u(x)*th), decode by matched filter.

    u_of_x maps a true value x to the coordinate the codebook is evaluated at
    (identity for flat FPE, the warp w for squeezed FPE).  Identical noise model,
    identical parabolic peak refinement, identical gross-error definition.
    """
    dx = GRID[1] - GRID[0]
    out, gross = {}, {}
    for region, bands in REGIONS:
        xs = np.concatenate([rng.uniform(lo, hi, trials // len(bands))
                             for lo, hi in bands])
        # (N, T) clean atoms, then complex Gaussian channel noise, same std as parts 1-3
        V = np.exp(1j * np.outer(th, u_of_x(xs)))
        V = V + NOISE * (rng.standard_normal(V.shape)
                         + 1j * rng.standard_normal(V.shape)) / np.sqrt(2)
        sim = np.real(decbook @ V)                      # (G, T)
        i = np.argmax(sim, axis=0)
        ic = np.clip(i, 1, len(GRID) - 2)
        t = np.arange(len(xs))
        y0, y1, y2 = sim[ic - 1, t], sim[ic, t], sim[ic + 1, t]
        den = y0 - 2 * y1 + y2
        d = np.where(den != 0, 0.5 * (y0 - y2) / np.where(den != 0, den, 1.0), 0.0)
        d = np.clip(d, -1.0, 1.0)
        d = np.where((i > 0) & (i < len(GRID) - 1), d, 0.0)
        err = GRID[i] + d * dx - xs
        out[region] = float(np.sqrt(np.mean(err ** 2)))
        gross[region] = float(np.mean(np.abs(err) > 0.5))
    return out, gross


def run_matched(seed, sigma):
    """One (seed, aperture) cell: flat arm plus the squeezed arm under three matchings.

    The three matchings differ ONLY in how the squeezed arm's codebook is scaled:
      base : th_sq = th_flat                  (identical codebook -- the current claim)
      rms  : th_sq = th_flat / rms_x w'(x)    (equal RMS effective spatial slope)
      max  : th_sq = th_flat / max_x w'(x)    (equal MAXIMUM effective spatial slope)
    """
    max_wp, rms_wp, _min_wp, xg, _wp = warp_slope()
    rng = np.random.default_rng(3000 + seed)
    th = truncated_freqs(rng, sigma)
    _xg, wg = warp()
    wgrid = np.interp(GRID, _xg, wg)

    flat, gflat = _rmse_batched(th, lambda x: x,
                                np.exp(-1j * np.outer(GRID, th)), rng)
    cell = {"flat": flat, "gross_flat": gflat,
            "theta_max_flat": float(np.max(np.abs(th))),
            "eff_slope_max_flat": float(np.max(np.abs(th)))}
    for name, scale in (("base", 1.0), ("rms", 1.0 / rms_wp), ("max", 1.0 / max_wp)):
        ths = scale * th
        sq, gsq = _rmse_batched(ths, lambda x, w=wg, g=_xg: np.interp(x, g, w),
                                np.exp(-1j * np.outer(wgrid, ths)), rng)
        cell[name] = sq
        cell["gross_" + name] = gsq
        cell["eff_slope_max_" + name] = float(np.max(np.abs(ths)) * max_wp)
        cell["eff_slope_rms_" + name] = float(np.std(ths) * rms_wp)
    cell["eff_slope_rms_flat"] = float(np.std(th))
    return cell


def warp_param(base, s):
    """The committed warp is (base, s) = (0.2, 1.0); this generalizes its steepness."""
    xg = np.linspace(A, B, 4000)
    dens = base + np.exp(-0.5 * ((xg - XF) / s) ** 2)
    cdf = np.cumsum(dens)
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return xg, A + (B - A) * cdf


def steepness_reductio(sigma=2.0, seeds=4):
    """Is 'same codebook == same aperture' a real RESOURCE CONSTRAINT?  Decisive test.

    Hold the codebook EXACTLY fixed -- so the manuscript's aperture is respected by
    construction, in its own terms -- and steepen the warp.  If "under a bandwidth
    aperture" were a genuine budget, the achievable fovea gain would be bounded.  It is
    not: measured at the fovea CENTRE the gain tracks max_x w'(x) essentially one-for-one,
    and max w' is a free modelling choice that costs nothing under this reading.  A
    resource definition under which the payoff is unbounded is not a resource definition.

    Two fovea windows are reported, because they answer different questions:
      wide   |x-5| < 1     -- the manuscript's window.  Saturates near 2.4x, but for a
                             geometric reason unrelated to the aperture: a narrow warp
                             leaves the EDGES of a fixed-width window coarse, and RMSE
                             over the window is dominated by its worst points.
      centre |x-5| < 0.15  -- the local resolution the warp actually buys.  Unbounded.
    """
    fams = [(0.2, 2.0), (0.2, 1.0), (0.2, 0.5), (0.05, 0.5), (0.02, 0.35), (0.01, 0.2)]
    wide = [("fovea", [(XF - 1, XF + 1)]), ("periphery", [(1.0, 3.0), (7.0, 9.0)])]
    narrow = [("fovea", [(XF - 0.15, XF + 0.15)]),
              ("periphery", [(1.0, 3.0), (7.0, 9.0)])]
    print("\n(5) IS 'same codebook == same aperture' a real constraint?  "
          "Steepen the warp, hold the codebook fixed:")
    print("      max w'   wide-window gain   centre gain   gain/max w'   "
          "squeezed periph. RMSE")
    rows = []
    global REGIONS
    saved = REGIONS
    try:
        for base, s in fams:
            xg, wg = warp_param(base, s)
            wp = np.gradient(wg, xg)
            wgrid = np.interp(GRID, xg, wg)
            acc = {}
            for tag, regs in (("wide", wide), ("centre", narrow)):
                REGIONS = regs
                ff, sf, sp, gr = [], [], [], []
                for seed in range(seeds):
                    rng = np.random.default_rng(4000 + seed)
                    th = truncated_freqs(rng, sigma)
                    flat, _ = _rmse_batched(th, lambda x: x,
                                            np.exp(-1j * np.outer(GRID, th)), rng)
                    sq, gs = _rmse_batched(
                        th, lambda x, w=wg, g=xg: np.interp(x, g, w),
                        np.exp(-1j * np.outer(wgrid, th)), rng)
                    ff.append(flat["fovea"])
                    sf.append(sq["fovea"])
                    sp.append(sq["periphery"])
                    gr.append(gs["periphery"])
                acc[tag] = (float(np.mean(np.array(ff) / np.array(sf))),
                            float(np.mean(sf)), float(np.mean(sp)), float(np.mean(gr)))
            rows.append(dict(base=base, width=s, max_wp=float(wp.max()),
                             fovea_gain_wide=acc["wide"][0],
                             fovea_gain_centre=acc["centre"][0],
                             centre_gain_over_max_wp=acc["centre"][0] / float(wp.max()),
                             squeezed_fovea_rmse_centre=acc["centre"][1],
                             squeezed_periphery=acc["wide"][2],
                             gross_periphery=acc["wide"][3]))
            print(f"      {wp.max():6.2f}   {acc['wide'][0]:14.2f}x   "
                  f"{acc['centre'][0]:9.2f}x   {acc['centre'][0]/wp.max():10.2f}   "
                  f"{acc['wide'][2]:18.4f}")
    finally:
        REGIONS = saved
    print("    -> the codebook never changes, yet the centre gain tracks max w' 1:1 up")
    print("       to ~10x.  So 'under the same aperture' bounds nothing; the 2.2-2.5x")
    print("       headline is a property of the chosen warp, not of the aperture.")
    return rows


def part4():
    max_wp, rms_wp, min_wp, _xg, _wp = warp_slope()
    print("\n(4) MATCHED-RESOURCE AUDIT of the aperture claim "
          f"({TRIALS4} trials x {SEEDS4} seeds, hard bound |theta|<={TRUNC:.0f}sigma).")
    # --- diagnostic: is sigma_cap in part 3 a cap at all? ---
    diag_frac, diag_ratio = [], []
    for s in range(SEEDS):
        r = np.random.default_rng(2000 + s)
        t = 1.0 * r.standard_normal(N)
        diag_frac.append(float(np.mean(np.abs(t) > 1.0)))
        diag_ratio.append(float(np.max(np.abs(t))))
    print(f"    part-3 'cap' diagnostic: frac(|theta|>sigma_cap) = "
          f"{np.mean(diag_frac):.4f}, max|theta|/sigma_cap = "
          f"{min(diag_ratio):.2f}-{max(diag_ratio):.2f}  -> sigma_cap is a bandwidth, "
          f"not a cap.")
    print(f"    warp slope w'(x): min {min_wp:.4f}, mean 1.0, rms {rms_wp:.4f}, "
          f"max {max_wp:.4f}  -> the warped arm's effective spatial frequency runs "
          f"{max_wp:.2f}x the flat arm's at the fovea.")

    rows = []
    for sigma in CAPS:
        cells = [run_matched(s, sigma) for s in range(SEEDS4)]
        ff = np.array([c["flat"]["fovea"] for c in cells])
        fp = np.array([c["flat"]["periphery"] for c in cells])
        row = dict(sigma=sigma, theta_max=TRUNC * sigma,
                   flat_fovea=float(ff.mean()), flat_fovea_std=float(ff.std()),
                   flat_periphery=float(fp.mean()), flat_periphery_std=float(fp.std()))
        print(f"    sigma={sigma:4.1f} (|theta|<={TRUNC*sigma:5.1f}):  "
              f"flat {ff.mean():.5f}/{fp.mean():.5f}")
        for name, label in (("base", "same base-freq law   (a)"),
                            ("rms", "same RMS eff. slope  (b)"),
                            ("max", "same MAX eff. slope  (c)")):
            sf = np.array([c[name]["fovea"] for c in cells])
            sp = np.array([c[name]["periphery"] for c in cells])
            gf = ff / sf          # per-seed gains, then summarize
            gp = fp / sp
            row[f"{name}_fovea"] = float(sf.mean())
            row[f"{name}_fovea_std"] = float(sf.std())
            row[f"{name}_periphery"] = float(sp.mean())
            row[f"{name}_periphery_std"] = float(sp.std())
            row[f"{name}_fovea_gain"] = float(gf.mean())
            row[f"{name}_fovea_gain_std"] = float(gf.std())
            row[f"{name}_fovea_gain_min"] = float(gf.min())
            row[f"{name}_fovea_gain_max"] = float(gf.max())
            row[f"{name}_periphery_gain"] = float(gp.mean())
            row[f"{name}_periphery_gain_std"] = float(gp.std())
            row[f"{name}_eff_slope_max"] = float(np.mean(
                [c["eff_slope_max_" + name] for c in cells]))
            row[f"{name}_eff_slope_rms"] = float(np.mean(
                [c["eff_slope_rms_" + name] for c in cells]))
            print(f"        {label}: squeezed {sf.mean():.5f}/{sp.mean():.5f}   "
                  f"fovea gain {gf.mean():.2f}x +/- {gf.std():.2f} "
                  f"[{gf.min():.2f}-{gf.max():.2f}]   periph {gp.mean():.2f}x   "
                  f"(max eff. slope {np.mean([c['eff_slope_max_'+name] for c in cells]):.2f} "
                  f"vs flat {np.mean([c['eff_slope_max_flat'] for c in cells]):.2f})")
        row["flat_eff_slope_max"] = float(np.mean(
            [c["eff_slope_max_flat"] for c in cells]))
        rows.append(row)
    for name, label in (("base", "(a) same base-frequency law"),
                        ("rms", "(b) same RMS effective slope"),
                        ("max", "(c) same MAX effective slope")):
        g = [r[f"{name}_fovea_gain"] for r in rows]
        print(f"    -> {label}: fovea gain {min(g):.2f}-{max(g):.2f}x across apertures")
    diag = dict(sigma_cap_is_not_a_cap=dict(
        frac_abs_theta_exceeds_sigma_cap=float(np.mean(diag_frac)),
        max_abs_theta_over_sigma_cap=[float(x) for x in diag_ratio]),
        warp_slope=dict(min=min_wp, mean=1.0, rms=rms_wp, max=max_wp),
        trials=TRIALS4, seeds=SEEDS4, trunc_sigmas=TRUNC)
    diag["steepness_reductio"] = steepness_reductio()
    return rows, diag


def main():
    res = [run(s) for s in range(SEEDS)]
    rows = []
    kinds = ("flat", "multiscale", "squeezed")
    print(f"decoding RMSE under channel noise (std {NOISE}/component), "
          f"{TRIALS} trials x {SEEDS} seeds:")
    print(f"  multi-scale bandwidths: {tuple(round(s, 4) for s in MS_SIGMAS)}  "
          f"(sum of squares {sum(s**2 for s in MS_SIGMAS):.2f} "
          f"= 3 x SIGMA^2 = {3*SIGMA**2:.2f}, matched budget)")
    means = {}
    for region in ("fovea", "periphery"):
        for kind in kinds:
            vals = [r[(region, kind)] for r in res]
            means[(region, kind)] = float(np.mean(vals))
            rows.append(dict(region=region, encoding=kind,
                             rmse_mean=float(np.mean(vals)),
                             rmse_std=float(np.std(vals))))
            print(f"  {region:>9} / {kind:>10}: RMSE = "
                  f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    gains = {}
    for region in ("fovea", "periphery"):
        for kind in ("multiscale", "squeezed"):
            gains[f"{region}_{kind}_gain_over_flat"] = (
                means[(region, "flat")] / means[(region, kind)])
    print("  gain over flat (>1 = better):")
    for k, v in gains.items():
        print(f"    {k:>34}: {v:.2f}x")
    # ---- PART 2: the uncapped control -- bandwidth is free, so flat wins ------
    print("\n(2) CONTROL: flat FPE with bandwidth FREE (nothing charges for sigma).")
    unc = []
    for sig in UNCAPPED:
        r = [run_uncapped(s, sig) for s in range(SEEDS)]
        fo = float(np.mean([x[0]["fovea"] for x in r]))
        pe = float(np.mean([x[0]["periphery"] for x in r]))
        gr = float(np.mean([x[1]["fovea"] for x in r]))
        unc.append(dict(sigma=sig, fovea=fo, periphery=pe, gross_fovea=gr))
        print(f"    sigma={sig:5.1f}: fovea {fo:.5f}  periphery {pe:.5f}  "
              f"gross-error rate {gr:.3f}")
    print("    -> widening sigma buys resolution for free; at sigma=5 the FLAT arm")
    print("       already beats squeezed-at-sigma=2 in the fovea AND the periphery.")

    # ---- PART 3: the aperture -- the regime where the budget actually binds ---
    print("\n(3) APERTURE: flat vs squeezed at a hard bandwidth cap (same codebook).")
    cap_rows = []
    for cap in CAPS:
        r = [run_capped(s, cap) for s in range(SEEDS)]
        ff = float(np.mean([x[0]["fovea"] for x in r]))
        fp = float(np.mean([x[0]["periphery"] for x in r]))
        sf = float(np.mean([x[1]["fovea"] for x in r]))
        sp = float(np.mean([x[1]["periphery"] for x in r]))
        gf = float(np.mean([x[2]["fovea"] for x in r]))
        gs = float(np.mean([x[3]["fovea"] for x in r]))
        cap_rows.append(dict(sigma_cap=cap, flat_fovea=ff, flat_periphery=fp,
                             squeezed_fovea=sf, squeezed_periphery=sp,
                             fovea_gain=ff / sf, periphery_gain=fp / sp,
                             gross_flat=gf, gross_squeezed=gs))
        print(f"    cap sigma<={cap:4.1f}: flat {ff:.5f}/{fp:.5f}  "
              f"squeezed {sf:.5f}/{sp:.5f}  "
              f"gain {ff/sf:.2f}x fovea / {fp/sp:.2f}x periphery")
    gvals = [r["fovea_gain"] for r in cap_rows]
    print(f"    -> fovea gain holds at every aperture across a "
          f"{CAPS[-1]/CAPS[0]:.0f}x range: {min(gvals):.2f}-{max(gvals):.2f}x, "
          f"always paid for in the periphery")

    # ---- PART 4: matched-resource audit (added after external review) --------
    matched_rows, matched_diag = part4()

    out = dict(params=dict(N=N, sigma=SIGMA, fovea=XF, noise=NOISE,
                           trials=TRIALS, seeds=SEEDS,
                           multiscale_sigmas=[float(s) for s in MS_SIGMAS],
                           caps=list(CAPS), uncapped=list(UNCAPPED)),
               rows=rows, gains_over_flat=gains,
               sweep_uncapped=unc, sweep_capped=cap_rows,
               matched_resource_diagnostics=matched_diag,
               sweep_matched_resource=matched_rows)
    with open("cwf_fpe_fovea_task_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_fovea_task_results.json")


if __name__ == "__main__":
    main()
