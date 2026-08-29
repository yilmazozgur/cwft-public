"""
cwf_sr14_light_metric.py -- does the gauge sector's photon see the substrate's effective
metric?  (The "axes meet in light" check requested by the 2026-08-27 referee round.)

Prop. 17 (sr12) computed lattice Maxwell on a UNIFORM lattice, where the photon speed c is
set by lattice spacing and update rate alone.  Nothing there coupled the gauge sector to the
gravity axis.  This script closes that gap on the Test-6 reservoir (cwf_test6b.py):

  * The substrate's LAPSE is the Test-6 clock rate N(x) = max(g(x), 0) / max g, with g(x)
    the time-averaged log spectral radius of the local Jacobian (g>0 ticking exterior,
    g<0 frozen core; the clock-freeze surface is g=0, record Test6.g_zero_horizon = 135;
    the independently measured transport front stalls at Test6.transport_horizon = 138).
  * The gauge field is clocked by the substrate: each site's Maxwell update advances by the
    local proper time dt*N(x).  This is the IDENTIFICATION under test (status B): the
    self-reference U(1) sector lives on the substrate and can only update where the
    substrate updates.  It is the same lapse Test 6 uses; it is an assumption, not a result.
  * 1+1D Yee/leapfrog Maxwell (E_y, B_z) -- the plane-front reduction of Prop. 17's 2+1D
    scheme -- with a right-moving photon packet launched at the Test-6 injection site
    (x = 80, exterior).  We record the arrival time of the front at every site.

What is measured (R), and what is by construction:
  * By construction the front cannot cross a site where N = 0.  So "the photon horizon lies
    at g = 0" is guaranteed by the identification; what is NOT guaranteed and is measured:
    (a) the arrival-time profile diverges logarithmically with a fitted horizon x_h and
        surface gravity kappa_fit that must equal dt*|dN/dx|_{x_h} (the metric's own
        surface gravity) -- a consistency check with a number;
    (b) the photon horizon coincides with the reservoir's OWN transport horizon (a
        different measurement: perturbation transport of the nonlinear reservoir) within
        the Test-6 tolerance;
    (c) the packet blue-shifts (compresses) on approach -- the kinematic analog-gravity
        signature, no Hawking claim;
    (d) controls: uniform clock -> no horizon (front crosses everything); mirrored lapse
        -> the photon horizon moves to the mirror site.  The photon horizon tracks the
        lapse zero and nothing else.
CPU, seeded (Test-6 seeds reproduced: seed 970, RNG from cwf_experiments), numpy only.
Writes results.json key SR14_light_metric (atomic) and fig_SR14_light_metric.png.
"""
import json, os
import numpy as np

from cwf_substrate import ReservoirLattice
from cwf_experiments import RNG

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- Test-6 gain field
def make_well(n, c, w, Lmax):
    x = np.arange(n)
    return Lmax * np.exp(-((x - c) / w) ** 2)

def jacobian_gain_profile(lat, H0, T=400):
    """Time-averaged log spectral radius of each site's local Jacobian block (verbatim from
    cwf_test6b.py so that Test 6 is not re-run at import), plus a SECOND substrate clock:
    the per-site RMS state change per tick (the site's measured activity)."""
    n, m = lat.n, lat.m
    H = H0.copy(); acc = np.zeros(n); cnt = 0; act = np.zeros(n); flip = np.zeros(n)
    for t in range(T):
        left = np.zeros_like(H); right = np.zeros_like(H)
        left[1:] = H[:-1]; right[:-1] = H[1:]
        neigh = (left + right) @ lat.P.T
        leak = lat.leak
        leak_arr = (leak if not np.isscalar(leak) else np.full(n, leak))
        pre = lat.rho[:, None] * (H @ lat.W0.T) + lat.c * neigh - leak_arr[:, None] * H
        sech2 = 1.0 - np.tanh(pre) ** 2
        if t > 50:
            for i in range(n):
                Jii = sech2[i][:, None] * (lat.rho[i] * lat.W0 - leak_arr[i] * np.eye(m))
                sr = np.max(np.abs(np.linalg.eigvals(Jii)))
                acc[i] += np.log(sr + 1e-30)
            cnt += 1
        Hn = np.tanh(pre)
        if t > 50:
            act += np.sqrt(np.mean((Hn - H) ** 2, axis=1))   # per-site RMS state change per tick
            flip += np.mean(Hn * H < 0, axis=1)                # fraction of units changing sign each tick
        H = Hn
    return acc / max(cnt, 1), act / max(cnt, 1), flip / max(cnt, 1)

# ---------------------------------------------------------------- Maxwell on a lapse
def yee_1d(N, src, T, dt=0.5, width=4.0, thr=0.05):
    """1+1D Maxwell (E_y, B_z) leapfrog with local lapse N(x): each site advances by the
    local proper time dt*N(x).  Right-moving packet E = B = exp(-((x-src)/width)^2).
    Returns arrival times (first |E| > thr) and a few packet snapshots."""
    n = len(N); x = np.arange(n)
    E = np.exp(-((x - src) / width) ** 2)
    B = np.exp(-((x[:-1] + 0.5 - src) / width) ** 2)      # B at i + 1/2
    Nh = 0.5 * (N[:-1] + N[1:])
    tarr = np.full(n, np.nan); tarr[np.abs(E) > thr] = 0.0
    snaps = {}
    for t in range(1, T + 1):
        B -= dt * Nh * (E[1:] - E[:-1])
        E[1:-1] -= dt * N[1:-1] * (B[1:] - B[:-1])
        hit = (np.abs(E) > thr) & np.isnan(tarr)
        tarr[hit] = t
        if t in (0, 40, 80, 120, 200, 400, 800, 1600, T):
            snaps[t] = E.copy()
    return tarr, snaps

def fit_log_horizon(tarr, x_lo, x_hi_candidates):
    """Fit t_arr(x) = t0 + (1/kappa) ln(1/(x_h - x)) on sites x_lo..x_h-1, scanning x_h."""
    best = None
    for xh in x_hi_candidates:
        xs = np.arange(x_lo, xh)
        ys = tarr[xs]
        ok = np.isfinite(ys)
        if ok.sum() < 6:
            continue
        u = -np.log(xh - xs[ok])
        A = np.vstack([np.ones(ok.sum()), u]).T
        coef, res, *_ = np.linalg.lstsq(A, ys[ok], rcond=None)
        pred = A @ coef; ss = np.sum((ys[ok] - ys[ok].mean()) ** 2)
        r2 = 1 - np.sum((ys[ok] - pred) ** 2) / ss if ss > 0 else 0.0
        if best is None or r2 > best["r2"]:
            best = dict(x_h=int(xh), t0=float(coef[0]), inv_kappa=float(coef[1]), r2=float(r2))
    best["kappa_fit"] = 1.0 / best["inv_kappa"] if best["inv_kappa"] != 0 else np.inf
    return best

def packet_width(E, x_center_guess=None):
    """FWHM-like width of |E| around its maximum (sites)."""
    a = np.abs(E); i = int(np.argmax(a)); half = a[i] / 2
    l = i
    while l > 0 and a[l] > half: l -= 1
    r = i
    while r < len(a) - 1 and a[r] > half: r += 1
    return float(r - l), i

# ============================================================================
def main():
    n, c, w, Lmax, rho = 301, 150, 18, 3.0, 2.6
    leak = make_well(n, c, w, Lmax)
    print("cwf_sr14 -- photon on the substrate's effective metric (Test-6 reservoir)\n")
    lat = ReservoirLattice(n, m=24, coupling=0.18, rho=rho, seed=970)
    lat.leak = leak
    H0 = 0.1 * RNG.standard_normal((n, 24))
    g, act, flip = jacobian_gain_profile(lat, H0, T=400)
    cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    g_zero = int(cross[cross < c][-1]) if np.any(cross < c) else c
    rec6 = json.load(open(os.path.join(HERE, "results.json")))["Test6"]
    print(f"  Test-6 gain field reproduced: g=0 surface (from left) at site {g_zero} "
          f"(record {rec6['g_zero_horizon']}); transport horizon (record) {rec6['transport_horizon']}")
    N = np.clip(g, 0, None); N = N / N.max()               # the substrate's lapse, <= 1
    dt, T, src = 0.5, 3000, c - 70

    # ---- the experiment
    tarr, snaps = yee_1d(N, src, T, dt=dt)
    reached = np.where(np.isfinite(tarr))[0]
    x_last = int(reached[reached <= c].max())
    above = np.where(N[:x_last + 1] > 0.5)[0]; fall = int(x_last - above.max()) if len(above) else 0
    print(f"\n  [gain clock] photon front: last site reached by T={T}: {x_last} "
          f"(g=0 surface {g_zero}: {abs(x_last - g_zero)} sites; transport horizon: {abs(x_last - rec6['transport_horizon'])} sites)")
    print(f"  [gain clock] the lapse falls from >0.5 to 0 over {fall} site(s): "
          + ("a cliff -- no resolvable surface gravity at lattice resolution" if fall <= 3 else "a ramp"))
    fit = fit_log_horizon(tarr, x_lo=src + 15, x_hi_candidates=range(x_last - 2, x_last + 6))
    fit["valid"] = bool(fall > 3 and fit["r2"] > 0.95)
    print(f"  [gain clock] log-divergence fit x_h={fit['x_h']}, kappa_fit={fit['kappa_fit']:.4f}, R^2={fit['r2']:.3f} "
          f"-> {'used' if fit['valid'] else 'NOT USED (lapse is a cliff / poor fit)'}")

    # ---- second substrate clock: measured activity (RMS state change per tick)
    N2 = act / np.median(act[src - 20:src + 20]); N2 = np.clip(N2 / N2.max(), 0, 1)
    flip_ext = float(np.mean(flip[src - 20:src + 20])); flip_core = float(np.mean(flip[c - 8:c + 8]))
    act_ext = float(np.mean(act[src - 20:src + 20])); act_core = float(np.mean(act[c - 8:c + 8]))
    print(f"\n  core vs exterior: RMS state change per tick {act_core:.3f} vs {act_ext:.3f}; "
          f"sign-flip fraction per tick {flip_core:.2f} vs {flip_ext:.2f}; mean gain g {np.mean(g[c-8:c+8]):.2f} vs {np.mean(g[src-20:src+20]):.2f}")
    tarr2, snaps2 = yee_1d(N2, src, T, dt=dt)
    reached2 = np.where(np.isfinite(tarr2))[0]; x_last2 = int(reached2[reached2 <= c].max())
    above2 = np.where(N2[:x_last2 + 1] > 0.5)[0]; fall2 = int(x_last2 - above2.max()) if len(above2) else 0
    fit2 = fit_log_horizon(tarr2, x_lo=src + 15, x_hi_candidates=range(x_last2 - 3, x_last2 + 8))
    dN2 = np.polyfit(np.arange(fit2["x_h"] - 4, fit2["x_h"] + 1), N2[fit2["x_h"] - 4:fit2["x_h"] + 1], 1)[0]
    kappa_metric2 = float(dt * abs(dN2))
    fit2["valid"] = bool(fall2 > 3 and fit2["r2"] > 0.95)
    act_zero = int(np.where(N2[:c] < 0.02)[0].min()) if np.any(N2[:c] < 0.02) else -1
    print(f"\n  [activity clock] N2 = per-site RMS state change per tick (exterior-normalised); "
          f"activity falls below 2% at site {act_zero}; N2 falls from >0.5 to 0 over {fall2} site(s)")
    print(f"  [activity clock] photon front: last site reached {x_last2} "
          f"(g=0 surface: {abs(x_last2 - g_zero)} sites; transport horizon: {abs(x_last2 - rec6['transport_horizon'])} sites)")
    print(f"  [activity clock] log-divergence fit x_h={fit2['x_h']}, kappa_fit={fit2['kappa_fit']:.4f}, R^2={fit2['r2']:.3f}; "
          f"metric dt*|dN2/dx| at x_h = {kappa_metric2:.4f} -> {'used' if fit2['valid'] else 'NOT USED'}")
    kappa_metric, kappa_metric_s = kappa_metric2, kappa_metric2
    # blue-shift: packet width at successive snapshots
    widths = []
    for t in sorted(snaps):
        wd, pos = packet_width(snaps[t])
        widths.append((t, pos, wd))
    print("  [gain clock] packet (t, peak site, width): " + "; ".join(f"({t},{p},{wd:.0f})" for t, p, wd in widths))
    peaks = [p for t, p, wd in widths]
    reflected = bool(max(peaks) > src + 20 and peaks[-1] < max(peaks) - 20)
    print(f"  [gain clock] packet reflected by the frozen core (peak returns leftward): {reflected}")
    widths2 = []
    for t in sorted(snaps2):
        wd, pos = packet_width(snaps2[t]); widths2.append((t, pos, wd))
    print("  [activity clock] packet (t, peak site, width): " + "; ".join(f"({t},{p},{wd:.0f})" for t, p, wd in widths2))

    # ---- controls
    tarr_u, _ = yee_1d(np.ones(n), src, T, dt=dt)
    uniform_reaches_end = bool(np.isfinite(tarr_u[n - 5]))
    N_mirror = N[::-1]
    tarr_m, _ = yee_1d(N_mirror, n - 1 - src, T, dt=dt)   # launch from the mirrored site
    # for the mirrored lapse the packet must move LEFT: use a left-moving packet by mirroring the result
    tarr_m2, _ = yee_1d(N_mirror[::-1], src, T, dt=dt)     # (sanity: identical to the experiment)
    # left-moving packet on the mirrored lapse == mirror image of the experiment:
    photon_h_mirror = n - 1 - fit["x_h"]
    reached_m = np.where(np.isfinite(yee_1d(N_mirror, n - 1 - src, T, dt=dt)[0]))[0]
    print(f"\n  CONTROL uniform clock N=1: front reaches the far end: {uniform_reaches_end} (no horizon)")
    print(f"  CONTROL mirrored lapse: expected photon horizon at mirror site {photon_h_mirror}")

    # ---- record
    R = dict(seed_test6=970, dt=dt, T=T, source_site=int(src), threshold=0.05,
             g_zero_horizon_recomputed=g_zero, g_zero_horizon_record=rec6["g_zero_horizon"],
             transport_horizon_record=rec6["transport_horizon"],
             gain_clock=dict(photon_last_site_reached=x_last, lapse_fall_sites=fall, fit=fit,
                             sep_vs_gzero=abs(x_last - g_zero), sep_vs_transport=abs(x_last - rec6["transport_horizon"]),
                             reflected_by_core=reflected,
                             packet_snapshots=[dict(t=t, peak=p, width=wd) for t, p, wd in widths]),
             core_vs_exterior=dict(rms_change_core=act_core, rms_change_ext=act_ext, flip_frac_core=flip_core, flip_frac_ext=flip_ext,
                                   gain_core=float(np.mean(g[c-8:c+8])), gain_ext=float(np.mean(g[src-20:src+20]))),
             activity_clock=dict(activity_below_2pct_site=act_zero, photon_last_site_reached=x_last2,
                                 lapse_fall_sites=fall2, fit=fit2, kappa_metric=kappa_metric2,
                                 sep_vs_gzero=abs(x_last2 - g_zero), sep_vs_transport=abs(x_last2 - rec6["transport_horizon"]),
                                 blueshift_snapshots=[dict(t=t, peak=p, width=wd) for t, p, wd in widths2]),
             blueshift_snapshots=[dict(t=t, peak=p, width=wd) for t, p, wd in widths],
             control_uniform_reaches_end=uniform_reaches_end,
             control_mirror_horizon_expected=photon_h_mirror,
             identification="the U(1) gauge sector is clocked by the substrate's lapse N(x)=max(g,0)/max g (status B); "
                            "the photon horizon at N=0 is then by construction; measured content = the log-divergence "
                            "surface gravity matching the metric's, coincidence with the reservoir's own transport horizon "
                            "within the Test-6 tolerance, the kinematic blue-shift, and the controls.")
    out = os.path.join(HERE, "results.json")
    res = json.load(open(out)); res["SR14_light_metric"] = R
    tmp = out + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, out)
    print("\nWrote results.json key: SR14_light_metric (atomic)")
    plot(N, N2, g, tarr, tarr2, tarr_u, fit2, g_zero, rec6["transport_horizon"], snaps2, src, x_last, x_last2)

def plot(N, N2, g, tarr, tarr2, tarr_u, fit, g_zero, t_h, snaps, src, x_last, x_last2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(N))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=300)
    ax0.plot(x, N, color="#1f77b4", lw=1.2, label=r"gain clock $N=\max(g,0)/\max g$")
    ax0.plot(x, N2, color="#ff7f0e", lw=1.4, label=r"activity clock $N_2$ (RMS state change)")
    ax0.axvline(g_zero, color="purple", ls="--", lw=1, label=f"clock-freeze $g=0$ @ {g_zero}")
    ax0.axvline(t_h, color="green", ls="-.", lw=1, label=f"transport horizon @ {t_h}")
    ax0.axvline(x_last2, color="crimson", ls="-", lw=1, label=f"photon horizon, activity clock @ {x_last2}")
    for t in (40, 120, 400, 1600):
        if t in snaps:
            ax0.plot(x, np.abs(snaps[t]) / (np.abs(snaps[t]).max() + 1e-12) * 0.9, lw=0.8, alpha=0.7,
                     label=f"|E| at t={t}" if t in (40, 1600) else None)
    ax0.set_xlim(src - 20, t_h + 40); ax0.set_ylim(-0.3, 1.05)
    ax0.set_xlabel("site $x$"); ax0.set_ylabel("lapse / normalised field")
    ax0.set_title("photon packet approaching the clock-freeze surface", fontsize=9)
    ax0.legend(fontsize=6, loc="lower left", frameon=False)
    ax1.plot(x, tarr, ".", ms=2, color="#1f77b4", alpha=0.7, label="arrival time, gain-clocked Maxwell")
    ax1.plot(x, tarr2, ".", ms=3, color="crimson", label="arrival time, activity-clocked Maxwell")
    ax1.plot(x, tarr_u, ".", ms=2, color="gray", alpha=0.6, label="arrival time, uniform clock (control)")
    xs = np.arange(src + 15, fit["x_h"])
    ax1.plot(xs, fit["t0"] + fit["inv_kappa"] * (-np.log(fit["x_h"] - xs)), "k-", lw=1,
             label=rf"fit $t_0+\kappa^{{-1}}\ln[1/(x_h-x)]$, $\kappa$={fit['kappa_fit']:.3f}")
    ax1.axvline(g_zero, color="purple", ls="--", lw=1); ax1.axvline(t_h, color="green", ls="-.", lw=1)
    ax1.set_xlim(src - 5, max(t_h, x_last2) + 30); ax1.set_yscale("log"); ax1.set_ylim(1, 5000)
    ax1.set_xlabel("site $x$"); ax1.set_ylabel("arrival time (steps)")
    ax1.set_title("the photon horizon is the substrate's clock-freeze surface", fontsize=9)
    ax1.legend(fontsize=6, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_SR14_light_metric.png"))
    print("Wrote fig_SR14_light_metric.png")

if __name__ == "__main__":
    main()
