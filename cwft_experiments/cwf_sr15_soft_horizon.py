"""
cwf_sr15_soft_horizon.py -- a SOFT clock-freeze reservoir: can the photon horizon's surface
gravity be resolved?  (Follow-up to cwf_sr14_light_metric.py, whose Test-6 lapse was a cliff.)

Same construction as sr14 -- the gauge field clocked by the substrate's information-production
lapse N(x) = max(g,0)/max g, g the local Jacobian gain -- on a FAMILY of damping wells of
increasing half-width w (w = 18 reproduces Test 6; then 40, 70, 100).  A wider well should make
the lapse vanish over many sites, so that the photon's arrival time diverges logarithmically
and its fitted surface gravity kappa_fit can be compared with the metric's own dt*|dN/dx|.
For every well we also RECOMPUTE the reservoir's transport horizon (perturbation reach, 8
seeds, as in Test 6), so the photon-vs-transport coincidence is a fresh measurement, not the
Test-6 record re-used.

Reported per well (R): the width over which N falls from half its exterior value to zero;
the g=0 site; the transport horizon; the photon horizon (last site reached, and the fitted
x_h); kappa_fit vs kappa_metric with the fit's R^2; packet width on approach (blue-shift);
whether the packet is reflected.  The identification (gauge sector runs on the substrate's
proper time) is the same status-B assumption as in sr14.  Seeded, CPU (minutes).
Writes results.json key SR15_soft_horizon (atomic) and fig_SR15_soft_horizon.png.
"""
import json, os, time
import numpy as np

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, RNG
from cwf_sr14_light_metric import make_well, jacobian_gain_profile, yee_1d, packet_width

HERE = os.path.dirname(os.path.abspath(__file__))
THETA = 1e-4

def transport_reach(n, leak, c, inject_off=-70, m=24, coupling=0.18, rho=2.6, T=1500, seeds=8):
    """Verbatim from cwf_test6b.py (kept here so that Test 6 is not re-run at import)."""
    inj = c + inject_off
    reach = np.zeros(n)
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=970 + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)
        reach += (D > THETA).any(axis=0)
    return reach / seeds, inj

def fit_window_horizon(tarr, x_lo, cands):
    """t_arr(x) = t0 + (1/kappa) ln(1/(x_h - x)) fitted on x_lo..x_h-1, scanning x_h."""
    best = None
    for xh in cands:
        xs = np.arange(x_lo, xh); ys = tarr[xs]; ok = np.isfinite(ys)
        if ok.sum() < 5:
            continue
        u = -np.log(xh - xs[ok]); A = np.vstack([np.ones(ok.sum()), u]).T
        coef = np.linalg.lstsq(A, ys[ok], rcond=None)[0]; pred = A @ coef
        ss = np.sum((ys[ok] - ys[ok].mean()) ** 2)
        r2 = 1 - np.sum((ys[ok] - pred) ** 2) / ss if ss > 0 else 0.0
        if best is None or r2 > best["r2"]:
            best = dict(x_h=int(xh), t0=float(coef[0]), inv_kappa=float(coef[1]), r2=float(r2), n_pts=int(ok.sum()))
    if best is None:
        return dict(x_h=int(cands[0]), t0=float("nan"), inv_kappa=float("nan"), r2=0.0, n_pts=0, kappa_fit=float("nan"))
    best["kappa_fit"] = float(1.0 / best["inv_kappa"]) if best["inv_kappa"] else float("inf")
    return best

def main():
    t0 = time.time()
    n, c, Lmax, rho = 301, 150, 3.0, 2.6
    src, dt, T = c - 70, 0.5, 6000
    widths = [18, 40, 70, 100]
    R = dict(seed=970, dt=dt, T=T, source_site=int(src), wells={})
    curves = {}
    print("cwf_sr15 -- soft clock-freeze wells: photon horizon, transport horizon, surface gravity\n")
    for w in widths:
        leak = make_well(n, c, w, Lmax)
        lat = ReservoirLattice(n, m=24, coupling=0.18, rho=rho, seed=970); lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, 24))
        g, act, flip = jacobian_gain_profile(lat, H0, T=400)
        cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        g_zero = int(cross[cross < c][-1]) if np.any(cross < c) else c
        reach, inj = transport_reach(n, leak, c, rho=rho)
        reached = np.where(reach >= 0.5)[0]
        t_h = int(reached[reached <= c].max()) if np.any(reached <= c) else inj
        N = np.clip(g, 0, None); N = N / N.max()
        N_ext = float(np.median(N[src - 20:src + 20]))
        # lapse fall: from the last site (left of g_zero) where N >= 0.5*N_ext to g_zero
        above = np.where(N[:g_zero] >= 0.5 * N_ext)[0]
        x_half = int(above.max()) if len(above) else g_zero
        fall = g_zero - x_half
        tarr, snaps = yee_1d(N, src, T, dt=dt)
        rr = np.where(np.isfinite(tarr))[0]; x_last = int(rr[rr <= c].max())
        x_lo = min(max(x_half, src + 10), x_last - 6)          # at least ~6 points even on a cliff
        fit = fit_window_horizon(tarr, x_lo=x_lo, cands=range(x_last, x_last + 9))
        win = np.arange(x_lo, fit["x_h"])
        slope = np.polyfit(win, N[win], 1)[0] if len(win) >= 3 else float("nan")
        kappa_metric = float(dt * abs(slope))
        ratio = fit["kappa_fit"] / kappa_metric if kappa_metric > 0 else float("nan")
        # eikonal test: predicted arrival from the lapse alone, t_pred(x) = t(x0) + sum_{x0<=x'<x} 1/(dt*N_half(x'))
        Nh = 0.5 * (N[:-1] + N[1:]); x0 = src + 12
        pred = np.full(n, np.nan); acc = tarr[x0] if np.isfinite(tarr[x0]) else 0.0
        for xx in range(x0, n - 1):
            pred[xx] = acc
            if Nh[xx] <= 1e-9: break
            acc += 1.0 / (dt * Nh[xx])
        sel = np.arange(x0, x_last + 1); sel = sel[np.isfinite(pred[sel]) & np.isfinite(tarr[sel]) & (N[sel] > 0.1)]
        eik_rel_rms = float(np.sqrt(np.mean(((tarr[sel] - pred[sel]) / np.maximum(tarr[sel], 1)) ** 2))) if len(sel) else float("nan")
        first_zero = int(np.where(N[src:] <= 1e-9)[0].min() + src) if np.any(N[src:] <= 1e-9) else -1
        launch_frozen = bool(N[src] <= 1e-9)
        # convention check: the book's acoustic metric treats f_c = max(g,0)/g_inf as the SQUARED lapse (clock rate sqrt f_c)
        Ns = np.sqrt(N); tarr_s, _ = yee_1d(Ns, src, T, dt=dt)
        rr_s = np.where(np.isfinite(tarr_s))[0]; x_last_s = int(rr_s[rr_s <= c].max()) if len(rr_s) else -1
        Nhs = 0.5 * (Ns[:-1] + Ns[1:]); pred_s = np.full(n, np.nan); acc_s = tarr_s[x0] if np.isfinite(tarr_s[x0]) else 0.0
        for xx in range(x0, n - 1):
            pred_s[xx] = acc_s
            if Nhs[xx] <= 1e-9: break
            acc_s += 1.0 / (dt * Nhs[xx])
        sel_s = np.arange(x0, max(x_last_s, x0) + 1); sel_s = sel_s[np.isfinite(pred_s[sel_s]) & np.isfinite(tarr_s[sel_s]) & (Ns[sel_s] > 0.3)]
        eik_s = float(np.sqrt(np.mean(((tarr_s[sel_s] - pred_s[sel_s]) / np.maximum(tarr_s[sel_s], 1)) ** 2))) if len(sel_s) else float("nan")
        print(f"          sqrt-lapse convention (clock rate = sqrt f_c): stall site {x_last_s}; eikonal rel. RMS {eik_s:.3f} over {len(sel_s)} sites")
        widths_t = []
        for t in sorted(snaps):
            wd, pos = packet_width(snaps[t]); widths_t.append((t, pos, wd))
        peaks = [p for t, p, wd in widths_t]
        reflected = bool(max(peaks) > src + 20 and peaks[-1] < max(peaks) - 20)
        approach = [(t, p, wd) for t, p, wd in widths_t if p >= src + 20 and p < fit["x_h"]]
        w_far = widths_t[1][2] if len(widths_t) > 1 else float("nan")
        w_near = approach[-1][2] if approach else float("nan")
        print(f"  w={w:3d}: g=0 @ {g_zero}; transport horizon @ {t_h}; lapse falls over {fall} sites (from {x_half}); "
              f"photon last site {x_last}, fit x_h={fit['x_h']} (R^2={fit['r2']:.3f}, {fit['n_pts']} pts)")
        print(f"          kappa_fit={fit['kappa_fit']:.4f}  vs  kappa_metric=dt|N'|={kappa_metric:.4f}  (ratio {ratio:.2f}); "
              f"packet width far {w_far:.0f} -> near {w_near:.0f}; reflected {reflected}   [{time.time()-t0:.0f}s]")
        print(f"          eikonal law: first lapse zero beyond the source @ {first_zero} (launch site frozen: {launch_frozen}); "
              f"relative RMS deviation of arrival from sum 1/(dt N) over {len(sel)} sites with N>0.1: {eik_rel_rms:.3f}")
        R["wells"][f"w={w}"] = dict(w=w, g_zero=g_zero, transport_horizon=t_h, lapse_fall_sites=int(fall), x_half=x_half,
                                    N_exterior=N_ext, photon_last_site=x_last, fit=fit, kappa_metric=kappa_metric,
                                    kappa_ratio=float(ratio), sep_photon_fit_vs_gzero=abs(fit["x_h"] - g_zero),
                                    sep_photon_last_vs_gzero=abs(x_last - g_zero),
                                    sep_photon_vs_transport=abs(x_last - t_h), sep_gzero_vs_transport=abs(g_zero - t_h),
                                    packet_width_far=float(w_far), packet_width_near=float(w_near), reflected=reflected,
                                    first_lapse_zero=first_zero, launch_site_frozen=launch_frozen,
                                    eikonal_rel_rms=eik_rel_rms, eikonal_sites=int(len(sel)),
                                    sqrt_lapse=dict(stall_site=x_last_s, eikonal_rel_rms=eik_s, eikonal_sites=int(len(sel_s))),
                                    snapshots=[dict(t=t, peak=p, width=wd) for t, p, wd in widths_t])
        curves[w] = dict(N=N, g=g, tarr=tarr, fit=fit, g_zero=g_zero, t_h=t_h, x_half=x_half, reach=reach)
    # verdict
    soft = [k for k, v in R["wells"].items() if v["lapse_fall_sites"] >= 5]
    good = [k for k in soft if R["wells"][k]["fit"]["r2"] >= 0.95]
    R["verdict"] = dict(
        wells_with_soft_lapse=soft, wells_with_log_divergence_fit=good,
        kappa_ratios={k: R["wells"][k]["kappa_ratio"] for k in soft},
        photon_at_gzero_all={k: R["wells"][k]["sep_photon_last_vs_gzero"] for k in R["wells"]},
        photon_vs_transport_all={k: R["wells"][k]["sep_photon_vs_transport"] for k in R["wells"]},
        eikonal_rel_rms={k: R["wells"][k]["eikonal_rel_rms"] for k in R["wells"]},
        first_zero_vs_transport={k: abs(R["wells"][k]["first_lapse_zero"] - R["wells"][k]["transport_horizon"]) for k in R["wells"]},
        launch_frozen={k: R["wells"][k]["launch_site_frozen"] for k in R["wells"]},
        note="No well produced a linearly vanishing lapse: the reservoir's gain field flickers rather than ramps, so the "
             "log-divergence kappa fits (3 free parameters on 6-7 points) are NOT evidence and are recorded only. The testable "
             "content is the eikonal law (arrival time = sum of 1/(dt N) along the path) and the stall site = the first lapse "
             "zero, compared with the freshly measured transport horizon.")
    print("\nVERDICT:", json.dumps(R["verdict"], indent=1)[:900])
    out = os.path.join(HERE, "results.json")
    res = json.load(open(out)); res["SR15_soft_horizon"] = R
    tmp = out + ".tmp"; json.dump(res, open(tmp, "w"), indent=2); os.replace(tmp, out)
    print("Wrote results.json key: SR15_soft_horizon (atomic)")
    plot(curves, widths, src, dt)

def plot(curves, widths, src, dt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=300)
    cols = {18: "#7f7f7f", 40: "#1f77b4", 70: "#2ca02c", 100: "#d62728"}
    ax = axes[0]
    for w in widths:
        cv = curves[w]; x = np.arange(len(cv["N"]))
        ax.plot(x, cv["N"], color=cols[w], lw=1.2, label=f"w={w}: g=0 @ {cv['g_zero']}, transport @ {cv['t_h']}")
    ax.set_xlim(src, 160); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("site $x$"); ax.set_ylabel("lapse $N(x)$")
    ax.set_title("substrate lapse for wells of increasing width", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[1]
    for w in widths:
        cv = curves[w]; x = np.arange(len(cv["N"]))
        ax.plot(x, cv["tarr"], ".", ms=2.5, color=cols[w], label=f"w={w}")
        f = cv["fit"]; xs = np.arange(min(max(cv["x_half"], src + 10), f["x_h"] - 6), f["x_h"])
        if len(xs) > 2:
            ax.plot(xs, f["t0"] + f["inv_kappa"] * (-np.log(f["x_h"] - xs)), "-", color=cols[w], lw=0.8, alpha=0.8)
        ax.axvline(cv["g_zero"], color=cols[w], ls="--", lw=0.6)
    ax.set_yscale("log"); ax.set_xlim(src, 160); ax.set_ylim(1, 8000)
    ax.set_xlabel("site $x$"); ax.set_ylabel("photon arrival time (steps)")
    ax.set_title("arrival times with log-divergence fits (dashed: $g=0$)", fontsize=9); ax.legend(fontsize=6, frameon=False)
    ax = axes[2]
    for w in widths:
        cv = curves[w]; f = cv["fit"]
        km = None
    ws = widths; kf = [curves[w]["fit"]["kappa_fit"] for w in ws]
    xs_ = np.arange(len(ws))
    km = []
    for w in ws:
        cv = curves[w]; win = np.arange(min(max(cv["x_half"], src + 10), cv["fit"]["x_h"] - 6), cv["fit"]["x_h"])
        km.append(dt * abs(np.polyfit(win, cv["N"][win], 1)[0]) if len(win) >= 3 else np.nan)
    ax.bar(xs_ - 0.18, kf, 0.36, color="#1f77b4", label=r"$\kappa_{\rm fit}$ (photon arrival)")
    ax.bar(xs_ + 0.18, km, 0.36, color="#ff7f0e", label=r"$\kappa_{\rm metric}=\mathrm{d}t\,|N'|$")
    ax.set_xticks(xs_); ax.set_xticklabels([f"w={w}\nR$^2$={curves[w]['fit']['r2']:.2f}" for w in ws], fontsize=7)
    ax.set_ylabel(r"surface gravity (steps$^{-1}$)"); ax.set_title("fitted vs metric surface gravity", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Soft clock-freeze wells: the photon horizon, the transport horizon, and the surface gravity", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_SR15_soft_horizon.png"))
    print("Wrote fig_SR15_soft_horizon.png")

if __name__ == "__main__":
    main()
