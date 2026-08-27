#!/usr/bin/env python3
"""Figures for "The Physics of Hyperdimensional Computing: A Phase-Space Lens for
Vector Symbolic Architectures".

Summary numbers are read from the committed results.json files in cwft_experiments;
the richer panels (kernels, Wigner heatmaps, autocorrelation curves, resolution
profiles) are recomputed inline with a fixed seed, so every figure is reproducible.
Emits 300 dpi PNGs into ./figures/.
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

def _find_experiments():
    """Locate the experiment results directory.

    Tries, in order: the CWFT_EXPERIMENTS environment variable; the sibling
    layout used in the source repository; and two layouts a reviewer is likely to
    get after unpacking an archive.  Falls back to the script's own directory, so
    a self-contained bundle with the JSONs beside make_figures.py also works.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get("CWFT_EXPERIMENTS"),
             os.path.join(here, "..", "..", "cwft_experiments"),
             os.path.join(here, "experiments"),
             os.path.join(here, "..", "experiments"),
             os.path.join(here, "code_vsa_phase"),
             here]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "cwf_fpe_uncertainty_results.json")):
            return os.path.abspath(c)
    raise SystemExit(
        "Could not find the experiment results.\n"
        "Set CWFT_EXPERIMENTS to the directory holding cwf_*_results.json, e.g.\n"
        "    CWFT_EXPERIMENTS=/path/to/results python3 make_figures.py")


EXP = _find_experiments()
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 300
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "mathtext.fontset": "cm",
})
C = {"coh": "#1f4e79", "dec": "#c0392b", "alt": "#7f8c8d",
     "acc": "#2e7d32", "warn": "#e67e22", "pur": "#6c3483"}


def load(name):
    with open(os.path.join(EXP, name)) as f:
        return json.load(f)


# ---- inlined, seeded recompute helpers (match the experiments) ---------------
def freqs(N, sigma, omega0=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    th = omega0 + sigma * rng.standard_normal(N)
    return th[th > 0] if omega0 > 0 else th


def kernel(theta, d):
    return np.cos(np.outer(d, theta)).mean(1)


def bundle(values, theta, phases=None):
    if phases is None:
        phases = np.zeros(len(values))
    return sum(np.exp(1j * p) * np.exp(1j * v * theta) for v, p in zip(values, phases))


def autocorr(b, theta, ds):
    return (np.cos(np.outer(ds, theta)) @ (np.abs(b) ** 2)) / len(theta)


def profile(theta, x0, xs):
    return np.exp(1j * np.outer(x0 - xs, theta)).mean(1)


def wigner_ville(psi):
    M = len(psi); W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n); m = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex); r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)


# =============================================================================
def fig_uncertainty():
    j = load("cwf_fpe_uncertainty_results.json")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    # (A) kernel = Fourier transform of the frequency distribution
    d = np.linspace(0, 8, 400)
    thg = rng.normal(0, 1.0, 6000)
    thu = rng.uniform(-2.0, 2.0, 6000)
    ax[0].plot(d, kernel(thg, d), color=C["coh"], lw=1.8, label="Gaussian freqs")
    ax[0].plot(d, np.exp(-d**2/2), "--", color="k", lw=1.0)
    ax[0].plot(d, kernel(thu, d), color=C["warn"], lw=1.8, label="uniform freqs")
    ax[0].plot(d, np.sinc(2*d/np.pi), "--", color="k", lw=1.0, label="analytic FT")
    ax[0].axhline(0, color="#bbbbbb", lw=0.6)
    ax[0].set_xlabel(r"separation $\Delta$"); ax[0].set_ylabel(r"kernel $K(\Delta)$")
    ax[0].set_title(r"(A) $K(\Delta)=\mathrm{FT}[\,p(\theta)\,]$")
    ax[0].legend(frameon=False, fontsize=8)
    # (B) reciprocal resolution-bandwidth law
    sw = j["part2_uncertainty"]["gaussian_sweep"]
    sig = [r["sigma"] for r in sw]; res = [r["resolution"] for r in sw]
    prod = np.mean([r["product"] for r in sw])
    ax[1].loglog(sig, res, "o-", color=C["coh"], lw=1.8, ms=5, label="resolution")
    ax[1].loglog(sig, [prod/s for s in sig], "--", color=C["alt"], lw=1.0,
                 label=r"$\propto 1/\sigma$")
    ax[1].set_xlabel(r"bandwidth $\sigma$")
    ax[1].set_ylabel("resolution (HWHM)", color=C["coh"])
    # The caption claims the PRODUCT is constant, so plot the product.
    axp = ax[1].twinx()
    axp.semilogx(sig, [r["product"] for r in sw], "s-", color=C["acc"], lw=1.6, ms=4,
                 label=r"product $\hbar_c$")
    axp.axhline(np.sqrt(2 * np.log(2)), color=C["acc"], ls=":", lw=0.9)
    axp.set_ylim(0, 2.0)
    axp.set_ylabel(r"$\Delta x\,\Delta\theta$", color=C["acc"])
    axp.annotate(r"$\sqrt{2\ln 2}$", (sig[0], np.sqrt(2*np.log(2)) + 0.08),
                 fontsize=6.5, color=C["acc"])
    ax[1].set_title(rf"(B) product flat at $\hbar_c\approx{prod:.2f}$")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = axp.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_uncertainty.png")); plt.close(fig)


def fig_wigner():
    j = load("cwf_fpe_wigner_results.json")
    rng = np.random.default_rng(0)
    th = (4.0 + 1.0 * rng.standard_normal(3000)); th = th[th > 0]
    xs = np.linspace(-6, 16, 220); dx = xs[1] - xs[0]; M = len(xs)
    cat = profile(th, 2.0, xs) + profile(th, 8.0, xs)
    W = np.fft.fftshift(wigner_ville(cat), axes=1)
    # true p axis under the p = theta convention: the WV FFT index maps to -2p
    # (pure-tone unit test: cwf_fpe_wigner_convention.py)
    pax = -0.5 * np.fft.fftshift(2 * np.pi * np.fft.fftfreq(M, dx))
    order = np.argsort(pax)
    W = W[:, order]; pax = pax[order]
    pc = pax[np.argmax(np.abs(W).sum(0))]                        # energy band centre
    keep = np.abs(pax - pc) < 3.5
    fig = plt.figure(figsize=(6.3, 2.55))
    # (A) Wigner heatmap of a cat state -- colour scale tuned to reveal the fringes
    axA = fig.add_subplot(1, 2, 1)
    vmax = 6.0
    im = axA.imshow(W[:, keep].T, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[xs[0], xs[-1], pax[keep][0]-pc, pax[keep][-1]-pc])
    axA.set_xlabel(r"value $x$"); axA.set_ylabel(r"frequency $p$ (rel.)")
    axA.set_title("(A) Wigner $W(x,p)$ of a 2-value bundle\n(blue$<$0: interference fringes at $x{=}5$)")
    fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    # (B) negativity vs separation + decoherence
    axB = fig.add_subplot(1, 2, 2)
    p1 = j["part1_separation"]; base = j["baseline"]["single"]
    seps = [r["sep"] for r in p1]; exc = [max(0, r["neg"]-base) for r in p1]
    axB.plot(seps, exc, "o-", color=C["coh"], lw=1.8, ms=4, label="genuine negativity")
    axB.axhline(0, color="#bbbbbb", lw=0.6)
    axB.set_xlabel(r"separation $\Delta$ (resolution $\approx1$)")
    axB.set_ylabel("excess Wigner negativity")
    axB.set_title("(B) onset with separation")
    # inset bars: coherent / decohered / floor
    dec = j["part2_decoherence"]
    axin = axB.inset_axes([0.55, 0.12, 0.4, 0.5])
    axin.bar([0, 1, 2], [dec["coherent"], dec["mixture"], dec["floor"]],
             color=[C["coh"], C["dec"], C["alt"]])
    axin.set_xticks([0, 1, 2]); axin.set_xticklabels(["coh", "dec", "floor"], fontsize=7)
    axin.set_title(f"decoh. $-{dec['frac_removed']*100:.0f}\\%$", fontsize=7)
    axin.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_wigner.png")); plt.close(fig)


def fig_resource():
    j = load("cwf_fpe_interference_results.json")
    rng = np.random.default_rng(0)
    th = rng.normal(0, 2.0, 4000)
    Delta = 3.0; m = 5
    bases = np.sort(rng.uniform(0, 25, m))
    vals = np.concatenate([bases, bases + Delta])
    ds = np.linspace(2, 12, 500)   # exclude the trivial self-term region d<2 (as in the experiment)
    sc = autocorr(bundle(vals, th), th, ds)
    # decohered (avg)
    accd = np.zeros(len(ds))
    for _ in range(40):
        ph = rng.uniform(0, 2*np.pi, len(vals))
        accd += autocorr(bundle(vals, th, ph), th, ds)
    accd /= 40
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    ax[0].plot(ds, sc, color=C["coh"], lw=1.6, label="coherent")
    ax[0].plot(ds, accd, color=C["dec"], lw=1.4, label="decohered")
    ax[0].axvline(Delta, color=C["acc"], ls=":", lw=1.2, label=rf"hidden period $\Delta={Delta:g}$")
    ax[0].set_xlabel(r"lag $d$"); ax[0].set_ylabel(r"autocorrelation score$(d)$")
    ax[0].set_title("(A) period-finding by interference")
    ax[0].legend(frameon=False, fontsize=8)
    p3 = j["part3_scaling"]
    ms = [r["m"] for r in p3]; coh = [r["coh"] for r in p3]; dec = [abs(r["dec"]) for r in p3]
    ax[1].loglog(ms, coh, "o-", color=C["coh"], lw=1.8, ms=5, label="coherent peak")
    ax[1].loglog(ms, [max(d, 1e-3) for d in dec], "s--", color=C["dec"], lw=1.4, ms=4,
                 label="decohered (noise)")
    ax[1].set_xlabel(r"number of pairs $s$"); ax[1].set_ylabel(r"period peak at $\Delta$")
    ax[1].set_title("(B) signal grows with data")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_resource.png")); plt.close(fig)


def fig_invariant():
    j = load("cwf_fpe_invariant_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    rows = j["rows"]
    R = [r["R"] for r in rows]

    def ci(p, n=100):
        """Clopper-Pearson 95% interval, as asymmetric yerr.

        The Wald interval 1.96*sqrt(p(1-p)/n) is identically ZERO at p=1, which is
        exactly where this figure's headline sits -- it would draw the 100% curve
        with no uncertainty at all.  Clopper-Pearson gives [0.964, 1] at 100/100.
        """
        from scipy.stats import beta
        p = np.asarray(p, float)
        x = np.round(p * n)
        lo = np.where(x > 0, beta.ppf(0.025, np.maximum(x, 1e-9), n - x + 1), 0.0)
        hi = np.where(x < n, beta.ppf(0.975, x + 1, np.maximum(n - x, 1e-9)), 1.0)
        return np.vstack([p - lo, hi - p])

    for key, style, col, lab in [("coherent", "o-", C["coh"], "coherent fingerprint"),
                                 ("decode", "d:", C["pur"], "decode-then-intervals"),
                                 ("refunbind", "*-.", C["warn"], "reference-unbind (SSP)"),
                                 ("naive", "s-", C["dec"], "naive bundle"),
                                 ("decohered", "^--", C["alt"], "decohered")]:
        p = [r[key] for r in rows]
        ax[0].errorbar(R, p, yerr=ci(p), fmt=style, color=col, lw=1.6, ms=4,
                       capsize=2, label=lab)
    ax[0].axhline(1/6, color="#bbbbbb", ls=":", lw=0.8, label="chance")
    ax[0].set_xlabel("transposition range $R$"); ax[0].set_ylabel("accuracy")
    ax[0].set_title("(A) translation-invariant recognition")
    ax[0].set_ylim(0, 1.05); ax[0].legend(frameon=False, fontsize=8)
    nz = j["noise"]
    pn = [r["coherent"] for r in nz]
    ax[1].errorbar([r["noise"] for r in nz], pn, yerr=ci(pn), fmt="o-",
                   color=C["coh"], lw=1.8, ms=5, capsize=2)
    ax[1].axhline(1/6, color="#bbbbbb", ls=":", lw=0.8)
    ax[1].set_xlabel(r"note jitter $\sigma$"); ax[1].set_ylabel("accuracy (full transposition)")
    ax[1].set_title("(B) graceful degradation with noise")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_invariant.png")); plt.close(fig)


def fig_squeezing():
    j = load("cwf_fpe_squeezing_results.json")
    # recompute resolution(x) over a fine grid for flat vs foveated
    rng = np.random.default_rng(0)
    th = rng.normal(0, 2.0, 4000)
    A, B, xf = 0.0, 10.0, 5.0
    xg = np.linspace(A, B, 4000)
    dens = 0.2 + np.exp(-0.5*((xg-xf)/1.0)**2)
    cdf = np.cumsum(dens); cdf -= cdf[0]; cdf /= cdf[-1]; wg = A + (B-A)*cdf

    def res_at(warp, x0, dmax=4.0, n=800):
        d = np.linspace(0, dmax, n)
        w0 = np.interp(x0, xg, warp); wd = np.interp(x0+d, xg, warp)
        K = np.cos(np.outer(wd-w0, th)).mean(1)
        below = np.where(K < 0.5)[0]
        return d[below[0]] if len(below) else dmax
    xpts = np.linspace(0.5, 8.5, 25)   # stop before the probe window saturates at the domain edge
    rflat = [res_at(xg, x) for x in xpts]
    rfov = [res_at(wg, x) for x in xpts]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    ax[0].plot(xpts, rflat, "o-", color=C["alt"], lw=1.6, ms=3, label="flat (uniform)")
    ax[0].plot(xpts, rfov, "o-", color=C["coh"], lw=1.8, ms=3, label="squeezed (fovea at 5)")
    ax[0].axvline(xf, color=C["acc"], ls=":", lw=1.0)
    ax[0].set_xlabel(r"value $x$"); ax[0].set_ylabel("local resolution (HWHM)")
    ax[0].set_title("(A) resolution redistributed")
    ax[0].legend(frameon=False, fontsize=8)
    # (B) the LOAD-BEARING measurement: the local cell, point by point.  The bar
    # chart this replaces plotted a structural identity (any endpoint-preserving
    # warp conserves the integral); the local cell is the one that has to be
    # measured, and cwf_fpe_localcell_results.json was committed but never plotted.
    lc = load("cwf_fpe_localcell_results.json")
    xs = [r["x"] for r in lc["rows"]]
    res = [r["res"] for r in lc["rows"]]
    bw = [r["bw"] for r in lc["rows"]]
    cell = [r["cell"] for r in lc["rows"]]
    hb = lc["params"]["hbar_c"]
    b = ax[1]
    b.plot(xs, res, "o-", color=C["coh"], ms=3, lw=1.4, label=r"resolution $\mathrm{res}(x)$")
    b.set_xlabel("value $x$"); b.set_ylabel(r"resolution", color=C["coh"])
    b.tick_params(axis="y", labelcolor=C["coh"])
    b2 = b.twinx()
    b2.plot(xs, bw, "s--", color=C["warn"], ms=3, lw=1.4, label=r"bandwidth $\mathrm{bw}(x)$")
    b2.set_ylabel(r"local bandwidth", color=C["warn"])
    b2.tick_params(axis="y", labelcolor=C["warn"])
    b3 = b.twinx(); b3.spines["right"].set_position(("axes", 1.28))
    b3.axhspan(hb * 0.92, hb * 1.08, color=C["pur"], alpha=0.13, zorder=0)
    b3.plot(xs, cell, "-", color=C["pur"], lw=2.0)
    b3.axhline(hb, color=C["pur"], ls=":", lw=1.0)
    b3.set_ylim(hb * 0.6, hb * 1.5)
    b3.set_ylabel(r"local cell $=\mathrm{res}\times\mathrm{bw}$", color=C["pur"])
    b3.tick_params(axis="y", labelcolor=C["pur"])
    b.set_title(r"(B) the books balance point by point")
    b3.annotate(r"cell flat to $8\%$ of $\hbar_c$", xy=(0.30, 0.92),
                xycoords="axes fraction", fontsize=7.4, color=C["pur"])
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_squeezing.png")); plt.close(fig)


def fig_complexity():
    j = load("cwf_fpe_complexity_results.json")
    rows = j["rows"]
    k = [r["k"] for r in rows]
    rd = [r["readout"]*1e3 for r in rows]
    nv = [(r["naive"]*1e3 if r["naive"] else None) for r in rows]
    ft = [r["fft"]*1e3 for r in rows]
    bd = [r["build"]*1e3 for r in rows]
    kn = [kk for kk, n in zip(k, nv) if n]; nvv = [n for n in nv if n]
    fig, ax = plt.subplots(figsize=(3.9, 2.70))
    ax.loglog(k, rd, "o-", color=C["coh"], lw=1.8, ms=5, label="bundle readout (given $b$)")
    ax.loglog(kn, nvv, "s-", color=C["dec"], lw=1.6, ms=4, label=r"naive $O(k^2)$")
    ax.loglog(k, ft, "^-", color=C["acc"], lw=1.6, ms=4, label=r"classical hist$\to$FFT")
    ax.loglog(k, bd, "v--", color=C["alt"], lw=1.4, ms=4, label=r"bundle build $O(kN)$")
    ax.set_xlabel("number of items $k$"); ax.set_ylabel("time (ms)")
    ax.set_title("Relational readout cost vs $k$")
    ax.legend(framealpha=1.0, frameon=True, edgecolor="none", fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_complexity.png")); plt.close(fig)


# fig_classical_limit removed: the tex no longer includes it.  Its panel A
# (orthogonality) and panel B (Husimi threshold curve) both live on in
# fig_offswitches, and the coarse-graining story is now WATCHED directly in
# fig_husimi_strip (three Wigner portraits under increasing smoothing).


def _resonator_width_ci(Ls, curve, n=60, boots=2000, rng=None):
    """Bootstrap CI for the transition width L20-L80 by resampling the binomials.

    The width is a derived quantity, so it carries real uncertainty at n=60 per
    point; without this the flatness claim is asserted rather than shown.
    """
    rng = rng or np.random.default_rng(0)
    Ls = np.asarray(Ls, float); p = np.asarray(curve, float)

    def width_of(pp):
        def cross(level):
            for i in range(len(pp) - 1):
                a, b = pp[i], pp[i + 1]
                if (a - level) * (b - level) <= 0 and a != b:
                    return Ls[i] + (a - level) / (a - b) * (Ls[i + 1] - Ls[i])
            return np.nan
        return cross(0.2) - cross(0.8)

    w = [width_of(rng.binomial(n, np.clip(p, 0, 1)) / n) for _ in range(boots)]
    w = np.array([x for x in w if np.isfinite(x)])
    if len(w) < 50:
        return np.nan, np.nan
    return float(np.percentile(w, 2.5)), float(np.percentile(w, 97.5))


def fig_resonator():
    jh = load("cwf_resonator_thetahi_results.json")   # n=500/point, with SE
    js = load("cwf_resonator_scaling_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    p2 = jh["rows"]
    th = np.array([r["theta"] for r in p2])
    su = np.array([r["succ"] for r in p2])
    se = np.array([r["se"] for r in p2])
    ax[0].plot(th, su, "o-", color=C["coh"], lw=1.8, ms=5, label="success")
    ax[0].fill_between(th, su - se, su + se, color=C["coh"], alpha=0.25, lw=0)
    ax[0].plot(th, [r["nohalt"] for r in p2], "s--", color=C["dec"], lw=1.4, ms=4,
               label="timeouts (capped runs)")
    best = max(p2, key=lambda r: r["succ"])
    ax[0].axvline(best["theta"], color=C["acc"], ls=":", lw=1.0)
    ax[0].set_xlabel(r"noise temperature $\theta_c$"); ax[0].set_ylabel("fraction")
    ax[0].set_title(r"(A) temperature sweep, fixed load ($n{=}500$; $\pm$1 SE)")
    ax[0].legend(frameon=False, fontsize=8)
    Ns = js["N"]; ws = js["width"]
    # Bootstrap CIs: the width is derived from 60-instance binomials per load
    # point, so the flatness claim needs visible uncertainty.
    los, his = [], []
    for N, w in zip(Ns, ws):
        lo, hi = _resonator_width_ci(js["Ls"], js["curves"][str(N)])
        los.append(w - lo if np.isfinite(lo) else 0.0)
        his.append(hi - w if np.isfinite(hi) else 0.0)
    ax[1].errorbar(Ns, ws, yerr=[los, his], fmt="o-", color=C["pur"], lw=1.8, ms=6,
                   capsize=3, label="measured (95% CI)")
    # what a mean-field critical narrowing would look like, anchored at the first N
    ax[1].plot(Ns, [ws[0] * (Ns[0] / N) ** 0.5 for N in Ns], ":", color=C["dec"],
               lw=1.4, label=r"critical $N^{-1/2}$ (not seen)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_ylim(0, max(ws) * 1.6)
    ax[1].set_xlabel("dimension $N$"); ax[1].set_ylabel(r"transition width $L_{20}-L_{80}$")
    ax[1].set_title("(B) no narrowing detected: a crossover")
    ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_resonator.png")); plt.close(fig)


def fig_advantage():
    jh = load("cwf_fpe_highdim_results.json")
    jb = load("cwf_fpe_beyond_capacity_results.json")
    jhash = load("cwf_fpe_hashjoin_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    # (A) dimension axis: grid time explodes as B^m, bundle flat; infeasible marked
    rows = jh["rows"]
    ms = [r["m"] for r in rows]
    tb = [r["t_bundle_ms"] for r in rows]
    tsp = [r["t_sparse_ms"] for r in rows]
    tg = [r["t_grid_ms"] for r in rows if r["grid_feasible"]]
    mg = [r["m"] for r in rows if r["grid_feasible"]]
    ax[0].semilogy(mg, tg, "s-", color=C["dec"], lw=1.8, ms=5,
                   label=r"dense grid ($B^m$ bins)")
    ax[0].semilogy(ms, tb, "o-", color=C["coh"], lw=1.8, ms=5,
                   label=r"bundle ($N{=}4000$, any $m$)")
    ax[0].semilogy(ms, tsp, "^--", color=C["acc"], lw=1.6, ms=4,
                   label=r"sparse pairwise (exact, $O(k^2m)$)")
    rh = jhash["part2"]["rows_m"]
    ax[0].semilogy([r["m"] for r in rh], [r["t_hash_ms"] for r in rh], "v--",
                   color=C["pur"], lw=1.6, ms=4,
                   label=r"spatial-hash join (exact, $O(k\,3^m m)$)")
    ax[0].set_ylim(top=max(tg) * 400)
    for r in rows:
        if not r["grid_feasible"]:
            gb = r["grid_mem_bytes"] / 1e9
            # Plotted in a labelled band well above the timed data: these points have
            # no measured time (the grid was never built), so the height is nominal.
            ax[0].plot([r["m"]], [max(tg) * 8], "x", color=C["dec"], ms=8, mew=2)
            ax[0].annotate(f"{gb:.0f} GB", (r["m"], max(tg) * 14),
                           ha="center", fontsize=7, color=C["dec"])
    ax[0].axhspan(max(tg) * 3, max(tg) * 400, color=C["dec"], alpha=0.05, zorder=0)
    ax[0].annotate("not run: over memory budget", (1.05, max(tg) * 120), fontsize=6.5,
                   color=C["dec"], ha="left", va="bottom")
    ax[0].set_xlabel("value dimension $m$"); ax[0].set_ylabel("readout time (ms)")
    ax[0].set_title("(A) the dimension axis (matched accuracy)")
    # Legend in the empty left part of the not-run band: every in-plot corner
    # is crossed by the red B^m diagonal or the flat bundle/sparse curves.
    ax[0].legend(frameon=False, fontsize=6.0, loc="upper left",
                 bbox_to_anchor=(0.0, 0.91), handlelength=1.0,
                 handletextpad=0.4, labelspacing=0.3)
    # (B) load axis: recall degrades past capacity; direct spectral readout stays
    rows = jb["rows"]
    k = [r["k"] for r in rows]
    ax[1].semilogx(k, [r["recall"] for r in rows], "o-", color=C["alt"], lw=1.6,
                   ms=4, label="item recall")
    ax[1].semilogx(k, [r["fp"] for r in rows], "^--", color=C["dec"], lw=1.4,
                   ms=4, label="false-positive rate")
    ax[1].semilogx(k, [r["direct"] for r in rows], "s-", color=C["coh"], lw=1.8,
                   ms=5, label=r"spectral readout, $s\propto k$")
    # Markers only, no cosmetic offset: this curve tracks "direct spectral
    # detection" closely, and open diamonds keep both visible honestly.
    ax[1].semilogx(k, [r["decode"] for r in rows], "d", color=C["pur"],
                   ms=7, mfc="none", mew=1.4, label=r"decode-then-difference, $s\propto k$")
    kf = [r["k"] for r in jb["rows_fixed_s"]]
    ax[1].semilogx(kf, [r["direct"] for r in jb["rows_fixed_s"]], "v-", color=C["warn"],
                   lw=1.8, ms=5, label=r"spectral readout, $s$ fixed $=35$")
    ax[1].axhline(jb["params"]["chance"], color="#999999", lw=0.8, ls="-.")
    ax[1].annotate("chance", (22, jb["params"]["chance"] + 0.02), fontsize=6.5,
                   color="#777777")
    ax[1].axvline(jb["params"]["capacity_007N"], color=C["acc"], ls=":", lw=1.0)
    ax[1].set_xlabel("stored items $k$"); ax[1].set_ylabel("fraction")
    ax[1].set_ylim(-0.03, 1.08)
    ax[1].set_title("(B) the load axis (capacity $\\approx$ 70, dotted)")
    # White background so the capacity guide (dotted vline) does not strike the
    # rows; the narrower font keeps the box clear of the orange fixed-s curve.
    ax[1].legend(fontsize=6.0, loc="center left", bbox_to_anchor=(0.0, 0.42),
                 framealpha=0.9, edgecolor="none", handlelength=1.2,
                 handletextpad=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_advantage.png")); plt.close(fig)


def fig_sketch():
    jv = load("cwf_fpe_variance_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    # (A) variance law: SD vs k/sqrt(N), collapse onto the diagonal
    rows = jv["rows"]
    pred = [r["pred"] for r in rows]
    sd = [r["sd"] for r in rows]
    ax[0].loglog(pred, sd, "o", color=C["coh"], ms=7)
    lo, hi = min(pred) * 0.7, max(pred) * 1.4
    ax[0].loglog([lo, hi], [lo, hi], "--", color=C["alt"], lw=1.2,
                 label=r"$\mathrm{SD}=k/\sqrt{N}$")
    ax[0].set_xlabel(r"prediction $k/\sqrt{N}$"); ax[0].set_ylabel(r"measured SD of score")
    ax[0].set_title(r"(A) variance law $\mathrm{Var}\approx k^2/N$")
    ax[0].legend(frameon=False, fontsize=8)
    # (B) signal at a planted lag ~ number of pairs
    sig = jv["signal"]
    npairs = [r["n_pairs"] for r in sig]; ms = [r["mean_score"] for r in sig]
    ax[1].plot(npairs, ms, "o-", color=C["acc"], lw=1.8, ms=6)
    ax[1].plot([0, max(npairs)], [0, max(npairs)], ":", color=C["alt"], lw=1.0,
               label=r"$\mathrm{signal}=s$")
    ax[1].set_xlabel(r"number of planted pairs $s$")
    ax[1].set_ylabel(r"$E[\mathrm{score}(\Delta)]$")
    ax[1].set_title("(B) signal at the lag $\\sim s$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_sketch.png")); plt.close(fig)


def fig_hardware():
    j = load("cwf_fpe_hardware_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    ph = j["photonic"]
    sp = [r["sigma_phi"] for r in ph]
    # Twin axes: the cell is an absolute phase-space quantity (envelope HWHM x sigma,
    # analytic sqrt(2 ln 2)); the coherence is a negativity fraction.  Plotting both
    # on one axis labelled "value" conflated them.
    ax[0].plot(sp, [r["cell"] for r in ph], "s--", color=C["acc"], lw=1.6, ms=5,
               label=r"cell $\hbar_c$ (envelope)")
    ax[0].axhline(np.sqrt(2 * np.log(2)), color=C["acc"], lw=0.8, ls=":", alpha=0.7)
    ax[0].annotate(r"$\sqrt{2\ln 2}$", (0.02, np.sqrt(2 * np.log(2)) + 0.09),
                   fontsize=7, color=C["acc"])
    ax[0].set_ylim(0.0, 1.45)
    ax[0].set_xlabel(r"photonic phase jitter $\sigma_\phi$ (rad)")
    ax[0].set_ylabel(r"cell $\hbar_c$", color=C["acc"])
    axc = ax[0].twinx()
    axc.plot(sp, [r["coherence"] for r in ph], "o-", color=C["coh"], lw=1.8, ms=5,
             label="coherence (Wigner excess)")
    axc.set_ylabel("coherence (Wigner excess)", color=C["coh"])
    axc.set_ylim(0.0, 0.20)
    ax[0].set_title("(A) cell holds; coherence decays gracefully")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = axc.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="lower left")
    pc = j["pcm"]
    bits = [r["bits"] for r in pc]
    ax[1].plot(bits, [r["coherence"] for r in pc], "o-", color=C["coh"], lw=1.8, ms=6,
               label="coherence (Wigner excess)")
    ax[1].axhline(0, color="#bbbbbb", lw=0.6)
    ax[1].axvspan(0.5, 1.5, color=C["alt"], alpha=0.15)
    ax[1].annotate("$q{=}2$\n(dead)", (1, 0.06), fontsize=7, ha="center", color=C["dec"])
    ax[1].set_xlabel("PCM phase bit-depth")
    ax[1].set_ylabel("coherence (Wigner excess)", color=C["coh"])
    axr = ax[1].twinx()
    axr.plot(bits, [r["period_peak"] for r in pc], "^--", color=C["warn"], lw=1.6, ms=6,
             label="relational readout (peak/bg)")
    axr.set_ylabel("period peak / background", color=C["warn"])
    ax[1].set_title("(B) threshold for coherence, dial for the readout")
    h1, l1 = ax[1].get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
    # Raised off the axes floor: the anchored position clears the steep blue
    # bits 1->2 segment and the low orange triangle at bits=1.
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.0,
                 loc="lower right", bbox_to_anchor=(1.0, 0.10),
                 handlelength=1.4, handletextpad=0.5)
    ax[1].set_xticks(bits)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_hardware.png")); plt.close(fig)


# fig_rope removed: it plotted one array (K) against a subsample of itself, so it
# demonstrated nothing.  RoPE's content-free logit IS the FPE kernel by construction
# (R(a)^T R(b) = R(b-a)); the check lives in cwf_fpe_rope.py, which records
# identity_max_abs_diff = 0.0.  That number is now quoted in the text instead.


def fig_ams():
    j = load("cwf_fpe_ams_results.json")
    rows = j["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    # (A) noise-floor SD: FPE and AGMS both track F2/sqrt(N)
    pred = [r["pred_F2_over_sqrtN"] for r in rows]
    sdf = [r["sd_fpe_null"] for r in rows]
    sda = [r["sd_agms_null"] for r in rows]
    ax[0].loglog(pred, sdf, "o", color=C["coh"], ms=7, label="FPE bundle")
    ax[0].loglog(pred, sda, "s", color=C["acc"], ms=6, label=r"AGMS $\pm1$ (optimal)")
    lo, hi = min(pred) * 0.7, max(pred) * 1.4
    ax[0].loglog([lo, hi], [lo, hi], "--", color=C["alt"], lw=1.1,
                 label=r"$F_2/\sqrt{N}$")
    ax[0].set_xlabel(r"$F_2/\sqrt{N}$"); ax[0].set_ylabel("noise-floor SD")
    ax[0].set_title("(A) both $\\Theta(F_2/\\sqrt{N})$")
    ax[0].legend(frameon=False, fontsize=8)
    # (B) ratio FPE/AGMS: a flat small constant
    k = [r["k"] for r in rows]
    ratio = np.array([r["ratio_fpe_agms"] for r in rows])
    # An SD estimated from n draws has relative error ~1/sqrt(2(n-1)); the ratio
    # of two such estimates carries sqrt(2) times that.  Without this the word
    # "flat" is asserted rather than shown.
    ndr = j["params"].get("draws", 200)
    rel = np.sqrt(2.0) / np.sqrt(2.0 * (ndr - 1))
    ax[1].errorbar(k, ratio, yerr=ratio * rel, fmt="o", color=C["pur"], ms=7,
                   capsize=3, lw=1.2)
    ax[1].set_xscale("log")
    ax[1].axhline(1.0, color=C["alt"], ls="--", lw=1.0, label="AGMS optimum")
    ax[1].set_ylim(0, 2.0)
    ax[1].set_xlabel("set size $k$"); ax[1].set_ylabel(r"SD$_{\rm FPE}$/SD$_{\rm AGMS}$")
    ax[1].set_title("(B) flat premium over optimal")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ams.png")); plt.close(fig)


def fig_trajectory():
    jt = load("cwf_fpe_trajectory_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    # (A) phase-space portrait of a bundled oscillator orbit (seed 10 = seed 1
    # of the experiment), true orbit overlaid
    N, A, T = 4000, 3.0, 200
    rng = np.random.default_rng(10)
    th = 2.0 * rng.standard_normal(N)
    om = 2.0 * rng.standard_normal(N)
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    b = np.exp(1j * (np.outer(A * np.cos(t), th)
                     + np.outer(-A * np.sin(t), om))).sum(axis=0)
    xs = np.linspace(-5, 5, 141)
    Ex = np.exp(-1j * np.outer(xs, th))
    Ep = np.exp(-1j * np.outer(xs, om))
    W = np.abs((Ex * b[None, :]) @ Ep.T / N) ** 2
    ax[0].imshow(W.T, origin="lower", extent=[-5, 5, -5, 5], cmap="Blues",
                 aspect="equal")
    cc = np.linspace(0, 2 * np.pi, 200)
    ax[0].plot(A * np.cos(cc), A * np.sin(cc), "--", color=C["acc"], lw=1.2,
               label="true orbit")
    ax[0].set_xlabel("value $x$"); ax[0].set_ylabel("momentum $p$")
    conc = jt["portrait"]["concentration"]
    ax[0].set_title("(A) portrait of one trajectory bundle")
    # Below the ring (radius 3), not across it: the label at the origin spanned
    # the dark band on both sides.
    ax[0].annotate(f"ring concentration {conc:.1f}$\\times$", (0, -4.45),
                   ha="center", fontsize=8, color=C["coh"])
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    # (B) time-reversal split: forward vs reversed similarity, two encodings
    sweep, half = jt["reversal"]["sweep"], jt["reversal"]["half_orbit"]
    pos = [sweep["sim_position_only"], half["sim_position_only"]]
    joint = [sweep["sim_joint"], half["sim_joint"]]
    xpos = np.arange(2)
    ax[1].bar(xpos - 0.17, pos, 0.3, color=C["dec"], label="position-only code")
    ax[1].bar(xpos + 0.17, joint, 0.3, color=C["coh"], label="joint $(x,p)$ code")
    for i, v in enumerate(pos):
        ax[1].annotate(f"{v:.2f}", (xpos[i] - 0.17, v + 0.03), ha="center",
                       fontsize=8)
    for i, v in enumerate(joint):
        ax[1].annotate(f"{v:+.2f}", (xpos[i] + 0.17, max(v, 0) + 0.03),
                       ha="center", fontsize=8)
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(["linear sweep", "oscillator half-orbit"])
    ax[1].set_ylim(-0.15, 1.62)
    ax[1].axhline(0, color="0.6", lw=0.8)
    ax[1].set_ylabel("forward$\\,\\leftrightarrow\\,$reversed similarity")
    ax[1].set_title("(B) same positions, mirrored momenta")
    mq = jt["reversal"]["momentum_query"]
    # Headroom added via ylim; annotation top right, legend top left, both
    # above the bars and their value labels.
    ax[1].annotate("momentum query at $x{=}0$:\n"
                   f"contrast ${mq['contrast']:+.3f}$", (0.98, 0.90),
                   xycoords="axes fraction", ha="right", va="top",
                   fontsize=7.5, color=C["coh"])
    ax[1].legend(frameon=False, fontsize=7, loc="upper left",
                 handlelength=1.2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_trajectory.png"))
    plt.close(fig)


def fig_chirp():
    jc = load("cwf_fpe_chirp_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    rows = jc["moving"]
    vs = [r["v"] for r in rows]
    ax[0].semilogy(vs, [r["x_rmse_static"] for r in rows], "s--", color=C["dec"],
                   lw=1.6, ms=5, label="static filter")
    ax[0].semilogy(vs, [r["x_rmse_chirp"] for r in rows], "o-", color=C["coh"],
                   lw=1.8, ms=5, label="chirped $(x,v)$ bank")
    ax[0].set_xlabel("true speed $v$")
    ax[0].set_ylabel("position RMSE")
    ax[0].set_title("(A) reading a moving value")
    ax[0].annotate("motion blur", (5.0, 1.1), fontsize=8, color=C["dec"])
    ax[0].annotate("speed read exactly\n(0.25-spaced bank)", (3.4, 0.03),
                   fontsize=8, color=C["coh"])
    ax[0].legend(frameon=False, fontsize=8, loc="center right")
    res = jc["resolution"]
    Ts = [r["T"] for r in res]
    dvs = [r["dv"] for r in res]
    ax[1].loglog(Ts, dvs, "o-", color=C["pur"], lw=1.8, ms=6,
                 label=r"measured $\Delta v$")
    guide = [dvs[0] * Ts[0] / T for T in Ts]
    ax[1].loglog(Ts, guide, ":", color=C["alt"], lw=1.2,
                 label=r"$\propto 1/T$")
    prods = ", ".join(f"{r['product']:.1f}" for r in res)
    ax[1].annotate(rf"$\Delta v\,T\,\omega_0 = {prods}$", (0.03, 0.06),
                   xycoords="axes fraction", fontsize=7.5,
                   bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax[1].set_xticks(Ts)
    ax[1].set_xticklabels([f"{T:g}" for T in Ts])
    ax[1].minorticks_off()
    ax[1].set_xlabel("encoding window $T$")
    ax[1].set_ylabel(r"speed resolution $\Delta v$ (HWHM)")
    ax[1].set_title(r"(B) the $(v,T)$ cell: $\Delta v \propto 1/T$")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_chirp.png"))
    plt.close(fig)


def fig_bispectrum():
    jb = load("cwf_fpe_bispectrum_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    rows = jb["detect_3d"]["rows"]
    ss = [r["s"] for r in rows]
    ax[0].errorbar(ss, [r["mean"] for r in rows], yerr=[r["sd"] for r in rows],
                   fmt="o-", color=C["coh"], lw=1.8, ms=6, capsize=3,
                   label="coherent readout")
    ax[0].plot(ss, [2 * s for s in ss], "--", color=C["acc"], lw=1.2,
               label="predicted $2s$")
    ax[0].axhline(jb["detect_3d"]["threshold"], color=C["alt"], ls=":",
                  lw=1.0)
    ax[0].annotate("3$\\sigma$ threshold\n(count-matched,\ndrawn at $s{=}0$ level)",
                   (0.05, 9.0), fontsize=7, va="bottom",
                   color=C["alt"])
    ax[0].errorbar([10], [jb["detect_3d"]["decohered_mean"]],
                   yerr=[jb["detect_3d"]["decohered_sd"]], fmt="x",
                   color=C["dec"], ms=9, mew=2, capsize=3, label="decohered")
    ax[0].set_xlabel("planted collinear triples $s$")
    ax[0].set_ylabel("centered score")
    ax[0].set_title("(A) third-order detection in $m{=}3$")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    el = jb["error_law"]
    ax[1].loglog(el["ks"], el["sd_k"], "o-", color=C["pur"], lw=1.8, ms=6,
                 label="measured null sd")
    guide = [el["sd_k"][0] * (k / el["ks"][0]) ** 1.5 for k in el["ks"]]
    ax[1].loglog(el["ks"], guide, ":", color=C["alt"], lw=1.2,
                 label=r"$\propto k^{3/2}$")
    ci = el.get("exp_k_ci")
    citxt = f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else ""
    ax[1].annotate(f"fitted $k^{{{el['exp_k']:.2f}}}$" + citxt + ";\n"
                   f"across $N$: $N^{{{el['exp_N']:.2f}}}$",
                   (0.05, 0.92), xycoords="axes fraction", fontsize=8,
                   va="top")
    ax[1].set_xticks(el["ks"])
    ax[1].set_xticklabels([str(k) for k in el["ks"]])
    ax[1].minorticks_off()
    ax[1].set_xlabel("stored items $k$")
    ax[1].set_ylabel("codebook noise sd (fixed sets)")
    ax[1].set_title("(B) the third-order error law")
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_bispectrum.png"))
    plt.close(fig)


def fig_ropefovea():
    jr = load("cwf_rope_fovea_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    pf = jr["position_foveation"]
    B1, B2 = jr["params"]["band"]
    ax[0].axvspan(B1, B2, color="0.9", label=f"target band [{B1},{B2}]")
    ax[0].semilogx(pf["probes"], pf["uniform_acc8"], "o-", color=C["coh"],
                   lw=1.8, ms=5, label="uniform geometric")
    ax[0].semilogx(pf["probes"], pf["banded_acc8"], "s--", color=C["dec"],
                   lw=1.6, ms=5, label="band-concentrated")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_xlabel("true offset $\\Delta$")
    ax[0].set_ylabel("identification accuracy ($|$err$|\\leq 8$)")
    ax[0].set_title("(A) position-foveation is impossible")
    # Between the two curves, clear of the upper-right legend.
    ax[0].annotate("flat: stationarity", (24, 0.695), fontsize=8,
                   color=C["coh"])
    # Below the red curve's lowest dip (0.52 at 2048), left of that marker.
    ax[0].annotate("loses everywhere,\nown band included", (160, 0.415),
                   fontsize=8, color=C["dec"], va="bottom")
    ax[0].legend(frameon=False, fontsize=8, loc="upper right")
    tilt = jr["tilt"]
    fts = [r["f_top"] for r in tilt]
    ax[1].plot(fts, [r["fine"] for r in tilt], "o-", color=C["coh"], lw=1.8,
               ms=6, label="fine accuracy ($|$err$|\\leq 2$)")
    ax[1].plot(fts, [r["gross"] for r in tilt], "s--", color=C["warn"],
               lw=1.6, ms=5, label="gross-error rate ($>$64)")
    lr = jr["long_range"]
    ax[1].annotate("at range 65,536:\n"
                   f"fine {lr[0]['fine']:.2f}$\\to${lr[1]['fine']:.2f}, "
                   f"gross {lr[0]['gross']:.2f} vs {lr[1]['gross']:.2f}",
                   (0.38, 0.22), xycoords="axes fraction", fontsize=8)
    ax[1].set_ylim(0, 1.0)
    ax[1].set_xlabel("budget fraction in the top two decades")
    ax[1].set_ylabel("rate")
    ax[1].set_title("(B) scale reallocation works, nearly free")
    ax[1].legend(frameon=False, fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_ropefovea.png"))
    plt.close(fig)


# ---- conceptual schematics ---------------------------------------------------
def _clock(ax, cx, cy, r, angles_colors):
    circ = plt.Circle((cx, cy), r, fill=False, color="0.6", lw=0.8)
    ax.add_patch(circ)
    for ang, col in angles_colors:
        ax.arrow(cx, cy, 0.82 * r * np.cos(ang), 0.82 * r * np.sin(ang),
                 head_width=0.05 * r, color=col, lw=1.6,
                 length_includes_head=True)


def fig_scheme_fpe():
    rng = np.random.default_rng(3)
    th = 2.0 * rng.standard_normal(8)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.61))
    # (A) clock hands: nearby vs far values
    a = ax[0]
    a.set_xlim(-0.6, 8.4); a.set_ylim(-0.9, 3.1)
    a.set_aspect("equal"); a.axis("off")
    x0 = 2.0
    for j in range(8):
        _clock(a, j + 0.15, 2.2, 0.42,
               [(x0 * th[j], C["coh"]), ((x0 + 0.15) * th[j], C["acc"])])
        _clock(a, j + 0.15, 0.4, 0.42,
               [(x0 * th[j], C["coh"]), ((x0 + 3.0) * th[j], C["dec"])])
    a.text(3.7, 2.95, "codes for $x$ and $x+0.15$: hands nearly agree "
                      r"$\rightarrow$ similar", ha="center", fontsize=8.5)
    a.text(3.7, -0.75, "codes for $x$ and $x+3$: hands scrambled "
                       r"$\rightarrow$ dissimilar", ha="center", fontsize=8.5)
    a.set_title("(A) a number is a pattern of clock angles")
    # (B) codebook width <-> kernel width (Fourier duality)
    b = ax[1]
    d = np.linspace(0, 6, 300)
    for sig, col, lab in [(0.5, C["warn"], r"narrow codebook $\sigma{=}0.5$"),
                          (2.0, C["coh"], r"wide codebook $\sigma{=}2$")]:
        b.plot(d, np.exp(-(sig * d) ** 2 / 2), color=col, lw=1.8, label=lab)
    b.set_xlabel(r"separation $\Delta$"); b.set_ylabel(r"similarity $K(\Delta)$")
    # Both annotations stacked in the empty wedge below the inset and above the
    # orange tail: the legend used to overprint the blue one, and the inset
    # clipped the orange one.
    b.annotate("wide codebook $\\rightarrow$ narrow\nkernel (fine resolution)",
               (3.65, 0.60), fontsize=6.8, color=C["coh"], ha="left", va="top")
    b.annotate("narrow codebook $\\rightarrow$ wide\nkernel (coarse)",
               (3.65, 0.39), fontsize=6.8, color=C["warn"], ha="left", va="top")
    axin = b.inset_axes([0.63, 0.70, 0.34, 0.27])
    t = np.linspace(-5, 5, 200)
    axin.plot(t, np.exp(-t**2 / (2 * 0.5**2)) / 0.5, color=C["warn"], lw=1.2)
    axin.plot(t, np.exp(-t**2 / (2 * 2.0**2)) / 2.0, color=C["coh"], lw=1.2)
    axin.set_title(r"codebooks $p(\theta)$", fontsize=6.5, pad=1.5)
    axin.set_xticks([]); axin.set_yticks([])
    # Frameless legend in a dead band opened below y=0, so it covers no curve;
    # yticks pinned to the original 0..1 labels.
    b.set_ylim(-0.36, 1.05)
    b.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    b.legend(frameon=False, fontsize=7.5, loc="lower left")
    b.set_title("(B) the kernel is the codebook's Fourier twin", pad=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scheme_fpe.png"))
    plt.close(fig)


def fig_scheme_cell():
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.46))
    a = ax[0]
    a.set_xlim(0, 10); a.set_ylim(0, 6); a.axis("off")
    from matplotlib.patches import Ellipse
    for (cx, cy, w, h, col, lab) in [
            (2.2, 3.0, 0.7, 3.2, C["dec"], "position-like\n(sharp $x$)"),
            (5.0, 3.0, 1.5, 1.5, C["coh"], "coherent\n(balanced)"),
            (8.0, 3.0, 3.2, 0.7, C["warn"], "momentum-like\n(sharp $\\theta$)")]:
        a.add_patch(Ellipse((cx, cy), w, h, fill=True, alpha=0.25,
                            facecolor=col, edgecolor=col, lw=1.6))
        a.text(cx, {2.2: 5.55, 5.0: 4.55, 8.0: 5.55}[cx], lab, ha="center",
               fontsize=7.6)
    a.annotate("", xy=(9.8, 0.4), xytext=(0.2, 0.4),
               arrowprops=dict(arrowstyle="->", color="k", lw=1))
    a.text(5.0, 0.05, "value $x$", ha="center", fontsize=9)
    a.annotate("", xy=(0.35, 5.9), xytext=(0.35, 0.4),
               arrowprops=dict(arrowstyle="->", color="k", lw=1))
    a.text(0.42, 5.95, r"$\theta$", fontsize=9, ha="left", va="top")
    a.text(5.4, 1.0, r"same area $\hbar_c$ in every shape", ha="center",
           fontsize=8.5, color=C["pur"])
    a.set_title("(A) one cell, three ways to wear it")
    # (B) the two capacity budgets as bars; the dashed line marks the binder.
    # Schematic, no data: the measured crossover lives in fig_budgets.
    b = ax[1]
    b.set_xlim(0, 10); b.set_ylim(0, 4); b.axis("off")
    bind = 5.6                          # right edge of the SHORTER bar
    b.plot([bind, bind], [0.45, 3.25], ls="--", color="k", lw=1.0)
    b.add_patch(plt.Rectangle((0.6, 2.35), 8.7, 0.72, facecolor=C["acc"],
                              alpha=0.30, edgecolor=C["acc"], lw=1.2))
    b.text(0.6, 3.32, r"geometric: up to $R\sigma/\hbar_c$ cells (no $N$)",
           ha="left", fontsize=8.2, color=C["acc"])
    b.add_patch(plt.Rectangle((0.6, 1.05), bind - 0.6, 0.72, facecolor=C["dec"],
                              alpha=0.30, edgecolor=C["dec"], lw=1.2))
    b.text(0.6, 0.62, r"statistical: $\propto N$ items (no $R$, $\sigma$)",
           ha="left", va="center", fontsize=8.2, color=C["dec"],
           bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.2))
    b.text(bind + 0.15, 0.12, "the smaller binds", ha="left", fontsize=8.2)
    b.set_title("(B) capacity: whichever budget runs out first")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scheme_cell.png"))
    plt.close(fig)


def fig_scheme_wigner():
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.12))
    # (A) anatomy of the fringe
    a = ax[0]
    x = np.linspace(-1, 11, 600)
    g1 = np.exp(-(x - 2) ** 2 / 0.5)
    g2 = np.exp(-(x - 8) ** 2 / 0.5)
    cross = 0.9 * np.exp(-(x - 5) ** 2 / 0.7) * np.cos(6 * (x - 5))
    a.fill_between(x, g1, color=C["coh"], alpha=0.35)
    a.fill_between(x, g2, color=C["coh"], alpha=0.35)
    a.plot(x, cross, color=C["dec"], lw=1.4)
    a.axhline(0, color="0.7", lw=0.7)
    a.text(1.4, 1.02, "bump at $x_1$", ha="center", fontsize=7.6)
    a.text(8.6, 1.02, "bump at $x_2$", ha="center", fontsize=7.6)
    # Caption kept inside the axes (extra ylim headroom below): at y=-1.25 with
    # ylim -1.15 it fell outside and the axis line struck its second row.
    a.text(5, -1.30, "cross term at the midpoint:\noscillates, dips negative",
           ha="center", va="top", fontsize=8, color=C["dec"])
    a.set_ylim(-1.55, 1.3); a.set_yticks([])
    a.set_xlabel("value $x$")
    a.set_title("(A) anatomy of a fringe")
    # (B) phasor addition: waves vs dice
    b = ax[1]
    b.set_xlim(-0.5, 10); b.set_ylim(-0.7, 6.4); b.axis("off")
    xc, yc = 0.4, 3.9
    x0c, y0c = xc, yc
    for angd in [0.30, 0.10, 0.42, 0.05, 0.25]:
        dx, dy = 1.45 * np.cos(angd), 1.45 * np.sin(angd)
        b.arrow(xc, yc, dx, dy, head_width=0.17, color=C["coh"],
                length_includes_head=True, lw=1.5)
        xc += dx; yc += dy
    b.annotate("", xy=(xc, yc - 0.35), xytext=(x0c, y0c - 0.35),
               arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0,
                               linestyle="--"))
    b.text(4.2, 6.1, r"in step: amplitudes add, $|\Sigma|$ long",
           fontsize=8.5, color=C["coh"], ha="center")
    xc, yc = 1.4, 1.5
    x0d, y0d = xc, yc
    for angd in [0.4, 2.4, 4.6, 1.2, 3.5]:
        dx, dy = 1.25 * np.cos(angd), 1.25 * np.sin(angd)
        b.arrow(xc, yc, dx, dy, head_width=0.17, color=C["dec"],
                length_includes_head=True, lw=1.5)
        xc += dx; yc += dy
    b.annotate("", xy=(xc, yc), xytext=(x0d, y0d),
               arrowprops=dict(arrowstyle="->", color="0.55", lw=1.0,
                               linestyle="--"))
    b.text(5.9, 0.0, r"scrambled: a random walk, $|\Sigma|$ short",
           fontsize=8.5, color=C["dec"], ha="center")
    b.set_title("(B) coherent vs. decohered")
    # (C) the beat writes the difference
    c = ax[2]
    th = np.linspace(0, 4, 400)
    x1, x2 = 3.0, 5.4
    c.plot(th, 2 + 2 * np.cos(th * (x2 - x1)), color=C["pur"], lw=1.6)
    c.set_xlabel(r"frequency $\theta$")
    c.set_ylabel(r"$|b(\theta)|^2$")
    c.set_title("(C) the beat writes $x_2{-}x_1$")
    # Above the curve (headroom added via ylim) instead of across it.
    c.set_ylim(-0.2, 5.35)
    c.annotate(r"beat period $=2\pi/(x_2{-}x_1)$", (2.05, 4.55),
               fontsize=8, color=C["pur"], ha="center", va="bottom",
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scheme_wigner.png"))
    plt.close(fig)


def fig_scheme_invariance():
    rng = np.random.default_rng(2)
    th = np.abs(4.0 + 1.0 * rng.standard_normal(3000))
    chord = np.array([0.0, 4.0, 7.0, 11.0])
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.30))
    a = ax[0]
    a.set_xlim(-1, 16); a.set_ylim(-0.6, 2.6); a.axis("off")
    for root, y, col in [(0.0, 1.9, C["coh"]), (3.0, 0.6, C["acc"])]:
        notes = chord + root
        a.plot([-0.5, 15.5], [y, y], color="0.8", lw=0.8)
        a.plot(notes, [y] * 4, "o", color=col, ms=7)
        for u, v in zip(notes[:-1], notes[1:]):
            a.annotate("", xy=(v, y + 0.28), xytext=(u, y + 0.28),
                       arrowprops=dict(arrowstyle="<->", color=col, lw=0.9))
            a.text((u + v) / 2, y + 0.36, f"{v-u:g}", ha="center", fontsize=7,
                   color=col)
    a.text(7.5, 2.45, "same intervals (4, 3, 4), different root", ha="center",
           fontsize=8.5)
    a.text(7.5, -0.45, "absolute positions disagree everywhere", ha="center",
           fontsize=8.5, color=C["dec"])
    a.set_title("(A) a chord is its intervals")
    b = ax[1]
    ds = np.linspace(2, 12, 300)
    for root, col, ls, lab in [(0.0, C["coh"], "-", "root at 0"),
                               (3.0, C["acc"], "--", "root at 3")]:
        bd = np.exp(1j * np.outer(chord + root, th)).sum(axis=0)
        sc = (np.cos(np.outer(ds, th)) @ (np.abs(bd) ** 2)) / len(th)
        b.plot(ds, sc, color=col, ls=ls, lw=1.7, label=lab)
    b.set_xlabel("lag $d$"); b.set_ylabel("fingerprint score$(d)$")
    b.legend(frameon=False, fontsize=8)
    b.set_title("(B) their fingerprints coincide")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_invariance.png"))
    plt.close(fig)


def fig_scheme_toolkit():
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.46))
    a = ax[0]
    x = np.linspace(0, 10, 400)
    w = x + 2.2 * np.tanh((x - 5) / 0.9) - 2.2 * np.tanh(-5 / 0.9) * 0
    w = 10 * (w - w[0]) / (w[-1] - w[0])
    a.plot(x, w, color=C["pur"], lw=1.8)
    a.set_xlabel("original coordinate $x$")
    a.set_ylabel("warped coordinate $w(x)$")
    lv = np.linspace(0.3, 9.7, 15)
    xt = np.interp(lv, w, x)
    for xx in xt:
        a.axvline(xx, ymin=0, ymax=0.045, color=C["coh"], lw=1.2)
    a.axvspan(4.1, 5.9, color="0.93")
    a.text(5.0, 8.7, "fovea:\ncells crowd where\n$w$ is steep", ha="center",
           fontsize=8, color=C["coh"])
    a.set_title("(A) squeezing = a coordinate warp", loc="left", fontsize=8.5)
    b = ax[1]
    b.set_xlim(0, 10); b.set_ylim(0, 6); b.axis("off")
    bx = dict(boxstyle="round,pad=0.35", fc="white", ec=C["coh"], lw=1.2)
    b.text(1.6, 3.0, "stream\n$x_1,x_2,\\dots,x_k$", ha="center", va="center",
           fontsize=8.5, bbox=bx)
    b.text(5.0, 3.0, "one bundle\n$b\\in\\mathbb{C}^N$\n(fixed size)",
           ha="center", va="center", fontsize=8.5,
           bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f9", ec=C["coh"],
                     lw=1.4))
    for y, lab in [(5.0, "$|b|^2$: pairwise differences"),
                   (3.0, "$\\mathrm{Re}\\,b^2$: pairwise sums"),
                   (1.0, "$\\overline{b^{(2)}}b^2$: triples")]:
        b.annotate("", xy=(8.0, y), xytext=(6.15, 3.0),
                   arrowprops=dict(arrowstyle="->", color=C["acc"], lw=1.1))
        b.text(8.1, y, lab, fontsize=8.5, va="center")
    b.annotate("", xy=(3.75, 3.0), xytext=(2.85, 3.0),
               arrowprops=dict(arrowstyle="->", color=C["coh"], lw=1.3))
    b.set_title("(B) the bundle as a relational sketch", loc="left",
                fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_toolkit.png"))
    plt.close(fig)


# =============================================================================
# NEW IN R4: the five figures the round-4 review asked for.
# =============================================================================

def fig_ecf():
    """THE KEYSTONE, drawn: a bundle and a histogram-FFT are ONE curve.

    (A) a set of stored values.  (B) two roads to summarise it -- bin and
    transform (grid), or turn each value into a clock hand and sum (bundle).
    (C) the results land on the SAME curve; the grid evaluates it on a regular
    lattice, the bundle samples it at N random frequencies.  That is why there
    is no one-dimensional speedup: the contest is decided by constants.
    """
    rng = np.random.default_rng(11)
    xs_vals = np.sort(rng.uniform(1.0, 9.0, 8))
    NB = 60                                     # bins for the grid route
    edges = np.linspace(0, 10, NB + 1)
    hb, _ = np.histogram(xs_vals, bins=edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])

    fig = plt.figure(figsize=(6.3, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.80, 0.90, 1.55], wspace=0.72)

    # ---- (A) the stored set --------------------------------------------------
    a = fig.add_subplot(gs[0])
    a.vlines(xs_vals, 0, 1, color=C["coh"], lw=1.2)
    a.plot(xs_vals, np.ones_like(xs_vals), "o", color=C["coh"], ms=5, zorder=3)
    a.set_xlim(-0.4, 10.4); a.set_ylim(0, 1.9)
    a.set_yticks([]); a.set_xticks([0, 5, 10])
    a.set_xlabel("value $x$")
    a.set_title("(A) eight stored values", fontsize=8.5)
    for sp in ("left", "right", "top"):
        a.spines[sp].set_visible(False)

    # ---- (B) two roads -------------------------------------------------------
    b = fig.add_subplot(gs[1])
    b.set_xlim(0, 10); b.set_ylim(0, 10); b.axis("off")
    b.set_title("(B) two roads", fontsize=8.5)
    # upper: bin, then transform
    b.text(5.0, 9.5, "bin, then transform", ha="center", fontsize=7.8,
           color=C["alt"])
    hs, _ = np.histogram(xs_vals, bins=np.linspace(0, 10, 21))
    for i, cnt in enumerate(hs):
        if cnt:
            b.add_patch(plt.Rectangle((0.6 + i * 0.42, 6.7), 0.30, 0.62 * cnt,
                                      color=C["alt"], ec="none"))
    b.plot([0.4, 9.4], [6.7, 6.7], color=C["alt"], lw=0.9)
    b.text(5.0, 5.75, r"ECF on a regular grid", ha="center", fontsize=7.2,
           color=C["alt"], style="italic")
    # lower: turn a clock hand, then sum
    b.text(5.0, 4.35, "turn a clock, then sum", ha="center", fontsize=7.8,
           color=C["coh"])
    for i, v in enumerate(xs_vals[:6]):
        cx = 1.1 + 1.56 * i
        b.add_patch(plt.Circle((cx, 2.6), 0.50, fill=False, color=C["coh"], lw=1.0))
        ang = 2.1 * v
        b.plot([cx, cx + 0.42 * np.cos(ang)], [2.6, 2.6 + 0.42 * np.sin(ang)],
               color=C["coh"], lw=1.3)
    b.text(5.0, 1.05, r"ECF at $N$ random $\theta$", ha="center", fontsize=7.2,
           color=C["coh"], style="italic")
    b.plot([0.2, 9.8], [5.25, 5.25], color="k", lw=0.5, ls=":")

    # ---- (C) the same curve --------------------------------------------------
    c = fig.add_subplot(gs[2])
    th = np.linspace(0, 3.0, 900)
    exact = np.array([np.cos(t * xs_vals).sum() for t in th])
    c.plot(th, exact, color=C["alt"], lw=3.4, alpha=0.5,
           label=r"histogram$\to$FFT", solid_capstyle="round")
    th_rand = np.sort(rng.uniform(0, 3.0, 40))
    b_rand = np.array([np.cos(t * xs_vals).sum() for t in th_rand])
    c.plot(th_rand, b_rand, "o", color=C["coh"], ms=3.4, zorder=3,
           label=r"bundle at random $\theta_j$")
    grid_at = np.array([(hb * np.cos(t * ctr)).sum() for t in th_rand])
    dev = float(np.max(np.abs(grid_at - b_rand)))
    c.set_xlabel(r"frequency $\theta$")
    c.set_ylabel(r"$\mathrm{Re}\,b(\theta)$", labelpad=1)
    c.set_title("(C) one curve, two samplings", fontsize=8.5)
    c.set_ylim(-8.6, 11.2)
    c.legend(fontsize=6.6, loc="upper right", framealpha=0.95, handlelength=1.4)
    c.annotate(f"grid vs bundle: $\\leq{dev:.2f}$ of $k={len(xs_vals)}$",
               xy=(0.03, 0.05), xycoords="axes fraction", fontsize=6.6,
               color=C["pur"])
    fig.savefig(os.path.join(OUT, "fig_ecf.png"), bbox_inches="tight")
    plt.close(fig)


def fig_offswitches():
    """NF3: the three off-switches in one row.  The Finding claims three
    mechanisms; the old fig_classical_limit showed only two."""
    st = load("cwf_fpe_stats_results.json")
    ph = load("cwf_fpe_phasespace_results.json")
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.45))

    # (A) ensemble dephasing -- from the multi-seed stats file
    a = ax[0]
    sv = st["seed_variation"]
    conv = st["convergence"]
    NT = [r["NT"] for r in conv]
    frac = [r["frac_removed_matched"] for r in conv]
    a.semilogx(NT, frac, "o-", color=C["coh"], ms=4.0, lw=1.5)
    a.axhline(1.0, color=C["alt"], ls="--", lw=1.0)
    a.text(NT[0] * 1.15, 1.012, "complete removal (analytic)", fontsize=6.5,
           color=C["alt"])
    a.set_ylim(0.80, 1.09)
    a.set_xlabel("draws averaged $N_T$")
    a.set_ylabel("fraction of excess removed")
    a.annotate(r"$%.2f\pm%.2f$ over 8 seeds" % (sv["frac_removed_matched"]["mean"],
                                                sv["frac_removed_matched"]["std"]),
               xy=(0.05, 0.10), xycoords="axes fraction", fontsize=6.4,
               color=C["coh"])
    a.set_title("(A) ensemble dephasing", fontsize=8.2)
    a.text(0.50, 0.30, "no cell-scale boundary", transform=a.transAxes,
           fontsize=6.8, color=C["dec"], ha="center")

    # (B) orthogonal codes
    b = ax[1]
    o = ph["part2a_orthogonality"]
    sg = [r["sigma"] for r in o]; ov = [r["overlap"] for r in o]
    b.loglog(sg, ov, "s-", color=C["warn"], ms=3.6, lw=1.4)
    b.axhline(1 / np.sqrt(4000), color=C["alt"], ls=":", lw=1.0)
    b.text(sg[0] * 1.1, 1 / np.sqrt(4000) * 1.25, r"$N^{-1/2}$ floor",
           fontsize=6.8, color=C["alt"])
    b.set_xlabel(r"bandwidth $\sigma$"); b.set_ylabel(r"$|$overlap$|$")
    b.set_title("(B) orthogonal codes", fontsize=8.2)
    # In the empty pocket right of the descending curve (the old top-center
    # position had the orange line through its second row).
    b.text(0.70, 0.62, "the READOUT stops\nresolving; the state\nstays coherent",
           transform=b.transAxes, fontsize=6.4, color=C["dec"], ha="center",
           va="center")

    # (C) coarse-graining -- the only one with a cell-scale boundary
    c = ax[2]
    h = ph["part2b_husimi"]
    # one pixel of smoothing is (s_x, s_p) = (0.050, 0.131), so s_x s_p = 6.55e-3 n^2;
    # the Husimi threshold is hbar/2 = 0.5 in the units of eq:wigner.
    sx = [max(6.55e-3 * r["smooth"] ** 2 / 0.5, 1e-3) for r in h]
    ng = [max(r["neg"], 1e-7) for r in h]
    c.loglog(sx, ng, "^-", color=C["pur"], ms=3.8, lw=1.4)
    c.axvline(1.0, color=C["acc"], ls=":", lw=1.2)
    c.axvspan(1.0, max(sx) * 1.5, color=C["acc"], alpha=0.10)
    c.text(1.25, max(ng) * 0.02, "positive\nHusimi", fontsize=6.6, color=C["acc"])
    c.set_xlabel(r"smoothing $s_xs_p$ / $(\hbar/2)$")
    c.set_ylabel("negativity")
    c.set_title("(C) coarse-graining", fontsize=8.2)
    c.text(0.34, 0.30, "cell-scale\nboundary", transform=c.transAxes,
           fontsize=6.8, color=C["acc"], ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_offswitches.png"))
    plt.close(fig)


def fig_decoherence():
    """NF4, redesigned: decoherence as a long exposure, watched in the POWER
    SPECTRUM -- where the relational readout actually looks.  The old x-space
    view showed no visible fringes at this separation, so the point it was
    built to make (the beat survives one draw, dies in the average) was
    invisible.  For a two-value bundle |b(theta)|^2 = 2 + 2cos(D*theta + phi)
    exactly, per codebook frequency; a random relative phase phi shifts the
    beat, and averaging phi flattens it to 2."""
    rng = np.random.default_rng(5)
    x1, x2 = 2.0, 5.0
    D = x2 - x1
    th = np.linspace(2.0, 6.0, 700)
    spec = lambda phi: 2.0 + 2.0 * np.cos(D * th + phi)

    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
    ax[0].set_ylim(-0.3, 5.9)
    # (A) coherent: the clean beat
    ax[0].plot(th, spec(0.0), color=C["coh"], lw=1.5)
    pk = 2 * np.pi / D
    ax[0].annotate("", xy=(2 * pk, 4.35), xytext=(pk, 4.35),
                   arrowprops=dict(arrowstyle="<->", color=C["coh"], lw=1.0))
    ax[0].annotate(r"beat period $2\pi/\Delta$:" "\nwhat the relational\nreadout reads",
                   xy=(1.5 * pk, 4.55), fontsize=6.2, color=C["coh"],
                   ha="center", va="bottom")
    ax[0].set_title("(A) coherent bundle", fontsize=8.2)
    ax[0].set_ylabel(r"$|b(\theta)|^2$")
    # (B) one phase-randomized draw: the beat survives, shifted
    phi = rng.uniform(0, 2 * np.pi)
    ax[1].plot(th, spec(0.0), color=C["alt"], lw=0.8, ls="--", alpha=0.55)
    ax[1].plot(th, spec(phi), color=C["coh"], lw=1.5)
    ax[1].annotate("one draw: still coherent,\nbeat moved", xy=(0.5, 0.92),
                   xycoords="axes fraction", fontsize=6.4, color=C["coh"],
                   ha="center", va="top")
    ax[1].set_title("(B) one randomized draw", fontsize=8.2)
    # (C) the 300-draw average: flat at the incoherent level 2
    acc = np.zeros_like(th)
    for i in range(300):
        s = spec(rng.uniform(0, 2 * np.pi))
        acc += s
        if i < 12:
            ax[2].plot(th, s, color=C["alt"], lw=0.5, alpha=0.30)
    ax[2].plot(th, acc / 300, color=C["dec"], lw=1.8)
    ax[2].annotate("ensemble average:\nbeat gone", xy=(0.5, 0.92),
                   xycoords="axes fraction", fontsize=6.4, color=C["dec"],
                   ha="center", va="top")
    ax[2].set_title("(C) average of 300 draws", fontsize=8.2)
    for a in ax:
        a.set_xlabel(r"frequency $\theta$")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decoherence.png"))
    plt.close(fig)


def fig_rulers():
    """NF5: one hyperbola, three rulers.  Defuses the paper's likeliest source
    of reviewer confusion -- three constants for one cell."""
    j = load("cwf_fpe_uncertainty_results.json")
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6))

    # ---- (A) the hyperbola ---------------------------------------------------
    a = ax[0]
    hc = np.sqrt(2 * np.log(2))
    dx = np.logspace(-1.1, 0.95, 300)
    a.loglog(dx, hc / dx, color=C["coh"], lw=2.2, zorder=4,
             label=r"Gaussian: $\Delta x\,\Delta\theta=\hbar_c=1.18$")
    a.loglog(dx, 1.0 / dx, color=C["dec"], lw=1.3, ls="--", zorder=3,
             label=r"proved bound $\geq1$")
    a.fill_between(dx, 1e-3, 1.0 / dx, color=C["dec"], alpha=0.09, zorder=1)
    sw = j["part2_uncertainty"]["gaussian_sweep"]
    a.plot([r["resolution"] for r in sw], [r["sigma"] for r in sw], "s",
           color=C["acc"], ms=4.6, zorder=5, label="measured sweep")
    a.text(0.115, 0.030, "forbidden", fontsize=7.0, color=C["dec"])
    # Below-left of its target point: the old (0.30, 12.0) spot was covered by
    # the opaque upper-right legend.
    a.annotate("wide band\n(position state)", xy=(0.16, hc / 0.16),
               xytext=(0.085, 2.55), fontsize=6.2, ha="left", va="top",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    a.annotate("narrow band\n(momentum state)", xy=(5.5, hc / 5.5),
               xytext=(0.9, 0.06), fontsize=6.2, ha="left",
               arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    a.set_xlabel(r"resolution $\Delta x$"); a.set_ylabel(r"bandwidth $\Delta\theta$")
    a.set_title("(A) choosing a codebook slides you along one curve", fontsize=7.4)
    a.set_ylim(2e-2, 4e1)
    a.legend(fontsize=5.9, loc="upper right", framealpha=0.95)

    # ---- (B) three rulers on one Gaussian ------------------------------------
    b = ax[1]
    u = np.linspace(-3.2, 3.2, 500)
    g = np.exp(-u ** 2 / 2)
    b.plot(u, g, color=C["coh"], lw=1.7, zorder=3)
    hw = np.sqrt(2 * np.log(2))          # HWHM of the amplitude Gaussian
    sd = 1.0                              # its standard deviation
    b.plot([-hw, hw], [0.5, 0.5], color=C["pur"], lw=1.6, marker="|", ms=7, zorder=4)
    b.text(0, 0.425, "HWHM", fontsize=6.6, color=C["pur"], ha="center")
    b.plot([-sd, sd], [np.exp(-0.5), np.exp(-0.5)], color=C["warn"], lw=1.6,
           marker="|", ms=7, zorder=4)
    b.text(0, 0.675, "std", fontsize=6.6, color=C["warn"], ha="center")
    b.axhline(0.5, color=C["alt"], ls=":", lw=0.7, zorder=1)
    b.set_xlim(-3.4, 8.2); b.set_ylim(-0.04, 1.20)
    b.set_yticks([]); b.set_xticks([])
    b.set_title("(B) three rulers, one cell", fontsize=7.4)
    rows = [(r"kernel HWHM $\times$ freq. std", r"$\sqrt{2\ln2}\approx1.18$", C["coh"]),
            (r"intensity std $\times$ std", r"$1/2$", C["warn"]),
            (r"intensity HWHM $\times$ HWHM", r"$\ln2\approx0.69$", C["pur"])]
    for i, (lab, val, col) in enumerate(rows):
        y = 1.02 - 0.24 * i
        b.text(3.9, y, lab, fontsize=6.1, color=col, ha="left")
        b.text(3.9, y - 0.105, val, fontsize=6.6, color=col, ha="left")
    b.text(3.9, 0.20, "one cell, three conventions.", fontsize=6.4, ha="left")
    b.text(3.9, 0.08, "A room is 4 m or 13 feet across.", fontsize=5.9,
           ha="left", style="italic", color=C["alt"])
    for sp in ("left", "right", "top"):
        b.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rulers.png"))
    plt.close(fig)



def fig_resultmap():
    """NF1: the whole paper on one page.  One assumption, three branches, each
    ending at a MEASURED limit.  A 62-page manuscript with 20-odd results and no
    picture of its own shape is asking a lot of a reader."""
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.set_xlim(0, 106); ax.set_ylim(0, 58); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=6.2, bold=False, tc="k"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                zorder=3, color=tc,
                fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, col="k", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0, ls=ls))

    # the single assumption, and the geometry it buys
    box(1, 24, 15, 8, "a VSA vector is\na classical WAVE", "#ffffff", C["coh"],
        fs=6.6, bold=True)
    ax.text(8.5, 21.5, "the only assumption", ha="center", fontsize=5.4,
            style="italic", color=C["alt"])
    box(19, 24, 15, 8, "FPE is a phase\nspace, cell $\\hbar_c$", "#eaf0f6", C["coh"],
        fs=6.6)
    arrow(16.2, 28, 18.8, 28)

    tracks = [
        (44, C["pur"], "IDENTITY",
         [("bundle = ECF\n$\\Rightarrow$ linear sketch", "ID"),
          ("error law $k^2/N$;\nAGMS parity", "ID+EXP"),
          ("third order from a\nsecond accumulator", "EXP")],
         "stops at: no 1-D\nspeedup; separations\nare conditional"),
        (28, C["acc"], "COHERENCE",
         [("Wigner negativity\nwitnesses $\\ell^2$", "EXP"),
          ("interference computes\nall pairwise relations", "EXP"),
          ("decode-free invariant\nrecognition", "EXP")],
         "stops at: three\nmeasured\noff-switches"),
        (12, C["warn"], "GEOMETRY",
         [("squeezing at a matched\nspectral budget", "ID+EXP", 5.0),
          ("velocity templates;\n$(x,p)$ trajectories", "EXP"),
          ("device bit rule;\nRoPE $=$ FPE kernel", "EMU+ID")],
         "stops at: no foveation\nover absolute\npositions"),
    ]
    for y, col, name, nodes, stop in tracks:
        ax.text(40, y + 7.8, name, fontsize=5.6, color=col, ha="left",
                fontweight="bold")
        arrow(34.2, 28, 39.4, y + 3.5, col=col)
        x = 40
        for i, node in enumerate(nodes):
            txt, tag = node[0], node[1]
            box(x, y, 14, 7, txt, "#ffffff", col,
                fs=node[2] if len(node) > 2 else 5.5)
            ax.text(x + 14 - 1.2, y + 0.6, tag, fontsize=4.3, color=col,
                    ha="right", va="bottom", zorder=4)
            if i < len(nodes) - 1:
                arrow(x + 14.2, y + 3.5, x + 15.8, y + 3.5, col=col)
            x += 16
        box(x, y, 14, 7, stop, "#fdeeee", C["dec"], fs=4.6, tc=C["dec"])
        arrow(x - 1.8, y + 3.5, x - 0.2, y + 3.5, col=C["dec"])

    # the one place two branches meet
    arrow(53.5, 35.2, 53.5, 43.6, col=C["alt"], ls=":")
    ax.text(54.8, 39.4, "the same object, read twice", fontsize=5.0,
            color=C["alt"], va="center")

    ax.text(1, 4.0, "TAGS: ID exact algebra $\\cdot$ EXP measured here $\\cdot$ "
                    "EMU device-inspired emulation", fontsize=5.2, color=C["alt"])
    ax.text(1, 0.8, "NOT claimed: quantum computation, or any speedup.",
            fontsize=5.6, color=C["dec"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_resultmap.png"), bbox_inches="tight")
    plt.close(fig)



def fig_decomposition():
    """What the relational estimator actually contains.

    The exact target T(d) is not the pair count: it is signal + diagonal +
    instance background, and finite N adds sampling noise on top.  Every scope
    condition in the paper is one of these terms.  Drawn on the kernel ENVELOPE
    (the carrier oscillation is suppressed for legibility -- it multiplies every
    term alike and does not change their relative sizes).
    """
    sig = 2.0
    K = lambda u: np.exp(-sig ** 2 * u ** 2 / 2)      # envelope
    hw = np.sqrt(2 * np.log(2)) / sig
    rng = np.random.default_rng(4)

    def build(k, s, R, dlag):
        base = np.sort(rng.uniform(0, R - dlag, s))
        rest = rng.uniform(0, R, k - 2 * s)
        return np.concatenate([base, base + dlag, rest])

    def terms(vals, ds, dlag):
        k = len(vals)
        du = vals[:, None] - vals[None, :]
        off = du[~np.eye(k, dtype=bool)]
        is_sig = np.abs(np.abs(off) - dlag) < 1e-9
        sg = np.array([0.5 * (K(off[is_sig] - d) + K(off[is_sig] + d)).sum() for d in ds])
        bg = np.array([0.5 * (K(off[~is_sig] - d) + K(off[~is_sig] + d)).sum() for d in ds])
        dg = np.array([k * K(d) for d in ds])
        return sg, bg, dg

    k, s, dlag = 60, 12, 4.0
    ds = np.linspace(0.0, 8.0, 800)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.75))

    a = ax[0]
    vals = build(k, s, 900.0, dlag)
    sg, bg, dg = terms(vals, ds, dlag)
    a.fill_between(ds, 0, dg, color=C["dec"], alpha=0.45, lw=0,
                   label=r"diagonal $kK(d)$")
    a.fill_between(ds, dg, dg + sg, color=C["coh"], alpha=0.60, lw=0,
                   label=rf"signal ($s={s}$ pairs)")
    a.fill_between(ds, dg + sg, dg + sg + bg, color=C["alt"], alpha=0.50, lw=0,
                   label="instance background")
    a.plot(ds, dg + sg + bg, color="k", lw=1.2)
    nz = k / np.sqrt(4000.0)
    for xq in (2.0, 6.0):
        i = int(np.argmin(np.abs(ds - xq)))
        a.errorbar([xq], [(dg + sg + bg)[i]], yerr=nz, color=C["pur"],
                   capsize=2.5, lw=1.3, zorder=6)
    a.annotate(r"finite-$N$ noise $\pm k/\sqrt{N}$", xy=(5.0, 15.5), fontsize=6.3,
               color=C["pur"])
    a.axvline(dlag, color=C["coh"], ls=":", lw=1.0)
    a.annotate("queried lag", xy=(dlag + 0.15, 20.5), fontsize=6.3, color=C["coh"])
    a.annotate(r"$|K(d)|\ll s/k$ is what", xy=(0.20, 33), fontsize=6.0, color=C["dec"])
    a.annotate("pushes this term down", xy=(0.20, 29.5), fontsize=6.0, color=C["dec"])
    a.set_xlabel("query lag $d$"); a.set_ylabel(r"contribution to $T(d)$")
    a.set_title("(A) what the estimator contains", fontsize=8.4)
    a.legend(fontsize=6.0, loc="upper right", framealpha=0.95)
    a.set_xlim(0, 8); a.set_ylim(0, 42)

    b = ax[1]
    for R, col, lab, st in [(900.0, C["acc"], "sparse range", "-"),
                            (60.0, C["warn"], "dense range", "--")]:
        v = build(k, s, R, dlag)
        _, bg2, _ = terms(v, ds, dlag)
        b.plot(ds, bg2, st, color=col, lw=1.6, label=lab)
    b.axhline(s, color=C["coh"], ls=":", lw=1.3)
    b.annotate(r"signal level $s$", xy=(4.4, s + 0.7), fontsize=6.4, color=C["coh"])
    b.set_xlabel("query lag $d$"); b.set_ylabel("background alone")
    b.set_title("(B) the background does not shrink with $N$", fontsize=8.4)
    b.legend(fontsize=6.2, loc="upper right", framealpha=0.95)
    b.set_xlim(0, 8)
    b.annotate("detection needs BOTH the background\nand the sampling noise below $s$;\n"
               "only the second shrinks with $N$",
               xy=(0.05, 0.58), xycoords="axes fraction", fontsize=6.0, color=C["alt"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_decomposition.png"))
    plt.close(fig)



def fig_fovearesource():
    """Which resource is held fixed decides what foveation is worth.

    The reviewer's Figure D.  Panel A shows why the question arises at all: the
    warped atom's phase turns at theta*w'(x), so its effective spatial frequency
    is w'(x) times the flat arm's, peaking at 2.66 in the fovea.  Panel B shows
    the consequence -- the measured gain under three matched resources.
    """
    j = load("cwf_fpe_fovea_task_results.json")
    rows = j["sweep_matched_resource"]
    diag = j["matched_resource_diagnostics"]["warp_slope"]
    A, B, XF = 0.0, 10.0, 5.0
    xg = np.linspace(A, B, 2000)
    dens = 0.2 + np.exp(-0.5 * ((xg - XF) / 1.0) ** 2)
    cdf = np.cumsum(dens); cdf -= cdf[0]; cdf /= cdf[-1]
    wg = A + (B - A) * cdf
    wp = np.gradient(wg, xg)

    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.7))

    # ---- (A) the warp and the effective slope it multiplies ------------------
    a = ax[0]
    a.plot(xg, wp, color=C["coh"], lw=1.8, label=r"$w'(x)$")
    a.axhline(1.0, color=C["alt"], ls="--", lw=1.1, label="flat arm")
    a.axhline(diag["rms"], color=C["acc"], ls=":", lw=1.3,
              label=rf"RMS $={diag['rms']:.2f}$")
    a.plot([XF], [diag["max"]], "o", color=C["dec"], ms=5, zorder=4)
    a.annotate(rf"$\max w'={diag['max']:.2f}$", xy=(XF, diag["max"]),
               xytext=(XF + 0.4, diag["max"] + 0.18), fontsize=6.6, color=C["dec"])
    a.fill_between(xg, 1.0, wp, where=wp > 1, color=C["dec"], alpha=0.14)
    a.fill_between(xg, wp, 1.0, where=wp < 1, color=C["acc"], alpha=0.12)
    a.set_xlabel("value $x$")
    a.set_ylabel(r"effective slope $/\,\theta$")
    a.set_title(r"(A) the warp multiplies the local frequency", fontsize=8.0)
    a.legend(fontsize=6.0, loc="upper left", framealpha=0.95)
    a.set_ylim(0.2, 3.15)

    # ---- (B) the gain under three matched resources --------------------------
    b = ax[1]
    sig = [r["sigma"] for r in rows]
    xpos = np.arange(len(sig))
    W = 0.26
    series = [("codebook $\\theta_j$", "base", C["alt"]),
              ("RMS eff. slope", "rms", C["coh"]),
              ("max eff. slope", "max", C["dec"])]
    for i, (lab, key, col) in enumerate(series):
        g = [r[f"{key}_fovea_gain"] for r in rows]
        e = [r.get(f"{key}_fovea_gain_std", 0.0) for r in rows]
        b.bar(xpos + (i - 1) * W, g, W, yerr=e, capsize=2, color=col, alpha=0.85,
              label=lab, error_kw=dict(lw=0.8))
    b.axhline(1.0, color="k", lw=1.0, ls="--")
    b.annotate("no gain", xy=(len(sig) - 0.55, 1.05), fontsize=6.2)
    b.set_xticks(xpos); b.set_xticklabels([f"{v:.0f}" for v in sig])
    b.set_xlabel(r"codebook bandwidth $\sigma$")
    b.set_ylabel("fovea gain (flat / squeezed)")
    b.set_title("(B) the answer depends on the budget", fontsize=8.0)
    b.legend(fontsize=6.0, loc="upper right", ncol=1, framealpha=0.95, title="held fixed",
             title_fontsize=6.0)
    b.set_ylim(0, 3.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_fovearesource.png"))
    plt.close(fig)


def fig_collisions():
    """What the translation-invariant fingerprint cannot tell apart.

    The reviewer's Figure E: the power spectrum is an autocorrelation, so it is
    invariant to translation (the point) and blind to reflection and to
    homometric partners (the price).  Measured similarities are the committed
    ones from cwf_fpe_invariant.py.
    """
    j = load("cwf_fpe_invariant_results.json")["collisions"]
    hp = j["homometric_pair"]
    sets = [("original", [0, 1, 4, 10, 12, 17], C["coh"], None),
            ("translated $+6$", [x + 6 for x in hp[0]], C["coh"], j.get("reflection_sim")),
            ("reflected", [17 - x for x in hp[0]][::-1], C["warn"], j["reflection_sim"]),
            ("homometric partner", hp[1], C["dec"], j["homometric_sim"]),
            ("one note moved", [0, 1, 4, 10, 12, 16], C["acc"], j["control_sim"])]
    sig = 5.0                    # narrow enough to resolve unit lags
    K = lambda u: np.exp(-sig ** 2 * u ** 2 / 2)
    ds = np.linspace(0, 20, 900)

    fig, ax = plt.subplots(2, 5, figsize=(6.3, 2.6),
                           gridspec_kw=dict(height_ratios=[0.5, 1.7], hspace=0.30, wspace=0.28))
    for i, (lab, pts, col, sim) in enumerate(sets):
        u = ax[0, i]
        u.plot([-1, 24], [0, 0], color="k", lw=0.7)
        u.plot(pts, np.zeros(len(pts)), "o", color=col, ms=4)
        u.set_xlim(-2, 25); u.set_ylim(-0.6, 0.6)
        u.axis("off")
        u.set_title(lab, fontsize=6.4, pad=2)
        d = ax[1, i]
        arr = np.array(pts, float)
        off = (arr[:, None] - arr[None, :])
        off = off[~np.eye(len(arr), dtype=bool)]
        prof = np.array([0.5 * (K(off - t) + K(off + t)).sum() for t in ds])
        d.fill_between(ds, 0, prof, color=col, alpha=0.55, lw=0)
        d.plot(ds, prof, color=col, lw=1.0)
        d.set_xlim(0, 19); d.set_ylim(0, 4.2)
        d.set_xticks([0, 10, 20]); d.tick_params(labelsize=5.6)
        if i: d.set_yticks([])
        else: d.set_ylabel("fingerprint", fontsize=6.4)
        d.set_xlabel("lag", fontsize=6.4)
        if sim is not None:
            same = sim > 0.999
            d.annotate(f"sim {sim:.3f}", xy=(0.5, 0.90), xycoords="axes fraction",
                       ha="center", fontsize=5.9,
                       color=C["dec"] if same else C["acc"],
                       fontweight="bold" if same else "normal")
            d.annotate("IDENTICAL" if same else "separated", xy=(0.5, 0.76),
                       xycoords="axes fraction", ha="center", fontsize=5.4,
                       color=C["dec"] if same else C["acc"])
    fig.suptitle("the fingerprint is translation-invariant, and not injective",
                 fontsize=8.2, y=1.0)
    fig.savefig(os.path.join(OUT, "fig_collisions.png"), bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# NEW IN R6: six schematics and two result figures.
# =============================================================================

def fig_scheme_decode():
    """What eq:decode produces: an ordinary curve over a grid of candidate
    values, one bump per stored value.  Profiles from an actual seeded FPE
    codebook (N=1500, Gaussian sigma=1.5), so the sidelobes are real."""
    rng = np.random.default_rng(7)
    N, sig = 1500, 1.5
    th = sig * rng.standard_normal(N)
    xs = np.linspace(0, 10, 500)
    grid = np.linspace(0, 10, 26)          # the candidate grid, made visible
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    a, b = ax
    # (A) single value
    a.plot(xs, np.abs(profile(th, 5.0, xs)), color=C["coh"], lw=1.5)
    a.plot(grid, np.full_like(grid, -0.045), "|", color="0.35", ms=5, mew=1.0,
           clip_on=False)
    a.plot(grid, np.full_like(grid, -0.045), ".", color="0.35", ms=1.6,
           clip_on=False)
    a.annotate("evaluated on a\ngrid of candidates", xy=(2.1, -0.035),
               xytext=(0.4, 0.30), fontsize=6.4, color="0.30",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    a.annotate("one stored value $\\rightarrow$ one bump", xy=(5.0, 1.0),
               xytext=(5.6, 1.14), fontsize=7.4, color=C["coh"],
               arrowprops=dict(arrowstyle="->", color=C["coh"], lw=0.8))
    a.text(9.7, 0.55, "an ordinary curve,\nnot a hypervector", fontsize=6.6,
           color=C["alt"], style="italic", ha="right")
    a.set_ylim(-0.08, 1.38)
    a.set_xlabel(r"candidate value $x$")
    a.set_ylabel(r"decoded profile $|\psi(x)|$")
    a.set_title("(A) single value", fontsize=8.5)
    # (B) bundle of three
    vals = (2.0, 5.0, 8.0)
    prof3 = np.abs(sum(profile(th, v, xs) for v in vals))
    b.plot(xs, prof3, color=C["coh"], lw=1.5)
    b.plot(grid, np.full_like(grid, -0.045), "|", color="0.35", ms=5, mew=1.0,
           clip_on=False)
    for v in vals:
        b.annotate("", xy=(v, 1.02), xytext=(v, 1.24),
                   arrowprops=dict(arrowstyle="->", color=C["acc"], lw=0.9))
    b.text(5.0, 1.27, "a bundle $\\rightarrow$ one bump per value",
           fontsize=7.4, color=C["acc"], ha="center", va="bottom")
    b.set_xlabel(r"candidate value $x$")
    b.set_title("(B) bundle of three (2, 5, 8)", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_decode.png"))
    plt.close(fig)


def fig_scheme_weyl():
    """The Weyl phase: translate-then-boost vs boost-then-translate differ by
    e^{i a p0}, the enclosed phase-space area.  Pure schematic."""
    aL, p0 = 3.0, 2.0
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    ax.add_patch(plt.Rectangle((0, 0), aL, p0, facecolor=C["pur"], alpha=0.10,
                               edgecolor="none", zorder=1))
    arr = dict(zorder=3)
    # route 1 (blue): translate right along the bottom, boost up the right side
    ax.annotate("", xy=(aL, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["coh"], lw=1.8,
                                shrinkA=4, shrinkB=2), **arr)
    ax.annotate("", xy=(aL + 0.07, p0), xytext=(aL + 0.07, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["coh"], lw=1.8,
                                shrinkA=2, shrinkB=4), **arr)
    ax.text(1.5, -0.22, "route 1: translate by $a$", ha="center", va="top",
            fontsize=7.2, color=C["coh"])
    ax.text(aL + 0.19, 1.0, "then boost\nby $p_0$", ha="left", va="center",
            fontsize=7.2, color=C["coh"])
    # route 2 (orange): boost up the left side, translate right along the top
    ax.annotate("", xy=(0, p0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.8,
                                shrinkA=4, shrinkB=2), **arr)
    ax.annotate("", xy=(aL, p0 + 0.07), xytext=(0, p0 + 0.07),
                arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=1.8,
                                shrinkA=2, shrinkB=4), **arr)
    ax.text(-0.13, 1.0, "route 2:\nboost first", ha="right", va="center",
            fontsize=7.2, color=C["warn"])
    ax.text(1.5, p0 + 0.20, "then translate", ha="center", va="bottom",
            fontsize=7.2, color=C["warn"])
    # the area, and the phase it buys
    ax.text(aL / 2, p0 / 2, r"area $= a\,p_0$", ha="center", va="center",
            fontsize=9, color=C["pur"])
    ax.text(aL / 2, p0 / 2 - 0.36, r"phase $e^{iap_0}$", ha="center",
            va="center", fontsize=8, color=C["pur"])
    # endpoints
    ax.plot([0], [0], "o", color="k", ms=5, zorder=4)
    ax.text(-0.13, -0.16, "start", ha="right", va="top", fontsize=7.2)
    ax.plot([aL], [p0], "o", color="k", ms=5, zorder=4)
    ax.text(aL + 0.16, p0 + 0.42, "same state,\nphase $e^{iap_0}$", ha="left",
            va="center", fontsize=7.2)
    ax.set_xlim(-1.15, 4.85); ax.set_ylim(-0.75, 2.95)
    ax.set_xticks([0, aL]); ax.set_xticklabels(["$0$", "$a$"])
    ax.set_yticks([0, p0]); ax.set_yticklabels(["$0$", "$p_0$"])
    ax.set_xlabel("$x$", labelpad=1); ax.set_ylabel("$p$", labelpad=1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_weyl.png"))
    plt.close(fig)


def fig_scheme_grid():
    """Why grids pay B^m: a strip, a sheet, a cube -- and the accumulator, one
    N-vector at every m.  Minimal, no data."""
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    ax.set_xlim(0, 12.6); ax.set_ylim(0, 4); ax.set_aspect("equal")
    ax.axis("off")
    cell_fc, cell_ec = "0.93", "0.55"
    # m=1: a strip of 12 cells
    x0, yc, s = 0.35, 2.0, 0.20
    for i in range(12):
        ax.add_patch(plt.Rectangle((x0 + i * s, yc - s / 2), s, s,
                                   facecolor=cell_fc, edgecolor=cell_ec, lw=0.5))
    ax.text(x0 + 6 * s, 0.42, "$m{=}1$: 12 cells", ha="center", fontsize=7.6)
    # m=2: a 12x12 sheet
    x1, y1, S = 3.35, 1.0, 2.0
    ax.add_patch(plt.Rectangle((x1, y1), S, S, facecolor=cell_fc,
                               edgecolor=cell_ec, lw=0.7))
    for i in range(1, 12):
        ax.plot([x1 + i * S / 12] * 2, [y1, y1 + S], color=cell_ec, lw=0.35)
        ax.plot([x1, x1 + S], [y1 + i * S / 12] * 2, color=cell_ec, lw=0.35)
    ax.text(x1 + S / 2, 0.42, "$m{=}2$: 144", ha="center", fontsize=7.6)
    # m=3: a cube sketch (front face + offset top and side)
    x2, y2, F, o = 6.15, 0.85, 1.8, 0.55
    ax.add_patch(plt.Polygon([(x2, y2 + F), (x2 + o, y2 + F + o),
                              (x2 + F + o, y2 + F + o), (x2 + F, y2 + F)],
                             facecolor="0.85", edgecolor=cell_ec, lw=0.7))
    ax.add_patch(plt.Polygon([(x2 + F, y2), (x2 + F + o, y2 + o),
                              (x2 + F + o, y2 + F + o), (x2 + F, y2 + F)],
                             facecolor="0.80", edgecolor=cell_ec, lw=0.7))
    ax.add_patch(plt.Rectangle((x2, y2), F, F, facecolor=cell_fc,
                               edgecolor=cell_ec, lw=0.7))
    for i in range(1, 12):
        ax.plot([x2 + i * F / 12] * 2, [y2, y2 + F], color=cell_ec, lw=0.3)
        ax.plot([x2, x2 + F], [y2 + i * F / 12] * 2, color=cell_ec, lw=0.3)
    for i in range(1, 4):
        d = i * o / 4
        ax.plot([x2 + d, x2 + F + d], [y2 + F + d] * 2, color=cell_ec, lw=0.3)
        ax.plot([x2 + F + d] * 2, [y2 + d, y2 + F + d], color=cell_ec, lw=0.3)
    ax.text(x2 + (F + o) / 2, 0.42, "$m{=}3$: 1,728", ha="center", fontsize=7.6)
    # the ellipsis toward m=6
    ax.text(10.0, 2.0, "$\\cdots$  $m{=}6$:\n$\\approx$3,000,000", ha="center",
            va="center", fontsize=7.6)
    # the accumulator: one tall thin N-vector, the same at every m
    # (cells 5-6 are left as a gap holding the vdots)
    xa, wa, ya, ha = 11.45, 0.34, 1.10, 2.55
    nseg = 12
    for i in range(nseg):
        if i in (5, 6):
            continue
        ax.add_patch(plt.Rectangle((xa, ya + i * ha / nseg), wa, ha / nseg,
                                   facecolor="#dce6f1", edgecolor=C["coh"],
                                   lw=0.6))
    ax.text(xa + wa / 2, ya + ha / 2, r"$\vdots$", ha="center", va="center",
            fontsize=8, color=C["coh"])
    ax.text(xa + wa / 2, ya + ha + 0.24, "$N$ entries", ha="center",
            fontsize=6.8, color=C["coh"])
    ax.text(xa + wa / 2, 0.30, "the accumulator:\n$N$ numbers at every $m$",
            ha="center", va="bottom", fontsize=6.8, color=C["coh"])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_grid.png"))
    plt.close(fig)


def fig_husimi_strip():
    """Coarse-graining watched directly: the two-value bundle's Wigner portrait
    unsmoothed, at a fifth of the cell, and at the Husimi width (non-negative).
    Same state and conventions as fig_wigner."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(0)
    th = (4.0 + 1.0 * rng.standard_normal(3000)); th = th[th > 0]
    xs = np.linspace(-6, 16, 220); dx = xs[1] - xs[0]; M = len(xs)
    cat = profile(th, 2.0, xs) + profile(th, 8.0, xs)
    W = np.fft.fftshift(wigner_ville(cat), axes=1)
    pax = -0.5 * np.fft.fftshift(2 * np.pi * np.fft.fftfreq(M, dx))
    order = np.argsort(pax)
    W = W[:, order]; pax = pax[order]; dp = pax[1] - pax[0]
    pc = pax[np.argmax(np.abs(W).sum(0))]
    keep = np.abs(pax - pc) < 3.5
    panels = [(0.0, "unsmoothed"),
              (0.1, r"smoothing $\approx$ cell/5"),
              (0.5, "at the Husimi width (non-negative)")]
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.15), sharey=True)
    vmax = 6.0
    for axp, (area, ttl) in zip(axs, panels):
        if area > 0:
            s = np.sqrt(area)                     # balanced: s_x = s_p
            Ws = gaussian_filter(W, sigma=(s / dx, s / abs(dp)), mode="nearest")
        else:
            Ws = W
        im = axp.imshow(Ws[:, keep].T, origin="lower", aspect="auto",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        extent=[xs[0], xs[-1], pax[keep][0] - pc,
                                pax[keep][-1] - pc])
        axp.set_title(ttl, fontsize=7.6)
        axp.set_xlabel("value $x$")
    axs[0].set_ylabel(r"frequency $p$ (rel.)")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_husimi_strip.png"))
    plt.close(fig)


def fig_scheme_device():
    """The two device channels acting on one clock hand: ideal, photonic
    jitter (a wobble cone each pass), PCM (snap to q levels + jitter)."""
    from matplotlib.patches import Wedge
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.35))
    ang = np.deg2rad(50.0)
    sig = np.deg2rad(26.0)

    def hand(a, angle, col, lw=2.0, alpha=1.0, r=0.86, ls="-"):
        a.plot([0, r * np.cos(angle)], [0, r * np.sin(angle)], color=col,
               lw=lw, alpha=alpha, ls=ls, solid_capstyle="round", zorder=3)
        a.plot([r * np.cos(angle)], [r * np.sin(angle)], "o", color=col,
               ms=2.6, alpha=alpha, zorder=3)

    for a in ax:
        a.set_xlim(-1.45, 1.45); a.set_ylim(-1.45, 1.45)
        a.set_aspect("equal"); a.axis("off")
        a.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.6", lw=1.0))
        a.plot([0], [0], "o", color="0.4", ms=2.5)
    # (left) ideal
    hand(ax[0], ang, C["coh"])
    ax[0].set_title("ideal phase", fontsize=8.5)
    # (middle) photonic: wobble cone +- sigma_phi, faint jittered hands
    ax[1].add_patch(Wedge((0, 0), 0.98, np.rad2deg(ang - sig),
                          np.rad2deg(ang + sig), color=C["dec"], alpha=0.15,
                          lw=0))
    for d in (-0.80, -0.40, 0.25, 0.55, 0.90):    # a balanced spray in the cone
        hand(ax[1], ang + d * sig, C["dec"], lw=1.0, alpha=0.35)
    hand(ax[1], ang, C["coh"])
    ax[1].annotate(r"$\pm\sigma_\phi$", xy=(1.06 * np.cos(ang + sig),
                   1.06 * np.sin(ang + sig)), fontsize=7.4, color=C["dec"],
                   ha="left", va="bottom")
    ax[1].set_title("photonic: jitter each pass", fontsize=8.5)
    # (right) PCM: q=4 rim ticks, hand snapped to the nearest, small wobble
    q = 4
    for k in range(q):
        tk = 2 * np.pi * k / q
        ax[2].plot([0.90 * np.cos(tk), 1.10 * np.cos(tk)],
                   [0.90 * np.sin(tk), 1.10 * np.sin(tk)], color="0.30",
                   lw=1.6, solid_capstyle="round")
    snap = np.pi / 2                       # nearest tick to the ideal 50 degrees
    ssig = np.deg2rad(9.0)
    hand(ax[2], ang, "0.55", lw=1.1, ls=":")       # the ideal, for reference
    ax[2].add_patch(Wedge((0, 0), 0.98, np.rad2deg(snap - ssig),
                          np.rad2deg(snap + ssig), color=C["dec"], alpha=0.18,
                          lw=0))
    hand(ax[2], snap, C["warn"])
    ax[2].annotate("", xy=(0.62 * np.cos(snap - 0.10),
                           0.62 * np.sin(snap - 0.10)),
                   xytext=(0.62 * np.cos(ang + 0.09),
                           0.62 * np.sin(ang + 0.09)),
                   arrowprops=dict(arrowstyle="->", color=C["warn"], lw=0.9,
                                   connectionstyle="arc3,rad=0.25"))
    ax[2].annotate(f"$q={q}$ levels", xy=(1.12, -1.05), fontsize=7.0,
                   color="0.30", ha="right")
    ax[2].set_title("PCM: snap to $q$ levels + jitter", fontsize=8.5)
    fig.text(0.5, 0.015, r"witness needs $\geq3$ levels $\;\cdot\;$ "
                         "readout survives at 2", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(OUT, "fig_scheme_device.png"))
    plt.close(fig)


def fig_scheme_conjugate():
    """What the conjugate pair buys beyond the reversal test: region queries,
    direction-at-a-crossing, regime templates (session Q25 schematic)."""
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.3))
    th = np.linspace(0, 2 * np.pi, 200)
    R = 2.6
    # (A) region/occupation query + Liouville cell grid
    a = ax[0]
    for g in np.arange(-4, 4.5, 1.0):                       # the area budget
        a.plot([g, g], [-4, 4], color="0.85", lw=0.4, zorder=0)
        a.plot([-4, 4], [g, g], color="0.85", lw=0.4, zorder=0)
    a.add_patch(plt.Rectangle((-4, 0), 4, 4, color=C["acc"], alpha=0.10, zorder=1))
    a.plot(R * np.cos(th), R * np.sin(th), color=C["coh"], lw=1.8, zorder=2)
    seg = th[(np.cos(th) < 0) & (np.sin(th) > 0)]
    a.plot(R * np.cos(seg), R * np.sin(seg), color=C["acc"], lw=3.4, zorder=3)
    a.annotate("", xy=(R * np.cos(2.2), R * np.sin(2.2)),
               xytext=(R * np.cos(2.0), R * np.sin(2.0)),
               arrowprops=dict(arrowstyle="->", color=C["coh"], lw=1.4))
    a.text(-2.05, 3.35, "region: $x<0$, $p>0$", fontsize=6.8, color=C["acc"])
    a.text(-3.7, -3.0, "grid: one $\\hbar_c$-area\ncell per patch", fontsize=6.2,
           color="0.45")
    a.text(-0.15, -1.15, "occupation =\none integral of\nthe portrait", fontsize=6.8,
           color=C["coh"], ha="center")
    a.set_title("(A) region (occupation) queries", fontsize=8.2)
    # (B) direction at a crossing
    b = ax[1]
    b.plot(R * np.cos(th), R * np.sin(th), color="0.7", lw=1.6)
    for t0 in (0.6, 2.2, 3.8, 5.4):
        b.annotate("", xy=(R * np.cos(t0 + 0.12), R * np.sin(t0 + 0.12)),
                   xytext=(R * np.cos(t0), R * np.sin(t0)),
                   arrowprops=dict(arrowstyle="->", color="0.55", lw=1.2))
    b.plot([0], [R], "o", color=C["coh"], ms=8)
    b.plot([0], [-R], "o", mfc="none", mec=C["dec"], ms=8, mew=1.6)
    b.text(0.25, R + 0.15, "probe $(0,+p_0)$: visited", fontsize=6.8,
           color=C["coh"], ha="left")
    b.text(0.25, -R - 0.75, "probe $(0,-p_0)$: not visited", fontsize=6.8,
           color=C["dec"], ha="left")
    b.text(-3.6, -0.2, "contrast\n$+0.996$", fontsize=7.0, color="0.25")
    b.set_title("(B) which way at $x=0$?", fontsize=8.2)
    # (C) regime templates
    c = ax[2]
    for cx, lab, col, direc in [(-3.0, "CW", C["coh"], -1), (0.0, "CCW", C["dec"], 1)]:
        r = 1.05
        c.plot(cx + r * np.cos(th), 0.9 + r * np.sin(th), color=col, lw=1.6)
        t0 = 0.9
        c.annotate("", xy=(cx + r * np.cos(t0 + direc * 0.14),
                           0.9 + r * np.sin(t0 + direc * 0.14)),
                   xytext=(cx + r * np.cos(t0), 0.9 + r * np.sin(t0)),
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        c.text(cx, -0.75, lab, ha="center", fontsize=7.2, color=col)
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(0, 0.3, (28, 2)), 0)
    walk -= walk.mean(0); walk *= 0.55
    c.plot(3.0 + walk[:, 0], 0.9 + walk[:, 1], color=C["alt"], lw=1.0, alpha=0.9)
    c.text(3.0, -0.75, "walk", ha="center", fontsize=7.2, color=C["alt"])
    c.text(0.0, -2.15, "one hypervector $\\to$ nearest portrait", ha="center",
           fontsize=7.0, color="0.25")
    c.set_xlim(-4.6, 4.6); c.set_ylim(-2.7, 2.6)
    c.set_title("(C) regime templates", fontsize=8.2)
    for A in ax:
        A.set_xticks([]); A.set_yticks([])
        if A is not ax[2]:
            A.set_xlim(-4, 4); A.set_ylim(-4, 4)
            A.set_xlabel("$x$", fontsize=7.5); A.set_ylabel("$p$", fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_scheme_conjugate.png")); plt.close(fig)


def fig_scheme_chirp():
    """What motion writes on a bundle: the shear in (x,t), and the sinc
    envelope D(v theta) it leaves across the codebook."""
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.45))
    # (A) the (x,t) plane: static = vertical, moving = tilted (the shear)
    a = ax[0]
    T = 3.0; v = 1.2; x0 = 4.0
    a.plot([2.0, 2.0], [0, T], color=C["coh"], lw=2.0)
    a.text(1.82, 1.5, "static value", rotation=90, ha="right", va="center",
           fontsize=7.6, color=C["coh"])
    a.plot([x0, x0 + v * T], [0, T], color=C["dec"], lw=2.0)
    a.text(6.45, 1.62, "moving value\n$x(t)=x_0+vt$",
           rotation=np.degrees(np.arctan2(1.0, v)) - 14, ha="center",
           va="center", fontsize=7.6, color=C["dec"])
    # the shear angle, against the vertical reference through x0
    a.plot([x0, x0], [0, 2.1], ls=":", color="0.6", lw=0.9)
    tt = np.linspace(np.pi / 2, np.arctan2(1.0, v), 40)
    rr = 1.55
    a.plot(x0 + rr * np.cos(tt), rr * np.sin(tt), color="0.35", lw=0.8)
    a.annotate("shear angle\n$\\propto v$", xy=(x0 + 0.85, 1.62),
               xytext=(2.9, 2.62), fontsize=6.8, color="0.25",
               arrowprops=dict(arrowstyle="->", color="0.45", lw=0.7))
    a.set_xlim(0.6, 8.6); a.set_ylim(-0.15, 3.35)
    a.set_xticks([]); a.set_yticks([])
    a.set_xlabel("value $x$"); a.set_ylabel("time $t$")
    a.set_title("(A) motion is a shear in $(x,t)$", fontsize=8.5)
    # (B) the envelope D(v theta) across the codebook
    b = ax[1]
    th = np.linspace(0, 8, 600); Tw = 1.0
    for vv, col, ls in [(0.0, C["alt"], "-"), (1.0, C["coh"], "-"),
                        (2.0, C["dec"], "--")]:
        D = np.ones_like(th) if vv == 0 else np.sinc(vv * th * Tw / (2 * np.pi))
        b.plot(th, D, color=col, lw=1.7, ls=ls, label=f"$v={vv:g}$")
    b.axhline(0, color="0.85", lw=0.6)
    b.annotate("speed pinches the envelope\nacross the codebook",
               xy=(4.3, 0.62), fontsize=7.0, color=C["dec"], ha="center")
    b.set_ylim(-0.33, 1.24)
    b.set_xlabel(r"codebook frequency $\theta$")
    b.set_ylabel(r"envelope $D(v\theta)$")
    b.set_title("(B) what survives the bundling", fontsize=8.5)
    b.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scheme_chirp.png"))
    plt.close(fig)


def fig_budgets():
    """Which budget binds, measured: k_max(sigma) at fixed R for two N -- the
    statistical plateau, the geometric packing branch, and the crossover
    sigma* that moves with N.  Arm B (the working protocol) throughout."""
    j = load("cwf_fpe_budgets_results.json")
    R = j["params"]["R"]; hc = j["params"]["hbar_c"]
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    sgl = np.logspace(np.log10(0.06), np.log10(30), 60)
    # the two reference lines: measured packing branch, and the ideal bound
    cgeo = np.mean([j["per_N"][n]["summary"]["geometric_slope_c"]
                    for n in ("512", "1024")])
    ax.loglog(sgl, cgeo * R * sgl, ls="--", color="k", lw=1.0,
              label=rf"packing $ {cgeo:.2f}\,R\sigma$ ($0.37\times$ ideal)")
    ax.loglog(sgl, R * sgl / hc, ls=":", color="k", lw=1.0,
              label=r"ideal $R\sigma/\hbar_c$")
    for N, col, mk in [(512, C["coh"], "o"), (1024, C["dec"], "s")]:
        d = j["per_N"][str(N)]
        sg = [r["sigma"] for r in d["rows"] if r["k_max"] > 0]
        km = [r["k_max"] for r in d["rows"] if r["k_max"] > 0]
        ax.loglog(sg, km, mk, ls="-", color=col, ms=4.2, lw=1.3,
                  label=f"$N={N}$")
        s = d["summary"]
        ax.axhline(s["plateau_k"], color=col, ls="--", lw=0.9, alpha=0.75)
        ax.annotate(rf"$\approx{s['plateau_over_N']:.2f}N$",
                    (23.5, s["plateau_k"] * 1.10), fontsize=6.2, color=col,
                    ha="right")
        # the measured crossover sigma*
        ax.annotate(r"$\sigma^*$", xy=(s["sigma_star_measured"],
                                       s["plateau_k"] * 0.92),
                    xytext=(s["sigma_star_measured"], s["plateau_k"] * 0.30),
                    fontsize=7.5, color=col, ha="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    ax.set_xlim(0.06, 30); ax.set_ylim(1.4, 260)
    ax.set_xlabel(r"bandwidth $\sigma$")
    ax.set_ylabel(r"item capacity $k_{\max}$")
    ax.set_title(r"$k_{\max}(\sigma)$ at $R=40$: the smaller budget binds",
                 fontsize=8.8)
    # Opaque background: the N=1024 plateau line crosses the legend region.
    ax.legend(fontsize=6.2, loc="upper left", handlelength=1.7,
              labelspacing=0.35, framealpha=0.95, edgecolor="none")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_budgets.png"))
    plt.close(fig)


def fig_invariant_md():
    """Invariant recognition in m dimensions: all routes stay at 100%
    accuracy, so the separation is cost -- time flat vs ~1,200x, memory
    fixed templates vs the B^m grid."""
    j = load("cwf_fpe_invariant_md_results.json")
    rows = j["per_m"]
    ms = [r["m"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.55))
    a = ax[0]
    for key, lab, col, mk, ls in [
            ("a", "raw spectrum (decode-free)", C["coh"], "o", "-"),
            ("b", "decode-then-classify (grid)", C["pur"], "d", ":"),
            ("c", "reference-unbind (grid)", C["warn"], "*", "-.")]:
        a.semilogy(ms, [r[key]["ms"] for r in rows], marker=mk, ls=ls,
                   color=col, lw=1.6, ms=5, label=lab)
    ratio = rows[-1]["b"]["ms"] / rows[0]["b"]["ms"]
    a.annotate(rf"$\sim\!{ratio/100:.0f}00\times$ by $m{{=}}4$",
               (0.97, 0.44), xycoords="axes fraction", ha="right",
               fontsize=6.8, color=C["pur"])
    a.annotate("all routes 100% accurate: the separation is cost",
               (0.5, 0.03), xycoords="axes fraction", ha="center",
               fontsize=6.2, color=C["alt"])
    a.set_ylim(3e-3, 4e3)
    a.set_xticks(ms)
    a.set_xlabel("value dimension $m$")
    a.set_ylabel("per-classification time (ms)")
    a.set_title("(A) time: only the raw spectrum stays flat", fontsize=8.4)
    a.legend(frameon=False, fontsize=6.6, loc="center left")
    b = ax[1]
    b.semilogy(ms, [r["a"]["mem_MB"] for r in rows], "o-", color=C["coh"],
               lw=1.6, ms=5, label="templates (raw spectrum)")
    b.semilogy(ms, [r["b"]["mem_MB"] for r in rows], "d:", color=C["pur"],
               lw=1.6, ms=5, label=r"decode grid ($B^m$ bins)")
    b.set_xticks(ms)
    b.set_xlabel("value dimension $m$")
    b.set_ylabel("memory (MB)")
    b.set_title("(B) memory: the grid pays $B^m$", fontsize=8.4)
    b.legend(frameon=False, fontsize=6.6, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_invariant_md.png"))
    plt.close(fig)


if __name__ == "__main__":
    for f in [fig_uncertainty, fig_wigner, fig_resource, fig_invariant,
              fig_squeezing, fig_complexity, fig_resonator,
              fig_advantage, fig_sketch, fig_hardware, fig_ams,
              fig_trajectory, fig_chirp, fig_bispectrum, fig_ropefovea,
              fig_scheme_fpe, fig_scheme_cell, fig_scheme_wigner,
              fig_scheme_invariance, fig_scheme_toolkit, fig_ecf, fig_offswitches,
              fig_decoherence, fig_rulers, fig_resultmap,
              fig_decomposition, fig_fovearesource, fig_collisions,
              fig_scheme_decode, fig_scheme_weyl, fig_scheme_grid,
              fig_husimi_strip, fig_scheme_device, fig_scheme_chirp,
              fig_budgets, fig_invariant_md, fig_scheme_conjugate]:
        f(); print("wrote", f.__name__)
    print("all figures in", OUT)
