#!/usr/bin/env python3
"""Publication regeneration of P1's four experiment figures.

  fig_Test6_metric.png   : RECOMPUTED with cwf_test6b.py's exact parameters, seeds,
                           and RNG draw order (validated against results.json[Test6];
                           +-1 site nondeterminism documented in the paper).
  fig_Clausius_2D.png    : RECOMPUTED with cwf_clausius2d.py's exact parameters and
                           seeds (validated against results.json[ClausiusTest2D]).
  fig_A3b2_happy_perfect.png : replotted purely from results.json[A3b2_happy_perfect].
  fig_A3b4_first_law.png     : replotted purely from results.json[A3b4_first_law].

This harness NEVER writes results.json. Styling: no in-figure titles or internal
codenames; captions in the paper carry the numbers.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "..", "cwft_experiments")
OUT = os.path.join(HERE, "MDPI_template_APA", "figures")
sys.path.insert(0, EXP)
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8})

RES = json.load(open(os.path.join(EXP, "results.json")))

# ============================================================ Test 6 (recompute)
from cwf_substrate import ReservoirLattice          # noqa: E402
from cwf_experiments import divergence_field, RNG   # noqa: E402

THETA = 1e-4

def make_well(n, c, w, Lmax):
    x = np.arange(n)
    return Lmax * np.exp(-((x - c) / w) ** 2)

def jacobian_gain_profile(lat, H0, T=400):
    n, m = lat.n, lat.m
    H = H0.copy()
    acc = np.zeros(n); cnt = 0
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
        H = np.tanh(pre)
    return acc / max(cnt, 1)

def transport_reach(n, leak, c, inject_off=-70, m=24, coupling=0.18, rho=2.6,
                    T=1500, seeds=8):
    inj = c + inject_off
    reach = np.zeros(n)
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=970 + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)
        reach += (D > THETA).any(axis=0)
    return reach / seeds, inj

print("[1/4] Test 6 recomputation (identical params/seeds/draw order) ...")
n, c, w, Lmax, rho = 301, 150, 18, 3.0, 2.6
leak = make_well(n, c, w, Lmax)
lat = ReservoirLattice(n, m=24, coupling=0.18, rho=rho, seed=970)
lat.leak = leak
H0 = 0.1 * RNG.standard_normal((n, 24))
g = jacobian_gain_profile(lat, H0, T=400)
reach, inj = transport_reach(n, leak, c, rho=rho)

cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
g_h = int(cross[cross < c][-1]) if np.any(cross < c) else c
reached = np.where(reach >= 0.5)[0]
t_h = int(reached[reached <= c].max()) if np.any(reached <= c) else inj
rec = RES["Test6"]
print(f"    clock-freeze {g_h} (record {rec['g_zero_horizon']}), "
      f"transport {t_h} (record {rec['transport_horizon']}), "
      f"sep {abs(g_h - t_h)} (record {rec['separation_sites']})")
assert abs(g_h - rec["g_zero_horizon"]) <= 1 and abs(t_h - rec["transport_horizon"]) <= 1, \
    "Test6 recomputation drifted beyond the documented +-1-site nondeterminism"

fig, ax = plt.subplots(figsize=(7.0, 3.4), constrained_layout=True)
xs = np.arange(n)
reg = (xs >= c - 80) & (xs <= c + 80)
ax.plot(xs, g, color="tab:blue", lw=1.6, label="gain field $g(x)$ (effective lapse / clock rate)")
ax.axhline(0, color="tab:blue", ls=":", lw=0.8)
ax.plot(xs, reach, color="tab:green", lw=1.6, alpha=0.85, label="transport reach (fraction of probe runs)")
ax.plot(xs, leak / leak.max() * np.nanmax(g[~reg]), color="orange", alpha=0.45, lw=1.2,
        label="damping well (scaled)")
ax.axvline(g_h, color="purple", ls="--", lw=1.3, label=f"clock-freeze $g(x){{=}}0$ (site {g_h})")
ax.axvline(t_h, color="crimson", ls="--", lw=1.3, label=f"transport horizon (site {t_h})")
ax.set_xlim(c - 80, c + 80)
ax.set_xlabel("lattice site"); ax.set_ylabel("gain / reach")
ax.legend(frameon=False, loc="center right")
fig.savefig(os.path.join(OUT, "fig_Test6_metric.png"), dpi=300)
plt.close(fig)
print("    wrote fig_Test6_metric.png")

# ========================================================= Clausius (recompute)
print("[2/4] Clausius 2D recomputation (identical params/seeds) ...")
RNG2 = np.random.default_rng(0)
N2, M2, RHO2, C2 = 61, 10, 2.6, 0.05
_W = RNG2.standard_normal((M2, M2)); _W /= max(abs(np.linalg.eigvals(_W)))
_P = RNG2.standard_normal((M2, M2)); _P /= max(abs(np.linalg.eigvals(_P)))
_EYE = np.eye(M2)
cx = cy = N2 // 2
Y, X = np.mgrid[0:N2, 0:N2]
RAD = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

def well2(Lmax, w):
    return Lmax * np.exp(-(RAD / w) ** 2)

def step2(H, leak):
    nb = (np.roll(H, 1, 0) + np.roll(H, -1, 0) + np.roll(H, 1, 1) + np.roll(H, -1, 1))
    pre = RHO2 * (H @ _W.T) + C2 * (nb @ _P.T) - leak[..., None] * H
    return np.tanh(pre), pre

def gain_field2(leak, T=240, burn=64, sample=8, seed=0):
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((N2, N2, M2))
    Mbase = RHO2 * _W[None, None] - leak[..., None, None] * _EYE[None, None]
    acc = np.zeros((N2, N2)); cnt = 0
    for t in range(T):
        H, pre = step2(H, leak)
        if t >= burn and (t - burn) % sample == 0:
            sech2 = 1.0 - np.tanh(pre) ** 2
            B = sech2[..., :, None] * Mbase
            ev = np.linalg.eigvals(B.reshape(-1, M2, M2))
            sr = np.abs(ev).max(axis=1).reshape(N2, N2)
            acc += np.log(sr + 1e-30); cnt += 1
    return acc / max(cnt, 1)

def radial_avg(field, nbins=34):
    bins = np.linspace(0, RAD.max(), nbins + 1)
    idx = np.digitize(RAD.ravel(), bins) - 1
    f = field.ravel()
    gg = np.array([f[idx == k].mean() if np.any(idx == k) else np.nan for k in range(nbins)])
    rc = 0.5 * (bins[:-1] + bins[1:])
    ok = ~np.isnan(gg)
    return rc[ok], gg[ok]

def measure(Lmax, w, seeds=2):
    leak = well2(Lmax, w)
    gf = np.mean([gain_field2(leak, seed=10 + s) for s in range(seeds)], axis=0)
    rc, gg = radial_avg(gf)
    gs = np.convolve(gg, np.ones(3) / 3, mode="same")
    cr = np.where((gs[:-1] < 0) & (gs[1:] >= 0))[0]
    if cr.size == 0 or gs[0] >= 0:
        return None
    k = cr[0]
    r_h = rc[k] + (rc[k + 1] - rc[k]) * (0 - gs[k]) / (gs[k + 1] - gs[k])
    lo, hi = max(0, k - 3), min(len(rc), k + 5)
    slope = np.polyfit(rc[lo:hi], gs[lo:hi], 1)[0]
    kappa = 0.5 * abs(slope)
    return dict(Lmax=Lmax, w=w, r_h=float(r_h), A=float(2 * np.pi * r_h),
                Tc=float(kappa / (2 * np.pi)),
                Mc=float(np.sum(np.clip(-gf, 0, None)[RAD < r_h])))

widths = [8.0, 11.0, 14.0, 17.0]
depths = [3.9, 4.5, 5.2, 5.9, 6.6, 7.4]
sweep = [m for w_ in widths for L_ in depths if (m := measure(L_, w_))]
allA = np.array([m["A"] for m in sweep]); allM = np.array([m["Mc"] for m in sweep])
p_fit = np.polyfit(np.log(allA), np.log(allM), 1)[0]

etas, pts = [], []
for w_ in widths:
    sl = sorted([m for m in sweep if m["w"] == w_ and m["r_h"] >= 5.0], key=lambda d: d["A"])
    for a, b in zip(sl[:-1], sl[1:]):
        dA = b["A"] - a["A"]; dM = b["Mc"] - a["Mc"]; Tc = 0.5 * (a["Tc"] + b["Tc"])
        if abs(dA) > 1e-3:
            etas.append(dM / (Tc * dA)); pts.append((Tc * dA, dM))
etas = np.array(etas); pts = np.array(pts)
x, y = pts[:, 0], pts[:, 1]
slope_fit = float(np.sum(x * y) / np.sum(x * x))
r2 = float(1 - np.sum((y - slope_fit * x) ** 2) / np.sum((y - y.mean()) ** 2))
eta_cv = float(np.std(etas) / (abs(np.mean(etas)) + 1e-12))
recC = RES["ClausiusTest2D"]
print(f"    p={p_fit:.3f} (record {recC['mass_area_exponent_p']:.3f}), "
      f"R2={r2:.3f} (record {recC['clausius_fit_R2']:.3f}), CV={eta_cv:.3f} (record {recC['eta_cv']:.3f})")
assert abs(p_fit - recC["mass_area_exponent_p"]) < 0.05 and abs(eta_cv - recC["eta_cv"]) < 0.05, \
    "Clausius recomputation drifted from the committed record"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
for w_, col in zip(widths, ["C0", "C1", "C2", "C3"]):
    sl = sorted([m for m in sweep if m["w"] == w_], key=lambda d: d["A"])
    a1.plot([m["A"] for m in sl], [m["Mc"] for m in sl], "o-", ms=4, color=col,
            label=f"well width $w={w_:.0f}$")
Aline = np.linspace(allA.min(), allA.max(), 20)
a1.plot(Aline, allM.mean() * (Aline / allA.mean()) ** p_fit, "k--", lw=1.0,
        label=f"global fit $M_c\\propto A^{{{p_fit:.2f}}}$")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("horizon area $A=2\\pi r_h$"); a1.set_ylabel("computational mass $M_c$")
a1.legend(frameon=False, fontsize=7)
a1.set_title("(a) mass--area scaling: no collapse across widths", fontsize=9)
a2.plot(x, y, "o", ms=5, alpha=0.85)
xl = np.linspace(0, x.max() * 1.05, 50)
a2.plot(xl, slope_fit * xl, "r-", lw=1.2,
        label=(f"first-law fit $dM_c=\\hat\\eta_c\\,T_c\\,dA$\n"
               f"$\\hat\\eta_c\\approx{slope_fit:.0f}$, $R^2={r2:.2f}$, CV$(\\eta_c)={eta_cv:.2f}$"))
a2.set_xlabel("$T_c\\,dA$"); a2.set_ylabel("$dM_c$")
a2.legend(frameon=False, fontsize=7)
a2.set_title("(b) first-law test: scatter, no universal $\\eta_c$", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_Clausius_2D.png"), dpi=300)
plt.close(fig)
print("    wrote fig_Clausius_2D.png")

# ================================================== A3b2 (replot from record)
print("[3/4] A3b2 replot from record ...")
rb = RES["A3b2_happy_perfect"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(rb["by_depth"])))
for col, (dk, dv) in zip(cols, sorted(rb["by_depth"].items())):
    cuts = [p["cut"] for p in dv["points"]]; Ss = [p["S"] for p in dv["points"]]
    jx = (np.random.default_rng(1).uniform(-0.12, 0.12, len(cuts)))
    a1.plot(np.array(cuts) + jx, Ss, "o", ms=3.5, alpha=0.6, color=col,
            label=f"depth {dk.split('=')[1]} ($N={dv['N_total']}$)")
mx = max(p["cut"] for dv in rb["by_depth"].values() for p in dv["points"])
a1.plot([0, mx], [0, mx], "k--", lw=1.0, label="$S(A)=|\\gamma_A|$ (slope 1)")
a1.set_xlabel("bulk min-cut $|\\gamma_A|$ (bonds)"); a1.set_ylabel("boundary entropy $S(A)$ (bits)")
a1.legend(frameon=False, fontsize=6.5)
a1.set_title("(a) entropy = min-cut, all depths", fontsize=9)
depths_n = [int(k.split('=')[1]) for k in sorted(rb["by_depth"])]
a2.plot(depths_n, [rb["slopes"][k] for k in sorted(rb["slopes"])], "s-", ms=6, color="crimson",
        label=f"fitted $\\eta_c$ (mean {rb['slope_mean']:.2f}, CV {rb['slope_CV']*100:.1f}\\%)")
a2.axhline(1.0, color="gray", ls=":", lw=1.0, label="canonical $\\eta_c=1$")
a2.set_ylim(0.9, 1.05)
a2.set_xlabel("network depth"); a2.set_ylabel("fitted RT density $\\eta_c$")
a2.set_xticks(depths_n)
a2.legend(frameon=False, fontsize=7)
a2.set_title("(b) $\\eta_c$ stays near 1 across depths", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_A3b2_happy_perfect.png"), dpi=300)
plt.close(fig)
print("    wrote fig_A3b2_happy_perfect.png")

# ================================================== A3b4 (replot from record)
print("[4/4] A3b4 replot from record ...")
lf = RES["A3b4_first_law"]["linearised"]; gr = RES["A3b4_first_law"]["graded"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.1), constrained_layout=True)
dl = sorted(lf.keys())
fr = [lf[k]["exact_frac"] * 100 for k in dl]
bars = a1.bar(range(len(dl)), fr, color="seagreen", width=0.55)
for b, v in zip(bars, fr):
    a1.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}\\%", ha="center", fontsize=8)
a1.set_xticks(range(len(dl)))
a1.set_xticklabels([f"depth {k.split('=')[1]}\n($N={lf[k]['N_total']}$)" for k in dl], fontsize=8)
a1.set_ylim(90, 102); a1.set_ylabel("exact (excitation, arc) pairs [\\%]")
a1.set_title("(a) linearised first law $\\delta S_A=[\\,b\\in\\mathrm{EW}(A)\\,]$", fontsize=9)
gl = list(gr.keys())
sl_ = [gr[k]["slope"] for k in gl]; ef = [gr[k]["exact_frac"] * 100 for k in gl]
xpos = np.arange(len(gl))
a2.plot(xpos, sl_, "D-", ms=7, color="navy", label="fitted response slope")
a2.axhline(1.0, color="gray", ls=":", lw=1.0, label="ideal slope 1")
for xv, sv, ev in zip(xpos, sl_, ef):
    a2.text(xv, sv - 0.018, f"{ev:.0f}\\% exact", ha="center", fontsize=7, color="navy")
a2.set_xticks(xpos); a2.set_xticklabels([k for k in gl], fontsize=8)
a2.set_ylim(0.88, 1.03)
a2.set_xlabel("configuration"); a2.set_ylabel("slope of $\\delta S_A$ vs $|B\\cap\\mathrm{EW}(A)|$")
a2.legend(frameon=False, fontsize=7)
a2.set_title("(b) graded / full-FLM variant", fontsize=9)
fig.savefig(os.path.join(OUT, "fig_A3b4_first_law.png"), dpi=300)
plt.close(fig)
print("    wrote fig_A3b4_first_law.png")
print("All four figures regenerated; results.json untouched.")
