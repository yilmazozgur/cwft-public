"""
The computational Clausius test (spherically-symmetric reduction).

We use the robust 1D gain physics (rho=2.6, as in Test 6, which gives an
expanding exterior g>0 and a clean g<0 damped core) and interpret the spatial
coordinate as a RADIUS in d dimensions. For each radial damping well we measure
three INDEPENDENT functionals of the gain field g(r):

  horizon radius r_h : outermost g(r)=0 crossing
  area      A   = Omega_{d-1} r_h^{d-1}        (Omega_1=2pi, Omega_2=4pi)
  temp      T_c = kappa_c/(2pi),  kappa_c=(1/2)|dg/dr|_{r_h}   (hbar_c:=1)
  mass      M_c = Integral_{g<0} (-g) Omega_{d-1} r^{d-1} dr   (gain deficit)

First law / Clausius:  dM_c = T_c dS_c,  S_c = eta_c A.
Non-circular test: eta_c = dM_c/(T_c dA) must be the SAME constant across a
2-parameter family (varying well depth Lmax AND width w). Constant eta_c =>
Clausius holds => computational Einstein equation grounded; G_c=1/(4 hbar_c eta_c).

Caveat: spherically-symmetric reduction approximates the d-dim dynamics by the
1D chain gain profile with a radial measure (drops the 1/r Laplacian term). It
is a first-pass test of whether the (M,A,T) functionals satisfy a first law.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
NX, M, RHO, C = 130, 24, 2.6, 0.18      # half-chain length (radius), internal dim
DDIM = 3                                # spatial dimension for area (Omega_2=4pi)
OMEGA = {2: 2 * np.pi, 3: 4 * np.pi}[DDIM]
_W = RNG.standard_normal((M, M)); _W /= max(abs(np.linalg.eigvals(_W)))
_P = RNG.standard_normal((M, M)); _P /= max(abs(np.linalg.eigvals(_P)))

def gain_profile(leak, T=360, burn=80, sample=4, seed=0):
    """1D-chain time-averaged log spectral radius of each site's local Jacobian."""
    n = leak.size
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((n, M))
    acc = np.zeros(n); cnt = 0
    for t in range(T):
        left = np.zeros_like(H); right = np.zeros_like(H)
        left[1:] = H[:-1]; right[:-1] = H[1:]
        nb = (left + right) @ _P.T
        pre = RHO * (H @ _W.T) + C * nb - leak[:, None] * H
        H = np.tanh(pre)
        if t >= burn and (t - burn) % sample == 0:
            s2 = 1 - np.tanh(pre) ** 2
            for i in range(n):
                B = s2[i][:, None] * (RHO * _W - leak[i] * np.eye(M))
                acc[i] += np.log(max(abs(np.linalg.eigvals(B))) + 1e-30)
            cnt += 1
    return acc / max(cnt, 1)

def measure(Lmax, w, seeds=6):
    r = np.arange(NX, dtype=float)            # radius = site index from r=0
    leak = Lmax * np.exp(-(r / w) ** 2)        # radial damping well
    g = np.mean([gain_profile(leak, seed=20 + s) for s in range(seeds)], axis=0)
    # light smoothing to stabilise crossing/slope
    ker = np.ones(3) / 3.0
    gs = np.convolve(g, ker, mode="same")
    # INNERMOST g=0 crossing (core is negative at r=0): core boundary = horizon
    cr = np.where((gs[:-1] < 0) & (gs[1:] >= 0))[0]
    if cr.size == 0 or gs[0] >= 0:
        return None
    k = cr[0]
    r_h = k + (0 - gs[k]) / (gs[k + 1] - gs[k])
    # surface gravity: robust linear fit of g(r) over a window around the horizon
    lo, hi = max(0, k - 3), min(NX, k + 6)
    slope = np.polyfit(r[lo:hi], gs[lo:hi], 1)[0]
    kappa = 0.5 * abs(slope)
    A = OMEGA * r_h ** (DDIM - 1)
    Tc = kappa / (2 * np.pi)
    # mass = gain deficit INSIDE the horizon only (radial measure)
    interior = r < r_h
    dV = OMEGA * np.clip(r, 1e-6, None) ** (DDIM - 1)
    Mc = float(np.sum(np.clip(-gs, 0, None)[interior] * dV[interior]))
    return dict(Lmax=Lmax, w=w, r_h=float(r_h), A=float(A), kappa=float(kappa),
                Tc=float(Tc), Mc=Mc)

print(f"Computational Clausius test (d={DDIM} spherical reduction):")
sweep = []
widths = [9.0, 12.0, 15.0, 18.0]
depths = [2.2, 2.7, 3.2, 3.8, 4.5, 5.3]
for w in widths:
    for Lmax in depths:
        m = measure(Lmax, w)
        if m:
            sweep.append(m)
            print(f"  w={w:4.1f} Lmax={Lmax:.1f}  r_h={m['r_h']:6.2f}  A={m['A']:8.1f}"
                  f"  T_c={m['Tc']:.4f}  M_c={m['Mc']:9.1f}")

# ---- Analysis 1: mass-area scaling law M_c ~ A^p (GR Schwarzschild d=3: p=0.5) ----
allA = np.array([m["A"] for m in sweep]); allM = np.array([m["Mc"] for m in sweep])
allrh = np.array([m["r_h"] for m in sweep]); allT = np.array([m["Tc"] for m in sweep])
p_fit, logA0 = np.polyfit(np.log(allA), np.log(allM), 1)
print(f"\n  mass-area scaling:  M_c ~ A^{p_fit:.2f}   (GR Schwarzschild d=3 would be p=0.5)")

# ---- Analysis 2: first-law eta_c = dM_c/(T_c dA), all vs well-formed horizons ----
def collect(rh_min):
    etas, pts = [], []
    for w in widths:
        sl = sorted([m for m in sweep if m["w"] == w and m["r_h"] >= rh_min],
                    key=lambda d: d["A"])
        for a, b in zip(sl[:-1], sl[1:]):
            dA = b["A"] - a["A"]; dM = b["Mc"] - a["Mc"]; Tc = 0.5 * (a["Tc"] + b["Tc"])
            if abs(dA) > 1e-3:
                etas.append(dM / (Tc * dA)); pts.append((Tc * dA, dM))
    return np.array(etas), np.array(pts)

for tag, rhmin in [("ALL black holes", 0.0), ("well-formed (r_h>=8)", 8.0)]:
    etas, pts = collect(rhmin)
    if etas.size < 2:
        continue
    cv = float(np.std(etas) / (abs(np.mean(etas)) + 1e-12))
    xx, yy = pts[:, 0], pts[:, 1]
    slope = float(np.sum(xx * yy) / np.sum(xx * xx))
    r2 = 1 - float(np.sum((yy - slope * xx) ** 2)) / float(np.sum((yy - yy.mean()) ** 2))
    print(f"  [{tag}]  eta_c median={np.median(etas):.0f}  CV={cv:.2f}  "
          f"fit slope={slope:.0f}  R^2={r2:.2f}  (n={etas.size})")

# ---- Analysis 3: per-width-slice eta_c (is it consistent across slices?) ----
print("  per-slice eta_c (well-formed):")
slice_etas = []
for w in widths:
    sl = sorted([m for m in sweep if m["w"] == w and m["r_h"] >= 8.0],
                key=lambda d: d["A"])
    es = [(b["Mc"] - a["Mc"]) / (0.5 * (a["Tc"] + b["Tc"]) * (b["A"] - a["A"]))
          for a, b in zip(sl[:-1], sl[1:]) if abs(b["A"] - a["A"]) > 1e-3]
    if es:
        slice_etas.append(np.median(es))
        print(f"    w={w:4.1f}:  eta_c={np.median(es):.0f}  (n={len(es)})")

# headline numbers from well-formed set
etas_wf, pts_wf = collect(8.0)
eta_med = float(np.median(etas_wf)); eta_cv = float(np.std(etas_wf) / (abs(np.mean(etas_wf)) + 1e-12))
xx, yy = pts_wf[:, 0], pts_wf[:, 1]
slope_fit = float(np.sum(xx * yy) / np.sum(xx * xx))
r2 = 1 - float(np.sum((yy - slope_fit * xx) ** 2)) / float(np.sum((yy - yy.mean()) ** 2))
G_c = 1.0 / (4 * eta_med) if eta_med else None

if eta_cv < 0.25 and r2 > 0.85:
    verdict = "CLAUSIUS HOLDS for well-formed horizons (near-universal eta_c)"
elif eta_cv < 0.5:
    verdict = "PARTIAL: approximate first law for well-formed horizons; breaks near threshold"
else:
    verdict = "Clausius does not hold cleanly (analog kinematics without Einstein dynamics)"
print(f"\n  G_c = 1/(4 hbar_c eta_c) = {G_c:.3g}")
print(f"  VERDICT: {verdict}")

res = json.load(open("results.json"))
res["ClausiusTest"] = {"d_dim": DDIM, "n_blackholes": len(sweep), "hbar_c": 1.0,
                       "mass_area_exponent_p": float(p_fit),
                       "eta_median_wellformed": eta_med, "eta_cv_wellformed": eta_cv,
                       "clausius_fit_slope": slope_fit, "clausius_fit_R2": r2,
                       "G_c": G_c, "verdict": verdict}
json.dump(res, open("results.json", "w"), indent=2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
for w, col in zip(widths, ["C0", "C1", "C2", "C3"]):
    sl = sorted([m for m in sweep if m["w"] == w], key=lambda d: d["A"])
    a1.plot([m["A"] for m in sl], [m["Mc"] for m in sl], "o-", color=col, label=f"w={w:.0f}")
a1.set_xlabel("horizon area  $A$"); a1.set_ylabel("computational mass  $M_c$")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_title(f"Mass vs area: $M_c\\sim A^{{{p_fit:.2f}}}$ (GR: 0.5)"); a1.legend(); a1.grid(alpha=0.3, which="both")
a2.plot(xx, yy, "o", ms=6, alpha=0.8, label="well-formed pairs")
xl = np.linspace(0, xx.max() * 1.05, 50)
a2.plot(xl, slope_fit * xl, "r-", label=f"$dM_c=\\eta_c T_c dA$\n$\\eta_c$={slope_fit:.0f}, $R^2$={r2:.2f}, CV={eta_cv:.2f}")
a2.set_xlabel("$T_c\\, dA$"); a2.set_ylabel("$dM_c$"); a2.grid(alpha=0.3)
a2.set_title("Clausius first-law test"); a2.legend()
plt.tight_layout(); plt.savefig("fig_Clausius_test.png", dpi=130); plt.close()
print("Wrote fig_Clausius_test.png")
