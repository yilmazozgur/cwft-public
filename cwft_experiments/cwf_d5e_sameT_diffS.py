"""
D5e (residual) -- the same-T, different-S isolation: does the gravitating charge respond
to entanglement BEYOND all local stress-energy?

The alpha-build (cwf_d5e_alpha_build.py) found alpha=0 for conserved-stress-energy sourcing
but varied S at fixed TOTAL energy while the LOCAL stress-energy rearranged. The genuine
residual: hold ALL local stress-energy fixed and vary ONLY entanglement.

KEY: for free fermions the state is the correlation matrix C (C_ij=<c_i^d c_j>). The stress-
energy is LOCAL -- density n_i=C_ii (diagonal) and bond energy e_i~Re(C_{i,i+1}) (first off-
diagonal); total E=Tr(hC) with h nearest-neighbour depends only on the first off-diagonal.
Entanglement S(A) reads C_A at ALL ranges. So scaling only the range>=2 elements of a state
holds density, bond energy AND total E BIT-IDENTICAL while changing S -- a genuine
same-T / different-S family, exact wherever C stays a valid correlation matrix.

We start from a FINITE-TEMPERATURE state (eigenvalues strictly interior), not a pure ground
state: a pure state's C is a projector (eigenvalues 0/1) on the boundary of the valid set,
where ANY off-diagonal perturbation is invalid (the first attempt at this gave a single valid
point and a vacuous null -- recorded honestly). At beta=2 a ~10% S-swing opens up at exactly
fixed T.

For each candidate gravitating-charge functional M_c[C] we report its RELATIVE VARIATION
across the same-T family (max-min)/mean|M|:
  - any functional of the LOCAL data (density, bond energy, range-1 correlations) -> 0 EXACTLY
    (it is a function of bit-identical inputs), so alpha_pure = dM_c/dS|_T = 0 by construction;
  - the region entropy S(A) and longer-range correlation functionals -> nonzero (they read the
    entanglement difference).

(Note: at half-filling the correlations vanish on EVEN separations -- C_{i,i+r}=0 for even r,
the sin(k_F r)/r structure with k_F=pi/2 -- so the range-2 functional is trivially invariant
and the entanglement response first enters at odd range r>=3.)

CONCLUSION: on a genuine same-T family where S varies by ~10%, every stress-energy functional
is EXACTLY invariant; only non-local (entanglement) functionals respond. So a gravitating
charge sourced by T_munu is exactly blind to entanglement at fixed T; entanglement gravitating
beyond energy (alpha_pure != 0) requires the charge to be a non-local functional that does NOT
derive from the stress-energy -- non-GR / div T != 0 sourcing. There is no stress-energy
channel for entanglement to gravitate. This closes the EP note's residual.

FIREWALL: internal effective quantities; not physical gravity. CPU-only, deterministic.
Writes results.json["D5e_sameT_diffS"] and fig_D5e_sameT_diffS.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 48
J = 1.0
BETA = 2.0


def hopping(n=N, j=J):
    h = np.zeros((n, n), float)
    for i in range(n - 1):
        h[i, i + 1] = -j
        h[i + 1, i] = -j
    return h


def thermal_corr(h, beta):
    """C = 1/(e^{beta h}+1); eigenvalues strictly in (0,1) -> interior of the valid set."""
    w, V = np.linalg.eigh(h)
    f = 1.0 / (np.exp(beta * w) + 1.0)
    return (V * f) @ V.conj().T


def scale_longrange(C0, lam):
    """Scale all range->=2 elements by lam; keep diagonal + first off-diagonal EXACT."""
    n = C0.shape[0]
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    return C0 * np.where(dist >= 2, lam, 1.0)


def region_entropy(C, idx):
    nu = np.linalg.eigvalsh(C[np.ix_(idx, idx)]).real
    nu = np.clip(nu, 1e-12, 1 - 1e-12)
    return float(-np.sum(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)))


def is_valid(C, tol=1e-9):
    ev = np.linalg.eigvalsh(0.5 * (C + C.conj().T)).real
    return bool(ev.min() >= -tol and ev.max() <= 1 + tol)


def corr_weight(C, dmax):
    n = C.shape[0]; idx = np.arange(n); dist = np.abs(idx[:, None] - idx[None, :])
    return float(np.sum(np.abs(C[(dist >= 1) & (dist <= dmax)]) ** 2))


print("D5e residual: same-T, different-S isolation (free-fermion correlation matrix)")
h = hopping()
C0 = thermal_corr(h, BETA)
left = list(range(N // 2))
E0 = float(np.real(np.trace(h @ C0)))
n0 = np.real(np.diag(C0)).copy()
e0 = np.array([np.real(C0[i, i + 1]) for i in range(N - 1)])
ev0 = np.linalg.eigvalsh(C0)
print(f"     n={N}, thermal beta={BETA}; E0={E0:.6f}, occ in [{ev0.min():.3f},{ev0.max():.3f}] "
      f"(interior); region A = left half")

lams = np.linspace(2.0, 0.0, 121)
rows = []
for lam in lams:
    C = scale_longrange(C0, lam)
    if not is_valid(C):
        continue
    rows.append(dict(
        lam=lam, S=region_entropy(C, left), E=float(np.real(np.trace(h @ C))),
        dn=float(np.max(np.abs(np.real(np.diag(C)) - n0))),
        de=float(np.max(np.abs(np.array([np.real(C[i, i + 1]) for i in range(N - 1)]) - e0))),
        M_dens=float(np.sum(np.abs(np.real(np.diag(C)) - 0.5))),
        M_bond=float(np.sum(np.abs([np.real(C[i, i + 1]) for i in range(N - 1)]))),
        M_r1=corr_weight(C, 1), M_r2=corr_weight(C, 2), M_r4=corr_weight(C, 4),
        M_rAll=corr_weight(C, N)))

lam_v = np.array([r["lam"] for r in rows]); S = np.array([r["S"] for r in rows])
E = np.array([r["E"] for r in rows])
dn = np.array([r["dn"] for r in rows]); de = np.array([r["de"] for r in rows])
swing_pct = 100 * (S.max() - S.min()) / S.max()
print(f"\n  valid same-T family: {len(rows)} states, lambda in [{lam_v.min():.3f},{lam_v.max():.3f}]")
print(f"  EXACT stress-energy hold:  max|dE|={np.max(np.abs(E - E0)):.2e}, "
      f"max density drift={dn.max():.2e}, max bond drift={de.max():.2e}  (bit-identical local T)")
print(f"  entanglement genuinely varies:  S in [{S.min():.4f},{S.max():.4f}]  "
      f"(swing {S.max() - S.min():.4f} = {swing_pct:.1f}% of S)")


def rel_var(key):
    M = np.array([r[key] for r in rows])
    return float((M.max() - M.min()) / (np.mean(np.abs(M)) + 1e-15))


cands = [
    ("M_dens  density functional (local T)",            "M_dens", "stress-energy (local)"),
    ("M_bond  bond-energy functional (local T)",        "M_bond", "stress-energy (local)"),
    ("M_r1    range-1 correlation (local)",             "M_r1",   "stress-energy (local)"),
    ("M_r2    range<=2 correlation",                    "M_r2",   "local (r=2 vanishes @ half-fill)"),
    ("M_r4    range<=4 correlation (reads r=3)",        "M_r4",   "non-local"),
    ("M_rAll  all-range correlation (non-local)",       "M_rAll", "non-local"),
]
print(f"\n  relative variation of each charge across the same-T family (max-min)/mean|M|:")
recc = {}
for label, key, kind in cands:
    rv = rel_var(key)
    flag = "EXACTLY invariant (alpha_pure=0)" if rv < 1e-12 else f"responds ({kind})"
    print(f"    {label:46s}: {rv:.3e}   {flag}")
    recc[label] = {"rel_var": rv, "kind": kind}
rv_S = float((S.max() - S.min()) / np.mean(S))
print(f"    {'M_S     region entropy S(A) (RT/encoding charge)':46s}: {rv_S:.3e}   responds (entanglement)")
recc["M_S region entropy"] = {"rel_var": rv_S, "kind": "entanglement"}

a_dens, a_bond, a_r1 = rel_var("M_dens"), rel_var("M_bond"), rel_var("M_r1")
a_r4, a_rAll = rel_var("M_r4"), rel_var("M_rAll")
print(f"\n  RESIDUAL CLOSED (exact construction):")
print(f"    same-T held bit-identical (dE={np.max(np.abs(E-E0)):.0e}, drift={max(dn.max(),de.max()):.0e}) "
      f"while S varied {swing_pct:.1f}%.")
print(f"    every LOCAL (stress-energy) functional is EXACTLY invariant: "
      f"dens {a_dens:.0e}, bond {a_bond:.0e}, range-1 {a_r1:.0e}.")
print(f"    only NON-LOCAL functionals respond: range<=4 {a_r4:.2e}, all-range {a_rAll:.2e}, S {rv_S:.2e}.")
print(f"    => a T_munu-sourced gravitating charge is EXACTLY blind to entanglement at fixed T; "
      f"alpha_pure != 0 requires non-local (non-T) sourcing = div T != 0. No stress-energy channel.")

verdict = (
    f"RESIDUAL CLOSED. Genuine same-T / different-S states constructed for free fermions by scaling "
    f"only the range>=2 correlations of a beta={BETA} thermal state: density, bond energy and total "
    f"energy stay BIT-IDENTICAL (drift 0 to machine precision) while the region entropy S varies by "
    f"{swing_pct:.0f}% over the valid (correlation-matrix-positive) range. On this exactly-fixed-T "
    f"family, EVERY local (stress-energy) functional is EXACTLY invariant -- alpha_pure=dM_c/dS|_T=0 "
    f"by construction (a function of bit-identical inputs) -- while the region entropy S and "
    f"longer-range (r>=3; even ranges vanish at half-filling) correlation functionals respond. So a "
    f"gravitating charge sourced by the stress-energy is exactly blind to entanglement at fixed T; "
    f"entanglement gravitating beyond energy (alpha_pure!=0) requires the charge to be a non-local "
    f"functional that does NOT derive from T_munu -- non-GR / stress-energy-non-conserving sourcing. "
    f"There is no stress-energy channel for entanglement to gravitate. This closes the EP note's "
    f"same-T/different-S residual and completes the conservation near-theorem. Honest scope: the "
    f"free-fermion S-lever at fixed uniform T is intrinsically modest (~10%), because the thermal "
    f"entropy dominates; the invariance conclusion is exact-by-construction and does not depend on "
    f"the swing magnitude. Internal effective quantities; not physical gravity.")
print("\n  VERDICT:", verdict)

rec = {
    "params": {"N": N, "J": J, "beta": BETA, "region": "left half",
               "construction": "scale range>=2 correlations of a thermal state; diagonal + first "
                               "off-diagonal exact; restrict to valid (PSD, <=I)"},
    "same_T_check": {"max_dE": float(np.max(np.abs(E - E0))), "max_density_drift": float(dn.max()),
                     "max_bond_drift": float(de.max()), "n_states": len(rows),
                     "lambda_range": [float(lam_v.min()), float(lam_v.max())]},
    "entanglement_swing": {"S_min": float(S.min()), "S_max": float(S.max()), "swing_pct": swing_pct},
    "rel_var_by_functional": recc,
    "summary": {"local_stress_energy": {"dens": a_dens, "bond": a_bond, "range1": a_r1},
                "nonlocal": {"range4": a_r4, "allrange": a_rAll, "entropy": rv_S}},
    "first_attempt_note": "a pure ground state (projector, boundary of valid set) gave a single valid "
                          "point and a vacuous null; the finite-T (interior) start is required.",
    "verdict": verdict,
    "firewall": ("Internal effective quantities. Same-T is bit-identical; alpha_pure=0 for stress-energy "
                 "sourcing is exact-by-construction; alpha_pure!=0 needs non-local (non-T_munu) sourcing "
                 "= div T != 0. Not physical gravity."),
    "status": "R (exact same-T construction + exact invariance of stress-energy functionals); residual closed.",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5e_sameT_diffS"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print("\n  wrote results.json['D5e_sameT_diffS']")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.7))
a = ax[0]
a.plot(lam_v, S, color="#d62728", lw=1.9, label=f"region entropy $S(A)$ ({swing_pct:.0f}% swing)")
a2 = a.twinx()
a2.plot(lam_v, np.abs(E - E0) + 1e-17, color="#2ca02c", lw=1.4, label="$|E-E_0|$")
a2.plot(lam_v, dn + 1e-17, color="#1f77b4", lw=1.2, ls=":", label="density drift")
a2.set_yscale("log"); a2.set_ylim(1e-18, 1e-1)
a.set_xlabel("$\\lambda$ (range$\\geq$2 correlation scale)"); a.set_ylabel("$S(A)$", color="#d62728")
a2.set_ylabel("stress-energy drift (log)", color="#2ca02c")
a.set_title("(a) Same-T family: $S$ varies, stress-energy bit-identical")
l1, lb1 = a.get_legend_handles_labels(); l2, lb2 = a2.get_legend_handles_labels()
a.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="center left"); a.grid(alpha=0.3)

a = ax[1]
def nrm(x):
    x = np.array(x); rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-14 else np.zeros_like(x)
order = np.argsort(S)
a.plot(S[order], nrm([rows[i]["M_dens"] for i in order]), color="#2ca02c", lw=2.4, label="density (local T): flat")
a.plot(S[order], nrm([rows[i]["M_bond"] for i in order]), color="#1f77b4", lw=1.6, ls="--", label="bond energy (local T): flat")
a.plot(S[order], nrm([rows[i]["M_r4"] for i in order]), color="#ff7f0e", lw=1.7, label="range$\\leq$4 corr: responds")
a.plot(S[order], nrm([rows[i]["M_rAll"] for i in order]), color="#d62728", lw=1.7, label="all-range corr: responds")
a.set_xlabel("entanglement $S(A)$ (at EXACTLY fixed $T$)"); a.set_ylabel("charge (normalised)")
a.set_title("(b) $dM_c/dS|_T$: local-T blind, non-local responds")
a.legend(fontsize=7.5, loc="center right"); a.grid(alpha=0.3)

a = ax[2]
names = ["dens", "bond", "range-1", "range$\\leq$4", "all-range", "$S$"]
rv = [a_dens, a_bond, a_r1, a_r4, a_rAll, rv_S]
cols = ["#2ca02c", "#2ca02c", "#2ca02c", "#ff7f0e", "#d62728", "#d62728"]
a.bar(names, [max(v, 1e-17) for v in rv], color=cols, alpha=0.85)
a.set_yscale("log"); a.set_ylim(1e-17, 1.0)
a.axhline(1e-12, color="k", ls=":", lw=0.8, label="machine zero")
a.set_ylabel("relative variation across same-T family")
a.set_title("(c) Residual closed: local-T exactly invariant, non-local responds")
a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
plt.setp(a.get_xticklabels(), rotation=20, fontsize=8)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5e_sameT_diffS.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
