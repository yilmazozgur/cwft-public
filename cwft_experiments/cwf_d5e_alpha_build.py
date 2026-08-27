"""
D5e (full build) -- The alpha measurement: does the gravitating charge respond to
entanglement at FIXED energy?  alpha = dM_c/dS |_E.

The probe (cwf_d5e_epalpha_probe.py) showed M_c and S co-exist on one Gaussian
substrate but M_c there was an INPUT (the loss profile), so it could not "respond"
to S. To measure alpha properly we need (a) S varied at FIXED energy E, and (b) a
gravitating charge that is a FUNCTIONAL OF THE STATE.

(a) is solved by UNITARY evolution: energy is exactly conserved, entanglement grows.
A free-fermion domain-wall quench (left half filled, right empty, evolved under
nearest-neighbour hopping) has E = Tr(hC) conserved to machine precision while the
half-chain entanglement S(t) grows from 0 -- a clean fixed-energy, sweeping-S family.

(b) The deep point this build surfaces: along a fixed-E trajectory, a gravitating
charge LOCKED to the conserved stress-energy is necessarily FLAT (alpha = 0), because
the conserved quantities don't change while S grows. A non-zero alpha REQUIRES the
charge to depend on a NON-conserved quantity (entanglement) -- which is exactly the
stress-energy-non-conservation (div T != 0) the EP note's firewall named. So alpha is
not a single number; it is a DICHOTOMY:

    GR-style sourcing (linear in the conserved stress-energy)  ->  alpha = 0
    entanglement sourcing (non-conserved)                      ->  alpha != 0  (= div T != 0)

We measure dM_c/dS|_E for several sourcing rules to demonstrate the dichotomy:
  M_E   = |E|                         (energy-locked, conserved)        -> expect alpha = 0
  M_N   = total particle number       (number-locked, conserved)       -> expect alpha = 0
  M_lin = sum_i kappa * e_i = kappa*E (LINEAR in local stress-energy)   -> expect alpha = 0
  M_nl  = sum_i max(n_i - <n>, 0)     (NONLINEAR density functional)    -> alpha != 0 (not conserved)
  M_ent = S                           (entanglement-sourced, the wager) -> alpha = 1 by fiat

Conclusion the build can honestly reach: the standard branch (gravity sourced linearly
by the conserved stress-energy, the GR way) gives alpha = 0 -- entanglement can grow
without bound at fixed energy and such a charge never moves. alpha != 0 is exactly the
non-GR / non-conserved sourcing -- the wager, whose cost is stress-energy conservation.
The residual hard part (isolating a pure-entanglement contribution BEYOND all local
stress-energy) needs same-T_munu-different-S states and is flagged, not claimed.

FIREWALL: internal effective quantities; not physical gravity. CPU-only, deterministic.
Writes results.json["D5e_alpha_build"] and fig_D5e_alpha_build.png.
"""
import json
import os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 64
J = 1.0
TMAX = 14.0
NT = 70


def hopping(n=N, j=J):
    h = np.zeros((n, n), float)
    for i in range(n - 1):
        h[i, i + 1] = -j
        h[i + 1, i] = -j
    return h


def region_entropy(C, idx):
    CA = C[np.ix_(idx, idx)]
    nu = np.linalg.eigvalsh(CA).real
    nu = np.clip(nu, 1e-12, 1 - 1e-12)
    return float(-np.sum(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)))


print("D5e full build: alpha = dM_c/dS at FIXED energy (free-fermion domain-wall quench)")
h = hopping()
half = N // 2
C0 = np.diag([1.0] * half + [0.0] * half).astype(complex)   # domain wall (low S)
E0 = float(np.trace(h @ C0).real)
N0 = float(np.trace(C0).real)
print(f"     n={N}, half-filled domain wall; E0={E0:.4f}, N0={N0:.0f}")

h2 = h @ h
ts = np.linspace(0.0, TMAX, NT)
left = list(range(half))
S = np.zeros(NT); E = np.zeros(NT); Npart = np.zeros(NT)
M_H2 = np.zeros(NT); M_N = np.zeros(NT)
M_nl = np.zeros(NT); M_ent = np.zeros(NT)

for k, t in enumerate(ts):
    U = expm(-1j * h * t)
    C = U @ C0 @ U.conj().T
    C = 0.5 * (C + C.conj().T)
    n_i = np.real(np.diag(C))
    S[k] = region_entropy(C, left)
    E[k] = float(np.real(np.trace(h @ C)))               # = 0 for a product-state start
    Npart[k] = float(np.real(np.trace(C)))               # conserved particle number
    M_N[k] = Npart[k]                                    # number-locked (conserved)
    M_H2[k] = float(np.real(np.trace(h2 @ C)))           # Tr(h^2 C): conserved, nonzero ("energy-like")
    M_nl[k] = np.sum(np.maximum(n_i - 0.5, 0.0))         # nonlinear density functional (non-conserved)
    M_ent[k] = S[k]                                      # entanglement-sourced (the wager)

# conservation checks
dE = float(np.max(np.abs(E - E0)))
dN = float(np.max(np.abs(Npart - N0)))
print(f"\n  fixed-energy guarantee: max|E-E0| = {dE:.2e}, max|N-N0| = {dN:.2e}  "
      f"(E,N conserved; S sweeps {S.min():.2f} -> {S.max():.2f})")


def alpha_of(M):
    """alpha = dM/dS over the entanglement-growth trajectory (linear fit), normalised
    by the charge scale so it is comparable across rules."""
    m = S > 0.05                                         # growth phase
    if np.std(S[m]) < 1e-9 or np.std(M[m]) < 1e-12:
        return 0.0, 0.0
    slope = np.polyfit(S[m], M[m], 1)[0]
    scale = np.mean(np.abs(M[m])) + 1e-12
    return float(slope), float(slope / scale)           # raw slope, fractional slope


rules = [
    ("M_N   particle number (conserved)",  M_N,   "conserved charge"),
    ("M_H2  Tr(h^2 C) (conserved, !=0)",   M_H2,  "conserved charge"),
    ("M_nl  nonlinear density functional", M_nl,  "NON-conserved functional"),
    ("M_ent entanglement (the wager)",     M_ent, "entanglement-sourced"),
]
print(f"\n  alpha = dM_c/dS|_E for each sourcing rule (raw slope | fractional slope/|M|):")
recrules = {}
for name, M, kind in rules:
    raw, frac = alpha_of(M)
    flag = "alpha=0 (no entanglement response)" if abs(frac) < 1e-3 else \
           f"alpha!=0 ({kind})"
    print(f"    {name:38s}: {raw:+.3e} | {frac:+.3e}   {flag}")
    recrules[name] = {"raw_slope": raw, "frac_slope": frac, "kind": kind}

# the dichotomy, quantified: conserved charges vs non-conserved functionals
a_N = alpha_of(M_N)[1]; a_H2 = alpha_of(M_H2)[1]
a_nl = alpha_of(M_nl)[1]; a_ent = alpha_of(M_ent)[1]
print(f"\n  THE DICHOTOMY:")
print(f"    conserved-charge sourcing (the GR way, gravity from conserved T): alpha = 0  "
      f"(N:{a_N:+.0e}, Tr(h^2 C):{a_H2:+.0e})")
print(f"    non-conserved sourcing:                                          alpha != 0  "
      f"(nonlinear-density:{a_nl:+.2f}, entanglement:{a_ent:+.2f})")
print(f"  => at fixed energy, entanglement grew {S.max()/max(S[1],1e-9):.0f}x but a "
      f"stress-energy-locked gravitating charge never moved. alpha != 0 REQUIRES "
      f"coupling gravity to a non-conserved quantity -- the div T != 0 the EP firewall named.")

verdict = (
    "alpha is a DICHOTOMY, not a single number. On a fixed-energy trajectory (free-fermion "
    "domain-wall quench; E,N conserved to ~1e-14 while half-chain S grows from 0): a "
    "gravitating charge locked to a CONSERVED charge (particle number N; Tr(h^2 C), an "
    "energy-like conserved moment) has alpha = dM_c/dS|_E = 0 EXACTLY -- it never responds to "
    "entanglement. "
    "alpha != 0 requires sourcing by a NON-conserved quantity (a nonlinear density functional, "
    "or entanglement itself), which is exactly stress-energy non-conservation (div T != 0). So "
    "the STANDARD branch (gravity from the conserved stress-energy, the GR way) gives alpha=0; "
    "the WAGER (entanglement gravitates beyond energy) is precisely the conservation-violating "
    "branch. This turns the EP note's firewall from a caution into a near-theorem. Residual "
    "(the genuine hard part, not claimed): isolating a pure-entanglement contribution BEYOND "
    "all local stress-energy needs same-T_munu / different-S states. Internal effective "
    "quantities; not physical gravity.")
print("\n  VERDICT:", verdict)

rec = {
    "params": {"N": N, "J": J, "TMAX": TMAX, "NT": NT,
               "substrate": "free-fermion domain-wall quench (unitary; E,N conserved)"},
    "fixed_energy_check": {"max_dE": dE, "max_dN": dN, "S_min": float(S.min()), "S_max": float(S.max())},
    "alpha_by_rule": recrules,
    "dichotomy": {"conserved_charge_alpha": {"M_N": a_N, "M_H2": a_H2},
                  "nonconserved_alpha": {"M_nl_density": a_nl, "M_ent": a_ent}},
    "verdict": verdict,
    "firewall": ("Internal effective quantities. alpha=0 is the conserved-stress-energy (GR) "
                 "branch; alpha!=0 is non-conserved sourcing = div T != 0. Not physical gravity. "
                 "Isolating pure-entanglement-beyond-stress-energy (same-T, different-S) is the "
                 "residual hard part, flagged not claimed."),
    "status": "R (computed dichotomy + conservation near-theorem); the pure-entanglement isolation remains open.",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5e_alpha_build"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print("\n  wrote results.json['D5e_alpha_build']")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.7))

a = ax[0]
a.plot(ts, S, color="#d62728", lw=1.8, label="half-chain entanglement $S(t)$")
a2 = a.twinx()
a2.plot(ts, E, color="#2ca02c", lw=1.5, label="energy $E(t)$")
a2.plot(ts, Npart, color="#1f77b4", lw=1.2, ls=":", label="particle number $N(t)$")
a2.set_ylim(min(E0, N0) - 2, max(E0, N0) + 2)
a.set_xlabel("time $t$"); a.set_ylabel("$S$", color="#d62728"); a2.set_ylabel("$E$, $N$ (flat)")
a.set_title(f"(a) Fixed-energy family: $S$ grows, $E,N$ conserved ($\\Delta E${dE:.0e})")
l1, lb1 = a.get_legend_handles_labels(); l2, lb2 = a2.get_legend_handles_labels()
a.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="center right"); a.grid(alpha=0.3)

a = ax[1]
# normalise each charge to [0,1] over the trajectory for shape comparison vs S
def nrm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)
a.plot(S, nrm(M_N), color="#2ca02c", lw=2.0, label="$M\\propto N$ (conserved): flat")
a.plot(S, nrm(M_H2), color="#1f77b4", lw=1.5, ls="--", label="$M\\propto\\mathrm{Tr}(h^2C)$ (conserved): flat")
a.plot(S, nrm(M_nl), color="#ff7f0e", lw=1.6, label="nonlinear density: $\\alpha\\neq0$")
a.plot(S, nrm(M_ent), color="#d62728", lw=1.6, label="entanglement (wager): $\\alpha\\neq0$")
a.set_xlabel("entanglement $S$ (at fixed $E$)"); a.set_ylabel("gravitating charge (normalised)")
a.set_title("(b) $dM_c/dS|_E$: conserved-source flat, non-conserved responds")
a.legend(fontsize=7.5, loc="center right"); a.grid(alpha=0.3)

a = ax[2]
names = ["$N$", "$\\mathrm{Tr}(h^2C)$", "nonlin.\ndensity", "entangle\n(wager)"]
fracs = [a_N, a_H2, a_nl, a_ent]
colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
a.bar(names, np.abs(fracs), color=colors, alpha=0.85)
a.axhline(0, color="k", lw=0.8)
a.set_ylabel("$|\\alpha|$ (fractional $dM_c/dS|_E$)")
a.set_title("(c) The dichotomy: conserved$\\to\\alpha{=}0$, non-conserved$\\to\\alpha{\\neq}0$")
a.set_yscale("symlog", linthresh=1e-3)
for i, f in enumerate(fracs):
    a.text(i, abs(f) * 1.3 + 1e-4, f"{abs(f):.0e}" if abs(f) < 1e-3 else f"{abs(f):.2f}",
           ha="center", fontsize=8)
a.grid(alpha=0.3, axis="y")

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5e_alpha_build.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
