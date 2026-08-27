"""
D5e (probe) -- Feasibility for the EP-alpha / horizon-wedge experiment:
can ONE Gaussian substrate carry BOTH the gain-deficit gravitating charge M_c
(the H_g side) AND an entanglement-wedge structure S(A) (the H_EW side), and do
the two horizons coincide?

WHY a dissipative Gaussian substrate. M_c is the gain DEFICIT (g<0, contracting).
A closed UNITARY system has |eigenvalues of the propagator| = 1, so g == 0 and
M_c == 0 -- which is exactly why the book puts H_g on DISSIPATIVE reservoirs and
H_EW on UNITARY codes (conj:horizon-wedge: they live on different substrate
classes; never yet on one object). To carry both we need a NON-HERMITIAN
(lossy) free-fermion chain: a localized loss well gives a gain-deficit M_c, and
the post-selected (no-click) correlation matrix still gives entanglement S(A).

SETUP. 1D tight-binding chain, n sites, half filled. H_eff = h - i*diag(gamma),
h = nearest-neighbour hopping (Hermitian), gamma(x) = a localized loss WELL
(gamma>0 = loss = local contraction). Occupied orbitals evolve Phi(t) =
expm(-i H_eff t) Phi(0); the Gaussian correlation matrix is the projector onto
the (renormalised) occupied subspace, C = Phi (Phi^d Phi)^{-1} Phi^d. Density
n_i = C_ii; entanglement of an interval from the eigenvalues of the block C_A
(Peschel).

QUANTITIES.
  gain field   g_i = -gamma_i        (on-site leading-order local-Jacobian gain;
                                       loss -> g<0, the framework's convention)
  M_c          = sum_i max(-g_i, 0)  (gain-deficit gravitating charge = the well)
  S(cut k)     = entanglement across cut k (pure state -> = bipartite entropy)

FEASIBILITY QUESTIONS (the point of the probe):
  (1) Is M_c well-defined and localized in the loss well?  (the H_g side)
  (2) Is S(A) well-defined and sensible?                   (the H_EW side)
  (3) Does the loss well IMPRINT on the entanglement -- i.e. is there an
      entanglement bottleneck (S(cut) dip) AT the well, so the gain horizon H_g
      coincides with an entanglement feature H_EW?
  (4) Coupling pre-look: as the well deepens (M_c grows), does the entanglement
      bottleneck deepen too? (a first hint that M_c and entanglement are linked
      on one object -- NOT yet the fixed-energy dM_c/dS measurement, which is the
      hard full experiment.)

This is a VIABILITY probe, not the alpha measurement. Outcome decides whether the
Gaussian route is worth the full build. FIREWALL: internal effective quantities;
not physical gravity. CPU-only, seeded-free (deterministic). Writes
results.json["D5e_epalpha_probe"] and fig_D5e_epalpha_probe.png.
"""
import json
import os
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 80
FILL = 0.5
J = 1.0
CENTER = N // 2
WIDTH = 6.0
T = 4.0


def hopping(n=N, j=J):
    h = np.zeros((n, n), complex)
    for i in range(n - 1):
        h[i, i + 1] = -j
        h[i + 1, i] = -j
    return h


def loss_profile(gmax, n=N, c=CENTER, w=WIDTH):
    x = np.arange(n)
    return gmax * np.exp(-((x - c) / w) ** 2)


def correlation_matrix(gamma):
    """Post-selected non-Hermitian free-fermion correlation matrix at time T."""
    h = hopping()
    Heff = h - 1j * np.diag(gamma)
    M = int(round(FILL * N))
    # initial half-filled Fermi sea of the Hermitian part
    w, V = np.linalg.eigh(h)
    Phi0 = V[:, :M]                       # lowest M orbitals (n x M), orthonormal
    U = expm(-1j * Heff * T)
    Phi = U @ Phi0                        # evolve occupied orbitals (non-unitary)
    # correlation matrix = projector onto the renormalised occupied subspace
    G = Phi.conj().T @ Phi
    C = Phi @ np.linalg.inv(G) @ Phi.conj().T
    C = 0.5 * (C + C.conj().T)            # hermitise (numerical)
    return C


def region_entropy(C, idx):
    CA = C[np.ix_(idx, idx)]
    nu = np.linalg.eigvalsh(CA).real
    nu = np.clip(nu, 1e-12, 1 - 1e-12)
    return float(-np.sum(nu * np.log(nu) + (1 - nu) * np.log(1 - nu)))


def cut_profile(C, n=N):
    """S([0,k)) for k=1..n-1 -- bipartite entanglement across each cut."""
    return np.array([region_entropy(C, list(range(k))) for k in range(1, n)])


print("D5e probe: can ONE Gaussian substrate carry both M_c (gain-deficit) and S(A)?")
print(f"     n={N}, half-filled, hopping J={J}, loss well center={CENTER} width={WIDTH}, T={T}")

# --- baseline (no well, unitary) vs a loss well ---
gamma0 = loss_profile(0.0)
gammaW = loss_profile(1.0)
C0 = correlation_matrix(gamma0)
CW = correlation_matrix(gammaW)
dens0 = np.real(np.diag(C0))
densW = np.real(np.diag(CW))
cut0 = cut_profile(C0)
cutW = cut_profile(CW)

g_field = -gammaW
M_c = float(np.sum(np.maximum(-g_field, 0.0)))
# density depletion in the well region
wellmask = np.abs(np.arange(N) - CENTER) <= WIDTH
depletion = float(np.mean(dens0[wellmask]) - np.mean(densW[wellmask]))
# entanglement bottleneck: S(cut) at the well vs baseline
kc = CENTER - 1                                  # cut index at the well centre
S_base_at_well = float(cut0[kc])
S_well_at_well = float(cutW[kc])
bottleneck = S_base_at_well - S_well_at_well

print("\n  (1) M_c side (gain-deficit):")
print(f"      M_c = sum max(-g,0) = {M_c:.2f}  (localized in the loss well)")
print(f"      density depletion in the well: {depletion:+.3f} "
      f"(the gain-deficit core freezes/empties)  -> H_g is real")
print("  (2) S(A) side (entanglement):")
print(f"      baseline mid-cut entropy {S_base_at_well:.3f}, with-well {S_well_at_well:.3f}  -> H_EW is real")
print("  (3) horizon coincidence:")
print(f"      entanglement bottleneck at the well = {bottleneck:+.3f} "
      f"({'DIP' if bottleneck > 0.02 else 'no clear feature'}) "
      f"-> the gain well {'IMPRINTS on' if bottleneck > 0.02 else 'does NOT imprint on'} the entanglement")

# --- (4) coupling pre-look: sweep well depth; M_c vs the entanglement bottleneck ---
gmaxes = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
Mcs = []; botts = []
for gm in gmaxes:
    g = loss_profile(gm)
    C = correlation_matrix(g)
    Mcs.append(float(np.sum(np.maximum(g, 0.0))))         # M_c = sum gamma
    botts.append(S_base_at_well - float(cut_profile(C)[kc]))
Mcs = np.array(Mcs); botts = np.array(botts)
# correlation between M_c and the bottleneck across the sweep
corr = float(np.corrcoef(Mcs, botts)[0, 1]) if np.std(botts) > 0 else float("nan")
print("\n  (4) coupling pre-look (deepen the well):")
for gm, mc, bt in zip(gmaxes, Mcs, botts):
    print(f"      gamma_max={gm:>4.2f}:  M_c={mc:6.2f}   entanglement bottleneck={bt:+.3f}")
print(f"      corr(M_c, bottleneck) = {corr:+.3f}  "
      f"(M_c and the entanglement feature are {'COUPLED' if corr > 0.7 else 'weakly/uncoupled'} via the well)")

viable = (M_c > 0) and (S_base_at_well > 0) and (bottleneck > 0.02)
verdict = (
    f"VIABLE: a single non-Hermitian free-fermion substrate carries BOTH a gain-deficit "
    f"M_c={M_c:.1f} (localized in the loss well, density depletes there) AND a sensible "
    f"entanglement structure; the gain well IMPRINTS an entanglement bottleneck "
    f"(dip {bottleneck:+.2f}), and deepening the well couples M_c to the bottleneck "
    f"(corr {corr:+.2f}). So H_g and an H_EW feature CAN coincide on one Gaussian object -> "
    f"the conj:horizon-wedge / EP-alpha route is worth the full build. NOT the alpha "
    f"measurement: that needs perturbing entanglement at FIXED energy and reading dM_c/dS. "
    f"Internal effective quantities; not physical gravity."
) if viable else (
    f"NOT cleanly viable on this setup: M_c={M_c:.1f}, bottleneck={bottleneck:+.2f}. The loss "
    f"well does not imprint clearly on the entanglement -> try a different open-Gaussian "
    f"design (Lindblad steady state, or gain+loss) before committing to the full build."
)
print("\n  VERDICT:", verdict)

rec = {
    "params": {"N": N, "fill": FILL, "J": J, "center": CENTER, "width": WIDTH, "T": T,
               "substrate": "non-Hermitian (lossy) free-fermion chain, post-selected no-click"},
    "M_c_side": {"M_c": M_c, "well_density_depletion": depletion},
    "S_side": {"baseline_mid_cut_entropy": S_base_at_well, "with_well_mid_cut_entropy": S_well_at_well},
    "horizon_coincidence": {"entanglement_bottleneck_at_well": bottleneck,
                            "imprints": bool(bottleneck > 0.02)},
    "coupling_prelook": {"gamma_max": gmaxes, "M_c": Mcs.tolist(),
                         "bottleneck": botts.tolist(), "corr_Mc_bottleneck": corr},
    "viable": bool(viable), "verdict": verdict,
    "firewall": ("Internal effective quantities (gain-deficit M_c; entanglement S(A)) on one "
                 "substrate; NOT physical gravity. This is a viability probe, not the alpha "
                 "(dM_c/dS at fixed energy) measurement."),
    "status": "R (computed feasibility); the full EP-alpha experiment remains the open frontier.",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5e_epalpha_probe"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print("\n  wrote results.json['D5e_epalpha_probe']")

# ---------------- figure ----------------
xs = np.arange(N)
fig, ax = plt.subplots(2, 2, figsize=(12.2, 8.4))

a = ax[0, 0]
a.plot(xs, gammaW, color="#d62728", lw=1.6, label="loss $\\gamma(x)$ = gain-deficit well")
a.fill_between(xs, gammaW, color="#d62728", alpha=0.2)
a.set_xlabel("site"); a.set_ylabel("$\\gamma(x)=-g(x)$")
a.set_title(f"(1) The gain-deficit well: $M_c={M_c:.1f}$ (the $H_g$ side)")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[0, 1]
a.plot(xs, dens0, color="#1f77b4", lw=1.3, label="no well (unitary)")
a.plot(xs, densW, color="#d62728", lw=1.6, label="with loss well")
a.axvspan(CENTER - WIDTH, CENTER + WIDTH, color="grey", alpha=0.15, label="well")
a.set_xlabel("site"); a.set_ylabel("density $n_i$")
a.set_title(f"(1) Density depletes in the well ({depletion:+.2f}): the core freezes")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 0]
kk = np.arange(1, N)
a.plot(kk, cut0, color="#1f77b4", lw=1.3, label="no well")
a.plot(kk, cutW, color="#d62728", lw=1.6, label="with loss well")
a.axvline(CENTER, color="k", ls="--", lw=0.8, label="well centre ($H_g$)")
a.set_xlabel("cut position $k$"); a.set_ylabel("entanglement $S([0,k))$")
a.set_title(f"(3) Entanglement bottleneck at the well ({bottleneck:+.2f}): $H_g$ imprints on $H_{{EW}}$")
a.legend(fontsize=8); a.grid(alpha=0.3)

a = ax[1, 1]
a.plot(Mcs, botts, "o-", color="#9467bd", lw=1.4, ms=6)
for gm, mc, bt in zip(gmaxes, Mcs, botts):
    a.annotate(f"{gm:.2g}", (mc, bt), fontsize=7, xytext=(3, 3), textcoords="offset points")
a.set_xlabel("$M_c$ (gain-deficit, deepening well)"); a.set_ylabel("entanglement bottleneck")
a.set_title(f"(4) $M_c$ couples to the entanglement feature (corr {corr:+.2f})")
a.grid(alpha=0.3)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5e_epalpha_probe.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
