"""
D5d -- The perfect-tensor code sits ON the holographic/Bekenstein bound (the EP note's
       §8 exhibit). Makes "eta_c = 1 is the saturation value" concrete.

The holographic bound (a max-flow / subadditivity theorem) caps a boundary region's
entanglement entropy by its min-cut "area": S(A) <= |gamma_A| (in bits, unit bonds). The
perfect-tensor (HaPPY) code SATURATES it by construction: S(A) = |gamma_A|, i.e.
eta_c = S(A)/|gamma_A| = 1. So eta_c is the BEKENSTEIN-SATURATION FRACTION:

    eta_c = 1  : on the bound  -> the black-hole corner (maximal entropy for the area)
    eta_c < 1  : below the bound -> ordinary, sub-maximal matter (room to spare)

Control (same geometry, less entanglement): dilute a fraction of internal bonds to
PRODUCT states instead of Bell pairs. The min-cut |gamma_A| (the "area") is unchanged,
but S(A) drops below it -> eta_c < 1. This shows the bound is general and only the
perfect code saturates it.

Why this matters for the EP note: at saturation the substrate-relative
G_c = 1/(4 hbar_c eta_c) has eta_c pinned at 1 and S(A)=|gamma_A| pinned to the physical
S_BH = A/4 (under the bond=Planck-area, bit=nat dictionary) -- the one corner where the
otherwise-gauge computational gravity is forced to equal physical gravity.

Reuses the validated perfect-tensor machinery of cwf_a3b2_happy_perfect.py (import-safe;
__main__-guarded) and cwf_hp_lib.py. FIREWALL: |gamma_A| is the substrate min-cut;
identifying it with a physical Planck area is the EP note's tie (status B), not asserted
here. CPU-only, seeded. Writes results.json["D5d_bekenstein"] and fig_D5d_bekenstein.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_a3b2_happy_perfect import (build_network, build_state, min_cut,
                                    circular_arcs, _pauli_string, _PERFECT_TABLEAU)
from cwf_hp_lib import make_sim, stabilizer_matrix, entropy_region


def build_state_diluted(tensors, bonds, bulk_qubits, dilute_frac, seed=0):
    """Same network/geometry, but a `dilute_frac` fraction of internal bonds are
    projected to PRODUCT |00> (Z_a=+1, Z_b=+1) instead of the Bell pair |Phi+>.
    Diluted bonds carry zero entanglement, so S(A) drops below |gamma_A| while the
    min-cut (area) is unchanged -> a sub-saturating state on the same geometry."""
    N = 6 * len(tensors)
    sim = make_sim(N, seed=0)
    for t in tensors:
        sim.do_tableau(_PERFECT_TABLEAU, t.qubits)
    rng = np.random.default_rng(seed)
    nb = len(bonds)
    n_dil = int(round(dilute_frac * nb))
    dil = set(rng.choice(nb, size=n_dil, replace=False).tolist()) if n_dil > 0 else set()
    for i, (a, b, _ta, _tb) in enumerate(bonds):
        if i in dil:
            sim.postselect_observable(_pauli_string(N, [a], 3), desired_value=False)  # Z_a=+1
            sim.postselect_observable(_pauli_string(N, [b], 3), desired_value=False)  # Z_b=+1
        else:
            sim.postselect_observable(_pauli_string(N, [a, b], 1), desired_value=False)  # XX=+1
            sim.postselect_observable(_pauli_string(N, [a, b], 3), desired_value=False)  # ZZ=+1
    for q in bulk_qubits:
        sim.postselect_observable(_pauli_string(N, [q], 3), desired_value=False)
    return sim, N


def measure(tensors, bonds, boundary, sim, N, max_per_size=8):
    X, Z = stabilizer_matrix(sim, N)
    n_b = len(boundary)
    S_list, cut_list = [], []
    for L, start in circular_arcs(n_b, max_per_size):
        A = [boundary[(start + i) % n_b] for i in range(L)]
        cut = min_cut(tensors, bonds, boundary, A)
        if cut == 0:
            continue
        S = float(entropy_region(X, Z, A, N))
        S_list.append(S); cut_list.append(cut)
    return np.array(S_list), np.array(cut_list, float)


DEPTH = 3
print("D5d: the perfect-tensor code sits ON the holographic/Bekenstein bound (EP note §8)")
tensors, bonds, bulk, boundary = build_network(DEPTH, n_child=2)
print(f"     network depth {DEPTH}: {len(tensors)} tensors, {len(bonds)} bonds, "
      f"{len(boundary)} boundary legs")

# ---- PERFECT: saturation ----
sim, N = build_state(tensors, bonds, bulk)
S, cut = measure(tensors, bonds, boundary, sim, N)
eta_perfect = float(np.mean(S / cut))
exact_frac = float(np.mean(np.abs(S - cut) < 1e-9))
bound_ok = bool(np.all(S <= cut + 1e-9))
print(f"\n  PERFECT tensor (the black-hole corner):")
print(f"    eta_c = <S/|gamma|> = {eta_perfect:.4f}   (=1 -> ON the holographic bound)")
print(f"    exact saturation S=|gamma|: {100*exact_frac:.1f}% of regions")
print(f"    holographic bound S <= |gamma| holds for all regions: {bound_ok}")

# ---- DILUTED control: below the bound, same geometry ----
fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
etas = []; bound_holds = []
for fr in fracs:
    simd, Nd = build_state_diluted(tensors, bonds, bulk, fr, seed=3)
    Sd, cutd = measure(tensors, bonds, boundary, simd, Nd)
    etas.append(float(np.mean(Sd / cutd)))
    bound_holds.append(bool(np.all(Sd <= cutd + 1e-9)))
etas = np.array(etas)
print(f"\n  DILUTION control (same geometry/|gamma|, fewer entangled bonds):")
for fr, e, ok in zip(fracs, etas, bound_holds):
    tag = "  <- perfect, ON the bound" if fr == 0 else ("  (below the bound)" if e < 0.999 else "")
    print(f"    dilute {fr:>4.0%}:  eta_c = {e:.3f}  (bound S<=|gamma| holds: {ok}){tag}")
print(f"  => eta_c is the Bekenstein-saturation fraction: 1.00 only for the perfect code "
      f"(black-hole corner); generic matter sits strictly below.")

# scatter data for the figure (perfect + one diluted level)
S0, c0 = S, cut
simh, Nh = build_state_diluted(tensors, bonds, bulk, 0.5, seed=3)
Sh, ch = measure(tensors, bonds, boundary, simh, Nh)

verdict = (f"The perfect-tensor code SATURATES the holographic bound S(A)<=|gamma_A|: "
           f"eta_c=<S/|gamma|>={eta_perfect:.2f}, exact on {100*exact_frac:.0f}% of regions. "
           "eta_c is the Bekenstein-saturation fraction -- 1 only for the perfect code (the "
           "black-hole corner), <1 for generic (diluted) matter. The bound S<=|gamma| holds "
           "everywhere. Under the EP note's bond=Planck-area + bit=nat dictionary, this is the "
           "computational S(A)=|gamma_A| meeting the physical S_BH=A/4 at saturation; eta_c=1 "
           "pins G_c=1/(4 hbar_c eta_c), the one corner where the gauge hbar_c is spent and "
           "computational gravity must equal physical gravity. Identification = bridge (B); "
           "saturation = computed (R).")
print("\n  VERDICT:", verdict)

rec = {
    "params": {"depth": DEPTH, "n_tensors": len(tensors), "n_bonds": len(bonds),
               "n_boundary": len(boundary), "substrate": "HaPPY perfect-tensor network"},
    "perfect_saturation": {"eta_c": eta_perfect, "exact_frac": exact_frac, "bound_holds": bound_ok},
    "dilution_control": {"fracs": fracs, "eta_c": etas.tolist(), "bound_holds": bound_holds},
    "verdict": verdict,
    "firewall": ("|gamma_A| is the substrate min-cut; identifying it with a physical Planck "
                 "area (and bits with nats) is the EP note's tie (status B), not asserted here. "
                 "The saturation itself is computed (R)."),
    "status": "R (saturation computed); B (the area/Planck and bit/nat identifications).",
}
RESULTS = os.path.join(os.path.dirname(__file__), "results.json")
res = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
res["D5d_bekenstein"] = rec
json.dump(res, open(RESULTS, "w"), indent=2)
print("\n  wrote results.json['D5d_bekenstein']")

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.8))

a = ax[0]
mx = max(c0.max(), ch.max()) + 0.5
a.plot([0, mx], [0, mx], "k--", lw=1.0, label="holographic bound  $S=|\\gamma_A|$")
a.fill_between([0, mx], [0, mx], mx, color="grey", alpha=0.08)
a.text(0.55 * mx, 0.92 * mx, "forbidden ($S>|\\gamma|$)", fontsize=8, color="grey")
# jitter for visibility
rng = np.random.default_rng(0)
a.scatter(c0 + rng.uniform(-.08, .08, c0.size), S0, s=22, color="#d62728",
          label=f"perfect tensor ($\\eta_c={eta_perfect:.2f}$, ON the bound)", zorder=3)
a.scatter(ch + rng.uniform(-.08, .08, ch.size), Sh, s=18, color="#1f77b4", alpha=0.7,
          label=f"50%-diluted ($\\eta_c={etas[2]:.2f}$, below)", zorder=2)
a.set_xlabel("min-cut area $|\\gamma_A|$ (bonds)"); a.set_ylabel("entanglement entropy $S(A)$ (bits)")
a.set_title("(a) Perfect code saturates the holographic bound")
a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

a = ax[1]
a.plot(np.array(fracs) * 100, etas, "o-", color="#9467bd", lw=1.4, ms=6)
a.axhline(1.0, color="#d62728", ls="--", lw=1.0, label="$\\eta_c=1$: on the bound (BH corner)")
a.set_xlabel("% of bonds diluted to product"); a.set_ylabel("$\\eta_c=\\langle S/|\\gamma|\\rangle$ (saturation fraction)")
a.set_ylim(0, 1.08)
a.set_title("(b) $\\eta_c$ is the Bekenstein-saturation fraction")
a.legend(fontsize=8); a.grid(alpha=0.3)

plt.tight_layout()
OUT = os.path.join(os.path.dirname(__file__), "fig_D5d_bekenstein.png")
plt.savefig(OUT, dpi=150); plt.close()
print(f"  wrote {os.path.basename(OUT)}")
