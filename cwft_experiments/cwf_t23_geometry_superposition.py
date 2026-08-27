"""
T2.2 / Option 1 -- reviving nonlinear Einstein: can a holographic area BACK-REACT to matter?

T2.2's negative was sharp: under code-magic the area operator FLUCTUATES (Var>0) but its
EXPECTATION does not move with the bulk matter, so a fixed-graph code carries no back-reaction
(nonlinear Einstein fails). The diagnosis: that magic was "intra-geometry" -- it deformed ONE
graph's entanglement spectrum but left the min-cut a fixed integer. Back-reaction needs
"inter-geometry" structure: the state must be a matter-weighted SUPERPOSITION of geometries with
DIFFERENT min-cuts, so the min-cut itself becomes a quantum variable whose expectation tracks the
matter (Cao et al.'s "non-trivial area operator = non-local magic", used ACROSS graphs).

This is the minimal decisive test, and it is the IDENTICAL pipeline to T2.2 with one change:
in T2.2 the matter (weight p) lived in a FIXED geometry -> area = S_A - S_bulk constant; here the
matter SELECTS the geometry (min-cut gamma_0=1 vs gamma_1=2) -> area should TRACK the matter-
weighted min-cut. Three curves, on a minimal holographic bridge (region A={a1,a2} | B={b1,b2}, the
A-B min-cut set by Bell-pair bonds):

  (control, T2.2 echo)  matter in a FIXED geometry (both branches gamma=2)  -> area FLAT (=2).
  (mixture)             matter SELECTS geometry, and is reconstructable from A (in the wedge)
                        -> area = p*gamma_0 + (1-p)*gamma_1, clean linear back-reaction.
  (coherent)            genuine quantum superposition of the two geometries -> S_A varies 1->2
                        with an interference correction (the genuinely quantum-geometry version).

Purpose distinct from Option 3 (computation-generated/undecidable geometry): Option 1 establishes
that a holographic area CAN back-react (the mechanism), independent of where the geometry comes
from. Keep this result regardless of Option 3.

CPU; tiny dense states. numpy only.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

_I2 = np.eye(2) / 2.0                  # maximally-mixed 1-qubit (reduced half of a Bell pair)
_K0 = np.array([[1.0, 0.0], [0.0, 0.0]])   # |0><0|
_K1 = np.array([[0.0, 0.0], [0.0, 1.0]])   # |1><1|


def vn(rho):
    w = np.linalg.eigvalsh(rho); w = w[w > 1e-13]
    return float(-np.sum(w * np.log2(w)))


def h2(p):
    return 0.0 if p <= 0 or p >= 1 else float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# Reduced state on A={a1,a2} for the two GEOMETRIES (B={b1,b2} traced out):
#   gamma=2: |Phi+>_{a1b1} |Phi+>_{a2b2}  -> rho_A = (I/2)x(I/2),  S_A = 2  (min-cut 2)
#   gamma=1: |Phi+>_{a1b1} |0>_{a2}|0>_{b2} -> rho_A = (I/2)x|0><0|, S_A = 1  (min-cut 1)
RHO_A_G2 = np.kron(_I2, _I2)               # S = 2
RHO_A_G1 = np.kron(_I2, _K0)               # S = 1
GAMMA = {"g2": 2.0, "g1": 1.0}


def area_curve(ps, mode):
    """Return area(p)=S_A-S_bulk over matter weight p, for one of:
      'control'  : matter (entropy h(p)) tags branches, but BOTH are gamma=2 (fixed geometry).
      'mixture'  : matter tags branches and SELECTS geometry (gamma=2 w.p. p, gamma=1 w.p. 1-p);
                   the tag is in A (matter reconstructable from A / in the wedge).
      'untagged' : same selection but matter NOT in A (no tag) -> back-reaction with a messier
                   (non-linear) form, since A cannot fully distinguish the geometries.
    """
    rows = []
    for p in ps:
        if mode == "control":
            # tag qubit tau in A; both branches gamma=2
            rhoA = p * np.kron(_K0, RHO_A_G2) + (1 - p) * np.kron(_K1, RHO_A_G2)
            S_bulk = h2(p)
            area = vn(rhoA) - S_bulk
        elif mode == "mixture":
            rhoA = p * np.kron(_K0, RHO_A_G2) + (1 - p) * np.kron(_K1, RHO_A_G1)
            S_bulk = h2(p)
            area = vn(rhoA) - S_bulk
        elif mode == "untagged":
            rhoA = p * RHO_A_G2 + (1 - p) * RHO_A_G1
            S_bulk = h2(p)
            area = vn(rhoA) - S_bulk
        else:
            raise ValueError(mode)
        rows.append(dict(p=float(p), area=float(area),
                         expected_mincut=float(p * GAMMA["g2"] + (1 - p) * GAMMA["g1"])))
    return rows


def coherent_curve(ps):
    """Genuine quantum superposition of the two geometries (non-orthogonal branches -> interference).
    a1b1 is always a Bell pair (factorises, contributes 1 to S_A); the a2b2 sector is
        |chi(p)> = sqrt(p) (|00>+|11>)/sqrt2  +  sqrt(1-p) |00>,  renormalised.
    S_A = 1 + S(rho_a2). We report S_A and compare to the classical expected min-cut, exposing
    the interference correction (the genuinely quantum-geometry term in <A_gamma>)."""
    rows = []
    for p in ps:
        v = np.zeros(4, dtype=complex)              # basis a2 b2: 00,01,10,11
        v[0] = np.sqrt(p) / np.sqrt(2) + np.sqrt(1 - p)   # |00>
        v[3] = np.sqrt(p) / np.sqrt(2)                    # |11>
        v /= np.linalg.norm(v)
        T = v.reshape(2, 2)                          # (a2, b2)
        rho_a2 = T @ T.conj().T
        S_A = 1.0 + vn(rho_a2)
        rows.append(dict(p=float(p), S_A=float(S_A),
                         expected_mincut=float(p * 2 + (1 - p) * 1)))
    return rows


def main():
    ps = list(np.linspace(0.0, 1.0, 11))
    print("T2.2/Option 1 -- can a holographic area back-react to matter?\n")

    ctrl = area_curve(ps, "control")
    mix = area_curve(ps, "mixture")
    unt = area_curve(ps, "untagged")
    coh = coherent_curve(ps)

    print(f"  {'p':>5} {'CONTROL':>9} {'MIXTURE':>9} {'<mincut>':>9} {'untagged':>9} "
          f"{'coherent S_A':>13}")
    for i, p in enumerate(ps):
        print(f"  {p:5.2f} {ctrl[i]['area']:9.4f} {mix[i]['area']:9.4f} "
              f"{mix[i]['expected_mincut']:9.4f} {unt[i]['area']:9.4f} {coh[i]['S_A']:13.4f}")

    ctrl_swing = max(r["area"] for r in ctrl) - min(r["area"] for r in ctrl)
    mix_swing = max(r["area"] for r in mix) - min(r["area"] for r in mix)
    # does the mixture area equal the matter-weighted min-cut (clean linear back-reaction)?
    lin_err = max(abs(r["area"] - r["expected_mincut"]) for r in mix)
    print(f"\n  CONTROL area swing (fixed geometry, T2.2 echo): {ctrl_swing:.2e}  (expect ~0, FLAT)")
    print(f"  MIXTURE area swing (matter selects geometry):   {mix_swing:.4f}  "
          f"(expect 1.0 = gamma_1->gamma_0)")
    print(f"  MIXTURE area vs matter-weighted min-cut p*2+(1-p)*1: max|err| = {lin_err:.2e} "
          f"(=0 => clean linear back-reaction)")
    coh_swing = max(r["S_A"] for r in coh) - min(r["S_A"] for r in coh)
    print(f"  COHERENT S_A swing (genuine superposition): {coh_swing:.4f} "
          f"(varies 1->2, with interference)")

    backreacts = mix_swing > 1e-2 and ctrl_swing < 1e-9
    verdict = (
        "BACK-REACTION RESTORED: when the matter selects the geometry (a superposition of "
        "min-cuts), the area S_A - S_bulk tracks the matter-weighted min-cut p*gamma_0+(1-p)*"
        "gamma_1 EXACTLY (clean linear back-reaction), while the fixed-geometry control stays "
        "flat (the T2.2 negative reproduced). So a holographic area CAN back-react to matter; "
        "T2.2's no-back-reaction was specifically the fixed-graph (intra-geometry-magic) case. "
        "The genuine coherent superposition gives the same variation plus an interference term. "
        "This is the mechanism nonlinear Einstein needs; where the geometry/superposition COMES "
        "FROM (and whether it can be undecidable) is the separate Option-3 question."
    ) if backreacts else (
        "UNEXPECTED: mixture did not back-react or control was not flat -- inspect."
    )
    print(f"\n  VERDICT: {verdict}")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["T23_geometry_superposition"] = dict(
        ps=[float(p) for p in ps], control=ctrl, mixture=mix, untagged=unt, coherent=coh,
        control_area_swing=float(ctrl_swing), mixture_area_swing=float(mix_swing),
        mixture_linear_error=float(lin_err), coherent_SA_swing=float(coh_swing),
        back_reacts=bool(backreacts), verdict=verdict,
        note="Option 1 (superposition of geometries). Minimal holographic bridge A|B with the "
             "A-B min-cut (gamma in {1,2}) set by Bell-pair bonds. CONTROL: matter in a fixed "
             "geometry -> area flat (reproduces T2.2). MIXTURE: matter selects geometry and is in "
             "EW(A) -> area = p*gamma_0+(1-p)*gamma_1 exactly (linear back-reaction). COHERENT: "
             "genuine quantum superposition -> S_A varies 1->2 with interference. Establishes that "
             "a holographic area CAN back-react; kept independent of Option 3 (computation-"
             "generated/undecidable geometry), which asks where the geometry comes from.")
    json.dump(R, open(out, "w"), indent=2)
    plot(ps, ctrl, mix, coh)
    print("Wrote results.json key: T23_geometry_superposition")


def plot(ps, ctrl, mix, coh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.array(ps)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    a = ax[0]
    a.plot(p, [r["area"] for r in ctrl], "o-", color="C0",
           label="control: fixed geometry (T2.2)")
    a.plot(p, [r["area"] for r in mix], "s-", color="C3",
           label="matter selects geometry (mixture)")
    a.plot(p, [r["expected_mincut"] for r in mix], "k:", lw=1,
           label=r"$p\,\gamma_0+(1{-}p)\,\gamma_1$")
    a.set_xlabel(r"matter weight $p$"); a.set_ylabel(r"area $=S_A-S_{\rm bulk}$")
    a.set_title("(a) the area back-reacts iff matter\nselects the geometry")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    b = ax[1]
    b.plot(p, [r["S_A"] for r in coh], "^-", color="C2", label=r"coherent $S_A$")
    b.plot(p, [r["expected_mincut"] for r in coh], "k:", lw=1, label="classical weighted min-cut")
    b.set_xlabel(r"matter weight $p$"); b.set_ylabel(r"$S_A$")
    b.set_title("(b) genuine superposition of geometries\n($S_A$ varies, with interference)")
    b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.suptitle("T2.2 / Option 1: reviving back-reaction via a superposition of geometries",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pth = os.path.join(HERE, "fig_T23_geometry_superposition.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
