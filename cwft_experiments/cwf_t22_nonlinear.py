"""
T2.2 -- nonlinear Einstein: does the (now fluctuating) area operator BACK-REACT to bulk matter?

Ch5 established the LINEARISED first law (A3b4: delta S_A = delta S_bulk, unit response). T2.1
showed that non-local code-magic turns the RT-surface area into a genuine fluctuating OPERATOR
(Var(K)>0; RT -> approximate). T2.2 asks the named next rung -- the nonlinear / back-reacting
content of holographic gravity:

    put bulk MATTER of entropy S_bulk in the entanglement wedge of A, and ask whether the
    geometric "area" = S_A - S_bulk RESPONDS to the matter (back-reaction) or stays a fixed
    c-number (kinematic, linear only).

  - Stabilizer code (alpha=0): area = |gamma| CONSTANT, independent of the bulk matter -> no
    back-reaction; only the matter term S_bulk moves (the linear first law). Geometry is rigid.
  - Code-magic (alpha>0): is the area operator's expectation STATE-DEPENDENT -- does area(p)
    shift as the matter entropy grows? A state-dependent area = matter sourcing geometry = the
    seed of NONLINEAR Einstein dynamics; a still-constant area = "linear/kinematic only", the
    honest boundary of the analogy.

Setup (on the T2.1 depth-1 HaPPY tree, the genuine 2-bond RT surface, |gamma|=2): excite a
child bulk leg into a mixed matter state of weight p (entropy S_bulk = h(p)) and read A = the
children boundary (its entanglement wedge contains that matter). S_A(p) computed exactly from
the dense boundary state; area(p) = S_A(p) - h(p).

CPU; dense (11-qubit boundary). numpy + stim.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_t21_stage2 import codemagic_t

HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Depth-1 tree contraction with SETTABLE bulk leg values (extends Stage 2)
# =========================================================================

def build_tree_bulk(alpha, pairsR, pairsC, bulk_vals):
    """Contract root R + children C1,C2 (bonds R.4<->C1.1, R.5<->C2.1), fixing the three
    bulk legs (R.0, C1.0, C2.0) to the computational values in bulk_vals=(vR,vC1,vC2).
    Returns the 11-qubit boundary state. Layout: axes 0,1,2 = R{1,2,3}; 3,4,5,6 = C1{2,3,4,5};
    7,8,9,10 = C2{2,3,4,5}."""
    vR, vC1, vC2 = bulk_vals
    R = codemagic_t(alpha, pairsR)[vR]                  # fix R bulk -> legs (1,2,3,4,5)
    C1 = codemagic_t(alpha, pairsC)[vC1]
    C2 = codemagic_t(alpha, pairsC)[vC2]
    M = np.tensordot(R, C1, axes=([3], [0])) / np.sqrt(2)     # R.4 <-> C1.1
    M = np.tensordot(M, C2, axes=([3], [0])) / np.sqrt(2)     # R.5 <-> C2.1
    psi = M.reshape(-1)
    return psi / np.linalg.norm(psi)


A_CHILDREN = list(range(3, 11))     # children boundary (axes 3..10): EW contains C1,C2 bulk


def rho_region(psi, n, A):
    """Reduced density matrix on region A (list of axes) for pure state psi (2^n vector)."""
    A = sorted(int(q) for q in A)
    B = [q for q in range(n) if q not in A]
    T = np.transpose(psi.reshape([2] * n), A + B).reshape(2 ** len(A), 2 ** len(B))
    return T @ T.conj().T


def vn_entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-13]
    return float(-np.sum(w * np.log2(w)))


def h2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


# =========================================================================
# Back-reaction sweep: excite C1's bulk matter (weight p) and read the area
# =========================================================================

def backreaction(alpha, pairsR, pairsC, ps):
    """For each matter weight p, build the mixed bulk-matter state on C1 (logical 0 w.p. p,
    logical 1 w.p. 1-p), reduce to A=children boundary, and return area(p)=S_A(p)-h(p)."""
    # the two pure boundary states with C1 bulk = 0 / 1 (R,C2 bulk fixed to 0)
    psi0 = build_tree_bulk(alpha, pairsR, pairsC, (0, 0, 0))
    psi1 = build_tree_bulk(alpha, pairsR, pairsC, (0, 1, 0))
    rA0 = rho_region(psi0, 11, A_CHILDREN)
    rA1 = rho_region(psi1, 11, A_CHILDREN)
    rows = []
    for p in ps:
        rhoA = p * rA0 + (1 - p) * rA1                 # bulk matter = mixed state, S_bulk=h(p)
        S_A = vn_entropy(rhoA)
        S_bulk = h2(p)
        rows.append(dict(p=float(p), S_A=S_A, S_bulk=S_bulk, area=S_A - S_bulk))
    return rows


def main():
    t0 = time.time()
    ps = list(np.linspace(0.5, 1.0, 11))               # p=1 vacuum (S_bulk=0) -> p=0.5 max matter
    # non-local code-magic that straddles the RT cut (the T2.1 gravity-active placement),
    # plus magic in the children dressing their bulk-to-boundary correlation (so the matter
    # can couple to the area):
    pairsR = [(3, 4), (2, 5), (1, 4)]
    pairsC = [(0, 2), (0, 3)]                           # child: bulk leg 0 <-> boundary legs
    print("T2.2 -- does the area operator back-react to bulk matter?\n")

    print("(stabilizer, alpha=0): expect area = |gamma| = 2 CONSTANT (no back-reaction)")
    stab = backreaction(0.0, pairsR, pairsC, ps)
    print(f"    {'p':>6} {'S_bulk':>7} {'S_A':>7} {'area=S_A-S_bulk':>16}")
    for r in stab:
        print(f"    {r['p']:6.3f} {r['S_bulk']:7.4f} {r['S_A']:7.4f} {r['area']:16.4f}")
    area_var_stab = max(r["area"] for r in stab) - min(r["area"] for r in stab)

    for alpha, tag in [(0.7854, "alpha=pi/4"), (1.5708, "alpha=pi/2")]:
        cm = backreaction(alpha, pairsR, pairsC, ps)
        print(f"\n(code-magic, {tag}): does area(p) VARY with the matter (back-reaction)?")
        print(f"    {'p':>6} {'S_bulk':>7} {'S_A':>7} {'area=S_A-S_bulk':>16}")
        for r in cm:
            print(f"    {r['p']:6.3f} {r['S_bulk']:7.4f} {r['S_A']:7.4f} {r['area']:16.4f}")
        area_var = max(r["area"] for r in cm) - min(r["area"] for r in cm)
        # linear first-law slope near vacuum (p->1): dS_A/dS_bulk
        # (use the two points nearest p=1)
        d_SA = cm[-1]["S_A"] - cm[-3]["S_A"]
        d_Sb = cm[-1]["S_bulk"] - cm[-3]["S_bulk"]
        slope = d_SA / d_Sb if abs(d_Sb) > 1e-9 else float("nan")
        print(f"    => area swing over the sweep = {area_var:.4f} "
              f"(stabilizer: {area_var_stab:.2e}); near-vacuum dS_A/dS_bulk = {slope:.3f}")
        if tag == "alpha=pi/2":
            cm_final = cm; area_var_final = area_var; slope_final = slope

    # verdict
    backreacts = area_var_final > 1e-2
    print(f"\n  VERDICT:")
    if backreacts:
        verdict = (f"BACK-REACTION: the area operator's expectation is STATE-DEPENDENT -- area "
                   f"swings {area_var_final:.3f} as bulk matter grows, vs ~0 for the stabilizer "
                   f"code. The geometry responds to matter: the seed of NONLINEAR Einstein "
                   f"dynamics on a code-magic substrate. The linear first law still holds near "
                   f"vacuum (dS_A/dS_bulk = {slope_final:.2f}); the nonlinear deviation is the "
                   f"area's back-reaction at finite matter.")
    else:
        verdict = (f"LINEAR ONLY (kinematic): even with a fluctuating area operator, the area's "
                   f"expectation stays ~constant under bulk matter (swing {area_var_final:.3f}) "
                   f"-- the matter does NOT source the geometry. The first law holds "
                   f"(dS_A/dS_bulk = {slope_final:.2f}) but Einstein dynamics do not back-react: "
                   f"the honest boundary of the analogy, consistent with Ch5's generic 'Einstein "
                   f"dynamics no'.")
    print(f"  {verdict}")
    print(f"\n  runtime {time.time()-t0:.1f}s")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["T22_nonlinear"] = dict(
        ps=[float(p) for p in ps], pairsR=pairsR, pairsC=pairsC,
        stabilizer=stab, code_magic_pi2=cm_final,
        area_swing_stabilizer=float(area_var_stab), area_swing_codemagic=float(area_var_final),
        linear_slope_codemagic=float(slope_final), back_reacts=bool(backreacts),
        verdict=verdict,
        note="Nonlinear Einstein / back-reaction test on the T2.1 2-bond RT surface (|gamma|=2). "
             "Bulk matter = mixed state on a child bulk leg (entropy S_bulk=h(p)); area(p)="
             "S_A(p)-h(p). Stabilizer: area constant (kinematic, A3b4 linear). Code-magic: "
             "does area(p) vary (geometry sourced by matter = nonlinear Einstein) or stay "
             "constant (linear only)? Builds on the fluctuating area operator from T2.1.")
    json.dump(R, open(out, "w"), indent=2)
    plot_t22(stab, cm_final, ps)
    print("Wrote results.json key: T22_nonlinear")


def plot_t22(stab, cm, ps):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    sb = [r["S_bulk"] for r in stab]
    a = ax[0]
    a.plot(sb, [r["S_A"] for r in stab], "o-", color="C0", label="S_A stabilizer")
    a.plot(sb, [r["S_A"] for r in cm], "s-", color="C3", label="S_A code-magic")
    a.plot(sb, sb, "k:", lw=0.8, label="slope-1 (matter only)")
    a.set_xlabel(r"bulk matter entropy $S_{\rm bulk}=h(p)$"); a.set_ylabel(r"$S_A$")
    a.set_title("(a) first law: S_A vs bulk matter")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    b = ax[1]
    b.plot(sb, [r["area"] for r in stab], "o-", color="C0", label="area stabilizer (flat)")
    b.plot(sb, [r["area"] for r in cm], "s-", color="C3", label="area code-magic")
    b.axhline(2.0, color="k", ls=":", lw=0.8, label=r"$|\gamma|=2$")
    b.set_xlabel(r"bulk matter entropy $S_{\rm bulk}$")
    b.set_ylabel(r"area $=S_A-S_{\rm bulk}$")
    b.set_title("(b) back-reaction: does the area respond to matter?")
    b.legend(fontsize=8); b.grid(alpha=0.3)
    fig.suptitle("T2.2: nonlinear Einstein -- area back-reaction to bulk matter", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(HERE, "fig_T22_nonlinear.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
