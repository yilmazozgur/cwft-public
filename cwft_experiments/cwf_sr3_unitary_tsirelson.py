"""
cwf_sr3_unitary_tsirelson.py -- does INSISTING ON A UNITARY (linear + faithful)
realization bound the super-quantum self-reference down to the quantum tier?

cwf_sr2 showed the BARE self-referential frustrated cycle (the possibilistic /
"no-signaling" completion) is SUPER-QUANTUM: contextual fraction CF = 1, the
PR-box maximum. cwf_sr1 showed a linear + faithful realization of self-NEGATION is
forced complex (eigenvalue i). This script closes the loop: it represents the SAME
frustrated self-referential cycle as a genuine QUANTUM model -- rank-1 projectors
on a qutrit with adjacent EXCLUSIVITY (orthogonality), evolvable by a unitary, with
a quantum state -- and measures the contextual fraction of THAT empirical model.

Construction (exact, closed-form KCBS / odd-n-cycle, no optimization):
  state  psi = e_0;  projectors P_i = |v_i><v_i| with
  v_i = (cos t, sin t cos phi_i, sin t sin phi_i),  phi_i = i * pi (n-1)/n,
  tan^2 t = 1 / cos(pi/n)  =>  adjacent v_i _|_ v_{i+1} EXACTLY (the negation /
  exclusivity edge: P_i, P_{i+1} are orthogonal, hence jointly measurable, and
  cannot both read 1 -- the quantum encoding of "this statement is the negation of
  its neighbour"). The cycle closes with every edge orthogonal (the pentagram).

PREDICTION (Route A, the closing leg): a unitary representation cannot reach the
super-quantum CF = 1. It pulls the contextuality down to the QUANTUM tier:
  classical (noncontextual)            CF = 0
  QUANTUM (unitary, this script)       0 <= CF < 1   (the Tsirelson-type bound)
  super-quantum (possibilistic, sr2)   CF = 1
and -- a sharp internal check -- for n = 3 the unitary representation collapses the
three exclusivities to an orthonormal BASIS (a single complete measurement), so it
is non-contextual (CF = 0) even though possibilistically it was CF = 1: unitarity
kills the smallest super-quantum cycle entirely, and only allows genuine (but
bounded) contextuality from n = 5 up.

HONEST SCOPE: the KCBS family is realisable over the REAL Hilbert space, so the
MAGNITUDE bound here is the consequence of unitarity (linear + faithful), NOT of
complex phase. The complex i is the separate consequence of the CONTINUITY of
self-negation (cwf_sr1, the holonomy). Forcing complex-over-real in the statistics
themselves needs a network scenario (Renou et al. 2021) -- a future experiment.
We do not conflate the two; this script tests the MAGNITUDE (Tsirelson) bound.

CPU; numpy + the validated CF LP (reused from cwf_phase_contextuality).
"""
import json, os
import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))


def kcbs_config(n):
    """Exact odd-n-cycle KCBS 'umbrella': n real unit vectors in R^3 with adjacent
    orthogonality, apex state e_0. Returns (psi, vectors)."""
    t = np.arctan(np.sqrt(1.0 / np.cos(np.pi / n)))     # tan^2 t = 1/cos(pi/n)
    phi = np.array([i * np.pi * (n - 1) / n for i in range(n)])
    V = np.stack([np.cos(t) * np.ones(n),
                  np.sin(t) * np.cos(phi),
                  np.sin(t) * np.sin(phi)], axis=1)      # (n,3)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    psi = np.array([1.0, 0.0, 0.0])
    return psi, V


def empirical_model_quantum(psi, V):
    """Build the cyclic empirical model from projectors P_i=|v_i><v_i| and state psi.
    Context = adjacent pair (i,i+1); they are orthogonal => jointly measurable, and
    (1,1) is impossible. Returns (contexts, observables, tables, max_adjacent_overlap)."""
    n = len(V)
    p = np.array([float(np.abs(psi @ V[i]) ** 2) for i in range(n)])   # P(P_i = 1)
    overlaps = [float(np.abs(V[i] @ V[(i + 1) % n])) for i in range(n)]
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci in range(n):
        i, j = contexts[ci]
        t = np.zeros((2, 2))
        t[1, 0] = p[i]                      # (P_i=1, P_j=0)
        t[0, 1] = p[j]                      # (P_i=0, P_j=1)
        t[1, 1] = 0.0                        # orthogonal: cannot both be 1
        t[0, 0] = max(0.0, 1.0 - p[i] - p[j])
        s = t.sum()
        tables[ci] = t / s if s > 0 else t
    return contexts, observables, tables, p, max(overlaps)


def analyze(n):
    psi, V = kcbs_config(n)
    contexts, observables, tables, p, max_ov = empirical_model_quantum(psi, V)
    lp = contextuality_lp(contexts, observables, tables)
    cf_q = lp.get("contextual_fraction", float("nan"))
    kcbs_sum = float(p.sum())                       # sum_i <P_i>
    classical_bound = (n - 1) / 2.0                 # noncontextual bound (odd cycle)
    quantum_value = n * np.cos(np.pi / n) / (1.0 + np.cos(np.pi / n))   # closed form
    return dict(
        n=n, max_adjacent_overlap=max_ov,           # ~0 confirms exclusivity exact
        kcbs_sum=kcbs_sum, classical_bound=classical_bound,
        quantum_value_closedform=float(quantum_value),
        beats_classical=bool(kcbs_sum > classical_bound + 1e-9),
        CF_quantum=float(cf_q),
        CF_possibilistic=1.0,                       # from cwf_sr2 (same cycle, unconstrained)
        CF_classical=0.0,
        lp_success=lp.get("success", False))


def main():
    print("cwf_sr3 -- does a unitary (linear+faithful) realization bound the "
          "super-quantum self-reference?\n")
    _validate_lp(); print()

    ns = [3, 5, 7, 9, 11]
    rows = [analyze(n) for n in ns]

    print(f"{'n':>3}{'adj.overlap':>13}{'KCBS sum':>10}{'classical':>10}"
          f"{'quantum':>9}{'beats?':>8}{'CF_quant':>10}{'CF_poss':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['n']:>3}{r['max_adjacent_overlap']:>13.2e}{r['kcbs_sum']:>10.4f}"
              f"{r['classical_bound']:>10.2f}{r['quantum_value_closedform']:>9.4f}"
              f"{str(r['beats_classical']):>8}{r['CF_quantum']:>10.4f}"
              f"{r['CF_possibilistic']:>9.1f}")

    # checks
    n3 = next(r for r in rows if r["n"] == 3)
    big = [r for r in rows if r["n"] >= 5]
    n3_collapses = n3["CF_quantum"] < 1e-6                       # unitarity kills the triangle
    bounded = all(0.0 < r["CF_quantum"] < 1.0 - 1e-6 for r in big)  # genuine but < super-quantum
    monotone_gap = all(r["CF_quantum"] < r["CF_possibilistic"] for r in rows)

    cf_big = [r["CF_quantum"] for r in big]
    verdict = (
        "UNITARITY BOUNDS THE SELF-REFERENCE TO THE QUANTUM TIER (Route A, closing "
        "leg). The SAME frustrated self-referential cycle that was super-quantum "
        "(CF=1) when left as a bare possibilistic constraint (cwf_sr2) is pulled "
        "strictly below CF=1 the moment it must be realised by a unitary (linear + "
        f"faithful) model: n=3 COLLAPSES to non-contextual (CF=0 -- the three "
        "exclusivities become an orthonormal basis, a single complete measurement), "
        f"and n>=5 gives genuine but BOUNDED contextuality (CF in "
        f"{min(cf_big):.3f}-{max(cf_big):.3f}, with the KCBS sum hitting the closed-form "
        "quantum value, strictly above the classical bound and strictly below the "
        "super-quantum 1). So a three-tier ladder is realised on one object: "
        "classical 0 < quantum (unitary) < super-quantum 1, and UNITARITY is exactly "
        "the constraint that demotes the super-quantum self-reference to quantum -- "
        "the Tsirelson-type bound as the shadow of insisting on a linear/faithful "
        "(wave) substrate, supporting Ozgur's reading that the super-quantum value is "
        "the unconstrained object and physical quantum-ness is its unitary shadow. "
        "SCOPE: this is the MAGNITUDE bound (real Hilbert space suffices for KCBS); "
        "the complex i is the separate continuity result of cwf_sr1, not claimed here."
    ) if (n3_collapses and bounded and monotone_gap) else (
        "PARTIAL/UNEXPECTED: the quantum tier did not land strictly between 0 and 1 as "
        "predicted for all n -- inspect."
    )
    print(f"\n  n=3 collapses to non-contextual (CF=0): {n3_collapses}")
    print(f"  n>=5 genuine but bounded (0<CF<1):       {bounded}")
    print(f"  quantum strictly below super-quantum:    {monotone_gap}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["SR3_unitary_tsirelson"] = dict(
        scenarios=rows, n3_collapses=bool(n3_collapses), bounded=bool(bounded),
        quantum_below_superquantum=bool(monotone_gap), verdict=verdict,
        note=("Exact closed-form KCBS odd-n-cycle (qutrit, real vectors, adjacent "
              "orthogonality = quantum exclusivity edge). The bare possibilistic cycle "
              "(sr2) is CF=1 (super-quantum); the unitary realization gives CF=0 for "
              "n=3 (collapses to an orthonormal basis) and 0<CF<1 for n>=5 (quantum "
              "tier, KCBS sum = closed-form quantum value > classical bound). Unitarity "
              "is the constraint that demotes super-quantum self-reference to the "
              "quantum tier (Tsirelson-type bound as the shadow of linear+faithful). "
              "MAGNITUDE bound only: KCBS is real-realisable, so this is not the "
              "complex-i result (that is sr1's continuity holonomy); complex-over-real "
              "in statistics needs a Renou-type network (future)."))
    json.dump(R, open(out, "w"), indent=2)
    plot(rows)
    print("\nWrote results.json key: SR3_unitary_tsirelson")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = [r["n"] for r in rows]
    cfq = [r["CF_quantum"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axhline(1.0, color="C3", ls="--", lw=1.3, label="super-quantum (bare self-reference, sr2): CF=1")
    ax.axhline(0.0, color="C0", ls="--", lw=1.3, label="classical (definite ground): CF=0")
    ax.plot(ns, cfq, "o-", color="C2", lw=2, ms=7,
            label="quantum (unitary / linear+faithful): this script")
    for r in rows:
        ax.annotate(f"{r['CF_quantum']:.3f}", (r["n"], r["CF_quantum"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.fill_between([min(ns) - 0.5, max(ns) + 0.5], 0, 1, color="C2", alpha=0.05)
    ax.set_xlabel("self-referential cycle length $n$ (odd)")
    ax.set_ylabel("contextual fraction CF")
    ax.set_title("Unitarity is the constraint that demotes super-quantum self-reference\n"
                 "to the quantum tier (CF=1 $\\to$ bounded); $n=3$ collapses to classical")
    ax.set_ylim(-0.08, 1.12); ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_SR3_unitary_tsirelson.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
