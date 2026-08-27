"""
cwf_sr2_selfref_contextuality.py -- does SELF-REFERENCE (no definite ground)
produce contextuality (CF>0), where passive ignorance over a DEFINITE ground does
not?

Prior CWF results (the Bell-Kochen-Specker wall): an embedded observer passively
coarse-graining a CLASSICAL substrate gives CF=0 -- a definite microstate is a
global non-contextual ('t Hooft beable) model. Spekkens restriction and
destructive back-action also gave CF=0. So no epistemic restriction over a
DEFINITE ground forces the phase. cwf_sr1 showed the conjecture's other half: a
linear + faithful realization of self-NEGATION is forced complex (eigenvalue i).

This script tests the missing ingredient on one object: a substrate with GENUINE
self-reference and NO definite ground. Model it as a cycle of n cells whose only
content is a relational constraint with their neighbour -- "cell i = cell i+1"
(equality edge) or "cell i = NOT cell i+1" (negation edge): the multi-statement
generalisation of the Liar. PARITY decides everything:

  - ODD number of negation edges  -> the cycle is FRUSTRATED: NO global {0,1}
    assignment is consistent, so the substrate has NO DEFINITE GROUND. The minimal
    case is the single self-loop "this cell is the negation of itself" (the Liar).
  - EVEN -> two consistent assignments exist -> a DEFINITE GROUND exists.

We feed the constraint structure to the VALIDATED contextual-fraction LP (imported
from cwf_phase_contextuality, checked PR=1 / Tsirelson=0.414 / local=0) and also
compute the loop HOLONOMY (product of edge operators), tying CF>0 back to the
forced-i of cwf_sr1.

HYPOTHESIS (Route A): the SAME parity controls three things at once --
  odd negations  <=>  no definite ground  <=>  CF > 0  <=>  holonomy = NOT (forces i)
  even negations <=>  definite ground      <=>  CF = 0  <=>  holonomy = I    (trivial)
The mixed controls (an even-length cycle made frustrated, an odd-length cycle made
consistent) prove the driver is the FRUSTRATION/self-reference, not the cycle size.

CPU; numpy + scipy LP (reused, validated). No quantum amplitudes injected anywhere:
the only inputs are classical relational constraints; CF measures whether they
admit a classical global section.
"""
import json, os
import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))

NOT = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)


def definite_grounds(n, neg):
    """Brute-force all 2^n cyclic assignments; return those consistent with the
    edge signs. neg[i]=True means edge (i,i+1) is a negation (x_i != x_{i+1}),
    False means equality (x_i == x_{i+1}). A 'definite ground' is a globally
    consistent truth assignment."""
    grounds = []
    for code in range(1 << n):
        x = [(code >> i) & 1 for i in range(n)]
        ok = all((x[i] ^ x[(i + 1) % n]) == (1 if neg[i] else 0) for i in range(n))
        if ok:
            grounds.append(x)
    return grounds


def empirical_model(n, neg):
    """Observables = the n cells; contexts = the n cyclic edges (pairs); each edge
    table is the uniform distribution over the outcomes its constraint allows:
      negation edge -> {(0,1),(1,0)} at 1/2 each;  equality edge -> {(0,0),(1,1)}.
    No-signaling: every single-cell marginal is (1/2,1/2)."""
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci in range(n):
        t = np.zeros((2, 2))
        if neg[ci]:
            t[0, 1] = t[1, 0] = 0.5
        else:
            t[0, 0] = t[1, 1] = 0.5
        tables[ci] = t
    return contexts, observables, tables


def holonomy(neg):
    """Product of edge operators around the loop: NOT for a negation edge, I for an
    equality edge. Equals NOT iff an ODD number of negations (frustrated), else I.
    If it is NOT, cwf_sr1 says the continuous reversible realization forces i."""
    M = I2.copy()
    for is_neg in neg:
        M = (NOT if is_neg else I2) @ M
    is_not = np.linalg.norm(M - NOT) < 1e-9
    return M, bool(is_not)


def analyze(name, n, neg):
    grounds = definite_grounds(n, neg)
    n_neg = int(sum(neg))
    odd = (n_neg % 2 == 1)
    contexts, observables, tables = empirical_model(n, neg)
    lp = contextuality_lp(contexts, observables, tables)
    cf = lp.get("contextual_fraction", float("nan"))
    Hmat, holo_is_not = holonomy(neg)
    return dict(name=name, n=n, n_negations=n_neg, odd_negations=odd,
                num_definite_grounds=len(grounds), has_definite_ground=len(grounds) > 0,
                contextual_fraction=float(cf),
                holonomy_is_NOT=holo_is_not,
                forces_i=holo_is_not,
                lp_success=lp.get("success", False))


def main():
    print("cwf_sr2 -- does self-reference (no definite ground) force contextuality?\n")
    val = _validate_lp()
    print()

    # scenarios: all-negation cycles of several sizes + mixed controls that flip
    # the parity independently of the cycle length.
    scenarios = [
        ("3-cycle all-neg (odd)",        3, [True, True, True]),
        ("4-cycle all-neg (even)",       4, [True, True, True, True]),
        ("5-cycle all-neg (odd)",        5, [True] * 5),
        ("6-cycle all-neg (even)",       6, [True] * 6),
        # mixed controls: parity decoupled from size
        ("4-cycle 3neg+1eq (odd)",       4, [True, True, True, False]),
        ("3-cycle 2neg+1eq (even)",      3, [True, True, False]),
        ("5-cycle 4neg+1eq (even)",      5, [True, True, True, True, False]),
    ]
    rows = [analyze(*s) for s in scenarios]

    print(f"{'scenario':<26}{'#neg':>5}{'odd?':>6}{'gnd?':>6}{'#gnd':>6}"
          f"{'CF':>9}{'holo=NOT':>10}{'forces i':>10}")
    print("-" * 88)
    for r in rows:
        print(f"{r['name']:<26}{r['n_negations']:>5}{str(r['odd_negations']):>6}"
              f"{str(r['has_definite_ground']):>6}{r['num_definite_grounds']:>6}"
              f"{r['contextual_fraction']:>9.4f}{str(r['holonomy_is_NOT']):>10}"
              f"{str(r['forces_i']):>10}")

    # the triple-coincidence check: odd_negations <=> no ground <=> CF>0 <=> holo=NOT
    tol = 1e-6
    coincidence = all(
        (r["odd_negations"]
         == (not r["has_definite_ground"])
         == (r["contextual_fraction"] > tol)
         == r["holonomy_is_NOT"])
        for r in rows)
    odd_rows = [r for r in rows if r["odd_negations"]]
    even_rows = [r for r in rows if not r["odd_negations"]]
    odd_cf = [r["contextual_fraction"] for r in odd_rows]
    even_cf = [r["contextual_fraction"] for r in even_rows]

    verdict = (
        "SELF-REFERENCE FORCES CONTEXTUALITY (Route A, positive). The SAME parity "
        "controls all three faces of the conjecture on one object: an ODD number of "
        "self-referential negations leaves the cycle with NO definite ground "
        f"(no consistent assignment), gives CF>0 (range {min(odd_cf):.3f}-{max(odd_cf):.3f}: "
        "no classical global section), and a loop holonomy = NOT, whose continuous "
        "reversible realization is forced complex (eigenvalue i, cwf_sr1). An EVEN "
        f"number leaves a definite ground, CF=0 (max {max(even_cf):.3f}), holonomy = I. "
        "The mixed controls confirm the driver is the FRUSTRATION (self-reference with "
        "no definite ground), not the cycle length. This is the missing ingredient the "
        "prior CF=0 negatives (beable/Spekkens/back-action over a DEFINITE ground) "
        "lacked: removing the definite ground -- which is exactly what self-reference "
        "does -- is what lets ignorance become genuinely quantum (CF>0) and forces the "
        "phase i. Both walls are now addressed on one construction."
    ) if coincidence else (
        "PARTIAL/UNEXPECTED: the odd<=>no-ground<=>CF>0<=>holonomy coincidence did not "
        "hold across all scenarios -- inspect the table."
    )
    print(f"\n  triple coincidence (odd-neg <=> no-ground <=> CF>0 <=> holo=NOT): "
          f"{coincidence}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["SR2_selfref_contextuality"] = dict(
        lp_validation=val, scenarios=rows, triple_coincidence=bool(coincidence),
        odd_CF=odd_cf, even_CF=even_cf, verdict=verdict,
        note=("Cyclic relational constraints (the multi-statement Liar): negation/"
              "equality edges; ODD negation parity => frustrated => no definite "
              "ground => CF>0 (validated ABM LP) => holonomy=NOT (forces i, cwf_sr1); "
              "EVEN => definite ground => CF=0 => holonomy=I. Mixed controls decouple "
              "parity from cycle length. No quantum amplitudes injected: inputs are "
              "classical constraints, CF measures absence of a classical global "
              "section. The missing ingredient over the prior CF=0 (definite-ground) "
              "negatives is exactly self-reference removing the definite ground."))
    json.dump(R, open(out, "w"), indent=2)
    plot(rows)
    print("\nWrote results.json key: SR2_selfref_contextuality")


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"].replace(" ", "\n", 1) for r in rows]
    cfs = [r["contextual_fraction"] for r in rows]
    cols = ["C3" if r["odd_negations"] else "C0" for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(range(len(rows)), cfs, color=cols)
    for i, r in enumerate(rows):
        tag = "no ground\nholo=NOT (i)" if r["odd_negations"] else "ground\nholo=I"
        ax.text(i, cfs[i] + 0.012, tag, ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("contextual fraction CF")
    ax.set_ylim(0, max(cfs) * 1.25 + 0.05)
    ax.set_title("Self-reference forces contextuality: ODD negation parity (red) "
                 "=\nno definite ground = CF>0 = holonomy NOT (forces i); EVEN (blue) = CF=0")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="C3", label="odd: frustrated / no definite ground"),
                       Patch(color="C0", label="even: definite ground exists")],
              fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_SR2_selfref_contextuality.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
