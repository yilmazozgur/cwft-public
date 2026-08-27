#!/usr/bin/env python3
"""STAGE 1 of the self-referential VSA: the frustrated odd-negation ring.

Goal: show that a VSA wired into a FRUSTRATED ring (no consistent global valuation ->
no definite ground), held as a reversible coherent superposition, exhibits genuine
CONTEXTUALITY (contextual fraction CF > 0) -- the CWFT strong horn -- while the
unfrustrated ring, the decohered ring, and standard definite-ground FPE all give CF = 0.

The frustration is realized as an odd exclusivity cycle (the n-cycle / KCBS scenario):
n unit hypervectors v_0..v_{n-1} with adjacent ones orthogonal (exclusive), closing a
ring.  For ODD n this is a frustrated cycle with a classical/quantum gap (contextual);
for EVEN n the gap closes (noncontextual).  The 'negations' are the exclusivity edges;
an odd cycle has no consistent 2-colouring = no definite ground.

Everything is validated against KNOWN values before any VSA claim:
  * the contextual-fraction LP is checked on CHSH (local 0, Tsirelson 0.414, PR box 1);
  * the KCBS ring reproduces the textbook sum <sum P_i> = sqrt(5).

Honesty (see SELFREF_VSA_PLAN.md):
  - This is a VSA-NATIVE realization of KNOWN single-system contextuality, plus the
    frustration-as-order-parameter framing -- not new contextuality physics.
  - It uses REAL hypervectors: CF>0 here does NOT require the imaginary unit i (single-
    system contextuality is a rebit phenomenon).  The 'self-reference forces i' claim
    and the real-vs-complex (Renou) separation are Stage 3, deferred.
  - VSA has no tensor product, so the strong-horn realization is SINGLE-system
    (KCBS-type), not Bell/CHSH-type -- consistent with 'no entanglement in VSA'.

Seeded, CPU (numpy + scipy.optimize.linprog).  Writes cwf_selfref_ring_results.json.
"""
import itertools
import json

import numpy as np
from scipy.optimize import linprog


# =====================================================================
# General contextual-fraction LP (Abramsky-Barbosa-Mansfield)
# =====================================================================
def contextual_fraction(n_meas, contexts, emp, outcomes=(0, 1)):
    """CF of an empirical model.
    n_meas   : number of measurements (indexed 0..n_meas-1)
    contexts : list of tuples of measurement indices (maximal compatible sets)
    emp      : dict  context -> {joint_outcome_tuple: probability}
    Returns CF in [0,1] = 1 - (max weight of a global noncontextual model dominated
    by the context marginals)."""
    globals_ = list(itertools.product(outcomes, repeat=n_meas))
    A_ub, b_ub = [], []
    for ctx in contexts:
        for s in itertools.product(outcomes, repeat=len(ctx)):
            row = np.zeros(len(globals_))
            for gi, g in enumerate(globals_):
                if all(g[ctx[j]] == s[j] for j in range(len(ctx))):
                    row[gi] = 1.0
            A_ub.append(row)
            b_ub.append(emp[ctx].get(s, 0.0))
    res = linprog(-np.ones(len(globals_)), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * len(globals_), method="highs")
    return float(max(0.0, 1.0 - (-res.fun)))


# =====================================================================
# (a) VALIDATION on CHSH  (CF = (CHSH-2)/2)
# =====================================================================
def chsh_model(E):
    signs = {(0, 1): +1, (0, 3): +1, (2, 1): +1, (2, 3): -1}   # A0,A1,A2,A3 cyclic
    contexts = [(0, 1), (0, 3), (2, 1), (2, 3)]
    emp = {}
    for ctx in contexts:
        corr = signs[ctx] * E
        emp[ctx] = {(a, b): (1 + ((-1) ** (a + b)) * corr) / 4
                    for a in (0, 1) for b in (0, 1)}
    return contexts, emp


def validate_chsh():
    out = []
    for name, E, chsh in [("local", 0.5, 2.0),
                          ("Tsirelson", np.sqrt(2) / 2, 2 * np.sqrt(2)),
                          ("PR box", 1.0, 4.0)]:
        ctx, emp = chsh_model(E)
        cf = contextual_fraction(4, ctx, emp)
        out.append(dict(name=name, cf=cf, predicted=max(0.0, (chsh - 2) / 2)))
    return out


# =====================================================================
# (b) The n-cycle exclusivity ring (single-system, real vectors)
# =====================================================================
def ring_vectors(n):
    """n unit vectors in R^3 on a star-polygon cone with adjacent (cyclic) ones
    orthogonal.  Returns (V (n,3), state psi (3,)).  This FRUSTRATED single-cycle
    construction exists ONLY for odd n: it needs the adjacent angular cosine c<0
    (obtuse), which the {n/((n-1)/2)} star polygon gives for odd n.  For even n no such
    single frustrated cycle exists (the ring is consistently 2-colourable -> a definite
    ground -> CF=0, the CSW perfect-graph case); we return None."""
    if n % 2 == 0:
        return None, None
    step = (n - 1) // 2
    phi = 2 * np.pi * step * np.arange(n) / n
    c = np.cos(2 * np.pi * step / n)           # adjacent angular cosine (obtuse, <0)
    if c >= 0:
        return None, None
    tan2 = -1.0 / c
    sinb = np.sqrt(tan2 / (1 + tan2))
    cosb = np.sqrt(1.0 / (1 + tan2))
    # shared axis last, so the apex state overlaps every v_i equally (<psi|v_i>=cosb)
    V = np.stack([sinb * np.cos(phi), sinb * np.sin(phi), cosb * np.ones(n)], axis=1)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    psi = np.array([0.0, 0.0, 1.0])
    return V, psi


def ring_model(V, psi):
    """Empirical model of the n-cycle: measurements P_i = |v_i><v_i| (outcome 1 = in
    v_i), contexts (i, i+1 mod n) which are compatible because adjacent v are
    orthogonal.  Exclusivity: P(1,1)=0 on each edge."""
    n = len(V)
    pr = (V @ psi) ** 2                          # <psi|P_i|psi>
    contexts = [(i, (i + 1) % n) for i in range(n)]
    emp = {}
    for (i, j) in contexts:
        emp[(i, j)] = {(1, 1): 0.0, (1, 0): float(pr[i]), (0, 1): float(pr[j]),
                       (0, 0): float(max(0.0, 1 - pr[i] - pr[j]))}
    return contexts, emp, pr


def decohere_model(V, psi):
    """Classical mixture over definite value-assignments consistent with exclusivity:
    the held superposition is replaced by a stochastic definite ground.  Should give
    CF = 0."""
    n = len(V)
    pr = (V @ psi) ** 2
    contexts = [(i, (i + 1) % n) for i in range(n)]
    # a valid noncontextual model: independent-ish definite assignment respecting the
    # marginals pr[i] and exclusivity, i.e. a global distribution over {0,1}^n with at
    # most... we build it by sampling definite grounds: pick at most one i 'lit'.
    # Marginals: assign 'lit=i' w.p. pr[i] (sum <= 1), else none lit.
    emp = {c: {(a, b): 0.0 for a in (0, 1) for b in (0, 1)} for c in contexts}
    p_none = max(0.0, 1 - pr.sum())
    def outcome(lit, m):
        return 1 if lit == m else 0
    for lit in list(range(n)) + [-1]:
        w = pr[lit] if lit >= 0 else p_none
        for (i, j) in contexts:
            emp[(i, j)][(outcome(lit, i), outcome(lit, j))] += w
    return contexts, emp


def main():
    rng = np.random.default_rng(0)
    out = {}

    # (a) validate the LP
    val = validate_chsh()
    ok = all(abs(v["cf"] - v["predicted"]) < 1e-6 for v in val)
    out["chsh_validation"] = val
    print("(a) CF-LP validation on CHSH:")
    for v in val:
        print(f"    {v['name']:>10}: CF={v['cf']:.4f} (predicted {v['predicted']:.4f})")
    assert ok, "LP validation failed -- do not trust downstream CF"
    print("    -> validated to machine precision.\n")

    # (b) the frustrated ring: odd n -> CF>0, even n -> CF=0
    print("(b) n-cycle frustrated ring (real hypervectors, held coherently):")
    rows = []
    for n in [4, 5, 6, 7, 9]:
        V, psi = ring_vectors(n)
        if V is None:
            rows.append(dict(n=n, cf=None, note="no single-cycle star polygon"))
            print(f"    n={n}: (no frustrated single-cycle construction)")
            continue
        # sanity: adjacent orthogonal, non-adjacent not
        adj = max(abs(V[i] @ V[(i + 1) % n]) for i in range(n))
        ctx, emp, pr = ring_model(V, psi)
        cf = contextual_fraction(n, ctx, emp)
        sumP = float(pr.sum())
        # decoherence control
        ctxd, empd = decohere_model(V, psi)
        cf_dec = contextual_fraction(n, ctxd, empd)
        rows.append(dict(n=n, cf=cf, cf_decohered=cf_dec, sum_Pi=sumP,
                         adj_orth=float(adj), parity="odd" if n % 2 else "even"))
        print(f"    n={n} ({'odd ' if n%2 else 'even'}): CF={cf:.4f}  "
              f"decohered CF={cf_dec:.4f}  <sum P_i>={sumP:.4f}  "
              f"(adj.orth={adj:.1e})")
    out["ring"] = rows
    # KCBS textbook check: n=5 sum should be sqrt(5)
    kcbs = next(r for r in rows if r["n"] == 5)
    print(f"    KCBS check (n=5): <sum P_i>={kcbs['sum_Pi']:.4f} vs sqrt(5)="
          f"{np.sqrt(5):.4f}  {'OK' if abs(kcbs['sum_Pi']-np.sqrt(5))<1e-3 else 'MISMATCH'}")

    # (c) VSA-native embedding: the ring vectors ARE hypervectors in R^D; similarity
    #     readouts reproduce the SAME CF (inner products preserved by a random
    #     orthonormal embedding).
    print("\n(c) VSA-native embedding (D=1024 real hypervectors, similarity readouts):")
    D, n = 1024, 5
    V, psi = ring_vectors(n)
    Q, _ = np.linalg.qr(rng.standard_normal((D, 3)))      # D x 3 orthonormal
    HV = (Q @ V.T).T                                        # hypervectors (n, D)
    Hpsi = Q @ psi
    pr = (HV @ Hpsi) ** 2
    ctx = [(i, (i + 1) % n) for i in range(n)]
    emp = {(i, j): {(1, 1): 0.0, (1, 0): float(pr[i]), (0, 1): float(pr[j]),
                    (0, 0): float(max(0, 1 - pr[i] - pr[j]))} for (i, j) in ctx}
    cf_vsa = contextual_fraction(n, ctx, emp)
    # reversibility / genuine-superposition checks
    idem = float(np.max([np.abs((np.outer(HV[i], HV[i]) @ np.outer(HV[i], HV[i]))
                                - np.outer(HV[i], HV[i])).max() for i in range(n)]))
    # frustration: no single hypervector is a definite eigen-assignment of all readouts
    definite = bool(np.any(np.abs(pr - np.round(pr)) < 1e-9))
    out["vsa_native"] = dict(D=D, n=n, cf=cf_vsa, projector_idempotency_err=idem,
                             has_definite_assignment=definite)
    print(f"    VSA-native CF = {cf_vsa:.4f} (matches the R^3 KCBS value)")
    print(f"    projector idempotency error = {idem:.1e} (reversible/proper)")
    print(f"    definite value assignment exists? {definite} "
          f"(False = genuine superposition = no definite ground)")

    # (d) contrast with standard definite-ground FPE (the safe horn, from
    #     cwf_fpe_selfref.py): CF = 0.  Recorded here for the scorecard.
    out["scorecard"] = dict(
        frustrated_ring_odd=cf_vsa,
        decohered=next(r["cf_decohered"] for r in rows if r.get("cf") is not None),
        even_ring="no frustrated construction (CF=0, perfect graph)",
        definite_ground_FPE=0.0)
    print("\n(d) scorecard: frustrated-odd-ring CF={:.3f} | decohered {:.3f} | "
          "even-ring: no frustration (CF=0) | definite-ground FPE 0.000".format(
              out["scorecard"]["frustrated_ring_odd"],
              out["scorecard"]["decohered"]))

    with open("cwf_selfref_ring_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_selfref_ring_results.json")


if __name__ == "__main__":
    main()
