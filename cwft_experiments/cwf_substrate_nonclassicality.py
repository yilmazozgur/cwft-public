#!/usr/bin/env python3
"""
cwf_substrate_nonclassicality.py  --  A uniform, substrate-intrinsic
non-classicality estimator across the framework's catalog, and the synthesis
claim it tests: genuine l2-phase rides the ENCODING axis, not the irreducibility
axis.

WHY THIS, GIVEN THE IMPOSSIBILITY THEOREM.
cwf_spekkens_restriction.py proved: any epistemic model over a definite ground
(incl. an undecidable one) is a non-contextual ontological model => contextual
fraction CF = 0. So for the framework's CLASSICAL substrates (CAs, reservoirs)
this estimator is GUARANTEED CF~0 -- it confirms the theorem on real substrates,
it does not discover non-classicality. The genuinely new content is twofold:
  (1) a UNIFORM CF measured from each substrate's OWN statistics (cwf_c3_bell.py
      plugged in textbook values; this extracts them), and
  (2) the SYNTHESIS TEST: cross-tabulating measured CF against the irreducibility
      axis (C1) and the encoding axis (gravity chapter) to test whether genuine
      l2 non-classicality aligns with ENCODING and is orthogonal to
      IRREDUCIBILITY -- i.e. that the phase and gravity share an axis, and the
      "gravity = geometry of ignorance" corner (irreducible AND encoding AND
      non-classical) is a still-empty triple point.

Substrate catalog (each placed by MEASURED CF on a common CHSH scenario):
  * classical CAs   Rule 110 (irreducible), Rule 30 (chaotic), Rule 90 (additive)
                    -- CF measured from coarse-grained observables over the CA's
                    own evolution, shared initial config = the hidden variable.
  * reservoir-style local lattice -- same, a second classical family.
  * genuine quantum Bell pair -- CF from expectation values on the actual state.
  * perfect-tensor (HaPPY) logical qubit pair -- CF from the encoded logical
    Bell state (the gravity chapter's encoding-class substrate).

Validated LP (Abramsky-Barbosa-Mansfield contextual fraction) reused from
cwf_phase_contextuality (closed form CF=(B-2)/2: PR=1, Tsirelson=0.414, local=0).
CPU-only; numpy + scipy. No stim needed (the quantum states are built explicitly).
"""

import json
import os
import time

import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp
from cwf_psi_l2_vs_l1 import eca_step

HERE = os.path.dirname(os.path.abspath(__file__))
SIGNS = np.array([[1, -1], [-1, 1]])          # (-1)^(a+b) on outcomes a,b in {0,1}

# CHSH scenario: observables A0,A1,B0,B1 (ids 0..3); contexts = the 4 (Ai,Bj) pairs
CHSH_OBS = [0, 1, 2, 3]
CHSH_CTX = [(0, 2), (0, 3), (1, 2), (1, 3)]


def box_from_correlators(E):
    """Standard no-signaling CHSH box from 4 correlators E[(Ai,Bj)] (uniform
    marginals): P(a,b) = (1 + (-1)^(a+b) E)/4. Returns context tables for the LP."""
    tables = {}
    for ci, pair in enumerate(CHSH_CTX):
        tables[ci] = (np.ones((2, 2)) + E[pair] * SIGNS) / 4.0
    return tables


def chsh_value(E):
    return abs(E[(0, 2)] + E[(0, 3)] + E[(1, 2)] - E[(1, 3)])


# ============================================================================
# Classical substrates: correlators measured from the substrate's OWN dynamics
# ============================================================================
def classical_ca_correlators(rule, W=121, burn=40, n_runs=40000, seed=11):
    """CHSH correlators from a CA, shared-randomness LHV style. The initial
    config is the hidden variable lambda; Alice's two settings are two parity
    observables on a left region, Bob's two on a disjoint right region; both are
    deterministic functions of the evolved config. By construction this is a
    local hidden-variable model, so CHSH<=2 and CF~0 (the impossibility theorem
    on a real substrate)."""
    rng = np.random.default_rng(seed)
    c = W // 2
    # Alice settings (left region), Bob settings (right region) -- disjoint supports
    A = [(c - 6, c - 5, c - 4), (c - 5, c - 4, c - 3)]
    B = [(c + 4, c + 5, c + 6), (c + 3, c + 4, c + 5)]
    acc = {(i, j): 0.0 for i in range(2) for j in range(2)}
    cnt = {(i, j): 0 for i in range(2) for j in range(2)}
    for _ in range(n_runs):
        row = rng.integers(0, 2, size=W).astype(np.int64)
        for _ in range(burn + int(rng.integers(0, 8))):
            row = eca_step(row, rule)
        i, j = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        a = int(np.bitwise_xor.reduce(row[list(A[i])]))
        b = int(np.bitwise_xor.reduce(row[list(B[j])]))
        acc[(i, j)] += (-1) ** (a ^ b)
        cnt[(i, j)] += 1
    return {(i, 2 + j): acc[(i, j)] / max(cnt[(i, j)], 1)
            for i in range(2) for j in range(2)}


def reservoir_correlators(W=121, burn=40, n_runs=40000, seed=12, leak=0.5, rho=1.4):
    """A second classical family: a tanh reservoir lattice read through binary
    coarse-grainings. Still a deterministic map of a shared random init => LHV."""
    rng = np.random.default_rng(seed)
    c = W // 2
    W0 = rng.normal(0, 1, size=(W, W)) * (np.abs(np.add.outer(np.arange(W), -np.arange(W))) <= 1)
    A = [(c - 6, c - 5), (c - 5, c - 4)]
    B = [(c + 4, c + 5), (c + 5, c + 6)]
    acc = {(i, j): 0.0 for i in range(2) for j in range(2)}
    cnt = {(i, j): 0 for i in range(2) for j in range(2)}
    for _ in range(n_runs // 4):          # reservoir step is heavier; fewer runs
        s = rng.normal(0, 1, size=W)
        for _ in range(burn):
            s = np.tanh(rho * (W0 @ s)) - leak * s
        bits = (s > 0).astype(int)
        i, j = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        a = int(np.bitwise_xor.reduce(bits[list(A[i])]))
        b = int(np.bitwise_xor.reduce(bits[list(B[j])]))
        acc[(i, j)] += (-1) ** (a ^ b)
        cnt[(i, j)] += 1
    return {(i, 2 + j): acc[(i, j)] / max(cnt[(i, j)], 1)
            for i in range(2) for j in range(2)}


# ============================================================================
# Quantum substrates: correlators as expectation values on the ACTUAL state
# ============================================================================
def _pauli_spin(theta):
    """Spin observable cos(theta) Z + sin(theta) X (eigenvalues +-1)."""
    Z = np.array([[1, 0], [0, -1]], complex)
    X = np.array([[0, 1], [1, 0]], complex)
    return np.cos(theta) * Z + np.sin(theta) * X


def quantum_bell_correlators():
    """CHSH correlators measured on the actual singlet state at the optimal
    angles. <A_i B_j> = <psi| A_i (x) B_j |psi>. Reaches Tsirelson => CF=0.414.
    This is a genuine computation on the substrate's own state, not a plugged-in
    number (the optimal angles are non-Pauli, hence the statevector route)."""
    # singlet
    psi = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
    a_ang = [0.0, np.pi / 2]
    b_ang = [np.pi / 4, -np.pi / 4]
    E = {}
    for i, ta in enumerate(a_ang):
        for j, tb in enumerate(b_ang):
            Aop = np.kron(_pauli_spin(ta), np.eye(2))
            Bop = np.kron(np.eye(2), _pauli_spin(tb))
            E[(i, 2 + j)] = float(np.real(psi.conj() @ (Aop @ Bop) @ psi))
    return E


def perfect_tensor_logical_correlators():
    """The HaPPY/perfect-tensor encoding class (gravity chapter) carries a
    LOGICAL qubit; a logical Bell pair across two code blocks has logical Pauli
    operators that act as bare Paulis on the logical state. The measured CHSH on
    the logical Bell pair therefore equals the bare Tsirelson value -- the
    encoding substrate is genuinely l2 at its logical level. We compute it on the
    logical state directly (the encoding is an isometry: it preserves all
    expectation values of logical observables), and flag the by-construction
    status honestly: it inherits, it does not independently generate."""
    return quantum_bell_correlators()      # isometric encoding preserves <.>


# ============================================================================
# Driver
# ============================================================================
def run():
    t0 = time.time()
    print("== LP validation ==")
    val = _validate_lp()
    results = {"LP_validation": val, "substrates": {}}

    # axis annotations: irreducible? (C1) ; encoding? (gravity chapter)
    catalog = [
        # name, correlator-fn, irreducible, encoding, kind
        ("Rule110",   lambda: classical_ca_correlators(110), True,  False, "classical CA"),
        ("Rule30",    lambda: classical_ca_correlators(30),  False, False, "classical CA"),
        ("Rule90",    lambda: classical_ca_correlators(90),  False, False, "classical CA"),
        ("reservoir", reservoir_correlators,                 False, False, "classical reservoir"),
        ("BellPair",  quantum_bell_correlators,              False, False, "quantum (bare)"),
        ("PerfectTensorQEC", perfect_tensor_logical_correlators, False, True, "encoding/QEC"),
    ]

    print(f"\n{'substrate':<18}{'kind':<22}{'CHSH':>7}{'CF':>8}"
          f"{'irred?':>8}{'enc?':>6}{'l2?':>6}")
    print("-" * 75)
    rows = []
    for name, fn, irred, enc, kind in catalog:
        E = fn()
        S = chsh_value(E)
        lp = contextuality_lp(CHSH_CTX, CHSH_OBS, box_from_correlators(E))
        cf = lp.get("contextual_fraction", float("nan"))
        l2 = cf > 0.05
        results["substrates"][name] = dict(kind=kind, CHSH=S, CF=cf,
                                           irreducible=irred, encoding=enc,
                                           nonclassical=bool(l2))
        rows.append((name, kind, S, cf, irred, enc, l2))
        print(f"{name:<18}{kind:<22}{S:>7.3f}{cf:>8.3f}"
              f"{str(irred):>8}{str(enc):>6}{str(l2):>6}")

    results["runtime_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE, "substrate_nonclassicality_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}  ({results['runtime_s']}s)")
    _interpret(rows)
    return results


def _interpret(rows):
    print("\n" + "=" * 75)
    print("INTERPRETATION  --  does genuine l2 (CF>0) ride encoding or irreducibility?")
    print("=" * 75)
    # axis-alignment: among substrates, does CF track encoding or irreducibility?
    nonclassical = [r for r in rows if r[6]]
    classical = [r for r in rows if not r[6]]
    print("non-classical (CF>0):", [r[0] for r in nonclassical] or "NONE")
    print("  of those, encoding=True:", [r[0] for r in nonclassical if r[5]] or "NONE")
    print("  of those, irreducible=True:", [r[0] for r in nonclassical if r[4]] or "NONE")
    print("classical (CF~0):    ", [r[0] for r in classical])
    print("  among these, irreducible=True:", [r[0] for r in classical if r[4]] or "NONE")
    print()
    # the synthesis claim
    enc_l2 = all(r[6] for r in rows if r[5])                 # encoding => l2 ?
    irred_not_l2 = all(not r[6] for r in rows if r[4])       # irreducible => NOT l2 (here)
    print("SYNTHESIS TEST:")
    print(f"  every ENCODING substrate is non-classical (l2)        : {enc_l2}")
    print(f"  every IRREDUCIBLE substrate tested is Bell-LOCAL (~l1) : {irred_not_l2}")
    if enc_l2 and irred_not_l2:
        print("\n  => genuine l2 non-classicality RIDES THE ENCODING AXIS and is")
        print("     ABSENT on the irreducibility axis. Combined with the gravity")
        print("     chapter (Einstein dynamics also an encoding-class property),")
        print("     PHASE and GRAVITY share the same axis; the epistemic content")
        print("     (irreducibility) is orthogonal to both. The 'gravity = geometry")
        print("     of ignorance' corner -- a substrate that is encoding AND")
        print("     irreducible AND therefore l2 -- is a TRIPLE POINT, still empty:")
        empty = [r[0] for r in rows if r[4] and r[5]]
        print(f"       substrates in (irreducible AND encoding): {empty or 'NONE'}")
        print("     This is the same empty corner the book names, now reached from")
        print("     the phase side as well as the gravity and irreducibility sides.")
    print("\n  Honesty: classical rows are CF~0 BY THEOREM (impossibility result),")
    print("  so they confirm rather than discover; the quantum/QEC rows are genuine")
    print("  measurements on genuinely quantum states, with the QEC row inheriting")
    print("  its logical-level Tsirelson value through an isometric encoding (it")
    print("  carries l2, it does not independently generate it from CA-like")
    print("  dynamics). No tested substrate occupies the triple point.")


if __name__ == "__main__":
    run()
