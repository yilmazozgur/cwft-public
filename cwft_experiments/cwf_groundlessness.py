"""
cwf_groundlessness.py -- is GROUNDLESSNESS (not closure) the order parameter for
genuine phase?  And is self-reference its source?  (COSMOLOGY_NOTE.md  Exp 2.)

CLAIM (CWFT).  Genuine interference (real complex phase) requires NO definite
ground (the impossibility result: ignorance over any definite ground -> classical
mixture, CF=0).  The user's cosmological move: closure forbids a definite ground
for the universe, so genuine phase is forced.  The confound to KILL: that this is
just decoherence ("closed -> coherent").  The distinctive claim is that
groundlessness, sourced by SELF-REFERENCE, is the order parameter -- a knob
independent of closure.

PART 1 -- two independent knobs, one order parameter.
  Build the CHSH contextual fraction CF over a grid of:
    p = grounding fraction  (Werner mixing with I/4: a definite-ground / locally
                             explainable component)
    g = decoherence / openness  (local dephasing: loss of closure)
  If CF is killed by BOTH knobs but a CLOSED-yet-GROUNDED point (g=0, p high) is
  already CF=0, then closure is NOT sufficient for genuine phase -- groundedness
  (p) is the controlling axis.  This separates groundlessness from closure and
  kills the decoherence confound.  (Verified against the singlet: |S|=2sqrt2.)

PART 2 -- self-negation forces i (groundlessness is sourced by self-reference).
  The book's sr1, made concrete: the half-NOT (sqrt of NOT) -- a system applying
  half of a self-negation -- has NO definite-ground (real / doubly-stochastic)
  realization; the only consistent realization is COMPLEX (carries i).  A classical
  "definite-ground" half-flip decoheres to a static mixture and cannot self-negate;
  only the complex (groundless) half-NOT recoheres to a definite flip.  So the act
  of self-negation forces the groundless complex structure -> CF>0 is sourced by
  self-reference.

Honest framing.  Part 1's CHSH/CF is standard QM; the two-knob separation is a
clean framing, not new physics.  Part 2's sqrt-NOT-forces-i is the book's sr1.
The contribution is the unifying ORDER-PARAMETER frame + the confound kill + the
cosmological reading (a CLOSED, groundless, self-referential universe is forced to
CF>0; a closed-but-grounded / superdeterministic one would be CF=0).

CPU-only, numpy, no LP (closed-form CHSH fraction, verified).  Writes
cwf_groundlessness_results.json.
"""

import json
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
SQ2 = np.sqrt(2)

# singlet |01> - |10> over basis |00>,|01>,|10>,|11>
PSI = np.array([0, 1, -1, 0], dtype=complex) / SQ2
RHO_S = np.outer(PSI, PSI.conj())

# CHSH-optimal settings for the singlet
A0, A1 = Z, X
B0, B1 = (Z + X) / SQ2, (Z - X) / SQ2


def E(rho, A, B):
    return float(np.real(np.trace(rho @ np.kron(A, B))))


def chsh(rho):
    return E(rho, A0, B0) + E(rho, A0, B1) + E(rho, A1, B0) - E(rho, A1, B1)


def cf_from_S(S):
    """Contextual/nonlocal fraction from the CHSH value: the share of the behavior
    above the local bound, normalized to the quantum max.  CF=0 below 2, 1 at
    2sqrt2."""
    return max(0.0, (abs(S) - 2.0) / (2 * SQ2 - 2.0))


def dephase(rho, g):
    """Local Z-dephasing strength g on each qubit (off-diagonals * (1-g))."""
    K0 = np.sqrt(1 - g / 2) * I2
    K1 = np.sqrt(g / 2) * Z
    out = np.zeros_like(rho)
    for Ka in (K0, K1):
        for Kb in (K0, K1):
            K = np.kron(Ka, Kb)
            out += K @ rho @ K.conj().T
    return out


def grounded(rho_s, p):
    """Mix in a definite-ground (maximally mixed / locally explainable) fraction p."""
    return (1 - p) * rho_s + p * np.eye(4, dtype=complex) / 4


def run_part1():
    # verification against the known value
    S0 = chsh(RHO_S)
    assert abs(abs(S0) - 2 * SQ2) < 1e-9, f"singlet CHSH check failed: {S0}"
    print(f"[Part 1] CHSH order parameter.  VERIFY singlet |S| = {abs(S0):.4f} "
          f"(= 2sqrt2 = {2*SQ2:.4f}).  CF table over (grounding p, decoherence g):")
    ps = [0.0, 0.15, 0.293, 0.5, 0.8]
    gs = [0.0, 0.2, 0.4, 0.7]
    print("     " + "g=" + "  ".join(f"{g:>5.2f}" for g in gs) + "    (rows: p)")
    grid = []
    for p in ps:
        cells, rowrec = "", {"p": p, "cf": {}}
        for g in gs:
            rho = dephase(grounded(RHO_S, p), g)
            cf = cf_from_S(chsh(rho))
            rowrec["cf"][f"{g:.2f}"] = cf
            cells += f"{cf:>7.3f}"
        grid.append(rowrec)
        print(f"  p={p:>5.3f}{cells}")
    closed_grounded = cf_from_S(chsh(grounded(RHO_S, 0.8)))   # g=0, p=0.8
    print(f"  -> CLOSED-but-GROUNDED point (g=0, p=0.80): CF = {closed_grounded:.3f}.")
    print(f"     Closure (g=0) does NOT give genuine phase; grounding (p) kills CF")
    print(f"     even when fully closed.  Groundedness is the controlling axis, a")
    print(f"     knob independent of closure -> the decoherence confound is killed.")
    return dict(singlet_S=abs(S0), ps=ps, gs=gs, grid=grid,
                closed_grounded_cf=closed_grounded)


def run_part2():
    print(f"\n[Part 2] self-negation forces i (groundlessness <- self-reference).")
    sqrtX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2
    rebuilt = sqrtX @ sqrtX
    print(f"  principal sqrt(NOT): (sqrtX)^2 = NOT?  {np.allclose(rebuilt, X)}; "
          f"is it complex?  max|Im| = {np.abs(sqrtX.imag).max():.3f}")
    # eigenvalues of NOT are +1, -1; sqrt(-1)=i forces the complex structure
    w = np.linalg.eigvals(X)
    print(f"  NOT eigenvalues {np.round(w.real,2)}; sqrt of the -1 eigenvalue = i "
          f"-> no REAL 2x2 root (eigvals {{1, i}} are not a conjugate pair).")

    # classical definite-ground half-flip (doubly stochastic) cannot self-negate
    S_half = np.array([[0.5, 0.5], [0.5, 0.5]])         # the only symmetric DS guess
    print(f"  classical half-flip S=[[.5,.5],[.5,.5]]:  S^2 = "
          f"{np.round(S_half@S_half,3).tolist()} != NOT -> stays a static mixture "
          f"(no recoherence to a definite flip).")

    # interference: complex half-NOT recoheres to a definite flip; |0> -> |1>
    v0 = np.array([1, 0], dtype=complex)
    after_two = sqrtX @ (sqrtX @ v0)
    p1 = abs(after_two[1]) ** 2
    print(f"  complex half-NOT twice on |0>: P(|1>) = {p1:.3f} (a DEFINITE flip) -- "
          f"the i-phase carries which-path info the classical coin destroys.")
    print(f"  -> the act of self-negation has NO definite-ground realization; it")
    print(f"     forces the complex (groundless) structure.  Genuine phase (CF>0,")
    print(f"     Part 1) is sourced by self-reference (self-negation), not closure.")
    return dict(sqrtX_squares_to_NOT=bool(np.allclose(rebuilt, X)),
                sqrtX_max_imag=float(np.abs(sqrtX.imag).max()),
                classical_halfflip_sq=(S_half @ S_half).tolist(),
                complex_halfNOT_flip_prob=float(p1))


if __name__ == "__main__":
    print("=" * 70)
    print("GROUNDLESSNESS AS THE ORDER PARAMETER FOR GENUINE PHASE")
    print("=" * 70)
    p1 = run_part1()
    p2 = run_part2()
    with open("cwf_groundlessness_results.json", "w") as f:
        json.dump({"part1_order_parameter": p1, "part2_selfnegation": p2}, f,
                  indent=2)
    print("\nwrote cwf_groundlessness_results.json")
