"""
cwf_sr1_liar_forced_i.py -- the trichotomy made exact.

Conjecture (Route A, aligned with Ozgur 2026-05-31): when a substrate's grammar
crosses the SELF-REFERENCE threshold and it INSISTS on a linear + faithful
(information-preserving) self-description, it cannot stay real -- the imaginary
unit i is the forced third corner of a trichotomy. Faced with the Liar's
no-fixed-point obstruction (L = NOT L has no consistent value), a system can:

  - NONLINEARIZE : keep it real & faithful, give up LINEARITY
                   (the value oscillates 0->1->0->1; a dynamical, never-settling map)
  - COARSE-GRAIN : keep it real & linear, give up FAITHFULNESS
                   (collapse to the maximally-mixed "don't know" gap, 1 bit lost)
  - COMPLEXIFY   : keep it linear & faithful, give up REALNESS -> i (the wave)

This script demonstrates all three corners EXACTLY on the minimal self-negation
operator NOT = [[0,1],[1,0]] (the truth-functional content of "this statement is
false"), and shows the COMPLEXIFY corner is FORCED, not chosen, by a topological
obstruction:

  NOT is a reflection (det = -1), DISCONNECTED from the identity in the real
  orthogonal group O(2) but CONNECTED in the unitary group U(2). So there is NO
  continuous, reversible, REAL path from "do nothing" to "self-negate"; the
  continuous reversible path exists only over the complex numbers and runs through
  sqrt(NOT), whose nontrivial eigenvalue is exactly i.

This is the discrete cartoon of Stone's theorem: the generator of any continuous,
linear, reversible flow is anti-symmetric, hence has imaginary spectrum -- so
"continuous + linear + reversible (faithful)" forces i. Self-reference is what
makes a system demand exactly those three of a description of ITSELF.

CPU; 2x2 matrices. numpy + scipy.linalg.
"""
import json, os
import numpy as np
from scipy.linalg import logm, sqrtm, expm

HERE = os.path.dirname(os.path.abspath(__file__))

NOT = np.array([[0.0, 1.0], [1.0, 0.0]])      # self-negation operator (swap)
I2 = np.eye(2)


def boolean_liar_fixed_points():
    """Over {0,1}: does v = NOT(v) have a solution? (truth value of the Liar)."""
    sols = [b for b in (0, 1) if (1 - b) == b]   # b == not b
    return sols                                   # [] -> no consistent truth value


def coarse_grain_corner():
    """COARSE-GRAIN corner: the real, LINEAR, column-stochastic fixed point of NOT
    (NOT is doubly stochastic). The stationary distribution is the maximally-mixed
    (1/2,1/2): linear and real, but it has thrown away one full bit -- the
    'don't know' gap. Faithfulness sacrificed."""
    w, V = np.linalg.eig(NOT)
    # eigenvalue 1 eigenvector, normalised to a probability vector
    k = int(np.argmin(np.abs(w - 1.0)))
    p = np.real(V[:, k]); p = np.abs(p) / np.abs(p).sum()
    entropy = float(-np.sum([pi * np.log2(pi) for pi in p if pi > 0]))
    return dict(fixed_distribution=p.tolist(), bits_lost=entropy,
                linear=True, faithful=False, real=True)


def nonlinearize_corner(steps=6):
    """NONLINEARIZE corner: iterate the deterministic Boolean map b -> 1-b. It is
    real and faithful (bijective), but it is NOT a fixed state -- it oscillates
    with period 2 and never settles. Linearity (a single static description)
    sacrificed for a nonlinear dynamical loop."""
    orbit = [0]
    for _ in range(steps):
        orbit.append(1 - orbit[-1])
    period = 2
    has_fixed = any(orbit[i] == orbit[i + 1] for i in range(len(orbit) - 1))
    return dict(orbit=orbit, period=period, has_static_fixed_point=has_fixed,
                linear=False, faithful=True, real=True)


def complexify_corner():
    """COMPLEXIFY corner: demand a CONTINUOUS, LINEAR, REVERSIBLE realization of
    self-negation -- and watch i be forced.

    (a) Real obstruction: NOT is a reflection (det = -1). det is continuous and
        valued in {+-1} on the real orthogonal group, so no continuous orthogonal
        path connects I (det +1) to NOT (det -1). Equivalently, NOT has no real
        anti-symmetric generator (exp of an anti-symmetric matrix lies in SO(2),
        det = +1 always). The matrix logarithm of NOT is therefore COMPLEX.
    (b) Complex realization: over U(2) the path exists. sqrt(NOT) is unitary,
        squares to NOT, and its nontrivial eigenvalue is i. The continuous group
        U(t) = exp(-i H t) with U(1) = NOT has Hermitian generator H (real
        spectrum), and U(1/2) = sqrt(NOT) carries the phase i.
    """
    detI, detNOT = float(np.linalg.det(I2)), float(np.linalg.det(NOT))

    # (a) real generator? exp(anti-symmetric) is always a rotation (det +1).
    L = logm(NOT)                                   # principal matrix log
    log_imag_norm = float(np.linalg.norm(L.imag))   # >0  => no real logarithm
    log_real_norm = float(np.linalg.norm(L.real))
    # an explicit witness: the best real anti-symmetric generator cannot reach NOT
    # because every exp(skew) has det +1 != det(NOT) = -1.
    real_reversible_possible = abs(detNOT - 1.0) < 1e-9

    # (b) complex realization
    S = sqrtm(NOT)                                  # sqrt(NOT) (complex, unitary)
    sq_err = float(np.linalg.norm(S @ S - NOT))
    unit_err = float(np.linalg.norm(S.conj().T @ S - I2))
    wS = np.linalg.eigvals(S)
    # the nontrivial eigenvalue (the one that is not +1) -- its argument is the phase
    nontrivial = wS[int(np.argmax(np.abs(wS - 1.0)))]
    phase = float(np.angle(nontrivial))             # = +/- pi/2  (i.e. i)

    # the continuous unitary group through NOT, sampled
    H = (1j * logm(NOT)).real if False else (1j * logm(NOT))  # H = i log(NOT)
    H = 0.5 * (H + H.conj().T)                       # Hermitian part (numerical hygiene)
    U_half = expm(-1j * H * 0.5)
    U_one = expm(-1j * H * 1.0)
    half_is_sqrt = float(np.linalg.norm(U_half @ U_half - NOT))
    one_is_not = float(np.linalg.norm(U_one - NOT))
    H_eigs = np.linalg.eigvalsh(H)

    return dict(
        det_I=detI, det_NOT=detNOT,
        real_continuous_reversible_possible=bool(real_reversible_possible),
        logm_imag_norm=log_imag_norm, logm_real_norm=log_real_norm,
        sqrtNOT=[[complex(x).__repr__() for x in row] for row in S],
        sqrt_squares_to_NOT_err=sq_err, sqrtNOT_unitary_err=unit_err,
        sqrtNOT_eigs=[complex(x).__repr__() for x in wS],
        nontrivial_eigenvalue=complex(nontrivial).__repr__(),
        forced_phase_rad=phase, forced_phase_over_pi=phase / np.pi,
        is_i=bool(abs(abs(phase) - np.pi / 2) < 1e-9),
        H_real_spectrum=[float(x) for x in H_eigs],
        Uhalf_squares_to_NOT_err=half_is_sqrt, Uone_is_NOT_err=one_is_not,
        linear=True, faithful=True, real=False)


def main():
    print("cwf_sr1 -- self-negation, the trichotomy, and the forced i\n")

    liar = boolean_liar_fixed_points()
    print(f"Boolean Liar  L = NOT L  over {{0,1}}: consistent truth values = {liar} "
          f"(=> {'NONE -- the obstruction' if not liar else 'has one'})\n")

    cg = coarse_grain_corner()
    print(f"[COARSE-GRAIN]  linear+real, NOT faithful: fixed distribution "
          f"{np.round(cg['fixed_distribution'],3).tolist()}, bits lost = "
          f"{cg['bits_lost']:.3f}  (the 'don't know' gap)")

    nl = nonlinearize_corner()
    print(f"[NONLINEARIZE]  real+faithful, NOT linear: orbit {nl['orbit']} "
          f"(period {nl['period']}, static fixed point: {nl['has_static_fixed_point']})")

    cx = complexify_corner()
    print(f"[COMPLEXIFY]    linear+faithful, NOT real:")
    print(f"   real continuous-reversible path I->NOT possible? "
          f"{cx['real_continuous_reversible_possible']}  "
          f"(det I={cx['det_I']:.0f}, det NOT={cx['det_NOT']:.0f}; reflection => NO)")
    print(f"   no real generator: ||Im logm(NOT)|| = {cx['logm_imag_norm']:.4f} "
          f"(>0 => matrix log is complex)")
    print(f"   sqrt(NOT) eigenvalues = {cx['sqrtNOT_eigs']}  -> nontrivial = "
          f"{cx['nontrivial_eigenvalue']}")
    print(f"   FORCED PHASE = {cx['forced_phase_over_pi']:+.3f} * pi  "
          f"=> i ? {cx['is_i']}   (sqrt^2=NOT err {cx['sqrt_squares_to_NOT_err']:.1e}, "
          f"unitary err {cx['sqrtNOT_unitary_err']:.1e})")
    print(f"   continuous group U(t)=exp(-iHt): H real spectrum "
          f"{np.round(cx['H_real_spectrum'],3).tolist()}, U(1/2)^2=NOT err "
          f"{cx['Uhalf_squares_to_NOT_err']:.1e}, U(1)=NOT err {cx['Uone_is_NOT_err']:.1e}")

    print("\n  TRICHOTOMY  (Linear / Faithful / Real -- pick two):")
    print(f"   {'corner':<14}{'Linear':>8}{'Faithful':>10}{'Real':>7}   sacrifices")
    for nm, d, sac in [("nonlinearize", nl, "LINEARITY"),
                       ("coarse-grain", cg, "FAITHFULNESS"),
                       ("complexify", cx, "REALNESS -> i")]:
        print(f"   {nm:<14}{str(d['linear']):>8}{str(d['faithful']):>10}"
              f"{str(d['real']):>7}   {sac}")

    forced = cx["is_i"] and not cx["real_continuous_reversible_possible"]
    verdict = (
        "FORCED-i CONFIRMED (ground truth for Route A). Self-negation has no "
        "consistent Boolean value (the Liar obstruction). A real, faithful "
        "description can only oscillate (nonlinear) or collapse to a 1-bit gap "
        "(coarse-grained); the ONLY linear + faithful realization is unitary and "
        "carries eigenvalue i. The imaginary unit is not chosen -- it is forced by "
        "a topological obstruction (NOT is a reflection, det=-1, disconnected from "
        "the identity over the reals, connected over the unitaries). Discrete "
        "Stone's theorem: continuous+linear+reversible => imaginary generator => i. "
        "This is the conceptual ground truth; cwf_sr2 tests whether self-reference "
        "with NO definite ground produces the matching contextuality (CF>0)."
    ) if forced else "UNEXPECTED -- inspect."
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["SR1_liar_forced_i"] = dict(
        boolean_liar_fixed_points=liar, coarse_grain=cg, nonlinearize=nl,
        complexify=cx, forced=bool(forced), verdict=verdict,
        note=("Trichotomy on NOT=[[0,1],[1,0]]: nonlinearize (give up linearity, "
              "period-2 oscillation), coarse-grain (give up faithfulness, "
              "maximally-mixed 1-bit gap), complexify (give up realness -> i). The "
              "complexify corner is FORCED: NOT is a reflection (det -1), so no "
              "continuous real-orthogonal path I->NOT exists; the continuous "
              "reversible path is unitary, through sqrt(NOT), eigenvalue i. "
              "Discrete Stone's theorem. Ground truth for the self-reference->phase "
              "conjecture (Route A)."))
    json.dump(R, open(out, "w"), indent=2)
    print("\nWrote results.json key: SR1_liar_forced_i")


if __name__ == "__main__":
    main()
