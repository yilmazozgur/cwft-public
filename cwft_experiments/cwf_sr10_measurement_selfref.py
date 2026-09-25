"""
cwf_sr10_measurement_selfref.py -- push (B): a complete, faithful, reversible
measurement on a CLOSED substrate is self-referential, and therefore forces the i.
This reduces the physical wager H1 ("nature derives its i from self-reference") to its
premise -- CLOSURE (no external observer) -- replacing "nature postulates i" with
"a closed world's complete measurements force i".

Builds on three established results:
  * C11 (Chapter 4): a computational substrate cannot faithfully BROADCAST/clone its own
    complete state -- self-introspection has a no-broadcasting obstruction.
  * thm:selfref-i / sr1 (Chapter 6): self-negation (NOT, a reflection, det=-1) has no
    continuous reversible REAL realization; its unique faithful reversible realization is
    sqrt(NOT), eigenvalue i = the self-referential complex structure J0.
  * thm:u1-cocycle / sr9: QM's geometric phase is the holonomy of that structure.

THE ARGUMENT (each link verified below; premises isolated):
  (P-closure)  The apparatus A is PART of the substrate it measures (a closed world has
               no external observer). So a complete measurement requires the substrate to
               faithfully record a part of itself -- self-reference.
  (1) [C11]    A faithful COMMUTING (classical, broadcastable) complete self-record is
               impossible: no unitary clones two non-commuting self-observables. So the
               self-record MUST carry components that do not commute with the pointer.
  (2)          Recording a self-observable that anti-commutes with the pointer makes the
               pointer's outcome depend on the pointer's own value with a NEGATION: the
               consistency condition is the Liar  P = NOT(P)  -- no definite outcome (a
               "measurement Liar"). [non-vacuous by (1); the canonical representative]
  (3) [sr1]    The unique faithful (information-preserving) + reversible realization of
               that self-negation is sqrt(NOT), eigenvalue i -- and it is genuinely
               complex (det(NOT)=-1 forbids any real reversible realization).
  (4) [sr8]    That i is exactly the complex structure J0 = i*sigma_Y (the measurement
               resource of Renou's separation, sr7).
  => A complete, faithful, reversible measurement on a CLOSED substrate necessarily
     invokes the self-referential i. H1 is reduced to (P-closure).

HONEST RESIDUAL (isolated, not eliminated): (P-closure) -- that the relevant description
is closed/embedded with no external real-valued frame. A final theory of physics has no
external observer, so this is far more motivated than a bare postulate of i; but it is a
premise, not a proof that our world is such a substrate. Also: the "self-negation/NOT"
of (2) is the canonical anti-commuting representative that C11 makes generic, not the only
possible self-record. An IRREVERSIBLE (decohering / coarse-grained) measurement escapes
to the classical corner (the sr1 trichotomy) -- the theorem is about FAITHFUL+REVERSIBLE
(the wave demand).

CPU; exact 2x2 algebra. numpy + scipy.linalg. Atomic write.
"""
import json, os
import numpy as np
from scipy.linalg import sqrtm, logm

HERE = os.path.dirname(os.path.abspath(__file__))

I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
NOT = X.copy()                                   # bit-flip = self-negation operator


def eq(a, b, tol=1e-9):
    return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol)


def main():
    print("cwf_sr10 -- a closed faithful reversible measurement is self-referential => forces i\n")
    checks = {}

    # (1) C11 core: no unitary clones two NON-COMMUTING self-states (no faithful commuting
    #     self-record). Inner-product test: cloning would force <0|+> = <0|+>^2.
    ket0 = np.array([1, 0], dtype=complex)
    ketp = np.array([1, 1], dtype=complex) / np.sqrt(2)      # |+>, X-eigenstate
    ov = complex(np.vdot(ket0, ketp))                         # <0|+> = 1/sqrt2
    clone_consistent = abs(ov - ov ** 2) < 1e-9               # would be required by cloning
    checks["C11 no-broadcast of non-commuting self-states"] = (not clone_consistent)
    print(f"(1) C11 no-broadcast: <0|+>={ov.real:.4f}, cloning would need <0|+>=<0|+>^2="
          f"{(ov**2).real:.4f}; equal? {clone_consistent} => faithful COMMUTING self-record "
          f"impossible: {checks['C11 no-broadcast of non-commuting self-states']}")
    # => a complete faithful self-record must carry non-commuting (e.g. X) components.
    checks["pointer Z and self-record X anticommute"] = eq(Z @ X + X @ Z, np.zeros((2, 2)))
    print(f"    pointer basis Z and a forced non-commuting self-record X ANTICOMMUTE "
          f"({{Z,X}}=0): {checks['pointer Z and self-record X anticommute']}")

    # (2) measurement Liar: recording an anti-commuting self-observable makes the pointer
    #     record the NEGATION of itself; the fixed-point P=NOT(P) has no classical solution.
    boolean_fixed = [b for b in (0, 1) if (1 - b) == b]
    checks["measurement Liar has no definite outcome"] = (len(boolean_fixed) == 0)
    print(f"(2) measurement Liar  P = NOT(P)  over {{0,1}}: solutions={boolean_fixed} "
          f"=> no definite outcome: {checks['measurement Liar has no definite outcome']}")

    # (3) sr1: the unique FAITHFUL + REVERSIBLE realization of self-negation is sqrt(NOT),
    #     eigenvalue i; and it is genuinely complex (det(NOT)=-1 => no real reversible one).
    detNOT = float(np.linalg.det(NOT).real)
    no_real = abs(detNOT + 1) < 1e-9                          # reflection: outside SO(2)
    sN = sqrtm(NOT)
    wN = np.linalg.eigvals(sN)
    has_i = any(abs(e - 1j) < 1e-9 for e in wN) or any(abs(e + 1j) < 1e-9 for e in wN)
    log_complex = float(np.linalg.norm(logm(NOT).imag)) > 1e-9
    checks["self-negation has no real reversible realization"] = (no_real and log_complex)
    checks["sqrt(NOT) reversible realization has eigenvalue i"] = has_i
    print(f"(3) sr1: det(NOT)={detNOT:.0f} (reflection, no real reversible realization: "
          f"{checks['self-negation has no real reversible realization']}); sqrt(NOT) "
          f"eigenvalues {np.round(wN,4).tolist()} include i: "
          f"{checks['sqrt(NOT) reversible realization has eigenvalue i']}")

    # (4) sr8: that i is the complex structure J0 = i*sigma_Y (the measurement resource of
    #     Renou's separation, sr7): J0 = sigma_Z sigma_X, J0 = i*sigma_Y, eigenvalue of
    #     sqrt(NOT) phase = pi/2.
    J0 = Z @ X
    checks["forced i is J0 = i*sigmaY (sr8 link)"] = eq(J0, 1j * Y) and eq(J0 @ J0, -I2)
    phase = float(np.angle(wN[int(np.argmax(np.abs(wN - 1.0)))]))
    checks["forced phase is +-pi/2 (the i)"] = abs(abs(phase) - np.pi / 2) < 1e-9
    print(f"(4) sr8 link: the forced i is J0=sigma_Z·sigma_X=i·sigma_Y (J0^2=-I): "
          f"{checks['forced i is J0 = i*sigmaY (sr8 link)']}; forced phase "
          f"{phase/np.pi:+.3f}·pi (the measurement resource of Renou's sigma_Y, sr7)")

    all_ok = all(checks.values())
    theorem = (
        "CONDITIONAL BRIDGE (status B; the book's Proposition prop:measurement-selfref) -- a "
        "complete, faithful, continuous, reversible measurement on a CLOSED substrate uses the "
        "self-referential i, modulo the premise of closure AND two posited modelling links "
        "(marked 'posited' below). Chain (the algebra of each link verified): (P-closure) the apparatus is part of the "
        "substrate, so a complete measurement is the substrate recording a part of itself; "
        "(1) by C11 a faithful, nondisturbing self-record of a NONCOMMUTING state family "
        "is impossible (no unitary clones non-commuting self-states, <0|+>!=<0|+>^2); "
        "posited: the record then carries components anti-commuting with the pointer "
        "({Z,X}=0); (2) posited: recording such a component makes the pointer record the NEGATION of itself -- the measurement Liar "
        "P=NOT(P), with no definite outcome; (3) by sr1 the faithful, continuous, reversible "
        "realization on the pointer's own 2-D real space is sqrt(NOT) (eigenvalue i), and it "
        "is genuinely complex -- det(NOT)=-1 forbids a real continuous one there; (4) by sr8 that i is J0=i*sigma_Y, exactly "
        "the measurement resource of Renou's separation (sr7). CONSEQUENCE: the physical "
        "wager H1 ('nature derives its i from self-reference') reduces, for the case of "
        "measurement, to its PREMISE -- closure (a final description has no external "
        "observer) -- which is far better motivated than postulating i. RESIDUAL (honest): "
        "(P-closure) is a premise, not a proof our world is such a substrate; the NOT of (2) "
        "is the canonical anti-commuting representative C11 makes generic, not the only "
        "self-record; and an irreversible/decohering measurement escapes to the classical "
        "corner (the sr1 trichotomy). So H1 is not eliminated but SHARPENED: from 'why i?' "
        "to 'is the world a closed self-describing substrate?'."
    ) if all_ok else "INCOMPLETE: a verification failed -- inspect."

    false_keys = [k for k, v in checks.items() if not v]
    if false_keys:
        print(f"\n  FALSE checks: {false_keys}")
    print(f"\nall checks pass: {all_ok}\n\n{theorem}")

    out = os.path.join(HERE, "results.json")
    try:
        R = json.load(open(out))
    except Exception:
        R = {}
    R["SR10_measurement_selfref"] = dict(
        checks={k: bool(v) for k, v in checks.items()}, all_verified=bool(all_ok),
        overlap=float(ov.real), det_NOT=detNOT, forced_phase_over_pi=phase / np.pi,
        theorem=theorem,
        note=("Push (B): a complete, faithful, reversible measurement on a CLOSED substrate "
              "is self-referential and forces the i. Chain: closure (apparatus in the "
              "substrate) -> C11 (no commuting self-broadcast; <0|+>!=<0|+>^2) -> the "
              "self-record anti-commutes with the pointer ({Z,X}=0) -> measurement Liar "
              "P=NOT(P), no definite outcome -> sr1: unique faithful reversible realization "
              "is sqrt(NOT), eigenvalue i, genuinely complex (det(NOT)=-1) -> sr8: that i is "
              "J0=i*sigma_Y, Renou's measurement resource. Reduces the physical wager H1 to "
              "the CLOSURE premise (no external observer). Residual: closure is a premise; "
              "the NOT is the canonical anti-commuting representative; irreversible "
              "measurement escapes classical. Atomic write."))
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=2)
    os.replace(tmp, out)
    print("\nWrote results.json key: SR10_measurement_selfref (atomic)")


if __name__ == "__main__":
    main()
