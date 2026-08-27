"""
cwf_roadA_bbm.py -- Road A: does the CWF gain g=Re(s)-1/2 CONSTRAIN the
Bender-Brody-Muller operator, or merely relabel it?

BBM (Bender, Brody, Muller, PRL 118, 130201, 2017; arXiv:1608.03679):
    H = (1 - e^{-i p})^{-1} (x p + p x) (1 - e^{-i p}),
a formal conjugate of the symmetrized dilation operator xp+px = 2xp - i.  On
x^{-s} the eigenvalue of xp+px is i(2s-1) = 2i(s-1/2); reality of the spectrum
<=> RH; H is non-Hermitian (iH is PT-symmetric, broken).  Known criticism
(Comment, arXiv:1704.02644; note arXiv:1704.04705): the change of basis
T = 1 - e^{-ip} is non-unitary, so reality does not transfer from 2xp.

CWF gain (Ch.5 / this note's Proposition):  g(psi_s) = Im(E_s)/hbar_c = Re(s)-1/2.
For the BBM eigenvalue E = 2i(s-1/2):  Im(E) = 2(Re(s)-1/2) = 2 g.
So  g = (1/2) Im(E_BBM).

This script tests whether g adds any CONSTRAINT beyond Im(E):
  [1] identity g = (1/2) Im(E)  -- a relabel, not a new equation.
  [2] the real gap is T's NON-INVERTIBILITY (zero at p=0), not its non-unitarity:
      ANY invertible similarity preserves eigenvalues, so only non-invertibility
      lets the spectrum leave the real axis.
  [3] the gain-conservation law  sum g = 0  IS the functional equation s<->1-s:
      it holds for OFF-line pairs too -> RH-blind -> constrains nothing.
  [4] the Ch.5 gain-field STRUCTURE (spatial g(x), horizon, M_c) does NOT
      transfer: the prime modes have spatially-UNIFORM gain -> no horizon.

CPU-only, numpy.  Prints a verdict.  (No results.json: this is a logic test.)
"""

import numpy as np

rng = np.random.default_rng(0)

print("=" * 72)
print("ROAD A:  does the CWF gain g = Re(s)-1/2 constrain the BBM operator?")
print("=" * 72)

# ---- [1] eigenvalue / gain identity -----------------------------------
print("\n[1] identity  g = Re(s)-1/2   vs   (1/2) Im(E),   E_BBM = 2i(s-1/2)")
for s in [0.5 + 14.134j, 0.3 + 21.0j, 0.7 + 25.0j, 0.5 + 0j, 0.9 + 10.0j]:
    E = 2j * (s - 0.5)
    g = s.real - 0.5
    print(f"  s={s!s:>16}:  g={g:+.4f}   (1/2)Im(E)={0.5*E.imag:+.4f}   "
          f"match={np.isclose(g, 0.5*E.imag)}")
print("  -> g is ALGEBRAICALLY half Im(E): a relabel of the PT-breaking")
print("     parameter, not an independent equation.")

# ---- [2] the gap is non-INVERTIBILITY of T, not non-unitarity ---------
print("\n[2] a true (invertible) similarity preserves eigenvalues -- even a")
print("    non-UNITARY one -- so the spectrum can go complex ONLY because")
print("    T = 1 - e^{-ip} is non-INVERTIBLE (it vanishes at p=0).")
A = rng.standard_normal((4, 4)); A = A + A.T            # real spectrum
T = rng.standard_normal((4, 4)) + 2.5 * np.eye(4)       # invertible, non-unitary
nonunit = np.linalg.norm(T.conj().T @ T - np.eye(4))
Ap = np.linalg.solve(T, A) @ T                          # T^{-1} A T
eA = np.sort(np.linalg.eigvals(A).real)
eAp = np.sort(np.linalg.eigvals(Ap).real)
print(f"  invertible non-unitary T (||T^dag T - I||={nonunit:.2f}):")
print(f"    eig(A)      = {np.round(eA,4)}  (real)")
print(f"    eig(T^-1AT) = {np.round(eAp,4)}  (same, still real) "
      f"-> non-unitarity alone moves nothing")
p = np.linspace(-2*np.pi, 2*np.pi, 200001)
Tmag = np.abs(1 - np.exp(-1j * p))
print(f"  BBM's T=1-e^-ip:  min|T| = {Tmag.min():.2e} at p={p[np.argmin(Tmag)]:+.3f}"
      f"  -> |T^-1| -> infinity: NOT boundedly invertible.")
print("  -> the loophole that admits complex eigenvalues (RH-failure) is T's")
print("     non-invertibility -- exactly the BBM gap; g says nothing new about it.")

# ---- [3] gain conservation = functional equation, RH-blind -------------
print("\n[3] gain conservation  g(rho) + g(1-conj rho) = 0  (functional eq s<->1-s)")
for rho in [0.5 + 14.134j, 0.3 + 21.0j, 0.7 + 30.0j]:   # on-line AND off-line
    partner = 1 - np.conj(rho)
    g1, g2 = rho.real - 0.5, partner.real - 0.5
    tag = "on-line" if np.isclose(rho.real, 0.5) else "OFF-line"
    print(f"  rho={rho!s:>14} ({tag:>8}):  g(rho)={g1:+.3f}  "
          f"g(1-conj rho)={g2:+.3f}  sum={g1+g2:+.3f}")
print("  -> conservation holds for OFF-line pairs too: RH-blind, constrains nothing.")

# ---- [4] spatial uniformity of the prime-mode gain (no horizon) --------
print("\n[4] local gain g(x) of psi_s(x)=x^{-s} under the (unitary) dilation flow")
print("    V_{e^t}psi(x)=e^{t/2}psi(e^t x).  A Ch.5 horizon needs g(x) to VARY")
print("    in x and cross zero.  Numerically differentiate log|V_{e^t}psi_s(x)|:")
sigma, gamma, dt = 0.7, 12.0, 1e-5
def logmod_after_flow(x, t):
    xf = np.exp(t) * x
    return np.log(np.exp(t/2) * xf ** (-sigma))         # |e^{t/2}(e^t x)^{-s}|
xs = np.geomspace(1e-3, 1e3, 9)
rate = (logmod_after_flow(xs, dt) - logmod_after_flow(xs, -dt)) / (2 * dt)
print(f"  local gain g(x) at x in [1e-3,1e3]: mean={rate.mean():+.4f}, "
      f"std={rate.std():.2e}  (= 1/2 - sigma = {0.5-sigma:+.3f})")
print("  -> spatially UNIFORM (std~0): no zero-crossing, no horizon surface, no")
print("     M_c profile.  The Ch.5 gravity STRUCTURE does not transfer; only the")
print("     scalar VALUE g=Re(s)-1/2 does.")

# ---- verdict ----------------------------------------------------------
print("\n" + "=" * 72)
print("VERDICT:  g does NOT constrain BBM.")
print("  * g = (1/2) Im(E_BBM): an identity (relabel of the PT-breaking parameter).")
print("  * the one law it supplies, sum g = 0, IS the functional equation -> RH-blind.")
print("  * gain-free is guaranteed only on the self-adjoint dilation operator")
print("    (continuous spectrum, NO zeros selected); selecting the zeros needs the")
print("    non-invertible T, which voids that guarantee -- the same gap BBM and its")
print("    critics already locate.  CWF relabels the gap; it does not close it.")
print("  * the gravity STRUCTURE (horizon, M_c) does not transfer (g(x) is flat).")
print("  => Road A is a wall: the operator-level form of 'stage, not actors'.")
print("=" * 72)
