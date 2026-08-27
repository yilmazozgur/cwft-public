"""
cwf_schrodinger_control.py -- CWFT reads stochastic optimal control / optimal
transport.  Demonstration (not new knowledge): the book's central metaphor -- a
tangled nonlinear process LIFTED into a clean linear wave (the Ch.0 dimensional-
lift figure) -- is, in this venue, a classical theorem: the Hopf-Cole transform.

  [Part 1] LINEARLY-SOLVABLE CONTROL = the lift to linearity, hbar_c = lambda.
    The value function V of a stochastic control problem obeys a NONLINEAR
    Hamilton-Jacobi-Bellman equation; the desirability z = exp(-V/lambda) obeys
    a LINEAR one (Todorov's linearly-solvable MDPs; Kappen's path-integral
    control).  lambda (control-cost / noise) IS hbar_c: lambda -> 0 is the
    classical (deterministic) limit; the optimal control is a Born-like
    reweighting of the passive dynamics by the "wavefunction" z.

  [Part 2] SCHRODINGER BRIDGE = entropic optimal transport, hbar^2 = epsilon.
    The entropic-regularized OT problem is exactly the Schrodinger bridge
    (Leonard).  The regularization epsilon plays the role of hbar^2: epsilon->0
    recovers classical (deterministic) optimal transport; epsilon>0 gives the
    diffuse "quantum" bridge.  Solved by Sinkhorn; the potentials are the
    bridge's wavefunctions.

Honest framing: pure re-description.  No new result.  The value is that the
book's lift-to-linearity is here a LITERAL wave with a LITERAL hbar_c -- a
cleaner fit than the (only semiclassical) prime case.  Honest fence: it is the
imaginary-time (heat) Schrodinger equation -- the real-amplitude / Wick-rotated
wave (the compression half), not the complex interfering phase.

CPU-only, numpy.  Writes cwf_schrodinger_control_results.json (new file).
"""

import json
import numpy as np


# =====================================================================
# Part 1 -- linearly-solvable control: the lift to linearity, hbar_c = lambda
# =====================================================================
def kl_control_chain(L, lam):
    """First-exit KL-control on a chain: goal=0 absorbing, reflecting at L-1,
    passive random walk, state cost 1 per interior step.  z solves a LINEAR
    system; V = -lambda log z is the (nonlinear) value function."""
    P = np.zeros((L, L))
    for x in range(1, L):
        if x == L - 1:
            P[x, x - 1] = 0.5; P[x, x] = 0.5
        else:
            P[x, x - 1] = 0.5; P[x, x + 1] = 0.5
    q = np.ones(L); q[0] = 0.0
    D = np.exp(-q / lam)
    idx = np.arange(1, L)
    G = D[idx, None] * P[np.ix_(idx, idx)]            # interior-interior
    b = D[idx] * P[idx, 0]                            # goal column (z[0]=1)
    z = np.empty(L); z[0] = 1.0
    z[idx] = np.linalg.solve(np.eye(L - 1) - G, b)    # LINEAR solve
    V = -lam * np.log(z)
    return z, V, P


def run_control():
    L = 20
    print("[Part 1] linearly-solvable control on a chain (L=20, goal=0)")
    print("  V obeys a nonlinear Bellman eq; z=exp(-V/lambda) obeys a LINEAR one")
    print("  (Hopf-Cole). lambda = hbar_c.  Reference x=10; det. shortest-path V=10.")
    print(f"  {'lambda':>8} {'V(10)':>8} {'V(10)/10':>9} {'u_left(10)':>11}")
    rows = []
    for lam in [0.2, 0.5, 1.0, 2.0, 5.0, 20.0]:
        z, V, P = kl_control_chain(L, lam)
        # optimal control u(x'|x) ~ p(x'|x) z(x') : Born-like reweighting
        x = 10
        w_left = P[x, x - 1] * z[x - 1]
        w_right = P[x, x + 1] * z[x + 1]
        u_left = w_left / (w_left + w_right)
        rows.append(dict(lam=lam, V10=float(V[10]), ratio=float(V[10] / 10),
                         u_left=float(u_left)))
        print(f"  {lam:>8.2f} {V[10]:>8.3f} {V[10]/10:>9.3f} {u_left:>11.3f}")
    print("  -> lambda->0 (classical): V(10)/10 -> 1 (deterministic shortest")
    print("     path) and u_left -> 1 (steer straight to goal).  lambda large:")
    print("     V grows and u_left -> 1/2 (passive diffusion).  z is the wave;")
    print("     the optimal control is z-reweighted passive dynamics (Born-like).")
    return dict(L=L, ref_state=10, sweep=rows)


# =====================================================================
# Part 2 -- Schrodinger bridge = entropic OT, hbar^2 = epsilon
# =====================================================================
def w2_1d(x, mu, nu, nq=4000):
    """Exact 1D quadratic OT cost via quantile (inverse-CDF) matching."""
    u = (np.arange(nq) + 0.5) / nq
    Qmu = x[np.searchsorted(np.cumsum(mu), u)]
    Qnu = x[np.searchsorted(np.cumsum(nu), u)]
    return float(np.mean((Qmu - Qnu) ** 2))


def sinkhorn(mu, nu, C, eps, iters=2000):
    """Entropic OT (Schrodinger bridge) via Sinkhorn; returns plan pi."""
    K = np.exp(-C / eps)
    u = np.ones_like(mu)
    for _ in range(iters):
        v = nu / (K.T @ u + 1e-300)
        u = mu / (K @ v + 1e-300)
    return u[:, None] * K * v[None, :]


def run_bridge():
    n = 60
    x = np.linspace(-2, 2, n)
    def gauss(c, s):
        g = np.exp(-0.5 * ((x - c) / s) ** 2); return g / g.sum()
    mu = gauss(-1.0, 0.3)
    nu = gauss(+1.0, 0.3)
    C = (x[:, None] - x[None, :]) ** 2
    ot = w2_1d(x, mu, nu)
    print(f"\n[Part 2] Schrodinger bridge = entropic OT (n={n}). hbar^2 = epsilon.")
    print(f"  exact classical OT cost W2^2 = {ot:.4f}")
    print(f"  {'epsilon':>8} {'<pi,C>':>9} {'cost/OT':>8} {'H(pi)':>8}")
    rows = []
    for eps in [4.0, 1.0, 0.25, 0.08]:
        pi = sinkhorn(mu, nu, C, eps)
        cost = float(np.sum(pi * C))
        H = float(-np.sum(pi * np.log(pi + 1e-300)))
        rows.append(dict(eps=eps, cost=cost, ratio=cost / ot, entropy=H))
        print(f"  {eps:>8.2f} {cost:>9.4f} {cost/ot:>8.3f} {H:>8.3f}")
    print("  -> epsilon -> 0 (classical): transport cost -> the exact OT cost")
    print("     and the plan sharpens (entropy falls).  epsilon large: the bridge")
    print("     diffuses (cost above OT, high entropy) -- the 'quantum' wave.")
    return dict(n=n, ot_cost=ot, sweep=rows)


if __name__ == "__main__":
    print("=" * 70)
    print("CWFT reads stochastic control / optimal transport (demonstration)")
    print("=" * 70)
    ctrl = run_control()
    bridge = run_bridge()
    with open("cwf_schrodinger_control_results.json", "w") as f:
        json.dump({"linearly_solvable_control": ctrl,
                   "schrodinger_bridge": bridge}, f, indent=2)
    print("\nwrote cwf_schrodinger_control_results.json")
