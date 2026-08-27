"""
T1.1 Phase 1b -- a continuous-state RANDOMLY-COUPLED RECURRENT LATTICE (RCL).

A third substrate class for the trade-off-law robustness leg, distinct from the
discrete CAs and the SYMMETRIC-diffusive CMLs already in the catalog.

  Design note / honest negative: the first attempt was an autonomous tanh
  echo-state reservoir with the spectral radius rho as the chaos knob. It does NOT
  work -- an autonomous tanh map saturates to a stable fixed point where tanh'
  collapses, so the local Jacobian gain dies regardless of rho(W); measured a_c = 0
  at every rho in [1, 12]. (Sustained autonomous chaos in discrete tanh nets is
  genuinely hard; ESNs show edge-of-chaos only when INPUT-DRIVEN, which would break
  the autonomous-closure protocol the CA/CML pipeline relies on.)

  So this class keeps a reliable autonomous chaos engine (a logistic local map) but
  wires the sites with RANDOM, ASYMMETRIC, ring-banded coupling -- the reservoir/RNN
  connectivity structure, vs the CML's symmetric uniform nearest-neighbour diffusion:

      x_{t+1,i} = (1 - eps) f(x_i) + eps * sum_j C_ij f(x_j),   f(x) = a x (1 - x)

  C is a nonneg, ROW-STOCHASTIC, ring-banded (radius R) random matrix. Because the
  update is a convex combination of f-values in [0,1] (for a <= 4), the state stays
  bounded in [0,1] -- no clipping. The logistic parameter a (and coupling eps) is the
  ordered->chaotic knob, so the class spans a_c the way the CMLs do, but through a
  random asymmetric connectivity rather than symmetric diffusion. Scalar-per-site +
  ring-banded C means the spatial-window ladder and locality functional apply
  unchanged. The analytic Jacobian
      J = [(1 - eps) I + eps C] diag(f'(x)),   f'(x) = a (1 - 2x)
  feeds the same Pesin-KS estimator (sum of positive Lyapunov exponents, bits/step)
  the CMLs use, so a_c is measured on a common footing across all three classes.
"""
import numpy as np

LN2 = float(np.log(2))


def make_ring_C(n, R, seed):
    """Nonneg, row-stochastic, ring-banded (periodic, radius R) random coupling.
    Row-stochastic + convex update keeps the logistic state bounded in [0,1]."""
    rng = np.random.default_rng(seed)
    C = np.zeros((n, n))
    for i in range(n):
        for d in range(-R, R + 1):
            j = (i + d) % n
            C[i, j] = rng.random()
    C /= C.sum(1, keepdims=True)           # row-stochastic
    return C


def _step(x, a, eps, C):
    fx = a * x * (1.0 - x)
    return (1.0 - eps) * fx + eps * (C @ fx)


def collect_rcl(n, a, eps, C, n_traj, T, burn, seed):
    """Snapshot pairs (x_t, x_{t+1}) over n_traj trajectories (mirrors collect_cml)."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_traj):
        x = rng.random(n)
        for _ in range(burn):
            x = _step(x, a, eps, C)
        for _ in range(T):
            xn = _step(x, a, eps, C)
            X.append(x); Y.append(xn); x = xn
    return np.asarray(X), np.asarray(Y)


def rcl_ks_entropy(a, eps, C, n, T=600, burn=120, seed=0):
    """Pesin KS entropy (bits/step) = sum of positive Lyapunov exponents, via the
    analytic Jacobian J = [(1-eps) I + eps C] diag(a(1-2x)) and QR renormalisation.
    Same convention as cml_ks_entropy (non-negative info production)."""
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    M = (1.0 - eps) * np.eye(n) + eps * C
    for _ in range(burn):
        x = _step(x, a, eps, C)
    Q = np.eye(n); acc = np.zeros(n)
    for _ in range(T):
        fp = a * (1.0 - 2.0 * x)
        J = M * fp[None, :]                 # [(1-eps)I + eps C] @ diag(fp)
        Qn, Rm = np.linalg.qr(J @ Q)
        acc += np.log(np.abs(np.diag(Rm)) + 1e-300)
        Q = Qn
        x = _step(x, a, eps, C)
    lam = acc / T                           # nats/step
    return float(np.sum(lam[lam > 0]) / LN2)  # bits/step
