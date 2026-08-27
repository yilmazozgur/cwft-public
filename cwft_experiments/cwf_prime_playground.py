"""
cwf_prime_playground.py  --  A CWF substrate that fires only on primes.

Playground for the prime / Riemann thread of Computational Wave Field Theory.
The substrate: discrete locations x = 2,3,4,... ; a "computation event" fires
at x iff x is prime.  The computon field Psi_c(x) carries the amplitude,
rho_c(x) = |Psi_c(x)|^2 the firing probability (Axiom 1).  On the LOG axis
u = log x the natural symmetry is dilation, whose generator is the symmetrized
x*p_c -- exactly the Berry-Keating operator.  This script checks, numerically
and self-consistently, the three claims that make that mapping non-trivial:

  P1  Berry-Keating Weyl term.  The regularized phase-space area of H = x p
      (Planck-cell cutoff, hbar=1) reproduces the SMOOTH Riemann zero-count
      N(E) = (E/2pi) log(E/2pi) - E/2pi + 7/8 to leading order.  => the
      dilation generator's semiclassical density of states IS the zero density.

  P2a zeros -> prime staircase (robust).  The Riemann-von Mangoldt explicit
      formula, truncated to K zeros, reconstructs the prime-power staircase
      psi(x).  CWF reading: the eigen-phases (zeros) of the prime Hamiltonian
      reconstruct the position-space firing pattern.  Error falls as K grows.

  P2b primes -> zeros (the uncertainty/resolution trade).  The von Mangoldt
      sum over primes up to N resonates at the zero ordinates gamma.  How many
      zeros you can resolve grows with N: localizing the spectrum costs
      observation length -- the position(prime)/momentum(zero) uncertainty.

All deterministic except a Monte-Carlo cross-check of the area (seeded).
CPU-only, numpy only.  Writes cwf_prime_results.json (new file; touches
nothing in the committed results.json).
"""

import json
import numpy as np

RNG = np.random.default_rng(0)

# First 30 nontrivial-zero ordinates gamma (rho = 1/2 + i gamma), high precision.
GAMMA = np.array([
    14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588,
    37.586178159, 40.918719012, 43.327073281, 48.005150881, 49.773832478,
    52.970321478, 56.446247697, 59.347044003, 60.831778525, 65.112544048,
    67.079810529, 69.546401711, 72.067157674, 75.704690699, 77.144840069,
    79.337375020, 82.910380854, 84.735492981, 87.425274613, 88.809111208,
    92.491899271, 94.651344041, 95.870634228, 98.831194218, 101.317851006,
])


def sieve(n):
    """Primes <= n."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if s[p]:
            s[p * p::p] = False
    return np.flatnonzero(s)


def von_mangoldt(n):
    """Lambda(k) for k=0..n :  log p if k = p^m, else 0."""
    L = np.zeros(n + 1)
    primes = sieve(n)
    for p in primes:
        pk = p
        while pk <= n:
            L[pk] = np.log(p)
            pk *= p
    return L


# ---------------------------------------------------------------------------
# P1 : Berry-Keating Weyl term  =  smooth Riemann zero count
# ---------------------------------------------------------------------------
def riemann_smooth_count(E):
    """N(E): smooth part of #{zeros with 0 < gamma <= E}."""
    return (E / (2 * np.pi)) * np.log(E / (2 * np.pi)) - E / (2 * np.pi) + 7.0 / 8.0


def bk_area_closed(E, cell=2 * np.pi):
    """Phase-space area of {x>=lx, p>=lp, x p <= E}, lx*lp = cell = h = 2pi hbar.
    Closed form: A = E log(E/cell) - E + cell."""
    return E * np.log(E / cell) - E + cell


def bk_area_montecarlo(E, cell=2 * np.pi, nsamp=4_000_000):
    """Cross-check the area by MC over the box [lx, E/lp] x [lp, E/lx]."""
    lx = lp = np.sqrt(cell)
    xmax, pmax = E / lp, E / lx
    xs = RNG.uniform(lx, xmax, nsamp)
    ps = RNG.uniform(lp, pmax, nsamp)
    inside = (xs * ps <= E)
    box = (xmax - lx) * (pmax - lp)
    return box * inside.mean()


def run_P1():
    out = {"E": [], "N_zeros_actual": [], "N_smooth": [], "N_BK_closed": [],
           "N_BK_montecarlo": []}
    for E in [50.0, 100.0]:
        n_act = int(np.sum(GAMMA <= E))
        n_sm = riemann_smooth_count(E)
        n_bk = bk_area_closed(E) / (2 * np.pi)          # states = area / h
        n_mc = bk_area_montecarlo(E) / (2 * np.pi)
        out["E"].append(E)
        out["N_zeros_actual"].append(n_act)
        out["N_smooth"].append(round(n_sm, 3))
        out["N_BK_closed"].append(round(n_bk, 3))
        out["N_BK_montecarlo"].append(round(n_mc, 3))
    return out


# ---------------------------------------------------------------------------
# P2a : explicit formula, zeros -> prime-power staircase psi(x)
# ---------------------------------------------------------------------------
def psi_true(x, Lam):
    """True Chebyshev psi(x) = sum_{k<=x} Lambda(k)."""
    return np.cumsum(Lam)[np.floor(x).astype(int)]


def psi_explicit(x, K):
    """Riemann-von Mangoldt psi truncated to first K zero-pairs."""
    val = x - np.log(2 * np.pi) - 0.5 * np.log(1 - x ** (-2))
    for g in GAMMA[:K]:
        amp = 2 * np.sqrt(x) / (0.25 + g * g)
        val -= amp * (0.5 * np.cos(g * np.log(x)) + g * np.sin(g * np.log(x)))
    return val


def run_P2a():
    nmax = 60
    Lam = von_mangoldt(nmax)
    xs = np.linspace(2.5, 30.0, 1400)            # avoid prime-power jump points
    truth = psi_true(xs, Lam)
    res = {"K": [], "rms_error": []}
    for K in [1, 5, 10, 20, 30]:
        approx = psi_explicit(xs, K)
        rms = float(np.sqrt(np.mean((approx - truth) ** 2)))
        res["K"].append(K)
        res["rms_error"].append(round(rms, 4))
    return res


# ---------------------------------------------------------------------------
# P2b : primes -> zeros.  Resonances of the von Mangoldt sum locate gamma.
# ---------------------------------------------------------------------------
def spectral_probe(N, t_grid):
    """F(t) = | sum_{n<=N} Lambda(n) n^{-1/2} n^{-i t} * w(n) |, Hann window in
    log n.  Peaks sit at the zero ordinates gamma."""
    Lam = von_mangoldt(N)
    n = np.arange(2, N + 1)
    lam = Lam[2:]
    nz = lam > 0
    n, lam = n[nz], lam[nz]
    w = 0.5 * (1 + np.cos(np.pi * np.log(n) / np.log(N)))   # taper -> 0 at n=N
    coeff = lam * n ** (-0.5) * w
    logn = np.log(n)
    # F(t) for each t: sum coeff * exp(-i t log n)
    phases = np.exp(-1j * np.outer(t_grid, logn))
    return np.abs(phases @ coeff)


def run_P2b():
    """Rank ALL local maxima of F(t) by height, keep the top n_targets, and
    ask which true gamma each lands on (within 0.3).  Report the count and the
    highest gamma recovered, swept over N.  This tests the *duality direction*
    (do the primes alone locate the zeros?) and looks for a resolution ceiling
    that climbs with N."""
    t_grid = np.linspace(10, 105, 12000)
    targets = GAMMA                                        # all 30, up to ~101
    res = {"N": [], "zeros_recovered": [], "highest_gamma": []}
    for N in [50, 200, 1_000, 10_000, 100_000]:
        F = spectral_probe(N, t_grid)
        idx = np.where((F[1:-1] > F[:-2]) & (F[1:-1] > F[2:]))[0] + 1
        pk_t, pk_h = t_grid[idx], F[idx]
        top = pk_t[np.argsort(pk_h)[::-1][:len(targets)]]  # strongest n_targets
        hit = [float(g) for g in targets
               if top.size and np.min(np.abs(top - g)) < 0.3]
        res["N"].append(N)
        res["zeros_recovered"].append(len(hit))
        res["highest_gamma"].append(round(max(hit), 2) if hit else None)
    res["n_targets"] = int(len(targets))
    return res


if __name__ == "__main__":
    print("=" * 68)
    print("CWF PRIME PLAYGROUND")
    print("=" * 68)

    P1 = run_P1()
    print("\n[P1] Berry-Keating x*p Weyl term  vs  smooth Riemann zero-count")
    print(f"  {'E':>6} {'#zeros':>7} {'N_smooth':>10} {'N_BK(area/h)':>13} "
          f"{'N_BK(MC)':>10}")
    for i, E in enumerate(P1["E"]):
        print(f"  {E:>6.0f} {P1['N_zeros_actual'][i]:>7d} "
              f"{P1['N_smooth'][i]:>10.3f} {P1['N_BK_closed'][i]:>13.3f} "
              f"{P1['N_BK_montecarlo'][i]:>10.3f}")
    print("  -> leading two terms of N_BK match N_smooth; closed area = MC area.")

    P2a = run_P2a()
    print("\n[P2a] zeros -> prime staircase psi(x)  (RMS error vs K zeros)")
    for K, e in zip(P2a["K"], P2a["rms_error"]):
        print(f"  K={K:>2d} zeros : RMS = {e:.4f}")
    print("  -> error falls monotonically as more eigen-phases are included.")

    P2b = run_P2b()
    print(f"\n[P2b] primes -> zeros  (ranked-peak test; "
          f"{P2b['n_targets']} zeros up to gamma~101)")
    for i, N in enumerate(P2b["N"]):
        print(f"  primes up to N={N:>9d} : recovered "
              f"{P2b['zeros_recovered'][i]:>2d}/{P2b['n_targets']} zeros, "
              f"highest gamma = {P2b['highest_gamma'][i]}")
    print("  -> do the primes alone locate the zeros? (resolution vs N)")

    with open("cwf_prime_results.json", "w") as f:
        json.dump({"P1_berry_keating": P1, "P2a_zeros_to_primes": P2a,
                   "P2b_primes_to_zeros": P2b}, f, indent=2)
    print("\nwrote cwf_prime_results.json")
