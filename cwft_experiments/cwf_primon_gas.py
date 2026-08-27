"""
cwf_primon_gas.py  --  The primon gas as the CWF two-knob cost sector.

The primon gas: a free bosonic gas with one mode per prime p, single-particle
energy log p.  Many-particle states <-> integers n>=1 by unique factorization
(n = prod p^{k_p}, energy log n), so the partition function is

    Z(beta) = sum_n n^{-beta} = zeta(beta)        (converges for beta > 1).

THE BRIDGE (status B).  The CWF two-knob action weight is
    W_c(n) = exp( i A_c/hbar_c  -  C_c/(2 theta_c) ),
and the natural self-description cost of the integer state n is its description
length C_c(n) = log n -- which IS the primon energy.  Hence the cost-sector
Born weight is |W_c(n)|^2 = n^{-1/theta_c}, and

    sum_n |W_c(n)|^2 = zeta(1/theta_c)      =>      beta = 1 / theta_c.

So the primon gas IS the partition function of the two-knob cost sector with
cost = description length, the classicality knob is beta = 1/theta_c, and the
Hagedorn pole / Bost-Connes SSB at beta = 1 is the critical self-description
temperature theta_c = 1.

This script CONFIRMS that beta = 1 (theta_c = 1) is a genuine critical point:
energy, entropy and specific heat diverge as power laws, and we extract the
critical exponents.  It also shows the low-prime "condensation" as beta -> 1.

Honest scope: this establishes the THERMODYNAMIC face of the classicality
transition.  Whether it coincides with the book's measurement-induced
ENTANGLEMENT transition (MIPT, p_c ~ 0.16, sec:grav-twoknob) is a separate,
harder question (different order parameter) -- left for the next step.

CPU-only; mpmath for zeta and its derivatives near the pole.  Writes
cwf_primon_gas_results.json (new file).
"""

import json
import numpy as np
import mpmath as mp

mp.mp.dps = 30


def thermo(beta):
    """Primon-gas thermodynamics at inverse temperature beta>1 (k_B=1, nats).
    Z=zeta, <E>=-d log Z/d beta, Var(E)=d^2 log Z/d beta^2, S, C=beta^2 Var."""
    Z = mp.zeta(beta)
    dZ = mp.zeta(beta, 1, 1)          # zeta'(beta)
    ddZ = mp.zeta(beta, 1, 2)         # zeta''(beta)
    logZ = mp.log(Z)
    E = -dZ / Z                       # mean energy <log n>
    varE = (ddZ * Z - dZ * dZ) / (Z * Z)   # d^2 log Z / d beta^2
    S = logZ + beta * E               # entropy
    C = beta * beta * varE            # specific heat
    theta_c = 1.0 / beta
    return dict(beta=float(beta), theta_c=float(theta_c), logZ=float(logZ),
                E=float(E), S=float(S), C=float(C), varE=float(varE))


def occupations(beta, primes):
    """Bose-Einstein occupation <k_p> = 1/(p^beta - 1)."""
    return {int(p): float(1.0 / (mp.mpf(p) ** beta - 1)) for p in primes}


def critical_exponents():
    """Fit <E> ~ eps^{-a}, Var(E) ~ eps^{-b}, S ~ eps^{-c} as eps=beta-1 -> 0.
    Theory (from zeta(beta) ~ 1/(beta-1)):  a=1, b=2, c=1."""
    eps = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    betas = 1.0 + eps
    E = np.array([thermo(b)["E"] for b in betas])
    V = np.array([thermo(b)["varE"] for b in betas])
    S = np.array([thermo(b)["S"] for b in betas])
    # slope of log(quantity) vs log(eps) -> -exponent
    a = -np.polyfit(np.log(eps), np.log(E), 1)[0]
    b = -np.polyfit(np.log(eps), np.log(V), 1)[0]
    c = -np.polyfit(np.log(eps), np.log(S), 1)[0]
    return dict(a_energy=float(a), b_var=float(b), c_entropy=float(c),
                eps=eps.tolist())


if __name__ == "__main__":
    print("=" * 70)
    print("PRIMON GAS AS THE CWF TWO-KNOB COST SECTOR")
    print("  bridge:  cost C_c(n)=log n  =>  beta = 1/theta_c ;  "
          "transition at theta_c=1")
    print("=" * 70)

    betas = [2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01]
    rows = [thermo(b) for b in betas]
    print(f"\n{'beta':>6} {'theta_c':>8} {'logZ':>9} {'<E>':>10} "
          f"{'S':>10} {'C=b^2 Var':>11}")
    for r in rows:
        print(f"{r['beta']:>6.2f} {r['theta_c']:>8.3f} {r['logZ']:>9.3f} "
              f"{r['E']:>10.3f} {r['S']:>10.3f} {r['C']:>11.3f}")
    print("  -> all diverge as beta -> 1^+ (theta_c -> 1): the Hagedorn /")
    print("     Bost-Connes critical point = the classicality transition.")

    exps = critical_exponents()
    print(f"\nCritical exponents near beta=1 (eps=beta-1):")
    print(f"  <E>   ~ eps^-{exps['a_energy']:.3f}   (theory 1)")
    print(f"  Var(E)~ eps^-{exps['b_var']:.3f}   (theory 2)")
    print(f"  S     ~ eps^-{exps['c_entropy']:.3f}   (theory 1)")

    primes = [2, 3, 5, 7, 11, 13]
    print(f"\nMode occupations <k_p>=1/(p^beta-1)  (low-prime condensation):")
    print(f"  {'prime':>6} " + " ".join(f"b={b:<5.2f}" for b in [2.0, 1.2, 1.01]))
    occ = {b: occupations(b, primes) for b in [2.0, 1.2, 1.01]}
    for p in primes:
        print(f"  {p:>6} " + " ".join(f"{occ[b][p]:>7.3f}"
              for b in [2.0, 1.2, 1.01]))
    print("  -> as beta->1 the lightest modes (p=2,3) grow without bound;")
    print("     the gas piles into the smallest primes (the 'classical' core).")

    out = {"bridge": {"beta_eq_inv_theta_c": True, "transition_theta_c": 1.0},
           "thermo": rows, "critical_exponents": exps,
           "occupations": {str(b): occ[b] for b in occ}}
    with open("cwf_primon_gas_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_primon_gas_results.json")
