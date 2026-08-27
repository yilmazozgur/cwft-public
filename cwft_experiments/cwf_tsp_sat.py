"""
cwf_tsp_sat.py -- CWFT reads NP-hard combinatorial optimization.

Demonstration (not new knowledge): the classical statistical-mechanics-of-
optimization results fall out of CWFT's own machinery.  Two structures TSP /
combinatorial optimization already carry, each landing a CWFT axis:

  [Part 1] PARTITION FUNCTION = the two-knob COST SECTOR.
    Z(beta) = sum_{tours} e^{-beta C(tour)},  beta = 1/theta_c.
    The cost C is the self-description/energy; theta_c is the annealing
    temperature (the classicality knob).  As theta_c -> 0 the Boltzmann measure
    CONDENSES onto the optimal tour -- the ground-state "record" -- exactly the
    primon-gas classicality/records story (cwf_primon_records.py), now on an
    optimization landscape.

  [Part 2] HARDNESS PHASE TRANSITION = the IRREDUCIBILITY axis.
    Random 3-SAT at clause density alpha=m/n has a SAT/UNSAT transition near
    alpha_c ~ 4.27, with a search-cost (hardness) PEAK at the transition
    (Monasson-Zecchina-Kirkpatrick-Selman-Troyansky, Nature 1999).  This is the
    irreducibility axis the PRIMES could NOT reach (primes are low-K, decidable):
    here computational irreducibility appears as a genuine phase transition.

Honest framing: pure re-description of established stat-mech-of-optimization in
CWFT vocabulary.  No new result.  The value is that the two-knob cost sector and
the irreducibility axis both land cleanly -- and that TSP/SAT lands the axis the
prime substrate missed (primes = encoding without irreducibility; SAT =
irreducibility as a phase transition).

CPU-only, numpy + itertools.  Writes cwf_tsp_sat_results.json (new file).
"""

import json
import itertools
import numpy as np

RNG = np.random.default_rng(0)


# =====================================================================
# Part 1 -- TSP partition function = two-knob cost sector
# =====================================================================
def tsp_tours_and_costs(N, coords):
    """All (N-1)!/2 undirected tours from city 0; return their costs."""
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    costs = []
    for perm in itertools.permutations(range(1, N)):
        if perm[0] < perm[-1]:                      # canonical direction
            route = (0,) + perm + (0,)
            c = sum(D[route[i], route[i + 1]] for i in range(N))
            costs.append(c)
    return np.array(sorted(costs))                  # ascending; costs[0]=optimum


def tsp_partition(costs, beta):
    """Boltzmann statistics over tours at inverse temperature beta."""
    w = np.exp(-beta * (costs - costs[0]))          # shift by optimum (stability)
    Z = w.sum()
    p = w / Z
    p_opt = p[0]
    n_eff = 1.0 / np.sum(p ** 2)                     # effective # of tours
    return p_opt, n_eff


def run_tsp():
    N = 8
    coords = RNG.uniform(0, 1, size=(N, 2))
    costs = tsp_tours_and_costs(N, coords)
    print(f"[Part 1] TSP partition function (N={N}, {len(costs)} undirected tours)")
    print(f"  optimal tour cost = {costs[0]:.4f}, worst = {costs[-1]:.4f}")
    print(f"  {'theta_c':>8} {'beta':>7} {'p_opt':>9} {'N_eff(tours)':>13}")
    rows = []
    for beta in [1, 2, 5, 10, 20, 50, 100]:
        p_opt, n_eff = tsp_partition(costs, beta)
        rows.append(dict(theta_c=1.0 / beta, beta=beta, p_opt=float(p_opt),
                         n_eff=float(n_eff)))
        print(f"  {1.0/beta:>8.3f} {beta:>7d} {p_opt:>9.4f} {n_eff:>13.2f}")
    print("  -> cooling (theta_c -> 0) CONDENSES the measure onto the optimal")
    print("     tour: p_opt -> 1, N_eff -> 1.  Optimal tour = the ground-state")
    print("     'record'; freezing = the classicality transition (two-knob cost")
    print("     sector, same machinery as the primon gas).")
    return dict(N=N, opt_cost=float(costs[0]), sweep=rows)


# =====================================================================
# Part 2 -- random 3-SAT hardness transition = irreducibility axis
# =====================================================================
def random_3sat(n, m):
    """m random 3-clauses over n vars; literal = signed 1..n."""
    clauses = []
    for _ in range(m):
        vs = RNG.choice(np.arange(1, n + 1), size=3, replace=False)
        signs = RNG.choice([-1, 1], size=3)
        clauses.append(tuple(int(s * v) for s, v in zip(signs, vs)))
    return clauses


def dpll(clauses, n):
    """DPLL with unit propagation; returns (sat, num_calls)."""
    calls = [0]

    def solve(cl, assign):
        calls[0] += 1
        simp = []
        for c in cl:
            nc, sat = [], False
            for lit in c:
                v = abs(lit)
                if v in assign:
                    if (assign[v] == 1) == (lit > 0):
                        sat = True
                        break
                else:
                    nc.append(lit)
            if sat:
                continue
            if not nc:
                return False                        # empty clause -> conflict
            simp.append(nc)
        if not simp:
            return True                             # all clauses satisfied
        for c in simp:                              # unit propagation
            if len(c) == 1:
                lit = c[0]
                a2 = dict(assign); a2[abs(lit)] = 1 if lit > 0 else 0
                return solve(simp, a2)
        v = abs(simp[0][0])                         # branch
        for val in (1, 0):
            a2 = dict(assign); a2[v] = val
            if solve(simp, a2):
                return True
        return False

    return solve(clauses, {}), calls[0]


def run_sat():
    n, n_inst = 22, 60
    alphas = [2.0, 3.0, 3.5, 4.0, 4.2, 4.5, 5.0, 5.5, 6.0, 7.0]
    print(f"\n[Part 2] random 3-SAT hardness transition (n={n} vars, "
          f"{n_inst} instances/point)")
    print(f"  alpha_c ~ 4.27 (literature).  Expect P(sat) to cross 1/2 there and")
    print(f"  the search cost (DPLL calls) to PEAK there.")
    print(f"  {'alpha':>6} {'m':>4} {'P(sat)':>8} {'median_calls':>13}")
    rows = []
    for a in alphas:
        m = int(round(a * n))
        sats, calls = [], []
        for _ in range(n_inst):
            cl = random_3sat(n, m)
            s, c = dpll(cl, n)
            sats.append(s); calls.append(c)
        psat = float(np.mean(sats))
        medc = float(np.median(calls))
        rows.append(dict(alpha=a, m=m, p_sat=psat, median_calls=medc))
        print(f"  {a:>6.2f} {m:>4d} {psat:>8.3f} {medc:>13.1f}")
    peak = max(rows, key=lambda r: r["median_calls"])
    print(f"  -> hardness peaks at alpha={peak['alpha']} (median "
          f"{peak['median_calls']:.0f} calls); P(sat) drops through ~1/2 near")
    print(f"     alpha_c.  The hardness peak = computational IRREDUCIBILITY as a")
    print(f"     phase transition -- the axis the prime substrate could not reach.")
    return dict(n=n, n_inst=n_inst, sweep=rows, hardness_peak_alpha=peak["alpha"])


if __name__ == "__main__":
    print("=" * 70)
    print("CWFT reads NP-hard combinatorial optimization (demonstration)")
    print("=" * 70)
    tsp = run_tsp()
    sat = run_sat()
    with open("cwf_tsp_sat_results.json", "w") as f:
        json.dump({"tsp_cost_sector": tsp, "sat_irreducibility": sat}, f, indent=2)
    print("\nwrote cwf_tsp_sat_results.json")
