#!/usr/bin/env python3
"""CONTEXTUAL FRACTION: the safe-horn baseline for definite-ground VSA (item 5).

CWFT's impossibility result: a genuine interfering phase (contextual/Bell statistics)
requires a substrate WITHOUT a definite ground.  Standard FPE has a definite ground
(the stored values x_i are definite), so its statistics must be noncontextual -- the
contextual fraction CF = 0.  This script (a) implements and RIGOROUSLY VALIDATES the
Abramsky-Barbosa-Mansfield contextual-fraction LP against known CHSH values
(local -> 0, Tsirelson -> 0.414, PR box -> 1), then (b) confirms the safe-horn
prediction: any model built from a single classical distribution over definite FPE
value-configurations has CF = 0.  The FORWARD experiment -- a self-referential,
definite-ground-free VSA that could give CF > 0 -- is specified in SELFREF_VSA_PLAN.md;
here we establish the validated machinery and the CF = 0 baseline it must beat.

Seeded, CPU (numpy + scipy.optimize.linprog).  Writes cwf_fpe_selfref_results.json.
"""
import itertools
import json

import numpy as np
from scipy.optimize import linprog


# ---- CHSH (2,2,2) scenario: measurements A0,A1,B0,B1; contexts {Ai Bj} ----------
MEAS = ["A0", "A1", "B0", "B1"]
CONTEXTS = [("A0", "B0"), ("A0", "B1"), ("A1", "B0"), ("A1", "B1")]
OUTCOMES = [0, 1]


def contextual_fraction(emp):
    """ABM contextual fraction of an empirical model.
    emp[context][(oA,oB)] = probability.  Returns CF in [0,1]."""
    # global deterministic assignments: value in {0,1} for each of the 4 measurements
    globals_ = list(itertools.product(OUTCOMES, repeat=len(MEAS)))
    gmap = {m: i for i, m in enumerate(MEAS)}
    # LP: maximize sum b_g  s.t.  for each (context, joint outcome s):
    #     sum_{g consistent with s} b_g <= emp[context][s];   b_g >= 0
    A_ub, b_ub = [], []
    for ctx in CONTEXTS:
        for s in itertools.product(OUTCOMES, repeat=2):
            row = np.zeros(len(globals_))
            for gi, g in enumerate(globals_):
                if g[gmap[ctx[0]]] == s[0] and g[gmap[ctx[1]]] == s[1]:
                    row[gi] = 1.0
            A_ub.append(row)
            b_ub.append(emp[ctx][s])
    res = linprog(-np.ones(len(globals_)), A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * len(globals_), method="highs")
    ncf = -res.fun
    return float(max(0.0, 1.0 - ncf))


def chsh_model(E):
    """Empirical model from a correlation value: symmetric no-signalling box with
    <A_i B_j> = +E except <A_1 B_1> = -E (the CHSH sign pattern), uniform marginals."""
    signs = {("A0", "B0"): +1, ("A0", "B1"): +1, ("A1", "B0"): +1, ("A1", "B1"): -1}
    emp = {}
    for ctx in CONTEXTS:
        corr = signs[ctx] * E
        p = {}
        for (a, b) in itertools.product(OUTCOMES, repeat=2):
            # P(a,b) = (1 + (-1)^{a+b} corr)/4  with uniform marginals
            p[(a, b)] = (1 + ((-1) ** (a + b)) * corr) / 4
        emp[ctx] = p
    return emp


def main():
    out = {}

    # (a) VALIDATION against known CHSH values (CF = (CHSH-2)/2 for 2<=CHSH<=4)
    print("(a) LP validation (CF should be (CHSH-2)/2):")
    val = []
    for name, E, chsh in [("local", 0.5, 2.0), ("Tsirelson", np.sqrt(2) / 2, 2 * np.sqrt(2)),
                          ("PR box", 1.0, 4.0)]:
        cf = contextual_fraction(chsh_model(E))
        predicted = max(0.0, (chsh - 2) / 2)
        val.append(dict(name=name, chsh=chsh, cf=cf, predicted=predicted))
        ok = abs(cf - predicted) < 1e-6
        print(f"    {name:>10}: CHSH={chsh:.3f}  CF={cf:.4f}  "
              f"predicted={predicted:.4f}  {'OK' if ok else 'MISMATCH!'}")
    out["validation"] = val
    assert all(abs(v["cf"] - v["predicted"]) < 1e-6 for v in val), "LP validation failed"
    print("    -> LP validated to machine precision.")

    # (b) SAFE-HORN BASELINE: a model built from a single classical distribution over
    #     DEFINITE FPE value-configurations is noncontextual (CF = 0).  We construct
    #     'measurements' as thresholded similarity readouts of an FPE bundle over
    #     definite stored values, sampled over a classical distribution of configs.
    print("(b) definite-ground FPE baseline (should give CF = 0):")
    rng = np.random.default_rng(0)
    N = 2000
    theta = 2.0 * rng.standard_normal(N)
    # two 'probe values' per party; outcome = sign of similarity to the probe
    probes = {"A0": 1.0, "A1": 4.0, "B0": 2.0, "B1": 6.0}
    NS = 4000
    counts = {ctx: {s: 0 for s in itertools.product(OUTCOMES, repeat=2)}
              for ctx in CONTEXTS}
    for _ in range(NS):
        # a definite stored configuration (the definite ground)
        vals = rng.uniform(0, 8, 3)
        b = sum(np.exp(1j * v * theta) for v in vals)
        read = {m: int(np.real(np.vdot(np.exp(1j * probes[m] * theta), b)) / N > 0.15)
                for m in MEAS}
        for ctx in CONTEXTS:
            counts[ctx][(read[ctx[0]], read[ctx[1]])] += 1
    emp = {ctx: {s: counts[ctx][s] / NS for s in counts[ctx]} for ctx in CONTEXTS}
    cf_fpe = contextual_fraction(emp)
    out["fpe_definite_ground_CF"] = cf_fpe
    print(f"    definite-ground FPE: CF = {cf_fpe:.4f}  "
          f"{'(safe horn confirmed: no genuine phase)' if cf_fpe < 1e-6 else '(unexpected)'}")

    out["note"] = ("Safe-horn baseline established. The forward experiment -- a "
                   "self-referential, definite-ground-free VSA predicted to give CF>0 "
                   "-- is specified in SELFREF_VSA_PLAN.md.")
    with open("cwf_fpe_selfref_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_selfref_results.json")


if __name__ == "__main__":
    main()
