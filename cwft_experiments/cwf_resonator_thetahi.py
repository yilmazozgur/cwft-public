#!/usr/bin/env python3
"""HIGH-STATISTICS theta_c sweep for the resonator (n = 500 instances per point).

Same construction as cwf_resonator_halting.py Part 2 (N=256, D=128, F=2, bipolar,
T=150), but 10x the instances and a finer temperature grid, so the optimum carries
error bars (binomial SE ~ 0.022 at p=0.5).  Writes
cwf_resonator_thetahi_results.json; the committed part2_theta_c (n=50) is untouched.
"""
import importlib.util
import json

import numpy as np

spec = importlib.util.spec_from_file_location("rh", "cwf_resonator_halting.py")
rh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rh)

N, D, F, T = 256, 128, 2, 150
NINST = 500
THETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def main():
    rows = []
    print(f"N={N}, D={D}, F={F}, T={T}, n={NINST}/point")
    for th in THETAS:
        r = rh.prob(N, D, F, "bipolar", T, th, NINST, 90000, full=True)
        succ, nohalt, spur = r['succ'], r['timeout'], r['spurious_fp']
        se = float(np.sqrt(succ * (1 - succ) / NINST))
        rows.append(dict(theta=th, succ=float(succ), nohalt=float(nohalt), se=se,
                         ever=r['ever'], ever_se=r['ever_se'],
                         occupancy=r['occupancy'],
                         median_first_hit=r['median_first_hit'],
                         cycle_detected=r['cycle_detected'],
                         spurious_fp=r['spurious_fp']))
        print(f"  theta_c={th:>5}: final={succ:.3f}+/-{se:.3f}  "
              f"ever-correct={r['ever']:.3f}+/-{r['ever_se']:.3f}  "
              f"timeout={nohalt:.3f}  cycles detected={r['cycle_detected']:.3f}  "
              f"median first-hit={r['median_first_hit']:.0f}")
    print("  -> 'timeout' is a capped run, NOT a detected cycle: a real state repeat")
    print("     is searched for only in the deterministic arm, where the Jacobi update")
    print("     at F=2 gives EVEN-period attractors (no period-1 fixed point exists).")
    print("     'ever-correct' is first-hitting; a resonator can verify a candidate for")
    print("     free, so it is the operationally natural readout.")
    out = dict(params=dict(N=N, D=D, F=F, T=T, n_inst=NINST,
                           readout="final-state at cap AND first-hitting"), rows=rows)
    with open("cwf_resonator_thetahi_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_resonator_thetahi_results.json")


if __name__ == "__main__":
    main()
