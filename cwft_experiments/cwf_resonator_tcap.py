#!/usr/bin/env python3
"""T-CAP CENSORING CONTROL for the resonator crossover width.

The transition width (cwf_resonator_scaling.py) is measured with the same iteration
cap T=120 at every N; if convergence times broaden with N, a flat width could be
cap-limited rather than dynamics-limited.  Control: the width at N=512 for
T in {120, 480, 1920} (60 instances/point, same load grid, first-crossing
interpolation).  If the width is T-stable, the crossover conclusion is not a
censoring artifact.  Writes cwf_resonator_tcap_results.json.
"""
import importlib.util
import json

import numpy as np

spec = importlib.util.spec_from_file_location("rs", "cwf_resonator_scaling.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)
spec_h = importlib.util.spec_from_file_location("rh", "cwf_resonator_halting.py")
rh = importlib.util.module_from_spec(spec_h)
spec_h.loader.exec_module(rh)

N, F, NINST = 512, 2, 60
LS = [0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.36]
TS = [120, 480, 1920]


def main():
    out = {"params": dict(N=N, F=F, n_inst=NINST, Ls=LS), "rows": []}
    print(f"N={N}, F={F}, {NINST} instances/point; T sweep {TS}")
    for T in TS:
        succ = []
        for L in LS:
            D = max(2, int(round(np.sqrt(L) * N)))
            s, _, _ = rh.prob(N, D, F, "bipolar", T, 0.0, NINST, 5000 + N)
            succ.append(float(s))
        L80 = rs.crossing(LS, succ, 0.8)
        L50 = rs.crossing(LS, succ, 0.5)
        L20 = rs.crossing(LS, succ, 0.2)
        w = (L20 - L80) if (L80 and L20) else None
        out["rows"].append(dict(T=T, succ=succ, L50=L50, width=w))
        print(f"  T={T:>5}: width(L20-L80) = {w}  L50 = {L50}")
    with open("cwf_resonator_tcap_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_resonator_tcap_results.json")


if __name__ == "__main__":
    main()
