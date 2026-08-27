"""
cwf_resonator_scaling.py -- does the resonator factorization crossover SHARPEN
into a genuine phase transition?  (Follow-on to cwf_resonator_halting.py §4b.)

§4b found the convergence transition at a CONSTANT load L_c~0.15 across N (the
M_c~N^2 collapse) but, at N<=512, it looked like a CROSSOVER, not a sharp
finite-size transition.  The decisive test: measure the transition WIDTH
    w(N) = L(success=0.2) - L(success=0.8)
on a fine load grid, for N up to 2048.
  - w(N) -> 0 as N grows (e.g. w ~ N^-a, a>0)  => a genuine phase transition
    (finite-size sharpening to a step function at L_c).
  - w(N) ~ const                                => a crossover (no transition).

Same verified F=2 bipolar resonator (imported).  Fixed L grid across N so the
curves are directly comparable.  CPU, numpy, seeded.  Writes
cwf_resonator_scaling_results.json.
"""

import json
import numpy as np
from cwf_resonator_halting import prob


def crossing(Ls, succ, level):
    """L where the (decreasing) success curve crosses `level`, linearly interp."""
    for i in range(len(succ) - 1):
        if succ[i] >= level >= succ[i + 1]:
            t = (succ[i] - level) / (succ[i] - succ[i + 1] + 1e-12)
            return float(Ls[i] + t * (Ls[i + 1] - Ls[i]))
    return None


if __name__ == "__main__":
    print("=" * 74)
    print("RESONATOR transition: does the crossover SHARPEN with N?  (width vs N)")
    print("=" * 74)
    F, T, NINST = 2, 120, 60
    Ls = [0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.36]
    Ns = [256, 512, 1024, 2048]
    print(f"  F={F}, {NINST} instances/point; load grid L = {Ls}")
    out = {"Ls": Ls, "N": Ns, "curves": {}, "L50": [], "width": []}
    for N in Ns:
        succ = []
        for L in Ls:
            D = max(2, int(round(np.sqrt(L) * N)))    # F=2: L = D^2/N^2 -> D = sqrt(L) N
            s, _, _ = prob(N, D, F, "bipolar", T, 0.0, NINST, 5000 + N)
            succ.append(s)
        L80 = crossing(Ls, succ, 0.8)
        L50 = crossing(Ls, succ, 0.5)
        L20 = crossing(Ls, succ, 0.2)
        w = (L20 - L80) if (L80 and L20) else None
        out["curves"][N] = succ
        out["L50"].append(L50); out["width"].append(w)
        print(f"  N={N:>5}: " + " ".join(f"{s:.2f}" for s in succ)
              + f"   | L50={L50:.3f}  width(L20-L80)={w}")
    print(f"\n  transition midpoint L50 vs N (should be ~constant = the M_c~N^2 collapse):")
    print(f"    " + "  ".join(f"N={n}:{l:.3f}" for n, l in zip(Ns, out['L50'])))
    ws = out["width"]
    print(f"  transition WIDTH vs N (the decisive number):")
    print(f"    " + "  ".join(f"N={n}:{(f'{w:.3f}' if w else 'NA')}" for n, w in zip(Ns, ws)))
    valid = [(n, w) for n, w in zip(Ns, ws) if w and w > 0]
    if len(valid) >= 2:
        lnN = np.log([n for n, _ in valid]); lnw = np.log([w for _, w in valid])
        slope = float(np.polyfit(lnN, lnw, 1)[0])
        out["width_scaling_exponent"] = slope
        ratio = valid[-1][1] / valid[0][1]
        print(f"  width ~ N^{slope:+.2f}  (width {valid[0][1]:.3f} at N={valid[0][0]} "
              f"-> {valid[-1][1]:.3f} at N={valid[-1][0]}, ratio {ratio:.2f})")
        verdict = ("SHARPENS -> genuine finite-size transition" if slope < -0.15
                   else "FLAT -> a crossover, not a sharp transition"
                   if slope > -0.05 else "weakly sharpening (inconclusive)")
        print(f"  -> VERDICT: {verdict}")
        out["verdict"] = verdict
    with open("cwf_resonator_scaling_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_resonator_scaling_results.json")
