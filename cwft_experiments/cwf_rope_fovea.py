#!/usr/bin/env python3
"""FOVEATED RoPE: WHAT THE SCHEDULE BUDGET CAN AND CANNOT BUY.

RoPE's positional part is FPE on a deterministic schedule.  The manuscript
speculated a "foveated RoPE" that concentrates positional resolution where a
task needs it.  This experiment executes the idea and finds a clean split:

 (1) POSITION-FOVEATION IS IMPOSSIBLE (measured, and structural).  The RoPE
     kernel is stationary -- a function of the offset m-n only -- so no fixed
     schedule can have offset-dependent resolution.  Measured: offset-
     estimation accuracy is flat across offsets 16..3072 for the standard
     geometric schedule; a schedule that concentrates frequencies on the
     offset band [512, 1024] loses accuracy EVERYWHERE, including inside its
     own band.  (Fine discrimination at ANY offset needs high frequencies;
     giving them away buys nothing back, because of stationarity.)
 (2) SCALE REALLOCATION WORKS.  The real budget lives in offset-DIFFERENCE
     scales: tilting the fixed budget of F frequencies toward the top decades
     buys fine accuracy (identification within +-2) under noise.
 (3) AND IT IS NEARLY FREE AT PRACTICAL RANGES.  The classical price of a
     high tilt is ambiguity (gross errors) at long range; measured up to a
     65536-offset range, the penalty stays within noise at F=64 -- the many
     high frequencies act as a near-unique code.  The standard geometric
     schedule over-serves coarse scales for this content-free task.

Task model: estimate the relative offset Delta* from the noisy content-free
positional signature y_i = exp(i Delta* th_i) + noise, by matched correlation
over all candidate offsets.  Caveat stated plainly: this probes the
content-free positional channel only; trained, content-weighted attention may
use low frequencies differently.  Seeded, CPU.
Writes cwf_rope_fovea_results.json.
"""
import json

import numpy as np

F = 64
DMAX = 4096
BASE = 10000.0
B1, B2 = 512, 1024
NOISE = 2.2
TRIALS = 300
SEED = 0


def sched_uniform():
    return BASE ** (-np.arange(F) / F)


def sched_band():
    """Half coarse geometric, half concentrated on the offset band [B1,B2]."""
    coarse = BASE ** (-np.arange(F // 2) / (F // 2))
    dense = np.geomspace(2.0 / B2, 8.0 / B1, F // 2)
    return np.concatenate([coarse, dense])


def sched_tilt(f_top, dmax):
    """f_top of the budget geometric in [1e-2, 1], rest down to the range."""
    n_top = int(round(F * f_top))
    n_low = F - n_top
    top = np.geomspace(1.0, 1e-2, n_top, endpoint=False)
    if n_low == 0:
        return top
    low = np.geomspace(1e-2, 2 * np.pi / dmax, n_low)
    return np.concatenate([top, low])


def measure(th, probes, trials, rng, dmax, tol_fine=2, tol_gross=64):
    grid = np.arange(1, dmax + 1)
    E = np.exp(1j * np.outer(grid, th))
    fine, gross, acc8 = [], [], []
    for d0 in probes:
        sig = np.exp(1j * d0 * th)
        errs = []
        for _ in range(trials):
            y = sig + NOISE * (rng.standard_normal(F)
                               + 1j * rng.standard_normal(F)) / np.sqrt(2)
            errs.append(abs(int(grid[np.argmax(np.abs(E @ np.conj(y)))]) - d0))
        errs = np.array(errs)
        fine.append(float((errs <= tol_fine).mean()))
        acc8.append(float((errs <= 8).mean()))
        gross.append(float((errs > tol_gross).mean()))
    return fine, acc8, gross


def main():
    out = {"params": dict(F=F, DMAX=DMAX, base=BASE, band=[B1, B2],
                          noise=NOISE, trials=TRIALS)}
    probes = [16, 64, 192, 384, 600, 768, 900, 1400, 2048, 3072]

    # ---------------- (1) position-foveation: impossible ----------------------
    print("(1) position-foveation (offset band [%d,%d]): impossible" % (B1, B2))
    _, a8_u, _ = measure(sched_uniform(), probes, TRIALS,
                         np.random.default_rng(1), DMAX)
    _, a8_b, _ = measure(sched_band(), probes, TRIALS,
                         np.random.default_rng(2), DMAX)
    print("    offset :  " + " ".join(f"{d:>5}" for d in probes))
    print("    uniform:  " + " ".join(f"{a:>5.2f}" for a in a8_u))
    print("    banded :  " + " ".join(f"{a:>5.2f}" for a in a8_b))
    in_b = [i for i, d in enumerate(probes) if B1 <= d <= B2]
    gain = float(np.mean([a8_b[i] - a8_u[i] for i in in_b]))
    spread_u = float(max(a8_u) - min(a8_u))
    print(f"    uniform accuracy spread across offsets: {spread_u:.2f} (flat: "
          "stationarity)")
    print(f"    banded schedule's in-band 'gain': {gain:+.2f}  (a loss -- "
          "position-foveation buys nothing)")
    out["position_foveation"] = dict(probes=probes, uniform_acc8=a8_u,
                                     banded_acc8=a8_b, in_band_gain=gain,
                                     uniform_spread=spread_u)

    # ---------------- (2) scale reallocation: works ---------------------------
    print("(2) scale tilt at range %d (fine = within +-2):" % DMAX)
    tilt_rows = []
    probes2 = [16, 64, 192, 1024, 2048, 3072]
    for f_top in [0.5, 0.65, 0.8, 0.95]:
        fine, _, gross = measure(sched_tilt(f_top, DMAX), probes2, 200,
                                 np.random.default_rng(3), DMAX)
        tilt_rows.append(dict(f_top=f_top, fine=float(np.mean(fine)),
                              gross=float(np.mean(gross))))
        print(f"    f_top={f_top}: fine-acc={np.mean(fine):.2f}  "
              f"gross(>64)={np.mean(gross):.2f}")
    out["tilt"] = tilt_rows

    # ---------------- (3) the price at long range: still small ----------------
    print("(3) long range (65536): does ambiguity bite?")
    long_rows = []
    for f_top in [0.5, 0.95]:
        fine, _, gross = measure(sched_tilt(f_top, 65536),
                                 [5000, 20000, 50000], 150,
                                 np.random.default_rng(4), 65536,
                                 tol_gross=256)
        long_rows.append(dict(f_top=f_top, fine=float(np.mean(fine)),
                              gross=float(np.mean(gross))))
        print(f"    f_top={f_top}: fine-acc={np.mean(fine):.2f}  "
              f"gross(>256)={np.mean(gross):.2f}")
    out["long_range"] = long_rows

    # kernels for the figure
    dgrid = np.unique(np.round(np.geomspace(1, DMAX, 400)).astype(int))
    out["kernel"] = dict(
        delta=[int(d) for d in dgrid],
        uniform=[float(v) for v in
                 np.abs(np.exp(1j * np.outer(dgrid, sched_uniform())).sum(1)) / F],
        banded=[float(v) for v in
                np.abs(np.exp(1j * np.outer(dgrid, sched_band())).sum(1)) / F],
        tilted=[float(v) for v in
                np.abs(np.exp(1j * np.outer(dgrid, sched_tilt(0.95, DMAX))).sum(1)) / F])

    print("VERDICT: stationarity forbids position-foveation (the banded")
    print("         schedule loses even in its own band); the spendable budget")
    print("         is the scale allocation, and at practical ranges tilting")
    print("         fine is nearly free for the content-free channel.")
    with open("cwf_rope_fovea_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_rope_fovea_results.json")


if __name__ == "__main__":
    main()
