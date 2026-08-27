"""
cwf_complex_attention_lm_sweep.py -- Route C on a real small transformer, CAPACITY
SWEEP.  Follow-on to cwf_complex_attention_lm.py.

The baseline run (d=48, 4000 steps) found the real transformer ALREADY solves
modular addition (~0.93) -- at capacity, so it groks the rotations itself and the
phase inductive bias has no room to help.  The route-C prediction is explicit that
the effect lives at LOW capacity (phase = a rotation inductive bias, not extra
expressivity).  So sweep model width d and measure the differential (phase - real)
on modular (U(1)) vs integer (no-wrap) addition at each capacity.

PREDICTION: at small d the real model cannot yet grok mod-add, and phase helps it
(positive (phase-real) on mod); the gap shrinks as d grows (real catches up); and
on int-add (no U(1) wraparound) phase does NOT help at any d.  The U(1)-specific
signature = a positive mod-gap that exceeds the int-gap, largest at low capacity.

Reports whatever happens, incl. no effect at any capacity (honest null).  Imports
the model + task from cwf_complex_attention_lm.py.  Seeded, CPU.  Writes
cwf_complex_attention_lm_sweep_results.json.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from cwf_complex_attention_lm import make_task, TinyTransformer


def run(kind, mode, seed, p, steps, d):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, n_cls, vocab = make_task(p, kind)
    N = len(X)
    perm = np.random.permutation(N)
    ntr = int(0.7 * N)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xte, yte = torch.tensor(X[te]), torch.tensor(y[te])
    torch.manual_seed(seed)
    model = TinyTransformer(vocab, n_cls, d=d, n_heads=2, mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1.0)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); lossf(model(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    return acc


if __name__ == "__main__":
    print("=" * 76)
    print("ROUTE C real transformer -- CAPACITY SWEEP (find where the U(1) bias shows)")
    print("=" * 76)
    P, STEPS, SEEDS = 31, 2500, [0, 1, 2]
    DIMS = [8, 12, 16, 24, 32]
    print(f"  p={P}, 2 heads, {STEPS} steps, mean over {len(SEEDS)} seeds; "
          f"sweeping d in {DIMS}")
    out = {"dims": DIMS, "mod": {}, "int": {}, "gap_mod": [], "gap_int": []}
    for kind in ["mod", "int"]:
        print(f"\n  [{kind}-add]   d:  " + "  ".join(f"{d:>5}" for d in DIMS))
        for m in ["real", "phase"]:
            row = []
            for d in DIMS:
                accs = [run(kind, m, sd, P, STEPS, d) for sd in SEEDS]
                row.append(float(np.mean(accs)))
            out[kind][m] = row
            print(f"            {m:>6}:  " + "  ".join(f"{a:>5.2f}" for a in row))
    out["gap_mod"] = [p - r for p, r in zip(out["mod"]["phase"], out["mod"]["real"])]
    out["gap_int"] = [p - r for p, r in zip(out["int"]["phase"], out["int"]["real"])]
    print(f"\n  (phase - real) gap:")
    print(f"     mod-add (U(1)):  " + "  ".join(f"{g:>+5.2f}" for g in out["gap_mod"]))
    print(f"     int-add (no-wrap):" + "  ".join(f"{g:>+5.2f}" for g in out["gap_int"]))
    best = max(range(len(DIMS)), key=lambda i: out["gap_mod"][i] - out["gap_int"][i])
    sep = out["gap_mod"][best] - out["gap_int"][best]
    print(f"  -> max U(1)-specific separation at d={DIMS[best]}: "
          f"mod-gap {out['gap_mod'][best]:+.2f} vs int-gap {out['gap_int'][best]:+.2f} "
          f"(sep {sep:+.2f})")
    print(f"     {'phase is U(1)-specific (helps mod>int at low capacity)' if sep > 0.05 and out['gap_mod'][best] > 0.03 else 'no clean U(1)-specific advantage -- honest null'}")
    with open("cwf_complex_attention_lm_sweep_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_complex_attention_lm_sweep_results.json")
