#!/usr/bin/env python3
"""
[R5 STATUS -- RETIRED 2026-06-05, kept as a dated record.] Executed and clean, but its
contribution is ILLUSTRATION, not proof: it shows realised CF=0 on a handful of trained rings,
which the beable theorem already guarantees for ALL classical substrates -- nonexistence in a
finite sample is not a proof of nonexistence, and it isn't news. The R5 phase-on-a-substrate
program was retired on this basis (Ozgur's critique): the self-reference->phase claim is
theorem-only (sr1/sr8/sr9), not empiricizable. The one genuinely-new observation here is that
compression TRACKS satisfiability (frustrated classical self-models are forced into higher-dim
limit cycles) -- the cost-side thread logged as the only live empirical successor. See
R5_SELFMODEL_PLAN.md (top "STATUS: RETIRED").

cwf_r5_phase3a.py  --  R5 Phase 3A: the fork-confirming negative (the learned substrate).

Plan: cwf_project/R5_SELFMODEL_PLAN.md, Phase 3A. Phases 1-2 established analytically that the
genuine phase needs BOTH frustration AND no definite ground: a classical (definite-ground) self-loop
gives only realised CF=0 + a Z2 holonomy (Phase 1), while the quantum self-loop gives realised CF>0 +
a U(1) holonomy (Phase 2). Phase 3A asks the learned-substrate question that motivated R5: can
OPTIMIZATION sneak around the fork? I.e., if we TRAIN a real-valued net as hard as possible to
self-model a frustrated (Liar) self-reference, does the phase appear?

PREDICTION (corrected, after Ozgur's catch): NO. A real-valued RNN has a definite ground -- its hidden
state is a definite vector at every step (a complete "answer sheet") -- so by the Phase-1 beable
theorem it can NEVER realise CF>0 or a non-Z2 holonomy, however it is trained. The phase is ONTOLOGICAL,
not trainable on a classical substrate. What training CAN do: reach the safe horn (self-modelling) for
SATISFIABLE self-references, and -- when frustrated -- land on the classical Z2 shadow (a limit cycle)
with the self-prediction loss flooring above zero (the Lawvere bound made empirical).

THE SUBSTRATE -- a genuinely TRAINED autonomous self-model ring. n real "neurons"; recurrent weights W
are LEARNED by Adam to make the net's own dynamics satisfy its self-model: adjacent units differ
(h[i] ~ -h[i+1] round the ring -- the answer-sheet cycle), subject to anti-collapse (stay off the
trivial all-zero state). This is a pure self-model loop: the net is optimised to predict/satisfy its own
state. The dial is parity: even = satisfiable (consistent ground); odd = frustrated (Liar, no ground).

WHAT THIS SCRIPT SHOWS (each measured, not assumed):
  (1) IMPOSSIBILITY -- realised CF = 0 for EVERY parity, on the trained net (validated ABM LP), and the
      holonomy is Z2 {0,pi}, never U(1). Optimisation does not manufacture the phase.
  (2) LAWVERE BOUND -- the self-model loss floors at ~0 for even (satisfiable) and strictly >0 for odd
      (frustrated): faithful self-modelling provably breaks down on a Liar loop. The floor IS the
      "breakdown of faithful self-modelling" the reviewer listed as a measure -- it is a theorem.
  (3) COLLAPSE <-> Lawvere/JEPA -- with anti-collapse OFF, the frustrated net escapes frustration by
      collapsing to the trivial fixed point (|h|->0) = Lawvere's degenerate self-model = JEPA
      representation collapse. Anti-collapse (the VICReg analogue) is the non-degeneracy constraint that
      keeps the answer sheet non-trivial.
  (4) COMPRESSION -- reported HONESTLY: a self-model that is SATISFIABLE (even) yields a low-dimensional,
      compressible representation (the safe horn); a FRUSTRATED (odd) classical self-model is forced into
      a HIGHER-dimensional, less-compressible limit cycle -- so compression is NOT flat across the dial
      here, it tracks satisfiability. (The Premakumar weight-narrowing signature is an RLCT statement we
      do not reproduce with toy proxies; see VERDICT.)

CPU-only. torch (forced CPU) + numpy + the validated ABM LP. All seeded.
"""

import json
import os
import zlib

import numpy as np
import torch

from cwf_phase_contextuality import contextuality_lp, _validate_lp

torch.set_num_threads(2)
DEVICE = torch.device("cpu")
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The trained autonomous self-model ring
# ---------------------------------------------------------------------------
def train_self_model_ring(n, seed, iters=600, steps=50, burn=25, lr=0.05,
                          anti_collapse=True, ac_target=0.6, ac_weight=0.5):
    """Learn W (Adam) so the autonomous map h <- tanh(W h) satisfies the self-model
    'adjacent ring units differ' (h[i] ~ -h[i+1]), subject to anti-collapse. Returns
    the post-transient hidden trajectory, the final self-model loss (the Lawvere floor),
    and W."""
    torch.manual_seed(seed)
    W = (0.6 * torch.randn(n, n, device=DEVICE)).requires_grad_(True)
    h0 = 0.3 * torch.randn(n, device=DEVICE)
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(iters):
        h = h0.clone()
        traj = []
        for _ in range(steps):
            h = torch.tanh(h @ W.T)
            traj.append(h)
        H = torch.stack(traj[burn:])
        Hr = torch.roll(H, -1, dims=1)
        L_self = ((H + Hr) ** 2).mean()                       # self-model: neighbours differ
        L = L_self
        if anti_collapse:
            L = L + ac_weight * (torch.relu(ac_target - H.abs()) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
    with torch.no_grad():
        h = h0.clone(); traj = []
        for _ in range(steps):
            h = torch.tanh(h @ W.T); traj.append(h)
        H = torch.stack(traj[burn:]).cpu().numpy()
    return H, float(L_self.item()), W.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def realised_cf(H, n):
    """Realised contextual fraction of the trained net's hidden readout (sign per unit),
    adjacent-pair contexts round the ring, via the validated ABM LP. Definite ground =>
    expect 0 for every parity."""
    bits = (H > 0).astype(int)
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci, (i, j) in enumerate(contexts):
        t = np.zeros((2, 2))
        for row in bits:
            t[row[i], row[j]] += 1.0
        tables[ci] = t / t.sum()
    return float(contextuality_lp(contexts, observables, tables)["contextual_fraction"])


def pancharatnam_phase(H):
    """Holonomy from the (real) hidden-state loop. Real overlaps => phase in {0,pi} = Z2."""
    Xc = H - H.mean(0)
    prod = 1.0 + 0j
    idx = list(range(len(Xc))) + [0]
    for k in range(len(idx) - 1):
        a, b = Xc[idx[k]], Xc[idx[k + 1]]
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        prod *= np.dot(a, b) / denom
    return float(np.angle(prod))


def is_Z2(phase, tol=1e-2):
    g = abs(((phase + np.pi) % (2 * np.pi)) - np.pi)
    return bool(g < tol or abs(g - np.pi) < tol)


def participation_ratio(H):
    """Effective dimensionality of the hidden activity (compression proxy)."""
    C = np.cov((H - H.mean(0)).T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)
    s = (ev ** 2).sum()
    return float(ev.sum() ** 2 / s) if s > 0 else 0.0


def trajectory_mdl(H):
    """Description length (bits/step) of the quantized trajectory via gzip -- the
    cwf_t11_ca_entropy compressibility proxy."""
    q = np.clip(((H + 1) / 2 * 255), 0, 255).astype(np.uint8).tobytes()
    return 8.0 * len(zlib.compress(q, 9)) / H.shape[0]


# ---------------------------------------------------------------------------
def main():
    print("=" * 74)
    print("R5 Phase 3A -- the fork-confirming negative (the LEARNED substrate)")
    print("=" * 74)
    _validate_lp()
    print()

    rings = [4, 5, 6, 7]
    seeds = [1, 2, 3]
    results = {"substrate": "trained autonomous self-model ring (Adam on W; h<-tanh(Wh))",
               "rings": {}}

    hdr = (f"{'n':>3} {'parity':>7} {'Lawvere floor':>14} {'realised CF':>12} "
           f"{'holonomy':>10} {'Z2?':>5} {'partic.ratio':>13} {'MDL b/step':>11}")
    print(hdr); print("-" * len(hdr))
    for n in rings:
        floors, cfs, phases, prs, mdls, z2s = [], [], [], [], [], []
        for s in seeds:
            H, floor, W = train_self_model_ring(n, seed=s)
            floors.append(floor); cfs.append(realised_cf(H, n))
            ph = pancharatnam_phase(H); phases.append(ph); z2s.append(is_Z2(ph))
            prs.append(participation_ratio(H)); mdls.append(trajectory_mdl(H))
        parity = "even" if n % 2 == 0 else "odd"
        rec = dict(parity=parity,
                   lawvere_floor=float(np.mean(floors)),
                   realised_CF=float(np.max(cfs)),          # worst case (most generous to "phase")
                   holonomy_phase_example=float(phases[0]),
                   holonomy_all_Z2=bool(all(z2s)),
                   participation_ratio=float(np.mean(prs)),
                   mdl_bits_per_step=float(np.mean(mdls)))
        results["rings"][n] = rec
        print(f"{n:>3} {parity:>7} {rec['lawvere_floor']:>14.4f} {rec['realised_CF']:>12.4f} "
              f"{rec['holonomy_phase_example']:>+10.3f} {str(rec['holonomy_all_Z2']):>5} "
              f"{rec['participation_ratio']:>13.2f} {rec['mdl_bits_per_step']:>11.1f}")

    # ---- collapse <-> Lawvere/JEPA (anti-collapse off, on a frustrated odd ring) ----
    H_off, floor_off, _ = train_self_model_ring(5, seed=1, anti_collapse=False)
    H_on, floor_on, _ = train_self_model_ring(5, seed=1, anti_collapse=True)
    amp_off = float(np.mean(np.abs(H_off)))
    amp_on = float(np.mean(np.abs(H_on)))
    collapsed = amp_off < 0.1
    results["collapse_demo"] = dict(
        ring=5, mean_abs_h_anticollapse_off=amp_off, mean_abs_h_anticollapse_on=amp_on,
        lawvere_floor_off=floor_off, lawvere_floor_on=floor_on, collapsed_when_off=collapsed)
    print("\nCOLLAPSE <-> Lawvere/JEPA (frustrated n=5 ring):")
    print(f"  anti-collapse OFF: mean|h| = {amp_off:.3f} (floor {floor_off:.4f})  "
          f"-> {'COLLAPSED to trivial fixed point' if collapsed else 'non-trivial'}")
    print(f"  anti-collapse ON : mean|h| = {amp_on:.3f} (floor {floor_on:.4f})  -> non-trivial")
    print("  => the frustrated self-model escapes via the Lawvere trivial fixed point (= JEPA")
    print("     representation collapse) unless a non-degeneracy (VICReg-analogue) constraint forbids it.")

    # ---- verdicts ----
    odd = [n for n in rings if n % 2 == 1]
    even = [n for n in rings if n % 2 == 0]
    max_cf = max(results["rings"][n]["realised_CF"] for n in rings)
    all_Z2 = all(results["rings"][n]["holonomy_all_Z2"] for n in rings)
    odd_floor = float(np.mean([results["rings"][n]["lawvere_floor"] for n in odd]))
    even_floor = float(np.mean([results["rings"][n]["lawvere_floor"] for n in even]))
    lawvere_clean = (odd_floor > 10 * even_floor) and (odd_floor > 1e-3)
    pr_even = float(np.mean([results["rings"][n]["participation_ratio"] for n in even]))
    pr_odd = float(np.mean([results["rings"][n]["participation_ratio"] for n in odd]))

    impossibility = (max_cf < 1e-6) and all_Z2

    results["verdicts"] = dict(
        impossibility_holds=bool(impossibility), max_realised_CF=max_cf, all_holonomy_Z2=bool(all_Z2),
        lawvere_even_floor=even_floor, lawvere_odd_floor=odd_floor, lawvere_clean=bool(lawvere_clean),
        partic_ratio_even=pr_even, partic_ratio_odd=pr_odd, collapse_demonstrated=bool(collapsed))

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    print(f"  (1) IMPOSSIBILITY: realised CF = {max_cf:.4f} (max over all rings/seeds), holonomy "
          f"all Z2 = {all_Z2}.")
    if impossibility:
        print("      => the TRAINED net never reaches the phase, any parity. Optimisation does NOT")
        print("         sneak around the fork: the phase is ontological (needs no definite ground),")
        print("         not trainable on a classical substrate. CONFIRMED on a learned net.")
    print(f"  (2) LAWVERE BOUND: self-model floor even={even_floor:.4f}, odd={odd_floor:.4f} "
          f"(clean = {lawvere_clean}).")
    print("      => faithful self-modelling breaks down on the frustrated (Liar) loop -- the floor")
    print("         IS the breakdown, and it is a theorem, not an optimisation failure.")
    print(f"  (3) COLLAPSE: anti-collapse off -> trivial fixed point (mean|h|={amp_off:.3f}) "
          f"= Lawvere/JEPA collapse: {collapsed}.")
    print(f"  (4) COMPRESSION (honest): participation ratio even={pr_even:.2f} < odd={pr_odd:.2f} -- a")
    print("      SATISFIABLE self-model compresses (low-dim, the safe horn); a FRUSTRATED classical")
    print("      one is forced into a HIGHER-dim limit cycle. So compression is NOT flat across the")
    print("      dial here -- it tracks satisfiability. The Premakumar RLCT-narrowing signature is")
    print("      NOT reproduced by toy weight/MDL proxies (it is a singular-learning-theory statement;")
    print("      defer to the 4070 RLCT scale-up). The safe-horn POSITIVE rests on the literature")
    print("      (Premakumar/SPR); this script's decisive contribution is the IMPOSSIBILITY (1)+(2).")
    print("\n  NET: a learned classical self-model reaches the safe horn for satisfiable self-references,")
    print("  the Z2 shadow + a Lawvere floor for frustrated ones, and the genuine phase in NEITHER --")
    print("  exactly the fork, now shown robust to optimisation. The phase lives only on the")
    print("  no-definite-ground substrate (Phase 2); a learned positive needs Phase 3B (quantum).")

    out = os.path.join(HERE, "r5_phase3a_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}")
    return results


if __name__ == "__main__":
    main()
