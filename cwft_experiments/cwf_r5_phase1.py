#!/usr/bin/env python3
"""
[R5 STATUS -- RETIRED 2026-06-05, KEPT as a record.] The R5 "learnable substrate for the
self-reference phase" program was retired after Phases 1/2/3A: the self-reference->phase claim
is theorem-only, not empiricizable (a classical substrate gives a theorem-guaranteed negative;
a quantum one makes "recovery" circular -- put i in, get i out). THIS script is kept because its
result is the genuinely useful piece: the two-CF TRAP -- constraint CF=1 vs realised CF=0, i.e.
"rotation + a contextual constraint != phase" -- a methodological caution worth keeping. See
R5_SELFMODEL_PLAN.md (top "STATUS: RETIRED") and FUTURE_WORK.md R5.

cwf_r5_phase1.py  --  R5 Phase 1: the control / the trap.

Plan: cwf_project/R5_SELFMODEL_PLAN.md, Phase 1 ("the control / the trap; PRIORITY,
do first"). This script builds the FALSE POSITIVE a naive neural-self-model
experiment would fall for, and shows the two guards correctly catch it.

THE SUBSTRATE -- a self-negation feedback ring (the dynamical face of the
answer-sheet cycle in cwf_answersheet_demo.py). n real "neurons" on a ring with
inhibitory (antiferromagnetic) coupling, integrated in continuous time:

    h[i] <- h[i] + dt * ( -h[i] + tanh( -g * ((1/2+a)h[i-1] + (1/2-a)h[i+1]) ) )

Each neuron is pushed to DISAGREE with its neighbours -- a "differ from your
neighbour" self-model constraint. The small directed bias `a` breaks the
synchronous period-2 degeneracy into a genuine traveling wave.

    EVEN ring -> a consistent 2-colouring exists (bipartite) -> the dynamics
                 SETTLES to a fixed point (the consistent self-model). No motion.
    ODD ring  -> no consistent 2-colouring (frustrated) -> NO fixed point ->
                 a LIMIT CYCLE: the frustration kink travels round the ring, so
                 the state vector ROTATES in a 2-plane (a "phase-like rotation").

THE TRAP.  An odd ring (a) visibly OSCILLATES -- a rotation in state space that
looks like a phase -- and (b) its self-model CONSTRAINT ("all neighbours differ,
simultaneously") is genuinely contextual: CF = 1 via the validated ABM LP (this
is exactly cwf_answersheet_demo.cycle_cf). A naive reading: "rotation + CF>0 =
genuine quantum phase." Both halves are real; the conclusion is WRONG.

THE DISCRIMINATION (three legs):
  (1) CONSTRAINT CF  -- idealised "must-differ" tables, the SELF-MODEL'S DEMAND.
      even 0, odd 1. This is the LURE.
  (2) REALISED-DYNAMICS CF  -- the SAME adjacent-pair observables, but tables
      built from the ACTUAL orbit, run through the SAME validated LP. ~0 for BOTH
      parities. Structural reason (= the beable argument in
      cwf_phase_contextuality._interpret): a deterministic trajectory is a
      classical mixture of definite global states; every coarse-grained
      observable is a function of a beable, so a single global joint always
      exists -> CF = 0. The contextuality of the abstract constraint does NOT
      transfer to the realised dynamics.
  (3) HOLONOMY CHARACTER  -- the limit cycle's Pancharatnam phase from the (real)
      state overlaps is pinned to {0, pi} (Z2), NEVER a generic U(1) value: real
      loops carry only a Z2 holonomy (thm:u1-cocycle / sr9). And the loop's
      signed winding is gauge-dependent (flips under an orientation-reversing
      change of coordinates), so "it rotates" is not even a convention-free
      statement -- whereas CF (which says 0) is convention-free.

VERDICT Phase 1 must produce: the oscillation is a CLASSICAL limit cycle. Guard
(2) reads CF=0 on the realised dynamics; guard (3) reads a Z2 (not U(1)) holonomy.
=> the trap is caught. KILL CRITERION (from the plan): if a single odd loop's
REALISED-dynamics CF were > 0, the contextual-vs-single-loop distinction would be
empirically empty. We check this explicitly; it must NOT fire.

CPU-only. Reuses the validated LP from cwf_phase_contextuality.py.
"""

import json
import os
import time

import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# The substrate: a continuous-time inhibitory (antiferromagnetic) ring
# ---------------------------------------------------------------------------
def ring_dynamics(n, steps, seed, g=1.8, asym=0.15, dt=0.15, burn_frac=0.6):
    """Integrate the self-negation ring. Returns (full_traj, tail) with tail the
    post-transient portion (the realised attractor: fixed point or limit cycle)."""
    rng = np.random.default_rng(seed)
    h = rng.normal(0.0, 0.3, n)
    traj = np.empty((steps + 1, n))
    traj[0] = h
    for t in range(steps):
        nb = (0.5 + asym) * np.roll(h, 1) + (0.5 - asym) * np.roll(h, -1)
        h = h + dt * (-h + np.tanh(-g * nb))
        traj[t + 1] = h
    tail = traj[int(burn_frac * steps):]
    return traj, tail


def classify(tail):
    """Settled (fixed point) vs oscillating, by the spread of the post-transient
    tail; plus the orbit's effective planar-ness (top-2 PCA variance fraction)."""
    spread = float(np.mean(np.std(tail, axis=0)))
    Xc = tail - tail.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = (s ** 2) / (s ** 2).sum()
    return dict(tail_spread=spread,
                settled=bool(spread < 1e-3),
                top2_pca_var=float(ev[:2].sum()))


# ---------------------------------------------------------------------------
# Leg 1 + 2: the two contextual fractions (same observables, different tables)
# ---------------------------------------------------------------------------
def cycle_scenario(n):
    """Observables = the n single cells; contexts = the n adjacent pairs round the
    ring (incl. wraparound). Identical to cwf_answersheet_demo.cycle_cf's scenario."""
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    return contexts, observables


def constraint_cf(n):
    """Leg 1 (the lure): the SELF-MODEL'S DEMAND -- every adjacent pair must
    DIFFER, simultaneously. Idealised anti-correlated tables. Odd -> CF=1."""
    contexts, observables = cycle_scenario(n)
    tables = {}
    for ci in range(n):
        t = np.zeros((2, 2))
        t[0, 1] = t[1, 0] = 0.5          # must-differ
        tables[ci] = t
    return float(contextuality_lp(contexts, observables, tables)["contextual_fraction"])


def realized_cf(tail, n):
    """Leg 2 (guard): the REALISED dynamics. Same observables/contexts, but the
    2x2 tables are the EMPIRICAL joint of (sign h[i], sign h[j]) over the orbit.
    By the beable argument this must be ~0 for any parity."""
    contexts, observables = cycle_scenario(n)
    bits = (tail > 0.0).astype(int)       # one definite global readout per timestep
    tables = {}
    for ci, (i, j) in enumerate(contexts):
        t = np.zeros((2, 2))
        for row in bits:
            t[row[i], row[j]] += 1.0
        tables[ci] = t / t.sum()
    return float(contextuality_lp(contexts, observables, tables)["contextual_fraction"])


# ---------------------------------------------------------------------------
# Leg 3: holonomy character (Z2 vs U(1)) + gauge-dependence of the apparent spin
# ---------------------------------------------------------------------------
def pancharatnam_phase(tail):
    """arg of the product of consecutive (real) normalised overlaps around the
    closed loop. Real vectors -> each overlap is real -> phase in {0, pi} (Z2)."""
    Xc = tail - tail.mean(0)
    prod = 1.0 + 0j
    idx = list(range(len(Xc))) + [0]      # close the loop
    for k in range(len(idx) - 1):
        a, b = Xc[idx[k]], Xc[idx[k + 1]]
        denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        prod *= np.dot(a, b) / denom
    return float(np.angle(prod))


def _winding(X):
    """Signed winding number of X (rows = points) in its own top-2 PCA plane."""
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T
    th = np.arctan2(P[:, 1], P[:, 0])
    dth = np.diff(th)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return float(dth.sum() / (2 * np.pi))


def winding_gauge_gate(tail, n_gauges=10, seed=11):
    """The loop's signed winding under random invertible changes of coordinate.
    A classical orbit's winding magnitude is roughly fixed but its SIGN flips
    under orientation-reversing gauges -> the apparent 'spin direction' is a
    convention, not an invariant. (CF, by contrast, is convention-free.)"""
    rng = np.random.default_rng(seed)
    n = tail.shape[1]
    w0 = _winding(tail)
    winds = []
    for _ in range(n_gauges):
        T = rng.normal(0, 1, (n, n))
        while abs(np.linalg.det(T)) < 0.3:
            T = rng.normal(0, 1, (n, n))
        winds.append(_winding(tail @ T.T))
    winds = np.array(winds)
    return dict(winding_native=w0,
                winding_abs_mean=float(np.mean(np.abs(winds))),
                winding_signed_std=float(np.std(winds)),
                sign_flips=int(np.sum(np.sign(winds) != np.sign(w0))),
                n_gauges=n_gauges)


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 72)
    print("R5 Phase 1 -- the control / the trap (self-negation feedback ring)")
    print("=" * 72)
    _validate_lp()                        # PR=1, Tsirelson=0.414, local=0 -- LP trustworthy
    print()

    rings = [4, 5, 6, 7]                   # even = consistent, odd = frustrated
    results = {"substrate": "continuous-time inhibitory ring, g=1.8 asym=0.15 dt=0.15",
               "rings": {}}

    hdr = f"{'n':>3} {'parity':>7} {'dynamics':>12} {'top2PCA':>8} " \
          f"{'CF_constraint':>14} {'CF_realised':>12} {'holonomy':>10}"
    print(hdr)
    print("-" * len(hdr))
    for n in rings:
        _, tail = ring_dynamics(n, steps=2000, seed=3)
        cl = classify(tail)
        cf_con = constraint_cf(n)
        cf_real = realized_cf(tail, n)
        parity = "even" if n % 2 == 0 else "odd"
        dyn = "fixed point" if cl["settled"] else "limit cycle"
        rec = dict(parity=parity, **cl,
                   CF_constraint=cf_con, CF_realised=cf_real)
        if not cl["settled"]:
            rec["pancharatnam_phase"] = pancharatnam_phase(tail)
            rec["gauge_gate"] = winding_gauge_gate(tail)
            holo = f"{rec['pancharatnam_phase']:+.3f}"
        else:
            holo = "--"
        results["rings"][n] = rec
        print(f"{n:>3} {parity:>7} {dyn:>12} {cl['top2_pca_var']:>8.3f} "
              f"{cf_con:>14.3f} {cf_real:>12.4f} {holo:>10}")

    # -------- verdict + kill-criterion check --------
    odd = [n for n in rings if n % 2 == 1]
    max_real_odd = max(results["rings"][n]["CF_realised"] for n in odd)
    pancharatnams = [results["rings"][n]["pancharatnam_phase"] for n in odd]
    # Z2 = within 1e-2 of 0 or pi
    is_Z2 = all(min(abs(p), abs(abs(p) - np.pi)) < 1e-2 for p in pancharatnams)
    kill_fires = max_real_odd > 0.05      # plan's Phase-1 kill criterion

    results["verdict"] = dict(
        max_realised_CF_odd=max_real_odd,
        odd_pancharatnam_phases=pancharatnams,
        odd_holonomy_is_Z2=bool(is_Z2),
        kill_criterion_fires=bool(kill_fires),
    )

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  Constraint CF (the lure):  even = 0, odd = 1  "
          f"(the self-model's DEMAND is frustrated for odd rings).")
    print(f"  Realised-dynamics CF:      <= {max_real_odd:.4f} for ALL parities  "
          f"(guard 2: the orbit is a")
    print(f"                             classical mixture of definite states -> "
          f"noncontextual).")
    print(f"  Odd-ring holonomy:         Pancharatnam phase in "
          f"{[round(p,3) for p in pancharatnams]} -> "
          f"{'Z2 {0,pi}' if is_Z2 else 'NOT pinned to Z2 (!)'}, "
          f"not U(1) (guard 3).")
    print()
    if not kill_fires and is_Z2:
        print("  => TRAP CAUGHT. The odd-ring oscillation is a CLASSICAL limit cycle:")
        print("     it winds (a real rotation) and its self-model constraint is")
        print("     contextual, but the REALISED dynamics is noncontextual (CF=0) and")
        print("     the holonomy is Z2, not a genuine U(1) phase. 'Rotation + a")
        print("     contextual constraint' does NOT equal genuine phase. Both guards")
        print("     fire correctly. Kill criterion does NOT fire -> the three-row")
        print("     structure of R5 holds; proceed to Phase 2 (genuine contextual phase).")
    else:
        print("  => KILL CRITERION FIRED or holonomy not Z2: revisit the R5 row")
        print("     structure before Phase 2. (max realised CF_odd ="
              f" {max_real_odd:.4f}; Z2={is_Z2}.)")

    results["runtime_s"] = round(time.time() - t0, 2)
    out = os.path.join(HERE, "r5_phase1_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}   ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
