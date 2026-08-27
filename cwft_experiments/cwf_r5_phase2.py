#!/usr/bin/env python3
"""
[R5 STATUS -- RETIRED 2026-06-05, kept as a dated record.] Executed and correct, but low added
value: it reassembles the validated sr3 (quantum CF) + sr9 (U(1) holonomy) into the R5 framing.
The R5 phase-on-a-substrate program was retired -- the self-reference->phase claim is theorem-only,
not empiricizable (classical = theorem-guaranteed negative; quantum = circular "put i in, get i
out"; the forcing of i is sr1/sr8/sr9, not a substrate experiment). See R5_SELFMODEL_PLAN.md
(top "STATUS: RETIRED").

cwf_r5_phase2.py  --  R5 Phase 2: genuine phase on the analytic host (the qubit self-loop).

Plan: cwf_project/R5_SELFMODEL_PLAN.md, Phase 2. Phase 1 (cwf_r5_phase1.py) proved that a
CLASSICAL deterministic realisation of the self-referential cycle washes the contextuality to
realised CF=0 and carries only a Z2 holonomy -- a classical limit cycle, not a phase. Phase 2
realises the SAME self-referential cycle QUANTUM-mechanically (no definite ground) and shows the
two genuine-quantum signatures the classical ring lacked, then contrasts the two directly.

It REUSES the validated sr-arc machinery rather than re-deriving it:
  * sr3 (cwf_sr3_unitary_tsirelson):  the unitary KCBS realisation of the odd self-referential
    cycle pulls the bare constraint's super-quantum CF=1 (sr2) down to the bounded QUANTUM TIER
    0<CF<1 -- genuine realised contextuality via the validated ABM LP. (n=3 collapses to 0.)
  * sr9 (cwf_sr9_u1_cocycle):  the loop's Bargmann/geometric phase gamma=-Omega/2 is a
    GAUGE-INVARIANT U(1) holonomy; a REAL (X-Z) loop gives only Z2 {0,pi}; the genuine U(1)
    phase requires sigma_Y, whose i is the self-referential complex structure (sr1/sr8).

THE DELIVERABLE: the explicit Phase-1 <-> Phase-2 split on the SAME self-referential cycle --
   classical realisation -> realised CF = 0,  Z2 holonomy   (Phase 1)
   quantum  realisation -> realised CF > 0,  U(1) holonomy (Phase 2)
-- plus the GAUGE GATE applied as the kill-criterion discriminator: the genuine U(1) phase
survives independent rephasing of the loop states (a true U(1) 1-cocycle), whereas Phase-1's
apparent rotation orientation did not. Genuine phase requires realised CF>0 AND a gauge-invariant
U(1) holonomy; both hold for the quantum self-loop, neither for the classical one.

HONEST SCOPE (preserve sr3's distinction; do NOT conflate the two faces):
  - The CF>0 MAGNITUDE is real-Hilbert realisable (KCBS, qutrit, n>=5) -- NOT a complex-i result.
  - The U(1) PHASE is the separate continuity/holonomy thread (qubit, sigma_Y).
The single QUBIT self-loop carries the genuine U(1) phase (sr9); the genuine no-signaling CF
magnitude lives one dimension up (qutrit, sr3). Both are absent classically and present quantumly,
via distinct mechanisms.

CPU-only. numpy + scipy LP. Phase 2 fixes the TARGET signatures Phase 3's learned substrate must hit.
"""

import json
import os

import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp
from cwf_sr3_unitary_tsirelson import kcbs_config, empirical_model_quantum
from cwf_sr9_u1_cocycle import ray, bargmann, bloch_vec, solid_angle, in_Z2

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Leg A -- the genuine phase: qubit self-loop holonomy (U(1) vs Z2) + gauge gate
# ---------------------------------------------------------------------------
def conn_sum(vecs):
    """The per-edge Berry connection summed round the loop (a gauge-dependent
    1-cochain). The loop holonomy is its gauge-invariant U(1) class."""
    n = len(vecs)
    return float(-sum(np.angle(np.vdot(vecs[k], vecs[(k + 1) % n])) for k in range(n)))


def gauge_gate(vecs, n_trials=8, seed=11):
    """Independently rephase each loop state v_k -> e^{i a_k} v_k. A GENUINE U(1)
    holonomy is invariant (cocycle); the per-edge connection changes (cochain).
    Returns the max loop-holonomy drift and whether the per-edge term moved."""
    rng = np.random.default_rng(seed)
    base = conn_sum(vecs)
    drifts, edge_moved = [], []
    for _ in range(n_trials):
        a = rng.uniform(0, 2 * np.pi, len(vecs))
        vs = [np.exp(1j * a[k]) * vecs[k] for k in range(len(vecs))]
        d = abs(((conn_sum(vs) - base + np.pi) % (2 * np.pi)) - np.pi)
        drifts.append(d)
        edge_moved.append(
            abs(np.angle(np.vdot(vs[0], vs[1])) - np.angle(np.vdot(vecs[0], vecs[1]))) > 1e-6)
    return dict(loop_holonomy_drift=float(max(drifts)),
                per_edge_connection_moves=bool(all(edge_moved)))


def phase_leg():
    """The qubit self-loop holonomy. octant Z->X->Y uses sigma_Y (the self-ref i) ->
    genuine U(1); the real X-Z loop is i-free -> only Z2 {0,pi} (= Phase-1's classical
    case). The gauge gate confirms the U(1) holonomy is a real (gauge-invariant) cocycle."""
    PZ, vZ = ray(0.0, 0.0)
    PX, vX = ray(np.pi / 2, 0.0)
    PY, vY = ray(np.pi / 2, np.pi / 2)
    g_oct, _ = bargmann([PZ, PX, PY])
    Om = solid_angle([bloch_vec(vZ), bloch_vec(vX), bloch_vec(vY)])
    gate = gauge_gate([vZ, vX, vY])

    # i-free (real X-Z plane) loop -> Z2
    Pr = [ray(t, 0.0)[0] for t in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)]
    g_real, _ = bargmann(Pr)

    return dict(
        gamma_qubit_selfloop=float(g_oct), solid_angle=float(Om),
        equals_minus_Omega_over_2=bool(abs(g_oct - (-Om / 2)) < 1e-9),
        is_U1_genuine=bool(not in_Z2(g_oct)),
        gauge_invariant=bool(gate["loop_holonomy_drift"] < 1e-9
                             and gate["per_edge_connection_moves"]),
        gauge_drift=gate["loop_holonomy_drift"],
        gamma_real_loop=float(g_real), real_loop_is_Z2=bool(in_Z2(g_real)),
    )


# ---------------------------------------------------------------------------
# Leg B -- genuine contextuality: unitary KCBS realised CF (quantum tier 0<CF<1)
# ---------------------------------------------------------------------------
def contextuality_leg(ns=(3, 5, 7)):
    """The SAME self-referential cycle realised by a unitary (KCBS qutrit). Genuine
    REALISED CF via the validated ABM LP. n=3 collapses to 0; n>=5 gives 0<CF<1."""
    rows = {}
    for n in ns:
        psi, V = kcbs_config(n)
        contexts, observables, tables, p, max_ov = empirical_model_quantum(psi, V)
        lp = contextuality_lp(contexts, observables, tables)
        rows[n] = dict(CF_realised_quantum=float(lp["contextual_fraction"]),
                       max_adjacent_overlap=float(max_ov))
    return rows


# ---------------------------------------------------------------------------
def load_phase1():
    """Pull the classical-realisation numbers (odd rings) from Phase 1 for the contrast."""
    p = os.path.join(HERE, "r5_phase1_results.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p))
    odd = {int(k): v for k, v in R["rings"].items() if int(k) % 2 == 1}
    return dict(
        max_realised_CF=max(v["CF_realised"] for v in odd.values()),
        holonomy_phases=[v.get("pancharatnam_phase") for v in odd.values()],
    )


def main():
    print("=" * 74)
    print("R5 Phase 2 -- genuine phase on the analytic host (the qubit self-loop)")
    print("=" * 74)
    _validate_lp()
    print()

    phase = phase_leg()
    ctx = contextuality_leg()
    p1 = load_phase1()

    # ---- Leg A: the genuine phase (qubit holonomy) ----
    print("LEG A -- qubit self-loop holonomy (the genuine phase):")
    print(f"  sigma_Y self-loop (Z->X->Y):  gamma = {phase['gamma_qubit_selfloop']:+.4f} "
          f"(= -Omega/2, Omega={phase['solid_angle']:.4f})")
    print(f"     genuine U(1) (not Z2)?      {phase['is_U1_genuine']}")
    print(f"     gauge-invariant cocycle?    {phase['gauge_invariant']} "
          f"(loop drift {phase['gauge_drift']:.1e}; per-edge connection moves)")
    print(f"  i-free real (X-Z) loop:        gamma = {phase['gamma_real_loop']:+.4f}  "
          f"-> Z2 {{0,pi}}? {phase['real_loop_is_Z2']}  (= Phase-1's classical case)")

    # ---- Leg B: genuine contextuality (unitary KCBS realised CF) ----
    print("\nLEG B -- unitary KCBS realised CF (genuine contextuality, quantum tier):")
    for n, r in ctx.items():
        tag = "collapses -> classical" if r["CF_realised_quantum"] < 1e-6 else "genuine 0<CF<1"
        print(f"  n={n}:  realised CF = {r['CF_realised_quantum']:.4f}   ({tag})")

    # ---- The Phase-1 <-> Phase-2 split (the deliverable) ----
    cf_q_big = max(r["CF_realised_quantum"] for n, r in ctx.items() if n >= 5)
    print("\n" + "-" * 74)
    print("THE SPLIT (same self-referential odd cycle, two substrates):")
    print(f"  {'substrate':<26}{'realised CF':>14}{'holonomy':>14}{'gauge-inv phase?':>18}")
    if p1 is not None:
        print(f"  {'classical ring (Phase 1)':<26}{p1['max_realised_CF']:>14.4f}"
              f"{'Z2 {0,pi}':>14}{'no':>18}")
    print(f"  {'quantum self-loop (Ph 2)':<26}{cf_q_big:>14.4f}"
          f"{'U(1) -Omega/2':>14}{'yes':>18}")

    # ---- kill-criterion check: genuine phase needs CF>0 AND gauge-invariant U(1) gamma ----
    cf_positive = cf_q_big > 1e-6
    u1_invariant = phase["is_U1_genuine"] and phase["gauge_invariant"]
    kill_fires = (cf_positive != u1_invariant)   # one without the other => identification fails

    # ---- compression comparability (structural; quantitative RLCT deferred to Phase 3) ----
    compression_note = (
        "Compression is NOT the discriminator at the analytic-host level: the SUBSTRATE "
        "description length is identical for the classical and quantum realisations of the "
        "same cycle (the n edge-signs + O(1) for the realisation choice), so it is flat across "
        "the classical<->quantum dial -- only CF and the holonomy character differ. The "
        "quantitative RLCT / C_q<C_mu comparison is the Phase-3 (learned-substrate) leg, where "
        "compression is the natural measure (sr3's CF magnitude is itself real-realisable, "
        "carrying no compression signature beyond the classical).")

    results = dict(
        phase_leg=phase, contextuality_leg=ctx, phase1_classical=p1,
        split=dict(classical_realised_CF=(p1["max_realised_CF"] if p1 else None),
                   quantum_realised_CF=cf_q_big,
                   classical_holonomy="Z2 {0,pi}", quantum_holonomy="U(1) -Omega/2"),
        kill_criterion_fires=bool(kill_fires),
        cf_positive=bool(cf_positive), u1_gauge_invariant=bool(u1_invariant),
        compression_note=compression_note,
    )

    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    if not kill_fires and cf_positive and u1_invariant:
        print("  => GENUINE PHASE CONFIRMED on the quantum self-loop. The SAME self-referential")
        print("     cycle that gave realised CF=0 + Z2 holonomy classically (Phase 1) gives, when")
        print("     realised quantum-mechanically (no definite ground): realised CF>0 (bounded")
        print(f"     quantum tier, max {cf_q_big:.3f}, sr3) AND a gauge-invariant U(1) holonomy")
        print(f"     (gamma=-pi/4 = -Omega/2, sr9 -- survives rephasing). Both signatures co-occur;")
        print("     the kill criterion does NOT fire. HONEST SCOPE: the CF magnitude is")
        print("     real-Hilbert (qutrit, sr3); the U(1) phase is the separate sigma_Y/i thread")
        print("     (qubit, sr9) -- not conflated. These are the TARGET signatures Phase 3's")
        print("     LEARNED substrate must reproduce to claim a genuine self-referential phase.")
    else:
        print(f"  => KILL CRITERION FIRED: CF>0={cf_positive}, U(1) gauge-invariant={u1_invariant}.")
        print("     The contextuality<->holonomy reading needs revision before Phase 3.")
    print(f"\n  compression: {compression_note}")

    out = os.path.join(HERE, "r5_phase2_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}")
    return results


if __name__ == "__main__":
    main()
