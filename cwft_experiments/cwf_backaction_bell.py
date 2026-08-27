#!/usr/bin/env python3
"""
cwf_backaction_bell.py  --  The back-action Bell/contextuality test.

The phase script (cwf_phase_contextuality.py) proved that PASSIVE coarse-grained
self-measurement of a classical substrate is never contextual (CF=0): a definite
microstate is a global non-contextual hidden variable. So the l2/phase upgrade,
IF forced at all, must come from genuine measurement BACK-ACTION -- the
disturbing self-measurement register of Ch.4 (no-broadcasting). This script
tests whether back-action forces the phase.

THE CIRCULARITY HAZARD (and how this avoids it).
If we implemented self-measurement as a quantum projective measurement,
contextuality would follow by Kochen-Specker -- true by construction, the eta_c=1
trap. Instead we implement CLASSICAL destructive back-action with a tunable
strength beta in [0,1] and ask WHAT KIND of non-classicality it produces. We
separate two things that are usually conflated:

  * SIGNALING  S(beta): does an observable's marginal depend on which other
    observable it is co-measured with?  Classical back-action generically
    CREATES this (measuring A first disturbs B). Signaling is the CHEAP
    non-classicality -- and it is exactly the pathology Ch.4's audit fights
    (it permits superluminal-style influence). It is NOT the phase.

  * NO-SIGNALING CONTEXTUALITY  CF_ns(beta): after removing the signaling part
    (symmetrising marginals), is there still no global joint?  THIS is the
    genuine signature of the l2/phase structure (it is what quantum mechanics
    has and the Spekkens toy model does not).

A genuine complex-amplitude qubit pair is run as a REFERENCE POINT (not a CWF
claim) to show what "forcing the phase" looks like: no signaling, but CF_ns>0.

Outcome logic:
  - back-action gives signaling but CF_ns=0  => phase NOT forced; back-action
    buys only the cheap (pathological) non-classicality. Connects to Ch.4 audit.
  - back-action gives CF_ns>0 with S=0       => phase FORCED bottom-up. The
    strong conjecture's phase clause vindicated, localized to the Ch.4 register.

CPU-only. numpy + scipy.optimize.linprog. Reuses the AB LP from the phase script.
"""

import json
import os
import time
import itertools

import numpy as np
from scipy.optimize import linprog

from cwf_phase_contextuality import contextuality_lp, pr_box_tables

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# A substrate with an ontic state and a DESTRUCTIVE self-measurement.
# ----------------------------------------------------------------------------
# Ontic state: 2 bits (s0, s1) -- a minimal self-modelling substrate. The
# observer reads PARITY-type observables of the ontic bits, but each read has
# back-action: with probability beta it RANDOMISES the bit(s) it did not report
# (a classical analog of measurement disturbing the conjugate variable). Two
# observables in a context are read SEQUENTIALLY (A then B), so A's back-action
# can disturb B. This is the genuine non-commuting case.
#
# Observables (each returns 1 bit):
#   X = s0           (reads bit 0; back-action randomises bit 1 w.p. beta)
#   Z = s1           (reads bit 1; back-action randomises bit 0 w.p. beta)
#   Y = s0 XOR s1    (reads parity; back-action randomises BOTH w.p. beta)
# X and Z "commute" (disjoint disturbance targets); Y is incompatible with both.

OBS = {"X": 0, "Z": 1, "Y": 2}            # ids
NAMES = ["X", "Z", "Y"]


def read_observable(state, oid, beta, rng):
    """Return (outcome, post_state) for a destructive read with back-action beta."""
    s0, s1 = state
    if oid == 0:          # X = s0, disturb s1
        out = s0
        if rng.random() < beta:
            s1 = int(rng.integers(0, 2))
    elif oid == 1:        # Z = s1, disturb s0
        out = s1
        if rng.random() < beta:
            s0 = int(rng.integers(0, 2))
    else:                 # Y = s0^s1, disturb both
        out = s0 ^ s1
        if rng.random() < beta:
            s0 = int(rng.integers(0, 2))
            s1 = int(rng.integers(0, 2))
    return out, (s0, s1)


def context_tables(beta, contexts, n_runs, rng, prep="mixed"):
    """For each ordered context (A,B), estimate joint P(a,b) over independent
    runs, reading A then B (so A's back-action can disturb B)."""
    tables = {}
    for ci, (oa, ob) in enumerate(contexts):
        t = np.zeros((2, 2))
        for _ in range(n_runs):
            if prep == "mixed":
                state = (int(rng.integers(0, 2)), int(rng.integers(0, 2)))
            else:  # a fixed pure-ish prep
                state = (0, 0)
            a, state = read_observable(state, oa, beta, rng)
            b, state = read_observable(state, ob, beta, rng)
            t[a, b] += 1.0
        t /= t.sum()
        tables[ci] = t
    return tables


# ----------------------------------------------------------------------------
# Signaling and no-signaling decomposition
# ----------------------------------------------------------------------------
def signaling_measure(contexts, observables, tables):
    """Max over observables of the total-variation spread of its single-observable
    marginal across the contexts it appears in (as first or second slot)."""
    marg = {o: [] for o in range(len(observables))}
    for ci, (oa, ob) in enumerate(contexts):
        pa = tables[ci].sum(axis=1)        # marginal of A (first slot)
        pb = tables[ci].sum(axis=0)        # marginal of B (second slot)
        marg[oa].append(pa)
        marg[ob].append(pb)
    worst = 0.0
    for o, ms in marg.items():
        if len(ms) < 2:
            continue
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                tv = 0.5 * np.abs(ms[i] - ms[j]).sum()
                worst = max(worst, tv)
    return float(worst)


def no_signaling_contextuality(contexts, observables, tables):
    """
    Contextual fraction restricted to the no-signaling content. We test whether a
    GLOBAL JOINT over all observables exists whose context-marginals match the
    SYMMETRISED (signaling-removed) tables. We symmetrise each observable's
    marginal to its across-context average and re-derive the closest no-signaling
    box (maximum-entropy joint per context with the averaged marginals and the
    observed correlation), then run the AB feasibility LP for the contextual
    fraction. CF_ns>0 means contextual even after signaling is removed.
    """
    # averaged single-observable marginals
    avg = {o: np.zeros(2) for o in range(len(observables))}
    cnt = {o: 0 for o in range(len(observables))}
    for ci, (oa, ob) in enumerate(contexts):
        avg[oa] += tables[ci].sum(axis=1); cnt[oa] += 1
        avg[ob] += tables[ci].sum(axis=0); cnt[ob] += 1
    for o in avg:
        if cnt[o]:
            avg[o] /= cnt[o]

    # rebuild each context table preserving the observed CORRELATION E[ab] but
    # forcing the no-signaling marginals avg[oa], avg[ob]
    ns_tables = {}
    for ci, (oa, ob) in enumerate(contexts):
        pa, pb = avg[oa], avg[ob]
        # observed correlation <(-1)^a (-1)^b>
        signs = np.array([[1, -1], [-1, 1]])   # (-1)^(a+b) for a,b in {0,1}
        corr = float((tables[ci] * signs).sum())
        # build joint with marginals pa,pb and correlation corr (max-ent-ish):
        # P(a,b) = pa(a)pb(b) + (-1)^(a+b)/4 * delta, delta solves correlation
        base = np.outer(pa, pb)
        c0 = float((base * signs).sum())
        # P = base + lam*signs/4 ; correlation(P) = c0 + lam  -> lam = corr - c0
        lam = corr - c0
        P = base + lam * signs / 4.0
        P = np.clip(P, 0, None)
        if P.sum() > 0:
            P /= P.sum()
        ns_tables[ci] = P
    return contextuality_lp(contexts, observables, ns_tables)


# ----------------------------------------------------------------------------
# Complex-amplitude qubit REFERENCE (what forcing the phase looks like)
# ----------------------------------------------------------------------------
def qubit_reference_cf():
    """A genuine qubit measured in 3 MUBs (X,Y,Z eigenbases) on a maximally
    mixed state gives no-signaling, and the {XZ, XY, ZY} pairwise statistics are
    contextual (state-independent KS-type). We compute the CF of the resulting
    box to exhibit a NONZERO no-signaling contextual fraction -- the reference
    for 'phase forced'. (Reference only; not a CWF substrate claim.)"""
    # Pauli measurement outcome correlations on the maximally mixed state are 0
    # for distinct axes (no two-point signal), so this simple version is NS but
    # NOT contextual -- the honest minimal qubit pair needs entanglement. We use
    # a Bell state across two qubits with CHSH settings as the reference instead.
    # CHSH box from a singlet at the optimal angles -> Tsirelson, which has NO
    # global joint (CF>0) and is no-signaling. Build it directly:
    obs = [(0,), (1,), (2,), (3,)]                 # A0,A1,B0,B1
    ctx = [(0, 2), (0, 3), (1, 2), (1, 3)]         # CHSH contexts
    E = {(0, 2): 1 / np.sqrt(2), (0, 3): 1 / np.sqrt(2),
         (1, 2): 1 / np.sqrt(2), (1, 3): -1 / np.sqrt(2)}
    tables = {}
    signs = np.array([[1, -1], [-1, 1]])
    for ci, pair in enumerate(ctx):
        corr = E[pair]
        P = np.full((2, 2), 0.25) + corr * signs / 4.0   # uniform marginals
        tables[ci] = P
    sig = signaling_measure(ctx, obs, tables)
    cf = contextuality_lp(ctx, obs, tables)
    return sig, cf


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run():
    t0 = time.time()
    rng = np.random.default_rng(20260530)
    results = {"controls": {}, "sweep": {}}

    # detector validation
    pr_ctx, pr_obs, pr_tab = pr_box_tables()
    pr = contextuality_lp(pr_ctx, pr_obs, pr_tab)
    print(f"[control +] PR box CF = {pr['contextual_fraction']:.3f} (expect >0)")
    results["controls"]["PR_box_CF"] = pr["contextual_fraction"]

    # complex-amplitude reference: what 'phase forced' looks like
    ref_sig, ref_cf = qubit_reference_cf()
    print(f"[reference] complex-QM CHSH box: signaling = {ref_sig:.4f} (~0), "
          f"no-signaling CF = {ref_cf['contextual_fraction']:.3f} (>0)")
    results["controls"]["complexQM_signaling"] = ref_sig
    results["controls"]["complexQM_CF_ns"] = ref_cf["contextual_fraction"]

    # the back-action substrate: 3 observables X,Z,Y; contexts = the 3 incompatible
    # pairs (X,Z commute; both incompatible with Y). Use all 3 ordered pairs.
    observables = [0, 1, 2]
    contexts = [(0, 1), (0, 2), (1, 2)]    # XZ, XY, ZY
    n_runs = 40000

    print(f"\nback-action sweep (observables X=s0, Z=s1, Y=s0^s1; "
          f"contexts XZ,XY,ZY; n={n_runs})")
    print(f"{'beta':>6}{'signaling S':>14}{'CF (raw)':>11}{'CF_ns':>9}{'verdict':>22}")
    print("-" * 62)
    for beta in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
        tab = context_tables(beta, contexts, n_runs, rng, prep="mixed")
        sig = signaling_measure(contexts, observables, tab)
        raw = contextuality_lp(contexts, observables, tab)
        ns = no_signaling_contextuality(contexts, observables, tab)
        cf_raw = raw.get("contextual_fraction", float("nan"))
        cf_ns = ns.get("contextual_fraction", float("nan"))
        if cf_ns > 1e-2:
            v = "PHASE FORCED"
        elif sig > 1e-2:
            v = "signaling only (cheap)"
        else:
            v = "fully classical"
        results["sweep"][f"beta={beta}"] = dict(
            signaling=sig, CF_raw=cf_raw, CF_ns=cf_ns, verdict=v)
        print(f"{beta:>6.2f}{sig:>14.4f}{cf_raw:>11.4f}{cf_ns:>9.4f}{v:>22}")

    results["runtime_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE, "backaction_bell_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}  ({results['runtime_s']}s)")
    _interpret(results)
    return results


def _interpret(results):
    print("\n" + "=" * 70)
    print("INTERPRETATION  --  does back-action force the phase?")
    print("=" * 70)
    sw = results["sweep"]
    forced = [b for b, d in sw.items() if d["CF_ns"] > 1e-2]
    sigs = [d["signaling"] for d in sw.values()]
    print(f"reference (complex QM): signaling~{results['controls']['complexQM_signaling']:.3f}, "
          f"CF_ns={results['controls']['complexQM_CF_ns']:.3f} -- this is 'phase forced'.")
    print(f"detector: PR box CF={results['controls']['PR_box_CF']:.3f} (validated).\n")
    if forced:
        print(f"CF_ns > 0 at: {forced}")
        print("=> BACK-ACTION FORCES no-signaling contextuality bottom-up. The l2/")
        print("   phase clause of the conjecture is vindicated and localized to the")
        print("   Ch.4 self-measurement register. This would be the first CWF")
        print("   substrate shown genuinely non-classical from its own dynamics.")
    else:
        print(f"CF_ns = 0 at ALL beta (max signaling seen: {max(sigs):.3f}).")
        print("=> THE SHARP NEGATIVE, and it is the informative one:")
        print("   classical back-action produces SIGNALING (marginals depend on")
        print("   context) but NOT no-signaling contextuality. Signaling is the")
        print("   CHEAP non-classicality -- precisely the pathology Ch.4's audit")
        print("   works to suppress. So back-action ALONE does not force the phase;")
        print("   it produces the very thing the framework must forbid.")
        print("\n   STRUCTURAL CONSEQUENCE: the genuine l2/phase (no-signaling")
        print("   contextuality, like the complex-QM reference) needs MORE than")
        print("   back-action -- it needs an epistemic RESTRICTION on preparable")
        print("   states (Spekkens knowledge-balance), which is exactly where")
        print("   Program II's undecidability could re-enter: not as the source of")
        print("   epistemic-ness, but as the restriction that, WITH back-action,")
        print("   could force no-signaling contextuality. That is the precise,")
        print("   newly-localized open target -- and the cleanest statement of what")
        print("   the framework still owes for the wave.")


if __name__ == "__main__":
    run()
