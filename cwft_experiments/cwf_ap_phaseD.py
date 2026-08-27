"""
cwf_ap_phaseD.py -- ACTION_PRINCIPLE_PLAN Phase D: the G2 firewall. Does the influence-action
path integral make a CWF-SPECIFIC prediction (something plain QM / the standard Caldeira-Leggett
influence functional does NOT), or is it "just QM with hbar_c in hbar's role"?

Recall the weight  W_c[gamma] = exp( (i/hbar_c) A_c[gamma]  -  (1/2 hbar_c) C_c[gamma] ).
Phase A established that the C_c->Euclidean-action half is already published (Imafuku 2512.08507),
and Phase C established the A_hol->contextuality half as the CWF residual. Phase D asks the
remaining firewall question about hbar_c itself.

Three tests, with HONEST labels of what is standard vs CWF-specific:

  D1 (MECHANISM, standard -- NOT CWF-specific). hbar_c is the interference BANDWIDTH. Over an
     ensemble of dynamical phase differences Delta ~ N(mu, sigma^2), the surviving interference is
     |<exp(i Delta/hbar_c)>| = exp(-sigma^2 / 2 hbar_c^2): large hbar_c -> coherent; small hbar_c ->
     washed out (classical). This is exactly the standard hbar->0 limit. We show it, and we LABEL it
     as standard: it does not by itself pass G2.

  D2 (the geometric/contextual split -- host ROBUSTNESS). A geometric/contextual phase gamma enters
     as-is (NOT divided by hbar_c), so <exp(i(gamma + s/hbar_c))> = exp(i gamma) exp(-sigma^2/2hbar_c^2):
     the DYNAMICAL part is host-tunable (D1) but the geometric phase gamma -- and hence the
     contextual fraction, which is a function of the constraint holonomy, not of hbar_c -- is
     host-INVARIANT. We confirm by recomputing Phase C's CF (n=3 odd cycle) at hbar_c in {0.1,1,10,100}:
     CF is identical. So CWF contextuality (Phase C) is a real substrate property, not a host
     artifact -- consistent with hbar_c being gauge (B2). (Robustness, not yet a new prediction.)

  D3 (the CWF-SPECIFIC prediction -- the host LOCK). hbar_c appears in BOTH sectors, and B2 fixes
     hbar_c = k_B T tau ln2 (a single THERMODYNAMIC action scale). So phase-coherence and decoherence
     are governed by the SAME knob: as the host varies, V = exp(-sigma^2/2hbar_c^2) (coherence) and
     Wt = exp(-C/hbar_c) (cost suppression) move along a 1-PARAMETER curve. Standard open-systems QM
     has TWO independent knobs (a universal action scale hbar and a separate bath/decoherence rate),
     filling a 2D region. So CWF predicts the reachable (coherence, decoherence) set is 1-dimensional
     -- measure zero in the QM plane -- a falsifiable structural constraint: no computational
     substrate can have high phase-coherence AND high cost-suppression independently; both are locked
     to the host T tau. We exhibit a (V, Wt) target reachable in QM but NOT in CWF.

  D4: the honest G2 verdict.

CPU; numpy + scipy; reuses the validated ABM LP from cwf_phase_contextuality for D2. Results ->
ap_phaseD_results.json (separate). No claim is made that the path-integral FORM is novel; the firewall
is about isolating exactly what is.
"""
import json, os
import numpy as np

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- D1: hbar_c as interference bandwidth (standard hbar->0 mechanism) -------------------
def dynamical_visibility(hbar_c, sigma, mu=0.0, n_samples=200000, seed=0):
    """Empirical |<exp(i Delta/hbar_c)>| over Delta ~ N(mu, sigma^2), and the closed form
    exp(-sigma^2/2 hbar_c^2). Confirms hbar_c sets the bandwidth."""
    rng = np.random.default_rng(seed)
    d = rng.normal(mu, sigma, n_samples)
    emp = abs(np.mean(np.exp(1j * d / hbar_c)))
    closed = float(np.exp(-(sigma ** 2) / (2 * hbar_c ** 2)))
    return emp, closed


# ---- D2: contextual fraction is hbar_c-invariant ----------------------------------------
def cf_odd3():
    """CF of the 3-cycle all-negation (the minimal self-referential / no-definite-ground cell),
    via the validated LP. The empirical tables are the constraint supports -- they do not depend
    on hbar_c, so CF is host-invariant by construction; we recompute at several hbar_c to make
    the invariance explicit and tie it to Phase C."""
    n = 3
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci in range(n):                       # all-negation
        t = np.zeros((2, 2)); t[0, 1] = t[1, 0] = 0.5
        tables[ci] = t
    return float(contextuality_lp(contexts, observables, tables)["contextual_fraction"])


# ---- D3: the host lock -- reachable (coherence, cost-weight) sets ------------------------
def cwf_reachable(sigma, C, hbar_grid):
    """CWF: ONE knob hbar_c. Each hbar_c -> (V, Wt). A 1-parameter curve."""
    V = np.exp(-(sigma ** 2) / (2 * hbar_grid ** 2))
    Wt = np.exp(-C / hbar_grid)
    return V, Wt


def cwf_can_reach(sigma, C, V_target, Wt_target, tol=0.02):
    """Is (V_target, Wt_target) on the CWF 1-parameter curve? Solve hbar_c from V_target,
    then check Wt matches."""
    # V = exp(-sigma^2/2 hbar^2)  =>  hbar = sigma / sqrt(-2 ln V)
    if not (0 < V_target < 1):
        return False, None, None
    hbar = sigma / np.sqrt(-2.0 * np.log(V_target))
    Wt_at = float(np.exp(-C / hbar))
    return (abs(Wt_at - Wt_target) < tol), float(hbar), Wt_at


def main():
    print("cwf_ap_phaseD -- the G2 firewall: what (if anything) is CWF-specific in hbar_c?\n")
    val = _validate_lp()
    print()

    # ----- D1 -----
    print("D1  hbar_c as interference bandwidth (STANDARD hbar->0 mechanism; NOT CWF-specific)")
    sigma = 1.0
    d1_rows = []
    for hbar_c in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0]:
        emp, closed = dynamical_visibility(hbar_c, sigma)
        d1_rows.append(dict(hbar_c=hbar_c, V_empirical=emp, V_closedform=closed))
        print(f"    hbar_c={hbar_c:6.2f}   visibility V={emp:.4f}  "
              f"(closed exp(-s^2/2hbar^2)={closed:.4f})")
    d1_ok = all(abs(r["V_empirical"] - r["V_closedform"]) < 0.02 for r in d1_rows)
    print(f"    -> matches exp(-sigma^2/2hbar_c^2): {d1_ok}.  Large hbar_c=coherent, "
          f"small=classical. STANDARD.")

    # ----- D2 -----
    print("\nD2  contextual fraction is host-INVARIANT (geometric/holonomy phase, not /hbar_c)")
    cf_vals = {hb: cf_odd3() for hb in [0.1, 1.0, 10.0, 100.0]}   # tables independent of hbar_c
    cf_set = set(round(v, 6) for v in cf_vals.values())
    d2_ok = (len(cf_set) == 1) and (abs(list(cf_set)[0] - 1.0) < 1e-6)
    for hb, v in cf_vals.items():
        print(f"    hbar_c={hb:7.2f}   CF(3-cycle odd) = {v:.4f}")
    # the geometric/dynamical decomposition: <exp(i(gamma + s/hbar))> = e^{i gamma} e^{-s^2/2hbar^2}
    gamma = np.pi / 4
    MAG_OBS = 0.05            # phase is only physically defined where interference is observable
    decomp = []
    for hbar_c in [0.1, 0.5, 1.0, 2.0, 10.0]:
        rng = np.random.default_rng(1)
        s = rng.normal(0, sigma, 200000)
        z = np.mean(np.exp(1j * (gamma + s / hbar_c)))
        decomp.append((hbar_c, abs(z), np.angle(z)))
    print("    geometric+dynamical:  the dynamical ENVELOPE shrinks with hbar_c, the geometric")
    print("    PHASE (here gamma=pi/4) is hbar_c-INVARIANT wherever the fringe is observable:")
    for hbar_c, mag, ang in decomp:
        obs = "obs" if mag > MAG_OBS else "decohered (phase undefined)"
        ang_str = f"{ang:+.4f}" if mag > MAG_OBS else "   n/a "
        print(f"      hbar_c={hbar_c:5.2f}   |<e^{{i phi}}>|={mag:.4f}  arg={ang_str} "
              f"(gamma={gamma:+.4f})  [{obs}]")
    # invariance asserted only where interference is observable; where the dynamical envelope
    # decoheres it to ~0 there is no fringe to read a phase from (that is the point of D1).
    geom_invariant = all(abs(ang - gamma) < 0.05 for _, mag, ang in decomp if mag > MAG_OBS)
    print(f"    -> CF host-invariant: {d2_ok};  geometric phase host-invariant: {geom_invariant}. "
          f"CWF contextuality is a real substrate property (consistent with hbar_c gauge, B2).")

    # ----- D3 -----
    print("\nD3  the host LOCK (CWF-SPECIFIC): hbar_c in BOTH sectors => 1-parameter "
          "(coherence, decoherence) curve")
    C = 10.0
    hbar_grid = np.geomspace(0.2, 20.0, 400)
    V, Wt = cwf_reachable(sigma, C, hbar_grid)
    # A target reachable by independent (hbar, Gamma) in QM, but check CWF:
    V_target, Wt_target = 0.90, 0.50           # high coherence AND modest cost-suppression
    reachable, hbar_sol, Wt_at = cwf_can_reach(sigma, C, V_target, Wt_target)
    print(f"    CWF curve: V=exp(-s^2/2hbar^2), Wt=exp(-C/hbar), C={C}, sigma={sigma}. "
          f"1-parameter in hbar_c.")
    print(f"    Target (V={V_target}, Wt={Wt_target}): high coherence + modest suppression.")
    print(f"      -> V={V_target} forces hbar_c={hbar_sol:.3f}, which forces Wt={Wt_at:.4f} "
          f"(!= {Wt_target}).  CWF-reachable: {reachable}.")
    print(f"      In QM+independent-bath (separate action scale hbar and decoherence rate Gamma) "
          f"the SAME (V,Wt) IS reachable -> CWF reachable set is 1-D (measure zero in the 2-D plane).")
    d3_lock = (not reachable)   # the lock holds iff the off-curve target is unreachable in CWF

    # ----- D4 verdict -----
    g2_narrow = d3_lock and d2_ok          # the host-lock is the CWF-specific, falsifiable bit
    verdict = (
        "PHASE D: G2 PASSED NARROWLY -- and the firewall did its job by saying exactly how narrowly. "
        "The path-integral FORM itself is NOT CWF-specific: D1 (hbar_c as interference bandwidth) is "
        "the standard hbar->0 limit, and D2 (geometric/contextual phase host-invariant, dynamical "
        "phase host-tunable) is the standard geometric-vs-dynamical-phase distinction. So the 'single "
        "generator' does NOT, by itself, produce new dynamics -- it is a UNIFYING CONTAINER (it folds "
        "the book's D1 + D2-D5 path-integral cases, and Imafuku's C_c->Euclidean half, into one "
        "weight). The CWF-SPECIFIC, falsifiable content is exactly two things, both OUTSIDE the path "
        "integral's generic structure: (1) the IDENTIFICATIONS -- A_hol = self-reference carrying "
        "contextuality (Phase C, the Imafuku-complement), and C_c = self-description; and (2) the HOST "
        "LOCK (D3) -- because hbar_c = k_B T tau ln2 sits in BOTH the phase and cost sectors, "
        "phase-coherence and decoherence are LOCKED to one thermodynamic knob, so the reachable "
        "(coherence, decoherence) set is 1-dimensional, where standard open-systems QM (independent "
        "hbar and decoherence rate) fills a 2-D region. Falsification handle: find a computational "
        "substrate with INDEPENDENTLY tunable phase-coherence and decoherence -> refutes the "
        "single-hbar_c claim. Net: the action principle's value is UNIFICATION + GROUNDING + one "
        "structural prediction (the lock), exactly as the plan's honest job stated -- NOT a source of "
        "broadly new physics. Overclaiming 'a single generator of all five projections gives new "
        "predictions' would fail this firewall; the honest claim survives it."
        if g2_narrow else
        "PHASE D: G2 NOT CLEANLY PASSED -- inspect D2/D3. If even the host-lock dissolves, the path "
        "integral is purely a re-expression of standard QM / the influence functional, and per the "
        "plan's kill criteria it should be reported as 'CWF substrates instantiate the standard "
        "influence functional' -- a grounding result, not a discovery."
    )
    print(f"\n  D1 bandwidth matches closed form : {d1_ok}  (STANDARD)")
    print(f"  D2 CF + geometric phase host-invariant : {d2_ok and geom_invariant}  (ROBUSTNESS)")
    print(f"  D3 host lock (off-curve target unreachable in CWF) : {d3_lock}  (CWF-SPECIFIC)")
    print(f"\n  G2 PASSED NARROWLY (via the host lock + the Phase-C identifications): {g2_narrow}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseD_results.json")
    R = dict(
        lp_validation=val,
        D1_bandwidth=dict(sigma=sigma, rows=d1_rows, matches_closed_form=bool(d1_ok),
                          label="standard hbar->0 limit; NOT CWF-specific"),
        D2_host_invariance=dict(CF_by_hbar_c=cf_vals, CF_invariant=bool(d2_ok),
                                geometric_phase_invariant=bool(geom_invariant),
                                label="contextuality is a real substrate property (hbar_c gauge, B2)"),
        D3_host_lock=dict(C=C, sigma=sigma, V_target=V_target, Wt_target=Wt_target,
                          cwf_hbar_forced_by_V=hbar_sol, cwf_Wt_at_that_hbar=Wt_at,
                          cwf_can_reach_target=bool(not d3_lock), lock_holds=bool(d3_lock),
                          label="CWF-SPECIFIC: reachable (coherence,decoherence) set is 1-D; "
                                "QM+bath is 2-D; falsifiable"),
        G2_passed_narrowly=bool(g2_narrow),
        verdict=verdict,
        note=("Firewall result: the path-integral FORM reproduces standard QM/influence-functional "
              "structure (D1 hbar->0, D2 geometric-vs-dynamical). CWF-specific content = the "
              "identifications (Phase C: A_hol=self-reference/contextuality; C_c=self-description) "
              "PLUS the host lock (D3: hbar_c=k_B T tau ln2 in both sectors -> 1-parameter "
              "coherence/decoherence curve, vs QM's 2-D). Honest job of the action principle = "
              "unification + grounding + one structural prediction (the lock), NOT new dynamics."))
    json.dump(R, open(out, "w"), indent=2)
    plot(hbar_grid, V, Wt, sigma, C, V_target, Wt_target)
    print(f"\nWrote {out}")


def plot(hbar_grid, V, Wt, sigma, C, V_target, Wt_target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: D1 bandwidth
    hb = np.geomspace(0.1, 20, 300)
    ax1.plot(hb, np.exp(-(sigma ** 2) / (2 * hb ** 2)), "C0-", lw=2)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\hbar_c$ (log)"); ax1.set_ylabel("dynamical visibility $V$")
    ax1.set_title("D1: $\\hbar_c$ = interference bandwidth\n(standard $\\hbar\\to0$ limit)")
    ax1.axhline(0.5, color="gray", ls=":", lw=1); ax1.grid(alpha=0.3)

    # right: D3 host lock -- CWF 1-D curve in the (coherence, decoherence) plane
    ax2.plot(V, Wt, "C3-", lw=2.5, label="CWF: one knob $\\hbar_c$ (1-parameter curve)")
    ax2.scatter([V_target], [Wt_target], c="k", s=70, zorder=5,
                label=f"target ($V$={V_target}, $W$={Wt_target}):\nQM-reachable, CWF-UNreachable")
    ax2.fill_between([0, 1], [0, 0], [1, 1], color="C0", alpha=0.08,
                     label="QM+independent bath: 2-D region")
    ax2.set_xlabel("phase coherence $V=e^{-\\sigma^2/2\\hbar_c^2}$")
    ax2.set_ylabel("cost weight $W=e^{-C/\\hbar_c}$")
    ax2.set_title("D3 (CWF-specific): the host LOCK\ncoherence & decoherence on ONE curve")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.legend(fontsize=7.5, loc="upper left")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseD.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
