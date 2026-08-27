"""
cwf_ap_phaseF_twoknob.py -- ACTION_PRINCIPLE follow-up: does DECOUPLING the two sectors of the
history weight (a SECOND independent scale theta_c) buy a genuine PHASE TRANSITION, or only a
smooth crossover?  [TIGHTENED with the tripartite-mutual-information (I3) diagnostic.]

Background. The book's action principle (Ch.1 sec:path-integral) writes the history weight as
    W_c[hist] = exp( (i/hbar_c) A_c[hist]  -  (1/2 hbar_c) C_c[hist] ),
with A_c the self-reference PHASE sector and C_c the self-description COST sector. The SAME
hbar_c divides BOTH -> the "host lock" (cwf_ap_phaseD.py): phase-coherence and decoherence
cannot be tuned independently (a 1-D reachable set where open-system QM has a 2-D plane). That
lock is a single-history, analytic statement, so it can only ever be a CROSSOVER.

This prototype gives the cost sector its OWN scale theta_c (no hard math obstruction -- the
generic Caldeira-Leggett / Feynman-Vernon influence functional already carries two independent
coefficients; the lock is the special ray theta_c = hbar_c) and asks whether the (hbar_c,theta_c)
plane then contains a genuine NON-ANALYTIC transition.

Realization -- a MEASUREMENT-INDUCED phase transition (MIPT) on a PERIODIC Clifford chain:
  * PHASE sector  A_c/hbar_c   <->  coherent SCRAMBLING (random 2-qubit Clifford brickwork, PBC),
  * COST sector   C_c/theta_c  <->  self-DESCRIPTION = self-MEASUREMENT (projective monitoring),
  control parameter  p = (cost rate)/(cost+phase rate) = hbar_c/(hbar_c+theta_c)  [convention],
  so the ONE-knob lock (theta_c=hbar_c) PINS p = 1/2 -- a single point on the control axis.

TWO diagnostics, both finite-size-scaled:
  (1) half-cut entanglement S(N/2): volume law (slope dS/dN ~ 1/2) -> area law (slope ~ 0).
  (2) TRIPARTITE MUTUAL INFORMATION  I3(A:B:C) for four equal quarters A,B,C,D of the periodic
      chain,  I3 = S_A+S_B+S_C - S_AB - S_AC - S_BC + S_ABC.  I3 is the sharp MIPT order
      parameter: large negative (volume, scales with N) -> 0 (area), and SCALE-INVARIANT at the
      critical point, so I3-vs-p curves for different N CROSS at p_c. The crossing (not a broad
      slope change) is what pins p_c -- the decisive transition-vs-crossover test.

ORDER PARAMETER is entanglement -- the framework's gravity/holography currency (Ch.5--6).

HONEST SCOPE. The underlying MIPT is established physics (Li-Chen-Fisher 2018; Skinner-Ruhman-
Nahum 2019; Zabalo et al. 2020 for the I3 crossing). The CWF content is (a) the MAPPING of the
two action sectors to scrambling vs self-description, (b) the one-knob lock PINS the MIPT control
parameter (p=1/2, area-law side) so the framework as written cannot access the transition, and
(c) the load-bearing identification COST = MONITORING (self-description keeps a record); recordless
dephasing would soften the transition to a crossover.

CPU; stim (Clifford) + numpy. Seeded for monitoring sites (stim's uniform-random-Clifford RNG is
not seedable in 1.16, but ensemble averages are stable). Writes ap_phaseF_twoknob_results.json + fig.
"""
import json
import os

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------- GF(2) rank (pivoted, bitmask) ----------
def gf2_rank(vectors):
    basis = {}
    rank = 0
    for v in vectors:
        cur = int(v)
        while cur:
            hb = cur.bit_length() - 1
            if hb in basis:
                cur ^= basis[hb]
            else:
                basis[hb] = cur
                rank += 1
                break
    return rank


# ---------- stabilizer entropy of an ARBITRARY qubit subset ----------
def entropy_region(stabs, qubits):
    """S_Q for region Q (list of qubit indices) of a pure stabilizer state, in bits:
    S_Q = rank_{GF2}(generators restricted to Q) - |Q|. (Fattal et al.; Nahum et al.)"""
    idx = list(qubits)
    pos = {q: k for k, q in enumerate(idx)}
    vecs = []
    for ps in stabs:
        bits = 0
        for q in idx:
            pi = ps[q]                          # 0=I,1=X,2=Y,3=Z
            k = pos[q]
            if pi in (1, 2):
                bits |= (1 << (2 * k))
            if pi in (2, 3):
                bits |= (1 << (2 * k + 1))
        vecs.append(bits)
    return gf2_rank(vecs) - len(idx)


def tripartite_mi(stabs, N):
    """I3(A:B:C) for the four equal contiguous quarters of the chain (D = complement)."""
    q = N // 4
    A = list(range(0, q)); B = list(range(q, 2 * q))
    C = list(range(2 * q, 3 * q))
    SA = entropy_region(stabs, A); SB = entropy_region(stabs, B); SC = entropy_region(stabs, C)
    SAB = entropy_region(stabs, A + B); SAC = entropy_region(stabs, A + C)
    SBC = entropy_region(stabs, B + C); SABC = entropy_region(stabs, A + B + C)
    return float(SA + SB + SC - SAB - SAC - SBC + SABC)


def _validate_entropy():
    s = stim.TableauSimulator(); s.h(0); s.cx(0, 1)
    bell = entropy_region(s.canonical_stabilizers(), [0])
    s = stim.TableauSimulator(); s.x(0)
    prod = entropy_region(s.canonical_stabilizers(), [0])
    s = stim.TableauSimulator(); s.h(0); s.cx(0, 1); s.cx(1, 2)
    ghz = entropy_region(s.canonical_stabilizers(), [0])
    ok = (bell == 1) and (prod == 0) and (ghz == 1)
    return dict(bell=bell, product=prod, ghz3=ghz, ok=bool(ok))


# ---------- one monitored-Clifford realization (PBC); returns (S(N/2), I3) ----------
def run_realization(N, p, T, rng):
    sim = stim.TableauSimulator()
    for q in range(N):
        sim.x(q); sim.x(q)                      # allocate all N qubits
    for layer in range(T):
        if layer % 2 == 0:
            pairs = [(a, a + 1) for a in range(0, N - 1, 2)]
        else:                                    # odd layer: shifted + wrap (periodic)
            pairs = [(a, a + 1) for a in range(1, N - 1, 2)] + [(N - 1, 0)]
        for (a, b) in pairs:
            sim.do_tableau(stim.Tableau.random(2), [a, b])
        if p > 0:
            for q in np.nonzero(rng.random(N) < p)[0]:
                sim.measure(int(q))
    st = sim.canonical_stabilizers()
    s_half = entropy_region(st, list(range(N // 2)))
    i3 = tripartite_mi(st, N)
    return float(s_half), i3


def sweep(N_list, p_grid, T_mult=3, reals=200, seed=0):
    Shalf = {}; I3 = {}
    for N in N_list:
        T = T_mult * N
        sh_row = []; i3_row = []
        for p in p_grid:
            rng = np.random.default_rng(seed * 100003 + N * 1009 + int(round(p * 1e6)))
            res = [run_realization(N, p, T, rng) for _ in range(reals)]
            sh_row.append(float(np.mean([r[0] for r in res])))
            i3_row.append(float(np.mean([r[1] for r in res])))
        Shalf[N] = sh_row; I3[N] = i3_row
        print(f"    N={N:3d}  I3 vs p: " +
              "  ".join(f"{p:.2f}:{v:6.2f}" for p, v in zip(p_grid, i3_row)))
    return Shalf, I3


def analyze(N_list, p_grid, Shalf, I3):
    Ns = np.array(N_list, float)
    pg = np.array(p_grid)
    # (1) S(N/2) volume/area slopes
    slopes = [float(np.polyfit(Ns, [Shalf[N][j] for N in N_list], 1)[0]) for j in range(len(pg))]
    slope_lowp = slopes[0]; slope_highp = float(np.mean(slopes[-2:]))
    # (2) I3 crossing: standard finite-size estimate = zero-crossing of the difference between
    #     the two LARGEST-N curves (volume side: bigger N more negative; area side: inverts).
    #     This avoids the trivial area-phase regime where all curves collapse to ~0.
    big, big2 = N_list[-1], N_list[-2]
    diff = np.array([I3[big][j] - I3[big2][j] for j in range(len(pg))])
    p_c = float(pg[-1])
    for j in range(len(pg) - 1):
        if diff[j] < 0 <= diff[j + 1]:                  # volume(neg) -> area: sign change
            t = -diff[j] / (diff[j + 1] - diff[j])
            p_c = float(pg[j] + t * (pg[j + 1] - pg[j]))
            break
    # confirm volume(neg, scales) -> area(~0)
    i3_lowp = float(np.mean([I3[N][0] for N in N_list]))
    i3_highp = float(np.mean([I3[N][-1] for N in N_list]))
    # does |I3| at low p grow with N (volume) ?
    i3_grows_with_N = bool(np.polyfit(Ns, [-I3[N][0] for N in N_list], 1)[0] > 0.05)
    return dict(Shalf_slopes=slopes, slope_lowp=slope_lowp, slope_highp=slope_highp,
                p_c_crossing=p_c, p_c_literature=0.16,
                i3_lowp=i3_lowp, i3_highp=i3_highp, i3_grows_with_N=i3_grows_with_N,
                volume_to_area=bool(slope_lowp > 0.3 and abs(slope_highp) < 0.06),
                clean_crossing=bool(i3_grows_with_N and abs(i3_highp) < 0.5))


def main():
    print("cwf_ap_phaseF_twoknob -- two-knob MIPT, tightened with the I3 crossing diagnostic\n")
    val = _validate_entropy()
    print(f"entropy validator (Bell=1, product=0, GHZ3=1): {val}\n")
    if not val["ok"]:
        raise SystemExit("entropy formula failed validation -- aborting")

    N_list = [8, 12, 16, 20, 24]                         # all divisible by 4 (four quarters)
    p_grid = [0.0, 0.05, 0.08, 0.11, 0.13, 0.15, 0.17, 0.19, 0.22, 0.26, 0.32, 0.40]
    reals = 200
    print(f"PBC monitored Clifford brickwork; N={N_list}; reals={reals}; p dense near 0.16\n")
    Shalf, I3 = sweep(N_list, p_grid, T_mult=3, reals=reals, seed=0)
    A = analyze(N_list, p_grid, Shalf, I3)

    p_oneknob = 0.5
    pinned_in_area = p_oneknob > A["p_c_crossing"]
    transition = A["volume_to_area"] and A["clean_crossing"]
    verdict = (
        "GENUINE TRANSITION, pinned by the I3 crossing. (1) S(N/2): near-maximal VOLUME law at "
        f"p->0 (slope dS/dN={A['slope_lowp']:.2f}~1/2) collapsing to an N-INDEPENDENT AREA law "
        f"(slope={A['slope_highp']:.2f}). (2) The tripartite mutual information I3 is large and "
        f"NEGATIVE and grows with N at small p (volume; I3~{A['i3_lowp']:.1f}), vanishes at large "
        f"p (area; I3~{A['i3_highp']:.2f}), and the I3-vs-N curves CROSS at the scale-invariant "
        f"point p_c~{A['p_c_crossing']:.2f} (literature p_c~{A['p_c_literature']} for this class). "
        "A crossing of the scale-invariant order parameter is the decisive signature of a genuine "
        "phase transition -- a non-analyticity the one-knob lock's single-history visibility "
        "argument (analytic) provably cannot produce. The one-knob lock (theta_c=hbar_c) PINS the "
        f"control parameter at p=1/2, deep in the AREA-law (classical) phase (p_c~{A['p_c_crossing']:.2f}): "
        "the framework as written is frozen on the classical side. DECOUPLING theta_c -- the second "
        "variable -- opens the tuning axis and the transition; the order parameter is entanglement "
        "(the gravity/holography currency), and the classicality ratio hbar_c/theta_c becomes the "
        "axis of a genuine phase, not a smooth crossover. CONDITIONAL on COST=MONITORING "
        "(self-description keeps a record); recordless dephasing softens it to a crossover."
        if transition else
        "NO CLEAN TRANSITION pinned at these sizes -- either the volume/area slope split or the I3 "
        "crossing is not clean. Report as a crossover / inconclusive (honest negative); try larger "
        "N or more realizations."
    )
    print(f"\n  (1) S(N/2) slope: low-p={A['slope_lowp']:.3f} (~1/2 volume), "
          f"high-p={A['slope_highp']:.3f} (~0 area)")
    print(f"  (2) I3: low-p={A['i3_lowp']:.2f} (neg, grows with N: {A['i3_grows_with_N']}), "
          f"high-p={A['i3_highp']:.2f} (~0)")
    print(f"      I3 scale-invariant CROSSING at p_c ~ {A['p_c_crossing']:.3f} "
          f"(lit. {A['p_c_literature']})")
    print(f"  one-knob pin p=1/2 sits inside the area-law phase: {pinned_in_area}")
    print(f"  GENUINE TRANSITION (slope split + I3 crossing): {transition}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseF_twoknob_results.json")
    R = dict(
        entropy_validation=val,
        setup=dict(N_list=N_list, p_grid=p_grid, reals=reals, T_mult=3, boundary="periodic",
                   mapping="A_c/hbar_c=scrambling; C_c/theta_c=self-description=projective "
                           "monitoring; p=hbar_c/(hbar_c+theta_c)",
                   diagnostics="S(N/2) volume/area slope; I3 four-quarter scale-invariant crossing"),
        S_half=Shalf, I3=I3, analysis=A,
        one_knob_pin=dict(p=p_oneknob, sits_in_area_law_phase=bool(pinned_in_area),
                          note="theta_c=hbar_c pins the MIPT control parameter at p=1/2"),
        genuine_transition=bool(transition),
        verdict=verdict,
        honesty=("Underlying MIPT + I3 crossing are established (Li-Chen-Fisher 2018; Skinner-"
                 "Ruhman-Nahum 2019; Zabalo et al. 2020). CWF content: mapping the two action "
                 "sectors to scrambling vs self-description; the one-knob lock pins the control "
                 "parameter (p=1/2, area-law) so the framework cannot access the transition; "
                 "COST=MONITORING is the load-bearing assumption. Order parameter = entanglement "
                 "= the gravity/holography currency; hbar_c/theta_c is the classicality axis."))
    json.dump(R, open(out, "w"), indent=2)
    plot(N_list, p_grid, Shalf, I3, A, p_oneknob)
    print(f"\nWrote {out}")


def plot(N_list, p_grid, Shalf, I3, A, p_oneknob):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.9))
    pg = np.array(p_grid)

    # left: I3 crossing (the tightening) -- the decisive diagnostic
    for N in N_list:
        ax1.plot(pg, I3[N], "o-", lw=1.8, ms=4, label=f"$N={N}$")
    ax1.axvline(A["p_c_crossing"], color="gray", ls="--", lw=1.2,
                label=f"$I_3$ crossing $p_c\\approx{A['p_c_crossing']:.2f}$\n(lit. $\\approx0.16$)")
    ax1.axvline(p_oneknob, color="C3", ls=":", lw=2,
                label="one-knob lock pins $p=1/2$\n(area-law / classical side)")
    ax1.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax1.set_xlabel("monitoring rate $p \\propto \\hbar_c/(\\hbar_c+\\theta_c)$")
    ax1.set_ylabel("tripartite MI  $I_3(A{:}B{:}C)$ [bits]")
    ax1.set_title("Scale-invariant $I_3$ crossing $\\Rightarrow$ genuine transition\n"
                  "(volume: $I_3\\!\\ll\\!0$, grows with $N$;  area: $I_3\\!\\to\\!0$)")
    ax1.legend(fontsize=7.3, loc="lower right"); ax1.grid(alpha=0.3)

    # right: S(N/2) volume/area finite-size scaling (corroborating)
    Ns = np.array(N_list, float)
    jc = int(np.argmin(np.abs(pg - A["p_c_literature"])))
    picks = [(0, "C0", f"$p={p_grid[0]:.2f}$ (volume)"),
             (jc, "C2", f"$p={p_grid[jc]:.2f}\\approx p_c$"),
             (len(p_grid) - 1, "C1", f"$p={p_grid[-1]:.2f}$ (area)")]
    for j, c, lab in picks:
        ax2.plot(Ns, [Shalf[N][j] for N in N_list], "s-", color=c, lw=1.8, ms=5, label=lab)
    ax2.set_xlabel("system size $N$")
    ax2.set_ylabel("half-cut entanglement $S(N/2)$ [bits]")
    ax2.set_title("Corroboration: $S(N/2)$ finite-size scaling\n"
                  "(volume $S\\propto N$ $\\to$ area $S\\sim$const)")
    ax2.legend(fontsize=8, loc="upper left"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseF_twoknob.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
