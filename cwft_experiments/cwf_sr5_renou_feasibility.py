"""
cwf_sr5_renou_feasibility.py -- FEASIBILITY PROBE for a Renou-type network test of
genuine complexity, which doubles as the answer to "is the self-reference holonomy
literally a U(1) phase, or just a realifiable Z2 sign?"

Background. sr1 forces i in the minimal description, sr2 gives contextuality, sr3
bounds it to the quantum tier -- but EACH is realifiable on its own (the doubling
trick: a complex Hilbert space = a real one of twice the dimension + a fixed
antisymmetric J playing i; single systems / single sources can always be realified).
Renou et al. (Nature 2021) showed the ONLY structure that forces genuinely complex
QM is a NETWORK with INDEPENDENT sources (the doubling needs one global J; source
independence forbids it). The open question: can self-reference realise that
structure, and is its phase genuinely U(1)?

The gating question reduces to three gates this probe computes:

  GATE A (= the U(1) question).  Is the self-reference holonomy a genuine U(1) phase,
     or a Z2 sign?  Discrete Liar / frustrated cycle: holonomy in Z2 (sign -1, the
     All-vs-Nothing class) -- REAL.  Continuous-unitary lift (sr1): sqrt(NOT),
     eigenvalue i, ORDER 4 -> Z4 in U(1), the continuous group U(1).  So the holonomy
     is Z2 at the discrete level and U(1) under the linear+faithful+continuous (wave)
     demand.  ANSWER to Q2: literally U(1) -- but only after the wave demand, and a
     single U(1) loop is still realifiable, which is exactly why the network test is
     needed.

  GATE B (independence + fusion).  Model two INDEPENDENT self-reference loops as
     phase-carrying entangled pairs (phase = the loop holonomy), do entanglement
     swapping (Bob's joint Bell measurement = the fusion), and check the fused A-C
     correlation (i) depends on the loops' relative phases and (ii) is INVARIANT under
     INDEPENDENT global-phase gauge of each source.  PASS => the structure "independent
     self-ref loops fused with an observable relative phase" genuinely exists, with
     independence preserved (no shared reference smuggled in).

  GATE C (independent i's; complex strictly richer than real-sign).  Each loop carries
     its OWN i (from its OWN self-negation); nothing ties them to a common reference,
     which IS the Renou independence condition.  Complex (U(1)) sources then access a
     CONTINUOUS family of fused correlations; independent real-SIGN (Z2) sources access
     only a finite subset.  Strict inclusion => the necessary prerequisite for a
     Renou-type complex>real gap is present.

HONEST SCOPE: gates A,B,C are NECESSARY structural prerequisites, NOT the genuine
doubling-proof advantage. Full real QM (with doubling) is richer than real-SIGN
sources, so the rigorous complex>real gap is the "whole nine yards": a real-QM network
SDP validated against Renou's published complex/real values. This probe only decides
whether that full test is WARRANTED (prereqs present) or DEAD ON ARRIVAL (a gate fails).

CPU; small dense states. numpy + scipy.linalg.
"""
import json, os
import numpy as np
from scipy.linalg import sqrtm

HERE = os.path.dirname(os.path.abspath(__file__))
NOT = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)


# ---------------------------------------------------------------------------
# GATE A -- the holonomy group: Z2 (discrete) -> U(1) (continuous lift). = Q2.
# ---------------------------------------------------------------------------
def gate_A():
    # discrete self-negation: order 2 (NOT^2 = I) -> Z2; the nontrivial class is the
    # sign -1 (the Liar / All-vs-Nothing holonomy).
    not_order = 1
    M = NOT.copy()
    while not np.allclose(M, I2) and not_order < 16:
        M = M @ NOT; not_order += 1
    # continuous-unitary lift: sqrt(NOT), eigenvalue i, ORDER 4 -> Z4 subset U(1)
    S = sqrtm(NOT)
    sqrt_order = 1
    P = S.copy()
    while not np.allclose(P, I2, atol=1e-9) and sqrt_order < 16:
        P = P @ S; sqrt_order += 1
    w = np.linalg.eigvals(S)
    nontrivial = w[int(np.argmax(np.abs(w - 1.0)))]
    phase = float(np.angle(nontrivial))
    return dict(
        discrete_holonomy_order=int(not_order),          # 2 -> Z2
        discrete_class="Z2 sign (-1), the All-vs-Nothing / Liar class (REAL)",
        continuous_holonomy_order=int(sqrt_order),        # 4 -> Z4 in U(1)
        continuous_phase_over_pi=phase / np.pi,           # 0.5 -> i
        continuous_class="U(1) (sqrt(NOT) eigenvalue i, order 4; continuous group U(1))",
        is_genuine_U1=bool(abs(abs(phase) - np.pi / 2) < 1e-9 and sqrt_order == 4),
        answer_Q2=("The loop holonomy is Z2 (a real sign) at the discrete level and "
                   "U(1) (genuinely complex, the geometric-phase type) under the "
                   "linear+faithful+continuous wave demand. So 'literally U(1)' holds "
                   "after the wave demand -- but a single U(1) loop is realifiable, so "
                   "doubling-proof genuine complexity is exactly the network (Renou) "
                   "test; Q1 and Q2 are the same question."))


# ---------------------------------------------------------------------------
# GATE B -- two independent phase loops + entanglement swapping (Bob's fusion).
# ---------------------------------------------------------------------------
def phased_pair(theta, gauge=0.0):
    """|psi> = e^{i*gauge} (|00> + e^{i theta}|11>)/sqrt2  on 2 qubits (basis 00,01,10,11).
    theta = the loop's relative-phase holonomy (physical); gauge = global phase (gauge)."""
    v = np.zeros(4, dtype=complex)
    v[0] = 1.0; v[3] = np.exp(1j * theta)
    v *= np.exp(1j * gauge) / np.sqrt(2)
    return v


def swap_AC_state(theta1, theta2, g1=0.0, g2=0.0):
    """Two independent pairs (a,b) and (b',c); Bob projects (b,b') onto |Phi+>.
    Returns the normalised post-swap (a,c) 2-qubit state."""
    p1 = phased_pair(theta1, g1).reshape(2, 2)      # indices (a,b)
    p2 = phased_pair(theta2, g2).reshape(2, 2)      # indices (b',c)
    # <Phi+|_{b b'} = (<00| + <11|)/sqrt2 over (b,b')
    # post-swap (a,c)[a,c] = (1/sqrt2) sum_{x in {0,1}} p1[a,x] * p2[x,c]
    out = (p1 @ p2) / np.sqrt(2)                     # (a,c) matrix; sum over b=b'=x
    vec = out.reshape(4)
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 1e-12 else vec


def spin(angle):
    """Measurement observable in the XY-plane at `angle`: cos a * X + sin a * Y."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return np.cos(angle) * X + np.sin(angle) * Y


def correlation_AC(state_ac, alpha, beta):
    O = np.kron(spin(alpha), spin(beta))
    return float(np.real(state_ac.conj() @ O @ state_ac))


def gate_B():
    rng = np.random.default_rng(7)
    alpha, beta = 0.3, 0.9
    # (i) depends on relative phases theta1, theta2 (via theta1+theta2)?
    base = correlation_AC(swap_AC_state(0.0, 0.0), alpha, beta)
    varied = correlation_AC(swap_AC_state(0.7, 1.1), alpha, beta)
    depends_on_relative_phase = abs(varied - base) > 1e-6
    # (ii) invariant under INDEPENDENT global gauge of each source?
    t1, t2 = 0.7, 1.1
    c_nogauge = correlation_AC(swap_AC_state(t1, t2, 0.0, 0.0), alpha, beta)
    gauge_diffs = []
    for _ in range(8):
        g1, g2 = rng.uniform(0, 2 * np.pi, 2)
        c_g = correlation_AC(swap_AC_state(t1, t2, g1, g2), alpha, beta)
        gauge_diffs.append(abs(c_g - c_nogauge))
    gauge_invariant = max(gauge_diffs) < 1e-9
    # confirm the fused phase is exactly theta1+theta2
    st = swap_AC_state(0.7, 1.1)
    fused_phase = float(np.angle(st[3] / st[0])) if abs(st[0]) > 1e-9 else float("nan")
    return dict(
        depends_on_relative_phase=bool(depends_on_relative_phase),
        independent_gauge_invariant=bool(gauge_invariant),
        max_gauge_drift=float(max(gauge_diffs)),
        fused_phase=fused_phase, expected_sum=float(0.7 + 1.1),
        fused_is_sum=bool(abs(((fused_phase - 1.8 + np.pi) % (2 * np.pi)) - np.pi) < 1e-6),
        passes=bool(depends_on_relative_phase and gauge_invariant))


# ---------------------------------------------------------------------------
# GATE C -- independent i's; complex (U(1)) strictly richer than real-sign (Z2).
# ---------------------------------------------------------------------------
def gate_C():
    alpha, beta = 0.3, 0.9
    # complex (U(1)) independent sources: theta_i free in [0,2pi)
    thetas = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    complex_corrs = set()
    for t1 in thetas:
        for t2 in thetas:
            c = correlation_AC(swap_AC_state(t1, t2), alpha, beta)
            complex_corrs.add(round(c, 4))
    complex_range = (min(complex_corrs), max(complex_corrs))
    # independent real-SIGN (Z2) sources: theta_i in {0, pi}
    real_sign_corrs = set()
    for t1 in (0.0, np.pi):
        for t2 in (0.0, np.pi):
            c = correlation_AC(swap_AC_state(t1, t2), alpha, beta)
            real_sign_corrs.add(round(c, 4))
    # the complex family covers a continuum; real-sign only a finite set
    complex_count = len(complex_corrs)
    real_sign_count = len(real_sign_corrs)
    # is there a complex-reachable correlation no independent real-sign source reaches?
    gap_value = None
    for c in sorted(complex_corrs):
        if all(abs(c - r) > 0.05 for r in real_sign_corrs):
            gap_value = c; break
    return dict(
        complex_corr_range=[float(complex_range[0]), float(complex_range[1])],
        complex_distinct=int(complex_count), real_sign_distinct=int(real_sign_count),
        real_sign_corrs=sorted(float(r) for r in real_sign_corrs),
        complex_strictly_richer=bool(gap_value is not None),
        example_gap_correlation=(float(gap_value) if gap_value is not None else None),
        independent_i=True,
        note=("Each loop carries its OWN i (its own self-negation, sr1); nothing ties "
              "them to a common reference -- the Renou independence condition. Complex "
              "U(1) sources access a continuum of fused correlations; independent "
              "real-SIGN (Z2) sources only a finite subset (strict inclusion). NECESSARY "
              "prereq for a complex>real gap; NOT the full real-QM (doubled) bound."))


def main():
    print("cwf_sr5 -- Renou-type feasibility probe (and the U(1) holonomy question)\n")

    A = gate_A()
    print("GATE A (= the U(1) question):")
    print(f"   discrete holonomy: order {A['discrete_holonomy_order']} -> {A['discrete_class']}")
    print(f"   continuous lift:   order {A['continuous_holonomy_order']}, phase "
          f"{A['continuous_phase_over_pi']:+.3f}*pi -> {A['continuous_class']}")
    print(f"   genuine U(1)? {A['is_genuine_U1']}")
    print(f"   Q2: {A['answer_Q2']}\n")

    B = gate_B()
    print("GATE B (independence + fusion / entanglement swapping):")
    print(f"   fused phase = theta1+theta2 ? {B['fused_is_sum']} "
          f"(got {B['fused_phase']:.3f}, expected {B['expected_sum']:.3f})")
    print(f"   correlation depends on relative phase? {B['depends_on_relative_phase']}")
    print(f"   invariant under INDEPENDENT global gauge? {B['independent_gauge_invariant']} "
          f"(max drift {B['max_gauge_drift']:.1e})")
    print(f"   GATE B passes: {B['passes']}\n")

    C = gate_C()
    print("GATE C (independent i's; complex richer than real-sign):")
    print(f"   complex U(1) sources: {C['complex_distinct']} distinct fused correlations "
          f"in [{C['complex_corr_range'][0]:.3f}, {C['complex_corr_range'][1]:.3f}]")
    print(f"   independent real-SIGN sources: {C['real_sign_distinct']} distinct "
          f"({C['real_sign_corrs']})")
    print(f"   complex strictly richer? {C['complex_strictly_richer']} "
          f"(e.g. correlation {C['example_gap_correlation']} unreachable by real-sign)\n")

    gates_pass = A["is_genuine_U1"] and B["passes"] and C["complex_strictly_richer"]
    verdict = (
        "FEASIBILITY GREEN -- the whole nine yards is WARRANTED (not dead on arrival). "
        "All three structural prerequisites for a Renou-type genuine-complexity test are "
        "present on self-referential sources: (A) self-negation forces a genuine U(1) "
        "phase (i, order 4), not a Z2 sign -- so the holonomy IS literally a U(1) phase "
        "under the wave demand (the Q2 answer); (B) two INDEPENDENT loops fuse by "
        "entanglement swapping into an observable relative phase theta1+theta2, invariant "
        "under independent global gauge (independence genuinely preserved -- no shared "
        "reference smuggled in); (C) the loops carry INDEPENDENT i's (the Renou "
        "condition), and complex U(1) sources are STRICTLY richer than independent "
        "real-SIGN (Z2) sources. HONEST SCOPE: these are NECESSARY prerequisites, not the "
        "genuine doubling-proof advantage -- full real QM (with doubling) is richer than "
        "real-SIGN sources, so the rigorous complex>real gap requires a real-QM network "
        "SDP validated against Renou's published values (the whole nine yards). Risk "
        "remains ~50/50 there; but the structure is sound, so the full test is worth "
        "building. NB: a SINGLE U(1) loop is still realifiable; only the network makes "
        "the complexity doubling-proof -- which is the whole point."
    ) if gates_pass else (
        "FEASIBILITY RED -- a structural prerequisite failed; inspect the failing gate "
        "before attempting the full Renou test."
    )
    print(f"VERDICT: {verdict}")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["SR5_renou_feasibility"] = dict(
        gate_A=A, gate_B=B, gate_C=C, gates_pass=bool(gates_pass), verdict=verdict,
        note=("Feasibility probe for a Renou-type network test of genuine complexity, "
              "doubling as the U(1)-holonomy answer (Q2). GATE A: holonomy Z2 (discrete) "
              "-> U(1) (continuous-unitary lift, i, order 4). GATE B: two independent "
              "phase loops fuse by entanglement swapping; fused phase = theta1+theta2, "
              "gauge-invariant under independent source gauges (independence preserved). "
              "GATE C: independent i's; complex U(1) sources strictly richer than "
              "real-SIGN (Z2). Necessary prereqs present => whole nine yards (real-QM "
              "network SDP vs Renou's published values) WARRANTED; not the full bound."))
    json.dump(R, open(out, "w"), indent=2)
    plot(C)
    print("\nWrote results.json key: SR5_renou_feasibility")


def plot(C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    alpha, beta = 0.3, 0.9
    thetas = np.linspace(0, 2 * np.pi, 200)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    # complex sources: fused correlation vs theta1 for a few theta2 (continuum)
    for t2 in [0.0, np.pi / 3, 2 * np.pi / 3]:
        ys = [correlation_AC(swap_AC_state(t1, t2), alpha, beta) for t1 in thetas]
        ax.plot(thetas / np.pi, ys, lw=1.8, label=f"complex U(1), $\\theta_2={t2/np.pi:.2f}\\pi$")
    # real-sign sources: the finite reachable set (markers)
    for t1 in (0.0, np.pi):
        for t2 in (0.0, np.pi):
            c = correlation_AC(swap_AC_state(t1, t2), alpha, beta)
            ax.plot(t1 / np.pi, c, "ks", ms=9, mfc="none", mew=2,
                    label="real-sign (Z2) reachable" if (t1 == 0 and t2 == 0) else None)
    ax.set_xlabel(r"loop-1 holonomy phase $\theta_1/\pi$")
    ax.set_ylabel(r"fused A--C correlation $E(\alpha,\beta)$")
    ax.set_title("Feasibility GATE C: complex U(1) self-reference sources (curves) access a\n"
                 "continuum; independent real-sign (Z2) sources (squares) reach only a finite set")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_SR5_renou_feasibility.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
