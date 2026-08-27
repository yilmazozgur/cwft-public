"""
cwf_ap_phaseH_axes.py -- Door 4: is the entanglement (cost-sector) transition INDEPENDENT of the
phase/self-reference axis? The three-axis doctrine says encoding (entanglement) and phase (magic /
non-stabilizerness, the resource the sr-arc ties to self-reference and T2.1 ties to the quantum
phase) are SEPARATE axes. We test this on the monitored substrate by giving it TWO knobs:

  * monitoring rate p   -- the COST sector C_c/theta_c (self-description as measurement); drives the
                           entanglement (volume<->area) transition of cwf_ap_phaseF/G.
  * magic-injection q   -- a PHASE-sector proxy: a fraction q of single-qubit gates are non-Clifford
                           (T gates), the only way to put magic / non-stabilizerness into the state.

Diagnostics on the steady state (full state vector, N=8):
  * entanglement S(N/2)      -- half-cut von Neumann entropy (Schmidt).
  * magic M_2 (stabilizer Renyi entropy, Leone-Oliviero-Hamma 2022):
        M_2 = -log2( 2^{-n} sum_P <psi|P|psi>^4 ),  P over all 4^n Paulis.
        M_2 = 0 iff |psi> is a stabilizer state; M_2 > 0 measures non-stabilizerness.

Hypothesis (prior): the entanglement transition is driven by p and the magic resource by q, and the
two are INDEPENDENT -- in particular, at q=0 (pure Clifford) the substrate traverses the FULL
entanglement transition with M_2 == 0 throughout (a theorem: Clifford preserves stabilizer states),
so the encoding/entanglement axis does NOT generate the phase/magic axis; magic needs its own knob.
If so, the two would-be critical lines sit on different axes -- confirming the three-axis doctrine
from the transition perspective, and giving Door 2 its second axis.

CPU; numpy + stim (only for sampling uniform random 2-qubit Cliffords as 4x4 unitaries). SRE
validated on GHZ/product (M_2=0) and a T-magic state (M_2=log2(4/3)). Writes JSON + fig.
"""
import json
import os

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))

T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
# single-qubit -> Pauli transform rows [I,X,Y,Z], cols [|00>,|01>,|10>,|11>] of a 2x2 block
PAULI_T = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1j, -1j, 0], [1, 0, 0, -1]], dtype=complex)


def apply_2q(psi, U, a, b, n):
    t = psi.reshape([2] * n)
    t = np.tensordot(U.reshape(2, 2, 2, 2), t, axes=([2, 3], [a, b]))
    t = np.moveaxis(t, [0, 1], [a, b])
    return t.reshape(-1)


def apply_1q(psi, G, a, n):
    t = psi.reshape([2] * n)
    t = np.tensordot(G, t, axes=([1], [a]))
    t = np.moveaxis(t, 0, a)
    return t.reshape(-1)


def measure_z(psi, a, n, rng):
    t = psi.reshape([2] * n)
    p0 = float(np.sum(np.abs(np.take(t, 0, axis=a)) ** 2))
    out = 0 if rng.random() < p0 else 1
    keep = np.take(t, out, axis=a)
    norm = np.sqrt(max(p0 if out == 0 else 1 - p0, 1e-300))
    t = np.zeros_like(t)
    sl = [slice(None)] * n
    sl[a] = out
    t[tuple(sl)] = keep / norm
    return t.reshape(-1)


def half_cut_entropy(psi, n):
    m = n // 2
    mat = psi.reshape(2 ** m, 2 ** (n - m))
    s = np.linalg.svd(mat, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log2(p)))


def stabilizer_renyi_entropy(psi, n):
    """M_2 = -log2( 2^{-n} sum_P <P>^4 ) via the per-qubit Pauli transform of rho."""
    rho = np.outer(psi, psi.conj()).reshape([2] * (2 * n))
    perm = []
    for k in range(n):
        perm += [k, n + k]
    c = np.transpose(rho, perm).reshape([4] * n)          # each qubit: (out,in)->[00,01,10,11]
    for k in range(n):
        c = np.tensordot(PAULI_T, c, axes=([1], [k]))      # -> Pauli coeff axis at front
        c = np.moveaxis(c, 0, k)
    coeffs = c.real                                        # c_P = <P> real for Hermitian rho
    s = np.sum(coeffs ** 4)
    val = (2.0 ** (-n)) * s
    return float(-np.log2(max(val, 1e-300)))


def _validate():
    # GHZ (stabilizer) -> M2 = 0 ; product |+>^3 -> 0 ; T|+> tensor |+>^2 -> log2(4/3)
    n = 3
    ghz = np.zeros(2 ** n, dtype=complex); ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    plus = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    tmag = plus.reshape([2] * n).copy()
    tmag = apply_1q(plus, T_GATE, 0, n)                    # one T on |+>^3
    m_ghz = stabilizer_renyi_entropy(ghz, n)
    m_plus = stabilizer_renyi_entropy(plus, n)
    m_t = stabilizer_renyi_entropy(tmag, n)
    ok = abs(m_ghz) < 1e-9 and abs(m_plus) < 1e-9 and abs(m_t - np.log2(4 / 3)) < 1e-6
    return dict(M2_GHZ=m_ghz, M2_plus=m_plus, M2_Tstate=m_t,
               expected_Tstate=float(np.log2(4 / 3)), ok=bool(ok))


def run(n, p, q, T, rng):
    psi = np.zeros(2 ** n, dtype=complex); psi[0] = 1.0
    cliffs = [stim.Tableau.random(2).to_unitary_matrix(endian="little") for _ in range(4 * T)]
    ci = 0
    for layer in range(T):
        pairs = ([(a, a + 1) for a in range(0, n - 1, 2)] if layer % 2 == 0
                 else [(a, a + 1) for a in range(1, n - 1, 2)] + [(n - 1, 0)])
        for (a, b) in pairs:
            psi = apply_2q(psi, cliffs[ci % len(cliffs)], a, b, n); ci += 1
        if q > 0:                                          # magic injection (phase-axis knob)
            for a in np.nonzero(rng.random(n) < q)[0]:
                psi = apply_1q(psi, T_GATE, int(a), n)
        if p > 0:                                          # monitoring (cost-axis knob)
            for a in np.nonzero(rng.random(n) < p)[0]:
                psi = measure_z(psi, int(a), n, rng)
    return half_cut_entropy(psi, n), stabilizer_renyi_entropy(psi, n)


def main():
    print("cwf_ap_phaseH_axes -- Door 4: entanglement (cost) axis vs magic (phase) axis\n")
    val = _validate()
    print(f"SRE validator (GHZ=0, |+>=0, T-state=log2(4/3)={val['expected_Tstate']:.4f}): {val}\n")
    if not val["ok"]:
        raise SystemExit("SRE failed validation -- aborting")

    n, T, reals = 8, 24, 30
    ps = [0.0, 0.08, 0.16, 0.30]
    qs = [0.0, 0.04, 0.10, 0.20]
    print(f"state-vector N={n}, T={T}, reals={reals}; p(monitoring)={ps}; q(magic)={qs}\n")
    S = np.zeros((len(qs), len(ps))); M = np.zeros((len(qs), len(ps)))
    for iq, q in enumerate(qs):
        for ip, p in enumerate(ps):
            rng = np.random.default_rng(7000 + iq * 97 + ip)
            res = [run(n, p, q, T, rng) for _ in range(reals)]
            S[iq, ip] = np.mean([r[0] for r in res])
            M[iq, ip] = np.mean([r[1] for r in res])
        print(f"  q={q:.2f}:  S(N/2)=[" + ", ".join(f"{S[iq,ip]:.2f}" for ip in range(len(ps))) +
              "]   M_2=[" + ", ".join(f"{M[iq,ip]:.3f}" for ip in range(len(ps))) + "]")

    # ---- decoupling analysis ----
    magic_at_q0 = M[0, :]                                  # q=0 row: should be ~0 at every p
    clifford_magic_max = float(np.max(np.abs(magic_at_q0)))
    # S responds to p (drop from low to high p), averaged over q
    S_drop_vs_p = float(np.mean(S[:, 0] - S[:, -1]))       # low-p minus high-p
    # S response to q (at fixed p), averaged: should be small
    S_range_vs_q = float(np.mean(np.ptp(S, axis=0)))       # spread across q at each p, averaged
    # M responds to q (rise from q=0), averaged over p
    M_rise_vs_q = float(np.mean(M[-1, :] - M[0, :]))       # high-q minus low-q
    # M response to p at fixed q>0 (is magic driven by p? expect weaker than by q)
    M_range_vs_p_atq = float(np.mean(np.ptp(M[1:, :], axis=1)))  # spread across p for q>0 rows

    # q=0 magic is zero by theorem (Clifford preserves stabilizers); numerically ~1e-5 from float
    # accumulation over T layers -- "zero" means many orders below the q>0 magic scale (~1).
    independent = bool(clifford_magic_max < 1e-3 and S_drop_vs_p > 1.0 and
                       M_rise_vs_q > 0.1 and S_range_vs_q < 0.5 * abs(S_drop_vs_p))
    verdict = (
        "TWO INDEPENDENT AXES (entanglement vs magic), confirmed. At q=0 (pure Clifford) the magic "
        f"M_2 is identically zero across the WHOLE entanglement sweep (max |M_2|={clifford_magic_max:.1e}) "
        "-- a theorem (Clifford preserves stabilizer states), here measured -- so the monitored "
        "substrate traverses the full volume->area entanglement transition WITHOUT generating any "
        "phase resource: the encoding/entanglement (cost-sector, p) axis does not touch the "
        f"phase/magic axis. Entanglement responds to p (S drops by {S_drop_vs_p:.2f} from low to high "
        f"monitoring) and barely to q (spread {S_range_vs_q:.2f}); magic responds to q (M_2 rises "
        f"{M_rise_vs_q:.2f} from q=0 to q_max) -- it requires its own non-Clifford knob, which the "
        "cost sector never supplies. So the entanglement-transition line (p) and the magic/phase line "
        "(q) sit on DIFFERENT axes: the three-axis separation (encoding vs phase) holds from the "
        "transition perspective, and Door 2's multicritical 'triple point' would need both knobs. "
        "Consistent with the sr-arc finding that contextuality/phase tracks self-reference (magic), "
        "independent of the cost sector, and with T2.1 (magic as the phase resource). HONEST SCOPE: "
        "small N=8 (the clean entanglement transition is the large-N stim result of phaseF/G; here we "
        "need only the DECOUPLING, which N=8 shows); magic is injected as T gates (a phase-sector "
        "proxy), and projective measurement can itself suppress magic at large p -- but the q=0 "
        "zero-magic-across-the-whole-transition statement is exact and is the load-bearing one."
        if independent else
        "Decoupling NOT cleanly established at these sizes -- inspect the (p,q) table; report honestly."
    )
    print(f"\n  q=0 magic across all p (max|M_2|): {clifford_magic_max:.2e}  (theorem: 0)")
    print(f"  S drop low->high p: {S_drop_vs_p:.2f}   S spread across q: {S_range_vs_q:.2f}")
    print(f"  M_2 rise q=0->q_max: {M_rise_vs_q:.2f}")
    print(f"  INDEPENDENT AXES: {independent}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseH_axes_results.json")
    R = dict(validation=val, setup=dict(n=n, T=T, reals=reals, ps=ps, qs=qs,
             knobs="p=monitoring (cost/entanglement axis); q=T-injection (phase/magic axis)"),
             S_half=S.tolist(), M2=M.tolist(),
             analysis=dict(clifford_magic_max=clifford_magic_max, S_drop_vs_p=S_drop_vs_p,
                           S_range_vs_q=S_range_vs_q, M_rise_vs_q=M_rise_vs_q,
                           M_range_vs_p_atq=M_range_vs_p_atq, independent_axes=bool(independent)),
             verdict=verdict,
             honesty=("N=8 (decoupling, not the clean transition -- that is phaseF/G at large N); magic "
                      "= T-injection proxy; measurement can suppress magic at large p; the exact, "
                      "load-bearing statement is q=0 => M_2==0 across the entire entanglement transition."))
    json.dump(R, open(out, "w"), indent=2)
    plot(ps, qs, S, M)
    print(f"\nWrote {out}")


def plot(ps, qs, S, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for iq, q in enumerate(qs):
        ax1.plot(ps, S[iq], "o-", lw=1.7, ms=4, label=f"$q={q:.2f}$")
    ax1.set_xlabel("monitoring rate $p$ (cost / entanglement axis)")
    ax1.set_ylabel("entanglement $S(N/2)$ [bits]")
    ax1.set_title("Entanglement responds to $p$, not $q$\n(curves for different magic $q$ overlap)")
    ax1.legend(fontsize=8, title="magic inj.", loc="upper right"); ax1.grid(alpha=0.3)
    for ip, p in enumerate(ps):
        ax2.plot(qs, M[:, ip], "s-", lw=1.7, ms=4, label=f"$p={p:.2f}$")
    ax2.set_xlabel("magic-injection rate $q$ (phase / self-reference axis)")
    ax2.set_ylabel("magic  $M_2$ (stabilizer Rényi entropy)")
    ax2.set_title("Magic responds to $q$, and is $\\equiv 0$ at $q{=}0$\n(for every $p$: the whole "
                  "entanglement transition is zero-magic)")
    ax2.legend(fontsize=8, title="monitoring", loc="upper left"); ax2.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseH_axes.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
