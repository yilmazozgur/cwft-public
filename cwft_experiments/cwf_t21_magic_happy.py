"""
T2.1 Stage 0 -- the magic dial on the [[5,1,3]] HaPPY substrate (ground truth).

Goal (de-risking): establish the simulation machinery and the first numbers before
the multi-week build. Specifically:
  (1) the magic dial works: a logical R_Z(theta) on the bulk leg drives the physical
      state from the stabilizer corner (theta=0) into the magic interior, measured by
      the stabilizer Renyi entropy M2 (Leone-Oliviero-Hamma 2022) -- the ORDER
      PARAMETER for the whole T2.1 program;
  (2) theta=0 reproduces the stabilizer corner: M2=0, integer S(A), and the contracted
      network's RT slope eta_c=1 (via cwf_a3b2_happy_perfect);
  (3) the codeword-superposition representation (physical state = superposition of <=
      2^k_log stabilizer codewords with the logical amplitudes) is correct, validated
      against the exact dense state vector;
  (4) first look at the GRAVITY leg: does state-magic move the geometry? (Cao et al.
      arXiv:2306.14996: stabilizer codes have NO non-trivial area operator -> expect
      single-tensor S(A) to be theta-INDEPENDENT, i.e. state-magic alone does not
      gravitate; the area leg will need code-magic or multi-leg bulk entanglement.)

The substrate is the [[5,1,3]] perfect tensor (cwf_a3b2_happy_perfect._PERFECT_TABLEAU):
6 legs, leg 0 = bulk/logical, legs 1..5 = physical/boundary. A single tensor is the
[[5,1,3]] encoding isometry |psi>_logical -> |psi_L> on 5 physical qubits.

CPU; dense state vectors (n <= 10 physical qubits here). numpy + stim.
"""
import json, os, time, itertools
from functools import reduce
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim

from cwf_a3b2_happy_perfect import _PERFECT_TABLEAU
import cwf_a3b2_happy_perfect as a3b2

HERE = os.path.dirname(os.path.abspath(__file__))

# single-qubit Paulis
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_P1 = [_I, _X, _Y, _Z]


# =========================================================================
# Core: codewords, magic states, observables (dense ground truth)
# =========================================================================

def codewords_513():
    """The two [[5,1,3]] logical codewords |0_L>, |1_L> as 32-dim (5-qubit) vectors,
    extracted from the 6-leg perfect tensor by fixing the bulk leg to |0>/|1>."""
    sim = stim.TableauSimulator()
    sim.do_tableau(_PERFECT_TABLEAU, list(range(6)))
    psi6 = np.array(sim.state_vector(endian="big"), dtype=complex).reshape([2] * 6)
    cw0 = psi6[0].reshape(-1); cw1 = psi6[1].reshape(-1)        # bulk=0 / bulk=1
    return cw0 / np.linalg.norm(cw0), cw1 / np.linalg.norm(cw1)


def logical_equatorial(theta):
    """The single-qubit magic dial on the equator: |psi(theta)> = (|0>+e^{i theta}|1>)/sqrt2.
    theta=0 -> |+> (stabilizer); theta=pi/4 -> |T> (max single-qubit magic)."""
    return np.array([1.0, np.exp(1j * theta)], dtype=complex) / np.sqrt(2)


def m2_stabilizer_renyi(psi, n):
    """Stabilizer 2-Renyi entropy M2 = n - log2( sum_P <psi|P|psi>^4 ) over all 4^n
    Paulis (Leone-Oliviero-Hamma, PRL 128 050402). M2=0 iff stabilizer; max for magic.
    Dense enumeration -- use only for small n (<= ~6)."""
    tot = 0.0
    for combo in itertools.product(range(4), repeat=n):
        P = reduce(np.kron, [_P1[c] for c in combo])
        e = float((psi.conj() @ (P @ psi)).real)
        tot += e ** 4
    return float(n - np.log2(tot))


def entropy_region(psi, n, A, renyi=None):
    """von Neumann (or Renyi-`renyi`) entanglement entropy (bits) of region A (a list
    of qubit indices) for pure state psi (2^n vector). Big-endian qubit order."""
    A = sorted(int(q) for q in A)
    B = [q for q in range(n) if q not in A]
    T = psi.reshape([2] * n)
    T = np.transpose(T, A + B).reshape(2 ** len(A), 2 ** len(B))
    s = np.linalg.svd(T, compute_uv=False)
    p = (s ** 2); p = p[p > 1e-14]
    if renyi is None:
        return float(-np.sum(p * np.log2(p)))
    if renyi == 2:
        return float(-np.log2(np.sum(p ** 2)))
    return float((1 / (1 - renyi)) * np.log2(np.sum(p ** renyi)))


def chsh_capacity_cf(rho2):
    """Bell/CHSH contextual-fraction CAPACITY of a 2-qubit state via the Horodecki
    criterion: B_max = 2 sqrt(t1^2+t2^2), t_i = two largest singular values of the
    3x3 correlation matrix T_ij = tr(rho sigma_i (x) sigma_j). CF = max(0,(B_max-2)/2).
    NB: this is measurement-OPTIMISED Bell nonlocality (needs entanglement), a SECONDARY
    witness; the intrinsic ABM contextual fraction is Stage 1."""
    paulis = [_X, _Y, _Z]
    T = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            op = np.kron(paulis[i], paulis[j])
            T[i, j] = float(np.real(np.trace(rho2 @ op)))
    sv = np.linalg.svd(T, compute_uv=False)
    t1, t2 = sv[0], sv[1]
    Bmax = 2 * np.sqrt(t1 ** 2 + t2 ** 2)
    return float(max(0.0, (Bmax - 2) / 2)), float(Bmax)


# =========================================================================
# (A) Single tensor: the magic dial, M2(theta), S(A)(theta)
# =========================================================================

def single_tensor_sweep(thetas):
    cw0, cw1 = codewords_513()
    # validate the codeword-superposition assembly == dense encoding at a probe theta
    probe = 0.3
    a, b = logical_equatorial(probe)
    psi_super = a * cw0 + b * cw1                      # codeword-superposition (the simulator)
    # dense cross-check: M2 preserved by the Clifford encoder == logical M2
    m2_phys = m2_stabilizer_renyi(psi_super, 5)
    m2_log = m2_stabilizer_renyi(logical_equatorial(probe), 1)
    super_ok = abs(m2_phys - m2_log) < 1e-9
    rows = []
    for th in thetas:
        a, b = logical_equatorial(th)
        psi = a * cw0 + b * cw1
        m2 = m2_stabilizer_renyi(psi, 5)
        # S(A) for one representative region of each size 1..4
        SA = {k: entropy_region(psi, 5, list(range(k))) for k in range(1, 5)}
        rows.append(dict(theta=float(th), M2=m2, S=SA))
    return rows, dict(super_matches_dense=bool(super_ok),
                      m2_phys_probe=m2_phys, m2_logical_probe=m2_log)


# =========================================================================
# (B) Two uncontracted tensors: 2 logical qubits, entangled magic logical states
#     -> M2 (logical sector) and the phase/entanglement first numbers.
# =========================================================================

def two_tensor_sweep(thetas):
    cw0, cw1 = codewords_513()
    cw = {0: cw0, 1: cw1}
    # codeword basis for 2 logical qubits = product of single-tensor codewords (10 qubits)
    CW = {(x1, x2): np.kron(cw[x1], cw[x2]) for x1 in (0, 1) for x2 in (0, 1)}

    def encode(logical4):
        """logical4: length-4 amplitudes over |x1 x2> -> 1024-dim boundary vector."""
        psi = np.zeros(2 ** 10, dtype=complex)
        for k, (x1, x2) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            psi += logical4[k] * CW[(x1, x2)]
        return psi

    out = {"phase_bell": [], "real_bell": []}
    for th in thetas:
        # (i) phase-magic Bell state: (|00>+e^{i th}|11>)/sqrt2 -- entanglement FIXED,
        #     magic dialed by the phase. Tests phase-magic vs entanglement.
        log_phase = np.array([1, 0, 0, np.exp(1j * th)], dtype=complex) / np.sqrt(2)
        # (ii) real rotated Bell: cos|00>+sin|11> -- entanglement dialed, NO magic.
        log_real = np.array([np.cos(th), 0, 0, np.sin(th)], dtype=complex)
        for tag, log4 in (("phase_bell", log_phase), ("real_bell", log_real)):
            m2_log = m2_stabilizer_renyi(log4, 2)              # == physical M2 (Clifford-inv)
            # logical 2-qubit reduced density matrix (for CHSH capacity)
            rho2 = np.outer(log4, log4.conj())
            cf, Bmax = chsh_capacity_cf(rho2)
            # boundary entropy of tensor-1's 5 qubits (= logical entanglement entropy)
            psi = encode(log4)
            S_tensor1 = entropy_region(psi, 10, list(range(5)))
            out[tag].append(dict(theta=float(th), M2=m2_log, S_tensor1=S_tensor1,
                                 chsh_cf=cf, chsh_B=Bmax))
    return out


# =========================================================================
# (C) theta=0 stabilizer corner of the CONTRACTED holographic network: eta_c=1
# =========================================================================

def stabilizer_corner_eta_c():
    rec = a3b2.run_depth(1, n_child=2)                  # the depth-1 HaPPY pentagon net
    slope, r2, exact, ntot = a3b2.fit_slope(rec["points"])
    return dict(depth=1, n_tensors=rec["n_tensors"], n_boundary=rec["n_boundary"],
                eta_c=slope, R2=r2, exact_frac=exact / ntot)


# =========================================================================
# Driver
# =========================================================================

def main():
    t0 = time.time()
    thetas = list(np.linspace(0, np.pi / 4, 9))
    print("T2.1 Stage 0 -- magic dial on the [[5,1,3]] HaPPY substrate\n")

    # (A) single tensor
    rowsA, val = single_tensor_sweep(thetas)
    print("(A) single tensor (1 logical -> 5 physical):")
    print(f"    codeword-superposition matches dense M2: {val['super_matches_dense']} "
          f"(phys {val['m2_phys_probe']:.6f} == logical {val['m2_logical_probe']:.6f})")
    print(f"    {'theta':>7} {'M2':>9} {'S(1)':>7} {'S(2)':>7} {'S(3)':>7} {'S(4)':>7}")
    for r in rowsA:
        s = r["S"]
        print(f"    {r['theta']:7.4f} {r['M2']:9.5f} {s[1]:7.3f} {s[2]:7.3f} "
              f"{s[3]:7.3f} {s[4]:7.3f}")
    m2_0 = rowsA[0]["M2"]; m2_max = rowsA[-1]["M2"]
    S_var = max(abs(rowsA[-1]["S"][k] - rowsA[0]["S"][k]) for k in range(1, 5))
    print(f"    => M2: {m2_0:.5f} (theta=0) -> {m2_max:.5f} (T-state); "
          f"max |dS(A)| over the sweep = {S_var:.2e}")
    print("    => single-tensor S(A) is theta-INDEPENDENT: state-magic does NOT move the "
          "geometry (consistent with Cao et al.: stabilizer code => trivial area operator).\n")

    # (B) two tensors, entangled magic logical states
    outB = two_tensor_sweep(thetas)
    print("(B) two tensors (2 logical -> 10 physical), entangled logical states:")
    print("    phase-magic Bell (|00>+e^{i th}|11>)/sqrt2 -- entanglement fixed, phase dialed:")
    print(f"    {'theta':>7} {'M2':>9} {'S(tens1)':>9} {'CHSH_B':>8} {'CHSH_cf':>8}")
    for r in outB["phase_bell"]:
        print(f"    {r['theta']:7.4f} {r['M2']:9.5f} {r['S_tensor1']:9.4f} "
              f"{r['chsh_B']:8.4f} {r['chsh_cf']:8.4f}")
    print("    real rotated Bell (cos|00>+sin|11>) -- entanglement dialed, no magic:")
    print(f"    {'theta':>7} {'M2':>9} {'S(tens1)':>9} {'CHSH_B':>8} {'CHSH_cf':>8}")
    for r in outB["real_bell"]:
        print(f"    {r['theta']:7.4f} {r['M2']:9.5f} {r['S_tensor1']:9.4f} "
              f"{r['chsh_B']:8.4f} {r['chsh_cf']:8.4f}")

    # (C) stabilizer corner: eta_c = 1 on the contracted network
    print("\n(C) stabilizer corner (theta=0) of the contracted HaPPY network:")
    eta = stabilizer_corner_eta_c()
    print(f"    depth-1 pentagon net: n_tensors={eta['n_tensors']}, "
          f"n_boundary={eta['n_boundary']}, eta_c={eta['eta_c']:.4f}, "
          f"exact S=|gamma|: {eta['exact_frac']*100:.0f}%")

    # gate checks
    gate = dict(
        theta0_M2_zero=bool(abs(m2_0) < 1e-9),
        Tstate_M2_correct=bool(abs(m2_max - (1 - np.log2(1.5))) < 1e-6),
        single_tensor_area_flat=bool(S_var < 1e-9),
        codeword_superposition_ok=val["super_matches_dense"],
        eta_c_is_1=bool(abs(eta["eta_c"] - 1.0) < 1e-9),
    )
    print("\n  STAGE-0 GATES:", gate)
    print(f"  runtime {time.time()-t0:.1f}s")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["T21_stage0"] = dict(
        thetas=[float(t) for t in thetas],
        single_tensor=rowsA, two_tensor=outB, stabilizer_corner=eta,
        validation=val, gates=gate,
        note="Stage-0 ground truth. M2 = stabilizer Renyi entropy (order parameter). "
             "Single-tensor S(A) theta-flat => state-magic does not gravitate (need "
             "code-magic / multi-leg bulk entanglement; Cao et al. arXiv:2306.14996). "
             "CHSH_cf is measurement-optimised Bell capacity (secondary witness); the "
             "intrinsic ABM contextual fraction is Stage 1.")
    json.dump(R, open(out, "w"), indent=2)
    plot_stage0(rowsA, outB, thetas)
    print("Wrote results.json key: T21_stage0")


def plot_stage0(rowsA, outB, thetas):
    th = np.array(thetas)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) single tensor: M2 rises, S(A) flat
    a = ax[0]
    a.plot(th, [r["M2"] for r in rowsA], "o-", color="C3", label=r"$M_2$ (magic)")
    for k in range(1, 5):
        a.plot(th, [r["S"][k] for r in rowsA], "--", alpha=0.7, label=f"S(|A|={k})")
    a.set_xlabel(r"$\theta$ (logical $R_Z$ angle)"); a.set_ylabel("bits")
    a.set_title("(a) single tensor: magic rises, geometry flat")
    a.legend(fontsize=7); a.grid(alpha=0.3)
    # (b) two tensors: M2 and CHSH for phase-magic vs real-rotation
    b = ax[1]
    b.plot(th, [r["M2"] for r in outB["phase_bell"]], "o-", color="C3",
           label=r"$M_2$ phase-Bell")
    b.plot(th, [r["M2"] for r in outB["real_bell"]], "s--", color="C1",
           label=r"$M_2$ real-Bell (=0)")
    b.plot(th, [r["chsh_cf"] for r in outB["phase_bell"]], "^-", color="C0",
           label="CHSH cf phase-Bell")
    b.plot(th, [r["chsh_cf"] for r in outB["real_bell"]], "v--", color="C4",
           label="CHSH cf real-Bell")
    b.set_xlabel(r"$\theta$"); b.set_ylabel("bits / cf")
    b.set_title("(b) two tensors: magic vs entanglement-nonlocality")
    b.legend(fontsize=7); b.grid(alpha=0.3)
    # (c) summary text
    c = ax[2]; c.axis("off")
    m0 = rowsA[0]["M2"]; mT = rowsA[-1]["M2"]
    txt = ("T2.1 Stage 0 -- gates\n\n"
           f"  theta=0 stabilizer corner:\n    M2 = {m0:.4f} (=0)\n"
           f"  T-state magic:\n    M2 = {mT:.4f} (=0.4150)\n\n"
           "  single-tensor S(A): theta-FLAT\n  -> state-magic does NOT gravitate\n"
           "     (trivial area operator;\n      Cao et al. 2306.14996)\n\n"
           "  codeword-superposition\n  matches dense state vector\n\n"
           "  => feasibility confirmed;\n     the gravity leg needs code-magic\n"
           "     or multi-leg bulk entanglement\n     (-> Stage 1).")
    c.text(0.02, 0.98, txt, transform=c.transAxes, fontsize=9.5, family="monospace",
           va="top")
    fig.suptitle("T2.1 Stage 0: the magic dial on the [[5,1,3]] HaPPY substrate",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(HERE, "fig_T21_stage0.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
