"""
T2.1 Stage 2 -- gravity vs phase on a GENUINE holographic RT surface, and the honest
limit of the "triple point" on a finite code.

Stage 1 showed: on a single code-magic tensor, NON-LOCAL code-magic turns on a non-trivial
area operator (Var(K)>0) in lockstep with magic; LOCAL magic raises M2 without gravitating;
and a single-bond network cut (rank 2) is capped and cannot fluctuate. Stage 2 builds the
smallest network with a >= 2-bond RT surface (a depth-1 HaPPY pentagon tree: root R bonded to
two children C1, C2) and asks, on ONE object, vs the code-magic dial:

  GRAVITY (area operator):   Var(K_A) across the 2-bond RT cut {C1,C2}|{R}; does it become
                             non-flat (a fluctuating area) and does RT survive (S ~ |gamma|=2)?
  PHASE (genuine l2):        M2 of the BOUNDARY state -- rigorous non-stabilizerness the
                             substrate presents (by Howard, Nature 2014, = its contextuality
                             capacity). Computed by a fast FWHT method (validated vs the slow
                             4^n Pauli enumeration).

Two magic modes test whether the axes co-onset or are distinguishable:
  - NON-LOCAL magic on the RT surface (CP between R's two bond legs) -> expect BOTH gravity
    and phase to turn on (the encoding corner of the triple point);
  - LOCAL magic inside one side (CP among R's boundary legs) -> expect phase (boundary M2>0)
    WITHOUT gravity (Var(K)=0): the axes are distinguishable.

The THIRD axis (irreducible ignorance / strong-ontic via undecidability, Program II) is NOT a
finite-code property: a finite code reconstructs its whole bulk (A3b3: unconstructable
fraction = 0). So the full triple point cannot be a single finite object -- the finite
holographic code realises the ENCODING face (gravity ^ phase, both via code-magic); the
irreducible-ignorance axis is asymptotic (Program II), orthogonal, and lives elsewhere. Stage 2
makes that explicit.

CPU; dense (boundary <= 11 qubits). numpy + stim.
"""
import json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim

from cwf_a3b2_happy_perfect import _PERFECT_TABLEAU
from cwf_t21_stage1 import area_stats, schmidt_p, m2_stabilizer_renyi

HERE = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Fast stabilizer Renyi entropy M2 via FWHT (validated vs slow enumeration)
# =========================================================================

def _fwht_rows(M):
    M = M.astype(complex).copy(); N = M.shape[-1]; h = 1
    while h < N:
        M = M.reshape(M.shape[0], N // (2 * h), 2, h)
        a = M[:, :, 0, :].copy(); b = M[:, :, 1, :].copy()
        M[:, :, 0, :] = a + b; M[:, :, 1, :] = a - b
        M = M.reshape(M.shape[0], N); h *= 2
    return M


def m2_fast(psi, n):
    """M2 = n - log2 sum_{a,b} |g(a,b)|^4, g(a,b)=sum_x conj(psi[x])psi[x^a](-1)^{b.x}.
    O(n 4^n); feasible to n ~ 12. Matches the 4^n Pauli enumeration exactly."""
    N = 2 ** n
    idx = np.arange(N)
    xor = idx[None, :] ^ idx[:, None]                  # xor[a,x] = x ^ a
    F = np.conj(psi)[None, :] * psi[xor]
    g = _fwht_rows(F)
    return float(n - np.log2(np.sum(np.abs(g) ** 4)))


# =========================================================================
# Code-magic perfect tensor
# =========================================================================

def perfect_tensor_t():
    sim = stim.TableauSimulator()
    sim.do_tableau(_PERFECT_TABLEAU, list(range(6)))
    v = np.array(sim.state_vector(endian="big"), dtype=complex)
    return v.reshape([2] * 6)


def apply_cp(T, i, j, alpha):
    T = T.copy()
    sl = [slice(None)] * T.ndim
    sl[i] = 1; sl[j] = 1
    T[tuple(sl)] *= np.exp(1j * alpha)
    return T


def codemagic_t(alpha, pairs):
    T = perfect_tensor_t()
    for (i, j) in pairs:
        T = apply_cp(T, i, j, alpha)
    return T / np.sqrt(np.vdot(T, T).real)


# =========================================================================
# The depth-1 HaPPY pentagon tree: root R + children C1, C2; 2 bonds.
# Legs per tensor: 0=bulk, 1..5 spatial. Bonds: R.4<->C1.1, R.5<->C2.1.
# Boundary: R{1,2,3}, C1{2,3,4,5}, C2{2,3,4,5} (11 legs). Bulk R.0,C1.0,C2.0 -> |0>.
# RT cut {C1,C2}|{R}: crosses both bonds -> |gamma| = 2.
# =========================================================================

def build_tree(alpha, pairsR, pairsC=()):
    R = codemagic_t(alpha, pairsR)                     # [2]*6, legs 0..5
    C1 = codemagic_t(alpha, pairsC)
    C2 = codemagic_t(alpha, pairsC)
    # fix bulk legs (leg 0) to |0>
    R = R[0]      # legs (1,2,3,4,5)  -> axes 0,1,2,3,4
    C1 = C1[0]    # legs (1,2,3,4,5)
    C2 = C2[0]
    # bond R.4 (axis 3 of R) <-> C1.1 (axis 0 of C1)
    M = np.tensordot(R, C1, axes=([3], [0])) / np.sqrt(2)
    #   M axes: R(1,2,3,5) , C1(2,3,4,5)  -> 8 axes; R.5 is axis 3
    # bond R.5 (axis 3 of M) <-> C2.1 (axis 0 of C2)
    M = np.tensordot(M, C2, axes=([3], [0])) / np.sqrt(2)
    #   M axes: R(1,2,3) , C1(2,3,4,5) , C2(2,3,4,5)  -> 11 axes
    psi = M.reshape(-1)
    return psi / np.linalg.norm(psi)


# boundary qubit layout (axes of the 11-qubit state):
#   0,1,2   -> R boundary legs 1,2,3
#   3,4,5,6 -> C1 boundary legs 2,3,4,5
#   7,8,9,10-> C2 boundary legs 2,3,4,5
R_SIDE = [0, 1, 2]                       # the {R} side of the 2-bond cut


def sweep(alphas, pairsR, pairsC, label):
    rows = []
    for a in alphas:
        psi = build_tree(a, pairsR, pairsC)
        S, VarK, flat = area_stats(schmidt_p(psi, 11, R_SIDE))
        m2b = m2_fast(psi, 11)                          # boundary non-stabilizerness (phase)
        rows.append(dict(alpha=float(a), S=S, VarK=VarK, flat=flat, M2_boundary=m2b))
    return rows


def spearman(x, y):
    rx = np.argsort(np.argsort(x)) - (len(x) - 1) / 2
    ry = np.argsort(np.argsort(y)) - (len(y) - 1) / 2
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))


def main():
    t0 = time.time()
    alphas = list(np.linspace(0, np.pi / 2, 9))
    print("T2.1 Stage 2 -- gravity vs phase on a 2-bond HaPPY RT surface\n")

    # sanity: stabilizer corner (alpha=0) -> S = |gamma| = 2, flat
    psi0 = build_tree(0.0, pairsR=[(4, 5)], pairsC=())
    S0, V0, f0 = area_stats(schmidt_p(psi0, 11, R_SIDE))
    print(f"(sanity) alpha=0 stabilizer net: S(R|rest)={S0:.4f} (=|gamma|=2?), "
          f"Var(K)={V0:.2e} (flat?), M2_boundary={m2_fast(psi0,11):.2e}\n")

    # NON-LOCAL magic STRADDLING the cut: CP between R-boundary legs (1,2,3) and R-bond
    # legs (4,5). (CP within the bonds {4,5} or within the boundary {1,2,3} is LOCAL to one
    # side of the cut and does NOT gravitate -- the Stage-1/2 design lesson.)
    nl = sweep(alphas, pairsR=[(3, 4), (2, 5), (1, 4)], pairsC=(), label="nonlocal")
    print("(A) NON-LOCAL magic STRADDLING the cut (CP between R boundary & bond legs):")
    print(f"    {'alpha':>7} {'S(cut)':>8} {'Var(K)':>9} {'M2_bndry':>9}")
    for r in nl:
        print(f"    {r['alpha']:7.4f} {r['S']:8.4f} {r['VarK']:9.5f} {r['M2_boundary']:9.5f}")

    # LOCAL magic inside the {R} side: CP among R's boundary legs (1,2),(2,3)
    loc = sweep(alphas, pairsR=[(1, 2), (2, 3)], pairsC=(), label="local")
    print("\n(B) LOCAL magic inside the {R} side (CP on R boundary legs):")
    print(f"    {'alpha':>7} {'S(cut)':>8} {'Var(K)':>9} {'M2_bndry':>9}")
    for r in loc:
        print(f"    {r['alpha']:7.4f} {r['S']:8.4f} {r['VarK']:9.5f} {r['M2_boundary']:9.5f}")

    # analysis
    def arr(rows, k): return np.array([r[k] for r in rows])
    rho_nl = spearman(arr(nl, "M2_boundary"), arr(nl, "VarK"))
    grav_nl = arr(nl, "VarK").max() > 1e-3
    grav_loc = arr(loc, "VarK").max() > 1e-3
    phase_nl = arr(nl, "M2_boundary").max() > 1e-3
    phase_loc = arr(loc, "M2_boundary").max() > 1e-3
    rt_dev_nl = float(abs(arr(nl, "S") - 2.0).max())    # RT survival: S vs |gamma|=2
    print(f"\n  CO-ONSET (genuine RT surface, |gamma|=2):")
    print(f"    NON-LOCAL magic: gravity(Var K>0)={grav_nl}, phase(M2>0)={phase_nl}, "
          f"Spearman(M2_b,VarK)={rho_nl:+.3f}, max|S-2|={rt_dev_nl:.3f}")
    print(f"    LOCAL magic:     gravity(Var K>0)={grav_loc}, phase(M2>0)={phase_loc}")
    if grav_nl and phase_nl and (not grav_loc) and phase_loc:
        verdict = ("ENCODING CORNER REALISED + AXES DISTINGUISHABLE: non-local code-magic "
                   "turns on BOTH gravity (fluctuating area on a genuine RT surface) and phase "
                   "(boundary non-stabilizerness) together; local magic gives phase WITHOUT "
                   "gravity. So gravity^phase co-onset via non-local magic, but the phase axis "
                   "can stand alone -- the two encoding-axis properties are distinct yet "
                   "share magic as their resource. RT survives as an APPROXIMATE relation "
                   f"(max|S-2|={rt_dev_nl:.2f}); the area is now an operator, not a c-number.")
    elif grav_nl and phase_nl:
        verdict = ("CO-ONSET: non-local magic turns on both gravity and phase; local-mode "
                   "separation weaker than expected -- inspect.")
    else:
        verdict = ("UNEXPECTED: gravity did not turn on on the 2-bond RT surface -- inspect "
                   "(geometry/magic placement).")
    print(f"\n  VERDICT: {verdict}")

    print("\n  THIRD AXIS (honest scope): the finite holographic code reconstructs its whole")
    print("  bulk (A3b3: unconstructable fraction = 0), so the irreducible-ignorance / strong-")
    print("  ontic axis is NOT a finite-code property -- it is asymptotic (Program II). The")
    print("  finite code realises the ENCODING face (gravity ^ phase); the full triple point")
    print("  (adding irreducible ignorance) cannot be a single finite object.")
    print(f"\n  runtime {time.time()-t0:.1f}s")

    out = os.path.join(HERE, "results.json")
    R = json.load(open(out)) if os.path.exists(out) else {}
    R["T21_stage2"] = dict(
        alphas=[float(a) for a in alphas], nonlocal_magic=nl, local_magic=loc,
        sanity=dict(S0=S0, VarK0=V0, M2_0=float(m2_fast(psi0, 11))),
        coonset=dict(spearman_M2b_VarK_nonlocal=rho_nl, gravity_nonlocal=bool(grav_nl),
                     gravity_local=bool(grav_loc), phase_nonlocal=bool(phase_nl),
                     phase_local=bool(phase_loc), RT_deviation_nonlocal=rt_dev_nl),
        verdict=verdict,
        third_axis_note="Finite code: unconstructable fraction = 0 (A3b3) => irreducible-"
            "ignorance/strong-ontic axis is asymptotic (Program II), not a finite-code "
            "property. The finite code realises the encoding face (gravity^phase); the full "
            "triple point is not a single finite object.",
        note="Depth-1 HaPPY pentagon tree (root+2 children), 2-bond RT cut {C1,C2}|{R}, "
            "|gamma|=2. Gravity = Var(K) across the cut (area-operator fluctuation); phase = "
            "M2 of the boundary state (Howard: = contextuality capacity), via fast FWHT "
            "(validated vs slow). Non-local vs local code-magic distinguishes the axes.")
    json.dump(R, open(out, "w"), indent=2)
    plot_stage2(alphas, nl, loc)
    print("Wrote results.json key: T21_stage2")


def plot_stage2(alphas, nl, loc):
    al = np.array(alphas)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    a = ax[0]
    a.plot(al, [r["VarK"] for r in nl], "o-", color="C3", label="Var(K) non-local")
    a.plot(al, [r["VarK"] for r in loc], "s--", color="C1", label="Var(K) local")
    a.set_xlabel(r"$\alpha$ (code-magic)"); a.set_ylabel(r"Var$(K)$ (area fluctuation)")
    a.set_title("(a) gravity: only NON-LOCAL magic\ngravitates (2-bond RT surface)")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    b = ax[1]
    b.plot(al, [r["M2_boundary"] for r in nl], "o-", color="C0", label="M2 boundary non-local")
    b.plot(al, [r["M2_boundary"] for r in loc], "s--", color="C4", label="M2 boundary local")
    b.set_xlabel(r"$\alpha$"); b.set_ylabel(r"$M_2$ boundary (phase)")
    b.set_title("(b) phase: BOTH modes raise\nboundary non-stabilizerness")
    b.legend(fontsize=8); b.grid(alpha=0.3)
    c = ax[2]
    c.plot([r["M2_boundary"] for r in nl], [r["VarK"] for r in nl], "o-", color="C3",
           label="non-local (gravity vs phase)")
    c.plot([r["M2_boundary"] for r in loc], [r["VarK"] for r in loc], "s--", color="C1",
           label="local (phase only)")
    c.set_xlabel(r"$M_2$ boundary (phase)"); c.set_ylabel(r"Var$(K)$ (gravity)")
    c.set_title("(c) gravity vs phase:\nco-onset (non-local) vs decouple (local)")
    c.legend(fontsize=8); c.grid(alpha=0.3)
    fig.suptitle("T2.1 Stage 2: gravity ^ phase on a genuine 2-bond RT surface", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(HERE, "fig_T21_stage2.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
