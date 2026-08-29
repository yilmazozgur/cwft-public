"""
cwf_ic1_sector_gate.py -- M1/M2-gate of the "self-reference is first" branch
(SELFREF_FIRST_BRANCH.md): what bubble-scoped i actually means, tested numerically.

Setting. A bubble b carries a REAL state space V_b and a complex structure J_b forced by
self-negation only up to sign (thm:selfref-i). The honest joint space of several bubbles is
the REAL tensor product, and each link (b,b') carries the real symmetric operator
    Q_{bb'} = J_b (x) J_b'      (Q^2 = 1),
whose -1 eigenspace is the standard complex tensor product H_b (x) H_b' and whose +1
eigenspace is the conjugate composite H_b (x) conj(H_b'). "Q = -1 everywhere" is
hypothesis H2 written as a sector choice.

Claims tested (all exact linear algebra, no fits):
  (a) CYCLE IDENTITY: around any ring of n bubbles, prod_k Q_k = (-1)^n as an operator,
      so the Z2 "sign holonomy" is fixed by algebra. A ring with an odd number of
      anti-aligned links has NO joint states (joint eigenspace of dimension 0) -- the
      Mobius / anti-i-wall configuration is impossible, not exotic.
  (b) GAUGE: conjugating one bubble (K_b) flips the sector label of every link at b.
      Hence every definite sector assignment is gauge-equivalent to all-aligned: the
      relative SIGN of i between bubbles carries no gauge-invariant content.
  (c) SUPERSELECTION: for observers whose observables are J-linear on each bubble, a
      superposition across sectors is indistinguishable from the mixture (cross terms
      vanish exactly). The only physical relative-i variable is therefore sector
      DEFINITENESS (eigenstate vs superposition), not sign.
  (d) THE WITNESS: on Renou et al.'s exact bilocal strategy (cwf_sr7), a link whose sector
      is indefinite averages the strategy with its one-sided conjugate; T falls from
      6*sqrt2 = 8.485 to the real-ablation value; a DEFINITE anti-aligned link is restored
      to 8.485 by relabeling. Effective witness T(m) for a link of local definiteness
      m = |<Q>|: linear, crossing the real bound 7.66 at m* (computed, not assumed).
  (e) CONSERVATION: J-linear (complex-linear) couplings commute with Q and conserve the
      sector weights; generic real orthogonal couplings do not. So ordinary "quantum"
      dynamics can neither create nor destroy definiteness; only J-breaking real dynamics
      (scrambling) or a Q-measurement ("comparison") can.

CPU, numpy only; real dims <= 256. Writes ic1_sector_gate_results.json and
fig_IC1_sector_gate.png. Seeded.
"""
import json, os, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20260827)
s2 = np.sqrt(2.0)

# ------------------------------------------------------------------ real forms
def real_form(M):
    """Complex n x n operator -> its real 2n x 2n form acting on [Re psi; Im psi]."""
    A, B = np.real(M), np.imag(M)
    return np.block([[A, -B], [B, A]])

def real_vec(psi):
    return np.concatenate([np.real(psi), np.imag(psi)])

def J_of(n):
    """Complex structure (multiplication by i) on C^n in real form."""
    return real_form(1j * np.eye(n))

def K_of(n):
    """Complex conjugation on C^n in real form (antilinear: K J K = -J)."""
    return np.diag(np.concatenate([np.ones(n), -np.ones(n)]))

def kron(*ops):
    out = np.array([[1.0]])
    for o in ops:
        out = np.kron(out, o)
    return out

# ---------------------------------------------------- embedding into sectors
def embed(psi_complex, dims, sectors):
    """Embed a complex multipartite state psi (dims = complex dims of the bubbles, in a
    CHAIN b_0 - b_1 - ... ) into the real tensor product, in the sector pattern
    `sectors` (one sign per link, -1 = standard composite, +1 = conjugate composite).
    Complex scalars act through the FIRST bubble's J. Returns a unit real vector."""
    nb = len(dims)
    Js = [J_of(d) for d in dims]
    Rdim = int(np.prod([2 * d for d in dims]))
    # sector projectors on each link
    P = np.eye(Rdim)
    for k, s in enumerate(sectors):
        ops = [np.eye(2 * d) for d in dims]
        ops[k] = Js[k]; ops[k + 1] = Js[k + 1]
        Q = kron(*ops)
        P = P @ (np.eye(Rdim) + s * Q) / 2.0
    Jfirst = kron(Js[0], *[np.eye(2 * d) for d in dims[1:]])
    v = np.zeros(Rdim)
    psi = psi_complex.reshape(dims)
    for idx in itertools.product(*[range(d) for d in dims]):
        c = psi[idx]
        if abs(c) < 1e-15:
            continue
        basis = [real_vec(np.eye(d)[i]) for d, i in zip(dims, idx)]
        e = P @ kron(*[b.reshape(-1, 1) for b in basis]).ravel()
        v += np.real(c) * e + np.imag(c) * (Jfirst @ e)
    v *= 2.0 ** (len(sectors) / 2.0)             # each projector halves the norm^2
    return v

def link_Q(dims, k):
    ops = [np.eye(2 * d) for d in dims]
    ops[k] = J_of(dims[k]); ops[k + 1] = J_of(dims[k + 1])
    return kron(*ops)

# ------------------------------------------------------------------- (a) ring
def ring_Qs(n, d=2):
    """Link charges Q_k = J_k (x) J_{k+1} on a ring of n bubbles of complex dim d."""
    Js = []
    for k in range(n):
        ops = [np.eye(2 * d)] * n
        ops = list(ops); ops[k] = J_of(d); ops[(k + 1) % n] = J_of(d)
        Js.append(kron(*ops))
    return Js

def test_cycle_identity(n=3, d=2):
    Qs = ring_Qs(n, d)
    prod = np.eye((2 * d) ** n)
    for Q in Qs:
        prod = prod @ Q
    ident_ok = np.allclose(prod, (-1) ** n * np.eye((2 * d) ** n))
    commute = all(np.allclose(Qs[i] @ Qs[j], Qs[j] @ Qs[i]) for i in range(n) for j in range(n))
    # joint eigenspace dimensions for every sign pattern
    dims = {}
    for pattern in itertools.product([-1, 1], repeat=n):
        Pj = np.eye((2 * d) ** n)
        for s, Q in zip(pattern, Qs):
            Pj = Pj @ (np.eye((2 * d) ** n) + s * Q) / 2.0
        dims["".join("-" if s < 0 else "+" for s in pattern)] = int(round(np.trace(Pj)))
    return ident_ok, commute, dims

# ---------------------------------------------------------------- (d) Renou T
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI = {"z": Z, "x": X, "y": Y}

def bell_states():
    p = 1 / s2
    return (np.array([1, 0, 0, 1], dtype=complex) * p, np.array([1, 0, 0, -1], dtype=complex) * p,
            np.array([0, 1, 1, 0], dtype=complex) * p, np.array([0, 1, -1, 0], dtype=complex) * p)

def alice_obs(x, ysign=1.0, use_Y=True):
    return {1: Z, 2: X, 3: (ysign * Y if use_Y else 0 * Y)}[x]

def charlie_obs(z, ysign=1.0, use_Y=True):
    P = {"z": Z, "x": X, "y": (ysign * Y if use_Y else 0 * Y)}
    table = {1: ("z", "x", +1), 2: ("z", "x", -1), 3: ("z", "y", +1),
             4: ("z", "y", -1), 5: ("x", "y", +1), 6: ("x", "y", -1)}
    i, j, sign = table[z]
    return (P[i] + sign * P[j]) / s2

def renou_state():
    phi = np.array([1, 0, 0, 1], dtype=complex) / s2
    return np.kron(phi, phi)                       # order a, B1, B2, c  (complex, 16)

LABELS = None   # the Bell-outcome labeling reproducing 6*sqrt2, found once by search (as in sr7)

def find_labeling():
    """Search the 24 (b1,b2) assignments on the complex reference; keep the one giving
    6*sqrt2 (unique quantum maximum) -- the same self-validation cwf_sr7 performs."""
    global LABELS
    twobits = [(0, 0), (0, 1), (1, 0), (1, 1)]
    best, best_lab = -1.0, None
    for perm in itertools.permutations(range(4)):
        lab = {perm[k]: twobits[k] for k in range(4)}
        T = T_complex_reference(labels=lab)
        if T > best:
            best, best_lab = T, lab
    LABELS = best_lab
    return best, best_lab

def T_from_real_state(v, dims, ya=1.0, yc=1.0, use_Y_a=True, use_Y_c=True, labels=None):
    """Renou's T evaluated on a REAL joint state v of the chain (Alice C^2, Bob C^4,
    Charlie C^2) with the observers' standard J-linear observables in real form."""
    bells = bell_states()
    labels = labels or LABELS
    def S(b, x, z):
        A = real_form(alice_obs(x, ya, use_Y_a))
        Pi = real_form(np.outer(bells[b], bells[b].conj()))
        C = real_form(charlie_obs(z, yc, use_Y_c))
        return float(v @ kron(A, Pi, C) @ v)
    def Tb(b):
        b1, b2 = labels[b]; sb = lambda x, z: S(b, x, z)
        return ((-1) ** b2 * (sb(1, 1) + sb(1, 2)) + (-1) ** b1 * (sb(2, 1) - sb(2, 2))
                + (-1) ** b2 * (sb(1, 3) + sb(1, 4)) - (-1) ** (b1 + b2) * (sb(3, 3) - sb(3, 4))
                + (-1) ** b1 * (sb(2, 5) + sb(2, 6)) - (-1) ** (b1 + b2) * (sb(3, 5) - sb(3, 6)))
    return float(sum(Tb(b) for b in range(4)))

def T_complex_reference(ya=1.0, yc=1.0, use_Y_a=True, use_Y_c=True, labels=None):
    """Same functional computed in ordinary complex QM (cwf_sr7's formula)."""
    psi = renou_state(); bells = bell_states()
    labels = labels or LABELS
    def ck(*ops):
        out = np.array([[1.0 + 0j]])
        for o in ops: out = np.kron(out, o)
        return out
    def S(b, x, z):
        Op = ck(alice_obs(x, ya, use_Y_a), np.outer(bells[b], bells[b].conj()), charlie_obs(z, yc, use_Y_c))
        return float(np.real(psi.conj() @ Op @ psi))
    def Tb(b):
        b1, b2 = labels[b]; sb = lambda x, z: S(b, x, z)
        return ((-1) ** b2 * (sb(1, 1) + sb(1, 2)) + (-1) ** b1 * (sb(2, 1) - sb(2, 2))
                + (-1) ** b2 * (sb(1, 3) + sb(1, 4)) - (-1) ** (b1 + b2) * (sb(3, 3) - sb(3, 4))
                + (-1) ** b1 * (sb(2, 5) + sb(2, 6)) - (-1) ** (b1 + b2) * (sb(3, 5) - sb(3, 6)))
    return float(sum(Tb(b) for b in range(4)))

# ------------------------------------------------------------- (e) couplings
def random_hermitian(n):
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (M + M.conj().T) / 2

def random_Jlinear_gate(dA, dB, strength=1.0):
    """Real orthogonal gate generated by a real antisymmetric G = (J_A (x) 1) H_R with
    H = sum_k A_k (x) B_k + local terms, A_k, B_k Hermitian. Commutes with J_A (x) 1 and
    1 (x) J_B: on the -1 sector it is exp(-iH), on the +1 sector the conjugate-composite
    counterpart. This is what an inter-bubble coupling that respects both complex
    structures looks like at the real level."""
    HR = np.zeros((4 * dA * dB, 4 * dA * dB))
    for _ in range(3):
        HR += np.kron(real_form(random_hermitian(dA)), real_form(random_hermitian(dB)))
    HR += np.kron(real_form(random_hermitian(dA)), np.eye(2 * dB))
    HR += np.kron(np.eye(2 * dA), real_form(random_hermitian(dB)))
    G = np.kron(J_of(dA), np.eye(2 * dB)) @ HR * strength
    assert np.allclose(G, -G.T)
    from scipy.linalg import expm
    return expm(G)

def random_orthogonal(n):
    M = rng.normal(size=(n, n))
    Qm, R = np.linalg.qr(M)
    Qm = Qm @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Qm) < 0:
        Qm[:, 0] *= -1
    return Qm

def sector_weights(v, Q):
    return float(v @ (np.eye(len(v)) - Q) @ v / 2), float(v @ (np.eye(len(v)) + Q) @ v / 2)

# ============================================================================
def main():
    R = {}
    print("cwf_ic1_sector_gate -- bubble-scoped i: what is gauge, what is physical\n")

    # ---- (a) cycle identity + frustration impossibility
    print("(a) CYCLE IDENTITY on rings of qubit-bubbles")
    R["a_cycle"] = {}
    for n in (3, 4):
        ok, comm, dims = test_cycle_identity(n, 2)
        odd = {k: v for k, v in dims.items() if k.count("+") % 2 == 1}
        even = {k: v for k, v in dims.items() if k.count("+") % 2 == 0}
        print(f"  n={n}: prod_k Q_k == (-1)^n * 1 : {ok};  charges commute: {comm}")
        print(f"        joint-eigenspace real dims, even # of anti-aligned links: {sorted(set(even.values()))}"
              f"  (expected 2*2^{n} = {2 * 2 ** n})")
        print(f"        joint-eigenspace real dims, odd  # of anti-aligned links: {sorted(set(odd.values()))}"
              f"  (Mobius ring: NO states)")
        R["a_cycle"][f"n={n}"] = dict(identity=bool(ok), commute=bool(comm), dims=dims)

    # ---- (b) gauge: conjugating one bubble flips its links' sectors
    print("\n(b) GAUGE: K on bubble 2 of a 3-ring flips Q_12 and Q_23, leaves Q_31")
    Qs = ring_Qs(3, 2)
    K2 = kron(np.eye(4), K_of(2), np.eye(4))
    flips = [np.allclose(K2 @ Q @ K2, -Q) for Q in Qs[:2]] + [np.allclose(K2 @ Qs[2] @ K2, Qs[2])]
    print(f"  Q_12 -> -Q_12: {flips[0]};  Q_23 -> -Q_23: {flips[1]};  Q_31 unchanged: {flips[2]}")
    print("  => any definite sector assignment (holonomy fixed by (a)) is gauge-equivalent to all-aligned.")
    R["b_gauge"] = dict(flip_12=bool(flips[0]), flip_23=bool(flips[1]), keep_31=bool(flips[2]))

    # ---- embedding validation on the Renou chain (Alice C^2 - Bob C^4 - Charlie C^2)
    dims = [2, 4, 2]
    psi = renou_state()
    v_mm = embed(psi, dims, [-1, -1])          # both links standard
    v_pm = embed(psi, dims, [+1, -1])          # Alice's link in the conjugate composite
    v_mp = embed(psi, dims, [-1, +1])
    v_pp = embed(psi, dims, [+1, +1])
    QAB, QBC = link_Q(dims, 0), link_Q(dims, 1)
    emb_ok = (abs(np.linalg.norm(v_mm) - 1) < 1e-12 and np.allclose(QAB @ v_mm, -v_mm)
              and np.allclose(QBC @ v_mm, -v_mm) and np.allclose(QAB @ v_pm, v_pm))
    print(f"\n  embedding: unit norm and correct sector eigenvalues: {emb_ok}")

    # ---- (c) superselection: cross terms vanish for J-linear observables
    print("\n(c) SUPERSELECTION for J-linear observers")
    v_sup = (v_mm + v_pm) / s2                       # coherent superposition across sectors
    v_sup2 = (v_mm - v_pm) / s2
    maxdev = 0.0
    for _ in range(50):
        O = kron(real_form(random_hermitian(2)), real_form(random_hermitian(4)), real_form(random_hermitian(2)))
        avg = 0.5 * (v_mm @ O @ v_mm) + 0.5 * (v_pm @ O @ v_pm)
        maxdev = max(maxdev, abs(v_sup @ O @ v_sup - avg), abs(v_sup2 @ O @ v_sup2 - avg))
    print(f"  max |<sup|O|sup> - sector average| over 50 random J-linear O: {maxdev:.2e}")
    R["c_superselection"] = dict(max_cross_term=maxdev)

    # ---- (d) the Renou witness under sector (in)definiteness
    print("\n(d) RENOU T under sector definiteness (exact strategy of cwf_sr7)")
    T_ref, lab = find_labeling()
    print(f"  Bell-outcome labeling found by search (as in sr7): {lab}")
    T_mm = T_from_real_state(v_mm, dims)
    T_pm_raw = T_from_real_state(v_pm, dims)
    T_pm_relabel = max(T_from_real_state(v_pm, dims, ya=s) for s in (1.0, -1.0))
    T_mp_relabel = max(T_from_real_state(v_mp, dims, yc=s) for s in (1.0, -1.0))
    T_pp_relabel = max(T_from_real_state(v_pp, dims, ya=s, yc=t) for s in (1.0, -1.0) for t in (1.0, -1.0))
    v_indAB = (v_mm + v_pm) / s2
    v_indBC = (v_mm + v_mp) / s2
    v_indboth = (v_mm + v_pm + v_mp + v_pp) / 2.0
    T_indAB = max(T_from_real_state(v_indAB, dims, ya=s) for s in (1.0, -1.0))
    T_indBC = max(T_from_real_state(v_indBC, dims, yc=s) for s in (1.0, -1.0))
    T_indboth = max(T_from_real_state(v_indboth, dims, ya=s, yc=t) for s in (1.0, -1.0) for t in (1.0, -1.0))
    T_noY_a = T_complex_reference(use_Y_a=False)
    T_noY_both = T_complex_reference(use_Y_a=False, use_Y_c=False)
    real_bound = 7.66
    rows = [("definite, both links aligned (H2)", T_mm),
            ("definite, Alice's link anti-aligned, raw labels", T_pm_raw),
            ("definite, Alice's link anti-aligned, Alice relabels sigma_Y", T_pm_relabel),
            ("definite, Charlie's link anti-aligned, relabeled", T_mp_relabel),
            ("definite, both anti-aligned, relabeled", T_pp_relabel),
            ("INDEFINITE Alice link (best labeling)", T_indAB),
            ("INDEFINITE Charlie link (best labeling)", T_indBC),
            ("INDEFINITE both links (best labeling)", T_indboth)]
    print(f"  complex-QM reference (sr7 formula): T = {T_ref:.4f}   (6*sqrt2 = {6 * s2:.4f});  real bound 7.66")
    for name, val in rows:
        flag = "complex" if val > real_bound + 1e-9 else "<= real bound"
        print(f"  {name:62s} T = {val:7.4f}   {flag}")
    print(f"  sr7 ablation (Alice's sigma_Y dropped):            T = {T_noY_a:7.4f}")
    print(f"  sr7 ablation (both sigma_Y dropped):               T = {T_noY_both:7.4f}")
    # effective witness for a link of local definiteness m = |<Q>| (weights (1+-m)/2):
    # T(m) = max over labelings of  w_- T_aligned + w_+ T_anti  =  T_ind + (T_def - T_ind) m
    T_def, T_ind = T_mm, T_indAB
    m_star = (real_bound - T_ind) / (T_def - T_ind)
    print(f"  effective single-link witness T(m) = {T_ind:.3f} + {T_def - T_ind:.3f} m ;"
          f" crosses 7.66 at m* = {m_star:.4f}")
    R["d_renou"] = dict(T_reference=T_ref, T_definite_aligned=T_mm, T_anti_raw=T_pm_raw,
                        T_anti_relabeled=T_pm_relabel, T_anti_charlie_relabeled=T_mp_relabel,
                        T_anti_both_relabeled=T_pp_relabel, T_indefinite_alice=T_indAB,
                        T_indefinite_charlie=T_indBC, T_indefinite_both=T_indboth,
                        T_sr7_noY_alice=T_noY_a, T_sr7_noY_both=T_noY_both, real_bound=real_bound,
                        witness_T_of_m=dict(intercept=T_ind, slope=T_def - T_ind, m_star=m_star))

    # ---- (e) conservation of sector weights
    print("\n(e) CONSERVATION of sector weights under couplings (Alice-Bob link, real dim 32)")
    dA, dB = 2, 4
    Q = np.kron(J_of(dA), J_of(dB))
    v0 = rng.normal(size=4 * dA * dB); v0 /= np.linalg.norm(v0)
    w0 = sector_weights(v0, Q)
    dev_J, dev_O, comm_J, comm_O = 0.0, 0.0, True, True
    for _ in range(20):
        UJ = random_Jlinear_gate(dA, dB)
        UO = random_orthogonal(4 * dA * dB)
        comm_J &= np.allclose(UJ @ Q, Q @ UJ)
        comm_O &= np.allclose(UO @ Q, Q @ UO)
        dev_J = max(dev_J, abs(sector_weights(UJ @ v0, Q)[0] - w0[0]))
        dev_O = max(dev_O, abs(sector_weights(UO @ v0, Q)[0] - w0[0]))
    print(f"  J-linear couplings: commute with Q: {comm_J};  max change of sector weight: {dev_J:.2e}")
    print(f"  generic real orthogonal couplings: commute with Q: {comm_O};  max change: {dev_O:.3f}")
    R["e_conservation"] = dict(Jlinear_commute=bool(comm_J), Jlinear_max_dev=dev_J,
                               orthogonal_commute=bool(comm_O), orthogonal_max_dev=dev_O)

    # ---- verdicts
    R["verdict"] = {
        "sign_is_gauge": bool(all(R["a_cycle"][k]["identity"] for k in R["a_cycle"]) and all(flips)),
        "mobius_impossible": bool(all(0 in set(v for kk, v in R["a_cycle"][k]["dims"].items() if kk.count("+") % 2 == 1)
                                      for k in R["a_cycle"])),
        "definiteness_is_the_witness": bool(T_mm > real_bound and T_indboth <= real_bound and T_pm_relabel > real_bound),
        "indefinite_equals_sr7_real_ablation": bool(abs(T_indboth - T_noY_both) < 1e-9 and abs(T_indAB - T_noY_a) < 1e-9),
        "Jlinear_dynamics_conserve_definiteness": bool(comm_J and dev_J < 1e-9 and dev_O > 1e-3),
    }
    print("\nVERDICTS:", json.dumps(R["verdict"], indent=2))
    with open(os.path.join(HERE, "ic1_sector_gate_results.json"), "w") as f:
        json.dump(R, f, indent=2)
    print("Wrote ic1_sector_gate_results.json")
    plot(rows, real_bound, T_ref)

def plot(rows, real_bound, T_ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["definite\naligned", "definite anti,\nraw", "definite anti,\nrelabeled",
             "indefinite\nAlice link", "indefinite\nCharlie link", "indefinite\nboth links"]
    vals = [rows[0][1], rows[1][1], rows[2][1], rows[5][1], rows[6][1], rows[7][1]]
    cols = ["#1f77b4", "#7f7f7f", "#1f77b4", "#d62728", "#d62728", "#d62728"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=300)
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    ax.axhline(real_bound, color="k", ls="--", lw=1)
    ax.text(len(vals) - 0.55, real_bound + 0.12, "real-QM bound 7.66", ha="right", fontsize=8)
    ax.axhline(T_ref, color="#1f77b4", ls=":", lw=1)
    ax.text(len(vals) - 0.55, T_ref + 0.12, r"$6\sqrt{2}$", ha="right", fontsize=8, color="#1f77b4")
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Renou functional $T$"); ax.set_ylim(0, 9.6)
    ax.set_title("Sector definiteness, not sign, is what the Renou witness sees", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_IC1_sector_gate.png"))
    print("Wrote fig_IC1_sector_gate.png")

if __name__ == "__main__":
    main()
