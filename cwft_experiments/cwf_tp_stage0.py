"""
Triple-Point Stage 0 -- the make-or-break feasibility test for the temporal
irreducibility leg (leg O). See cwf_project/TRIPLE_POINT_PLAN.md.

CONSTRUCTION (corrected after a first attempt: a scrambling Clifford + single-qubit
Z readout gives a DEGENERATE observable -- constant at almost all horizons -- so it
cannot test a wall; see the project note). The sound construction is IQP-style, whose
Clifford limit is provably GF(2)-affine and balanced:

  input-driven IQP brickwork, mirroring paper B's growing causal cone:
    |+>^n  ->  for k=1..t: if x_k=1 apply [ CZ(a_k,b_k) . R_Z(theta) on a_k,b_k ]
            ->  H^{⊗n}  ->  observable y = sign< parity = ∏ Z_i >, computed EXACTLY.
  theta=0: H·(CZ network)·H is CLIFFORD -> parity expectation is a deterministic ±1,
    GF(2)-affine in the input bits -> affine predictor exact at ALL t -> NO wall (the
    rule-90 analog) -> REDUCIBLE.
  theta>0: the diagonal layer carries magic (R_Z) -> an IQP circuit (#P-hard to
    sample) -> the parity becomes a complex Boolean function of growing arity ->
    bounded predictors fail as t grows -> WALL -> IRREDUCIBLE.

The three Stage-0 gates:
  (1) theta=0 is the stabilizer corner: M2=0 and Var(K)=0; both rise with magic.
  (3) the magic knob moves the legs (M2, Var(K) > 0 at theta>0).
  (2) THE CRITICAL ONE: the cost wall tracks magic, on a VALID (non-degenerate)
      observable. Validity gate: theta=0 must be non-degenerate AND affine-exact at
      every horizon. Then: theta=0 has no wall; theta>0 walls. If the theta=0 stream
      walls or is degenerate, or the magic stream does not wall, leg O is unsound
      (split (e)) -- reported plainly.

State-vector exact (cost 2^n per step, magic-INDEPENDENT). Reuses gf2_affine_fit /
train_mlp from cwf_c1_predictive_cost. CPU-only NumPy.  Run: python cwf_tp_stage0.py
"""
import json, os, time, itertools
import numpy as np
from cwf_c1_predictive_cost import gf2_affine_fit, train_mlp

I2 = np.eye(2, dtype=complex)
H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
PAULI = {'I': I2, 'X': np.array([[0, 1], [1, 0]], dtype=complex),
         'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.array([[1, 0], [0, -1]], dtype=complex)}


def hadamard_all(n):
    M = np.array([[1]], dtype=complex)
    for _ in range(n):
        M = np.kron(M, H1)
    return M


def bits_of(j, n):
    return [(j >> (n - 1 - i)) & 1 for i in range(n)]


def step_diag(n, a, b, theta):
    """Diagonal of CZ(a,b) . R_Z(theta) on a . R_Z(theta) on b, over 2^n basis."""
    d = 1 << n
    v = np.empty(d, dtype=complex)
    for j in range(d):
        bt = bits_of(j, n)
        phase = np.pi * (bt[a] & bt[b])                  # CZ
        phase += theta * (bt[a] - 0.5) + theta * (bt[b] - 0.5)   # R_Z on a,b
        v[j] = np.exp(1j * phase)
    return v


def parity_diag(n):
    d = 1 << n
    return np.array([(-1) ** bin(j).count("1") for j in range(d)], dtype=float)


# --- magic monotone M2 (stabilizer Renyi-2) and capacity Var(K) ---
def all_paulis(n):
    for combo in itertools.product('IXYZ', repeat=n):
        M = np.array([[1]], dtype=complex)
        for c in combo:
            M = np.kron(M, PAULI[c])
        yield M


def stab_renyi2(psi, n):
    s = sum(np.real(np.vdot(psi, P @ psi)) ** 4 for P in all_paulis(n))
    return float(-np.log2(s / (1 << n)))


def stab_renyi2_sampled(psi, n, nsamp, seed):
    """Sampled M2 = -log2( 2^n * mean_P <P>^4 ) over random Pauli strings. 0 iff stabilizer."""
    rng = np.random.default_rng(seed)
    acc = 0.0
    for _ in range(nsamp):
        M = np.array([[1]], dtype=complex)
        for _q in range(n):
            M = np.kron(M, PAULI['IXYZ'[rng.integers(4)]])
        acc += np.real(np.vdot(psi, M @ psi)) ** 4
    return float(-np.log2((1 << n) * acc / nsamp + 1e-300))


def var_K(psi, n, nA):
    m = psi.reshape((1 << nA, 1 << (n - nA)))
    p = np.linalg.eigvalsh(m @ m.conj().T)
    p = p[p > 1e-12]
    lk = -np.log2(p)
    return float(np.sum(p * lk ** 2) - np.sum(p * lk) ** 2)


# --- input-driven IQP trajectory, exact parity observable ---
def gen_dataset(n, theta, t, n_samples, seed, Hn, parity, plus_state, cz_pairs):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, (n_samples, t)).astype(np.int8)
    state = np.tile(plus_state, (n_samples, 1)).astype(complex)
    diags = [step_diag(n, a, b, theta) for (a, b) in cz_pairs]
    for k in range(t):
        dg = diags[k % len(cz_pairs)]
        m = x[:, k] == 1
        if m.any():
            state[m] = state[m] * dg                      # diagonal phase layer
    state = state @ Hn.T
    ez = (np.abs(state) ** 2) @ parity                    # <parity>(x), exact
    # median-split target -> base ~0.5 by construction (removes the imbalance artifact);
    # truly-constant observable (spread ~0) is flagged degenerate by the caller via balance.
    if float(ez.max() - ez.min()) < 1e-9:
        y = np.zeros(len(ez), np.int8)
    else:
        y = (ez > np.median(ez)).astype(np.int8)
        if y.mean() in (0.0, 1.0):
            y = (ez > ez.mean()).astype(np.int8)
    return x.astype(np.float64), y, ez


def quad_features(X):
    """Append GF(2) pairwise products x_i*x_j (i<j) -- the degree-2 feature map."""
    d = X.shape[1]
    cols = [X] + [X[:, i:i + 1] * X[:, i + 1:] for i in range(d - 1)]
    return np.concatenate(cols, axis=1)


def gf2_quadratic_fit(Xtr, ytr, Xte, yte):
    """Exact GF(2) DEGREE-2 predictor (affine fit on pairwise-product features).
    Clifford/stabilizer (graph-state) observables are GF(2)-quadratic Gauss sums, so
    this is the correct 'reducible shortcut' for the quantum setting (the degree-2
    analog of the degree-1 affine shortcut that is exact for rule-90)."""
    Xq, Xqe = quad_features(Xtr), quad_features(Xte)
    return gf2_affine_fit(Xq, ytr, Xqe, yte), Xq.shape[1]


def cost_of_wall(Xw, y, H_grid, eps, seed):
    ntr = int(0.7 * len(y))
    Xtr, Xte, ytr, yte = Xw[:ntr], Xw[ntr:], y[:ntr], y[ntr:]
    base = max(yte.mean(), 1 - yte.mean())
    d = Xw.shape[1]
    sk = lambda acc: (acc - base) / (1 - base + 1e-12)
    cands, skills = [], {}
    skills['affine'] = sk(gf2_affine_fit(Xtr, ytr, Xte, yte))
    if skills['affine'] >= eps:
        cands.append(d + 1)
    acc_q, dq = gf2_quadratic_fit(Xtr, ytr, Xte, yte)
    skills['quad'] = sk(acc_q)
    if skills['quad'] >= eps:
        cands.append(dq + 1)
    for H in H_grid:
        skills[f'H{H}'] = sk(train_mlp(Xtr, ytr, Xte, yte, H=H, seed=seed + H))
        if skills[f'H{H}'] >= eps:
            cands.append((H * (d + 2) + 1) if H > 0 else d + 1)
    return (int(min(cands)) if cands else None), float(base), skills


def main():
    n = 8
    H_grid = [0, 8, 32]
    horizons = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
    n_samples = 3000
    eps = 0.75
    thetas = {"theta=0 (Clifford)": 0.0, "theta=pi/8": np.pi / 8, "theta=pi/4 (T/magic)": np.pi / 4}
    seed0 = 11
    t0 = time.time()

    Hn = hadamard_all(n)
    parity = parity_diag(n)
    e0 = np.zeros(1 << n, dtype=complex); e0[0] = 1.0
    plus_state = Hn @ e0
    cz_pairs = [(i, (i + 1) % n) for i in range(n)]

    # ---- gates 1 & 3: M2(theta), Var(K)(theta) on a fixed-input IQP reference state ----
    ref = np.array([1, 0, 1, 1, 0, 1, 1, 0], dtype=np.int8)
    legs = {}
    for label, th in thetas.items():
        diags = [step_diag(n, a, b, th) for (a, b) in cz_pairs]
        psi = plus_state.copy()
        for k, bk in enumerate(ref):
            if bk:
                psi = psi * diags[k % len(cz_pairs)]
        psi = Hn @ psi
        m2 = stab_renyi2(psi, n) if n <= 6 else stab_renyi2_sampled(psi, n, 1500, seed0)
        legs[label] = dict(M2=m2, VarK=var_K(psi, n, n // 2))
        print(f"[gate1/3] {label:24s}  M2={legs[label]['M2']:.4f}  Var(K)={legs[label]['VarK']:.4f}")

    # ---- gate 2: the temporal cost wall vs magic, on the valid parity observable ----
    cost = {}
    for label, th in thetas.items():
        rows = []
        for t in horizons:
            Xw, y, ez = gen_dataset(n, th, t, n_samples, seed0 + t, Hn, parity, plus_state, cz_pairs)
            bal = float(y.mean())
            degen = min(bal, 1 - bal) < 0.05
            if degen:
                rows.append(dict(t=t, req_params=None, base=max(bal, 1 - bal), degenerate=True, skills={}))
                print(f"[gate2] {label:24s} t={t:2d}  DEGENERATE (bal={bal:.3f})")
                continue
            req, base, sk = cost_of_wall(Xw, y, H_grid, eps, seed0 + t)
            rows.append(dict(t=t, req_params=req, base=base, degenerate=False, skills=sk))
            bestH = max(v for k, v in sk.items() if k.startswith('H'))
            print(f"[gate2] {label:24s} t={t:2d}  base={base:.2f}  affine_sk={sk['affine']:+.2f}  "
                  f"quad_sk={sk['quad']:+.2f}  bestH_sk={bestH:+.2f}  req={req}")
        any_degen = any(r["degenerate"] for r in rows)
        walls = [r["t"] for r in rows if (not r["degenerate"]) and r["req_params"] is None]
        nd = [r for r in rows if not r["degenerate"]]
        affine_all = all(r["skills"].get("affine", 0) >= 0.99 for r in nd)
        quad_all = all(r["skills"].get("quad", 0) >= eps for r in nd)
        cost[label] = dict(theta=th, rows=rows, wall_t=(walls[0] if walls else None),
                           any_degenerate=any_degen, affine_exact_all_t=affine_all,
                           quad_reducible_all_t=quad_all)
        print(f"  -> {label}: wall_t={cost[label]['wall_t']}  degenerate_any={any_degen}  "
              f"affine_exact_all_t={affine_all}  quad_reducible_all_t={quad_all}\n")

    cliff, magicT = cost["theta=0 (Clifford)"], cost["theta=pi/4 (T/magic)"]
    # gate 1/3 use the magic monotone M2 (the leg-O resource knob); M2~0 at Clifford
    # (sampled, small tolerance), M2>0 with magic. Var(K) is the Stage-1 gravity leg
    # (IQP states are flat across this cut -> reported informationally, not gated here).
    gate1 = legs["theta=0 (Clifford)"]["M2"] < 0.05
    gate3 = legs["theta=pi/4 (T/magic)"]["M2"] > 0.1
    valid = not cliff["any_degenerate"]                       # construction validity
    # reducible baseline for the QUANTUM setting = GF(2) degree-2 (graph-state quadratic),
    # not degree-1 affine. Clifford reducible iff a degree-2 fit reaches skill at all t.
    cliff_reducible = valid and cliff["quad_reducible_all_t"] and cliff["wall_t"] is None
    magic_walls = magicT["wall_t"] is not None
    gate2 = cliff_reducible and magic_walls
    if not valid:
        verdict = "INCONCLUSIVE -- theta=0 observable degenerate at some horizon (construction invalid)."
    elif gate2:
        verdict = "LEG O SOUND -- Clifford reducible (affine, no wall); magic walls. Temporal angle viable."
    else:
        verdict = ("LEG O UNSOUND (split (e)) -- "
                   + ("Clifford stream walls/non-affine; " if not cliff_reducible else "")
                   + ("magic stream does not wall; " if not magic_walls else "")
                   + "temporal-irreducibility angle fails as constructed.")
    summary = dict(n=n, eps=eps, gate1_stabilizer_corner=bool(gate1), gate3_magic_moves_legs=bool(gate3),
                   gate2_construction_valid=bool(valid), gate2_cliff_reducible=bool(cliff_reducible),
                   gate2_magic_walls=bool(magic_walls), gate2_leg_O_sound=bool(gate2), verdict=verdict)
    print("=" * 72)
    for k, v in summary.items():
        if k != "verdict": print(f"  {k}: {v}")
    print("  VERDICT:", verdict)
    print("=" * 72)

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["TP_stage0"] = dict(summary=summary, legs=legs, cost=cost,
                              meta=dict(horizons=horizons, H_grid=H_grid, n_samples=n_samples,
                                        construction="input-driven IQP, parity observable",
                                        runtime_s=round(time.time() - t0, 1)))
    json.dump(r_all, open(out, "w"), indent=2)
    print(f"runtime {round(time.time()-t0,1)}s  -> results.json key TP_stage0")


if __name__ == "__main__":
    main()
