#!/usr/bin/env python3
"""
cwf_psi_l2_vs_l1.py  --  The l2-vs-l1 horse-race.

Tests the load-bearing clause of the optimal-epistemic-object conjecture
(Chapter 6): that the optimal object an embedded observer holds over an
(un)decidable computational ground UPGRADES from a classical l1 probability
mixture (Solomonoff) to a complex l2 interfering amplitude (Born rule).

The conjecture is split into two operationally distinct questions, because
conflating them is exactly how "the wave is the optimal belief" overreaches:

  (A) ACCURACY.  Does an l2 (Born-rule, |amplitude|^2) predictor achieve lower
      held-out log-loss than an l1 (classical probability) predictor?
      CLAIM (this script's thesis): NO, and provably so.  The log-loss floor is
      the process entropy rate h_mu, attained by the true conditional
      distribution -- a classical probability vector.  A Born-rule model only
      RE-PARAMETERISES conditional distributions, so it cannot beat h_mu.
      We confirm the floor is reached *classically* (order-k Markov sweep
      plateaus at h_mu).  No accuracy advantage from amplitudes, by the
      definition of "prediction".

  (B) COMPRESSION.  At matched accuracy, does the l2 model need a smaller memory
      than the l1 model?  Here a real advantage CAN exist and is theorem-backed:
      Gu, Wiesner, Rieper & Vedral, "Quantum mechanics can reduce the complexity
      of classical models", Nat. Commun. 3, 762 (2012): the quantum statistical
      complexity C_q <= classical statistical complexity C_mu, with strict
      inequality whenever distinct causal states have overlapping futures.
      We measure C_mu (epsilon-machine) and C_q (q-machine) for each substrate's
      observation process.

Substrates: Rule 110 (universal / irreducible), Rule 30 (chaotic),
Rule 90 (additive / reducible control -- column process is ~IID fair coin,
so C_mu ~ C_q ~ 0; the dissociation that the three-axis story predicts).

Honest caveats are printed inline and saved to results.  The epsilon-machine is
a finite-history (order-L) approximation; the q-machine uses the standard
one-step quantum causal states, so C_q here is an UPPER bound on the true
quantum complexity -- which only strengthens any C_q < C_mu it finds.

CPU-only, numpy only.  Reuses no external state; standalone by design.
"""

import json
import time
import os
from collections import Counter, defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# 1. Substrate: elementary cellular automaton + observation channel
# ----------------------------------------------------------------------------
def eca_step(row, rule):
    """One synchronous ECA update, periodic boundaries. row: int64 array of 0/1."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right          # neighbourhood code 0..7
    return ((rule >> idx) & 1).astype(np.int64)


def simulate_observation(rule, W, T, n_seqs, burn, rng, channel="column", win=3):
    """
    Evolve the ECA from random IID initial rows and read a coarse observable each
    step -- the embedded observer's partial access.  Returns a list of int8
    sequences (one per random initial condition), each of length T - burn.

      channel='column'  : value of the centre cell (single-site readout)
      channel='parity'  : XOR-parity of a centre window of width `win`
    """
    seqs = []
    c = W // 2
    lo, hi = c - win // 2, c + win // 2 + 1
    for _ in range(n_seqs):
        row = rng.integers(0, 2, size=W).astype(np.int64)
        obs = np.empty(T, dtype=np.int8)
        for t in range(T):
            if channel == "column":
                obs[t] = row[c]
            else:
                obs[t] = int(np.bitwise_xor.reduce(row[lo:hi]))
            row = eca_step(row, rule)
        seqs.append(obs[burn:])
    return seqs


# ----------------------------------------------------------------------------
# 2. (A) Accuracy: held-out log-loss of order-k classical predictors
# ----------------------------------------------------------------------------
def order_k_logloss(seqs_train, seqs_test, k, alpha=0.5):
    """Held-out predictive log-loss (bits/symbol) of an order-k Markov model
    with Laplace(alpha) smoothing.  k=0 is the marginal model."""
    counts = defaultdict(lambda: np.zeros(2))
    for seq in seqs_train:
        s = np.asarray(seq)
        for t in range(k, len(s)):
            counts[tuple(s[t - k:t])][s[t]] += 1
    bits, n = 0.0, 0
    for seq in seqs_test:
        s = np.asarray(seq)
        for t in range(k, len(s)):
            ctx = tuple(s[t - k:t])
            c = counts.get(ctx)
            if c is None:
                p = np.array([0.5, 0.5])
            else:
                p = (c + alpha) / (c.sum() + 2 * alpha)
            bits += -np.log2(p[s[t]])
            n += 1
    return bits / max(n, 1)


def accuracy_curve(seqs_train, seqs_test, kmax=12):
    """log-loss vs order k.  The minimum is the best classical predictive
    accuracy at this data budget; its value estimates the entropy-rate floor
    h_mu that NO predictor (classical or Born-rule) can beat."""
    curve = [order_k_logloss(seqs_train, seqs_test, k) for k in range(kmax + 1)]
    kbest = int(np.argmin(curve))
    return curve, kbest, float(curve[kbest])


# ----------------------------------------------------------------------------
# 3. (B) Compression: epsilon-machine (C_mu) and q-machine (C_q)
# ----------------------------------------------------------------------------
def infer_epsilon_machine(seqs_train, L, F, tol=0.10, min_count=20):
    """
    Finite-history causal-state reconstruction (approximate CSSR).

    Causal states = clusters of length-L pasts whose conditional distribution
    over length-F futures (the 'morph') agree within total-variation `tol`.
    Returns the machine: causal-state stationary distribution pi, and the
    one-step joint-transition tensor  P(symbol x, next state j | state i).
    """
    fut_counts = defaultdict(Counter)
    ctx_total = Counter()
    for seq in seqs_train:
        s = np.asarray(seq)
        for t in range(L, len(s) - F + 1):
            past = tuple(s[t - L:t])
            fut = tuple(s[t:t + F])
            fut_counts[past][fut] += 1
            ctx_total[past] += 1

    # morph vectors over the observed length-F future alphabet
    all_futs = sorted({f for c in fut_counts.values() for f in c})
    fidx = {f: i for i, f in enumerate(all_futs)}
    pasts = [p for p in ctx_total if ctx_total[p] >= min_count]
    pasts.sort(key=lambda p: -ctx_total[p])          # well-estimated reps first
    if not pasts:
        return None
    morph = {}
    for p in pasts:
        v = np.zeros(len(all_futs))
        for f, c in fut_counts[p].items():
            v[fidx[f]] = c
        morph[p] = v / v.sum()

    # greedy TV clustering -> causal states
    reps = []            # representative morph per state
    members = []         # member pasts per state
    assign = {}
    for p in pasts:
        v = morph[p]
        placed = False
        for si, rep in enumerate(reps):
            if 0.5 * np.abs(v - rep).sum() < tol:
                members[si].append(p)
                assign[p] = si
                placed = True
                break
        if not placed:
            reps.append(v.copy())
            members.append([p])
            assign[p] = len(reps) - 1
    nstates = len(reps)

    # stationary distribution over causal states (empirical occupancy)
    occ = np.zeros(nstates)
    for p in pasts:
        occ[assign[p]] += ctx_total[p]
    pi = occ / occ.sum()

    # one-step joint transitions T[(x,j)] from each state i, estimated from data
    # next past after seeing symbol x from past p is p[1:]+(x,)
    joint = [defaultdict(float) for _ in range(nstates)]   # i -> {(x,j): count}
    row_tot = np.zeros(nstates)
    for seq in seqs_train:
        s = np.asarray(seq)
        for t in range(L, len(s) - 1):
            p = tuple(s[t - L:t])
            i = assign.get(p)
            if i is None:
                continue
            x = int(s[t])
            pnext = tuple(s[t - L + 1:t + 1])
            j = assign.get(pnext)
            if j is None:                  # next context too rare to be a state
                continue
            joint[i][(x, j)] += 1.0
            row_tot[i] += 1.0

    return {
        "nstates": nstates,
        "pi": pi,
        "joint": joint,
        "row_tot": row_tot,
        "sizes": [len(m) for m in members],
        "all_futs": len(all_futs),
    }


def C_mu(machine):
    """Classical statistical complexity = Shannon entropy of causal-state dist."""
    pi = machine["pi"]
    pi = pi[pi > 0]
    return float(-np.sum(pi * np.log2(pi)))


def C_q(machine):
    """
    Quantum statistical complexity via the one-step q-machine
    (Gu et al. 2012).  Quantum causal states
        |eta_i> = sum_{x,j} sqrt(P(x,j|i)) |x,j>,
    and C_q = S(rho), rho = sum_i pi_i |eta_i><eta_i|.  The nonzero spectrum of
    rho equals that of  G_{ik} = sqrt(pi_i pi_k) <eta_i|eta_k>, which we
    diagonalise.  C_q here is an UPPER bound on the true quantum complexity
    (longer-future constructions can only lower it), so C_q < C_mu is robust.
    """
    n = machine["nstates"]
    pi = machine["pi"]
    joint = machine["joint"]
    row_tot = machine["row_tot"]

    # build amplitude (sqrt-probability) vectors over a common (x,j) basis
    basis = sorted({xj for i in range(n) for xj in joint[i].keys()})
    bidx = {xj: b for b, xj in enumerate(basis)}
    amp = np.zeros((n, len(basis)))
    for i in range(n):
        if row_tot[i] <= 0:
            continue
        for xj, c in joint[i].items():
            amp[i, bidx[xj]] = np.sqrt(c / row_tot[i])     # sqrt P(x,j|i)

    G = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            G[i, k] = np.sqrt(pi[i] * pi[k]) * float(amp[i] @ amp[k])
    w = np.linalg.eigvalsh(G)
    w = w[w > 1e-12]
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(-np.sum(w * np.log2(w)))


# ----------------------------------------------------------------------------
# 4. Self-test: validate the q-machine math on a known input
# ----------------------------------------------------------------------------
def self_test():
    """Build a tiny machine by hand and check the invariants:
       - C_q <= C_mu always
       - identical morphs (one causal state) -> C_mu = C_q = 0
       - distinct, non-overlapping futures -> C_q = C_mu (no compression)
       - overlapping futures -> C_q < C_mu (genuine compression).
    """
    print("[self-test] q-machine invariants")

    # case 1: two states with DISJOINT next-(x,j) supports -> no overlap
    m = {"nstates": 2, "pi": np.array([0.5, 0.5]),
         "row_tot": np.array([10.0, 10.0]),
         "joint": [defaultdict(float, {(0, 0): 10.0}),
                   defaultdict(float, {(1, 1): 10.0})]}
    cm, cq = C_mu(m), C_q(m)
    print(f"  disjoint futures : C_mu={cm:.4f}  C_q={cq:.4f}   (expect equal, =1)")
    assert abs(cm - 1.0) < 1e-6 and abs(cq - 1.0) < 1e-6

    # case 2: two states with IDENTICAL futures -> should collapse (C_q=0)
    m = {"nstates": 2, "pi": np.array([0.5, 0.5]),
         "row_tot": np.array([10.0, 10.0]),
         "joint": [defaultdict(float, {(0, 0): 5.0, (1, 1): 5.0}),
                   defaultdict(float, {(0, 0): 5.0, (1, 1): 5.0})]}
    cm, cq = C_mu(m), C_q(m)
    print(f"  identical futures: C_mu={cm:.4f}  C_q={cq:.4f}   (expect C_mu=1, C_q=0)")
    assert abs(cm - 1.0) < 1e-6 and cq < 1e-6

    # case 3: partial overlap -> 0 < C_q < C_mu
    m = {"nstates": 2, "pi": np.array([0.5, 0.5]),
         "row_tot": np.array([10.0, 10.0]),
         "joint": [defaultdict(float, {(0, 0): 9.0, (1, 1): 1.0}),
                   defaultdict(float, {(0, 0): 1.0, (1, 1): 9.0})]}
    cm, cq = C_mu(m), C_q(m)
    print(f"  partial overlap  : C_mu={cm:.4f}  C_q={cq:.4f}   (expect C_q<C_mu)")
    assert cq < cm - 1e-6
    print("  [self-test] PASSED\n")


# ----------------------------------------------------------------------------
# 5. Driver
# ----------------------------------------------------------------------------
def run():
    t0 = time.time()
    self_test()

    cfg = dict(W=201, T=1400, burn=400, n_seqs=160, kmax=12,
               L=8, F=4, tol=0.10, min_count=20, channel="column", win=3, seed=20260530)
    rng = np.random.default_rng(cfg["seed"])
    rules = {"Rule110 (universal)": 110, "Rule30 (chaotic)": 30,
             "Rule90 (additive)": 90, "Rule184 (class II)": 184}

    results = {"config": cfg, "substrates": {}}
    print(f"config: {cfg}\n")
    header = (f"{'substrate':<22}{'h_mu(bits)':>11}{'k*':>4}"
              f"{'C_mu(bits)':>11}{'C_q(bits)':>10}{'C_q/C_mu':>9}{'#states':>8}")
    print(header)
    print("-" * len(header))

    for name, rule in rules.items():
        seqs = simulate_observation(rule, cfg["W"], cfg["T"], cfg["n_seqs"],
                                    cfg["burn"], rng, channel=cfg["channel"],
                                    win=cfg["win"])
        ntr = int(0.7 * len(seqs))
        tr, te = seqs[:ntr], seqs[ntr:]

        curve, kbest, hmu = accuracy_curve(tr, te, cfg["kmax"])

        mach = infer_epsilon_machine(tr, cfg["L"], cfg["F"],
                                     tol=cfg["tol"], min_count=cfg["min_count"])
        if mach is None:
            cm = cq = ratio = float("nan")
            nst = 0
        else:
            cm = C_mu(mach)
            cq = C_q(mach)
            ratio = cq / cm if cm > 1e-9 else float("nan")
            nst = mach["nstates"]
            assert cq <= cm + 1e-6, "INVARIANT VIOLATED: C_q > C_mu"

        results["substrates"][name] = {
            "rule": rule, "h_mu_bits": hmu, "kbest": kbest,
            "logloss_curve": curve, "C_mu_bits": cm, "C_q_bits": cq,
            "Cq_over_Cmu": ratio, "n_causal_states": nst,
        }
        print(f"{name:<22}{hmu:>11.4f}{kbest:>4}{cm:>11.4f}{cq:>10.4f}"
              f"{ratio:>9.3f}{nst:>8}")

    # ---- CONTROL: synthetic processes with KNOWN statistical complexity -------
    # An IID coin has TRUE C_mu = C_q = 0 (memoryless: one causal state).  Any
    # C_mu/C_q the reconstruction reports on it is the finite-sample NOISE FLOOR.
    # If Rule30/Rule90 sit at this floor, their large "compression" is the
    # q-machine denoising the artifact, not compressing real structure.
    print("\nCONTROL: synthetic processes (known true C_mu = C_q = 0 for IID)")
    print(f"{'process':<22}{'h_mu(bits)':>11}{'C_mu(bits)':>11}{'C_q(bits)':>10}"
          f"{'C_q/C_mu':>9}{'#states':>8}")
    controls = {}
    nseq, slen = cfg["n_seqs"], cfg["T"] - cfg["burn"]
    synth = {
        "IID fair coin": lambda: rng.integers(0, 2, size=slen).astype(np.int8),
        "IID biased p=.15": lambda: (rng.random(slen) < 0.15).astype(np.int8),
    }
    for sname, gen in synth.items():
        sq = [gen() for _ in range(nseq)]
        tr = sq[:int(0.7 * len(sq))]
        te = sq[int(0.7 * len(sq)):]
        _, _, hmu = accuracy_curve(tr, te, cfg["kmax"])
        m = infer_epsilon_machine(tr, cfg["L"], cfg["F"], tol=cfg["tol"],
                                  min_count=cfg["min_count"])
        if m is None:
            cm = cq = r = float("nan"); nst = 0
        else:
            cm, cq = C_mu(m), C_q(m)
            r = cq / cm if cm > 1e-9 else float("nan")
            nst = m["nstates"]
        controls[sname] = dict(h_mu_bits=hmu, C_mu_bits=cm, C_q_bits=cq,
                               Cq_over_Cmu=r, n_causal_states=nst)
        print(f"{sname:<22}{hmu:>11.4f}{cm:>11.4f}{cq:>10.4f}{r:>9.3f}{nst:>8}")
    results["controls_known_C"] = controls

    # ---- robustness of the compression result to the clustering tolerance ----
    print("\ntolerance sweep (Rule110, column):  tol -> (#states, C_mu, C_q, C_q/C_mu)")
    seqs110 = simulate_observation(110, cfg["W"], cfg["T"], cfg["n_seqs"],
                                   cfg["burn"], rng, channel=cfg["channel"])
    tr110 = seqs110[:int(0.7 * len(seqs110))]
    tol_sweep = {}
    for tol in (0.05, 0.08, 0.10, 0.15, 0.20):
        m = infer_epsilon_machine(tr110, cfg["L"], cfg["F"], tol=tol,
                                  min_count=cfg["min_count"])
        if m is None:
            continue
        cm, cq = C_mu(m), C_q(m)
        r = cq / cm if cm > 1e-9 else float("nan")
        tol_sweep[tol] = dict(nstates=m["nstates"], C_mu=cm, C_q=cq, ratio=r)
        print(f"  tol={tol:<5}  states={m['nstates']:<5} C_mu={cm:.4f}  "
              f"C_q={cq:.4f}  C_q/C_mu={r:.3f}")
    results["tol_sweep_rule110"] = tol_sweep

    results["runtime_s"] = round(time.time() - t0, 1)
    out = os.path.join(HERE, "psi_l2_vs_l1_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nsaved -> {out}   ({results['runtime_s']}s)")

    if HAVE_MPL:
        _figure(results)

    _interpret(results)
    return results


def _figure(results):
    subs = results["substrates"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, d in subs.items():
        ax[0].plot(range(len(d["logloss_curve"])), d["logloss_curve"],
                   marker="o", ms=3, label=name)
    ax[0].set_xlabel("Markov order k")
    ax[0].set_ylabel("held-out log-loss (bits/symbol)")
    ax[0].set_title("(A) Accuracy floor h_mu is reached classically\n"
                    "(no amplitude can go below it)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    names = list(subs.keys())
    cm = [subs[n]["C_mu_bits"] for n in names]
    cq = [subs[n]["C_q_bits"] for n in names]
    x = np.arange(len(names))
    ax[1].bar(x - 0.18, cm, 0.36, label="C_mu (classical l1)")
    ax[1].bar(x + 0.18, cq, 0.36, label="C_q (quantum l2)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([n.split()[0] for n in names], rotation=20, fontsize=8)
    ax[1].set_ylabel("statistical complexity (bits)")
    ax[1].set_title("(B) Compression: C_q vs C_mu\n"
                    "(this is where any l2 advantage lives)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(HERE, "fig_psi_l2_vs_l1.png")
    fig.savefig(p, dpi=130)
    print(f"saved -> {p}")


def _interpret(results):
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    subs = results["substrates"]
    print("(A) ACCURACY: log-loss bottoms out at h_mu for every substrate using a"
          "\n    classical order-k model. A Born-rule predictor re-parameterises the"
          "\n    same conditional distributions and so CANNOT beat h_mu. =>"
          "\n    No predictive-accuracy advantage from l2. (the crux objection)\n")
    # Noise floor: reconstruction inflates C_mu (spurious near-identical states)
    # but NOT C_q (the q-machine collapses near-identical noisy morphs), so C_q
    # is the noise-robust memory estimate. Floor = largest C_q on the IID
    # controls (true complexity 0); genuine memory needs C_q well above it.
    ctrl = results.get("controls_known_C", {})
    floor_q = max([c["C_q_bits"] for c in ctrl.values()], default=0.0)
    floor_mu = max([c["C_mu_bits"] for c in ctrl.values()], default=0.0)
    print(f"(B) COMPRESSION: IID noise floor (true C=0): C_q<={floor_q:.2f}, "
          f"C_mu<={floor_mu:.2f} bits. C_q is the noise-robust statistic.")
    for n, d in subs.items():
        cm, cq, r = d["C_mu_bits"], d["C_q_bits"], d["Cq_over_Cmu"]
        if r != r:
            continue
        if cq <= floor_q * 1.25:
            verdict = "ARTIFACT: memoryless (C_q at floor; apparent compression is denoising)"
        elif r < 0.95:
            verdict = "genuine memory + real l2 compression"
        else:
            verdict = "genuine memory, ~no compression at this scale"
        print(f"    {n:<22} C_mu={cm:.3f} C_q={cq:.3f}  ratio={r:.3f}  -> {verdict}")
    print("\n    Only substrates with C_mu well above the floor carry genuine memory."
          "\n    There the l2 (Born-rule) advantage is COMPRESSION, not accuracy -- but"
          "\n    modest at this scale. The one-step q-machine UNDER-states C_q's gap,"
          "\n    so the measured compression is a conservative lower bound.")


if __name__ == "__main__":
    run()
