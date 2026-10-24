#!/usr/bin/env python3
"""UNIT TESTS FOR THE COMPRESSION PIPELINE (cwf_psi_unifilar.py, version 2.6).

Run with `python3 test_psi_unifilar.py`.  No pytest, no arguments, no files written;
about a minute.  Exit status is 0 iff every test passes.

WHY THIS FILE EXISTS.  Audit A001 (Codex, 2026-10-21, round 1) found that the
pipeline's checks were weaker than their labels.  Each test below covers the whole
class of one finding, in every layer that consumes the fix (the helper and the driver):

  F1  a zero-saving process of infinite Markov order (the hidden clock) gives large
      finite-history savings; its exact memories are pinned here;
  F2  a small step-to-step change of the Gram recursion was taken as convergence; C_q
      now comes from the exact infinite-depth overlaps;
  F3  NaN, infinite, negative or wrongly sized spectra, and malformed machines or
      occupancies, became a zero entropy (a 100% saving) instead of an error;
  F5  the docstring promised the largest compatible group, the code takes the first;
  F6  the bootstrap's stationary law was a 4000-step average and its minimisation
      rounded emission rows to a 1e-9 grid;
  F8  the share of transitions routed through rare-successor proxies wasn't recorded.

Round 2 (R2-F1 to R2-F3): the direct overlap solve returned wrong limits on slow
multi-state cycles, and sqrt(p * q) underflowed; the stationary solve lost slow rates;
complex values and wrong-length occupancies slipped through.

Round 3 (R3-F1 to R3-F3): the overlap bound left accumulated rounding out; GTH
intermediates could overflow on valid chains; q_entropy took a malformed occupancy.

Round 4 (R4-F1, R4-F2): the rescaled GTH back-substitution could underflow and drop a
component; rows within 1e-9 of normalised were accepted as they were, and an overlap could
be clipped by more than its bound.

Before round 5 (M10, the master's own finding in R4-F1's class): the transient elimination
in stationary_from could underflow the same way when a transient state leaves through a
subnormal rate.  The round-4 tests reached only the helpers, so the bootstrap's generator,
built inside run(), is now generator_truth() and has a driver test, and evaluate has one
for the normalisation.

Round 5 (R5-F1): the count of minimal states left out states occupied below 1e-12, and
the driver test printed it without checking it.  Driver tests now assert every value
they get back.

Every test is deterministic (seeded).
"""
import copy
import itertools
import json
import math
import os
import sys
import traceback

import numpy as np

import cwf_psi_unifilar as M

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = []


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def slow_machine(eps):
    """Two states that agree on symbol 0 (and swap) and merge on symbol 1, which has
    probability eps.  The limiting overlap is 1, so C_q = 0, reached only slowly."""
    return M.Machine([[1 - eps, eps], [1 - eps, eps]], [[1, 0], [0, 0]],
                     pi=[1 / (2 - eps), (1 - eps) / (2 - eps)])


def random_machine(rng, n, deterministic_share=0.25):
    p1 = rng.uniform(0.05, 0.95, size=n)
    P = np.column_stack([1 - p1, p1])
    for i in range(n):
        if rng.random() < deterministic_share:
            P[i] = [0.0, 1.0] if rng.random() < 0.5 else [1.0, 0.0]
    succ = rng.integers(0, n, size=(n, 2))
    succ[P == 0.0] = -1
    pi = M.stationary_from(M.Machine(P, succ, pi=np.full(n, 1 / n)), np.full(n, 1 / n))
    return M.Machine(P, succ, pi=pi)


def gram_by_enumeration(m, d):
    """Independent depth-d Gram matrix: sum over words w of sqrt(P(w|j) P(w|k)), counted
    only when j and k end in the same state."""
    G = np.zeros((m.n, m.n))
    for w in itertools.product((0, 1), repeat=d):
        prob, end = np.ones(m.n), np.arange(m.n)
        for x in w:
            for j in range(m.n):
                if prob[j] > 0:
                    p = m.P[end[j], x]
                    prob[j] *= p
                    end[j] = m.succ[end[j], x] if p > 0 else -1
        for j in range(m.n):
            for k in range(m.n):
                if prob[j] > 0 and prob[k] > 0 and end[j] == end[k]:
                    G[j, k] += math.sqrt(prob[j] * prob[k])
    return G


def predicted_depth(eps, tol=1e-12):
    """The certified depth for a pair whose unmerged weight is exactly (1 - eps)^d: the
    first d at which the returned bound, rounding allowance included, falls below tol;
    None if the rounding allowance reaches tol first."""
    d = 0
    while True:
        d += 1
        if M.ROUNDING_PER_STEP * d >= tol:
            return None
        if M.ROUNDING_PER_STEP * d + (1 - eps) ** d * math.exp(6 * M.U * d) < tol:
            return d


def reset_cycle(n, eps):
    """n states, every row [1 - eps, eps]; symbol 0 cycles the states, symbol 1 resets all
    of them to state 0.  Every pair merges eventually, so every overlap's limit is 1."""
    return M.Machine(np.tile([1 - eps, eps], (n, 1)),
                     np.column_stack([(np.arange(n) + 1) % n, np.zeros(n, dtype=int)]), pi=np.full(n, 1 / n))


def mp_overlap_limit(m, dps=50):
    """High-precision reference for the infinite-depth overlaps: the same pair system,
    built and solved in mpmath at dps digits (an independent implementation)."""
    import mpmath as mp
    mp.mp.dps = dps
    n = m.n
    pairs = [(j, k) for j in range(n) for k in range(j + 1, n)]
    idx = {pq: i for i, pq in enumerate(pairs)}
    succs, b = {i: {} for i in range(len(pairs))}, [mp.mpf(0)] * len(pairs)
    for i, (j, k) in enumerate(pairs):
        for x in (0, 1):
            if m.P[j, x] > 0 and m.P[k, x] > 0:
                a = mp.sqrt(mp.mpf(float(m.P[j, x]))) * mp.sqrt(mp.mpf(float(m.P[k, x])))
                sj, sk = int(m.succ[j, x]), int(m.succ[k, x])
                if sj == sk:
                    b[i] += a
                else:
                    c = idx[(min(sj, sk), max(sj, sk))]
                    succs[i][c] = succs[i].get(c, mp.mpf(0)) + a
    reach = {i for i in range(len(pairs)) if b[i] > 0}
    while True:
        grown = reach | {i for i in range(len(pairs)) if any(c in reach for c in succs[i])}
        if grown == reach:
            break
        reach = grown
    R = sorted(reach)
    pos = {i: r for r, i in enumerate(R)}
    G = np.eye(n)
    if R:
        A = mp.eye(len(R))
        for i in R:
            for c, a in succs[i].items():
                if c in pos:
                    A[pos[i], pos[c]] -= a
        g = mp.lu_solve(A, mp.matrix([b[i] for i in R]))
        for i in R:
            j, k = pairs[i]
            G[j, k] = G[k, j] = float(g[pos[i]])
    return G


# =============================================================================
# F2: C_q from the exact infinite-depth overlaps
# =============================================================================

@test("F2/R2-F1/R3-F1 helper: a slow pair is certified at the predicted depth within its returned bound, or refused")
def t_slow_coalescence():
    out = []
    for eps in (0.1, 0.2, 0.5):
        G, info = M.q_overlap_limit(slow_machine(eps))
        err = abs(1.0 - G[0, 1])                                 # the exact limit is 1
        assert info["steps"] == predicted_depth(eps) and err <= info["bound"] < 1e-12, (eps, info, err)
        assert M.c_q_unifilar(slow_machine(eps))[0] < 1e-9
        out.append(f"eps={eps:g}: depth {info['steps']}, error {err:.1e} <= bound {info['bound']:.1e}")
    for eps in (1e-2, 1e-3, 307 / 2 ** 20, 1e-8, 1e-13, 1e-200):
        assert predicted_depth(eps) is None or eps >= 1e-2
        assert raises(lambda eps=eps: M.c_q_unifilar(slow_machine(eps))), f"eps={eps} was not refused"
    return "; ".join(out) + "; eps = 1e-2, 1e-3, 307/2^20, 1e-8, 1e-13, 1e-200 refused"


@test("R2-F1/R3-F1 helper: reset cycles of 2 to 5 states are certified within the returned bound of the exact limit, or refused")
def t_reset_cycles():
    worst, done = 0.0, 0
    for n in (2, 3, 4, 5):
        for eps in (0.05, 0.1, 0.3, 0.5, 2.0 ** -3, 2.0 ** -4):
            m = reset_cycle(n, eps)
            G, info = M.q_overlap_limit(m)
            err = float(np.max(np.abs(G - 1.0)))
            assert info["steps"] == predicted_depth(eps) and err <= info["bound"] < 1e-12, (n, eps, info, err)
            worst, done = max(worst, err / info["bound"]), done + 1
        for eps in (0.01, 1e-13, 2.0 ** -30):
            assert raises(lambda n=n, eps=eps: M.c_q_unifilar(reset_cycle(n, eps))), (n, eps)
    return f"{done} certified, each within its bound (largest error/bound {worst:.2f}); eps 0.01, 1e-13, 2^-30 refused"


@test("R3-F1 helper: slow pairs and random machines: every returned overlap is within its bound of the exact or 50-digit reference, or refused")
def t_bound_vs_reference():
    # Slow pairs, where accumulated rounding matters: the limit is exactly 1, so a returned
    # overlap must be within the returned bound of 1, or the computation must refuse.
    # (Version 2.2 returned a bound of 1.0e-12 with an actual error of 1.4e-12 at 307/2^20.)
    for eps in (307 / 2 ** 20, 314 / 2 ** 20, 321 / 2 ** 20, 2.0 ** -11, 1e-3, 0.01, 0.05):
        try:
            G, info = M.q_overlap_limit(slow_machine(eps))
        except ValueError:
            continue
        assert abs(1.0 - G[0, 1]) <= info["bound"], (eps, abs(1.0 - G[0, 1]), info)
    rng = np.random.default_rng(29)
    worst, certified, refused = 0.0, 0, 0
    for _ in range(40):
        m = random_machine(rng, int(rng.integers(2, 7)))
        try:
            G, info = M.q_overlap_limit(m)
        except ValueError:
            refused += 1
            continue
        err = float(np.max(np.abs(G - mp_overlap_limit(m))))
        assert err <= max(info["bound"], 1e-15), (err, info)     # bound 0 only when nothing can merge
        worst, certified = max(worst, err), certified + 1
    assert certified >= 30, (certified, refused)
    return f"{certified} certified within their bounds (largest error {worst:.1e}); {refused} refused"


@test("R2-F1 helper and driver: probabilities whose product underflows are refused, not read as orthogonal")
def t_underflow():
    m = slow_machine(1e-200)
    assert raises(lambda: M.q_overlap_limit(m)) and raises(lambda: M.c_q_unifilar(m))
    assert raises(lambda: evaluate_with_machine(m))
    return "eps = 1e-200 refused by q_overlap_limit, c_q_unifilar and evaluate"


@test("F2 helper: the certified limit agrees with the dense recursion run 2000 steps, and the recursion with explicit words")
def t_limit_vs_recursion():
    rng = np.random.default_rng(11)
    worst_lim, worst_enum, done = 0.0, 0.0, 0
    for _ in range(25):
        m = random_machine(rng, int(rng.integers(2, 8)))
        try:
            G_lim, _ = M.q_overlap_limit(m)
        except ValueError:
            continue
        curve, G_2000 = M.q_depth_curve(m, 2000)
        worst_lim = max(worst_lim, float(np.max(np.abs(G_lim - G_2000))), abs(M.q_entropy(G_lim, m.pi) - curve[-1]))
        for d in (1, 3, 6):
            worst_enum = max(worst_enum, abs(M.q_entropy(gram_by_enumeration(m, d), m.pi) - curve[d - 1]))
        done += 1
    assert done >= 20 and worst_lim < 1e-9, (done, worst_lim)
    assert worst_enum < 1e-12, worst_enum
    return f"{done} random machines: limit vs depth 2000 {worst_lim:.1e}; recursion vs enumeration {worst_enum:.1e}"


@test("F2 helper: pairs that never meet keep overlap 0 (two disjoint copies: C_q + 1 bit)")
def t_never_meet():
    gm = M.known_processes()["Golden mean"]
    two = M.Machine([[.5, .5], [0, 1], [.5, .5], [0, 1]], [[1, 0], [-1, 0], [3, 2], [-1, 2]],
                    pi=np.r_[gm.pi, gm.pi] / 2)
    a, b = M.c_q_unifilar(gm)[0], M.c_q_unifilar(two)[0]
    assert abs(b - (a + 1)) < 1e-12, (a, b)
    return f"golden mean {a:.6f}, two copies {b:.6f}"


def evaluate_with_machine(m, eps_label=""):
    """evaluate() with reconstruct() replaced by a fixed two-state machine and partition."""
    orig = M.reconstruct

    def fake(seqs, L, alpha, c_min, c_tr, clustering=None, return_partition=False):
        diag = dict(n_states=2, n_frequent_pasts=2, frequent_mass=1.0, dropped_transition_mass=0.0,
                    imputed_successor_mass=0.0, rare_successor_mass=0.0, rare_successor_mean_match=None,
                    clustering="suffix", stationarity_L1=0.0, refinement_iterations=1)
        return m, diag, {0: 0, 1: 1}, {0: float(m.pi[0]), 1: float(m.pi[1])}

    rng = np.random.default_rng(3)
    seqs = [rng.integers(0, 2, 200).astype(np.int8) for _ in range(4)]
    M.reconstruct = fake
    try:
        return M.evaluate(seqs, seqs, dict(L=1, alpha=1e-3, c_min=1, c_tr=1))
    finally:
        M.reconstruct = orig


@test("F2/R2-F1 driver: evaluate records a certified C_q for a slow pair, and refuses one too slow to certify")
def t_evaluate_slow():
    r = evaluate_with_machine(slow_machine(0.1))
    assert r["C_q"] < 1e-9 and r["q_converged"] is True and r["q_depth"] == predicted_depth(0.1) and r["q_bound"] < 1e-12, r
    assert raises(lambda: evaluate_with_machine(slow_machine(1e-8)))
    return f"eps=0.1: C_q={r['C_q']:.1e} at depth {r['q_depth']}; eps=1e-8 refused"


# =============================================================================
# F3: malformed machines, distributions and spectra raise instead of reading as zero
# =============================================================================

@test("F3 helper: Machine rejects every malformed input")
def t_machine_validation():
    good_P, good_s = [[.5, .5], [0., 1.]], [[1, 0], [-1, 0]]
    bad = {
        "NaN probability": ([[np.nan, .5], [0, 1]], good_s, None),
        "infinite probability": ([[np.inf, .5], [0, 1]], good_s, None),
        "negative probability": ([[1.2, -.2], [0, 1]], good_s, None),
        "row not summing to 1": ([[.5, .4], [0, 1]], good_s, None),
        "three symbols": ([[.3, .3, .4]], [[0, 0, 0]], None),
        "successor shape": (good_P, [[1, 0]], None),
        "float successors": (good_P, [[1.0, 0.0], [-1.0, 0.0]], None),
        "successor -1 where emitted": (good_P, [[-1, 0], [-1, 0]], None),
        "successor out of range": (good_P, [[2, 0], [-1, 0]], None),
        "occupancy NaN": (good_P, good_s, [np.nan, 1.0]),
        "occupancy negative": (good_P, good_s, [1.5, -.5]),
        "occupancy not normalised": (good_P, good_s, [.5, .4]),
        "occupancy wrong length": (good_P, good_s, [1.0]),
        "occupancy empty": (good_P, good_s, []),
        "occupancy all zero": (good_P, good_s, [0.0, 0.0]),
        "complex probability": ([[.5 + .1j, .5 - .1j], [0, 1]], good_s, None),
        "complex occupancy": (good_P, good_s, [.5 + .1j, .5 - .1j]),
        "complex probability array": (np.array([[.5 + .1j, .5 - .1j], [0, 1]]), good_s, None),
        "complex occupancy array": (good_P, good_s, np.array([.5 + .1j, .5 - .1j])),
    }
    missed = [k for k, (P, s, pi) in bad.items() if not raises(lambda: M.Machine(P, s, pi=pi))]
    assert not missed, f"accepted: {missed}"
    M.Machine(good_P, good_s)                     # the well-formed one is accepted
    return f"{len(bad)} malformed machines rejected"


@test("F3 helper: distributions and entropies reject NaN, infinity, negatives, empty and unnormalised input")
def t_distribution_validation():
    bad = [[np.nan, 1.0], [np.inf, 0.0], [1.5, -.5], [], [.5, .4], [[.5, .5]], ["a", "b"], 0.5,
           np.array([.5 + .1j, .5 - .1j])]
    missed = [b for b in bad if not raises(lambda b=b: M.shannon_bits(b))]
    assert not missed, f"accepted: {missed}"
    assert abs(M.shannon_bits([.5, .5]) - 1.0) < 1e-15
    return f"{len(bad)} malformed distributions rejected"


@test("F3 helper and driver: invalid spectra raise in q_entropy, c_q_unifilar and evaluate")
def t_spectrum_validation():
    gm = M.known_processes()["Golden mean"]
    rng = np.random.default_rng(5)
    tr = [gm.sample(2000, rng) for _ in range(20)]
    te = [gm.sample(2000, rng) for _ in range(8)]
    good = M.evaluate(tr, te, M.DEFAULT)
    spectra = {"NaN": lambda n: np.full(n, np.nan), "infinite": lambda n: np.r_[np.inf, np.zeros(n - 1)],
               "negative": lambda n: np.r_[-0.1, 1.1, np.zeros(n - 2)],
               "trace 0.9": lambda n: np.r_[0.9, np.zeros(n - 1)], "wrong size": lambda n: np.array([1.0]),
               "empty": lambda n: np.array([]), "complex": lambda n: np.r_[.5 + .1j, .5 - .1j, np.zeros(n - 2)]}
    orig = np.linalg.eigvalsh
    missed = []
    try:
        for name, make in spectra.items():
            np.linalg.eigvalsh = lambda a, make=make: make(a.shape[0])
            for layer, call in (("q_entropy", lambda: M.q_entropy(np.eye(2), [.5, .5])),
                                ("c_q_unifilar", lambda: M.c_q_unifilar(gm)),
                                ("evaluate", lambda: M.evaluate(tr, te, M.DEFAULT))):
                if not raises(call):
                    missed.append(f"{name} in {layer}")
    finally:
        np.linalg.eigvalsh = orig
    assert not missed, f"accepted: {missed}"
    again = M.evaluate(tr, te, M.DEFAULT)
    assert again["C_q"] == good["C_q"]            # the patch was undone
    return f"{len(spectra)} invalid spectra rejected in all three layers; normal saving {good['saving']:.3f}"


@test("R3-F3 helper: q_entropy rejects complex, two-dimensional and wrong-length occupancies and non-square Gram matrices")
def t_q_entropy_inputs():
    bad = [(np.eye(2), np.array([.5 + .1j, .5 - .1j])), (np.eye(2), np.array([[.5, .5]])), (np.eye(2), np.array([1.0])),
           (np.ones((2, 3)), np.array([.5, .5])), (np.eye(2) + 0j, np.array([.5, .5]))]
    missed = [i for i, (G, pi) in enumerate(bad) if not raises(lambda G=G, pi=pi: M.q_entropy(G, pi))]
    assert not missed, missed
    assert abs(M.q_entropy(np.eye(2), np.array([.5, .5])) - 1.0) < 1e-15
    return f"{len(bad)} malformed inputs rejected"


@test("F3 driver: the summaries reject a non-finite saving instead of carrying it")
def t_summary_validation():
    rows = [dict(setting=dict(L=7), C_mu=1.0, saving=0.5), dict(setting=dict(L=7), C_mu=1.0, saving=float("nan"))]
    assert raises(lambda: M.summary(rows)) and raises(lambda: M.by_L(rows))
    rows[1]["saving"] = 0.25
    assert M.summary(rows)["median"] == 0.375
    return "summary and by_L reject NaN"


# =============================================================================
# F6: exact stationary laws and exact minimisation
# =============================================================================

@test("F6/R2-F2/R3-F2/R4-F1/M10 helper: stationary laws are right on irreducible, periodic, slow, absorbing, transient, wide-range, two-peak and subnormal-exit chains")
def t_stationary():
    close = lambda a, b: np.allclose(a, b, rtol=0.0, atol=1e-15)
    gm = M.known_processes()["Golden mean"]
    assert close(gm.pi, [2 / 3, 1 / 3]), gm.pi
    p3 = M.known_processes()["Period-3 (001)"]
    assert close(p3.pi, [1 / 3] * 3), p3.pi
    for eps in (1e-16, 1e-8):
        sym = M.Machine([[1 - eps, eps], [eps, 1 - eps]], [[0, 1], [0, 1]])
        assert close(sym.pi, [.5, .5]), (eps, sym.pi)
    absorb = M.Machine([[1 - 1e-4, 1e-4], [0., 1.]], [[0, 1], [-1, 1]], pi=[1., 0.])
    assert close(M.stationary_from(absorb, [1., 0.]), [0., 1.])
    e = 2.0 ** -30
    for n in (2, 3, 4, 5):                           # a transient cycle leaking slowly into one state
        P = np.vstack([np.tile([1 - e, e], (n, 1)), [1., 0.]])
        succ = np.vstack([np.column_stack([(np.arange(n) + 1) % n, np.full(n, n)]), [n, -1]])
        pi0 = np.r_[1., np.zeros(n)]
        got = M.stationary_from(M.Machine(P, succ, pi=pi0), pi0)
        assert close(got, np.r_[np.zeros(n), 1.]), (n, got)
    split = M.Machine([[.5, .5], [1., 0.], [0., 1.]], [[1, 2], [1, -1], [-1, 2]], pi=[1 / 3] * 3)
    assert close(M.stationary_from(split, [1., 0., 0.]), [0., .5, .5])
    n = 400                                          # pi_i proportional to 7^i: GTH must not overflow
    bd = M.Machine(np.tile([1 / 8, 7 / 8], (n, 1)),
                   np.column_stack([np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)]))
    ref = 7.0 ** (np.arange(n, dtype=float) - (n - 1))
    err_bd = float(np.max(np.abs(bd.pi - ref / ref.sum())))
    assert err_bd < 1e-13, err_bd                    # GTH on logarithms: about 1e-14 here
    e2 = 2.0 ** -52                                  # two peaks joined through 2^-52 rates (R4-F1)
    tp = M.Machine(np.array([[1 - e2, e2] if i < 21 else [e2, 1 - e2] for i in range(42)]),
                   np.column_stack([np.maximum(np.arange(42) - 1, 0), np.minimum(np.arange(42) + 1, 41)]))
    assert abs(tp.pi[0] - 0.5) < 1e-13 and abs(tp.pi[-1] - 0.5) < 1e-13, (tp.pi[0], tp.pi[-1])
    for s in (1e-300, 1e-310, 1e-320, 5e-324):        # a transient state leaving at a subnormal rate (M10)
        sub = M.Machine([[1., 0.], [0., 1.], [1., s], [1e-4, 1 - 1e-4]],
                        [[0, -1], [-1, 1], [2, 3], [0, 1]], pi=[0., 0., 1., 0.])
        got = M.stationary_from(sub, [0., 0., 1., 0.])     # its mass must still split 1e-4 / 1 - 1e-4
        assert abs(got[0] - 1e-4) < 1e-15 and abs(got[1] - (1 - 1e-4)) < 1e-12, (s, got)
    e = 1e-310                                       # subnormal rates
    assert close(M.stationary_from(M.Machine([[1 - e, e], [0., 1.]], [[0, 1], [0, 1]], pi=[1., 0.]), [1., 0.]), [0., 1.])
    assert close(M.Machine([[0., 1.], [e, 1 - e]], [[0, 1], [0, 1]]).pi, [e / (1 + e), 1 / (1 + e)])
    bad = [[np.nan, 1, 0], [-1, 2, 0], [0, 0, 0], [1, 0], [.5 + .1j, .5, 0]]
    missed = [b for b in bad if not raises(lambda b=b: M.stationary_from(split, b))]
    assert not missed, missed
    return ("golden mean, period 3, slow symmetric (1e-16, 1e-8), absorbing, 4 transient cycles, two classes, "
            "a 400-state chain with weights up to 7^399 (error %.1e), two peaks joined at 2^-52, transient exits "
            "at 1e-300 to 5e-324, subnormal rates; 5 bad starts rejected" % err_bd)


@test("F6/R2-F3/R5-F1 helper: minimisation compares rows exactly, checks the occupancy's length, and counts every occupied state")
def t_minimisation():
    e = 2e-10
    near = M.Machine([[.5, .5], [.5 + e, .5 - e]], [[0, 1], [0, 1]])
    mu, k = M.minimal_cmu(near, near.pi)
    assert k == 2 and abs(mu - M.shannon_bits(near.pi)) < 1e-12, (mu, k)
    dup = M.Machine([[.5, .5], [.5, .5], [0., 1.]], [[2, 1], [2, 0], [-1, 0]])
    mu, k = M.minimal_cmu(dup, dup.pi)
    gm = M.known_processes()["Golden mean"]
    assert k == 2 and abs(mu - M.shannon_bits(gm.pi)) < 1e-12, (mu, k)
    ratio = M.Machine([[1 / 3, 2 / 3], [2 / 6, 4 / 6]], [[0, 1], [0, 1]])
    assert M.minimal_cmu(ratio, ratio.pi)[1] == 1     # equal count ratios, equal floats
    assert raises(lambda: M.minimal_cmu(gm, [.4, .4, .2])) and raises(lambda: M.minimal_cmu(gm, [1.0]))
    tiny = M.minimal_cmu(gm, [1 - 1e-14, 1e-14])            # a minimal state with occupancy 1e-14 still counts (R5-F1)
    assert tiny[1] == 2 and tiny[0] > 0, tiny
    return ("near rows kept apart; duplicates and equal ratios merged; wrong-length occupancies rejected; "
            "a state occupied at 1e-14 counted")


@test("R4-F1/M10/R4-F2/R5-F1 driver: the bootstrap's generator (generator_truth, as run() uses it) gets its stationary law, true values, state count and rows right")
def t_generator_truth():
    e2, n = 2.0 ** -52, 42                                   # R4-F1's two-peak chain
    tp = M.Machine(np.array([[1 - e2, e2] if i < 21 else [e2, 1 - e2] for i in range(n)]),
                   np.column_stack([np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)]),
                   pi=np.full(n, 1.0 / n))
    # Every returned value is asserted, the state count included (R5-F1).
    g1, cmu1, cq1, k1 = M.generator_truth(tp)
    assert abs(g1.pi[0] - 0.5) < 1e-13 and abs(g1.pi[-1] - 0.5) < 1e-13, (g1.pi[0], g1.pi[-1])
    assert np.all(g1.pi > 0), "an irreducible chain has every state occupied"
    assert abs(cmu1 - 1.0) < 1e-9 and abs(cq1 - 1.0) < 1e-9, (cmu1, cq1)   # two nearly orthogonal peaks
    assert k1 == 42, f"42 distinct, occupied minimal states, reported {k1}"
    sub = M.Machine([[1., 0.], [0., 1.], [1., 1e-320], [1e-4, 1 - 1e-4]],   # M10's subnormal exit
                    [[0, -1], [-1, 1], [2, 3], [0, 1]], pi=[0., 0., 1., 0.])
    g2, cmu2, cq2, k2 = M.generator_truth(sub)
    h = M.shannon_bits([1e-4, 1 - 1e-4])
    assert abs(g2.pi[0] - 1e-4) < 1e-15 and abs(g2.pi[1] - (1 - 1e-4)) < 1e-12 and g2.pi[2] == g2.pi[3] == 0, g2.pi
    assert abs(cmu2 - h) < 1e-12 and abs(cq2 - h) < 1e-12 and k2 == 2, (cmu2, cq2, k2)
    near = M.Machine([[.5, .5 + 2.0 ** -32], [.5, .5 + 2.0 ** -32]], [[1, 0], [0, 0]], pi=[.5, .5])
    g3, cmu3, cq3, k3 = M.generator_truth(near)             # R4-F2: the generator's rows are normalised too
    assert np.all(np.abs(g3.P.sum(axis=1) - 1.0) <= 4 * M.U), g3.P.sum(axis=1)
    p00 = g3.P[0, 0]                                        # state 0 emits 0 and moves to 1; state 1 returns to 0
    assert abs(g3.pi[0] - 1 / (1 + p00)) < 1e-15 and abs(g3.pi[1] - p00 / (1 + p00)) < 1e-15, (g3.pi, p00)
    assert cmu3 == 0.0 and cq3 < 1e-9 and k3 == 1, (cmu3, cq3, k3)   # its two states are one minimal state
    return (f"two peaks: endpoints {g1.pi[0]:.3f}, {g1.pi[-1]:.3f}, smallest weight {g1.pi.min():.1e}, true C_mu "
            f"{cmu1:.9f}, C_q {cq1:.9f}, {k1} occupied minimal states; subnormal exit: C_mu = C_q = h(1e-4) = "
            f"{h:.6f}, {k2} occupied; near-normalised: rows within 4u, C_mu 0, C_q {cq3:.1e}, {k3} minimal state")


# =============================================================================
# F1, F5, F8 and the periodicity diagnostic
# =============================================================================

@test("R4-F2 helper: rows are normalised on construction, and no overlap is clipped by more than its bound")
def t_normalisation():
    for delta in (2.0 ** -50, 2.0 ** -40, 2.0 ** -32, 1e-10):
        x = M.Machine([[.5, .5 + delta], [.5, .5 + delta]], [[1, 0], [0, 0]], pi=[.5, .5])
        assert np.all(np.abs(x.P.sum(axis=1) - 1.0) <= 4 * M.U), (delta, x.P.sum(axis=1))
        G, info = M.q_overlap_limit(x)
        assert abs(1.0 - G[0, 1]) <= info["bound"], (delta, G[0, 1], info)   # exact limit for normalised rows: 1
    bad = M.Machine([[.5, .5], [.5, .5]], [[1, 0], [0, 0]], pi=[.5, .5])
    bad.P = np.array([[.5, .5 + 2.0 ** -32], [.5, .5 + 2.0 ** -32]])        # bypass the constructor
    assert raises(lambda: M.q_overlap_limit(bad)), "an overlap above 1 + bound was clipped silently"
    assert raises(lambda: M.Machine([[.5, .5 + 1e-8]], [[0, 0]]))          # beyond 1e-9: rejected
    return "4 near-normalised rows normalised and certified; an unnormalised machine refused; 1e-8 surplus rejected"


@test("R4-F2 driver: evaluate certifies a near-normalised machine and refuses an unnormalised one at the overlap check")
def t_evaluate_normalisation():
    near = M.Machine([[.5, .5 + 2.0 ** -32], [.5, .5 + 2.0 ** -32]], [[1, 0], [0, 0]], pi=[.5, .5])
    r = evaluate_with_machine(near)
    assert r["C_q"] < 1e-9 and r["q_converged"] is True and r["q_bound"] < 1e-12, r
    bad = M.Machine([[.5, .5], [.5, .5]], [[1, 0], [0, 0]], pi=[.5, .5])
    bad.P = np.array([[.5, .5 + 2.0 ** -32], [.5, .5 + 2.0 ** -32]])        # bypass the constructor
    try:
        evaluate_with_machine(bad)
    except ValueError as e:
        assert "overlap" in str(e), f"refused, but not at the overlap check: {e}"
    else:
        raise AssertionError("evaluate accepted an overlap above 1 + its bound")
    return f"near-normalised: C_q {r['C_q']:.1e} within bound {r['q_bound']:.1e}; unnormalised: refused at the overlap check"


@test("F1: the hidden clock's exact memories are log2(10) and its overlaps the identity")
def t_hidden_clock():
    clock = M.known_processes()[M.CLOCK]
    G, info = M.q_overlap_limit(clock)
    assert np.allclose(G, np.eye(10)) and info["merging_pairs"] == 0
    assert abs(M.shannon_bits(clock.pi) - math.log2(10)) < 1e-12
    assert abs(M.c_q_unifilar(clock)[0] - math.log2(10)) < 1e-12
    return "C_mu = C_q = log2(10); no pair of phases ever merges"


@test("F5: wildcard members join the first compatible group, as the docstring now says")
def t_first_compatible():
    N = [[56, 52], [39, 3], [116, 2], [0, 34], [57, 2], [94, 1], [38, 21], [2, 44], [41, 0],
         [0, 62], [42, 31], [3, 64], [68, 2], [5, 85], [1, 25], [3, 88]]
    seqs = [np.array([(p >> j) & 1 for j in range(3, -1, -1)] + [x], dtype=np.int8)
            for p in range(16) for x in (0, 1) for _ in range(N[p][x])]
    _, _, label, _ = M.reconstruct(seqs, 4, .001, 5, 3, return_partition=True)
    assert label[9] == label[15] and label[9] != label[13], label
    return "past 9 joins the group of 15 (opened first), not the heavier group of 13 and 11"


@test("F8: rare_successor_mass counts the transitions routed through proxies")
def t_rare_successors():
    N = {0: (50, 50), 1: (50, 50), 2: (50, 50), 3: (1, 1)}       # past 3 = '11' is rare
    seqs = [np.array([(p >> 1) & 1, p & 1, x], dtype=np.int8)
            for p, cnt in N.items() for x in (0, 1) for _ in range(cnt[x])]
    _, diag = M.reconstruct(seqs, 2, 1e-3, 10, 3)
    # frequent pasts 0, 1, 2 carry 300 transitions; past 1 ('01') on symbol 1 leads to
    # the rare past 3 fifty times; its proxy shares one recent symbol
    assert abs(diag["rare_successor_mass"] - 50 / 300) < 1e-12, diag
    assert diag["rare_successor_mean_match"] == 1.0, diag
    return f"rare_successor_mass = {diag['rare_successor_mass']:.4f} (1/6), mean match 1"


@test("periodicity: exact periods are found, aperiodic sequences are not called periodic")
def t_periodicity():
    rng = np.random.default_rng(9)
    base = np.array([0, 1, 1, 0, 1, 0, 0])
    periodic = np.tile(base, 300)[:2000]
    assert M.sequence_period(periodic, 1000) == 7
    assert M.sequence_period(np.zeros(50, int), 10) == 1
    assert M.sequence_period(rng.integers(0, 2, 2000), 1000) is None
    s = M.periodicity([periodic, rng.integers(0, 2, 2000)], 1000)
    assert s["n_periodic"] == 1 and s["period_min"] == 7
    return "period 7 found; a random sequence is not periodic"


@test("compare_to_record reports moved numbers and structural changes")
def t_compare():
    rec = json.load(open(os.path.join(HERE, "psi_unifilar_results.json")))
    st, pr = M.compare_to_record(copy.deepcopy(rec))
    assert st["above_tol"] == 0 and not pr and st["numbers"] > 0
    r = copy.deepcopy(rec)
    r["validation"]["Golden mean"]["default"]["C_q"] += 1e-6
    r["validation"]["Golden mean"]["default"]["n_states"] = 2.0
    r["validation"]["Even process"].pop("by_L")
    st, pr = M.compare_to_record(r)
    assert st["above_tol"] == 1 and len(pr) == 2, (st, pr)
    return f"{st['numbers']} numbers compared; 1 moved number and 2 structural changes caught"


def main():
    print("=" * 78)
    print("COMPRESSION PIPELINE TESTS (cwf_psi_unifilar.py)")
    print("=" * 78)
    fails = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print(f"  PASS  {name}\n        {detail}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}\n        {e}")
        except Exception:
            fails += 1
            print(f"  ERROR {name}")
            traceback.print_exc()
    print("-" * 78)
    print(f"{len(TESTS) - fails}/{len(TESTS)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
