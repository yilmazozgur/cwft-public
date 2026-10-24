#!/usr/bin/env python3
"""
cwf_psi_unifilar.py  --  The compression clause, re-estimated with a unifilar
causal-state reconstruction and a validated q-machine.

Why this script exists.  A review of the unified manuscript (2026-09-06) found that
the causal-state reconstruction in cwf_psi_l2_vs_l1.py (used by cwf_psi_scaleup.py)
is not unifilar.  Pasts are clustered by their 5-step future distributions, and the
clusters are never refined so that (state, symbol) fixes the next state.  The depth
recursion  G <- sum_x A^x G (A^x)^T  used in cwf_psi_scaleup.cq_depth_curve is the
Gram recursion of the q-machine only for a unifilar machine.  On a non-unifilar
machine the diagonal of G leaves 1, and the renormalisation in that function then
changes the model.  This script leaves the committed record psi_scaleup_results.json
untouched.  It re-estimates C_mu and C_q on the SAME sequences (same seed, same draw
order) with a repaired pipeline, and it validates the pipeline first on processes
whose answers are known.

Pipeline, per process:
  1. Pasts of length L (most recent symbol = least significant bit), and next-symbol
     counts N[p, x] over the training sequences.
  2. Frequent pasts: N[p] >= c_min.
  3. Initial partition by suffix inheritance (the homogenisation step of CSSR,
     Shalizi and Klinkner 2004).  Pasts grow one symbol into the past at a time, from
     length 0 to L.  A past keeps the state of its suffix (one symbol shorter) unless
     its next-symbol counts reject that state's pooled counts in a 2x2 G-test at the
     Bonferroni-style threshold alpha / (number of frequent pasts of that length); a
     rejected past moves to the best-fitting other state, or opens a new one.  (The
     G-test p-values are asymptotic and the counts come from overlapping windows of
     single sequences, so the thresholds are heuristics, not calibrated error rates:
     audit A001, F7.)
  4. Unifilar refinement (Moore).  The successor of past p on symbol x is the past
     (p, x) shifted by one; if that past is rare (count below c_min) it is replaced by
     the frequent past sharing its longest recent history (longest common suffix,
     ties to the larger count).  A transition counts as observed when N[p, x] >= c_tr.
     Any state whose members disagree on the successor state of a symbol is split; a
     member with no observed transition on that symbol joins the first compatible
     group, in the order the groups were opened (members are visited in decreasing
     count, so each group is opened by its highest-count member; audit A001, F5).
     Iterate to a fixed point.  The code asserts that every (state, symbol)
     has at most one successor.  Under-sampled transitions still count toward the
     state's emission probabilities; a state none of whose members observed a symbol's
     transition takes the successor of its most frequent member (reported as the
     imputed successor mass).
  5. C_mu = Shannon entropy of the state occupancy.  C_q = von Neumann entropy of the
     q-machine ensemble (Gu et al. 2012; depth-unfolded as in Mahoney et al. 2016) at
     infinite depth.  Over pairs of states the Gram recursion is linear.  It is iterated
     until the weight not yet merged, plus an allowance for accumulated floating-point
     rounding, bounds every overlap's error by 1e-12 (q_overlap_limit); a pair that merges
     too slowly to reach that bound (within about 750 steps, beyond which the rounding
     allowance alone exceeds it) makes the computation raise instead of answering.  q_depth is the
     number of steps the bound needed.  (Version 2 stopped when one step changed little,
     which bounds nothing; version 2.1 solved the linear system directly, which lost slow
     merges: audit A001, F2 and R2-F1.)  Machines, distributions and spectra are
     validated before use (A001, F3 and R2-F3).
  6. Fit checks on held-out sequences: total-variation distance between the machine's
     word distribution and the empirical one (word lengths 1 to 6), and the
     stationarity of the occupancy under the machine.
  7. A conservative classical baseline, C_mu_merged.  A finite-history reconstruction
     over-splits a process of infinite Markov order: states then differ only in where
     the length-L window will lose information, their futures are nearly the same, and
     C_mu is inflated while C_q (which sees only the futures) is not.  The baseline
     coarsens the reconstructed partition: states whose pooled length-8 future-word
     counts pass a G-test of homogeneity are merged, most similar pair first, at the
     Bonferroni-style threshold alpha / (number of state pairs), a heuristic as in
     step 3.  A coarsening of the same occupancy can only lower its entropy, so
     C_mu_merged <= C_mu by construction, and the saving against it is the smaller
     one.  (Over-merging lowers it further, the conservative direction.)  Savings are
     reported against both.  The baseline is conservative relative to the
     reconstructed occupancy only: it is not a bound on the process's own minimal
     classical memory, and it does not remove every finite-history artefact (the
     hidden clock below keeps savings of 73 to 81% against it).

Version history.  Version 1 (the first record, 2026-10-21 morning) used a greedy
first-fit initial partition, treated transitions into rare pasts as unobserved and
dropped their counts, and used a second baseline built from pasts merged by their
futures without refinement.  An independent check the same day found (a) that the
greedy partition plus the wildcard fill shattered processes whose next-symbol
distributions are close (an order-2 Markov chain with 4 causal states came out with
91 to 1232 states at L = 9 to 13), (b) that dropping counts biased the emission
probabilities (perturbed coin: C_q 0.774 against 0.722 at L = 13), and (c) that the
futures-merged baseline was not a coarsening of the unifilar partition, so its memory
could exceed C_mu.  Version 2 fixes all three; the version-1 record is not kept in the
repository, so the version-1 numbers above come from its validation data and cannot be
reproduced exactly.  `clustering="greedy"` reproduces the version-1 partition step and
its count dropping, and shatters the same order-2 chain on other draws (audit A001,
control X1).  Version 2.1 (2026-10-21, after audit A001 round 1) changed no
reconstruction step.  It computed C_q at infinite depth, validated machines,
distributions and spectra, solved the bootstrap generator's stationary law directly and
minimised it with exact row equality, recorded the rare-successor share and the
periodicity of the automata's sequences, and added a zero-saving control (the hidden
clock) to the validation and to the bootstrap.  Version 2.2 (after round 2)
replaces 2.1's two direct solves, which lost small rates: the overlaps come from the
iteration with a certified bound (step 5), and stationary laws from the GTH elimination,
which uses no subtraction.  It also rejects complex input and wrong-length occupancies.
Version 2.3 (after round 3) added the rounding allowance to the overlap bound, kept the
GTH intermediates from overflowing, and validated q_entropy's occupancy.  Version 2.4
(after round 4) runs GTH on logarithms, so nothing underflows either, normalises every
machine's rows, and never clips an overlap by more than its bound.  Version 2.5 (before
round 5) moves the transient elimination in stationary_from onto logarithms
as well (audit A001, M10), and puts the bootstrap's generator in generator_truth, so the
driver can be tested.  Version 2.6 (this file, after round 5) counts every occupied
minimal state, as minimal_cmu's docstring says (R5-F1).  No reconstruction step changed
in any of these versions.

Validation.  The same pipeline, at the same data budget as the automata, on ten
processes whose epsilon-machines are known exactly, so their true C_mu and C_q can be
computed from them (the golden-mean and perturbed-coin values are also checked against
their closed forms, and the overlap recursion against explicit word enumeration in
audit A001).  Five stress
the steps that failed in version 1, or the reach of a finite history: two order-2
Markov chains (one with two causal states sharing a next-symbol distribution, so only
refinement separates them) and three processes of infinite Markov order: the even
process; a three-state machine whose two equal-emission states are told apart only by
the parity of the current run of 1s; and a ten-phase hidden clock (audit A001, F1),
whose phase advances whatever is emitted, so a finite history can only estimate it.
The clock's distinct phases never meet, so its saving is zero, yet its finite-history
reconstructions show large savings that the merged baseline does not remove, with good
held-out word fits.  A finite-history saving is therefore not evidence that the
process itself saves memory.  A parametric bootstrap then checks the automata: the
machine fitted to each headline rule at the default setting is resampled at the same
budget and re-reconstructed across the grid.  It measures recovery of the fitted
finite-history machine, so it inherits that machine's artefacts; the same bootstrap on
the hidden clock recovers the spurious saving.

Periodicity.  The record gives, for each rule, how many of its sequences are exactly
periodic over the recorded window.  A periodic sequence is deterministic once its
phase is known, so its q-machine states have orthogonal futures, and a process made of
such sequences has no quantum saving at infinite history.  Its finite-history saving
comes from the window's uncertainty about orbit and phase.

Sensitivity.  L in {7, 9, 11, 13}, alpha in {1e-3, 1e-5}, c_min in {30, 100}, on the
original data (seed 20260530) and on a fresh draw (seed 20261021).

CPU only, numpy + scipy.  Seeded.  Writes psi_unifilar_results.json.

Usage.  `python3 cwf_psi_unifilar.py` runs everything (about 6 min), writes the record and
the figure.  `--replot` redraws the figure from the record.  `--no-write` runs everything,
writes nothing, and compares the run with the committed record number by number (for a
read-only checkout; added 2026-10-21 for an audit, the computation is unchanged).
"""

import json
import math
import os
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import spsolve
from scipy.special import logsumexp
from scipy.stats import chi2

from cwf_psi_l2_vs_l1 import simulate_observation, infer_epsilon_machine, C_mu as C_mu_orig
from cwf_psi_scaleup import cq_depth_curve as cq_depth_curve_orig

HERE = os.path.dirname(os.path.abspath(__file__))

# Same data configuration as cwf_psi_scaleup.py (and the same rule order, so the
# random stream reproduces the committed sequences exactly).
DATA_CFG = dict(W=251, T=2600, burn=600, n_seqs=260, channel="column", train_frac=0.7)
RULES = [("Rule110 (IV)", 110), ("Rule54 (IV)", 54), ("Rule62 (II/IV)", 62),
         ("Rule184 (II)", 184), ("Rule94 (II)", 94), ("Rule22 (III)", 22),
         ("Rule30 (III)", 30), ("Rule90 (III/add)", 90)]
SEED_ORIG = 20260530
SEED_FRESH = 20261021
DEFAULT = dict(L=9, alpha=1e-3, c_min=30, c_tr=3)
BOOT_RULES = ("Rule184 (II)", "Rule54 (IV)", "Rule110 (IV)", "Rule22 (III)")
GRID = [dict(L=L, alpha=a, c_min=c, c_tr=3)
        for L in (7, 9, 11, 13) for a in (1e-3, 1e-5) for c in (30, 100)]
FUTURE_F = 8
MAX_DEPTH = 400                      # kept in the config record (version 2's cap)
Q_MAX_STEPS = 100_000                # hard cap on the certified overlap iteration
U = 2.0 ** -53                       # unit roundoff of IEEE double precision
ROUNDING_PER_STEP = 12 * U           # rounding allowance per overlap-iteration step (see q_overlap_limit)
WORD_K = 6
CLOCK = "Ten-phase hidden clock"     # the zero-saving control (audit A001, F1)
PERIOD_MAX = 1000                    # longest period tested in periodicity()


# ----------------------------------------------------------------------------
# Information-theory helpers
# ----------------------------------------------------------------------------
def check_distribution(p, what, tol=1e-9):
    """Validate a probability vector: one-dimensional, non-empty, finite, no entry below
    -tol, summing to 1 within tol.  Returns a float copy with tiny negatives set to 0.
    Raises ValueError otherwise (audit A001, F3)."""
    try:
        raw = np.asarray(p)
        if np.iscomplexobj(raw):
            raise ValueError(f"{what}: complex entries")
        a = np.array(raw, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what}: not a real numeric vector ({e})") from None
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"{what}: expected a non-empty vector, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{what}: non-finite entries")
    if a.min() < -tol:
        raise ValueError(f"{what}: negative entry {a.min():.3g}")
    s = float(a.sum())
    if abs(s - 1.0) > tol:
        raise ValueError(f"{what}: sums to {s:.12g}, not 1")
    return np.clip(a, 0.0, None)


def shannon_bits(p, what="distribution"):
    """Shannon entropy in bits of a validated probability vector."""
    p = check_distribution(p, what)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def gtest_2x2(a, b):
    """G-test of homogeneity for two count vectors a, b over a binary alphabet.
    Returns the p-value (chi-square, 1 dof)."""
    tab = np.array([a, b], float)
    tot = tab.sum()
    if tot <= 0:
        return 1.0
    rows = tab.sum(axis=1, keepdims=True)
    cols = tab.sum(axis=0, keepdims=True)
    exp = rows * cols / tot
    mask = tab > 0
    g = 2.0 * float(np.sum(tab[mask] * np.log(tab[mask] / exp[mask])))
    return float(chi2.sf(max(g, 0.0), 1))


# ----------------------------------------------------------------------------
# Unifilar machines: construction, sampling, and the q-machine
# ----------------------------------------------------------------------------
class Machine:
    """A unifilar machine over the binary alphabet.
    P[i, x] = Pr(symbol x | state i); succ[i, x] = next state (-1 if P[i, x] == 0)."""

    def __init__(self, P, succ, pi=None):
        # Validation (audit A001, F3): shapes, finite non-negative emission rows that sum
        # to 1, integer successors in range wherever a symbol can be emitted, and a
        # valid occupancy.  A malformed machine raises ValueError here, before any
        # entropy is computed from it.
        try:
            raw = np.asarray(P)
            if np.iscomplexobj(raw):
                raise ValueError("complex entries")
            P = np.array(raw, dtype=float)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Machine: emission matrix is not real and numeric ({e})") from None
        succ = np.array(succ)
        if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 1:
            raise ValueError(f"Machine: emission matrix must have shape (n, 2), got {P.shape}")
        if succ.shape != P.shape:
            raise ValueError(f"Machine: successor table has shape {succ.shape}, not {P.shape}")
        if not np.issubdtype(succ.dtype, np.integer):
            raise ValueError("Machine: successors must be integers")
        if not np.all(np.isfinite(P)) or P.min() < 0.0:
            raise ValueError("Machine: emission probabilities must be finite and non-negative")
        if np.max(np.abs(P.sum(axis=1) - 1.0)) > 1e-9:
            raise ValueError("Machine: every emission row must sum to 1")
        # Normalise (audit A001, R4-F2): the overlap certificate assumes rows that sum to 1,
        # and rows within 1e-9 of it were accepted unnormalised before.  A row whose sum is
        # already 1.0 in floating point is left unchanged.
        P = P / P.sum(axis=1, keepdims=True)
        n = P.shape[0]
        active = P > 0
        if np.any((succ[active] < 0) | (succ[active] >= n)):
            raise ValueError("Machine: an emitted symbol has no successor in range")
        self.P = P
        self.succ = succ.astype(int)
        self.n = n
        if pi is None:
            self.pi = self.stationary()
        else:
            self.pi = check_distribution(pi, "Machine occupancy")
            if self.pi.shape != (n,):
                raise ValueError(f"Machine: occupancy has {self.pi.size} entries for {n} states")

    def transition_matrix(self):
        Tm = np.zeros((self.n, self.n))
        for i in range(self.n):
            for x in (0, 1):
                if self.P[i, x] > 0:
                    Tm[i, self.succ[i, x]] += self.P[i, x]
        return Tm

    def stationary(self):
        """The limiting occupancy from a uniform start (unique for an irreducible
        machine, periodic or not), computed by stationary_from.  Version 2 took the
        absolute value of an eigenvector, which is fragile for reducible machines."""
        return stationary_from(self, np.full(self.n, 1.0 / self.n))

    def word_probs(self, k):
        """Probability of every length-k word (as an integer, first symbol = MSB),
        starting from self.pi."""
        probs = np.zeros(2 ** k)
        # dynamic programming over (word prefix, state)
        cur = {(0, i): self.pi[i] for i in range(self.n) if self.pi[i] > 0}
        for _ in range(k):
            nxt = defaultdict(float)
            for (w, i), pr in cur.items():
                for x in (0, 1):
                    q = self.P[i, x]
                    if q > 0:
                        nxt[((w << 1) | x, int(self.succ[i, x]))] += pr * q
            cur = nxt
        for (w, _), pr in cur.items():
            probs[w] += pr
        return probs

    def sample(self, length, rng):
        s = int(rng.choice(self.n, p=self.pi))
        out = np.empty(length, dtype=np.int8)
        for t in range(length):
            x = 1 if rng.random() < self.P[s, 1] else 0
            out[t] = x
            s = int(self.succ[s, x])
        return out


def q_entropy(G, pi, what="q-machine spectrum"):
    """Von Neumann entropy (bits) of sum_j pi_j |eta_j><eta_j| from the Gram matrix G of
    the unit vectors |eta_j>: its non-zero spectrum is that of sqrt(pi) G sqrt(pi).  The
    spectrum is validated before anything is dropped (real, finite, the right size, trace
    1, no eigenvalue below -1e-9); eigenvalues at or below 1e-14 are then dropped as in
    version 2.  The occupancy and the Gram matrix are validated first (real, matching
    sizes).  Raises ValueError on invalid input (audit A001, F3, R2-F3 and R3-F3)."""
    pi = check_distribution(pi, f"{what}: occupancy")
    G = np.asarray(G)
    if np.iscomplexobj(G) or G.ndim != 2 or G.shape != (pi.size, pi.size):
        raise ValueError(f"{what}: Gram matrix of shape {G.shape} for {pi.size} states")
    sp = np.sqrt(pi)
    w = np.linalg.eigvalsh((sp[:, None] * sp[None, :]) * G.astype(float))
    if np.iscomplexobj(w):
        raise ValueError(f"{what}: complex eigenvalues")
    w = np.asarray(w, dtype=float)
    if w.shape != pi.shape or not np.all(np.isfinite(w)):
        raise ValueError(f"{what}: not a finite spectrum of the right size")
    if abs(float(w.sum()) - 1.0) > 1e-9:
        raise ValueError(f"{what}: trace {float(w.sum()):.12g}, not 1")
    if w.min() < -1e-9:
        raise ValueError(f"{what}: negative eigenvalue {w.min():.3g}")
    w = w[w > 1e-14]
    return shannon_bits(w / w.sum(), what)


def q_overlap_limit(m, tol=1e-12, max_steps=Q_MAX_STEPS):
    """Overlaps of the depth-unfolded q-machine at infinite depth, each known to within tol,
    rounding included (audit A001, F2, R2-F1 and R3-F1).

    For a pair of states r = (j, k), j < k, and a symbol x that both emit, the Gram
    recursion moves weight sqrt(P(x|j)) sqrt(P(x|k)) to the successor pair; when the two
    successors coincide the pair has merged and the weight goes to b_r.  So the
    off-diagonal overlaps obey g(d) = A g(d-1) + b with g(0) = 0, and the limit is
    g = sum_t A^t b.  Pairs that cannot reach a merge (a graph search over the positive
    weights) keep g = 0.  On the others, the partial sum g(d) bounds g from below, and
    g(d) + A^d 1 bounds it from above, because g <= 1.  Both sequences are sums of
    non-negative terms, so nothing cancels.

    Rounding (audit A001, R3-F1).  In IEEE double arithmetic with correctly rounded
    square roots, each weight sqrt(p) sqrt(q) carries a relative error of at most 3u
    (u = 2^-53), and 4u where two symbols' weights are summed; each step adds at most two
    products and two sums, another 4u.  Machine normalises its rows, so each sums to 1
    within about 2u, and every row of weights then sums to at most 1 + 2u
    (Cauchy-Schwarz); the computed partial sums drift from the exact ones by at most
    about 8u per step, with the margin of the 12u allowance covering the rest, and the computed unmerged weight understates the exact one by a
    factor of at least (1 - 5u)^d.  The returned bound is therefore
        ROUNDING_PER_STEP * d + max(A^d 1) * exp(6 u d),  with ROUNDING_PER_STEP = 12u,
    and the iteration stops when it is below tol, so every returned overlap is within the
    bound of the exact limit.  Since the rounding term grows with d, a pair that merges
    too slowly cannot be certified: once that term alone reaches tol (about 750 steps at
    tol = 1e-12), or after max_steps, ValueError is raised.  Version 2.1 solved the linear system directly;
    on slow multi-pair cycles that returned wrong limits with zero residual, and
    sqrt(p * q) underflowed.  Version 2.2's bound left rounding out.  Version 2 stopped on a
    small step, which bounds nothing.  Returns (G, info): the steps taken, the final bound
    and the number of pairs that can merge."""
    n = m.n
    if n == 1:
        return np.ones((1, 1)), dict(pairs=0, merging_pairs=0, steps=0, bound=0.0)
    J, K = np.triu_indices(n, 1)
    npairs = J.size
    idx = np.full((n, n), -1, dtype=np.int64)
    idx[J, K] = np.arange(npairs)
    idx[K, J] = idx[J, K]
    root = np.sqrt(m.P)            # roots first, so sqrt(p) * sqrt(q) cannot underflow to 0
    b = np.zeros(npairs)
    rows, cols, vals = [], [], []
    for x in (0, 1):
        live = np.flatnonzero((m.P[J, x] > 0) & (m.P[K, x] > 0))
        a = root[J[live], x] * root[K[live], x]
        sj, sk = m.succ[J[live], x], m.succ[K[live], x]
        same = sj == sk
        np.add.at(b, live[same], a[same])
        rows.append(live[~same])
        cols.append(idx[sj[~same], sk[~same]])
        vals.append(a[~same])
    rows, cols, vals = (np.concatenate(v) for v in (rows, cols, vals))
    if np.any(vals <= 0.0):
        raise ValueError("q-machine overlaps: a transition both states can make has zero weight")
    merging = np.flatnonzero(b > 0)
    rev = csr_matrix((np.ones(cols.size + merging.size),
                      (np.concatenate([cols, np.full(merging.size, npairs)]),
                       np.concatenate([rows, merging]))), shape=(npairs + 1, npairs + 1))
    seen = breadth_first_order(rev, npairs, directed=True, return_predecessors=False)
    R = np.sort(seen[seen < npairs])
    g = np.zeros(npairs)
    steps, bound = 0, 0.0
    if R.size:
        A = csr_matrix((vals, (rows, cols)), shape=(npairs, npairs))[R][:, R].tocsr()
        bR = b[R]
        low = np.zeros(R.size)
        mass = np.ones(R.size)
        while True:
            low = A @ low + bR        # g(d): weight merged within d steps
            mass = A @ mass           # A^d 1: weight not yet merged nor lost
            steps += 1
            rounding = ROUNDING_PER_STEP * steps
            bound = rounding + float(mass.max()) * math.exp(6 * U * steps)
            if bound < tol:
                break
            if rounding >= tol or steps >= max_steps:
                raise ValueError(f"q-machine overlap limit not certified after {steps} steps "
                                 f"(unmerged weight {float(mass.max()):.3g}, rounding allowance "
                                 f"{rounding:.3g}, tolerance {tol:g}): a pair merges too slowly")
        g[R] = low
    if not np.all(np.isfinite(g)) or g.min() < 0.0 or g.max() > 1.0 + bound:
        raise ValueError(f"q-machine overlaps out of range [{g.min():.3g}, {g.max():.3g}] "
                         f"(an overlap may exceed 1 by at most its bound, {bound:.3g})")
    g = np.minimum(g, 1.0)                        # moves no overlap by more than the bound
    G = np.eye(n)
    G[J, K] = g
    G[K, J] = g
    return G, dict(pairs=int(npairs), merging_pairs=int(R.size), steps=int(steps), bound=bound)


def q_depth_curve(m, depth):
    """C_q of the depth-d q-machine for d = 1..depth, from the dense Gram recursion
    G(d) = sum_x A^x G(d-1) (A^x)^T, G(0) = I, A^x[i, succ(i, x)] = sqrt(P(x|i)).  For a
    unifilar machine every state stays a unit vector and G grows entrywise; both are
    asserted at every depth.  Returns (curve, G(depth))."""
    n = m.n
    A = [np.zeros((n, n)), np.zeros((n, n))]
    for i in range(n):
        for x in (0, 1):
            if m.P[i, x] > 0:
                A[x][i, m.succ[i, x]] = math.sqrt(m.P[i, x])
    G = np.eye(n)
    curve = []
    for _ in range(depth):
        Gn = A[0] @ G @ A[0].T + A[1] @ G @ A[1].T
        assert np.max(np.abs(np.diag(Gn) - 1.0)) < 1e-9, "q-machine state left the unit sphere"
        assert np.min(Gn - G) > -1e-9, "Gram recursion not monotone (non-unifilar input?)"
        G = Gn
        curve.append(q_entropy(G, m.pi))
    return curve, G


def c_q_unifilar(m, tol=1e-12, max_steps=Q_MAX_STEPS, return_info=False, with_curve=False):
    """Quantum statistical complexity C_q of the depth-unfolded q-machine of a unifilar
    machine at infinite depth: the von Neumann entropy of its ensemble, from overlaps
    known to within tol (q_overlap_limit, which raises if it can't certify them).  `depth`
    is the number of recursion steps the bound needed.  With with_curve, `curve` holds
    C_q at depths 1..depth from the dense recursion (q_depth_curve), whose last Gram
    matrix is checked against the certified one.  Returns (C_q, depth, curve), plus the
    info dict when return_info is True."""
    G_lim, info = q_overlap_limit(m, tol=tol, max_steps=max_steps)
    cq = q_entropy(G_lim, m.pi)
    curve = []
    if with_curve and info["steps"] > 0:
        curve, G_d = q_depth_curve(m, info["steps"])
        gap = float(np.max(np.abs(G_d - G_lim)))
        assert gap <= max(tol, 1e-12) + 1e-12, f"dense recursion disagrees with the bound ({gap:.3g})"
        info["dense_gap"] = gap
    return (cq, info["steps"], curve, info) if return_info else (cq, info["steps"], curve)


# ----------------------------------------------------------------------------
# Reconstruction from data (the repaired pipeline)
# ----------------------------------------------------------------------------
def past_codes(seq, L):
    """Integer code of the length-L past ending just before position t, for every
    t in [L, len(seq)).  Most recent symbol = least significant bit."""
    s = np.asarray(seq, np.int64)
    n = len(s)
    codes = np.zeros(n - L, np.int64)
    for k in range(L):
        codes = (codes << 1) | s[k:n - L + k]
    return codes


def count_transitions(seqs, L):
    """N[p, x]: number of times past p (length L) is followed by symbol x."""
    N = np.zeros((2 ** L, 2), np.int64)
    for seq in seqs:
        s = np.asarray(seq, np.int64)
        if len(s) <= L:
            continue
        codes = past_codes(s, L)
        nxt = s[L:]
        np.add.at(N, (codes, nxt), 1)
    return N


def homogenize_greedy(N, order, level):
    """Version-1 initial partition (kept for the record of the comparison): greedy
    first fit in order of decreasing count; a past joins the first cluster whose
    pooled counts pass the G-test at `level`.  An independent check (2026-10-21) found
    that this misassigns low-count pasts when two next-symbol distributions are close,
    and that the Moore refinement then shatters the partition (an order-2 Markov chain
    with 4 causal states came out with 91 to 1232 states at L = 9 to 13)."""
    label, pooled = {}, []
    for p in order:
        for c, cnt in enumerate(pooled):
            if gtest_2x2(N[p], cnt) > level:
                label[p] = c
                pooled[c] = cnt + N[p]
                break
        else:
            label[p] = len(pooled)
            pooled.append(N[p].copy())
    return label


def homogenize_suffix(seqs_train, L, alpha, c_min):
    """Initial partition by suffix inheritance (the homogenisation step of CSSR,
    Shalizi and Klinkner 2004).  Pasts are grown one symbol into the past at a time,
    from length 0 to L.  A past keeps the state of its suffix (the past one symbol
    shorter) unless its next-symbol counts reject that state's pooled counts in a G-test
    at the Bonferroni level alpha / (number of frequent pasts of that length).  A
    rejected past moves to the best-fitting other state if one passes, or else opens a
    new state.  A rare past therefore inherits the state of its well-sampled suffix
    instead of being matched on its own few counts.  Returns the labels of the
    frequent length-L pasts."""
    lab_prev = {0: 0}                               # length-0 past: the empty past
    ref = [count_transitions(seqs_train, 0)[0].astype(float)]
    for k in range(1, L + 1):
        Nk = count_transitions(seqs_train, k)
        totk = Nk.sum(axis=1)
        freq_k = np.flatnonzero(totk >= c_min)
        level_k = alpha / max(len(freq_k), 1)
        lab = {}
        new_ref = defaultdict(lambda: np.zeros(2))
        n_old = len(ref)
        for p in sorted(freq_k.tolist(), key=lambda p: (-int(totk[p]), p)):
            parent = p & ((1 << (k - 1)) - 1)       # the k-1 most recent symbols
            s = lab_prev.get(parent)
            if s is not None and gtest_2x2(Nk[p], ref[s]) > level_k:
                lab[p] = s
            else:
                best, best_pv = None, level_k
                cands = list(range(len(ref)))
                for j in cands:
                    if j == s:
                        continue
                    r = ref[j] if j < n_old else new_ref[j]
                    pv = gtest_2x2(Nk[p], r)
                    if pv > best_pv:
                        best, best_pv = j, pv
                if best is None:
                    best = len(ref)
                    ref.append(np.zeros(2))
                lab[p] = best
            new_ref[lab[p]] = new_ref[lab[p]] + Nk[p]
        # reference counts for the next length: pooled over this length's members
        for j in range(len(ref)):
            if j in new_ref:
                ref[j] = new_ref[j]
        lab_prev = lab
    # compact the labels of the length-L frequent pasts
    ids = {}
    return {p: ids.setdefault(s, len(ids)) for p, s in sorted(lab_prev.items())}


CLUSTERING = "suffix"     # "suffix" (version 2, used for the record) or "greedy" (version 1)


def reconstruct(seqs_train, L, alpha, c_min, c_tr, clustering=None, return_partition=False):
    """Unifilar causal-state reconstruction.  Returns (Machine, diagnostics), plus the
    partition (label, occupancy weight per frequent past) when return_partition."""
    clustering = clustering or CLUSTERING
    N = count_transitions(seqs_train, L)
    mask = (1 << L) - 1
    tot = N.sum(axis=1)
    freq = np.flatnonzero(tot >= c_min)
    if len(freq) == 0:
        if return_partition:
            return None, {"note": "no frequent pasts"}, None, None
        return None, {"note": "no frequent pasts"}
    is_freq = np.zeros(2 ** L, bool)
    is_freq[freq] = True
    order = sorted(freq.tolist(), key=lambda p: (-int(tot[p]), p))
    level = alpha / max(len(freq), 1)

    # 3. initial partition by next-symbol distribution
    if clustering == "greedy":
        label = homogenize_greedy(N, order, level)
    else:
        label = homogenize_suffix(seqs_train, L, alpha, c_min)
        assert set(label) == set(freq.tolist()), "partition does not cover the frequent pasts"

    # Successor pasts.  Version 1 used the literal successor past and treated a
    # transition into a rare (non-frequent) past as unobserved, a wildcard in the
    # refinement, and dropped its counts.  At long L most transitions end in rare
    # pasts, the greedy wildcard fill then assigns members inconsistently, and the
    # refinement shatters the partition.  Version 2 (clustering == "suffix") maps a rare
    # successor past to the frequent past that shares its longest recent history (the
    # longest common suffix; ties to the larger count), as CSSR does with suffixes.
    proxy = np.arange(2 ** L)
    match_len = np.full(2 ** L, L)            # length of the shared recent history
    if clustering != "greedy":
        by_suffix = [dict() for _ in range(L + 1)]
        for q in order:                       # decreasing count: first seen = best
            for j in range(L + 1):
                by_suffix[j].setdefault(q & ((1 << j) - 1), q)
        for q in range(2 ** L):
            if not is_freq[q]:
                for j in range(L - 1, -1, -1):
                    hit = by_suffix[j].get(q & ((1 << j) - 1))
                    if hit is not None:
                        proxy[q] = hit
                        match_len[q] = j
                        break

    def succ_past(p, x):
        return int(proxy[((p << 1) | x) & mask])

    def observed(p, x):
        return N[p, x] >= c_tr and is_freq[succ_past(p, x)]

    # 4. unifilar (Moore) refinement with a wildcard for unobserved transitions
    n_iter = 0
    while True:
        n_iter += 1
        groups = defaultdict(list)
        for p in order:
            groups[label[p]].append(p)
        new_label = {}
        nid = 0
        split = False
        for cid in sorted(groups):
            members = groups[cid]          # already in decreasing-count order
            subs = []                      # list of [signature(list of 2), members]
            for p in members:
                sig = [label[succ_past(p, x)] if observed(p, x) else None for x in (0, 1)]
                for g in subs:
                    if all(g[0][x] is None or sig[x] is None or g[0][x] == sig[x] for x in (0, 1)):
                        for x in (0, 1):
                            if g[0][x] is None:
                                g[0][x] = sig[x]
                        g[1].append(p)
                        break
                else:
                    subs.append([sig, [p]])
            if len(subs) > 1:
                split = True
            for g in subs:
                for p in g[1]:
                    new_label[p] = nid
                nid += 1
        label = new_label
        if not split:
            break
        if n_iter > 4 * len(freq) + 10:
            raise RuntimeError("refinement did not converge")

    n = 1 + max(label.values())
    Pc = np.zeros((n, 2))
    succ = -np.ones((n, 2), int)
    occ = np.zeros(n)
    dropped = 0.0
    imputed = 0.0
    best_member = {}
    for p in order:
        i = label[p]
        occ[i] += tot[p]
        for x in (0, 1):
            if N[p, x] == 0:
                continue
            if observed(p, x):
                j = label[succ_past(p, x)]
                assert succ[i, x] in (-1, j), "non-unifilar after refinement"
                succ[i, x] = j
                Pc[i, x] += N[p, x]
            elif clustering == "greedy":
                dropped += N[p, x]
            else:
                # version 2: an under-sampled transition still counts toward the state's
                # emission probability; its successor is the state's observed one.
                Pc[i, x] += N[p, x]
                if (i, x) not in best_member:
                    best_member[(i, x)] = p      # most frequent member (order is by count)
    for (i, x), p in best_member.items():
        if succ[i, x] == -1:                    # no member observed this transition
            succ[i, x] = label[succ_past(p, x)]
            imputed += Pc[i, x]
    rows = Pc.sum(axis=1)
    assert np.all(rows > 0), "a reconstructed state has no observed transition"
    Pc = Pc / rows[:, None]
    pi_occ = occ / occ.sum()
    m = Machine(Pc, succ, pi=pi_occ)
    # stationarity of the occupancy under the machine: || pi T - pi ||_1.  (Valid for
    # reducible machines too, where the stationary distribution is not unique.)
    stat_l1 = float(np.abs(pi_occ @ m.transition_matrix() - pi_occ).sum())
    # Rare successors (audit A001, F8): the share of the retained transition counts whose
    # literal successor past is rare.  Version 2 replaces each such past by a proxy (the
    # frequent past sharing its longest recent history) and treats the transition as
    # observed when N[p, x] >= c_tr, so imputed_successor_mass does not count them.  The
    # greedy mode drops them instead (dropped_transition_mass).
    rare_mass, rare_len = 0.0, 0.0
    for p in order:
        for x in (0, 1):
            if N[p, x] > 0:
                q = ((p << 1) | x) & mask
                if not is_freq[q]:
                    rare_mass += N[p, x]
                    rare_len += N[p, x] * match_len[q]
    diag = dict(n_states=int(n), n_frequent_pasts=int(len(freq)),
                frequent_mass=float(tot[freq].sum() / tot.sum()),
                dropped_transition_mass=float(dropped / tot[freq].sum()),
                imputed_successor_mass=float(imputed / tot[freq].sum()),
                rare_successor_mass=float(rare_mass / tot[freq].sum()),
                rare_successor_mean_match=(float(rare_len / rare_mass)
                                           if rare_mass > 0 and clustering != "greedy" else None),
                clustering=clustering,
                stationarity_L1=stat_l1,
                refinement_iterations=int(n_iter))
    if return_partition:
        return m, diag, label, {p: float(tot[p]) for p in label}
    return m, diag


def gtest_sparse(a, b):
    """G-test of homogeneity between two sparse count dicts (category -> count).
    Returns the p-value (chi-square with (number of categories - 1) dof)."""
    cats = set(a) | set(b)
    if len(cats) <= 1:
        return 1.0
    na, nb = float(sum(a.values())), float(sum(b.values()))
    tot = na + nb
    g = 0.0
    for c in cats:
        oa, ob = a.get(c, 0), b.get(c, 0)
        col = oa + ob
        if oa > 0:
            g += oa * math.log(oa / (na * col / tot))
        if ob > 0:
            g += ob * math.log(ob / (nb * col / tot))
    return float(chi2.sf(max(2.0 * g, 0.0), len(cats) - 1))


def future_word_counts(seqs_train, L, F):
    """Dense counts C[p, w] of the length-F future word w following length-L past p."""
    C = np.zeros((2 ** L, 2 ** F))
    for seq in seqs_train:
        s_ = np.asarray(seq, np.int64)
        n = len(s_)
        if n < L + F:
            continue
        pc = past_codes(s_, L)[: n - L - F + 1]
        fc = np.zeros(n - L - F + 1, np.int64)
        for k in range(F):
            fc = (fc << 1) | s_[L + k: n - F + 1 + k]
        np.add.at(C, (pc, fc), 1)
    return C


def _gtest_rows(a, B):
    """G-test of homogeneity of count vector a against each row of B (dense, same
    categories).  Returns p-values, one per row."""
    na = a.sum()
    nb = B.sum(axis=1)
    col = a[None, :] + B
    tot = na + nb
    with np.errstate(divide="ignore", invalid="ignore"):
        ea = na * col / tot[:, None]
        eb = nb[:, None] * col / tot[:, None]
        ta = np.where(a[None, :] > 0, a[None, :] * np.log(a[None, :] / ea), 0.0)
        tb = np.where(B > 0, B * np.log(B / eb), 0.0)
    g = 2.0 * (ta.sum(axis=1) + tb.sum(axis=1))
    df = (col > 0).sum(axis=1) - 1
    pv = chi2.sf(np.maximum(g, 0.0), np.maximum(df, 1))
    pv[df <= 0] = 1.0
    return pv


def merged_cmu(seqs_train, L, label, weight, F, alpha):
    """Conservative classical baseline (step 7): coarsen the reconstructed partition.
    label: frequent length-L past -> state; weight: past -> occupancy count (the same
    counts that give C_mu).  States whose pooled length-F future-word counts pass the
    G-test are merged, most similar pair first, at the Bonferroni level
    alpha / (number of state pairs).  Returns (C_mu_merged, number of merged states)."""
    n = 1 + max(label.values())
    C = future_word_counts(seqs_train, L, F)
    S_ = np.zeros((n, 2 ** F))
    w = np.zeros(n)
    for p, i in label.items():
        S_[i] += C[p]
        w[i] += weight[p]
    if n == 1:
        return shannon_bits(w / w.sum()), 1
    level = alpha / (n * (n - 1) / 2.0)
    active = np.ones(n, bool)
    P = np.full((n, n), -1.0)
    for i in range(n - 1):
        P[i, i + 1:] = _gtest_rows(S_[i], S_[i + 1:])
    P = np.maximum(P, P.T)
    np.fill_diagonal(P, -1.0)
    while True:
        k = int(np.argmax(P))
        i, j = divmod(k, n)
        if P[i, j] <= level:
            break
        S_[i] += S_[j]
        w[i] += w[j]
        active[j] = False
        P[j, :] = -1.0
        P[:, j] = -1.0
        idx = np.flatnonzero(active & (np.arange(n) != i))
        if len(idx):
            pv = _gtest_rows(S_[i], S_[idx])
            P[i, idx] = pv
            P[idx, i] = pv
    ww = w[active]
    return shannon_bits(ww / ww.sum()), int(active.sum())


def empirical_word_probs(seqs, k):
    counts = np.zeros(2 ** k)
    for seq in seqs:
        s = np.asarray(seq, np.int64)
        if len(s) < k:
            continue
        codes = np.zeros(len(s) - k + 1, np.int64)
        for j in range(k):
            codes = (codes << 1) | s[j:len(s) - k + 1 + j]
        counts += np.bincount(codes, minlength=2 ** k)
    return counts / counts.sum()


def evaluate(seqs_train, seqs_test, setting):
    m, diag, label, weight = reconstruct(seqs_train, setting["L"], setting["alpha"],
                                         setting["c_min"], setting["c_tr"],
                                         return_partition=True)
    if m is None:
        return dict(setting=setting, **diag)
    cmu = shannon_bits(m.pi)
    cq, depth, _, qinfo = c_q_unifilar(m, return_info=True)
    tv = []
    for k in range(1, WORD_K + 1):
        tv.append(0.5 * float(np.abs(m.word_probs(k) - empirical_word_probs(seqs_test, k)).sum()))
    cmu_m, n_m = merged_cmu(seqs_train, setting["L"], label, weight, FUTURE_F,
                            setting["alpha"])
    assert cmu_m <= cmu + 1e-9, "a coarsening raised the occupancy entropy"
    return dict(setting=setting, C_mu=cmu, C_q=cq,
                saving=(1.0 - cq / cmu) if cmu > 1e-9 else None,
                C_mu_merged=cmu_m, n_merged_states=n_m,
                saving_vs_merged=(1.0 - cq / cmu_m) if cmu_m > 1e-9 else None,
                # q_depth: recursion steps until every overlap is known to within 1e-12
                # (q_bound, the largest unmerged weight); an uncertified limit raises, so
                # q_converged is always True in a written record.
                q_depth=depth, q_converged=True, q_bound=qinfo["bound"],
                word_TV=tv, **diag)


# ----------------------------------------------------------------------------
# Validation processes with known answers
# ----------------------------------------------------------------------------
def order2_chain(p1):
    """Order-2 binary Markov chain: state = last two symbols (2 * older + newer),
    P(1 | state) = p1[state]."""
    P, succ = [], []
    for st in range(4):
        newer = st & 1
        P.append([1.0 - p1[st], p1[st]])
        succ.append([(newer << 1) | 0, (newer << 1) | 1])
    return Machine(P, succ)


def known_processes():
    p = 0.1
    return {
        "IID fair coin": Machine([[0.5, 0.5]], [[0, 0]]),
        "IID biased 0.15": Machine([[0.85, 0.15]], [[0, 0]]),
        "Golden mean": Machine([[0.5, 0.5], [0.0, 1.0]], [[1, 0], [-1, 0]]),
        "Even process": Machine([[0.5, 0.5], [0.0, 1.0]], [[0, 1], [-1, 0]]),
        "Perturbed coin p=0.1": Machine([[1 - p, p], [p, 1 - p]], [[0, 1], [0, 1]]),
        "Period-3 (001)": Machine([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                                  [[1, -1], [2, -1], [-1, 0]]),
        # stress tests added in version 2 (see the version history above)
        "Order-2, shared emission": order2_chain([0.3, 0.3, 0.8, 0.6]),
        "Order-2, distinct emissions": order2_chain([0.1, 0.4, 0.8, 0.6]),
        "Three-state parity machine": Machine([[0.7, 0.3], [0.7, 0.3], [0.2, 0.8]],
                                              [[0, 1], [2, 0], [0, 1]]),
        # zero-saving control added in version 2.1 (audit A001, F1); last, so the other
        # processes' validation draws are unchanged
        "Ten-phase hidden clock": hidden_clock(),
    }


def hidden_clock():
    """Zero-saving control of infinite Markov order (audit A001, F1).  Ten phases are
    visited in a fixed cycle whatever is emitted, with P(1 | phase) below.  Distinct
    phases never meet, so the q-machine states stay orthogonal and C_q = C_mu = log2(10):
    the saving is zero.  A finite history can only estimate the phase, so a
    finite-history reconstruction over-splits it into many overlapping states."""
    p = np.array([.1, .8, .2, .9, .3, .6, .15, .7, .4, .85])
    succ = np.repeat(((np.arange(10) + 1) % 10)[:, None], 2, axis=1)
    return Machine(np.column_stack([1 - p, p]), succ, pi=np.full(10, 0.1))


def minimal_partition(m):
    """Moore minimisation of a unifilar machine: refine the partition by emission row
    and successor classes to a fixed point.  Emission rows are compared exactly, as
    floats.  For machines built from count ratios, equal ratios give equal floats;
    version 2 rounded rows to a 1e-9 grid, which could merge distinct states (audit
    A001, F6).  Returns the class label of every state."""
    key = {}
    cls = [key.setdefault(tuple(m.P[i].tolist()), len(key)) for i in range(m.n)]
    while True:
        sig = {}
        new = [sig.setdefault((cls[i],) + tuple(cls[m.succ[i, x]] if m.P[i, x] > 0 else -1
                                                for x in (0, 1)), len(sig))
               for i in range(m.n)]
        if len(sig) == len(set(cls)):
            return cls
        cls = new


def minimal_cmu(m, pi):
    """C_mu of the minimal machine (the occupancy pi aggregated over the classes of
    minimal_partition) and its number of occupied states: every minimal state with
    positive occupancy, however small (before version 2.6 the count left out states
    below 1e-12; audit A001, R5-F1).  Used to give the true C_mu of a fitted machine run
    as a generator, which may itself contain states with identical futures."""
    pi = check_distribution(pi, "occupancy for minimisation")
    if pi.shape != (m.n,):
        raise ValueError(f"minimal_cmu: occupancy has {pi.size} entries for {m.n} states")
    cls = minimal_partition(m)
    agg = np.zeros(1 + max(cls))
    for i in range(m.n):
        agg[cls[i]] += pi[i]
    return shannon_bits(agg / agg.sum()), int((agg > 0).sum())


def _gth(T):
    """Stationary law of an irreducible stochastic matrix by the Grassmann-Taksar-Heyman
    elimination (Grassmann, Taksar and Heyman 1985), carried out on logarithms.  The
    elimination never forms 1 - p, so small transition rates keep their relative
    accuracy, and on logarithms no intermediate can overflow or underflow: version 2.3
    rescaled the back-substitution in linear arithmetic, and a product there could
    underflow and silently drop a whole component (audit A001, R4-F1); version 2.2 could
    overflow (R3-F2)."""
    with np.errstate(divide="ignore"):
        LP = np.log(np.array(T, dtype=float))             # -inf where a rate is 0
    n = LP.shape[0]
    log_out = np.zeros(n)
    for k in range(n - 1, 0, -1):
        lo = logsumexp(LP[k, :k])                          # log of k's outflow to states < k
        if not np.isfinite(lo):
            raise ValueError("stationary law: the class is not irreducible")
        log_out[k] = lo
        # censored chain on states < k: P[i, j] += P[i, k] * P[k, j] / out_k
        LP[:k, :k] = np.logaddexp(LP[:k, :k], LP[:k, k][:, None] + (LP[k, :k] - lo)[None, :])
    logpi = np.full(n, -np.inf)
    logpi[0] = 0.0
    for k in range(1, n):
        logpi[k] = logsumexp(logpi[:k] + LP[:k, k]) - log_out[k]
    return np.exp(logpi - logsumexp(logpi))


def stationary_from(m, pi0, tol=1e-12):
    """The limiting occupancy reached from pi0 (the Cesaro limit of pi0 T^t), computed
    without subtraction (audit A001, F6 and R2-F2).  Closed classes are the strongly
    connected components with no exit.  pi0's mass on transient states is carried into
    them by eliminating the transient states one at a time, GTH-style: each state's
    outflow is the sum of its remaining off-diagonal and absorbing rates, never 1 minus
    its self-loop.  Each class's stationary law then comes from _gth.  Valid for
    reducible and periodic machines.  The result is checked: a distribution, stationary
    to tol in L1.  (Version 2.1 solved I - T directly, which lost slow rates; version 2
    averaged 4000 steps.)"""
    raw = np.asarray(pi0)
    if np.iscomplexobj(raw):
        raise ValueError("stationary_from: complex initial occupancy")
    pi0 = np.array(raw, dtype=float)
    if pi0.shape != (m.n,) or not np.all(np.isfinite(pi0)) or pi0.min() < 0 or not pi0.sum() > 0:
        raise ValueError("stationary_from: the initial occupancy must be finite, "
                         "non-negative, of the right size, and not all zero")
    pi0 = pi0 / pi0.sum()
    T = m.transition_matrix()
    k, lab = connected_components(csr_matrix(T > 0), directed=True, connection="strong")
    closed = [ids for ids in (np.flatnonzero(lab == c) for c in range(k))
              if not np.any(T[np.ix_(ids, np.flatnonzero(lab != lab[ids[0]]))] > 0)]
    in_closed = np.zeros(m.n, bool)
    for ids in closed:
        in_closed[ids] = True
    weights = np.array([pi0[ids].sum() for ids in closed])
    trans = np.flatnonzero(~in_closed)
    if trans.size and pi0[trans].sum() > 0:
        # On logarithms, as in _gth (audit A001, M10).  In linear arithmetic a rate times a
        # proportion could underflow before the later division by an outflow that small:
        # a transient state leaving through a rate of 1e-320 sent all its mass to one class.
        Q = T[np.ix_(trans, trans)].copy()
        np.fill_diagonal(Q, 0.0)                       # a self-loop only delays leaving
        Rk = np.column_stack([T[np.ix_(trans, ids)].sum(axis=1) for ids in closed])
        with np.errstate(divide="ignore"):             # log(0) = -inf marks a zero rate or mass
            LQ, LR = np.log(Q), np.log(Rk)
            LS, LW = np.log(pi0[trans]), np.log(weights)
        for j in range(trans.size - 1, -1, -1):        # eliminate transient state j
            lo = logsumexp(np.concatenate([LQ[j, :j], LR[j]]))     # log of j's outflow
            if not np.isfinite(lo):
                raise ValueError("stationary_from: a transient state never leaves")
            lq, lr = LQ[j, :j] - lo, LR[j] - lo                     # log proportions
            LQ[:j, :j] = np.logaddexp(LQ[:j, :j], LQ[:j, j][:, None] + lq[None, :])
            LR[:j] = np.logaddexp(LR[:j], LQ[:j, j][:, None] + lr[None, :])
            LS[:j] = np.logaddexp(LS[:j], LS[j] + lq)
            LW = np.logaddexp(LW, LS[j] + lr)
        weights = np.exp(LW)
    result = np.zeros(m.n)
    for w, ids in zip(weights, closed):
        if w > 0:
            result[ids] = w * _gth(T[np.ix_(ids, ids)])
    result = check_distribution(result, "stationary occupancy")
    resid = float(np.abs(result @ T - result).sum())
    if resid > tol:
        raise ValueError(f"stationary_from: residual {resid:.3g} exceeds {tol:g}")
    return result


def generator_truth(m_fit):
    """The parametric bootstrap's generator and its true values, as run() uses them.  The
    generator is the fitted machine started from its stationary law, which stationary_from
    reaches from the fitted occupancy.  Its true C_mu is that of its minimisation (it may
    hold duplicate states), and its true C_q comes from c_q_unifilar.  Returns (generator,
    C_mu, C_q, number of occupied minimal states, as minimal_cmu counts them).  (Before
    version 2.5 these steps were written out twice inside run(), where no test reached
    them.)"""
    gen = Machine(m_fit.P, m_fit.succ, pi=stationary_from(m_fit, m_fit.pi))
    cq = c_q_unifilar(gen, with_curve=False)[0]
    cmu, n_min = minimal_cmu(gen, gen.pi)
    return gen, cmu, cq, n_min


def sequence_period(s, pmax):
    """Smallest p <= pmax with s[t] = s[t + p] over the whole sequence, or None."""
    s = np.asarray(s)
    for p in range(1, min(pmax, len(s) - 1) + 1):
        if np.array_equal(s[p:], s[:-p]):
            return p
    return None


def periodicity(seqs, pmax=1000):
    """How many sequences are exactly periodic over the recorded window, with the
    smallest, median and largest period.  A periodic sequence is deterministic once its
    phase is known, so its q-machine states have orthogonal futures."""
    per = [sequence_period(s, pmax) for s in seqs]
    found = [q for q in per if q is not None]
    return dict(n_sequences=len(per), n_periodic=len(found), max_period_tested=pmax,
                period_min=min(found) if found else None,
                period_median=float(np.median(found)) if found else None,
                period_max=max(found) if found else None)


def closed_forms():
    h = lambda q: shannon_bits([q, 1 - q])
    # golden mean: rho = [[2/3, 1/3], [1/3, 1/3]]; perturbed coin: overlap 2 sqrt(p(1-p)) = 0.6
    gm = np.linalg.eigvalsh(np.array([[2 / 3, 1 / 3], [1 / 3, 1 / 3]]))
    return {"Golden mean": dict(C_mu=h(1 / 3), C_q=shannon_bits(gm)),
            "Perturbed coin p=0.1": dict(C_mu=1.0, C_q=h(0.8))}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def automaton_data(seed):
    """Regenerate the automaton column sequences exactly as cwf_psi_scaleup.py does
    (same configuration, same rule order, one random stream)."""
    rng = np.random.default_rng(seed)
    data = {}
    for name, rule in RULES:
        seqs = simulate_observation(rule, DATA_CFG["W"], DATA_CFG["T"], DATA_CFG["n_seqs"],
                                    DATA_CFG["burn"], rng, channel=DATA_CFG["channel"])
        ntr = int(DATA_CFG["train_frac"] * len(seqs))
        data[name] = (rule, seqs[:ntr], seqs[ntr:])
    return data


def _finite_values(rows, key):
    """The non-None values of `key` across rows; a non-finite one raises ValueError
    (audit A001, F3: the summaries must not carry an invalid saving forward)."""
    s = [r[key] for r in rows if r.get(key) is not None]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in s):
        raise ValueError(f"summary: non-finite or non-numeric {key}")
    return s


def summary(rows, key="saving"):
    s = _finite_values(rows, key)
    if not s:
        return None
    return dict(median=float(np.median(s)), min=float(np.min(s)), max=float(np.max(s)), n=len(s))


def by_L(rows):
    """Median over the grid cells at each history length L, for both savings and both
    classical baselines (the L-dependence separates long memory from truncation)."""
    out = {}
    for L in sorted({r["setting"]["L"] for r in rows}):
        sub = [r for r in rows if r["setting"]["L"] == L and "C_mu" in r]
        if not sub:
            out[str(L)] = None
            continue
        med = lambda k: (float(np.median(_finite_values(sub, k)))
                         if any(r.get(k) is not None for r in sub) else None)
        out[str(L)] = dict(C_mu=med("C_mu"), C_mu_merged=med("C_mu_merged"),
                           C_q=med("C_q"), saving=med("saving"),
                           saving_vs_merged=med("saving_vs_merged"), n=len(sub))
    return out


def run(write=True):
    t0 = time.time()
    results = {"config": dict(data=DATA_CFG, rules=[r for _, r in RULES], seed_original=SEED_ORIG,
                              seed_fresh=SEED_FRESH, default=DEFAULT, grid=GRID,
                              bootstrap_rules=list(BOOT_RULES), pipeline_version=2.6,
                              bootstrap_controls=[CLOCK],
                              future_length_F=FUTURE_F, max_depth=MAX_DEPTH,
                              word_lengths=list(range(1, WORD_K + 1)),
                              periodicity_max_period=PERIOD_MAX)}

    # ---- 1. true memories of the known processes, and the closed-form cross-check ----
    procs = known_processes()
    exact = {}
    for name, m in procs.items():
        cq, depth, _ = c_q_unifilar(m)
        exact[name] = dict(C_mu=shannon_bits(m.pi), C_q=cq, q_depth=depth)
    for name, cf in closed_forms().items():
        assert abs(exact[name]["C_mu"] - cf["C_mu"]) < 1e-9, name
        assert abs(exact[name]["C_q"] - cf["C_q"]) < 1e-9, name
    results["exact"] = exact
    print("true memories of the known processes:")
    for k, v in exact.items():
        print(f"  {k:<22} C_mu={v['C_mu']:.4f}  C_q={v['C_q']:.4f}")

    # ---- 2. validation: reconstruct known processes at the automata's data budget ----
    rng_v = np.random.default_rng(SEED_FRESH + 7)
    slen = DATA_CFG["T"] - DATA_CFG["burn"]
    ntr = int(DATA_CFG["train_frac"] * DATA_CFG["n_seqs"])
    nte = DATA_CFG["n_seqs"] - ntr
    validation = {}
    print("\nvalidation (default setting, then the full grid):")
    for name, m in procs.items():
        tr = [m.sample(slen, rng_v) for _ in range(ntr)]
        te = [m.sample(slen, rng_v) for _ in range(nte)]
        if name == CLOCK:
            clock_data = (tr, te)               # reused by the bootstrap control in 3b
        rows = [evaluate(tr, te, st) for st in GRID]
        dflt = evaluate(tr, te, DEFAULT)
        ok = [r for r in rows if "C_mu" in r]      # cells with no frequent past are skipped
        err_mu = [abs(r["C_mu"] - exact[name]["C_mu"]) for r in ok]
        err_q = [abs(r["C_q"] - exact[name]["C_q"]) for r in ok]
        validation[name] = dict(default=dflt, grid=rows, by_L=by_L(rows),
                                grid_cells_without_frequent_pasts=
                                len(rows) - len(ok),
                                max_abs_err_C_mu=float(max(err_mu)),
                                max_abs_err_C_q=float(max(err_q)))
        print(f"  {name:<22} default C_mu={dflt['C_mu']:.4f} C_q={dflt['C_q']:.4f} "
              f"states={dflt['n_states']}   grid max|err| C_mu={max(err_mu):.4f} C_q={max(err_q):.4f}")
    results["validation"] = validation

    # ---- 3. automata: original sequences (and the original estimator, for reference) ----
    rec = json.load(open(os.path.join(HERE, "psi_scaleup_results.json")))
    data = automaton_data(SEED_ORIG)
    autom = {}
    print("\nautomata, original data (seed %d):" % SEED_ORIG)
    for name, (rule, tr, te) in data.items():
        # reproduce the committed estimator on these sequences (data-identity check)
        mo = infer_epsilon_machine(tr, 9, 5, tol=0.10, min_count=30)
        _, cq_o, _ = cq_depth_curve_orig(mo, D=16, tol=1e-4)
        cmu_o = C_mu_orig(mo)
        r_o = rec["substrates"][name]
        same = abs(cmu_o - r_o["C_mu_bits"]) < 1e-9 and abs(cq_o - r_o["Cq_depth_bits"]) < 1e-9
        rows = [evaluate(tr, te, st) for st in GRID]
        dflt = evaluate(tr, te, DEFAULT)
        autom[name] = dict(rule=rule, original_estimator=dict(C_mu=cmu_o, C_q=cq_o,
                           saving=1 - cq_o / cmu_o, reproduces_record=bool(same)),
                           default=dflt, grid=rows, grid_saving=summary(rows),
                           grid_saving_vs_merged=summary(rows, "saving_vs_merged"),
                           grid_cells_without_frequent_pasts=sum("C_mu" not in r for r in rows),
                           by_L=by_L(rows),
                           periodicity=periodicity(list(tr) + list(te), PERIOD_MAX))
        gs = summary(rows)
        print(f"  {name:<18} orig {1 - cq_o / cmu_o:6.3f} (record match {same})  "
              f"repaired default C_mu={dflt['C_mu']:.3f} C_q={dflt['C_q']:.3f} "
              f"saving={dflt['saving'] if dflt['saving'] is None else round(dflt['saving'], 3)} "
              f"states={dflt['n_states']} TV6={dflt['word_TV'][-1]:.4f}  "
              f"grid median {gs['median'] if gs else None}")
    results["automata_original_data"] = autom

    # ---- 3b. parametric bootstrap: does the pipeline recover a known automaton-like machine? ----
    # The machine fitted to each headline rule at the default setting is run as a
    # generator at the same data budget, and the pipeline is re-run across the grid.
    # Its true C_mu is that of its minimisation (emission rows compared exactly; it may hold
    # duplicate states), with its stationary law from the GTH elimination.
    boot = {}
    rng_b = np.random.default_rng(SEED_FRESH + 11)
    slen_b = DATA_CFG["T"] - DATA_CFG["burn"]
    ntr_b = int(DATA_CFG["train_frac"] * DATA_CFG["n_seqs"])
    nte_b = DATA_CFG["n_seqs"] - ntr_b
    print("\nparametric bootstrap (fitted machine as generator):")
    for name in BOOT_RULES:
        rule, tr, te = data[name]
        m_fit, _ = reconstruct(tr, DEFAULT["L"], DEFAULT["alpha"], DEFAULT["c_min"], DEFAULT["c_tr"])
        gen, cmu_t, cq_t, n_min = generator_truth(m_fit)
        trs = [gen.sample(slen_b, rng_b) for _ in range(ntr_b)]
        tes = [gen.sample(slen_b, rng_b) for _ in range(nte_b)]
        rows = [evaluate(trs, tes, st) for st in GRID]
        true_sav = 1.0 - cq_t / cmu_t
        boot[name] = dict(rule=rule, generator_states=int(gen.n), minimal_states=n_min,
                          true_C_mu=cmu_t, true_C_q=cq_t, true_saving=true_sav, grid=rows,
                          grid_saving=summary(rows), grid_saving_vs_merged=summary(rows, "saving_vs_merged"),
                          by_L=by_L(rows))
        gs, gm = summary(rows), summary(rows, "saving_vs_merged")
        print(f"  {name:<18} true saving {true_sav:.3f} (C_mu {cmu_t:.3f}, C_q {cq_t:.3f}, "
              f"{n_min} minimal states)  recovered: unifilar {gs['min']:.3f} to {gs['max']:.3f}, "
              f"merged {gm['min']:.3f} to {gm['max']:.3f}")
    results["bootstrap_automata"] = boot

    # The same bootstrap on the zero-saving hidden clock (audit A001, F1), drawn after the
    # four rules so their draws are unchanged.  The fitted generator is the clock's
    # finite-history machine, so the bootstrap recovers that machine's (spurious) saving.
    tr_c, _ = clock_data
    m_fit, _ = reconstruct(tr_c, DEFAULT["L"], DEFAULT["alpha"], DEFAULT["c_min"], DEFAULT["c_tr"])
    gen, cmu_t, cq_t, n_min = generator_truth(m_fit)
    trs = [gen.sample(slen_b, rng_b) for _ in range(ntr_b)]
    tes = [gen.sample(slen_b, rng_b) for _ in range(nte_b)]
    rows = [evaluate(trs, tes, st) for st in GRID]
    results["bootstrap_controls"] = {CLOCK: dict(
        process_saving=0.0, generator_states=int(gen.n), minimal_states=n_min,
        generator_C_mu=cmu_t, generator_C_q=cq_t, generator_saving=1.0 - cq_t / cmu_t,
        grid=rows, grid_saving=summary(rows),
        grid_saving_vs_merged=summary(rows, "saving_vs_merged"), by_L=by_L(rows))}
    gm = summary(rows, "saving_vs_merged")
    print(f"  {CLOCK} (control, process saving 0): generator saving {1.0 - cq_t / cmu_t:.3f}, "
          f"recovered against the merged baseline {gm['min']:.3f} to {gm['max']:.3f}")

    # ---- 4. automata: a fresh draw ----
    data2 = automaton_data(SEED_FRESH)
    autom2 = {}
    print("\nautomata, fresh data (seed %d):" % SEED_FRESH)
    for name, (rule, tr, te) in data2.items():
        rows = [evaluate(tr, te, st) for st in GRID]
        dflt = evaluate(tr, te, DEFAULT)
        autom2[name] = dict(rule=rule, default=dflt, grid=rows, grid_saving=summary(rows),
                            grid_saving_vs_merged=summary(rows, "saving_vs_merged"),
                            grid_cells_without_frequent_pasts=sum("C_mu" not in r for r in rows),
                            by_L=by_L(rows),
                            periodicity=periodicity(list(tr) + list(te), PERIOD_MAX))
        gs = summary(rows)
        print(f"  {name:<18} repaired default saving="
              f"{dflt['saving'] if dflt['saving'] is None else round(dflt['saving'], 3)} "
              f"states={dflt['n_states']}  grid median {gs['median'] if gs else None}")
    results["automata_fresh_data"] = autom2

    results["runtime_s"] = round(time.time() - t0, 1)
    if write:
        out = os.path.join(HERE, "psi_unifilar_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=1, default=float)
        print(f"\nsaved -> {out}   ({results['runtime_s']} s)")
    return results


def compare_to_record(results, path=None, tol=1e-9):
    """Compare a run with a record, leaf by leaf, after a JSON round trip of the run (so
    numpy types become what the record holds).  The top-level runtime_s is skipped.
    Returns (stats, problems).  stats: numbers compared, the largest absolute
    difference, and how many differ by more than tol.  problems: keys present on one
    side only, list-length changes, type changes, unequal non-numeric leaves (strings,
    booleans, None), and non-finite numbers, which are always reported."""
    path = path or os.path.join(HERE, "psi_unifilar_results.json")
    with open(path) as f:
        rec = json.load(f)
    new = json.loads(json.dumps(results, default=float))
    stats = dict(numbers=0, max_abs_diff=0.0, above_tol=0)
    problems = []

    def walk(a, b, where):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) ^ set(b)):
                problems.append(f"{where}/{k}: key only in the {'record' if k in a else 'run'}")
            for k in sorted(set(a) & set(b)):
                if where == "" and k == "runtime_s":
                    continue
                walk(a[k], b[k], f"{where}/{k}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                problems.append(f"{where}: length {len(a)} in the record, {len(b)} in the run")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{where}[{i}]")
        elif (isinstance(a, (int, float)) and not isinstance(a, bool)
              and isinstance(b, (int, float)) and not isinstance(b, bool)):
            if type(a) is not type(b):
                problems.append(f"{where}: {type(a).__name__} in the record, {type(b).__name__} in the run")
            if not (math.isfinite(a) and math.isfinite(b)):
                problems.append(f"{where}: non-finite number ({a!r} in the record, {b!r} in the run)")
                return
            stats["numbers"] += 1
            d = abs(a - b)
            stats["max_abs_diff"] = max(stats["max_abs_diff"], d)
            if d > tol:
                stats["above_tol"] += 1
        elif type(a) is not type(b) or a != b:
            problems.append(f"{where}: {a!r} in the record, {b!r} in the run")

    walk(rec, new, "")
    return stats, problems


def figure(results=None, out=None, dpi=300):
    """Plot the record: (a) validation against the true values at the default setting,
    (b) the saving against history length L for both classical baselines.
    Reads psi_unifilar_results.json when results is None (no experiment is re-run)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for path in ("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
                 "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf"):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix",
                         "font.size": 8.5, "axes.titlesize": 9, "legend.fontsize": 7})
    if results is None:
        results = json.load(open(os.path.join(HERE, "psi_unifilar_results.json")))
    out = out or os.path.join(HERE, "fig_psi_unifilar.png")

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw=dict(width_ratios=[1.1, 1.05]))

    # (a) validation
    names = ["IID fair coin", "IID biased 0.15", "Golden mean", "Perturbed coin p=0.1",
             "Period-3 (001)", "Order-2, shared emission", "Order-2, distinct emissions",
             "Three-state parity machine", "Even process"]
    labels = ["fair\ncoin", "biased\ncoin", "golden\nmean", "pert.\ncoin", "period\n3",
              "order 2\n(shared)", "order 2", "3-state\nparity", "even\nprocess"]
    xs = np.arange(len(names))
    ex = results["exact"]
    va = results["validation"]
    a = ax[0]
    a.scatter(xs - 0.12, [ex[n]["C_mu"] for n in names], marker="o", facecolors="none",
              edgecolors="k", s=28, label=r"true $C_\mu$", zorder=3)
    a.scatter(xs + 0.12, [ex[n]["C_q"] for n in names], marker="s", facecolors="none",
              edgecolors="k", s=28, label=r"true $C_q$", zorder=3)
    a.scatter(xs - 0.12, [va[n]["default"]["C_mu"] for n in names], marker="o", c="#1f5fa8",
              s=10, label=r"$C_\mu$, unifilar", zorder=4)
    a.scatter(xs - 0.12, [va[n]["default"]["C_mu_merged"] for n in names], marker="^",
              c="#d9822b", s=12, label=r"$C_\mu$, merged (conservative)", zorder=4)
    a.scatter(xs + 0.12, [va[n]["default"]["C_q"] for n in names], marker="s", c="#1f5fa8",
              s=10, label=r"$C_q$ (q-machine)", zorder=4)
    a.set_xticks(xs)
    a.set_xticklabels(labels, fontsize=5.6)
    a.set_ylabel("memory (bits)")
    a.set_title("(a) validation, history length 9")
    a.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    a.set_ylim(-0.15, 3.2)
    a.grid(alpha=0.25, lw=0.5)

    # (b) saving versus history length
    b = ax[1]
    series = [("Rule184 (II)", "rule 184 (periodic)", "#1f5fa8"), ("Rule54 (IV)", "rule 54", "#2e8b57"),
              ("Rule110 (IV)", "rule 110", "#7b3fa0"), ("Rule62 (II/IV)", "rule 62 (periodic)", "#8c8c8c"),
              ("Rule22 (III)", "rule 22", "#c0392b")]
    for key, lab, col in series:
        bl = results["automata_original_data"][key]["by_L"]
        bl2 = results["automata_fresh_data"][key]["by_L"]
        Ls = [int(L) for L in bl if bl[L] and bl[L]["saving_vs_merged"] is not None]
        b.plot(Ls, [100 * bl[str(L)]["saving_vs_merged"] for L in Ls], "-o", c=col, ms=3.5,
               lw=1.1, label=lab)
        b.plot(Ls, [100 * bl[str(L)]["saving"] for L in Ls], "--", c=col, lw=0.8)
        b.plot(Ls, [100 * bl2[str(L)]["saving_vs_merged"] for L in Ls], "o", mfc="none",
               mec=col, ms=5.5, lw=0)
    ev = results["validation"]["Even process"]["by_L"]
    Ls = [int(L) for L in ev if ev[L]]
    b.plot(Ls, [100 * ev[str(L)]["saving_vs_merged"] for L in Ls], "-^", c="k", ms=3.5,
           lw=1.0, label="even process (true 0)")
    b.plot(Ls, [100 * ev[str(L)]["saving"] for L in Ls], "--", c="k", lw=0.8)
    if CLOCK in results["validation"]:
        hc = results["validation"][CLOCK]["by_L"]
        Ls = [int(L) for L in hc if hc[L]]
        b.plot(Ls, [100 * hc[str(L)]["saving_vs_merged"] for L in Ls], "-s", c="#b8860b", ms=3.2,
               lw=1.0, label="hidden clock (true 0)")
        b.plot(Ls, [100 * hc[str(L)]["saving"] for L in Ls], "--", c="#b8860b", lw=0.8)
    b.axhline(0, c="k", lw=0.5)
    b.set_xlabel("history length $L$")
    b.set_ylabel(r"saving $1-C_q/C_\mu$ (%)")
    b.set_title("(b) saving against history length")
    b.set_xticks([7, 9, 11, 13])
    b.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, handlelength=1.6)
    b.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"saved -> {out}")
    return out


if __name__ == "__main__":
    import sys
    if "--replot" in sys.argv:
        figure()
    elif "--no-write" in sys.argv:
        st, probs = compare_to_record(run(write=False))
        print(f"\ncompared with the committed record: {st['numbers']} numbers, largest "
              f"|difference| {st['max_abs_diff']:.3g}, {st['above_tol']} above 1e-9; "
              f"{len(probs)} structural problems")
        for p in probs[:20]:
            print("  ", p)
        sys.exit(0 if (st["above_tol"] == 0 and not probs and st["numbers"] > 0) else 1)
    else:
        figure(run())
