#!/usr/bin/env python3
"""GAP-HAMMING EMBEDDING CHECK: the combinatorics behind the shifted-self-join
lower-bound sketch (manuscript sec. "Remark: a lower bound for the shifted self-join").

The reduction maps set membership to position on the line using a Sidon set
(all pairwise differences distinct).  With A, B subsets of [n], S the Sidon set,
and M a large integer:

    Alice streams  {S_a * M     : a in A}
    Bob   streams  {S_b * M + 1 : b in B}

Two claims are checked exhaustively on random instances:

  (1) J(1) == |A intersect B|  EXACTLY, where J(d) counts ordered pairs of
      streamed values at difference exactly d.  Within-party differences are
      multiples of M (never 1); a cross difference equals 1 iff the two parties
      hold the SAME element, since S_b - S_a = 0 iff a == b.

  (2) every NON-SIGNAL difference has multiplicity at most 2.  A difference can
      arise once inside Alice's set and once inside Bob's, and only when both
      parties hold the same PAIR {i,j} -- the Sidon property forbids any other
      coincidence.  Multiplicity 2 genuinely occurs, so "at most twice" is tight
      and "at most once" would be wrong.

This is the easy half of the argument (it verifies the embedding, not the cost
accounting); the manuscript says so.  Seeded, CPU.
Writes cwf_gaphamming_embedding_results.json.
"""
import json
from collections import Counter

import numpy as np

N_ELEM = 24          # |[n]|: universe size
M_MULT = 10 ** 6     # the multiplier separating within- from cross-party diffs
INSTANCES = 500
SEED = 20260728


def sidon_set(n, rng=None):
    """n integers with all pairwise differences distinct (Erdos-Turan / Singer form).

    Uses the classical modular construction {2*p*i + (i^2 mod p) : i < p} for a
    prime p >= n, which is a Sidon set (a "perfect difference set" family) inside
    [0, 2 p^2).  We verify the Sidon property directly rather than trusting it.
    """
    def is_prime(v):
        return v > 1 and all(v % f for f in range(2, int(v ** 0.5) + 1))

    p = n
    while not is_prime(p):
        p += 1
    S = np.array([2 * p * i + (i * i) % p for i in range(n)], dtype=np.int64)
    d = S[:, None] - S[None, :]
    off = d[~np.eye(n, dtype=bool)]
    assert len(set(off.tolist())) == len(off), "Sidon property violated"
    return S


def diff_multiset(vals):
    """Counter of all ordered pairwise differences of a 1-D integer array."""
    d = vals[:, None] - vals[None, :]
    off = d[~np.eye(len(vals), dtype=bool)]
    return Counter(off.tolist())


def main():
    rng = np.random.default_rng(SEED)
    S = sidon_set(N_ELEM)
    failures_j1 = 0
    failures_mult = 0
    max_nonsignal_mult = 0
    mult2_instances = 0

    for _ in range(INSTANCES):
        # random non-trivial subsets
        A = np.where(rng.random(N_ELEM) < 0.5)[0]
        B = np.where(rng.random(N_ELEM) < 0.5)[0]
        if len(A) == 0 or len(B) == 0:
            continue
        vals = np.concatenate([S[A] * M_MULT, S[B] * M_MULT + 1])
        assert len(set(vals.tolist())) == len(vals), "streamed values not distinct"

        cnt = diff_multiset(vals)
        j1 = cnt.get(1, 0)
        truth = len(np.intersect1d(A, B))
        if j1 != truth:
            failures_j1 += 1

        nonsignal = [c for d, c in cnt.items() if abs(d) != 1]
        m = max(nonsignal) if nonsignal else 0
        max_nonsignal_mult = max(max_nonsignal_mult, m)
        if m > 2:
            failures_mult += 1
        if m == 2:
            mult2_instances += 1

    out = dict(params=dict(n=N_ELEM, M=M_MULT, instances=INSTANCES, seed=SEED),
               failures_J1_ne_intersection=failures_j1,
               failures_multiplicity_gt_2=failures_mult,
               max_nonsignal_multiplicity=int(max_nonsignal_mult),
               instances_attaining_multiplicity_2=mult2_instances)
    print(f"Sidon set of {N_ELEM} elements inside [0, {S.max()}]  (verified Sidon)")
    print(f"{INSTANCES} random instances:")
    print(f"  J(1) != |A cap B| failures : {failures_j1}")
    print(f"  non-signal multiplicity > 2: {failures_mult}")
    print(f"  max non-signal multiplicity: {max_nonsignal_mult}"
          f"   (attained in {mult2_instances} instances)")
    print("  -> embedding CONFIRMED: J(1) is exact, and 'at most twice' is tight."
          if failures_j1 == 0 and failures_mult == 0 else "  -> EMBEDDING FAILED")
    with open("cwf_gaphamming_embedding_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_gaphamming_embedding_results.json")


if __name__ == "__main__":
    main()
