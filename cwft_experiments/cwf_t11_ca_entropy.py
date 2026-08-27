"""
T1.1 Phase 2 (cont.) -- a FAITHFUL symbolic entropy estimator for cellular automata.

Both CA Lyapunov-tangent estimators (defect-max, QR-Boolean-Jacobian) measure sensitivity
SPREADING, not metric entropy, and contradict gzip compressibility (see ../cwft_manuscripts/P5_tradeoff_law/THEOREM_ac_bound.md).
This replaces them with a measure-theoretic quantity: the TEMPORAL BLOCK-ENTROPY RATE of a
single cell's symbol sequence,

    h_t = lim_{L->inf} [ H(L+1) - H(L) ]   (bits / step)

where H(L) is the Shannon entropy of length-L temporal blocks (one site's values over L
consecutive steps), pooled over all sites / trajectories. h_t(L) = H(L+1)-H(L) is the
conditional entropy of the next symbol given the previous L -- a monotone-decreasing upper
bound on the entropy rate; we report it at the largest block length L* with adequate
sampling. This is the entropy an observer watching one cell would measure: 0 for frozen /
periodic rules, ~1 for maximally chaotic, intermediate for complex. Plug-in entropies carry
the Miller-Madow bias correction.

Faithfulness checks (run as __main__): ordering must match gzip and the Wolfram classes
(I/II ~ 0  <  IV  <  III), with rule 30's column near-maximal and majority/traffic low.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from cwf_b1_tradeoff import ca_table, ca_step

LN2 = float(np.log(2))


def _spacetime(rule, n, T, burn, n_traj, seed):
    tbl = ca_table(rule); rng = np.random.default_rng(seed)
    cols = np.empty((n_traj, T, n), dtype=np.int8)
    for k in range(n_traj):
        s = rng.integers(0, 2, n).astype(np.int8)
        for _ in range(burn):
            s = ca_step(s, tbl)
        for t in range(T):
            cols[k, t] = s; s = ca_step(s, tbl)
    return cols                                    # (n_traj, T, n)


def _block_entropy(seqs, L):
    """Miller-Madow-corrected Shannon entropy (bits) of length-L blocks over a set of
    integer sequences. seqs: 2D array (n_seq, T) of {0,1}. Pools windows from all rows."""
    if L == 0:
        return 0.0
    W = sliding_window_view(seqs, L, axis=1)        # (n_seq, T-L+1, L)
    W = W.reshape(-1, L)
    weights = (1 << np.arange(L, dtype=np.int64))   # encode block -> int
    codes = W.astype(np.int64) @ weights
    counts = np.bincount(codes)
    counts = counts[counts > 0]
    N = counts.sum(); p = counts / N
    H = -np.sum(p * np.log2(p))
    H += (counts.size - 1) / (2.0 * N * LN2)        # Miller-Madow bias correction
    return float(H), int(N), int(counts.size)


def ca_symbolic_entropy(rule, n=160, T=260, burn=160, n_traj=40, seed=0,
                        Lmax=14, min_per_state=50, return_detail=False):
    """Temporal block-entropy rate h_t (bits/step) of a single CA cell, at the largest
    block length L* with adequate sampling (#samples >= min_per_state * 2^(L*+1))."""
    cols = _spacetime(rule, n, T, burn, n_traj, seed)
    # treat every site's time-series as a sequence: reshape to (n_traj*n, T)
    seqs = np.moveaxis(cols, 2, 1).reshape(-1, T)   # (n_traj*n, T)
    Hs = {0: (0.0, seqs.shape[0] * T, 1)}
    htL = {}
    for L in range(1, Lmax + 2):
        Hs[L] = _block_entropy(seqs, L)
    # h_t(L) = H(L+1) - H(L); pick largest L where the (L+1)-block sampling is adequate
    Lstar = 1
    for L in range(1, Lmax + 1):
        H_L1, N_L1, _ = Hs[L + 1]
        htL[L] = Hs[L + 1][0] - Hs[L][0]
        if N_L1 >= min_per_state * (1 << (L + 1)):
            Lstar = L
    h = max(0.0, htL[Lstar])                        # entropy rate is non-negative
    if return_detail:
        return h, dict(Lstar=Lstar, htL=htL, H={L: Hs[L][0] for L in Hs})
    return h


if __name__ == "__main__":
    import zlib
    def gzip_rate(rule, n=400, T=400, burn=200, seed=0):
        tbl = ca_table(rule); rng = np.random.default_rng(seed)
        s = rng.integers(0, 2, n).astype(np.int8)
        for _ in range(burn):
            s = ca_step(s, tbl)
        rows = [s.copy()]
        for _ in range(T):
            s = ca_step(s, tbl); rows.append(s.copy())
        D = np.array(rows, dtype=np.uint8); raw = np.packbits(D.ravel()).tobytes()
        return len(zlib.compress(raw, 9)) / (D.size / 8)

    panel = [(0, "I"), (8, "I"), (232, "II"), (184, "II"), (108, "II"), (94, "II"),
             (110, "IV"), (54, "IV"), (30, "III"), (90, "III"), (150, "III"),
             (18, "III"), (126, "III"), (45, "III"), (60, "III")]
    print(f"{'rule':>5}{'class':>6}{'h_t(bits/step)':>16}{'L*':>4}{'gzip':>8}")
    for rule, cls in panel:
        h, det = ca_symbolic_entropy(rule, return_detail=True)
        print(f"{rule:5d}{cls:>6}{h:16.4f}{det['Lstar']:4d}{gzip_rate(rule):8.3f}")
