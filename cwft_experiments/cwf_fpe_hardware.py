#!/usr/bin/env python3
"""hbar_c AS A MEASURED SUBSTRATE CONSTANT UNDER DEVICE NOISE (Move B).

On a physical VSA (photonic / phase-change-memory), the FPE phase is stored with
finite precision and read back with noise.  We emulate two realistic device-noise
channels and measure, as functions of the noise, (a) the phase-space cell hbar_c
(does the resolution x bandwidth product survive?), (b) the coherence order
parameter (Wigner excess) and the relational readout (period-finding), and (c) the
effective phase-bit floor -- the point where the interference-borne capabilities die.

Noise models (parameters from the PCM/photonic literature):
 - PCM: q-level phase quantization (q ~ 2^bits) + Gaussian programming/read jitter
   sigma_phi on each stored phase (memristive conductance noise ~ few %).
 - Photonic: continuous Gaussian phase jitter sigma_phi per component (thermal /
   shot noise), no quantization.
The manuscript's design rule (q>=3 carries coherence, q=2 kills it) is here given a
noise-margin: how much jitter can be tolerated at each bit depth.  This supplies the
MECHANISM behind the empirical 3-4 bit findings of qFHRR (arXiv:2604.25939) and PCM
phase resolution.  Seeded, CPU.  Writes cwf_fpe_hardware_results.json.
"""
import json

import numpy as np

N = 3000
SIGMA = 1.0
OMEGA0 = 4.0
XS = np.linspace(-6, 16, 220)
PF_N, PF_SIGMA, PF_DELTA, PF_PAIRS = 4000, 2.0, 3.0, 5
PF_DS = np.linspace(2, 12, 400)
SEEDS = 4


def wigner_ville(psi):
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n)
        m = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex)
        r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)


def negativity(W):
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def channel(z, q, sigma_phi, rng):
    """Apply q-level phase quantization then Gaussian phase jitter."""
    ang = np.angle(z)
    if q:
        ang = np.round(ang * q / (2 * np.pi)) * (2 * np.pi / q)
    ang = ang + sigma_phi * rng.standard_normal(len(ang))
    return np.exp(1j * ang)


DRAWS_CELL = 24   # noise draws averaged (COMPLEX) before locating the half-crossing
SEEDS_CELL = 12   # codebook seeds for the cell estimate


def cell_hwhm(theta, q, sigma_phi, rng, dmax=8.0, n=400):
    """Half-width of the kernel ENVELOPE |K|, not of Re K.

    This codebook is a carrier (theta ~ N(OMEGA0, SIGMA^2)), so
    Re K(d) = cos(omega0 d) exp(-sigma^2 d^2 / 2): its first 1/2-crossing is the
    carrier half-period arccos(1/2)/omega0, which is a property of omega0 and NOT
    the phase-space cell -- it drifts ~6x as SIGMA is swept.  The envelope
    |K(d)| = exp(-sigma^2 d^2 / 2) carries the cell: its HWHM times SIGMA is
    sqrt(2 ln 2) at every bandwidth, matching the zero-mean result of the paper's
    phase-space section.  Normalizing at d=0 divides out the e^{-sigma_phi^2}
    attenuation, which is what makes this a consistency check of that identity.
    TWO estimator details matter, and getting either wrong moves the answer more
    than the physics does.

    (1) AVERAGE THE COMPLEX CORRELATION, NOT ITS MAGNITUDE.  The identity that
        licenses this measurement is E[K_noisy] = e^{-sigma_phi^2} K, which says the
        NORMALIZED width cannot move at all.  It governs the complex mean.  Taking
        |.| inside the draw loop instead estimates E|X|, and E|X| != |E[X]|: the
        magnitude of a finite-N complex mean has a Rice floor sqrt(pi/(4N))
        (0.0162 at N=3000), which is FLAT in the number of draws -- averaging more
        does not remove it.  Normalized by K(0) = e^{-sigma_phi^2} that floor grows
        as 0.0162 e^{+sigma_phi^2}: 0.15 at 1.5 rad and 0.88 at 2.0 rad, at which
        point the half-crossing disappears entirely.  So we accumulate the complex
        correlation and take the magnitude once, at the end.

    (2) INTERPOLATE THE CROSSING.  Returning the first grid point below 1/2 biases
        the width up by up to one grid step (0.02 here, i.e. ~1.7% of the cell).
        That alone accounted for the difference between the analytic sqrt(2 ln 2) =
        1.177 and the 1.193 reported from a coarse grid.
    """
    d = np.linspace(0, dmax, n)
    acc = np.zeros(n, dtype=complex)
    for _ in range(DRAWS_CELL):
        a0 = channel(np.exp(1j * 0.0 * theta), q, sigma_phi, rng)
        acc += np.array([np.vdot(a0, channel(np.exp(1j * dd * theta),
                                             q, sigma_phi, rng))
                         for dd in d]) / len(theta)
    K = np.abs(acc / DRAWS_CELL)          # magnitude ONCE, after the complex average
    K = K / (K[0] + 1e-12)
    below = np.where(K < 0.5)[0]
    if not len(below):
        return float("nan")               # no crossing: estimator out of validity
    i = below[0]
    if i == 0:
        return 0.0
    y0, y1 = K[i - 1], K[i]               # linear interpolation of the 1/2 crossing
    t = (y0 - 0.5) / (y0 - y1) if y0 != y1 else 0.0
    return float(d[i - 1] + t * (d[i] - d[i - 1]))


def coherence(theta, q, sigma_phi, rng):
    dec = lambda v: (np.exp(-1j * np.outer(XS, theta)) @ v) / len(theta)
    b1 = channel(np.exp(1j * 2.0 * theta), q, sigma_phi, rng) + \
        channel(np.exp(1j * 8.0 * theta), q, sigma_phi, rng)
    b0 = channel(np.exp(1j * 5.0 * theta), q, sigma_phi, rng)
    return negativity(wigner_ville(dec(b1))) - negativity(wigner_ville(dec(b0)))


def period(q, sigma_phi, rng, planted=True):
    """Relational readout under the channel.

    Returns (ratio, raw_peak, raw_background, hit) rather than the ratio alone.
    The ratio is blind to the dominant effect of phase noise BY CONSTRUCTION: the
    peak and the background are both cross-term quantities, so the channel
    attenuates both by the same e^{-sigma_phi^2} and the quotient barely moves while
    the raw signal collapses.  `hit` is whether the global argmax lands on the
    planted lag -- the measure that actually tracks usefulness.  With planted=False
    no period is planted, which calibrates the ratio's chance floor (it is ~1.5,
    not 1) and the argmax null rate.
    """
    th = PF_SIGMA * rng.standard_normal(PF_N)
    base = np.sort(rng.uniform(0, 25, PF_PAIRS))
    vals = np.concatenate([base, base + PF_DELTA]) if planted else \
        np.sort(rng.uniform(0, 25, 2 * PF_PAIRS))
    b = sum(channel(np.exp(1j * v * th), q, sigma_phi, rng) for v in vals)
    score = (np.cos(np.outer(PF_DS, th)) @ (np.abs(b) ** 2)) / PF_N
    i0 = int(np.argmin(np.abs(PF_DS - PF_DELTA)))
    peak = float(score[i0 - 8:i0 + 9].max())
    bg = float(np.median(np.abs(score)))
    hit = bool(abs(PF_DS[int(np.argmax(score))] - PF_DELTA) < 0.3)
    return float(peak / (bg + 1e-9)), peak, bg, hit


def main():
    out = {"params": dict(N=N, sigma=SIGMA, omega0=OMEGA0), }

    # (A) photonic: continuous phase jitter sweep, cell + coherence
    print("(A) photonic phase-jitter sweep (q=continuous):")
    print("    cell: complex-averaged correlation, interpolated 1/2-crossing, "
          f"{SEEDS_CELL} seeds")
    print("    readout: RAW peak and background reported alongside the ratio, plus "
          "argmax accuracy")
    NTRIAL = 200
    photonic = []
    for sp in [0.0, 0.1, 0.2, 0.4, 0.8, 1.5]:
        cells, cohs = [], []
        for s in range(SEEDS_CELL):
            rng = np.random.default_rng(100 + s)
            th = OMEGA0 + SIGMA * rng.standard_normal(N); th = th[th > 0]
            c = cell_hwhm(th, 0, sp, rng) * SIGMA
            if np.isfinite(c):
                cells.append(c)
            if s < SEEDS:
                cohs.append(coherence(th, 0, sp, rng))
        pr = [period(0, sp, np.random.default_rng(5000 + t)) for t in range(NTRIAL)]
        nul = [period(0, sp, np.random.default_rng(9000 + t), planted=False)
               for t in range(NTRIAL)]
        cell = float(np.mean(cells)) if cells else float("nan")
        rec = dict(sigma_phi=sp, cell=cell,
                   cell_sd=float(np.std(cells)) if cells else float("nan"),
                   cell_seeds_with_crossing=len(cells), cell_seeds=SEEDS_CELL,
                   coherence=float(np.mean(cohs)),
                   period_peak=float(np.mean([x[0] for x in pr])),
                   raw_peak=float(np.mean([x[1] for x in pr])),
                   raw_background=float(np.mean([x[2] for x in pr])),
                   argmax_accuracy=float(np.mean([x[3] for x in pr])),
                   argmax_accuracy_se=float(np.std([x[3] for x in pr]) / np.sqrt(NTRIAL)),
                   null_ratio=float(np.mean([x[0] for x in nul])),
                   null_argmax_accuracy=float(np.mean([x[3] for x in nul])),
                   trials=NTRIAL)
        photonic.append(rec)
        print(f"    sigma_phi={sp:>4}: cell={cell:.4f}  coherence={np.mean(cohs):+.3f}"
              f"  ratio={rec['period_peak']:.2f} (null {rec['null_ratio']:.2f})"
              f"  RAW peak={rec['raw_peak']:.3f} bg={rec['raw_background']:.3f}"
              f"  argmax acc={rec['argmax_accuracy']:.3f}")
    print("    -> the RATIO is blind by construction: the channel attenuates peak and")
    print("       background by the same e^{-sigma_phi^2}.  Accuracy is the measure")
    print("       that tracks usefulness, and the ratio's chance floor is ~1.5, not 1.")
    out["photonic"] = photonic

    # (B) PCM: bit-depth (q=2^bits) at a fixed realistic programming jitter
    print("(B) PCM bit-depth at programming jitter sigma_phi=0.15 rad:")
    pcm = []
    for bits in [1, 2, 3, 4, 6]:
        q = 2 ** bits
        cohs = []
        for s in range(SEEDS):
            rng = np.random.default_rng(200 + s)
            th = OMEGA0 + SIGMA * rng.standard_normal(N); th = th[th > 0]
            cohs.append(coherence(th, q, 0.15, rng))
        pr = [period(q, 0.15, np.random.default_rng(6000 + t)) for t in range(NTRIAL)]
        nul = [period(q, 0.15, np.random.default_rng(9500 + t), planted=False)
               for t in range(NTRIAL)]
        rec = dict(bits=bits, q=q, coherence=float(np.mean(cohs)),
                   period_peak=float(np.mean([x[0] for x in pr])),
                   raw_peak=float(np.mean([x[1] for x in pr])),
                   argmax_accuracy=float(np.mean([x[3] for x in pr])),
                   argmax_accuracy_se=float(np.std([x[3] for x in pr]) / np.sqrt(NTRIAL)),
                   null_ratio=float(np.mean([x[0] for x in nul])),
                   null_argmax_accuracy=float(np.mean([x[3] for x in nul])),
                   trials=NTRIAL)
        pcm.append(rec)
        print(f"    {bits} bit (q={q:>2}): Wigner excess={np.mean(cohs):+.3f}  "
              f"ratio={rec['period_peak']:.2f} (null {rec['null_ratio']:.2f})  "
              f"argmax acc={rec['argmax_accuracy']:.3f} "
              f"(null {rec['null_argmax_accuracy']:.3f})")
    out["pcm"] = pcm
    print("  -> the WIGNER-EXCESS WITNESS is indistinguishable from zero at 1 bit and")
    print("     positive from 2 bits up.  The RELATIONAL READOUT is a different story:")
    print("     it is already far above its null at 1 bit.  The threshold is a property")
    print("     of the witness, not of coherence-borne capability.")

    with open("cwf_fpe_hardware_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_hardware_results.json")


if __name__ == "__main__":
    main()
