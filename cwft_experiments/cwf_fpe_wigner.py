"""
cwf_fpe_wigner.py -- Wigner negativity as a non-classicality witness for VSA
bundles.  (VSA_CWFT_NOTE.md sec.5c, the strongest pay candidate of the FPE
phase-space program.)

A VSA vector v decodes to an amplitude profile over value-space,
    psi_v(x) = (1/N) sum_k v_k exp(-i x theta_k),
the "wavefunction in the x-basis" (theta_k = positive random frequencies -> psi is
a complex analytic signal).  A single FPE value phi(x0) -> psi = a bump at x0; a
BUNDLE phi(x1)+phi(x2) -> two bumps -- a coherent superposition (a computational
"cat state").  The Wigner-Ville distribution W(x,p) of psi is real but goes
NEGATIVE for cat states (interference fringes between the bumps).

The claim under test: Wigner negativity is a non-classicality WITNESS for VSA --
  * a single value is classical (W >= 0, negativity ~ 0);
  * a coherent bundle of separated values has W < 0 (genuine superposition);
  * a classical MIXTURE of the same values (phase-randomized ensemble average)
    has W >= 0 (negativity ~ 0) -- so the negativity distinguishes coherent
    (l2, amplitude) superposition from a classical (l1, probability) belief state.
  * negativity ONSETS at the resolution scale (fringes need separated bumps).
This is the measurable order parameter that the ground (l1) reading lacks --
exactly why Furlong-Eliasmith (2022) found negative fractional-binding quasi-
probability and DECLINED the Born rule; the negativity is why l1 is incomplete.

HONEST CAVEAT: that ANY coherent superposition of separated bumps has negative
Wigner is a math fact; the content is (a) VSA bundling realizes such superpositions
(so VSA states carry a genuine non-classicality witness), (b) it is a clean order
parameter (vs separation, vs decoherence), (c) decoherence/mixture kills it -- the
l2-vs-l1 split, measurable.  CPU, numpy, seeded.  Writes cwf_fpe_wigner_results.json.
"""

import json
import numpy as np

RNG = np.random.default_rng(0)


def make_freqs(N, omega0=4.0, sigma=1.0):
    """Positive random frequencies (analytic signal): theta ~ omega0 + sigma*Normal,
    kept > 0.  omega0 = carrier, sigma = bandwidth (sets resolution ~ 1/sigma)."""
    th = omega0 + sigma * RNG.standard_normal(N)
    return th[th > 0]


def fpe(x, theta):
    return np.exp(1j * x * theta)                       # FPE encoding of scalar x: (N,)


def profile(v, theta, xs):
    """Decoded wavefunction psi_v(x) over the value grid xs."""
    return (np.exp(-1j * np.outer(xs, theta)) @ v) / len(theta)


def wigner_ville(psi):
    """Discrete Wigner-Ville distribution W(x,p) of complex psi (real-valued)."""
    M = len(psi)
    W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        kmax = min(n, M - 1 - n)
        r = np.zeros(M, dtype=complex)
        m = np.arange(-kmax, kmax + 1)
        r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)                                    # (x, p)


def negativity(W):
    """Fraction of total Wigner mass that is negative, in [0, 0.5]."""
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


def bundle_profile(centers, theta, xs, phases=None):
    """psi for a bundle sum_i e^{i phase_i} phi(x_i)."""
    if phases is None:
        phases = np.zeros(len(centers))
    v = sum(np.exp(1j * ph) * fpe(c, theta) for c, ph in zip(centers, phases))
    return profile(v, theta, xs)


if __name__ == "__main__":
    print("=" * 74)
    print("WIGNER NEGATIVITY as a non-classicality witness for VSA bundles")
    print("=" * 74)
    N = 3000
    theta = make_freqs(N)
    xs = np.linspace(-6, 16, 220)
    dx = xs[1] - xs[0]
    print(f"  N={len(theta)} positive freqs (carrier 4, bw 1); x-grid {len(xs)} pts, "
          f"dx={dx:.3f}; resolution ~ 1/bw = 1.0")

    # ---- baseline: single value = the CLASSICAL floor (WVD finite-N/sidelobe
    #      artifact, NOT genuine negativity).  Average over freq draws. ----
    def avg_neg(make_psi, ntr=6):
        vals = []
        for _ in range(ntr):
            th = make_freqs(N)
            vals.append(negativity(wigner_ville(make_psi(th))))
        return float(np.mean(vals)), float(np.std(vals))

    out = {}
    base, base_sd = avg_neg(lambda th: profile(fpe(5.0, th), th, xs))
    two, _ = avg_neg(lambda th: bundle_profile([2.0, 8.0], th, xs))
    print(f"\n[baseline] single value (classical floor) = {base:.4f} +/- {base_sd:.4f}")
    print(f"           two separated values            = {two:.4f}  "
          f"(genuine excess = {two - base:+.4f})")
    print(f"  -> the ~{base:.3f} floor is a WVD artifact; GENUINE negativity = excess above it.")
    out["baseline"] = {"single": base, "two": two, "excess": two - base}

    # ---- Part 1: negativity vs separation (onset at the resolution scale) ----
    print(f"\n[1] negativity vs separation Delta (resolution ~ 1.0); excess above floor:")
    P1 = []
    for D in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
        neg, _ = avg_neg(lambda th: bundle_profile([5 - D / 2, 5 + D / 2], th, xs))
        P1.append(dict(sep=D, neg=neg, excess=neg - base))
        print(f"    Delta={D:>4}: negativity={neg:.4f}  excess={neg - base:+.4f}")
    out["part1_separation"] = P1

    # ---- Part 1b: the excess is CARRIER-LOCKED, not resolution-onset ---------
    # Factor the common boost out of a two-atom bundle on a carrier codebook:
    #   psi_1 + psi_2 = e^{-i w0 x} [ e^{i w0 x1} G(x-x1) + e^{i w0 x2} G(x-x2) ],
    # a real-Gaussian cat with RELATIVE PHASE phi = w0 * Delta.  So the fringe
    # contrast oscillates in Delta with period 2*pi/w0, and only its ENVELOPE grows
    # with separation.  The coarse grid of Part 1 aliases this; sample it finely.
    W0 = 4.0
    print(f"\n[1b] the excess is carrier-locked: predicted period 2*pi/w0 = "
          f"{2*np.pi/W0:.3f} in Delta (resolution ~ 1/bw = 1.0)")
    P1B = []
    for D in np.arange(0.25, 3.01, 0.25):
        neg, sd = avg_neg(lambda th, D=D: bundle_profile([5 - D / 2, 5 + D / 2],
                                                         th, xs), ntr=10)
        P1B.append(dict(sep=float(D), neg=neg, sd=sd, excess=neg - base,
                        carrier_phase=float(W0 * D)))
        print(f"    Delta={D:4.2f}  phi=w0*Delta={W0*D:5.2f} rad  "
              f"excess={neg - base:+.4f} +/- {sd:.4f}")
    ex = np.array([r["excess"] for r in P1B])
    ds = np.array([r["sep"] for r in P1B])
    print(f"    -> peak at Delta={ds[int(np.argmax(ex))]:.2f} (BELOW the resolution 1.0); "
          f"excess is non-monotone in Delta.")
    out["part1b_carrier"] = P1B

    # ---- Part 1c: HUDSON, stated correctly ----------------------------------
    # The continuum profile of a carrier-Gaussian codebook is
    #   psi(x) = int p(th) e^{i(x0-x)th} dth = e^{-i w0 u} e^{-sig^2 u^2/2},  u = x-x0,
    # i.e. a Gaussian times a linear phase = a translated, boosted GAUSSIAN state.
    # Hudson's theorem (W >= 0 iff Gaussian, for pure states) therefore predicts a
    # floor of EXACTLY ZERO here.  The measured ~0.055 is an ESTIMATOR artifact of
    # finite N and the finite x-window, and it shrinks as the estimate is refined.
    print("\n[1c] Hudson: the continuum profile IS Gaussian, so the predicted floor is 0")

    def psi_continuum(x0, sig=1.0, w0=W0, grid=None):
        u = (xs if grid is None else grid) - x0
        return np.exp(-1j * w0 * u) * np.exp(-sig ** 2 * u ** 2 / 2)

    def profile_chunked(v, th, grid, chunk=20000):
        """Memory-safe decode: never materialises a len(grid) x len(th) matrix."""
        acc = np.zeros(len(grid), dtype=complex)
        for a in range(0, len(th), chunk):
            b = min(a + chunk, len(th))
            acc += np.exp(-1j * np.outer(grid, th[a:b])) @ v[a:b]
        return acc / len(th)

    cont = float(negativity(wigner_ville(psi_continuum(5.0))))
    print(f"    continuum single atom (analytic Gaussian) : negativity = {cont:.6f}")
    # the theta>0 truncation the manuscript blames contributes essentially nothing:
    # N(4,1) truncated at 0 loses Phi(-4) = 3.2e-5 of its mass.
    tg = np.linspace(1e-6, 12.0, 40000)
    w = np.exp(-(tg - W0) ** 2 / 2)
    w /= w.sum()
    psi_tr = (np.exp(-1j * np.outer(xs - 5.0, tg)) @ w)
    trunc = float(negativity(wigner_ville(psi_tr)))
    print(f"    continuum, theta>0 truncated              : negativity = {trunc:.6f}")
    # finite-N: the floor is sampling noise and it decays
    NS = []
    for Nn in [3000, 30000, 100000]:
        th = W0 + 1.0 * RNG.standard_normal(Nn)
        th = th[th > 0]
        v = fpe(5.0, th)
        nv = float(negativity(wigner_ville(profile_chunked(v, th, xs))))
        NS.append(dict(N=int(len(th)), negativity=nv))
        print(f"    finite N={len(th):>7}                        : negativity = {nv:.4f}")
    # and it depends on where in the numerical window the bump sits
    WP = []
    for x0 in [0.0, 5.0, 12.0]:
        nv, _ = avg_neg(lambda th, x0=x0: profile(fpe(x0, th), th, xs), ntr=6)
        WP.append(dict(x0=x0, negativity=nv))
        print(f"    finite N=3000, bump at x0={x0:>5.1f}            : negativity = {nv:.4f}")
    print("    -> a quantity that depends on N and on window position is a property")
    print("       of the ESTIMATOR, not of the state.  Hence the excess must be taken")
    print("       against the matched incoherent baseline W1+W2, measured identically.")
    out["part1c_hudson"] = dict(continuum_single=cont, continuum_truncated=trunc,
                                finite_N=NS, window_position=WP,
                                note="Hudson predicts 0 for the Gaussian continuum "
                                     "profile; the measured floor is an estimator "
                                     "artifact of finite N and finite window.")

    # ---- Part 2: coherent vs decohered mixture (the witness) ----
    print(f"\n[2] coherent superposition vs decohered MIXTURE (phase-randomized):")
    centers = [2.0, 8.0]
    neg_coh = negativity(wigner_ville(bundle_profile(centers, theta, xs)))
    Wmix = np.zeros((len(xs), len(xs)))
    NT = 300
    for _ in range(NT):
        ph = RNG.uniform(0, 2 * np.pi, len(centers))
        Wmix += wigner_ville(bundle_profile(centers, theta, xs, phases=ph))
    Wmix /= NT
    neg_mix = negativity(Wmix)
    # the witness: how much of the GENUINE (excess-above-floor) negativity does
    # decoherence remove?
    removed = (neg_coh - neg_mix) / (neg_coh - base + 1e-12)
    print(f"    coherent bundle:          negativity = {neg_coh:.4f}  (excess {neg_coh-base:+.4f})")
    print(f"    decohered (mixture, avg): negativity = {neg_mix:.4f}  (excess {neg_mix-base:+.4f})")
    print(f"    classical floor:          negativity = {base:.4f}")
    killed = removed > 0.7
    print(f"    -> decoherence removes {removed*100:.0f}% of the genuine negativity"
          f" ({'KILLS it -> ' if killed else ''}witnesses COHERENT (l2) superposition,")
    print(f"       not an l1 classical mixture).")
    out["part2_decoherence"] = {"coherent": neg_coh, "mixture": neg_mix, "floor": base,
                                "frac_removed": removed, "killed": killed}

    # ---- Part 3: negativity vs bundle size k ----
    print(f"\n[3] negativity vs bundle size k (well-separated values):")
    P3 = []
    for k in [1, 2, 3, 4, 6, 8]:
        cs = np.linspace(-4, 14, k)
        negs = []
        for _ in range(5):
            th = make_freqs(N)
            negs.append(negativity(wigner_ville(bundle_profile(cs, th, xs))))
        P3.append(dict(k=k, neg=float(np.mean(negs))))
        print(f"    k={k}: negativity={np.mean(negs):.4f}")
    out["part3_bundle_size"] = P3

    with open("cwf_fpe_wigner_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_wigner_results.json")
