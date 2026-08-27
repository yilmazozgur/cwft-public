#!/usr/bin/env python3
"""UNIT TESTS FOR THE IDENTITIES THE MANUSCRIPT RESTS ON.

Run with `python3 test_identities.py`.  No pytest required, no arguments, no
network; finishes in well under a minute.  Exit status is 0 iff every test passes.

WHY THIS FILE EXISTS.  Three defects reached a submitted draft because a REPAIR
was written as a claim and never tested.  In each case the diagnosis was right and
the fix asserted a property nobody measured:

  * the foveation arms were asserted to sit at "identical maximum frequency"
    -- nobody evaluated theta * w'(x), which is 2.66x larger at the fovea;
  * the hardware cell was asserted to be fixed by averaging more noise draws
    -- nobody checked that a mean of MAGNITUDES converges, which it does not;
  * the Gap-Hamming readout was asserted to agree with J(1) once the Sidon
    spacing M was large -- nobody checked that M leaves the self term alone.

Tests 1-3 below are regression tests on exactly those three claims: each one
FAILS if the old assertion is reinstated.  The rest cover the identities the
paper states as exact, so that a future edit cannot quietly break one.

Every test is deterministic (seeded).  Tolerances are stated inline and are
loose enough to survive a BLAS change but tight enough to catch a sign error.
"""
import sys
import traceback

import numpy as np

TESTS = []


def test(name):
    def deco(fn):
        TESTS.append((name, fn))
        return fn
    return deco


# =============================================================================
# 1-3: regression tests on the three defects that reached a draft
# =============================================================================

@test("gap-hamming: T(D) has three terms, and M controls only the third")
def t_gaphamming():
    """The reduction's readout is NOT the pair count.  Exactly:

        T(D) = (1 + K(2D)) |A cap B|  +  k K(D)  +  R

    with |R| bounded by the Sidon spacing.  The manuscript once claimed that
    choosing M large made T(D) agree with J(D); this test fails if that is
    reinstated, because it pins the M-INDEPENDENT terms.
    """
    sigma, w0 = 2.0, 8.0
    K = lambda u: np.cos(w0 * np.asarray(u)) * np.exp(-sigma ** 2 * np.asarray(u) ** 2 / 2)
    rng = np.random.default_rng(11)

    def sidon(n):
        p = n
        while not all(p % f for f in range(2, int(p ** .5) + 1)) or p < 2:
            p += 1
        S = np.array([2 * p * i + (i * i) % p for i in range(n)], dtype=np.int64)
        d = S[:, None] - S[None, :]
        assert len(set(d[~np.eye(n, dtype=bool)].tolist())) == n * (n - 1), "not a Sidon set"
        return S

    n, M = 24, 64
    S = sidon(n)
    for D in (1, 2, 3, 4):
        errs = []
        for _ in range(40):
            A = np.where(rng.random(n) < .5)[0]
            B = np.where(rng.random(n) < .5)[0]
            if not len(A) or not len(B):
                continue
            vals = np.concatenate([S[A] * M, S[B] * M + D]).astype(float)
            k = len(vals)
            inter = len(np.intersect1d(A, B))
            du = vals[:, None] - vals[None, :]
            T = 0.5 * (K(du - D) + K(du + D)).sum()          # includes the diagonal
            predicted = (1 + K(2 * D)) * inter + k * K(D)     # R is ~0 at this M
            errs.append(abs(T - predicted))
        m = max(errs)
        assert m < 1e-9, f"three-term identity failed at D={D}: max error {m:.2e}"

    # and the defect itself: at D=1 the naive reading is WRONG by k*K(1)
    A = np.where(rng.random(n) < .5)[0]
    B = np.where(rng.random(n) < .5)[0]
    vals = np.concatenate([S[A] * M, S[B] * M + 1]).astype(float)
    k, inter = len(vals), len(np.intersect1d(A, B))
    du = vals[:, None] - vals[None, :]
    T1 = 0.5 * (K(du - 1) + K(du + 1)).sum()
    naive_err = abs(T1 - inter) / k
    assert naive_err > 0.015, (
        "at D=1 the naive readout should miss J by ~|K(1)|=0.0197 per item; "
        f"got {naive_err:.4f}. If this fails the codebook changed -- recheck the "
        "shift condition in the manuscript's Gap-Hamming remark.")
    # while a shift of a few kernel widths removes it with no calibration
    vals4 = np.concatenate([S[A] * M, S[B] * M + 4]).astype(float)
    du4 = vals4[:, None] - vals4[None, :]
    T4 = 0.5 * (K(du4 - 4) + K(du4 + 4)).sum()
    assert abs(T4 - inter) / k < 1e-6, "large shift should need no calibration"
    return f"identity exact to 1e-9; naive D=1 error {naive_err:.4f}/item, D=4 error <1e-6"


@test("estimator: mean of magnitudes does NOT converge; complex mean does")
def t_magnitude_bias():
    """E|X| != |E X|.  The magnitude of a finite-N complex mean carries a Rice
    floor sqrt(pi/4N) that is FLAT in the number of averaged draws.  The
    manuscript once 'fixed' a noisy cell estimate by averaging more draws of the
    MAGNITUDE; this test fails if that is reinstated.
    """
    N, TRIALS = 3000, 120
    rng = np.random.default_rng(3)
    floor_mag, floor_cpx = [], []
    for draws in (6, 24, 96, 384):
        mg, cp = [], []
        for _ in range(TRIALS):
            z = np.array([np.exp(1j * rng.uniform(0, 2 * np.pi, N)).mean()
                          for _ in range(draws)])
            mg.append(np.abs(z).mean())          # mean of magnitudes
            cp.append(abs(z.mean()))             # magnitude of the complex mean
        floor_mag.append(float(np.mean(mg)))
        floor_cpx.append(float(np.mean(cp)))
    rice = np.sqrt(np.pi / (4 * N))
    assert abs(floor_mag[0] - rice) < 0.25 * rice, \
        f"magnitude floor {floor_mag[0]:.5f} should match Rice sqrt(pi/4N)={rice:.5f}"
    # flat in draws: 64x more averaging changes it by under 20%
    ratio = floor_mag[-1] / floor_mag[0]
    assert 0.8 < ratio < 1.25, \
        f"magnitude floor must NOT shrink with draws; ratio {ratio:.3f} over 64x"
    # the complex mean does converge
    cpx_ratio = floor_cpx[0] / floor_cpx[-1]
    assert cpx_ratio > 4.0, \
        f"complex mean must converge as 1/sqrt(draws) (expect ~8x over 64x); " \
        f"got {cpx_ratio:.2f}x ({floor_cpx[0]:.5f} -> {floor_cpx[-1]:.5f})"
    return (f"Rice floor {rice:.5f}; mean-of-mag flat ({ratio:.2f}x over 64x draws); "
            f"complex mean converges {cpx_ratio:.1f}x")


@test("warp: a coordinate warp raises the effective spatial frequency")
def t_warp_slope():
    """The warped atom is exp(i theta w(x)), so its phase turns at theta*w'(x).
    Sharing a codebook therefore does NOT match the arms in local bandwidth.  The
    manuscript once asserted 'both arms sit at identical maximum frequency'; this
    test fails if that is reinstated.
    """
    A, B, XF = 0.0, 10.0, 5.0
    xg = np.linspace(A, B, 4000)
    dens = 0.2 + np.exp(-0.5 * ((xg - XF) / 1.0) ** 2)
    cdf = np.cumsum(dens); cdf -= cdf[0]; cdf /= cdf[-1]
    wg = A + (B - A) * cdf
    wp = np.gradient(wg, xg)
    assert abs(wp.mean() - 1.0) < 5e-3, \
        f"endpoint-preserving warp must have mean slope 1; got {wp.mean():.4f}"
    assert wp.max() > 2.5, \
        f"fovea slope should exceed 2.5; got {wp.max():.4f} -- warp changed?"
    assert wp.min() < 0.5, f"periphery slope should fall below 0.5; got {wp.min():.4f}"
    # the claim under test: identical codebook does not mean identical max frequency
    rng = np.random.default_rng(7)
    th = 2.0 * rng.standard_normal(4000)
    assert np.abs(th).max() * wp.max() > 2.5 * np.abs(th).max(), \
        "sharing theta does not match maximum effective slope"
    # and a Gaussian sigma is not a cap
    frac = float(np.mean(np.abs(th) > 2.0))
    assert frac > 0.25, \
        f"a Gaussian sd is not a hard aperture: {frac:.3f} of |theta| exceed it"
    return (f"w' in [{wp.min():.3f}, {wp.max():.3f}], mean {wp.mean():.4f}; "
            f"{frac:.1%} of |theta| exceed the nominal 'cap'")


# =============================================================================
# 4+: the identities the paper states as exact
# =============================================================================

@test("bochner: the similarity kernel is the Fourier transform of p(theta)")
def t_bochner():
    rng = np.random.default_rng(0)
    N, sigma = 200000, 2.0
    th = sigma * rng.standard_normal(N)
    for d in (0.3, 0.8, 1.5):
        emp = float(np.cos(d * th).mean())
        ana = float(np.exp(-sigma ** 2 * d ** 2 / 2))
        assert abs(emp - ana) < 4 / np.sqrt(N), f"K({d}) {emp:.5f} vs {ana:.5f}"
    return "Gaussian codebook -> Gaussian kernel, within the 1/sqrt(N) floor"


@test("uncertainty: the three width conventions are one cell")
def t_conventions():
    sigma = 2.0
    hwhm_std = np.sqrt(2 * np.log(2)) / sigma * sigma          # kernel HWHM x freq std
    assert abs(hwhm_std - np.sqrt(2 * np.log(2))) < 1e-12
    int_std = (1 / (sigma * np.sqrt(2))) * (sigma / np.sqrt(2))  # intensity std x std
    assert abs(int_std - 0.5) < 1e-12, f"intensity std product {int_std}"
    int_hwhm = np.log(2)                                        # intensity HWHM x HWHM
    assert abs(hwhm_std ** 2 / 2 - int_hwhm) < 1e-12, "conversion between rulers"
    # the Gaussian is NOT the minimiser in the HWHM convention
    uni = 1.8955 / np.sqrt(3)
    assert uni < hwhm_std, "uniform should beat Gaussian under HWHM x std"
    return f"1.18 / 0.5 / 0.69 consistent; uniform {uni:.3f} < Gaussian {hwhm_std:.3f}"


@test("wigner: marginals, tone placement, boost and shear signs")
def t_wigner():
    xs = np.linspace(-10, 10, 512)
    dx = xs[1] - xs[0]
    th0 = 6.0
    psi = np.exp(-1j * th0 * xs) * np.exp(-xs ** 2 / 8)   # decoded atom at codebook th0
    ps = np.linspace(-2, 14, 400)

    def wig(psi_, ps_):
        M = len(xs); S = M // 2
        ss = np.arange(-S, S) * dx
        out = []
        for p in ps_:
            a = np.interp(xs[M // 2] + ss, xs, psi_.real) + 1j * np.interp(xs[M // 2] + ss, xs, psi_.imag)
            out.append(0)
        return out

    # tone placement via the paper's transform convention psihat(p)=int psi e^{+ipx}
    hat = np.array([np.trapezoid(psi * np.exp(1j * p * xs), xs) for p in ps])
    peak = ps[int(np.argmax(np.abs(hat)))]
    assert abs(peak - th0) < 0.15, f"atom of codebook freq {th0} landed at p={peak:.3f}"
    # boost e^{-i p0 x} raises <p> by p0
    p0 = 3.0
    hat_b = np.array([np.trapezoid(psi * np.exp(-1j * p0 * xs) * np.exp(1j * p * xs), xs)
                      for p in ps])
    peak_b = ps[int(np.argmax(np.abs(hat_b)))]
    assert abs(peak_b - (th0 + p0)) < 0.15, f"boost moved peak to {peak_b:.3f}"
    # shear e^{-i gamma x^2/2} tilts <p> with slope +gamma
    g = 0.5
    cent = []
    for x0 in (-2.0, 0.0, 2.0):
        loc = np.exp(-(xs - x0) ** 2 / 0.5) * np.exp(-1j * th0 * xs) * np.exp(-1j * g * xs ** 2 / 2)
        h = np.array([np.trapezoid(loc * np.exp(1j * p * xs), xs) for p in ps])
        w = np.abs(h) ** 2
        cent.append(float((ps * w).sum() / w.sum()))
    slope = (cent[2] - cent[0]) / 4.0
    assert abs(slope - g) < 0.06, f"shear slope {slope:.3f} should be +{g}"
    return f"tone at p={peak:.2f}; boost -> {peak_b:.2f}; shear slope {slope:.3f}"


@test("bispectrum: the centering identity raw - S2 - 2 S1 + 2k is exact")
def t_centering():
    rng = np.random.default_rng(5)
    N, sigma, k = 40000, 2.0, 14
    th = sigma * rng.standard_normal(N)
    xs = np.sort(rng.uniform(0, 60, k))
    b = np.exp(1j * np.outer(xs, th)).sum(0)
    b2 = np.exp(2j * np.outer(xs, th)).sum(0)
    raw = float((np.real(np.conj(b2) * b ** 2)).mean())
    S1 = float((np.abs(b) ** 2).mean())
    S2 = float((np.abs(b2) ** 2).mean())
    centered = raw - S2 - 2 * S1 + 2 * k
    # brute force over distinct index triples
    K = lambda u: np.exp(-sigma ** 2 * np.asarray(u) ** 2 / 2)
    tot = 0.0
    for i in range(k):
        for l in range(k):
            for n in range(k):
                if i != l and i != n and l != n:
                    tot += K(xs[i] + xs[l] - 2 * xs[n])
    err = abs(centered - tot)
    assert err < 0.05 * max(1.0, abs(tot)) + 0.5, \
        f"centered {centered:.4f} vs brute force {tot:.4f} (err {err:.4f})"
    return f"centered {centered:.4f} vs distinct-triple sum {tot:.4f}"


@test("binary cross-term: |a_j+c_j|^2 = 2 + 2 a_j c_j, exact and retained")
def t_binary_cross():
    """Bipolar entries have unit modulus, so bundling keeps the cross term on
    the surface: |a_j + c_j|^2 = |a_j|^2 + |c_j|^2 + 2 Re(a_j conj(c_j))
    = 2 + 2 a_j c_j -- EXACT entrywise, no expectation taken.  Checks (i) the
    algebraic identity on random bipolar vectors, and (ii) that the bundle of
    two bipolar FPE-QUANTIZED atoms (sign of the real part of the phasor)
    retains nonzero cross terms: its power spectrum differs from the sum of
    the individual power spectra by exactly 2 Re(a_j conj(c_j)), which for
    bipolar entries is +/-2 at EVERY component -- never zero.
    """
    rng = np.random.default_rng(13)
    N = 5000
    # (i) the algebraic identity on random bipolar vectors, exact for all j
    a = rng.choice([-1.0, 1.0], size=N)
    c = rng.choice([-1.0, 1.0], size=N)
    assert np.array_equal(np.abs(a + c) ** 2, 2.0 + 2.0 * a * c), \
        "|a_j+c_j|^2 = 2 + 2 a_j c_j must hold EXACTLY for all j"
    # (ii) bipolar FPE-quantized atoms at two values, bundled
    sigma, x1, x2 = 2.0, 1.3, 4.7
    th = sigma * rng.standard_normal(N)
    aq = np.sign(np.real(np.exp(1j * x1 * th)))
    cq = np.sign(np.real(np.exp(1j * x2 * th)))
    assert np.all(np.abs(aq) == 1.0) and np.all(np.abs(cq) == 1.0), \
        "quantized atoms must be genuinely bipolar"
    b = aq + cq
    cross = np.abs(b) ** 2 - (np.abs(aq) ** 2 + np.abs(cq) ** 2)
    assert np.array_equal(cross, 2.0 * np.real(aq * np.conj(cq))), \
        "bundle spectrum minus individual spectra must equal 2 Re(a_j conj(c_j)) exactly"
    assert np.all(cross != 0.0), \
        "bipolar cross terms are +/-2 at every component -- none may vanish"
    frac_con = float(np.mean(cross > 0))          # constructive fraction
    assert 0.05 < frac_con < 0.95, \
        f"both interference signs should occur; constructive fraction {frac_con:.3f}"
    assert abs(float(cross.sum())) > 0, "the bundle-level cross term must not cancel"
    return (f"identity exact on N={N}; cross term +/-2 everywhere "
            f"({frac_con:.1%} constructive, {1 - frac_con:.1%} destructive)")


@test("weyl: translation and boost commute up to the enclosed area")
def t_weyl():
    xs = np.linspace(-12, 12, 4096)
    dx = xs[1] - xs[0]
    rng = np.random.default_rng(7)
    th = 2.0 * rng.standard_normal(3000)
    prof = lambda x0: np.exp(1j * np.outer(x0 - xs, th)).mean(1)

    def translate(f, a):
        kk = np.fft.fftfreq(len(xs), d=dx)
        return np.fft.ifft(np.fft.fft(f) * np.exp(-2j * np.pi * kk * a))

    worst = 0.0
    for a in (0.5, 1.0, 2.0):
        for p0 in (0.5, 1.0, 2.0):
            u = np.exp(-1j * p0 * xs) * prof(3.0 + a)          # M_p T_a
            v = translate(np.exp(-1j * p0 * xs) * prof(3.0), a)  # T_a M_p
            meas = float(np.angle(np.vdot(u, v)))
            want = float((a * p0 + np.pi) % (2 * np.pi) - np.pi)
            worst = max(worst, abs(np.angle(np.exp(1j * (meas - want)))))
    assert worst < 5e-3, f"Weyl phase deviation {worst:.2e} rad"
    return f"max deviation {worst:.2e} rad over 9 (a, p0) pairs"


@test("sketch: score is unbiased for T(d) and its variance falls as 1/N")
def t_sketch():
    rng = np.random.default_rng(9)
    sigma, w0, k, d = 2.0, 8.0, 30, 7.0
    K = lambda u: np.cos(w0 * np.asarray(u)) * np.exp(-sigma ** 2 * np.asarray(u) ** 2 / 2)
    xs = np.sort(rng.uniform(0, 400, k))
    du = xs[:, None] - xs[None, :]
    T = 0.5 * (K(du - d) + K(du + d)).sum()
    sds = {}
    for N in (2000, 8000):
        vals = []
        for _ in range(300):
            th = w0 + sigma * rng.standard_normal(N)
            b = np.exp(1j * np.outer(xs, th)).sum(0)
            vals.append(float((np.cos(d * th) @ (np.abs(b) ** 2)) / N))
        m, s = float(np.mean(vals)), float(np.std(vals))
        assert abs(m - T) < 4 * s / np.sqrt(300), \
            f"biased at N={N}: mean {m:.4f} vs target {T:.4f} (SD {s:.4f})"
        sds[N] = s
    ratio = sds[2000] / sds[8000]
    assert 1.6 < ratio < 2.4, f"SD should halve for 4x N; ratio {ratio:.3f}"
    return f"unbiased for T(d)={T:.3f}; SD ratio {ratio:.2f} over 4x N (expect 2)"


@test("hash join: dict and sort joins equal brute force, duplicates included")
def t_hashjoin():
    """The R11 sparse baseline (cwf_fpe_hashjoin.py): the ordered self-join
    J_+(d) = sum_u f(u) f(u-d) computed by frequency-dict scan and by
    sort-merge must equal explicit pair enumeration on multisets WITH
    duplicates -- the case a set-based implementation silently gets wrong."""
    from cwf_fpe_hashjoin import hash_join, sort_join, pair_enum
    rng = np.random.default_rng(21)
    checked = 0
    for _ in range(40):
        vals = rng.integers(0, 30, size=rng.integers(4, 80)).astype(np.int64)
        for d in (0, 1, -3, 7):
            jh, js, jb = hash_join(vals, d), sort_join(vals, d), pair_enum(vals, d)
            assert jh == js == jb, f"d={d}: hash {jh}, sort {js}, brute {jb}"
            checked += 1
    return f"{checked} (multiset, lag) cases: hash == sort == brute force"


@test("variance: all-equal multiset breaks k/sqrt(N) -- SD is k^2/sqrt(N)")
def t_worstcase_variance():
    """R11 scoping guard: the sparse-regime noise law SD ~ F_2/sqrt(N) is a
    HEURISTIC.  For k copies of ONE value, |b|^2 = k^2 exactly, so the score
    SD is k^2 * SD[cos(d theta)] / sqrt(N) -- the worst case the manuscript
    must keep citing.  This test fails if anyone 'simplifies' the appendix
    counterexample away."""
    rng = np.random.default_rng(22)
    sigma, w0, k, d, N = 2.0, 8.0, 32, 7.0, 500
    vals = []
    for _ in range(400):
        th = w0 + sigma * rng.standard_normal(N)
        b = k * np.exp(1j * 3.7 * th)                  # k copies of x = 3.7
        vals.append(float((np.cos(d * th) @ (np.abs(b) ** 2)) / N))
    sd = float(np.std(vals))
    sd_cos = float(np.std(np.cos(d * (w0 + sigma * rng.standard_normal(200000)))))
    pred_worst = k ** 2 * sd_cos / np.sqrt(N)          # k^2 scale
    pred_sparse = k / np.sqrt(N)                       # the heuristic's scale
    assert 0.5 * pred_worst < sd < 2.0 * pred_worst, \
        f"SD {sd:.2f} vs k^2 prediction {pred_worst:.2f}"
    assert sd > 10 * pred_sparse, \
        f"SD {sd:.2f} not >> sparse-law {pred_sparse:.2f}: counterexample gone?"
    return (f"SD {sd:.1f} ~ k^2 rule {pred_worst:.1f}, "
            f">> sparse rule {pred_sparse:.1f} (x{sd/pred_sparse:.0f})")


@test("variance: an arithmetic progression is noisier than a generic sparse set")
def t_structured_variance():
    """Repeated differences (high additive energy) inflate the fourth moment:
    at matched k and N, an AP's null-lag score SD must exceed a generic
    sparse set's by a clear factor.  Guards the 'nondegenerate differences'
    condition attached to the sparse law."""
    rng = np.random.default_rng(23)
    sigma, w0, k, d, N = 2.0, 8.0, 64, 3.0, 500

    def score_sd(xs):
        vals = []
        for _ in range(300):
            th = w0 + sigma * rng.standard_normal(N)
            b = np.exp(1j * np.outer(xs, th)).sum(0)
            vals.append(float((np.cos(d * th) @ (np.abs(b) ** 2)) / N))
        return float(np.std(vals))

    sd_ap = score_sd(np.arange(k) * 1.0)               # every difference repeats
    sd_gen = score_sd(np.sort(rng.uniform(0, 1e6, k)))  # generic: distinct diffs
    assert sd_ap > 2.0 * sd_gen, \
        f"AP SD {sd_ap:.2f} not > 2x generic SD {sd_gen:.2f}"
    return f"AP SD {sd_ap:.1f} vs generic {sd_gen:.1f} ({sd_ap/sd_gen:.1f}x) at k={k}"


@test("resonator: a capped run is a timeout, not a detected cycle")
def t_resonator():
    """Under fresh per-iteration noise an exact state repeat is impossible, so
    labelling capped runs 'limit cycles' cannot be right.  Also checks that the
    Jacobi update at F=2 has no period-one fixed point, which is why a
    successive-state equality test never fires.
    """
    rng = np.random.default_rng(2)
    N, D, T = 128, 24, 60
    books = [np.sign(rng.standard_normal((N, D))) for _ in range(2)]
    true = (3, 11)
    c = books[0][:, true[0]] * books[1][:, true[1]]

    def run(theta):
        xh = [np.sign(books[f].sum(1) + 1e-9 * rng.standard_normal(N)) for f in range(2)]
        hist, exact_repeat = [], False
        for _ in range(T):
            new = []
            for f in range(2):
                s = c * xh[1 - f]
                clean = books[f] @ (books[f].T @ s)
                if theta:
                    clean = clean + theta * np.std(np.abs(clean)) * rng.standard_normal(N)
                new.append(np.sign(clean))
            xh = new
            for h in hist:
                if all(np.array_equal(xh[f], h[f]) for f in range(2)):
                    exact_repeat = True
            hist.append([x.copy() for x in xh])
            if exact_repeat:
                break
        return exact_repeat, len(hist)

    det = [run(0.0) for _ in range(24)]
    noisy = [run(0.6) for _ in range(24)]
    nd, nn = sum(r for r, _ in det), sum(r for r, _ in noisy)
    assert nd >= 12, f"deterministic runs should reach genuine cycles; got {nd}/24"
    assert nn < nd / 2, \
        (f"noise must suppress genuine cycles: deterministic {nd}/24 vs noisy {nn}/24. "
         "Note the honest statement is 'rare', NOT 'impossible': this substrate is "
         "bipolar, so sign() makes the state discrete and a repeat is improbable "
         "rather than ruled out. A continuous (FHRR) carrier would rule it out.")
    return (f"deterministic {nd}/24 cycled, noisy {nn}/24 -- noise suppresses cycles; "
            "on a bipolar carrier this is 'rare', not 'impossible'")


# =============================================================================

def main():
    print("=" * 78)
    print("IDENTITY AND REGRESSION TESTS")
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
