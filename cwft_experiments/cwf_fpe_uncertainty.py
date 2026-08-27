"""
cwf_fpe_uncertainty.py -- FPE as a computational phase space: the Fourier-Gabor
uncertainty principle of Fractional Power Encoding.  (VSA_CWFT_NOTE.md sec.5;
the VSA entry point.)

Fractional Power Encoding (FPE / Spatial Semantic Pointers) encodes a continuous
value x as a vector of unit phasors with phases that scale linearly with x:
    phi(x)_k = exp(i x theta_k),   theta_k ~ p(theta)   (random fixed frequencies).
The similarity kernel is
    K(Delta) = (1/N) Re <phi(x), phi(x+Delta)> = (1/N) sum_k cos(Delta theta_k)
            -> E_theta[cos(Delta theta)] = (real) FOURIER TRANSFORM of p(theta).
So x (the encoded value) and theta (frequency) are CONJUGATE variables, and the
encoding is a coherent-state-like object on a computational phase space.  The CWFT
reading: phi(x) is the computon field; theta is computational momentum; hbar_c is
the phase-space cell; and the resolution<->bandwidth tradeoff IS the position-
momentum uncertainty (postulate P5).  None of this is framed as an uncertainty
principle in the VSA literature (white space), though it IS the Fourier limit.

FIRST RESULTS (verified numerically; honest fences noted):
  [1] Fourier-pair: K(Delta) = FT(p(theta)) -- CLEAN.  Gaussian freqs -> Gaussian
      kernel e^{-sigma^2 Delta^2/2}; uniform -> sinc; Laplace -> Lorentzian (dev
      ~1/sqrt(N)).  x and theta are conjugate.
  [2] Uncertainty: resolution Delta_x (kernel HWHM) * bandwidth Delta_theta (freq
      std) = const ~1.19 -- CLEAN reciprocal, the Gabor limit; that constant is
      hbar_c (the phase-space cell).  NOTE: 'Gaussian = the minimum-uncertainty
      coherent state' is an ANALYTIC claim (sinc->infinite variance) NOT cleanly
      verified at finite N (HWHM favors uniform; RMS is noise-floor-corrupted) --
      a direction, not a result.
  [3] Phase-space HOLDING capacity (bundling): k_max ~ 0.07 N (matches N/(2 ln L))
      -- the wave holds ~0.07N phase-space cells in superposition; area/hbar_c ~ N.
      This RE-DESCRIBES the known bundling capacity (new content = the phase-space
      framing).  [The clean-decode 'capacity=N' framing was a BUG -- self-similarity
      is always max; the N-bound is a superposition bound, not a decode bound.]

CPU, numpy, seeded.  Writes cwf_fpe_uncertainty_results.json.
"""

import json
import numpy as np

RNG = np.random.default_rng(0)


def freqs(dist, N, scale):
    if dist == "gaussian":
        return RNG.normal(0, scale, N)
    if dist == "uniform":
        return RNG.uniform(-scale, scale, N)
    if dist == "laplace":
        return RNG.laplace(0, scale, N)
    raise ValueError(dist)


def kernel(theta, deltas):
    return np.cos(np.outer(deltas, theta)).mean(axis=1)


def hwhm(theta, dmax=50.0, n=4000):
    """Half-width at half-max of the kernel (a simple resolution measure)."""
    d = np.linspace(0, dmax, n)
    K = kernel(theta, d)
    below = np.where(K < 0.5)[0]
    return float(d[below[0]]) if len(below) else dmax


def rms_width(theta, dmax=60.0, n=20000):
    """RMS (second-moment) width of the kernel ENERGY -- the proper conjugate
    measure for the Heisenberg-Gabor product (Gaussian is the unique minimum;
    heavy-tailed kernels like sinc diverge)."""
    d = np.linspace(0, dmax, n)
    K = kernel(theta, d) ** 2
    return float(np.sqrt(np.sum(d**2 * K) / (np.sum(K) + 1e-12)))


# ---------------------------------------------------------------------------
def part1():
    print("[1] Fourier-pair: empirical kernel vs analytic FT(p(theta))")
    N = 4000
    out = {}
    for dist, scale, analytic in [
        ("gaussian", 1.0, lambda d: np.exp(-1.0**2 * d**2 / 2)),
        ("uniform", 2.0, lambda d: np.sinc(2.0 * d / np.pi)),       # np.sinc(x)=sin(pi x)/(pi x)
        ("laplace", 1.0, lambda d: 1.0 / (1 + 1.0**2 * d**2)),
    ]:
        th = freqs(dist, N, scale)
        d = np.linspace(0, 8, 400)
        Kemp = kernel(th, d)
        Kana = analytic(d)
        err = float(np.max(np.abs(Kemp - Kana)))
        out[dist] = {"scale": scale, "max_dev": err}
        print(f"  {dist:>9} (scale={scale}): max|empirical-analytic| = {err:.3f} "
              f"(~1/sqrt(N)={1/np.sqrt(N):.3f})")
    print("  -> the similarity kernel IS the Fourier transform of the frequency"
          " distribution.")
    return out


def part2():
    print("\n[2] uncertainty: resolution (kernel HWHM) x bandwidth (freq std)")
    N = 8000
    out = {}
    # reciprocal relation: scale the SAME distribution, resolution ~ 1/bandwidth
    print("  Gaussian, sweep bandwidth sigma -> resolution should be ~1/sigma:")
    rows = []
    for sig in [0.5, 1.0, 2.0, 4.0, 8.0]:
        th = freqs("gaussian", N, sig)
        res = hwhm(th)
        rows.append(dict(sigma=sig, resolution=res, product=res * sig))
        print(f"    sigma={sig:>4}: resolution={res:.3f}  res*sigma={res*sig:.3f}")
    out["gaussian_sweep"] = rows
    print(f"  -> res*sigma ~ constant ({np.mean([r['product'] for r in rows]):.2f}):"
          f" the reciprocal resolution-bandwidth law (the uncertainty).")
    # the uncertainty product is O(1) for ALL shapes (universal); which shape is
    # the strict minimum is convention-dependent and NOT cleanly resolved at
    # finite N (HWHM favors uniform; the RMS measure is corrupted by the 1/sqrt(N)
    # noise-floor tail).  Report honestly; the 'Gaussian = coherent state' minimum
    # is an ANALYTIC claim (sinc has infinite variance), a direction, not verified here.
    print("  HWHM x freq-std product across shapes (the uncertainty is O(1) for all):")
    comp = {}
    for dist, scale in [("gaussian", 2.0), ("uniform", 2.0), ("laplace", 2.0)]:
        th = freqs(dist, N, scale)
        res = hwhm(th); bw = float(np.std(th)); prod = res * bw
        comp[dist] = {"hwhm": res, "bandwidth": bw, "product": prod}
        print(f"    {dist:>9}: HWHM={res:.3f}  bandwidth(std)={bw:.3f}  product={prod:.3f}")
    out["distributions"] = comp
    print("  -> products all O(1) (~1.1-1.4): the uncertainty is universal across")
    print("     frequency shapes.  HONEST: 'Gaussian = the minimum-uncertainty")
    print("     coherent state' is an ANALYTIC claim (sinc->infinite variance), NOT")
    print("     cleanly verified at finite N here -- a direction, not a result.")
    return out


def part3():
    """Phase-space HOLDING capacity: how many resolved phase-space cells can the
    wave hold in SUPERPOSITION (bundle) and still read out?  (The clean-decode
    'capacity' is not N -- that was the bug; the N-bound is a superposition bound.)
    Bundle k well-separated FPE cells, read out by thresholded similarity; find
    k_max (95% recall, low false-positive).  k_max should scale ~ N."""
    print("\n[3] phase-space HOLDING capacity (bundling): k_max vs N")
    out = {}
    sigma, spacing = 4.0, 3.0                    # cells well-separated (K(spacing)~0)
    for N in [128, 256, 512, 1024]:
        th = freqs("gaussian", N, sigma)
        L = 4 * N                                # library of L candidate cells
        xs = np.arange(L) * spacing
        Phi = np.exp(1j * np.outer(xs, th))      # (L,N)
        kmax = 0
        for k in range(2, L, max(2, N // 16)):
            rec, fp = [], []
            for trial in range(8):
                S = RNG.choice(L, size=k, replace=False)
                b = Phi[S].sum(0)
                sim = np.real(Phi @ b.conj()) / N
                pred = sim > 0.5
                true = np.zeros(L, bool); true[S] = True
                rec.append((pred & true).sum() / k)
                fp.append((pred & ~true).sum() / max(1, (~true).sum()))
            if np.mean(rec) >= 0.95 and np.mean(fp) < 0.01:
                kmax = k
            else:
                break
        out[N] = {"k_max": kmax, "k_max_over_N": kmax / N}
        print(f"  N={N:>5}: k_max={kmax:>4}  (k_max/N = {kmax/N:.2f})")
    ratios = [out[N]["k_max_over_N"] for N in out]
    print(f"  -> k_max/N ~ constant ({np.mean(ratios):.2f}): the wave holds ~{np.mean(ratios):.2f}N")
    print(f"     phase-space cells in superposition.  Phase-space area accessible")
    print(f"     (#cells x hbar_c) scales with the dimension N (re-describes the known")
    print(f"     bundling capacity; the NEW framing is cells = area / hbar_c).")
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("FPE as a computational phase space: the Fourier-Gabor uncertainty")
    print("=" * 74)
    P1 = part1(); P2 = part2(); P3 = part3()
    with open("cwf_fpe_uncertainty_results.json", "w") as f:
        json.dump({"part1_fourier_pair": P1, "part2_uncertainty": P2,
                   "part3_phase_space_capacity": P3}, f, indent=2)
    print("\nwrote cwf_fpe_uncertainty_results.json")
