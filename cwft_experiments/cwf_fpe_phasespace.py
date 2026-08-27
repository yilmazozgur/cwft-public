"""
cwf_fpe_phasespace.py -- completing the story: the full position-momentum phase
space, and the two interference-suppressing regimes.  (VSA_CWFT_NOTE sec.5; foundational
rounding-out, does NOT change the no-separation verdict.)

FPE encodes a value x with phases e^{i x theta}; the conjugate coordinate is the
frequency theta (= 'momentum' p, the Fourier-dual of x).  A state is a blob in the
(x, p) plane:
  - POSITION state: full bandwidth -> sharp in x, broad in p.
  - MOMENTUM state: narrow bandwidth -> broad in x, sharp in p.
The resolution<->bandwidth tradeoff is the position-momentum uncertainty (the
1/2 bound holds in the intensity-std convention for the continuum profile; the
HWHM widths here show the same conjugate tradeoff).  The two conjugate flows:
  - translation in x  = binding with e^{i a theta}  (the native FPE shift),
  - boost in p        = modulation psi(x) -> e^{-i p0 x} psi(x)
    (the sign pairs with the decoder e^{-i x theta}; p = theta convention,
     see cwf_fpe_wigner_convention.py -- the moments below report <p> on that
     axis, so a decoded atom at carrier omega0 sits at p = +omega0).

Two distinct interference-suppressing regimes (NOT one 'classical limit'):
  (a) bandwidth -> infinity: the kernel -> a delta; distinct values become
      orthogonal TO THE SIMILARITY READOUT (the codes decorrelate) -- the
      lookup-table / orthogonal-code regime.  A bundle of such codes remains
      a coherent superposition; orthogonality is not decoherence.
  (b) coarse-graining the Wigner past the cell gives the (non-negative)
      HUSIMI distribution -- interference is visible only when phase space is
      resolved finer than the cell.

CPU, numpy/scipy, seeded.  Writes cwf_fpe_phasespace_results.json.
"""

import json
import numpy as np
from scipy.ndimage import gaussian_filter

RNG = np.random.default_rng(0)


def freqs(N, omega0, sigma):
    th = omega0 + sigma * RNG.standard_normal(N)
    return th[th > 0]


def profile(theta, x0, xs, boost=0.0):
    """psi(x) for a value x0 (optionally momentum-boosted by `boost`)."""
    psi = np.exp(1j * np.outer(x0 - xs, theta)).mean(1)   # (1/N) sum_k e^{i(x0-x)theta}
    return psi * np.exp(1j * boost * xs)


def _hwhm(dist, grid):
    """half-width at half-max around the peak (robust to the finite-N floor)."""
    pk = int(np.argmax(dist)); half = dist[pk] / 2
    l = pk
    while l > 0 and dist[l] > half:
        l -= 1
    r = pk
    while r < len(dist) - 1 and dist[r] > half:
        r += 1
    return float((grid[r] - grid[l]) / 2)


def moments(psi, xs):
    """(<x>, Dx) from |psi|^2, (<p>, Dp) from |FFT psi|^2; widths = HWHM (the
    second moment is corrupted by the 1/sqrt(N) floor over a wide window).
    <p> is reported on the p = theta axis: the decoder pairs x with -theta, so
    the np.fft frequency axis is negated (pure-tone unit test:
    cwf_fpe_wigner_convention.py)."""
    dx = xs[1] - xs[0]
    Px = np.abs(psi) ** 2
    mx = (xs * Px).sum() / Px.sum(); Dx = _hwhm(Px, xs)
    ps = np.fft.fftshift(2 * np.pi * np.fft.fftfreq(len(xs), dx))
    Pp = np.fft.fftshift(np.abs(np.fft.fft(psi)) ** 2)
    mp = (ps * Pp).sum() / Pp.sum(); Dp = _hwhm(Pp, ps)
    return float(mx), float(Dx), float(-mp), float(Dp)


def wigner_ville(psi):
    M = len(psi); W = np.zeros((M, M), dtype=complex)
    for n in range(M):
        k = min(n, M - 1 - n); m = np.arange(-k, k + 1)
        r = np.zeros(M, dtype=complex); r[m % M] = psi[n + m] * np.conj(psi[n - m])
        W[n] = np.fft.fft(r)
    return np.real(W)


def negativity(W):
    a = np.abs(W).sum()
    return float(-W[W < 0].sum() / a) if a > 0 else 0.0


if __name__ == "__main__":
    print("=" * 74)
    print("THE FULL PHASE SPACE + the two interference-suppressing regimes")
    print("=" * 74)
    N, omega0 = 4000, 16.0                    # high carrier: no positive-freq truncation
    xs = np.linspace(-12, 12, 480)
    out = {}

    # ---- Part 1: position / momentum / coherent states ----
    print("\n[1] phase-space states: Dx (position) and Dp (momentum) vs bandwidth")
    print(f"    {'bandwidth sigma':>16} | {'Dx':>6} | {'Dp':>6} | {'Dx*Dp':>7} | state")
    rows = []
    for sig in [0.5, 1.0, 2.0, 4.0]:
        th = freqs(N, omega0, sig)
        _, Dx, _, Dp = moments(profile(th, 0.0, xs), xs)
        kind = "momentum-localized" if sig <= 1 else ("position-localized" if sig >= 4 else "balanced")
        rows.append(dict(sigma=sig, Dx=Dx, Dp=Dp, prod=Dx * Dp))
        print(f"    {sig:>16} | {Dx:>6.3f} | {Dp:>6.3f} | {Dx*Dp:>7.3f} | {kind}")
    prod = np.mean([r["prod"] for r in rows])
    out["part1_states"] = rows
    print(f"    -> Dx down, Dp up with bandwidth (the conjugate tradeoff); Dx*Dp ~ O(1)")
    print(f"       ({prod:.2f}, shape-dependent as in sec.5b): narrow bandwidth = momentum")
    print(f"       state (sharp p, broad x), wide bandwidth = position state. The")
    print(f"       resolution<->bandwidth law IS the position-momentum uncertainty.")

    # ---- Part 1b: the two conjugate flows ----
    print("\n[1b] conjugate phase-space flows (translation in x, boost in p):")
    th = freqs(N, omega0, 2.0)
    mx0, _, mp0, _ = moments(profile(th, 0.0, xs), xs)
    # translation: bind with e^{i a theta} <=> encode value x0=a -> shifts <x>
    mxT, _, mpT, _ = moments(profile(th, 3.0, xs), xs)        # x0 = 3
    # boost: multiply psi by e^{-i p0 x} -> raises <p> by p0 (p = theta axis)
    mxB, _, mpB, _ = moments(profile(th, 0.0, xs, boost=-4.0), xs)
    print(f"    base state:         <x>={mx0:+.2f}, <p>={mp0:.2f}")
    print(f"    translate (x0=3):   <x>={mxT:+.2f} (shifted ~+3), <p>={mpT:.2f} (unchanged)")
    print(f"    boost (p0=4):       <x>={mxB:+.2f} (unchanged), <p>={mpB:.2f} (shifted ~+4)")
    print(f"    -> translation (binding) moves the blob in x; boost (modulation) in p:")
    print(f"       the two conjugate flows of a genuine phase space.")
    out["part1b_flows"] = {"base_x": mx0, "trans_x": mxT, "base_p": mp0, "boost_p": mpB}

    # ---- Part 2a: classical limit via bandwidth -> infinity (orthogonality) ----
    print("\n[2a] bandwidth -> infinity: distinct values become orthogonal codes")
    Delta = 1.0
    P2a = []
    for sig in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        th = freqs(N, omega0, sig)
        # overlap envelope of two values separated by Delta (magnitude; the signed
        # value just carries the carrier phase cos(omega0 Delta))
        ov = float(np.abs(np.vdot(np.exp(1j * 0.0 * th), np.exp(1j * Delta * th))) / len(th))
        P2a.append(dict(sigma=sig, overlap=ov))
        print(f"    bandwidth sigma={sig:>5}: |overlap| K({Delta})={ov:.3f}")
    out["part2a_orthogonality"] = P2a
    print(f"    -> overlap -> 0 as bandwidth grows: distinct values become perfectly")
    print(f"       distinguishable to the similarity readout -- the orthogonal-code")
    print(f"       (lookup-table) regime. (A bundle of orthogonal codes is still a")
    print(f"       coherent superposition: orthogonality is not decoherence.)")

    # ---- Part 2b: classical limit via coarse-graining (Wigner -> Husimi) ----
    print("\n[2b] coarse-graining the Wigner -> the (positive) Husimi = classical limit")
    th = freqs(N, omega0, 3.0)
    cat = profile(th, -3.0, xs) + profile(th, 3.0, xs)        # a cat state (negativity)
    W = wigner_ville(cat)
    P2b = []
    for s in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]:
        Ws = W if s == 0 else gaussian_filter(W, sigma=s, mode="nearest")
        P2b.append(dict(smooth=s, neg=negativity(Ws)))
        print(f"    coarse-grain width={s:>5}: Wigner negativity = {negativity(Ws):.4f}")
    out["part2b_husimi"] = P2b
    print(f"    -> negativity -> 0 as the coarse-graining reaches the hbar_c cell (the")
    print(f"       Husimi is non-negative): the wave (interference) is visible only when")
    print(f"       you resolve phase space finer than the cell. (A distinct mechanism")
    print(f"       from the ensemble-dephasing control of the Wigner experiment.)")

    with open("cwf_fpe_phasespace_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_phasespace_results.json")
