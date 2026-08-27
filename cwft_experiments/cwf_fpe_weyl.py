#!/usr/bin/env python3
"""THE WEYL RELATION on the FPE substrate: measure the symplectic phase.

Translation T_a (native VSA binding with e^{i a theta}) and boost M_p (modulation of
the decoded profile by e^{i p x}) are the two conjugate Heisenberg--Weyl flows.
Their symplectic content is the COMMUTATION PHASE: the two orderings differ by the
global phase e^{i a p} (the phase-space area cocycle),

    T_a M_p = e^{-i a p} M_p T_a .

We realize both orders on the decoded field (T_a implemented natively, by binding
the hypervector, before decoding) and measure arg <T_a M_p psi, M_p T_a psi> over a
grid of (a, p).  This verifies an operator identity -- the value is that the
symplectic structure asserted in sec. 4.3 is exhibited on the substrate, not just
named.  Seeded, CPU.  Writes cwf_fpe_weyl_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
X0 = 3.0
GRID = np.linspace(-8, 20, 1400)


def profile(theta, x0, xs):
    return np.exp(1j * np.outer(x0 - xs, theta)).mean(1)


def translate(f, a, xs):
    """Numerically translate a sampled field by a: (T_a f)(x) = f(x - a).

    Done by Fourier shift, INDEPENDENTLY of the identity under test.  The earlier
    version of this script built the second operator ordering from the analytic
    relation e^{ip(x-a)} psi(x-a) = e^{-ipa} u(x), so the measured phase was
    a*p by floating-point construction and the reported 1e-16 was roundoff, not
    evidence.  Nothing below uses the commutator it is trying to verify.
    """
    n = len(xs)
    dx = xs[1] - xs[0]
    k = np.fft.fftfreq(n, d=dx)
    return np.fft.ifft(np.fft.fft(f) * np.exp(-2j * np.pi * k * a))


def main():
    rng = np.random.default_rng(7)
    theta = SIGMA * rng.standard_normal(N)
    rows, devs = [], []
    for a in [0.5, 1.0, 2.0, 3.5]:
        for p in [0.5, 1.0, 2.0, 3.0]:
            # Boost sign follows the manuscript: M_p psi = e^{-i p x} psi, which
            # raises <p> by p (the earlier script used e^{+ipx}, the opposite).
            # order 1: M_p T_a psi -- bind (native translation x0 -> x0+a), decode,
            #          then modulate.
            u = np.exp(-1j * p * GRID) * profile(theta, X0 + a, GRID)
            # order 2: T_a M_p psi -- decode at x0, modulate, then TRANSLATE the
            #          modulated field numerically.  Expect v = e^{+i a p} u.
            g = np.exp(-1j * p * GRID) * profile(theta, X0, GRID)
            v = translate(g, a, GRID)
            meas = float(np.angle(np.vdot(u, v)))          # expect +a*p (mod 2pi)
            want = float((a * p + np.pi) % (2 * np.pi) - np.pi)
            dev = abs(np.angle(np.exp(1j * (meas - want))))
            rows.append(dict(a=a, p=p, measured=meas, expected=want, dev=dev))
            devs.append(dev)
    out = dict(params=dict(N=N, sigma=SIGMA, x0=X0),
               rows=rows, max_dev_rad=float(np.max(devs)))
    print(f"Weyl commutation phase over {len(rows)} (a,p) points: "
          f"max |measured - a*p| = {out['max_dev_rad']:.2e} rad")
    with open("cwf_fpe_weyl_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_weyl_results.json")


if __name__ == "__main__":
    main()
