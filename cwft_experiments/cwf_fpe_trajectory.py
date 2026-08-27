#!/usr/bin/env python3
"""JOINT POSITION-MOMENTUM ENCODING: TRAJECTORIES AS PHASE-SPACE OBJECTS.

A 2-D FPE over the conjugate pair (x, p) -- atom phi(x,p)_j = exp(i(x th_j + p om_j))
with independent random codebooks th, om -- lets ONE hypervector store a whole
sampled trajectory as a phase-space measure.  Three demonstrations:

 (1) KNOWN-CASE CHECK: a single encoded phase-space point decodes to a peak at the
     right (x, p) location (validates the 2-D decode before anything else).
 (2) PORTRAIT: a bundled harmonic-oscillator orbit decodes to its energy shell (a
     ring) in the (x, p) plane, read from the single bundle; ring concentration
     is quantified against the uniform baseline.
 (3) TIME-REVERSAL SPLIT (headline): a trajectory traversed backward visits the
     SAME positions with MIRRORED momenta.  Position-only bundles of forward and
     reversed runs are identical (similarity 1 by construction); joint (x,p)
     bundles separate.  Variants: linear sweep (left-to-right vs right-to-left)
     and oscillator half-orbit; plus a momentum query at a single position
     ("which way is it moving at x=0?") answered from the bundle alone.

Honest size: the machinery is 2-D FPE (standard in the SSP literature for spatial
coordinates); the content is the CHOICE to make the second axis the conjugate
variable p, which turns a bundle into a phase-space object for dynamics.
Seeded, CPU.  Writes cwf_fpe_trajectory_results.json.
"""
import json

import numpy as np

N = 4000
SIG_X = 2.0     # codebook bandwidth for the x axis
SIG_P = 2.0     # codebook bandwidth for the p axis
SEEDS = 3


def codebooks(rng):
    return SIG_X * rng.standard_normal(N), SIG_P * rng.standard_normal(N)


def atom(x, p, th, om):
    return np.exp(1j * (x * th + p * om))


def decode2d(b, th, om, xs, ps):
    """psi(x,p) = (1/N) sum_j b_j exp(-i(x th_j + p om_j)) on the xs x ps grid."""
    Ex = np.exp(-1j * np.outer(xs, th))
    Ep = np.exp(-1j * np.outer(ps, om))
    return (Ex * b[None, :]) @ Ep.T / len(th)


def cossim(a, b):
    return float(np.real(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    out = {"params": dict(N=N, sig_x=SIG_X, sig_p=SIG_P, seeds=SEEDS)}

    # ---------------- (1) known-case check: single point decodes in place -----
    print("(1) known-case check: single phase-space point")
    rng = np.random.default_rng(0)
    th, om = codebooks(rng)
    x0, p0 = 2.0, -1.5
    xs = np.linspace(-4, 4, 161)
    ps = np.linspace(-4, 4, 161)
    W = np.abs(decode2d(atom(x0, p0, th, om), th, om, xs, ps)) ** 2
    i, j = np.unravel_index(np.argmax(W), W.shape)
    err = float(np.hypot(xs[i] - x0, ps[j] - p0))
    ok = err < 0.1
    print(f"    peak at ({xs[i]:.2f},{ps[j]:.2f}) vs true ({x0},{p0}): "
          f"err={err:.3f}  {'PASS' if ok else 'FAIL'}")
    out["known_case"] = dict(err=err, ok=bool(ok))

    # ---------------- (2) portrait: oscillator orbit -> energy shell ----------
    print("(2) portrait: harmonic-oscillator orbit decodes to its energy shell")
    A, T = 3.0, 200
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    xs = np.linspace(-5, 5, 141)
    ps = np.linspace(-5, 5, 141)
    band = 0.4
    concs = []
    for s in range(SEEDS):
        rng = np.random.default_rng(10 + s)
        th, om = codebooks(rng)
        b = sum(atom(A * np.cos(tt), -A * np.sin(tt), th, om) for tt in t)
        W = np.abs(decode2d(b, th, om, xs, ps)) ** 2
        R = np.hypot(*np.meshgrid(xs, ps, indexing="ij"))
        in_band = np.abs(R - A) < band
        mass_frac = float(W[in_band].sum() / W.sum())
        area_frac = float(in_band.mean())
        concs.append(mass_frac / area_frac)
    conc = float(np.mean(concs))
    print(f"    ring concentration (mass frac / area frac) = {conc:.1f}x"
          f"  (band |r-A|<{band}; {SEEDS} seeds: "
          + ", ".join(f"{c:.1f}" for c in concs) + ")")
    out["portrait"] = dict(A=A, T=T, band=band, concentration=conc,
                           per_seed=[float(c) for c in concs])

    # ---------------- (3) time-reversal split --------------------------------
    print("(3) time-reversal split: same positions, mirrored momenta")
    res = {}
    # (3a) linear sweep left-to-right vs right-to-left
    v, Tl = 1.5, 60
    xs_line = np.linspace(0, 10, Tl)
    sims_pos, sims_joint = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(20 + s)
        th, om = codebooks(rng)
        b_fwd = sum(atom(x, +v, th, om) for x in xs_line)
        b_rev = sum(atom(x, -v, th, om) for x in xs_line[::-1])
        bx_fwd = sum(np.exp(1j * x * th) for x in xs_line)
        bx_rev = sum(np.exp(1j * x * th) for x in xs_line[::-1])
        sims_pos.append(cossim(bx_fwd, bx_rev))
        sims_joint.append(cossim(b_fwd, b_rev))
    res["sweep"] = dict(v=v, T=Tl,
                        sim_position_only=float(np.mean(sims_pos)),
                        sim_joint=float(np.mean(sims_joint)),
                        sim_joint_sd=float(np.std(sims_joint)))
    print(f"    linear sweep  : position-only sim = {np.mean(sims_pos):.3f}, "
          f"joint sim = {np.mean(sims_joint):+.3f} +- {np.std(sims_joint):.3f}")

    # (3b) oscillator half-orbit (lower vs upper semicircle)
    Th = 100
    th_t = np.linspace(0, np.pi, Th)
    sims_pos, sims_joint = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(30 + s)
        th, om = codebooks(rng)
        xf = A * np.cos(th_t)
        pf = -A * np.sin(th_t)          # forward: lower semicircle
        b_fwd = sum(atom(x, p, th, om) for x, p in zip(xf, pf))
        b_rev = sum(atom(x, -p, th, om) for x, p in zip(xf[::-1], pf[::-1]))
        bx_fwd = sum(np.exp(1j * x * th) for x in xf)
        bx_rev = sum(np.exp(1j * x * th) for x in xf[::-1])
        sims_pos.append(cossim(bx_fwd, bx_rev))
        sims_joint.append(cossim(b_fwd, b_rev))
    res["half_orbit"] = dict(T=Th,
                             sim_position_only=float(np.mean(sims_pos)),
                             sim_joint=float(np.mean(sims_joint)),
                             sim_joint_sd=float(np.std(sims_joint)))
    print(f"    half-orbit    : position-only sim = {np.mean(sims_pos):.3f}, "
          f"joint sim = {np.mean(sims_joint):+.3f} +- {np.std(sims_joint):.3f}")

    # (3d) order-aware baseline: a time-stamped (x,t) code also separates
    # reversal (VSA's standard sequence tools are order-aware); what the (x,p)
    # code adds is the dynamics-native readout, not the bare separation.
    sims_xt = []
    for s in range(SEEDS):
        rng = np.random.default_rng(60 + s)
        th, tau = codebooks(rng)          # tau: independent time codebook
        xf = A * np.cos(th_t)
        b_fwd = sum(atom(x, t, th, tau) for x, t in zip(xf, th_t))
        b_rev = sum(atom(x, t, th, tau) for x, t in zip(xf[::-1], th_t))
        sims_xt.append(cossim(b_fwd, b_rev))
    res["timestamped_baseline"] = dict(sim=float(np.mean(sims_xt)),
                                       sd=float(np.std(sims_xt)))
    print(f"    (x,t) baseline : forward-reversed sim = "
          f"{np.mean(sims_xt):+.3f} +- {np.std(sims_xt):.3f}"
          "  (order-aware codes separate reversal too)")

    # (3c) momentum query at one position: which way is it moving at x=0?
    contrasts = []
    for s in range(SEEDS):
        rng = np.random.default_rng(40 + s)
        th, om = codebooks(rng)
        xf = A * np.cos(th_t)
        pf = -A * np.sin(th_t)
        b_fwd = sum(atom(x, p, th, om) for x, p in zip(xf, pf))
        i_neg = np.abs(decode2d(b_fwd, th, om, np.array([0.0]),
                                np.array([-A]))[0, 0]) ** 2
        i_pos = np.abs(decode2d(b_fwd, th, om, np.array([0.0]),
                                np.array([+A]))[0, 0]) ** 2
        contrasts.append((i_neg - i_pos) / (i_neg + i_pos))
    res["momentum_query"] = dict(contrast=float(np.mean(contrasts)),
                                 sd=float(np.std(contrasts)))
    print(f"    momentum query: contrast (down vs up at x=0) = "
          f"{np.mean(contrasts):+.3f} +- {np.std(contrasts):.3f}")
    out["reversal"] = res

    print("VERDICT: one hypervector stores a trajectory as a phase-space object;")
    print("         position-only encoding cannot see the arrow of traversal,")
    print("         the joint (x,p) encoding can.")
    with open("cwf_fpe_trajectory_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_trajectory_results.json")


if __name__ == "__main__":
    main()
