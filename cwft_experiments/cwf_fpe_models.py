#!/usr/bin/env python3
"""THE CELL ACROSS VSA MODELS: is hbar_c substrate-structure or convention?

CWFT (book B2): a substrate's hbar_c has a substrate-specific VALUE but an invariant
STRUCTURE.  This is exactly what the VSA zoo shows at a matched width convention
(kernel HWHM x frequency std): the phase-space cell EXISTS and is O(1) for every
carrier (structure invariant), but its VALUE is carrier-specific -- 1.18 for FHRR
(= sqrt(2 ln 2), the continuous-phase Gaussian), rising to 1.36 for the bipolar/binary
carriers (MAP, BSC).  "Each substrate its own Planck constant" is thus literal and
measured.  The ell^2 COHERENCE resource (Wigner excess) is separately graded: full for
continuous-phase FHRR, halved by the real projection in HRR, and -- per the dedicated
phase-quantization sweep (cwf_fpe_quantize.py) -- extinguished at true 2-level phase.

For each model we build an FPE-style encoding of a continuous value and measure
(a) the similarity kernel and its resolution (HWHM), (b) resolution x bandwidth
(the cell), (c) the Wigner excess negativity of a two-value bundle (the coherence
order parameter of the manuscript's sec. 5).  Seeded, CPU.  Writes
cwf_fpe_models_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0
XS = np.linspace(-6, 16, 220)


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


def atom(model, x, theta, base=None):
    """FPE-style encoding of value x for each carrier."""
    if model == "FHRR":                       # unit complex phasors, continuous
        return np.exp(1j * x * theta)
    if model == "HRR":                        # real circ-conv; fractional power
        # base is a real unit vector; its DFT phases scale linearly with x, output real
        ph = np.angle(np.fft.fft(base))
        return np.fft.ifft(np.exp(1j * x * ph))          # complex; Re is the HRR vector
    if model == "MAP":                        # bipolar: phase snapped to {0, pi}
        z = np.exp(1j * x * theta)
        return np.sign(np.real(z)) + 0j       # +/-1 (q=2)
    if model == "BSC":                        # binary {0,1} -> map to +/-1
        z = np.exp(1j * x * theta)
        return (np.real(z) > 0).astype(float) * 2 - 1
    raise ValueError(model)


def decode(v, theta, xs, base=None, model="FHRR"):
    if model == "HRR":
        ph = np.angle(np.fft.fft(base))
        return np.array([np.vdot(np.fft.ifft(np.exp(1j * x * ph)), v) for x in xs]) / len(v)
    return (np.exp(-1j * np.outer(xs, theta)) @ v) / len(theta)


def kernel_hwhm(model, theta, base=None, dmax=8.0, n=800):
    d = np.linspace(0.0, dmax, n)
    a0 = atom(model, 0.0, theta, base)
    K = np.array([np.real(np.vdot(a0, atom(model, dd, theta, base))) for dd in d])
    K = K / K[0]
    below = np.where(K < 0.5)[0]
    return float(d[below[0]]) if len(below) else float(dmax)


def main():
    rng = np.random.default_rng(0)
    theta = SIGMA * rng.standard_normal(N)
    base = rng.standard_normal(N)
    base /= np.linalg.norm(base)
    out = {"params": dict(N=N, sigma=SIGMA), "rows": []}
    print(f"cell across VSA models (N={N}, sigma={SIGMA}, HWHMxstd convention):")
    for model in ["FHRR", "HRR", "MAP", "BSC"]:
        res = kernel_hwhm(model, theta, base)
        cell = res * SIGMA
        # coherence: Wigner excess of a two-value bundle vs single-value floor
        b1 = atom(model, 2.0, theta, base) + atom(model, 8.0, theta, base)
        b0 = atom(model, 5.0, theta, base)
        exc = negativity(wigner_ville(decode(b1, theta, XS, base, model))) - \
              negativity(wigner_ville(decode(b0, theta, XS, base, model)))
        cont = model in ("FHRR", "HRR")
        out["rows"].append(dict(model=model, resolution=res, cell=cell,
                                wigner_excess=float(exc),
                                continuous_phase=cont))
        print(f"  {model:>5}: resolution(HWHM)={res:.3f}  cell(res x bw)={cell:.3f}  "
              f"Wigner excess={exc:+.3f}  {'[continuous phase]' if cont else '[q=2 carrier]'}")
    # Matched-base control: HRR whose DFT phases follow a GAUSSIAN law like the
    # FHRR codebook.  The default random base carries uniform DFT phases, so its
    # cell value reflects codebook SHAPE, not the carrier; this control separates
    # the two.  Structural fact: DFT phases live on (-pi, pi], so an HRR carrier
    # cannot hold a Gaussian law that spills past +-pi -- we use sigma=1 (wrap
    # mass 0.2%; the HWHM x std cell is scale-invariant, so the shape test is
    # unaffected).  (Conjugate-symmetric phases keep the base real.)
    SIGMA_M = 1.0
    ph = np.zeros(N)
    half = SIGMA_M * np.random.default_rng(7).standard_normal(N // 2 - 1)
    ph[1:N // 2] = half
    ph[N // 2 + 1:] = -half[::-1]
    base_m = np.real(np.fft.ifft(np.exp(1j * ph)))
    res_m = kernel_hwhm("HRR", theta, base_m)
    cell_m = res_m * float(np.std(ph))
    ph_def = np.angle(np.fft.fft(base))
    res_def = kernel_hwhm("HRR", theta, base)
    cell_shape_true = res_def * float(np.std(ph_def))
    out["hrr_matched"] = dict(resolution=res_m, cell=cell_m,
                              phase_std=float(np.std(ph)),
                              default_phase_std=float(np.std(ph_def)),
                              default_cell_shape_true=cell_shape_true)
    print(f"  HRR matched-base control: cell = {cell_m:.3f} (Gaussian DFT phases;"
          " compare FHRR 1.18)")
    print(f"  HRR default base: DFT-phase std = {np.std(ph_def):.3f} (uniform law);"
          f" shape-true cell = {cell_shape_true:.3f} (compare uniform-law 1.09)")
    print("  -> the cell EXISTS and is O(1) for every carrier construction, but its")
    print("     value follows the CODEBOOK SHAPE: shape-matched HRR equals FHRR.")

    # Two-level construction checks (the algebraic scope of the MAP/BSC rows).
    # (a) HOMOMORPHISM: sign quantization breaks the FPE group law --
    #     q((x+y)th) != q(x th) q(y th) in general (e.g. x th = y th = 0.6 pi:
    #     both factors -1, product +1, but q(1.2 pi) = -1).  Measured: the
    #     entrywise violation fraction for a few (x, y).
    # (b) STATIONARITY: the two-level similarity need not depend on the gap
    #     alone.  Measured: K(x, x+D) across absolute positions x at fixed
    #     gaps D -- the sd across x quantifies the wobble a stationary kernel
    #     would not have.
    print("  two-level (MAP) construction checks:")
    viols = []
    for x_, y_ in [(0.3, 0.6), (1.1, 0.7), (2.4, 1.9)]:
        qx = np.sign(np.cos(x_ * theta))
        qy = np.sign(np.cos(y_ * theta))
        qxy = np.sign(np.cos((x_ + y_) * theta))
        viols.append(float(np.mean(qx * qy != qxy)))
    print(f"    (a) homomorphism: q(x th) q(y th) != q((x+y) th) on "
          f"{np.mean(viols):.0%} of entries (3 value pairs: "
          + ", ".join(f"{v:.2f}" for v in viols) + ")")
    stat_rows = []
    for D in [0.5, 1.0, 2.0]:
        sims = []
        for x0 in np.linspace(0.0, 10.0, 41):
            a1 = atom("MAP", x0, theta)
            a2 = atom("MAP", x0 + D, theta)
            sims.append(float(np.real(np.vdot(a1, a2)) / N))
        stat_rows.append(dict(gap=D, mean=float(np.mean(sims)),
                              sd=float(np.std(sims)),
                              minmax=[float(np.min(sims)), float(np.max(sims))]))
        print(f"    (b) stationarity, gap D={D}: K = {np.mean(sims):.3f} "
              f"+- {np.std(sims):.3f} across absolute positions "
              f"(range {np.min(sims):.3f}..{np.max(sims):.3f})")
    print("    -> the quantized construction is NOT an FPE homomorphism; its")
    print("       origin-referenced width is a property of the projected")
    print("       construction, and stationarity holds only approximately.")
    out["map_checks"] = dict(homomorphism_violation=viols,
                             stationarity=stat_rows)
    with open("cwf_fpe_models_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_models_results.json")


if __name__ == "__main__":
    main()
