"""
cwf_fpe_thetac_bridge.py -- placing the RESONATOR TEMPERATURE on the static
DECOHERENCE AXIS: what does the search's optimal noise level theta_c = 0.75 cost
a static bundle in coherence and readout?

THE CHANNEL (matched to cwf_resonator_halting.py / cwf_resonator_thetahi.py,
verified by direct read of resonator(), lines 78-84): at every iteration the
resonator adds Gaussian noise to the cleaned-up state BEFORE the nonlinearity,
    state <- state + theta_c * sd * noise,     sd = std(|state|) over components,
with real noise for the bipolar branch and COMPLEX noise (g + i g') for the
FHRR branch, and then projects (sign / phasor normalization).  Here the same
channel is applied ONE-SHOT to a static complex FPE bundle, LITERALLY:
    b <- b + theta_c * std(|b|) * (g + i g'),
i.e. the FHRR branch of the identical code path (noise sd theta_c*std(|b|) on
the real and imaginary parts separately).  STATED MISMATCHES
(inevitable): (1) the resonator injects the noise EVERY iteration and follows
it with the phasor projection; a static bundle has no iteration loop, and
applying the projection would set |b_j| = 1 and erase the power spectrum that
carries the relational readout, so the projection is omitted; (2) the
resonator's theta_c=0.75 optimum was measured on the BIPOLAR (Z_2, sign
nonlinearity) resonator; the complex-state analogue is the same formula's FHRR
branch.  For single-atom states (unit-modulus components) std(|b|) = 0, so the
matched incoherent baselines reuse the SCALE MEASURED ON THE COHERENT BUNDLE --
otherwise the baseline would be noiseless and the excess would conflate
channel-induced spurious negativity with lost coherence.

TWO READOUTS vs theta_c in {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}:
 (1) WIGNER EXCESS (the coherence order parameter), cwf_fpe_wigner/stats/
     hardware construction: N=3000, theta ~ omega0=4 + N(0,1), cat state at
     {2,8}, decode on 220 grid points over [-6,16], Wigner-Ville negativity;
     excess = neg(cat) - neg(matched incoherent baseline W(psi_2)+W(psi_8))
     (each single-value state pushed through the same channel at the same
     absolute noise scale); the single-value {5} floor is logged too.  8 seeds.
 (2) RELATIONAL ARGMAX ACCURACY (the useful readout), cwf_fpe_hardware
     period-finding arm: N=4000, sigma=2, s=5 pairs at Delta=3 hidden in
     values U(0,25), autocorrelation score on 400 lags in [2,12]; hit = global
     argmax within 0.3 of Delta.  200 planted + 200 null trials per theta_c.

RESULTS (measured, this run) -- the headline is MILDNESS:
  - Wigner excess: 0.1644 +- 0.0088 at theta_c=0 -> 0.1561 +- 0.0090 at the
    resonator optimum 0.75 (95% retained, 5% cost) -> 0.1522 at 1.0 -> 0.1354
    at 2.0 (82% retained).  Monotone decline, no knee in range.  The matched
    baseline matters: the RAW cat negativity barely moves (0.204 -> 0.210)
    while the incoherent baseline RISES with noise (0.040 -> 0.075, spurious
    channel-induced fringes) -- only the excess is a meaningful signal.
  - Relational argmax accuracy: 0.910 (null 0.115) at theta_c=0 -> 0.900 at
    0.75 (99% of the above-null margin retained) -> 0.815 at 2.0 (90% of
    margin).  No failure knee anywhere in the sweep.
  -> A static bundle is nearly INSENSITIVE to the resonator's channel at
     one-shot dosage: the search's optimal temperature theta_c=0.75, which
     costs the resonator its entire deterministic-failure mode (success 0.28
     -> 0.39), costs the static substrate only ~5% of its coherence witness
     and ~1% of its readout.  Mechanism: the static decode is a matched
     filter, so white complex noise is averaged down by ~1/sqrt(N); the
     resonator's sensitivity comes from RE-INJECTING the noise every iteration
     through the sign nonlinearity, not from the per-step dose.  The witness
     does decay measurably faster than the readout (82% vs 90% retained at
     theta_c=2), same ordering as the hardware-noise study, but both are far
     from any threshold.

CPU, numpy, seeded.  Writes cwf_fpe_thetac_bridge_results.json.
"""

import json

import numpy as np

THETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
THETA_OPT = 0.75                  # resonator optimum (cwf_resonator_thetahi)

# Wigner arm (cwf_fpe_wigner / cwf_fpe_stats / cwf_fpe_hardware construction)
NW = 3000
OMEGA0, SIGMA_W = 4.0, 1.0
XS = np.linspace(-6, 16, 220)
CENTERS = [2.0, 8.0]
FLOOR_C = 5.0
SEEDS_W = 8

# period arm (cwf_fpe_hardware period-finding construction)
PF_N, PF_SIGMA, PF_DELTA, PF_PAIRS = 4000, 2.0, 3.0, 5
PF_DS = np.linspace(2, 12, 400)
NTRIAL = 200


def channel(b, theta_c, rng, scale=None):
    """Resonator noise channel, FHRR branch, LITERAL (resonator() lines 78-84):
    b + theta_c * sd * (g + i g'), sd = std(|b|) unless given.  Note the
    resonator adds theta_c*sd to the real AND imaginary parts separately (total
    complex noise power 2*(theta_c*sd)^2); that exact form is kept here."""
    if theta_c <= 0:
        return b.copy()
    sd = float(np.std(np.abs(b))) if scale is None else scale
    n = (rng.standard_normal(len(b)) + 1j * rng.standard_normal(len(b)))
    return b + theta_c * sd * n


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


def wigner_arm(theta_c, seed):
    rng = np.random.default_rng(1000 + seed)
    th = OMEGA0 + SIGMA_W * rng.standard_normal(NW)
    th = th[th > 0]
    dec = lambda v: (np.exp(-1j * np.outer(XS, th)) @ v) / len(th)

    b_cat = np.exp(1j * CENTERS[0] * th) + np.exp(1j * CENTERS[1] * th)
    scale = float(np.std(np.abs(b_cat)))          # channel scale, from the
    #                                               coherent object (see header)
    neg_cat = negativity(wigner_ville(dec(channel(b_cat, theta_c, rng))))
    W_inc = (wigner_ville(dec(channel(np.exp(1j * CENTERS[0] * th),
                                      theta_c, rng, scale))) +
             wigner_ville(dec(channel(np.exp(1j * CENTERS[1] * th),
                                      theta_c, rng, scale))))
    neg_inc = negativity(W_inc)
    neg_floor = negativity(wigner_ville(dec(channel(np.exp(1j * FLOOR_C * th),
                                                    theta_c, rng, scale))))
    return neg_cat, neg_inc, neg_floor, scale


def period_arm(theta_c, seed, planted=True):
    rng = np.random.default_rng(5000 + seed if planted else 9000 + seed)
    th = PF_SIGMA * rng.standard_normal(PF_N)
    base = np.sort(rng.uniform(0, 25, PF_PAIRS))
    vals = (np.concatenate([base, base + PF_DELTA]) if planted
            else np.sort(rng.uniform(0, 25, 2 * PF_PAIRS)))
    b = sum(np.exp(1j * v * th) for v in vals)
    b = channel(b, theta_c, rng)                  # one-shot resonator channel
    score = (np.cos(np.outer(PF_DS, th)) @ (np.abs(b) ** 2)) / PF_N
    return bool(abs(PF_DS[int(np.argmax(score))] - PF_DELTA) < 0.3)


if __name__ == "__main__":
    print("=" * 74)
    print("RESONATOR TEMPERATURE ON THE STATIC DECOHERENCE AXIS")
    print("=" * 74)
    print("channel: b + theta_c*std(|b|)*(g+ig')  (resonator FHRR branch, literal,")
    print("one-shot, no phasor projection; scale from the coherent bundle)")
    print(f"Wigner arm: N={NW}, omega0={OMEGA0}, cat {CENTERS}, {SEEDS_W} seeds; "
          f"period arm: N={PF_N}, {PF_PAIRS} pairs at Delta={PF_DELTA}, "
          f"{NTRIAL}+{NTRIAL} trials")

    out = {"params": dict(
        thetas=THETAS, theta_optimum_resonator=THETA_OPT,
        wigner=dict(N=NW, omega0=OMEGA0, sigma=SIGMA_W, centers=CENTERS,
                    floor_center=FLOOR_C, grid=[float(XS[0]), float(XS[-1]),
                                                len(XS)], seeds=SEEDS_W),
        period=dict(N=PF_N, sigma=PF_SIGMA, delta=PF_DELTA, pairs=PF_PAIRS,
                    ds=[2.0, 12.0, len(PF_DS)], trials=NTRIAL,
                    hit_window=0.3),
        channel=("cwf_resonator_halting.py resonator() noise, FHRR branch, "
                 "LITERAL: state + theta_c*std(|state|)*(g+ig'), i.e. sd per "
                 "real and imaginary part separately; applied ONE-SHOT to the "
                 "static bundle; phasor projection OMITTED (would erase |b|^2, "
                 "the readout carrier); single-atom baselines use the coherent "
                 "bundle's scale since std(|atom|)=0; resonator optimum was "
                 "measured on the BIPOLAR branch; the resonator injects this "
                 "noise EVERY iteration, the static analogue once -- these are "
                 "the stated mismatches")),
        "rows": []}

    print(f"\n  {'theta_c':>7} | {'excess':>7} +- sd    | {'cat':>6} {'incoh':>6} "
          f"{'floor':>6} | {'acc':>5} (null)")
    for th in THETAS:
        W = np.array([wigner_arm(th, s) for s in range(SEEDS_W)])
        cat, inc, flo = W[:, 0], W[:, 1], W[:, 2]
        exc = cat - inc
        hits = [period_arm(th, t, True) for t in range(NTRIAL)]
        nulls = [period_arm(th, t, False) for t in range(NTRIAL)]
        acc, nul = float(np.mean(hits)), float(np.mean(nulls))
        row = dict(theta_c=th,
                   wigner_excess=float(exc.mean()),
                   wigner_excess_sd=float(exc.std()),
                   neg_cat=float(cat.mean()), neg_incoherent=float(inc.mean()),
                   neg_floor=float(flo.mean()),
                   channel_scale=float(W[:, 3].mean()),
                   argmax_accuracy=acc,
                   argmax_accuracy_se=float(np.sqrt(acc * (1 - acc) / NTRIAL)),
                   null_accuracy=nul)
        out["rows"].append(row)
        mark = "  <-- resonator optimum" if th == THETA_OPT else ""
        print(f"  {th:>7} | {exc.mean():.4f} +- {exc.std():.4f} | "
              f"{cat.mean():.4f} {inc.mean():.4f} {flo.mean():.4f} | "
              f"{acc:.3f} ({nul:.3f}){mark}")

    r0 = out["rows"][0]
    ro = next(r for r in out["rows"] if r["theta_c"] == THETA_OPT)
    exc_frac = ro["wigner_excess"] / (r0["wigner_excess"] + 1e-12)
    acc_frac = ((ro["argmax_accuracy"] - ro["null_accuracy"]) /
                max(r0["argmax_accuracy"] - r0["null_accuracy"], 1e-12))
    out["at_optimum"] = dict(theta_c=THETA_OPT,
                             excess_fraction_retained=float(exc_frac),
                             accuracy_fraction_retained=float(acc_frac))
    print(f"\nAT THE RESONATOR OPTIMUM theta_c={THETA_OPT}: Wigner excess retains "
          f"{exc_frac*100:.0f}% ({r0['wigner_excess']:.4f} -> "
          f"{ro['wigner_excess']:.4f});")
    print(f"relational readout retains {acc_frac*100:.0f}% above null "
          f"({r0['argmax_accuracy']:.3f} -> {ro['argmax_accuracy']:.3f}, "
          f"null {ro['null_accuracy']:.3f}).")
    print("-> a static bundle is nearly insensitive to this channel at one-shot")
    print("   dosage (matched-filter decode averages white noise down ~1/sqrt(N));")
    print("   the resonator's fragility comes from re-injecting it every iteration")
    print("   through the nonlinearity, not from the per-step dose.  The witness")
    print("   decays faster than the readout, but neither hits a knee by theta_c=2.")

    with open("cwf_fpe_thetac_bridge_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_thetac_bridge_results.json")
