#!/usr/bin/env python3
"""CHIRPED FPE: SHEAR IN PHASE SPACE, AND CODES FOR VALUES IN MOTION.

Chirp = shear, the third elementary symplectic operation after translation and
squeezing: (x, p) -> (x, p + gamma x).  Three results:

 (1) ALGEBRA CHECK: multiplying the decoded profile by the quadratic phase
     exp(-i gamma x^2 / 2) shifts the momentum center of a state at x0 by
     gamma * x0 (p = theta convention; the sign pairs with the decoder
     e^{-i x theta}, see cwf_fpe_wigner_convention.py).  Measured: the slope
     of <p> vs x0 equals gamma.  (Like the boost, this is a codebook-level
     operation, not an elementwise vector op.)
 (2) MOVING VALUES (native, operational): a value that MOVES during the encoding
     window, x(t) = x0 + v t for t in [-T/2, T/2], bundles into
     b_j = e^{i x0 th_j} D(v th_j) -- motion imprints a v-dependent REAL
     envelope (radar's range-Doppler coupling).  Each component's temporal
     phase is LINEAR (constant Doppler v*th_j): a frequency-proportional
     Doppler slope across coordinates -- the (t, theta) bilinear phase -- not
     a time-domain linear-FM chirp; the (x,p) shear proper is the
     quadratic-phase operation of part (1).  A static matched filter blurs
     as v grows; a chirped matched-filter bank over (x, v) recovers the
     position AND the speed.  Metric: x-RMSE vs v for both decoders, under
     fixed additive noise; speed-RMSE for the chirped bank.
     TWO STRUCTURAL FACTS, measured rather than hidden:
       (a) SIGN DEGENERACY: bundling is order-blind, so b(+v) = b(-v) exactly
           (cos-sim = 1).  Speed is readable from the moving bundle; direction
           is not.  Direction needs the joint (x,p) encoding of
           cwf_fpe_trajectory.py -- the two experiments are one lesson.
       (b) CARRIER REQUIRED: on a zero-mean codebook the components near
           theta ~ 0 never see motion and leave a similarity pedestal >= 0.5
           at all speed lags; motion reading wants a carrier codebook -- the
           same carrier caveat as the sketch error law (and the reason radar
           uses a carrier).
 (3) WINDOW-RESOLUTION SCALING: on the carrier codebook the speed
     resolution Delta_v shrinks as the window T grows, with
     Delta_v * T * omega0 ~ const.  The constant is convention- and
     window-specific (an analytic prediction for THIS window/codebook/
     similarity is computed and compared); the 1/T scaling is the law.
     Off-grid speeds are tested (part 2b): nearest-template error is grid-
     bounded, parabolic interpolation refines it.

Honest size: matched filtering of linear-FM signals is classical radar; the
content is that FPE inherits it natively -- the velocity templates are
ordinary bundles of moving codes -- because motion writes a (t, theta) shear
into the code's joint time-frequency structure.
Seeded, CPU.  Writes cwf_fpe_chirp_results.json.
"""
import json

import numpy as np

N = 4000
SIGMA = 2.0         # zero-mean codebook bandwidth (part 1, shear algebra)
OM0, SIGC = 4.0, 1.0  # carrier codebook (parts 2-3, motion reading)
S = 25              # samples across the encoding window
NOISE = 0.15        # complex noise sd per component (per unit-normalized bundle)
TRIALS = 30
VS = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
T_WIN = 1.0


def moving_bundle(x0, v, th, tgrid):
    return np.mean(np.exp(1j * np.outer(x0 + v * tgrid, th)), axis=0)


def main():
    out = {"params": dict(N=N, sigma=SIGMA, S=S, noise=NOISE, trials=TRIALS,
                          T=T_WIN)}

    # ---------------- (1) algebra check: quadratic phase = shear -------------
    # p = theta convention (cwf_fpe_wigner_convention.py): the decoder pairs x
    # with -theta, so the shear operator is e^{-i gamma x^2/2} and the FFT
    # frequency axis is negated.  Slope of <p> vs x0 should equal +gamma.
    print("(1) algebra check: quadratic phase = shear (slope of <p> vs x0 = gamma)")
    rng = np.random.default_rng(0)
    th = SIGMA * rng.standard_normal(N)
    xs = np.linspace(-8, 8, 2048)
    dx = xs[1] - xs[0]
    p_axis = -2 * np.pi * np.fft.fftfreq(len(xs), d=dx)   # p = theta axis
    slopes = {}
    for gamma in [0.5, 1.0]:
        x0s = np.linspace(-4, 4, 9)
        pbars = []
        for x0 in x0s:
            psi = np.exp(-1j * np.outer(xs, th)) @ np.exp(1j * x0 * th) / N
            win = np.exp(-((xs - x0) ** 2) / (2 * 2.0 ** 2))
            psi_c = psi * np.exp(-1j * gamma * xs ** 2 / 2) * win
            P = np.abs(np.fft.fft(psi_c)) ** 2
            pbars.append(float((p_axis * P).sum() / P.sum()))
        slope = float(np.polyfit(x0s, pbars, 1)[0])
        slopes[gamma] = slope
        print(f"    gamma={gamma}: fitted slope = {slope:.3f}"
              f"  (relative error {abs(slope - gamma) / gamma:.1%})")
    out["shear_check"] = {str(g): s for g, s in slopes.items()}

    # ---------------- (2) moving values: static vs chirped decoding ----------
    print("(2) moving values: static matched filter vs chirped (x,v) bank")
    print("    (carrier codebook omega0=4, sigma=1 -- see structural fact (b))")
    tgrid = T_WIN * np.linspace(-0.5, 0.5, S)
    xhat = np.arange(0.0, 10.0, 0.02)
    vhat = np.arange(0.0, 9.0 + 1e-9, 0.25)   # speed grid: sign is degenerate
    rng = np.random.default_rng(1)
    th = OM0 + SIGC * rng.standard_normal(N)
    th = th[th > 0]
    Nc = len(th)
    Ex = np.exp(-1j * np.outer(xhat, th))                    # (nx, N)
    env = np.stack([np.mean(np.exp(1j * np.outer(v * tgrid, th)), axis=0)
                    for v in vhat])                          # (nv, N)
    env_norm = np.linalg.norm(env, axis=1)

    # structural fact (a): sign degeneracy, measured
    bp = moving_bundle(5.0, +3.0, th, tgrid)
    bm = moving_bundle(5.0, -3.0, th, tgrid)
    sign_sim = float(np.abs(np.vdot(bp, bm)) /
                     (np.linalg.norm(bp) * np.linalg.norm(bm)))
    print(f"    sign degeneracy: cos-sim(b(+v), b(-v)) = {sign_sim:.6f}"
          "  (order-blind bundle: speed yes, direction no)")
    # structural fact (b): zero-mean pedestal, measured
    th0 = SIGMA * np.random.default_rng(7).standard_normal(N)
    tails = []
    b0 = moving_bundle(5.0, 2.0, th0, tgrid)
    for dv in [4.0, 6.0]:
        b1 = moving_bundle(5.0, 2.0 + dv, th0, tgrid)
        tails.append(float(np.abs(np.vdot(b1, b0)) /
                           (np.linalg.norm(b1) * np.linalg.norm(b0))))
    print(f"    zero-mean pedestal: speed-lag similarity stays at "
          f"{tails[0]:.2f}/{tails[1]:.2f} (dv=4/6) -- carrier required")
    out["structural"] = dict(sign_sim=sign_sim, zero_mean_tails=tails)

    rows = []
    for v in VS:
        ex_s, ex_c, ev_c, amp_s = [], [], [], []
        for tr in range(TRIALS):
            r = np.random.default_rng(1000 + 17 * tr + int(v * 10))
            x0 = r.uniform(2, 8)
            b = moving_bundle(x0, v, th, tgrid)
            b = b + NOISE * (r.standard_normal(Nc) + 1j * r.standard_normal(Nc)) \
                / np.sqrt(2)
            # static filter: correlate with stationary codes
            sc_s = np.abs(Ex @ b) / Nc
            ex_s.append(xhat[np.argmax(sc_s)] - x0)
            amp_s.append(float(sc_s.max()))
            # chirped bank: correlate with moving-code templates over (x, speed)
            best, bx, bv = -1.0, 0.0, 0.0
            for iv in range(len(vhat)):
                sc = np.abs(Ex @ (np.conj(env[iv]) * b)) / (env_norm[iv] + 1e-12)
                i = int(np.argmax(sc))
                if sc[i] > best:
                    best, bx, bv = float(sc[i]), float(xhat[i]), float(vhat[iv])
            ex_c.append(bx - x0)
            ev_c.append(bv - abs(v))
        rows.append(dict(v=v,
                         x_rmse_static=float(np.sqrt(np.mean(np.square(ex_s)))),
                         x_rmse_chirp=float(np.sqrt(np.mean(np.square(ex_c)))),
                         speed_rmse_chirp=float(np.sqrt(np.mean(np.square(ev_c)))),
                         static_peak=float(np.mean(amp_s))))
        print(f"    v={v:>3}: x-RMSE static={rows[-1]['x_rmse_static']:.3f}  "
              f"chirped={rows[-1]['x_rmse_chirp']:.3f}  "
              f"speed-RMSE={rows[-1]['speed_rmse_chirp']:.3f}  "
              f"static peak={rows[-1]['static_peak']:.3f}")
    out["moving"] = rows

    # (2b) OFF-GRID speeds: truth placed between template gridpoints, so exact
    # recovery cannot come from grid alignment.  Nearest-template error is
    # bounded by the grid (<= 0.125); 3-point parabolic interpolation over the
    # per-speed peak scores refines it.
    print("    off-grid speeds (truth between gridpoints; grid spacing 0.25):")
    offrows = []
    for v in [0.87, 1.63, 3.41, 5.12, 7.06]:
        ev_near, ev_par = [], []
        for tr in range(TRIALS):
            r = np.random.default_rng(3000 + 17 * tr + int(v * 100))
            x0 = r.uniform(2, 8)
            b = moving_bundle(x0, v, th, tgrid)
            b = b + NOISE * (r.standard_normal(Nc) + 1j * r.standard_normal(Nc)) \
                / np.sqrt(2)
            sc_v = np.empty(len(vhat))
            for iv in range(len(vhat)):
                sc = np.abs(Ex @ (np.conj(env[iv]) * b)) / (env_norm[iv] + 1e-12)
                sc_v[iv] = sc.max()
            iv = int(np.argmax(sc_v))
            ev_near.append(vhat[iv] - v)
            if 0 < iv < len(vhat) - 1:
                y0, y1, y2 = sc_v[iv - 1], sc_v[iv], sc_v[iv + 1]
                d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-15)
                ev_par.append(vhat[iv] + d * 0.25 - v)
            else:
                ev_par.append(vhat[iv] - v)
        offrows.append(dict(v=v,
                            rmse_nearest=float(np.sqrt(np.mean(np.square(ev_near)))),
                            rmse_parabolic=float(np.sqrt(np.mean(np.square(ev_par))))))
        print(f"      v={v}: speed-RMSE nearest={offrows[-1]['rmse_nearest']:.3f}"
              f"  parabolic={offrows[-1]['rmse_parabolic']:.3f}")
    out["offgrid"] = offrows

    # ---------------- (3) speed resolution vs window: Delta_v * T ------------
    print("(3) speed resolution (carrier): Delta_v * T * omega0 ~ const")
    v_true, x_true = 2.0, 5.0
    prods = []
    for T in [0.5, 1.0, 2.0, 4.0]:
        tg = T * np.linspace(-0.5, 0.5, S)
        b = moving_bundle(x_true, v_true, th, tg)
        dvs = np.arange(0, 6.01, 0.02)
        sims = np.array([np.abs(np.vdot(moving_bundle(x_true, v_true + dv, th, tg), b))
                         / (np.linalg.norm(moving_bundle(x_true, v_true + dv, th, tg))
                            * np.linalg.norm(b)) for dv in dvs])
        below = np.where(sims < 0.5)[0]
        hw = float(dvs[below[0]]) if len(below) else float("inf")
        prods.append(dict(T=T, dv=hw, product=float(hw * T * OM0)))
        print(f"    T={T}: Delta_v={hw:.3f}  Delta_v*T*omega0={hw * T * OM0:.2f}")
    out["resolution"] = prods

    # analytic prediction for THIS convention (cosine-similarity half-crossing,
    # S-sample window, codebook law N(omega0, sigma_c^2)): integrate the
    # envelope correlation over the law instead of sampling it.  The constant
    # is convention- and window-specific -- e.g. a continuous flat window read
    # at the intensity half-width gives 2.78, at the amplitude half-width 3.79.
    thg = np.linspace(OM0 - 6 * SIGC, OM0 + 6 * SIGC, 2001)
    wg = np.exp(-((thg - OM0) ** 2) / (2 * SIGC ** 2))
    wg /= wg.sum()

    def env_pred(v, tg):
        return np.mean(np.cos(np.outer(v * tg, thg)), axis=0)   # D(v theta), real

    preds = []
    for T in [0.5, 1.0, 2.0, 4.0]:
        tg = T * np.linspace(-0.5, 0.5, S)
        D0 = env_pred(v_true, tg)
        dvs = np.arange(0, 6.001, 0.005)
        sims = np.array([abs((wg * env_pred(v_true + dv, tg) * D0).sum())
                         / np.sqrt((wg * env_pred(v_true + dv, tg) ** 2).sum()
                                   * (wg * D0 ** 2).sum()) for dv in dvs])
        below = np.where(sims < 0.5)[0]
        hwp = float(dvs[below[0]]) if len(below) else float("inf")
        preds.append(float(hwp * T * OM0))
    print(f"    predicted constant for this window/codebook/similarity: "
          f"{min(preds):.2f}-{max(preds):.2f} (measured above: "
          f"{min(p['product'] for p in prods):.2f}-"
          f"{max(p['product'] for p in prods):.2f})")
    out["resolution_predicted"] = preds

    print("VERDICT: motion writes a (t,theta) shear into the code; velocity-")
    print("         template banks read position AND speed from one bundle")
    print("         (direction needs the joint (x,p) code), and Delta_v ~ 1/T")
    print("         holds with the window/convention-specific constant.")
    with open("cwf_fpe_chirp_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_chirp_results.json")


if __name__ == "__main__":
    main()
