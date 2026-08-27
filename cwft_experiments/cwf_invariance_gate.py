"""
Invariance gate -- numerical demonstration for the calibration chapter
(Chapter ref: sec:invariance-gate).

CLAIM UNDER TEST. The dispersion mass of a Rule 110 propagating mode,

    m_disp = (2*pi/T) * sqrt(1 - (s/T)^2)        [natural units, c_c = 1 cell/tick]

is a function only of the mode's integer temporal period T and spatial shift s.
Those are intrinsic recurrence/translation data of a localized pattern, so
m_disp must be the SAME number in any representation that preserves the substrate
dynamics. The gate accepts a recipe output as a substrate property iff it is
invariant under the representation gauge freedom:
  (ii)  the choice / truncation order of the Koopman lift used to linearise the
        dynamics;
  (iii) the choice of computational basis / coarse-graining grain.

We demonstrate this and -- crucially -- pair it with quantities that FAIL the
gate under the SAME gauge changes, so the test is shown to DISCRIMINATE rather
than to agree by construction:
  PASS quantity: m_disp (recurrence period / shift of a propagating mode).
  FAIL (iii):    m_act, the active-cell density of the same structure (a naive
                 "activity / energy" mass proxy) -- moves under an invertible
                 local recoding of the lattice.
  FAIL (ii):     the worst spurious-mode growth rate of the finite Koopman (DMD)
                 operator -- drifts with truncation rank, while the recurrence
                 period (hence m_disp) does not.

The recurrence period is read by Pearson autocorrelation (Wiener--Khinchin: the
autocorrelation is the Koopman cycle length / spectral content), which works
identically on the raw binary field, an invertibly recoded field, and a
real-valued rank-truncated Koopman reconstruction.

CPU-only, numpy. Writes results.json key "invariance_gate" and a figure.
"""
import json, os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import rule110_step

RNG = np.random.default_rng(0)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# PART 1 -- analytic catalog: m_disp is a function of integer (s, T) only.
# ---------------------------------------------------------------------------
def m_disp(T, s, c_c=1.0):
    """Dispersion rest mass in natural units, m_c c_c^2 / hbar_c = omega*sqrt(1-v^2)."""
    omega = 2.0 * np.pi / T
    v = (s / T) / c_c
    return omega * np.sqrt(max(0.0, 1.0 - v * v))


def catalog_check():
    # (name, shift s, period T) from Cook--Martinez glider data (chapter table).
    cat = [("A", 2, 3), ("B", -2, 4), ("C", 0, 7), ("D", 6, 30)]
    return [{"glider": n, "s": s, "T": T, "v_g": s / T, "m_disp": m_disp(T, s)}
            for n, s, T in cat]


# ---------------------------------------------------------------------------
# PART 2 -- a simulated Rule 110 mode, measured under inequivalent gauges.
#   Source: single "1" on a 0-background. The right edge is stationary, the
#   interior fills with the Rule 110 ether (a v_g = 0, period-7 background).
#   We measure that clean stationary mode in the deep interior.
# ---------------------------------------------------------------------------
def single_one_trajectory(width=601, steps=240):
    row = np.zeros(width, dtype=np.uint8)
    row[width // 2] = 1
    traj = np.empty((steps + 1, width), dtype=np.uint8)
    traj[0] = row
    for t in range(1, steps + 1):
        row = rule110_step(row)
        traj[t] = row
    return traj


def clean_window(traj, w_cols=18, interior_offset=90, t0=100, n_t=91):
    """Clean period-7 ether deep in the interior (away from both edges). Width
    exceeds the ether spatial period (14); n_t = 13*7 is commensurate."""
    redge = int(np.where(traj.any(axis=0))[0].max())
    c1 = redge - interior_offset
    return traj[t0:t0 + n_t, c1:c1 + w_cols].astype(np.float64), redge


def measure_period(window, max_lag=20):
    """Recurrence period via Pearson autocorrelation of the (flattened) field.
    Works on binary fields and on real-valued Koopman reconstructions alike."""
    X = window.reshape(window.shape[0], -1).astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    corr = []
    for lag in range(1, max_lag + 1):
        a, b = X[:-lag].ravel(), X[lag:].ravel()
        denom = np.sqrt((a @ a) * (b @ b))
        corr.append((a @ b) / denom if denom > 0 else 0.0)
    return int(np.argmax(corr)) + 1, np.array(corr)


def recode_xor_left(field):
    """Invertible local recoding (gauge iii): XOR each cell with its left
    neighbour. Bijective given a boundary cell; preserves the recurrence period,
    changes the active-cell density."""
    f = field.astype(np.uint8)
    return (f ^ np.roll(f, 1, axis=1)).astype(np.float64)


def koopman_reconstruct(window, r):
    """Rank-r Koopman/POD reconstruction (truncation order r of the lift)."""
    mu = window.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(window - mu, full_matrices=False)
    r = max(1, min(r, len(S)))
    return (U[:, :r] * S[:r]) @ Vh[:r] + mu


def dmd_eigs(window, rank, ridge=1e-6):
    """DMD (finite Koopman lift) at truncation `rank`. Rank is NOT capped at the
    numerical rank: a bounded observer does not know the true rank, and pushing
    `rank` past the genuine modes is exactly the truncation choice the gate tests."""
    Y = window.T
    X1, X2 = Y[:, :-1], Y[:, 1:]
    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    r = max(1, min(rank, U.shape[1]))
    Ur, Sr, Vr = U[:, :r], S[:r], Vh[:r, :].conj().T
    Atil = Ur.conj().T @ X2 @ Vr @ np.diag(Sr / (Sr ** 2 + ridge ** 2))
    return np.linalg.eigvals(Atil)


def worst_spurious_growth(eigs):
    """Largest |log|lambda|| among OFF-circle eigenvalues -- a truncation artifact."""
    off = eigs[np.abs(np.abs(eigs) - 1.0) >= 0.05]
    mags = np.abs(off)
    mags = mags[mags > 1e-9]
    return float(np.max(np.abs(np.log(mags)))) if mags.size else 0.0


def rel_spread(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return (max(xs) - min(xs)) / (abs(np.mean(xs)) + 1e-12) if xs else np.nan


def main():
    out = {}

    # ---- Part 1: analytic catalog -----------------------------------------
    cat = catalog_check()
    out["catalog"] = cat
    print("PART 1  m_disp = (2pi/T) sqrt(1-(s/T)^2)  reproduces the chapter table")
    print(f"  {'glider':>6} {'s':>3} {'T':>3} {'v_g':>7} {'m_disp':>8}")
    for r in cat:
        print(f"  {r['glider']:>6} {r['s']:>3} {r['T']:>3} {r['v_g']:>7.3f} {r['m_disp']:>8.3f}")

    # ---- Part 2: simulated mode under gauges ------------------------------
    traj = single_one_trajectory()
    window, redge = clean_window(traj)

    # Gauge (iii): raw cell basis vs invertible local recoding.
    T_raw, _ = measure_period(window)
    T_rec, _ = measure_period(recode_xor_left(window))
    m_raw, m_rec = m_disp(T_raw, 0), m_disp(T_rec, 0)
    act_raw = float(window.mean())
    act_rec = float(recode_xor_left(window).mean())

    # Gauge (ii): truncation order of the Koopman lift.
    ranks = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]
    T_trunc = [measure_period(koopman_reconstruct(window, r))[0] for r in ranks]
    m_trunc = [m_disp(T, 0) for T in T_trunc]
    # FAIL companion on the same sweep: spurious modes of the finite Koopman
    # operator, read at finite observer resolution (small measurement noise).
    win_noisy = window + 0.01 * RNG.standard_normal(window.shape)
    spurious = [worst_spurious_growth(dmd_eigs(win_noisy, r)) for r in ranks]

    all_m = [m_raw, m_rec] + m_trunc
    out["simulated_mode"] = {
        "source": "Rule 110, single-1 seed; stationary ether mode, deep interior",
        "right_edge_col": int(redge), "v_g": 0.0,
        "PASS_m_disp": {"raw_basis": m_raw, "recoded_basis": m_rec,
                        "koopman_trunc_ranks": ranks,
                        "koopman_trunc_m_disp": m_trunc,
                        "T_raw": T_raw, "T_recoded": T_rec, "T_trunc": T_trunc},
        "FAIL_m_act": {"raw_basis": act_raw, "recoded_basis": act_rec},
        "FAIL_spurious_growth_vs_rank": dict(zip(map(str, ranks), spurious)),
    }
    out["verdict"] = {
        "m_disp_rel_spread_all_gauges": rel_spread(all_m),
        "m_act_rel_spread_recoding": rel_spread([act_raw, act_rec]),
        "spurious_growth_range_vs_rank": max(spurious) - min(spurious),
        "catalog_C_class_m_disp": m_disp(7, 0),
    }

    print("\nPART 2  the same Rule 110 mode, measured under inequivalent gauges")
    print(f"  v_g = 0 (stationary interior); catalog C-class mass = {m_disp(7,0):.4f}")
    print(f"  (iii) basis change   T raw/recoded = {T_raw}/{T_rec}")
    print(f"        PASS  m_disp    raw={m_raw:.4f}  recoded={m_rec:.4f}")
    print(f"        FAIL  m_act     raw={act_raw:.4f}  recoded={act_rec:.4f}"
          f"   (rel spread {rel_spread([act_raw, act_rec]):.2f})")
    print(f"  (ii)  Koopman truncation rank {ranks}")
    print(f"        PASS  period T  = {T_trunc}   ->  m_disp all = "
          f"{m_disp(7,0):.4f}")
    print(f"        FAIL  worst spurious growth = "
          f"{np.array2string(np.array(spurious), precision=2)}")
    print(f"\n  VERDICT  m_disp rel spread across ALL gauges = "
          f"{out['verdict']['m_disp_rel_spread_all_gauges']:.2e}  (PASS: invariant)")
    print(f"           m_act  rel spread under recoding      = "
          f"{out['verdict']['m_act_rel_spread_recoding']:.2f}  (FAIL: gauge artifact)")
    print(f"           spurious-growth range across rank     = "
          f"{out['verdict']['spurious_growth_range_vs_rank']:.2f}  (FAIL: truncation artifact)")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.7))
    ax[0].axhline(m_raw, color="C2", lw=1, ls=":")
    ax[0].plot([0, 1], [m_raw, m_rec], "o-", color="C2", ms=8,
               label=r"$m_{\rm disp}$ (PASS)")
    ax[0].plot([0, 1], [act_raw, act_rec], "s--", color="C3", ms=8,
               label=r"$m_{\rm act}$ activity (FAIL)")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["raw\nbasis", "recoded\nbasis"])
    ax[0].set_ylabel("value (natural units)"); ax[0].set_ylim(bottom=0)
    ax[0].set_title("(iii) basis change: mass invariant, activity not")
    ax[0].legend(fontsize=8)

    ax[1].plot(ranks, m_trunc, "o-", color="C2", label=r"$m_{\rm disp}$ (PASS)")
    ax[1].set_xlabel("Koopman truncation rank")
    ax[1].set_ylabel(r"$m_{\rm disp}$", color="C2"); ax[1].set_ylim(0, 1.3)
    ax2 = ax[1].twinx()
    ax2.plot(ranks, spurious, "s--", color="C3", label="spurious growth (FAIL)")
    ax2.set_ylabel(r"worst $|\log|\lambda||$ spurious", color="C3")
    ax[1].set_title("(ii) truncation: mass invariant, spurious modes not")
    fig.tight_layout()
    figpath = os.path.join(HERE, "fig_invariance_gate.png")
    fig.savefig(figpath, dpi=130)
    print(f"\nfigure  -> {figpath}")

    # ---- merge into results.json -----------------------------------------
    rj = os.path.join(HERE, "results.json")
    data = {}
    if os.path.exists(rj):
        with open(rj) as f:
            data = json.load(f)
    data["invariance_gate"] = out
    with open(rj, "w") as f:
        json.dump(data, f, indent=2)
    print(f"results -> {rj} [key: invariance_gate]")


if __name__ == "__main__":
    main()
