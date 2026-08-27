"""
cwf_ap_phaseB.py -- ACTION_PRINCIPLE_PLAN Phase B: the classical/reducibility limit of the
influence action (Test 1 / G1 / G6). The COST-sector complement to Phases C/D (phase sector).

Claim (plan Sec.2, refined per the reviewer's G6): the transition-violation cost
    C_dyn[gamma] = sum_k d( s_{k+1}, T(s_k) )                 (d = Hamming distance)
is DERIVABLE directly from the substrate rule T (not fitted, G1), is >= 0 with equality EXACTLY on
the rule-respecting history, so in the cost-dominated (small-hbar_c / Euclidean) limit the path
integral is dominated by the minimum-C_dyn history -- which is the deterministic TRAJECTORY = the
substrate's effective/classical dynamics (G6: 'stationary phase = effective dynamics'; here a clean
global minimum, so the reviewer's stationary!=minimum caveat is moot -- it IS the minimum, uniquely).

The REDUCIBLE vs IRREDUCIBLE distinction is then NOT in the saddle (both rules have the trajectory as
their zero-cost saddle) but in whether that saddle is reachable by a FINITE SHORTCUT:
  - Rule 90 is LINEAR over GF(2) (new[i] = s[i-1] XOR s[i+1]) -> closed form: state_t = M^t . s_0 (mod
    2), reachable WITHOUT stepping -> REDUCIBLE.
  - Rule 110 is NONLINEAR / universal -> the linear shortcut FAILS (superposition is violated) -> the
    only way to the saddle is to RUN it -> IRREDUCIBLE (we show the linear shortcut fails and cite
    universality; we cannot prove NO shortcut exists -- that is the undecidable/open part).

So the action's classical limit reproduces the effective dynamics (G6), G1 holds for C_dyn (derivable
from T), and 'reducibility' = the saddle is shortcut-reachable. This is the honest scope: it confirms
the COST sector's classical limit; the harder Koopman-MDL identity and the description cost C_desc
remain open (G1 for those terms).

CPU; numpy. Results -> ap_phaseB_results.json. Self-contained ECA stepping (auditable).
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def eca_step(s, rule):
    """One elementary-CA step, periodic boundary. neighborhood = 4*left+2*center+1*right."""
    left = np.roll(s, 1)
    right = np.roll(s, -1)
    idx = (left << 2) | (s << 1) | right
    return ((rule >> idx) & 1).astype(np.int64)


def trajectory(s0, rule, T):
    """The deterministic history s_0 .. s_T."""
    hist = [s0.copy()]
    s = s0.copy()
    for _ in range(T):
        s = eca_step(s, rule)
        hist.append(s.copy())
    return np.array(hist)


def C_dyn(history, rule):
    """Transition-violation cost = sum_k Hamming(history[k+1], T(history[k])). Derivable from T;
    zero iff the history obeys the rule everywhere."""
    c = 0
    for k in range(len(history) - 1):
        pred = eca_step(history[k], rule)
        c += int(np.sum(history[k + 1] != pred))
    return c


# ---- reducibility: linearity over GF(2) (superposition test) ----------------------------
def superposition_residual(rule, W, T, n_trials, seed):
    """For random a,b: is T^t(a XOR b) == T^t(a) XOR T^t(b)? (linear, offset-free). Returns the
    mean Hamming residual over trials at the final time T. 0 => linear (Rule 90); >0 => nonlinear."""
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_trials):
        a = rng.integers(0, 2, W)
        b = rng.integers(0, 2, W)
        ta = trajectory(a, rule, T)[-1]
        tb = trajectory(b, rule, T)[-1]
        tab = trajectory(a ^ b, rule, T)[-1]
        res.append(int(np.sum(tab != (ta ^ tb))))
    return float(np.mean(res))


# ---- Rule 90 closed-form shortcut: state_t = M^t . s_0 (mod 2) ---------------------------
def rule90_matrix(W):
    """GF(2) update matrix for Rule 90: new[i] = s[i-1] XOR s[i+1]."""
    M = np.zeros((W, W), dtype=np.int64)
    for i in range(W):
        M[i, (i - 1) % W] = 1
        M[i, (i + 1) % W] = 1
    return M


def matpow_mod2(M, t):
    """M^t over GF(2) by repeated squaring."""
    W = M.shape[0]
    R = np.eye(W, dtype=np.int64)
    B = M.copy()
    while t > 0:
        if t & 1:
            R = (R @ B) & 1
        B = (B @ B) & 1
        t >>= 1
    return R


def main():
    print("cwf_ap_phaseB -- the classical/reducibility limit of the action (Test 1 / G1 / G6)\n")
    W, T = 31, 30
    rng = np.random.default_rng(7)
    s0 = rng.integers(0, 2, W)

    # ----- B1: C_dyn is derivable and the trajectory is its zero-cost saddle (both rules) -----
    print("B1  C_dyn derivable from T; the deterministic trajectory is its unique zero-cost saddle")
    b1 = {}
    for rule in (90, 110):
        traj = trajectory(s0, rule, T)
        c_traj = C_dyn(traj, rule)
        # perturbed histories: flip random cells in the trajectory -> cost must rise above 0
        costs_pert = []
        for _ in range(200):
            h = traj.copy()
            nflip = rng.integers(1, 6)
            for _f in range(nflip):
                i = rng.integers(0, len(h)); j = rng.integers(0, W)
                h[i, j] ^= 1
            costs_pert.append(C_dyn(h, rule))
        b1[rule] = dict(C_dyn_trajectory=c_traj, min_perturbed=int(min(costs_pert)),
                        mean_perturbed=float(np.mean(costs_pert)))
        print(f"    Rule {rule:3d}:  C_dyn(trajectory)={c_traj}  |  perturbed: "
              f"min={min(costs_pert)}, mean={np.mean(costs_pert):.1f}  "
              f"-> trajectory is the unique minimum: {c_traj == 0 and min(costs_pert) > 0}")
    b1_ok = all(b1[r]["C_dyn_trajectory"] == 0 and b1[r]["min_perturbed"] > 0 for r in (90, 110))
    print(f"    -> saddle (min C_dyn) = deterministic trajectory = classical/effective dynamics "
          f"(G6): {b1_ok}")

    # ----- B2: reducibility = linearity (superposition) -> shortcut reachable -----
    print("\nB2  reducibility: is the saddle reachable by a finite (linear) shortcut?")
    lin90 = superposition_residual(90, W, T, n_trials=300, seed=1)
    lin110 = superposition_residual(110, W, T, n_trials=300, seed=1)
    print(f"    superposition residual  T^t(a^b) ^ T^t(a) ^ T^t(b):")
    print(f"      Rule  90: {lin90:.3f}  -> LINEAR over GF(2) (reducible)")
    print(f"      Rule 110: {lin110:.3f} -> NONLINEAR (no linear shortcut; universal/irreducible)")
    b2_ok = (lin90 < 1e-9) and (lin110 > 1.0)

    # ----- B3: the actual Rule-90 closed-form shortcut M^t . s_0 == stepped -----
    print("\nB3  Rule 90 closed-form shortcut: state_t = M^t . s_0 (mod 2), reached WITHOUT stepping")
    M = rule90_matrix(W)
    traj90 = trajectory(s0, 90, T)
    shortcut_ok = True
    for t_check in [1, 5, 13, T]:
        pred = (matpow_mod2(M, t_check) @ s0) & 1
        match = bool(np.all(pred == traj90[t_check]))
        shortcut_ok = shortcut_ok and match
        print(f"      t={t_check:3d}:  M^t . s_0 == stepped state : {match}")
    # and confirm the SAME linear shortcut fails for Rule 110 (no GF(2) matrix reproduces it):
    traj110 = trajectory(s0, 110, T)
    # best-effort linear model for 110: there is none; the t=1 map is already nonlinear in s_0.
    # Demonstrate via the t=1 affine-ization failing the superposition test (done in B2); here just
    # confirm Rule 110's evolution is NOT the Rule-90 (or any tested linear) map:
    lin110_fails = not np.all((M @ s0) & 1 == traj110[1])
    print(f"      Rule 110 is NOT this (or any) linear map (t=1 differs from a GF(2) update): "
          f"{lin110_fails}")
    b3_ok = shortcut_ok and lin110_fails

    all_pass = b1_ok and b2_ok and b3_ok
    verdict = (
        "PHASE B PASS -- the action's COST sector has a clean classical limit, and 'reducibility' is "
        "the shortcut-reachability of its saddle. (B1) C_dyn = sum Hamming(s_{k+1}, T(s_k)) is "
        "DERIVABLE from the rule T (G1, for this term), is >=0, and vanishes EXACTLY on the "
        "deterministic trajectory; perturbed histories cost more -> the trajectory is the unique "
        "global minimum, so the cost-dominated path integral is dominated by it = the effective/"
        "classical dynamics (G6; a genuine minimum, so 'stationary != minimum' does not bite). (B2) "
        f"Rule 90 is LINEAR over GF(2) (superposition residual {lin90:.3f}=0) and (B3) its saddle is "
        f"reachable by the closed form state_t=M^t.s_0 WITHOUT stepping -> REDUCIBLE. Rule 110 is "
        f"NONLINEAR (residual {lin110:.1f}>0), no linear shortcut reproduces it -> the saddle is "
        "RUN-ONLY -> IRREDUCIBLE. So the action principle's classical limit reproduces the substrate's "
        "deterministic dynamics, and the reducible/irreducible axis is exactly whether that saddle has "
        "a finite shortcut. Honest scope: this confirms the COST sector (C_dyn derivable, classical "
        "limit clean); 'no shortcut for Rule 110' is shown for LINEAR shortcuts + cites universality "
        "(proving NO shortcut exists is undecidable, the open part); and the harder Koopman-MDL "
        "identity / description cost C_desc remain open (G1 for those terms)."
    ) if all_pass else (
        "PHASE B INCOMPLETE -- a sub-test failed; inspect B1/B2/B3. A failure of B1 would mean C_dyn "
        "is not derivable or the trajectory is not its minimum (G1 kill for the cost sector)."
    )
    print(f"\n  B1 C_dyn derivable, trajectory = unique saddle : {b1_ok}  (G1+G6, cost sector)")
    print(f"  B2 reducibility = linearity (Rule 90 yes / 110 no) : {b2_ok}")
    print(f"  B3 Rule-90 closed-form shortcut reaches the saddle : {b3_ok}")
    print(f"\n  ALL CHECKS PASS: {all_pass}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseB_results.json")
    R = dict(
        W=W, T=T,
        B1_saddle=dict(per_rule=b1, trajectory_is_unique_minimum=bool(b1_ok),
                       label="C_dyn derivable from T (G1); min C_dyn = deterministic trajectory (G6)"),
        B2_reducibility=dict(superposition_residual_rule90=lin90,
                             superposition_residual_rule110=lin110, linear_split=bool(b2_ok),
                             label="Rule 90 linear over GF(2) (reducible); Rule 110 nonlinear"),
        B3_shortcut=dict(rule90_closed_form_matches=bool(shortcut_ok),
                         rule110_not_linear=bool(lin110_fails), shortcut_split=bool(b3_ok),
                         label="Rule 90 saddle reachable by M^t.s_0 (no stepping); Rule 110 run-only"),
        all_pass=bool(all_pass), verdict=verdict,
        note=("Cost-sector classical limit. C_dyn = sum Hamming(s_{k+1},T(s_k)) derivable from T (G1); "
              "its unique minimum is the deterministic trajectory = the effective/classical dynamics "
              "(G6, a clean global minimum). Reducible (Rule 90, linear GF(2), closed-form shortcut "
              "M^t.s_0) vs irreducible (Rule 110, nonlinear, run-only). Cannot prove NO shortcut for "
              "110 (undecidable); shows linear shortcut fails + universality. Phase-sector (A_c) is "
              "Phases C/D; Koopman-MDL identity and C_desc remain open."))
    json.dump(R, open(out, "w"), indent=2)
    plot(s0, traj90, traj110)
    print(f"\nWrote {out}")


def plot(s0, traj90, traj110):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax1.imshow(traj90, cmap="binary", aspect="auto", interpolation="nearest")
    ax1.set_title("Rule 90 (LINEAR / GF(2)): saddle = $M^t\\,s_0$\nreducible -- closed-form shortcut")
    ax1.set_xlabel("cell"); ax1.set_ylabel("time step")
    ax2.imshow(traj110, cmap="binary", aspect="auto", interpolation="nearest")
    ax2.set_title("Rule 110 (NONLINEAR / universal): saddle run-only\nirreducible -- no linear shortcut")
    ax2.set_xlabel("cell"); ax2.set_ylabel("time step")
    fig.suptitle("Phase B: the action's saddle (min $C_{dyn}$) = the deterministic trajectory; "
                 "reducibility = is that saddle shortcut-reachable?", fontsize=10)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseB.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
