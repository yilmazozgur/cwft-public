"""
cwf_ap_phaseE.py -- ACTION_PRINCIPLE_PLAN Phase E: the photon/black-hole Wick-rotation corner
duality, as a BRIDGE (G5), and the G7 refinement (do A_c and C_c form one analytic object?).

Movement VI of the book reads the photon and the black hole as the two exact corners, "mirror
images, and the mirror is time." This phase asks whether the influence-action picture makes that
PRECISE: are the photon corner (Lorentzian, pure phase exp(i A_c/hbar_c)) and the black-hole corner
(Euclidean, thermal exp(-C_c^E/hbar_c)) the two analytic continuations t -> i tau of ONE partition
object Z_c, with the horizon temperature emerging as the imaginary-time period (Gibbons-Hawking)?

The reviewer (Reviewer_Suggests_Actional.txt, G5) cautioned: Wick rotation is NOT automatic for an
arbitrary computational substrate. It needs (1) a time variable admitting analytic continuation; (2)
a stable Euclidean functional; (3) reflection positivity or an analogue; (4) a relation between
Euclidean periodicity and horizon temperature; (5) a substrate horizon with well-defined surface
gravity. So the honest claim is conditional: WHEN a substrate admits both a Lorentzian phase
functional and a Euclidean cost functional (the analog-gravity kinematics of Ch5 supply the
near-horizon Rindler structure), the two corners are continuations of one Z_c. Status B, not a
general theorem; the Gibbons-Hawking smoothness->temperature and the t->i tau rotation are STANDARD
machinery -- the CWF content is only the IDENTIFICATIONS (A_c = self-ref phase, C_c = self-description
cost, kappa from the gain-horizon g'(x_h)).

Three tests:
  E1  Gibbons-Hawking on the gain horizon: g(x)=0 with slope kappa gives a near-horizon Rindler
      metric; Wick rotation t->i tau is smooth (no conical singularity) ONLY at period beta=2pi/kappa
      -> T_c = kappa/2pi. (STANDARD mechanism; CWF input = kappa from the gain gradient.)
  E2  the actual rotation: the free propagator K(x;t) rotated t = T0 e^{-i theta} from theta=0
      (Lorentzian, OSCILLATORY = photon corner) to theta=pi/2 (Euclidean, DAMPED heat kernel =
      black-hole/thermal corner). "The mirror is time," made concrete.
  E3  the G7 refinement: the DYNAMICAL phase A_dyn = int L dt Wick-rotates into a real Euclidean
      cost (exp(i A_dyn) -> exp(-S_E)); but the GEOMETRIC/contextual holonomy A_sr (Phase C; host-
      invariant per Phase D) is a topological phase (oint A), NOT an action int L dt -- it does NOT
      rotate into a cost. So A_c and C_c are one analytic object only in the DYNAMICAL sector; the
      contextual phase stands apart. The corner duality is the dynamical sector's; contextuality is
      the separate (third-axis) thing.

CPU; numpy. Standard machinery + CWF identifications. Results -> ap_phaseE_results.json.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- E1: Gibbons-Hawking smoothness on the gain horizon ---------------------------------
def gh_temperature(kappa, n_beta=4000):
    """Near-horizon Euclidean metric ds^2 = (kappa rho)^2 dtau^2 + drho^2. In polar form the
    angle is (kappa tau); periodicity of tau with period beta gives total angle kappa*beta. Smooth
    (no conical singularity) iff kappa*beta = 2pi, i.e. beta = 2pi/kappa, T_c = kappa/2pi. We find
    the root of the conical deficit Delta(beta) = 2pi - kappa*beta numerically."""
    betas = np.linspace(1e-3, 4 * np.pi / kappa, n_beta)
    deficit = 2 * np.pi - kappa * betas
    i0 = int(np.argmin(np.abs(deficit)))
    beta_star = float(betas[i0])
    return dict(kappa=float(kappa), beta_smooth=beta_star, T_c=1.0 / beta_star,
                T_c_closed=float(kappa / (2 * np.pi)),
                deficit_at_beta_star=float(deficit[i0]))


# ---- E2: the actual t -> i tau rotation of the propagator -------------------------------
def propagator(x, t, m=1.0, hbar=1.0):
    """Free-particle kernel K(x;t) = sqrt(m/(2 pi i hbar t)) exp(i m x^2 / (2 hbar t)). t may be
    complex: t = T0 e^{-i theta}. theta=0 -> oscillatory (Lorentzian, photon corner); theta=pi/2 ->
    t=-i T0, K -> real Gaussian heat kernel (Euclidean, black-hole/thermal corner)."""
    return np.sqrt(m / (2 * np.pi * 1j * hbar * t)) * np.exp(1j * m * x ** 2 / (2 * hbar * t))


def rotation_scan(T0=1.0):
    """Rotate theta in [0, pi/2]; report the oscillation 'visibility' of Re K over x (a proxy for
    interference) -- high at theta=0 (oscillatory photon corner), -> monotone/damped at theta=pi/2
    (Euclidean black-hole corner)."""
    x = np.linspace(-6, 6, 1201)
    rows = []
    for theta in np.linspace(0, np.pi / 2, 7):
        t = T0 * np.exp(-1j * theta)
        K = propagator(x, t)
        reK = K.real
        # number of sign changes of Re K = a clean proxy for oscillatory (Lorentzian) vs
        # monotone-decaying (Euclidean) behaviour:
        sign_changes = int(np.sum(np.abs(np.diff(np.sign(reK))) > 0))
        # imaginary part magnitude relative to total (Euclidean kernel is purely real):
        imag_frac = float(np.mean(np.abs(K.imag)) / (np.mean(np.abs(K)) + 1e-15))
        rows.append(dict(theta=float(theta), sign_changes_ReK=sign_changes,
                         imag_fraction=imag_frac))
    return rows


# ---- E3: the G7 refinement -- dynamical phase rotates, geometric phase does not ----------
def dynamical_action_rotates(m=1.0, hbar=1.0, T0=1.0, x0=0.0, x1=2.0):
    """For a classical free path x(t)=x0+(x1-x0)t/T over [0,T], the Lorentzian on-shell action is
    S_L = m (x1-x0)^2 / (2 T). Under t = -i tau (T -> -i T), exp(i S_L/hbar) -> exp(-S_E/hbar) with
    S_E = m (x1-x0)^2 / (2 T) (the Euclidean action, positive). We verify the analytic continuation
    numerically by rotating T = T0 e^{-i theta} and showing i S_L(T)/hbar goes from pure-imaginary
    (oscillatory) at theta=0 to pure-NEGATIVE-real (damping) at theta=pi/2."""
    dq2 = (x1 - x0) ** 2
    rows = []
    for theta in np.linspace(0, np.pi / 2, 7):
        T = T0 * np.exp(-1j * theta)
        S_L = m * dq2 / (2 * T)               # complex on-shell action
        expo = 1j * S_L / hbar                # the exponent of the weight exp(iS/hbar)
        rows.append(dict(theta=float(theta), re_exponent=float(expo.real),
                         im_exponent=float(expo.imag)))
    S_E = m * dq2 / (2 * T0)
    # at theta=pi/2: exponent should be -S_E/hbar (real, negative), i.e. exp -> exp(-S_E/hbar)
    end = rows[-1]
    rotates_to_minus_SE = abs(end["re_exponent"] - (-S_E / hbar)) < 1e-6 and abs(end["im_exponent"]) < 1e-6
    return rows, float(S_E), bool(rotates_to_minus_SE)


def geometric_phase_does_not_rotate(gamma=np.pi / 4):
    """A geometric/contextual holonomy enters as a bare phase exp(i gamma) (gamma = oint A, a pure
    number, NOT int L dt). Under t -> i tau it is unaffected: |exp(i gamma)| = 1 at every theta -- it
    never becomes a real (damping) cost. So the contextual phase A_sr stands OUTSIDE the Lorentzian-
    Euclidean rotation; only the dynamical phase A_dyn rotates into the cost C_c^E."""
    rows = []
    for theta in np.linspace(0, np.pi / 2, 7):
        # the geometric phase is theta-independent by construction; the magnitude stays 1.
        z = np.exp(1j * gamma)
        rows.append(dict(theta=float(theta), magnitude=float(abs(z)), phase=float(np.angle(z))))
    never_damps = all(abs(r["magnitude"] - 1.0) < 1e-12 for r in rows)
    return rows, bool(never_damps)


def main():
    print("cwf_ap_phaseE -- the photon/black-hole Wick-rotation corner duality (BRIDGE, G5) + G7\n")

    # ----- E1 -----
    print("E1  Gibbons-Hawking on the gain horizon: smoothness fixes T_c = kappa/2pi "
          "(STANDARD mechanism; CWF input = kappa from g'(x_h))")
    e1 = [gh_temperature(k) for k in [0.5, 1.0, 2.0, 4.0]]
    for r in e1:
        print(f"    kappa={r['kappa']:4.1f}  smooth period beta={r['beta_smooth']:.4f}  "
              f"T_c=1/beta={r['T_c']:.4f}  (kappa/2pi={r['T_c_closed']:.4f})")
    e1_ok = all(abs(r["T_c"] - r["T_c_closed"]) < 1e-2 for r in e1)
    print(f"    -> T_c = kappa/2pi recovered (smoothness/no-conical-singularity): {e1_ok}")

    # ----- E2 -----
    print("\nE2  the t -> i tau rotation: propagator from OSCILLATORY (photon) to DAMPED/thermal "
          "(black hole)")
    e2 = rotation_scan()
    for r in e2:
        print(f"    theta={r['theta']:.3f}  sign-changes(Re K)={r['sign_changes_ReK']:3d}  "
              f"imag_fraction={r['imag_fraction']:.3f}")
    e2_photon = e2[0]["sign_changes_ReK"]     # many oscillations at theta=0
    e2_bh = e2[-1]["sign_changes_ReK"]        # ~0 at theta=pi/2 (real Gaussian heat kernel)
    e2_ok = (e2_photon > 5) and (e2_bh <= 1) and (e2[-1]["imag_fraction"] < 1e-2)
    print(f"    -> photon corner (theta=0): {e2_photon} oscillations; black-hole corner "
          f"(theta=pi/2): {e2_bh} (real heat kernel). Rotation realised: {e2_ok}")

    # ----- E3 (G7) -----
    print("\nE3  G7 refinement: the DYNAMICAL phase rotates into the cost; the GEOMETRIC/contextual "
          "phase does NOT")
    e3_dyn, S_E, dyn_ok = dynamical_action_rotates()
    print(f"    dynamical: exp(i S_L/hbar) exponent rotates from pure-imaginary (theta=0) to "
          f"-S_E/hbar (theta=pi/2):")
    for r in e3_dyn:
        print(f"      theta={r['theta']:.3f}  exponent = {r['re_exponent']:+.4f} "
              f"{r['im_exponent']:+.4f} i")
    print(f"      -> at theta=pi/2 the exponent = -S_E/hbar = {-S_E:+.4f} (real, damping): {dyn_ok}")
    e3_geo, geo_never_damps = geometric_phase_does_not_rotate()
    print(f"    geometric/contextual holonomy exp(i*pi/4): magnitude stays 1 at every theta "
          f"(never becomes a cost): {geo_never_damps}")
    e3_ok = dyn_ok and geo_never_damps

    # ----- E4 verdict -----
    bridge_ok = e1_ok and e2_ok and e3_ok
    verdict = (
        "PHASE E: the corner duality holds as a BRIDGE (Status B), not a general theorem, exactly as "
        "G5 cautioned. (E1) On a gain horizon with surface gravity kappa = g'(x_h), the Euclidean "
        "continuation is smooth only at period beta = 2pi/kappa, giving the substrate temperature "
        "T_c = kappa/2pi (the Gibbons-Hawking mechanism -- STANDARD; the CWF input is only kappa from "
        "the gain gradient). (E2) Rotating t = T0 e^{-i theta} carries the free propagator from "
        "oscillatory (theta=0, the photon/Lorentzian corner, pure phase) to a real damped heat kernel "
        "(theta=pi/2, the black-hole/Euclidean/thermal corner) -- Movement VI's 'the mirror is time' "
        "made precise as t -> i tau. (E3 / G7) BUT the unity is PARTIAL: only the DYNAMICAL phase "
        "A_dyn = int L dt rotates into a real Euclidean cost (exp(i S_L) -> exp(-S_E)); the "
        "GEOMETRIC/contextual holonomy A_sr (Phase C; host-invariant, Phase D) is a topological phase "
        "(oint A), NOT an action, and does NOT rotate into a cost (|exp(i gamma)|=1 at every theta). "
        "So A_c and C_c form one analytic (Wick-dual) object in the DYNAMICAL sector only; the "
        "contextual phase -- the deepest CWF content -- stands OUTSIDE the photon/black-hole rotation, "
        "consistent with it being the separate (irreducibility/self-reference) axis. Net: the two "
        "exact corners are dynamical-sector Wick duals of one Z_c (a genuine, if conditional, bridge "
        "that sharpens Movement VI); the corner duality is rigorous WHERE the analog-gravity Rindler "
        "near-horizon and an analytic action hold (Ch5 kinematics-yes), and S in general (no clean "
        "horizon / no analytic continuation). The reviewer's 'do not say exact' is honoured: it is a "
        "bridge conjecture, made concrete, not an exact universal claim."
        if bridge_ok else
        "PHASE E INCOMPLETE -- a sub-test failed; inspect E1/E2/E3. The corner duality would then be "
        "only formal (Status S), not even a clean conditional bridge."
    )
    print(f"\n  E1 GH temperature T_c=kappa/2pi : {e1_ok}  (STANDARD mechanism)")
    print(f"  E2 photon->black-hole rotation  : {e2_ok}  (t -> i tau)")
    print(f"  E3 G7: dynamical rotates, contextual does NOT : {e3_ok}  (PARTIAL unity)")
    print(f"\n  CORNER DUALITY AS A BRIDGE (B): {bridge_ok}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseE_results.json")
    R = dict(
        E1_gibbons_hawking=dict(rows=e1, T_c_recovered=bool(e1_ok),
                                label="STANDARD GH smoothness->temperature; CWF input = kappa=g'(x_h)"),
        E2_wick_rotation=dict(rows=e2, photon_oscillations=e2_photon, blackhole_oscillations=e2_bh,
                              rotation_realised=bool(e2_ok),
                              label="photon (oscillatory) -> black hole (thermal heat kernel) via t->i tau"),
        E3_G7_refinement=dict(dynamical=e3_dyn, S_E=S_E, dynamical_rotates=bool(dyn_ok),
                              geometric=e3_geo, geometric_never_damps=bool(geo_never_damps),
                              label=("dynamical phase Wick-dual to cost; geometric/contextual holonomy "
                                     "topological, does NOT rotate -> A_c & C_c one analytic object in "
                                     "the dynamical sector ONLY")),
        corner_duality_bridge=bool(bridge_ok),
        verdict=verdict,
        note=("Corner duality = BRIDGE (B), not a theorem (G5). GH smoothness->T_c=kappa/2pi and the "
              "t->i tau propagator rotation are STANDARD; CWF content = identifications (A_c=self-ref "
              "phase, C_c=self-description cost, kappa from gain horizon). G7: A_c and C_c are one "
              "analytic object only in the dynamical sector; the contextual holonomy stands apart "
              "(topological, host-invariant) -- so the photon/black-hole Wick duality is the dynamical "
              "sector's, and contextuality is the separate axis. Sharpens Movement VI ('mirror is time' "
              "= t->i tau) as a conditional bridge, not an exact universal claim."))
    json.dump(R, open(out, "w"), indent=2)
    plot(e1, e2, e3_dyn)
    print(f"\nWrote {out}")


def plot(e1, e2, e3_dyn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    # E1: T_c vs kappa
    ks = [r["kappa"] for r in e1]; Tc = [r["T_c"] for r in e1]
    ax1.plot(ks, Tc, "o-", color="C0", label="T_c = 1/beta (smoothness)")
    kk = np.linspace(0.4, 4.2, 100)
    ax1.plot(kk, kk / (2 * np.pi), "C3--", label=r"$\kappa/2\pi$ (Gibbons-Hawking)")
    ax1.set_xlabel(r"surface gravity $\kappa$ (gain gradient)")
    ax1.set_ylabel(r"$T_c$"); ax1.set_title("E1: horizon temperature\nfrom Euclidean smoothness")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # E2: oscillations vs theta (photon->black hole)
    th = [r["theta"] for r in e2]; sc = [r["sign_changes_ReK"] for r in e2]
    ax2.plot(th, sc, "s-", color="C4")
    ax2.set_xlabel(r"rotation angle $\theta$  ($0$=Lorentzian, $\pi/2$=Euclidean)")
    ax2.set_ylabel("oscillations of Re$\\,K$")
    ax2.set_title("E2: $t\\to i\\tau$\nphoton (oscillatory) $\\to$ black hole (thermal)")
    ax2.axvline(0, color="C3", ls=":", lw=1); ax2.axvline(np.pi / 2, color="C0", ls=":", lw=1)
    ax2.grid(alpha=0.3)

    # E3: dynamical exponent rotates (real part goes negative); geometric stays |.|=1
    th3 = [r["theta"] for r in e3_dyn]
    re = [r["re_exponent"] for r in e3_dyn]; im = [r["im_exponent"] for r in e3_dyn]
    ax3.plot(th3, re, "o-", color="C3", label="Re exponent (dynamical) -> $-S_E/\\hbar$ (cost)")
    ax3.plot(th3, im, "o-", color="C0", label="Im exponent (dynamical, oscillatory)")
    ax3.axhline(1.0, color="C2", ls="--", lw=1.5, label="geometric phase $|e^{i\\gamma}|=1$ (no rotation)")
    ax3.set_xlabel(r"rotation angle $\theta$")
    ax3.set_ylabel("weight exponent")
    ax3.set_title("E3 (G7): dynamical phase rotates to cost;\ncontextual holonomy does NOT")
    ax3.legend(fontsize=7.5); ax3.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseE.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
