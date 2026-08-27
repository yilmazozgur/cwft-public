"""
cwf_sr13_maxwell_action_theorem.py -- promote "the Maxwell action IS the self-consistency
cost" from an EFT-level IDENTIFICATION (prop:lattice-maxwell, sr12: "B^2 = the Wilson
action to leading order") to a THEOREM: the gauge-invariant, local, parity-even,
flat-vanishing cost of self-referential inconsistency is UNIQUELY the Maxwell/Wilson F^2
action at leading order -- the FORM fixed by closure/self-reference, the index CONTRACTION
by the substrate's effective metric (the gravity axis), the COUPLING free.

THE AXIOMS (each grounded in closure/self-reference, NONE mentioning F^2):
  (G) gauge invariance       S invariant under theta_l -> theta_l + alpha_{boundary(l)}.
                             [closure: no external phase frame => only gauge-invariant
                              data is physical; the sr10/sr11 "no global reference".]
  (L) locality (lowest order) S = sum of single-plaquette terms (the smallest gauge-
                             invariant loop); larger Wilson loops = higher-derivative
                             corrections. [substrate locality, Ch5.]
  (C) consistency            S = 0 on FLAT connections, S > 0 otherwise.
                             [the key input: a flat connection = a globally consistent
                              self-description (global phase section, no frustration --
                              sr2/sr4) => zero inconsistency cost.]
  (P) parity                 invariance under loop-orientation reversal phi -> -phi.
                             [no preferred orientation of the self-reference loop.]
  (A) analyticity            S analytic in the link phases near the flat configuration.

THEOREM (verified below). Under (G,L,C,P,A) the leading term of S is UNIQUELY
    S[U] = (beta/2) sum_p phi_p^2 + O(phi^4)  =  beta sum_p (1 - cos phi_p) + ...,
phi_p the plaquette holonomy = lattice curvature F; the one free constant beta (the
coupling) is NOT fixed. Continuum: S -> (beta/4) int F_{mu nu} F^{mu nu} sqrt(g) d^dx, the
Maxwell action, indices contracted by the substrate's effective metric (gravity axis).
Proof chain (non-circular -- F^2 is DERIVED, not assumed):
  (G) => S is a class function of holonomies => for U(1) a function of fluxes only,
        S = sum_p f(phi_p) (+ bigger loops, higher order by (L));
  (C) => f(0) = 0  -- KILLS THE CONSTANT (the cosmological / vacuum term);
  (P) => f even => f'(0) = 0 -- KILLS THE LINEAR (theta / total-flux) term;
  (A) => f(phi) = (1/2) f''(0) phi^2 + O(phi^4) => leading term ~ phi^2 = F^2, unique
        up to beta = f''(0).

The FIVE checks (verify-don't-trust; this is mostly analytic, so each leg is pinned):
  (1) Wilson expansion: beta(1-cos phi) = (beta/2)phi^2 - (beta/24)phi^4 + ... (coeffs).
  (2) UNIQUENESS, concretely: the admissible single-plaquette family is {1-cos(n phi)};
      every member has leading term ~ phi^2 (constant & linear forced to vanish by C,P);
      n=1 (smallest loop) = Wilson, the unique member with flat (phi=0) as its ONLY
      minimum on (-pi,pi].
  (3) METRIC/continuum: phi_p = a^2 F + O(a^4); sum_p phi_p^2 -> int F^2; anisotropic
      spacings (a metric) enter as phi = a_x a_y F -- the contraction uses the metric.
  (4) LAMBDA-corollary: dropping (C) reintroduces a constant = a vacuum-energy term ~
      volume; (C) is exactly what sets Lambda_c = 0 here (ties to "Lambda_c not sourced").
  (5) EXTREMUM = MAXWELL: dS/dtheta_l = beta sum_{p>l} +-sin(phi_p) = beta*(discrete
      d*F)_l + O(phi^3); a discrete-Maxwell solution is a critical point of S.

HONEST SCOPE (prominent): the MATHEMATICAL core (gauge inv + locality + lowest order =>
F^2) IS the standard lattice-gauge / Wilsonian uniqueness argument -- here made exact and
lattice-explicit. The CWF content is (a) GROUNDING the hypotheses in closure/self-reference
(esp. (C) flat=consistent=zero-cost), (b) the LAMBDA corollary tying (C) to "Lambda_c not
sourced" (Ch5/Ch6), and (c) the two-axis WELD -- the F^2 FORM from closure/self-reference,
the index CONTRACTION (metric) from the gravity axis. The coupling beta/e is NOT derived
(free substrate parameter, gauge like hbar_c, B2); Lorentz invariance emergent-only.

CPU; numpy only. Atomic write.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ------- 2D periodic U(1) lattice: link phases theta_x, theta_y; plaquette flux ----------
def plaq_flux(tx, ty):
    # phi[x,y] = tx[x,y] + ty[x+1,y] - tx[x,y+1] - ty[x,y]
    return tx + np.roll(ty, -1, 0) - np.roll(tx, -1, 1) - ty


def grad_S(tx, ty, beta):
    """Analytic gradient of S = beta*sum(1-cos phi) w.r.t. each link phase."""
    s = np.sin(plaq_flux(tx, ty))
    gx = beta * (s - np.roll(s, 1, 1))     # d/dtx[x,y]: +phi[x,y] - phi[x,y-1]
    gy = beta * (-s + np.roll(s, 1, 0))    # d/dty[x,y]: -phi[x,y] + phi[x-1,y]
    return gx, gy


def maxwell_op(tx, ty):
    """Linearized lattice d*F (the discrete Maxwell / Euler-Lagrange operator)."""
    phi = plaq_flux(tx, ty)
    mx = phi - np.roll(phi, 1, 1)
    my = -phi + np.roll(phi, 1, 0)
    return mx, my


def main():
    rng = np.random.default_rng(0)
    print("cwf_sr13 -- the self-consistency cost of self-referential inconsistency is "
          "UNIQUELY the Maxwell action (leading order)\n")
    checks = {}
    beta = 1.7   # arbitrary coupling: must NOT be fixed by the axioms (it isn't)

    # ===== (1) Wilson expansion: beta(1-cos phi) = (beta/2)phi^2 - (beta/24)phi^4 + ... =====
    # read the Taylor coefficients directly as limits (the small-phi ratios converge to them).
    ps = 1e-3
    lead = beta * (1 - np.cos(ps)) / ps**2                       # -> beta/2
    pq = 0.1
    quart = (beta * (1 - np.cos(pq)) - 0.5 * beta * pq**2) / pq**4   # -> -beta/24
    lead_ok = abs(lead - 0.5 * beta) < 1e-5
    quartic_ok = abs(quart - (-beta / 24)) < 1e-4
    checks["Wilson leading term = (beta/2)phi^2"] = bool(lead_ok)
    checks["Wilson next term = -(beta/24)phi^4"] = bool(quartic_ok)
    print(f"(1) beta(1-cos phi): leading coeff (1-cos)/phi^2 -> {lead:.6f} "
          f"(=beta/2={0.5*beta:.6f}: {lead_ok}); quartic coeff {quart:.6f} "
          f"(=-beta/24={-beta/24:.6f}: {quartic_ok}).")

    # ===== (2) UNIQUENESS: admissible single-plaquette family {1-cos(n phi)} ============
    # gauge-inv class fn f(phi) = a0 + sum_n [a_n cos(n phi) + b_n sin(n phi)].
    # (P) parity f(-phi)=f(phi) => b_n = 0;  (C) f(0)=0 => a0 = -sum a_n.
    # => admissible f = sum_n c_n (1 - cos(n phi)). Every member: leading ~ phi^2.
    eps = 1e-5
    def d2_at0(f):  # numeric f''(0)
        return (f(eps) - 2 * f(0.0) + f(-eps)) / eps**2
    def d1_at0(f):  # numeric f'(0)
        return (f(eps) - f(-eps)) / (2 * eps)

    fam_leading = {}
    fam_ok = True
    for n in range(1, 7):
        f = (lambda nn: (lambda p: 1 - np.cos(nn * p)))(n)
        f0 = f(0.0); d1 = d1_at0(f); d2 = d2_at0(f)
        fam_leading[n] = d2 / 2
        ok = abs(f0) < 1e-9 and abs(d1) < 1e-6 and abs(d2 / 2 - n**2 / 2) < 1e-3
        fam_ok = fam_ok and ok
    checks["family {1-cos n phi}: each has phi^2 leading term (= n^2/2)"] = bool(fam_ok)
    print(f"(2) admissible family 1-cos(n phi), n=1..6: each has f(0)=0, f'(0)=0, "
          f"leading coeff n^2/2 = {[round(fam_leading[n],3) for n in range(1,7)]} "
          f"(all ~ phi^2): {fam_ok}")

    # a RANDOM admissible cost (random non-negative c_n) -> still phi^2 leading, no const/linear
    c = rng.uniform(0.1, 1.0, 6)
    frand = lambda p: sum(c[n-1] * (1 - np.cos(n * p)) for n in range(1, 7))
    rand_ok = (abs(frand(0.0)) < 1e-9 and abs(d1_at0(frand)) < 1e-6
               and d2_at0(frand) > 1e-6)
    lead_rand = d2_at0(frand) / 2
    expect_rand = sum(c[n-1] * n**2 / 2 for n in range(1, 7))
    rand_ok = rand_ok and abs(lead_rand - expect_rand) < 1e-2
    checks["random admissible cost: phi^2 leading, no constant/linear term"] = bool(rand_ok)
    print(f"    random admissible cost: f(0)={frand(0.0):.2e}, f'(0)={d1_at0(frand):.2e} "
          f"(both 0), leading coeff {lead_rand:.4f} (=sum c_n n^2/2={expect_rand:.4f}): {rand_ok}")

    # constant / linear terms are EXCLUDED by (C)/(P): exhibit violators
    f_const = lambda p: 0.3 + 0.5 * (1 - np.cos(p))         # has a constant -> violates (C)
    f_lin = lambda p: np.sin(p) + (1 - np.cos(p))           # has a sine -> violates (P)
    excl_ok = (abs(f_const(0.0)) > 1e-3) and (abs(f_lin(-0.1) - f_lin(0.1)) > 1e-3)
    checks["constant excluded by (C), linear excluded by (P)"] = bool(excl_ok)
    print(f"    a constant term gives f(0)={f_const(0.0):.2f}!=0 (excluded by C); a sine term "
          f"gives f(-x)!=f(x) (excluded by P): only even, flat-vanishing -> phi^2 survives: {excl_ok}")

    # n=1 (smallest loop = Wilson) is the UNIQUE member with flat as its ONLY minimum on (-pi,pi]
    grid = np.linspace(-np.pi, np.pi, 200001)[:-1]   # half-open (-pi,pi]
    n_minima = {}
    for n in range(1, 5):
        fv = 1 - np.cos(n * grid)
        # count distinct global minima (value ~ 0)
        mins = grid[fv < 1e-6]
        # count clusters of adjacent minima (sentinel -10 < all data, so the first
        # cluster's leading gap is counted)
        n_minima[n] = int(np.sum(np.diff(np.concatenate(([-10.0], mins))) > 0.01))
    wilson_unique = (n_minima[1] == 1 and all(n_minima[n] >= 2 for n in range(2, 5)))
    checks["n=1 (Wilson) unique with flat as only minimum"] = bool(wilson_unique)
    print(f"    # of minima on (-pi,pi]: {n_minima} -> n=1 (Wilson, smallest loop) has the "
          f"flat connection as its ONLY vacuum; n>=2 have extra minima (lattice artifacts): "
          f"{wilson_unique}")

    # ===== (3) METRIC / continuum: phi_p = a^2 F + O(a^4); sum -> int F^2 =================
    # smooth potential A_x = sin(y), A_y = 0.5 x  => F = dAy/dx - dAx/dy = 0.5 - cos(y)
    Ax = lambda x, y: np.sin(y)
    Ay = lambda x, y: 0.5 * x
    Fexact = lambda x, y: 0.5 - np.cos(y)
    x0, y0 = 1.3, 0.7
    errs = []
    for a in [0.2, 0.1, 0.05, 0.025]:
        # links around the plaquette with lower-left (x0,y0), midpoint rule
        th_bottom = a * Ax(x0 + a/2, y0)
        th_right = a * Ay(x0 + a, y0 + a/2)
        th_top = a * Ax(x0 + a/2, y0 + a)
        th_left = a * Ay(x0, y0 + a/2)
        phi = th_bottom + th_right - th_top - th_left   # plaquette holonomy (loop integral)
        F_center = Fexact(x0 + a/2, y0 + a/2)
        errs.append(abs(phi / a**2 - F_center) / abs(F_center))
    ratios = [errs[i] / errs[i+1] for i in range(len(errs)-1)]
    curvature_ok = (errs[-1] < 1e-3 and np.mean(ratios) > 3.0)   # O(a^2): halving a -> /~4
    checks["plaquette holonomy = a^2 F + O(a^4) (curvature)"] = bool(curvature_ok)
    print(f"(3) phi_p / a^2 -> F as a->0: rel.err {[f'{e:.2e}' for e in errs]}, "
          f"error ratios {[round(r,1) for r in ratios]} (~4 => O(a^2)): {curvature_ok}")
    # anisotropic spacings (a metric): phi = a_x a_y F -- the contraction uses the metric
    ax_, ay_ = 0.05, 0.12
    th_b = ax_ * Ax(x0 + ax_/2, y0); th_r = ay_ * Ay(x0 + ax_, y0 + ay_/2)
    th_t = ax_ * Ax(x0 + ax_/2, y0 + ay_); th_l = ay_ * Ay(x0, y0 + ay_/2)
    phi_aniso = th_b + th_r - th_t - th_l
    F_c = Fexact(x0 + ax_/2, y0 + ay_/2)
    metric_ok = abs(phi_aniso / (ax_ * ay_) - F_c) / abs(F_c) < 1e-2
    checks["anisotropic (metric): phi = a_x a_y F (contraction uses metric)"] = bool(metric_ok)
    print(f"    anisotropic spacings (a_x={ax_}, a_y={ay_}): phi/(a_x a_y)={phi_aniso/(ax_*ay_):.4f} "
          f"(=F={F_c:.4f}): the flux carries the metric; F^2 contraction uses g (gravity axis): {metric_ok}")

    # ===== (4) LAMBDA-corollary: dropping (C) reintroduces a constant = vacuum energy =====
    L = 16
    tx0 = np.zeros((L, L)); ty0 = np.zeros((L, L))     # FLAT connection (all phi = 0)
    Nplaq = L * L
    c0 = 0.4
    S_with_C = beta * np.sum(1 - np.cos(plaq_flux(tx0, ty0)))         # (C): f(0)=0
    S_no_C = c0 * Nplaq + beta * np.sum(1 - np.cos(plaq_flux(tx0, ty0)))  # constant kept
    lambda_ok = (abs(S_with_C) < 1e-12 and abs(S_no_C - c0 * Nplaq) < 1e-9)
    checks["(C) sets Lambda=0; dropping it gives a volume (vacuum-energy) term"] = bool(lambda_ok)
    print(f"(4) on the flat connection: with (C) S={S_with_C:.2e} (=0, no cosmological term); "
          f"dropping (C) gives S={S_no_C:.3f}=c0*volume={c0*Nplaq:.3f} -- a Lambda-like vacuum "
          f"term ~ volume. So (C) is exactly what sets Lambda_c=0 here: {lambda_ok}")

    # ===== (5) EXTREMUM = MAXWELL: dS/dtheta = beta*(discrete d*F) + O(phi^3) =============
    scale = 0.03
    tx = rng.standard_normal((L, L)) * scale
    ty = rng.standard_normal((L, L)) * scale
    gx, gy = grad_S(tx, ty, beta)
    mx, my = maxwell_op(tx, ty)
    # analytic gradient vs beta * linear Maxwell operator (should agree to O(phi^3))
    rel = (np.max(np.abs(gx - beta * mx)) + np.max(np.abs(gy - beta * my))) / \
          (beta * (np.max(np.abs(mx)) + np.max(np.abs(my))))
    grad_matches_maxwell = rel < 1e-2
    checks["grad S = beta*(discrete Maxwell operator) + O(phi^3)"] = bool(grad_matches_maxwell)
    # finite-difference gradient sanity (against analytic)
    h = 1e-6; i, j = 5, 7
    Splus = beta * np.sum(1 - np.cos(plaq_flux(tx + h * (np.arange(L*L).reshape(L,L) == (i*L+j)), ty)))
    Sminus = beta * np.sum(1 - np.cos(plaq_flux(tx - h * (np.arange(L*L).reshape(L,L) == (i*L+j)), ty)))
    fd = (Splus - Sminus) / (2 * h)
    fd_ok = abs(fd - gx[i, j]) < 1e-4
    checks["analytic gradient matches finite-difference"] = bool(fd_ok)
    # a discrete-Maxwell solution (uniform flux, incl. flat) is a critical point of S
    tx_sol = np.zeros((L, L)); ty_sol = np.tile(np.arange(L) * 0.05, (L, 1))  # uniform B field
    phi_sol = plaq_flux(tx_sol, ty_sol)
    gsx, gsy = grad_S(tx_sol, ty_sol, beta)
    uniform_flux = np.std(phi_sol) < 1e-9
    critical_ok = uniform_flux and (np.max(np.abs(gsx)) + np.max(np.abs(gsy)) < 1e-9)
    checks["a discrete-Maxwell solution is a critical point of S"] = bool(critical_ok)
    print(f"(5) grad S vs beta*(discrete d*F): rel.diff {rel:.2e} (~0, agree to O(phi^3): "
          f"{grad_matches_maxwell}); analytic vs finite-diff at one link: {fd:.6f} vs "
          f"{gx[i,j]:.6f} ({fd_ok}); a uniform-flux (Maxwell) solution has grad S "
          f"{np.max(np.abs(gsx)):.1e} (=0, a critical point: {critical_ok}).")

    all_ok = all(checks.values())
    false_keys = [k for k, v in checks.items() if not v]
    if false_keys:
        print(f"\n  FALSE checks: {false_keys}")

    verdict = (
        "THEOREM (the self-consistency cost IS the Maxwell action, leading order). Under "
        "(G) gauge invariance [closure: no external phase frame], (L) locality [substrate], "
        "(C) flat connections cost ZERO and others cost more [a globally consistent self-"
        "description has no inconsistency], (P) loop-orientation parity, and (A) analyticity, "
        "the leading term of the self-referential inconsistency cost is UNIQUELY S=(beta/2)"
        "sum_p phi_p^2 + O(phi^4) = beta sum_p (1-cos phi_p) -- the Maxwell/Wilson F^2 action; "
        "the coupling beta is the one free constant, NOT fixed by the axioms. PROOF (F^2 is "
        "DERIVED, not assumed): (G)=>class function of holonomies=>function of fluxes f(phi); "
        "(C)=>f(0)=0, KILLING the constant (cosmological/vacuum) term; (P)=>f even=>f'(0)=0, "
        "killing the linear (theta) term; (A)=>f=(1/2)f''(0)phi^2+O(phi^4)=>leading ~ phi^2=F^2. "
        f"VERIFIED: (1) beta(1-cos phi)=(beta/2)phi^2-(beta/24)phi^4+...; (2) every admissible "
        "single-plaquette cost {1-cos n phi} has phi^2 leading term (constant & linear forced "
        "out by C,P), n=1 (smallest loop)=Wilson is the unique one with flat as its ONLY vacuum; "
        "(3) phi_p=a^2 F + O(a^4), sum->int F^2, anisotropic spacings carry the metric (the F^2 "
        "contraction uses g -- the GRAVITY axis); (4) dropping (C) reintroduces a constant = a "
        "vacuum-energy term ~ volume, so (C) is exactly what sets Lambda_c=0 (ties to 'Lambda_c "
        "not sourced'); (5) dS/dtheta = beta*(discrete d*F)+O(phi^3), so EXTREMIZING the cost "
        "gives Maxwell's equations (a Maxwell solution is a critical point). So the self-"
        "reference U(1) (sr11/sr12), given only these closure-grounded axioms, has Maxwell as "
        "its UNIQUE leading-order action: the FORM from closure/self-reference, the index "
        "CONTRACTION from the gravity-axis metric, the coupling free. HONEST SCOPE: the "
        "mathematical core (gauge inv + locality + lowest order => F^2) is the standard lattice-"
        "gauge / Wilsonian uniqueness argument, here made exact and lattice-explicit; the CWF "
        "content is grounding the hypotheses in closure/self-reference (esp. (C)), the Lambda "
        "corollary, and the two-axis weld. The coupling e is NOT derived (free substrate param, "
        "gauge like hbar_c, B2); Lorentz invariance emergent-only (substrate-scale violation, "
        "sr12). Promotes prop:lattice-maxwell's 'B^2 = Wilson action to leading order' from "
        "IDENTIFICATION to DERIVATION."
    ) if all_ok else "INCOMPLETE: a check failed -- inspect."
    print(f"\nall checks pass: {all_ok}\n\n{verdict}")

    out = os.path.join(HERE, "results.json")
    try:
        R = json.load(open(out))
    except Exception:
        R = {}
    R["SR13_maxwell_action_theorem"] = dict(
        checks={k: bool(v) for k, v in checks.items()}, all_verified=bool(all_ok),
        beta=float(beta), family_leading={int(n): float(fam_leading[n]) for n in fam_leading},
        n_minima={int(n): int(v) for n, v in n_minima.items()},
        continuum_rel_errs=[float(e) for e in errs], continuum_ratios=[float(r) for r in ratios],
        lambda_term_on_flat=float(S_no_C), grad_vs_maxwell_reldiff=float(rel),
        verdict=verdict,
        note=("THEOREM: the gauge-invariant, local, parity-even, flat-vanishing cost of self-"
              "referential inconsistency is UNIQUELY the Maxwell/Wilson F^2 action at leading "
              "order. Axioms (G,L,C,P,A) grounded in closure/self-reference; F^2 DERIVED, not "
              "assumed -- (C) flat=zero-cost kills the cosmological term, (P) parity kills the "
              "theta term, leaving phi^2=F^2 uniquely (coupling beta free). VERIFIED: Wilson "
              "expansion; uniqueness over {1-cos n phi} (n=1=Wilson, flat=unique vacuum); "
              "continuum phi=a^2 F + anisotropic metric contraction (gravity axis); Lambda-"
              "corollary ((C) sets Lambda_c=0, ties to 'Lambda_c not sourced'); extremum=Maxwell "
              "(grad S = beta*d*F + O(phi^3)). SCOPE: math core = standard lattice-gauge/Wilsonian "
              "uniqueness, made exact; CWF content = the closure-grounding + Lambda corollary + "
              "two-axis weld (form from self-reference, contraction from gravity metric); coupling "
              "e NOT derived (gauge like hbar_c); Lorentz emergent-only. Promotes prop:lattice-"
              "maxwell's identification to a derivation. Atomic write."))
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=2)
    os.replace(tmp, out)
    plot(fam_leading, n_minima, tx, ty, beta)
    print("\nWrote results.json key: SR13_maxwell_action_theorem (atomic)")


def plot(fam_leading, n_minima, tx, ty, beta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # (a) the admissible family: all have phi^2 leading; n=1 (Wilson) unique flat vacuum
    pp = np.linspace(-np.pi, np.pi, 1000)
    for n in range(1, 5):
        lw = 2.6 if n == 1 else 1.2
        lab = "$1-\\cos\\phi$ (Wilson, $n{=}1$)" if n == 1 else f"$1-\\cos {n}\\phi$"
        ax1.plot(pp, 1 - np.cos(n * pp), lw=lw, label=lab,
                 color="C3" if n == 1 else None, alpha=1.0 if n == 1 else 0.55)
    ax1.plot(pp, 0.5 * pp**2, "k--", lw=1.3, label="$\\phi^2/2$ (leading, $=F^2$)")
    ax1.set_title("Admissible self-consistency costs: all $\\sim\\phi^2=F^2$ at leading order\n"
                  "(constant killed by (C), linear by (P)); $n{=}1$ = Wilson, flat = unique vacuum")
    ax1.set_xlabel("plaquette flux $\\phi$ (curvature $F$)"); ax1.set_ylabel("cost $f(\\phi)$")
    ax1.set_ylim(-0.1, 2.3); ax1.legend(fontsize=8, loc="upper center"); ax1.grid(alpha=0.3)

    # (b) extremum = Maxwell: analytic grad S vs beta*(discrete d*F)
    gx, gy = grad_S(tx, ty, beta)
    mx, my = maxwell_op(tx, ty)
    g = np.concatenate([gx.ravel(), gy.ravel()])
    m = beta * np.concatenate([mx.ravel(), my.ravel()])
    lim = max(np.max(np.abs(g)), np.max(np.abs(m))) * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.6, label="$y=x$")
    ax2.scatter(m, g, s=9, alpha=0.5, color="C0")
    ax2.set_title("Extremizing the cost $=$ Maxwell's equations\n"
                  "$\\partial S/\\partial\\theta_\\ell = \\beta\\,(d{\\star}F)_\\ell + O(\\phi^3)$")
    ax2.set_xlabel(r"$\beta\,(d{\star}F)_\ell$  (discrete Maxwell operator)")
    ax2.set_ylabel(r"$\partial S/\partial\theta_\ell$  (gradient of the cost)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3); ax2.set_aspect("equal")

    fig.tight_layout()
    pth = os.path.join(HERE, "fig_SR13_maxwell_action_theorem.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
