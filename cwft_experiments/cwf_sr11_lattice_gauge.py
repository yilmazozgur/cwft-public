"""
cwf_sr11_lattice_gauge.py -- Segment 2 of the (C) bridge: lattice-gauge-on-substrate.
Concrete test of the chain  closure -> referenceless local self-reference phase ->
lattice U(1) gauge connection -> Wilson-loop holonomy = Aharonov-Bohm phase = sr9's
self-reference cocycle, spatially gauged.

WHERE THIS SITS. (B)/sr10 gives, from CLOSURE (no external observer): the self-referential
i, AND -- the same root -- no global phase reference (only relative phases physical: the
gauge premise). sr9 gives the abstract self-reference U(1) holonomy (a geometric phase).
Segment 2 asks: spread that referenceless U(1) phase over the substrate's SPATIAL/causal
structure, demand LOCAL (per-site) phase invariance, and does the standard gauge step
yield a U(1) lattice gauge field whose Wilson-loop holonomy is the Aharonov-Bohm phase --
the spatial, observable realization of the same self-reference cocycle?

WHAT IS COMPUTED + VERIFIED here (the gauge kinematics, all exact):
  (1) GAUGE STRUCTURE. Sites carry a U(1) phase; links carry U_{nm}=e^{i theta_{nm}}
      transforming as theta_{nm} -> theta_{nm} + alpha_n - alpha_m under a local rephasing
      alpha. The plaquette flux F (lattice curl) and Wilson loops W = prod U are
      GAUGE-INVARIANT.
  (2) AHARONOV-BOHM. With flux Phi confined to ONE plaquette p0 (a lattice solenoid; the
      discrete-vortex vector potential), the local field strength F=0 on every other
      plaquette, yet a loop ENCLOSING p0 has Wilson holonomy W=e^{i Phi} != 1, while a loop
      NOT enclosing it has W=1 -- even though both loop paths run through the F=0 region.
      The textbook AB effect.
  (3) NO GLOBAL SECTION (= the obstruction; the tie to (B)/sr9). The plaquette flux is
      gauge-invariant, so a nonzero Phi CANNOT be gauged away: there is no global phase
      section trivializing all links. The Wilson holonomy IS that obstruction -- a
      gauge-invariant U(1) 1-cocycle, exactly the type of object as sr9's geometric phase
      (-arg Tr prod P_k): both are products of U(1) phases around a loop, gauge-invariant,
      non-trivial iff "something is enclosed" (flux here; solid angle in sr9).

THE CHAIN (what the construction shows, with its inputs flagged):
  closure (B) ==> [self-referential i] + [no global phase reference]
              ==> a referenceless local U(1) phase on the substrate's sites
              ==> (this construction) lattice U(1) connection, Wilson holonomy = AB
              ==> AB holonomy = sr9's self-reference cocycle, spatially gauged.

HONEST SCOPE (kept explicit, this is NOT a derivation of electromagnetism):
  * KINEMATICS ONLY. This builds the connection/holonomy (the AB / Wilson-loop structure).
    It does NOT derive Maxwell DYNAMICS (no field action / equations of motion) nor the
    coupling constant e. Those need an additional dynamical principle.
  * THE GAUGE MACHINERY IS TEXTBOOK (Wegner-Wilson lattice gauge theory). The CWF content
    is the IDENTIFICATION of its inputs with self-reference (the per-site U(1) = the
    self-referential phase of (B); referencelessness = closure) and of its holonomy with
    sr9's self-reference cocycle. The (B) inputs are ASSUMED here, not re-derived.
  * So Segment 2 is closed at the KINEMATIC level: the chain to the AB holonomy is now a
    computed construction, not an analogy -- the double duty's payoff touches observable
    physics (AB). Dynamics + coupling + the (B) premises remain.

CPU; numpy only. Atomic write.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def vortex_links(L, p0, Phi):
    """Discrete-vortex (lattice solenoid) vector potential: link phases whose curl is
    Phi at plaquette p0 and 0 elsewhere. Sites (x,y), x,y in 0..L-1; plaquette (px,py)
    has lower-left site (px,py), px,py in 0..L-2; its centre is (px+0.5, py+0.5)."""
    xc, yc = p0[0] + 0.5, p0[1] + 0.5
    ang = np.zeros((L, L))
    for x in range(L):
        for y in range(L):
            ang[x, y] = np.arctan2(y - yc, x - xc)
    H = np.zeros((L, L))   # H[x,y]: link (x,y)->(x+1,y), x in 0..L-2
    V = np.zeros((L, L))   # V[x,y]: link (x,y)->(x,y+1), y in 0..L-2
    for x in range(L - 1):
        for y in range(L):
            H[x, y] = (Phi / (2 * np.pi)) * wrap(ang[x + 1, y] - ang[x, y])
    for x in range(L):
        for y in range(L - 1):
            V[x, y] = (Phi / (2 * np.pi)) * wrap(ang[x, y + 1] - ang[x, y])
    return H, V


def plaquette_flux(H, V, L):
    F = np.zeros((L - 1, L - 1))
    for px in range(L - 1):
        for py in range(L - 1):
            F[px, py] = wrap(H[px, py] + V[px + 1, py] - H[px, py + 1] - V[px, py])
    return F


def wilson_loop(H, V, x0, y0, x1, y1):
    """Holonomy (sum of link phases, = arg of prod U) around the rectangular loop with
    corners (x0,y0)->(x1,y0)->(x1,y1)->(x0,y1)->back, x0<x1, y0<y1."""
    s = 0.0
    for x in range(x0, x1):          # bottom edge ->
        s += H[x, y0]
    for y in range(y0, y1):          # right edge ^
        s += V[x1, y]
    for x in range(x1 - 1, x0 - 1, -1):  # top edge <-
        s -= H[x, y1]
    for y in range(y1 - 1, y0 - 1, -1):  # left edge v
        s -= V[x0, y]
    return wrap(s)


def gauge_transform(H, V, L, alpha):
    Hg, Vg = H.copy(), V.copy()
    for x in range(L - 1):
        for y in range(L):
            Hg[x, y] += alpha[x + 1, y] - alpha[x, y]
    for x in range(L):
        for y in range(L - 1):
            Vg[x, y] += alpha[x, y + 1] - alpha[x, y]
    return Hg, Vg


def main():
    print("cwf_sr11 -- lattice-gauge-on-substrate: closure -> ... -> AB holonomy\n")
    checks = {}
    L = 7
    p0 = (3, 3)                     # central plaquette = the solenoid
    Phi = 2.0                       # confined flux (rad), generic (not a multiple of pi)
    H, V = vortex_links(L, p0, Phi)
    F = plaquette_flux(H, V, L)

    # (1) the flux is confined to p0
    flux_p0 = F[p0[0], p0[1]]
    flux_elsewhere = max(abs(F[px, py]) for px in range(L - 1) for py in range(L - 1)
                         if (px, py) != p0)
    checks["flux confined to one plaquette"] = (abs(flux_p0 - Phi) < 1e-9 and flux_elsewhere < 1e-9)
    print(f"(1) flux at p0 = {flux_p0:.4f} (=Phi={Phi}); max |flux| elsewhere = "
          f"{flux_elsewhere:.2e} (=0): {checks['flux confined to one plaquette']}")

    # (2) Aharonov-Bohm: loop ENCLOSING p0 vs NOT, both running through F=0 region
    W_enc = wilson_loop(H, V, 1, 1, 6, 6)        # large loop around centre, encloses p0
    W_not = wilson_loop(H, V, 0, 0, 3, 3)         # loop in a corner, does NOT enclose p0
    enc_ok = abs(wrap(W_enc - Phi)) < 1e-9
    not_ok = abs(W_not) < 1e-9
    checks["AB: enclosing loop holonomy = Phi"] = enc_ok
    checks["AB: non-enclosing loop holonomy = 0"] = not_ok
    # confirm the enclosing loop's PATH runs through zero-field plaquettes (AB hallmark)
    border_field = max(abs(F[px, py]) for px in (1, 5) for py in range(1, 6)) if L >= 7 else 0
    checks["AB: loop path in zero-field region"] = (border_field < 1e-9)
    print(f"(2) AB: enclosing-loop holonomy = {W_enc:.4f} (=Phi: {enc_ok}); "
          f"non-enclosing = {W_not:.2e} (=0: {not_ok}); loop path field "
          f"{border_field:.1e} (=0): the holonomy is nonzero where B=0 -- the AB effect.")

    # (3) NO GLOBAL SECTION: plaquette flux is gauge-invariant => Phi can't be gauged away
    rng = np.random.default_rng(0)
    inv = 0.0
    for _ in range(8):
        alpha = rng.uniform(0, 2 * np.pi, (L, L))
        Hg, Vg = gauge_transform(H, V, L, alpha)
        Fg = plaquette_flux(Hg, Vg, L)
        Wg = wilson_loop(Hg, Vg, 1, 1, 6, 6)
        inv = max(inv, np.max(np.abs(wrap(Fg - F))), abs(wrap(Wg - W_enc)))
    checks["flux and holonomy gauge-invariant (cocycle)"] = (inv < 1e-9)
    checks["nonzero flux cannot be gauged away (no global section)"] = (abs(flux_p0) > 1e-6)
    print(f"(3) under random gauge transformations: max drift of flux & holonomy = "
          f"{inv:.1e} (gauge-invariant: {checks['flux and holonomy gauge-invariant (cocycle)']}); "
          f"so the nonzero Phi is a genuine obstruction -- NO global phase section "
          f"trivializes the links.")

    # (4) same cocycle TYPE as sr9: a gauge-invariant U(1) holonomy = product of phases
    #     around a loop, non-trivial iff something is enclosed (flux here; solid angle in
    #     sr9). The Wilson loop = sum of enclosed plaquette fluxes (lattice Stokes).
    enclosed_sum = wrap(sum(F[px, py] for px in range(1, 6) for py in range(1, 6)))
    checks["Wilson loop = sum of enclosed fluxes (Stokes)"] = abs(wrap(W_enc - enclosed_sum)) < 1e-9
    print(f"(4) lattice Stokes: enclosing holonomy {W_enc:.4f} = sum of enclosed plaquette "
          f"fluxes {enclosed_sum:.4f}: {checks['Wilson loop = sum of enclosed fluxes (Stokes)']}. "
          f"This is a gauge-invariant U(1) 1-cocycle -- the SAME object-type as sr9's "
          f"geometric phase (product of U(1) phases around a loop, obstruction to a global "
          f"section); the AB holonomy is the spatial, gauged realization of sr9's self-"
          f"reference cocycle.")

    all_ok = all(checks.values())
    false_keys = [k for k, v in checks.items() if not v]
    if false_keys:
        print(f"\n  FALSE checks: {false_keys}")

    verdict = (
        "SEGMENT 2 CLOSED AT THE KINEMATIC LEVEL -- the chain to the AB holonomy is now a "
        "computed construction. Feeding the substrate's spatial structure a referenceless "
        "local U(1) phase and demanding local invariance yields a U(1) lattice gauge "
        "connection whose Wilson-loop holonomy is the Aharonov-Bohm phase: nonzero around "
        f"confined flux Phi={Phi} (W=e^{{i{Phi}}}) even where the local field is zero, "
        "gauge-invariant, and the genuine obstruction to a global phase section. That "
        "holonomy is the SAME gauge-invariant U(1) cocycle as sr9's self-reference phase, "
        "spatially gauged -- so the (C) chain closure(B) -> {self-ref i + no global phase "
        "reference} -> referenceless local U(1) -> lattice gauge -> AB holonomy is concrete, "
        "not an analogy. The closure premise's DOUBLE DUTY now touches observable physics: "
        "the no-global-reference it supplies (the gauge premise) gauges, spatially, into the "
        "AB holonomy. HONEST SCOPE: KINEMATICS ONLY -- this builds the connection/holonomy, "
        "NOT Maxwell dynamics (no field action/equations of motion) nor the coupling e; the "
        "gauge machinery is textbook lattice gauge theory, the CWF content being the "
        "identification of its inputs with self-reference (B) and its holonomy with sr9's "
        "cocycle; the (B) premises (the phase IS self-referential; closure) are assumed, not "
        "re-derived. So the chain reaches the AB holonomy at the kinematic level; full "
        "electromagnetism (dynamics + coupling) remains beyond this construction."
    ) if all_ok else "INCOMPLETE: a check failed -- inspect."
    print(f"\nall checks pass: {all_ok}\n\n{verdict}")

    out = os.path.join(HERE, "results.json")
    try:
        R = json.load(open(out))
    except Exception:
        R = {}
    R["SR11_lattice_gauge"] = dict(
        checks={k: bool(v) for k, v in checks.items()}, all_verified=bool(all_ok),
        L=L, p0=list(p0), Phi=float(Phi), W_enclosing=float(W_enc), W_not_enclosing=float(W_not),
        flux_at_p0=float(flux_p0), max_flux_elsewhere=float(flux_elsewhere),
        gauge_drift=float(inv), verdict=verdict,
        note=("Segment 2 of the (C) bridge, kinematic level. Lattice U(1) gauge theory on "
              "the substrate: referenceless local self-reference phase -> link variables "
              "-> Wilson-loop holonomy = Aharonov-Bohm phase (confined flux Phi=2.0; "
              "enclosing loop W=e^{iPhi}, non-enclosing W=1, holonomy nonzero where local "
              "field=0). Flux & holonomy gauge-invariant (a U(1) 1-cocycle); nonzero flux "
              "has no global section -- the same obstruction TYPE as sr9's self-reference "
              "cocycle, spatially gauged. Wilson loop = sum of enclosed fluxes (lattice "
              "Stokes). Closes the (C) chain closure(B)->referenceless local U(1)->lattice "
              "gauge->AB holonomy at the KINEMATIC level (the double duty's gauge premise "
              "gauges into AB). SCOPE: kinematics only -- NOT Maxwell dynamics or the "
              "coupling e; gauge machinery is textbook (Wegner-Wilson), CWF content is the "
              "identification of inputs with self-reference (B) and holonomy with sr9; the "
              "(B) premises are assumed. Atomic write."))
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=2)
    os.replace(tmp, out)
    plot(F, L, p0)
    print("\nWrote results.json key: SR11_lattice_gauge (atomic)")


def plot(F, L, p0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(F.T, origin="lower", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.add_patch(plt.Rectangle((0.5, 0.5), 5, 5, fill=False, ec="k", lw=2, ls="-"))
    ax.text(3, 5.3, "enclosing loop: holonomy $=\\Phi$ (AB)", ha="center", va="top", fontsize=9)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 3, fill=False, ec="gray", lw=1.5, ls="--"))
    ax.text(1, -0.9, "non-enclosing: $0$", ha="center", fontsize=8, color="gray")
    ax.plot(p0[0], p0[1], "k*", ms=16)
    ax.text(p0[0] + 0.2, p0[1], r"$\Phi$ confined", fontsize=9)
    fig.colorbar(im, ax=ax, label="plaquette flux (local field)")
    ax.set_title("Lattice-gauge-on-substrate: AB holonomy = self-reference cocycle, gauged\n"
                 "field confined to one plaquette; loop holonomy nonzero where field $=0$",
                 fontsize=10.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_SR11_lattice_gauge.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
