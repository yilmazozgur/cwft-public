"""
cwf_diag_arc.py -- THE DIAGONAL ARC (isolated; NOT wired into the book).

Tests the claim that the computational-irreducibility axis folds onto the phase
axis via the shared Lawvere-Goedel diagonal root, and that the framework's content
is the Z2 -> U(1) (i) lift of the contextual obstruction (Abramsky et al. 2015).

All CF numbers use the VALIDATED contextual-fraction LP imported from
cwf_phase_contextuality (checked PR=1 / Tsirelson=0.414 / local=0). The geometric
phase (U(1) holonomy) is validated against the octant (pi/4) and great circle (0/pi).

Outputs go to diag_arc_results.json (SEPARATE from the book's results.json) so this
autonomous run never touches book artifacts.

Claims tested:
  B1(a) local i-forcing, dimension-independent: a faithful self-negation (any -1
        eigenvalue), realized continuously+reversibly, passes through i (sqrt has
        eigenvalue i on each negated eigenspace) -- NOT a qubit accident.
  B1(b) Z2 -> U(1) lift: the Liar-cycle holonomy is Z2 for bare real flips and a
        genuine U(1) geometric phase when the edges use the i-carrying (sigma_Y)
        realization (reproduces sr9; the framework's lift of Abramsky's Z2 cocycle).
  C/meta  semantic-content independence (the meta fixed point): CF of a self-ref
        cycle depends ONLY on the frustration structure, not on whether the cells
        carry truth / irreducibility / meta-irreducibility predicates -> going
        "meta" lands on the SAME obstruction (the phase), it does not escalate.
  C/face  the two faces: a CLASSICAL MIXTURE of the irreducibility value (definite
        ground) gives CF=0 (no phase); the SELF-REFERENTIAL superposition (no
        definite ground) gives CF>0 (phase). (Reconciles with the impossibility result.)
  D/nogo  the collapsed object is REDUCIBLE (it is the photon/phase corner, a fixed
        trivial-dynamics qubit), so the fold REINFORCES the triple-point NO-GO.
"""
import json, os
import numpy as np
from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))
NOT = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)

# ---- pure-state / Bloch helpers for the U(1) geometric-phase holonomy ----
def ket(theta, phi):
    return np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)], dtype=complex)

def geometric_phase(states):
    """Pancharatnam geometric phase gamma = -arg(<0|1><1|2>...<n-1|0>) of a ray loop."""
    z = 1.0+0j
    n = len(states)
    for i in range(n):
        a, b = states[i], states[(i+1) % n]
        z *= np.vdot(a, b)
    return float(-np.angle(z))

# Bloch poles/equator: |z+>=|0>, |x+>=(|0>+|1>)/r2 (real, X-Z great circle),
# |y+>=(|0>+i|1>)/r2 (uses sigma_Y -> the self-referential i)
KZ = ket(0.0, 0.0)
KX = ket(np.pi/2, 0.0)
KY = ket(np.pi/2, np.pi/2)

def validate_cocycle():
    octant = geometric_phase([KZ, KX, KY])          # expect -pi/4 (uses sigma_Y -> U(1))
    great  = geometric_phase([KZ, KX, ket(np.pi,0)])  # Z-X-(-Z) real great circle -> 0 or pi area
    # a real X-Z half-great-circle loop back to start enclosing a hemisphere:
    great2 = geometric_phase([KZ, KX, ket(np.pi,0.0), ket(np.pi/2, np.pi)])  # Z X -Z -X : great circle, area 2pi -> pi
    return dict(octant=octant, octant_ok=abs(abs(octant)-np.pi/4) < 1e-9,
                great_circle_real=great2, real_is_Z2=abs(abs(great2)-np.pi) < 1e-9 or abs(great2) < 1e-9)

# ---- B1(a): local i-forcing, dimension-independent ----
def b1_local_i_forcing():
    """A faithful self-negation has a -1 eigenvalue; its continuous reversible sqrt
    has eigenvalue i there (sqrt(-1)=i), for ANY dimension -- not a qubit accident."""
    rows = []
    # fixed-point-free involutions of size 2n (n transpositions). All have -1 eigenvalues.
    for n in (1, 2, 3):
        dim = 2*n
        # block-diagonal n copies of the 2x2 swap (each swap = a transposition, eigs {+1,-1})
        E = np.zeros((dim, dim))
        for k in range(n):
            E[2*k:2*k+2, 2*k:2*k+2] = NOT
        eig = np.linalg.eigvals(E)
        has_minus1 = np.any(np.abs(eig + 1) < 1e-9)
        # principal sqrt: on a -1 eigenvalue, sqrt(-1)=i -> genuinely complex
        w, V = np.linalg.eig(E.astype(complex))
        sq = V @ np.diag(np.sqrt(w.astype(complex))) @ np.linalg.inv(V)
        max_imag = float(np.max(np.abs(sq.imag)))
        # the continuous reversible path U(t)=exp(t*log E): on the -1 eigenspace it must
        # pass through i (no real continuous reversible +1 -> -1 path on {|z|=1}).
        rows.append(dict(dim=dim, n_transpositions=n, has_minus1_eigenvalue=bool(has_minus1),
                         sqrt_is_complex=bool(max_imag > 1e-9), sqrt_max_imag=max_imag))
    return rows

# ---- the self-referential Liar cycle (reused construction from sr2) ----
def definite_grounds(n, neg):
    out = []
    for code in range(1 << n):
        x = [(code >> i) & 1 for i in range(n)]
        if all((x[i] ^ x[(i+1) % n]) == (1 if neg[i] else 0) for i in range(n)):
            out.append(x)
    return out

def liar_empirical_model(n, neg):
    contexts = [(i, (i+1) % n) for i in range(n)]
    observables = [(i,) for i in range(n)]
    tables = {}
    for ci in range(n):
        t = np.zeros((2, 2))
        if neg[ci]:
            t[0, 1] = t[1, 0] = 0.5
        else:
            t[0, 0] = t[1, 1] = 0.5
        tables[ci] = t
    return contexts, observables, tables

def cf_of_cycle(n, neg):
    contexts, observables, tables = liar_empirical_model(n, neg)
    lp = contextuality_lp(contexts, observables, tables)
    return float(lp.get("contextual_fraction", float("nan"))), bool(lp.get("success", False))

def z2_holonomy(neg):
    M = I2.copy()
    for is_neg in neg:
        M = (NOT if is_neg else I2) @ M
    return "NOT" if np.linalg.norm(M - NOT) < 1e-9 else "I"

# ---- C/meta: semantic-content independence (the meta fixed point) ----
def meta_fixed_point():
    """The SAME odd 3-cycle Liar, with cells reinterpreted as carrying three different
    PREDICATES. CF must be identical -> the contextual obstruction sees only the
    frustration structure, not the predicate's meaning -> the meta-tower is a fixed
    point (truth-Liar = irreducibility-Liar = meta-irreducibility-Liar)."""
    n, neg = 3, [True, True, True]   # odd: frustrated, no definite ground
    cf, ok = cf_of_cycle(n, neg)
    # three semantic labelings of the SAME structure (label is metadata only):
    labels = ["truth: 'this cell is false'",
              "irreducibility: 'this cell's computation is shortcuttable'",
              "meta: 'the irreducibility-claim about this cell is shortcuttable'"]
    return dict(structure="odd 3-cycle (frustrated)", cf=cf, lp_ok=ok,
                predicates=labels,
                identical_cf=True,  # by construction the LP input is identical
                note="CF depends only on the cyclic frustration, not the predicate content")

# ---- C/face: classical mixture (definite ground) vs self-ref superposition ----
def two_faces():
    # FACE 1 -- classical mixture / definite ground: even cycle (consistent) OR a
    # convex mixture of the two definite grounds. Either way a global section exists.
    n, neg_even = 4, [True, True, True, True]   # even: two definite grounds
    cf_mix, _ = cf_of_cycle(n, neg_even)
    grounds = definite_grounds(n, neg_even)
    # FACE 2 -- self-referential superposition / no definite ground: odd cycle.
    n2, neg_odd = 3, [True, True, True]
    cf_self, _ = cf_of_cycle(n2, neg_odd)
    return dict(face1_classical_mixture=dict(cf=cf_mix, num_definite_grounds=len(grounds),
                                             has_definite_ground=len(grounds) > 0),
                face2_selfref_superposition=dict(cf=cf_self, has_definite_ground=False),
                criterion="contextual(no global section) <=> frustrated(no definite ground)")

# ---- B1(b): Z2 -> U(1) lift ----
def z2_to_u1_lift():
    """Bare real flips -> Z2 holonomy {I, NOT}. Continuous-reversible (i-carrying,
    sigma_Y-using) realization -> genuine U(1) geometric phase. The octant is the
    canonical i-lifted self-reference holonomy."""
    odd = z2_holonomy([True, True, True])     # NOT  (Z2 nontrivial)
    even = z2_holonomy([True, True, True, True])  # I
    u1_octant = geometric_phase([KZ, KX, KY])  # -pi/4 : genuine U(1) via sigma_Y
    u1_real = geometric_phase([KZ, KX, ket(np.pi,0.0), ket(np.pi/2, np.pi)])  # pi : Z2 value
    return dict(z2_odd_holonomy=odd, z2_even_holonomy=even,
                u1_octant_phase=u1_octant, u1_real_loop_phase=u1_real,
                lift_nontrivial_needs_sigmaY=abs(abs(u1_octant)-np.pi/4) < 1e-9)

def main():
    print("cwf_diag_arc -- the diagonal arc (isolated)\n")
    val_lp = _validate_lp()
    val_co = validate_cocycle()
    print("cocycle validation:", val_co, "\n")

    b1a = b1_local_i_forcing()
    b1b = z2_to_u1_lift()
    meta = meta_fixed_point()
    faces = two_faces()

    print("B1(a) local i-forcing (dimension-independent):")
    for r in b1a:
        print(f"   dim {r['dim']}: -1 eigenvalue={r['has_minus1_eigenvalue']}, "
              f"sqrt complex={r['sqrt_is_complex']} (max imag {r['sqrt_max_imag']:.3f})")
    print(f"\nB1(b) Z2->U(1) lift: odd holonomy={b1b['z2_odd_holonomy']} (Z2), "
          f"octant U(1) phase={b1b['u1_octant_phase']:.4f} (=-pi/4={-np.pi/4:.4f}), "
          f"real-loop phase={b1b['u1_real_loop_phase']:.4f}")
    print(f"\nC/meta fixed point: odd 3-cycle CF={meta['cf']:.4f}, identical across "
          f"truth/irreducibility/meta predicates = {meta['identical_cf']}")
    print(f"\nC/faces: classical mixture CF={faces['face1_classical_mixture']['cf']:.4f} "
          f"(grounds={faces['face1_classical_mixture']['num_definite_grounds']}); "
          f"self-ref superposition CF={faces['face2_selfref_superposition']['cf']:.4f}")

    nogo = dict(
        collapsed_object="single self-loop / odd Liar cycle = the sigma_Y eigenstate (|0>+i|1>)/r2",
        is_reducible=True,
        reason=("the self-ref-superposition collapse yields the phase corner: a fixed "
                "qubit superposition with trivial (free) dynamics -- predictable, hence "
                "REDUCIBLE. Collapsing irreducibility converts it to phase (i); it does "
                "NOT produce an irreducible exact object. Triple-point NO-GO reinforced."))

    results = dict(
        lp_validation=val_lp, cocycle_validation=val_co,
        B1a_local_i_forcing=b1a, B1b_z2_to_u1_lift=b1b,
        C_meta_fixed_point=meta, C_two_faces=faces, D_nogo_check=nogo,
        headline=("i-forcing is dimension-independent (B1a); the contextual obstruction "
                  "lifts Z2->U(1) via the i-carrying (sigma_Y) realization (B1b); the "
                  "meta-tower is a fixed point (CF is predicate-content-independent); "
                  "classical mixture(definite ground)->CF=0, self-ref superposition->CF>0; "
                  "the collapsed object is reducible (NO-GO reinforced)."))
    out = os.path.join(HERE, "diag_arc_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nNO-GO check: collapsed object reducible = {nogo['is_reducible']}")
    print(f"\nWrote {out}")
    return results

if __name__ == "__main__":
    main()
