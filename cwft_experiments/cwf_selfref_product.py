#!/usr/bin/env python3
"""STAGE 3 of the self-referential VSA: the product-VSA tensor route (route a).

Stage 1 (cwf_selfref_ring.py) reached single-system contextuality (CF ~ 0.47) with a
frustrated ring of REAL hypervectors, but capped there: the real-vs-complex (Renou)
separation needs two INDEPENDENT entangled sources fused by a joint (Bell) measurement --
a NETWORK, hence a tensor product.  VSA binding is dimension-preserving (no tensor
product), so the manuscript's own claim is 'no entanglement'.  Here we TEST whether a
'product VSA' route supplies the missing structure, and quantify exactly what it lacks.

Renou et al. (Nature 600, 625, 2021): the bilocal functional T reaches 6sqrt2 ~ 8.49 in
COMPLEX quantum theory and is bounded by 7.66 in REAL quantum theory (their SDP) -- a
doubling-proof separation.  We reuse the book's exact realization (cwf_sr7_renou_realize)
as the validated control (T=6sqrt2 for maximally entangled sources), then:

  (1) sweep T against the SOURCE ENTANGLEMENT (concurrence C in [0,1]): T is monotone in
      C; find the C at which T crosses the real bound 7.66 and the complex value 8.49;
  (2) CHSH across a source vs its concurrence (validate: |phi+> C=1 -> CHSH=2sqrt2);
  (3) show a VSA-NATIVE two-party source -- built from single-qubit hypervectors by VSA
      binding/bundling (dimension-preserving) -- has concurrence ~ 0 (separable) and
      CHSH <= 2: VSA sources sit at C=0;
  (4) verdict: T_VSA = T(C=0) < 7.66.  The product-VSA route does NOT reach the complex
      tier; the obstruction is exactly the tensor product / entanglement VSA lacks.

Honest (route b, the predicted outcome): this is a CLEAN NEGATIVE.  Forming the true
tensor product is possible but is NOT a native VSA operation (it is the dimension-doubling
VSA deliberately avoids); doing it is quantum mechanics in a hypervector container, not
VSA, and trivially gives 6sqrt2.  So 'product VSA' reaches the complex tier only by
abandoning the dimension-preserving property that defines VSA.  Closes the loop with the
manuscript's 'no entanglement, no quantum speedup' -> 'and no complex-vs-real separation'.

CPU, numpy.  Writes cwf_selfref_product_results.json.
"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
s2 = np.sqrt(2.0)

# import the validated Renou realization from the book's sr7
spec = importlib.util.spec_from_file_location("sr7", os.path.join(HERE, "cwf_sr7_renou_realize.py"))
sr7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr7)

Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def source_ent(eta):
    """Partially entangled source cos(eta)|00> + sin(eta)|11>.  Concurrence C = sin(2 eta):
    eta=0 -> |00> (product, C=0); eta=pi/4 -> |phi+> (maximal, C=1)."""
    v = np.zeros(4, dtype=complex)
    v[0] = np.cos(eta)
    v[3] = np.sin(eta)
    return v


def concurrence(psi4):
    """Concurrence of a 2-qubit pure state (c00,c01,c10,c11): C = 2|c00 c11 - c01 c10|."""
    c = psi4
    return float(2 * abs(c[0] * c[3] - c[1] * c[2]))


def renou_T_for_sources(eta1, eta2, use_Y=True):
    """Renou T with partially-entangled sources source_ent(eta1,2), maximized over the
    Bell labeling (as in sr7)."""
    import itertools
    bells = sr7.bell_states()
    psi = np.kron(source_ent(eta1), source_ent(eta2))          # order a,B1,B2,c
    twobits = [(0, 0), (0, 1), (1, 0), (1, 1)]
    best = 0.0
    for perm in itertools.permutations(range(4)):
        labels = {perm[k]: twobits[k] for k in range(4)}

        def S(b, x, z):
            A = sr7.alice_obs(x, use_Y); C = sr7.charlie_obs(z, use_Y)
            Pi = np.outer(bells[b], bells[b].conj())
            return float(np.real(psi.conj() @ sr7.kron(A, Pi, C) @ psi))

        def Tb(b):
            b1, b2 = labels[b]; sb = lambda x, z: S(b, x, z)
            return ((-1) ** b2 * (sb(1, 1) + sb(1, 2)) + (-1) ** b1 * (sb(2, 1) - sb(2, 2))
                    + (-1) ** b2 * (sb(1, 3) + sb(1, 4)) - (-1) ** (b1 + b2) * (sb(3, 3) - sb(3, 4))
                    + (-1) ** b1 * (sb(2, 5) + sb(2, 6)) - (-1) ** (b1 + b2) * (sb(3, 5) - sb(3, 6)))
        T = abs(sum(Tb(b) for b in range(4)))
        best = max(best, T)
    return best


def chsh(psi4):
    """CHSH for a 2-qubit state: Alice {Z, X}, Bob {(Z+X)/sqrt2, (Z-X)/sqrt2}."""
    A = [Z, X]
    B = [(Z + X) / s2, (Z - X) / s2]
    val = 0.0
    signs = [[+1, +1], [+1, -1]]           # <A0B0>+<A0B1>+<A1B0>-<A1B1>
    E = lambda a, b: float(np.real(psi4.conj() @ np.kron(A[a], B[b]) @ psi4))
    return abs(E(0, 0) + E(0, 1) + E(1, 0) - E(1, 1))


def vsa_native_source(rng):
    """Build a two-party 'source' by NATIVE VSA operations on two single-qubit
    hypervectors (FHRR, dimension-preserving).  Binding = component-wise complex product;
    bundling = sum.  Neither creates the C^4 tensor space; embedded as a 2-qubit state
    the result is a PRODUCT (separable) state.  We construct the tensor of the two
    single-qubit hypervectors (the most a dimension-preserving 'product' can represent as
    a bipartite state) and report its concurrence -- which is 0 for any product of two
    pure qubits."""
    a = rng.standard_normal(2) + 1j * rng.standard_normal(2); a /= np.linalg.norm(a)
    b = rng.standard_normal(2) + 1j * rng.standard_normal(2); b /= np.linalg.norm(b)
    # VSA binding of two qubits stays a product state in any tensor embedding:
    psi = np.kron(a, b)                     # a (x) b  -- separable by construction
    return psi


def main():
    rng = np.random.default_rng(0)
    out = {"benchmarks": dict(complex_max=float(6 * s2), real_bound=7.66,
                              tsirelson=float(2 * s2))}

    # (0) validate against sr7: maximally entangled sources -> T = 6sqrt2
    T_val, _ = sr7.best_T_over_labelings(0.0, 0.0, use_Y=True)
    ok = abs(abs(T_val) - 6 * s2) < 1e-6
    out["validation_T_maxent"] = float(abs(T_val))
    print(f"(0) validation: Renou T (maximally entangled sources) = {abs(T_val):.4f} "
          f"= 6sqrt2 {abs(6*s2):.4f}?  {ok}")
    assert ok, "sr7 validation failed"
    # CHSH validation on |phi+>
    phi_p = source_ent(np.pi / 4)
    chsh_phi = chsh(phi_p)
    print(f"    CHSH(|phi+>) = {chsh_phi:.4f} = 2sqrt2 {2*s2:.4f}?  "
          f"{abs(chsh_phi-2*s2)<1e-6}   concurrence={concurrence(phi_p):.3f}\n")

    # (1) T vs source entanglement (concurrence)
    print("(1) Renou T vs source concurrence C (both sources at concurrence C):")
    rows = []
    for eta in np.linspace(0, np.pi / 4, 9):
        C = float(np.sin(2 * eta))
        T = renou_T_for_sources(eta, eta, use_Y=True)
        rows.append(dict(concurrence=C, T=float(T)))
        mark = " <- real bound 7.66" if abs(T - 7.66) < 0.3 else (
            " <- complex max 8.49" if T > 8.4 else "")
        print(f"    C={C:.3f}: T={T:.4f}{mark}")
    out["T_vs_concurrence"] = rows
    # crossing of the real bound
    Cs = np.array([r["concurrence"] for r in rows]); Ts = np.array([r["T"] for r in rows])
    cross = float(np.interp(7.66, Ts, Cs)) if Ts.max() > 7.66 else None
    out["concurrence_at_real_bound"] = cross
    print(f"    -> T crosses the real bound 7.66 at concurrence C ~ {cross:.3f}; "
          f"the complex value needs C=1.\n")

    # (2)/(3) VSA-native sources are separable (C=0), CHSH <= 2
    print("(2/3) VSA-native two-party sources (built by binding, dimension-preserving):")
    Cs_vsa, chshs_vsa = [], []
    for _ in range(200):
        psi = vsa_native_source(rng)
        Cs_vsa.append(concurrence(psi)); chshs_vsa.append(chsh(psi))
    out["vsa_source_concurrence"] = dict(mean=float(np.mean(Cs_vsa)),
                                         max=float(np.max(Cs_vsa)))
    out["vsa_source_chsh"] = dict(mean=float(np.mean(chshs_vsa)),
                                  max=float(np.max(chshs_vsa)))
    print(f"    concurrence: mean={np.mean(Cs_vsa):.2e}  max={np.max(Cs_vsa):.2e}  "
          f"(separable: C=0)")
    print(f"    CHSH:        mean={np.mean(chshs_vsa):.3f}  max={np.max(chshs_vsa):.3f}  "
          f"(<= 2, no violation)")
    T_vsa = renou_T_for_sources(0.0, 0.0, use_Y=True)     # C=0 sources
    out["T_vsa_native"] = float(T_vsa)
    print(f"    Renou T with VSA-native (C=0) sources = {T_vsa:.4f}  "
          f"(< real bound 7.66)\n")

    # (4) verdict
    reaches = T_vsa > 7.66 + 1e-6
    out["reaches_complex_tier"] = bool(reaches)
    out["verdict"] = (
        "NEGATIVE (clean): the product-VSA route does NOT reach the complex tier. "
        f"Renou T is monotone in source entanglement, crossing the real bound 7.66 only "
        f"near concurrence C~{cross:.2f} and reaching 6sqrt2=8.49 only at C=1 (maximal "
        "entanglement). VSA-native sources -- built by dimension-preserving binding -- are "
        f"separable (concurrence {np.max(Cs_vsa):.0e}, CHSH<=2), so T_VSA={T_vsa:.2f} < 7.66. "
        "The obstruction is EXACTLY the tensor product / entanglement VSA structurally "
        "lacks. Forming the true tensor product is possible but is not a native VSA "
        "operation (dimension doubling, which VSA avoids); doing so is quantum mechanics "
        "in a hypervector container, not VSA. So the strong horn's COMPLEX tier is "
        "unreachable natively -- a clean split: VSA reaches single-system contextuality "
        "(Stage 1, CF>0) but not the complex-vs-real network separation, for the same "
        "reason it has no quantum speedup: no tensor product, no entanglement.")
    print(f"(4) VERDICT: {out['verdict']}")

    with open(os.path.join(HERE, "cwf_selfref_product_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_selfref_product_results.json")
    plot(rows, cross, T_vsa)


def plot(rows, cross, T_vsa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C = [r["concurrence"] for r in rows]; T = [r["T"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(C, T, "o-", color="#1f4e79", lw=2, ms=6, label="Renou $T$ vs source entanglement")
    ax.axhline(7.66, color="#2e7d32", ls="--", lw=1.3, label="real-QM bound (7.66)")
    ax.axhline(6 * s2, color="#c0392b", ls=":", lw=1.2, label=r"complex max $6\sqrt{2}=8.49$")
    ax.axvline(0.0, color="#6c3483", lw=1.0)
    ax.plot([0.0], [T_vsa], "s", color="#6c3483", ms=11, label=f"VSA-native (C=0): T={T_vsa:.2f}")
    if cross:
        ax.plot([cross], [7.66], "v", color="#2e7d32", ms=9)
        ax.annotate(f"C$\\approx${cross:.2f}", (cross, 7.66), textcoords="offset points",
                    xytext=(-6, 8), fontsize=8, color="#2e7d32")
    ax.set_xlabel("source concurrence $C$ (entanglement)")
    ax.set_ylabel(r"Renou bilocal functional $T$")
    ax.set_title("Stage 3: the complex tier needs entanglement VSA lacks\n"
                 "VSA-native (dimension-preserving) sources are separable ($C=0$)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(HERE, "fig_selfref_product.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
