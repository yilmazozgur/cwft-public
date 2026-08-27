"""
cwf_ap_phaseC.py -- ACTION_PRINCIPLE_PLAN Phase C: the holonomy/contextuality split
in the influence-action language. Does the contextual fraction live in the PHASE
sector A_c (the self-referential holonomy) and NOT in the COST sector C_c (the
self-description / Euclidean cost)?

Context (see cwf_project/ACTION_PRINCIPLE_PLAN.md). The proposed CWF history weight is
the complex influence action

    W_c[gamma] = exp( (i/hbar_c) A_c[gamma]  -  (1/2 hbar_c) C_c[gamma] ),

so that |W_c|^2 = exp(-C_c/hbar_c). The reviewer (Reviewer_Suggests_Actional.txt) corrected
the v1 framing: genuine (contextual, CF>0) interference is NOT "a real action vs a complex
action"; it is the presence of a NONTRIVIAL self-referential phase HOLONOMY A_hol. The cost
C_c is a positive Euclidean/MDL weight that compresses but cannot, by itself, create
contextuality. Phase A then found that the description-cost->Euclidean-action half is already
published (Imafuku, arXiv:2512.08507, 2025-12-09); what is CWF-specific is exactly that the
phase i is self-referential (thm:selfref-i) and that it carries CONTEXTUALITY. So Phase C must
show, on one family, that CF is a function of A_hol and NOT of C_c.

Substrate family (the multi-statement Liar, reused from cwf_sr2): a cycle of n cells with
edge constraints "x_i = x_{i+1}" (equality) or "x_i = NOT x_{i+1}" (negation). PARITY of the
negation count decides everything:
  - ODD negations  -> frustrated, NO definite ground (self-referential, like the Liar self-loop)
  - EVEN negations -> a definite ground exists.

For each scenario we compute three independent quantities:
  CF    -- contextual fraction, via the VALIDATED Abramsky-Barbosa-Mansfield LP imported from
           cwf_phase_contextuality (checked PR=1 / Tsirelson=0.414 / local=0).
  C_c   -- description cost = sum over edges of the Shannon entropy of that edge's table (bits).
           This is additive over local segments (Imafuku-style) and, for the uniform tables
           here, equals n bits -> MONOTONE in cycle length n.
  A_hol -- the loop holonomy: the Z2 product of edge operators (NOT for odd, I for even), and
           its continuous-reversible Theorem-B1 lift (the principal sqrt of the holonomy
           operator; eigenvalue i / matrix-entry imaginary part ~0.5 iff the holonomy is NOT).

THE DECOUPLING TEST. Sweep all-negation cycles n=3..10: C_c = n rises monotonically, but the
parity (hence A_hol, hence CF) OSCILLATES. If CF oscillates while C_c climbs monotonically,
then CF is NOT a function of C_c -- it is a function of A_hol. That is the operational form of
the reviewer's correction and the CWF-specific residual from Phase A.

CPU; numpy + scipy. No quantum amplitudes injected: inputs are classical relational
constraints; CF measures the absence of a classical global section; A_hol is computed from the
constraint operators; C_c from the table entropies. Results -> ap_phaseC_results.json (SEPARATE
from the main results.json, per the plan's isolation discipline).
"""
import json, os
import numpy as np
from scipy.linalg import sqrtm

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))
NOT = np.array([[0.0, 1.0], [1.0, 0.0]])
I2 = np.eye(2)


def empirical_model(n, neg):
    """Contexts = the n cyclic edges; each edge table is uniform over the pairs its
    constraint allows (negation -> {(0,1),(1,0)}; equality -> {(0,0),(1,1)}). Mirrors
    cwf_sr2.empirical_model. No-signaling: every single-cell marginal is (1/2,1/2)."""
    observables = [(i,) for i in range(n)]
    contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci in range(n):
        t = np.zeros((2, 2))
        if neg[ci]:
            t[0, 1] = t[1, 0] = 0.5
        else:
            t[0, 0] = t[1, 1] = 0.5
        tables[ci] = t
    return contexts, observables, tables


def definite_grounds(n, neg):
    """Brute-force all 2^n cyclic assignments consistent with the edge signs."""
    grounds = []
    for code in range(1 << n):
        x = [(code >> i) & 1 for i in range(n)]
        if all((x[i] ^ x[(i + 1) % n]) == (1 if neg[i] else 0) for i in range(n)):
            grounds.append(x)
    return grounds


def description_cost(tables):
    """C_c = sum over edges of the Shannon entropy (bits) of the edge table. Additive
    over local segments (Imafuku-style). For the uniform tables here -> 1 bit/edge."""
    C = 0.0
    for t in tables.values():
        p = t.flatten()
        p = p[p > 0]
        C += float(-(p * np.log2(p)).sum())
    return C


def holonomy(neg):
    """Z2 loop holonomy: product of edge operators (NOT for a negation, I for equality).
    Equals NOT iff an ODD number of negations (frustrated/self-referential), else I.
    Continuous-reversible lift (Theorem B1): the principal sqrt of the holonomy operator
    is complex (eigenvalue i; matrix-entry |Im| ~ 0.5 = the sqrt(NOT) gate) iff holonomy=NOT."""
    M = I2.copy()
    for is_neg in neg:
        M = (NOT if is_neg else I2) @ M
    is_not = np.linalg.norm(M - NOT) < 1e-9
    R = sqrtm(M.astype(complex))                     # the continuous-reversible realization
    sqrt_max_abs_im = float(np.max(np.abs(R.imag)))  # ~0.5 for sqrt(NOT), 0 for sqrt(I)
    eigvals = np.linalg.eigvals(R)
    eig_max_abs_im = float(np.max(np.abs(eigvals.imag)))  # ~1 (the eigenvalue i) for NOT
    return dict(is_NOT=bool(is_not), sqrt_max_abs_imag=sqrt_max_abs_im,
                sqrt_eig_max_abs_imag=eig_max_abs_im)


def analyze(name, n, neg):
    contexts, observables, tables = empirical_model(n, neg)
    lp = contextuality_lp(contexts, observables, tables)
    cf = float(lp.get("contextual_fraction", float("nan")))
    Cc = description_cost(tables)
    hol = holonomy(neg)
    grounds = definite_grounds(n, neg)
    return dict(name=name, n=n, n_negations=int(sum(neg)),
                odd_negations=bool(sum(neg) % 2 == 1),
                num_definite_grounds=len(grounds),
                has_definite_ground=len(grounds) > 0,
                contextual_fraction=cf,
                C_c_bits=Cc,
                A_hol_is_NOT=hol["is_NOT"],
                A_hol_sqrt_maxImag=hol["sqrt_max_abs_imag"],
                A_hol_sqrtEig_maxImag=hol["sqrt_eig_max_abs_imag"],
                lp_success=lp.get("success", False))


def spearman(x, y):
    """Spearman rho via rank-Pearson (small n; ties broken by argsort-of-argsort)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / denom) if denom > 0 else 0.0


def main():
    print("cwf_ap_phaseC -- does contextual fraction track the holonomy A_hol, not the "
          "cost C_c?\n")
    val = _validate_lp()
    print()

    # The DECOUPLING sweep: all-negation cycles n=3..10. C_c = n (monotone up); parity
    # (hence A_hol, hence CF) oscillates.
    sweep = [(f"{n}-cycle all-neg ({'odd' if n % 2 else 'even'})", n, [True] * n)
             for n in range(3, 11)]
    # Mixed controls: parity decoupled from cycle length (from sr2).
    controls = [
        ("4-cycle 3neg+1eq (odd)",  4, [True, True, True, False]),
        ("3-cycle 2neg+1eq (even)", 3, [True, True, False]),
        ("5-cycle 4neg+1eq (even)", 5, [True, True, True, True, False]),
        ("6-cycle 5neg+1eq (odd)",  6, [True, True, True, True, True, False]),
    ]
    rows = [analyze(*s) for s in sweep + controls]

    hdr = (f"{'scenario':<26}{'n':>3}{'#neg':>5}{'odd?':>6}{'gnd?':>6}"
           f"{'C_c(bit)':>9}{'CF':>8}{'A_hol':>7}{'sqrt|Im|':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<26}{r['n']:>3}{r['n_negations']:>5}"
              f"{str(r['odd_negations']):>6}{str(r['has_definite_ground']):>6}"
              f"{r['C_c_bits']:>9.2f}{r['contextual_fraction']:>8.4f}"
              f"{('NOT' if r['A_hol_is_NOT'] else 'I'):>7}{r['A_hol_sqrt_maxImag']:>9.3f}")

    tol = 1e-6
    # CHECK 1: A_hol nontrivial  <=>  CF > 0  <=>  no definite ground.
    check1 = all((r["A_hol_is_NOT"] == (r["contextual_fraction"] > tol)
                  == (not r["has_definite_ground"])) for r in rows)

    # CHECK 2 (the decoupling): on the all-neg sweep, C_c is monotone in n while CF
    # oscillates with parity. Quantify: corr(C_c, CF) ~ 0 (or negative), but CF is
    # PERFECTLY predicted by A_hol.
    sweep_rows = rows[:len(sweep)]
    Cc_sweep = [r["C_c_bits"] for r in sweep_rows]
    CF_sweep = [r["contextual_fraction"] for r in sweep_rows]
    Ahol_sweep = [1.0 if r["A_hol_is_NOT"] else 0.0 for r in sweep_rows]
    CFpos_sweep = [1.0 if c > tol else 0.0 for c in CF_sweep]
    rho_Cc_CF = spearman(Cc_sweep, CF_sweep)
    Cc_monotone = all(Cc_sweep[i] < Cc_sweep[i + 1] for i in range(len(Cc_sweep) - 1))
    CF_oscillates = any((CF_sweep[i] > tol) != (CF_sweep[i + 1] > tol)
                        for i in range(len(CF_sweep) - 1))
    Ahol_predicts_CF = all(a == p for a, p in zip(Ahol_sweep, CFpos_sweep))

    # CHECK 3 (i-forcing): the continuous-reversible holonomy lift is complex EXACTLY for
    # the nontrivial (CF>0) cases (Theorem B1: sqrt(NOT) eigenvalue i, entry |Im|~0.5).
    check3 = all(
        ((r["A_hol_sqrt_maxImag"] > 0.4) == r["A_hol_is_NOT"]
         and (r["A_hol_sqrtEig_maxImag"] > 0.9) == r["A_hol_is_NOT"])
        for r in rows)

    # CHECK 4 (the corners / the modulus identity): |exp(iA - C/2)|^2 = exp(-C), confirmed
    # numerically across the scenarios' costs and a phase sweep -> the phase drops out of the
    # probability, so the COST sector alone (A=0) is phase-free and (Check 1) gives CF=0; the
    # PHASE sector (A_hol=NOT) is what carries CF>0.
    rng_A = np.linspace(0, 2 * np.pi, 7)
    mod_ok = True
    for r in rows:
        C = r["C_c_bits"]
        for A in rng_A:
            W = np.exp(1j * A - C / 2.0)
            if abs(abs(W) ** 2 - np.exp(-C)) > 1e-9:
                mod_ok = False
    # the two corners, stated from the sweep:
    pure_cost_cases = [r for r in rows if not r["A_hol_is_NOT"]]    # A_hol=0 (any C_c)
    pure_phase_cases = [r for r in rows if r["A_hol_is_NOT"]]       # A_hol!=0
    corner_cost_max_CF = max((r["contextual_fraction"] for r in pure_cost_cases), default=0.0)
    corner_phase_min_CF = min((r["contextual_fraction"] for r in pure_phase_cases), default=0.0)

    all_pass = check1 and Cc_monotone and CF_oscillates and Ahol_predicts_CF and check3 and mod_ok

    print(f"\n  CHECK 1  A_hol nontrivial <=> CF>0 <=> no definite ground : {check1}")
    print(f"  CHECK 2  decoupling: C_c monotone-up={Cc_monotone}, CF oscillates="
          f"{CF_oscillates}, A_hol predicts CF perfectly={Ahol_predicts_CF}, "
          f"corr(C_c,CF)={rho_Cc_CF:+.2f}")
    print(f"  CHECK 3  i-forcing: continuous holonomy lift complex iff CF>0 (Thm B1): {check3}")
    print(f"  CHECK 4  |exp(iA-C/2)|^2 = exp(-C) (phase drops out of prob): {mod_ok}")
    print(f"           pure-COST corner (A_hol=0, any C_c): max CF = {corner_cost_max_CF:.4f}")
    print(f"           pure-PHASE corner (A_hol=NOT)      : min CF = {corner_phase_min_CF:.4f}")

    verdict = (
        "PHASE C PASS -- the contextual fraction tracks the self-referential HOLONOMY A_hol, "
        "NOT the description cost C_c. On the all-negation sweep n=3..10 the cost C_c=n rises "
        f"monotonically (corr(C_c,CF)={rho_Cc_CF:+.2f}) while CF OSCILLATES with parity, and "
        "A_hol predicts CF perfectly: nontrivial holonomy (NOT, odd, no definite ground) <=> "
        "CF>0; trivial (I, even, definite ground) <=> CF=0 -- including the LARGE-C_c even "
        "cycles, which carry the most description cost yet stay CF=0 (the pure-cost / Euclidean "
        "corner). The holonomy that drives CF>0 is the Theorem-B1 self-referential i (the "
        "continuous-reversible sqrt of NOT is complex; eigenvalue i). |exp(iA-C/2)|^2=exp(-C) "
        "confirms the phase drops out of the single-history probability, so the cost sector is "
        "phase-free and cannot create contextuality. This is the operational form of the "
        "reviewer's correction (genuine interference = nontrivial phase holonomy, not 'complex "
        "vs real action') and isolates the CWF-specific residual from Phase A: our phase is "
        "self-referential AND carries contextuality, distinct from the cost-based Euclidean "
        "(Imafuku/OS-positivity) route, which addresses neither which substrates are contextual "
        "nor the self-reference origin of i."
    ) if all_pass else (
        "PHASE C INCOMPLETE/UNEXPECTED -- one or more checks failed; inspect the table. The "
        "claim is CF=f(A_hol), independent of C_c; a failure here would mean the holonomy/cost "
        "decoupling does not hold on this family."
    )
    print(f"\n  ALL CHECKS PASS: {all_pass}")
    print(f"\nVERDICT: {verdict}")

    out = os.path.join(HERE, "ap_phaseC_results.json")
    R = dict(
        lp_validation=val,
        scenarios=rows,
        checks=dict(
            check1_holonomy_iff_CF=bool(check1),
            check2_decoupling=dict(C_c_monotone_up=bool(Cc_monotone),
                                   CF_oscillates=bool(CF_oscillates),
                                   A_hol_predicts_CF=bool(Ahol_predicts_CF),
                                   corr_Cc_CF=rho_Cc_CF),
            check3_i_forcing_thmB1=bool(check3),
            check4_modulus_identity=bool(mod_ok),
            pure_cost_corner_maxCF=corner_cost_max_CF,
            pure_phase_corner_minCF=corner_phase_min_CF,
        ),
        all_pass=bool(all_pass),
        verdict=verdict,
        note=("Influence-action history weight W_c=exp(iA_c/hbar_c - C_c/2hbar_c); CF via the "
              "validated ABM LP; C_c = sum of per-edge table entropies (bits, additive); A_hol "
              "= Z2 loop-operator product + its continuous Theorem-B1 sqrt (complex iff NOT). "
              "Decoupling: all-neg cycle C_c=n monotone vs CF oscillating with parity => "
              "CF=f(A_hol) not f(C_c). Reuses cwf_phase_contextuality.contextuality_lp "
              "(validated). No quantum amplitudes injected; inputs are classical constraints."))
    json.dump(R, open(out, "w"), indent=2)
    plot(rows, len(sweep))
    print(f"\nWrote {out}")


def plot(rows, n_sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sweep = rows[:n_sweep]
    ns = [r["n"] for r in sweep]
    Cc = [r["C_c_bits"] for r in sweep]
    CF = [r["contextual_fraction"] for r in sweep]
    odd = [r["odd_negations"] for r in sweep]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ns, Cc, "o-", color="C7", label="description cost $C_c$ (bits) -- monotone")
    cols = ["C3" if o else "C0" for o in odd]
    ax.bar(ns, CF, width=0.5, color=cols, alpha=0.85,
           label="contextual fraction CF -- oscillates with parity")
    for r in sweep:
        tag = "no gnd\nA_hol=NOT" if r["odd_negations"] else "gnd\nA_hol=I"
        ax.text(r["n"], r["contextual_fraction"] + 0.02, tag, ha="center",
                va="bottom", fontsize=6.5)
    ax.set_xlabel("cycle length $n$  (all-negation)")
    ax.set_ylabel("$C_c$ (bits)  /  CF")
    ax.set_title("Phase C: CF tracks the holonomy $A_{hol}$ (parity), NOT the cost $C_c$\n"
                 "$C_c=n$ climbs monotonically while CF oscillates -- contextuality lives in "
                 "the phase sector")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_ap_phaseC.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
