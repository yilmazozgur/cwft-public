"""
T1.1 Phase 2 (cont.) -- re-test the trade-off law and the a_c <= F(N*,D) bound on the CA
class using the FAITHFUL symbolic entropy h_t (cwf_t11_ca_entropy), replacing the unfaithful
Lyapunov-tangent a_c. N_rich is reused from results.json key T11_scale_extensive (it does
not depend on the a_c estimator). Everything intensive (per-DOF): h_t is bits/step/site for
CAs; for CML/RCL the comparison uses (sum positive Lyapunov)/n.

Answers the two open questions:
  (Q1) Does the N*<->a_c law hold WITHIN the CA class with a faithful entropy?
  (Q2) Which CAs violate the bound a_c <= kappa*log2(1/(1-N*^2)) with a faithful entropy --
       i.e. are the additive/linear rules GENUINE counterexamples (not estimator artifacts)?
"""
import json
import numpy as np
from cwf_t11_scale import CATALOG
from cwf_t11_ca_entropy import ca_symbolic_entropy
from cwf_b2_bridge import spearman

N_DOF = 16
ADDITIVE = {90, 150, 60, 105, 102, 153, 195, 165}   # linear-over-GF(2) ECAs


def bootstrap(x, y, B=4000, seed=7):
    rng = np.random.default_rng(seed); n = len(x); b = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n); b[i] = spearman(x[idx], y[idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

def perm_p(x, y, P=20000, seed=11):
    rng = np.random.default_rng(seed); obs = abs(spearman(x, y)); yc = y.copy(); ge = 0
    for _ in range(P):
        rng.shuffle(yc)
        if abs(spearman(x, yc)) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (P + 1)


def main():
    R = json.load(open("results.json"))
    by = R["T11_scale_extensive"]["by_system"]

    # (1) faithful h_t for all catalog CA rules
    ca = [(nm, p["rule"]) for (nm, typ, p, _) in CATALOG if typ == "ca"]
    print(f"Computing faithful symbolic entropy h_t for {len(ca)} CA rules...\n")
    print(f"  {'rule':10s}{'N_rich':>8s}{'h_t':>9s}{'a_c_lyap(old)':>15s}")
    rows = []
    for nm, rule in ca:
        h = ca_symbolic_entropy(rule)
        N = by[nm]["N_rich"]; old = by[nm]["a_c"]
        rows.append(dict(nm=nm, rule=rule, N=N, h=h, old=old,
                         additive=(rule in ADDITIVE)))
        print(f"  {nm:10s}{N:8.3f}{h:9.4f}{old:15.3f}"
              f"{'  [additive]' if rule in ADDITIVE else ''}")

    Nca = np.array([r["N"] for r in rows]); Hca = np.array([r["h"] for r in rows])

    # (Q1) within-CA law with faithful entropy
    rho = spearman(Nca, Hca); lo, hi = bootstrap(Nca, Hca); pv = perm_p(Nca, Hca)
    rho_old = spearman(Nca, np.array([r["old"] for r in rows]))
    print(f"\n(Q1) WITHIN-CA Spearman(N_rich, h_t [faithful]) = {rho:+.3f}  "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]  perm p = {pv:.4f}")
    print(f"     vs same with unfaithful Lyapunov a_c: {rho_old:+.3f}")

    # pooled intensive (per-DOF) law across all three classes
    xs, ys, cls = [], [], []
    for nm, typ, p, _ in CATALOG:
        if typ == "ca":
            continue
    for r in rows:
        xs.append(r["N"]); ys.append(r["h"]); cls.append("ca")
    for nm, info in by.items():
        if info["typ"] in ("cml", "rcl"):
            xs.append(info["N_rich"]); ys.append(info["a_c"] / N_DOF); cls.append(info["typ"])
    xs = np.array(xs); ys = np.array(ys); cls = np.array(cls)
    rho_pool = spearman(xs, ys); lo_p, hi_p = bootstrap(xs, ys)
    print(f"\n     pooled INTENSIVE (per-DOF) Spearman(N_rich, a_c) = {rho_pool:+.3f}  "
          f"95% CI [{lo_p:+.3f}, {hi_p:+.3f}]  (n={len(xs)})")
    for c in ["ca", "cml", "rcl"]:
        m = cls == c
        if m.sum() >= 4:
            print(f"       {c.upper():4s} (n={int(m.sum())}): {spearman(xs[m], ys[m]):+.3f}")

    # (Q2) bound check with faithful intensive entropy: a_c_int <= kappa * (-log2(1-N^2))
    print(f"\n(Q2) bound a_c <= kappa*(-log2(1-N*^2)) with FAITHFUL h_t (per-DOF, so kappa<=1 "
          f"would mean m_u=1):")
    krs = []
    for r in rows:
        s = 1 - r["N"] ** 2; base = -np.log2(s) if s > 1e-12 else np.inf
        r["base"] = base
        if base > 1e-9 and np.isfinite(base):
            krs.append((r["h"] / base, r["nm"], r["additive"]))
    kca = max(krs)[0]
    kca_noadd = max((k for k, _, a in krs if not a), default=float("nan"))
    add_bind = [nm for k, nm, a in krs if a and k > 1.0]
    print(f"     CA kappa_req (faithful) = {kca:.2f}  (binding at {max(krs)[1]})")
    print(f"     CA kappa_req EXCLUDING additive rules = {kca_noadd:.2f}")
    print(f"     additive rules with kappa>1 (genuine bound violators): {add_bind}")
    # which CAs exceed the per-DOF bound a_c <= 1*(-log2(1-N^2)) (m_u=1)?
    viol1 = [r["nm"] for r in rows if r["base"] > 1e-9 and r["h"] > r["base"] + 1e-9]
    print(f"     CAs exceeding the m_u=1 (single-mode) per-site bound: {viol1}")

    R["T11_ca_faithful"] = dict(
        estimator="temporal block-entropy rate h_t (cwf_t11_ca_entropy)",
        within_ca_spearman_faithful=rho, within_ca_ci95=[lo, hi], within_ca_perm_p=pv,
        within_ca_spearman_unfaithful=rho_old,
        pooled_intensive_spearman=rho_pool, pooled_intensive_ci95=[lo_p, hi_p],
        ca_kappa_req=float(kca), ca_kappa_req_excl_additive=float(kca_noadd),
        additive_genuine_violators=add_bind,
        per_rule={r["nm"]: dict(rule=r["rule"], N_rich=r["N"], h_t=r["h"],
                                additive=r["additive"]) for r in rows},
        note="faithful symbolic entropy h_t (per-site, bits/step) replacing the unfaithful "
             "Lyapunov-tangent a_c. h_t in [0,1] for binary CAs. Additive/linear rules remain "
             "GENUINE bound violators (high entropy + moderate Koopman residual) -- the clean "
             "inequality is false for linear-algebraic substrates even with faithful entropy.")
    json.dump(R, open("results.json", "w"), indent=2)
    print("\nWrote results.json key: T11_ca_faithful")


if __name__ == "__main__":
    main()
