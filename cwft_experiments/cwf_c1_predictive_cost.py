"""
C1 -- predictive-cost scaling on elementary cellular automata (Program II).

The finite, measurable SHADOW of the irreducible-ignorance thesis. We do NOT
(and cannot) demonstrate undecidability here: predicting any observable a finite
t steps ahead is decidable by brute-force simulation. What we measure is
computational IRREDUCIBILITY -- whether any predictor cheaper than simulation
can forecast a coarse observable, and how that ability scales with the forecast
horizon t. Irreducibility is NECESSARY for the asymptotic undecidability story
(Conjecture: Irreducible-Ignorance Theorem) but is a separate, weaker, finite
claim; the asymptotic-to-finite bridge is argued elsewhere, not shown here.

Setup. Target observable: the centre cell value t steps ahead, A_t(x_0) =
(S^t x_0)_c, predicted from the initial light-cone window [c-t, c+t] (the exact
causal dependency set). For each rule we sweep the horizon t and a ladder of
predictors of growing cost:
  (0) majority-class baseline           -- the trivial "guess the marginal" cost.
  (1) GF(2)-affine fit (exact solve)    -- the ADDITIVE shortcut. Exact iff the
      horizon map is GF(2)-affine (additive/shift/identity rules); this is the
      precise sense in which Rule 90 is "linearly closable" (cf. B1/B2).
  (2) real logistic regression          -- the linear-over-reals shortcut.
  (3) small MLP (tanh, H hidden)        -- a bounded nonlinear shortcut.
  (4) full simulation                   -- exact, cost proportional to t (the
      "no shortcut" reference; not learned).

The falsifiable split. For reducible rules (90 additive, 170 shift) a fixed-cost
shortcut stays exact at ALL horizons. For the universal rule 110 and the chaotic
rule 30, every bounded predictor's skill decays toward the baseline as t grows --
the resources to forecast grow with the horizon, i.e. you cannot escape
simulating. Rule 184 (class II) is the structured middle case.

Honesty notes baked into the report:
  - "no shortcut found" over the tested predictor families is EVIDENCE, not proof.
  - finite-t prediction is always decidable; this is irreducibility, not undecidability.

Reuses ca_table / ca_step semantics from cwf_b1_tradeoff. CPU-only NumPy.
Run:  python cwf_c1_predictive_cost.py
"""
import json, os, time
import numpy as np
from cwf_b1_tradeoff import ca_table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================================
# Substrates: vectorised elementary-CA evolution over many initial conditions
# =========================================================================

def ca_evolve_batch(X0, table, t):
    """Evolve a batch of initial rows (m, n) for t steps under an elementary CA
    (periodic BC). Returns the configuration at step t."""
    s = X0.astype(np.int8)
    for _ in range(t):
        l = np.roll(s, 1, axis=1)
        r = np.roll(s, -1, axis=1)
        s = table[(4 * l + 2 * s + r).astype(np.int64)]
    return s


def gen_dataset(rule, t, n_samples, seed):
    """Predict the centre cell t steps ahead from its initial light-cone window.

    Lattice width n = 4t + 21 with periodic BC guarantees the opposite boundary
    cannot influence the centre's radius-t cone within t steps, so the periodic
    simulation is exact for the target. Input = initial cells [c-t, c+t]."""
    n = 4 * t + 21
    c = n // 2
    table = ca_table(rule)
    rng = np.random.default_rng(seed)
    X0 = rng.integers(0, 2, (n_samples, n)).astype(np.int8)
    st = ca_evolve_batch(X0, table, t)
    y = st[:, c].astype(np.int8)
    Xwin = X0[:, c - t:c + t + 1].astype(np.float64)   # width 2t+1
    return Xwin, y


# =========================================================================
# Predictor ladder
# =========================================================================

def gf2_affine_fit(Xtr, ytr, Xte, yte):
    """Exact GF(2)-affine predictor via Gaussian elimination. If an affine map
    over GF(2) reproduces the horizon function, this finds it (test acc = 1).
    For non-affine functions it returns the solution of a maximal consistent
    subsystem -- its test accuracy is then low, certifying 'no affine shortcut'."""
    A = np.concatenate([Xtr.astype(np.uint8) & 1, np.ones((len(Xtr), 1), np.uint8)], axis=1)
    b = ytr.astype(np.uint8) & 1
    M = np.concatenate([A, b[:, None]], axis=1)
    m, k1 = M.shape
    k = k1 - 1
    piv, row = [], 0
    for col in range(k):
        sel = -1
        nz = np.nonzero(M[row:, col])[0]
        if nz.size == 0:
            continue
        sel = row + nz[0]
        M[[row, sel]] = M[[sel, row]]
        mask = M[:, col].astype(bool).copy(); mask[row] = False
        M[mask] ^= M[row]
        piv.append(col); row += 1
        if row == m:
            break
    x = np.zeros(k, np.uint8)
    for i, col in enumerate(piv):
        x[col] = M[i, k]
    Ate = np.concatenate([Xte.astype(np.uint8) & 1, np.ones((len(Xte), 1), np.uint8)], axis=1)
    yhat = (Ate @ x.astype(np.int64)) & 1
    return float((yhat == (yte & 1)).mean())


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_mlp(Xtr, ytr, Xte, yte, H, seed, epochs=350, lr=5e-2, l2=1e-4):
    """Binary classifier by full-batch Adam. H=0 reduces to logistic regression.
    Returns test accuracy. Capacity (cost) is set by H."""
    rng = np.random.default_rng(seed)
    Xtr, Xte = _standardize(Xtr, Xte)
    d = Xtr.shape[1]
    ytr_ = ytr.astype(np.float64); yte_ = yte.astype(np.float64)

    def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    if H == 0:
        w = np.zeros(d); b = 0.0
        params = [w, b]; mom = [np.zeros_like(p) for p in params]
        vel = [np.zeros_like(p) for p in params]
        for ep in range(epochs):
            z = Xtr @ w + b; p = sig(z); g = (p - ytr_) / len(ytr_)
            gw = Xtr.T @ g + l2 * w; gb = g.sum()
            for i, (gr) in enumerate([gw, gb]):
                mom[i] = 0.9 * mom[i] + 0.1 * gr
                vel[i] = 0.999 * vel[i] + 0.001 * gr ** 2
                step = lr * mom[i] / (np.sqrt(vel[i]) + 1e-8)
                if i == 0: w = w - step
                else: b = b - step
        acc = (((Xte @ w + b) > 0).astype(int) == yte).mean()
        return float(acc)

    # one hidden layer (tanh)
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, H)); b1 = np.zeros(H)
    W2 = rng.normal(0, 1 / np.sqrt(H), H); b2 = 0.0
    params = [W1, b1, W2, b2]
    mom = [np.zeros_like(p) for p in params]
    vel = [np.zeros_like(p) for p in params]
    for ep in range(epochs):
        h = np.tanh(Xtr @ W1 + b1)
        z = h @ W2 + b2; p = sig(z)
        g = (p - ytr_) / len(ytr_)
        gW2 = h.T @ g + l2 * W2; gb2 = g.sum()
        gh = np.outer(g, W2) * (1 - h ** 2)
        gW1 = Xtr.T @ gh + l2 * W1; gb1 = gh.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        for i, gr in enumerate(grads):
            mom[i] = 0.9 * mom[i] + 0.1 * gr
            vel[i] = 0.999 * vel[i] + 0.001 * gr ** 2
            params[i] = params[i] - lr * mom[i] / (np.sqrt(vel[i]) + 1e-8)
        W1, b1, W2, b2 = params
    h = np.tanh(Xte @ W1 + b1)
    acc = (((h @ W2 + b2) > 0).astype(int) == yte).mean()
    return float(acc)


# =========================================================================
# Sweep
# =========================================================================

def main():
    RULES = {                      # rule : (Wolfram class label, expected)
        110: "IV (universal)",
        30:  "III (chaotic)",
        90:  "III (additive/XOR)",
        184: "II (traffic)",
        170: "shift (trivial)",
    }
    horizons = [1, 2, 3, 4, 6, 8, 11, 14, 18, 23, 28]
    n_samples = 4000
    H_mlp = 32
    eps = 0.75                      # skill threshold for the half-life horizon
    seed0 = 7

    t0 = time.time()
    by_rule = {}
    for rule, label in RULES.items():
        rows = []
        for t in horizons:
            Xw, y = gen_dataset(rule, t, n_samples, seed0 + rule + t)
            ntr = int(0.7 * len(y))
            Xtr, Xte = Xw[:ntr], Xw[ntr:]
            ytr, yte = y[:ntr], y[ntr:]
            base = max(yte.mean(), 1 - yte.mean())           # majority-class baseline
            a_gf2 = gf2_affine_fit(Xtr, ytr, Xte, yte)
            a_log = train_mlp(Xtr, ytr, Xte, yte, H=0, seed=seed0 + t)
            a_mlp = train_mlp(Xtr, ytr, Xte, yte, H=H_mlp, seed=seed0 + t)
            # skill above baseline of the best bounded (sub-simulation) predictor
            best_bounded = max(a_gf2, a_log, a_mlp)
            skill = (best_bounded - base) / (1 - base + 1e-12)
            rows.append(dict(t=t, base=float(base), acc_gf2=a_gf2,
                             acc_logistic=a_log, acc_mlp=a_mlp,
                             best_bounded=float(best_bounded),
                             skill=float(np.clip(skill, 0, 1))))
            print(f"  rule {rule:3d} [{label:16s}] t={t:2d}  base={base:.2f}  "
                  f"GF2={a_gf2:.2f}  log={a_log:.2f}  MLP={a_mlp:.2f}  skill={skill:.2f}")
        # derived metrics
        gf2_min = min(r["acc_gf2"] for r in rows)
        affine_shortcut = bool(gf2_min >= 0.99)              # exact additive shortcut at all t?
        skills = [r["skill"] for r in rows]
        skill_tmax = skills[-1]
        # half-life horizon: first t where skill drops below eps (None if never)
        thalf = next((rows[i]["t"] for i in range(len(rows)) if skills[i] < eps), None)
        by_rule[str(rule)] = dict(
            rule=rule, label=label, rows=rows,
            affine_shortcut=affine_shortcut, gf2_min=float(gf2_min),
            skill_at_tmax=float(skill_tmax), skill_halflife_t=thalf,
            verdict=("reducible (affine shortcut)" if affine_shortcut
                     else ("reducible (bounded shortcut holds)" if skill_tmax >= eps
                           else "irreducible (no sub-simulation shortcut)")))
        print(f"  -> rule {rule}: {by_rule[str(rule)]['verdict']}  "
              f"(skill@t={horizons[-1]}={skill_tmax:.2f}, affine={affine_shortcut})\n")

    meta = dict(horizons=horizons, n_samples=n_samples, H_mlp=H_mlp,
                skill_threshold=eps, target="centre cell at step t from initial light cone",
                note=("C1 measures computational IRREDUCIBILITY (finite, decidable), the "
                      "necessary finite shadow of asymptotic undecidability -- not undecidability "
                      "itself. 'No shortcut' is over the tested predictor families: evidence, not proof."),
                runtime_s=round(time.time() - t0, 1))
    print(f"runtime {meta['runtime_s']}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["C1_predictive_cost"] = dict(meta=meta, by_rule=by_rule)
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(by_rule, meta)
    print("Wrote results.json key: C1_predictive_cost")


def plot_results(by_rule, meta):
    order = ["90", "170", "184", "30", "110"]
    order = [k for k in order if k in by_rule] + [k for k in by_rule if k not in order]
    color = {"110": "#e7298a", "30": "#7570b3", "90": "#1b9e77",
             "184": "#d95f02", "170": "#66a61e"}
    mark = {"110": "o", "30": "s", "90": "^", "184": "D", "170": "v"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) skill of best bounded predictor vs horizon
    a = ax[0]
    for k in order:
        r = by_rule[k]; ts = [x["t"] for x in r["rows"]]
        sk = [x["skill"] for x in r["rows"]]
        a.plot(ts, sk, mark.get(k, "o") + "-", color=color.get(k, "0.4"),
               lw=1.6, ms=5, label=f"rule {r['rule']} — {r['label']}")
    a.axhline(meta["skill_threshold"], ls="--", c="0.5", lw=1)
    a.text(ts[-1], meta["skill_threshold"] + 0.02, f"skill$={meta['skill_threshold']}$",
           ha="right", fontsize=8, color="0.4")
    a.set_xlabel("forecast horizon  $t$ (steps)")
    a.set_ylabel("skill of best sub-simulation predictor")
    a.set_title("(a) Forecast skill collapses — irreducibility")
    a.set_ylim(-0.03, 1.05)
    a.legend(fontsize=7.5, loc="lower left")

    # (b) GF(2)-affine shortcut accuracy vs horizon
    b = ax[1]
    for k in order:
        r = by_rule[k]; ts = [x["t"] for x in r["rows"]]
        ag = [x["acc_gf2"] for x in r["rows"]]
        b.plot(ts, ag, mark.get(k, "o") + "-", color=color.get(k, "0.4"),
               lw=1.6, ms=5, label=f"rule {r['rule']}")
    b.axhline(1.0, ls=":", c="0.6", lw=1)
    b.set_xlabel("forecast horizon  $t$ (steps)")
    b.set_ylabel("GF(2)-affine shortcut accuracy")
    b.set_title("(b) The additive shortcut: exact only if linear")
    b.set_ylim(0.45, 1.03)
    b.legend(fontsize=7.5, loc="lower left")

    # (c) verdict bars: skill at the longest horizon
    c = ax[2]
    ks = order
    vals = [by_rule[k]["skill_at_tmax"] for k in ks]
    cols = [color.get(k, "0.4") for k in ks]
    bars = c.barh(range(len(ks)), vals, color=cols, edgecolor="white")
    c.set_yticks(range(len(ks)))
    c.set_yticklabels([f"rule {by_rule[k]['rule']}" for k in ks], fontsize=9)
    c.invert_yaxis()
    c.axvline(meta["skill_threshold"], ls="--", c="0.4", lw=1)
    c.set_xlim(0, 1.05)
    c.set_xlabel(f"skill at $t={meta['horizons'][-1]}$")
    c.set_title("(c) Reducible (right) vs irreducible (left)")
    for i, k in enumerate(ks):
        tag = "affine" if by_rule[k]["affine_shortcut"] else ""
        if tag:
            c.text(vals[i] + 0.02, i, tag, va="center", fontsize=7.5, color="0.3")

    fig.suptitle("C1: predictive-cost scaling — computational irreducibility "
                 "as the finite shadow of irreducible ignorance",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_C1_predictive_cost.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
