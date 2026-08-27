"""
C1b -- predictive-cost SCALING (Program II, strengthening of C1).

C1 showed forecast skill collapses with horizon at FIXED predictor cost. C1b
makes the cost-scaling claim literal: how do the predictive RESOURCES required to
forecast the centre cell to a fixed skill threshold scale with the horizon t?

Two cost axes:
  - capacity: parameter count of the cheapest predictor reaching skill >= eps,
    minimised over BOTH the GF(2)-affine fit and an MLP capacity ladder
    H in {0(logistic),...}. Minimising over predictor TYPES is essential: the
    additive shortcut (exact for rules 90/170) costs only O(window) ~ O(t)
    parameters, so an MLP-only sweep would falsely wall those rules (SGD cannot
    learn a wide XOR even though the closed-form map is trivial).
  - data: training samples needed to reach skill >= eps at fixed large capacity.

Expected (and the falsifiable content):
  - reducible (90 additive, 170 shift): required cost grows only ~linearly with t
    (the affine predictor), never walls;
  - irreducible (110 universal, 30 chaotic): required cost diverges past a short
    horizon -- no feasible capacity OR data reaches eps, the operational
    "must simulate" wall;
  - 184 (class II): intermediate -- walls later.

Same honesty caveat as C1: this is computational IRREDUCIBILITY (finite,
decidable), the necessary finite shadow of asymptotic undecidability, not a
demonstration of undecidability; "no shortcut" is over the tested families.

Reuses gen_dataset / train_mlp / gf2_affine_fit from cwf_c1_predictive_cost.
Run:  python cwf_c1b_cost_scaling.py
"""
import json, os, time
import numpy as np
from cwf_c1_predictive_cost import gen_dataset, train_mlp, gf2_affine_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mlp_params(d, H):
    return (d + 1) if H == 0 else (d * H + 2 * H + 1)


def skill_of(acc, base):
    return float(np.clip((acc - base) / (1 - base + 1e-12), 0, 1))


def main():
    RULES = {110: "IV (universal)", 30: "III (chaotic)", 90: "III (additive/XOR)",
             184: "II (traffic)", 170: "shift (trivial)"}
    horizons = [1, 2, 4, 6, 8, 12, 16]
    H_grid = [0, 4, 16, 64, 128]            # capacity ladder (0 = logistic)
    N_grid = [300, 1000, 3000, 6000, 10000] # sample-complexity ladder
    eps = 0.75                              # skill threshold
    seeds = [1, 2]
    POOL, TEST, NTR_CAP, H_DATA = 12500, 1500, 4000, 64

    CEILING = None  # represented as None -> "wall" (cost beyond tested budget)
    t0 = time.time()
    by_rule = {}
    for rule, label in RULES.items():
        rows = []
        for t in horizons:
            d = 2 * t + 1
            Xw, y = gen_dataset(rule, t, POOL, seed=rule + t)
            Xte, yte = Xw[-TEST:], y[-TEST:]
            Xpool, ypool = Xw[:-TEST], y[:-TEST]
            base = float(max(yte.mean(), 1 - yte.mean()))

            # affine predictor (exact iff GF(2)-affine); cost = d+1 params
            sk_aff = skill_of(gf2_affine_fit(Xpool[:NTR_CAP], ypool[:NTR_CAP], Xte, yte), base)

            # capacity sweep at fixed n_train = NTR_CAP
            cap = {}
            for H in H_grid:
                accs = [train_mlp(Xpool[:NTR_CAP], ypool[:NTR_CAP], Xte, yte, H=H,
                                  seed=s, epochs=220) for s in seeds]
                cap[H] = skill_of(float(np.mean(accs)), base)

            # data sweep at fixed capacity H_DATA
            dat = {}
            for nt in N_grid:
                accs = [train_mlp(Xpool[:nt], ypool[:nt], Xte, yte, H=H_DATA,
                                  seed=s, epochs=220) for s in seeds]
                dat[nt] = skill_of(float(np.mean(accs)), base)

            # required capacity (param count) to reach eps: min over predictor types
            cands = []
            if sk_aff >= eps:
                cands.append(d + 1)                       # affine cost
            for H in H_grid:
                if cap[H] >= eps:
                    cands.append(mlp_params(d, H))
            req_params = int(min(cands)) if cands else CEILING

            # required samples to reach eps at fixed capacity
            req_n = next((nt for nt in N_grid if dat[nt] >= eps), CEILING)

            rows.append(dict(t=t, d=d, base=base, skill_affine=sk_aff,
                             cap={str(H): cap[H] for H in H_grid},
                             dat={str(nt): dat[nt] for nt in N_grid},
                             req_params=req_params, req_samples=req_n))
            print(f"  rule {rule:3d} t={t:2d}  affine_sk={sk_aff:.2f}  "
                  f"cap[max H]={cap[H_grid[-1]]:.2f}  req_params={req_params}  req_n={req_n}")

        walls = [r["t"] for r in rows if r["req_params"] is None]
        wall_t = walls[0] if walls else None
        by_rule[str(rule)] = dict(rule=rule, label=label, rows=rows,
                                  wall_horizon=wall_t,
                                  verdict=("no wall (cost bounded at all tested t)"
                                           if wall_t is None else f"cost wall at t={wall_t}"))
        print(f"  -> rule {rule}: {by_rule[str(rule)]['verdict']}\n")

    meta = dict(horizons=horizons, H_grid=H_grid, N_grid=N_grid, skill_threshold=eps,
                n_train_capacity=NTR_CAP, H_data=H_DATA, seeds=seeds,
                cost="parameter count of cheapest predictor (GF2-affine OR MLP-H) reaching skill>=eps",
                note=("computational IRREDUCIBILITY (finite, decidable), the necessary finite shadow "
                      "of asymptotic undecidability; 'wall' = no tested predictor/budget reaches eps."),
                runtime_s=round(time.time() - t0, 1))
    print(f"runtime {meta['runtime_s']}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["C1_cost_scaling"] = dict(meta=meta, by_rule=by_rule)
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(by_rule, meta)
    print("Wrote results.json key: C1_cost_scaling")


def plot_results(by_rule, meta):
    order = [k for k in ["90", "170", "184", "30", "110"] if k in by_rule]
    color = {"110": "#e7298a", "30": "#7570b3", "90": "#1b9e77",
             "184": "#d95f02", "170": "#66a61e"}
    mark = {"110": "o", "30": "s", "90": "^", "184": "D", "170": "v"}
    ts = meta["horizons"]
    ceil_y = 6e5  # plotting height for "wall"

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    # (a) required capacity (params) to reach eps vs horizon
    a = ax[0]
    for k in order:
        r = by_rule[k]
        xs = [x["t"] for x in r["rows"]]
        ys = [x["req_params"] if x["req_params"] is not None else None for x in r["rows"]]
        xf = [x for x, y in zip(xs, ys) if y is not None]
        yf = [y for y in ys if y is not None]
        a.plot(xf, yf, mark[k] + "-", color=color[k], lw=1.7, ms=6,
               label=f"rule {r['rule']} — {r['label']}")
        # wall markers (open, at ceiling)
        xw = [x for x, y in zip(xs, ys) if y is None]
        if xw:
            a.scatter(xw, [ceil_y] * len(xw), marker="x", color=color[k], s=55, zorder=5)
    a.axhline(ceil_y, ls=":", c="0.6", lw=1)
    a.text(ts[0], ceil_y * 1.15, "wall (no predictor/budget reaches skill $\\epsilon$)",
           fontsize=8, color="0.4")
    a.set_yscale("log")
    a.set_xlabel("forecast horizon  $t$ (steps)")
    a.set_ylabel("required predictor cost (parameters)")
    a.set_title("(a) Required cost vs horizon")
    a.legend(fontsize=7.5, loc="center right")

    # (b) capacity does not rescue irreducibility: skill vs H for rule 110
    b = ax[1]
    r110 = by_rule["110"]
    show_t = [t for t in [2, 4, 6, 8, 12] if t in ts]
    cmap = plt.cm.viridis
    for i, t in enumerate(show_t):
        row = next(x for x in r110["rows"] if x["t"] == t)
        Hs = meta["H_grid"]
        sk = [row["cap"][str(H)] for H in Hs]
        xx = [max(H, 1) for H in Hs]   # H=0 -> 1 for log axis
        b.plot(xx, sk, "o-", color=cmap(i / max(len(show_t) - 1, 1)),
               lw=1.5, ms=5, label=f"$t={t}$")
    b.axhline(meta["skill_threshold"], ls="--", c="0.5", lw=1)
    b.set_xscale("log")
    b.set_xlabel("MLP hidden units $H$ (capacity)")
    b.set_ylabel("forecast skill")
    b.set_title("(b) Rule 110: capacity does not buy horizon")
    b.set_ylim(-0.03, 1.05)
    b.legend(fontsize=8, title="horizon", loc="upper right")

    # (c) sample-complexity: required training samples vs horizon
    c = ax[2]
    ceil_n = meta["N_grid"][-1] * 3
    for k in order:
        r = by_rule[k]
        xs = [x["t"] for x in r["rows"]]
        ys = [x["req_samples"] for x in r["rows"]]
        xf = [x for x, y in zip(xs, ys) if y is not None]
        yf = [y for y in ys if y is not None]
        c.plot(xf, yf, mark[k] + "-", color=color[k], lw=1.7, ms=6, label=f"rule {r['rule']}")
        xw = [x for x, y in zip(xs, ys) if y is None]
        if xw:
            c.scatter(xw, [ceil_n] * len(xw), marker="x", color=color[k], s=55, zorder=5)
    c.axhline(ceil_n, ls=":", c="0.6", lw=1)
    c.text(ts[0], ceil_n * 1.05, "data wall", fontsize=8, color="0.4")
    c.set_yscale("log")
    c.set_xlabel("forecast horizon  $t$ (steps)")
    c.set_ylabel(f"training samples to reach skill $\\epsilon$ (H={meta['H_data']})")
    c.set_title("(c) Sample cost vs horizon")
    c.legend(fontsize=7.5, loc="center right")

    fig.suptitle("C1b: predictive-cost SCALING — required resources diverge with horizon "
                 "for irreducible substrates", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_C1b_cost_scaling.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
