"""
cwf_complex_attention.py -- Route C: the complex self-attention DIAGONAL.
(TRANSFORMER_PHASE_NOTE.md  experiment.)

Route C: in self-attention Q,K,V all come from X, so a token attends to itself
(the diagonal a_ii) -- a literal self-measurement (the C11 trilemma).  We give
that diagonal a self-SOURCED coefficient that modulates how the context is read:
the query token's own head sets a coefficient applied to the context aggregate,
    o = coef(x_query) * c,   c = sum_{j != query} alpha_j v_j ,   logit = W o.
The four variants differ ONLY in the coefficient function (matched params):
    base   : coef = 1                 (standard; cannot suppress or flip)
    gate   : coef = sigmoid(h) in [0,1]   (real; can suppress, NOT flip)
    signed : coef = tanh(h)    in [-1,1]  (real; CAN flip -- the STRONG RIVAL)
    phase  : coef = cos(theta) in [-1,1]  (route C; the self-reference phase)

THE DIFFERENTIAL PREDICTION (TRANSFORMER_PHASE_NOTE.md sec.6).  Phase/negation at
the diagonal should help SPECIFICALLY when the task needs self-referential
negation, and NOT otherwise.  Three matched tasks isolate the two ingredients:
    SELF-NEG : y = agg * (-1)^{b_self}   (negate bit on the QUERY -> at the diagonal)  [criterion MET]
    CTX-NEG  : y = agg * (-1)^{b_ctx}    (negate bit on a CONTEXT token -> off-diagonal) [control: negation, wrong site]
    NO-NEG   : y = agg                   (bit is a distractor)                           [control: no negation]
Prediction: signed & phase >> gate & base on SELF-NEG; ALL equal on CTX-NEG (bit
not at the diagonal, unreachable by the self-coefficient) and on NO-NEG.

THE HONEST CRUX (do NOT rig).  For a single sign flip, tanh and cos both span
[-1,1], so we EXPECT phase ~ signed on SELF-NEG: phase = a sign = the Z_2
holonomy (sr2), NOT special.  Genuine phase (U(1)) should beat a real scalar only
when the task needs a ROTATION, not a sign.  Part 3 tests exactly that:
    U(1)-ROT : answer = quadrant of R(theta_query) * context_vector  (a 2D rotation)
    phase coef = e^{i theta} (rotates); real-signed coef = scalar (can only
    scale/flip -> cannot rotate).  Matched params.  This is where phase-qua-phase
    should pay -- the sr9 / thm:u1-cocycle distinction (U(1) needs the genuine i).

Reports whatever happens, including phase ~ signed on Z_2 (the honest negative for
"phase is special" at the sign level).  Seeded, CPU, torch.  Writes
cwf_complex_attention_results.json.
"""

import json
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)
DEV = "cpu"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_data(task, n_ctx=8, d_in=8, N=4000):
    """Token slots: 0=value, 1=negate-bit, 2=is_query, 3..=noise.
    Position 0 = query; positions 1..n_ctx = context."""
    n = n_ctx + 1
    X = 0.3 * np.random.randn(N, n, d_in).astype(np.float32)
    X[:, :, 2] = 0.0
    X[:, 0, 2] = 1.0                                   # is_query flag
    # context values +-1 (odd count -> no ties)
    vals = np.random.choice([-1.0, 1.0], size=(N, n_ctx)).astype(np.float32)
    X[:, 1:, 0] = vals
    X[:, 0, 0] = 0.0
    agg = np.sign(vals.sum(axis=1)).astype(np.float32)  # +-1 majority
    # negate bit
    b = np.random.randint(0, 2, size=N).astype(np.float32)
    X[:, :, 1] = 0.0
    if task in ("self_neg",):
        X[:, 0, 1] = 2 * b - 1                         # bit on QUERY (diagonal)
    elif task in ("ctx_neg",):
        X[:, 1, 1] = 2 * b - 1                         # bit on a CONTEXT token
    elif task in ("no_neg",):
        X[:, 0, 1] = 2 * b - 1                         # present but irrelevant
    # label
    if task == "self_neg":
        y = agg * (1 - 2 * b)
    elif task == "ctx_neg":
        y = agg * (1 - 2 * b)
    else:  # no_neg
        y = agg
    y = (y > 0).astype(np.int64)                       # {0,1}
    return torch.tensor(X), torch.tensor(y)


def make_rot_data(n_ctx=8, d_in=8, N=4000, Q=4):
    """U(1) task: context tokens carry an angle; answer = quadrant of the context
    vector ROTATED by the query's angle."""
    n = n_ctx + 1
    X = 0.3 * np.random.randn(N, n, d_in).astype(np.float32)
    X[:, :, 2] = 0.0
    X[:, 0, 2] = 1.0
    # context angles -> store (cos,sin) in slots 0,1
    cang = np.random.uniform(-np.pi, np.pi, size=(N, n_ctx)).astype(np.float32)
    X[:, 1:, 0] = np.cos(cang)
    X[:, 1:, 1] = np.sin(cang)
    # context resultant vector (uniform aggregate)
    cx = np.cos(cang).sum(1)
    cy = np.sin(cang).sum(1)
    # query angle in slot 3 (single scalar the head will read)
    qang = np.random.uniform(-np.pi, np.pi, size=N).astype(np.float32)
    X[:, 0, 3] = qang
    # rotate context resultant by query angle
    rx = np.cos(qang) * cx - np.sin(qang) * cy
    ry = np.sin(qang) * cx + np.cos(qang) * cy
    ang = np.arctan2(ry, rx)
    y = (((ang + np.pi) / (2 * np.pi) * Q).astype(np.int64) % Q)
    return torch.tensor(X), torch.tensor(y), Q


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class DiagScalarAttn(nn.Module):
    """Part 1/2: self-sourced SCALAR coefficient modulates the context aggregate."""
    def __init__(self, d_in, d_model, mode, n_cls=2):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_model)
        self.Wk = nn.Linear(d_in, d_model)
        self.Wv = nn.Linear(d_in, d_model)
        self.head = nn.Linear(d_in, 1)                 # self-coefficient head
        self.readout = nn.Linear(d_model, n_cls)
        self.mode = mode
        self.dm = d_model

    def coef(self, h):
        if self.mode == "base":   return torch.ones_like(h)
        if self.mode == "gate":   return torch.sigmoid(h)
        if self.mode == "signed": return torch.tanh(h)
        if self.mode == "phase":  return torch.cos(h)
        raise ValueError(self.mode)

    def forward(self, X):
        q = self.Wq(X[:, :1]); k = self.Wk(X[:, 1:]); v = self.Wv(X[:, 1:])
        s = (q @ k.transpose(1, 2)) / np.sqrt(self.dm)  # (B,1,n_ctx)
        a = torch.softmax(s, dim=-1)
        c = (a @ v).squeeze(1)                          # (B,d_model) context aggregate
        h = self.head(X[:, 0]).squeeze(-1)              # (B,) from query token
        coef = self.coef(h)
        o = coef.unsqueeze(-1) * c
        return self.readout(o), coef


class DiagRotAttn(nn.Module):
    """Part 3: self-sourced coefficient is a ROTATION (phase) or a SCALAR (real)."""
    def __init__(self, d_in, mode, n_cls=4):
        super().__init__()
        self.Wq = nn.Linear(d_in, 8)
        self.Wk = nn.Linear(d_in, 8)
        self.Wv = nn.Linear(d_in, 2)                    # 2D context vector
        self.head = nn.Linear(d_in, 1)                  # one scalar (angle or scale)
        self.readout = nn.Linear(2, n_cls)
        self.mode = mode

    def forward(self, X):
        q = self.Wq(X[:, :1]); k = self.Wk(X[:, 1:]); v = self.Wv(X[:, 1:])
        s = (q @ k.transpose(1, 2)) / np.sqrt(8)
        a = torch.softmax(s, dim=-1)
        c = (a @ v).squeeze(1)                          # (B,2) context resultant
        h = self.head(X[:, 0]).squeeze(-1)              # (B,)
        if self.mode == "phase":                        # genuine rotation
            cs, sn = torch.cos(h), torch.sin(h)
            ox = cs * c[:, 0] - sn * c[:, 1]
            oy = sn * c[:, 0] + cs * c[:, 1]
            o = torch.stack([ox, oy], dim=-1)
        elif self.mode == "real":                       # scalar: can scale/flip, not rotate
            o = torch.tanh(h).unsqueeze(-1) * c
        else:
            raise ValueError(self.mode)
        return self.readout(o), h


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------
def train_eval(model, Xtr, ytr, Xte, yte, epochs=300, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logit, _ = model(Xtr)
        lossf(logit, ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = model(Xte)[0].argmax(1)
        acc = (pred == yte).float().mean().item()
        coef = model(Xte)[1]
    return acc, coef


def n_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    print("=" * 74)
    print("ROUTE C: complex self-attention diagonal -- differential prediction")
    print("=" * 74)

    SEEDS = [1, 2, 3, 4]
    # ---- Part 1: Z_2 negation factorial (seed-averaged) ----
    tasks = ["self_neg", "ctx_neg", "no_neg"]
    modes = ["base", "gate", "signed", "phase"]
    print(f"\n[Part 1] test accuracy, mean over {len(SEEDS)} seeds")
    print(f"  {'task':>9} | " + " ".join(f"{m:>7}" for m in modes) + "   n_params")
    P1 = {}
    for task in tasks:
        row, npar = {}, None
        cells = ""
        for m in modes:
            accs = []
            for sd in SEEDS:
                torch.manual_seed(sd); np.random.seed(sd)
                Xtr, ytr = make_data(task, N=4000)
                Xte, yte = make_data(task, N=2000)
                torch.manual_seed(sd)
                model = DiagScalarAttn(Xtr.shape[-1], 16, m)
                acc, _ = train_eval(model, Xtr, ytr, Xte, yte)
                accs.append(acc); npar = n_params(model)
            row[m] = float(np.mean(accs)); row[m + "_sd"] = float(np.std(accs))
            cells += f"{np.mean(accs):>8.3f}"
        P1[task] = row
        print(f"  {task:>9} |{cells}    {npar}")
    sn = P1["self_neg"]
    print(f"  -> self_neg: signed={sn['signed']:.3f} phase={sn['phase']:.3f} "
          f"gate={sn['gate']:.3f} base={sn['base']:.3f}")
    print(f"     phase vs signed on self_neg: "
          f"phase is {'WORSE' if sn['phase'] < sn['signed'] - 0.05 else ('~tie' if abs(sn['phase']-sn['signed'])<0.05 else 'better')}"
          f" (cos periodicity = bad landscape for a pure sign flip)")

    # ---- Part 2: order parameter (learned coef on self_neg) ----
    print("\n[Part 2] order parameter: learned coef vs the query's negate bit "
          "(self_neg, phase model)")
    Xtr, ytr = make_data("self_neg", N=4000)
    Xte, yte = make_data("self_neg", N=2000)
    torch.manual_seed(1)
    model = DiagScalarAttn(Xtr.shape[-1], 16, "phase")
    acc, coef = train_eval(model, Xtr, ytr, Xte, yte)
    b = (Xte[:, 0, 1] > 0).numpy()                      # query negate bit
    coef = coef.detach().numpy()
    c0, c1 = float(coef[~b].mean()), float(coef[b].mean())
    print(f"  acc={acc:.3f};  mean coef | bit=0: {c0:+.3f}   | bit=1: {c1:+.3f}")
    print(f"  -> bimodal +/-1 tracking the self bit? "
          f"{'YES (phase emerged, ~+1 vs ~-1)' if c0 > 0.5 and c1 < -0.5 else 'no'}")

    # ---- Part 3: U(1) rotation -- phase vs real-signed (seed-averaged) ----
    print(f"\n[Part 3] U(1) rotation task: phase (rotates) vs real-signed (scalar), "
          f"mean over {len(SEEDS)} seeds")
    P3 = {}
    for m in ["phase", "real"]:
        accs = []
        for sd in SEEDS:
            torch.manual_seed(sd); np.random.seed(sd)
            Xtr, ytr, Q = make_rot_data(N=5000)
            Xte, yte, _ = make_rot_data(N=2000)
            torch.manual_seed(sd)
            model = DiagRotAttn(Xtr.shape[-1], m, n_cls=Q)
            acc, _ = train_eval(model, Xtr, ytr, Xte, yte, epochs=500)
            accs.append(acc); npar = n_params(model)
        P3[m] = float(np.mean(accs)); P3[m + "_sd"] = float(np.std(accs))
        print(f"  {m:>6}: acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}   "
              f"(chance={1.0/Q:.2f})   n_params={npar}")
    print(f"  -> phase {'BEATS' if P3['phase'] > P3['real'] + 0.08 else 'ties'} "
          f"real-signed on U(1): phase-qua-phase "
          f"{'pays (needs rotation; sr9/thm:u1-cocycle)' if P3['phase'] > P3['real'] + 0.08 else 'no clearer than sign'}")

    with open("cwf_complex_attention_results.json", "w") as f:
        json.dump({"part1_z2_factorial": P1,
                   "part2_order_param": {"acc": acc, "coef_bit0": c0, "coef_bit1": c1},
                   "part3_u1_rotation": P3}, f, indent=2)
    print("\nwrote cwf_complex_attention_results.json")
