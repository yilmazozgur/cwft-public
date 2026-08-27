"""
cwf_complex_attention_lm.py -- Route C on a REAL small transformer.
(TRANSFORMER_PHASE_NOTE.md  experiment, the real-task follow-on to the synthetic
cwf_complex_attention.py.)

Moves route C (a self-sourced PHASE at the self-attention diagonal) off the
synthetic single-coefficient toy onto a full small transformer (embed + multi-head
attention + MLP + LayerNorm + readout) on an ALGORITHMIC task -- the modular-
arithmetic / grokking testbed (Power et al. 2022; Nanda et al. 2023), where
transformers are KNOWN to solve the task by representing numbers as rotations
e^{i w a} and adding angles: a genuine U(1) mechanism on a real transformer.

Honest framing: this is "a small transformer on algorithmic data," NOT a natural-
language LM.  Reason: the route-C prediction is U(1)-SPECIFIC (phase helps U(1)/
rotation, hurts Z_2/sign -- cwf_complex_attention.py).  Modular addition is the
canonical U(1) task; integer addition (same inputs, NO wraparound) is the matched
non-cyclic control.  A language negation task would test the Z_2 horn, where the
prediction is a NULL -- a weak test.

THE DIFFERENTIAL PREDICTION (on a real transformer).  Route C (phase at the
diagonal) should help MODULAR addition (U(1)) MORE than integer addition (no
wraparound): (phase - real)_mod  >  (phase - real)_int.  The effect should live at
LOW capacity / fixed budget (phase is an inductive bias for rotation, not extra
expressivity -- a big enough real transformer learns the rotations itself).

Variants (matched params; differ only in the DIAGONAL self-term treatment):
    real   : standard transformer (diagonal untouched)
    phase  : rotate the diagonal self-term's complex pairs by a self-sourced angle
    signed : scale the diagonal self-term's pairs by tanh(.) (the real Z_2 rival)

Reports whatever happens, incl. phase helping both (=capacity, not U(1)-specific)
or neither (MLP compensates).  Seeded, CPU, torch.  Writes
cwf_complex_attention_lm_results.json.
"""

import json
import numpy as np
import torch
import torch.nn as nn


def make_task(p, kind):
    """All (a,b) pairs; sequence [a, b, EQ]; label at EQ position."""
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    EQ = p                                              # '=' token id
    X = np.stack([a, b, np.full_like(a, EQ)], axis=1)   # (N,3)
    if kind == "mod":
        y = (a + b) % p                                 # U(1): p classes
        n_cls = p
    elif kind == "int":
        y = a + b                                       # no wraparound: 2p-1 classes
        n_cls = 2 * p - 1
    else:
        raise ValueError(kind)
    return X.astype(np.int64), y.astype(np.int64), n_cls, p + 1  # vocab = p numbers + EQ


class RouteCAttn(nn.Module):
    def __init__(self, d, n_heads, mode):
        super().__init__()
        assert d % n_heads == 0 and (d // n_heads) % 2 == 0
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mode = mode
        if mode != "real":
            self.thead = nn.Linear(d, d // 2)           # one angle/scale per complex pair

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                # (B,h,T,dh)
        att = torch.softmax((q @ k.transpose(-2, -1)) / np.sqrt(self.dh), dim=-1)
        out = att @ v                                   # (B,h,T,dh)  incl. diagonal A_ii V_i
        if self.mode != "real":
            diag = torch.diagonal(att, dim1=-2, dim2=-1)        # (B,h,T) = A_ii
            self_term = diag.unsqueeze(-1) * v                  # (B,h,T,dh)
            t = self.thead(x).view(B, T, self.h, self.dh // 2).permute(0, 2, 1, 3)
            st = self_term.view(B, self.h, T, self.dh // 2, 2)
            if self.mode == "phase":                            # rotate each pair
                cs, sn = torch.cos(t), torch.sin(t)
                x0, x1 = st[..., 0], st[..., 1]
                mod = torch.stack([cs * x0 - sn * x1, sn * x0 + cs * x1], dim=-1)
            elif self.mode == "signed":                         # scale each pair (Z_2 rival)
                mod = torch.tanh(t).unsqueeze(-1) * st
            mod = mod.view(B, self.h, T, self.dh)
            out = out - self_term + mod                         # swap diagonal contribution
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class TinyTransformer(nn.Module):
    def __init__(self, vocab, n_cls, d=48, n_heads=4, T=3, mode="real"):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(0.02 * torch.randn(T, d))
        self.ln1 = nn.LayerNorm(d); self.attn = RouteCAttn(d, n_heads, mode)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.head = nn.Linear(d, n_cls)

    def forward(self, idx):
        x = self.emb(idx) + self.pos
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return self.head(x[:, -1])                       # predict at EQ position


def run(kind, mode, seed, p=31, steps=4000, d=48):
    torch.manual_seed(seed); np.random.seed(seed)
    X, y, n_cls, vocab = make_task(p, kind)
    N = len(X)
    perm = np.random.permutation(N)
    ntr = int(0.7 * N)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
    Xte = torch.tensor(X[te]); yte = torch.tensor(y[te])
    torch.manual_seed(seed)
    model = TinyTransformer(vocab, n_cls, d=d, mode=mode)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1.0)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad(); lossf(model(Xtr), ytr).backward(); opt.step()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean().item()
    npar = sum(pp.numel() for pp in model.parameters())
    return acc, npar


if __name__ == "__main__":
    print("=" * 76)
    print("ROUTE C on a REAL small transformer: modular (U(1)) vs integer (no-wrap)")
    print("=" * 76)
    SEEDS = [0, 1, 2]
    modes = ["real", "phase", "signed"]
    P, STEPS, D = 31, 4000, 48
    print(f"  p={P}, d={D}, 1 layer/4 heads, {STEPS} steps, mean over {len(SEEDS)} seeds")
    res = {}
    for kind in ["mod", "int"]:
        print(f"\n  [{kind}-add]  {'mode':>7} {'test_acc':>10}  n_params")
        res[kind] = {}
        for m in modes:
            accs, npar = [], None
            for sd in SEEDS:
                a, npar = run(kind, m, sd, p=P, steps=STEPS, d=D)
                accs.append(a)
            res[kind][m] = float(np.mean(accs)); res[kind][m + "_sd"] = float(np.std(accs))
            print(f"            {m:>7} {np.mean(accs):>9.3f} +/-{np.std(accs):.3f}  {npar}")
    dm = res["mod"]["phase"] - res["mod"]["real"]
    di = res["int"]["phase"] - res["int"]["real"]
    print(f"\n  DIFFERENTIAL (phase - real):  mod-add = {dm:+.3f}   int-add = {di:+.3f}")
    print(f"  -> phase is U(1)-specific? {'YES' if dm > di + 0.03 and dm > 0.02 else 'NO / inconclusive'}"
          f"  (predict mod >> int if route C buys the rotation inductive bias)")
    res["differential"] = {"phase_minus_real_mod": dm, "phase_minus_real_int": di}
    with open("cwf_complex_attention_lm_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote cwf_complex_attention_lm_results.json")
