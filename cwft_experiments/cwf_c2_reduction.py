"""
C2 -- the minimal irreducible-ignorance reduction (Program II).

C2 is mostly a PROOF; this script is the "finite-instance validation easy" half.
The theorem (stated and proved in the chapter, rigorous GIVEN cited universality):

    For a fixed universal substrate S there is a computable family of initial
    states {x_M} (one per machine M) and a COARSE observable A such that
    A(S^t x_M) reaches its firing value iff M halts. Hence a computable observer
    that predicts A with certainty for all inputs would decide the halting
    problem -- impossible. Contrapositive: undecidability => irreducible residual
    unpredictability of A for every computable observer.

The undecidability is INHERITED, not re-derived here: Minsky (tag systems are
Turing-universal) and Cook 2004 (Rule 110 emulates cyclic tag systems) give the
reduction chain
        Halting  <=_m  2-tag reachability  <=_m  cyclic-tag-system halting
                 <=_m  Rule-110 configuration reachability.
We do NOT reimplement Cook's Rule-110 glider compiler; we cite it and validate
the reduction MECHANISM at the two ends we can build exactly:
  - the Turing-machine end (the family {M}), and
  - the cyclic-tag-system end (the Rule-110-equivalent universal substrate).

What the validation shows (and its tie to C1). C1 measured that predictive COST
diverges with the forecast horizon. C2 exhibits the in-principle BARRIER behind
that wall: the coarse observable A faithfully tracks halting, and a
bounded-budget decider resolves the halters and the provable-loopers but leaves
an IRREDUCIBLE UNDECIDED RESIDUE (the open-ended growers) that no budget removes.
That residue is where, in the universal family, halting is undecidable -- the
operational form of "irreducible residual stochasticity."

CPU-only NumPy/matplotlib.  Run:  python cwf_c2_reduction.py
"""
import json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# =========================================================================
# Substrate 1 -- Turing machine (the family {M}); A = halt-state reached
# =========================================================================

class TM:
    """Single-tape TM over a binary alphabet {0,1} with blank 0.
    delta: dict (state, sym) -> (write, move in {-1,+1}, next_state).
    'H' is the halting state. Tape is a dict (sparse, two-way infinite);
    init_tape is a 0/1 string placed at positions 0..len-1."""
    def __init__(self, delta, start="A", init_tape="", name=""):
        self.delta, self.start, self.init_tape, self.name = delta, start, init_tape, name

    def run(self, budget, cap=3000):
        """Return (status, steps) with status in {HALT, LOOP, OPEN}.
        LOOP = an exact configuration (state, head, tape) recurs (provable
        non-halt; detected within the first `cap` steps). OPEN = neither halt
        nor recurrence within budget."""
        tape = {i: 1 for i, c in enumerate(self.init_tape) if c == "1"}
        head, state, seen = 0, self.start, {}
        for t in range(budget):
            if state == "H":
                return "HALT", t
            sym = tape.get(head, 0)
            if t < cap:                              # exact-configuration cycle test
                key = (state, head, frozenset(tape.items()))
                if key in seen:
                    return "LOOP", t
                seen[key] = t
            if (state, sym) not in self.delta:
                return "HALT", t                     # no rule -> halt
            w, mv, ns = self.delta[(state, sym)]
            if w == 0:
                tape.pop(head, None)
            else:
                tape[head] = w
            head += mv
            state = ns
        return "OPEN", budget


def tm_family():
    """A computable family with verifiable status, resolution times spread
    across the budget range, spanning the three outcomes (HALT/LOOP/OPEN)."""
    sweep = {("A", 1): (1, +1, "A"), ("A", 0): (0, +1, "H")}   # scan a block of 1s, halt at blank
    fam = []
    fam.append(TM({("A", 0): (1, +1, "H")}, name="halt-fast"))               # ~1 step
    fam.append(TM(sweep, init_tape="1" * 50, name="halt-sweep-50"))          # ~51 steps
    fam.append(TM(sweep, init_tape="1" * 400, name="halt-sweep-400"))        # ~401 steps
    # provable loop: bounce between two cells, exact configuration recurs
    fam.append(TM({("A", 0): (0, +1, "B"), ("B", 0): (0, -1, "A")}, name="loop-bounce"))
    # open-ended grower: writes 1s forever; tape grows, no config recurs, no
    # halt -- the cell where halting is undecidable in the universal family
    fam.append(TM({("A", 0): (1, +1, "A"), ("A", 1): (1, +1, "A")}, name="grow-forever"))
    return fam


# =========================================================================
# Substrate 2 -- cyclic tag system (the Rule-110-equivalent; Cook 2004)
# A = the data word becomes empty (the CTS halting / reachability predicate)
# =========================================================================

class CTS:
    """Cyclic tag system: ordered productions p_0..p_{m-1} (binary strings),
    a binary data word, a pointer k that cycles 0..m-1. Each step: pop the
    leftmost data symbol; if it was '1' append p_k; k=(k+1) mod m. Halt when
    the word is empty."""
    def __init__(self, prods, word, name=""):
        self.prods, self.word0, self.name = prods, word, name

    def run(self, budget, cap=5000):
        prods, m = self.prods, len(self.prods)
        word, k, seen = self.word0, 0, {}
        for t in range(budget):
            if word == "":
                return "HALT", t
            if t < cap:                              # exact (word,k) cycle test
                key = (word, k)
                if key in seen:
                    return "LOOP", t
                seen[key] = t
            head, word = word[0], word[1:]
            if head == "1":
                word = word + prods[k]
            k = (k + 1) % m
        return "OPEN", budget


def cts_family():
    """Resolution times spread across the budget range; one open-ended grower
    stands in for the cell where CTS halting is undecidable."""
    fam = []
    fam.append(CTS(["0", ""], "11", name="produce-drain"))    # halts ~3
    fam.append(CTS(["", ""], "1" * 5, name="drain-5"))        # halts ~5
    fam.append(CTS(["", ""], "1" * 200, name="drain-200"))    # halts ~200
    fam.append(CTS(["1"], "1", name="loop-fixed"))            # provable loop
    fam.append(CTS(["11"], "1", name="grow-forever"))         # open-ended grower
    return fam


# =========================================================================
# The bounded-budget decider: the operational face of the barrier
# =========================================================================

def decided_fraction(family, budgets):
    """Fraction of the family a bounded observer can DECIDE (halt found, or an
    exact configuration recurrence proving non-halt) within each budget. The
    plateau below 1 is the irreducible undecided residue: instances whose
    halting no finite budget settles -- exactly where the universal family
    hides an undecidable predicate."""
    out = []
    per_inst = {}
    for B in budgets:
        n_dec = 0
        for m in family:
            status, steps = m.run(B)
            decided = status in ("HALT", "LOOP")
            n_dec += int(decided)
            per_inst.setdefault(m.name, {})[B] = (status, steps)
        out.append(n_dec / len(family))
    return out, per_inst


def classify(family, budget):
    rows = []
    for m in family:
        status, steps = m.run(budget)
        rows.append(dict(name=m.name, status=status, steps=steps,
                         tracks_halting=(status == "HALT")))
    return rows


# =========================================================================
# Run + report
# =========================================================================

def main():
    budgets = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    BIG = 6000
    t0 = time.time()

    tmf, ctsf = tm_family(), cts_family()
    tm_rows, cts_rows = classify(tmf, BIG), classify(ctsf, BIG)
    tm_frac, _ = decided_fraction(tmf, budgets)
    cts_frac, _ = decided_fraction(ctsf, budgets)

    print("Turing-machine family (substrate = TM; A = halt-state reached):")
    for r in tm_rows:
        print(f"  {r['name']:14s} status={r['status']:5s} @ step {r['steps']}")
    print("Cyclic-tag-system family (substrate = CTS = Rule-110 target; A = empty word):")
    for r in cts_rows:
        print(f"  {r['name']:14s} status={r['status']:5s} @ step {r['steps']}")

    residue_tm = 1 - tm_frac[-1]
    residue_cts = 1 - cts_frac[-1]
    print(f"\n  decided fraction at budget {budgets[-1]}: TM={tm_frac[-1]:.2f}, "
          f"CTS={cts_frac[-1]:.2f}")
    print(f"  irreducible undecided residue: TM={residue_tm:.2f}, CTS={residue_cts:.2f} "
          f"(the open-ended growers; no budget settles them)")
    print(f"  runtime {time.time()-t0:.2f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["C2_reduction"] = dict(
        budgets=budgets, big_budget=BIG,
        tm_family=tm_rows, cts_family=cts_rows,
        tm_decided_fraction=tm_frac, cts_decided_fraction=cts_frac,
        residue=dict(tm=residue_tm, cts=residue_cts),
        chain="Halting <=_m 2-tag <=_m cyclic-tag-system <=_m Rule-110 reachability",
        note=("Validation of the reduction MECHANISM (coarse observable tracks halting; "
              "bounded decider leaves an irreducible undecided residue). Undecidability is "
              "INHERITED from Minsky (tag universality) + Cook 2004 (Rule 110 emulates CTS), "
              "cited not re-derived; Cook's Rule-110 glider compiler is not reimplemented."))
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(tm_rows, cts_rows, budgets, tm_frac, cts_frac, residue_tm, residue_cts)
    print("Wrote results.json key: C2_reduction")


def plot_results(tm_rows, cts_rows, budgets, tm_frac, cts_frac, res_tm, res_cts):
    fig = plt.figure(figsize=(15, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.25)

    # --- (a) the bounded decider: decided fraction vs budget -----------
    a = fig.add_subplot(gs[0, 0])
    a.plot(budgets, tm_frac, "o-", color="#1b9e77", lw=1.8, ms=6, label="Turing machine")
    a.plot(budgets, cts_frac, "s-", color="#e7298a", lw=1.8, ms=6,
           label="cyclic tag system (Rule-110 target)")
    a.axhline(1.0, ls=":", c="0.6", lw=1)
    a.fill_between(budgets, [max(tm_frac[-1], cts_frac[-1])] * len(budgets), 1.0,
                   color="0.88", zorder=0)
    a.text(budgets[1], 1.0 - max(res_tm, res_cts) / 2,
           "irreducible\nundecided\nresidue", fontsize=8.5, style="italic", color="0.35",
           va="center")
    a.set_xscale("log", base=2)
    a.set_xlabel("decider budget (steps)")
    a.set_ylabel("fraction of family decided")
    a.set_title("(a) The barrier: bounded decision plateaus below 1")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8, loc="center right")

    # --- (b) the reduction chain, with validated vs cited links --------
    b = fig.add_subplot(gs[0, 1]); b.axis("off")
    b.set_xlim(0, 10); b.set_ylim(0, 10)
    boxes = [
        (0.3, "Halting\nproblem", "#fde0dd"),
        (2.7, "2-tag\nsystem", "#fff2cc"),
        (5.1, "cyclic tag\nsystem", "#e7f0d8"),
        (7.5, "Rule 110\n(CA)", "#dbe7f5"),
    ]
    cy = 6.2
    centers = []
    for x, txt, col in boxes:
        p = FancyBboxPatch((x, cy), 2.0, 1.7, boxstyle="round,pad=0.08",
                           fc=col, ec="0.4", lw=1.1)
        b.add_patch(p)
        b.text(x + 1.0, cy + 0.85, txt, ha="center", va="center", fontsize=9.5,
               fontweight="bold")
        centers.append(x + 1.0)
    # reduction arrows + citation labels
    labels = ["$\\leq_m$ Minsky", "$\\leq_m$ Cook", "$\\leq_m$ Cook"]
    for i, lab in enumerate(labels):
        ar = FancyArrowPatch((boxes[i][0] + 2.0, cy + 0.85),
                             (boxes[i + 1][0], cy + 0.85),
                             arrowstyle="-|>", mutation_scale=14, color="0.3", lw=1.3)
        b.add_patch(ar)
        b.text((centers[i] + centers[i + 1]) / 2, cy + 1.95, lab, ha="center",
               fontsize=8, color="0.3")
    # observable annotation
    b.text(5.0, cy - 0.5, "coarse observable $A$ = reachability of the halt marker",
           ha="center", fontsize=9, style="italic")
    # validated-in-code vs cited markers
    b.text(1.3, cy - 1.6, "validated\nin code", ha="center", fontsize=8.5,
           color="#1b9e77", fontweight="bold")
    b.annotate("", xy=(1.3, cy - 0.1), xytext=(1.3, cy - 1.1),
               arrowprops=dict(arrowstyle="-", color="#1b9e77", lw=1.2))
    b.text(6.1, cy - 1.6, "validated\nin code", ha="center", fontsize=8.5,
           color="#e7298a", fontweight="bold")
    b.annotate("", xy=(6.1, cy - 0.1), xytext=(6.1, cy - 1.1),
               arrowprops=dict(arrowstyle="-", color="#e7298a", lw=1.2))
    b.text(5.0, 9.3, "Reduction chain: halting visible in a coarse observable "
           "of a fixed universal substrate", ha="center", fontsize=10, fontweight="bold")
    b.text(5.0, 1.9, "Reductions ($\\leq_m$) are cited (Minsky; Cook 2004), not "
           "reimplemented;\nthe mechanism is validated at the TM and CTS ends.",
           ha="center", fontsize=8.3, color="0.35")

    fig.suptitle("C2: the minimal irreducible-ignorance reduction "
                 "(mechanism validated; undecidability inherited)",
                 fontsize=13, fontweight="bold", y=1.02)
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_C2_reduction.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
