"""
cwf_answersheet_demo.py -- intuition demo: three states of the hidden "answer sheet."

The hidden "answer sheet" = a single definite global configuration the wave would be
ignorance ABOUT (a noncontextual hidden-variable assignment). Whether it EXISTS, and
how hard it is to READ, splits the framework's deep states into three:

  EASY   (reducible)   -- classical substrate, affine rule (Rule 90, s'=L XOR R, degree 1):
                          the answer sheet exists AND has a closed-form shortcut
                          (row t = M^t . s0). Definite ground -> CF = 0 -> no phase.
  HARD   (irreducible) -- classical substrate, nonlinear rule (Rule 110,
                          s'=(C OR R) XOR (L AND C AND R), degree 3, universal): the answer sheet
                          exists but there is NO shortcut -- you must run it.
                          Still a definite ground -> CF = 0 -> no phase.
  ABSENT (no ground)   -- self-referential constraint (odd Liar cycle): NO consistent
                          assignment exists at all. No answer sheet -> no definite
                          ground -> CF = 1 -> genuine phase (holonomy = NOT forces i).

EASY vs HARD is the IRREDUCIBILITY axis (shortcut or not; algebraic degree); both keep
CF = 0. ABSENT is the PHASE axis (off the cube). The idiom shift from the classical ECA
(panels 1-2) to the frustrated cycle (panel 3) IS the lesson: a classical substrate
ALWAYS has an answer sheet; only self-reference removes it. Uses the validated ABM
contextual-fraction LP for the cycle. Intuition aid (not a book figure unless adopted).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_phase_contextuality import contextuality_lp, _validate_lp

HERE = os.path.dirname(os.path.abspath(__file__))
NOT = np.array([[0.0, 1.0], [1.0, 0.0]]); I2 = np.eye(2)


def eca_spacetime(rule, steps=44, width=89):
    """Single central seed, periodic boundary; returns the steps x width spacetime."""
    grid = np.zeros((steps, width), dtype=int)
    grid[0, width // 2] = 1
    for t in range(steps - 1):
        s = grid[t]
        left, right = np.roll(s, 1), np.roll(s, -1)
        idx = (left << 2) | (s << 1) | right
        grid[t + 1] = (rule >> idx) & 1
    return grid


def cycle_cf(n):
    observables = [(i,) for i in range(n)]; contexts = [(i, (i + 1) % n) for i in range(n)]
    tables = {}
    for ci in range(n):
        t = np.zeros((2, 2)); t[0, 1] = t[1, 0] = 0.5; tables[ci] = t
    return float(contextuality_lp(contexts, observables, tables)["contextual_fraction"])


def holonomy_is_NOT(n):
    M = I2.copy()
    for _ in range(n):
        M = NOT @ M
    return bool(np.linalg.norm(M - NOT) < 1e-9)


def draw_cycle(ax, n):
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    xs, ys = np.cos(ang), np.sin(ang); color = [i % 2 for i in range(n)]
    for i in range(n):
        j = (i + 1) % n
        if color[i] != color[j]:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="C2", lw=2.6, zorder=1)
        else:
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="C3", lw=3.2, ls="--", zorder=1)
            mx, my = (xs[i] + xs[j]) / 2, (ys[i] + ys[j]) / 2
            ax.text(mx * 1.55, my * 1.55, "X conflict:\nmust differ,\nforced equal", color="C3",
                    ha="center", va="center", fontsize=8, fontweight="bold")
    for i in range(n):
        fc = "white" if color[i] == 0 else "black"
        ax.scatter([xs[i]], [ys[i]], s=430, c=fc, edgecolors="black", linewidths=1.5, zorder=3)
        ax.text(xs[i], ys[i], str(color[i]), color=("black" if color[i] == 0 else "white"),
                ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.05, 1.7); ax.set_aspect("equal"); ax.axis("off")


def main():
    _validate_lp(); print()
    n_odd = 5; cf_o = cycle_cf(n_odd)
    g90, g110 = eca_spacetime(90), eca_spacetime(110)
    print("EASY   Rule 90  (affine, degree 1): definite ground, CF=0, closed-form shortcut")
    print("HARD   Rule 110 (nonlinear, degree 3, universal): definite ground, CF=0, no shortcut")
    print(f"ABSENT odd cycle n={n_odd}: CF={cf_o:.3f}  holonomy=NOT? {holonomy_is_NOT(n_odd)}")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    # EASY
    axes[0].imshow(g90, cmap="binary", interpolation="nearest", aspect="auto")
    axes[0].set_title("EASY (reducible)\nRule 90:  $s' = L \\oplus R$   (degree 1, affine)", fontsize=10.5)
    axes[0].set_xlabel("answer sheet EXISTS + closed-form shortcut (row $t = M^{t}s_0$)\n"
                       "$\\rightarrow$ definite ground, CF $=0$, no phase", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([]); axes[0].set_ylabel("time $\\downarrow$")
    # HARD
    axes[1].imshow(g110, cmap="binary", interpolation="nearest", aspect="auto")
    axes[1].set_title("HARD (irreducible)\nRule 110:  $s' = (C \\vee R) \\oplus (L \\wedge C \\wedge R)$   (degree 3, universal)", fontsize=10.5)
    axes[1].set_xlabel("answer sheet EXISTS but NO shortcut (must run it, step by step)\n"
                       "$\\rightarrow$ definite ground, CF $=0$, no phase", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])
    # ABSENT
    draw_cycle(axes[2], n_odd)
    axes[2].set_title("ABSENT (no ground)\nodd Liar cycle:  no consistent assignment", fontsize=10.5)
    axes[2].text(0, -1.86, "NO answer sheet (frustrated) $\\rightarrow$ no definite ground,\n"
                 f"CF $={cf_o:.0f}$, genuine phase (holonomy $=$ NOT forces $i$)",
                 ha="center", va="top", fontsize=9, color="C3")

    # (No suptitle: the LaTeX \caption carries the top-level framing in the book; the panel
    #  titles + per-panel captions keep the standalone figure self-explanatory.)
    fig.tight_layout()
    pth = os.path.join(HERE, "fig_answersheet_demo.png")
    plt.savefig(pth, dpi=140, bbox_inches="tight"); plt.close()
    print(f"\nWrote {pth}")


if __name__ == "__main__":
    main()
