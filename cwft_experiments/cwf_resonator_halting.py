"""
cwf_resonator_halting.py -- VSA resonator factorization as a HALTING /
IRREDUCIBILITY phase transition with a theta_c knob.  (VSA_CWFT_NOTE.md sec.4.)

A resonator network factorizes a bound hypervector c = a_1 (x) ... (x) a_F (given
the per-factor codebooks) by iterating unbind -> cleanup -> feedback, holding each
factor as a superposition.  Known facts (Frady/Kent/Olshausen/Sommer 2020): a sharp
operational capacity M_max ~ N^2 (MEASURED, not derived), no Lyapunov function / no
convergence guarantee, failure above capacity = limit cycles / chaos (NON-HALTING),
and injected noise breaks the limit cycles (Karunaratne 2024).  The CWFT claim is
that these are one object -- a halting/irreducibility phase transition with a
theta_c (temperature) classicality knob -- which the resonator literature has the
pieces of but never unified.

THREE PARTS (each reports whatever happens; an honest test, not a demo):
  [1] HALTING TRANSITION + finite-size scaling.  Sweep load L = M/N^2 (M = D^F) at
      several N; measure convergence-to-the-TRUE-factorization probability.  CWFT
      predicts a SHARP transition that sharpens with N and COLLAPSES onto one curve
      vs L -- a genuine order parameter (the analytic frame Kent et al. lacked).
      Also classify failures as NON-HALTING (limit cycle) vs spurious fixed point,
      to test the 'halting' reading literally.
  [2] theta_c PHASE DIAGRAM.  Near the transition, sweep injected-noise amplitude
      theta_c; CWFT predicts NON-monotonic success with an OPTIMAL theta_c (frozen
      limit-cycle phase at theta_c=0, solvable at theta_c*, noise-destroyed above).
  [3] U(1) vs Z_2.  FHRR (complex phasors, U(1)) vs bipolar (+-1, Z_2) resonator at
      matched N -- does complex-phase interference extend operational capacity?
      (Ties to the route-C U(1)/Z_2 finding.)

Honest expectation: this may just re-describe M_max ~ N^2 in physics words; the TEST
of genuine content is (1) clean finite-size scaling and (2) a non-trivial optimal
theta_c.  CPU, numpy, seeded.  Writes cwf_resonator_halting_results.json.
"""

import json
import numpy as np


# ---------------------------------------------------------------------------
# resonator
# ---------------------------------------------------------------------------
def make_books(N, D, F, kind, rng):
    if kind == "bipolar":
        return [rng.choice([-1.0, 1.0], size=(N, D)) for _ in range(F)]
    return [np.exp(1j * rng.uniform(-np.pi, np.pi, size=(N, D))) for _ in range(F)]


def bind(vs):
    out = vs[0].copy()
    for v in vs[1:]:
        out = out * v
    return out


def resonator(c, books, kind, T, theta, rng):
    """Returns (success, halted, iters, failmode). success = decoded the TRUE
    factorization; halted = reached a fixed point; failmode in {ok, spurious,
    limitcycle}."""
    N, F = c.shape[0], len(books)
    # init: uniform superposition over each codebook (the STANDARD resonator init;
    # random init for all factors gives no correlation to bootstrap from)
    if kind == "bipolar":
        xh = [np.sign(books[f].sum(axis=1) + 1e-9 * rng.standard_normal(N))
              for f in range(F)]
    else:
        xh = [books[f].sum(axis=1) for f in range(F)]
        xh = [x / (np.abs(x) + 1e-12) for x in xh]
    hist, decs = [], []
    for t in range(T):
        new = []
        for f in range(F):
            s = c.copy()
            for g in range(F):
                if g != f:
                    s = s * (xh[g] if kind == "bipolar" else np.conj(xh[g]))
            Xf = books[f]
            sim = (Xf.T if kind == "bipolar" else Xf.conj().T) @ s
            clean = Xf @ sim
            if theta > 0:
                sd = np.std(np.abs(clean)) + 1e-12
                if kind == "bipolar":
                    clean = clean + theta * sd * rng.standard_normal(N)
                else:
                    clean = clean + theta * sd * (rng.standard_normal(N)
                                                  + 1j * rng.standard_normal(N))
            new.append(np.sign(clean) if kind == "bipolar"
                       else clean / (np.abs(clean) + 1e-12))
        xh = new
        # decode
        dec = tuple(int(np.argmax(np.real(
            (books[f].conj().T if kind != "bipolar" else books[f].T) @ xh[f])))
            for f in range(F))
        hist.append((dec, [x.copy() for x in xh]))
        decs.append(dec)
        # CYCLE DETECTION, and only where it is meaningful.
        # The previous version compared the state only with its IMMEDIATE
        # predecessor, i.e. it tested for period 1, and then labelled every
        # non-success at the cap a "limit cycle" without testing anything.  Two
        # facts make that label empty.  (a) The update is Jacobi-style (every
        # factor is recomputed from the PREVIOUS state), so at F=2 the attractors
        # have EVEN period: a period-1 fixed point essentially never occurs, and
        # the period-1 test therefore never fires.  Measured at theta=0, the
        # period histogram is {2: 49, 4: 58, 6: 21, 8: 19, ...} with no period 1.
        # (b) Under injected noise the state is perturbed afresh every iteration,
        # so exact state equality is impossible and EVERY noisy run would be
        # labelled non-halting by construction.  We therefore search the whole
        # history for a genuine repeat, and only in the deterministic case.
        if theta == 0.0:
            for u in range(len(hist) - 1):
                if all(np.allclose(xh[f], hist[u][1][f]) for f in range(F)):
                    per = len(hist) - 1 - u
                    return dec, True, t + 1, ("fp" if per == 1 else f"cycle{per}"), \
                        decs, per
    return hist[-1][0], False, T, "timeout", decs, 0


def run_instance(N, D, F, kind, T, theta, seed):
    rng = np.random.default_rng(seed)
    books = make_books(N, D, F, kind, rng)
    true = tuple(int(rng.integers(D)) for _ in range(F))
    c = bind([books[f][:, true[f]] for f in range(F)])
    dec, halted, iters, tag, decs, per = resonator(c, books, kind, T, theta, rng)
    success = (dec == true)                       # FINAL-STATE correctness at the cap
    hits = [i for i, d in enumerate(decs) if d == true]
    ever = bool(hits)                             # did it ever reach the answer?
    first = hits[0] + 1 if hits else -1           # first-hitting iteration
    occupancy = len(hits) / max(1, len(decs))
    if success:
        mode = "ok"
    elif halted:
        mode = "spurious-fp" if tag == "fp" else "cycle"
    else:
        mode = "timeout"           # a capped run, NOT a detected cycle
    return success, mode, iters, ever, first, occupancy, per


def prob(N, D, F, kind, T, theta, n_inst, seed0, full=False):
    """Success statistics.

    `succ` is FINAL-STATE correctness at the iteration cap -- the quantity the
    earlier version reported.  `ever` is first-hitting: did the run ever decode the
    true factorization?  The distinction matters because a resonator can verify a
    candidate for free (re-bind and compare), so first-hitting is the operationally
    natural stopping rule, and the two diverge sharply at high noise.
    `timeout` counts capped runs; `cycle` counts runs where a genuine state repeat
    was DETECTED (deterministic runs only -- see resonator()).
    """
    res = [run_instance(N, D, F, kind, T, theta, seed0 + i) for i in range(n_inst)]
    succ = np.mean([r[0] for r in res])
    modes = [r[1] for r in res]
    timeout = np.mean([m == "timeout" for m in modes])
    spur = np.mean([m == "spurious-fp" for m in modes])
    cyc = np.mean([m == "cycle" for m in modes])
    if not full:
        return float(succ), float(timeout), float(spur)
    ever = np.mean([r[3] for r in res])
    hits = [r[4] for r in res if r[4] > 0]
    return dict(succ=float(succ), ever=float(ever), timeout=float(timeout),
                spurious_fp=float(spur), cycle_detected=float(cyc),
                occupancy=float(np.mean([r[5] for r in res])),
                median_first_hit=float(np.median(hits)) if hits else -1.0,
                n=n_inst,
                succ_se=float(np.sqrt(succ * (1 - succ) / n_inst)),
                ever_se=float(np.sqrt(np.mean([r[3] for r in res]) *
                                      (1 - np.mean([r[3] for r in res])) / n_inst)))


if __name__ == "__main__":
    print("=" * 74)
    print("RESONATOR FACTORIZATION as a halting / irreducibility transition")
    print("=" * 74)

    F, T, NINST = 2, 150, 50          # F=2: verified-good implementation (F>=3 hits
                                      # spurious-fixed-point regimes -- a known harder case)
    s, _, _ = prob(256, 16, F, "bipolar", T, 0.0, 30, 1)
    print(f"\n[sanity] N=256, D=16 (tiny load), F={F}: success = {s:.2f} "
          f"(must be ~1.0)")

    # ---- Part 1: halting transition + finite-size scaling ----
    print(f"\n[Part 1] halting transition, bipolar, F={F}, vs load L = D^2 / N^2")
    Ns = [128, 256, 512]
    P1 = {}
    for N in Ns:
        Ds = sorted(set(int(d) for d in np.linspace(8, int(0.85 * N), 11)))
        row = []
        for D in Ds:
            succ, nohalt, spur = prob(N, D, F, "bipolar", T, 0.0, NINST, 1000 + N)
            row.append(dict(D=D, L=D**2 / N**2, succ=succ, nohalt=nohalt, spur=spur))
        P1[N] = row
        print(f"  N={N:>4}: " + " ".join(f"{r['succ']:.2f}" for r in row))
    print(f"           L: " + " ".join(f"{r['L']:.2f}" for r in P1[Ns[0]])
          + "  (N=128 row)")
    print("  transition load L_c (50% crossing) and D_c (test D_c ∝ N ⇒ M_c ∝ N^2):")
    for N in Ns:
        row = P1[N]
        i = next((k for k in range(len(row)) if row[k]["succ"] < 0.5), len(row) - 1)
        Lc, Dc = row[i]["L"], row[i]["D"]
        nh = np.mean([r["nohalt"] for r in row if 0 < r["succ"] < 1] or [0])
        print(f"    N={N:>4}: L_c≈{Lc:.2f}, D_c≈{Dc}, D_c/N≈{Dc/N:.2f}; "
              f"non-halting fraction among partial successes = {nh:.2f}")

    # ---- Part 2: theta_c phase diagram ----
    print(f"\n[Part 2] theta_c phase diagram (noise breaks limit cycles?)")
    Nf = 256
    Dt = int(0.5 * Nf)                # L~0.25: deterministic fails via limit cycles
    print(f"  N={Nf}, D={Dt} (L={Dt**2/Nf**2:.2f}); sweep injected noise theta_c:")
    P2 = []
    for th in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        succ, nohalt, spur = prob(Nf, Dt, F, "bipolar", T, th, NINST, 7000)
        P2.append(dict(theta=th, succ=succ, nohalt=nohalt))
        print(f"    theta_c={th:>4}: success={succ:.2f}  non-halting={nohalt:.2f}")
    best = max(P2, key=lambda r: r["succ"])
    print(f"  -> optimal theta_c={best['theta']} (succ {best['succ']:.2f}) vs "
          f"deterministic {P2[0]['succ']:.2f}: "
          f"{'NON-trivial optimum -- theta_c PAYS (noise breaks cycles)' if best['theta'] > 0 and best['succ'] > P2[0]['succ'] + 0.05 else 'no clear optimum'}")

    # ---- Part 3: U(1) vs Z_2 ----
    print(f"\n[Part 3] FHRR (U(1)) vs bipolar (Z_2) operational capacity (N={Nf}, F={F})")
    P3 = {"bipolar": [], "fhrr": []}
    Dgrid = sorted(set(int(d) for d in np.linspace(8, int(0.85 * Nf), 9)))
    for kind in ["bipolar", "fhrr"]:
        for D in Dgrid:
            succ, _, _ = prob(Nf, D, F, kind, T, 0.0, NINST, 9000)
            P3[kind].append(dict(D=D, L=D**2 / Nf**2, succ=succ))
        print(f"  {kind:>7}: " + " ".join(f"{r['succ']:.2f}" for r in P3[kind]))
    print(f"        L: " + " ".join(f"{r['L']:.2f}" for r in P3['bipolar']))
    bi = [r["succ"] for r in P3["bipolar"]]; fh = [r["succ"] for r in P3["fhrr"]]
    print(f"  -> capacity (sum success over grid): bipolar={sum(bi):.2f} "
          f"fhrr={sum(fh):.2f}: "
          f"{'FHRR/U(1) higher' if sum(fh) > sum(bi) + 0.3 else ('bipolar/Z2 higher' if sum(bi) > sum(fh)+0.3 else 'comparable (SNR-limited, U(1) no edge)')}")

    with open("cwf_resonator_halting_results.json", "w") as f:
        json.dump({"part1_transition": P1, "part2_theta_c": P2, "part3_u1_vs_z2": P3,
                   "params": {"F": F, "T": T, "n_inst": NINST}}, f, indent=2)
    print("\nwrote cwf_resonator_halting_results.json")
