"""
cwf_fpe_regime.py -- dynamical-REGIME recognition from ONE hypervector: the
joint (x,p) phase-space code separates what any position-only code cannot.
(Follows cwf_fpe_trajectory.py's joint code; the recognition-task complement of
its time-reversal split.)

Three trajectory classes on the SAME spatial ring (2-D positions, radius 3, so
the position marginals are near-identical):
  (i)  CW orbit    -- constant angular velocity -omega,
  (ii) CCW orbit   -- constant angular velocity +omega,
  (iii) ring WALK  -- random angular steps, same speed magnitude (a fair-coin
        tangential direction each step).
A trajectory is S=48 samples of (x_t, p_t) with x in R^2 on the ring and
p = velocity (tangent, magnitude r*omega), bundled into ONE hypervector by the
joint code  phi(x,p)_j = exp(i (x . theta_j + p . omega_j)),  theta_j, omega_j
in R^2 with i.i.d. N(0, sigma^2) entries (sigma_x = sigma_p = 2, N = 4000) --
cwf_fpe_trajectory.py's code with each conjugate axis widened to 2-D.
Nuisances: random starting phase, +-10% radius jitter per trajectory (speed
scales with it).  NOTE the full-orbit design: in 1-D phase space a full CW and
CCW oscillator orbit trace the SAME (x,p) ellipse as SETS -- only the 2-D ring
makes the two directions distinct point-SETS ((q, v) vs (q, -v)); a bundle is
an unordered set, so this is the honest minimal arena for the split.

Two arms at matched N: JOINT (x,p) code vs POSITION-ONLY code (x codebook
alone).  Class templates from 8 prototype trajectories per class (separate
seed); 100 test trajectories per class; two readouts -- cosine to the MEAN
class template, and nearest-prototype (1-NN) cosine over the 24 prototypes;
3 codebook seeds.

RESULTS (measured, this run; 3 seeds x 300 test trajectories):
  - CHIRALITY (the headline): CW and CCW rows are PERFECT under the joint code
    (1.00 / 1.00, zero cross-talk, both readouts) and a COIN FLIP under the
    position-only code (0.56/0.44 and 0.55/0.45, both readouts) -- the two
    orbits are the same position set by construction, and only the momentum
    axis separates them.
  - WALK is the hard class, and it exposes the READOUT, not the code: the mean
    walk template is nearly collinear with the CW+CCW mixture (a random walk
    bundles both tangent branches), so mean-template scores WALK at only 0.37
    (joint overall 0.789 +- 0.028).  The 1-NN readout lifts WALK to 0.71 and
    the joint overall to 0.904 +- 0.056.
  - Position-only overall: 0.461 +- 0.009 (template), 0.587 +- 0.031 (1-NN).
    HONEST WRINKLE: position-only 1-NN actually classifies WALK slightly
    BETTER than the joint code (0.75 vs 0.71) -- a finite walk covers a
    diffusive arc (~1 rad), so its position distribution alone is a good walk
    detector; what position statistics can NEVER carry is the chirality.
  -> One phase-space hypervector separates all three regimes only through the
     momentum axis; the walk-vs-orbit part is also partly visible to position
     statistics, the CW-vs-CCW part is not at all.

CPU, numpy, seeded.  Writes cwf_fpe_regime_results.json.
"""

import json

import numpy as np

N = 4000
SIG_X = 2.0
SIG_P = 2.0
RADIUS = 3.0
OMEGA = 1.0                      # angular speed; |p| = r * omega
S = 48                           # samples per trajectory
DT = 2 * np.pi / S               # one revolution for the orbits
N_PROTO = 8                      # prototype trajectories per class template
N_TEST = 100                     # test trajectories per class
SEEDS = 3
CLASSES = ["CW", "CCW", "WALK"]


def make_codebooks(rng):
    """theta, omega: (N,2) each -- 2-D position and 2-D momentum codebooks."""
    return SIG_X * rng.standard_normal((N, 2)), SIG_P * rng.standard_normal((N, 2))


def trajectory(cls, rng):
    """(Q, V): S x 2 positions on the ring and velocities (tangent)."""
    a0 = rng.uniform(0, 2 * np.pi)
    r = RADIUS * (1 + rng.uniform(-0.1, 0.1))
    if cls == "CW":
        signs = -np.ones(S)
    elif cls == "CCW":
        signs = np.ones(S)
    else:                                    # WALK: fair-coin tangential steps
        signs = rng.choice([-1.0, 1.0], S)
    ang = a0 + np.concatenate([[0.0], np.cumsum(signs[:-1]) * OMEGA * DT])
    Q = r * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    V = r * OMEGA * signs[:, None] * np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    return Q, V


def encode(Q, V, th, om, joint):
    """Bundle of S joint (or position-only) atoms, normalized."""
    ph = Q @ th.T
    if joint:
        ph = ph + V @ om.T
    b = np.exp(1j * ph).sum(axis=0)
    return b / (np.linalg.norm(b) + 1e-12)


def cossim(a, b):
    return float(np.real(np.vdot(a, b)))


def run_seed(seed):
    rng_code = np.random.default_rng(100 + seed)
    rng_proto = np.random.default_rng(200 + seed)      # separate template seed
    rng_test = np.random.default_rng(300 + seed)
    th, om = make_codebooks(rng_code)

    res = {}
    for arm, joint in [("joint", True), ("position_only", False)]:
        templates, protos = [], []
        rp = np.random.default_rng(rng_proto.integers(1 << 31))
        for ci, cls in enumerate(CLASSES):
            bs = [encode(*trajectory(cls, rp), th, om, joint)
                  for _ in range(N_PROTO)]
            protos += [(b, ci) for b in bs]
            t = sum(bs)
            templates.append(t / (np.linalg.norm(t) + 1e-12))
        conf = np.zeros((3, 3), int)          # mean-template readout
        conf_nn = np.zeros((3, 3), int)       # nearest-prototype (1-NN) readout
        rt = np.random.default_rng(rng_test.integers(1 << 31))
        for ci, cls in enumerate(CLASSES):
            for _ in range(N_TEST):
                b = encode(*trajectory(cls, rt), th, om, joint)
                pred = int(np.argmax([cossim(t, b) for t in templates]))
                conf[ci, pred] += 1
                sims = [cossim(p, b) for p, _ in protos]
                conf_nn[ci, protos[int(np.argmax(sims))][1]] += 1
        res[arm] = dict(confusion=conf.tolist(),
                        accuracy=float(np.trace(conf) / conf.sum()),
                        confusion_nn=conf_nn.tolist(),
                        accuracy_nn=float(np.trace(conf_nn) / conf_nn.sum()))
    return res


if __name__ == "__main__":
    print("=" * 74)
    print("REGIME RECOGNITION FROM ONE HYPERVECTOR: joint (x,p) vs position-only")
    print("=" * 74)
    print(f"ring r={RADIUS} (+-10%), omega={OMEGA}, S={S} samples, N={N}, "
          f"sigma_x=sigma_p={SIG_X}; {N_PROTO} prototypes/template, "
          f"{N_TEST} tests/class, {SEEDS} seeds")

    out = {"params": dict(N=N, sig_x=SIG_X, sig_p=SIG_P, radius=RADIUS,
                          omega=OMEGA, S=S, n_proto=N_PROTO, n_test=N_TEST,
                          seeds=SEEDS, classes=CLASSES,
                          nuisance="random start phase; +-10% radius jitter",
                          note=("2-D ring positions: full CW/CCW 1-D oscillator "
                                "orbits coincide as (x,p) SETS; the 2-D ring is "
                                "the minimal arena where chirality is a set "
                                "property")),
           "per_seed": []}
    for s in range(SEEDS):
        r = run_seed(s)
        out["per_seed"].append(r)
        print(f"  seed {s}: joint acc={r['joint']['accuracy']:.3f} "
              f"(1-NN {r['joint']['accuracy_nn']:.3f})  "
              f"position-only acc={r['position_only']['accuracy']:.3f} "
              f"(1-NN {r['position_only']['accuracy_nn']:.3f})")

    for arm in ["joint", "position_only"]:
        accs = [r[arm]["accuracy"] for r in out["per_seed"]]
        accs_nn = [r[arm]["accuracy_nn"] for r in out["per_seed"]]
        conf = np.mean([np.array(r[arm]["confusion"], float)
                        for r in out["per_seed"]], axis=0) / N_TEST
        conf_nn = np.mean([np.array(r[arm]["confusion_nn"], float)
                           for r in out["per_seed"]], axis=0) / N_TEST
        out[arm] = dict(accuracy_mean=float(np.mean(accs)),
                        accuracy_sd=float(np.std(accs)),
                        confusion_mean=conf.tolist(),
                        accuracy_nn_mean=float(np.mean(accs_nn)),
                        accuracy_nn_sd=float(np.std(accs_nn)),
                        confusion_nn_mean=conf_nn.tolist())
        print(f"\n  {arm}: mean-template acc {np.mean(accs):.3f} +- "
              f"{np.std(accs):.3f} | 1-NN acc {np.mean(accs_nn):.3f} +- "
              f"{np.std(accs_nn):.3f}")
        print(f"    mean-template confusion (rows true {CLASSES}) | 1-NN confusion:")
        for ci, cls in enumerate(CLASSES):
            print(f"      {cls:>4}: " + "  ".join(f"{v:.2f}" for v in conf[ci])
                  + "   |   " + "  ".join(f"{v:.2f}" for v in conf_nn[ci]))

    j, p = out["joint"], out["position_only"]
    cw_ccw_pos = (p["confusion_mean"][0][0] + p["confusion_mean"][1][1]) / 2
    cw_ccw_joint = (j["confusion_mean"][0][0] + j["confusion_mean"][1][1]) / 2
    print(f"\nVERDICT: chirality (CW/CCW diagonal): position-only "
          f"{cw_ccw_pos:.2f} (~chance among the orbit pair), joint "
          f"{cw_ccw_joint:.2f}; overall {p['accuracy_mean']:.2f} vs "
          f"{j['accuracy_mean']:.2f}.")
    print("         The momentum axis of the phase-space code carries the "
          "regime; position statistics cannot.")

    with open("cwf_fpe_regime_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_regime_results.json")
