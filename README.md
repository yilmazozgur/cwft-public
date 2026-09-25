# Computational Wave Field Theory — code base

Experiments and code only: the scripts, result records and reference figures behind
**Computational Wave Field Theory (CWFT)**, Ozgur Yilmaz's research program on modelling computational substrates
with the complex-wave mathematics of quantum mechanics: each substrate with its own effective
constants (ℏ_c, c_c, G_c), the epistemic wave as the best object an embedded observer can hold
over a computationally irreducible ground, and the clean splits the program produced —
analog-gravity kinematics yes / Einstein dynamics only on holographic error-correcting codes;
the ℓ² wave as compression yes / phase only under self-reference; and the self-reference
route to the imaginary unit.

The canonical statement of the framework is the book *Computational Wave Field Theory*
(Zenodo concept DOI, which resolves to the latest edition: [10.5281/zenodo.19862755](https://doi.org/10.5281/zenodo.19862755)).
This repository is its replicability companion: every number in the book, and in the
manuscripts derived from it, traces to a script here and to a committed record. The book and
the manuscripts themselves are not part of this repository. Everything is seeded, CPU-only
and toy-scale; negative results are kept, not hidden.

## Layout

| path | contents |
|---|---|
| `cwft_experiments/` | all experiment scripts (~155), their records (`results.json`, one key per experiment, plus per-script `*_results.json`), the reference figures used in the book (`fig_*.png`), `REPORT.md` (the running write-up of the gravity testbed, including what failed), `run_all.sh`, and `SCRIPT_INDEX.md` — an auto-generated one-line-per-script index |
| `cwft_experiments/README.md` | notes for the VSA / hyperdimensional-computing subset: runtimes of the long scripts, two estimator details that matter |
| `requirements.txt`, `CITATION.cff`, `LICENSE` | environment, citation metadata, MIT license |
| `make_script_index.py` | generates `cwft_experiments/SCRIPT_INDEX.md` (run by the mirroring script in the working repository) |

The analog-gravity suite is also published on its own, with a tagged release matching the
paper that uses it: [cwft_analog_gravity](https://github.com/yilmazozgur/cwft_analog_gravity)
(release `v1.1-jpc`). The scripts here are the same code in its working form.

## Running

```bash
pip install -r requirements.txt
cd cwft_experiments
python3 cwf_test6b.py            # example: transport horizon = clock-freeze surface (Test 6)
python3 cwf_sr7_renou_realize.py # example: the Renou real-vs-complex separation with the self-referential i
bash run_all.sh                  # every script; logs to /tmp/<script>.log
```

Each script is standalone: it seeds its own RNG, writes its record next to itself and prints
its findings. Most finish in seconds to minutes; the handful of long ones (an hour or two) are
listed in `cwft_experiments/README.md`. Records were produced with Python 3.10.12, NumPy 2.2.6,
SciPy 1.15.3, Matplotlib 3.10.3, stim 1.16.0, networkx 3.4.2. Wall-clock timings that some
scripts print depend on the machine and are not portable; the mathematical outputs are, with one
exception. The scripts that draw random two-qubit Clifford gates (`cwf_pagecurve.py`,
`cwf_a3_clifford_horizon.py`, the `cwf_hp_*` scripts, `cwf_a3b_emergent_geometry.py`,
`cwf_ap_phaseF/G/H/I_*`) originally used `stim.Tableau.random`, which takes no seed, so their
committed records are one draw of a random ensemble. The gate streams are now seeded; a re-run is
deterministic and returns the same reported behaviour, but individual numbers differ from the
committed records (the largest shift: the two-knob I3 crossing, 0.16 committed vs 0.20 re-run).

## Map by cluster

Script prefixes group into the program's threads (chapter numbers refer to the book; the
exact purpose of every script is one line in `cwft_experiments/SCRIPT_INDEX.md`):

| cluster | scripts |
|---|---|
| Substrate and gravity testbed (Ch5) | `cwf_substrate` (library: reservoir lattice, Rule 110), `cwf_experiments` (E-series battery), `cwf_e2v2/3`, `cwf_refine*`, `cwf_scrambler`, `cwf_test5`, `cwf_test6`, `cwf_test6b`, `cwf_clausius`, `cwf_clausius2d`, `cwf_a1_hysteresis`, `cwf_a2_*`, `cwf_kappa`, `cwf_p1_ensembles`, `_debug_a2*` |
| Holographic / error-correcting codes (Ch5) | `cwf_a3_clifford_horizon`, `cwf_a3b_emergent_geometry`, `cwf_a3b2_happy_perfect` (η_c = 1), `cwf_a3b3_cosmological`, `cwf_a3b4_first_law`, `cwf_a3d_dyngeom`, `cwf_a3e_modular_firstlaw`, `cwf_pagecurve`, `cwf_hp_lib` (library) + `cwf_hp_phase1–5` (decoding, Page curve, scrambling), `cwf_diag_arc` |
| Gravitating charge / equivalence-principle arc | `cwf_d5_bifurcation`, `cwf_d5b_spatial`, `cwf_d5c_hardening`, `cwf_d5d_bekenstein`, `cwf_d5e_*` |
| Programs I–III: trade-off law, ℏ_c gauge, predictive cost, reduction, Bell, reconstruction | `cwf_b1_*`, `cwf_b1b2_figs`, `cwf_b2_bridge`, `cwf_c1_predictive_cost`, `cwf_c1b_cost_scaling`, `cwf_c2_reduction`, `cwf_c2b_progII_theorem`, `cwf_c3_bell`, `cwf_d1_reconstruction`, `cwf_d2_entanglement`, `cwf_invariance_gate` |
| The epistemic wave (Ch6) | `cwf_psi_l2_vs_l1`, `cwf_psi_scaleup`, `cwf_phase_contextuality`, `cwf_backaction_bell`, `cwf_spekkens_restriction`, `cwf_substrate_nonclassicality`, `cwf_answersheet_demo` |
| Self-reference arc (Ch6, sr1–sr13) | `cwf_sr1_liar_forced_i` … `cwf_sr13_maxwell_action_theorem`, `cwf_selfref_lyapunov`, `cwf_selfref_ring`, `cwf_selfref_product`, `cwf_groundlessness`, `cwf_nonlocalmagic_eca` |
| Action principle (phases B–I) | `cwf_ap_phaseB` … `cwf_ap_phaseI_multicritical` (two-knob MIPT, Ryu–Takayanagi line, axes, multicriticality) |
| Three axes / T-series | `cwf_t11_*` (trade-off law at scale), `cwf_t21_*` (magic as order parameter), `cwf_t22_nonlinear`, `cwf_t23_geometry_superposition`, `cwf_t24_computation_generated_geometry`, `cwf_t25_both_axes`, `cwf_tp_stage0` |
| VSA / hyperdimensional computing (phase-space lens) | `cwf_fpe_*`, `cwf_resonator_*`, `cwf_gaphamming_*`, `cwf_rope_fovea` |
| Transformers / complex attention (needs `torch`) | `cwf_complex_attention`, `cwf_complex_attention_lm`, `cwf_complex_attention_lm_sweep` |
| Readings of other problems | `cwf_schrodinger_control` (stochastic optimal control), `cwf_tsp_sat` (NP-hard optimization) |
| Number-theoretic playground (closed thread, kept for the record) | `cwf_prime_playground`, `cwf_primon_gas`, `cwf_primon_entanglement`, `cwf_primon_records`, `cwf_roadA_bbm` |
| Retired, kept as dated records | `cwf_r5_phase1/2/3a` (learnable-substrate line, retired 2026-06-05) |
| Checks | `test_identities.py` (unit tests for the identities the manuscripts rest on) |

## Records and figures

`results.json` is the multi-key record of the gravity, holography, action-principle and
T-series experiments (each script owns its keys and rewrites only those). The per-script
`*_results.json` files hold the epistemic-wave, self-reference, VSA and remaining records.
Every figure in the book and manuscripts is plotted from these files; the `fig_*.png` here are
the reference outputs, so a re-run can be compared against what was published.

## Provenance and honesty notes

This is research code in its working form: exploratory, toy-scale, with retired lines and
debug scripts left in place rather than curated away. Scripts state their own status in
their docstrings (retired, isolated, not wired into the book). `REPORT.md` records the gravity
testbed's negative results alongside the positive ones — the Clausius / area-law failure is
as much a result as the horizon test that passed.

## Citing

Please cite the book (DOI above) for the framework and this repository for the code; see
`CITATION.cff`. Author: Ozgur Yilmaz, Adana Alparslan Türkeş Science and Technology
University, ORCID [0009-0002-8781-5386](https://orcid.org/0009-0002-8781-5386).

## License

MIT — see `LICENSE`.
