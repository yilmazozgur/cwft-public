# Experiments — *The Physics of Hyperdimensional Computing*

Every number in the manuscript traces to a script here and to its committed
`*_results.json`. All scripts are CPU-only, seeded, and standalone: no shared state, no
network, no GPU.

## Environment

```
python >= 3.10
numpy >= 1.24
scipy >= 1.10          # only cwf_fpe_highdim.py (separable smoothing)
matplotlib >= 3.6      # figure generation only
stim                   # only cwf_clifford_*.py (not used by this manuscript)
networkx               # only cwf_a3b2_happy_perfect.py (not used by this manuscript)
```

Install the manuscript's dependencies with `pip install -r requirements.txt`.
Results were produced on Python 3.10.12 / NumPy 2.2.6 / SciPy 1.15.3 / Matplotlib 3.10.3.
Numerical identities are
platform-independent; **wall-clock timings are not** — see the note below.

## Reproducing

Each script writes its own JSON next to itself and prints its findings:

```bash
python3 cwf_fpe_uncertainty.py        # -> cwf_fpe_uncertainty_results.json
```

Regenerating every figure in the paper, from the committed JSONs:

```bash
cd ../cwft_manuscripts/vsa_phase_space
python3 make_figures.py               # -> figures/*.png
```

`make_figures.py` locates the results directory automatically. If you have moved things,
point it explicitly:

```bash
CWFT_EXPERIMENTS=/path/to/results python3 make_figures.py
```

## Runtimes

Most scripts finish in under a minute. Five are genuinely long and should not be run
under a short timeout:

| script | approximate runtime | why |
|---|---|---|
| `cwf_fpe_highdim.py` | ~1–2 h | dense grids up to `24^6` cells with separable smoothing at every `B` |
| `cwf_fpe_beyond_capacity.py` | ~20–40 min | 9 loads × 2 arms × 20 trials, rejection-sampled continuum supports |
| `cwf_resonator_scaling.py` | ~1–2 h | 4 sizes × 9 loads × 60 instances, `N` up to 2048 |
| `cwf_resonator_halting.py` | ~30–60 min | load sweep at 50 instances/point |
| `cwf_resonator_tcap.py` | ~1 h | iteration cap swept to `T = 1920` |

`cwf_resonator_thetahi.py` (the temperature sweep that carries the θ_c result) runs in
about two minutes at n = 500.

## Timings in the manuscript are illustrative, not portable

Several scripts report wall-clock numbers. Those depend on CPU, BLAS build, thread count
and machine load, and we have seen them move by orders of magnitude under parallel load
while every mathematical output stayed identical. Treat them as one machine's
illustration of an asymptotic ordering, never as a benchmark. The operation counts in the
manuscript's resource table are the portable statement.

## Scripts by cluster

**Phase space and uncertainty** — `cwf_fpe_uncertainty`, `cwf_fpe_phasespace`,
`cwf_fpe_localcell`, `cwf_fpe_weyl`, `cwf_fpe_models`

**Coherence and its diagnostics** — `cwf_fpe_wigner`, `cwf_fpe_wigner_convention`,
`cwf_fpe_stats`, `cwf_fpe_quantize`, `cwf_fpe_interference`

**The sketch** — `cwf_fpe_variance`, `cwf_fpe_background`, `cwf_fpe_ams`,
`cwf_fpe_complexity`, `cwf_fpe_highdim`, `cwf_fpe_beyond_capacity`,
`cwf_gaphamming_embedding`, `cwf_gaphamming_calibration`, `cwf_fpe_bispectrum`

**Operations** — `cwf_fpe_squeezing`, `cwf_fpe_fovea_task`, `cwf_fpe_invariant`,
`cwf_fpe_chirp`, `cwf_fpe_trajectory`

**Dynamics** — `cwf_resonator_halting`, `cwf_resonator_scaling`, `cwf_resonator_thetahi`,
`cwf_resonator_tcap`

**Hardware and connections** — `cwf_fpe_hardware`, `cwf_fpe_rope`, `cwf_rope_fovea`,
`cwf_fpe_ssm`

## Two estimator details that matter more than they look

Recorded here because getting either wrong moves a headline number more than the physics
does, and both were wrong in an earlier version:

1. **`cwf_fpe_hardware.py`** averages the **complex** correlation over noise draws and
   takes the magnitude once, at the end. Taking `|·|` inside the loop estimates `E|X|`
   rather than `|E X|`, which carries a Rice floor `sqrt(pi/4N)` that is *flat in the
   number of draws* — averaging more does not remove it.
2. **`cwf_fpe_fovea_task.py`** reports the foveation gain under three different matched
   resources, because the answer depends on which one a device fixes: 2.2× at a fixed
   codebook, 1.8× at fixed RMS effective slope, 0.86× at fixed maximum effective slope.
   A Gaussian standard deviation is not a hard aperture.
