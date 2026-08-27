"""Quick diagnostic: trace D.max(), D.min(), D.mean() over time for one
seed of the power-law substrate. If max never crosses theta, the substrate
is saturated; if max grows but min lags, full-saturation criterion is too
strict and we should use a different t* definition."""
import numpy as np
from cwf_a2_phase1_scrambling import make_kernel, step_pl, M, RHO, C

for alpha in [6.0, 2.0, 0.5]:
    for N in [21, 41]:
        K = make_kernel(N, alpha)
        K_fft = np.fft.rfft2(K)
        no_leak = np.zeros((N, N))
        out_shape = (N, N)
        rng = np.random.default_rng(42)
        H = 0.1 * rng.standard_normal((N, N, M))
        for _ in range(200):
            H = step_pl(H, no_leak, K_fft, out_shape)
        # post-warmup statistics
        h_rms = np.sqrt((H**2).mean())
        h_sat = (np.abs(H) > 0.99).mean()
        print(f"\nalpha={alpha}  N={N}  warmup: H rms={h_rms:.3f}, "
              f"fraction saturated (|h|>0.99)={h_sat:.3f}")
        # perturb
        c = N // 2
        Hp = H.copy()
        Hp[c, c, 0] += 1e-7
        ref, per = H, Hp
        print("  t  D.min     D.max     D.mean    n_above_theta")
        for t in range(1, 60):
            ref = step_pl(ref, no_leak, K_fft, out_shape)
            per = step_pl(per, no_leak, K_fft, out_shape)
            D = np.linalg.norm(per - ref, axis=2)
            if t in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]:
                n_ok = int((D > 1e-4).sum())
                print(f"  {t:3d}  {D.min():.2e}  {D.max():.2e}  {D.mean():.2e}  {n_ok}/{N*N}")
