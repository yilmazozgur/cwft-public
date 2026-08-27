"""Find a parameter setting where the 2D substrate is genuinely chaotic
(divergence grows). Sweep (m, coupling, rho) on NN coupling first; we
already know the divergence collapses for the cwf_clausius2d setting
(m=10, coupling=0.05, rho=2.6)."""
import numpy as np

def step_nn(H, leak, W, P, rho, c):
    nb = (np.roll(H, 1, 0) + np.roll(H, -1, 0)
          + np.roll(H, 1, 1) + np.roll(H, -1, 1))
    pre = rho * (H @ W.T) + c * (nb @ P.T) - leak[..., None] * H
    return np.tanh(pre)

def lyapunov_test(m, coupling, rho, N=41, warmup=200, T=80, eps=1e-7, seeds=6):
    rng_init = np.random.default_rng(0)
    W = rng_init.standard_normal((m, m)); W /= max(abs(np.linalg.eigvals(W)))
    P = rng_init.standard_normal((m, m)); P /= max(abs(np.linalg.eigvals(P)))
    no_leak = np.zeros((N, N))
    growth = []
    saturated_fracs = []
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        H = 0.1 * rng.standard_normal((N, N, m))
        for _ in range(warmup):
            H = step_nn(H, no_leak, W, P, rho, coupling)
        saturated_fracs.append(float((np.abs(H) > 0.99).mean()))
        # perturb at center with random m-vector
        c0 = N // 2
        Hp = H.copy()
        Hp[c0, c0] += eps * rng.standard_normal(m)
        ref, per = H, Hp
        D_traj = []
        for t in range(T):
            ref = step_nn(ref, no_leak, W, P, rho, coupling)
            per = step_nn(per, no_leak, W, P, rho, coupling)
            D = np.linalg.norm(per - ref, axis=2)
            D_traj.append(float(np.linalg.norm(D)))  # global norm
        D_traj = np.array(D_traj)
        # estimate Lyapunov from early growth (or decay)
        # fit log(D) vs t in steps 5..min(T, 30)
        t_fit = np.arange(5, min(T, 30))
        slope = np.polyfit(t_fit, np.log(D_traj[t_fit] + 1e-30), 1)[0]
        growth.append(slope)
    return np.mean(growth), np.std(growth), np.mean(saturated_fracs)

# Sweep parameters
configs = [
    ("c=0.20_rho=2.6", 10, 0.20, 2.6),
    ("c=0.30_rho=2.6", 10, 0.30, 2.6),
    ("c=0.40_rho=2.6", 10, 0.40, 2.6),
    ("c=0.50_rho=2.6", 10, 0.50, 2.6),
    ("c=0.70_rho=2.6", 10, 0.70, 2.6),
    ("c=1.00_rho=2.6", 10, 1.00, 2.6),
    ("c=0.30_rho=2.0", 10, 0.30, 2.0),
    ("c=0.30_rho=3.0", 10, 0.30, 3.0),
    ("c=0.50_rho=3.0", 10, 0.50, 3.0),
    ("c=0.30_m=24", 24, 0.30, 2.6),
    ("c=0.50_m=24", 24, 0.50, 2.6),
]
print(f"  {'config':30s}  {'m':3s} {'c':5s} {'rho':4s}   {'lambda':8s} sat_frac")
for name, m, c, rho in configs:
    lam, lam_std, sat = lyapunov_test(m, c, rho)
    flag = "CHAOTIC" if lam > 0.005 else ("EDGE" if lam > -0.005 else "CONTRACTIVE")
    print(f"  {name:30s}  {m:3d} {c:.2f}  {rho:.1f}   "
          f"{lam:+.4f} +/- {lam_std:.4f}   {sat:.2f}   {flag}")
