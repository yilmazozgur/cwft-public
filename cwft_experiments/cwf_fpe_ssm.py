#!/usr/bin/env python3
"""THE SSM / LINEAR-ATTENTION STATE IS A RUNNING ECF SKETCH (Move A / Bridge 2).

A diagonal state-space model with unit-modulus eigenvalues e^{i omega_k} accumulates
a stream of scalar tokens u_1..u_T into a state

    S_k = sum_t e^{i omega_k (T - t)} phi(u_t)_k ,   phi(u)_k = e^{i u theta_k},

which is exactly an FPE bundle of the token values (shifted by the position phase) --
i.e. the empirical characteristic function of the token multiset at frequencies theta.
Consequences, all tested here:
 (1) |S|^2 recovers the token-difference histogram (the relational readout), so an SSM
     state answers relational queries without unrolling the stream;
 (2) the recovery error follows the manuscript's k^2/N variance law (Prop.);
 (3) the number of distinguishable tokens tracks the VSA bundling capacity ~ N/(2 ln L).
This lands VSA capacity theory and SSM state-compression in one identification.
Seeded, CPU.  Writes cwf_fpe_ssm_results.json.
"""
import json

import numpy as np

N = 2000               # SSM state width (= VSA dimension)
SIGMA = 2.0            # value-axis bandwidth
SIGMA_W = 2.0          # age-axis bandwidth (the position phase; MUST be nonzero)
R = 200.0
DRAWS = 200
T_JOINT = 12           # stream length for the joint (age, value) tests
UGRID = np.linspace(0, R, 4001)


def running_state(tokens, theta, omega):
    """Diagonal-SSM state after streaming `tokens` (position-phase included)."""
    T = len(tokens)
    pos = np.arange(T)[::-1]                       # T-t
    S = np.zeros(N, dtype=complex)
    for t, u in enumerate(tokens):
        S += np.exp(1j * omega * pos[t]) * np.exp(1j * u * theta)
    return S


def value_at_age(S, a, theta, omega):
    """Decode the stored value at age a: the joint (age, value) query."""
    ph = a * omega[None, :] + np.outer(UGRID, theta)
    prof = np.abs(np.exp(-1j * ph) @ S) / N
    return float(UGRID[int(np.argmax(prof))])


def main():
    rng = np.random.default_rng(3)
    out = {"params": dict(N=N, sigma=SIGMA, sigma_w=SIGMA_W, R=R, T_joint=T_JOINT)}

    theta = SIGMA * rng.standard_normal(N)         # value axis
    omega = SIGMA_W * rng.standard_normal(N)       # age axis -- ON

    # (1) THE JOINT (age, value) QUERY -- the claim the identification actually makes.
    # The state is the running ECF of the PAIRS (T-t, u_t), so it should answer
    # "which value arrived a steps ago?" without unrolling the stream.  The control
    # is the same query on a state built with omega = 0, which carries no age
    # information and must therefore fail.
    tokens = rng.uniform(0, R, T_JOINT)
    ages = np.arange(T_JOINT)                      # token t has age T-1-t
    truth = tokens[::-1]                           # truth[a] = value at age a
    S_joint = running_state(tokens, theta, omega)
    S_flat = running_state(tokens, theta, np.zeros(N))

    rec_joint = np.array([value_at_age(S_joint, a, theta, omega) for a in ages])
    rec_flat = np.array([value_at_age(S_flat, a, theta, np.zeros(N)) for a in ages])
    tol = 0.5
    acc_joint = float(np.mean(np.abs(rec_joint - truth) < tol))
    acc_flat = float(np.mean(np.abs(rec_flat - truth) < tol))
    rmse_joint = float(np.sqrt(np.mean((rec_joint - truth) ** 2)))
    out["joint_query"] = dict(T=T_JOINT, tol=tol, acc_joint=acc_joint,
                              acc_omega_zero=acc_flat, rmse_joint=rmse_joint,
                              truth=[float(v) for v in truth],
                              recovered=[float(v) for v in rec_joint])
    print(f"(1) joint (age,value) query, T={T_JOINT} tokens:")
    print(f"    omega ON : {acc_joint*100:5.1f}% of ages decoded within {tol} "
          f"(RMSE {rmse_joint:.4f})")
    print(f"    omega = 0: {acc_flat*100:5.1f}%  <- control: no age axis, no answer")

    # (1b) the omega = 0 SLICE recovers the plain value-difference histogram.
    # This is the marginal of the joint object, not the joint object itself.
    m, Delta = 6, 5.0
    base = np.sort(rng.uniform(0, R, m))
    toks2 = np.concatenate([base, base + Delta])
    S0 = running_state(toks2, theta, np.zeros(N))
    ds = np.linspace(2, 12, 500)
    score = (np.cos(np.outer(ds, theta)) @ (np.abs(S0) ** 2)) / N
    found = bool(abs(ds[np.argmax(score)] - Delta) < 0.3)
    out["readout_omega0_slice"] = dict(planted_Delta=Delta,
                                       recovered=float(ds[np.argmax(score)]),
                                       found=found)
    print(f"(1b) omega=0 slice -> value-difference histogram: planted {Delta}, "
          f"recovered {ds[np.argmax(score)]:.2f}  found={found}")

    # (2) variance law: SD of score at a null lag vs k / sqrt(N), carrier codebook
    OMEGA0 = 8.0
    rows = []
    for k in [50, 200, 800]:
        vals = np.sort(rng.uniform(0, R, k))
        sc = []
        for _ in range(DRAWS):
            th = OMEGA0 + SIGMA * rng.standard_normal(N)
            S = np.exp(1j * np.outer(vals, th)).sum(0)
            sc.append(float((np.cos(1.7 * th) @ (np.abs(S) ** 2)) / N))
        sd = float(np.std(sc))
        # The k^2/N law assumes well-separated values.  Flag when the mean spacing
        # R/k drops below one kernel half-width, i.e. when we have left the sparse
        # regime the proposition is stated for.
        hwhm = float(np.sqrt(2 * np.log(2)) / SIGMA)
        sparse = bool((R / k) > hwhm)
        rows.append(dict(k=k, sd=sd, pred=k / np.sqrt(N), ratio=sd / (k / np.sqrt(N)),
                         mean_spacing=float(R / k), kernel_hwhm=hwhm, sparse=sparse))
        print(f"(2) k={k:>4}: SD={sd:8.3f}  k/sqrt(N)={k/np.sqrt(N):7.3f}  "
              f"ratio={sd/(k/np.sqrt(N)):.2f}   spacing={R/k:.2f} vs HWHM={hwhm:.2f}"
              f"  {'sparse' if sparse else '<- DENSE, outside the law'}")
    out["variance"] = rows

    # (3) distinguishable-token capacity ~ N/(2 ln L)
    caps = []
    for Nk in [500, 1000, 2000]:
        th = 4.0 * rng.standard_normal(Nk)
        L = 4 * Nk
        xs = np.arange(L) * 3.0
        Phi = np.exp(1j * np.outer(xs, th))
        kmax = 0
        for kk in range(2, L, max(2, Nk // 16)):
            recs = []
            for _ in range(8):
                Ssel = rng.choice(L, kk, replace=False)
                S = Phi[Ssel].sum(0)
                sim = np.real(Phi @ S.conj()) / Nk
                pred = sim > 0.5
                tru = np.zeros(L, bool); tru[Ssel] = True
                recs.append((pred & tru).sum() / kk)
            if np.mean(recs) >= 0.95:
                kmax = kk
            else:
                break
        caps.append(dict(N=Nk, kmax=kmax, over_N=kmax / Nk,
                         pred=Nk / (2 * np.log(4 * Nk))))
        print(f"(3) N={Nk:>4}: distinguishable tokens k_max={kmax}  "
              f"(k/N={kmax/Nk:.3f}, N/(2lnL)={Nk/(2*np.log(4*Nk))/Nk:.3f}N)")
    out["capacity"] = caps

    with open("cwf_fpe_ssm_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote cwf_fpe_ssm_results.json")


if __name__ == "__main__":
    main()
