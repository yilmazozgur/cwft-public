"""
cwf_primon_records.py  --  Classical records from the Bost-Connes SSB. (Option B.)

Reading (status B/S): the Bost-Connes phase transition of the primon gas at
beta=1 (theta_c=1) is the emergence of definite CLASSICAL RECORDS.
  * beta>1 (theta_c<1): broken symmetry -> a family of definite extremal KMS
    states = pointer states / records.  Test A showed these are PRODUCT states
    (zero entanglement): the records are classical and a partial read does not
    disturb the rest.
  * beta<=1 (theta_c>=1): a unique symmetric KMS state = record-free, scrambled.
  * beta=1: record formation (the classicality transition).

This script supports the reading with EXACT order parameters (no Bost-Connes
Galois formula is reconstructed; only the thermodynamic/condensation face):

  * Definiteness fraction p_1(beta) = 1/zeta(beta) = weight on the definite
    "vacuum" record.  >0 for beta>1, -> 0 at beta=1 (records carry finite
    weight only in the classical phase).  More generally p_m = m^{-beta}/zeta
    is the weight of the definite record m.
  * Effective number of dominant records  N_eff = zeta(beta)^2 / zeta(2 beta)
    (inverse participation).  O(1) deep in the classical phase, diverges at
    beta=1 (definiteness lost).
  * Truncated p_1(beta,N): shows the order parameter -> 0 with N for beta<=1
    (no equilibrium / no record) and -> 1/zeta(beta) for beta>1.

Honest scope: this is the CONDENSATION face of the SSB (exact, computed).  The
full Galois multiplicity of the broken-phase states is the cited Bost-Connes
theorem, not recomputed here.

CPU-only; mpmath + numpy.  Writes cwf_primon_records_results.json (new file).
"""

import json
import numpy as np
import mpmath as mp

mp.mp.dps = 30


def condensate_fraction(beta):
    """p_1 = weight on the definite vacuum record = 1/zeta(beta), beta>1."""
    return float(1 / mp.zeta(beta))


def n_eff_records(beta):
    """Effective number of dominant records = zeta(beta)^2 / zeta(2 beta)."""
    return float(mp.zeta(beta) ** 2 / mp.zeta(2 * beta))


def record_weights(beta, ms):
    Z = mp.zeta(beta)
    return {int(m): float(mp.mpf(m) ** (-beta) / Z) for m in ms}


def truncated_p1(beta, N):
    n = np.arange(1, N + 1, dtype=np.float64)
    return float(1.0 / np.sum(n ** (-beta)))


if __name__ == "__main__":
    print("=" * 70)
    print("CLASSICAL RECORDS FROM THE BOST-CONNES SSB (Option B)")
    print("  records exist for theta_c<1 (beta>1); dissolve at theta_c=1")
    print("=" * 70)

    betas = [3.0, 2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01]
    print(f"\n[classical phase beta>1]  exact order parameters")
    print(f"  {'beta':>6} {'theta_c':>8} {'p_1=1/zeta':>11} "
          f"{'N_eff':>9} {'p_2':>8} {'p_3':>8}")
    rows = []
    for be in betas:
        p1 = condensate_fraction(be)
        ne = n_eff_records(be)
        w = record_weights(be, [2, 3])
        rows.append(dict(beta=be, theta_c=1.0 / be, p1=p1, n_eff=ne,
                         p2=w[2], p3=w[3]))
        print(f"  {be:>6.2f} {1.0/be:>8.3f} {p1:>11.5f} {ne:>9.3f} "
              f"{w[2]:>8.5f} {w[3]:>8.5f}")
    print("  -> p_1>0 and N_eff=O(1): definite records carry finite weight.")
    print("     As beta->1+, p_1->0 and N_eff->inf: definiteness dissolves.")

    print(f"\n[order parameter vanishing]  truncated p_1(beta,N) across beta=1")
    Ns = [10**3, 10**5, 10**7]
    sweep = [0.6, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]
    print(f"  {'beta':>6} " + " ".join(f"N=1e{int(np.log10(N))}" for N in Ns)
          + "   phase")
    trunc = []
    for be in sweep:
        ps = [truncated_p1(be, N) for N in Ns]
        phase = "classical (record)" if be > 1.0 else (
            "transition" if be == 1.0 else "scrambled (no record)")
        trunc.append(dict(beta=be, Ns=Ns, p1=ps, phase=phase))
        print(f"  {be:>6.2f} " + " ".join(f"{p:>7.4f}" for p in ps)
              + f"   {phase}")
    print("  -> for beta<=1, p_1 -> 0 as N grows (no definite record);")
    print("     for beta>1, p_1 converges to 1/zeta(beta)>0 (record persists).")

    out = {"classical_phase": rows, "order_parameter_vanishing": trunc}
    with open("cwf_primon_records_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_primon_records_results.json")
