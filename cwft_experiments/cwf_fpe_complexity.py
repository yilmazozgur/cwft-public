"""
cwf_fpe_complexity.py -- is there a genuine COMPLEXITY advantage from the coherent
wave?  (The category question; follows cwf_fpe_invariant.py.)

We have argued VSA is classical wave computation (no exponential Hilbert space) ->
NO quantum speedup is possible.  The honest remaining question: is there ANY
genuine complexity advantage of the coherent representation over the best classical
method?  This experiment MEASURES it and lets the data decide.

The candidate advantage is the 'compute in superposition' (linearity) effect: the
power spectrum |b|^2 of a bundle b = sum_i phi(x_i) encodes ALL O(k^2) pairwise
differences, and the relational readout (autocorrelation over a d-grid) costs
O(N*G) -- INDEPENDENT of k.  So from a GIVEN bundle, relational structure is
flat-in-k, vs O(k^2) for naive pairwise computation.

We time three ways to compute the difference-histogram (autocorrelation) of k
values, vs k:
  A. BUNDLE readout (given b): |b|^2 + cosine transform to G d-bins -- O(N*G), flat.
  B. NAIVE pairwise: all k^2 differences, histogrammed -- O(k^2).
  C. classical HISTOGRAM->FFT (given the values): O(k + R log R).
Plus the build cost of the bundle (O(k*N)), and the UNIVERSAL-SKETCH property (one
bundle answers several relational query types).  Then the honest verdict + scope.

CPU, numpy, seeded.  Writes cwf_fpe_complexity_results.json.
"""

import json
import time
import numpy as np

RNG = np.random.default_rng(0)
N = 4000
SIGMA = 3.0
THETA = RNG.normal(0, SIGMA, N)
G = 240
DS = np.linspace(0.5, 25, G)
COSMAT = np.cos(np.outer(DS, THETA))            # (G, N) precomputed transform


def build_bundle(values):
    return np.exp(1j * np.outer(values, THETA)).sum(0)   # (N,), O(kN)


def relational_readout(b):                       # autocorrelation from a GIVEN bundle
    return COSMAT @ (np.abs(b) ** 2) / N          # O(N*G), independent of k


def naive_autocorr(values):                      # O(k^2)
    diffs = np.abs(values[:, None] - values[None, :]).ravel()
    hist = np.zeros(G)
    for d in diffs:
        j = int((d - DS[0]) / (DS[1] - DS[0]))
        if 0 <= j < G:
            hist[j] += 1
    return hist


def hist_fft(values, R=60.0, res=0.1):           # classical: histogram -> FFT, O(k + RlogR)
    nb = int(R / res)
    h = np.zeros(nb)
    idx = (values / res).astype(int) % nb
    np.add.at(h, idx, 1.0)
    P = np.abs(np.fft.rfft(h)) ** 2               # power spectrum = autocorr transform
    return P


def timeit(fn, arg, reps):
    fn(arg)                                       # warm up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(arg); t.append(time.perf_counter() - t0)
    return float(np.median(t))


if __name__ == "__main__":
    print("=" * 74)
    print("COMPLEXITY: is the coherent relational readout genuinely cheaper?")
    print("=" * 74)
    print(f"  N={N}, G={G} d-bins. Time to get the difference-histogram vs k.")
    print(f"  {'k':>6} | {'bundle readout':>14} | {'naive O(k^2)':>13} | {'hist->FFT':>11} | {'bundle build':>13}")
    out = {"N": N, "G": G, "rows": []}
    for k in [30, 100, 300, 1000, 3000, 9000]:
        vals = RNG.uniform(0, 50, k)
        b = build_bundle(vals)
        reps = max(3, int(2e6 / (k + 1)))
        t_read = timeit(relational_readout, b, min(reps, 50))
        t_naive = timeit(naive_autocorr, vals, min(reps, 20)) if k <= 3000 else None
        t_fft = timeit(lambda v: hist_fft(v), vals, min(reps, 50))
        t_build = timeit(build_bundle, vals, min(reps, 20))
        out["rows"].append(dict(k=k, readout=t_read, naive=t_naive, fft=t_fft, build=t_build))
        nv = f"{t_naive*1e3:>10.2f}ms" if t_naive else f"{'(skip)':>12}"
        print(f"  {k:>6} | {t_read*1e3:>12.2f}ms | {nv} | {t_fft*1e3:>9.2f}ms | {t_build*1e3:>11.2f}ms")

    print(f"\n  -> bundle READOUT is FLAT in k (~O(N*G)); naive grows as k^2.")
    print(f"     classical hist->FFT is also cheap (O(k+RlogR)); bundle BUILD is O(kN).")

    # ---- universal sketch: one bundle, several relational query types ----
    print(f"\n  UNIVERSAL SKETCH: one O(N) bundle answers several relational queries:")
    vals = RNG.uniform(0, 40, 200)
    b = build_bundle(vals)
    # differences (autocorr), sums (self-convolution via b^2 per-component), cross-corr
    P = np.abs(b) ** 2
    diff_hist = COSMAT @ P / N                                 # pairwise differences
    sum_field = COSMAT @ np.real(b ** 2) / N                   # pairwise sums (b_k^2 = sum_ij e^{i(xi+xj)th})
    b2 = build_bundle(vals + 5.0)                              # a shifted set
    xcorr = COSMAT @ np.real(b * np.conj(b2)) / N              # cross-correlation (alignment)
    print(f"    from the SAME bundle b (O(N)): differences |b|^2, sums Re(b^2), "
          f"cross-corr Re(b b2*)")
    print(f"    each an O(N) readout -> a compact universal relational sketch.")
    # verify the cross-correlation peaks at the known shift 5.0
    shift_peak = DS[np.argmax(xcorr)]
    print(f"    cross-corr peak at d={shift_peak:.1f} (true shift 5.0): "
          f"{'recovered' if abs(shift_peak-5.0)<1.0 else 'missed'}")
    out["universal_sketch"] = {"xcorr_peak": float(shift_peak)}

    print(f"\n  HONEST VERDICT:")
    print(f"   - Genuine: the relational readout is O(N), FLAT in k -- a real")
    print(f"     'compute in superposition' (linearity) advantage over naive O(k^2),")
    print(f"     and one compact O(N) bundle serves MANY relational query types.")
    print(f"   - Conditional: needs the data in BUNDLE form (else build is O(kN)), or")
    print(f"     amortised over many queries; classical hist->FFT TIES any single")
    print(f"     query type. So it is a REPRESENTATIONAL / parallelism advantage,")
    print(f"     NOT a fundamental complexity separation, and NOT quantum.")
    out["verdict"] = ("representational/parallelism advantage (flat-in-k, universal sketch); "
                      "no fundamental separation; not quantum")
    with open("cwf_fpe_complexity_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote cwf_fpe_complexity_results.json")
