"""
T2.5 -- THE BOTH-AXES SUBSTRATE: one finite object carrying BOTH a holographic encoding
geometry (RT areas + entanglement-wedge reconstruction) AND computational irreducibility,
with the two HORIZONS coinciding -- the literal "gravity = geometry of irreducible
ignorance" at the encoding/\\irreducible edge.

Where this sits. T2.1 built the encoding face (gravity /\\ phase via magic). t24 showed a
min-cut SELECTOR can inherit a CA's irreducibility. A3d made the geometry dynamic
(back-reaction). None put a genuine holographic CODE (RT + entanglement-wedge reconstruction)
and computational irreducibility on the SAME object with their horizons compared. This does.

The construction (the unifying identification: radial bulk depth = CA computation time --
the standard "bulk depth = RG scale / computation time" of holography). A weighted disk
(A3d's geometry): boundary shell = the screen (holds a CA seed); each inward shell = one CA
step; the centre is the deepest bulk. So a bulk node at shell s holds the value the substrate
computes after s steps, and RECONSTRUCTING it from the screen means RUNNING the CA s steps.
 * ENCODING axis: S(A) = min-cut(A : Abar) (RT area, eta_c=1); the entanglement wedge of A
   is the source side; entanglement-wedge reconstruction = "a bulk node in the wedge is
   recoverable from A". Geometry present and measured. (A3d machinery.)
 * IRREDUCIBILITY axis: the reconstruction map (screen seed) -> (bulk value at shell s) is
   f_{rule,s}; its ANF degree over GF(2) (t24 machinery) is how hard the reconstruction is.
   Universal Rule 110 -> degree climbs with depth (IRREDUCIBLE: must run the substrate);
   additive Rule 90 -> degree 1 for all depth (REDUCIBLE: closed-form decode).

Four legs:
  LEG A  ENCODING geometry present (RT areas + nested entanglement wedges by depth).
  LEG B  IRREDUCIBILITY of reconstruction vs depth, per rule (ANF degree).
  LEG C  THE COINCIDENCE (headline).  At observer budget kappa (affordable reconstruction
         degree), a bulk node is DARK if its reconstruction degree > kappa. For the UNIVERSAL
         substrate the dark nodes form a CORE at large depth, bounded by a geometric horizon
         shell s* (a circle in the disk) -- the same object is an encoding geometry that
         HIDES an irreducible-ignorance core behind a geometric horizon. For the ADDITIVE
         substrate dark fraction = 0 (encoded AND fully reconstructable -- a normal finite
         code). Clean split: encoding-with-ignorance vs encoding-without.
  LEG D  HONEST SCOPE.  finite => IRREDUCIBLE, not UNDECIDABLE (asymptotic only). The PHASE
         is NOT dragged along (irreducibility over a definite computed ground does not force
         contextual interference, Ch6 impossibility) -- the full TRIPLE point still needs
         self-reference. Back-reaction (dynamic geometry) is the separate A3d leg, compatible.

CPU; numpy + networkx. Seeded. Atomic write. Reuses A3d (disk/min-cut) + t24 (CA/ANF).
"""
import json, os
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
BIG = 1e6


# ---- t24 CA + ANF machinery (reconstruction-hardness measure) ----
def all_seeds(n):
    xs = np.arange(1 << n, dtype=np.uint32)
    return ((xs[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.int8)


def eca_step_batch(states, rule):
    left = np.roll(states, 1, axis=1); right = np.roll(states, -1, axis=1)
    idx = ((left << 2) | (states << 1) | right).astype(np.int64)
    return ((np.int64(rule) >> idx) & 1).astype(np.int8)


def truth_table(rule, n, T, c):
    s = all_seeds(n)
    for _ in range(T):
        s = eca_step_batch(s, rule)
    return s[:, c].astype(np.uint8)


def anf_degree(tt, n):
    a = tt.copy().astype(np.uint8); N = 1 << n; idx = np.arange(N)
    for i in range(n):
        bit = 1 << i; sel = (idx & bit) > 0
        a[sel] ^= a[idx[sel] ^ bit]
    masks = np.nonzero(a)[0]
    return 0 if masks.size == 0 else int(max(int(m).bit_count() for m in masks))


# ---- A3d disk geometry (RT = min-cut), with radial shells = CA depth ----
def build_disk(R=5.0):
    a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, np.sqrt(3) / 2]); pos = {}
    rng = range(-int(2 * R), int(2 * R) + 1)
    for i in rng:
        for j in rng:
            p = i * a1 + j * a2
            if np.hypot(*p) <= R + 1e-9:
                pos[(i, j)] = p
    G = nx.Graph()
    for node, p in pos.items():
        G.add_node(node, pos=p, r=float(np.hypot(*p)))
    for (i, j) in list(pos):
        for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]:
            m = (i + di, j + dj)
            if m in pos and not G.has_edge((i, j), m):
                G.add_edge((i, j), m, cap=1.0)
    for node in G.nodes:
        G.nodes[node]["boundary"] = (G.degree(node) < 6)
    bnodes = [n for n in G.nodes if G.nodes[n]["boundary"]]
    bnodes.sort(key=lambda n: np.arctan2(G.nodes[n]["pos"][1], G.nodes[n]["pos"][0]))
    bulk = [n for n in G.nodes if not G.nodes[n]["boundary"]]
    rmax = max(G.nodes[n]["r"] for n in G.nodes)
    for n in G.nodes:                       # shell = depth from boundary = CA time
        G.nodes[n]["shell"] = int(round(rmax - G.nodes[n]["r"]))
    return G, bnodes, bulk, rmax


def arc(bnodes, frac, start=0):
    k = max(1, int(round(frac * len(bnodes))))
    A = [bnodes[(start + t) % len(bnodes)] for t in range(k)]
    return A, [n for n in bnodes if n not in set(A)]


def mincut(G, A, B):
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, capacity=d["cap"])
    for a in A:
        H.add_edge("SRC", a, capacity=BIG)
    for b in B:
        H.add_edge(b, "SNK", capacity=BIG)
    val, (reach, _) = nx.minimum_cut(H, "SRC", "SNK", capacity="capacity")
    return float(val), set(reach)


def main():
    print("T2.5 -- the BOTH-AXES SUBSTRATE: holographic encoding /\\ computational "
          "irreducibility on one object, horizons compared\n")
    checks = {}
    G, bnodes, bulk, rmax = build_disk(R=5.0)
    n_ca, c_ca = 12, 6
    kappa = 4                       # observer budget: affordable reconstruction ANF degree
    shells = sorted({G.nodes[v]["shell"] for v in bulk})
    print(f"disk: {G.number_of_nodes()} nodes ({len(bnodes)} boundary, {len(bulk)} bulk), "
          f"shells (depth=CA time) {shells}; observer budget kappa={kappa}.\n")

    # ---- LEG A: ENCODING geometry -- RT areas + nested wedges by depth ----
    regions = {f"f{int(100*fr)}": arc(bnodes, fr) for fr in [0.25, 0.5, 0.75]}
    rt = {}
    for name, (A, B) in regions.items():
        val, wedge = mincut(G, A, B)
        depth_in_wedge = max((G.nodes[v]["shell"] for v in wedge if v in set(bulk)), default=0)
        rt[name] = dict(area_S=val, wedge_bulk=len([v for v in wedge if v in set(bulk)]),
                        max_depth_reached=int(depth_in_wedge))
    # encoding present: bigger boundary region -> deeper wedge (reaches the core)
    depths = [rt[n]["max_depth_reached"] for n in ["f25", "f50", "f75"]]
    encoding_ok = bool(rt["f25"]["area_S"] > 0 and depths[2] >= depths[0]
                       and rt["f75"]["wedge_bulk"] > rt["f25"]["wedge_bulk"])
    checks["encoding geometry: RT areas + wedges deepen with region size"] = encoding_ok
    print("(A) ENCODING geometry (RT = min-cut):")
    for name in ["f25", "f50", "f75"]:
        print(f"    region {name}: S(A)={rt[name]['area_S']:.0f}, wedge holds "
              f"{rt[name]['wedge_bulk']} bulk nodes, reaches depth {rt[name]['max_depth_reached']}")
    print(f"    -> entanglement wedges deepen with region size (bigger A reconstructs deeper "
          f"bulk): {encoding_ok}")

    # ---- LEG B: IRREDUCIBILITY of reconstruction vs depth, per rule ----
    rules = {90: "additive (reducible)", 110: "universal (irreducible)"}
    recon_deg = {}     # rule -> {shell: ANF degree of reconstruction map at that depth}
    for rule in rules:
        recon_deg[rule] = {s: anf_degree(truth_table(rule, n_ca, max(s, 1), c_ca), n_ca)
                           for s in shells}
    irreducible_split = bool(max(recon_deg[110].values()) >= 6 and max(recon_deg[90].values()) <= 1)
    checks["reconstruction irreducible for universal, reducible for additive"] = irreducible_split
    print("\n(B) IRREDUCIBILITY of reconstruction (ANF degree of screen->bulk map vs depth):")
    for rule, lab in rules.items():
        print(f"    Rule {rule:>3} ({lab}): degree by shell "
              f"{ {s: recon_deg[rule][s] for s in shells} }")
    print(f"    -> universal reconstruction climbs (must run the substrate); additive stays "
          f"affine (closed-form decode): {irreducible_split}")

    # ---- LEG C: THE COINCIDENCE -- dark (irreducible) core behind a geometric horizon ----
    coincidence = {}
    for rule in rules:
        # a bulk node is DARK if reconstructing it costs more than the budget kappa
        dark = {v: (recon_deg[rule][G.nodes[v]["shell"]] > kappa) for v in bulk}
        dark_nodes = [v for v in bulk if dark[v]]
        dark_frac = len(dark_nodes) / len(bulk)
        # is the dark set a geometric CORE? (all dark nodes deeper than all bright nodes)
        dark_shells = sorted({G.nodes[v]["shell"] for v in dark_nodes})
        bright_shells = sorted({G.nodes[v]["shell"] for v in bulk if not dark[v]})
        horizon = min(dark_shells) if dark_shells else None        # the horizon shell s*
        is_core = bool(dark_nodes) and (not bright_shells or min(dark_shells) > max(bright_shells))
        # dark-fraction profile by shell
        prof = {s: float(np.mean([dark[v] for v in bulk if G.nodes[v]["shell"] == s]))
                for s in shells}
        coincidence[rule] = dict(dark_fraction=dark_frac, horizon_shell=horizon,
                                 dark_is_geometric_core=is_core, profile=prof,
                                 n_dark=len(dark_nodes))
    univ = coincidence[110]; addi = coincidence[90]
    # headline: universal hides a geometric dark core; additive is transparent
    headline = bool(univ["dark_fraction"] > 0 and univ["dark_is_geometric_core"]
                    and addi["dark_fraction"] == 0)
    checks["universal: irreducible-ignorance core behind a geometric horizon"] = bool(
        univ["dark_fraction"] > 0 and univ["dark_is_geometric_core"])
    checks["additive control: encoding transparent (dark fraction 0)"] = bool(addi["dark_fraction"] == 0)
    print("\n(C) THE COINCIDENCE -- encoding geometry /\\ irreducible ignorance on ONE object:")
    print(f"    UNIVERSAL (Rule 110): dark fraction {univ['dark_fraction']:.2f}, horizon at "
          f"shell s*={univ['horizon_shell']} (a geometric circle), dark set is a radial CORE: "
          f"{univ['dark_is_geometric_core']}; dark-by-shell {univ['profile']}")
    print(f"    ADDITIVE  (Rule  90): dark fraction {addi['dark_fraction']:.2f} -- encoded AND "
          f"fully reconstructable (a normal finite code).")
    print(f"    => the SAME holographic geometry hides an irreducible-ignorance core behind a "
          f"geometric horizon for the universal substrate, and is transparent for the additive "
          f"one: {headline}. The dark core is encoded (in wedges) yet irreducible -- 'gravity = "
          f"geometry of irreducible ignorance' as a measured statement on a finite object.")

    all_ok = all(checks.values())
    false_keys = [k for k, v in checks.items() if not v]
    if false_keys:
        print(f"\n  NOTE -- checks reading False (report honestly): {false_keys}")

    verdict = (
        "BOTH-AXES SUBSTRATE REALIZED at the encoding/\\irreducible edge, on one finite object. "
        "A holographic disk with radial depth = CA computation time carries (A) a genuine "
        "ENCODING geometry -- RT areas S(A)=min-cut and entanglement wedges that deepen with "
        "region size (bigger boundary reconstructs deeper bulk) -- and (B) computational "
        "IRREDUCIBILITY of reconstruction: the screen->bulk map's ANF degree climbs with depth "
        f"for universal Rule 110 (to {max(recon_deg[110].values())}, must run the substrate) "
        f"while additive Rule 90 stays affine (degree {max(recon_deg[90].values())}, closed-form "
        "decode). (C) The HEADLINE: at observer budget kappa, the universal substrate's geometry "
        f"hides an irreducible-ignorance CORE (dark fraction {univ['dark_fraction']:.2f}) behind a "
        f"GEOMETRIC horizon (shell s*={univ['horizon_shell']}, a radial circle), the dark set a "
        "clean core; the additive control is fully transparent (dark fraction 0 -- a normal "
        "finite code). So the SAME holographic geometry either hides or does not hide an "
        "irreducible-ignorance core, and when it does the horizon is GEOMETRIC: 'gravity = the "
        "geometry of irreducible ignorance' as a measured statement on a finite object -- the "
        "encoding and irreducibility axes coincide. (The radial SHAPE of the horizon follows "
        "from the standard depth=computation-time identification, not as an independent surprise; "
        "the substantive content is that an irreducible substrate HIDES such a core while a "
        "reducible one does NOT -- the additive control's dark fraction 0, on the identical "
        "geometry, is what makes the coupling non-vacuous.) HONEST SCOPE: (D) finite => IRREDUCIBLE "
        "(ANF/no-shortcut), NOT undecidable (asymptotic only -- the dark core is decidable by "
        "running the substrate; genuine undecidability needs the non-halting limit). The PHASE is "
        "NOT dragged along: irreducibility over a definite computed ground does not force "
        "contextual interference (the Ch6 Bell-Kochen-Specker impossibility), so the full TRIPLE "
        "point (gravity /\\ irreducibility /\\ phase) still requires self-reference and stays "
        "open. Back-reaction (dynamic geometry) is the separate, compatible A3d leg. "
        "Reconstruction-irreducibility is ANF algebraic complexity (a strong no-low-degree-"
        "shortcut proxy); unconditional time-hardness of Rule 110 is a separate open question."
    ) if (encoding_ok and irreducible_split and headline) else \
        "INCOMPLETE: a core leg failed -- inspect."
    print(f"\nall checks pass: {all_ok}\n\n{verdict}")

    out = os.path.join(HERE, "results.json")
    try:
        R = json.load(open(out))
    except Exception:
        R = {}
    R["T25_both_axes"] = dict(
        checks={k: bool(v) for k, v in checks.items()}, all_verified=bool(all_ok),
        n_nodes=G.number_of_nodes(), n_bulk=len(bulk), shells=shells, kappa=kappa,
        encoding_rt={k: v for k, v in rt.items()},
        recon_degree={int(r): {int(s): recon_deg[r][s] for s in shells} for r in rules},
        coincidence={int(r): coincidence[r] for r in rules}, verdict=verdict,
        note=("T2.5 both-axes substrate. A holographic disk (RT=min-cut, A3d geometry) with "
              "radial depth = CA computation time (t24 reconstruction-hardness). ENCODING: RT "
              "areas + entanglement wedges deepen with region size. IRREDUCIBILITY: screen->bulk "
              "reconstruction ANF degree climbs with depth for universal Rule 110, stays affine "
              "for additive Rule 90. HEADLINE: at budget kappa the universal geometry hides an "
              "irreducible-ignorance CORE behind a GEOMETRIC horizon shell s* (encoded yet "
              "irreducible = geometry of irreducible ignorance on a finite object); additive "
              "fully transparent (dark fraction 0). SCOPE: finite=>irreducible not undecidable "
              "(asymptotic); phase NOT dragged (definite ground, Ch6 impossibility) so the triple "
              "point still needs self-reference; back-reaction = A3d (compatible); ANF degree is "
              "a strong shortcut proxy, unconditional Rule-110 time-hardness open. Atomic write."))
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(R, f, indent=2)
    os.replace(tmp, out)
    plot(G, bulk, recon_deg, coincidence, kappa, shells)
    print("\nWrote results.json key: T25_both_axes (atomic)")


def plot(G, bulk, recon_deg, coincidence, kappa, shells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 5.2))
    pos = {n: G.nodes[n]["pos"] for n in G.nodes}
    # (a) disk: universal substrate -- bulk nodes coloured dark(red)/bright(green), horizon circle
    nx.draw_networkx_edges(G, pos, ax=ax0, edge_color="0.88", width=0.5)
    dark = {v: (recon_deg[110][G.nodes[v]["shell"]] > kappa) for v in bulk}
    bx = [pos[v][0] for v in bulk if not dark[v]]; by = [pos[v][1] for v in bulk if not dark[v]]
    dx = [pos[v][0] for v in bulk if dark[v]]; dy = [pos[v][1] for v in bulk if dark[v]]
    ax0.scatter(bx, by, c="C2", s=30, label="reconstructable (bright)", zorder=3)
    ax0.scatter(dx, dy, c="C3", s=30, label="irreducible (dark core)", zorder=3)
    bnd = [n for n in G.nodes if G.nodes[n]["boundary"]]
    ax0.scatter([pos[n][0] for n in bnd], [pos[n][1] for n in bnd], c="0.4", s=14,
                marker="s", label="screen (boundary)", zorder=2)
    s_star = coincidence[110]["horizon_shell"]
    if s_star is not None:
        rmax = max(G.nodes[n]["r"] for n in G.nodes)
        r_h = rmax - (s_star - 0.5)
        th = np.linspace(0, 2 * np.pi, 200)
        ax0.plot(r_h * np.cos(th), r_h * np.sin(th), "k--", lw=1.4,
                 label=f"reconstruction horizon $s^*={s_star}$")
    ax0.set_title("(a) universal substrate (Rule 110):\nencoding geometry hides an irreducible "
                  "core\nbehind a geometric horizon")
    ax0.legend(fontsize=7.5, loc="lower left"); ax0.set_aspect("equal"); ax0.axis("off")
    # (b) dark-fraction-by-depth profiles, universal vs additive
    prof_u = coincidence[110]["profile"]; prof_a = coincidence[90]["profile"]
    ax1.plot(shells, [prof_u[s] for s in shells], "o-", color="C3", lw=2,
             label="universal (Rule 110)")
    ax1.plot(shells, [prof_a[s] for s in shells], "s--", color="C0", lw=2,
             label="additive (Rule 90)")
    ax1.set_xlabel("bulk depth = CA time (shell)")
    ax1.set_ylabel("dark (irreducible) fraction at the observer budget $\\kappa_{\\mathrm{obs}}$")
    ax1.set_title(f"(b) the same geometry: irreducible-ignorance core\n(universal) vs "
                  f"transparent (additive), $\\kappa_{{\\mathrm{{obs}}}}={kappa}$")
    ax1.set_ylim(-0.05, 1.05); ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
    fig.suptitle("T2.5: the both-axes substrate -- holographic encoding /\\ computational "
                 "irreducibility, with coinciding horizons", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pth = os.path.join(HERE, "fig_T25_both_axes.png")
    plt.savefig(pth, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Wrote {pth}")


if __name__ == "__main__":
    main()
