#!/usr/bin/env python3
"""merge_bib.py -- bibliography merger for the TJP unified manuscript (BUILD_PLAN 1.4, gate G4).

Reads the per-source bibliographies extracted at Phase 1 (bib_A.tex, bib_B.tex, bib_C.tex),
dedupes (same-key exact; cross-key via curated ALIAS map; reports similarity candidates),
then emits bibliography_merged.tex ordered by first citation in the main tex.

Checks (exit 1 on error):
  - every \\cite key in the main tex resolves to a merged entry           [error]
  - dropped keys (DROP set) must not be cited                             [error]
  - entries never cited                                                   [warn list; --strict makes it an error]
  - cross-source same-key entries with materially different text          [warn: canonical-priority applied]

Usage: merge_bib.py [--strict] [--main cwft_unified.tex] [--supp cwft_supplement.tex]
Both tex files are scanned for cites (supplement optional); ordering follows the main tex, with
supplement-only cites appended after.
"""
import re, sys, json, argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Curated cross-key aliases: alias_key -> canonical_key (canonical must exist in some source).
ALIAS = {
    "abramsky2017": "abramskybarbosamansfield2017",  # B key == C key, same paper (PRL 119, 050504)
}
# Keys deleted in the merge (become internal cross-references, never cited in merged doc).
DROP = {"cwfselfref", "cwfgravity"}
# Canonical-source priority when the same key appears with different entry text (C newest: v6 DOI etc.)
# "X" = tools/bib_extra.tex, post-review primary-source additions (highest priority, no conflicts expected).
PRIORITY = ["X", "C", "B", "A"]

def parse_bib(path):
    text = path.read_text(encoding="utf-8")
    body = re.search(r"\\begin\{thebibliography\}\{[^}]*\}(.*)\\end\{thebibliography\}", text, re.S)
    if not body:
        sys.exit(f"ERROR: no thebibliography block in {path}")
    entries = {}
    parts = re.split(r"\\bibitem\{([^}]+)\}", body.group(1))
    # parts[0] = preamble junk; then alternating key, text
    for key, entry in zip(parts[1::2], parts[2::2]):
        entries[key.strip()] = " ".join(entry.split())
    return entries

def norm_tokens(s):
    s = re.sub(r"\\[a-zA-Z]+", " ", s)          # tex commands
    s = re.sub(r"[^a-zA-Z0-9 ]", " ", s.lower())
    return set(t for t in s.split() if len(t) > 2)

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--main", default=str(ROOT / "cwft_unified.tex"))
    ap.add_argument("--supp", default=str(ROOT / "cwft_supplement.tex"))
    args = ap.parse_args()

    sources = {s: parse_bib(HERE / f"bib_{s}.tex") for s in ("A", "B", "C")}
    if (HERE / "bib_extra.tex").exists():
        sources["X"] = parse_bib(HERE / "bib_extra.tex")
    warn, err = [], []

    # ---- merge by key with priority; track provenance ----
    merged, prov = {}, {}
    for s in reversed(PRIORITY):            # lowest priority first, higher overwrites
        for k, v in sources[s].items():
            if k in merged and jaccard(norm_tokens(merged[k]), norm_tokens(v)) < 0.5:
                warn.append(f"same-key different-text: {k} ({prov[k]} vs {s}) -- {PRIORITY[0]}-priority applied")
            merged[k] = v
            prov.setdefault(k, []).append(s)
            prov[k] = sorted(set(prov[k]))
    # apply aliases and drops
    for a, c in ALIAS.items():
        if a in merged:
            if c not in merged:
                err.append(f"alias target missing: {a} -> {c}")
            else:
                prov[c] = sorted(set(prov.get(c, []) + prov.pop(a, [])))
                del merged[a]
    for d in DROP:
        merged.pop(d, None); prov.pop(d, None)

    # ---- cross-key similarity report (undetected duplicates) ----
    keys = list(merged)
    toks = {k: norm_tokens(merged[k]) for k in keys}
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            j = jaccard(toks[k1], toks[k2])
            if j > 0.65:
                warn.append(f"possible duplicate ({j:.2f}): {k1} <-> {k2}  [add to ALIAS if same paper]")

    # ---- citation scan, first-appearance order ----
    def cites_in(path):
        p = Path(path)
        if not p.exists(): return []
        txt = p.read_text(encoding="utf-8")
        txt = re.sub(r"(?<!\\)%.*", "", txt)   # strip comments
        out = []
        for m in re.finditer(r"\\cite[tp]?\{([^}]+)\}", txt):
            for k in m.group(1).split(","):
                out.append(k.strip())
        return out

    def first_order(path):
        order, seen = [], set()
        for k in cites_in(path):
            k = ALIAS.get(k, k)
            if k in DROP:
                err.append(f"dropped key still cited in {Path(path).name}: {k}")
                continue
            if k not in merged:
                err.append(f"cite with no entry in {Path(path).name}: {k}")
                continue
            if k not in seen:
                seen.add(k); order.append(k)
        return order

    order_main = first_order(args.main)
    order_supp = first_order(args.supp)
    cited_any = set(order_main) | set(order_supp)
    uncited = [k for k in merged if k not in cited_any]
    if uncited:
        msg = f"{len(uncited)} entries cited nowhere: {', '.join(sorted(uncited)[:12])}{' ...' if len(uncited)>12 else ''}"
        (err if args.strict else warn).append(msg)

    def emit(fname, order, pending):
        out = [f"% AUTO-GENERATED by tools/merge_bib.py -- do not edit by hand.",
               f"% {len(order)} cited entries in first-appearance order"
               + (f"; {len(pending)} pending appended." if pending else "."),
               "\\begin{thebibliography}{99}\\setlength{\\itemsep}{2pt plus 1pt}"]
        for k in order:
            out.append(f"\\bibitem{{{k}}} {merged[k]}")
        if pending:
            out.append("% ---- entries below are cited nowhere yet (dropped by --strict) ----")
            for k in sorted(pending):
                out.append(f"\\bibitem{{{k}}} {merged[k]}")
        out.append("\\end{thebibliography}")
        (ROOT / fname).write_text("\n".join(out) + "\n", encoding="utf-8")

    emit("bibliography_merged.tex", order_main, [] if args.strict else uncited)
    emit("bibliography_supp.tex", order_supp, [])
    (HERE / "bib_merged.json").write_text(json.dumps(
        {"provenance": prov, "aliases": ALIAS, "dropped": sorted(DROP),
         "cited_main": order_main, "cited_supp": order_supp,
         "uncited": sorted(uncited)}, indent=1), encoding="utf-8")

    for w in warn: print(f"WARN: {w}")
    for e in err:  print(f"ERROR: {e}")
    print(f"merged={len(merged)} main={len(order_main)} supp={len(order_supp)} "
          f"uncited={len(uncited)} "
          f"(raw A={len(sources['A'])} B={len(sources['B'])} C={len(sources['C'])})")
    sys.exit(1 if err else 0)

if __name__ == "__main__":
    main()
