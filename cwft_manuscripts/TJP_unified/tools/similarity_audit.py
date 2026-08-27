#!/usr/bin/env python3
"""similarity_audit.py -- gate S1 (vs the public book) and gate X3 (self-duplication).

Default mode: 10-word shingle overlap of the MAIN text against the Zenodo book's tex
(the only public overlap surface iThenticate can match). Reports the fraction of the main
text's shingles that appear in the book, and the longest contiguous verbatim runs with
their locations. The supplement is reported separately (proofs are expected to overlap
heavily by design; the cover letter discloses this).

--self mode (X3): near-duplicate paragraph detection WITHIN the main text (8-word shingle
Jaccard > 0.5 between distinct paragraphs).

Normalization: strip comments, TeX commands, math ($...$ replaced by a MATH token so
verbatim formula reuse still counts as overlap), braces, punctuation; lowercase.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BOOK = ROOT.parent.parent / "cwft_book" / "cwf_book.tex"

def norm_words(text):
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " MATH ", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}~]", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())
    return text.split()

def shingles(words, k):
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

def main():
    if "--self" in sys.argv:
        text = (ROOT / "cwft_unified.tex").read_text(encoding="utf-8")
        text = re.sub(r"(?<!\\)%.*", "", text)
        paras = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) > 40]
        sets = [shingles(norm_words(p), 8) for p in paras]
        dups = 0
        for i in range(len(paras)):
            for j in range(i + 1, len(paras)):
                if not sets[i] or not sets[j]:
                    continue
                jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
                if jac > 0.5:
                    dups += 1
                    print(f"DUP ({jac:.2f}): para {i} <-> para {j}:")
                    print("  A:", " ".join(paras[i].split()[:15]), "...")
                    print("  B:", " ".join(paras[j].split()[:15]), "...")
        print(f"self-duplication (X3): {'FAIL' if dups else 'PASS'} ({len(paras)} paragraphs, {dups} near-duplicate pairs)")
        sys.exit(1 if dups else 0)

    K = 10
    book_words = norm_words(BOOK.read_text(encoding="utf-8"))
    book_sh = shingles(book_words, K)
    for name in ("cwft_unified.tex", "cwft_supplement.tex"):
        words = norm_words((ROOT / name).read_text(encoding="utf-8"))
        sh = [tuple(words[i:i+K]) for i in range(len(words) - K + 1)]
        hits = [s in book_sh for s in sh]
        frac = sum(hits) / len(hits) if hits else 0.0
        # longest verbatim runs (consecutive hit shingles => run of len K + extra)
        runs, start = [], None
        for i, h in enumerate(hits):
            if h and start is None:
                start = i
            elif not h and start is not None:
                runs.append((i - start + K - 1, start)); start = None
        if start is not None:
            runs.append((len(hits) - start + K - 1, start))
        runs.sort(reverse=True)
        print(f"== {name}: {frac*100:.1f}% of {K}-word shingles appear in the book "
              f"({sum(hits)}/{len(hits)})")
        for ln, st in runs[:8]:
            print(f"   run of ~{ln} words: \"{' '.join(words[st:st+12])} ...\"")
    print("note: main-text target <15-20% before disclosure; supplement overlap is by design "
          "(verbatim proofs), covered by the cover-letter disclosure.")

if __name__ == "__main__":
    main()
