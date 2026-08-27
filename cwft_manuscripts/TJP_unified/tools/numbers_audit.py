#!/usr/bin/env python3
"""numbers_audit.py -- gate C2: every numeric token in the merged text must trace to a source.

Extracts numeric tokens (>=2 digits, or any decimal) from the merged main/supplement tex and
checks membership in the union of the three frozen sources' tokens. Tokens absent from every
source must appear in tools/numbers_whitelist.json with a justification, else exit 1.

Single-digit integers are not audited (they are ubiquitous in both directions and carry no
transcription-risk signal). Normalization strips TeX brace-commas (4{,}924 -> 4924) and commas.

Usage: numbers_audit.py [files...]   (default: ../cwft_unified.tex ../cwft_supplement.tex)
"""
import re, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MS = ROOT.parent  # cwft_manuscripts/

SOURCES = [
    MS / "P1_computational_gravity_split/Journal of Physics Communications/analog_gravity_neutral.tex",
    MS / "combined_epistemic_wave_program/epistemic_wave_program.tex",
    MS / "combined_self_reference_arc/self_reference_arc.tex",
    # U5 repoint (UPGRADE_PLAN D-4): the book is now a first-class source -- the cited
    # public deposit whose synthesis layer the 2026-08-23 upgrade imported (pinned e579c3b).
    MS.parent / "cwft_book/cwf_book.tex",
]
WL_PATH = HERE / "numbers_whitelist.json"

NUM = re.compile(r"\d+\.\d+|\d{2,}")

def tokens(text):
    text = re.sub(r"(?<!\\)%.*", "", text)          # strip comments
    # strip pure-layout numerics: \includegraphics options, p{..cm}/tabularx column widths,
    # and \\[..pt] spacing -- these are typography, not scientific content (gate C2 scope)
    text = re.sub(r"\\includegraphics\[[^\]]*\]", r"\\includegraphics", text)
    text = re.sub(r"[pm]\{[0-9.]+(?:cm|in|pt|em)\}", "", text)
    text = re.sub(r"\\\\\[[0-9.]+(?:pt|em|cm)\]", "", text)
    text = text.replace("{,}", "")  # TeX thousands-comma only: 4{,}924 -> 4924
    # plain commas stay as separators (set literals {1,2,3} must NOT fuse into '123')
    return set(NUM.findall(text))

def main():
    targets = [Path(a) for a in sys.argv[1:]] or [ROOT / "cwft_unified.tex", ROOT / "cwft_supplement.tex"]
    src = set()
    for s in SOURCES:
        src |= tokens(s.read_text(encoding="utf-8"))
    wl = json.loads(WL_PATH.read_text(encoding="utf-8")) if WL_PATH.exists() else {}

    bad = []
    for t in targets:
        if not t.exists():
            continue
        for tok in sorted(tokens(t.read_text(encoding="utf-8"))):
            if tok in src or tok in wl:
                continue
            bad.append((t.name, tok))
    for name, tok in bad:
        print(f"ERROR: {name}: numeric token '{tok}' not found in any source and not whitelisted")
    n_wl = len(wl)
    print(f"audit: {'FAIL' if bad else 'PASS'} ({len(bad)} unsourced tokens; whitelist has {n_wl})")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
