#!/usr/bin/env python3
"""thm_diff.py -- gate C3: imported theorem-family statements must match source verbatim.

Extracts every labeled theorem-family environment (theorem, proposition, lemma, corollary,
definition, conjecture, remark) from the merged documents and from the three frozen sources;
for each label present in both, compares whitespace-normalized bodies. Differences are printed
as word-level unified diffs and exit 1 (justified deltas go in tools/thm_diff_waivers.json:
{label: reason}).

Labels present in merged but in no source are reported as NEW (informational -- fresh
statements are allowed, they are simply not sourced).
"""
import re, sys, json, difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MS = ROOT.parent

SOURCES = [
    MS / "P1_computational_gravity_split/Journal of Physics Communications/analog_gravity_neutral.tex",
    MS / "combined_epistemic_wave_program/epistemic_wave_program.tex",
    MS / "combined_self_reference_arc/self_reference_arc.tex",
    # U5 repoint (UPGRADE_PLAN D-4): the book is now a first-class source -- the cited
    # public deposit whose synthesis layer the 2026-08-23 upgrade imported (pinned e579c3b).
    MS.parent / "cwft_book/cwf_book.tex",
]
TARGETS = [ROOT / "cwft_unified.tex", ROOT / "cwft_supplement.tex"]
ENVS = "theorem|proposition|lemma|corollary|definition|conjecture|remark"
WAIVERS = HERE / "thm_diff_waivers.json"

def extract(path):
    """label -> normalized env body (env kind + title + text, label line removed)."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?<!\\)%.*", "", text)
    out = {}
    for m in re.finditer(rf"\\begin\{{({ENVS})\}}(.*?)\\end\{{\1\}}", text, re.S):
        body = m.group(2)
        lab = re.search(r"\\label\{([^}]+)\}", body)
        if not lab:
            continue
        norm = re.sub(r"\\label\{[^}]+\}", "", body)
        norm = " ".join(norm.split())
        out[lab.group(1)] = (m.group(1), norm)
    return out

def main():
    src = {}
    for s in SOURCES:
        src.update(extract(s))
    waivers = json.loads(WAIVERS.read_text(encoding="utf-8")) if WAIVERS.exists() else {}
    fails, new = [], []
    checked = 0
    for t in TARGETS:
        if not t.exists():
            continue
        for lab, (env, body) in extract(t).items():
            if lab not in src:
                new.append(f"{t.name}: {lab} ({env}) -- NEW, no source counterpart")
                continue
            checked += 1
            senv, sbody = src[lab]
            if body == sbody:
                continue
            if lab in waivers:
                print(f"WAIVED: {lab} -- {waivers[lab]}")
                continue
            diff = "\n".join(difflib.unified_diff(sbody.split(), body.split(),
                                                  "source", t.name, lineterm="", n=2))
            fails.append(f"DIFF {lab} ({env} vs source {senv}):\n{diff[:2000]}")
    for n in new:
        print(f"NEW: {n}")
    for f in fails:
        print(f"ERROR: {f}")
    print(f"thm_diff: {'FAIL' if fails else 'PASS'} ({checked} statements checked, "
          f"{len(fails)} diffs, {len(new)} new, {len(waivers)} waived)")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
