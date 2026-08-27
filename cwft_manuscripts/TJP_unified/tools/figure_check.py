#!/usr/bin/env python3
"""figure_check.py -- gate G5: figures exist, are used, and meet TJP print specs.

Checks, for every \\includegraphics in the main and supplement tex:
  - the file exists in figures/                                        [error]
  - printed width  = frac * LINEWIDTH_CM <= 16 cm                      [error]
  - printed height = native aspect * printed width <= 20 cm            [error]
  - effective resolution at printed size >= 300 dpi                    [error]
  - PNG carries dpi metadata                                           [warn]
Also reports figures/ files never used by either document              [warn].
"""
import re, sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGS = ROOT / "figures"
LINEWIDTH_CM = 15.92  # a4paper, 1in margins (geometry)

def main():
    errs, warns, used = [], [], set()
    for tex in (ROOT / "cwft_unified.tex", ROOT / "cwft_supplement.tex"):
        text = re.sub(r"(?<!\\)%.*", "", tex.read_text(encoding="utf-8"))
        for m in re.finditer(r"\\includegraphics\[width=([0-9.]*)\\linewidth\]\{([^}]+)\}", text):
            frac = float(m.group(1)) if m.group(1) else 1.0
            name = m.group(2)
            used.add(name)
            f = FIGS / name
            if not f.exists():
                errs.append(f"{tex.name}: missing figure file {name}")
                continue
            im = Image.open(f)
            w_px, h_px = im.size
            pw = frac * LINEWIDTH_CM
            ph = pw * h_px / w_px
            eff_dpi = w_px / (pw / 2.54)
            tag = f"{name:32s} printed {pw:5.2f}x{ph:5.2f}cm  eff_dpi={eff_dpi:4.0f}"
            if pw > 16.0 + 1e-6: errs.append(f"{tex.name}: {tag}  WIDTH>16cm")
            if ph > 20.0 + 1e-6: errs.append(f"{tex.name}: {tag}  HEIGHT>20cm")
            if eff_dpi < 300 - 1e-6: errs.append(f"{tex.name}: {tag}  <300dpi")
            if "dpi" not in im.info: warns.append(f"{name}: no dpi metadata")
            print(f"ok  {tex.name[:14]:14s} {tag}")
    for f in sorted(FIGS.glob("*.png")):
        if f.name not in used:
            warns.append(f"unused figure file: {f.name}")
    for w in warns: print(f"WARN: {w}")
    for e in errs:  print(f"ERROR: {e}")
    print(f"figure_check: {'FAIL' if errs else 'PASS'} ({len(used)} used, {len(warns)} warnings)")
    sys.exit(1 if errs else 0)

if __name__ == "__main__":
    main()
