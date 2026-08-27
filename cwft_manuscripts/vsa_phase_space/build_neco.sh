#!/bin/bash
# Build the Neural Computation journal version (main + supplement).
set -e
pdflatex -interaction=nonstopmode vsa_phase_space_neco.tex >/dev/null
pdflatex -interaction=nonstopmode vsa_phase_space_neco.tex >/dev/null
# filtered aux for xr (drop citation labels so the supplement's own bibliography wins)
python3 - <<'PY'
import re
aux = open("vsa_phase_space_neco.aux", errors="ignore").read()
keep = []
for line in aux.split("\n"):
    if line.startswith(r'\bibcite'): continue
    m = re.match(r'\\newlabel\{([^}]+)\}', line)
    if m and (':' not in m.group(1) or m.group(1).startswith('cite.')): continue
    keep.append(line)
open("neco_xr.aux","w").write("\n".join(keep))
PY
pdflatex -interaction=nonstopmode vsa_phase_space_neco_supp.tex >/dev/null
pdflatex -interaction=nonstopmode vsa_phase_space_neco_supp.tex >/dev/null
echo "built: vsa_phase_space_neco.pdf + vsa_phase_space_neco_supp.pdf"
