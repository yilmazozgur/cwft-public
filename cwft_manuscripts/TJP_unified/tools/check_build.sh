#!/usr/bin/env bash
# check_build.sh -- gate G1/G3/G4-lite for the TJP unified build.
# Compiles a tex file twice and fails on: TeX errors, undefined refs/cites, multiply-defined labels.
# Warns on overfull hbox >= 10pt. Prints the page count.
# Usage: check_build.sh <file.tex>
set -u
f="${1:?usage: check_build.sh <file.tex>}"
base="${f%.tex}"
dir="$(dirname "$f")"
cd "$dir" || exit 2
name="$(basename "$base")"

for pass in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error "$name.tex" > /dev/null 2>&1
  rc=$?
done
log="$name.log"
[ -f "$log" ] || { echo "FAIL: no log produced"; exit 2; }

errs=$(grep -c "^!" "$log")
undef_ref=$(grep -c "LaTeX Warning: Reference .* undefined" "$log")
undef_cite=$(grep -c "LaTeX Warning: Citation .* undefined" "$log")
multi=$(grep -c "multiply defined" "$log")
over=$(grep "Overfull \\\\hbox" "$log" | awk -F'[( ]' '{for(i=1;i<=NF;i++) if($i ~ /pt$/){sub("pt","",$i); if($i+0>=10) c++}} END{print c+0}')
pages=$(pdfinfo "$name.pdf" 2>/dev/null | awk '/^Pages/{print $2}')

echo "== $name: errors=$errs undef_refs=$undef_ref undef_cites=$undef_cite multiply_defined=$multi overfull>=10pt=$over pages=${pages:-?} (pdflatex rc=$rc)"
grep "^!" "$log" | head -5
grep "LaTeX Warning: Reference .* undefined" "$log" | head -5
grep "LaTeX Warning: Citation .* undefined" "$log" | head -5

if [ "$errs" -ne 0 ] || [ "$undef_ref" -ne 0 ] || [ "$undef_cite" -ne 0 ] || [ "$multi" -ne 0 ] || [ "$rc" -ne 0 ]; then
  echo "GATE: FAIL"
  exit 1
fi
[ "$over" -gt 0 ] && echo "GATE: PASS (with $over overfull-hbox warnings >=10pt)" || echo "GATE: PASS"
exit 0
