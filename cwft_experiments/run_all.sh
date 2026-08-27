#!/usr/bin/env bash
# Regenerate every figure and every results.json record for the CWF book.
# CPU-only, ~minutes total. Library-only modules (imported, not run) are skipped.
# Usage:  bash run_all.sh        (logs go to /tmp/<script>.log)
set -u
cd "$(dirname "$0")"

skip="cwf_substrate.py cwf_hp_lib.py"     # imported by others; no standalone run
fail=0; n=0
for s in cwf_*.py; do
  case " $skip " in *" $s "*) echo "skip (library): $s"; continue;; esac
  n=$((n+1))
  printf "run %-46s " "$s"
  if python3 "$s" > "/tmp/${s%.py}.log" 2>&1; then
    echo "ok"
  else
    echo "FAIL  (see /tmp/${s%.py}.log)"; fail=$((fail+1))
  fi
done
echo "------------------------------------------------------------"
echo "ran $n scripts; failures: $fail"
exit $fail
