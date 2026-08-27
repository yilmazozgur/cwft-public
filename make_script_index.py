#!/usr/bin/env python3
"""Emit a Markdown index of every *.py in a directory: script name -> first line of its module
docstring (or first comment line). Used by tools/sync_public.sh; accurate by construction."""
import ast, sys, pathlib

d = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(d.glob("*.py")):
    src = p.read_text(encoding="utf-8", errors="replace")
    line = ""
    try:
        doc = ast.get_docstring(ast.parse(src)) or ""
        line = next((l.strip() for l in doc.splitlines() if l.strip()), "")
    except SyntaxError:
        pass
    if not line:
        for l in src.splitlines():
            s = l.strip()
            if s.startswith("#") and not s.startswith("#!") and len(s) > 2:
                line = s.lstrip("#").strip(); break
    rows.append((p.name, line.replace("|", "\\|")))
print("# Script index\n")
print("Auto-generated at sync time by `make_script_index.py` (repository root): each script's")
print("first docstring (or comment) line. Library modules (imported, not run): `cwf_substrate.py`,")
print("`cwf_hp_lib.py`. Scripts prefixed `_debug_` are diagnostics kept for the record.\n")
print("| script | purpose |\n|---|---|")
for n, l in rows:
    print(f"| `{n}` | {l} |")
