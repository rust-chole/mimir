"""Import every ai.* module of a code root (feature flag unset); report failures."""
from __future__ import annotations

import importlib
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
os.environ.pop("MIMIR_PRO_EDIT", None)
os.chdir(root)
sys.path.insert(0, str(root))
modules = sorted(".".join(p.with_suffix("").parts) for p in pathlib.Path("ai").rglob("*.py") if p.name != "__init__.py")
modules += ["ai", "ai.editor", "ai.editor.pro_edit", "ai.editor.pro_edit.vision", "ai.video_brain"]
failures = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as error:  # report every failure
        failures.append((name, f"{type(error).__name__}: {error}"))
sys.stdout.reconfigure(encoding="utf-8")
print(f"{pathlib.Path(sys.executable).parents[1].name} python {sys.version.split()[0]} | "
      f"modules imported {len(modules) - len(failures)}/{len(modules)} | failures {failures}")
raise SystemExit(1 if failures else 0)
