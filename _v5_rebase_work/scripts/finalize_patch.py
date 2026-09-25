"""Build the FINAL extract-safe patch into the repository root and prove it.

1. build   baseline_root (untouched original) vs final_root (integrated) -> ZIP in repo root
2. scan    excluded paths / media / archives / caches / secrets
3. repro   clean copy of baseline + extract (Python zipfile AND PowerShell Expand-Archive -Force)
           -> SHA256 manifest compare against final_root (mismatch_count must be 0)
4. live    the live repository's ORIGINAL source files are still byte-identical to the
           session-start manifest (nothing was modified in place)
Writes evidence JSON under _v5_rebase_work/evidence/patch/.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent
PY = sys.executable
TOOLS = WORK / "tools"
ZIP = REPO / "mimir_current_root_to_pro_edit_v5_patch.zip"
EVIDENCE = WORK / "evidence" / "patch"


def run(*args: str) -> tuple[int, str]:
    result = subprocess.run([PY, "-X", "utf8", *args], capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={"PYTHONDONTWRITEBYTECODE": "1", **__import__("os").environ})
    return result.returncode, (result.stdout + result.stderr).strip()


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    code, out = run(str(TOOLS / "patch_tools.py"), "build", str(WORK / "baseline_root"), str(WORK / "final_root"),
                    str(ZIP), str(EVIDENCE / "listing.json"))
    report["build"] = {"exit": code, "output": out}
    if code != 0:
        print(out)
        return 1
    report["zip"] = {"path": str(ZIP), "bytes": ZIP.stat().st_size,
                     "sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest()}
    code, out = run(str(TOOLS / "patch_tools.py"), "scan", str(ZIP))
    report["scan"] = {"exit": code, **json.loads(out)}
    repro_dir = WORK / "repro"
    code, out = run(str(TOOLS / "patch_tools.py"), "repro", str(WORK / "baseline_root"), str(WORK / "final_root"),
                    str(ZIP), str(repro_dir))
    report["repro"] = {"exit": code, **json.loads(out)}
    excludes = []
    for name in (".venv", "venv", "vod_output", "__pycache__", "_v5_reference", "_v5_rebase_work", ".pytest_cache"):
        excludes += ["--exclude-dir", name]
    code, out = run(str(TOOLS / "tree_tools.py"), "manifest", str(REPO), str(EVIDENCE / "live_source_at_end.json"),
                    *excludes)
    code, out = run(str(TOOLS / "tree_tools.py"), "compare", str(WORK / "manifests" / "baseline_source.json"),
                    str(EVIDENCE / "live_source_at_end.json"))
    live = json.loads(out)
    # The live root may only have GAINED the deliverables (report, zip) - never a changed original.
    report["live_root_original_sources"] = {
        "content_differs": live["content_differs"], "missing_originals": live["only_left"],
        "new_top_level_files": live["only_right"], "unchanged": not live["content_differs"] and not live["only_left"]}
    (EVIDENCE / "finalize_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    summary = {
        "zip": report["zip"], "members": report["scan"]["members"], "scan_failures": report["scan"]["failures"],
        "python_zipfile_mismatch": report["repro"]["python_zipfile"]["mismatch_count"],
        "expand_archive_exit": report["repro"]["expand_archive_exit"],
        "expand_archive_mismatch": report["repro"]["powershell_expand_archive"]["mismatch_count"],
        "entries_compared": report["repro"]["python_zipfile"]["entries"],
        "live_root_original_sources": report["live_root_original_sources"],
    }
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    ok = (not summary["scan_failures"] and summary["python_zipfile_mismatch"] == 0
          and summary["expand_archive_exit"] == 0 and summary["expand_archive_mismatch"] == 0
          and summary["live_root_original_sources"]["unchanged"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
