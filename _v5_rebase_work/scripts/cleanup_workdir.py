"""Post-validation cleanup of the repository-local work directory.

KEEP   baseline_root/ (untouched original sources = rollback), manifests, logs, scripts,
       tools, caption-regression results, real-footage JSON/PNG evidence, patch evidence.
DELETE bulky / reproducible trees: final_root (== baseline + patch, SHA256-proven),
       verify copies, runtime sandbox, reproduction trees, candidate patch, the OpenCV
       test venv, real-footage MP4s, and the V5 extraction inside _v5_reference
       (the user's ZIP itself is left untouched).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
REPO = WORK.parent
EVIDENCE = WORK / "evidence"


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        print("deleted dir ", path.relative_to(REPO))
    elif path.exists():
        path.unlink()
        print("deleted file", path.relative_to(REPO))


def move(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print("kept        ", dst.relative_to(REPO))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    EVIDENCE.mkdir(exist_ok=True)
    for log in sorted(WORK.glob("logs_*.txt")):
        move(log, EVIDENCE / "logs" / log.name)
    move(WORK / "manifests", EVIDENCE / "manifests")
    move(WORK / "regress", EVIDENCE / "regress")
    move(WORK / "diffs", EVIDENCE / "diffs")
    real = WORK / "real_kai16"
    for media in real.rglob("*.mp4"):
        remove(media)
    move(real, EVIDENCE / "real_kai16")
    for name in ("final_root", "verify_cv", "verify_prod", "runtime_root", "repro", "patch_candidate", "venv_cv",
                 "report_draft.md"):
        remove(WORK / name)
    remove(REPO / "_v5_reference" / "extracted")
    print("remaining in _v5_rebase_work:", sorted(p.name for p in WORK.iterdir()))
    print("remaining in _v5_reference:", sorted(p.name for p in (REPO / "_v5_reference").iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
