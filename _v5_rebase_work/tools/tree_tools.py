"""Tree snapshot / manifest / compare utilities for the MIMIR V5 rebase.

Windows-native, stdlib only. Paths are handled with pathlib and all text
output is UTF-8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Runtime/generated/secret material that is never part of a source snapshot.
SNAPSHOT_EXCLUDE_DIRS = {".venv", "venv", "vod_output", "__pycache__", "_v5_reference", ".pytest_cache"}
SNAPSHOT_EXCLUDE_FILES = {".env"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path, exclude_dirs: set[str], exclude_files: set[str]):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dirs)
        for name in sorted(filenames):
            if name in exclude_files:
                continue
            if name.startswith(".env") and name != ".env.example":
                continue
            full = Path(dirpath) / name
            yield full, full.relative_to(root).as_posix()


def manifest(root: Path, exclude_dirs: set[str], exclude_files: set[str]) -> dict[str, str]:
    return {rel: sha256_file(full) for full, rel in iter_files(root, exclude_dirs, exclude_files)}


def cmd_snapshot(args: argparse.Namespace) -> int:
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if dst.exists():
        print(f"refusing: destination exists: {dst}")
        return 2
    count = 0
    for full, rel in iter_files(src, SNAPSHOT_EXCLUDE_DIRS, SNAPSHOT_EXCLUDE_FILES):
        target = dst / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full, target)
        count += 1
    print(f"snapshot files copied: {count}")
    return 0


def cmd_manifest(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    exclude_dirs = set(args.exclude_dir or []) if args.exclude_dir is not None else set(SNAPSHOT_EXCLUDE_DIRS)
    exclude_files = set(SNAPSHOT_EXCLUDE_FILES) if not args.include_env_hash else set()
    data = manifest(root, exclude_dirs, exclude_files)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    print(f"manifest entries: {len(data)} -> {out}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    differ = sorted(k for k in set(left) & set(right) if left[k] != right[k])
    mismatch = len(only_left) + len(only_right) + len(differ)
    report = {
        "left_entries": len(left),
        "right_entries": len(right),
        "only_left": only_left,
        "only_right": only_right,
        "content_differs": differ,
        "mismatch_count": mismatch,
    }
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if mismatch == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("snapshot"); p.add_argument("src"); p.add_argument("dst")
    p = sub.add_parser("manifest"); p.add_argument("root"); p.add_argument("out")
    p.add_argument("--exclude-dir", action="append"); p.add_argument("--include-env-hash", action="store_true")
    p = sub.add_parser("compare"); p.add_argument("left"); p.add_argument("right")
    args = parser.parse_args()
    return {"snapshot": cmd_snapshot, "manifest": cmd_manifest, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
