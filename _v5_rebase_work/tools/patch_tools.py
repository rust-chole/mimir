"""Extract-safe patch builder / scanner / reproduction test (stdlib, Windows).

build   : diff baseline_root vs final_root by SHA256 -> zip of added/changed files
scan    : reject excluded paths, media, archives, caches, secrets
repro   : copy baseline_root, extract zip over it (python zipfile AND Windows
          PowerShell Expand-Archive -Force), compare SHA256 with final_root
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tree_tools import SNAPSHOT_EXCLUDE_DIRS, SNAPSHOT_EXCLUDE_FILES, manifest  # noqa: E402

FORBIDDEN_DIR_PARTS = {".venv", "venv", "vod_output", "__pycache__", "_v5_reference", ".pytest_cache",
                       "cache", "discovery", ".git", "node_modules"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts", ".mp3",
                      ".wav", ".m4a", ".aac", ".flac", ".ogg", ".png", ".jpg", ".jpeg", ".gif", ".webp",
                      ".zip", ".rar", ".7z", ".tar", ".gz", ".bak", ".log", ".exe", ".dll", ".ass", ".srt"}
SECRET_PATTERNS = [
    ("openai_key", re.compile(r"sk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("bearer_header", re.compile(r"Authorization\s*[:=]\s*['\"]?Bearer\s+[A-Za-z0-9._\-]{12,}", re.I)),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("service_account", re.compile(r"\"type\"\s*:\s*\"service_account\"")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{30,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("assigned_secret_literal", re.compile(
        r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][A-Za-z0-9_\-]{24,}['\"]")),
]


def changed_files(baseline: Path, final: Path) -> tuple[list[str], list[str], list[str]]:
    left = manifest(baseline, set(SNAPSHOT_EXCLUDE_DIRS), set(SNAPSHOT_EXCLUDE_FILES))
    right = manifest(final, set(SNAPSHOT_EXCLUDE_DIRS), set(SNAPSHOT_EXCLUDE_FILES))
    added = sorted(set(right) - set(left))
    modified = sorted(k for k in set(left) & set(right) if left[k] != right[k])
    deleted = sorted(set(left) - set(right))
    return added, modified, deleted


def path_violations(rel: str) -> list[str]:
    problems = []
    pure = PurePosixPath(rel)
    if pure.is_absolute() or ".." in pure.parts or ":" in rel or "\\" in rel:
        problems.append("unsafe path")
    if any(part in FORBIDDEN_DIR_PARTS for part in pure.parts[:-1]):
        problems.append("forbidden directory")
    name = pure.name
    if name == ".env" or (name.startswith(".env") and name != ".env.example"):
        problems.append("env file")
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        problems.append(f"forbidden suffix {pure.suffix}")
    if re.search(r"(?i)(credential|secret|token).*\.json$", name):
        problems.append("credential-like json")
    return problems


def secret_hits(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="replace")
    return [label for label, pattern in SECRET_PATTERNS if pattern.search(text)]


def cmd_build(args: argparse.Namespace) -> int:
    baseline, final = Path(args.baseline).resolve(), Path(args.final).resolve()
    added, modified, deleted = changed_files(baseline, final)
    if deleted:
        print("DELETIONS REQUIRED (cannot be expressed by extract-over patch):", deleted)
        return 3
    members = added + modified
    problems = {rel: path_violations(rel) for rel in members}
    problems = {k: v for k, v in problems.items() if v}
    if problems:
        print("PATH VIOLATIONS:", json.dumps(problems, indent=1))
        return 4
    out = Path(args.zip).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for rel in members:
            source = final / Path(rel)
            info = zipfile.ZipInfo.from_file(source, arcname=rel)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.read_bytes())
    listing = {"added": added, "modified": modified, "deleted": deleted, "zip": str(out)}
    Path(args.listing).write_text(json.dumps(listing, indent=1), encoding="utf-8")
    print(json.dumps({"added": len(added), "modified": len(modified), "zip": str(out)}, indent=1))
    for rel in added:
        print("  A", rel)
    for rel in modified:
        print("  M", rel)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    failures = []
    with zipfile.ZipFile(args.zip) as archive:
        bad = archive.testzip()
        if bad:
            failures.append(f"CRC failure: {bad}")
        for info in archive.infolist():
            rel = info.filename
            if info.is_dir():
                continue
            for problem in path_violations(rel):
                failures.append(f"{rel}: {problem}")
            hits = secret_hits(archive.read(info))
            for label in hits:
                failures.append(f"{rel}: secret pattern {label}")
        count = len([i for i in archive.infolist() if not i.is_dir()])
    print(json.dumps({"members": count, "failures": failures}, indent=1))
    return 0 if not failures else 1


def _copy_clean(baseline: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(baseline, target)


def cmd_repro(args: argparse.Namespace) -> int:
    baseline, final = Path(args.baseline).resolve(), Path(args.final).resolve()
    work = Path(args.work).resolve()
    zip_path = Path(args.zip).resolve()
    final_manifest = manifest(final, set(SNAPSHOT_EXCLUDE_DIRS), set(SNAPSHOT_EXCLUDE_FILES))
    report = {}

    # A) Python zipfile extraction with overwrite.
    target_a = work / "repro_python_zipfile"
    _copy_clean(baseline, target_a)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target_a)
    report["python_zipfile"] = _compare(manifest(target_a, set(SNAPSHOT_EXCLUDE_DIRS), set(SNAPSHOT_EXCLUDE_FILES)), final_manifest)

    # B) Windows-native PowerShell Expand-Archive -Force (what a user runs).
    target_b = work / "repro_expand_archive"
    _copy_clean(baseline, target_b)
    import os
    env = dict(os.environ)
    env["MIMIR_PATCH_ZIP"] = str(zip_path)
    env["MIMIR_PATCH_DEST"] = str(target_b)
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         "$ErrorActionPreference='Stop'; "
         "Expand-Archive -LiteralPath $env:MIMIR_PATCH_ZIP -DestinationPath $env:MIMIR_PATCH_DEST -Force"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
    )
    report["expand_archive_exit"] = completed.returncode
    if completed.returncode != 0:
        report["expand_archive_stderr"] = completed.stderr[-2000:]
    report["powershell_expand_archive"] = _compare(
        manifest(target_b, set(SNAPSHOT_EXCLUDE_DIRS), set(SNAPSHOT_EXCLUDE_FILES)), final_manifest)

    print(json.dumps(report, indent=1))
    ok = (report["python_zipfile"]["mismatch_count"] == 0
          and report["expand_archive_exit"] == 0
          and report["powershell_expand_archive"]["mismatch_count"] == 0)
    return 0 if ok else 1


def _compare(left: dict[str, str], right: dict[str, str]) -> dict:
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    differ = sorted(k for k in set(left) & set(right) if left[k] != right[k])
    return {"entries": len(left), "only_patched": only_left, "only_final": only_right,
            "content_differs": differ, "mismatch_count": len(only_left) + len(only_right) + len(differ)}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build"); p.add_argument("baseline"); p.add_argument("final"); p.add_argument("zip"); p.add_argument("listing")
    p = sub.add_parser("scan"); p.add_argument("zip")
    p = sub.add_parser("repro"); p.add_argument("baseline"); p.add_argument("final"); p.add_argument("zip"); p.add_argument("work")
    args = parser.parse_args()
    return {"build": cmd_build, "scan": cmd_scan, "repro": cmd_repro}[args.cmd](args)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
