"""Batch B4: verify script, requirements, .env.example.

verify_unified_clean.py : current == V5 base except V5's additive Pro Edit block
                          (checked) -> V5 file + ONE Windows/UTF-8 hardening:
                          the AST pre-check reads sources with utf-8-sig, because
                          three current-root sources carry a UTF-8 BOM (Python
                          itself compiles them fine; ast.parse(str) does not).
requirements.txt        : V5 adds opencv-python-headless (optional at runtime)
                          and a commented uharfbuzz hint; the current two lines
                          must be an exact prefix.
.env.example            : new documentation file (no values for keys).
"""
from __future__ import annotations

import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
FINAL = WORK / "final_root"
BASE = WORK / "baseline_root"
V5 = WORK.parent / "_v5_reference" / "extracted" / "MIMIR_SHORTS_V7_1_PRO_EDIT_V5"

AST_READ_OLD = '        tree = ast.parse(path.read_text(encoding="utf-8"))\n'
AST_READ_NEW = ('        # utf-8-sig: Windows editors may add a BOM; the interpreter accepts it,\n'
                '        # ast.parse(str) does not.\n'
                '        tree = ast.parse(path.read_text(encoding="utf-8-sig"))\n')


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    cur = (BASE / "verify_unified_clean.py").read_text(encoding="utf-8")
    ref = (V5 / "verify_unified_clean.py").read_text(encoding="utf-8")
    block_start = ref.index("# Pro Edit (V1-V5): feature flag default OFF")
    block_end = ref.index('if shutil.which("ffmpeg") is None:')
    summary = ' - Pro Edit V5 (default OFF) contracts + tests: OK ({pro_edit_summary})")\n'
    reconstructed = ref[:block_start] + ref[block_end:]
    reconstructed = reconstructed.replace('print(f"' + summary, "")
    if reconstructed != cur:
        raise SystemExit("verify_unified_clean.py: current != V5 base minus the Pro Edit block")
    if ref.count(AST_READ_OLD) != 1:
        raise SystemExit("AST read line not found exactly once")
    merged = ref.replace(AST_READ_OLD, AST_READ_NEW)
    compile(merged, "verify_unified_clean.py", "exec", dont_inherit=True)
    (FINAL / "verify_unified_clean.py").write_bytes(merged.encode("utf-8"))
    print("B4 verify_unified_clean.py: V5 Pro Edit block ported + utf-8-sig AST read")

    cur_req = (BASE / "requirements.txt").read_bytes()
    ref_req = (V5 / "requirements.txt").read_bytes()
    if not ref_req.startswith(cur_req):
        raise SystemExit("requirements.txt: current lines are not a prefix of the V5 file")
    (FINAL / "requirements.txt").write_bytes(ref_req)
    print("B4 requirements.txt: + opencv-python-headless>=4.9,<5 (optional at runtime), uharfbuzz comment")

    env_example = (V5 / ".env.example").read_bytes()
    import re

    for line in env_example.decode("utf-8").splitlines():
        key, sep, value = line.partition("=")
        name = key.strip().lstrip("#").strip()
        if sep and re.fullmatch(r"[A-Z0-9_]*(_KEY|_TOKEN|_SECRET|_PASSWORD)", name) and value.strip():
            raise SystemExit(f".env.example has a value for {name}")
    (FINAL / ".env.example").write_bytes(env_example)
    print("B4 .env.example: added (every key empty / placeholder-free)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
