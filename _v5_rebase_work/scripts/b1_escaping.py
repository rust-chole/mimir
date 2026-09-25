"""Batch B1: FFmpeg subtitle filter-path escaping hardening (ported from Pro Edit V5).

caption_renderer.py : the current file equals the V5 base except for V5's
                      escape_filter_path hunk -> port that hunk, then prove the
                      result is byte-identical to the V5 file.
intro_renderer.py   : current-root file (BOM, one CRLF, cp1254 mojibake, LOCKED
                      PEAK validation) -> replace ONLY the escape_filter_path body
                      at byte level; every other byte is preserved.
"""
from __future__ import annotations

import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
FINAL = WORK / "final_root"
V5 = WORK.parent / "_v5_reference" / "extracted" / "MIMIR_SHORTS_V7_1_PRO_EDIT_V5"


def port_caption_renderer() -> None:
    cur = FINAL / "ai" / "editor" / "caption_renderer.py"
    ref = V5 / "ai" / "editor" / "caption_renderer.py"
    src = cur.read_text(encoding="utf-8")
    ref_src = ref.read_text(encoding="utf-8")
    old_start = src.index("    # Windows drive colon:\n")
    old_end = src.index("    return value\n", old_start)
    new_start = ref_src.index("    # Filtre seçeneği (2. seviye) kaçışı")
    new_end = ref_src.index("    return value\n", new_start)
    old_block = src[old_start:old_end]
    assert "value[1] == \":\"" in old_block and "r\"\\'\"" in old_block, old_block
    merged = src[:old_start] + ref_src[new_start:new_end] + src[old_end:]
    if merged != ref_src:
        raise SystemExit("caption_renderer: merged file differs from V5 beyond the escape hunk")
    cur.write_bytes(merged.encode("utf-8"))
    compile(merged, str(cur), "exec", dont_inherit=True)
    print("B1 caption_renderer.py: V5 escape_filter_path ported (result == V5 file, LF, no BOM)")


INTRO_OLD = (
    b"    # Windows drive letter for FFmpeg filter syntax:\n"
    b"    # C:/... -> C\\:/...\n"
    b"    if len(value) >= 2 and value[1] == \":\":\n"
    b"        value = value[0] + r\"\\:\" + value[2:]\n"
    b"\n"
    b"    return value.replace(\"'\", r\"\\'\")\n"
)
INTRO_NEW = (
    b"    # Windows drive letter for FFmpeg filter syntax (the only ':' in a Windows path):\n"
    b"    # C:/... -> C\\:/...\n"
    b"    value = value.replace(\":\", r\"\\:\")\n"
    b"\n"
    b"    # A quoted filter value cannot contain \\' (it closes the quote and the\n"
    b"    # apostrophe is lost): close the quote, add an escaped quote, reopen.\n"
    b"    return value.replace(\"'\", \"'\\\\\\\\\\\\''\")\n"
)


def port_intro_renderer() -> None:
    path = FINAL / "ai" / "editor" / "intro_renderer.py"
    raw = path.read_bytes()
    if raw.count(INTRO_OLD) != 1:
        raise SystemExit(f"intro_renderer: expected 1 old escape block, found {raw.count(INTRO_OLD)}")
    ref = (V5 / "ai" / "editor" / "intro_renderer.py").read_bytes()
    if ref.count(INTRO_NEW) != 1:
        raise SystemExit("intro_renderer: V5 replacement block not found verbatim in the V5 file")
    merged = raw.replace(INTRO_OLD, INTRO_NEW)
    compile(merged, str(path), "exec", dont_inherit=True)
    # Everything outside the replaced block is byte-identical.
    head = raw.index(INTRO_OLD)
    assert merged[:head] == raw[:head]
    assert merged[head + len(INTRO_NEW):] == raw[head + len(INTRO_OLD):]
    path.write_bytes(merged)
    print(f"B1 intro_renderer.py: escape_filter_path body replaced at byte {head}; "
          f"BOM kept={merged.startswith(bytes.fromhex('efbbbf'))}, CRLF count={merged.count(b'\r\n')}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    port_caption_renderer()
    port_intro_renderer()
