"""One text file for labeling: export it, fill it in any editor while watching the videos, import it.

    python -m ai.clip_eval export            -> vod_output/clip_eval/label_sheet.txt
    (a person watches each video, writes its moments and marks the block [x])
    python -m ai.clip_eval import vod_output/clip_eval/label_sheet.txt

A block per video:

    === <case_id> [ ]                          [x] = these labels are complete
    # video: <file> (<length>; split dev)
    # [0:00.0] transcript of the first 30 s     (read-only: lines starting with # are ignored)
    # [0:30.0] ...
    moment: 12:30 - 13:05 | great | chair breaks, chat explodes
    never: 1:02:00 - 1:04:30 | sponsor read

Times are H:MM:SS(.s), M:SS(.s) or seconds on the source video's clock. Blind like the editor: the
sheet holds the transcript and this annotator's own labels only, never MIMIR's picks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ai.clip_eval import cases as case_store

SHEET_NAME = "label_sheet.txt"
_HEADER = re.compile(r"^===\s+(\S+)\s*\[([ xX]?)\]\s*$")
_RANGE = re.compile(r"^\s*([0-9:.,]+)\s*-\s*([0-9:.,]+)\s*(?:\|(.*))?$")
_INTRO = """# MIMIR clip label sheet ({when}).
# For each video: watch it (the file on its "# video:" line) and write the moments you would publish as a
# Short, one per line, from the first second you would keep (setup / hook) to the last (payoff + reaction):
#     moment: <start> - <end> | great | <short note>      grade: great (you would post it) / ok / bad
#     never:  <start> - <end> | <why it must never be picked>   (sponsor read, private info, dead air...)
# Times: H:MM:SS, M:SS or seconds, on the video's own clock. Aim for 2-6 moments per video. The transcript
# lines (starting with #) are there to help you find things; they are ignored. When a video is done, change
# its [ ] to [x]. Only [x] blocks are scored.
annotator: {annotator}
"""


@dataclass
class Block:
    case_id: str
    done: bool
    moments: list[dict[str, Any]] = field(default_factory=list)
    negatives: list[dict[str, Any]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def export_sheet(*, root: str | Path | None = None, only: Sequence[str] | None = None, out: str | Path | None = None,
                 annotator: str | None = None, window: float = 30.0) -> tuple[Path, int]:
    cases = case_store.list_cases(root, only)
    if not cases:
        raise FileNotFoundError(f"no cases under {case_store.eval_root(root) / 'cases'}; run `prepare` first")
    name = case_store.clean_annotator(annotator) if annotator else case_store.default_annotator()
    parts = [_INTRO.format(when=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"), annotator=name)]
    for case in cases:
        labels = case.labels(name)
        lines = [f"=== {case.case_id} [{'x' if labels and labels.finished else ' '}]",
                 f"# video: {case.video_path} ({case_store.clock(case.duration)}; split {case.split})"]
        if case.has_transcript():
            for start, _end, text in case_store.transcript_windows(case.words(), window, case.duration):
                lines.append(f"# [{case_store.clock(start)}] {text}")
        else:
            lines.append("# (no transcript yet: " + case_store.transcribe_command(case.video_path) + ")")
        for moment in labels.moments if labels else []:
            lines.append(f"moment: {case_store.clock(moment.start)} - {case_store.clock(moment.end)} | {moment.grade}"
                         + (f" | {moment.note}" if moment.note else ""))
        for never in labels.negatives if labels else []:
            lines.append(f"never: {case_store.clock(never.start)} - {case_store.clock(never.end)}"
                         + (f" | {never.note}" if never.note else ""))
        if not labels or not labels.moments:
            lines.append("moment: ")
        lines.append("")
        parts.append("\n".join(lines))
    target = Path(out) if out else case_store.eval_root(root) / SHEET_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    return target, len(cases)


def parse_sheet(text: str) -> tuple[str | None, list[Block]]:
    """-> (annotator named in the sheet, blocks)."""
    annotator: str | None = None
    blocks: list[Block] = []
    for number, raw in enumerate(str(text).replace("\r\n", "\n").splitlines(), start=1):
        line = raw.strip()
        header = _HEADER.match(line)
        if header:
            blocks.append(Block(header.group(1), header.group(2).lower() == "x"))
            continue
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "annotator" and not blocks:
            annotator = value.strip() or None
            continue
        if not blocks or key not in ("moment", "never"):
            if blocks:
                blocks[-1].problems.append(f"line {number}: not understood: {raw.strip()[:80]!r}")
            continue
        if not value.strip():
            continue                                             # the empty template line
        block = blocks[-1]
        match = _RANGE.match(value)
        if not match:
            block.problems.append(f"line {number}: expected '<start> - <end> | ...', got {value.strip()[:80]!r}")
            continue
        fields = [f.strip() for f in (match.group(3) or "").split("|")]
        try:
            start, end = case_store.parse_clock(match.group(1)), case_store.parse_clock(match.group(2))
        except ValueError as error:
            block.problems.append(f"line {number}: {error}")
            continue
        if key == "moment":
            grade = (fields[0].lower() if fields and fields[0] else "great")
            note = " | ".join(fields[1:]) if len(fields) > 1 else ""
            if grade not in case_store.GRADES:                   # "moment: 1:00 - 1:30 | funny" -> note, grade great
                grade, note = "great", " | ".join(fields)
            block.moments.append({"start": start, "end": end, "grade": grade, "note": note})
        else:
            block.negatives.append({"start": start, "end": end, "note": " | ".join(f for f in fields if f)})
    return annotator, blocks


def import_sheet(path: str | Path, *, root: str | Path | None = None, annotator: str | None = None
                 ) -> list[dict[str, str]]:
    """Write every block into its case's labels (for the sheet's annotator, or ``annotator``)."""
    named, blocks = parse_sheet(Path(path).read_text(encoding="utf-8-sig"))
    name = case_store.clean_annotator(annotator or named) if (annotator or named) else case_store.default_annotator()
    known = {c.case_id: c for c in case_store.list_cases(root)}
    rows: list[dict[str, str]] = []
    for block in blocks:
        case = known.get(block.case_id)
        if case is None:
            rows.append({"case_id": block.case_id, "status": "unknown case (not prepared on this machine)"})
            continue
        if block.problems:
            rows.append({"case_id": block.case_id, "status": "not saved: " + "; ".join(block.problems)})
            continue
        current = case.labels(name)
        payload = {"moments": block.moments, "negatives": block.negatives, "finished": block.done}
        new = case_store.labels_from_dict(payload, name).to_dict(case.case_id)
        if current is not None:
            old = current.to_dict(case.case_id)
            if (old["moments"], old["negatives"], old["finished"]) == (new["moments"], new["negatives"],
                                                                       new["finished"]):
                rows.append({"case_id": block.case_id, "status": "unchanged"})
                continue
        if not block.moments and not block.negatives and not block.done:
            rows.append({"case_id": block.case_id, "status": "skipped: empty"})
            continue
        status, problems, warnings = case_store.save_labels(case, payload, annotator=name)
        text = f"saved for {name} (labels {status})" if not problems else "not saved: " + "; ".join(problems)
        if warnings:
            text += " - check: " + "; ".join(warnings)
        rows.append({"case_id": block.case_id, "status": text})
    return rows
