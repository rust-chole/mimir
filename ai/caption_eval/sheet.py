"""One text file for the whole gold set: export it, correct it in any editor, send / import it.

    python -m ai.caption_eval export            -> vod_output/caption_eval/gold_sheet.txt
    (a person corrects each block and marks it [x])
    python -m ai.caption_eval import gold_sheet.txt   -> each case's gold.txt
    python -m ai.caption_eval score-sheet gold_sheet.txt   -> MIMIR's captions vs the corrected text

A block per case:

    === <case_id> [ ]                      [x] = this transcript is checked word for word
    # listen: <audio file> (<seconds> s; <which part of the short>)
    names: Alex, Sam                       verified names said in it (editable)
    # MIMIR: <what MIMIR captioned, one line, never edited>
    <the transcript to correct>            any number of lines

Lines starting with ``#`` are ignored. The sheet holds text only (no audio), so it can be
sent around; ``score-sheet`` needs nothing else to score the published captions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ai.caption_eval import cases as case_store
from ai.caption_eval import metrics, report

SHEET_NAME = "gold_sheet.txt"
_HEADER = re.compile(r"^===\s+(\S+)\s*\[([ xX]?)\]\s*$")
_INTRO = """# MIMIR caption gold sheet ({when}).
# For each short: listen to the file on its "listen:" line and correct the text under the "# MIMIR:" line
# WORD FOR WORD: slang, profanity, repeats and cut-off words as said; names spelled as they should appear;
# [inaudible] where even you cannot tell. Fillers (uh/um), case and punctuation do not matter. Do not trust
# the draft: it is what is being measured. When a short is checked, change its [ ] to [x].
# Lines starting with # are ignored. The file holds text only (no audio).
"""


@dataclass
class Block:
    case_id: str
    done: bool
    names: list[str] = field(default_factory=list)
    mimir: str | None = None
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def _listen_note(case: case_store.Case) -> str:
    source = case.meta.get("source") or {}
    part = ("the story part of the published short, right after the cold open" if source.get("kind") == "final"
            else "the edited clip")
    return f"# listen: {case.audio_path} ({case.duration:.1f} s; {part})"


def export_sheet(*, root: str | Path | None = None, only: Sequence[str] | None = None,
                 out: str | Path | None = None) -> tuple[Path, int]:
    """Write the sheet (current gold text, else MIMIR's draft) -> (path, cases)."""
    cases = case_store.list_cases(root, only)
    if not cases:
        raise FileNotFoundError(f"no cases under {case_store.eval_root(root) / 'cases'}; run `prepare` first")
    parts = [_INTRO.format(when=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"))]
    for case in cases:
        published = case.published_words()
        mimir = " ".join(str(w.get("word", "")) for w in published or [] if str(w.get("word", "")).strip())
        status = case.gold_status()
        body = metrics.gold_body(case.gold_text()).strip() if case.gold_path.is_file() else ""
        if body == "(write the transcript here)":
            body = ""
        parts.append("\n".join([
            f"=== {case.case_id} [{'x' if status == 'ready' else ' '}]",
            _listen_note(case),
            f"names: {', '.join(case.names)}",
            f"# MIMIR: {mimir}" if published is not None else "# MIMIR: (no captions recorded for this case)",
            body or mimir,
            "",
        ]))
    target = Path(out) if out else case_store.eval_root(root) / SHEET_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(parts), encoding="utf-8")
    return target, len(cases)


def parse_sheet(text: str) -> list[Block]:
    blocks: list[Block] = []
    for raw in str(text).replace("\r\n", "\n").splitlines():
        line = raw.strip()
        header = _HEADER.match(line)
        if header:
            blocks.append(Block(header.group(1), header.group(2).lower() == "x"))
            continue
        if not blocks:
            continue
        block = blocks[-1]
        if line.startswith("# MIMIR:"):
            value = line[len("# MIMIR:"):].strip()
            block.mimir = None if value.startswith("(no captions") else value
        elif line.startswith("#") or not line:
            continue
        elif line.lower().startswith("names:"):              # several names lines add up
            for name in (x.strip() for x in line.split(":", 1)[1].split(",")):
                if name and name not in block.names:
                    block.names.append(name)
        else:
            block.lines.append(line)
    return blocks


def import_sheet(path: str | Path, *, root: str | Path | None = None) -> list[dict[str, str]]:
    """Write every finished (or edited) block into its case's gold.txt / names."""
    from ai.caption_eval.editor import save_gold

    known = {c.case_id: c for c in case_store.list_cases(root)}
    rows: list[dict[str, str]] = []
    for block in parse_sheet(Path(path).read_text(encoding="utf-8-sig")):
        case = known.get(block.case_id)
        if case is None:
            rows.append({"case_id": block.case_id, "status": "unknown case (not prepared on this machine)"})
            continue
        current = metrics.gold_body(case.gold_text()).strip() if case.gold_path.is_file() else ""
        if not block.done and block.text == current and block.names == case.names:
            rows.append({"case_id": block.case_id, "status": "unchanged"})
            continue
        if not block.text:
            rows.append({"case_id": block.case_id, "status": "skipped: empty transcript"})
            continue
        status = save_gold(case, block.text, names=", ".join(block.names), done=block.done)
        rows.append({"case_id": block.case_id, "status": f"saved (gold {status})"})
    return rows


def score_sheet(path: str | Path, *, out: str | Path | None = None) -> dict[str, Any]:
    """Score MIMIR's captions against the corrected text of every [x] block; text only, no audio."""
    blocks = parse_sheet(Path(path).read_text(encoding="utf-8-sig"))
    scores: list[metrics.CaseScore] = []
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for block in blocks:
        if not block.done:
            skipped.append({"case_id": block.case_id, "reason": "not marked [x]"})
            continue
        if block.mimir is None:
            skipped.append({"case_id": block.case_id, "reason": "no MIMIR captions in the sheet"})
            continue
        if not metrics.gold_display_tokens(block.text):
            skipped.append({"case_id": block.case_id, "reason": "empty transcript"})
            continue
        score = metrics.score_case(block.case_id, metrics.gold_tokens(block.text, names=block.names),
                                   metrics.hypothesis_tokens([{"word": w} for w in block.mimir.split()]))
        scores.append(score)
        rows.append(score.to_dict())
    result = {"run": f"published (from {Path(path).name})", "system": "published", "overrides": {},
              "git_commit": "", "scored_at": datetime.now().astimezone().isoformat(timespec="seconds"),
              "metrics_version": metrics.METRICS_VERSION, "aggregate": metrics.aggregate(scores), "cases": rows,
              "skipped": skipped, "configurations": []}
    target = Path(out) if out else Path(path).with_suffix(".report.md")
    target.write_text(report.run_report(result), encoding="utf-8")
    result["report_path"] = str(target)
    return result
