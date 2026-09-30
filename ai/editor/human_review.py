"""Human review packet: the final acceptance layer is a person, not a model.

MIMIR publishes only a candidate that passed deterministic QC on the rendered
MP4; a human then accepts or rejects it. This module turns evidence MIMIR
already holds into a short checklist of the places worth looking at, on the
FINAL clock of the published file:

* caption words the evidence could not settle (shown as the primary ear heard them, marked uncertain);
* caption words changed from the primary ear's transcript (by independent-ear agreement or the
  caption judge; who decided is shown);
* verified-name spelling decisions;
* the cold open (window, headline or none) and any effect;
* every disclosed degradation (anything that made the run ``published_degraded``).

Deterministic and cheap: no model call, no media decode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

HUMAN_REVIEW_VERSION = 1


def _clock(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.-"
    minutes, rest = divmod(max(0.0, float(seconds)), 60.0)
    return f"{int(minutes)}:{rest:04.1f}"


def _to_final(doc: Mapping[str, Any], paced: float) -> float | None:
    from ai.editor import intro_renderer

    try:
        return intro_renderer.paced_to_final(doc, float(paced))
    except Exception:
        return None


def _word_rows(truth: Mapping[str, Any]) -> dict[int, list[Any]]:
    rows: dict[int, list[Any]] = {}
    for row in truth.get("words", []) or []:
        if isinstance(row, list) and len(row) >= 4:
            try:
                rows[int(row[0])] = row
            except (TypeError, ValueError):
                continue
    return rows


def build_review_packet(*, published: Path, source: Path, status: str, qc_rows: Sequence[Mapping[str, Any]],
                        timeline_doc: Mapping[str, Any], truth: Mapping[str, Any] | None, headline: str,
                        effect_windows: Sequence[tuple[float, float]] = (),
                        degradations: Sequence[str] = (), repairs: Sequence[str] = ()) -> dict[str, Any]:
    truth = truth or {}
    words = _word_rows(truth)
    focus: list[dict[str, Any]] = []

    for row in truth.get("uncertain", []) or []:
        ids = [int(i) for i in row.get("word_ids", []) or [] if int(i) in words]
        if not ids:
            continue
        final = _to_final(timeline_doc, float(words[ids[0]][2]))
        if final is None:           # trimmed by the restart: not in the published file
            continue
        focus.append({"kind": "caption_uncertain", "final_s": round(final, 3),
                      "text": " ".join(str(words[i][1]) for i in ids), "reason": str(row.get("reason", "")),
                      "near": row.get("near")})

    for row in truth.get("lexical_decisions", []) or []:
        if not row.get("changed"):
            continue
        final = _to_final(timeline_doc, float(row.get("paced_start", 0.0)))
        if final is None:
            continue
        focus.append({"kind": "caption_changed_by_judge", "final_s": round(final, 3),
                      "from": str(row.get("from", "")), "to": str(row.get("to", "")),
                      "decided_by": str(row.get("decided_by", "")), "confidence": row.get("confidence")})

    for row in truth.get("entity_decisions", []) or []:
        index = row.get("word_id")
        if row.get("verdict") != "canonical" or index not in words:
            continue
        final = _to_final(timeline_doc, float(words[index][2]))
        if final is None:
            continue
        focus.append({"kind": "name_spelling_by_judge", "final_s": round(final, 3),
                      "from": str(row.get("token", "")), "to": str(words[index][1]),
                      "confidence": row.get("confidence")})

    focus.sort(key=lambda item: item["final_s"])
    speakers: dict[str, int] = {}
    colours: dict[str, int] = {}
    for row in words.values():
        label = str(row[5] if len(row) > 5 else "") or "(unlabeled lane)"
        speakers[label] = speakers.get(label, 0) + 1
        colour = str(row[6] if len(row) > 6 else "") or "plain"
        colours[colour] = colours.get(colour, 0) + 1
    intro = timeline_doc.get("intro") or {}
    failed = [r for r in qc_rows if r.get("status") == "fail"]
    return {
        "version": HUMAN_REVIEW_VERSION,
        "published": str(published),
        "source": str(source),
        "status": status,
        "qc": {"checks": len(qc_rows), "passed": sum(1 for r in qc_rows if r.get("status") == "pass"),
               "failed": [r.get("check") for r in failed]},
        "cold_open": {"final_window_s": [0.0, round(float(intro.get("duration", 0.0) or 0.0), 3)],
                      "paced_window_s": intro.get("paced"), "headline": headline or ""},
        "effects_final_s": [[round(a, 3), round(b, 3)] for a, b in effect_windows],
        "speakers": speakers,
        "speaker_colors": colours,
        "focus": focus,
        "degradations": list(degradations),
        "repairs": list(repairs),
    }


def render_markdown(packet: Mapping[str, Any]) -> str:
    name = Path(str(packet.get("published", ""))).name
    qc = packet.get("qc") or {}
    status = str(packet.get("status", ""))
    lines = [f"# Human review: {name}", "",
             f"- Status: **{status}**" + (" (disclosed degradations below)" if status != "published" else ""),
             f"- Source: `{packet.get('source', '')}`",
             f"- Final QC on the rendered MP4: {qc.get('passed', 0)}/{qc.get('checks', 0)} checks passed",
             "", "MIMIR's checks are deterministic; **you** are the acceptance step. Times are on the clock of the "
                 "published file.", ""]
    focus = packet.get("focus") or []
    lines.append("## Look here first")
    if not focus:
        lines.append("- Nothing flagged: no caption word was left unsettled, changed from the primary ear's "
                     "transcript or respelled as a name.")
    for item in focus:
        at = _clock(item.get("final_s"))
        kind = item.get("kind")
        if kind == "caption_uncertain":
            near = f" (maybe '{item['near']}')" if item.get("near") else ""
            lines.append(f"- [ ] {at} caption \"{item['text']}\": not settled by the evidence{near}; "
                         f"shown as heard, marked uncertain")
        elif kind == "caption_changed_by_judge":
            lines.append(f"- [ ] {at} caption changed from the primary ear's transcript: \"{item['from']}\" -> "
                         f"\"{item['to']}\" ({item.get('decided_by', '')}, confidence {item.get('confidence')})")
        elif kind == "name_spelling_by_judge":
            lines.append(f"- [ ] {at} name spelling \"{item['from']}\" -> \"{item['to']}\" "
                         f"(confidence {item.get('confidence')})")
    cold = packet.get("cold_open") or {}
    window = cold.get("final_window_s") or [0.0, 0.0]
    lines += ["", "## Cold open",
              f"- [ ] {_clock(window[0])}-{_clock(window[1])}: moving peak footage with its real audio, no speech "
              "captions, hard cut into the story",
              "- [ ] Headline: " + (f"\"{cold['headline']}\" (grounded in the transcript / verified names)"
                                    if cold.get("headline") else "none (no grounded headline was approved)")]
    effects = packet.get("effects_final_s") or []
    lines += ["", "## Effects"]
    lines += [f"- [ ] {_clock(a)}-{_clock(b)}: effect placed away from captions" for a, b in effects] or \
        ["- none"]
    colours = packet.get("speaker_colors") or {}
    lines += ["", "## Speakers"]
    if any(key in colours for key in ("A", "B", "C")):
        counts = ", ".join(f"{key}: {count} word(s)" for key, count in sorted(colours.items()))
        lines.append(f"- [ ] Each voice keeps one caption colour and colour changes follow the speakers ({counts})")
        lines.append("- A colour names a voice, never a person; words with an uncertain voice are neutral.")
    else:
        lines.append("- Speaker colours off (one voice, or speaker turns too uncertain): plain captions.")
    lines += ["", "## Disclosed degradations"]
    lines += [f"- {row}" for row in packet.get("degradations") or []] or ["- none"]
    if packet.get("repairs"):
        lines.append("- deterministic repair applied: " + ", ".join(packet["repairs"]))
    lines += ["", "## Decision", "- [ ] Accept", "- [ ] Reject. Reason:", ""]
    return "\n".join(lines)


def review_path_for(published: Path) -> Path:
    return Path(published).with_suffix(".review.md")


def write_review_packet(published: Path, packet: Mapping[str, Any]) -> Path:
    target = review_path_for(published)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(render_markdown(packet), encoding="utf-8")
    os.replace(temp, target)
    return target


def load_truth(path: Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
