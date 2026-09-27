"""Stage ``caption_truth``: merge the four authorities and FREEZE the result.

lexical (caption_verify + name lock) + timing (caption_verify clock) +
speaker (speakers assignment) + identity (confirmed names only).
The frozen document carries a signature over (id, text, start, end, speaker);
captions and final QC prove the burned captions still equal this truth.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from mimir.config import Settings
from mimir.core.jsonio import content_hash
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.transcript.name_lock import build_roster, letters, lock_names


def truth_rows(words: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [[w["id"], w["text"], round(float(w["start"]), 3), round(float(w["end"]), 3), w.get("speaker") or ""]
            for w in words]


def truth_signature(words: Sequence[Mapping[str, Any]]) -> str:
    return content_hash(truth_rows(words))


def timing_issues(words: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = []
    previous_start = -1.0
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        if end <= start:
            issues.append(f"{word['id']}: non-positive duration")
        if start + 1e-6 < previous_start:
            issues.append(f"{word['id']}: onset earlier than previous word")
        previous_start = start
    return issues


def flagged_names(caption: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    for detail in (caption.get("micro") or {}).get("details", []):
        for reason in detail.get("reasons", []):
            match = re.search(r"known-name spelling:\s*(\S+)\s+vs\s+(\S+)", str(reason))
            if match:
                found.add(letters(match.group(1)) + "|" + letters(match.group(2)))
    return found


class CaptionTruthStage:
    name = "caption_truth"
    version = 1
    deps = ("caption_verify", "speakers", "identity")

    def params(self, settings: Settings) -> Any:
        return {"entities": list(settings.identity.entities), "creator": settings.identity.creator}

    def run(self, ctx: StageContext) -> StageOutput:
        caption = ctx.dep("caption_verify").json("caption_words")
        speakers = ctx.dep("speakers").json("speakers")
        identity = ctx.dep("identity").json("identity")
        participants = {p["id"] for p in speakers["participants"]}
        assignment = speakers.get("assignment", {})
        words = []
        for word in caption["words"]:
            row = assignment.get(word["id"], {})
            speaker = row.get("speaker") or ""
            if speaker and speaker not in participants:
                raise StageError(self.name, f"word {word['id']} assigned to unknown speaker {speaker}")
            words.append({**word, "speaker": speaker, "speaker_confidence": float(row.get("confidence", 0.0)),
                          "speaker_source": row.get("source", "unresolved")})
        roster = build_roster(identity["speakers"], ctx.settings.identity.entities, ctx.settings.identity.creator)
        locked, audit = lock_names(words, roster, flagged_names(caption))
        issues = timing_issues(locked)
        if issues:
            raise StageError(self.name, "caption timing invariants violated: " + "; ".join(issues[:5]))
        for correction in audit["corrections"]:
            ctx.ledger.info("name_lock", f"{correction['token']} -> {correction['to']}",
                            evidence=correction["evidence"])
        names = {sid: row["name"] for sid, row in identity["speakers"].items() if row.get("confirmed")}
        return StageOutput(data={"caption_truth": {
            "window": caption["window"],
            "words": locked,
            "signature": truth_signature(locked),
            "speaker_mode": speakers["mode"],
            "speaker_resolution": speakers.get("resolution", {"status": "confirmed"}),
            "overlaps": speakers.get("overlaps", []),
            "confirmed_names": names,
            "name_lock": audit,
            "quality": caption.get("quality", {}),
        }})
