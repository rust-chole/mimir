from __future__ import annotations

import json
from typing import Any

from ai import model_config
from ai.openai_client import client


ROLE_JUDGE_VERSION = 2
SPEAKER_ROLE_MODEL = getattr(model_config, "SPEAKER_ROLE_MODEL", model_config.DEFAULT_LUNA_MODEL)
SPEAKER_ROLE_REASONING_EFFORT = getattr(model_config, "SPEAKER_ROLE_REASONING_EFFORT", "low")

ROLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["single", "dual", "triple", "crowd", "unresolved"]},
        "primary_speaker": {"type": ["string", "null"]},
        "secondary_speaker": {"type": ["string", "null"]},
        "human_speakers": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "background_speakers": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"},
    },
    "required": [
        "mode", "primary_speaker", "secondary_speaker", "human_speakers",
        "background_speakers", "confidence", "reason",
    ],
}

INSTRUCTIONS = """
You are MIMIR's cheap speaker-role judge. You DO NOT identify real people.
You only decide which diarization speaker IDs correspond to real conversational
participants versus crowd/laughter/room/game/background audio.

Rules:
- Speaker IDs are arbitrary labels such as speaker_0, speaker_1.
- Accuracy beats forced identity. If participant count is ambiguous, return unresolved; do not force a human prompt.
- If two different speaker IDs each contain coherent lexical human speech,
  classify as dual unless one is clearly only crowd/noise/background.
- A crowd cheer, laughter, scream, chant, game announcer, TV/radio, or very short
  non-conversational background fragment is not a named participant.
- For exactly 2 real humans + crowd, return mode=dual and put crowd IDs in background_speakers.
- For exactly 3 genuine conversational humans, return mode=triple and list all 3 in human_speakers.
- Use crowd only for 4+ genuine conversational humans, not ambient crowd audio.
- Do not infer names, creator identity, gender, age, or any personal attribute.
- Do not invent speaker IDs. Use only IDs provided in the evidence.
- primary_speaker should normally be the strongest recurring real participant.
- secondary_speaker is the other named participant in dual mode.
- Use dual/triple only when the evidence clearly supports exactly 2/3 real conversational humans.
- If evidence is ambiguous between single/dual/triple/background, return unresolved with appropriately low confidence.
""".strip()


def _compact_evidence(segments: list[dict[str, Any]], stats: list[dict[str, Any]]) -> str:
    stat_rows = []
    for item in stats[:8]:
        stat_rows.append({
            "speaker": str(item.get("speaker", "")),
            "speaking_seconds": float(item.get("speaking_seconds", 0.0) or 0.0),
            "segment_count": int(item.get("segment_count", 0) or 0),
            "word_count": int(item.get("word_count", 0) or 0),
            "substantial_segment_count": int(item.get("substantial_segment_count", 0) or 0),
            "noise_segment_ratio": float(item.get("noise_segment_ratio", 0.0) or 0.0),
        })

    seg_rows = []
    for item in segments[:80]:
        text = " ".join(str(item.get("text", "")).split())
        if len(text) > 180:
            text = text[:177] + "..."
        seg_rows.append({
            "start": float(item.get("start", 0.0) or 0.0),
            "end": float(item.get("end", 0.0) or 0.0),
            "speaker": str(item.get("speaker", "")),
            "text": text,
        })

    payload = {
        "speaker_stats": stat_rows,
        "timeline_segments": seg_rows,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _parse_json(text: str) -> dict[str, Any]:
    data = json.loads(str(text).strip())
    if not isinstance(data, dict):
        raise RuntimeError("Speaker role judge object döndürmedi.")
    return data


def classify_speaker_roles(
    segments: list[dict[str, Any]],
    stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Cheap Luna-low classification over already-produced diarization evidence."""
    known = {str(item.get("speaker", "")) for item in stats if str(item.get("speaker", ""))}
    if not known:
        return {
            "status": "fallback",
            "mode": "single",
            "primary_speaker": None,
            "secondary_speaker": None,
            "human_speakers": [],
            "background_speakers": [],
            "confidence": 0.0,
            "reason": "no speaker ids",
        }

    response = client.responses.create(
        model=SPEAKER_ROLE_MODEL,
        reasoning={"effort": SPEAKER_ROLE_REASONING_EFFORT},
        instructions=INSTRUCTIONS,
        input=_compact_evidence(segments, stats),
        text={
            "format": {
                "type": "json_schema",
                "name": "mimir_speaker_role_judge_v2",
                "strict": True,
                "schema": ROLE_SCHEMA,
            }
        },
    )
    raw = _parse_json(getattr(response, "output_text", ""))

    human = [str(x) for x in raw.get("human_speakers", []) if str(x) in known]
    background = [str(x) for x in raw.get("background_speakers", []) if str(x) in known]
    primary = str(raw.get("primary_speaker") or "") or None
    secondary = str(raw.get("secondary_speaker") or "") or None
    if primary not in known:
        primary = None
    if secondary not in known or secondary == primary:
        secondary = None

    mode = str(raw.get("mode", "unresolved"))
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.0) or 0.0)))

    # Normalize the model's structure. The model can judge roles, but it cannot
    # create IDs or make an impossible dual without two known labels.
    if mode == "dual":
        if primary is None and human:
            primary = human[0]
        if secondary is None:
            secondary = next((x for x in human if x != primary), None)
        if not primary or not secondary or len(human) < 2:
            mode = "unresolved"
    elif mode == "triple":
        if primary is None and human:
            primary = human[0]
        if secondary is None:
            secondary = next((x for x in human if x != primary), None)
        if not primary or not secondary or len(dict.fromkeys(human)) < 3:
            mode = "unresolved"
        else:
            human = list(dict.fromkeys(human))[:3]
    elif mode == "single":
        if primary is None and human:
            primary = human[0]
        secondary = None
    elif mode == "crowd":
        if primary is None and human:
            primary = human[0]
        secondary = None
    else:
        mode = "unresolved"
        secondary = None

    return {
        "status": "ok",
        "mode": mode,
        "primary_speaker": primary,
        "secondary_speaker": secondary,
        "human_speakers": human,
        "background_speakers": background,
        "confidence": confidence,
        "reason": " ".join(str(raw.get("reason", "")).split())[:260],
        "model": SPEAKER_ROLE_MODEL,
        "reasoning_effort": SPEAKER_ROLE_REASONING_EFFORT,
        "version": ROLE_JUDGE_VERSION,
    }
