"""MIMIR production run record: explicit fallback registry, manifest and final gate.

The single production path = frozen caption truth (``caption_truth.py``) +
Pro Edit presentation with the evidence director and pixel proof
(``pro_edit/direction.py``, ``render_proof.py``) + rendered-MP4 QC
(``final_qc.py``) + bounded review (``final_review.py``) + this gate.

* ``--rerender`` (alias ``--force-v6``) re-runs only the presentation /
  render / QC stages; every upstream cache stays valid.
* No silent fallback: every subsystem that degrades is recorded here as
  (subsystem, reason, level) and printed. A fallback that touches caption truth
  or the verified render path BLOCKS publishing; a presentation-quality
  fallback (face tracking, planner/director, placement level, energy) is
  published only as an explicit DEGRADED result, and only after the rendered
  MP4 itself passed QC.
* Final gate: deterministic checks on the FINAL candidate (rendered MP4,
  frozen caption truth, burned captions, camera plan, pixel proofs). A camera
  JSON or an importable module is never accepted as evidence on its own.

Nothing here calls a model, and nothing here mutates caption truth, story or
media: the gate only reads.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

V6_VERSION = 1
_TRUE = {"1", "true", "yes", "on"}

# Stage keys (StageBook) that ``--force-v6`` re-runs. Everything upstream
# (transcription, clip analysis, pacing, timeline, speaker preflight, pacing
# cut, caption ASR, visual support, teaser, intro analysis, memes) keeps its
# cache; downstream renders re-run because their inputs change.
FORCE_V6_STAGES = frozenset({"caption_truth_v6", "pro_edit", "caption_render", "intro_final_base", "meme_render",
                             "publish"})

# Fallbacks that change what the viewer reads or which render was verified.
BLOCKING_FALLBACKS = frozenset({"caption_truth", "pro_edit", "pro_edit_render", "caption_presentation",
                                "caption_presentation_render", "vision_runtime"})

GATE_PASSED = "passed"
GATE_FAILED = "failed"


def v6_requested(override: bool | None = None) -> bool:
    if override is not None:
        return bool(override)
    return str(os.getenv("MIMIR_V6", "0")).strip().casefold() in _TRUE


def caption_entities_from_env() -> list[str]:
    """User-verified extra entities (names/brands/games/places), comma separated."""
    return [" ".join(part.split()) for part in str(os.getenv("MIMIR_CAPTION_ENTITIES", "")).split(",")
            if " ".join(part.split())]


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class V6Run:
    """What the V6 path did in this run (subsystem status + explicit fallbacks)."""

    enabled: bool
    force: bool = False
    subsystems: dict[str, dict[str, Any]] = field(default_factory=dict)
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    gate: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def record(self, subsystem: str, status: str, **info: Any) -> None:
        self.subsystems[subsystem] = {"status": status, **{k: v for k, v in info.items() if v is not None}}

    def fallback(self, subsystem: str, reason: str, level: str) -> dict[str, Any]:
        row = {"subsystem": subsystem, "reason": " ".join(str(reason).split())[:600], "level": level}
        if row not in self.fallbacks:
            self.fallbacks.append(row)
        return row

    def artifact(self, name: str, path: str | Path | None) -> None:
        if path is not None:
            self.artifacts[name] = str(Path(path))

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.gate:
            return "incomplete"
        return str(self.gate.get("status", "incomplete"))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "mimir_v6_manifest", "version": V6_VERSION, "enabled": self.enabled, "force": self.force,
                "status": self.status, "subsystems": self.subsystems, "fallbacks": list(self.fallbacks),
                "gate": self.gate, "artifacts": dict(self.artifacts), "written_at": _now()}

    def summary(self) -> dict[str, Any]:
        failed = [c["check"] for c in self.gate.get("checks", []) if c.get("status") == "fail"]
        return {"status": self.status, "fallbacks": len(self.fallbacks), "failed_checks": failed,
                "subsystems": {k: v.get("status") for k, v in self.subsystems.items()}}

    def console_lines(self) -> list[str]:
        lines = [f"[MIMIR_V6] status={self.status} force={int(self.force)}"]
        for name, info in self.subsystems.items():
            detail = info.get("detail") or info.get("reason") or ""
            lines.append(f"[MIMIR_V6] {name}: {info.get('status')}" + (f" ({detail})" if detail else ""))
        for row in self.fallbacks:
            lines.append(f"[MIMIR_V6_FALLBACK] {row['subsystem']} -> {row['level']}: {row['reason']}")
        for check in self.gate.get("checks", []):
            if check.get("status") != "pass":
                lines.append(f"[MIMIR_V6_GATE] {check['check']}: {check['status']} ({check.get('detail', '')})")
        return lines


def write_manifest(path: str | Path, run: V6Run) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temp, target)
    return target


# ============================================================
# FINAL QUALITY GATE
# ============================================================

_BLOCK = re.compile(r"(\{[^}]*\})")
_DRAWING = re.compile(r"\\p(\d+)")


def burned_caption_tokens(ass_path: str | Path | None) -> list[str]:
    """Visible words of every Dialogue event of the ASS that was burned.

    Override blocks are removed and vector drawings (``\\p1`` .. ``\\p0``,
    e.g. legibility backplates) are skipped, so only rendered text remains."""
    if ass_path is None or not Path(ass_path).is_file():
        return []
    tokens: list[str] = []
    for line in Path(ass_path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        drawing = False
        visible: list[str] = []
        for piece in _BLOCK.split(parts[9]):
            if piece.startswith("{") and piece.endswith("}"):
                for level in _DRAWING.findall(piece):
                    drawing = int(level) > 0
                continue
            if not drawing:
                visible.append(piece)
        text = " ".join(visible).replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
        tokens.extend(t for t in text.split() if t)
    return tokens


def _key(token: str) -> str:
    return "".join(ch for ch in str(token).casefold() if ch.isalnum())


def _check(name: str, status: str, detail: str = "", **data: Any) -> dict[str, Any]:
    return {"check": name, "status": status, "detail": detail, **data}


def _load(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).is_file():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def check_output(final_output: str | Path) -> dict[str, Any]:
    from ai.editor.pro_edit.media import probe_media

    path = Path(final_output)
    if not path.is_file() or path.stat().st_size < 1024:
        return _check("output_file", "fail", f"missing or empty: {path}")
    try:
        info = probe_media(path)
    except Exception as error:
        return _check("output_file", "fail", f"ffprobe failed: {type(error).__name__}: {error}")
    problems = []
    if info.frame_count <= 0 or info.duration_s <= 0:
        problems.append("no frames/duration")
    if not info.has_audio:
        problems.append("no audio stream")
    if problems:
        return _check("output_file", "fail", "; ".join(problems))
    return _check("output_file", "pass", f"{info.width}x{info.height} {info.fps} {info.duration_s:.2f}s "
                                         f"{info.frame_count} frames, audio")


def check_caption_truth(profile_path: str | Path | None, truth_path: str | Path | None) -> list[dict[str, Any]]:
    """Frozen truth unchanged + timing / speaker ownership valid (re-derived from the final profile)."""
    from ai.editor import caption_truth

    if profile_path is None or truth_path is None:
        return [_check("caption_truth_frozen", "fail", "caption truth document or final profile missing")]
    ok, detail = caption_truth.verify_frozen_truth(profile_path, truth_path)
    rows = [_check("caption_truth_frozen", "pass" if ok else "fail", detail)]
    profile = _load(profile_path) or {}
    timing = caption_truth.timing_issues(profile)
    rows.append(_check("word_timing", "pass" if not timing else "fail",
                       "monotonic, inside the clip" if not timing else "; ".join(timing[:5])))
    speakers = caption_truth.speaker_issues(profile)
    rows.append(_check("speaker_ownership", "pass" if not speakers else "fail",
                       "raw ids known, labels human-confirmed" if not speakers else "; ".join(speakers[:5])))
    return rows


def check_burned_names(truth_path: str | Path | None, burned_ass: str | Path | None,
                       display_labels: Sequence[str] = ()) -> dict[str, Any]:
    """The burned captions show only truth words (+ confirmed labels) and never a
    spelling the entity lock replaced (no verified-name contradiction)."""
    truth = _load(truth_path)
    if truth is None:
        return _check("verified_names", "fail", "caption truth document missing")
    burned = burned_caption_tokens(burned_ass)
    if not burned:
        return _check("verified_names", "fail", f"burned caption ASS unreadable: {burned_ass}")
    def keys(text: str) -> set[str]:
        # A truth word may hold several spoken tokens ("date with"); the ASS shows them split.
        return {_key(text), *(_key(part) for part in str(text).split())}

    truth_keys: set[str] = set()
    for row in truth.get("words", []):
        if isinstance(row, list) and len(row) > 1:
            truth_keys |= keys(str(row[1]))
    for label in display_labels:
        truth_keys |= keys(label)
    truth_keys.discard("")
    replaced: set[str] = set()
    for row in (truth.get("entity_lock", {}) or {}).get("corrections", []) or []:
        if isinstance(row, Mapping):
            replaced |= keys(str(row.get("from", "")))
    replaced.discard("")
    contradictions = sorted({t for t in burned if _key(t) in replaced and _key(t) not in truth_keys})
    foreign = sorted({t for t in burned if _key(t) and _key(t) not in truth_keys})
    if contradictions:
        return _check("verified_names", "fail", "burned captions still show replaced spelling(s): "
                      + ", ".join(contradictions[:6]), contradictions=contradictions[:12])
    if foreign:
        return _check("verified_names", "fail", "burned caption words not in the frozen truth: "
                      + ", ".join(foreign[:6]), foreign=foreign[:12])
    corrected = [f"{row.get('from')}->{row.get('to')}" for row in
                 (truth.get("entity_lock", {}) or {}).get("corrections", []) or [] if isinstance(row, Mapping)]
    return _check("verified_names", "pass", f"{len(set(_key(t) for t in burned))} burned word forms == truth"
                  + (f"; canonicalized {', '.join(corrected[:6])}" if corrected else ""))


def check_presentation(manifest_path: str | Path | None, expected_signature: str) -> dict[str, Any]:
    manifest = _load(manifest_path)
    if manifest is None:
        return _check("caption_presentation", "fail", "presentation manifest missing")
    pages = manifest.get("pages", []) or []
    too_many = [p.get("page_id") for p in pages if len(p.get("lines", []) or []) > 2]
    signature = str((manifest.get("truth", {}) or {}).get("caption_signature", ""))
    problems = []
    if too_many:
        problems.append(f"{len(too_many)} page(s) with more than 2 lines")
    if expected_signature and signature != expected_signature:
        problems.append("presentation was built from different caption truth")
    retimed = int((manifest.get("truth", {}) or {}).get("retimed_words", 0) or 0)
    if retimed:
        problems.append(f"{retimed} retimed word(s)")
    metrics = manifest.get("metrics", {}) or {}
    if problems:
        return _check("caption_presentation", "fail", "; ".join(problems))
    return _check("caption_presentation", "pass",
                  f"{len(pages)} pages, max 2 lines, legibility={json.dumps(metrics.get('legibility', {}))}")


def check_camera(prep: Any, direction_required: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved = getattr(prep, "resolved", None)
    if resolved is None:
        return [_check("camera_plan", "fail", "no resolved camera plan")]
    direction = dict(getattr(prep, "direction", {}) or {})
    if direction.get("status") != "directed":
        # The validated planner intent (or a static camera) was rendered instead:
        # a degraded but legitimate edit; its pixels are still proven below.
        rows.append(_check("camera_plan", "fail" if direction_required else "warn",
                           f"evidence direction not applied: {direction.get('reason', direction)}"))
    else:
        rows.append(_check("camera_plan", "pass", f"{len(resolved.ops)} op(s), peak zoom "
                           f"{resolved.metrics.get('peak_zoom', 1.0)}, intents {direction.get('intents', {})}"))
    plan_doc = _load(getattr(getattr(prep, "artifacts", None), "camera_plan", None))
    holds = (plan_doc or {}).get("holds", []) if plan_doc is not None else []
    unexplained = [h for h in holds if not str(h.get("reason", "")).strip()]
    if direction_required and plan_doc is None:
        rows.append(_check("hold_reasons", "fail", "camera plan artifact missing"))
    elif unexplained:
        rows.append(_check("hold_reasons", "fail", f"{len(unexplained)} long static region(s) without a reason"))
    else:
        rows.append(_check("hold_reasons", "pass", f"{len(holds)} long HOLD region(s), all explained"))
    return rows


def untraceable_final_proof(main_proof: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Final-proof record when the published short cannot be proven (None: run the final proof).

    The final is proven by tracing camera frames VERIFIED in the main render.
    When the main proof verified nothing (unavailable / failed / inconclusive),
    the final is not an innocent "no camera op" pass: it is unproven too."""
    status = str((main_proof or {}).get("status", ""))
    if status == "passed":
        return None
    if status == "no_camera_ops":
        return {"status": "no_camera_ops", "samples": [],
                "reason": "the main render has no provable camera change to trace into the final"}
    return {"status": "inconclusive" if status == "inconclusive" else "unavailable", "samples": [],
            "reason": f"main render proof {status or 'missing'}: no verified camera frame to trace into the final"}


def check_proof(name: str, proof: Mapping[str, Any] | None, *, required: bool) -> dict[str, Any]:
    if proof is None:
        return _check(name, "fail" if required else "skip", "proof not run")
    status = str(proof.get("status", ""))
    reason = str(proof.get("reason", ""))
    if status == "passed":
        return _check(name, "pass", reason)
    if status == "no_camera_ops":
        return _check(name, "pass", reason or "no camera op to prove")
    if status == "inconclusive":
        return _check(name, "warn", reason)
    return _check(name, "fail", f"{status}: {reason}")


def check_geometry(proof: Mapping[str, Any] | None) -> dict[str, Any]:
    geometry = (proof or {}).get("story_geometry") if proof else None
    if not geometry:
        return _check("story_regions_visible", "fail", "geometry check not run")
    if geometry.get("status") == "passed":
        return _check("story_regions_visible", "pass", f"{geometry.get('frames_checked', 0)} active frame(s) checked")
    return _check("story_regions_visible", "fail",
                  f"{geometry.get('violation_count', 0)} violation(s): {json.dumps(geometry.get('violations', [])[:2])}")


def final_quality_gate(*, run: V6Run, final_output: str | Path, profile_path: str | Path | None,
                       truth_path: str | Path | None, burned_ass: str | Path | None, prep: Any,
                       main_proof: Mapping[str, Any] | None, final_proof: Mapping[str, Any] | None,
                       intro_proof: Mapping[str, Any] | None, story_intact: tuple[bool, str],
                       intro_handoff: Mapping[str, Any] | None, render_status: str,
                       display_labels: Sequence[str] = (), qc_rows: Sequence[Mapping[str, Any]] = (),
                       review: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic acceptance on the final candidate. Never raises."""
    checks: list[dict[str, Any]] = []

    def guarded(label: str, function: Any, *args: Any, **kwargs: Any) -> None:
        try:
            result = function(*args, **kwargs)
            checks.extend(result if isinstance(result, list) else [result])
        except Exception as error:  # a gate bug is a failed check, never a crash
            checks.append(_check(label, "fail", f"gate error: {type(error).__name__}: {error}"))

    guarded("output_file", check_output, final_output)
    guarded("caption_truth_frozen", check_caption_truth, profile_path, truth_path)
    guarded("verified_names", check_burned_names, truth_path, burned_ass, display_labels)
    presentation = getattr(getattr(prep, "artifacts", None), "caption_presentation", None)
    integrity = getattr(prep, "caption_integrity", None)
    if getattr(prep, "presentation_ass", None) is not None:
        guarded("caption_presentation", check_presentation, presentation,
                getattr(integrity, "profile_signature", ""))
    else:
        checks.append(_check("caption_presentation", "fail", "V6 captions were not rendered by the presentation layer"))
    ok, detail = story_intact
    checks.append(_check("story_preserved", "pass" if ok else "fail", detail))
    handoff = dict(intro_handoff or {})
    if handoff:
        checks.append(_check("intro_handoff", "pass" if handoff.get("status") == "verified" else "fail",
                             str(handoff.get("reason", handoff.get("status", "")))))
    guarded("camera_plan", check_camera, prep, False)
    guarded("story_regions_visible", check_geometry, main_proof)
    checks.append(check_proof("camera_pixels_main", main_proof, required=True))
    checks.append(check_proof("camera_pixels_final", final_proof, required=True))
    if intro_proof is not None:
        checks.append(check_proof("camera_pixels_intro", intro_proof, required=False))
    checks.append(_check("v6_render_path", "pass" if render_status == "pro_edit" else "fail",
                         f"main render status={render_status}"))
    blocking = [r for r in run.fallbacks if r["subsystem"] in BLOCKING_FALLBACKS]
    degraded = [r for r in run.fallbacks if r["subsystem"] not in BLOCKING_FALLBACKS]
    if blocking:
        checks.append(_check("no_blocking_fallback", "fail", "; ".join(f"{r['subsystem']}->{r['level']}"
                                                                       for r in blocking[:6]),
                             fallbacks=list(blocking)))
    else:
        checks.append(_check("no_blocking_fallback", "pass", "caption truth and the verified render path held"))
    if degraded:
        checks.append(_check("presentation_degradations", "warn", "; ".join(f"{r['subsystem']}->{r['level']}"
                                                                            for r in degraded[:6]),
                             fallbacks=list(degraded)))
    checks.extend(dict(row) for row in qc_rows)
    if review:
        status = str(review.get("status", ""))
        if status == "reviewed" and not review.get("warnings"):
            checks.append(_check("final_review", "pass", str(review.get("summary", "")) or "no finding"))
        else:
            detail = "; ".join(str(w) for w in (review.get("warnings") or [])[:4]) or str(review.get("reason", status))
            checks.append(_check("final_review", "warn", detail, review=dict(review)))
    failed = [c for c in checks if c["status"] == "fail"]
    warned = [c for c in checks if c["status"] == "warn"]
    status = GATE_FAILED if failed else ("passed_with_warnings" if warned else GATE_PASSED)
    return {"version": V6_VERSION, "status": status, "checks": checks, "failed": [c["check"] for c in failed],
            "warnings": [c["check"] for c in warned], "at": _now()}
