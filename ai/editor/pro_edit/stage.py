"""Pipeline boundary for Pro Edit (the only functions the pipeline calls).

Runs AFTER MIMIR has fixed story, pacing, captions and the mandatory intro
selection, and BEFORE caption burn-in and intro composition:

    prepare_pro_edit    context (+ tracks, spans, caption region, intro clock)
                        -> planner (model | replay | rules | static)
                        -> validator -> [V6: evidence director -> validator]
                        -> resolver (main + intro camera) [V6: camera plan + HOLD reasons]
                        -> caption presentation (pages/lines/emphasis over
                           the same caption truth; its ASS becomes the
                           caption safe region the camera avoids)
    render_with_fallback    main: camera -> subtitles, one encode
    render_intro_source     clean paced clip with camera only on the teaser
    prove_* (V6)            planned camera vs rendered main / intro / final pixels

Fallback hierarchy: planner unavailable -> static; bad events -> sanitized or
dropped; event unresolvable -> original pixels; caption presentation invalid
-> baseline ASS; render/validation/integrity failure -> camera + baseline ASS
-> the existing MIMIR render. No Pro Edit error escapes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ai.editor.pro_edit.camera import base_window
from ai.editor.pro_edit.caption_guard import (
    CaptionIntegrity,
    CaptionIntegrityError,
    caption_safe_region,
    load_profile,
    trusted_display_names,
)
from ai.editor.pro_edit import caption_background as background_mod
from ai.editor.pro_edit.caption_action import derive_action_regions, detect_people_in_spans
from ai.editor.pro_edit.caption_brand import load_brand
from ai.editor.pro_edit.caption_layout import UNKNOWN_LAYOUT, classify_layout
from ai.editor.pro_edit.caption_occupancy import OutputMappedOccupancy, load_or_analyze
from ai.editor.pro_edit.caption_placement import PlacementEvidence, TimedBox
from ai.editor.pro_edit.caption_platform import NormBox, load_platform_profile
from ai.editor.pro_edit.caption_presentation import (
    CAPTION_PRESENTATION_VERSION,
    apply_effect_overrides,
    CAPTION_STYLE_PACK,
    CAPTION_STYLE_PACK_VERSION,
    CaptionPresentation,
    build_presentation,
    check_intro_handoff,
    mark_caption_channel,
    write_presentation,
)
from ai.editor.pro_edit.config import ProEditConfig
from ai.editor.pro_edit.context import ContextInputs, EditContext, build_edit_context
from ai.editor.pro_edit.direction import DIRECTION_VERSION, DirectionReport, direct_plan, explain_holds
from ai.editor.pro_edit import editorial_energy
from ai.editor.pro_edit.diagnostics import (
    ArtifactPaths,
    Diagnostics,
    event_blocks,
    load_json,
    now_iso,
    plan_block,
    summarize_issues,
    write_artifacts,
    write_json_atomic,
)
from ai.editor.pro_edit.errors import (
    EditContextError,
    PlannerOutputError,
    PlannerUnavailableError,
    ProEditError,
)
from ai.editor.pro_edit.executor import RENDERER_VERSION, render_camera_captions
from ai.editor.pro_edit.font_metrics import parse_font
from ai.editor.pro_edit.ffmpeg_filters import FILTER_BUILDER_VERSION, FfmpegCapabilities, probe_capabilities
from ai.editor.pro_edit.intro_timeline import IntroTimeline, build_intro_timeline, hook_text_band, teaser_signature
from ai.editor.pro_edit.media import MediaInfo, probe_media
from ai.editor.pro_edit.planner import (
    EditPlanner,
    PlannerOutcome,
    ProviderEditPlanner,
    RuleBasedEditPlanner,
    StaticEditPlanner,
    outcome_from_cached,
)
from ai.editor.pro_edit.presets import PRESET_ENGINE_VERSION, ResolvedPlan, resolve_intro_plan, resolve_plan
from ai.editor.pro_edit import render_proof
from ai.editor.pro_edit.providers import OpenAIResponsesProvider, ReplayProvider
from ai.editor.pro_edit.schema import EDIT_PLAN_SCHEMA_VERSION, plan_hash
from ai.editor.pro_edit.story import source_story_signature
from ai.editor.pro_edit.style import get_style_pack
from ai.editor.pro_edit.subjects import JsonSidecarSubjectProvider, NullSubjectProvider, SubjectTrackProvider
from ai.editor.pro_edit.timebase import ClipTimelineMap, TimelineDomain, Timestamp
from ai.editor.pro_edit.validator import validate_plan

STATUS_READY = "ready"
STATUS_STATIC = "static"
STATUS_FALLBACK = "fallback"


@dataclass(frozen=True)
class ProEditRequest:
    config: ProEditConfig
    timeline_path: Path
    clip_index: int
    edited_clip_path: Path
    caption_path: Path
    output_path: Path
    artifact_dir: Path
    input_signature: str
    analysis_clip: Mapping[str, Any] | None = None
    speaker_profile_path: Path | None = None
    video_report_path: Path | None = None
    teaser_record: Mapping[str, Any] | None = None
    intro_record: Mapping[str, Any] | None = None
    intro_output_path: Path | None = None
    force: bool = False
    # --force-v6: recompute evidence/plan/render but keep the planner's intent cache
    # when its semantic input is unchanged (no re-billed model call).
    reuse_plan_cache: bool = False
    planner: EditPlanner | None = None
    capabilities: FfmpegCapabilities | None = None
    subject_provider: SubjectTrackProvider | None = None


@dataclass
class ProEditPreparation:
    status: str
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    output_path: Path | None = None
    plan_id: str = ""
    render_options: dict[str, Any] = field(default_factory=dict)
    resolved: ResolvedPlan | None = None
    intro_resolved: ResolvedPlan | None = None
    intro_output_path: Path | None = None
    intro_timeline: IntroTimeline | None = None
    media: MediaInfo | None = None
    artifacts: ArtifactPaths | None = None
    model_calls: int = 0
    caps: FfmpegCapabilities | None = None
    interpolation: str = "cubic"
    keep_failed: bool = False
    caption_integrity: CaptionIntegrity | None = None
    source_story_signature: str = ""
    story_signature: str = ""
    timeline_path: Path | None = None
    clip_index: int = 1
    speaker_profile_path: Path | None = None
    caption_path: Path | None = None
    planner_cache_key: str = ""
    presentation_ass: Path | None = None
    caption_fonts_dir: str | None = None
    presentation_signature: str = ""
    captions: dict[str, Any] = field(default_factory=dict)
    energy: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    v6: bool = False
    direction: dict[str, Any] = field(default_factory=dict)
    context: EditContext | None = None

    @property
    def ready(self) -> bool:
        return self.status == STATUS_READY and self.resolved is not None and self.output_path is not None

    @property
    def intro_ready(self) -> bool:
        return (self.status in (STATUS_READY, STATUS_STATIC) and self.intro_resolved is not None
                and self.intro_output_path is not None and not self.intro_resolved.path.is_identity)

    def summary(self) -> dict[str, Any]:
        return {"status": self.status, "reason": self.reason, "plan_id": self.plan_id,
                "model_calls": self.model_calls, "intro_camera": self.intro_ready,
                "story_signature": self.story_signature[:16],
                "caption_signature": (self.caption_integrity.profile_signature[:16]
                                      if self.caption_integrity else ""),
                "captions": dict(self.captions),
                "tracking": dict(self.tracking),
                **({"direction": dict(self.direction)} if self.v6 else {})}


# ============================================================
# SELECTION
# ============================================================

def select_planner(config: ProEditConfig) -> EditPlanner:
    if config.planner == "rules":
        return RuleBasedEditPlanner()
    if config.planner == "static":
        return StaticEditPlanner()
    if config.planner == "replay" and config.replay_path:
        return ProviderEditPlanner(ReplayProvider(config.replay_path))
    return ProviderEditPlanner(OpenAIResponsesProvider(config.model, config.reasoning_effort,
                                                       timeout_s=config.planner_timeout_s))


def _scene_changes(report_path: Path | None) -> list[float]:
    from ai.editor.pro_edit.context import visual_report_body

    data = visual_report_body(load_json(report_path)) if report_path else None
    if not data:
        return []
    times: list[float] = []
    support = data.get("editing_support", {}) if isinstance(data.get("editing_support"), dict) else {}
    for value in support.get("scene_change_times", []) or []:
        try:
            times.append(float(value))
        except (TypeError, ValueError):
            continue
    return sorted(times)


def select_subject_provider(request: ProEditRequest, media: MediaInfo, artifacts: ArtifactPaths) -> SubjectTrackProvider:
    if request.subject_provider is not None:
        return request.subject_provider
    config = request.config
    if config.tracker == "off":
        return NullSubjectProvider()
    if config.subjects_path and config.tracker in ("auto", "sidecar"):
        return JsonSidecarSubjectProvider(config.subjects_path)
    from ai.editor.pro_edit.vision.provider import LocalVisionSubjectProvider

    return LocalVisionSubjectProvider(media.path, width=media.width, height=media.height,
                                      cache_path=artifacts.subjects,
                                      scene_changes=_scene_changes(request.video_report_path), force=request.force)


# ============================================================
# PREPARE
# ============================================================

# Bumped only when previously cached plans become semantically invalid. Compatible
# extensions (e.g. optional emphasis_reasons) keep the epoch: old plans stay valid.
PLANNER_CACHE_EPOCH = 1


def planner_cache_key(context: EditContext, config: ProEditConfig) -> str:
    """Plan cache identity = the true semantic planning input: the payload sent
    to the planner + planner kind/model/effort + style + epoch. Presentation,
    placement, brand, fonts and render code never enter it."""
    from ai.editor.pro_edit.request import build_payload

    style = get_style_pack(config.style)
    blob = json.dumps({"payload": build_payload(context, style), "planner": config.planner,
                       "model": config.model if config.planner == "model" else "",
                       "effort": config.reasoning_effort if config.planner == "model" else "",
                       "replay": (config.replay_path or "") if config.planner == "replay" else "",
                       "style": [style.name, style.version], "schema": EDIT_PLAN_SCHEMA_VERSION,
                       "epoch": PLANNER_CACHE_EPOCH}, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _plan_outcome(request: ProEditRequest, context: EditContext, prep: ProEditPreparation) -> PlannerOutcome:
    style = get_style_pack(request.config.style)
    artifacts = prep.artifacts
    assert artifacts is not None
    cached = None if (request.force and not request.reuse_plan_cache) else load_json(artifacts.plan)
    semantic_key = prep.planner_cache_key
    same_input = cached is not None and (
        (semantic_key and cached.get("planner_cache_key") == semantic_key)
        or cached.get("input_signature") == request.input_signature)
    if (cached and same_input
            and int(cached.get("schema_version", -1)) == EDIT_PLAN_SCHEMA_VERSION
            and cached.get("style") == {"name": style.name, "version": style.version}
            and isinstance(cached.get("plan"), dict)):
        try:
            return outcome_from_cached(cached["plan"], context, style, str(cached.get("planner", "cache")))
        except PlannerOutputError as error:
            prep.warnings.append(f"Pro Edit cached plan ignored: {error}")
    planner = request.planner or select_planner(request.config)
    try:
        return planner.plan(context, style)
    except (PlannerUnavailableError, PlannerOutputError) as error:
        prep.warnings.append(
            f"Pro Edit planner ({getattr(planner, 'name', 'planner')}) unavailable/invalid; static fallback "
            f"(existing MIMIR framing). Detay: {error}")
        return StaticEditPlanner().plan(context, style)


def _write_exchanges(outcome: PlannerOutcome, artifacts: ArtifactPaths) -> None:
    """Debug artifacts: request + raw response (no keys, no headers)."""
    if not outcome.exchanges:
        return
    first_request, first_response = outcome.exchanges[0]
    request_doc = first_request.to_artifact()
    response_doc = first_response.to_artifact()
    if len(outcome.exchanges) > 1:
        repair_request, repair_response = outcome.exchanges[1]
        request_doc["repair"] = repair_request.to_artifact()
        response_doc["repair"] = repair_response.to_artifact()
    write_json_atomic(artifacts.request, request_doc)
    write_json_atomic(artifacts.raw_response, response_doc)


def _file_signature(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


EVIDENCE_BIN_S = 0.5


def _output_box(x0: float, y0: float, x1: float, y1: float, width: int, height: int,
                base: tuple[int, int, int, int]) -> NormBox | None:
    """Source-normalized box -> output-normalized box (base window crop); None if outside."""
    bx, by, bw, bh = base
    ox0, oy0 = (x0 * width - bx) / bw, (y0 * height - by) / bh
    ox1, oy1 = (x1 * width - bx) / bw, (y1 * height - by) / bh
    ox0, oy0, ox1, oy1 = max(0.0, ox0), max(0.0, oy0), min(1.0, ox1), min(1.0, oy1)
    if ox1 - ox0 <= 1e-4 or oy1 - oy0 <= 1e-4:
        return None
    return NormBox(ox0, oy0, ox1, oy1)


def _track_boxes(track: Any, kind: str, start: float, end: float, width: int, height: int,
                 base: tuple[int, int, int, int]) -> list[TimedBox]:
    """Union of reliable samples per 0.5 s bin (bounded evidence size)."""
    bins: dict[int, list[float]] = {}
    for sample in track.samples:
        if sample.confidence < 0.5 or not start <= sample.t <= end:
            continue
        key = int(sample.t // EVIDENCE_BIN_S)
        box = [sample.cx - sample.width / 2, sample.cy - sample.height / 2,
               sample.cx + sample.width / 2, sample.cy + sample.height / 2]
        current = bins.get(key)
        bins[key] = box if current is None else [min(current[0], box[0]), min(current[1], box[1]),
                                                  max(current[2], box[2]), max(current[3], box[3])]
    rows = []
    for key, (x0, y0, x1, y1) in sorted(bins.items()):
        out = _output_box(x0, y0, x1, y1, width, height, base)
        if out is not None:
            rows.append(TimedBox(key * EVIDENCE_BIN_S, (key + 1) * EVIDENCE_BIN_S, out, kind))
    return rows


def _placement_evidence(request: ProEditRequest, prep: ProEditPreparation, context: EditContext,
                        media: MediaInfo, platform: Any) -> tuple[PlacementEvidence, dict[str, Any], Any]:
    """Faces, story geometry, action regions, layout, UI/text occupancy, activity and
    platform zones in OUTPUT coordinates (+ the background map for legibility)."""
    config = request.config
    width, height = context.clip.width, context.clip.height
    base = base_window(width, height, config.output_profile)
    boxes: list[TimedBox] = []
    sources: list[str] = []
    info: dict[str, Any] = {}
    for track in context.subject_tracks:
        if track.kind == "face":
            boxes.extend(_track_boxes(track, "face", 0.0, context.clip.duration_s, width, height, base))
    if any(b.kind == "face" for b in boxes):
        sources.append("faces")
    by_id = {t.subject_id: t for t in context.subject_tracks}
    for span in context.spans:
        for region in span.required_regions:
            out = _output_box(region.x0, region.y0, region.x1, region.y1, width, height, base)
            if out is not None:
                boxes.append(TimedBox(span.start, span.end, out, "story_region"))
        for subject_id in span.required_subject_ids:
            if subject_id in by_id:
                boxes.extend(_track_boxes(by_id[subject_id], "story_subject", span.start, span.end, width, height,
                                          base))
    if any(b.kind.startswith("story") for b in boxes):
        sources.append("story_geometry")
    occupancy = None
    raw_occupancy = None
    full_frame = base == (0, 0, width, height)
    if config.caption_activity:
        assert prep.artifacts is not None
        try:
            found, status = load_or_analyze(prep.artifacts.caption_occupancy, media, context.clip_identity,
                                            force=request.force)
            raw_occupancy = found
            occupancy = found if full_frame else OutputMappedOccupancy(found, base, (width, height))
            info["activity"] = {"status": status, "samples": len(found.times), "confidence": round(found.confidence, 3)}
            sources.append("activity")
        except Exception as error:  # activity is optional evidence; placement falls back to faces/story/platform
            info["activity"] = {"status": "failed", "reason": f"{type(error).__name__}: {str(error)[:200]}"}
    else:
        info["activity"] = {"status": "disabled"}
    # V5: background map (legibility + TEXT_LIKE occupancy + panel lines), one sparse decode.
    background = None
    if config.caption_legibility or config.caption_ui or config.caption_layout:
        assert prep.artifacts is not None
        try:
            found_bg, status = background_mod.load_or_analyze(prep.artifacts.caption_background, media,
                                                              context.clip_identity, force=request.force)
            background = found_bg
            info["background"] = {"status": status, "samples": len(found_bg.samples),
                                  "confidence": round(found_bg.confidence, 3),
                                  "kind": "luminance/edges + TEXT_LIKE occupancy (no OCR)"}
        except Exception as error:  # optional evidence: legibility / UI / layout are skipped
            info["background"] = {"status": "failed", "reason": f"{type(error).__name__}: {str(error)[:200]}"}
    else:
        info["background"] = {"status": "disabled"}
    mapped_bg = None
    if background is not None:
        mapped_bg = background if full_frame else background_mod.OutputMappedBackground(background, base,
                                                                                        (width, height))
    ui = mapped_bg if (config.caption_ui and mapped_bg is not None) else None
    if ui is not None:
        sources.append("text_like_occupancy")
    layout = UNKNOWN_LAYOUT
    if config.caption_layout:
        try:
            layout = classify_layout(context.subject_tracks, context.clip.duration_s, occupancy=raw_occupancy,
                                     background=background)
        except Exception as error:  # layout is optional; UNKNOWN keeps V4 weights
            layout = dataclasses.replace(UNKNOWN_LAYOUT, evidence={"reason": f"{type(error).__name__}: {error}"})
        info["layout"] = layout.to_dict()
        if layout.facecam_box is not None:
            out = _output_box(*layout.facecam_box, width, height, base)
            if out is not None:
                boxes.append(TimedBox(0.0, context.clip.duration_s, out, "layout_region"))
                sources.append("layout_facecam")
    else:
        info["layout"] = {"status": "disabled"}
    objects = []
    if config.caption_objects == "hog_person":
        objects, object_status = detect_people_in_spans(media.path, media.width, media.height, context.spans)
        info["objects"] = {"mode": config.caption_objects, "status": object_status,
                           "classes": ["person (upright full body)"]}
    else:
        info["objects"] = {"mode": "off", "classes": []}
    actions = derive_action_regions(context.spans, raw_occupancy, objects=objects)
    for action in actions:
        out = _output_box(*action.bbox, width, height, base)
        if out is not None:
            boxes.append(TimedBox(action.start, action.end, out, "action_region", weight=action.confidence))
    info["action_regions"] = [a.to_dict() for a in actions[:50]]
    if actions:
        sources.append("action_regions")
    if platform.reserved_regions or platform.preferred_caption_regions:
        sources.append(f"platform:{platform.profile_id}")
    evidence = PlacementEvidence(tuple(boxes), occupancy, tuple(context.scene_changes), platform, tuple(sources),
                                 ui=ui, multipliers=dict(layout.weights), layout=layout.layout.value)
    return evidence, info, (mapped_bg if config.caption_legibility else None)


def _brand_font(brand: Any, profile: Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Custom brand font -> (build kwargs, note). Falls back to V3.1 font resolution."""
    if not brand.optional_font_path:
        return {}, ""
    try:
        metrics = parse_font(brand.optional_font_path, family=brand.font_family)
    except (OSError, ValueError) as error:
        return {}, f"brand font unreadable ({type(error).__name__}); V3.1 font resolution used"
    words = [str(w.get("word", "")) + str(w.get("speaker_label", "")) for w in (profile or {}).get("words", []) or []
             if isinstance(w, dict)]
    words += list(trusted_display_names(profile).values())   # names the renderer may print
    missing = sorted({ch for text in words for ch in text if not metrics.covers(ch)})
    if missing:
        return {}, f"brand font lacks glyphs {''.join(missing[:12])!r}; V3.1 font resolution used"
    family = metrics.resolved_family or brand.font_family
    return {"metrics": metrics, "font_family": family, "bold": metrics.weight_class >= 600,
            "fonts_dir": str(Path(brand.optional_font_path).resolve().parent)}, f"brand font {family}"


def _prepare_captions(request: ProEditRequest, prep: ProEditPreparation, clip_timeline: Mapping[str, Any],
                      plan: Any, output_size: tuple[int, int], context: EditContext | None = None,
                      media: MediaInfo | None = None) -> CaptionPresentation | None:
    """Caption presentation over the SAME truth.

    Fallback ladder: full V4 placement (faces/story/activity/platform) ->
    V3.1 placement with the configured brand -> pure V3.1 presentation ->
    the existing caption ASS (baseline).
    """
    config = request.config
    if not config.captions:
        prep.captions = {"status": "disabled", "reason": "MIMIR_PRO_EDIT_CAPTIONS=0"}
        return None
    assert prep.artifacts is not None and prep.caption_integrity is not None
    profile = load_profile(request.speaker_profile_path)
    platform, platform_notes = load_platform_profile(
        config.caption_platform, config.caption_platform_file, variant=config.caption_platform_variant,
        device=config.caption_platform_device, description=config.caption_description)
    brand, brand_notes = load_brand(config.caption_brand)
    font_kwargs, font_note = _brand_font(brand, profile)
    for note in (*platform_notes, *brand_notes, *([font_note] if "used" in font_note else [])):
        prep.warnings.append(f"Pro Edit captions: {note}")
    fonts_dir = font_kwargs.pop("fonts_dir", None)
    # Verified names = identities the caption renderer may print (captions V24: human-confirmed only).
    verified = list(trusted_display_names(profile).values())
    if config.v6:
        # V6: + the verified entity roster frozen by Caption Truth V6 (user-verified creator /
        # entities, full multi-word identities) for name emphasis reasons and name-safe breaks.
        lock_audit = (profile or {}).get("participant_name_lock") if isinstance(profile, Mapping) else None
        for row in (lock_audit or {}).get("roster", []) if isinstance(lock_audit, Mapping) else []:
            identity = " ".join(str(row.get("identity", "")).split()) if isinstance(row, Mapping) else ""
            if identity and identity.casefold() not in {v.casefold() for v in verified}:
                verified.append(identity)
    common = dict(profile=profile, clip_timeline=clip_timeline, plan=plan, width=output_size[0],
                  height=output_size[1], spans=context.spans if context is not None else (), verified_names=verified,
                  shaper_mode=config.caption_shaper)
    evidence_info: dict[str, Any] = {}
    fallbacks: list[str] = []
    attempts: list[tuple[str, dict[str, Any]]] = []
    if context is not None and media is not None:
        try:
            evidence, evidence_info, legibility_bg = _placement_evidence(request, prep, context, media, platform)
            attempts.append(("v5_placement", dict(brand=brand, platform=platform, evidence=evidence,
                                                  background=legibility_bg, **font_kwargs)))
            if legibility_bg is not None or evidence.ui is not None:
                # Same evidence without the V5 background layers (V4 behaviour) if they fail.
                attempts.append(("v4_placement", dict(
                    brand=brand, platform=platform,
                    evidence=dataclasses.replace(evidence, ui=None, multipliers={}, layout="",
                                                 boxes=tuple(b for b in evidence.boxes
                                                             if b.kind in ("face", "story_region", "story_subject"))),
                    **font_kwargs)))
        except Exception as error:  # evidence is optional
            fallbacks.append(f"evidence: {type(error).__name__}: {str(error)[:200]}")
    attempts.append(("v31_placement", dict(brand=brand, platform=platform, **font_kwargs)))
    attempts.append(("v31_presentation", {}))
    presentation = None
    level = ""
    for label, kwargs in attempts:
        try:
            presentation = build_presentation(**common, **kwargs)
            ass_path = write_presentation(presentation, prep.artifacts.caption_ass)
            check_intro_handoff(ass_path, Path(request.caption_path))
            level = label
            break
        except Exception as error:  # next rung of the ladder
            fallbacks.append(f"{label}: {type(error).__name__}: {str(error)[:300]}")
            presentation = None
    if presentation is None:
        prep.captions = {"status": "baseline_ass", "reason": " | ".join(fallbacks)[:600], "fallbacks": fallbacks}
        prep.warnings.append(f"Pro Edit caption presentation kapalı; mevcut caption ASS kullanılıyor. "
                             f"Detay: {prep.captions['reason']}")
        return None
    if level == "v31_presentation":
        fonts_dir = None
    manifest = presentation.manifest(caption_signature=prep.caption_integrity.profile_signature)
    manifest["evidence"] = evidence_info
    manifest["fallbacks"] = fallbacks
    manifest["level"] = level
    write_json_atomic(prep.artifacts.caption_presentation, manifest)
    prep.presentation_ass = ass_path.resolve()
    prep.presentation_signature = _file_signature(ass_path)
    prep.caption_fonts_dir = fonts_dir
    metrics = manifest["metrics"]
    prep.captions = {
        "status": "presentation", "version": CAPTION_PRESENTATION_VERSION, "level": level,
        "profile": manifest["layout_profile"], "brand": presentation.brand.profile_id,
        "platform": f"{presentation.platform.profile_id}{'' if presentation.platform.verified else ' (unverified)'}",
        "pages": metrics["pages"], "avg_words_per_page": metrics["avg_words_per_page"],
        "max_words_per_page": metrics["max_words_per_page"], "two_line_pages": metrics["two_line_pages"],
        "speaker_pages": metrics["speaker_pages"], "emphasized_words": metrics["emphasized_words"],
        "invalid_emphasis_ignored": metrics["invalid_emphasis_ignored"], "impact_pages": metrics["impact_pages"],
        "overflow_prevented": metrics["fit_scaled_pages"], "terminal_flash_pages": metrics["terminal_flash_pages"],
        "max_simultaneous_words": metrics["max_simultaneous_words"], "safe_band": metrics["safe_band"],
        "placement_zones": metrics["placement_zones"], "placement_switches": metrics["placement_switches"],
        "activity": evidence_info.get("activity", {}).get("status", "not_run"),
        "primitives": metrics["primitives"], "shaper": presentation.shaper_note,
        "legibility": metrics["legibility"],
        "layout": evidence_info.get("layout", {}).get("layout", "not_run"),
        "background": evidence_info.get("background", {}).get("status", "not_run"),
        "fallbacks": len(fallbacks),
        "font": manifest["font"]["resolved_family"] + (" (substitute)" if manifest["font"]["substituted"] else ""),
        "font_source": manifest["font"]["source"], "ass": str(prep.presentation_ass),
    }
    return presentation


def _coordinate_energy(request: ProEditRequest, prep: ProEditPreparation, context: EditContext, plan: Any,
                       resolved: ResolvedPlan, presentation: CaptionPresentation | None, style: Any
                       ) -> tuple[ResolvedPlan, CaptionPresentation | None]:
    """Cross-channel energy pass (after captions AND camera; see editorial_energy)."""
    config = request.config
    assert prep.artifacts is not None
    table = dict(config.energy_table) or dict(editorial_energy.ENERGY_TABLE)
    intro = prep.intro_timeline
    transition = (intro.main_restart, intro.main_restart + intro.transition) if intro is not None else None
    events = editorial_energy.collect_events(presentation=presentation, resolved=resolved, plan=plan,
                                             recommendations=resolved.recommendations, transition=transition,
                                             table=table)
    result = editorial_energy.coordinate(events, presentation=presentation, window_ms=config.energy_window_ms,
                                         budget=config.energy_budget, table=table)
    if result.caption_overrides and presentation is not None and prep.presentation_ass is not None:
        updated = apply_effect_overrides(presentation, result.caption_overrides, result.caption_notes)
        ass_path = write_presentation(updated, prep.artifacts.caption_ass)
        check_intro_handoff(ass_path, Path(request.caption_path))
        previous = load_json(prep.artifacts.caption_presentation) or {}
        manifest = updated.manifest(caption_signature=prep.caption_integrity.profile_signature
                                    if prep.caption_integrity else "")
        for key in ("evidence", "fallbacks", "level"):
            if key in previous:
                manifest[key] = previous[key]
        write_json_atomic(prep.artifacts.caption_presentation, manifest)
        prep.presentation_ass = ass_path.resolve()
        prep.presentation_signature = _file_signature(ass_path)
        prep.captions["energy_degraded_pages"] = len(result.caption_overrides)
        presentation = updated
        result.applied["captions"] = len(result.caption_overrides)
    if result.camera_motions:
        try:
            softened = resolve_plan(editorial_energy.soften_plan(plan, result.camera_motions), context, style,
                                    output_profile=config.output_profile, strict_center=config.v6)
            resolved = dataclasses.replace(softened, recommendations=resolved.recommendations)
            result.applied["camera"] = {k: v.value for k, v in sorted(result.camera_motions.items())}
        except ProEditError as error:
            result.applied["camera"] = f"kept original camera ({type(error).__name__}: {str(error)[:200]})"
    if result.dropped_recommendations:
        resolved = dataclasses.replace(resolved, recommendations=editorial_energy.mark_recommendations(
            resolved.recommendations, result.dropped_recommendations))
        result.applied["recommendations_dropped"] = len(result.dropped_recommendations)
    report = result.to_dict()
    write_json_atomic(prep.artifacts.energy, report)
    prep.energy = {"status": "applied" if result.changed else "within_budget",
                   "max_load": [result.max_load_before, result.max_load_after], "budget": result.budget,
                   "window_ms": config.energy_window_ms, "degradations": len(result.decisions),
                   "unresolved": len(result.unresolved),
                   "signature": hashlib.sha256(json.dumps(report, sort_keys=True).encode("utf-8")).hexdigest()[:16]}
    return resolved, presentation


def _direct(plan: Any, context: EditContext, style: Any) -> tuple[Any, DirectionReport, str]:
    """V6: evidence-directed copy of the validated plan; the same validator keeps
    policy/density authority (and decides which added emphasis may stay)."""
    def validate(candidate: Any) -> Any:
        return validate_plan(candidate, context, style)

    directed, report = direct_plan(plan, context, style, validate=validate)
    checked = validate(directed)
    return checked.raise_if_fatal(), report, checked.status.value


def camera_plan_document(report: DirectionReport, resolved: ResolvedPlan) -> dict[str, Any]:
    """Compact V6 camera plan: intent + reason per rendered op, and every HOLD reason."""
    intents: dict[str, dict[str, Any]] = {row["event_id"]: row for row in report.decisions}
    intents.update({row["event_id"]: row for row in report.additions})
    rate = resolved.fps.fps
    ops = []
    for op in resolved.ops:
        decision = intents.get(op.source_event_id, {})
        params = op.params
        ops.append({"event_id": op.source_event_id, "intent": decision.get("intent", "PLANNER"),
                    "reason": decision.get("reason", "validated planner intent"),
                    "window_s": [round(op.start_frame / rate, 3), round(op.end_frame / rate, 3)],
                    "frames": [op.start_frame, op.end_frame], "camera": params.camera, "motion": params.preset,
                    "peak_zoom": round(params.scale_peak, 4), "anchor": [round(params.anchor_x, 4),
                                                                         round(params.anchor_y, 4)],
                    "target": params.target_subject or "center_safe", "target_reliable": params.target_reliable,
                    "notes": list(params.notes)[:8]})
    return {"kind": "mimir_v6_camera_plan", "version": DIRECTION_VERSION,
            "question": "what must the viewer see right now?", "layout": report.layout_mode,
            "layout_reason": report.layout_reason, "subjects": report.subjects,
            "excluded_subjects": report.excluded_subjects, "decisions": report.decisions,
            "additions": report.additions, "restraint": report.restraint, "ops": ops,
            "dropped": [{"event_id": e, "reason": r} for e, r in resolved.dropped],
            "holds": report.holds, "metrics": dict(resolved.metrics)}


def prepare_pro_edit(request: ProEditRequest) -> ProEditPreparation:
    config = request.config
    prep = ProEditPreparation(status=STATUS_FALLBACK, interpolation=config.interpolation, keep_failed=config.debug,
                              timeline_path=Path(request.timeline_path), clip_index=int(request.clip_index),
                              speaker_profile_path=request.speaker_profile_path,
                              caption_path=Path(request.caption_path))
    prep.warnings.extend(f"Pro Edit config: {p}" for p in config.problems)
    prep.artifacts = ArtifactPaths(Path(request.artifact_dir), int(request.clip_index))
    try:
        # Caption truth is captured BEFORE anything else runs.
        prep.caption_integrity = CaptionIntegrity.capture(request.speaker_profile_path, request.caption_path)
        caps = request.capabilities or probe_capabilities()
        prep.caps = caps
        if not caps.camera_ready:
            raise ProEditError(f"FFmpeg camera support unavailable: {caps.to_dict()}")
        media = probe_media(request.edited_clip_path)
        prep.media = media
        if not media.is_cfr:
            raise ProEditError("paced clip is not constant frame rate; frame-domain camera disabled")
        timeline_data = load_json(Path(request.timeline_path))
        if timeline_data is None:
            raise EditContextError(f"timeline unreadable: {request.timeline_path}")
        clip_timeline = next((c for c in timeline_data.get("timelines", []) or []
                              if isinstance(c, dict) and int(c.get("clip_index", -1)) == int(request.clip_index)), None)
        if clip_timeline is None:
            raise EditContextError(f"timeline has no clip {request.clip_index}")
        prep.source_story_signature = source_story_signature(clip_timeline)
        clip_map = ClipTimelineMap.from_timeline_clip(clip_timeline)

        from ai.editor import intro_renderer

        restart, first_caption = intro_renderer.calculate_main_restart_seconds(
            caption_path=request.caption_path, main_duration=media.duration_s,
            protected_ranges=intro_renderer.protected_edited_ranges(clip_timeline))
        restart = intro_renderer.snap_to_frame(restart, media.fps.fps)
        output_size = base_window(media.width, media.height, config.output_profile)[2:]
        intro = None
        hook_band = None
        if request.teaser_record is not None:
            intro = build_intro_timeline(request.teaser_record, caption_path=request.caption_path,
                                         clean_duration=media.duration_s, main_duration=media.duration_s,
                                         clip_map=clip_map, clip_timeline=clip_timeline, fps=media.fps.fps)
            prep.intro_timeline = intro
            try:
                hook_band = hook_text_band(request.intro_record, intro, width=output_size[0],
                                           height=output_size[1])
            except Exception as error:  # evidence only; the intro camera then ignores hook text
                prep.warnings.append(f"Pro Edit intro hook band unavailable: {type(error).__name__}: {error}")
        provider = select_subject_provider(request, media, prep.artifacts)
        context = build_edit_context(ContextInputs(
            timeline_data=timeline_data, clip_index=request.clip_index, media=media,
            analysis_clip=request.analysis_clip, speaker_profile_path=request.speaker_profile_path,
            video_report_path=request.video_report_path, visible_start_s=restart,
            intro_metadata={"main_restart_s": round(restart, 3),
                            "first_caption_s": None if first_caption is None else round(first_caption, 3),
                            "intro_footage_source": "clean paced clip; camera only via intro directive"},
            subject_provider=provider, caption_path=request.caption_path, intro=intro, hook_band=hook_band,
        ))
        prep.tracking = dict(getattr(provider, "diagnostics", {}) or {"provider": getattr(provider, "name", "?")})
        prep.tracking["tracks"] = len(context.subject_tracks)
        prep.tracking["speaker_links"] = sum(1 for t in context.subject_tracks if t.speaker_id)
        prep.story_signature = context.story_signature
        if context.caption_signature != prep.caption_integrity.profile_signature:
            raise CaptionIntegrityError("context caption references differ from the authoritative profile")

        style = get_style_pack(config.style)
        prep.planner_cache_key = planner_cache_key(context, config)
        outcome = _plan_outcome(request, context, prep)
        plan = outcome.plan
        report = outcome.report
        prep.model_calls = outcome.model_calls
        if config.debug:
            _write_exchanges(outcome, prep.artifacts)
        planner_plan = plan
        direction_report: DirectionReport | None = None
        if config.v6:
            prep.v6 = True
            try:
                plan, direction_report, directed_status = _direct(plan, context, style)
                prep.direction = {"status": "directed", "validation": directed_status,
                                  **direction_report.summary()}
            except Exception as error:  # evidence-only layer: the validated planner intent stays
                plan, direction_report = planner_plan, None
                prep.direction = {"status": "fallback", "reason": f"{type(error).__name__}: {str(error)[:300]}"}
                prep.warnings.append("Pro Edit V6 direction kullanılamadı; doğrulanmış planner niyeti korundu. "
                                     f"Detay: {prep.direction['reason']}")
        presentation = _prepare_captions(request, prep, clip_timeline, plan, output_size, context, media)
        if presentation is not None and prep.presentation_ass is not None:
            # The camera must avoid the captions that will ACTUALLY be burned.
            context = dataclasses.replace(context, caption_region=caption_safe_region(prep.presentation_ass))
        prep.context = context
        resolved = resolve_plan(plan, context, style, output_profile=config.output_profile, strict_center=config.v6)
        resolved = dataclasses.replace(resolved, recommendations=mark_caption_channel(resolved.recommendations,
                                                                                      presentation))
        if config.energy:
            try:
                resolved, presentation = _coordinate_energy(request, prep, context, plan, resolved, presentation,
                                                            style)
            except Exception as error:  # the coordinator is optional; V4 behaviour stays
                prep.energy = {"status": "failed", "reason": f"{type(error).__name__}: {str(error)[:300]}"}
                prep.warnings.append(f"Pro Edit editorial energy pass skipped: {prep.energy['reason']}")
        else:
            prep.energy = {"status": "disabled"}
        prep.resolved = resolved
        if direction_report is not None:
            rate = context.clip.fps.fps
            explain_holds([(a / rate, b / rate) for a, b in resolved.path.active_ranges()], context, style,
                          direction_report)
            prep.direction["holds"] = len(direction_report.holds)
            prep.direction["camera_ops"] = len(resolved.ops)
            write_json_atomic(prep.artifacts.camera_plan, camera_plan_document(direction_report, resolved))
        intro_resolved = None
        if config.intro_camera and plan.intro is not None and request.intro_output_path is not None:
            try:
                intro_resolved = resolve_intro_plan(plan.intro, context, style,
                                                    output_profile=config.output_profile)
            except ProEditError as error:
                prep.warnings.append(f"Pro Edit intro camera dropped (original intro framing kept): {error}")
        prep.intro_resolved = intro_resolved
        prep.intro_output_path = Path(request.intro_output_path).resolve() if request.intro_output_path else None
        plan_id = plan_hash(plan, clip_identity=context.clip_identity, style_name=style.name,
                            style_version=style.version, engine_version=PRESET_ENGINE_VERSION)
        prep.plan_id = plan_id
        prep.render_options = {
            "plan_id": plan_id, "style": f"{style.name}@{style.version}", "engine": PRESET_ENGINE_VERSION,
            "filter_builder": FILTER_BUILDER_VERSION, "renderer": RENDERER_VERSION, "ffmpeg": caps.version,
            "output_profile": config.output_profile.value, "interpolation": config.interpolation,
            "caption_signature": prep.caption_integrity.profile_signature,
            "ass_signature": prep.caption_integrity.ass_signature,
            "energy": prep.energy.get("signature", prep.energy.get("status", "")),
            "caption_presentation": (
                {"version": CAPTION_PRESENTATION_VERSION,
                 "style_pack": f"{CAPTION_STYLE_PACK}@{CAPTION_STYLE_PACK_VERSION}",
                 "ass_signature": prep.presentation_signature, "fonts_dir": prep.caption_fonts_dir or "",
                 "font_source": prep.captions.get("font_source", "")}
                if prep.presentation_ass is not None else "baseline_ass"),
        }
        write_artifacts(
            prep.artifacts,
            context=context.to_dict(),
            plan={
                "schema_version": EDIT_PLAN_SCHEMA_VERSION, "created_at": now_iso(),
                "input_signature": request.input_signature,
                "planner_cache_key": prep.planner_cache_key,
                "style": {"name": style.name, "version": style.version},
                "planner": outcome.planner, "model_calls": outcome.model_calls,
                "repair_attempted": outcome.repair_attempted, "plan_id": plan_id,
                "plan": planner_plan.to_dict(), "validation": report.to_dict(),
                **({"directed_plan_id": plan_hash(plan, clip_identity=context.clip_identity, style_name=style.name,
                                                  style_version=style.version, engine_version=PRESET_ENGINE_VERSION)}
                   if direction_report is not None else {}),
                "integrity": {"caption": prep.caption_integrity.to_dict(), "story": context.story_signature,
                              "source_story": prep.source_story_signature,
                              "teaser": teaser_signature(request.teaser_record)},
            },
            resolved={**resolved.to_dict(), "plan_id": plan_id, "ffmpeg": caps.to_dict(),
                      "render_options": prep.render_options},
        )
        if intro_resolved is not None:
            write_json_atomic(prep.artifacts.intro_resolved,
                              {**intro_resolved.to_dict(), "intro_timeline": intro.to_dict() if intro else None})
        plan_block(prep.diagnostics, enabled=True, plan=plan, style=style.name, style_version=style.version,
                   duration=context.clip.duration_s, report=report, planner=outcome.planner,
                   model_calls=outcome.model_calls, plan_id=plan_id)
        prep.diagnostics.block("PRO_EDIT_EVIDENCE", subjects=json.dumps(prep.tracking, separators=(",", ":")),
                               spans=len(context.spans), caption_region=context.caption_region.source,
                               intro=intro is not None, intro_camera=intro_resolved is not None,
                               hook_text_band=hook_band is not None)
        prep.diagnostics.block("PRO_EDIT_ENERGY", **{
            k: (json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v)
            for k, v in prep.energy.items() if k != "signature"})
        prep.diagnostics.block("PRO_EDIT_CAPTIONS", **{
            k: (json.dumps(v, separators=(",", ":")) if isinstance(v, (dict, list)) else v)
            for k, v in prep.captions.items() if k not in ("ass", "font_source")})
        event_blocks(prep.diagnostics, resolved, plan, debug=config.debug)
        if report.issues:
            prep.diagnostics.block("PRO_EDIT_VALIDATION", issues=",".join(summarize_issues(report)))
        if resolved.is_identity and prep.presentation_ass is None:
            prep.status = STATUS_STATIC
            prep.reason = "plan has no main camera change; existing caption render used"
            return prep
        prep.output_path = Path(request.output_path).resolve()
        prep.status = STATUS_READY
        camera = (f"{len(resolved.ops)} camera op(s) over {int(resolved.metrics.get('active_frames', 0))} frames"
                  if not resolved.is_identity else "no camera change")
        captions = (f"caption presentation ({prep.captions.get('pages', 0)} pages)"
                    if prep.presentation_ass is not None else "baseline captions")
        prep.reason = f"{camera}; {captions}"
        return prep
    except ProEditError as error:
        prep.status = STATUS_FALLBACK
        prep.reason = f"{type(error).__name__}: {error}"
    except Exception as error:  # subsystem boundary: never a single point of failure
        prep.status = STATUS_FALLBACK
        prep.reason = f"unexpected {type(error).__name__}: {error}"
        prep.diagnostics.block("PRO_EDIT_TRACE", trace=traceback.format_exc(limit=6).replace("\n", " | "))
    prep.intro_resolved = None
    prep.presentation_ass = None
    prep.warnings.append(f"Pro Edit devre dışı kaldı; mevcut MIMIR caption render kullanılıyor. Detay: {prep.reason}")
    prep.diagnostics.block("PRO_EDIT", enabled=1, status=prep.status, reason=prep.reason)
    return prep


# ============================================================
# RENDER
# ============================================================

def verify_truth(prep: ProEditPreparation, stage: str) -> None:
    """Caption truth and story structure must be unchanged by Pro Edit."""
    if prep.caption_integrity is not None:
        prep.caption_integrity.verify(prep.speaker_profile_path, prep.caption_path, stage=stage)
    if prep.timeline_path is not None and prep.source_story_signature:
        data = load_json(prep.timeline_path) or {}
        clip = next((c for c in data.get("timelines", []) or []
                     if isinstance(c, dict) and int(c.get("clip_index", -1)) == prep.clip_index), None)
        if clip is None or source_story_signature(clip) != prep.source_story_signature:
            raise ProEditError(f"story/timeline truth changed during {stage}")


def render_with_fallback(
    prep: ProEditPreparation,
    *,
    caption_file: Path,
    baseline: Callable[[], Path | str],
    outcome: dict[str, Any],
) -> Path:
    """Main: camera BEFORE subtitles in one graph.

    Order: camera + presentation ASS -> camera + baseline ASS (only when the
    camera changes something) -> the existing caption renderer. ``outcome``
    receives status/reason for the main thread (may run in a worker).
    """
    if not prep.ready:
        outcome.update(status="baseline", reason=prep.reason or prep.status)
        return Path(baseline())
    assert prep.resolved is not None and prep.output_path is not None and prep.media is not None
    assert prep.caps is not None and prep.artifacts is not None
    attempts: list[tuple[str, Path]] = []
    if prep.presentation_ass is not None:
        attempts.append(("presentation", prep.presentation_ass))
    if not prep.resolved.is_identity:
        attempts.append(("baseline_ass", Path(caption_file)))
    failures: list[str] = []
    for label, ass in attempts:
        try:
            verify_truth(prep, "pre-render")
            if label == "presentation" and _file_signature(ass) != prep.presentation_signature:
                raise CaptionIntegrityError("presentation ASS changed after preparation")
            result = render_camera_captions(
                edited_clip=prep.media.path, caption_file=ass, output_path=prep.output_path,
                resolved=prep.resolved, caps=prep.caps, source_media=prep.media,
                script_path=prep.artifacts.filter_script, interpolation=prep.interpolation,
                keep_failed=prep.keep_failed,
                fonts_dir=prep.caption_fonts_dir if label == "presentation" else None,
            )
            verify_truth(prep, "main render")
            outcome.update(status="pro_edit", captions=label, reason=prep.reason, output=str(result.output_path),
                           failures=failures)
            return result.output_path
        except Exception as error:  # boundary: next fallback level, report upward
            failures.append(f"{label}: {type(error).__name__}: {str(error)[:1200]}")
    outcome.update(status="fallback", reason=" | ".join(failures) or "no render attempt")
    return Path(baseline())


def render_static_camera(prep: ProEditPreparation, *, output_path: Path) -> tuple[Path, dict[str, Any]]:
    """Bounded repair: the same verified caption presentation with NO camera move.

    A stable wide shot is always a valid edit; this is what the final reviewer's
    'cropped subject' repair renders (once). Captions stay the frozen-truth
    presentation; only the camera path becomes identity. Returns the render and
    its proof (no camera op to prove; story geometry re-checked on the identity path)."""
    from ai.editor.pro_edit.camera import CameraPath

    if prep.resolved is None or prep.media is None or prep.caps is None or prep.artifacts is None:
        raise ProEditError("static repair needs a prepared Pro Edit render")
    ass = prep.presentation_ass if prep.presentation_ass is not None else prep.caption_path
    if ass is None:
        raise ProEditError("static repair has no caption file")
    static = dataclasses.replace(prep.resolved, ops=(), path=CameraPath(prep.resolved.frame_count, ()),
                                 recommendations=())
    verify_truth(prep, "static repair")
    result = render_camera_captions(
        edited_clip=prep.media.path, caption_file=Path(ass), output_path=Path(output_path), resolved=static,
        caps=prep.caps, source_media=prep.media, script_path=prep.artifacts.filter_script.with_name(
            prep.artifacts.filter_script.stem + "_static" + prep.artifacts.filter_script.suffix),
        interpolation=prep.interpolation, keep_failed=prep.keep_failed,
        fonts_dir=prep.caption_fonts_dir if prep.presentation_ass is not None else None)
    verify_truth(prep, "static repair render")
    geometry = (render_proof.story_geometry_check(static, prep.context, get_style_pack(static.style_name))
                if prep.context is not None else
                {"status": "passed", "frames_checked": 0, "violations": [], "violation_count": 0})
    proof = {"status": "no_camera_ops", "reason": "static camera repair: identity camera path", "samples": [],
             "story_geometry": geometry, "render": str(result.output_path)}
    return result.output_path, proof


def render_intro_source(prep: ProEditPreparation, *, clean_clip: Path, outcome: dict[str, Any]) -> Path:
    """Clean paced clip with camera ONLY on the selected teaser frames.

    The intro renderer then trims exactly the same teaser range from it, so
    intro selection, source range, handoff and order are unchanged. Any
    failure returns the clean clip (existing MIMIR intro).
    """
    if not prep.intro_ready:
        outcome.update(status="clean", reason="no intro camera")
        return Path(clean_clip)
    assert prep.intro_resolved is not None and prep.intro_output_path is not None
    assert prep.caps is not None and prep.media is not None and prep.artifacts is not None
    try:
        result = render_camera_captions(
            edited_clip=clean_clip, caption_file=None, output_path=prep.intro_output_path,
            resolved=prep.intro_resolved, caps=prep.caps, source_media=prep.media,
            script_path=prep.artifacts.intro_filter_script, interpolation=prep.interpolation,
            keep_failed=prep.keep_failed,
        )
        verify_truth(prep, "intro camera render")
        outcome.update(status="pro_edit", output=str(result.output_path))
        return result.output_path
    except Exception as error:  # boundary: keep the clean intro source
        outcome.update(status="fallback", reason=f"{type(error).__name__}: {str(error)[:800]}")
        return Path(clean_clip)


# ============================================================
# V6 PIXEL PROOF (planned camera -> rendered -> final)
# ============================================================

def prove_main_render(prep: ProEditPreparation, rendered: Path, burned_ass: Path | None) -> dict[str, Any]:
    """Planned main camera vs the rendered captioned main (+ story geometry on every active frame)."""
    if prep.resolved is None or prep.media is None:
        return {"status": "unavailable", "reason": "no resolved camera plan"}
    region = caption_safe_region(burned_ass) if burned_ass is not None and Path(burned_ass).is_file() else None
    proof = render_proof.prove_camera(source_path=prep.media.path, rendered_path=rendered, resolved=prep.resolved,
                                      source_size=(prep.media.width, prep.media.height), caption_region=region)
    if prep.context is not None:
        proof["story_geometry"] = render_proof.story_geometry_check(
            prep.resolved, prep.context, get_style_pack(prep.resolved.style_name))
    proof["render"] = str(Path(rendered))
    return proof


def prove_intro_render(prep: ProEditPreparation, intro_source: Path) -> dict[str, Any]:
    """Planned intro camera vs the camera-rendered intro source (teaser frames only)."""
    if prep.intro_resolved is None or prep.media is None:
        return {"status": "no_camera_ops", "reason": "no intro camera"}
    proof = render_proof.prove_camera(source_path=prep.media.path, rendered_path=intro_source,
                                      resolved=prep.intro_resolved,
                                      source_size=(prep.media.width, prep.media.height))
    if prep.context is not None and prep.context.intro_span is not None:
        proof["story_geometry"] = render_proof.story_geometry_check(
            prep.intro_resolved, prep.context, get_style_pack(prep.intro_resolved.style_name),
            spans=(prep.context.intro_span,))
    proof["render"] = str(Path(intro_source))
    return proof


def prove_final_output(prep: ProEditPreparation, *, final_path: Path, main_render: Path, main_proof: Mapping[str, Any],
                       intro_source: Path | None, intro_proof: Mapping[str, Any] | None,
                       burned_ass: Path | None) -> dict[str, Any]:
    """Verified camera frames of the main (and intro) renders are present in the FINAL short."""
    if prep.media is None or prep.resolved is None:
        return {"status": "unavailable", "reason": "no Pro Edit media"}
    try:
        final_info = probe_media(final_path)
    except Exception as error:  # an unreadable final is reported by the gate as well
        return {"status": "unavailable", "reason": f"final unreadable: {type(error).__name__}: {error}"}
    rate = prep.media.fps.fps
    final_rate = final_info.fps.fps
    intro = prep.intro_timeline
    region = caption_safe_region(burned_ass) if burned_ass is not None and Path(burned_ass).is_file() else None
    hook = prep.context.hook_band if prep.context is not None else None
    pairs: list[render_proof.FinalPair] = []
    for row in main_proof.get("samples", []) or []:
        if row.get("kind") != "camera" or row.get("verdict") != "reached":
            continue
        frame = int(row["frame"])
        t = frame / rate
        if intro is not None:
            if t < intro.main_restart + intro.transition + 2.0 / rate:
                continue
            mapped = intro.main_to_final(Timestamp(t, TimelineDomain.PACED_CLIP))
            if mapped is None:
                continue
            final_t = float(mapped.seconds)
        else:
            final_t = t
        bands = tuple((b.y0, b.y1) for b in region.active_bands(t - 0.3, t + 0.3)) if region is not None else ()
        pairs.append(render_proof.FinalPair(str(main_render), frame, int(round(final_t * final_rate)), bands, "main",
                                            str(row.get("event_id", ""))))
    if intro is not None and intro_source is not None and intro_proof:
        for row in intro_proof.get("samples", []) or []:
            if row.get("kind") != "camera" or row.get("verdict") != "reached":
                continue
            frame = int(row["frame"])
            local = frame / rate - intro.teaser_start
            if not 0.0 <= local < intro.teaser_duration - intro.transition - 2.0 / rate:
                continue
            bands = ((hook.y0, hook.y1),) if hook is not None and hook.active(frame / rate - 0.1,
                                                                             frame / rate + 0.1) else ()
            pairs.append(render_proof.FinalPair(str(intro_source), frame, int(round(local * final_rate)), bands,
                                                "intro", str(row.get("event_id", ""))))
    proof = render_proof.prove_final(final_path=final_path, pairs=pairs, source_path=prep.media.path,
                                     source_size=(prep.media.width, prep.media.height),
                                     output_size=prep.resolved.output_size, base=prep.resolved.base)
    proof["final"] = str(Path(final_path))
    return proof
