"""MIMIR Pro Edit Engine V1 - deterministic presentation layer.

MIMIR decides WHAT story is shown (truth/story/captions/speakers/intro).
Pro Edit decides HOW that already-selected short is framed and captioned:

    EditContext -> EditPlanner -> EditPlan -> validator -> preset resolver
    -> ResolvedPlan (frame domain) -> executor (one FFmpeg graph, one encode)
    caption truth (read-only) + plan caption intent -> caption presentation ASS

V6 (MIMIR_V6=1 / --v6): evidence director between validator and resolver
(direction), camera plan + HOLD reasons, and pixel proof of the render
(render_proof).

Feature switch: MIMIR_PRO_EDIT=0|1 (default 0 = existing behavior).
"""
from __future__ import annotations

from ai.editor.pro_edit import (
    camera,
    caption_action,
    caption_background,
    caption_brand,
    caption_guard,
    caption_layout,
    caption_legibility,
    caption_occupancy,
    caption_placement,
    caption_platform,
    caption_presentation,
    caption_primitives,
    config,
    context,
    diagnostics,
    direction,
    director_vision,
    editorial_energy,
    errors,
    executor,
    ffmpeg_filters,
    font_metrics,
    framing,
    intro_timeline,
    media,
    planner,
    policy,
    presets,
    providers,
    render_proof,
    request,
    schema,
    speaker_link,
    stage,
    story,
    style,
    subjects,
    timebase,
    validator,
)
from ai.editor.pro_edit.vision import association as vision_association
from ai.editor.pro_edit.vision import cv_runtime as vision_cv_runtime
from ai.editor.pro_edit.vision import detectors as vision_detectors
from ai.editor.pro_edit.vision import frames as vision_frames
from ai.editor.pro_edit.vision import provider as vision_provider
from ai.editor.pro_edit.vision import tracker as vision_tracker

PRO_EDIT_VERSION = 5

# Fingerprinted by the pipeline stage signatures when Pro Edit is enabled.
MODULES = (
    camera, caption_action, caption_background, caption_brand, caption_guard, caption_layout, caption_legibility,
    caption_occupancy, caption_placement, caption_platform,
    caption_presentation, caption_primitives, config, context, diagnostics, direction, director_vision, editorial_energy, errors,
    executor,
    ffmpeg_filters, font_metrics, framing,
    intro_timeline, media, planner, policy, presets, providers, request, schema, speaker_link, stage,
    story, style, subjects, timebase, validator, vision_association, vision_cv_runtime, vision_detectors,
    vision_frames, vision_provider, vision_tracker,
)
# Presentation-only modules run after planning; they are left out of the plan
# cache signature so a caption-presentation change never re-bills the planner.
PRESENTATION_MODULES = (caption_action, caption_background, caption_brand, caption_layout, caption_legibility,
                        caption_occupancy, caption_placement, caption_platform, caption_presentation,
                        caption_primitives, editorial_energy, font_metrics)
PLAN_MODULES = tuple(module for module in MODULES if module not in PRESENTATION_MODULES)
# Post-render verification only (V6 pixel proof): never part of a render/plan cache key.
VERIFY_MODULES = (render_proof,)

__all__ = ["PRO_EDIT_VERSION", "MODULES", "config", "stage"]
