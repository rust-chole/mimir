"""Pro Edit configuration (environment / .env, same style as MIMIR flags).

MIMIR_PRO_EDIT=0|1                 feature switch (default 0 = current MIMIR)
MIMIR_V6=0|1                       V6 direction layer inside Pro Edit: evidence-directed shot intent,
                                   camera plan + HOLD reasons, pixel proof (default 0; the pipeline's
                                   --v6 / --force-v6 turn it on together with Pro Edit)
MIMIR_PRO_EDIT_PLANNER=model|replay|rules|static  (default model; unavailable -> static)
MIMIR_PRO_EDIT_REPLAY=<recorded raw planner response JSON>  (planner=replay)
MIMIR_PRO_EDIT_TRACKER=auto|local|sidecar|off  (auto: sidecar if set, else local OpenCV)
MIMIR_PRO_EDIT_FACE_MODEL=<YuNet ONNX path>    (optional; else bundled Haar cascades)
MIMIR_PRO_EDIT_INTRO=0|1           camera presentation of the existing intro (default 1)
MIMIR_PRO_EDIT_CAPTIONS=0|1        caption presentation layer (default 1; 0 = burn the baseline ASS)
MIMIR_CAPTION_PLATFORM_PROFILE=generic|generic_conservative|custom|tiktok|reels|shorts (default generic;
                                   named platforms have no verified geometry -> generic_conservative)
MIMIR_CAPTION_PLATFORM_PROFILE_FILE=<normalized safe-zone JSON>  (custom profile)
MIMIR_CAPTION_BRAND_PROFILE=mimir_default|<brand JSON path>      (default mimir_default == V3.1 look)
MIMIR_CAPTION_ACTIVITY=0|1         local motion-activity map for caption placement (default 1)
MIMIR_CAPTION_SHAPER=naive|auto|harfbuzz   width shaper (default naive; harfbuzz needs uharfbuzz)
MIMIR_CAPTION_LEGIBILITY=0|1       background contrast analysis -> outline/shadow/plate (default 1)
MIMIR_CAPTION_UI_OCCUPANCY=0|1     static TEXT_LIKE / HUD occupancy in placement (default 1)
MIMIR_CAPTION_LAYOUT=0|1           content layout classification for placement weights (default 1)
MIMIR_CAPTION_OBJECTS=off|hog_person  optional OpenCV HOG person evidence in payoff/reaction spans (default off)
MIMIR_CAPTION_PLATFORM_VARIANT=<ui variant id>   selects a variant of a v2 platform profile file
MIMIR_CAPTION_PLATFORM_DEVICE=<device class>     (e.g. phone / tablet; v2 profile files only)
MIMIR_CAPTION_DESCRIPTION=none|short|long        description/caption footprint (v2 profile files only)
MIMIR_PRO_EDIT_ENERGY=0|1          global editorial energy coordinator (default 1)
MIMIR_PRO_EDIT_ENERGY_WINDOW_MS=<300..2000>      energy window (default 800)
MIMIR_PRO_EDIT_ENERGY_BUDGET=<2..10>             max energy inside one window (default 4)
MIMIR_PRO_EDIT_ENERGY_TABLE=<JSON object>        override event energies (known keys, integers 0..5)
MIMIR_PRO_EDIT_PLANNER_TIMEOUT=<seconds>       (default 300: the director reasons at high effort)
MIMIR_EDIT_DIRECTOR_MODEL / MIMIR_EDIT_DIRECTOR_REASONING   (ai/model_config.py; default Astra 6 / high)
MIMIR_PRO_EDIT_MODEL / MIMIR_PRO_EDIT_REASONING             (legacy per-run override of the two above)
MIMIR_PRO_EDIT_STYLE=pro_stream_v1
MIMIR_PRO_EDIT_OUTPUT_PROFILE=preserve|16:9|9:16|1:1  (default preserve)
MIMIR_PRO_EDIT_INTERPOLATION=linear|cubic             (default cubic)
MIMIR_PRO_EDIT_DEBUG=0|1           keep failed temp renders / verbose trace
MIMIR_PRO_EDIT_SUBJECTS=<path to subject sidecar JSON> (optional tracker output)

The director model comes from ai/model_config.py (routing lives in one place);
stage signatures carry the exact model ids, so routing changes never re-bill
unrelated cached stages.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from ai import model_config
from ai.editor.pro_edit.camera import OutputProfile
from ai.editor.pro_edit.style import available_style_packs

_FALSE = {"0", "false", "no", "off", ""}
_TRUE = {"1", "true", "yes", "on"}
_PLANNERS = {"model", "replay", "rules", "static"}
_TRACKERS = {"auto", "local", "sidecar", "off"}
_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class ProEditConfig:
    enabled: bool = False
    planner: str = "model"
    model: str = model_config.EDIT_DIRECTOR_MODEL
    reasoning_effort: str = model_config.EDIT_DIRECTOR_REASONING_EFFORT
    style: str = "pro_stream_v1"
    output_profile: OutputProfile = OutputProfile.PRESERVE
    interpolation: str = "cubic"
    debug: bool = False
    subjects_path: str | None = None
    tracker: str = "auto"
    intro_camera: bool = True
    captions: bool = True
    caption_platform: str = "generic"
    caption_platform_file: str | None = None
    caption_brand: str = "mimir_default"
    caption_activity: bool = True
    caption_shaper: str = "naive"
    caption_legibility: bool = True
    caption_ui: bool = True
    caption_layout: bool = True
    caption_objects: str = "off"
    caption_platform_variant: str = ""
    caption_platform_device: str = ""
    caption_description: str = ""
    energy: bool = True
    energy_window_ms: int = 800
    energy_budget: int = 4
    energy_table: tuple[tuple[str, int], ...] = ()
    replay_path: str | None = None
    planner_timeout_s: float = 300.0
    v6: bool = False
    problems: tuple[str, ...] = field(default_factory=tuple)

    def signature_payload(self) -> dict[str, object]:
        return {
            "planner": self.planner,
            "model": self.model if self.planner == "model" else "",
            "reasoning_effort": self.reasoning_effort if self.planner == "model" else "",
            "style": self.style,
            "output_profile": self.output_profile.value,
            "interpolation": self.interpolation,
            "subjects_path": self.subjects_path or "",
            "tracker": self.tracker,
            "intro_camera": self.intro_camera,
            "captions": self.captions,
            "caption_platform": self.caption_platform,
            "caption_platform_file": self.caption_platform_file or "",
            "caption_brand": self.caption_brand,
            "caption_activity": self.caption_activity,
            "caption_shaper": self.caption_shaper,
            "caption_legibility": self.caption_legibility,
            "caption_ui": self.caption_ui,
            "caption_layout": self.caption_layout,
            "caption_objects": self.caption_objects,
            "caption_platform_variant": self.caption_platform_variant,
            "caption_platform_device": self.caption_platform_device,
            "caption_description": self.caption_description,
            "energy": self.energy,
            "energy_window_ms": self.energy_window_ms,
            "energy_budget": self.energy_budget,
            "energy_table": [list(item) for item in self.energy_table],
            "replay_path": (self.replay_path or "") if self.planner == "replay" else "",
            **({"v6": True} if self.v6 else {}),
        }

    def plan_signature_payload(self) -> dict[str, object]:
        """Plan-cache key: presentation-only switches (captions) never re-bill the planner."""
        payload = self.signature_payload()
        for key in ("captions", "caption_platform", "caption_platform_file", "caption_brand", "caption_activity",
                    "caption_shaper", "caption_legibility", "caption_ui", "caption_layout", "caption_objects",
                    "caption_platform_variant", "caption_platform_device", "caption_description", "energy",
                    "energy_window_ms", "energy_budget", "energy_table"):
            payload.pop(key, None)
        return payload


def _flag(value: str | None, default: bool) -> bool:
    text = str(value if value is not None else "").strip().casefold()
    if not text:
        return default
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def _bounded_int(env: Mapping[str, str], key: str, default: int, low: int, high: int,
                 problems: list[str]) -> int:
    raw = str(env.get(key, "")).strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if not low <= value <= high:
            raise ValueError
    except ValueError:
        problems.append(f"{key}={raw!r} invalid (integer {low}..{high}); using {default}")
        return default
    return value


def _ident(value: str | None) -> str:
    text = str(value or "").strip().casefold()
    return text if len(text) <= 48 and all(c.isalnum() or c in "_-." for c in text) else ""


def load_config(override_enabled: bool | None = None, environ: Mapping[str, str] | None = None,
                v6: bool | None = None) -> ProEditConfig:
    """Never raises: invalid values fall back to safe defaults and are reported."""
    env = os.environ if environ is None else environ
    problems: list[str] = []
    enabled = _flag(env.get("MIMIR_PRO_EDIT"), False) if override_enabled is None else bool(override_enabled)
    v6_enabled = _flag(env.get("MIMIR_V6"), False) if v6 is None else bool(v6)

    planner = str(env.get("MIMIR_PRO_EDIT_PLANNER", "model")).strip().casefold() or "model"
    if planner not in _PLANNERS:
        problems.append(f"MIMIR_PRO_EDIT_PLANNER={planner!r} invalid; using static")
        planner = "static"
    model = str(env.get("MIMIR_PRO_EDIT_MODEL", "")).strip() or model_config.EDIT_DIRECTOR_MODEL
    default_effort = model_config.EDIT_DIRECTOR_REASONING_EFFORT
    effort = str(env.get("MIMIR_PRO_EDIT_REASONING", default_effort)).strip().casefold() or default_effort
    if effort not in _EFFORTS:
        problems.append(f"MIMIR_PRO_EDIT_REASONING={effort!r} invalid; using {default_effort}")
        effort = default_effort
    style = str(env.get("MIMIR_PRO_EDIT_STYLE", "pro_stream_v1")).strip() or "pro_stream_v1"
    if style not in available_style_packs():
        problems.append(f"MIMIR_PRO_EDIT_STYLE={style!r} unknown; using pro_stream_v1")
        style = "pro_stream_v1"
    raw_profile = str(env.get("MIMIR_PRO_EDIT_OUTPUT_PROFILE", "preserve")).strip() or "preserve"
    try:
        profile = OutputProfile(raw_profile)
    except ValueError:
        problems.append(f"MIMIR_PRO_EDIT_OUTPUT_PROFILE={raw_profile!r} invalid; using preserve")
        profile = OutputProfile.PRESERVE
    interpolation = str(env.get("MIMIR_PRO_EDIT_INTERPOLATION", "cubic")).strip().casefold() or "cubic"
    if interpolation not in {"linear", "cubic"}:
        problems.append(f"MIMIR_PRO_EDIT_INTERPOLATION={interpolation!r} invalid; using cubic")
        interpolation = "cubic"
    subjects = str(env.get("MIMIR_PRO_EDIT_SUBJECTS", "")).strip() or None
    tracker = str(env.get("MIMIR_PRO_EDIT_TRACKER", "auto")).strip().casefold() or "auto"
    if tracker not in _TRACKERS:
        problems.append(f"MIMIR_PRO_EDIT_TRACKER={tracker!r} invalid; using auto")
        tracker = "auto"
    if tracker == "sidecar" and not subjects:
        problems.append("MIMIR_PRO_EDIT_TRACKER=sidecar without MIMIR_PRO_EDIT_SUBJECTS; using auto")
        tracker = "auto"
    replay = str(env.get("MIMIR_PRO_EDIT_REPLAY", "")).strip() or None
    if planner == "replay" and not replay:
        problems.append("MIMIR_PRO_EDIT_PLANNER=replay without MIMIR_PRO_EDIT_REPLAY; using static")
        planner = "static"
    try:
        timeout = float(str(env.get("MIMIR_PRO_EDIT_PLANNER_TIMEOUT", "300")).strip() or 300)
        if not 5 <= timeout <= 900:
            raise ValueError
    except ValueError:
        problems.append("MIMIR_PRO_EDIT_PLANNER_TIMEOUT invalid; using 300")
        timeout = 300.0
    shaper = str(env.get("MIMIR_CAPTION_SHAPER", "naive")).strip().casefold() or "naive"
    if shaper not in ("naive", "auto", "harfbuzz"):
        problems.append(f"MIMIR_CAPTION_SHAPER={shaper!r} invalid; using naive")
        shaper = "naive"
    objects = str(env.get("MIMIR_CAPTION_OBJECTS", "off")).strip().casefold() or "off"
    if objects not in ("off", "hog_person"):
        problems.append(f"MIMIR_CAPTION_OBJECTS={objects!r} invalid; using off")
        objects = "off"
    description = str(env.get("MIMIR_CAPTION_DESCRIPTION", "")).strip().casefold()
    if description not in ("", "none", "short", "long"):
        problems.append(f"MIMIR_CAPTION_DESCRIPTION={description!r} invalid; ignored")
        description = ""
    window_ms = _bounded_int(env, "MIMIR_PRO_EDIT_ENERGY_WINDOW_MS", 800, 300, 2000, problems)
    budget = _bounded_int(env, "MIMIR_PRO_EDIT_ENERGY_BUDGET", 4, 2, 10, problems)
    from ai.editor.pro_edit.editorial_energy import parse_energy_table

    table, table_problem = parse_energy_table(env.get("MIMIR_PRO_EDIT_ENERGY_TABLE"))
    if table_problem:
        problems.append(table_problem)
    return ProEditConfig(
        enabled=enabled,
        planner=planner,
        model=model,
        reasoning_effort=effort,
        style=style,
        output_profile=profile,
        interpolation=interpolation,
        debug=_flag(env.get("MIMIR_PRO_EDIT_DEBUG"), False),
        subjects_path=subjects,
        tracker=tracker,
        intro_camera=_flag(env.get("MIMIR_PRO_EDIT_INTRO"), True),
        captions=_flag(env.get("MIMIR_PRO_EDIT_CAPTIONS"), True),
        caption_platform=str(env.get("MIMIR_CAPTION_PLATFORM_PROFILE", "generic")).strip().casefold() or "generic",
        caption_platform_file=str(env.get("MIMIR_CAPTION_PLATFORM_PROFILE_FILE", "")).strip() or None,
        caption_brand=str(env.get("MIMIR_CAPTION_BRAND_PROFILE", "mimir_default")).strip() or "mimir_default",
        caption_activity=_flag(env.get("MIMIR_CAPTION_ACTIVITY"), True),
        caption_shaper=shaper,
        caption_legibility=_flag(env.get("MIMIR_CAPTION_LEGIBILITY"), True),
        caption_ui=_flag(env.get("MIMIR_CAPTION_UI_OCCUPANCY"), True),
        caption_layout=_flag(env.get("MIMIR_CAPTION_LAYOUT"), True),
        caption_objects=objects,
        caption_platform_variant=_ident(env.get("MIMIR_CAPTION_PLATFORM_VARIANT")),
        caption_platform_device=_ident(env.get("MIMIR_CAPTION_PLATFORM_DEVICE")),
        caption_description=description,
        energy=_flag(env.get("MIMIR_PRO_EDIT_ENERGY"), True),
        energy_window_ms=window_ms,
        energy_budget=budget,
        energy_table=tuple(sorted(table.items())),
        replay_path=replay,
        planner_timeout_s=timeout,
        v6=v6_enabled,
        problems=tuple(problems),
    )
