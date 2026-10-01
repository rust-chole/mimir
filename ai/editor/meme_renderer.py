from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ai.editor import render_backend


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"
MEME_LIBRARY_DIR = PROJECT_ROOT / "meme_library"

FINAL_PREVIEWS_DIR = VOD_OUTPUT_DIR / "final_previews"
FINAL_MEMES_DIR = VOD_OUTPUT_DIR / "final_memes"

REPORT_DIR = (
    VOD_OUTPUT_DIR
    / "memes"
    / "render_reports"
)


# ============================================================
# VERSION
# ============================================================

MEME_RENDERER_VERSION = 6


# ============================================================
# CORE POLICY
# ============================================================

# Visual meme merkezde, küçük ve hafif transparan.
VISUAL_WIDTH_PERCENT = 24.0
VISUAL_MAX_HEIGHT_PERCENT = 28.0
VISUAL_OPACITY = 0.78
# A visual meme goes where the picture is least busy and never over captions:
# candidate anchors (normalized centre) tried in this order on equal activity.
VISUAL_EDGE_MARGIN = 0.04
VISUAL_ANCHORS = (
    ("top_right", 1.0, 0.0), ("top_left", 0.0, 0.0), ("top_center", 0.5, 0.0),
    ("mid_right", 1.0, 0.5), ("mid_left", 0.0, 0.5), ("bottom_right", 1.0, 1.0), ("bottom_left", 0.0, 1.0),
)
MAIN_OPENING_PROTECTION = 0.70   # no effect in the first moments after the story restart

# Smooth giriş/çıkış.
VISUAL_FADE_IN = 0.12
VISUAL_FADE_OUT = 0.18

# Audio meme tek, kısa ve mix-aware. Sabit yüksek volume yerine slot bağlamına göre
# gain/ducking hesaplanır; amaç konuşmayı ezmeden ritmik punctuation yaratmak.
AUDIO_VOLUME_SUBTLE = 0.42
AUDIO_VOLUME_MEDIUM = 0.53
AUDIO_VOLUME_STRONG = 0.63
AUDIO_FADE_IN = 0.040
AUDIO_FADE_OUT = 0.180

# Sidechain değerleri event bazında yumuşatılır.
SIDECHAIN_THRESHOLD = 0.035
SIDECHAIN_ATTACK_MS = 12
SIDECHAIN_RELEASE_MS = 135

OUTPUT_LIMITER = 0.95


# ============================================================
# ENCODE
# ============================================================

# A visual meme re-encodes video with the shared render backend
# (ai/editor/render_backend.py); an audio-only meme stream-copies the video.
# The overlay graph itself still normalizes to this pixel format.
PIXEL_FORMAT = "yuv420p"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 48000

DURATION_TOLERANCE = 0.35


# ============================================================
# FILE TYPES
# ============================================================

STATIC_VISUAL_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


# ============================================================
# JSON HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı:\n{path}"
        )

    raw = path.read_text(
        encoding="utf-8"
    ).strip()

    if not raw:
        raise RuntimeError(
            f"JSON dosyası boş:\n{path}"
        )

    try:
        data = json.loads(raw)

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "JSON formatı bozuk:\n"
            f"{path}\n\n"
            f"Satır: {error.lineno}\n"
            f"Sütun: {error.colno}\n"
            f"Hata: {error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(
            f"JSON root object değil:\n{path}"
        )

    return data


def save_json(
    path: str | Path,
    data: dict[str, Any],
) -> None:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_filename(
    value: str,
) -> str:

    value = re.sub(
        r'[<>:"/\\|?*]+',
        "_",
        str(value).strip(),
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return (
        value.strip(" ._")
        or "clip"
    )


def parse_fraction(
    value: Any,
) -> float:

    text = str(
        value or ""
    ).strip()

    if not text:
        return 0.0

    if "/" in text:
        left, right = text.split(
            "/",
            1,
        )

        try:
            numerator = float(left)
            denominator = float(right)
        except ValueError:
            return 0.0

        if denominator == 0:
            return 0.0

        return numerator / denominator

    try:
        return float(text)
    except ValueError:
        return 0.0


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:

    return max(
        minimum,
        min(
            value,
            maximum,
        ),
    )


# ============================================================
# PACKAGE VALIDATION
# ============================================================

def validate_discovery_package(
    package: dict[str, Any],
) -> None:

    from ai.editor import meme_discovery

    # One source of truth: the producer's own version constant. A hard-coded
    # number here once disabled every meme/SFX render without anyone noticing.
    if package.get("version") != meme_discovery.DISCOVERY_VERSION:
        raise RuntimeError(
            f"Meme Renderer, Meme Discovery V{meme_discovery.DISCOVERY_VERSION} çıktısı bekliyor "
            f"(gelen: V{package.get('version')})."
        )

    if not isinstance(
        package.get("clips"),
        list,
    ):
        raise RuntimeError(
            "Discovery JSON içinde clips listesi yok."
        )

    inputs = package.get("inputs")

    if not isinstance(inputs, dict):
        raise RuntimeError(
            "Discovery JSON inputs eksik."
        )

    meme_slots = inputs.get(
        "meme_slots"
    )

    if (
        not isinstance(meme_slots, str)
        or not meme_slots.strip()
    ):
        raise RuntimeError(
            "Discovery JSON inputs.meme_slots eksik."
        )


def validate_slot_package(
    package: dict[str, Any],
) -> None:

    from ai.editor import meme_analyzer

    if package.get("version") != meme_analyzer.MEME_ANALYZER_VERSION:
        raise RuntimeError(
            f"Meme Renderer, Meme Analyzer V{meme_analyzer.MEME_ANALYZER_VERSION} çıktısı bekliyor "
            f"(gelen: V{package.get('version')})."
        )

    inputs = package.get("inputs")

    if not isinstance(inputs, dict):
        raise RuntimeError(
            "Meme slot JSON inputs eksik."
        )

    timeline = inputs.get(
        "timeline"
    )

    if (
        not isinstance(timeline, str)
        or not timeline.strip()
    ):
        raise RuntimeError(
            "Meme slot JSON inputs.timeline eksik."
        )


def validate_timeline(
    timeline: dict[str, Any],
) -> None:

    if timeline.get("version") != 3:
        raise RuntimeError(
            "Meme Renderer V2 yalnızca Timeline V3 ile çalışır."
        )

    if not isinstance(
        timeline.get("source"),
        dict,
    ):
        raise RuntimeError(
            "Timeline source bilgisi eksik."
        )


# ============================================================
# REFERENCED INPUTS
# ============================================================

def get_slot_json_path(
    discovery_package: dict[str, Any],
) -> Path:

    path = Path(
        discovery_package[
            "inputs"
        ][
            "meme_slots"
        ]
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Meme slot JSON bulunamadı:\n{path}"
        )

    return path


def get_timeline_path(
    slot_package: dict[str, Any],
) -> Path:

    path = Path(
        slot_package[
            "inputs"
        ][
            "timeline"
        ]
    ).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Timeline bulunamadı:\n{path}"
        )

    return path


# ============================================================
# VIDEO STEM
# ============================================================

def get_video_stem(
    timeline: dict[str, Any],
) -> str:

    source = timeline.get(
        "source",
        {},
    )

    for key in (
        "video_stem",
        "video_name",
        "video_path",
    ):

        value = source.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):

            if key == "video_stem":
                return value.strip()

            return Path(value).stem

    raise RuntimeError(
        "Timeline source içinde video stem/path bulunamadı."
    )


# ============================================================
# FFPROBE
# ============================================================

def get_media_info(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Media bulunamadı:\n{path}"
        )

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe bulunamadı. FFmpeg kurulumunu kontrol et."
        ) from error

    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe media bilgisini okuyamadı:\n"
            + result.stderr
        )

    try:
        data = json.loads(
            result.stdout
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "ffprobe geçersiz JSON döndürdü."
        ) from error

    width = 0
    height = 0
    fps = 30.0

    has_video = False
    has_audio = False

    for stream in data.get(
        "streams",
        [],
    ):

        if not isinstance(stream, dict):
            continue

        codec_type = stream.get(
            "codec_type"
        )

        if codec_type == "video":
            has_video = True

            if width <= 0:
                try:
                    width = int(
                        stream.get(
                            "width",
                            0,
                        )
                    )

                    height = int(
                        stream.get(
                            "height",
                            0,
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    width = 0
                    height = 0

                parsed_fps = parse_fraction(
                    stream.get(
                        "avg_frame_rate"
                    )
                    or stream.get(
                        "r_frame_rate"
                    )
                )

                if parsed_fps > 0:
                    fps = parsed_fps

        elif codec_type == "audio":
            has_audio = True

    try:
        duration = float(
            data.get(
                "format",
                {},
            ).get(
                "duration",
                0.0,
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        duration = 0.0

    return {
        "has_video": has_video,
        "has_audio": has_audio,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": max(
            0.0,
            duration,
        ),
    }


# ============================================================
# FIND BASE FINAL VIDEO
# ============================================================

FINAL_VERSION_PATTERN = re.compile(
    r"final_preview_v(\d+)\.mp4$",
    re.IGNORECASE,
)


def preview_sort_key(
    path: Path,
) -> tuple[
    int,
    float,
]:

    match = FINAL_VERSION_PATTERN.search(
        path.name
    )

    version = (
        int(
            match.group(1)
        )
        if match
        else 0
    )

    try:
        modified = path.stat().st_mtime
    except OSError:
        modified = 0.0

    return (
        version,
        modified,
    )


def find_base_final_video(
    timeline: dict[str, Any],
    clip_index: int,
) -> Path:

    video_stem = get_video_stem(
        timeline
    )

    directory = (
        FINAL_PREVIEWS_DIR
        / video_stem
    )

    if not directory.exists():
        raise FileNotFoundError(
            "Final preview klasörü bulunamadı:\n"
            f"{directory}\n\n"
            "Önce intro_renderer çalıştır."
        )

    candidates: list[
        Path
    ] = []

    for pattern in (
        f"clip_{clip_index:02d}_*final_preview_v*.mp4",
        f"clip_{clip_index:02d}_*final_preview.mp4",
    ):

        candidates.extend(
            directory.glob(
                pattern
            )
        )

    unique: dict[
        str,
        Path
    ] = {}

    for path in candidates:

        key = str(
            path.resolve()
        ).lower()

        unique[key] = path

    candidates = list(
        unique.values()
    )

    if not candidates:
        raise FileNotFoundError(
            "Bu clip için final preview bulunamadı:\n"
            f"{directory}"
        )

    candidates.sort(
        key=preview_sort_key,
        reverse=True,
    )

    return candidates[0].resolve()


# ============================================================
# DISCOVERY CLIP
# ============================================================

def get_discovery_clip(
    package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for position, clip in enumerate(
        package.get(
            "clips",
            [],
        ),
        start=1,
    ):

        if not isinstance(clip, dict):
            continue

        try:
            current = int(
                clip.get(
                    "clip_index",
                    position,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            current = position

        if current == clip_index:
            return clip

    raise IndexError(
        f"Discovery JSON içinde clip_index={clip_index} bulunamadı."
    )


# ============================================================
# ASSET PATH
# ============================================================

def resolve_local_asset(
    discovery: dict[str, Any],
) -> Path | None:

    asset = discovery.get(
        "asset"
    )

    if not isinstance(asset, dict):
        return None

    absolute_path = asset.get(
        "absolute_path"
    )

    if (
        isinstance(absolute_path, str)
        and absolute_path.strip()
    ):

        path = Path(
            absolute_path
        ).resolve()

        if path.exists():
            return path

    local_file = asset.get(
        "local_file"
    )

    if (
        isinstance(local_file, str)
        and local_file.strip()
    ):

        path = (
            MEME_LIBRARY_DIR
            / local_file
        ).resolve()

        if path.exists():
            return path

    return None


# ============================================================
# SINGLE BEST MEME
# ============================================================

def discovery_quality_score(
    discovery: dict[str, Any],
) -> tuple[
    float,
    float,
]:

    try:
        fit_score = float(
            discovery.get(
                "fit_score",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        fit_score = 0.0

    try:
        confidence = float(
            discovery.get(
                "confidence",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = 0.0

    return (
        fit_score,
        confidence,
    )


def choose_single_best_discovery(
    discovery_clip: dict[str, Any],
) -> dict[str, Any] | None:

    raw_discoveries = discovery_clip.get(
        "discoveries",
        [],
    )

    if not isinstance(
        raw_discoveries,
        list,
    ):
        return None

    selected = [
        item
        for item in raw_discoveries
        if (
            isinstance(item, dict)
            and item.get("selected") is True
        )
    ]

    if not selected:
        return None

    # Önce fit_score, sonra confidence.
    selected.sort(
        key=discovery_quality_score,
        reverse=True,
    )

    return selected[0]


# ============================================================
# BUILD SINGLE EVENT
# ============================================================

def build_single_event(
    discovery_clip: dict[str, Any],
    base_duration: float,
    timeline_doc: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any] | None,
    list[str],
]:

    warnings: list[
        str
    ] = []

    discovery = choose_single_best_discovery(
        discovery_clip
    )

    if discovery is None:
        return (
            None,
            warnings,
        )

    slot = discovery.get(
        "slot"
    )

    candidate = discovery.get(
        "candidate"
    )

    if (
        not isinstance(slot, dict)
        or not isinstance(candidate, dict)
    ):
        warnings.append(
            "En iyi discovery içinde slot/candidate eksik."
        )

        return (
            None,
            warnings,
        )

    asset_path = resolve_local_asset(
        discovery
    )

    if asset_path is None:
        warnings.append(
            "Seçilen discovery asset dosyası cache içinde bulunamadı."
        )

        return (
            None,
            warnings,
        )

    media_type = str(
        candidate.get(
            "media_type",
            "",
        )
    ).strip().lower()

    if media_type not in {
        "audio",
        "visual",
    }:
        warnings.append(
            f"Geçersiz media_type: {media_type}"
        )

        return (
            None,
            warnings,
        )

    timing = slot.get(
        "timing"
    )

    if not isinstance(timing, dict):
        warnings.append(
            "Slot timing bilgisi eksik."
        )

        return (
            None,
            warnings,
        )

    try:
        start = float(
            timing[
                "final_start"
            ]
        )

        requested_duration = float(
            timing.get(
                "max_duration",
                0.0,
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):
        warnings.append(
            "final_start / max_duration geçersiz."
        )

        return (
            None,
            warnings,
        )

    start = max(
        0.0,
        start,
    )

    # The final clock is the intro renderer's timeline document (cold open +
    # hard restart at main_restart), never "teaser length + main time": the
    # restart trim would otherwise land the meme seconds late.
    intro_end = None
    if isinstance(timeline_doc, dict):
        from ai.editor import intro_renderer

        try:
            paced = float(timing["main_edited_start"])
        except (KeyError, TypeError, ValueError):
            warnings.append("main_edited_start yok; meme final saatine eşlenemedi.")
            return (None, warnings)
        mapped = intro_renderer.paced_to_final(timeline_doc, paced)
        intro_end = float((timeline_doc.get("intro") or {}).get("duration", 0.0) or 0.0)
        if mapped is None:
            warnings.append("Meme anı story restart'ından önce kesilmiş bölgede; meme atlandı.")
            return (None, warnings)
        if mapped < intro_end + MAIN_OPENING_PROTECTION:
            warnings.append("Meme cold open'a veya restart'ın hemen başına denk geliyor; meme atlandı.")
            return (None, warnings)
        start = mapped

    if start >= base_duration:
        warnings.append(
            "Meme başlangıcı final videonun dışında."
        )

        return (
            None,
            warnings,
        )

    if requested_duration <= 0:
        warnings.append(
            "Meme duration geçersiz."
        )

        return (
            None,
            warnings,
        )

    asset_info = get_media_info(
        asset_path
    )

    if (
        media_type == "audio"
        and not asset_info[
            "has_audio"
        ]
    ):
        warnings.append(
            "Audio meme dosyasında audio stream yok."
        )

        return (
            None,
            warnings,
        )

    if (
        media_type == "visual"
        and not asset_info[
            "has_video"
        ]
    ):
        warnings.append(
            "Visual meme dosyasında video/image stream yok."
        )

        return (
            None,
            warnings,
        )

    suffix = asset_path.suffix.lower()

    is_static = (
        media_type == "visual"
        and suffix in STATIC_VISUAL_EXTENSIONS
        and asset_info[
            "duration"
        ] <= 0.10
    )

    asset_duration = float(
        asset_info[
            "duration"
        ]
    )

    preserve_full_local_audio = (
        media_type == "audio"
        and str(candidate.get("provider", "")).strip().lower() == "local_sfx"
        and asset_duration > 0
    )

    if preserve_full_local_audio:
        duration = asset_duration

        timing_main_start = timing.get("main_edited_start", 0.0)
        try:
            timing_main_start = float(timing_main_start)
        except (TypeError, ValueError):
            timing_main_start = 0.0
        teaser_offset = max(0.0, start - timing_main_start)
        earliest_start = min(base_duration, (intro_end if intro_end is not None else teaser_offset)
                             + MAIN_OPENING_PROTECTION)

        if duration > (base_duration - earliest_start) + 1e-6:
            warnings.append(
                f"Local SFX tam süresi ({duration:.3f}s) final videoya kesmeden sığmıyor; meme atlandı."
            )
            return (None, warnings)

        if start + duration > base_duration:
            shifted = max(earliest_start, base_duration - duration)
            warnings.append(
                f"Local SFX kesilmemesi için başlangıç {start:.3f}s -> {shifted:.3f}s kaydırıldı."
            )
            start = shifted

    elif is_static:
        duration = min(requested_duration, base_duration - start)

    elif asset_duration > 0:
        duration = min(requested_duration, asset_duration, base_duration - start)

    else:
        duration = min(requested_duration, base_duration - start)

    minimum = (
        0.18
        if media_type == "audio"
        else 0.30
    )

    if duration < minimum:
        warnings.append(
            f"Meme süresi çok kısa: {duration:.3f}s"
        )

        return (
            None,
            warnings,
        )

    event = {
        "media_type": media_type,

        "name": str(
            candidate.get(
                "name",
                asset_path.stem,
            )
        ),

        "provider": str(
            candidate.get(
                "provider",
                "",
            )
        ),

        "asset_path": str(
            asset_path
        ),

        "asset_suffix": suffix,

        "is_static_visual": is_static,

        "start": start,

        "duration": duration,

        "full_asset_preserved": bool(preserve_full_local_audio),
        "asset_duration": asset_duration,

        "end": (
            start
            + duration
        ),

        "fit_score": discovery.get(
            "fit_score"
        ),

        "confidence": discovery.get(
            "confidence"
        ),

        "slot_intent": str(
            slot.get(
                "intent",
                "",
            )
        ),

        "sound_function": str(
            slot.get(
                "sound_function",
                "none",
            )
        ),

        "strength": str(
            slot.get(
                "strength",
                "subtle",
            )
        ),

        "audio_mix": (
            slot.get("audio_mix", {})
            if isinstance(slot.get("audio_mix"), dict)
            else {}
        ),
    }

    if media_type == "audio":
        event["render_audio_mix"] = resolve_audio_mix_params(event)

    return (
        event,
        warnings,
    )


# ============================================================
# OUTPUT
# ============================================================

def get_output_path(
    timeline: dict[str, Any],
    discovery_clip: dict[str, Any],
    clip_index: int,
) -> Path:

    directory = (
        FINAL_MEMES_DIR
        / get_video_stem(
            timeline
        )
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    title = safe_filename(
        discovery_clip.get(
            "title",
            "",
        )
    )

    return (
        directory
        / (
            f"clip_{clip_index:02d}_"
            f"{title}_"
            "meme_final.mp4"
        )
    ).resolve()


def get_report_path(
    timeline: dict[str, Any],
    clip_index: int,
) -> Path:

    directory = (
        REPORT_DIR
        / get_video_stem(
            timeline
        )
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        directory
        / (
            f"clip_{clip_index:02d}_"
            "meme_render_v2.json"
        )
    ).resolve()


# ============================================================
# PASSTHROUGH
# ============================================================

def passthrough_copy(
    source: Path,
    target: Path,
) -> None:

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        target,
    )


# ============================================================
# FILTERS — AUDIO
# ============================================================

def resolve_audio_mix_params(event: dict[str, Any]) -> dict[str, float | str]:
    strength = str(event.get("strength", "subtle")).strip().lower()
    base_volume = {
        "subtle": AUDIO_VOLUME_SUBTLE,
        "medium": AUDIO_VOLUME_MEDIUM,
        "strong": AUDIO_VOLUME_STRONG,
    }.get(strength, AUDIO_VOLUME_SUBTLE)

    mix = event.get("audio_mix", {})
    if not isinstance(mix, dict):
        mix = {}

    risk = str(mix.get("speech_overlap_risk", "unknown")).strip().lower()
    try:
        headroom = float(mix.get("insertion_headroom_score", 0.50))
    except (TypeError, ValueError):
        headroom = 0.50
    headroom = clamp(headroom, 0.0, 1.0)

    # High speech-overlap => SFX stays underneath; do not duck dialogue hard.
    if risk == "high":
        volume_factor = 0.68
        sidechain_ratio = 1.15
        release_ms = 95
    elif risk == "medium":
        volume_factor = 0.84
        sidechain_ratio = 1.55
        release_ms = 120
    else:
        volume_factor = 1.00
        sidechain_ratio = 1.95
        release_ms = SIDECHAIN_RELEASE_MS

    # Clean pause/headroom can tolerate a touch more; crowded audio gets less.
    headroom_factor = 0.90 + (0.16 * headroom)
    volume = clamp(base_volume * volume_factor * headroom_factor, 0.24, 0.68)

    # Full-length local reaction sounds can be several seconds long. Keep them
    # audible, but do not hold the source dialogue under an aggressive duck for
    # the whole asset. This changes mix only; the asset itself remains intact.
    try:
        event_duration = float(event.get("duration", 0.0))
    except (TypeError, ValueError):
        event_duration = 0.0
    if bool(event.get("full_asset_preserved")) and event_duration >= 3.0:
        volume = clamp(volume * 0.84, 0.22, 0.58)
        sidechain_ratio = min(sidechain_ratio, 1.40)
        release_ms = max(release_ms, 165)
    elif bool(event.get("full_asset_preserved")) and event_duration >= 1.6:
        volume = clamp(volume * 0.93, 0.23, 0.62)
        sidechain_ratio = min(sidechain_ratio, 1.70)
        release_ms = max(release_ms, 145)

    return {
        "speech_overlap_risk": risk,
        "volume": round(volume, 4),
        "sidechain_ratio": round(sidechain_ratio, 3),
        "sidechain_threshold": SIDECHAIN_THRESHOLD,
        "sidechain_attack_ms": SIDECHAIN_ATTACK_MS,
        "sidechain_release_ms": release_ms,
    }


def build_audio_filter(
    input_index: int,
    event: dict[str, Any],
    base_duration: float,
    base_has_audio: bool,
) -> tuple[
    str,
    str,
]:

    start = float(
        event[
            "start"
        ]
    )

    duration = float(
        event[
            "duration"
        ]
    )

    mix_params = resolve_audio_mix_params(event)
    meme_volume = float(mix_params["volume"])
    sidechain_ratio = float(mix_params["sidechain_ratio"])
    sidechain_threshold = float(mix_params["sidechain_threshold"])
    sidechain_attack_ms = int(mix_params["sidechain_attack_ms"])
    sidechain_release_ms = int(mix_params["sidechain_release_ms"])

    fade_in = min(
        AUDIO_FADE_IN,
        duration * 0.28,
    )

    fade_out = min(
        AUDIO_FADE_OUT,
        duration * 0.40,
    )

    fade_out_start = max(
        0.0,
        duration - fade_out,
    )

    delay_ms = max(
        0,
        int(
            round(
                start * 1000
            )
        ),
    )

    chains = [
        (
            f"[{input_index}:a]"
            f"atrim=start=0:end={duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"aresample={AUDIO_SAMPLE_RATE},"
            f"aformat="
            f"sample_fmts=fltp:"
            f"sample_rates={AUDIO_SAMPLE_RATE}:"
            f"channel_layouts=stereo,"
            "highpass=f=35,"
            f"volume={meme_volume:.6f},"
            f"afade="
            f"t=in:"
            f"st=0:"
            f"d={fade_in:.6f},"
            f"afade="
            f"t=out:"
            f"st={fade_out_start:.6f}:"
            f"d={fade_out:.6f},"
            f"adelay={delay_ms}|{delay_ms},"
            f"apad,"
            f"atrim=duration={base_duration:.6f}"
            f"[meme_audio]"
        ),

        (
            "[meme_audio]"
            "asplit=2"
            "[meme_sidechain]"
            "[meme_mix]"
        ),
    ]

    if base_has_audio:

        chains.extend(
            [
                (
                    "[0:a]"
                    f"aresample={AUDIO_SAMPLE_RATE},"
                    f"aformat="
                    f"sample_fmts=fltp:"
                    f"sample_rates={AUDIO_SAMPLE_RATE}:"
                    f"channel_layouts=stereo,"
                    f"atrim=duration={base_duration:.6f},"
                    f"asetpts=PTS-STARTPTS"
                    "[base_audio]"
                ),

                (
                    "[base_audio]"
                    "[meme_sidechain]"
                    "sidechaincompress="
                    f"threshold={sidechain_threshold}:"
                    f"ratio={sidechain_ratio}:"
                    f"attack={sidechain_attack_ms}:"
                    f"release={sidechain_release_ms}"
                    "[ducked]"
                ),

                (
                    "[ducked]"
                    "[meme_mix]"
                    "amix="
                    "inputs=2:"
                    "duration=first:"
                    "dropout_transition=0:"
                    "normalize=0,"
                    f"alimiter=limit={OUTPUT_LIMITER}"
                    "[out_a]"
                ),
            ]
        )

    else:

        chains.append(
            (
                "[meme_mix]"
                f"alimiter=limit={OUTPUT_LIMITER}"
                "[out_a]"
            )
        )

    return (
        ";".join(chains),
        "[out_a]",
    )


# ============================================================
# FILTERS — VISUAL
# ============================================================

def overlay_size(base_info: dict[str, Any], asset_info: dict[str, Any] | None) -> tuple[int, int]:
    width, height = int(base_info["width"]), int(base_info["height"])
    box_w = max(96, int(round(width * VISUAL_WIDTH_PERCENT / 100.0)))
    box_h = max(96, int(round(height * VISUAL_MAX_HEIGHT_PERCENT / 100.0)))
    try:
        aw, ah = int((asset_info or {}).get("width") or 0), int((asset_info or {}).get("height") or 0)
    except (TypeError, ValueError):
        aw = ah = 0
    if aw <= 0 or ah <= 0:
        return box_w, box_h
    scale = min(box_w / aw, box_h / ah)
    return max(2, int(aw * scale)), max(2, int(ah * scale))


def choose_overlay_position(
    base_video: Path,
    base_info: dict[str, Any],
    start: float,
    duration: float,
    size: tuple[int, int],
    forbidden_bands: list[tuple[float, float]],
) -> tuple[int, int, str] | None:
    """Least-busy anchor for a visual meme, never over a caption band.

    Busyness = spatial detail + motion of the ACTUAL rendered frames under the
    overlay during its window (faces, action and HUD are busy; sky, walls and
    letterbox are not). None when every anchor would cover captions."""
    from ai.editor import final_qc

    np = final_qc._numpy()
    width, height = int(base_info["width"]), int(base_info["height"])
    fps = float(base_info["fps"])
    ow, oh = size
    margin_x, margin_y = int(width * VISUAL_EDGE_MARGIN), int(height * VISUAL_EDGE_MARGIN)
    candidates = []
    for name, ax, ay in VISUAL_ANCHORS:
        x = int(round(margin_x + ax * (width - ow - 2 * margin_x)))
        y = int(round(margin_y + ay * (height - oh - 2 * margin_y)))
        y0, y1 = y / height, (y + oh) / height
        if any(y0 < b1 and b0 < y1 for b0, b1 in forbidden_bands):
            continue
        candidates.append((name, x, y))
    if not candidates:
        return None
    aw = final_qc.ANALYSIS_WIDTH
    ah = max(16, int(round(height * aw / max(1, width))))
    ah -= ah % 2
    frames = [int(round((start + duration * k / 4.0) * fps)) for k in range(5)]
    decoded = [f for _, f in sorted(final_qc.decode_frames(base_video, frames, aw, ah).items())]
    if not decoded:
        return candidates[0][1], candidates[0][2], candidates[0][0] + " (no frames decoded)"
    stack = np.stack(decoded)
    detail = np.abs(np.diff(stack, axis=2)).mean(axis=0)
    detail = np.pad(detail, ((0, 0), (0, 1)), mode="edge")
    motion = np.abs(np.diff(stack, axis=0)).mean(axis=0) if len(decoded) > 1 else np.zeros_like(detail)
    busy = detail + 2.0 * motion
    fx, fy = aw / width, ah / height
    best = None
    for name, x, y in candidates:
        region = busy[int(y * fy):max(int(y * fy) + 1, int((y + oh) * fy)),
                      int(x * fx):max(int(x * fx) + 1, int((x + ow) * fx))]
        score = float(region.mean()) if region.size else float("inf")
        if best is None or score < best[0] - 1e-6:
            best = (score, name, x, y)
    assert best is not None
    return best[2], best[3], best[1]


def build_visual_filter(
    input_index: int,
    event: dict[str, Any],
    base_info: dict[str, Any],
) -> tuple[
    str,
    str,
]:

    width = int(
        base_info[
            "width"
        ]
    )

    height = int(
        base_info[
            "height"
        ]
    )

    fps = float(
        base_info[
            "fps"
        ]
    )

    start = float(
        event[
            "start"
        ]
    )

    duration = float(
        event[
            "duration"
        ]
    )

    target_width = max(
        96,
        int(
            round(
                width
                * (
                    VISUAL_WIDTH_PERCENT
                    / 100.0
                )
            )
        ),
    )

    target_height = max(
        96,
        int(
            round(
                height
                * (
                    VISUAL_MAX_HEIGHT_PERCENT
                    / 100.0
                )
            )
        ),
    )

    fade_in = min(
        VISUAL_FADE_IN,
        duration * 0.28,
    )

    fade_out = min(
        VISUAL_FADE_OUT,
        duration * 0.34,
    )

    fade_out_start = max(
        0.0,
        duration - fade_out,
    )

    is_static = bool(
        event[
            "is_static_visual"
        ]
    )

    if is_static:

        source_chain = (
            f"[{input_index}:v]"
            "loop="
            "loop=-1:"
            "size=1:"
            "start=0,"
            f"setpts=N/({fps:.6f}*TB),"
            f"trim=duration={duration:.6f},"
        )

    else:

        source_chain = (
            f"[{input_index}:v]"
            f"trim=start=0:end={duration:.6f},"
            "setpts=PTS-STARTPTS,"
            f"fps={fps:.6f},"
        )

    # Önce boyutlandır.
    source_chain += (
        f"scale="
        f"w={target_width}:"
        f"h={target_height}:"
        "force_original_aspect_ratio=decrease:"
        "flags=lanczos,"
        "setsar=1,"
        "format=rgba,"
    )

    # Hafif transparan.
    source_chain += (
        f"colorchannelmixer=aa={VISUAL_OPACITY:.6f},"
    )

    # Smooth fade alpha.
    source_chain += (
        "fade="
        "t=in:"
        "st=0:"
        f"d={fade_in:.6f}:"
        "alpha=1,"
        "fade="
        "t=out:"
        f"st={fade_out_start:.6f}:"
        f"d={fade_out:.6f}:"
        "alpha=1,"
    )

    # Final timeline'a kaydır.
    source_chain += (
        f"setpts=PTS-STARTPTS+{start:.6f}/TB"
        "[meme_visual]"
    )

    filter_complex = ";".join(
        [
            (
                "[0:v]"
                f"fps={fps:.6f},"
                "setsar=1,"
                f"format={PIXEL_FORMAT},"
                "setpts=PTS-STARTPTS"
                "[base_v]"
            ),

            source_chain,

            (
                "[base_v]"
                "[meme_visual]"
                "overlay="
                f"x={int(event.get('overlay_x', 0))}:"
                f"y={int(event.get('overlay_y', 0))}:"
                "eof_action=pass:"
                "repeatlast=0:"
                "shortest=0,"
                f"format={PIXEL_FORMAT},"
                "setsar=1"
                "[out_v]"
            ),
        ]
    )

    return (
        filter_complex,
        "[out_v]",
    )


# ============================================================
# RENDER
# ============================================================

def build_render_command(
    *,
    base_video: Path,
    asset_path: Path,
    event: dict[str, Any],
    base_info: dict[str, Any],
    output_path: Path,
    profile: render_backend.EncoderProfile | None,
) -> list[str]:
    """One meme event. Audio: video stream-copied (``profile`` unused). Visual:
    CPU overlay graph, video encoder from the render backend, audio copied."""

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(
            base_video
        ),
        "-i",
        str(
            asset_path
        ),
    ]

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    if event[
        "media_type"
    ] == "audio":

        (
            filter_complex,
            audio_label,
        ) = build_audio_filter(
            input_index=1,
            event=event,
            base_duration=float(
                base_info[
                    "duration"
                ]
            ),
            base_has_audio=bool(
                base_info[
                    "has_audio"
                ]
            ),
        )

        command.extend(
            [
                "-filter_complex",
                filter_complex,

                "-map",
                "0:v:0",

                "-map",
                audio_label,

                "-c:v",
                "copy",

                "-c:a",
                AUDIO_CODEC,

                "-b:a",
                AUDIO_BITRATE,
            ]
        )

    # --------------------------------------------------------
    # VISUAL
    # --------------------------------------------------------

    else:

        if profile is None:
            raise ValueError("visual meme render needs a render-backend encoder profile")

        (
            filter_complex,
            video_label,
        ) = build_visual_filter(
            input_index=1,
            event=event,
            base_info=base_info,
        )

        command.extend(
            [
                "-filter_complex",
                filter_complex,

                "-map",
                video_label,

                *profile.video_args(),
            ]
        )

        if base_info[
            "has_audio"
        ]:

            command.extend(
                [
                    "-map",
                    "0:a:0?",

                    "-c:a",
                    "copy",
                ]
            )

    command.extend(
        [
            "-map_metadata",
            "0",

            "-movflags",
            "+faststart",

            # No -shortest: both branches already keep the base length exactly
            # (audio apad+atrim to it, overlay eof_action=pass). With a stream-
            # copied video, -shortest cut the short at the muxer's interleave
            # point and silently dropped its last frames.
            str(
                output_path
            ),
        ]
    )

    return command


def render_clip(
    discovery_package: dict[str, Any],
    timeline: dict[str, Any],
    clip_index: int,
    *,
    base_video_path: str | Path | None = None,
    forbidden_bands: list[tuple[float, float]] | None = None,
) -> Path:

    discovery_clip = get_discovery_clip(
        discovery_package,
        clip_index,
    )

    # The pipeline hands over the exact composed short; a directory glob could
    # adopt another source's or an older run's preview.
    if base_video_path is not None:
        base_video = Path(base_video_path).expanduser().resolve()
        if not base_video.is_file():
            raise FileNotFoundError(f"Base final video bulunamadı:\n{base_video}")
    else:
        base_video = find_base_final_video(
            timeline,
            clip_index,
        )
    from ai.editor import intro_renderer

    timeline_doc = intro_renderer.load_final_timeline(base_video)

    base_info = get_media_info(
        base_video
    )

    if (
        not base_info[
            "has_video"
        ]
        or base_info[
            "duration"
        ] <= 0
        or base_info[
            "width"
        ] <= 0
        or base_info[
            "height"
        ] <= 0
    ):
        raise RuntimeError(
            "Base final video bilgisi geçersiz."
        )

    output_path = get_output_path(
        timeline,
        discovery_clip,
        clip_index,
    )

    report_path = get_report_path(
        timeline,
        clip_index,
    )

    (
        event,
        warnings,
    ) = build_single_event(
        discovery_clip,
        float(
            base_info[
                "duration"
            ]
        ),
        timeline_doc,
    )

    if event is not None and event["media_type"] == "visual":
        asset_info = get_media_info(event["asset_path"])
        size = overlay_size(base_info, asset_info)
        placement = choose_overlay_position(
            base_video, base_info, float(event["start"]), float(event["duration"]), size,
            list(forbidden_bands or []),
        )
        if placement is None:
            warnings.append("Visual meme her konumda caption bandını kapatıyor; meme atlandı.")
            event = None
        else:
            event["overlay_x"], event["overlay_y"], event["overlay_anchor"] = placement
            event["overlay_size"] = list(size)

    print()
    print(
        "=" * 74
    )

    print(
        f"🎬 MEME RENDERER V2 — CLIP {clip_index}"
    )

    print(
        "=" * 74
    )

    print(
        f"📥 Base:\n{base_video}"
    )

    print(
        f"📐 "
        f"{base_info['width']}x"
        f"{base_info['height']} "
        f"@ {base_info['fps']:.3f}"
    )

    print(
        f"⏱️ {base_info['duration']:.3f}s"
    )

    # ========================================================
    # NO MEME
    # ========================================================

    if event is None:

        print()
        print(
            "👌 Kullanılacak meme yok."
        )

        print(
            "📦 Videoya dokunmadan passthrough kopyalanıyor..."
        )

        passthrough_copy(
            base_video,
            output_path,
        )

        save_json(
            report_path,
            {
                "version": MEME_RENDERER_VERSION,

                "clip_index": clip_index,

                "mode": "passthrough",

                "base_video": str(
                    base_video
                ),

                "output_video": str(
                    output_path
                ),

                "event_count": 0,

                "warnings": warnings,
            },
        )

        print()
        print(
            "✅ PASSTHROUGH FINAL HAZIR"
        )

        print(
            f"📂 {output_path}"
        )

        return output_path

    # ========================================================
    # SINGLE MEME
    # ========================================================

    print()
    print(
        "🏆 TEK MEME SEÇİLDİ"
    )

    print(
        (
            f"   {event['name']}\n"
            f"   type={event['media_type']}\n"
            f"   start={event['start']:.3f}s\n"
            f"   duration={event['duration']:.3f}s\n"
            f"   fit={event.get('fit_score')}\n"
            f"   confidence={event.get('confidence')}"
        )
    )

    asset_path = Path(
        event[
            "asset_path"
        ]
    ).resolve()

    print()
    print(
        "🎞️ Render başlıyor..."
    )

    def run_meme_command(command: list[str]) -> None:
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                "FFmpeg bulunamadı."
            ) from error

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg Meme Renderer V2 hatası:\n\n"
                + result.stderr
            )

        if not output_path.exists():
            raise RuntimeError(
                "Meme final video oluşmadı."
            )

    if event[
        "media_type"
    ] == "audio":

        # Audio-only effect: the video stream is copied, no encoder involved.
        run_meme_command(
            build_render_command(
                base_video=base_video,
                asset_path=asset_path,
                event=event,
                base_info=base_info,
                output_path=output_path,
                profile=None,
            )
        )

    else:

        def encode(profile: render_backend.EncoderProfile) -> None:
            run_meme_command(
                build_render_command(
                    base_video=base_video,
                    asset_path=asset_path,
                    event=event,
                    base_info=base_info,
                    output_path=output_path,
                    profile=profile,
                )
            )
            render_backend.check_output(
                output_path,
                profile,
            )

        render_backend.run_encode(
            "meme_render",
            encode,
        )

    output_info = get_media_info(
        output_path
    )

    duration_difference = abs(
        float(
            output_info[
                "duration"
            ]
        )
        - float(
            base_info[
                "duration"
            ]
        )
    )

    if duration_difference > DURATION_TOLERANCE:
        warnings.append(
            (
                "Output duration farkı yüksek: "
                f"{duration_difference:.3f}s"
            )
        )

    save_json(
        report_path,
        {
            "version": MEME_RENDERER_VERSION,

            "clip_index": clip_index,

            "mode": "single_best_meme",

            "policy": {
                "max_memes_per_clip": 1,

                "visual_position": event.get("overlay_anchor", "n/a"),

                "visual_width_percent": (
                    VISUAL_WIDTH_PERCENT
                ),

                "visual_opacity": (
                    VISUAL_OPACITY
                ),
            },

            "base_video": str(
                base_video
            ),

            "output_video": str(
                output_path
            ),

            "base_duration": round(
                float(
                    base_info[
                        "duration"
                    ]
                ),
                3,
            ),

            "output_duration": round(
                float(
                    output_info[
                        "duration"
                    ]
                ),
                3,
            ),

            "event_count": 1,

            "event": event,

            "warnings": warnings,
        },
    )

    print()
    print(
        "✅ MEME FINAL HAZIR"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        f"🧾 {report_path}"
    )

    return output_path


# ============================================================
# RENDER ALL
# ============================================================

def render_memes(
    discovery_json_path: str | Path,
    clip_index: int | None = None,
    *,
    base_video_path: str | Path | None = None,
    forbidden_bands: list[tuple[float, float]] | None = None,
) -> list[Path]:

    discovery_json_path = Path(
        discovery_json_path
    ).resolve()

    discovery_package = load_json(
        discovery_json_path
    )

    validate_discovery_package(
        discovery_package
    )

    slot_json_path = get_slot_json_path(
        discovery_package
    )

    slot_package = load_json(
        slot_json_path
    )

    validate_slot_package(
        slot_package
    )

    timeline_path = get_timeline_path(
        slot_package
    )

    timeline = load_json(
        timeline_path
    )

    validate_timeline(
        timeline
    )

    if clip_index is not None:

        return [
            render_clip(
                discovery_package=discovery_package,
                timeline=timeline,
                clip_index=clip_index,
                base_video_path=base_video_path,
                forbidden_bands=forbidden_bands,
            )
        ]

    outputs: list[
        Path
    ] = []

    for position, clip in enumerate(
        discovery_package.get(
            "clips",
            [],
        ),
        start=1,
    ):

        if not isinstance(clip, dict):
            continue

        try:
            current_index = int(
                clip.get(
                    "clip_index",
                    position,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            current_index = position

        outputs.append(
            render_clip(
                discovery_package=discovery_package,
                timeline=timeline,
                clip_index=current_index,
            )
        )

    return outputs


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "MIMIR Meme Renderer V3 — Smooth Audio SFX"
    )

    print(
        "Single best meme only"
    )

    print(
        "Visual: least-busy anchor, never over captions"
    )

    print()

    discovery_json_path = input(
        "Meme discovery JSON yolunu gir: "
    ).strip().strip('"')

    clip_input = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    try:

        selected_clip = (
            int(
                clip_input
            )
            if clip_input
            else None
        )

        render_memes(
            discovery_json_path=discovery_json_path,
            clip_index=selected_clip,
        )

    except Exception as error:

        print()
        print(
            "❌ MEME RENDERER V3 HATASI:"
        )

        print(
            error
        )
