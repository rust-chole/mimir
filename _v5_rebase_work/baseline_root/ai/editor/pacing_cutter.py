from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "vod_output"
    / "edited_clips"
)


# ============================================================
# CONFIG
# ============================================================

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = "18"

CUTTER_REVISION = 9

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"

MIN_SEGMENT_DURATION = 0.02
DURATION_WARNING_TOLERANCE = 0.20

# Final safety net. Even if an upstream timeline accidentally contains a
# cut over a Terra money moment, renderer refuses that cut.
PROTECTED_GUARD_SECONDS = 0.015

# Defense-in-depth: if a malformed/stale timeline would shorten the clip below
# the story-duration floor, drop automatic cuts rather than destroy the Short.
STORY_DURATION_GUARD_TOLERANCE = 0.08


# ============================================================
# GENERIC HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def round_time(
    value: float,
) -> float:

    return round(
        float(value),
        3,
    )


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


def safe_filename(
    text: str,
) -> str:

    invalid = '<>:"/\\|?*'

    result = str(text)

    for character in invalid:
        result = result.replace(
            character,
            "_",
        )

    result = result.strip(
        " ."
    )

    if not result:
        return "clip"

    return result


# ============================================================
# FFPROBE
# ============================================================

def video_has_audio(
    video_path: str | Path,
) -> bool:

    video_path = Path(
        video_path
    ).resolve()

    command = [
        "ffprobe",
        "-v",
        "error",

        "-select_streams",
        "a:0",

        "-show_entries",
        "stream=index",

        "-of",
        "csv=p=0",

        str(video_path),
    ]

    try:

        result = subprocess.run(
            command,
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

    return bool(
        result.stdout.strip()
    )


def get_video_duration(
    video_path: str | Path,
) -> float:

    video_path = Path(
        video_path
    ).resolve()

    command = [
        "ffprobe",
        "-v",
        "error",

        "-show_entries",
        "format=duration",

        "-of",
        "default=noprint_wrappers=1:nokey=1",

        str(video_path),
    ]

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

    except FileNotFoundError as error:

        raise RuntimeError(
            "ffprobe bulunamadı."
        ) from error

    except subprocess.CalledProcessError as error:

        raise RuntimeError(
            f"Video süresi okunamadı:\n{error.stderr}"
        ) from error

    try:

        return float(
            result.stdout.strip()
        )

    except ValueError as error:

        raise RuntimeError(
            "ffprobe geçersiz duration döndürdü."
        ) from error


# ============================================================
# TIMELINE
# ============================================================

def validate_timeline(
    timeline_data: dict[str, Any],
) -> None:

    version = int(
        timeline_data.get(
            "version",
            -1,
        )
    )

    if version != 3:

        raise RuntimeError(
            f"Timeline V3 gerekiyor. Bulunan version: {version}"
        )

    timelines = timeline_data.get(
        "timelines",
        [],
    )

    if not timelines:

        raise RuntimeError(
            "Timeline içinde klip bulunamadı."
        )

    source = timeline_data.get(
        "source",
        {},
    )

    video_path = str(
        source.get(
            "video_path",
            "",
        )
    ).strip()

    if not video_path:

        raise RuntimeError(
            "Timeline source.video_path içermiyor."
        )


def get_clip_timeline(
    timeline_data: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for timeline in timeline_data.get(
        "timelines",
        [],
    ):

        if int(
            timeline.get(
                "clip_index",
                -1,
            )
        ) == int(
            clip_index
        ):

            return timeline

    raise RuntimeError(
        f"clip_index={clip_index} bulunamadı."
    )


# ============================================================
# CUT RANGES
# ============================================================


# ============================================================
# TERRA PROTECTED-RANGE SAFETY NET
# ============================================================

def normalize_protected_ranges(
    clip_timeline: dict[str, Any],
) -> list[dict[str, Any]]:

    source = clip_timeline.get(
        "source",
        {},
    )

    clip_duration = float(
        source.get(
            "duration",
            0.0,
        )
    )

    if clip_duration <= 0:
        return []

    raw = clip_timeline.get(
        "protected_ranges",
        [],
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    result: list[
        dict[str, Any]
    ] = []

    for item in raw:

        if not isinstance(
            item,
            dict,
        ):
            continue

        start = clamp(
            float(
                item.get(
                    "start",
                    0.0,
                )
            )
            - PROTECTED_GUARD_SECONDS,
            0.0,
            clip_duration,
        )

        end = clamp(
            float(
                item.get(
                    "end",
                    start,
                )
            )
            + PROTECTED_GUARD_SECONDS,
            start,
            clip_duration,
        )

        if end <= start:
            continue

        result.append(
            {
                "start": round_time(
                    start
                ),

                "end": round_time(
                    end
                ),

                "reason": str(
                    item.get(
                        "reason",
                        "Terra protected range.",
                    )
                ),
            }
        )

    result.sort(
        key=lambda item: (
            float(
                item["start"]
            ),
            float(
                item["end"]
            ),
        )
    )

    return result


def ranges_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:

    return (
        start_a < end_b
        and end_a > start_b
    )


def sanitize_cut_ranges_against_protected(
    cut_ranges: list[dict[str, float]],
    protected_ranges: list[dict[str, Any]],
) -> tuple[
    list[dict[str, float]],
    list[dict[str, Any]],
]:

    if not protected_ranges:
        return (
            cut_ranges,
            [],
        )

    safe: list[
        dict[str, float]
    ] = []

    blocked: list[
        dict[str, Any]
    ] = []

    for cut in cut_ranges:

        cut_start = float(
            cut["start"]
        )

        cut_end = float(
            cut["end"]
        )

        hit: dict[str, Any] | None = None

        for protected in protected_ranges:

            if ranges_overlap(
                cut_start,
                cut_end,
                float(
                    protected["start"]
                ),
                float(
                    protected["end"]
                ),
            ):

                hit = protected
                break

        if hit is None:

            safe.append(
                cut
            )

            continue

        blocked.append(
            {
                "start": round_time(
                    cut_start
                ),

                "end": round_time(
                    cut_end
                ),

                "reason": (
                    "Renderer blocked cut over Terra protected range: "
                    + str(
                        hit.get(
                            "reason",
                            "",
                        )
                    )
                ),
            }
        )

    return (
        safe,
        blocked,
    )

def normalize_cut_ranges(
    clip_timeline: dict[str, Any],
) -> list[dict[str, float]]:

    source = clip_timeline.get(
        "source",
        {},
    )

    clip_duration = float(
        source.get(
            "duration",
            0,
        )
    )

    if clip_duration <= 0:

        raise RuntimeError(
            "Klip duration geçersiz."
        )

    ranges: list[
        dict[str, float]
    ] = []

    for item in clip_timeline.get(
        "cut_ranges",
        [],
    ):

        start = clamp(
            float(
                item.get(
                    "start",
                    0,
                )
            ),
            0.0,
            clip_duration,
        )

        end = clamp(
            float(
                item.get(
                    "end",
                    0,
                )
            ),
            0.0,
            clip_duration,
        )

        if end <= start:
            continue

        ranges.append(
            {
                "start": start,
                "end": end,
            }
        )

    ranges.sort(
        key=lambda item: item[
            "start"
        ]
    )

    if not ranges:
        return []

    merged: list[
        dict[str, float]
    ] = []

    current = dict(
        ranges[0]
    )

    for item in ranges[1:]:

        if (
            item["start"]
            <= current["end"]
            + 0.001
        ):

            current["end"] = max(
                current["end"],
                item["end"],
            )

        else:

            merged.append(
                current
            )

            current = dict(
                item
            )

    merged.append(
        current
    )

    return [
        {
            "start": round_time(
                item["start"]
            ),

            "end": round_time(
                item["end"]
            ),
        }

        for item in merged
    ]


# ============================================================
# SYNC FIX (V25): TEK NOKTADAN NİHAİ CUT_RANGES + GERİ YAZMA
# ============================================================
#
# KÖK NEDEN: `sanitize_cut_ranges_against_protected` ve
# `enforce_story_duration_guard`, timeline.py tarafından zaten hesaplanıp
# JSON'a yazılmış `cut_ranges`'i render anında SESSİZCE değiştirebiliyor
# (kesim iptal etme / protected-range çakışmasını engelleme). Ama
# captions.py, "edited_time" hesaplarken timeline dosyasındaki ORİJİNAL
# (guard öncesi) cut_ranges'i okuyor. İki taraf farklı cut listeleri
# kullandığı an, o noktadan sonraki tüm altyazı/emphasis/SFX zamanlaması
# gerçek sesle senkronunu kaybediyor - bildirilen "ses kayması" budur.
# Bu guard'lar SADECE belirli klip/kesim kombinasyonlarında devreye
# girdiği için sorun "bazı videolarda" ortaya çıkıyor.
#
# ÇÖZÜM: guard'lar bir şeyi değiştirdiğinde, düzeltilmiş cut_ranges'i
# clip_timeline'a geri yazıyoruz ve (mümkünse) timeline JSON dosyasını
# diske kaydediyoruz - render, captions.py'den ÖNCE çalıştığı için
# captions.py bir sonraki adımda her zaman aynı (nihai) cut listesini
# okumuş oluyor.
# ============================================================

def resolve_final_cut_ranges(
    clip_timeline: dict[str, Any],
    clip_duration: float,
    timeline_data: dict[str, Any] | None = None,
    timeline_path: str | Path | None = None,
) -> tuple[
    list[dict[str, float]],
    list[dict[str, Any]],
    bool,
    float,
]:

    raw_cut_ranges = normalize_cut_ranges(clip_timeline)

    protected_ranges = normalize_protected_ranges(clip_timeline)

    guarded_cut_ranges, blocked_protected_cuts = sanitize_cut_ranges_against_protected(
        cut_ranges=raw_cut_ranges,
        protected_ranges=protected_ranges,
    )

    (
        final_cut_ranges,
        story_duration_guard_triggered,
        minimum_story_duration,
    ) = enforce_story_duration_guard(
        clip_timeline=clip_timeline,
        clip_duration=clip_duration,
        cut_ranges=guarded_cut_ranges,
    )

    if final_cut_ranges != raw_cut_ranges:

        _persist_corrected_cut_ranges(
            clip_timeline=clip_timeline,
            clip_duration=clip_duration,
            final_cut_ranges=final_cut_ranges,
            timeline_data=timeline_data,
            timeline_path=timeline_path,
        )

    return (
        final_cut_ranges,
        blocked_protected_cuts,
        story_duration_guard_triggered,
        minimum_story_duration,
    )


def _persist_corrected_cut_ranges(
    clip_timeline: dict[str, Any],
    clip_duration: float,
    final_cut_ranges: list[dict[str, float]],
    timeline_data: dict[str, Any] | None,
    timeline_path: str | Path | None,
) -> None:

    rounded = [
        {
            "start": round_time(float(item["start"])),
            "end": round_time(float(item["end"])),
        }
        for item in final_cut_ranges
    ]

    clip_timeline["cut_ranges"] = rounded

    removed_duration = round_time(
        sum(item["end"] - item["start"] for item in rounded)
    )

    estimated_duration = round_time(
        max(0.0, clip_duration - removed_duration)
    )

    edited = clip_timeline.setdefault("edited", {})
    edited["removed_duration"] = removed_duration
    edited["estimated_duration"] = estimated_duration

    if clip_duration > 0:
        edited["retained_ratio"] = round(
            estimated_duration / clip_duration, 4
        )

    clip_index = clip_timeline.get("clip_index", "?")

    print(
        f"⚠️  Clip {clip_index}: render-time guard cut_ranges'i değiştirdi "
        f"(korumalı aralık çakışması veya süre tabanı). Altyazı kaymasını "
        f"önlemek için timeline dosyası güncelleniyor."
    )

    if timeline_data is not None and timeline_path is not None:

        timeline_path = Path(timeline_path)

        with timeline_path.open("w", encoding="utf-8") as file:
            json.dump(
                timeline_data,
                file,
                ensure_ascii=False,
                indent=2,
            )


# ============================================================
# STORY-DURATION SAFETY NET
# ============================================================

def enforce_story_duration_guard(
    clip_timeline: dict[str, Any],
    clip_duration: float,
    cut_ranges: list[dict[str, float]],
) -> tuple[
    list[dict[str, float]],
    bool,
    float,
]:
    """
    Final stale-timeline safety net.

    Older behavior cancelled ALL cuts when the duration floor was crossed,
    which could resurrect every dead-air gap. V8 keeps as much high-value
    silence removal as possible while still respecting the emergency floor.
    """

    edited = clip_timeline.get(
        "edited",
        {},
    )

    if not isinstance(
        edited,
        dict,
    ):
        edited = {}

    try:
        minimum_duration = float(
            edited.get(
                "minimum_story_duration",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        minimum_duration = 0.0

    if (
        minimum_duration <= 0
        or not cut_ranges
    ):
        return (
            cut_ranges,
            False,
            max(
                0.0,
                minimum_duration,
            ),
        )

    allowed_removal = max(
        0.0,
        clip_duration
        - minimum_duration,
    )

    total_removal = sum(
        max(
            0.0,
            float(
                item["end"]
            )
            - float(
                item["start"]
            ),
        )
        for item in cut_ranges
    )

    if (
        total_removal
        <= allowed_removal
        + STORY_DURATION_GUARD_TOLERANCE
    ):
        return (
            cut_ranges,
            False,
            minimum_duration,
        )

    # Keep the biggest obvious dead-air compressions first. They give the most
    # pacing improvement per splice and avoid the old all-or-nothing fallback.
    ordered = sorted(
        cut_ranges,
        key=lambda item: (
            -(
                float(
                    item["end"]
                )
                - float(
                    item["start"]
                )
            ),
            float(
                item["start"]
            ),
        ),
    )

    accepted: list[
        dict[str, float]
    ] = []

    spent = 0.0

    for item in ordered:
        duration = max(
            0.0,
            float(
                item["end"]
            )
            - float(
                item["start"]
            ),
        )

        if duration <= 0:
            continue

        if (
            spent
            + duration
            <= allowed_removal
            + STORY_DURATION_GUARD_TOLERANCE
        ):
            accepted.append(
                item
            )
            spent += duration

    accepted.sort(
        key=lambda item: float(
            item["start"]
        )
    )

    return (
        accepted,
        True,
        minimum_duration,
    )


# ============================================================
# KEEP RANGES
# ============================================================

def build_keep_ranges(
    clip_duration: float,
    cut_ranges: list[dict[str, float]],
) -> list[dict[str, float]]:

    clip_duration = float(
        clip_duration
    )

    if not cut_ranges:

        return [
            {
                "start": 0.0,
                "end": round_time(
                    clip_duration
                ),
            }
        ]

    keep_ranges: list[
        dict[str, float]
    ] = []

    cursor = 0.0

    for cut in cut_ranges:

        cut_start = float(
            cut["start"]
        )

        cut_end = float(
            cut["end"]
        )

        if (
            cut_start
            - cursor
            >= MIN_SEGMENT_DURATION
        ):

            keep_ranges.append(
                {
                    "start": round_time(
                        cursor
                    ),

                    "end": round_time(
                        cut_start
                    ),
                }
            )

        cursor = max(
            cursor,
            cut_end,
        )

    if (
        clip_duration
        - cursor
        >= MIN_SEGMENT_DURATION
    ):

        keep_ranges.append(
            {
                "start": round_time(
                    cursor
                ),

                "end": round_time(
                    clip_duration
                ),
            }
        )

    if not keep_ranges:

        raise RuntimeError(
            "Cut ranges tüm klibi siliyor."
        )

    return keep_ranges


# ============================================================
# ABSOLUTE SOURCE RANGES
# ============================================================

def to_absolute_ranges(
    clip_start: float,
    keep_ranges: list[dict[str, float]],
) -> list[dict[str, float]]:

    result = []

    for item in keep_ranges:

        result.append(
            {
                "start": round_time(
                    clip_start
                    + float(
                        item["start"]
                    )
                ),

                "end": round_time(
                    clip_start
                    + float(
                        item["end"]
                    )
                ),
            }
        )

    return result


# ============================================================
# FILTER COMPLEX
# ============================================================

def build_filter_complex(
    absolute_ranges: list[dict[str, float]],
    has_audio: bool,
) -> tuple[
    str,
    str,
    str | None,
]:

    if not absolute_ranges:

        raise RuntimeError(
            "Render edilecek segment bulunamadı."
        )

    filters: list[str] = []

    # --------------------------------------------------------
    # TEK SEGMENT
    # --------------------------------------------------------

    if len(
        absolute_ranges
    ) == 1:

        segment = absolute_ranges[0]

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        filters.append(
            "[0:v:0]"
            f"trim=start={start:.6f}:end={end:.6f},"
            "setpts=PTS-STARTPTS"
            "[vout]"
        )

        if has_audio:

            filters.append(
                "[0:a:0]"
                f"atrim=start={start:.6f}:end={end:.6f},"
                "asetpts=PTS-STARTPTS"
                "[aout]"
            )

        return (
            ";".join(
                filters
            ),
            "[vout]",
            (
                "[aout]"
                if has_audio
                else None
            ),
        )

    # --------------------------------------------------------
    # MULTIPLE SEGMENTS
    # --------------------------------------------------------

    concat_inputs = []

    for index, segment in enumerate(
        absolute_ranges
    ):

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        video_label = (
            f"v{index}"
        )

        filters.append(
            "[0:v:0]"
            f"trim=start={start:.6f}:end={end:.6f},"
            "setpts=PTS-STARTPTS"
            f"[{video_label}]"
        )

        concat_inputs.append(
            f"[{video_label}]"
        )

        if has_audio:

            audio_label = (
                f"a{index}"
            )

            filters.append(
                "[0:a:0]"
                f"atrim=start={start:.6f}:end={end:.6f},"
                "asetpts=PTS-STARTPTS"
                f"[{audio_label}]"
            )

            concat_inputs.append(
                f"[{audio_label}]"
            )

    if has_audio:

        filters.append(
            "".join(
                concat_inputs
            )
            + f"concat=n={len(absolute_ranges)}:v=1:a=1"
            + "[vout][aout]"
        )

        audio_output = (
            "[aout]"
        )

    else:

        filters.append(
            "".join(
                concat_inputs
            )
            + f"concat=n={len(absolute_ranges)}:v=1:a=0"
            + "[vout]"
        )

        audio_output = None

    return (
        ";".join(
            filters
        ),
        "[vout]",
        audio_output,
    )


# ============================================================
# OUTPUT PATH
# ============================================================

def get_output_path(
    timeline_data: dict[str, Any],
    clip_timeline: dict[str, Any],
) -> Path:

    source = timeline_data.get(
        "source",
        {},
    )

    video_stem = str(
        source.get(
            "video_stem",
            "vod",
        )
    ).strip()

    if not video_stem:

        video_stem = "vod"

    clip_index = int(
        clip_timeline[
            "clip_index"
        ]
    )

    title = safe_filename(
        clip_timeline.get(
            "title",
            f"clip_{clip_index}",
        )
    )

    # Dosya adı fazla uzamasın.
    title = title[:70]

    output_dir = (
        OUTPUT_ROOT
        / safe_filename(
            video_stem
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_dir
        / (
            f"clip_{clip_index:02d}_"
            f"{title}_edited.mp4"
        )
    )


# ============================================================
# EXPECTED DURATION
# ============================================================

def calculate_expected_duration(
    keep_ranges: list[dict[str, float]],
) -> float:

    total = sum(
        float(
            item["end"]
        )
        - float(
            item["start"]
        )

        for item in keep_ranges
    )

    return round_time(
        total
    )


# ============================================================
# RENDER SINGLE CLIP
# ============================================================

def render_clip(
    timeline_data: dict[str, Any],
    clip_timeline: dict[str, Any],
    timeline_path: str | Path | None = None,
) -> Path:

    source_data = timeline_data.get(
        "source",
        {},
    )

    source_video = Path(
        str(
            source_data.get(
                "video_path",
                "",
            )
        )
    ).resolve()

    if not source_video.exists():

        raise FileNotFoundError(
            f"Orijinal VOD bulunamadı:\n{source_video}"
        )

    clip_index = int(
        clip_timeline[
            "clip_index"
        ]
    )

    source = clip_timeline.get(
        "source",
        {},
    )

    clip_start = float(
        source.get(
            "absolute_start",
            0,
        )
    )

    clip_end = float(
        source.get(
            "absolute_end",
            0,
        )
    )

    clip_duration = float(
        source.get(
            "duration",
            clip_end - clip_start,
        )
    )

    if (
        clip_end <= clip_start
        or clip_duration <= 0
    ):

        raise RuntimeError(
            f"Clip {clip_index} source aralığı geçersiz."
        )

    # --------------------------------------------------------
    # CUTS
    # --------------------------------------------------------

    (
        cut_ranges,
        blocked_protected_cuts,
        story_duration_guard_triggered,
        minimum_story_duration,
    ) = resolve_final_cut_ranges(
        clip_timeline=clip_timeline,
        clip_duration=clip_duration,
        timeline_data=timeline_data,
        timeline_path=timeline_path,
    )

    keep_ranges = build_keep_ranges(
        clip_duration=clip_duration,
        cut_ranges=cut_ranges,
    )

    protected_ranges = normalize_protected_ranges(
        clip_timeline
    )

    absolute_ranges = to_absolute_ranges(
        clip_start=clip_start,
        keep_ranges=keep_ranges,
    )

    expected_duration = (
        calculate_expected_duration(
            keep_ranges
        )
    )

    # --------------------------------------------------------
    # AUDIO
    # --------------------------------------------------------

    has_audio = video_has_audio(
        source_video
    )

    # --------------------------------------------------------
    # FILTER
    # --------------------------------------------------------

    (
        filter_complex,
        video_output,
        audio_output,
    ) = build_filter_complex(
        absolute_ranges=absolute_ranges,
        has_audio=has_audio,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_path = get_output_path(
        timeline_data=timeline_data,
        clip_timeline=clip_timeline,
    )

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------

    print()
    print(
        "=" * 68
    )

    print(
        f"🎬 CLIP {clip_index}"
    )

    print(
        f"📛 {clip_timeline.get('title', '')}"
    )

    print(
        f"📍 VOD: "
        f"{clip_start:.2f}"
        f" → "
        f"{clip_end:.2f}"
    )

    print(
        f"⏱️ Ham süre: "
        f"{clip_duration:.2f}s"
    )

    print(
        f"🛡️ Terra protected range: "
        f"{len(protected_ranges)}"
    )

    if story_duration_guard_triggered:

        print(
            "🚨 Story-duration safety net: stale/bozuk timeline cut listesi "
            "minimum süreyi aşıyordu; yalnızca en değerli dead-air cut'ları tutuldu."
        )

        print(
            f"   Minimum story duration: {minimum_story_duration:.2f}s"
        )

    if blocked_protected_cuts:

        print(
            "🚫 Renderer güvenlik ağı: "
            f"{len(blocked_protected_cuts)} cut engellendi."
        )

        for blocked in blocked_protected_cuts:

            print(
                "   🛡️ "
                f"{blocked['start']:.2f}"
                " → "
                f"{blocked['end']:.2f}"
                " | "
                f"{blocked['reason']}"
            )

    print(
        f"✂️ Uygulanacak cut sayısı: "
        f"{len(cut_ranges)}"
    )

    for index, cut in enumerate(
        cut_ranges,
        start=1,
    ):

        print(
            f"   {index}. "
            f"{cut['start']:.2f}"
            f" → "
            f"{cut['end']:.2f}"
            f" "
            f"(-{cut['end'] - cut['start']:.2f}s)"
        )

    print(
        f"🧩 Kalacak segment: "
        f"{len(keep_ranges)}"
    )

    print(
        f"⚡ Beklenen edit süresi: "
        f"{expected_duration:.2f}s"
    )

    # --------------------------------------------------------
    # FFMPEG COMMAND
    # --------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",

        "-i",
        str(
            source_video
        ),

        "-filter_complex",
        filter_complex,

        "-map",
        video_output,

        "-c:v",
        VIDEO_CODEC,

        "-preset",
        VIDEO_PRESET,

        "-crf",
        VIDEO_CRF,

        "-pix_fmt",
        "yuv420p",
    ]

    if audio_output is not None:

        command.extend(
            [
                "-map",
                audio_output,

                "-c:a",
                AUDIO_CODEC,

                "-b:a",
                AUDIO_BITRATE,
            ]
        )

    command.extend(
        [
            "-movflags",
            "+faststart",

            str(
                output_path
            ),
        ]
    )

    print()
    print(
        "⚙️ Frame-accurate pacing edit render ediliyor..."
    )

    try:

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    except FileNotFoundError as error:

        raise RuntimeError(
            "FFmpeg bulunamadı."
        ) from error

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg pacing render hatası:\n\n"
            + result.stderr
        )

    if not output_path.exists():

        raise RuntimeError(
            "FFmpeg tamamlandı fakat çıktı oluşmadı."
        )

    # --------------------------------------------------------
    # VERIFY DURATION
    # --------------------------------------------------------

    actual_duration = get_video_duration(
        output_path
    )

    difference = abs(
        actual_duration
        - expected_duration
    )

    print()
    print(
        f"✅ Gerçek çıktı süresi: "
        f"{actual_duration:.3f}s"
    )

    print(
        f"🎯 Beklenen süre: "
        f"{expected_duration:.3f}s"
    )

    if (
        difference
        > DURATION_WARNING_TOLERANCE
    ):

        print(
            f"⚠️ Süre farkı biraz yüksek: "
            f"{difference:.3f}s"
        )

    else:

        print(
            f"✅ Timestamp doğrulaması başarılı "
            f"(fark {difference:.3f}s)"
        )

    print()
    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 68
    )

    return output_path



# ============================================================
# PRE-RENDER AUDIO-ONLY SPEAKER SCAN
# ============================================================

SPEAKER_SCAN_AUDIO_ROOT = PROJECT_ROOT / "vod_output" / "temp" / "speaker_prerender"


def get_speaker_scan_audio_path(
    timeline_data: dict[str, Any],
    clip_timeline: dict[str, Any],
) -> Path:
    source_data = timeline_data.get("source", {})
    source_video = Path(str(source_data.get("video_path", ""))).resolve()
    clip_index = int(clip_timeline.get("clip_index", 0) or 0)
    folder = SPEAKER_SCAN_AUDIO_ROOT / safe_filename(source_video.stem or "video")
    folder.mkdir(parents=True, exist_ok=True)
    return (folder / f"clip_{clip_index:02d}_speaker_prerender.wav").resolve()


def _speaker_audio_filter(absolute_ranges: list[dict[str, float]]) -> tuple[str, str]:
    if not absolute_ranges:
        raise RuntimeError("Speaker scan için audio segment bulunamadı.")
    filters: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(absolute_ranges):
        start = float(item["start"])
        end = float(item["end"])
        label = f"sa{index}"
        filters.append(
            f"[0:a:0]atrim=start={start:.6f}:end={end:.6f},"
            "asetpts=PTS-STARTPTS,aresample=48000,"
            f"aformat=sample_fmts=s16:channel_layouts=mono[{label}]"
        )
        labels.append(f"[{label}]")
    if len(labels) == 1:
        filters.append(f"{labels[0]}anull[aout]")
    else:
        filters.append("".join(labels) + f"concat=n={len(labels)}:v=0:a=1[aout]")
    return ";".join(filters), "[aout]"


def render_audio_only_for_clip(
    timeline_path: str | Path,
    clip_index: int,
) -> Path:
    """Create the exact pacing-cut AUDIO before any video render.

    This uses the same protected-range and story-duration guards as the real
    pacing render, but decodes/encodes audio only. It is intentionally cheap
    and is used for speaker diarization + human A/B naming before the first
    heavy video encode begins.
    """
    timeline_data = load_json(timeline_path)
    validate_timeline(timeline_data)
    clip_timeline = get_clip_timeline(timeline_data=timeline_data, clip_index=clip_index)

    source_data = timeline_data.get("source", {})
    source_video = Path(str(source_data.get("video_path", ""))).resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"Orijinal VOD bulunamadı:\n{source_video}")
    if not video_has_audio(source_video):
        raise RuntimeError("Seçilen VOD audio stream içermiyor.")

    source = clip_timeline.get("source", {})
    clip_start = float(source.get("absolute_start", 0.0))
    clip_end = float(source.get("absolute_end", 0.0))
    clip_duration = float(source.get("duration", clip_end - clip_start))
    if clip_end <= clip_start or clip_duration <= 0:
        raise RuntimeError(f"Clip {clip_index} source aralığı geçersiz.")

    cut_ranges, _blocked, _guarded, _minimum = resolve_final_cut_ranges(
        clip_timeline=clip_timeline,
        clip_duration=clip_duration,
        timeline_data=timeline_data,
        timeline_path=timeline_path,
    )
    keep_ranges = build_keep_ranges(clip_duration=clip_duration, cut_ranges=cut_ranges)
    absolute_ranges = to_absolute_ranges(clip_start=clip_start, keep_ranges=keep_ranges)

    filter_complex, audio_output = _speaker_audio_filter(absolute_ranges)
    output = get_speaker_scan_audio_path(timeline_data, clip_timeline)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source_video),
        "-filter_complex", filter_complex,
        "-map", audio_output,
        "-vn", "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size < 1024:
        raise RuntimeError(
            "Pre-render speaker audio çıkarılamadı. "
            + (completed.stderr.strip() or "ffmpeg başarısız")
        )
    return output


# ============================================================
# RENDER ONE
# ============================================================

def render_one_clip(
    timeline_path: str | Path,
    clip_index: int,
) -> Path:

    timeline_data = load_json(
        timeline_path
    )

    validate_timeline(
        timeline_data
    )

    clip_timeline = get_clip_timeline(
        timeline_data=timeline_data,
        clip_index=clip_index,
    )

    return render_clip(
        timeline_data=timeline_data,
        clip_timeline=clip_timeline,
        timeline_path=timeline_path,
    )


# ============================================================
# RENDER ALL
# ============================================================

def render_all_clips(
    timeline_path: str | Path,
) -> list[Path]:

    timeline_data = load_json(
        timeline_path
    )

    validate_timeline(
        timeline_data
    )

    outputs: list[Path] = []

    timelines = timeline_data.get(
        "timelines",
        [],
    )

    print()
    print(
        "=" * 68
    )

    print(
        "✂️ MIMIR PACING CUTTER"
    )

    print(
        f"🎬 Toplam klip: "
        f"{len(timelines)}"
    )

    print(
        "=" * 68
    )

    for clip_timeline in timelines:

        output = render_clip(
            timeline_data=timeline_data,
            clip_timeline=clip_timeline,
            timeline_path=timeline_path,
        )

        outputs.append(
            output
        )

    return outputs


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 68
    )

    print(
        "✂️ MIMIR PACING CUTTER"
    )

    print(
        "=" * 68
    )

    timeline_path = input(
        "Timeline V3 JSON yolunu gir: "
    ).strip().strip('"')

    clip_index_text = input(
        "Clip index "
        "(boş = tüm klipler): "
    ).strip()

    try:

        if clip_index_text:

            render_one_clip(
                timeline_path=timeline_path,
                clip_index=int(
                    clip_index_text
                ),
            )

        else:

            outputs = render_all_clips(
                timeline_path=timeline_path,
            )

            print()
            print(
                "=" * 68
            )

            print(
                f"✅ TOPLAM {len(outputs)} KLİP HAZIR"
            )

            for output in outputs:

                print(
                    f"📹 {output}"
                )

            print(
                "=" * 68
            )

    except Exception as error:

        print()
        print(
            "❌ PACING CUTTER HATASI:"
        )

        print(
            error
        )