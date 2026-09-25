from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import winreg
except ImportError:
    winreg = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"

EDITED_CLIPS_DIR = VOD_OUTPUT_DIR / "edited_clips"
CAPTIONED_PREVIEWS_DIR = VOD_OUTPUT_DIR / "previews"
INTRO_ASSETS_DIR = VOD_OUTPUT_DIR / "intros" / "render_assets"
FINAL_PREVIEWS_DIR = VOD_OUTPUT_DIR / "final_previews"


# ============================================================
# RENDER CONFIG
# ============================================================

RENDERER_VERSION = 10

# Renderer also enforces the analyzer quality gate.
MIN_RENDER_SCORE = 8.0

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = "18"
PIXEL_FORMAT = "yuv420p"

# Final output is rebuilt from two sources:
# 1) clean edited clip -> teaser segment + neon hook (NO normal captions)
# 2) captioned preview -> full main clip with normal dynamic captions
AUDIO_SAMPLE_RATE = 48000
AUDIO_BITRATE = "192k"


# ============================================================
# HOOK OVERLAY TIMING
# ============================================================

# There is NO separate intro card anymore.
# The hook copy appears directly on the moving teaser scene.
#
# The analyzer's old intro duration is only used as a hint.
# We extend it because the text can now stay on screen while the teaser plays.
DISPLAY_EXTENSION_SECONDS = 0.30
MIN_DISPLAY_DURATION = 1.70
PREFERRED_MAX_DISPLAY_DURATION = 2.45

# Hook'u teaser'ın son nefesine kadar taşımıyoruz; transition öncesi temiz alan bırak.
TEASER_END_CLEARANCE = 0.12

# Teaser -> main restart geçişi. Kısa tutulur; edit hissini öldürmez.
SMOOTH_TRANSITION_SECONDS = 0.16
MIN_TRANSITION_SECONDS = 0.08
MAX_TRANSITION_SECONDS = 0.22

# Text reveal begins almost instantly.
LINE_1_DELAY = 0.015
LINE_2_DELAY = 0.095

# Intro -> main handoff. The old renderer always restarted the captioned main
# from 0:00. When the selected story has a long quiet/setup lead, that creates a
# dead gap immediately after the cold open and can make burned-in captions feel
# late even though their internal timing is correct.
#
# Use the FIRST rendered caption as a conservative speech clock. We only trim
# when the lead is clearly long, and we keep a short natural preroll before the
# first spoken word. Caption text/timing generation itself is untouched.
POST_INTRO_GAP_TRIGGER_SECONDS = 1.60
POST_INTRO_SPEECH_PREROLL_SECONDS = 0.65
POST_INTRO_MAX_TRIM_SECONDS = 8.00
POST_INTRO_MIN_MAIN_SECONDS = 0.80


# ============================================================
# VIRAL TEXT LOOK
# ============================================================

# The reference look is closer to Anton / Impact / heavy condensed italic.
# Do NOT use Arial Black.
FONT_CANDIDATES = (
    ("Anton", "Anton"),
    ("Montserrat ExtraBold", "Montserrat ExtraBold"),
    ("Impact", "Impact"),
    ("Bahnschrift SemiCondensed", "Bahnschrift SemiCondensed"),
    ("Bahnschrift", "Bahnschrift SemiCondensed"),
)

FALLBACK_FONT = "Impact"

# Neon lime inspired by the supplied reference.
# ASS colors are &HAABBGGRR.
NEON_LIME = "&H0000FFD7"       # #D7FF00
BLACK = "&H00000000"
TRANSPARENT = "&HFF000000"

MAIN_OUTLINE = 5
MAIN_SHADOW = 0

# Slight artificial shear to make even fallback fonts feel more aggressive.
TEXT_SHEAR = -0.085
TEXT_SPACING = -1


# ============================================================
# TEXT POSITION / SIZE
# ============================================================

# Upper-middle placement. The teaser itself has NO normal dynamic captions,
# so the neon hook is the only text visible during the cold open.
LINE_1_Y_RATIO = 0.355
LINE_2_Y_RATIO = 0.465
SINGLE_LINE_Y_RATIO = 0.405

# Maximum horizontal area occupied by the hook text.
MAX_TEXT_WIDTH_RATIO = 0.80

# Large on purpose.
BASE_FONT_SIZE_1080 = 148
MIN_FONT_SIZE_1080 = 84


# ============================================================
# ANIMATION
# ============================================================

# A short-form "hit" rather than a smooth corporate UI animation:
# oversized -> slam smaller -> rebound -> settle.
START_SCALE = 142
LAND_SCALE = 94
BOUNCE_SCALE = 106
FINAL_SCALE = 100

LAND_MS = 85
BOUNCE_MS = 145
SETTLE_MS = 225

# Small opposing lateral motion for the two lines.
LINE_SLIDE_RATIO = 0.030

# A very subtle second pulse while the hook is held.
PULSE_START_MS = 520
PULSE_PEAK_MS = 610
PULSE_END_MS = 700
PULSE_SCALE = 103

# Fade out just before the teaser ends.
FADE_OUT_MS = 90


# ============================================================
# BASIC JSON HELPERS
# ============================================================


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"JSON bulunamadı:\n{path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root object değil:\n{path}")

    return data



def safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", str(value).strip())
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ._")


# ============================================================
# PACKAGE VALIDATION
# ============================================================


def validate_intro_package(package: dict[str, Any]) -> None:
    if package.get("version") != 1:
        raise RuntimeError(
            "Intro Renderer V8, Intro Analyzer V1-compatible package bekliyor."
        )

    intros = package.get("intros")

    if not isinstance(intros, list) or not intros:
        raise RuntimeError(
            "Intro JSON içinde 'intros' listesi bulunamadı."
        )

    inputs = package.get("inputs")

    if not isinstance(inputs, dict):
        raise RuntimeError("Intro JSON içinde 'inputs' bulunamadı.")

    for key in ("timeline", "teaser"):
        value = inputs.get(key)

        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"Intro JSON inputs.{key} eksik.")



def validate_timeline(timeline: dict[str, Any]) -> None:
    if timeline.get("version") != 3:
        raise RuntimeError("Timeline V3 bekleniyordu.")

    if not isinstance(timeline.get("source"), dict):
        raise RuntimeError("Timeline source bilgisi bulunamadı.")



def validate_teaser_package(package: dict[str, Any]) -> None:
    if package.get("version") != 1:
        raise RuntimeError("Teaser Analyzer V1 çıktısı bekleniyordu.")

    teasers = package.get("teasers")

    if not isinstance(teasers, list) or not teasers:
        raise RuntimeError(
            "Teaser JSON içinde 'teasers' listesi bulunamadı."
        )


# ============================================================
# GET PACKAGE ITEMS
# ============================================================


def get_intro(
    package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:
    for position, intro in enumerate(package["intros"], start=1):
        if not isinstance(intro, dict):
            continue

        try:
            current_index = int(intro.get("clip_index", position))
        except (TypeError, ValueError):
            current_index = position

        if current_index == clip_index:
            return intro

    raise IndexError(
        f"Intro JSON içinde clip_index={clip_index} bulunamadı."
    )



def get_teaser(
    package: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:
    for position, teaser in enumerate(package["teasers"], start=1):
        if not isinstance(teaser, dict):
            continue

        try:
            current_index = int(teaser.get("clip_index", position))
        except (TypeError, ValueError):
            current_index = position

        if current_index == clip_index:
            return teaser

    raise IndexError(
        f"Teaser JSON içinde clip_index={clip_index} bulunamadı."
    )


# ============================================================
# REFERENCED PATHS
# ============================================================


def get_referenced_paths(
    intro_package: dict[str, Any],
) -> tuple[Path, Path]:
    inputs = intro_package["inputs"]

    timeline_path = Path(inputs["timeline"]).resolve()
    teaser_path = Path(inputs["teaser"]).resolve()

    if not timeline_path.exists():
        raise FileNotFoundError(
            f"Intro JSON'un referans verdiği Timeline bulunamadı:\n"
            f"{timeline_path}"
        )

    if not teaser_path.exists():
        raise FileNotFoundError(
            f"Intro JSON'un referans verdiği Teaser JSON bulunamadı:\n"
            f"{teaser_path}"
        )

    return timeline_path, teaser_path


# ============================================================
# VIDEO STEM / PREVIEW DISCOVERY
# ============================================================


def get_video_stem(timeline: dict[str, Any]) -> str:
    source = timeline.get("source", {})

    for key in ("video_stem", "video_name", "video_path"):
        value = source.get(key)

        if isinstance(value, str) and value.strip():
            if key == "video_stem":
                return value.strip()

            return Path(value).stem

    raise RuntimeError(
        "Timeline source içinde video_stem/video_name/video_path bulunamadı."
    )



def find_edited_clip(
    timeline: dict[str, Any],
    clip_index: int,
) -> Path:
    directory = EDITED_CLIPS_DIR / get_video_stem(timeline)

    if not directory.exists():
        raise FileNotFoundError(
            f"Edited clips klasörü bulunamadı:\n{directory}\n\n"
            "Önce pacing_cutter çalıştır."
        )

    matches = sorted(
        directory.glob(
            f"clip_{clip_index:02d}_*_edited.mp4"
        )
    )

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            "Birden fazla edited clip bulundu:\n"
            + "\n".join(f"- {path.name}" for path in matches)
        )

    raise FileNotFoundError(
        f"Edited clip bulunamadı:\n{directory}\n\n"
        "Önce pacing_cutter çalıştır."
    )



def find_captioned_preview(
    timeline: dict[str, Any],
    clip_index: int,
) -> Path:
    directory = CAPTIONED_PREVIEWS_DIR / get_video_stem(timeline)

    if not directory.exists():
        raise FileNotFoundError(
            f"Captioned preview klasörü bulunamadı:\n{directory}\n\n"
            "Önce caption_renderer çalıştır."
        )

    exact = (
        directory
        / f"clip_{clip_index:02d}_captioned_preview.mp4"
    )

    if exact.exists():
        return exact.resolve()

    matches = [
        path
        for path in sorted(
            directory.glob(
                f"clip_{clip_index:02d}_*captioned*.mp4"
            )
        )
        if "teaser" not in path.name.lower()
        and "final" not in path.name.lower()
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        raise RuntimeError(
            "Birden fazla captioned preview bulundu:\n"
            + "\n".join(f"- {path.name}" for path in matches)
        )

    raise FileNotFoundError(
        f"Captioned preview bulunamadı:\n{directory}\n\n"
        "Önce caption_renderer çalıştır."
    )


# ============================================================
# FFPROBE
# ============================================================


def run_ffprobe_json(video_path: str | Path) -> dict[str, Any]:
    path = Path(video_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Video bulunamadı:\n{path}")

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
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe bulunamadı. FFmpeg kurulumunu kontrol et."
        ) from error

    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe video bilgisini okuyamadı:\n" + result.stderr
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("ffprobe geçersiz JSON döndürdü.") from error

    if not isinstance(data, dict):
        raise RuntimeError("ffprobe çıktısı geçersiz.")

    return data



def parse_fraction(value: str) -> float:
    value = str(value).strip()

    if not value:
        return 0.0

    if "/" in value:
        left, right = value.split("/", 1)

        try:
            numerator = float(left)
            denominator = float(right)
        except ValueError:
            return 0.0

        if denominator == 0:
            return 0.0

        return numerator / denominator

    try:
        return float(value)
    except ValueError:
        return 0.0



def get_video_info(video_path: str | Path) -> dict[str, Any]:
    data = run_ffprobe_json(video_path)

    width: int | None = None
    height: int | None = None
    fps = 0.0
    has_audio = False

    for stream in data.get("streams", []):
        if not isinstance(stream, dict):
            continue

        if stream.get("codec_type") == "video" and width is None:
            try:
                width = int(stream["width"])
                height = int(stream["height"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "Video çözünürlüğü okunamadı."
                ) from error

            fps = parse_fraction(
                stream.get("avg_frame_rate")
                or stream.get("r_frame_rate")
                or ""
            )

        elif stream.get("codec_type") == "audio":
            has_audio = True

    if width is None or height is None:
        raise RuntimeError("Video stream bulunamadı.")

    if fps <= 0:
        fps = 30.0

    try:
        duration = float(data.get("format", {}).get("duration", 0))
    except (TypeError, ValueError) as error:
        raise RuntimeError("Video süresi okunamadı.") from error

    if duration <= 0:
        raise RuntimeError("Video süresi geçersiz.")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "has_audio": has_audio,
        "duration": duration,
    }


# ============================================================
# FONT DETECTION
# ============================================================


def detect_font() -> str:
    """
    Prefer a real installed condensed/bold creator font.

    On a normal Windows install Impact is available, so we always have
    a strong non-Arial fallback.
    """

    if winreg is None:
        return FALLBACK_FONT

    installed_names: list[str] = []

    registry_locations = (
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        ),
        (
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        ),
    )

    for root, registry_path in registry_locations:
        try:
            with winreg.OpenKey(root, registry_path) as key:
                index = 0

                while True:
                    try:
                        name, _, _ = winreg.EnumValue(key, index)
                    except OSError:
                        break

                    installed_names.append(str(name).casefold())
                    index += 1

        except OSError:
            continue

    for registry_token, ass_family in FONT_CANDIDATES:
        needle = registry_token.casefold()

        if any(needle in item for item in installed_names):
            return ass_family

    return FALLBACK_FONT


# ============================================================
# TIMING
# ============================================================


def get_teaser_duration(teaser: dict[str, Any]) -> float:
    edited = teaser.get("edited")

    if not isinstance(edited, dict):
        raise RuntimeError("Teaser JSON içinde edited bilgisi bulunamadı.")

    # Preferred field from Teaser Analyzer V1.
    try:
        duration = float(edited.get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0

    if duration > 0:
        return duration

    # Fallback.
    try:
        start = float(edited["teaser_start"])
        end = float(edited["teaser_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Teaser duration hesaplanamadı."
        ) from error

    duration = end - start

    if duration <= 0:
        raise RuntimeError("Teaser duration geçersiz.")

    return duration



def get_teaser_bounds(
    teaser: dict[str, Any],
    edited_duration: float,
) -> tuple[float, float]:
    edited = teaser.get("edited")

    if not isinstance(edited, dict):
        raise RuntimeError(
            "Teaser JSON içinde edited bilgisi bulunamadı."
        )

    try:
        start = float(edited["teaser_start"])
        end = float(edited["teaser_end"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            "Teaser edited.teaser_start/end geçersiz."
        ) from error

    start = max(0.0, start)
    end = min(float(edited_duration), end)

    if end <= start:
        raise RuntimeError(
            "Teaser kesim aralığı edited clip içinde geçersiz."
        )

    return round(start, 6), round(end, 6)



def _ass_timestamp_to_seconds(value: str) -> float | None:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
    except (TypeError, ValueError):
        return None
    return max(0.0, hours * 3600.0 + minutes * 60.0 + seconds)


def get_first_caption_start(caption_path: str | Path | None) -> float | None:
    """Return the earliest ASS Dialogue start, or None on any uncertainty.

    Fail-open is deliberate: if the caption file cannot be parsed, the renderer
    keeps the historical 0:00 main restart rather than guessing.
    """
    if caption_path is None:
        return None

    path = Path(caption_path).expanduser().resolve()
    if not path.is_file():
        return None

    earliest: float | None = None
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line.startswith("Dialogue:"):
                    continue
                payload = line.split(":", 1)[1].lstrip()
                parts = payload.split(",", 3)
                if len(parts) < 3:
                    continue
                start = _ass_timestamp_to_seconds(parts[1])
                if start is None:
                    continue
                earliest = start if earliest is None else min(earliest, start)
    except OSError:
        return None

    return earliest


def calculate_main_restart_seconds(
    *,
    caption_path: str | Path | None,
    main_duration: float,
) -> tuple[float, float | None]:
    """Choose a conservative post-intro main restart point.

    The first caption is already built from the exact rendered clip clock. If it
    starts late, remove only the dead lead before it while preserving a short
    preroll. Video, audio and burned-in captions are then trimmed together.
    """
    first_caption = get_first_caption_start(caption_path)
    if first_caption is None or first_caption <= POST_INTRO_GAP_TRIGGER_SECONDS:
        return 0.0, first_caption

    desired = max(0.0, first_caption - POST_INTRO_SPEECH_PREROLL_SECONDS)
    desired = min(desired, POST_INTRO_MAX_TRIM_SECONDS)
    safe_max = max(0.0, float(main_duration) - POST_INTRO_MIN_MAIN_SECONDS)
    restart = min(desired, safe_max)

    if restart < 0.05:
        restart = 0.0

    return round(restart, 6), first_caption


def get_analyzer_intro_duration(intro: dict[str, Any]) -> float:
    intro_info = intro.get("intro")

    if not isinstance(intro_info, dict):
        return 0.0

    try:
        return max(0.0, float(intro_info.get("duration", 0)))
    except (TypeError, ValueError):
        return 0.0



def calculate_display_duration(
    intro: dict[str, Any],
    teaser_duration: float,
) -> float:
    """
    The text is on top of the teaser, so keeping it longer does NOT add
    another intro section or increase the video duration.
    """

    available = max(
        0.30,
        teaser_duration - TEASER_END_CLEARANCE,
    )

    analyzer_hint = get_analyzer_intro_duration(intro)

    desired = max(
        MIN_DISPLAY_DURATION,
        analyzer_hint + DISPLAY_EXTENSION_SECONDS,
    )

    desired = min(
        desired,
        PREFERRED_MAX_DISPLAY_DURATION,
    )

    return round(
        min(desired, available),
        3,
    )


# ============================================================
# TEXT LAYOUT
# ============================================================


def clean_hook_text(text: str) -> str:
    return " ".join(
        str(text)
        .strip()
        .upper()
        .replace("’", "'")
        .replace("‘", "'")
        .split()
    )



def split_balanced_two_lines(text: str) -> tuple[str, str]:
    words = clean_hook_text(text).split()

    if not words:
        raise RuntimeError("intro_text boş.")

    if len(words) <= 2:
        return " ".join(words), ""

    best_split = 1
    best_score = float("inf")

    for split_index in range(1, len(words)):
        first = " ".join(words[:split_index])
        second = " ".join(words[split_index:])

        # Balance visual width; overly long lines are penalized.
        score = abs(len(first) - len(second))

        if len(first) > 20:
            score += (len(first) - 20) * 2.5

        if len(second) > 20:
            score += (len(second) - 20) * 2.5

        if score < best_score:
            best_score = score
            best_split = split_index

    return (
        " ".join(words[:best_split]),
        " ".join(words[best_split:]),
    )



def estimate_font_size(
    line_1: str,
    line_2: str,
    width: int,
    height: int,
) -> int:
    """
    Keep the text intentionally huge, but ensure the longest line fits.

    Heavy condensed fonts average roughly 0.50-0.58 em per character.
    This conservative estimate works well for Impact/Anton-style faces.
    """

    scale = height / 1080.0

    base = BASE_FONT_SIZE_1080 * scale
    minimum = MIN_FONT_SIZE_1080 * scale

    longest = max(
        len(line_1),
        len(line_2),
        1,
    )

    max_width = width * MAX_TEXT_WIDTH_RATIO

    fit_size = max_width / (longest * 0.54)

    final_size = min(base, fit_size)
    final_size = max(minimum, final_size)

    return int(round(final_size))


# ============================================================
# ASS HELPERS
# ============================================================


def escape_ass_text(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )



def seconds_to_ass_time(seconds: float) -> str:
    centiseconds = int(round(max(0.0, float(seconds)) * 100))

    hours = centiseconds // 360000
    centiseconds %= 360000

    minutes = centiseconds // 6000
    centiseconds %= 6000

    secs = centiseconds // 100
    cs = centiseconds % 100

    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"



def build_motion_tags(
    center_x: int,
    y: int,
    slide_pixels: int,
    direction: int,
    duration: float,
) -> str:
    start_x = center_x + (slide_pixels * direction)

    pulse_start = min(PULSE_START_MS, int(duration * 1000 * 0.55))
    pulse_peak = min(PULSE_PEAK_MS, int(duration * 1000 * 0.70))
    pulse_end = min(PULSE_END_MS, int(duration * 1000 * 0.82))

    parts = [
        "\\an5",
        f"\\move({start_x},{y},{center_x},{y},0,{LAND_MS + 45})",
        f"\\fax{TEXT_SHEAR}",
        f"\\fsp{TEXT_SPACING}",
        "\\alpha&HFF&",
        "\\frz-2.0" if direction < 0 else "\\frz2.0",
        f"\\fscx{START_SCALE}",
        f"\\fscy{START_SCALE}",
        (
            f"\\t(0,{LAND_MS},"
            f"\\alpha&H00&"
            f"\\frz0"
            f"\\fscx{LAND_SCALE}"
            f"\\fscy{LAND_SCALE})"
        ),
        (
            f"\\t({LAND_MS},{BOUNCE_MS},"
            f"\\fscx{BOUNCE_SCALE}"
            f"\\fscy{BOUNCE_SCALE})"
        ),
        (
            f"\\t({BOUNCE_MS},{SETTLE_MS},"
            f"\\fscx{FINAL_SCALE}"
            f"\\fscy{FINAL_SCALE})"
        ),
    ]

    # Only add the mid-hold pulse if there is enough time.
    if pulse_end > pulse_start + 80:
        parts.extend(
            [
                (
                    f"\\t({pulse_start},{pulse_peak},"
                    f"\\fscx{PULSE_SCALE}"
                    f"\\fscy{PULSE_SCALE})"
                ),
                (
                    f"\\t({pulse_peak},{pulse_end},"
                    f"\\fscx100"
                    f"\\fscy100)"
                ),
            ]
        )

    parts.append(f"\\fad(18,{FADE_OUT_MS})")

    return "{" + "".join(parts) + "}"




# ============================================================
# ASS BUILD
# ============================================================


def build_intro_ass(
    text: str,
    display_duration: float,
    width: int,
    height: int,
    font_name: str,
) -> str:
    line_1, line_2 = split_balanced_two_lines(text)

    font_size = estimate_font_size(
        line_1=line_1,
        line_2=line_2,
        width=width,
        height=height,
    )

    center_x = width // 2
    slide_pixels = max(18, int(round(width * LINE_SLIDE_RATIO)))

    if line_2:
        line_1_y = int(round(height * LINE_1_Y_RATIO))
        line_2_y = int(round(height * LINE_2_Y_RATIO))
    else:
        line_1_y = int(round(height * SINGLE_LINE_Y_RATIO))
        line_2_y = line_1_y

    end_ass = seconds_to_ass_time(display_duration)

    line_1_start = min(
        LINE_1_DELAY,
        max(0.0, display_duration - 0.05),
    )

    line_2_start = min(
        LINE_2_DELAY,
        max(0.0, display_duration - 0.05),
    )

    line_1_start_ass = seconds_to_ass_time(line_1_start)
    line_2_start_ass = seconds_to_ass_time(line_2_start)

    line_1_text = escape_ass_text(line_1)
    line_2_text = escape_ass_text(line_2)

    events: list[str] = []

    def add_line(
        line_text: str,
        start_ass: str,
        y: int,
        direction: int,
    ) -> None:
        # One crisp foreground layer only; no soft-effect layer.
        events.append(
            (
                f"Dialogue: 20,{start_ass},{end_ass},Main,,0,0,0,,"
                f"{build_motion_tags(center_x, y, slide_pixels, direction, display_duration)}"
                f"{line_text}"
            )
        )

    add_line(
        line_text=line_1_text,
        start_ass=line_1_start_ass,
        y=line_1_y,
        direction=-1,
    )

    if line_2:
        add_line(
            line_text=line_2_text,
            start_ass=line_2_start_ass,
            y=line_2_y,
            direction=1,
        )

    event_text = "\n".join(events)

    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{font_name},{font_size},{NEON_LIME},{NEON_LIME},{BLACK},{TRANSPARENT},-1,-1,0,0,100,100,{TEXT_SPACING},0,1,{MAIN_OUTLINE},{MAIN_SHADOW},5,35,35,35,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
{event_text}
"""


# ============================================================
# ASSET / OUTPUT PATHS
# ============================================================


def get_intro_base(intro_json_path: str | Path) -> str:
    stem = Path(intro_json_path).stem

    suffix = "_intros"

    if stem.endswith(suffix):
        stem = stem[:-len(suffix)]

    return stem or "intro"



def save_intro_ass(
    intro_json_path: str | Path,
    clip_index: int,
    text: str,
    display_duration: float,
    width: int,
    height: int,
    font_name: str,
) -> Path:
    directory = INTRO_ASSETS_DIR / get_intro_base(intro_json_path)
    directory.mkdir(parents=True, exist_ok=True)

    output_path = directory / f"clip_{clip_index:02d}_intro_v5.ass"

    content = build_intro_ass(
        text=text,
        display_duration=display_duration,
        width=width,
        height=height,
        font_name=font_name,
    )

    output_path.write_text(content, encoding="utf-8-sig")

    return output_path.resolve()



def escape_filter_path(path: str | Path) -> str:
    value = str(Path(path).resolve()).replace("\\", "/")

    # Windows drive letter for FFmpeg filter syntax (the only ':' in a Windows path):
    # C:/... -> C\:/...
    value = value.replace(":", r"\:")

    # A quoted filter value cannot contain \' (it closes the quote and the
    # apostrophe is lost): close the quote, add an escaped quote, reopen.
    return value.replace("'", "'\\\\\\''")



def get_output_path(
    timeline: dict[str, Any],
    intro: dict[str, Any],
    clip_index: int,
) -> Path:
    directory = FINAL_PREVIEWS_DIR / get_video_stem(timeline)
    directory.mkdir(parents=True, exist_ok=True)

    title = safe_filename(str(intro.get("title", "")))

    if title:
        filename = (
            f"clip_{clip_index:02d}_{title}_final_preview_v"
            f"{RENDERER_VERSION}.mp4"
        )
    else:
        filename = (
            f"clip_{clip_index:02d}_final_preview_v"
            f"{RENDERER_VERSION}.mp4"
        )

    return (directory / filename).resolve()


# ============================================================
# RENDER
# ============================================================


def calculate_transition_duration(
    teaser_duration: float,
    main_duration: float,
) -> float:
    """
    Very short crossfade: enough to remove the harsh splice without turning
    the edit into a slow cinematic dissolve.
    """

    safe_max = min(
        MAX_TRANSITION_SECONDS,
        max(0.0, teaser_duration * 0.22),
        max(0.0, main_duration * 0.22),
    )

    if safe_max < MIN_TRANSITION_SECONDS:
        return 0.0

    return round(
        min(
            SMOOTH_TRANSITION_SECONDS,
            safe_max,
        ),
        3,
    )


def build_filter_complex(
    teaser_start: float,
    teaser_end: float,
    ass_path: str,
    width: int,
    height: int,
    fps: float,
    edited_has_audio: bool,
    main_has_audio: bool,
    transition_duration: float,
    main_start: float = 0.0,
) -> tuple[str, list[str]]:
    fps_value = f"{fps:.6f}"
    teaser_duration = max(0.0, teaser_end - teaser_start)

    teaser_video = (
        f"[0:v]"
        f"trim=start={teaser_start:.6f}:end={teaser_end:.6f},"
        f"setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:flags=lanczos,"
        f"setsar=1,"
        f"format={PIXEL_FORMAT},"
        f"subtitles=filename='{ass_path}',"
        f"setsar=1,"
        f"settb=AVTB,"
        f"setpts=N/({fps_value}*TB),"
        f"fps={fps_value}"
        f"[teaser_v];"
    )

    main_start = max(0.0, float(main_start))

    main_video = (
        f"[1:v]"
        f"trim=start={main_start:.6f},"
        f"setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:flags=lanczos,"
        f"setsar=1,"
        f"format={PIXEL_FORMAT},"
        f"settb=AVTB,"
        f"setpts=N/({fps_value}*TB),"
        f"fps={fps_value}"
        f"[main_v];"
    )

    if edited_has_audio != main_has_audio:
        raise RuntimeError(
            "Edited clip ile captioned preview audio yapısı uyuşmuyor. "
            "İkisinde de ses olmalı."
        )

    transition_duration = max(0.0, float(transition_duration))

    # Extremely short clips: keep deterministic concat instead of invalid xfade.
    if transition_duration <= 0.0:
        if edited_has_audio and main_has_audio:
            audio_filters = (
                f"[0:a]"
                f"atrim=start={teaser_start:.6f}:end={teaser_end:.6f},"
                f"asetpts=PTS-STARTPTS,"
                f"aresample={AUDIO_SAMPLE_RATE},"
                f"aformat=sample_fmts=fltp:"
                f"sample_rates={AUDIO_SAMPLE_RATE}:"
                f"channel_layouts=stereo"
                f"[teaser_a];"

                f"[1:a]"
                f"atrim=start={main_start:.6f},"
                f"asetpts=PTS-STARTPTS,"
                f"aresample={AUDIO_SAMPLE_RATE},"
                f"aformat=sample_fmts=fltp:"
                f"sample_rates={AUDIO_SAMPLE_RATE}:"
                f"channel_layouts=stereo"
                f"[main_a];"
            )

            concat_filter = (
                f"[teaser_v][teaser_a]"
                f"[main_v][main_a]"
                f"concat=n=2:v=1:a=1"
                f"[out_v][out_a]"
            )

            return (
                teaser_video + main_video + audio_filters + concat_filter,
                ["-map", "[out_v]", "-map", "[out_a]"],
            )

        concat_filter = (
            f"[teaser_v][main_v]"
            f"concat=n=2:v=1:a=0"
            f"[out_v]"
        )

        return (
            teaser_video + main_video + concat_filter,
            ["-map", "[out_v]"],
        )

    xfade_offset = max(
        0.0,
        teaser_duration - transition_duration,
    )

    video_transition = (
        f"[teaser_v][main_v]"
        f"xfade=transition=fade:"
        f"duration={transition_duration:.6f}:"
        f"offset={xfade_offset:.6f},"
        f"format={PIXEL_FORMAT}"
        f"[out_v]"
    )

    if edited_has_audio and main_has_audio:
        audio_filters = (
            f"[0:a]"
            f"atrim=start={teaser_start:.6f}:end={teaser_end:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"aresample={AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:"
            f"sample_rates={AUDIO_SAMPLE_RATE}:"
            f"channel_layouts=stereo"
            f"[teaser_a];"

            f"[1:a]"
            f"atrim=start={main_start:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"aresample={AUDIO_SAMPLE_RATE},"
            f"aformat=sample_fmts=fltp:"
            f"sample_rates={AUDIO_SAMPLE_RATE}:"
            f"channel_layouts=stereo"
            f"[main_a];"

            f"[teaser_a][main_a]"
            f"acrossfade=d={transition_duration:.6f}:"
            f"c1=tri:c2=tri"
            f"[out_a]"
        )

        return (
            teaser_video
            + main_video
            + audio_filters
            + ";"
            + video_transition,
            ["-map", "[out_v]", "-map", "[out_a]"],
        )

    return (
        teaser_video + main_video + video_transition,
        ["-map", "[out_v]"],
    )



def intro_quality_status(
    intro: dict[str, Any],
) -> tuple[
    bool,
    float,
    str,
]:

    try:
        score = float(
            intro.get(
                "score",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        score = 0.0

    if intro.get(
        "recommended"
    ) is not True:
        return (
            False,
            score,
            "recommended=false",
        )

    quality_gate = intro.get(
        "quality_gate",
        {},
    )

    locked_intro_override = bool(
        isinstance(quality_gate, dict)
        and quality_gate.get("locked_intro_override") is True
        and quality_gate.get("accepted") is not False
    )

    if score < MIN_RENDER_SCORE and not locked_intro_override:
        return (
            False,
            score,
            (
                f"score {score:.1f}/10 < "
                f"{MIN_RENDER_SCORE:.1f}/10"
            ),
        )

    if isinstance(
        quality_gate,
        dict,
    ):

        accepted = quality_gate.get(
            "accepted"
        )

        if accepted is False:
            return (
                False,
                score,
                "quality_gate.accepted=false",
            )

    return (
        True,
        score,
        "locked_intro_override" if locked_intro_override else "accepted",
    )


def render_clip(
    intro_json_path: str | Path,
    intro_package: dict[str, Any],
    teaser_package: dict[str, Any],
    timeline: dict[str, Any],
    clip_index: int,
    *,
    edited_clip_path: str | Path | None = None,
    captioned_preview_path: str | Path | None = None,
    caption_path: str | Path | None = None,
) -> Path:
    intro = get_intro(intro_package, clip_index)
    teaser = get_teaser(teaser_package, clip_index)

    accepted, intro_score, gate_reason = (
        intro_quality_status(
            intro
        )
    )

    if not accepted:
        raise RuntimeError(
            "Intro kalite kapısından geçmedi: "
            f"{gate_reason}. "
            "No intro is better than a bad intro."
        )

    text = clean_hook_text(str(intro.get("intro_text", "")))

    if not text:
        raise RuntimeError("intro_text boş.")

    # Teaser CLEAN edited clip'ten gelir -> normal dynamic caption YOK.
    # Main CAPTIONED preview'dan gelir -> restart sonrası normal caption VAR.
    # Pipeline may pass the exact edited clip identity. Prefer it over directory
    # discovery so stale/multiple edited variants can never hijack the mandatory intro.
    if edited_clip_path is not None:
        edited_clip = Path(edited_clip_path).expanduser().resolve()
        if not edited_clip.is_file():
            raise FileNotFoundError(
                f"Pipeline'in verdiği edited clip bulunamadı:\n{edited_clip}"
            )
    else:
        edited_clip = find_edited_clip(
            timeline=timeline,
            clip_index=clip_index,
        )

    if captioned_preview_path is not None:
        captioned_preview = Path(captioned_preview_path).expanduser().resolve()
        if not captioned_preview.is_file():
            raise FileNotFoundError(
                f"Pipeline'in verdiği captioned preview bulunamadı:\n{captioned_preview}"
            )
    else:
        captioned_preview = find_captioned_preview(
            timeline=timeline,
            clip_index=clip_index,
        )

    edited_info = get_video_info(edited_clip)
    main_info = get_video_info(captioned_preview)

    main_restart, first_caption_start = calculate_main_restart_seconds(
        caption_path=caption_path,
        main_duration=float(main_info["duration"]),
    )
    main_effective_duration = max(
        0.0,
        float(main_info["duration"]) - main_restart,
    )

    teaser_start, teaser_end = get_teaser_bounds(
        teaser=teaser,
        edited_duration=float(edited_info["duration"]),
    )

    teaser_duration = teaser_end - teaser_start

    display_duration = calculate_display_duration(
        intro=intro,
        teaser_duration=teaser_duration,
    )

    display_duration = min(
        display_duration,
        max(0.25, teaser_duration - 0.03),
    )

    transition_duration = calculate_transition_duration(
        teaser_duration=teaser_duration,
        main_duration=main_effective_duration,
    )

    width = int(main_info["width"])
    height = int(main_info["height"])
    fps = float(main_info["fps"])

    font_name = detect_font()

    ass_file = save_intro_ass(
        intro_json_path=intro_json_path,
        clip_index=clip_index,
        text=text,
        display_duration=display_duration,
        width=width,
        height=height,
        font_name=font_name,
    )

    output_path = get_output_path(
        timeline=timeline,
        intro=intro,
        clip_index=clip_index,
    )

    filter_complex, maps = build_filter_complex(
        teaser_start=teaser_start,
        teaser_end=teaser_end,
        ass_path=escape_filter_path(ass_file),
        width=width,
        height=height,
        fps=fps,
        edited_has_audio=bool(edited_info["has_audio"]),
        main_has_audio=bool(main_info["has_audio"]),
        transition_duration=transition_duration,
        main_start=main_restart,
    )

    expected_duration = (
        teaser_duration
        + main_effective_duration
        - transition_duration
    )

    print()
    print("=" * 72)
    print(f"⚡ MIMIR INTRO RENDERER V{RENDERER_VERSION}")
    print("=" * 72)
    print(f"🎞️ Clip          : {clip_index}")
    print(f"🧲 Hook text     : {text}")
    print(f"⭐ Terra score   : {intro_score:.1f}/10")
    print(f"🛡️ Quality gate  : >= {MIN_RENDER_SCORE:.1f}/10")
    print(f"🔤 Font          : {font_name}")
    print(f"🔥 Teaser cut    : {teaser_start:.3f} → {teaser_end:.3f}s")
    print(f"🔥 Teaser süre   : {teaser_duration:.3f}s")
    print(f"📝 Neon süre     : {display_duration:.3f}s")
    print(f"🌊 Transition    : {transition_duration:.3f}s smooth crossfade")
    if main_restart > 0.0:
        lead_after_restart = (
            max(0.0, float(first_caption_start) - main_restart)
            if first_caption_start is not None
            else 0.0
        )
        print(
            f"✂️ Main restart  : {main_restart:.3f}s "
            f"(ilk caption {float(first_caption_start):.3f}s, preroll ≈ {lead_after_restart:.3f}s)"
        )
    else:
        print("✂️ Main restart  : 0.000s (uzun post-intro gap yok / güvenli fallback)")
    print("🚫 Teaser altyazı: YOK")
    print("✅ Main altyazı  : VAR")
    print(
        "🎬 Yapı          : CLEAN TEASER + HOOK → SMOOTH MAIN RESTART "
        f"{main_restart:.2f}s"
    )
    print(f"📥 Clean teaser source:\n{edited_clip}")
    print(f"📥 Captioned main:\n{captioned_preview}")
    print(f"📝 Hook ASS:\n{ass_file}")
    print()
    print("🎞️ Render başlıyor...")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(edited_clip),
        "-i",
        str(captioned_preview),
        "-filter_complex",
        filter_complex,
        *maps,
        "-r",
        f"{fps:.6f}",
        "-c:v",
        VIDEO_CODEC,
        "-preset",
        VIDEO_PRESET,
        "-crf",
        VIDEO_CRF,
        "-pix_fmt",
        PIXEL_FORMAT,
    ]

    if bool(edited_info["has_audio"]) and bool(main_info["has_audio"]):
        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                AUDIO_BITRATE,
            ]
        )

    command.extend(
        [
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("FFmpeg bulunamadı.") from error

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg Intro Render V8 hatası:\n\n"
            + result.stderr
        )

    if not output_path.exists():
        raise RuntimeError("Final intro preview V8 oluşmadı.")

    actual_info = get_video_info(output_path)
    actual_duration = float(actual_info["duration"])
    difference = abs(actual_duration - expected_duration)

    print()
    print(f"✅ Expected final : {expected_duration:.3f}s")
    print(f"✅ Actual final   : {actual_duration:.3f}s")

    if difference > 0.30:
        print(f"⚠️ Duration farkı: {difference:.3f}s")

    print(f"✅ Çıktı:\n{output_path}")
    print("=" * 72)

    return output_path



def render_all(
    intro_json_path: str | Path,
    intro_package: dict[str, Any],
    teaser_package: dict[str, Any],
    timeline: dict[str, Any],
) -> list[Path]:
    outputs: list[Path] = []

    for position, intro in enumerate(intro_package["intros"], start=1):
        if not isinstance(intro, dict):
            continue

        try:
            clip_index = int(intro.get("clip_index", position))
        except (TypeError, ValueError):
            clip_index = position

        accepted, score, gate_reason = (
            intro_quality_status(
                intro
            )
        )

        if not accepted:
            print(
                f"⏭️ Clip {clip_index}: intro reddedildi "
                f"({score:.1f}/10, {gate_reason})."
            )
            continue

        outputs.append(
            render_clip(
                intro_json_path=intro_json_path,
                intro_package=intro_package,
                teaser_package=teaser_package,
                timeline=timeline,
                clip_index=clip_index,
            )
        )

    return outputs


# ============================================================
# PUBLIC ENTRY
# ============================================================


def run_renderer(
    intro_json_path: str | Path,
    clip_index: int | None = None,
    *,
    edited_clip_path: str | Path | None = None,
    captioned_preview_path: str | Path | None = None,
    caption_path: str | Path | None = None,
) -> list[Path]:
    intro_json_path = Path(intro_json_path).resolve()

    intro_package = load_json(intro_json_path)
    validate_intro_package(intro_package)

    timeline_path, teaser_path = get_referenced_paths(intro_package)

    timeline = load_json(timeline_path)
    teaser_package = load_json(teaser_path)

    validate_timeline(timeline)
    validate_teaser_package(teaser_package)

    if (
        edited_clip_path is not None
        or captioned_preview_path is not None
        or caption_path is not None
    ) and clip_index is None:
        raise ValueError(
            "Explicit edited/captioned path yalnız belirli bir clip_index ile kullanılabilir."
        )

    if clip_index is not None:
        return [
            render_clip(
                intro_json_path=intro_json_path,
                intro_package=intro_package,
                teaser_package=teaser_package,
                timeline=timeline,
                clip_index=clip_index,
                edited_clip_path=edited_clip_path,
                captioned_preview_path=captioned_preview_path,
                caption_path=caption_path,
            )
        ]

    return render_all(
        intro_json_path=intro_json_path,
        intro_package=intro_package,
        teaser_package=teaser_package,
        timeline=timeline,
    )


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    print()
    print("MIMIR Intro Renderer V8")
    print("Clean teaser + hook -> smooth captioned main restart")
    print()

    intro_json_path = input(
        "Intro JSON yolunu gir: "
    ).strip().strip('"')

    clip_input = input(
        "Clip index (boş = tüm önerilen klipler): "
    ).strip()

    try:
        selected_clip = int(clip_input) if clip_input else None

        run_renderer(
            intro_json_path=intro_json_path,
            clip_index=selected_clip,
        )

    except Exception as error:
        print()
        print("❌ INTRO RENDERER V8 HATASI:")
        print(error)
