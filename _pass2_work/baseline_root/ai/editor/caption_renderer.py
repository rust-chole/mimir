from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EDITED_CLIPS_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "edited_clips"
)

CAPTIONS_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "captions"
)

PREVIEW_DIR = (
    PROJECT_ROOT
    / "vod_output"
    / "previews"
)


# ============================================================
# CONFIG
# ============================================================

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "fast"
VIDEO_CRF = "18"

PIXEL_FORMAT = "yuv420p"

DURATION_WARNING_TOLERANCE = 0.20


# ============================================================
# HELPERS
# ============================================================

def load_json(
    path: str | Path,
) -> dict[str, Any]:

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSON bulunamadı:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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
# FFMPEG / FFPROBE
# ============================================================

def check_subtitles_filter() -> None:

    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-filters",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    except FileNotFoundError as error:

        raise RuntimeError(
            "FFmpeg bulunamadı."
        ) from error

    output = result.stdout.casefold()

    if "subtitles" not in output:

        raise RuntimeError(
            "Bu FFmpeg kurulumunda 'subtitles' filtresi "
            "bulunamadı. libass destekli FFmpeg gerekiyor."
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
            "Geçersiz video duration."
        ) from error


# ============================================================
# WINDOWS FFMPEG PATH ESCAPING
# ============================================================

def escape_filter_path(
    path: str | Path,
) -> str:
    """
    FFmpeg filter içindeki Windows path'ini güvenli hale getirir.

    C:\\Users\\...\\file.ass

    →

    C\\:/Users/.../file.ass
    """

    path = Path(
        path
    ).resolve()

    value = path.as_posix()

    # Filtre seçeneği (2. seviye) kaçışı: önce ters bölü.
    # Windows'ta as_posix() sonrası ters bölü kalmaz (çıktı aynı).
    value = value.replace(
        "\\",
        "\\\\",
    )

    # Windows drive colon (Windows path'inde tek ':' budur):
    # C:/... -> C\:/...
    value = value.replace(
        ":",
        "\\:",
    )

    # FFmpeg filter string içindeki tek tırnak: tırnaklı değer
    # içinde \' çalışmaz (tırnağı kapatır, tırnak kaybolur).
    # Tırnağı kapat, kaçışlı tırnak ekle, tırnağı yeniden aç.
    value = value.replace(
        "'",
        "'\\\\\\''",
    )

    return value


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
            f"Timeline V3 gerekiyor. Bulunan: {version}"
        )

    if not timeline_data.get(
        "timelines"
    ):

        raise RuntimeError(
            "Timeline içinde klip bulunamadı."
        )


def get_clip_timeline(
    timeline_data: dict[str, Any],
    clip_index: int,
) -> dict[str, Any]:

    for clip in timeline_data.get(
        "timelines",
        [],
    ):

        if int(
            clip.get(
                "clip_index",
                -1,
            )
        ) == int(
            clip_index
        ):

            return clip

    raise RuntimeError(
        f"clip_index={clip_index} bulunamadı."
    )


# ============================================================
# INPUT PATHS
# ============================================================

def get_video_stem(
    timeline_data: dict[str, Any],
) -> str:

    source = timeline_data.get(
        "source",
        {},
    )

    stem = str(
        source.get(
            "video_stem",
            "vod",
        )
    ).strip()

    return (
        safe_filename(stem)
        if stem
        else "vod"
    )


def find_edited_clip(
    timeline_data: dict[str, Any],
    clip_timeline: dict[str, Any],
) -> Path:

    video_stem = get_video_stem(
        timeline_data
    )

    clip_index = int(
        clip_timeline[
            "clip_index"
        ]
    )

    folder = (
        EDITED_CLIPS_DIR
        / video_stem
    )

    if not folder.exists():

        raise FileNotFoundError(
            "Edited clips klasörü bulunamadı:\n"
            f"{folder}\n\n"
            "Önce pacing_cutter çalıştır."
        )

    # Önce index üzerinden ara.
    matches = sorted(
        folder.glob(
            f"clip_{clip_index:02d}_*_edited.mp4"
        )
    )

    if len(matches) == 1:
        return matches[0].resolve()

    # Tam beklenen ismi dene.
    title = safe_filename(
        clip_timeline.get(
            "title",
            f"clip_{clip_index}",
        )
    )[:70]

    exact = (
        folder
        / (
            f"clip_{clip_index:02d}_"
            f"{title}_edited.mp4"
        )
    )

    if exact.exists():
        return exact.resolve()

    if not matches:

        raise FileNotFoundError(
            f"Clip {clip_index} için edited MP4 bulunamadı:\n"
            f"{folder}"
        )

    raise RuntimeError(
        f"Clip {clip_index} için birden fazla edited MP4 bulundu."
    )


def find_caption_file(
    timeline_path: str | Path,
    clip_index: int,
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve the caption file without hard-coding an old ASS version."""
    if explicit_path is not None:
        explicit = Path(explicit_path).resolve()
        if explicit.is_file():
            return explicit
        raise FileNotFoundError(f"Caption ASS bulunamadı:\n{explicit}")

    timeline_path = Path(timeline_path).resolve()
    base_name = timeline_path.stem.replace("_timeline_v3", "")
    folder = CAPTIONS_DIR / base_name

    candidates = list(folder.glob(f"clip_{int(clip_index):02d}_captions_v*.ass"))
    if not candidates:
        raise FileNotFoundError(
            "Caption ASS bulunamadı:\n"
            f"{folder}\n\n"
            "Pipeline caption aşamasını tekrar çalıştır."
        )

    def version_key(path: Path) -> tuple[int, float]:
        import re
        match = re.search(r"_captions_v(\d+)\.ass$", path.name)
        version = int(match.group(1)) if match else -1
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0.0
        return version, modified

    return max(candidates, key=version_key).resolve()


# ============================================================
# OUTPUT
# ============================================================

def get_output_path(
    timeline_data: dict[str, Any],
    clip_timeline: dict[str, Any],
) -> Path:

    video_stem = get_video_stem(
        timeline_data
    )

    clip_index = int(
        clip_timeline[
            "clip_index"
        ]
    )

    folder = (
        PREVIEW_DIR
        / video_stem
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        folder
        / f"clip_{clip_index:02d}_captioned_preview.mp4"
    )


# ============================================================
# RENDER
# ============================================================

def render_captioned_clip(
    timeline_path: str | Path,
    clip_index: int,
    caption_path: str | Path | None = None,
) -> Path:

    timeline_path = Path(
        timeline_path
    ).resolve()

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

    edited_clip = find_edited_clip(
        timeline_data=timeline_data,
        clip_timeline=clip_timeline,
    )

    caption_file = find_caption_file(
        timeline_path=timeline_path,
        clip_index=clip_index,
        explicit_path=caption_path,
    )

    output_path = get_output_path(
        timeline_data=timeline_data,
        clip_timeline=clip_timeline,
    )

    # --------------------------------------------------------
    # VERIFY INPUT DURATION
    # --------------------------------------------------------

    actual_input_duration = get_video_duration(
        edited_clip
    )

    expected_duration = float(
        clip_timeline.get(
            "edited",
            {},
        ).get(
            "estimated_duration",
            actual_input_duration,
        )
    )

    duration_difference = abs(
        actual_input_duration
        - expected_duration
    )

    print()
    print(
        "=" * 68
    )

    print(
        f"🎬 CAPTION PREVIEW — CLIP {clip_index}"
    )

    print(
        "=" * 68
    )

    print(
        f"📹 Video:\n{edited_clip}"
    )

    print()

    print(
        f"📝 Captions:\n{caption_file}"
    )

    print()

    print(
        f"⏱️ Video süresi: "
        f"{actual_input_duration:.3f}s"
    )

    print(
        f"🎯 Timeline süresi: "
        f"{expected_duration:.3f}s"
    )

    if (
        duration_difference
        > DURATION_WARNING_TOLERANCE
    ):

        print(
            f"⚠️ Duration farkı: "
            f"{duration_difference:.3f}s"
        )

    else:

        print(
            "✅ Duration eşleşiyor."
        )

    # --------------------------------------------------------
    # SUBTITLE FILTER
    # --------------------------------------------------------

    escaped_caption_path = escape_filter_path(
        caption_file
    )

    subtitle_filter = (
        "subtitles="
        f"filename='{escaped_caption_path}'"
    )

    # --------------------------------------------------------
    # FFMPEG
    # --------------------------------------------------------

    command = [
        "ffmpeg",
        "-y",

        "-i",
        str(
            edited_clip
        ),

        "-vf",
        subtitle_filter,

        "-c:v",
        VIDEO_CODEC,

        "-preset",
        VIDEO_PRESET,

        "-crf",
        VIDEO_CRF,

        "-pix_fmt",
        PIXEL_FORMAT,

        # Video tekrar encode olmak zorunda,
        # ama sesi tekrar encode etmeye gerek yok.
        "-c:a",
        "copy",

        "-movflags",
        "+faststart",

        str(
            output_path
        ),
    ]

    print()
    print(
        "📝 Dinamik altyazılar videoya basılıyor..."
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
            "FFmpeg caption render hatası:\n\n"
            + result.stderr
        )

    if not output_path.exists():

        raise RuntimeError(
            "Render tamamlandı fakat çıktı oluşmadı."
        )

    # --------------------------------------------------------
    # OUTPUT VERIFY
    # --------------------------------------------------------

    output_duration = get_video_duration(
        output_path
    )

    print()
    print(
        "✅ CAPTION PREVIEW HAZIR"
    )

    print(
        f"⏱️ Final süre: "
        f"{output_duration:.3f}s"
    )

    print(
        f"📂 {output_path}"
    )

    print(
        "=" * 68
    )

    return output_path


# ============================================================
# RENDER ALL
# ============================================================

def render_all_captioned_clips(
    timeline_path: str | Path,
) -> list[Path]:

    timeline_path = Path(
        timeline_path
    ).resolve()

    timeline_data = load_json(
        timeline_path
    )

    validate_timeline(
        timeline_data
    )

    outputs: list[Path] = []

    for clip in timeline_data.get(
        "timelines",
        [],
    ):

        clip_index = int(
            clip[
                "clip_index"
            ]
        )

        output = render_captioned_clip(
            timeline_path=timeline_path,
            clip_index=clip_index,
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
        "📝 MIMIR CAPTION PREVIEW RENDERER"
    )

    print(
        "=" * 68
    )

    try:

        check_subtitles_filter()

        print(
            "✅ FFmpeg subtitles/libass hazır."
        )

        timeline_path = input(
            "\nTimeline V3 JSON yolunu gir: "
        ).strip().strip('"')

        clip_index_text = input(
            "Clip index "
            "(boş = tüm klipler): "
        ).strip()

        if clip_index_text:

            render_captioned_clip(
                timeline_path=timeline_path,
                clip_index=int(
                    clip_index_text
                ),
            )

        else:

            outputs = (
                render_all_captioned_clips(
                    timeline_path=timeline_path,
                )
            )

            print()
            print(
                "=" * 68
            )

            print(
                f"✅ TOPLAM {len(outputs)} PREVIEW HAZIR"
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
            "❌ CAPTION RENDERER HATASI:"
        )

        print(
            error
        )