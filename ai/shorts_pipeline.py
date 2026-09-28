from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from ai import model_config, vod_processor
from ai.editor import (
    caption_renderer,
    caption_truth,
    clip_analyzer,
    captions,
    final_qc,
    final_review,
    intro_analyzer,
    intro_bounds,
    intro_peak_support,
    intro_renderer,
    meme_audio_support,
    meme_analyzer,
    meme_discovery,
    meme_renderer,
    pacing,
    pacing_cutter,
    participant_name_lock,
    speaker_caption_support,
    speaker_naming,
    speaker_role_judge,
    teaser_analyzer,
    timeline,
    v6_runtime,
)


PIPELINE_VERSION = 10
TOTAL_STAGES = 15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOD_OUTPUT_DIR = PROJECT_ROOT / "vod_output"
STATE_DIR = VOD_OUTPUT_DIR / "pipeline"
PUBLISHED_DIR = VOD_OUTPUT_DIR / "final"
REJECTED_DIR = VOD_OUTPUT_DIR / "rejected"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts"}
MIN_VIDEO_BYTES = 1024
MIN_JSON_BYTES = 8
MIN_CAPTION_BYTES = 80
MIN_INTRO_ACCEPT_SCORE = 8.0
PREFERRED_SHORT_DURATION = 32.0
SAFE_PIPELINE_PARALLEL = str(os.getenv("MIMIR_SAFE_PIPELINE_PARALLEL", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _load_pro_edit() -> tuple[Any, Any]:
    """Pro Edit is the production presentation layer (caption presentation over
    frozen truth, evidence-directed camera, pixel proof). It is always on; an
    import/config failure stops the run loudly instead of silently rendering
    the legacy caption path."""
    from ai.editor import pro_edit as pro_edit_pkg

    config = pro_edit_pkg.config.load_config(override_enabled=True, v6=True)
    return pro_edit_pkg, config


def _speaker_profile_path(edited_clip_path: Path, clip_index: int) -> Path | None:
    """Locate the final speaker/caption profile (Pro Edit word IDs come from it)."""
    finder = getattr(speaker_caption_support, "_output_path", None)
    if not callable(finder):
        return None
    try:
        path = Path(finder(edited_clip_path, clip_index)).resolve()
    except (OSError, TypeError, ValueError):
        return None  # Pro Edit then runs without word evidence (recorded in its context artifact)
    return path if path.is_file() else None


class ShortsPipelineError(RuntimeError):
    pass


class NoStrongClipError(ShortsPipelineError):
    pass


# ============================================================
# BASIC HELPERS
# ============================================================

def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_name(value: str) -> str:
    text = str(value).strip()
    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")
    return " ".join(text.split()).strip(" ._") or "video"


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"JSON bulunamadı:\n{path}")

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(f"JSON dosyası boş:\n{path}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"JSON formatı bozuk:\n{path}\n"
            f"Satır={error.lineno} Sütun={error.colno} Hata={error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root object değil:\n{path}")
    return data


def _write_json_atomic(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return path


def _valid_file(path: str | Path, min_bytes: int = 1) -> bool:
    path = Path(path)
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _probe_video_duration(path: str | Path) -> float:
    """Return video duration in seconds without making pipeline success depend on ffprobe."""
    target = Path(path).resolve()
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(target),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            timeout=20,
        )
        return max(0.0, float(completed.stdout.strip()))
    except Exception:
        return 0.0


def _file_size_mb(path: str | Path) -> float:
    try:
        return max(0.0, Path(path).stat().st_size / (1024 * 1024))
    except OSError:
        return 0.0


def _human_time(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} sa {minutes} dk {secs} sn"
    if minutes:
        return f"{minutes} dk {secs} sn"
    return f"{secs} sn"


def _print_friendly_result(result: dict[str, Any]) -> None:
    output = Path(str(result.get("final_output", ""))).resolve()
    selected = result.get("selected_clip", {})
    if not isinstance(selected, dict):
        selected = {}
    stats = result.get("stats", {})
    if not isinstance(stats, dict):
        stats = {}

    title = str(selected.get("title", "Short"))
    try:
        score = float(selected.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0

    try:
        source_duration = float(stats.get("selected_duration_seconds", 0.0))
    except (TypeError, ValueError):
        source_duration = 0.0
    try:
        final_duration = float(stats.get("final_duration_seconds", 0.0))
    except (TypeError, ValueError):
        final_duration = 0.0
    try:
        intro_duration = float(stats.get("intro_duration_seconds", 0.0))
    except (TypeError, ValueError):
        intro_duration = 0.0
    try:
        meme_count = int(stats.get("meme_count", 0))
    except (TypeError, ValueError):
        meme_count = 0
    try:
        anchor_count = int(stats.get("anchor_count", selected.get("anchor_count", 0)))
    except (TypeError, ValueError):
        anchor_count = 0
    try:
        protected_count = int(
            stats.get("protected_range_count", selected.get("protected_range_count", 0))
        )
    except (TypeError, ValueError):
        protected_count = 0
    try:
        size_mb = float(stats.get("file_size_mb", 0.0))
    except (TypeError, ValueError):
        size_mb = 0.0
    warnings = result.get("warnings", [])
    warning_count = len(warnings) if isinstance(warnings, list) else 0

    print()
    if result.get("status") == "published_degraded":
        print("⚠️ SHORT HAZIR — QC geçti, DEGRADED uyarılarla (aşağıya bak)")
    else:
        print("✅ SHORT HAZIR — render edilmiş MP4 final QC'den geçti")
    print("────────────────────────────────────────")
    print(f"🎬 {title}")
    print()
    print("📊 İstatistikler")
    print(f"   Terra skoru : {score:.1f}/10")
    if source_duration > 0:
        print(f"   Seçilen klip: {source_duration:.1f} sn")
    if final_duration > 0:
        print(f"   Final video : {final_duration:.1f} sn")
    if intro_duration > 0:
        print(f"   Intro       : {intro_duration:.1f} sn")
    print(f"   Meme        : {meme_count}")
    print(f"   Anchor      : {anchor_count}")
    print(f"   Protected   : {protected_count}")
    if size_mb > 0:
        print(f"   Dosya boyutu: {size_mb:.1f} MB")
    if result.get("fast_resume"):
        print("   İşlem       : hazır çıktı kullanıldı")
    else:
        print(f"   İşlem süresi: {_human_time(float(result.get('run_seconds', 0.0)))}")
    print(f"   Uyarı       : {warning_count}")

    profile = result.get("profile", {})
    if not result.get("fast_resume") and isinstance(profile, dict):
        slowest = profile.get("slowest", [])
        if isinstance(slowest, list) and slowest:
            print("   En yavaş    :")
            for item in slowest[:5]:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", item.get("key", "stage")))
                try:
                    seconds = float(item.get("seconds", 0.0))
                except (TypeError, ValueError):
                    seconds = 0.0
                if seconds > 0.0:
                    print(f"      ↳ {label}: {seconds:.1f}s")
    if warning_count and isinstance(warnings, list):
        for warning in warnings[:2]:
            text = " ".join(str(warning).split())
            if len(text) > 150:
                text = text[:147] + "..."
            print(f"      ↳ {text}")
    v6 = result.get("v6")
    if isinstance(v6, dict) and v6:
        print(f"   Final gate  : {v6.get('status', '?')}")
        for name in (v6.get("failed_checks") or [])[:6]:
            print(f"      ↳ gate: {name}")
        for row in (v6.get("fallback_rows") or [])[:4]:
            print(f"      ↳ fallback: {row.get('subsystem')} -> {row.get('level')}")
            reason = " ".join(str(row.get("reason", "")).split())
            if reason:
                print(f"        neden: {reason[:220] + ('...' if len(reason) > 220 else '')}")
    print()
    print("📁 Video burada:")
    print(f"   {output}")
    print("────────────────────────────────────────")


def _valid_json(path: str | Path, expected_version: int | None = None) -> bool:
    if not _valid_file(path, MIN_JSON_BYTES):
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False

    if expected_version is None:
        return True

    try:
        return int(data.get("version", -1)) == expected_version
    except (TypeError, ValueError):
        return False


def _package_clip(path: str | Path, list_key: str, clip_index: int) -> dict[str, Any] | None:
    try:
        package = _load_json(path)
    except Exception:
        return None

    items = package.get(list_key, [])
    if not isinstance(items, list):
        return None

    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("clip_index", position))
        except (TypeError, ValueError):
            current = position
        if current == clip_index:
            return item
    return None


def _contains_clip(path: str | Path, list_key: str, clip_index: int, version: int) -> bool:
    try:
        package = _load_json(path)
        if int(package.get("version", -1)) != version:
            return False
    except Exception:
        return False
    return _package_clip(path, list_key, clip_index) is not None


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


_MODULE_HASHES: dict[tuple[str, int, int], str] = {}


def _module_fingerprint(module: Any) -> dict[str, Any]:
    """Code identity by CONTENT, not by path/mtime.

    A fresh checkout, a re-extracted archive or a moved project directory must
    not re-bill paid stages whose code is byte-identical; any real change to a
    module's source still invalidates every stage that fingerprints it."""
    raw = getattr(module, "__file__", None)
    name = str(getattr(module, "__name__", "<unknown>"))
    if not raw:
        return {"module": name, "sha256": None}
    path = Path(raw).resolve()
    try:
        stat = path.stat()
    except OSError:
        return {"module": name, "sha256": None}
    key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _MODULE_HASHES.get(key)
    if digest is None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _MODULE_HASHES[key] = digest
    return {"module": name, "sha256": digest}


def _hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stage_signature(
    name: str,
    *,
    inputs: Iterable[str | Path] = (),
    modules: Iterable[Any] = (),
    options: dict[str, Any] | None = None,
) -> str:
    return _hash(
        {
            "pipeline_version": PIPELINE_VERSION,
            "stage": name,
            "inputs": [_file_fingerprint(path) for path in inputs],
            "modules": [_module_fingerprint(module) for module in modules],
            "options": options or {},
        }
    )


def _module_paths(modules: Iterable[Any]) -> list[Path]:
    result: list[Path] = []
    for module in modules:
        raw = getattr(module, "__file__", None)
        if raw:
            result.append(Path(raw).resolve())
    return result




# ============================================================
# VIDEO PATH
# ============================================================

def resolve_video_path(video_path: str | Path) -> Path:
    raw_text = str(video_path).strip().strip('"')
    if not raw_text:
        raise FileNotFoundError("Video yolu boş.")

    raw = Path(raw_text).expanduser()
    candidates = [raw]

    if not raw.is_absolute():
        candidates.extend(
            [
                Path.cwd() / raw,
                Path.home() / "Downloads" / raw,
                Path.home() / "Desktop" / raw,
                Path.home() / "Documents" / raw,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue

        key = str(candidate).casefold()
        if key in seen:
            continue
        seen.add(key)

        if candidate.is_file():
            return candidate

    if not raw.is_absolute() and raw.parent == Path("."):
        query = raw.name.casefold()
        roots = [
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            Path.home() / "Documents",
        ]
        exact: list[Path] = []
        partial: list[Path] = []

        for root in roots:
            if not root.is_dir():
                continue
            try:
                for item in root.rglob("*"):
                    if not item.is_file():
                        continue
                    suffix = item.suffix.casefold()
                    if suffix and suffix not in VIDEO_EXTENSIONS:
                        continue

                    name = item.name.casefold()
                    if name == query:
                        exact.append(item.resolve())
                    elif query in name:
                        partial.append(item.resolve())

                    if len(exact) + len(partial) >= 30:
                        break
            except (OSError, PermissionError):
                continue

        matches = exact if exact else partial
        unique = {str(item).casefold(): item for item in matches}
        matches = list(unique.values())

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            preview = "\n".join(str(item) for item in matches[:8])
            raise FileNotFoundError(
                "Birden fazla video eşleşti. Tam dosya yolunu kullan:\n" + preview
            )

    raise FileNotFoundError(f"Video bulunamadı:\n{raw_text}")


# ============================================================
# STATE / STAGE BOOK
# ============================================================

def _source_fingerprint(video_path: Path) -> dict[str, Any]:
    stat = video_path.stat()
    return {
        "path": str(video_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _state_path(video_path: Path) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return (STATE_DIR / f"{_safe_name(video_path.stem)}_pipeline.json").resolve()


def _load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        state = _load_json(path)
        if int(state.get("pipeline_version", -1)) != PIPELINE_VERSION:
            return None
        return state
    except Exception:
        return None


def _same_source(state: dict[str, Any] | None, source: dict[str, Any]) -> bool:
    if not isinstance(state, dict):
        return False
    old = state.get("source", {})
    if not isinstance(old, dict):
        return False

    return (
        str(old.get("path", "")).casefold() == str(source["path"]).casefold()
        and int(old.get("size", -1)) == int(source["size"])
        and int(old.get("mtime_ns", -1)) == int(source["mtime_ns"])
    )


def _new_state(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_version": PIPELINE_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "source": source,
        "request_signature": None,
        "options": {},
        "selected_clip": None,
        "stages": {},
        "warnings": [],
        "video_brain_source_report": None,
        "video_brain_report": None,
        "final_output": None,
        "run_status": "running",
        "last_run_seconds": None,
    }


class StageBook:
    def __init__(self, state_path: Path, state: dict[str, Any], force: bool, rerun: Iterable[str] = ()) -> None:
        self.state_path = state_path
        self.state = state
        self.force = force
        # --force-v6: only these stage keys are recomputed (upstream caches stay valid).
        self.rerun = frozenset(rerun)

    def save(self) -> None:
        self.state["updated_at"] = _now()
        _write_json_atomic(self.state_path, self.state)

    def warn(self, text: str) -> None:
        warnings = self.state.setdefault("warnings", [])
        if text not in warnings:
            warnings.append(text)
        self.save()
        print(f"⚠️ {text}")

    def record(
        self,
        name: str,
        status: str,
        signature: str | None,
        *,
        path: str | Path | None = None,
        note: str | None = None,
        elapsed: float | None = None,
    ) -> None:
        stages = self.state.setdefault("stages", {})
        record: dict[str, Any] = {"status": status, "at": _now()}

        if signature is not None:
            record["signature"] = signature
        if path is not None:
            record["path"] = str(Path(path).resolve())
        if note:
            record["note"] = note
        if elapsed is not None:
            record["duration_seconds"] = round(max(0.0, elapsed), 3)

        stages[name] = record
        self.save()

    def reusable(
        self,
        name: str,
        signature: str,
        validator: Callable[[], bool],
        *,
        output: str | Path | None = None,
        freshness: Iterable[str | Path] = (),
    ) -> bool:
        """Reuse only artifacts recorded with the exact current stage signature.

        ``output`` and ``freshness`` remain accepted for call-site stability, but V8
        intentionally does not use mtime-only fallback. Old/untracked artifacts must
        never bypass a changed model, option, source or code signature.
        """
        del output, freshness

        if self.force or name in self.rerun:
            return False

        try:
            if not validator():
                return False
        except Exception:
            return False

        stages = self.state.get("stages", {})
        record = stages.get(name) if isinstance(stages, dict) else None
        if not isinstance(record, dict):
            return False
        if record.get("status") in {"fallback", "failed"}:
            return False
        return record.get("signature") == signature


class RuntimeProfiler:
    """Near-zero-overhead wall-time profiler for existing work only.

    StageBook already records foreground stage wall time. This helper records
    background worker time so V14 parallel work is not hidden by a fast join.
    It never changes scheduling or adds work.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, float] = {}

    def timed(self, key: str, func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            with self._lock:
                self._tasks[key] = round(max(0.0, elapsed), 3)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._tasks)


_PROFILE_LABELS = {
    "transcription": "Transcription",
    "clip_analysis": "Clip analysis",
    "pacing": "Pacing analysis",
    "timeline": "Timeline",
    "speaker_preflight": "Speaker",
    "pacing_cut": "Pacing render/join",
    "captions": "Caption accuracy",
    "caption_name_lock": "Participant name lock",
    "caption_truth_v6": "Caption truth V6",
    "video_brain": "Visual support",
    "teaser": "Teaser/peak",
    "intro_analysis": "Intro analysis",
    "caption_render": "Caption render/join",
    "intro_render": "Intro render",
    "meme_analysis": "Meme analysis",
    "meme_discovery": "Meme discovery",
    "meme_render": "Meme render",
    "publish": "Publish",
    "pacing_encode_worker": "Pacing encode (worker)",
    "visual_support_worker": "Visual support (worker)",
    "caption_render_worker": "Caption render (worker)",
}


def _build_profile(
    state: dict[str, Any],
    run_seconds: float,
    background_tasks: dict[str, float],
) -> dict[str, Any]:
    stage_wall: dict[str, float] = {}
    stage_status: dict[str, str] = {}
    stages = state.get("stages", {})
    if isinstance(stages, dict):
        for key, record in stages.items():
            if not isinstance(record, dict):
                continue
            try:
                seconds = max(0.0, float(record.get("duration_seconds", 0.0)))
            except (TypeError, ValueError):
                seconds = 0.0
            stage_wall[str(key)] = round(seconds, 3)
            stage_status[str(key)] = str(record.get("status", ""))

    candidates: list[dict[str, Any]] = []
    for key, seconds in stage_wall.items():
        if seconds <= 0.0 or stage_status.get(key) == "skipped":
            continue
        candidates.append({
            "key": key,
            "label": _PROFILE_LABELS.get(key, key.replace("_", " ").title()),
            "seconds": round(seconds, 3),
            "kind": "stage_wall",
        })
    for key, seconds in background_tasks.items():
        if seconds <= 0.0:
            continue
        candidates.append({
            "key": key,
            "label": _PROFILE_LABELS.get(key, key.replace("_", " ").title()),
            "seconds": round(seconds, 3),
            "kind": "background_worker",
        })

    candidates.sort(key=lambda item: (-float(item["seconds"]), str(item["key"])))
    return {
        "run_seconds": round(max(0.0, float(run_seconds)), 3),
        "stage_wall_seconds": stage_wall,
        "stage_status": stage_status,
        "background_worker_seconds": {
            str(key): round(max(0.0, float(value)), 3)
            for key, value in background_tasks.items()
        },
        "slowest": candidates[:8],
        "note": (
            "Stage wall süreleri paralel çalışan işleri overlap nedeniyle çift sayabilir; "
            "background_worker_seconds gerçek worker sürelerini ayrıca gösterir."
        ),
    }


# ============================================================
# PROGRESS / GENERIC STAGE
# ============================================================

def _print_header(
    video_path: Path,
    *,
    force: bool,
    memes: bool,
    video_brain: bool,
    keep_temp: bool,
    pro_edit: str | None = None,
    v6: str | None = None,
) -> None:
    print()
    print("╔════════════════════════════════════════════════════════════╗")
    print("║               MIMIR OPTIMAL V9 SHORTS                   ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\n🎬 {video_path}")
    print(f"♻️ Force: {'EVET' if force else 'HAYIR'}")
    print(f"🧠 Video Brain: {'AÇIK' if video_brain else 'KAPALI'}")
    print("🎯 Clip can alıcılığı + seçim: SADECE TERRA")
    print("👁️ Visual Observer: whole-VOD FACTS ONLY")
    print("🧠 Selected-clip visual support: intro için")
    print(f"😂 Meme sistemi: {'AÇIK' if memes else 'KAPALI'}")
    print(f"🧹 Temp koru: {'EVET' if keep_temp else 'HAYIR'}")
    if pro_edit is not None:
        print(f"🎥 Pro Edit: AÇIK ({pro_edit})")
    if v6 is not None:
        print(f"🧪 MIMIR V6: AÇIK ({v6})")
    model_config.print_model_plan()


def _stage_start(number: int, name: str) -> float:
    print(f"\n[{number}/{TOTAL_STAGES}] {name}")
    print("-" * 60)
    return time.perf_counter()


def _stage_done(path: str | Path | None = None) -> None:
    print("✅ Tamamlandı." if path is None else f"✅ Hazır:\n{Path(path).resolve()}")


def _stage_skip(path: str | Path | None = None) -> None:
    print(
        "⏭️ Hazır çıktı bulundu, skip."
        if path is None
        else f"⏭️ Hazır çıktı bulundu, skip:\n{Path(path).resolve()}"
    )


def _run_simple_stage(
    *,
    book: StageBook,
    number: int,
    title: str,
    key: str,
    signature: str,
    output: Path,
    validator: Callable[[], bool],
    runner: Callable[[], Any],
    freshness: Iterable[str | Path],
    error_message: str,
) -> Path:
    started = _stage_start(number, title)

    if book.reusable(
        key,
        signature,
        validator,
        output=output,
        freshness=freshness,
    ):
        _stage_skip(output)
        book.record(
            key,
            "skipped",
            signature,
            path=output,
            elapsed=time.perf_counter() - started,
        )
        return output

    runner()

    if not validator():
        raise ShortsPipelineError(error_message)

    _stage_done(output)
    book.record(
        key,
        "done",
        signature,
        path=output,
        elapsed=time.perf_counter() - started,
    )
    return output


# ============================================================
# PATHS / CLIP HELPERS
# ============================================================

def _transcript_path(video_path: Path) -> Path:
    return (Path(vod_processor.TRANSCRIPT_DIR) / f"{video_path.stem}.json").resolve()


def _analysis_path(transcript_path: Path) -> Path:
    return (
        Path(clip_analyzer.ANALYSIS_DIR)
        / f"{transcript_path.stem}_clips.json"
    ).resolve()


def _pacing_path(analysis_path: Path) -> Path:
    base = analysis_path.stem.replace("_clips", "")
    return (Path(pacing.PACING_OUTPUT_DIR) / f"{base}_pacing.json").resolve()


def _timeline_path(analysis_path: Path) -> Path:
    base = analysis_path.stem.replace("_clips", "")
    version = int(getattr(timeline, "TIMELINE_VERSION", 3))
    return (
        Path(timeline.TIMELINE_OUTPUT_DIR)
        / f"{base}_timeline_v{version}.json"
    ).resolve()


def _select_clip(
    analysis: dict[str, Any],
    requested_clip_index: int | None,
) -> tuple[int, dict[str, Any]]:
    clips = analysis.get("clips", [])
    if not isinstance(clips, list) or not clips:
        raise NoStrongClipError(
            "Terra bu videoda geçerli bir Short adayı üretemedi."
        )

    valid = [
        (position, clip)
        for position, clip in enumerate(clips, start=1)
        if isinstance(clip, dict)
    ]
    if not valid:
        raise NoStrongClipError("Clip Analyzer geçerli bir klip döndürmedi.")

    if requested_clip_index is not None:
        for position, clip in valid:
            if position == requested_clip_index:
                return position, clip
        raise ShortsPipelineError(
            f"clip_index={requested_clip_index} bulunamadı. Toplam klip: {len(valid)}"
        )

    # Terra final judge is the authoritative selector. Do not override that
    # decision with a duration heuristic.
    terra_selected = [
        item
        for item in valid
        if item[1].get(
            "terra_selected"
        ) is True
    ]

    if terra_selected:
        return terra_selected[0]

    def score_key(item: tuple[int, dict[str, Any]]) -> tuple[float, int, float, int]:
        position, clip = item
        try:
            score = float(clip.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        duration_raw: Any = clip.get("duration")
        if duration_raw is None and isinstance(clip.get("source"), dict):
            duration_raw = clip["source"].get("duration")
        try:
            duration = float(duration_raw)
        except (TypeError, ValueError):
            duration = 999.0

        covered = clip.get(
            "covered_moment_ids",
            [],
        )

        covered_count = (
            len(covered)
            if isinstance(
                covered,
                list,
            )
            else 0
        )

        return (
            score,
            covered_count,
            -abs(
                duration
                - PREFERRED_SHORT_DURATION
            ),
            -position,
        )

    valid.sort(key=score_key, reverse=True)
    return valid[0]


def _timeline_clip(timeline_data: dict[str, Any], clip_index: int) -> dict[str, Any]:
    items = timeline_data.get("timelines", [])
    if not isinstance(items, list):
        raise ShortsPipelineError("Timeline package içindeki timelines listesi geçersiz.")

    for position, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        try:
            current = int(item.get("clip_index", position))
        except (TypeError, ValueError):
            current = position
        if current == clip_index:
            return item

    raise ShortsPipelineError(f"Timeline içinde clip_index={clip_index} yok.")


def _selected_title(timeline_data: dict[str, Any], clip_index: int) -> str:
    return str(_timeline_clip(timeline_data, clip_index).get("title", f"clip_{clip_index}"))


# ============================================================
# FALLBACK PACKAGES
# ============================================================

def _clip_edited_duration(
    timeline_data: dict[str, Any],
    clip_index: int,
    edited_video_path: str | Path | None = None,
) -> float:
    clip = _timeline_clip(timeline_data, clip_index)
    edited = clip.get("edited", {})
    if isinstance(edited, dict):
        try:
            duration = float(edited.get("estimated_duration", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0.0:
            return duration

    if edited_video_path is not None:
        duration = _probe_video_duration(edited_video_path)
        if duration > 0.0:
            return duration

    return 0.0


def _deterministic_peak_candidate(
    *,
    clip: dict[str, Any],
    clip_duration: float,
    peak_support: dict[str, Any],
    restart_floor: float = 0.0,
) -> dict[str, Any]:
    """The strongest MEASURED peak, or the editor's payoff region; never a guessed position.

    The cold open must preview a future event that recurs in the story, so a
    candidate before the story restart (it would be trimmed away) is never used.
    """
    candidates = peak_support.get("candidates", [])
    valid: list[dict[str, Any]] = []
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start", 0.0))
                end = float(item.get("end", start))
            except (TypeError, ValueError):
                continue
            if end <= start or start < restart_floor - 1e-3 or end > clip_duration + 0.15:
                continue
            valid.append(item)

    if valid:
        # Cold-open must feel like a preview of a future event, not a duplicate
        # of the first frame. Prefer a later real peak when one exists.
        later_floor = max(restart_floor, min(1.0, max(0.55, clip_duration * 0.08)))
        later = [item for item in valid if float(item.get("start", 0.0)) >= later_floor]
        pool = later or valid

        def _rank(item: dict[str, Any]) -> tuple[float, float, float]:
            try:
                combined = float(item.get("combined_score", 0.0))
            except (TypeError, ValueError):
                combined = 0.0
            multimodal_bonus = 0.05 if bool(item.get("multimodal")) else 0.0
            return combined + multimodal_bonus, combined, float(item.get("start", 0.0))

        return dict(max(pool, key=_rank))

    # No measured peak survived: the editor's own payoff region is the event.
    payoff = clip.get("payoff", {})
    if isinstance(payoff, dict):
        try:
            cuts = teaser_analyzer.normalize_cut_ranges(clip)
            core_start = teaser_analyzer.source_to_edited_time(float(payoff.get("source_start")), cuts)
            core_end = teaser_analyzer.source_to_edited_time(float(payoff.get("source_end")), cuts)
        except (TypeError, ValueError, KeyError):
            core_start = core_end = -1.0
        if core_end > core_start >= restart_floor:
            return {
                "peak_id": 999001,
                "start": round(core_start, 3),
                "end": round(min(clip_duration, core_end), 3),
                "teaser_start": round(core_start, 3),
                "teaser_end": round(min(clip_duration, core_end), 3),
                "audio_score": 0.0,
                "visual_score": 0.0,
                "combined_score": 0.0,
                "multimodal": False,
                "signals": ["editor_payoff_region"],
                "visual_description": "editor-marked payoff region (no measured audio/visual peak)",
            }
    raise ShortsPipelineError(
        "Cold open için ölçülebilir bir peak ya da editör payoff bölgesi yok; tahmini bir an "
        "cold open yapılmaz."
    )


def _write_fallback_teaser(
    teaser_path: Path,
    timeline_path: Path,
    transcript_path: Path,
    timeline_data: dict[str, Any],
    clip_index: int,
    reason: str,
    *,
    edited_video_path: str | Path | None = None,
    video_report_path: str | Path | None = None,
    caption_profile_path: str | Path | None = None,
    caption_path: str | Path | None = None,
) -> Path:
    """Write a REAL cold open (measured peak, evidence-bounded) when the teaser model failed."""
    clip = _timeline_clip(timeline_data, clip_index)
    transcript_data = _load_json(transcript_path)
    try:
        available_words = teaser_analyzer.build_available_words(transcript_data, clip)
    except Exception:
        # A non-verbal physical peak is still a valid cold-open.
        available_words = []
    clip_duration = _clip_edited_duration(
        timeline_data,
        clip_index,
        edited_video_path=edited_video_path,
    )
    if clip_duration <= 0.0:
        raise ShortsPipelineError(
            "Mandatory intro fallback için edited clip süresi bulunamadı."
        )
    caption_profile = None
    if caption_profile_path and Path(caption_profile_path).is_file():
        try:
            caption_profile = _load_json(caption_profile_path)
        except Exception:
            caption_profile = None
    boundary_words = teaser_analyzer.boundary_words_for(available_words, caption_profile)
    envelope = intro_bounds.audio_envelope(edited_video_path)
    restart_floor, _ = intro_renderer.calculate_main_restart_seconds(
        caption_path=caption_path, main_duration=clip_duration,
        protected_ranges=intro_renderer.protected_edited_ranges(clip))

    peak_support = intro_peak_support.build_peak_support(
        edited_video_path=edited_video_path,
        video_report_path=video_report_path,
        clip_duration=clip_duration,
        envelope=envelope,
        words=boundary_words,
    )
    intro_peak_support.attach_nearby_words(peak_support, available_words)
    chosen = _deterministic_peak_candidate(
        clip=clip,
        clip_duration=clip_duration,
        peak_support=peak_support,
        restart_floor=restart_floor,
    )

    candidates = peak_support.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    if not any(
        isinstance(item, dict)
        and int(item.get("peak_id", -1)) == int(chosen.get("peak_id", -2))
        for item in candidates
    ):
        candidates.append(chosen)
        peak_support["candidates"] = candidates

    result = teaser_analyzer.build_teaser_result(
        clip=clip,
        available_words=available_words,
        ai_result={
            "recommended": True,
            "score": 0.0,
            "teaser_type": "reaction",
            "selection_mode": "peak_window",
            "peak_id": int(chosen.get("peak_id", 999001)),
            "peak_alignment_score": 0.0,
            "start_word_id": -1,
            "end_word_id": -1,
            "reason": reason,
            "viewer_question": "How did this happen?",
            "spoiler_risk": "medium",
        },
        peak_support=peak_support,
        boundary_words=boundary_words,
        envelope=envelope,
        media_path=edited_video_path,
    )
    result["recommended"] = True
    result["ai_recommended"] = False
    result["mandatory_cold_open"] = True
    result["reason"] = reason
    warnings = result.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warnings.append("Deterministic cold open: strongest measured peak, evidence-bounded (teaser model unavailable).")
    result["warnings"] = warnings

    return _write_json_atomic(
        teaser_path,
        {
            "version": 1,
            "mode": "mandatory_cold_open_fallback",
            "inputs": {
                "timeline": str(timeline_path),
                "transcript": str(transcript_path),
                "edited_video": str(Path(edited_video_path).resolve()) if edited_video_path else None,
                "video_report": str(Path(video_report_path).resolve()) if video_report_path else None,
            },
            "clip_count": 1,
            "teasers": [result],
        },
    )


def _write_fallback_intro(
    intro_path: Path,
    teaser_path: Path,
    timeline_path: Path,
    transcript_path: Path,
    timeline_data: dict[str, Any],
    clip_index: int,
    reason: str,
) -> Path:
    """Cold open WITHOUT a headline: a strong peak with no text beats generic copy."""
    return _write_json_atomic(
        intro_path,
        {
            "version": 1,
            "mode": "no_headline_intro",
            "inputs": {
                "teaser": str(teaser_path),
                "timeline": str(timeline_path),
                "transcript": str(transcript_path),
            },
            "clip_count": 1,
            "intros": [
                {
                    "clip_index": clip_index,
                    "title": _selected_title(timeline_data, clip_index),
                    "recommended": True,
                    "ai_recommended": False,
                    "headline_status": "none",
                    "score": 0.0,
                    "intro_text": "",
                    "reason": reason,
                    "quality_gate": {
                        "threshold": MIN_INTRO_ACCEPT_SCORE,
                        "accepted": True,
                        "no_headline": True,
                        "rejection_reason": reason,
                    },
                    "intro": {"duration": 0.0, "word_count": 0, "character_count": 0},
                    "background": {"source": "edited_clip", "freeze_frame_time": 0.0, "style": "moving_teaser"},
                    "sequence": ["moving_peak_without_headline", "hard_main_clip_restart"],
                    "render_plan": {
                        "intro_background": "moving_teaser",
                        "then_play": "main_clip_restart",
                        "restart_main_clip_after_teaser": True,
                        "transition": "hard_cut",
                    },
                }
            ],
        },
    )


def _write_fallback_meme_slots(
    slot_path: Path,
    intro_path: Path,
    teaser_path: Path,
    timeline_path: Path,
    transcript_path: Path,
    timeline_data: dict[str, Any],
    clip_index: int,
    reason: str,
) -> Path:
    return _write_json_atomic(
        slot_path,
        {
            "version": meme_analyzer.MEME_ANALYZER_VERSION,
            "mode": "pipeline_fallback_no_meme",
            "inputs": {
                "intro": str(intro_path),
                "teaser": str(teaser_path),
                "timeline": str(timeline_path),
                "transcript": str(transcript_path),
            },
            "clip_count": 1,
            "clips": [
                {
                    "clip_index": clip_index,
                    "title": _selected_title(timeline_data, clip_index),
                    "recommended": False,
                    "main_clip_duration": 0.0,
                    "opening_teaser_duration": 0.0,
                    "max_slots_allowed": 1,
                    "slots": [],
                    "ai_summary": {"use_meme": False, "decision_reason": reason},
                    "quality_gate": {"single_meme_only": True, "rejections": []},
                }
            ],
        },
    )


def _write_fallback_discovery(
    discovery_path: Path,
    slot_path: Path,
    timeline_data: dict[str, Any],
    clip_index: int,
    reason: str,
) -> Path:
    return _write_json_atomic(
        discovery_path,
        {
            "version": meme_discovery.DISCOVERY_VERSION,
            "mode": "pipeline_fallback_no_discovery",
            "created_at": _now(),
            "inputs": {"meme_slots": str(slot_path)},
            "clip_count": 1,
            "clips": [
                {
                    "clip_index": clip_index,
                    "title": _selected_title(timeline_data, clip_index),
                    "searched": False,
                    "selected_asset_count": 0,
                    "discoveries": [],
                    "reason": reason,
                }
            ],
        },
    )


# ============================================================
# MEDIA ARTIFACT HELPERS
# ============================================================

def _is_fresh_copy(output: str | Path, dependency: str | Path) -> bool:
    """Return True when a derived copy is at least as new as its source."""
    output_path = Path(output)
    dependency_path = Path(dependency)
    if not output_path.is_file() or not dependency_path.is_file():
        return False
    try:
        return output_path.stat().st_mtime_ns >= dependency_path.stat().st_mtime_ns
    except OSError:
        return False


def _fallback_final_preview_path(
    timeline_data: dict[str, Any],
    clip_index: int,
) -> Path:
    video_stem = intro_renderer.get_video_stem(timeline_data)
    directory = Path(intro_renderer.FINAL_PREVIEWS_DIR) / video_stem
    directory.mkdir(parents=True, exist_ok=True)

    renderer_version = int(getattr(intro_renderer, "RENDERER_VERSION", 6))
    return (
        directory
        / f"clip_{clip_index:02d}_fallback_final_preview_v{renderer_version}.mp4"
    ).resolve()


def _prepare_fallback_final_preview(
    captioned_preview: Path,
    timeline_data: dict[str, Any],
    clip_index: int,
    *,
    force: bool,
) -> Path:
    output = _fallback_final_preview_path(timeline_data, clip_index)

    if (
        force
        or not _valid_file(output, MIN_VIDEO_BYTES)
        or not _is_fresh_copy(output, captioned_preview)
    ):
        temp = output.with_suffix(output.suffix + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(captioned_preview, temp)
        os.replace(temp, output)

    os.utime(output, None)
    return output


def _activate_final_preview(path: str | Path) -> Path:
    path = Path(path).resolve()
    if not _valid_file(path, MIN_VIDEO_BYTES):
        raise ShortsPipelineError("Final base preview geçersiz.")
    os.utime(path, None)
    return path


PACED_CLIP_TOLERANCE_S = 0.25   # same floor as Pro Edit's timeline/media check


def _resolve_edited_clip(expected_path: Path) -> Path | None:
    """The paced clip of THIS timeline clip: only the exact file pacing_cutter writes for it.

    Never searched by pattern: after the temp cleanup removed this clip, a glob
    over edited_clips/ returned another source's (or an older selection's)
    clip_01 video, and every downstream stage consumed that foreign media."""
    if _valid_file(expected_path, MIN_VIDEO_BYTES):
        return expected_path.resolve()
    return None


def _declared_paced_duration(clip_timeline: dict[str, Any]) -> float | None:
    """Paced duration the timeline declares (source duration minus its persisted final cuts)."""
    try:
        source = clip_timeline.get("source") or {}
        keep = pacing_cutter.build_keep_ranges(float(source.get("duration", 0.0)),
                                               pacing_cutter.normalize_cut_ranges(clip_timeline))
        return float(pacing_cutter.calculate_expected_duration(keep))
    except Exception:
        return None


def _paced_clip_matches(path: Path | None, clip_timeline: dict[str, Any]) -> bool:
    """The media is the paced clip the timeline describes (the clock all consumers assume)."""
    expected = _declared_paced_duration(clip_timeline)
    actual = _probe_video_duration(path) if path is not None else 0.0
    return expected is not None and actual > 0.0 and abs(actual - expected) <= PACED_CLIP_TOLERANCE_S


def _media_identity(path: Path) -> dict[str, Any]:
    """Content identity (size + SHA-256) and file clock of a rendered media file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()}


def _keep_render_clock(path: Path, previous: dict[str, Any], signature: str) -> dict[str, Any]:
    """A re-rendered paced clip keeps the file clock downstream caches recorded when its bytes are identical.

    The temp cleanup deletes the paced clip after a successful run, so a rerun
    (e.g. --force-v6) re-renders it. Downstream signatures fingerprint it by
    path + size + mtime: a byte-identical re-render gets the recorded mtime
    back instead of re-billing captions / visual support / intro analysis. Any
    byte difference keeps the new clock, so every dependent stage re-runs."""
    identity = _media_identity(path)
    media = previous.get("media") if isinstance(previous.get("media"), dict) else None
    if (media is not None and previous.get("signature") == signature
            and Path(str(previous.get("path", ""))).resolve() == path.resolve()
            and media.get("size") == identity["size"] and media.get("sha256") == identity["sha256"]):
        os.utime(path, ns=(path.stat().st_atime_ns, int(media["mtime_ns"])))
        identity["mtime_ns"] = int(media["mtime_ns"])
        identity["clock"] = "restored_identical_render"
    return identity


def _meme_slots(slot_path: Path, clip_index: int) -> list[dict[str, Any]]:
    clip = _package_clip(slot_path, "clips", clip_index)
    if not isinstance(clip, dict):
        return []

    slots = clip.get("slots", [])
    if not isinstance(slots, list):
        return []

    return [item for item in slots if isinstance(item, dict)]


def _selected_meme_count(discovery_path: Path, clip_index: int) -> int:
    clip = _package_clip(discovery_path, "clips", clip_index)
    if not isinstance(clip, dict):
        return 0

    try:
        explicit = int(clip.get("selected_asset_count", 0))
    except (TypeError, ValueError):
        explicit = 0

    if explicit > 0:
        return explicit

    discoveries = clip.get("discoveries", [])
    if not isinstance(discoveries, list):
        return 0

    return sum(
        1
        for item in discoveries
        if isinstance(item, dict) and item.get("selected") is True
    )


# ============================================================
# VIDEO BRAIN SUPPORT
# ============================================================

def _video_brain_config() -> tuple[bool, str]:
    try:
        from ai.video_brain.config import VIDEO_BRAIN_ENABLED, VIDEO_BRAIN_MODEL

        return bool(VIDEO_BRAIN_ENABLED), str(VIDEO_BRAIN_MODEL)
    except Exception:
        return False, ""


def _video_brain_code_info() -> dict[str, Any]:
    try:
        from ai.video_brain import video_analyzer

        return {
            "available": True,
            "module": _module_fingerprint(video_analyzer),
            "analyzer_version": int(getattr(video_analyzer, "VIDEO_ANALYZER_VERSION", 0)),
            "schema_version": int(getattr(video_analyzer, "REPORT_SCHEMA_VERSION", 0)),
        }
    except Exception as error:
        return {"available": False, "error": type(error).__name__}


def _run_video_brain(
    edited_clip_path: Path,
    *,
    force: bool,
) -> tuple[dict[str, Any] | None, Path | None]:
    from ai.video_brain import video_analyzer

    # Let the real exception propagate to the caller. The whole-VOD caller is
    # fail-closed, while the selected-clip caller already catches errors and
    # continues fail-open. This preserves the correct policy AND exposes the
    # actual Gemini/Terra fallback error instead of the vague "report üretmedi".
    report = video_analyzer.analyze_video(edited_clip_path, force=force)

    try:
        report_path = Path(video_analyzer.get_output_path(edited_clip_path)).resolve()
        if not _valid_json(report_path):
            raise RuntimeError(
                f"Video Brain analiz döndürdü ama report dosyası geçersiz: {report_path}"
            )
    except Exception:
        raise

    return report, report_path


# ============================================================
# PUBLISH / CLEANUP
# ============================================================

def _publish_final(source: Path, video_path: Path) -> Path:
    if not _valid_file(source, MIN_VIDEO_BYTES):
        raise ShortsPipelineError(f"Publish source geçersiz:\n{source}")

    PUBLISHED_DIR.mkdir(parents=True, exist_ok=True)
    output = (PUBLISHED_DIR / f"{_safe_name(video_path.stem)}_short.mp4").resolve()

    if source.resolve() == output:
        return output

    temp = output.with_suffix(output.suffix + ".tmp")
    try:
        temp.unlink(missing_ok=True)
    except OSError:
        pass

    try:
        os.link(source, temp)
    except Exception:
        shutil.copy2(source, temp)

    os.replace(temp, output)
    return output


def _reject_candidate(candidate: Path, video_path: Path, gate: dict[str, Any], state: dict[str, Any]) -> Path:
    """Keep a rejected candidate for inspection OUTSIDE vod_output/final, with its QC report."""
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    target = (REJECTED_DIR / f"{_safe_name(video_path.stem)}_short_REJECTED.mp4").resolve()
    if _valid_file(candidate, MIN_VIDEO_BYTES) and candidate.resolve() != target:
        temp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(candidate, temp)
        os.replace(temp, target)
    _write_json_atomic(target.with_suffix(".qc.json"), {
        "status": "rejected", "source": str(video_path), "candidate": str(candidate),
        "failed": gate.get("failed"), "warnings": gate.get("warnings"), "gate": gate,
        "final_qc": state.get("final_qc"), "at": _now(),
    })
    return target


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _cleanup_temp_files(
    final_output: Path,
    candidates: Iterable[str | Path | None],
) -> list[str]:
    removable_suffixes = {
        ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".ts",
        ".mp3", ".wav", ".m4a", ".aac",
    }

    removed: list[str] = []
    seen: set[str] = set()

    for raw in candidates:
        if raw is None:
            continue

        path = Path(raw).resolve()
        key = str(path).casefold()

        if key in seen:
            continue
        seen.add(key)

        if path == final_output.resolve():
            continue
        if not _inside(path, VOD_OUTPUT_DIR):
            continue
        if not path.is_file():
            continue
        if path.suffix.casefold() not in removable_suffixes:
            continue

        try:
            path.unlink()
            removed.append(str(path))
        except OSError:
            pass

    return removed


# ============================================================
# FAST RESUME
# ============================================================

def _code_signature(video_brain_enabled: bool, video_brain_model: str) -> str:
    modules = [
        model_config,
        vod_processor,
        clip_analyzer,
        pacing,
        timeline,
        captions,
        participant_name_lock,
        pacing_cutter,
        caption_renderer,
        caption_truth,
        teaser_analyzer,
        intro_peak_support,
        intro_analyzer,
        intro_renderer,
        meme_analyzer,
        meme_discovery,
        meme_renderer,
        final_qc,
        final_review,
        v6_runtime,
    ]

    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "modules": [_module_fingerprint(module) for module in modules],
        "video_brain_enabled": video_brain_enabled,
        "video_brain_model": video_brain_model if video_brain_enabled else "",
        "model_plan": model_config.model_plan(),
    }

    if video_brain_enabled:
        payload["video_brain"] = _video_brain_code_info()

    return _hash(payload)


def _request_signature(
    source: dict[str, Any],
    *,
    creator_name: str | None,
    clip_index: int | None,
    enable_memes: bool,
    enable_video_brain: bool,
    video_brain_model: str,
    presentation_signature: str,
) -> str:
    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "source": source,
        "creator_name": creator_name or "",
        "clip_index": clip_index,
        "enable_memes": enable_memes,
        "enable_video_brain": enable_video_brain,
        "video_brain_model": video_brain_model if enable_video_brain else "",
        "code_signature": _code_signature(enable_video_brain, video_brain_model),
    }
    payload["presentation"] = presentation_signature
    return _hash(payload)


def _fast_resume(
    state: dict[str, Any] | None,
    request_signature: str,
    state_path: Path,
) -> dict[str, Any] | None:
    if not isinstance(state, dict):
        return None
    if state.get("run_status") != "success":
        return None
    if state.get("request_signature") != request_signature:
        return None

    final_raw = state.get("final_output")
    if not final_raw or not _valid_file(final_raw, MIN_VIDEO_BYTES):
        return None

    selected = state.get("selected_clip", {})
    if not isinstance(selected, dict):
        selected = {}

    source = state.get("source", {})
    source_video = source.get("path", "") if isinstance(source, dict) else ""

    return {
        "success": True,
        "pipeline_version": PIPELINE_VERSION,
        "source_video": str(source_video),
        "selected_clip": selected,
        "final_output": str(Path(final_raw).resolve()),
        "state_file": str(state_path),
        "video_brain_source_report": state.get("video_brain_source_report"),
        "video_brain_report": state.get("video_brain_report"),
        "warnings": list(state.get("warnings", [])),
        "cleanup_removed": [],
        "run_seconds": 0.0,
        "stats": state.get("summary_stats", {}),
        "profile": state.get("profile", {}),
        "fast_resume": True,
        "status": str(state.get("publish_status", "published")),
        **({"v6": state["v6"].get("summary", {})} if isinstance(state.get("v6"), dict) else {}),
    }


# ============================================================
# MAIN
# ============================================================

def _verify_pro_edit_intro(
    pro_edit_pkg: Any,
    prep: Any,
    *,
    book: "StageBook",
    state: dict[str, Any],
    teaser_path: Path,
    intro_path: Path,
    caption_path: Path,
    clean_clip: Path,
    intro_source: Path,
    main_preview: Path,
    final_preview: Path,
    clip_index: int,
) -> Path:
    """Intro/handoff integrity after composition when Pro Edit is active.

    Re-derives the intro clock from the files actually used and compares it to
    the clock captured before Pro Edit (teaser identity, source range, xfade,
    main restart, main duration) and to the rendered final duration. If the
    Pro Edit intro source is implicated, the intro is re-rendered from the
    clean clip (existing MIMIR intro); otherwise the problem is reported.
    """
    timeline_mod = pro_edit_pkg.intro_timeline
    media_mod = pro_edit_pkg.media
    report: dict[str, Any] = {}
    try:
        teaser_record = _package_clip(teaser_path, "teasers", clip_index) or {}
        clean_info = media_mod.probe_media(clean_clip)
        main_info = media_mod.probe_media(main_preview)
        clip_timeline = _timeline_clip(_load_json(prep.timeline_path), clip_index) if prep.timeline_path else None
        after = timeline_mod.build_intro_timeline(
            teaser_record, caption_path=caption_path, clean_duration=media_mod.probe_media(intro_source).duration_s,
            main_duration=main_info.duration_s, clip_timeline=clip_timeline, fps=clean_info.fps.fps)
        frame = clean_info.fps.frame_duration
        timeline_mod.verify_handoff(prep.intro_timeline, after, frame_duration=frame)
        final_info = media_mod.probe_media(final_preview)
        report["final_duration_delta_s"] = round(
            timeline_mod.verify_final_duration(after, final_info.duration_s, frame_duration=frame), 4)
        report["status"] = "verified"
        report["timeline"] = after.to_dict()
    except Exception as error:  # report, then fall back to the clean intro source if it was replaced
        report["status"] = "failed"
        report["reason"] = f"{type(error).__name__}: {error}"
        book.warn(f"Pro Edit intro/handoff doğrulaması başarısız: {report['reason']}")
        if Path(intro_source).resolve() != Path(clean_clip).resolve():
            try:
                rendered = intro_renderer.run_renderer(
                    intro_json_path=intro_path,
                    clip_index=clip_index,
                    edited_clip_path=clean_clip,
                    captioned_preview_path=main_preview,
                    caption_path=caption_path,
                )
                final_preview = _activate_final_preview(rendered[0])
                report["recovery"] = "intro re-rendered from the clean paced clip"
            except Exception as recovery_error:
                report["recovery"] = f"failed: {type(recovery_error).__name__}: {recovery_error}"
    if isinstance(state.get("pro_edit"), dict):
        state["pro_edit"]["intro_verification"] = report
    book.save()
    return final_preview


def _run_caption_truth_v6(
    *,
    book: "StageBook",
    run: Any,
    truth_mod: Any,
    profile_path: Path | None,
    caption_path: Path,
    transcript_path: Path,
    timeline_path: Path,
    edited_clip_path: Path,
    clip_index: int,
    creator_name: str | None,
    entities: list[str],
) -> tuple[Path, Path | None]:
    """Stage 7b under V6: resolve -> targeted escalation -> FREEZE caption truth.

    Lexical/timing/speaker/identity authorities stay separate: only verified
    entity spelling may change (word id/time/speaker kept), the caption ASS is
    regenerated from the frozen truth, and every degradation is recorded as an
    explicit V6 fallback. Never re-transcribes the clip."""
    started = time.perf_counter()
    if profile_path is None:
        reason = "final speaker profile not found"
        run.fallback("caption_truth", reason, "unfrozen_asr_captions")
        run.record("caption_truth", "fallback", detail=reason)
        book.warn(f"Caption Truth V6 çalışmadı; ASR caption kullanıldı. Detay: {reason}")
        book.record("caption_truth_v6", "fallback", None, note=reason, elapsed=time.perf_counter() - started)
        return caption_path, None
    truth_path = Path(truth_mod.truth_path_for(profile_path)).resolve()

    def signature() -> str:
        return _stage_signature(
            "caption_truth_v6",
            inputs=[profile_path, caption_path, timeline_path, transcript_path, edited_clip_path],
            modules=[truth_mod, participant_name_lock, captions],
            options={
                "clip_index": clip_index,
                "version": int(getattr(truth_mod, "CAPTION_TRUTH_VERSION", 1)),
                "name_lock_version": int(getattr(participant_name_lock, "NAME_LOCK_VERSION", 1)),
                "creator": creator_name or "",
                "entities": list(entities),
                "max_escalations": int(getattr(truth_mod, "MAX_ESCALATIONS", 0)),
            },
        )

    def reusable() -> bool:
        return (_valid_file(caption_path, MIN_CAPTION_BYTES) and _valid_json(profile_path)
                and truth_path.is_file() and bool(truth_mod.verify_frozen_truth(profile_path, truth_path)[0]))

    if book.reusable("caption_truth_v6", signature(), reusable, output=truth_path):
        _stage_skip(truth_path)
        document = _load_json(truth_path)
        lock = document.get("entity_lock", {}) if isinstance(document.get("entity_lock"), dict) else {}
        run.record("caption_truth", "frozen", cached=True, corrections=len(lock.get("corrections") or []),
                   uncertain=len(document.get("uncertain") or []), timing_issues=len(document.get("timing_issues") or []),
                   speaker_issues=len(document.get("speaker_issues") or []))
        for row in document.get("fallbacks") or []:
            if isinstance(row, dict):
                run.fallback(str(row.get("subsystem", "caption_truth")), str(row.get("reason", "")),
                             str(row.get("level", "text_evidence_only")))
        book.record("caption_truth_v6", "skipped", signature(), path=truth_path,
                    elapsed=time.perf_counter() - started)
        return caption_path, truth_path
    try:
        result = truth_mod.run_caption_truth(profile_path, edited_clip_path=edited_clip_path,
                                             creator_name=creator_name, entities=entities)
        if result.changed_text:
            caption_path = Path(
                captions.create_clip_captions(
                    transcript_path=transcript_path,
                    timeline_path=timeline_path,
                    clip_index=clip_index,
                    speaker_profile_path=profile_path,
                )
            ).resolve()
        audit = result.audit
        for row in audit.get("corrections") or []:
            print(f"🔤 Caption truth: {row.get('from')} → {row.get('to')} "
                  f"({row.get('entity_kind', 'entity')} {row.get('canonical')}; {', '.join(row.get('evidence') or [])})")
        for row in result.fallbacks:
            run.fallback(str(row.get("subsystem", "caption_truth")), str(row.get("reason", "")),
                         str(row.get("level", "text_evidence_only")))
        run.record("caption_truth", "frozen", corrections=len(audit.get("corrections") or []),
                   escalations=audit.get("escalations"), escalation_calls=audit.get("escalation_calls"),
                   uncertain_words=audit.get("uncertain_words"), timing_issues=audit.get("timing_issues"),
                   speaker_issues=audit.get("speaker_issues"), signature=str(audit.get("signature", ""))[:16])
        print(f"🔒 Caption truth frozen: {len(audit.get('corrections') or [])} canonicalization(s), "
              f"{audit.get('escalations', 0)} targeted escalation(s), {audit.get('uncertain_words', 0)} uncertain word(s)")
        _stage_done(truth_path)
        book.record("caption_truth_v6", "done", signature(), path=truth_path,
                    note=f"{len(audit.get('corrections') or [])} correction(s); signature={str(audit.get('signature'))[:12]}",
                    elapsed=time.perf_counter() - started)
        return caption_path, truth_path
    except Exception as error:  # fail closed: ASR spelling and the existing ASS stay in use, loudly
        reason = f"{type(error).__name__}: {error}"
        run.fallback("caption_truth", reason, "unfrozen_asr_captions")
        run.record("caption_truth", "fallback", detail=reason)
        book.warn(f"Caption Truth V6 başarısız; ASR yazımı korundu (V6 kapısı başarısız sayılacak). Detay: {reason}")
        book.record("caption_truth_v6", "fallback", None, note=reason, elapsed=time.perf_counter() - started)
        return caption_path, None


def _verified_names(profile_path: Path | None, creator_name: str | None, entities: Iterable[str]) -> list[str]:
    """Names a human confirmed: creator, human-confirmed speakers, user-verified entities."""
    names: list[str] = []
    if creator_name:
        names.append(creator_name)
    if profile_path is not None:
        try:
            from ai.editor.pro_edit.caption_guard import load_profile, trusted_display_names

            names.extend(trusted_display_names(load_profile(profile_path)).values())
        except Exception:
            pass
    names.extend(entities)
    result: list[str] = []
    for name in names:
        value = " ".join(str(name).split())
        if value and value.casefold() not in {n.casefold() for n in result}:
            result.append(value)
    return result


def _truth_text(truth_path: Path | None) -> str:
    """The frozen caption words as plain text (headline grounding evidence)."""
    if truth_path is None or not Path(truth_path).is_file():
        return ""
    try:
        rows = _load_json(truth_path).get("words", [])
    except Exception:
        return ""
    return " ".join(str(row[1]) for row in rows if isinstance(row, list) and len(row) > 1)


def _final_caption_bands(burned_ass: Path | None) -> list[tuple[float, float]]:
    """Vertical bands (normalized) where the burned speech captions live."""
    if burned_ass is None or not Path(burned_ass).is_file():
        return []
    from ai.editor.pro_edit.caption_guard import caption_safe_region

    return sorted({(round(b.y0, 4), round(b.y1, 4)) for b in caption_safe_region(burned_ass).bands})


def _record_pro_edit_v6(run: Any, prep: Any) -> None:
    """Expose every V6-relevant Pro Edit degradation as an explicit fallback."""
    from ai.editor.pro_edit.vision.cv_runtime import opencv_status

    # Root cause first: without OpenCV there are no faces, no caption activity /
    # background evidence and no pixel proof (e.g. Smart App Control refusing cv2.pyd).
    vision = opencv_status()
    run.record("vision_runtime", vision["status"], detail=vision["reason"])
    if vision["status"] != "ok":
        run.fallback("vision_runtime", vision["reason"], "no_local_vision")
    run.record("pro_edit", str(prep.status), detail=str(prep.reason)[:300])
    if prep.status != "ready":
        run.fallback("pro_edit", prep.reason or prep.status, "baseline_caption_render")
    direction = dict(getattr(prep, "direction", {}) or {})
    run.record("camera_direction", str(direction.get("status", "not_run")), intents=direction.get("intents"),
               holds=direction.get("holds"), camera_ops=direction.get("camera_ops"),
               subjects=direction.get("subjects"), linked_subjects=direction.get("linked_subjects"),
               layout=direction.get("layout"), detail=direction.get("reason"))
    if direction.get("status") != "directed":
        run.fallback("camera_direction", str(direction.get("reason", "V6 direction not applied")),
                     "validated_planner_intent")
    tracking = dict(getattr(prep, "tracking", {}) or {})
    tracking_status = str(tracking.get("status", tracking.get("provider", "unknown")))
    run.record("face_tracking", tracking_status, detector=tracking.get("detector"), tracks=tracking.get("tracks"),
               speaker_links=tracking.get("speaker_links"), detail=tracking.get("reason"))
    if tracking_status in {"unavailable", "failed"}:
        run.fallback("face_tracking", str(tracking.get("reason", tracking_status)), "center_safe_framing")
    caption_info = dict(getattr(prep, "captions", {}) or {})
    run.record("caption_presentation", str(caption_info.get("status", "not_run")), level=caption_info.get("level"),
               pages=caption_info.get("pages"), legibility=caption_info.get("legibility"))
    if caption_info.get("status") != "presentation":
        run.fallback("caption_presentation", str(caption_info.get("reason", caption_info.get("status"))),
                     "baseline_ass")
    elif caption_info.get("level") not in (None, "v5_placement"):
        run.fallback("caption_placement", f"evidence placement degraded to {caption_info.get('level')}",
                     str(caption_info.get("level")))
    for evidence in ("activity", "background"):
        if str(caption_info.get(evidence, "")) == "failed":
            run.fallback(f"caption_evidence_{evidence}", f"{evidence} analysis failed"
                         + ("" if vision["status"] == "ok" else " (OpenCV unavailable, see vision_runtime)"),
                         f"no_{evidence}_evidence")
    energy = dict(getattr(prep, "energy", {}) or {})
    run.record("editorial_energy", str(energy.get("status", "not_run")))
    if energy.get("status") == "failed":
        run.fallback("editorial_energy", str(energy.get("reason", "")), "uncoordinated_channels")


def _finish_v6(
    *,
    v6_mod: Any,
    run: Any,
    pro_edit_pkg: Any,
    prep: Any,
    book: "StageBook",
    state: dict[str, Any],
    published_path: Path,
    main_render: Path,
    main_proof: dict[str, Any] | None,
    intro_source: Path | None,
    intro_proof: dict[str, Any] | None,
    burned_ass: Path | None,
    render_status: str,
    profile_path: Path | None,
    truth_path: Path | None,
    artifact_dir: Path,
    clip_index: int,
    qc_rows: list[dict[str, Any]] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Final pixel proof + deterministic quality gate + compact manifest (runs BEFORE publishing).

    ``published_path`` is the CANDIDATE that would be published."""
    final_proof = None
    story = (False, "Pro Edit was not prepared")
    labels: list[str] = []
    if prep is not None and pro_edit_pkg is not None:
        if main_proof is not None:
            final_proof = v6_mod.untraceable_final_proof(main_proof) or pro_edit_pkg.stage.prove_final_output(
                prep, final_path=published_path, main_render=main_render, main_proof=main_proof,
                intro_source=intro_source, intro_proof=intro_proof, burned_ass=burned_ass)
        try:
            pro_edit_pkg.stage.verify_truth(prep, "final gate")
            story = (True, "timeline story + caption truth unchanged since planning")
        except Exception as error:
            story = (False, f"{type(error).__name__}: {error}")
        from ai.editor.pro_edit.caption_guard import load_profile, trusted_display_names

        labels = list(trusted_display_names(load_profile(profile_path)).values())
    for name, proof in (("camera_pixels_main", main_proof), ("camera_pixels_intro", intro_proof),
                        ("camera_pixels_final", final_proof)):
        if proof is not None:
            run.record(name, str(proof.get("status")), detail=str(proof.get("reason", ""))[:240])
    gate = v6_mod.final_quality_gate(
        run=run, final_output=published_path, profile_path=profile_path, truth_path=truth_path,
        burned_ass=burned_ass, prep=prep, main_proof=main_proof, final_proof=final_proof, intro_proof=intro_proof,
        story_intact=story, intro_handoff=(state.get("pro_edit") or {}).get("intro_verification"),
        render_status=render_status, display_labels=labels, qc_rows=qc_rows or (), review=review)
    run.gate = gate
    artifacts = getattr(prep, "artifacts", None)
    proof_path = getattr(artifacts, "render_proof", None)
    if proof_path is not None:
        from ai.editor.pro_edit.diagnostics import write_json_atomic

        write_json_atomic(proof_path, {"main": main_proof, "intro": intro_proof, "final": final_proof})
        run.artifact("render_proof", proof_path)
    camera_plan = getattr(artifacts, "camera_plan", None)
    if camera_plan is not None and Path(camera_plan).is_file():
        run.artifact("camera_plan", camera_plan)
    run.artifact("caption_truth", truth_path)
    run.artifact("final_output", published_path)
    manifest = getattr(artifacts, "v6_manifest", None) or (artifact_dir / f"clip_{clip_index:02d}_v6_manifest.json")
    run.artifact("manifest", manifest)
    v6_mod.write_manifest(manifest, run)
    summary = {**run.summary(), "manifest": str(manifest),
               "fallback_rows": [{"subsystem": r["subsystem"], "level": r["level"], "reason": r["reason"][:300]}
                                 for r in run.fallbacks]}
    state["v6"] = {"summary": summary, "gate": {k: gate.get(k) for k in ("status", "failed", "warnings")}}
    for line in run.console_lines():
        print(line)
    if gate.get("status") == v6_mod.GATE_FAILED:
        book.warn("Final kalite kapısı başarısız: " + ", ".join(gate.get("failed") or []))
    elif gate.get("warnings"):
        book.warn("Final kalite kapısı DEGRADED uyarılarla geçti: " + ", ".join(gate.get("warnings") or []))
    book.save()
    return summary


def run_pipeline(
    video_path: str | Path,
    *,
    creator_name: str | None = None,
    force: bool = False,
    clip_index: int | None = None,
    enable_memes: bool = True,
    enable_video_brain: bool | None = None,
    keep_temp: bool = False,
    rerender: bool = False,
    force_v6: bool = False,
) -> dict[str, Any]:
    """The one production path: VOD -> story -> truth-locked captions -> cold
    open -> directed presentation -> effects -> rendered-MP4 QC -> publish.

    ``rerender`` (legacy alias ``force_v6``) recomputes only the presentation,
    render, QC and publish stages; transcription, story discovery, caption ASR
    and the other paid upstream stages keep their caches."""
    rerender = bool(rerender or force_v6)
    run_started = time.perf_counter()
    runtime_profiler = RuntimeProfiler()
    video_path = resolve_video_path(video_path)
    creator_name = str(creator_name).strip() if creator_name else None

    config_vb, video_brain_model = _video_brain_config()
    resolved_vb = config_vb if enable_video_brain is None else bool(enable_video_brain)

    source = _source_fingerprint(video_path)
    state_path = _state_path(video_path)
    previous = _load_state(state_path)
    same_source = _same_source(previous, source)

    v6_mod: Any = v6_runtime
    caption_truth_mod: Any = caption_truth
    v6_run: Any = v6_runtime.V6Run(enabled=True, force=rerender)
    pro_edit_pkg, pro_edit_cfg = _load_pro_edit()
    v6_entities = v6_runtime.caption_entities_from_env()
    presentation_sig = _hash(
        {
            "version": int(getattr(pro_edit_pkg, "PRO_EDIT_VERSION", 1)),
            "config": pro_edit_cfg.signature_payload(),
            "entities": v6_entities,
            "modules": [
                _module_fingerprint(module)
                for module in (*pro_edit_pkg.MODULES, caption_truth, v6_runtime, participant_name_lock,
                               *getattr(pro_edit_pkg, "VERIFY_MODULES", ()))
            ],
        }
    )

    request_sig = _request_signature(
        source,
        creator_name=creator_name,
        clip_index=clip_index,
        enable_memes=bool(enable_memes),
        enable_video_brain=resolved_vb,
        video_brain_model=video_brain_model,
        presentation_signature=presentation_sig,
    )

    if not force and not rerender and same_source:
        resumed = _fast_resume(previous, request_sig, state_path)
        if resumed is not None:
            print(f"\n⚡ MIMIR fast-resume (hazır, doğrulanmış çıktı)\n📂 {resumed['final_output']}")
            return resumed

    source_changed = previous is not None and not same_source
    effective_force = bool(force or source_changed)

    state = previous if same_source and previous is not None else _new_state(source)
    state["source"] = source
    state["request_signature"] = request_sig
    state["options"] = {
        "creator_name": creator_name,
        "clip_index": clip_index,
        "enable_memes": bool(enable_memes),
        "enable_video_brain": resolved_vb,
        "keep_temp": bool(keep_temp),
        "rerender": rerender,
    }
    state.pop("pro_edit", None)
    state.pop("v6", None)
    state.pop("final_qc", None)
    state.pop("publish_status", None)
    state["warnings"] = []
    state["video_brain_source_report"] = None
    state["video_brain_report"] = None
    state["final_output"] = None
    state["run_status"] = "running"

    book = StageBook(
        state_path,
        state,
        effective_force,
        rerun=v6_runtime.FORCE_V6_STAGES if rerender else (),
    )
    book.save()

    _print_header(
        video_path,
        force=effective_force,
        memes=bool(enable_memes),
        video_brain=resolved_vb,
        keep_temp=bool(keep_temp),
        pro_edit=(
            f"planner={pro_edit_cfg.planner}, style={pro_edit_cfg.style}, "
            f"profile={pro_edit_cfg.output_profile.value}"
        ),
        v6="rerender: sunum/render/QC yeniden" if rerender else "caption truth + kamera + piksel QC + final gate",
    )

    temp_candidates: list[Path | None] = []

    # --------------------------------------------------------
    # V23 SOURCE VISUAL FACTS — PARALLEL WITH TRANSCRIPTION
    # --------------------------------------------------------
    # V9 correctly removed whole-VOD visual work from the critical path for
    # speed, but that made clip discovery blind to silent visual-only payoffs
    # that happen OUTSIDE whatever transcript moment was already selected
    # (object break/destruction/fall/etc.). V18 can recover visual setup only
    # around an ALREADY-selected moment; it cannot discover a different later
    # payoff. V23 restores factual source-video observation, but starts the
    # existing Video Brain job at the same time as transcription so we pay as
    # little extra wall-clock latency as possible. It remains support-only:
    # Terra still owns every editorial decision.
    source_visual_report_path: Path | None = None
    source_vb_sig: str | None = None
    source_vb_expected_report: Path | None = None
    source_vb_cached = False
    source_vb_prepare_error: Exception | None = None
    source_vb_executor: ThreadPoolExecutor | None = None
    source_vb_future = None

    if resolved_vb:
        try:
            from ai.video_brain import video_analyzer

            source_vb_expected_report = Path(
                video_analyzer.get_output_path(video_path)
            ).resolve()
            source_vb_sig = _hash(
                {
                    "stage": "source_visual_facts_v23",
                    "source": _file_fingerprint(video_path),
                    "model": video_brain_model,
                    "fallback_model": model_config.VISUAL_SUPPORT_MODEL,
                    "fallback_reasoning": model_config.VISUAL_SUPPORT_REASONING_EFFORT,
                    "analyzer": _video_brain_code_info(),
                }
            )
            source_vb_cached = book.reusable(
                "visual_fact_observer",
                source_vb_sig,
                lambda: _valid_json(source_vb_expected_report),
                output=source_vb_expected_report,
                freshness=[video_path],
            )
            if source_vb_cached:
                source_visual_report_path = source_vb_expected_report
            elif SAFE_PIPELINE_PARALLEL:
                source_vb_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="mimir-source-visual",
                )
                source_vb_future = source_vb_executor.submit(
                    runtime_profiler.timed,
                    "source_visual_worker",
                    _run_video_brain,
                    video_path,
                    force=effective_force,
                )
        except Exception as error:
            source_vb_prepare_error = error

    # --------------------------------------------------------
    # 1. TRANSCRIPTION
    # --------------------------------------------------------

    transcript_path = _transcript_path(video_path)
    transcript_sig = _stage_signature(
        "transcription",
        inputs=[video_path],
        modules=[vod_processor],
        options={
            "vod_text_model": str(getattr(vod_processor, "ACCURATE_MODEL", "gpt-4o-mini-transcribe")),
            "timing_model": str(getattr(vod_processor, "TIMING_MODEL", "whisper-1")),
        },
    )
    try:
        _run_simple_stage(
            book=book,
            number=1,
            title="Transcription",
            key="transcription",
            signature=transcript_sig,
            output=transcript_path,
            validator=lambda: _valid_json(transcript_path, 2),
            runner=lambda: vod_processor.process_vod(video_path),
            freshness=[video_path, *_module_paths([vod_processor])],
            error_message="Transcription çıktı JSON'u oluşmadı/geçersiz.",
        )
    except Exception:
        # Do not leave a background visual worker alive if transcription itself
        # aborts the run. This changes no successful-run output and avoids a
        # pointless wait on a failed pipeline.
        if source_vb_future is not None:
            source_vb_future.cancel()
        if source_vb_executor is not None:
            source_vb_executor.shutdown(wait=False, cancel_futures=True)
        raise

    try:
        transcript_data = _load_json(transcript_path)
        source_data = transcript_data.get("source", {})
        if isinstance(source_data, dict) and source_data.get("audio_path"):
            temp_candidates.append(Path(str(source_data["audio_path"])))
    except Exception:
        pass

    # --------------------------------------------------------
    # 2. SOURCE VISUAL FACTS JOIN → CLIP ANALYSIS
    # --------------------------------------------------------
    # Join only here, exactly when clip selection actually needs the facts.
    # If visual support fails we warn and preserve the transcript-only fallback;
    # the pipeline never dies solely because support intelligence failed.
    if resolved_vb:
        visual_started = time.perf_counter()
        try:
            if source_vb_cached and source_vb_expected_report is not None:
                source_visual_report_path = source_vb_expected_report
                book.record(
                    "visual_fact_observer",
                    "skipped",
                    source_vb_sig,
                    path=source_visual_report_path,
                    note="V23 cached whole-source factual visual inventory.",
                    elapsed=0.0,
                )
            elif source_vb_prepare_error is not None:
                raise source_vb_prepare_error
            else:
                if source_vb_future is not None:
                    _report, report_path = source_vb_future.result()
                else:
                    _report, report_path = _run_video_brain(
                        video_path,
                        force=effective_force,
                    )
                if report_path is None or not _valid_json(report_path):
                    raise RuntimeError("Source visual observer geçerli report üretmedi.")
                source_visual_report_path = Path(report_path).resolve()
                book.record(
                    "visual_fact_observer",
                    "done",
                    source_vb_sig,
                    path=source_visual_report_path,
                    note="V23 whole-source factual visual inventory; editorial owner remains Terra.",
                    elapsed=time.perf_counter() - visual_started,
                )
        except Exception as error:
            source_visual_report_path = None
            reason = (
                "Source visual facts kullanılamadı; transcript-only clip selection ile devam edildi. "
                f"Detay: {error}"
            )
            book.warn(reason)
            book.record(
                "visual_fact_observer",
                "fallback",
                source_vb_sig,
                note=reason,
                elapsed=time.perf_counter() - visual_started,
            )
        finally:
            if source_vb_executor is not None:
                source_vb_executor.shutdown(wait=True)
    else:
        book.record(
            "visual_fact_observer",
            "disabled",
            None,
            note="Video Brain disabled; transcript-only clip selection.",
        )

    # --------------------------------------------------------
    # 2. CLIP ANALYSIS — TERRA FINAL EDITOR
    # --------------------------------------------------------

    analysis_path = _analysis_path(transcript_path)
    analyzer_version = int(getattr(clip_analyzer, "ANALYZER_VERSION", 2))
    clip_analysis_inputs: list[Path] = [
        transcript_path,
    ]

    if source_visual_report_path is not None:
        clip_analysis_inputs.append(
            source_visual_report_path
        )

    analysis_sig = _stage_signature(
        "clip_analysis",
        inputs=clip_analysis_inputs,
        modules=[clip_analyzer, model_config],
        options={
            "scout_model": model_config.CLIP_SCOUT_MODEL,
            "scout_reasoning_effort": model_config.CLIP_SCOUT_REASONING_EFFORT,
            "judge_model": model_config.CLIP_JUDGE_MODEL,
            "judge_reasoning_effort": model_config.CLIP_JUDGE_REASONING_EFFORT,
            "judge_escalation_reasoning_effort": model_config.CLIP_JUDGE_ESCALATION_REASONING_EFFORT,
            "judge_high_margin": float(getattr(clip_analyzer, "CLIP_HIGH_ESCALATION_MARGIN", 0.35)),
            "editorial_owner": "Terra",
            "visual_facts_available": (
                source_visual_report_path is not None
            ),
        },
    )
    _run_simple_stage(
        book=book,
        number=2,
        title="Luna scout → Terra final clip judge",
        key="clip_analysis",
        signature=analysis_sig,
        output=analysis_path,
        validator=lambda: _valid_json(analysis_path, analyzer_version),
        runner=lambda: clip_analyzer.create_clip_analysis(
            transcript_path,
            video_report_path=source_visual_report_path,
        ),
        freshness=[
            transcript_path,
            *(
                [source_visual_report_path]
                if source_visual_report_path is not None
                else []
            ),
            *_module_paths([clip_analyzer]),
        ],
        error_message="Clip analysis JSON oluşmadı/geçersiz.",
    )

    analysis_data = _load_json(analysis_path)
    selected_clip_index, selected_clip = _select_clip(analysis_data, clip_index)

    try:
        selected_score = float(selected_clip.get("score", 0.0))
    except (TypeError, ValueError):
        selected_score = 0.0

    selected_title = str(
        selected_clip.get("title", f"clip_{selected_clip_index}")
    )

    try:
        selected_duration = float(
            selected_clip.get(
                "duration",
                0.0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        selected_duration = 0.0

    selected_covered = selected_clip.get(
        "covered_moment_ids",
        [],
    )

    selected_covered_count = (
        len(
            selected_covered
        )
        if isinstance(
            selected_covered,
            list,
        )
        else 0
    )

    strongest_anchor = selected_clip.get(
        "strongest_anchor",
        {},
    )

    if not isinstance(
        strongest_anchor,
        dict,
    ):
        strongest_anchor = {}

    anchor_count = len(
        selected_clip.get(
            "anchor_moments",
            [],
        )
        if isinstance(
            selected_clip.get(
                "anchor_moments",
                [],
            ),
            list,
        )
        else []
    )

    protected_count = len(
        selected_clip.get(
            "must_keep_ranges",
            [],
        )
        if isinstance(
            selected_clip.get(
                "must_keep_ranges",
                [],
            ),
            list,
        )
        else []
    )

    state["selected_clip"] = {
        "clip_index": selected_clip_index,
        "title": selected_title,
        "score": round(
            selected_score,
            2,
        ),
        "duration_seconds": round(selected_duration, 3),
        "covered_moment_count": selected_covered_count,
        "score_policy": "ranking_only_no_threshold",
        "anchor_count": anchor_count,
        "protected_range_count": protected_count,
        "strongest_anchor": strongest_anchor,
        "clip_selection_owner": "Terra",
    }

    book.save()

    print()
    print(
        "🏆 TERRA'NIN SEÇTİĞİ KLİP"
    )

    print(
        f"   Clip : {selected_clip_index}"
    )

    print(
        f"   Score: {selected_score:.1f}/10 "
        "(yalnızca sıralama)"
    )

    print(
        f"   Title: {selected_title}"
    )

    print(
        f"   Süre : {selected_duration:.2f}s "
        f"(dense story merkezi ~{PREFERRED_SHORT_DURATION:.0f}s)"
    )

    print(
        f"   🧩 Covered money moments: {selected_covered_count}"
    )

    print(
        "   Gate : YOK"
    )

    print(
        f"   💎 Anchor: {anchor_count}"
    )

    print(
        f"   🛡️ Protected: {protected_count}"
    )

    if strongest_anchor:

        print(
            "   🎯 En can alıcı nokta: "
            f"{float(strongest_anchor.get('start', 0.0)):.2f}"
            " → "
            f"{float(strongest_anchor.get('end', 0.0)):.2f}"
            " | "
            f"{float(strongest_anchor.get('strength', 0.0)):.1f}/10"
        )

    # --------------------------------------------------------
    # 3. PACING

    # --------------------------------------------------------

    pacing_path = _pacing_path(analysis_path)
    pacing_sig = _stage_signature(
        "pacing",
        inputs=[analysis_path, transcript_path],
        modules=[pacing],
    )
    _run_simple_stage(
        book=book,
        number=3,
        title="Pacing analysis",
        key="pacing",
        signature=pacing_sig,
        output=pacing_path,
        validator=lambda: _valid_json(pacing_path),
        runner=lambda: pacing.create_pacing_analysis(
            analysis_path=analysis_path,
            transcript_path=transcript_path,
        ),
        freshness=[
            analysis_path,
            transcript_path,
            *_module_paths([pacing]),
        ],
        error_message="Pacing JSON oluşmadı/geçersiz.",
    )

    # --------------------------------------------------------
    # 4. TIMELINE
    # --------------------------------------------------------

    timeline_path = _timeline_path(analysis_path)
    timeline_version = int(getattr(timeline, "TIMELINE_VERSION", 3))
    timeline_sig = _stage_signature(
        "timeline",
        inputs=[analysis_path, transcript_path, pacing_path],
        modules=[timeline],
        options={"clip_index": selected_clip_index},
    )
    _run_simple_stage(
        book=book,
        number=4,
        title=f"Timeline V{timeline_version}",
        key="timeline",
        signature=timeline_sig,
        output=timeline_path,
        validator=lambda: (
            _valid_json(timeline_path, timeline_version)
            and _contains_clip(
                timeline_path,
                "timelines",
                selected_clip_index,
                timeline_version,
            )
        ),
        runner=lambda: timeline.create_edit_timeline(
            analysis_path=analysis_path,
            transcript_path=transcript_path,
            pacing_path=pacing_path,
        ),
        freshness=[
            analysis_path,
            transcript_path,
            pacing_path,
            *_module_paths([timeline]),
        ],
        error_message="Timeline seçilen klibi içermiyor.",
    )

    timeline_data = _load_json(timeline_path)
    selected_timeline = _timeline_clip(timeline_data, selected_clip_index)

    # --------------------------------------------------------
    # 5A. PRE-RENDER SPEAKER PREFLIGHT
    # --------------------------------------------------------
    # Correct order:
    #   exact pacing AUDIO only -> diarization -> Luna-low participant/crowd
    #   role judge -> optional high-confidence 2/3-person audio identity -> heavy video render.
    # No video frame is encoded before the identity question.
    expected_edited = Path(
        pacing_cutter.get_output_path(timeline_data, selected_timeline)
    ).resolve()
    speaker_audio_path = Path(
        pacing_cutter.get_speaker_scan_audio_path(timeline_data, selected_timeline)
    ).resolve()
    speaker_scan_path = Path(
        speaker_caption_support.get_speaker_scan_path(
            speaker_audio_path, selected_clip_index
        )
    ).resolve()
    # V14 safe speed scheduling: pacing video output is independent of speaker
    # identity. Pre-compute its cache state now so, after the small speaker WAV
    # exists, an otherwise-needed encode may run in the background while
    # diarization/naming waits on network or user input. Render content is
    # unchanged; only scheduling overlaps.
    existing_edited = _resolve_edited_clip(expected_edited)
    previous_pacing = dict((state.get("stages") or {}).get("pacing_cut") or {})
    pacing_cut_sig = _stage_signature(
        "pacing_cut",
        inputs=[timeline_path],
        modules=[pacing_cutter],
        options={"clip_index": selected_clip_index},
    )
    pacing_cut_cached = book.reusable(
        "pacing_cut",
        pacing_cut_sig,
        lambda: _paced_clip_matches(existing_edited, selected_timeline),
        output=existing_edited or expected_edited,
        freshness=[timeline_path, *_module_paths([pacing_cutter])],
    )
    pacing_render_executor: ThreadPoolExecutor | None = None
    pacing_render_future = None

    speaker_preflight_sig = _stage_signature(
        "speaker_preflight",
        inputs=[timeline_path, video_path],
        modules=[pacing_cutter, speaker_caption_support, speaker_naming, speaker_role_judge],
        options={
            "clip_index": selected_clip_index,
            "speaker_profile_version": int(
                getattr(speaker_caption_support, "SPEAKER_PROFILE_VERSION", 13)
            ),
            "speaker_naming_version": int(getattr(speaker_naming, "SPEAKER_NAMING_VERSION", 11)),
            "speaker_role_version": int(getattr(speaker_role_judge, "ROLE_JUDGE_VERSION", 2)),
            "speaker_role_model": str(getattr(model_config, "SPEAKER_ROLE_MODEL", "gpt-5.6-luna")),
            "speaker_role_effort": str(getattr(model_config, "SPEAKER_ROLE_REASONING_EFFORT", "low")),
            "diarization_vad_threshold": float(getattr(speaker_caption_support, "DIARIZATION_VAD_THRESHOLD", 0.35)),
            "diarization_vad_prefix_padding_ms": int(getattr(speaker_caption_support, "DIARIZATION_VAD_PREFIX_PADDING_MS", 300)),
            "diarization_vad_silence_ms": int(getattr(speaker_caption_support, "DIARIZATION_VAD_SILENCE_MS", 220)),
            "speaker_identity_policy": "prompt_only_confident_2_or_3_else_plain; blank_name=plain",
            "render_order": "speaker_resolution_before_video_render",
        },
    )

    started = _stage_start(5, "Speaker preflight (optional confident 2/3-person identity)")
    print("🎧 Speaker taraması: yalnız yüksek güvenli 2/3 gerçek seste isim sorulur; diğer durumlarda normal altyazıyla devam...")
    if book.reusable(
        "speaker_preflight",
        speaker_preflight_sig,
        lambda: _valid_json(
            speaker_scan_path,
            int(getattr(speaker_caption_support, "SPEAKER_PROFILE_VERSION", 13)),
        ),
        output=speaker_scan_path,
        freshness=[timeline_path],
    ):
        _stage_skip(speaker_scan_path)
        book.record(
            "speaker_preflight",
            "skipped",
            speaker_preflight_sig,
            path=speaker_scan_path,
            elapsed=time.perf_counter() - started,
        )
    else:
        speaker_audio_path = Path(
            pacing_cutter.render_audio_only_for_clip(
                timeline_path=timeline_path,
                clip_index=selected_clip_index,
            )
        ).resolve()

        # The audio-only speaker asset is ready. If the exact pacing video is
        # not cached, start that identical encode now while diarization/naming
        # proceeds. No speaker result is consumed by the video renderer.
        if SAFE_PIPELINE_PARALLEL and not pacing_cut_cached:
            pacing_render_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mimir-pacing-render",
            )
            pacing_render_future = pacing_render_executor.submit(
                runtime_profiler.timed,
                "pacing_encode_worker",
                pacing_cutter.render_one_clip,
                timeline_path=timeline_path,
                clip_index=selected_clip_index,
            )

        speaker_scan_path = Path(
            speaker_caption_support.create_speaker_scan_from_audio(
                audio_path=speaker_audio_path,
                clip_index=selected_clip_index,
                reference_path=expected_edited,
            )
        ).resolve()
        scan_data = _load_json(speaker_scan_path)
        role = scan_data.get("role_judge", {}) if isinstance(scan_data, dict) else {}
        if isinstance(role, dict) and role:
            print(
                "🤖 Luna speaker role: "
                f"{scan_data.get('mode', '?')} "
                f"(confidence={float(role.get('confidence', 0.0) or 0.0):.2f})"
            )

        naming = speaker_naming.resolve_interactive_speaker_names(
            speaker_profile_path=speaker_scan_path,
            edited_clip_path=expected_edited,
            clip_index=selected_clip_index,
            creator_name=creator_name,
        )
        speaker_scan_path = Path(
            naming.get("profile_path", speaker_scan_path)
        ).resolve()
        preview_path = naming.get("preview_path")
        if preview_path:
            state["speaker_preview"] = str(Path(preview_path).resolve())
        book.record(
            "speaker_preflight",
            "done",
            speaker_preflight_sig,
            path=speaker_scan_path,
            note=(f"audio_preview={preview_path}" if preview_path else "speaker scan complete"),
            elapsed=time.perf_counter() - started,
        )

    # --------------------------------------------------------
    # 5B. PACING VIDEO RENDER
    # --------------------------------------------------------
    # V14: the encode may already be running behind speaker preflight. We still
    # join it here before any stage that consumes edited_clip_path, preserving
    # the exact dependency graph and output bytes.
    started = _stage_start(6, "Frame-accurate pacing cut")

    if pacing_cut_cached:
        edited_clip_path = Path(existing_edited or expected_edited).resolve()
        _stage_skip(edited_clip_path)
        paced_media = previous_pacing.get("media") if isinstance(previous_pacing.get("media"), dict) else None
        paced_stat = edited_clip_path.stat()
        if not (paced_media and paced_media.get("size") == paced_stat.st_size
                and paced_media.get("mtime_ns") == paced_stat.st_mtime_ns):
            paced_media = _media_identity(edited_clip_path)
        book.record(
            "pacing_cut",
            "skipped",
            pacing_cut_sig,
            path=edited_clip_path,
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            if pacing_render_future is not None:
                result_path = pacing_render_future.result()
            else:
                result_path = pacing_cutter.render_one_clip(
                    timeline_path=timeline_path,
                    clip_index=selected_clip_index,
                )
        finally:
            if pacing_render_executor is not None:
                pacing_render_executor.shutdown(wait=True)
        edited_clip_path = Path(result_path).resolve()

        if not _valid_file(edited_clip_path, MIN_VIDEO_BYTES):
            recovered = _resolve_edited_clip(expected_edited)
            if recovered is None:
                raise ShortsPipelineError("Edited clip oluşmadı.")
            edited_clip_path = recovered

        paced_media = _keep_render_clock(edited_clip_path, previous_pacing, pacing_cut_sig)
        if paced_media.get("clock") == "restored_identical_render":
            print("♻️ Paced clip byte-identical to the cached render; downstream caches stay valid.")
        _stage_done(edited_clip_path)
        book.record(
            "pacing_cut",
            "done",
            pacing_cut_sig,
            path=edited_clip_path,
            elapsed=time.perf_counter() - started,
        )
    # Content identity of the paced clip: lets a later re-render (after the temp
    # cleanup) prove it produced the same media.
    book.state["stages"]["pacing_cut"]["media"] = paced_media
    book.save()

    temp_candidates.append(edited_clip_path)

    # V14 safe speed scheduling: selected-clip visual support and final caption
    # analysis both depend on edited_clip_path, but not on each other. Start the
    # existing visual-support work now and join it at Stage 8. No call is added.
    video_brain_report_path: Path | None = None
    vb_sig: str | None = None
    vb_expected_report: Path | None = None
    vb_cached = False
    vb_prepare_error: Exception | None = None
    vb_executor: ThreadPoolExecutor | None = None
    vb_future = None
    if resolved_vb:
        try:
            vb_info = _video_brain_code_info()
            vb_sig = _hash(
                {
                    "stage": "video_brain",
                    "edited_clip": _file_fingerprint(edited_clip_path),
                    "model": video_brain_model,
                    "fallback_model": model_config.VISUAL_SUPPORT_MODEL,
                    "fallback_reasoning": model_config.VISUAL_SUPPORT_REASONING_EFFORT,
                    "analyzer": vb_info,
                }
            )
            from ai.video_brain import video_analyzer

            vb_expected_report = Path(
                video_analyzer.get_output_path(edited_clip_path)
            ).resolve()
            vb_cached = book.reusable(
                "video_brain",
                vb_sig,
                lambda: _valid_json(vb_expected_report),
                output=vb_expected_report,
                freshness=[edited_clip_path],
            )
            if SAFE_PIPELINE_PARALLEL and not vb_cached:
                vb_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="mimir-visual-support",
                )
                vb_future = vb_executor.submit(
                    runtime_profiler.timed,
                    "visual_support_worker",
                    _run_video_brain,
                    edited_clip_path,
                    force=effective_force,
                )
        except Exception as error:
            vb_prepare_error = error

    # --------------------------------------------------------
    # 6. FINAL-CLIP SPEAKER-AWARE CAPTIONS
    # --------------------------------------------------------

    caption_version = int(getattr(captions, "CAPTION_VERSION", 5))
    caption_dir = Path(captions.build_output_directory(timeline_path))
    caption_path = (
        caption_dir
        / f"clip_{selected_clip_index:02d}_captions_v{caption_version}.ass"
    ).resolve()

    captions_sig = _stage_signature(
        "captions",
        inputs=[edited_clip_path, transcript_path, timeline_path, speaker_scan_path],
        modules=[captions, speaker_caption_support, speaker_naming, speaker_role_judge, vod_processor],
        options={
            "clip_index": selected_clip_index,
            "caption_version": caption_version,
            "speaker_caption_version": int(
                getattr(speaker_caption_support, "SPEAKER_PROFILE_VERSION", 13)
            ),
            "speaker_naming_version": int(getattr(speaker_naming, "SPEAKER_NAMING_VERSION", 11)),
            "caption_text_model": str(getattr(vod_processor, "CAPTION_ACCURATE_MODEL", "gpt-transcribe")),
            "caption_crosscheck_model": str(getattr(vod_processor, "CAPTION_CROSSCHECK_MODEL", "gpt-4o-transcribe")),
            "caption_consensus_policy": "3-pass full ASR -> semantic/ASR suspicion -> micro-audio 2-of-3/3-of-5",
            "caption_semantic_model": str(getattr(speaker_caption_support, "CAPTION_SEMANTIC_MODEL", "gpt-5.6-luna")),
            "caption_micro_max_spans": int(getattr(speaker_caption_support, "CAPTION_MICRO_MAX_SPANS", 8)),
            "caption_quality_target": float(getattr(speaker_caption_support, "CAPTION_WORD_ACCURACY_TARGET", 0.97)),
            "caption_quality_retry_limit": int(getattr(speaker_caption_support, "CAPTION_QUALITY_RETRY_LIMIT", 1)),
            "timing_basis": "edited_clip",
        },
    )

    def _create_speaker_aware_captions() -> Path:
        # Speaker scan + optional human identity happened BEFORE video render.
        # Final word timing comes from the rendered edited clip; if identity was
        # skipped/uncertain, the accurate transcript stays as one unlabeled lane.
        speaker_profile_path = speaker_caption_support.create_speaker_profile(
            edited_clip_path=edited_clip_path,
            clip_index=selected_clip_index,
            transcript_path=transcript_path,
            timeline_path=timeline_path,
            speaker_scan_path=speaker_scan_path,
        )

        try:
            speaker_profile = _load_json(speaker_profile_path)
        except Exception:
            speaker_profile = {}

        if str(speaker_profile.get("status", "")) != "ok":
            detail = str(
                speaker_profile.get("error", "final-clip caption transcription unavailable")
            ).strip()
            # Accuracy is mandatory: never silently publish legacy transcript
            # timing after the exact-final-48k profile failed. That fallback was
            # the direct cause of a real two-speaker short losing both V3 timing
            # corrections and the human name display mapping. Fail BEFORE caption render so
            # the defect is visible and downstream media is not mislabeled.
            raise RuntimeError(
                "Mandatory exact-final caption profile failed; legacy timing fallback disabled. "
                f"Detay: {detail}"
            )
        elif str(speaker_profile.get("diarization_status", "")) != "ok":
            detail = str(speaker_profile.get("diarization_error", "")).strip()
            book.warn(
                "Speaker ayrımı kullanılamadı; senkronize tek-konuşmacı caption ile devam edildi. "
                f"Detay: {detail or 'diarization unavailable'}"
            )
        elif str(speaker_profile.get("mode", "")) == "unresolved":
            quality = speaker_profile.get("assignment", {}).get("separation_quality", "?")
            name_source = str((speaker_profile.get("speaker_names") or {}).get("source", ""))
            book.warn(
                "Speaker isimlendirmesi atlandı veya ayrım yeterince güvenli değildi; "
                f"tek isimsiz caption lane ile devam edildi. quality={quality}, source={name_source or 'plain'}"
            )

        caption_quality = speaker_profile.get("caption_quality", {})
        if isinstance(caption_quality, dict) and caption_quality:
            # The current caption stage reports ``quality_score`` (final text vs the independent
            # cross-check) and ``full_asr_passes``; older profiles used the other keys.
            best_agreement = float(caption_quality.get("best_reference_agreement",
                                                       caption_quality.get("quality_score", 0.0)) or 0.0)
            target = float(caption_quality.get("target", 0.97) or 0.97)
            local_fixes = int(caption_quality.get("local_corrections", 0) or 0)
            unresolved = int(caption_quality.get("unresolved_local_conflicts", 0) or 0)
            passes = int(caption_quality.get("strong_asr_passes", caption_quality.get("full_asr_passes", 0)) or 0)
            micro = caption_quality.get("micro_accuracy", {}) if isinstance(caption_quality.get("micro_accuracy", {}), dict) else {}
            micro_checked = int(micro.get("checked_spans", 0) or 0)
            micro_corrected = int(micro.get("corrected_spans", 0) or 0)
            target_met = caption_quality.get("target_met")
            if target_met is None:
                target_met = best_agreement >= target and unresolved == 0
            if target_met is True:
                print(
                    f"   ✅ Caption QA: {best_agreement * 100:.1f}% ASR consensus "
                    f"(target {target * 100:.0f}%, full_passes={passes}, micro={micro_checked}/{micro_corrected}, local_fixes={local_fixes})"
                )
            else:
                book.warn(
                    "Caption QA hedefi precision consensus sonrasında da karşılanmadı; "
                    f"ASR consensus={best_agreement * 100:.1f}% / hedef={target * 100:.0f}%, "
                    f"unresolved_local_conflicts={unresolved}. "
                    "Whisper/diarization metniyle kelime uydurulmadan en güvenli ASR metni korunarak devam edildi."
                )

        return captions.create_clip_captions(
            transcript_path=transcript_path,
            timeline_path=timeline_path,
            clip_index=selected_clip_index,
            speaker_profile_path=speaker_profile_path,
        )

    _run_simple_stage(
        book=book,
        number=7,
        title="Final-clip synchronized captions",
        key="captions",
        signature=captions_sig,
        output=caption_path,
        validator=lambda: _valid_file(caption_path, MIN_CAPTION_BYTES),
        runner=_create_speaker_aware_captions,
        freshness=[
            edited_clip_path,
            transcript_path,
            timeline_path,
            speaker_scan_path,
            *_module_paths([captions, speaker_caption_support, speaker_naming, speaker_role_judge, vod_processor]),
        ],
        error_message="Caption ASS dosyası oluşmadı.",
    )

    # --------------------------------------------------------
    # 7B. VERIFIED PARTICIPANT NAME LOCK (caption truth)
    # --------------------------------------------------------
    # Runs on the identity-resolved final profile, where WHO is known (the V7
    # text-level vocative lock runs before speaker assignment). Only word TEXT
    # may change; word ids, times and speakers are verified unchanged, and the
    # caption ASS is regenerated from the corrected truth. Local and cheap:
    # the paid caption stage above is never re-run for it.
    name_lock_profile = _speaker_profile_path(edited_clip_path, selected_clip_index)
    print(f"\n[7b/{TOTAL_STAGES}] Caption truth (verified names → targeted escalation → freeze)")
    print("-" * 60)
    caption_truth_path: Path | None = None
    caption_path, caption_truth_path = _run_caption_truth_v6(
        book=book,
        run=v6_run,
        truth_mod=caption_truth_mod,
        profile_path=name_lock_profile,
        caption_path=caption_path,
        transcript_path=transcript_path,
        timeline_path=timeline_path,
        edited_clip_path=edited_clip_path,
        clip_index=selected_clip_index,
        creator_name=creator_name,
        entities=v6_entities,
    )
    if caption_truth_path is None:
        raise ShortsPipelineError(
            "Caption truth dondurulamadı; doğrulanmamış altyazıyla Short yayınlanmaz. "
            "Ayrıntı yukarıdaki Caption Truth uyarısında."
        )

    # V14: caption rendering is deterministic FFmpeg work and does not feed
    # visual/teaser/intro analysis. Start the exact same render now and join it
    # only before intro rendering/fallback actually needs the captioned preview.
    # Explicit paced clip: the baseline render (and the Pro Edit fallback, which
    # calls this job) consumes exactly the media every other stage used.
    caption_render_job: Callable[[], Any] = functools.partial(
        caption_renderer.render_captioned_clip,
        timeline_path=timeline_path,
        clip_index=selected_clip_index,
        caption_path=caption_path,
        edited_clip_path=edited_clip_path,
    )
    captioned_preview_path = Path(
        caption_renderer.get_output_path(timeline_data, selected_timeline)
    ).resolve()
    caption_render_sig = _stage_signature(
        "caption_render",
        inputs=[edited_clip_path, caption_path, timeline_path],
        modules=[caption_renderer],
        options={"clip_index": selected_clip_index},
    )

    def _start_caption_render(
        preview_path: Path,
        render_sig: str,
        freshness_modules: Iterable[Any],
        job: Callable[[], Any],
    ) -> tuple[bool, ThreadPoolExecutor | None, Any, float]:
        cached = book.reusable(
            "caption_render",
            render_sig,
            lambda: _valid_file(preview_path, MIN_VIDEO_BYTES),
            output=preview_path,
            freshness=[
                edited_clip_path,
                caption_path,
                timeline_path,
                *_module_paths(freshness_modules),
            ],
        )
        executor: ThreadPoolExecutor | None = None
        future = None
        started_at = time.perf_counter()
        if SAFE_PIPELINE_PARALLEL and not cached:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="mimir-caption-render",
            )
            future = executor.submit(
                runtime_profiler.timed,
                "caption_render_worker",
                job,
            )
        return cached, executor, future, started_at

    caption_render_cached = False
    caption_render_executor: ThreadPoolExecutor | None = None
    caption_render_future = None
    caption_render_started = time.perf_counter()
    pro_render_outcome: dict[str, Any] = {}

    # --------------------------------------------------------
    # 7. VIDEO BRAIN SUPPORT
    # --------------------------------------------------------

    started = _stage_start(8, "Selected-clip visual support")

    if not resolved_vb:
        print("⏭️ Video Brain kapalı.")
        book.record(
            "video_brain",
            "disabled",
            None,
            note="Support-only Video Brain bu run için kapalı.",
            elapsed=time.perf_counter() - started,
        )
    elif vb_prepare_error is not None:
        reason = (
            "Video Brain destek katmanı çalışmadı; ana Shorts pipeline "
            f"normal devam ediyor. Detay: {vb_prepare_error}"
        )
        book.warn(reason)
        book.record(
            "video_brain",
            "fallback",
            vb_sig,
            note=reason,
            elapsed=time.perf_counter() - started,
        )
    elif vb_cached and vb_expected_report is not None:
        video_brain_report_path = vb_expected_report
        _stage_skip(vb_expected_report)
        book.record(
            "video_brain",
            "skipped",
            vb_sig,
            path=vb_expected_report,
            note="Selected-clip visual support only. Terra already selected the clip.",
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            if vb_future is not None:
                _, report_path = vb_future.result()
            else:
                _, report_path = _run_video_brain(
                    edited_clip_path,
                    force=effective_force,
                )
            if report_path is None:
                raise RuntimeError("Video Brain report üretmedi.")
            video_brain_report_path = Path(report_path).resolve()
            _stage_done(video_brain_report_path)
            book.record(
                "video_brain",
                "done",
                vb_sig,
                path=video_brain_report_path,
                note="Selected-clip visual support only. Terra already selected the clip.",
                elapsed=time.perf_counter() - started,
            )
        except Exception as error:
            reason = (
                "Video Brain destek katmanı çalışmadı; ana Shorts pipeline "
                f"normal devam ediyor. Detay: {error}"
            )
            book.warn(reason)
            book.record(
                "video_brain",
                "fallback",
                vb_sig,
                note=reason,
                elapsed=time.perf_counter() - started,
            )
        finally:
            if vb_executor is not None:
                vb_executor.shutdown(wait=True)

    if video_brain_report_path is not None:
        state["video_brain_report"] = str(video_brain_report_path)
        book.save()

    # Caption render is running in the background (or already cached).

    # --------------------------------------------------------
    # 9. TEASER
    # --------------------------------------------------------

    teaser_path = Path(teaser_analyzer.get_output_path(timeline_path)).resolve()
    started = _stage_start(10, "Cold-open teaser analysis")

    teaser_analysis_inputs: list[Path] = [
        timeline_path,
        transcript_path,
        edited_clip_path,
    ]
    if video_brain_report_path is not None:
        teaser_analysis_inputs.append(video_brain_report_path)
    # The cold-open boundaries are measured on the frozen caption word clock.
    if name_lock_profile is not None:
        teaser_analysis_inputs.append(name_lock_profile)

    teaser_sig = _stage_signature(
        "teaser_analysis",
        inputs=teaser_analysis_inputs,
        modules=[teaser_analyzer, intro_peak_support, intro_bounds, model_config],
        options={
            "clip_index": selected_clip_index,
            "model": model_config.TEASER_MODEL,
            "reasoning_effort": model_config.TEASER_REASONING_EFFORT,
        },
    )

    if book.reusable(
        "teaser_analysis",
        teaser_sig,
        lambda: _contains_clip(
            teaser_path, "teasers", selected_clip_index, 1
        ),
        output=teaser_path,
        freshness=[
            timeline_path,
            transcript_path,
            edited_clip_path,
            *(
                [video_brain_report_path]
                if video_brain_report_path is not None
                else []
            ),
            *_module_paths([teaser_analyzer, intro_peak_support]),
        ],
    ):
        _stage_skip(teaser_path)
        book.record(
            "teaser_analysis",
            "skipped",
            teaser_sig,
            path=teaser_path,
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            teaser_analyzer.analyze_teasers(
                timeline_path=timeline_path,
                transcript_path=transcript_path,
                clip_index=selected_clip_index,
                edited_video_path=edited_clip_path,
                video_report_path=video_brain_report_path,
                caption_profile_path=name_lock_profile,
            )
            if not _contains_clip(teaser_path, "teasers", selected_clip_index, 1):
                raise RuntimeError("Teaser package seçilen klibi içermiyor.")

            _stage_done(teaser_path)
            book.record(
                "teaser_analysis",
                "done",
                teaser_sig,
                path=teaser_path,
                elapsed=time.perf_counter() - started,
            )
        except Exception as error:
            reason = (
                "Teaser AI yolu üretilemedi; V27.1 mandatory peak fallback devreye girdi. "
                f"Detay: {error}"
            )
            book.warn(reason)
            try:
                _write_fallback_teaser(
                    teaser_path,
                    timeline_path,
                    transcript_path,
                    timeline_data,
                    selected_clip_index,
                    reason,
                    edited_video_path=edited_clip_path,
                    video_report_path=video_brain_report_path,
                    caption_profile_path=name_lock_profile,
                    caption_path=caption_path,
                )
            except Exception as fallback_error:
                raise ShortsPipelineError(
                    "Mandatory cold-open üretilemedi; introsuz Short yayınlamak yasak. "
                    f"AI hata: {error}; deterministic fallback hata: {fallback_error}"
                ) from fallback_error
            book.record(
                "teaser_analysis",
                "fallback",
                teaser_sig,
                path=teaser_path,
                note=reason,
                elapsed=time.perf_counter() - started,
            )

    teaser_record = _package_clip(
        teaser_path,
        "teasers",
        selected_clip_index,
    )

    teaser_recommended = bool(
        teaser_record
        and teaser_record.get(
            "recommended"
        )
        is True
    )

    teaser_duration = 0.0

    if isinstance(
        teaser_record,
        dict,
    ):
        edited_teaser = teaser_record.get(
            "edited",
            {},
        )

        if isinstance(
            edited_teaser,
            dict,
        ):
            try:
                teaser_duration = float(
                    edited_teaser.get(
                        "duration",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                teaser_duration = 0.0

    # Important V8 reliability change:
    # teaser "recommended" is an advisory score, not a hard prerequisite for
    # intro generation. A valid teaser can become excellent when paired with a
    # strong Terra hook. The intro's own 8/10 judge remains the final gate.
    teaser_usable = bool(
        isinstance(
            teaser_record,
            dict,
        )
        and intro_bounds.MIN_INTRO_S - 0.05
        <= teaser_duration
        <= intro_bounds.MAX_INTRO_S + 0.05
    )

    if not teaser_usable:
        reason = (
            "Teaser package teknik olarak kullanılamaz durumda; "
            "V27.1 mandatory peak fallback ile yeniden kuruluyor."
        )
        book.warn(reason)
        try:
            _write_fallback_teaser(
                teaser_path,
                timeline_path,
                transcript_path,
                timeline_data,
                selected_clip_index,
                reason,
                edited_video_path=edited_clip_path,
                video_report_path=video_brain_report_path,
                caption_profile_path=name_lock_profile,
                caption_path=caption_path,
            )
            teaser_record = _package_clip(
                teaser_path,
                "teasers",
                selected_clip_index,
            )
            edited_teaser = teaser_record.get("edited", {}) if isinstance(teaser_record, dict) else {}
            teaser_duration = float(edited_teaser.get("duration", 0.0)) if isinstance(edited_teaser, dict) else 0.0
            teaser_recommended = bool(teaser_record and teaser_record.get("recommended") is True)
            teaser_usable = bool(
                isinstance(teaser_record, dict)
                and intro_bounds.MIN_INTRO_S - 0.05 <= teaser_duration <= intro_bounds.MAX_INTRO_S + 0.05
            )
        except Exception as fallback_error:
            raise ShortsPipelineError(
                "Mandatory cold-open recovery başarısız; introsuz Short yayınlanmadı. "
                f"Detay: {fallback_error}"
            ) from fallback_error

    if not teaser_usable:
        raise ShortsPipelineError(
            "Mandatory cold-open teknik doğrulamadan geçmedi; introsuz Short yayınlanmadı."
        )

    if (
        teaser_usable
        and not teaser_recommended
    ):
        print(
            "ℹ️ Teaser 7/10 recommendation eşiğinin altında ama teknik olarak "
            "geçerli. Intro Terra judge yine deneyecek."
        )

    # --------------------------------------------------------
    # 11. INTRO + BASE PREVIEW
    # --------------------------------------------------------

    intro_path = Path(intro_analyzer.get_output_path(teaser_path)).resolve()
    started = _stage_start(11, "Hook intro + final base preview")

    intro_analysis_inputs: list[Path] = [
        teaser_path,
    ]

    if video_brain_report_path is not None:
        intro_analysis_inputs.append(
            video_brain_report_path
        )
    # Headline grounding evidence: the frozen caption truth and verified names.
    intro_analysis_inputs.append(caption_truth_path)
    headline_verified_names = _verified_names(name_lock_profile, creator_name, v6_entities)
    headline_caption_text = _truth_text(caption_truth_path)

    intro_analysis_sig = _stage_signature(
        "intro_analysis",
        inputs=intro_analysis_inputs,
        modules=[intro_analyzer, model_config],
        options={
            "clip_index": selected_clip_index,
            "creator_name": creator_name or "",
            "verified_names": headline_verified_names,
            "quality_threshold": MIN_INTRO_ACCEPT_SCORE,
            "draft_model": model_config.INTRO_DRAFT_MODEL,
            "draft_reasoning_effort": model_config.INTRO_DRAFT_REASONING_EFFORT,
            "judge_model": model_config.INTRO_JUDGE_MODEL,
            "judge_reasoning_effort": model_config.INTRO_JUDGE_REASONING_EFFORT,
        },
    )

    intro_record: dict[str, Any] | None = None

    if not teaser_usable:
        raise ShortsPipelineError(
            "Mandatory intro aşamasına geçildi fakat teaser_usable=false; introsuz çıktı engellendi."
        )
    if book.reusable(
        "intro_analysis",
        intro_analysis_sig,
        lambda: _contains_clip(intro_path, "intros", selected_clip_index, 1),
        output=intro_path,
    ):
        _stage_skip(intro_path)
        book.record(
            "intro_analysis",
            "skipped",
            intro_analysis_sig,
            path=intro_path,
        )
    else:
        try:
            intro_analyzer.analyze_intros(
                teaser_json_path=teaser_path,
                clip_index=selected_clip_index,
                manual_creator_name=creator_name,
                video_report_path=video_brain_report_path,
                verified_names=headline_verified_names,
                caption_text=headline_caption_text,
            )
            if not _contains_clip(intro_path, "intros", selected_clip_index, 1):
                raise RuntimeError("Intro package seçilen klibi içermiyor.")
            book.record(
                "intro_analysis",
                "done",
                intro_analysis_sig,
                path=intro_path,
            )
        except Exception as error:
            reason = (
                "Headline AI yolu üretilemedi; cold open headline'sız (yalnız hareketli peak) çıkacak. "
                f"Detay: {error}"
            )
            book.warn(reason)
            _write_fallback_intro(
                intro_path,
                teaser_path,
                timeline_path,
                transcript_path,
                timeline_data,
                selected_clip_index,
                reason,
            )
            book.record(
                "intro_analysis",
                "fallback",
                intro_analysis_sig,
                path=intro_path,
                note=reason,
            )

    intro_record = _package_clip(intro_path, "intros", selected_clip_index)
    intro_score = 0.0
    if isinstance(intro_record, dict):
        try:
            intro_score = float(intro_record.get("score", 0.0))
        except (TypeError, ValueError):
            intro_score = 0.0
    intro_quality_gate = intro_record.get("quality_gate", {}) if isinstance(intro_record, dict) else {}
    if not isinstance(intro_quality_gate, dict):
        intro_quality_gate = {}
    headline_text = str(intro_record.get("intro_text", "")).strip() if isinstance(intro_record, dict) else ""
    headline_approved = bool(
        isinstance(intro_record, dict)
        and intro_record.get("recommended") is True
        and intro_quality_gate.get("accepted") is not False
        and headline_text
        and intro_score >= MIN_INTRO_ACCEPT_SCORE
        and not intro_quality_gate.get("locked_intro_override")
    )
    no_headline = bool(isinstance(intro_record, dict) and intro_quality_gate.get("no_headline") is True
                       and not headline_text)
    if not headline_approved and not no_headline:
        reason = (
            f"Headline {intro_score:.1f}/10 (< {MIN_INTRO_ACCEPT_SCORE:.1f}) veya onaysız; "
            "cold open headline'sız çıkacak (kötü headline yerine yalnız hareketli peak)."
        )
        print("ℹ️ " + reason)
        _write_fallback_intro(
            intro_path,
            teaser_path,
            timeline_path,
            transcript_path,
            timeline_data,
            selected_clip_index,
            reason,
        )
        intro_record = _package_clip(intro_path, "intros", selected_clip_index)
        headline_text = ""
    if not isinstance(intro_record, dict):
        raise ShortsPipelineError("Intro package doğrulanamadı; cold open'sız Short yayınlanmaz.")
    print(f"🧲 Headline: {headline_text or '(yok — cold open yalnız hareketli peak)'}")

    # Intro source defaults to the clean paced clip (existing MIMIR behavior).
    intro_source_path = edited_clip_path
    intro_source_executor: ThreadPoolExecutor | None = None
    intro_source_future = None
    pro_intro_outcome: dict[str, Any] = {}
    pro_prep_ref: Any = None

    # --------------------------------------------------------
    # 11B. PRO EDIT DIRECTION (production presentation layer)
    # --------------------------------------------------------
    # Runs AFTER story, pacing, captions and the
    # mandatory intro selection are final (all consumed read-only) and BEFORE
    # caption burn-in and intro composition. The main camera is placed BEFORE
    # the subtitles filter in one FFmpeg graph (captions never zoomed, audio
    # stream-copied). The intro camera, if any, is rendered into a copy of the
    # clean paced clip ONLY on the selected teaser frames; the intro renderer
    # then trims the same range, so selection/order/handoff are unchanged.
    baseline_preview_path = captioned_preview_path
    baseline_render_sig = caption_render_sig
    baseline_render_job = caption_render_job
    render_modules: list[Any] = [caption_renderer]
    try:
        pro_started = time.perf_counter()
        print(f"\n[11b/{TOTAL_STAGES}] Pro Edit direction (presentation only)")
        print("-" * 60)
        pro_stage = pro_edit_pkg.stage
        pro_intro_output = captioned_preview_path.with_name(
            f"{captioned_preview_path.stem}_intro_source_pro_edit_v"
            f"{int(getattr(pro_edit_pkg, 'PRO_EDIT_VERSION', 1))}{captioned_preview_path.suffix}"
        )
        pro_speaker_profile = _speaker_profile_path(edited_clip_path, selected_clip_index)
        pro_output = captioned_preview_path.with_name(
            f"{captioned_preview_path.stem}_pro_edit_v{int(getattr(pro_edit_pkg, 'PRO_EDIT_VERSION', 1))}"
            f"{captioned_preview_path.suffix}"
        )
        pro_sig = _stage_signature(
            "pro_edit_plan",
            inputs=[
                path
                for path in (
                    timeline_path,
                    edited_clip_path,
                    caption_path,
                    analysis_path,
                    teaser_path,
                    intro_path,
                    pro_speaker_profile,
                    video_brain_report_path,
                )
                if path is not None
            ],
            modules=list(getattr(pro_edit_pkg, "PLAN_MODULES", pro_edit_pkg.MODULES)),
            options={
                "clip_index": selected_clip_index,
                "config": getattr(pro_edit_cfg, "plan_signature_payload", pro_edit_cfg.signature_payload)(),
            },
        )
        pro_prep = pro_stage.prepare_pro_edit(
            pro_stage.ProEditRequest(
                config=pro_edit_cfg,
                timeline_path=timeline_path,
                clip_index=selected_clip_index,
                edited_clip_path=edited_clip_path,
                caption_path=caption_path,
                output_path=pro_output,
                artifact_dir=VOD_OUTPUT_DIR / "pro_edit" / _safe_name(video_path.stem),
                input_signature=pro_sig,
                analysis_clip=selected_clip,
                speaker_profile_path=pro_speaker_profile,
                video_report_path=video_brain_report_path,
                teaser_record=teaser_record,
                intro_record=intro_record,
                intro_output_path=pro_intro_output,
                force=effective_force or rerender,
                reuse_plan_cache=rerender and not effective_force,
            )
        )
        pro_prep.diagnostics.emit()
        for warning in pro_prep.warnings:
            book.warn(warning)
        plan_artifact = pro_prep.artifacts.plan if pro_prep.artifacts is not None else None
        book.record(
            "pro_edit",
            pro_prep.status,
            pro_sig,
            path=plan_artifact if plan_artifact is not None and plan_artifact.is_file() else None,
            note=pro_prep.reason,
            elapsed=time.perf_counter() - pro_started,
        )
        pro_prep_ref = pro_prep
        state["pro_edit"] = {
            **pro_prep.summary(),
            "planner": pro_edit_cfg.planner,
            "style": pro_edit_cfg.style,
            "output_profile": pro_edit_cfg.output_profile.value,
        }
        book.save()
        print(f"🎥 Pro Edit: {pro_prep.status} — {pro_prep.reason}")
        if v6_run is not None:
            _record_pro_edit_v6(v6_run, pro_prep)

        if pro_prep.ready:
            captioned_preview_path = pro_prep.output_path
            caption_render_sig = _stage_signature(
                "caption_render",
                inputs=[edited_clip_path, caption_path, timeline_path],
                modules=[caption_renderer, *pro_edit_pkg.MODULES],
                options={
                    "clip_index": selected_clip_index,
                    "pro_edit": pro_prep.render_options,
                },
            )
            caption_render_job = functools.partial(
                pro_stage.render_with_fallback,
                pro_prep,
                caption_file=caption_path,
                baseline=caption_render_job,
                outcome=pro_render_outcome,
            )
            render_modules = [caption_renderer, *pro_edit_pkg.MODULES]
        else:
            # Static plan / fallback: exactly the existing caption render + cache.
            render_modules = [caption_renderer]
    except Exception as error:  # Pro Edit must never be a single point of failure
        captioned_preview_path = baseline_preview_path
        caption_render_sig = baseline_render_sig
        caption_render_job = baseline_render_job
        render_modules = [caption_renderer]
        pro_render_outcome.clear()
        pro_prep_ref = None
        state["pro_edit"] = {"status": "fallback", "reason": f"{type(error).__name__}: {error}"}
        book.warn(
            "Pro Edit aşaması beklenmedik şekilde başarısız; mevcut MIMIR caption render kullanılıyor. "
            f"Detay: {type(error).__name__}: {error}"
        )
        if v6_run is not None:
            v6_run.fallback("pro_edit", f"{type(error).__name__}: {error}", "baseline_caption_render")
    (
        caption_render_cached,
        caption_render_executor,
        caption_render_future,
        caption_render_started,
    ) = _start_caption_render(
        captioned_preview_path,
        caption_render_sig,
        render_modules,
        caption_render_job,
    )
    if pro_prep_ref is not None and pro_prep_ref.intro_ready:
        # Independent of the caption burn-in: render concurrently, join
        # right before the intro renderer consumes it.
        intro_source_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mimir-pro-intro")
        intro_source_future = intro_source_executor.submit(
            runtime_profiler.timed,
            "pro_edit_intro_worker",
            pro_edit_pkg.stage.render_intro_source,
            pro_prep_ref,
            clean_clip=edited_clip_path,
            outcome=pro_intro_outcome,
        )

    # Join the deterministic caption render at the first true consumer.
    # Teaser and intro analysis above have already overlapped with this encode.
    if caption_render_cached:
        _stage_skip(captioned_preview_path)
        book.record(
            "caption_render",
            "skipped",
            caption_render_sig,
            path=captioned_preview_path,
            elapsed=time.perf_counter() - caption_render_started,
        )
    else:
        try:
            if caption_render_future is not None:
                rendered_caption = caption_render_future.result()
            else:
                rendered_caption = caption_render_job()
            if rendered_caption:
                captioned_preview_path = Path(rendered_caption).resolve()
        finally:
            if caption_render_executor is not None:
                caption_render_executor.shutdown(wait=True)
        if not _valid_file(captioned_preview_path, MIN_VIDEO_BYTES):
            raise ShortsPipelineError("Captioned preview oluşmadı.")
        caption_render_status = "done"
        if pro_render_outcome.get("status") == "fallback":
            # Recorded as fallback so the next run retries Pro Edit instead of
            # reusing a baseline file under the Pro Edit signature.
            caption_render_status = "fallback"
            book.warn(
                "Pro Edit render başarısız; mevcut MIMIR caption render kullanıldı. "
                f"Detay: {pro_render_outcome.get('reason', '')}"
            )
        elif pro_render_outcome.get("failures"):
            # A lower Pro Edit level (camera + baseline ASS) rendered: keep the
            # output but do not cache it under the presentation signature.
            caption_render_status = "fallback"
            book.warn(
                "Pro Edit caption presentation render başarısız; kamera + mevcut caption ASS kullanıldı. "
                f"Detay: {' | '.join(pro_render_outcome.get('failures', []))}"
            )
        if pro_render_outcome and isinstance(state.get("pro_edit"), dict):
            state["pro_edit"]["render"] = dict(pro_render_outcome)
        _stage_done(captioned_preview_path)
        book.record(
            "caption_render",
            caption_render_status,
            caption_render_sig,
            path=captioned_preview_path,
            elapsed=time.perf_counter() - caption_render_started,
        )
    temp_candidates.append(captioned_preview_path)

    v6_main_proof: dict[str, Any] | None = None
    v6_intro_proof: dict[str, Any] | None = None
    v6_burned_ass: Path | None = caption_path
    v6_render_status = "baseline"
    if v6_run is not None:
        if pro_prep_ref is not None and pro_prep_ref.ready:
            if caption_render_cached:
                v6_render_status = "pro_edit"
                burned_label = "presentation" if pro_prep_ref.presentation_ass is not None else "baseline_ass"
            else:
                v6_render_status = str(pro_render_outcome.get("status", "unknown"))
                burned_label = str(pro_render_outcome.get("captions", ""))
            if burned_label == "presentation" and pro_prep_ref.presentation_ass is not None:
                v6_burned_ass = Path(pro_prep_ref.presentation_ass)
            if pro_render_outcome.get("failures"):
                v6_run.fallback("caption_presentation_render", " | ".join(pro_render_outcome.get("failures", [])),
                                "camera_with_baseline_ass")
        if v6_render_status == "pro_edit" and pro_prep_ref is not None:
            v6_main_proof = pro_edit_pkg.stage.prove_main_render(pro_prep_ref, captioned_preview_path, v6_burned_ass)
        else:
            v6_run.fallback("pro_edit_render", str(pro_render_outcome.get("reason", "")) or
                            f"Pro Edit render status={v6_render_status}", "baseline_caption_render")

    if intro_source_future is not None:
        try:
            intro_source_path = Path(intro_source_future.result()).resolve()
        except Exception as error:  # render_intro_source is contained; defensive only
            pro_intro_outcome.update(status="fallback", reason=f"{type(error).__name__}: {error}")
            intro_source_path = edited_clip_path
        finally:
            if intro_source_executor is not None:
                intro_source_executor.shutdown(wait=True)
        if pro_intro_outcome.get("status") == "fallback":
            book.warn(
                "Pro Edit intro camera render başarısız; mevcut intro (temiz klip) kullanıldı. "
                f"Detay: {pro_intro_outcome.get('reason', '')}"
            )
        if isinstance(state.get("pro_edit"), dict):
            state["pro_edit"]["intro_render"] = dict(pro_intro_outcome)
        if intro_source_path != edited_clip_path:
            temp_candidates.append(intro_source_path)
        if v6_run is not None:
            if pro_intro_outcome.get("status") == "fallback":
                v6_run.fallback("intro_camera", str(pro_intro_outcome.get("reason", "")), "clean_intro_source")
            elif intro_source_path != edited_clip_path and pro_prep_ref is not None:
                v6_intro_proof = pro_edit_pkg.stage.prove_intro_render(pro_prep_ref, intro_source_path)

    intro_render_sig = _stage_signature(
        "intro_final_base",
        inputs=[
            intro_path,
            teaser_path,
            timeline_path,
            intro_source_path,
            captioned_preview_path,
            caption_path,
        ],
        modules=[intro_renderer],
        options={
            "clip_index": selected_clip_index,
            "headline": headline_text,
            "restart": "hard_cut_protected_restart_v1",
            "post_intro_preroll_seconds": float(
                getattr(intro_renderer, "POST_INTRO_SPEECH_PREROLL_SECONDS", 0.65)
            ),
        },
    )

    expected_intro_video = Path(
        intro_renderer.get_output_path(
            timeline=timeline_data,
            intro=intro_record,
            clip_index=selected_clip_index,
        )
    ).resolve()

    if book.reusable(
        "intro_final_base",
        intro_render_sig,
        lambda: (_valid_file(expected_intro_video, MIN_VIDEO_BYTES)
                 and intro_renderer.load_final_timeline(expected_intro_video) is not None),
        output=expected_intro_video,
    ):
        final_preview_path = _activate_final_preview(expected_intro_video)
        _stage_skip(final_preview_path)
    else:
        try:
            rendered = intro_renderer.run_renderer(
                intro_json_path=intro_path,
                clip_index=selected_clip_index,
                edited_clip_path=intro_source_path,
                captioned_preview_path=captioned_preview_path,
                caption_path=caption_path,
            )
            if not rendered:
                raise RuntimeError("Intro renderer çıktı döndürmedi.")
            final_preview_path = _activate_final_preview(rendered[0])
        except Exception as error:
            raise ShortsPipelineError(
                "Mandatory intro render başarısız; main-only fallback YASAK. "
                f"Detay: {error}"
            ) from error

    if not _valid_file(final_preview_path, MIN_VIDEO_BYTES):
        raise ShortsPipelineError("Final base preview oluşmadı.")
    final_timeline_doc = intro_renderer.load_final_timeline(final_preview_path)
    if final_timeline_doc is None:
        raise ShortsPipelineError("Final timeline belgesi yok; cold open ve restart doğrulanamaz.")
    # Structural contract: a real cold open was prepended (the renderer already
    # verified the composed duration against intro + main to within 2 frames).
    if float((final_timeline_doc.get("intro") or {}).get("duration", 0.0) or 0.0) < intro_bounds.MIN_INTRO_S - 0.05:
        raise ShortsPipelineError("Mandatory intro structural guard: cold open eksik; introsuz çıktı yayınlanmaz.")

    if pro_prep_ref is not None and pro_prep_ref.intro_timeline is not None:
        final_preview_path = _verify_pro_edit_intro(
            pro_edit_pkg,
            pro_prep_ref,
            book=book,
            state=state,
            teaser_path=teaser_path,
            intro_path=intro_path,
            caption_path=caption_path,
            clean_clip=edited_clip_path,
            intro_source=intro_source_path,
            main_preview=captioned_preview_path,
            final_preview=final_preview_path,
            clip_index=selected_clip_index,
        )

    _stage_done(final_preview_path)
    book.record(
        "intro_final_base",
        "done",
        intro_render_sig,
        path=final_preview_path,
        elapsed=time.perf_counter() - started,
    )
    temp_candidates.append(final_preview_path)

    # --------------------------------------------------------
    # 11. MEME ANALYSIS
    # --------------------------------------------------------

    meme_slot_path = Path(meme_analyzer.get_output_path(intro_path)).resolve()
    started = _stage_start(12, "Single-meme analysis")

    meme_analysis_sig = _stage_signature(
        "meme_analysis",
        inputs=[intro_path, teaser_path, timeline_path, transcript_path, final_preview_path],
        modules=[meme_analyzer, meme_audio_support, model_config],
        options={
            "clip_index": selected_clip_index,
            "enabled": bool(enable_memes),
            "model": model_config.MEME_MODEL,
            "reasoning_effort": model_config.MEME_REASONING_EFFORT,
        },
    )

    if not enable_memes:
        reason = "Meme pipeline kullanıcı tarafından kapatıldı."
        _write_fallback_meme_slots(
            meme_slot_path,
            intro_path,
            teaser_path,
            timeline_path,
            transcript_path,
            timeline_data,
            selected_clip_index,
            reason,
        )
        print("⏭️ Meme sistemi kapalı.")
        book.record(
            "meme_analysis",
            "disabled",
            meme_analysis_sig,
            path=meme_slot_path,
            note=reason,
            elapsed=time.perf_counter() - started,
        )
    elif book.reusable(
        "meme_analysis",
        meme_analysis_sig,
        lambda: _contains_clip(meme_slot_path, "clips", selected_clip_index, meme_analyzer.MEME_ANALYZER_VERSION),
        output=meme_slot_path,
        freshness=[
            intro_path,
            teaser_path,
            timeline_path,
            transcript_path,
            final_preview_path,
            *_module_paths([meme_analyzer, meme_audio_support]),
        ],
    ):
        _stage_skip(meme_slot_path)
        book.record(
            "meme_analysis",
            "skipped",
            meme_analysis_sig,
            path=meme_slot_path,
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            meme_analyzer.analyze_memes(
                intro_json_path=intro_path,
                clip_index=selected_clip_index,
                base_video_path=final_preview_path,
            )
            if not _contains_clip(meme_slot_path, "clips", selected_clip_index, meme_analyzer.MEME_ANALYZER_VERSION):
                raise RuntimeError("Meme slot JSON seçilen klibi içermiyor.")

            _stage_done(meme_slot_path)
            book.record(
                "meme_analysis",
                "done",
                meme_analysis_sig,
                path=meme_slot_path,
                elapsed=time.perf_counter() - started,
            )
        except Exception as error:
            reason = (
                "Meme analizi başarısız; meme olmadan devam edildi. "
                f"Detay: {error}"
            )
            book.warn(reason)
            _write_fallback_meme_slots(
                meme_slot_path,
                intro_path,
                teaser_path,
                timeline_path,
                transcript_path,
                timeline_data,
                selected_clip_index,
                reason,
            )
            book.record(
                "meme_analysis",
                "fallback",
                meme_analysis_sig,
                path=meme_slot_path,
                note=reason,
                elapsed=time.perf_counter() - started,
            )

    slots = _meme_slots(meme_slot_path, selected_clip_index)

    # --------------------------------------------------------
    # 12. MEME DISCOVERY
    # --------------------------------------------------------

    meme_discovery_path = Path(
        meme_discovery.get_output_path(meme_slot_path)
    ).resolve()
    started = _stage_start(13, "Automatic meme discovery")

    discovery_sig = _stage_signature(
        "meme_discovery",
        inputs=[meme_slot_path],
        modules=[meme_discovery, model_config],
        options={
            "clip_index": selected_clip_index,
            "slot_count": len(slots),
            "model": model_config.MEME_DISCOVERY_MODEL,
            "reasoning_effort": model_config.MEME_DISCOVERY_REASONING_EFFORT,
        },
    )

    if not enable_memes or not slots:
        reason = (
            "Meme sistemi kapalı; discovery yapılmadı."
            if not enable_memes
            else "Meme Analyzer slot açmadı; web search/download atlandı."
        )
        _write_fallback_discovery(
            meme_discovery_path,
            meme_slot_path,
            timeline_data,
            selected_clip_index,
            reason,
        )
        print("⏭️ Meme discovery skip.")
        book.record(
            "meme_discovery",
            "disabled" if not enable_memes else "skipped",
            discovery_sig,
            path=meme_discovery_path,
            note=reason,
            elapsed=time.perf_counter() - started,
        )
    elif book.reusable(
        "meme_discovery",
        discovery_sig,
        lambda: _contains_clip(
            meme_discovery_path, "clips", selected_clip_index, meme_discovery.DISCOVERY_VERSION
        ),
        output=meme_discovery_path,
        freshness=[
            meme_slot_path,
            *_module_paths([meme_discovery]),
        ],
    ):
        _stage_skip(meme_discovery_path)
        book.record(
            "meme_discovery",
            "skipped",
            discovery_sig,
            path=meme_discovery_path,
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            meme_discovery.discover_memes(
                slot_json_path=meme_slot_path,
                clip_index=selected_clip_index,
            )
            if not _contains_clip(
                meme_discovery_path, "clips", selected_clip_index, meme_discovery.DISCOVERY_VERSION
            ):
                raise RuntimeError("Meme discovery JSON seçilen klibi içermiyor.")

            _stage_done(meme_discovery_path)
            book.record(
                "meme_discovery",
                "done",
                discovery_sig,
                path=meme_discovery_path,
                elapsed=time.perf_counter() - started,
            )
        except Exception as error:
            reason = (
                "Meme discovery başarısız; meme olmadan devam edildi. "
                f"Detay: {error}"
            )
            book.warn(reason)
            _write_fallback_discovery(
                meme_discovery_path,
                meme_slot_path,
                timeline_data,
                selected_clip_index,
                reason,
            )
            book.record(
                "meme_discovery",
                "fallback",
                discovery_sig,
                path=meme_discovery_path,
                note=reason,
                elapsed=time.perf_counter() - started,
            )

    selected_meme_count = _selected_meme_count(
        meme_discovery_path,
        selected_clip_index,
    )

    # --------------------------------------------------------
    # 13. MEME RENDER
    # --------------------------------------------------------

    started = _stage_start(14, "Final meme render")
    final_source = final_preview_path

    meme_render_sig = _stage_signature(
        "meme_render",
        inputs=[meme_discovery_path, final_preview_path, timeline_path],
        modules=[meme_renderer],
        options={
            "clip_index": selected_clip_index,
            "selected_meme_count": selected_meme_count,
        },
    )

    if not enable_memes or selected_meme_count <= 0:
        print("⏭️ Render edilecek meme yok; base preview kullanılıyor.")
        book.record(
            "meme_render",
            "disabled" if not enable_memes else "skipped",
            meme_render_sig,
            path=final_source,
            elapsed=time.perf_counter() - started,
        )
    else:
        try:
            discovery_package = _load_json(meme_discovery_path)
            discovery_clip = meme_renderer.get_discovery_clip(
                discovery_package,
                selected_clip_index,
            )
            expected_meme_final = Path(
                meme_renderer.get_output_path(
                    timeline=timeline_data,
                    discovery_clip=discovery_clip,
                    clip_index=selected_clip_index,
                )
            ).resolve()

            if book.reusable(
                "meme_render",
                meme_render_sig,
                lambda: _valid_file(expected_meme_final, MIN_VIDEO_BYTES),
                output=expected_meme_final,
                freshness=[
                    meme_discovery_path,
                    final_preview_path,
                    timeline_path,
                    *_module_paths([meme_renderer]),
                ],
            ):
                final_source = expected_meme_final
                _stage_skip(final_source)
                book.record(
                    "meme_render",
                    "skipped",
                    meme_render_sig,
                    path=final_source,
                    elapsed=time.perf_counter() - started,
                )
            else:
                rendered = meme_renderer.render_memes(
                    discovery_json_path=meme_discovery_path,
                    clip_index=selected_clip_index,
                    base_video_path=final_preview_path,
                    forbidden_bands=_final_caption_bands(v6_burned_ass),
                )
                if not rendered:
                    raise RuntimeError("Meme renderer çıktı döndürmedi.")

                final_source = Path(rendered[0]).resolve()
                if not _valid_file(final_source, MIN_VIDEO_BYTES):
                    raise RuntimeError("Meme renderer output dosyası geçersiz.")

                _stage_done(final_source)
                book.record(
                    "meme_render",
                    "done",
                    meme_render_sig,
                    path=final_source,
                    elapsed=time.perf_counter() - started,
                )

        except Exception as error:
            reason = (
                "Meme render başarısız; final base preview kullanıldı. "
                f"Detay: {error}"
            )
            book.warn(reason)
            final_source = final_preview_path
            book.record(
                "meme_render",
                "fallback",
                meme_render_sig,
                path=final_source,
                note=reason,
                elapsed=time.perf_counter() - started,
            )

    if final_source != final_preview_path:
        temp_candidates.append(final_source)

    # --------------------------------------------------------
    # 15. FINAL QC -> (one bounded repair) -> PUBLISH or REJECT
    # --------------------------------------------------------
    # Nothing reaches vod_output/final before the RENDERED candidate passed
    # deterministic QC and the gate. The reviewer can trigger at most one
    # deterministic repair round; the repaired candidate is re-checked by QC
    # (never re-reviewed in a loop).

    started = _stage_start(15, "Final QC + publish")
    labels = _verified_names(name_lock_profile, None, ())
    qc_common = dict(
        intro_source=intro_source_path,
        burned_ass=v6_burned_ass,
        truth_path=caption_truth_path,
        protected_paced=[tuple(r) for r in (final_timeline_doc.get("restart") or {}).get("protected_paced") or []],
        display_labels=labels,
    )

    def _effect_windows(candidate: Path) -> list[tuple[float, float]]:
        if candidate == final_preview_path:
            return []
        try:
            report = _load_json(meme_renderer.get_report_path(timeline_data, selected_clip_index))
        except Exception:
            return []
        event = report.get("event") if isinstance(report.get("event"), dict) else None
        if not event:
            return []
        start = float(event.get("start", 0.0))
        return [(start, start + float(event.get("duration", 0.0)))]

    def _evaluate(candidate: Path, main_render: Path, doc: dict[str, Any], *, review: bool
                  ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        context = final_qc.QcContext(candidate=candidate, timeline_doc=doc, main_render=main_render,
                                     effect_windows=_effect_windows(candidate), **qc_common)
        rows = final_qc.run_final_qc(context)
        for row in rows:
            mark = "✅" if row["status"] == "pass" else ("⚠️" if row["status"] == "warn" else "❌")
            print(f"   {mark} {row['check']}: {row['detail']}")
        if not review or any(row["status"] == "fail" for row in rows):
            return rows, None
        beats = [float(b) for b in ((doc.get("intro") or {}).get("peak") or {}).values() if b is not None]
        beats += [float(r[0]) for r in qc_common["protected_paced"]]
        result = final_review.review_final(final_review.ReviewInput(
            candidate=candidate, intro_source=intro_source_path, timeline_doc=doc, headline=headline_text,
            transcript=_truth_text(caption_truth_path), story_beats=beats,
            effect_windows=_effect_windows(candidate)))
        print(f"   🧐 final review: {result.status}" + (f" — {result.summary}" if result.summary else "")
              + (f" ({result.reason})" if result.reason else ""))
        return rows, result.to_dict()

    candidate = final_source
    composed_base = final_preview_path
    main_render_used = captioned_preview_path
    candidate_doc = final_timeline_doc
    qc_rows, review_doc = _evaluate(candidate, main_render_used, candidate_doc, review=True)

    repairs: list[str] = []
    failed_checks = {row["check"] for row in qc_rows if row["status"] == "fail"}
    # An effect is optional presentation: whatever QC failure a candidate WITH an
    # effect shows, the bounded repair is to drop the effect (the base is then
    # re-checked; a failure that was the base's own still rejects).
    if candidate != final_preview_path and failed_checks:
        repairs.append("drop_effects")
    if "intro_headline" in failed_checks and headline_text:
        repairs.append("drop_headline")
    for repair in (review_doc or {}).get("repairs", []) or []:
        if repair not in repairs:
            repairs.append(repair)
    if repairs and review_doc is None:
        # The reviewer only looks at a QC-clean candidate; this one was repaired
        # after QC failed and is re-checked by QC only. Say so in the gate.
        review_doc = {"status": "not_run", "reason": "candidate failed QC before review; the repaired "
                      "candidate is re-checked by QC only", "warnings": [f"not reviewed: QC failed "
                      f"({', '.join(sorted(failed_checks))}) and was repaired"], "repairs": []}
    if repairs:
        print(f"\n🛠️ Bounded repair (tek tur): {', '.join(repairs)}")
        try:
            if "static_camera" in repairs and pro_prep_ref is not None and pro_prep_ref.ready:
                static_path = captioned_preview_path.with_name(captioned_preview_path.stem + "_static"
                                                               + captioned_preview_path.suffix)
                main_render_used, v6_main_proof = pro_edit_pkg.stage.render_static_camera(
                    pro_prep_ref, output_path=static_path)
                temp_candidates.append(main_render_used)
                v6_run.fallback("camera_direction", "final review: subject crop -> static camera repair",
                                "static_camera_repair")
                if intro_source_path != edited_clip_path:
                    # The crop may be in the cold open: it goes back to the clean paced footage too.
                    intro_source_path = edited_clip_path
                    qc_common["intro_source"] = edited_clip_path
                    v6_intro_proof = {"status": "no_camera_ops", "reason": "static camera repair: clean cold open",
                                      "samples": []}
            if "drop_headline" in repairs or "static_camera" in repairs:
                if "drop_headline" in repairs:
                    _write_fallback_intro(intro_path, teaser_path, timeline_path, transcript_path, timeline_data,
                                          selected_clip_index, "bounded repair: headline removed")
                    intro_record = _package_clip(intro_path, "intros", selected_clip_index)
                    headline_text = ""
                rendered = intro_renderer.run_renderer(
                    intro_json_path=intro_path, clip_index=selected_clip_index,
                    edited_clip_path=intro_source_path, captioned_preview_path=main_render_used,
                    caption_path=caption_path)
                final_preview_path = _activate_final_preview(rendered[0])
                candidate_doc = intro_renderer.load_final_timeline(final_preview_path) or candidate_doc
                candidate = final_preview_path
                if "drop_effects" not in repairs and final_source != composed_base:
                    candidate = Path(meme_renderer.render_memes(
                        discovery_json_path=meme_discovery_path, clip_index=selected_clip_index,
                        base_video_path=final_preview_path,
                        forbidden_bands=_final_caption_bands(v6_burned_ass))[0]).resolve()
            elif "drop_effects" in repairs:
                candidate = final_preview_path
            if "drop_effects" in repairs:
                selected_meme_count = 0
                candidate = final_preview_path
                v6_run.fallback("memes", "bounded repair: effect removed after the effect candidate failed "
                                + ", ".join(sorted(failed_checks) or ["review"]), "effect_dropped")
            qc_rows, _ = _evaluate(candidate, main_render_used, candidate_doc, review=False)
            if review_doc is not None:
                review_doc = {**review_doc, "applied_repairs": list(repairs)}
        except Exception as error:
            book.warn(f"Bounded repair başarısız: {type(error).__name__}: {error}")
            qc_rows = [*qc_rows, final_qc._check("bounded_repair", "fail",
                                                 f"{', '.join(repairs)} failed: {type(error).__name__}: {error}")]

    v6_summary = _finish_v6(
        v6_mod=v6_mod,
        run=v6_run,
        pro_edit_pkg=pro_edit_pkg,
        prep=pro_prep_ref,
        book=book,
        state=state,
        published_path=candidate,
        main_render=main_render_used,
        main_proof=v6_main_proof,
        intro_source=intro_source_path if intro_source_path != edited_clip_path else None,
        intro_proof=v6_intro_proof,
        burned_ass=v6_burned_ass,
        render_status=v6_render_status,
        profile_path=name_lock_profile,
        truth_path=caption_truth_path,
        artifact_dir=VOD_OUTPUT_DIR / "pro_edit" / _safe_name(video_path.stem),
        clip_index=selected_clip_index,
        qc_rows=qc_rows,
        review=review_doc,
    )
    gate = dict(v6_run.gate)
    state["final_qc"] = {"checks": qc_rows, "review": review_doc, "repairs": repairs}
    publish_sig = _stage_signature("publish", inputs=[candidate], options={"gate": gate.get("status")})
    if gate.get("status") == v6_runtime.GATE_FAILED:
        rejected = _reject_candidate(candidate, video_path, gate, state)
        book.record("publish", "failed", None, path=rejected, note="final QC rejected the candidate",
                    elapsed=time.perf_counter() - started)
        state["run_status"] = "rejected"
        state["publish_status"] = "rejected"
        book.save()
        failed_names = ", ".join(gate.get("failed") or [])
        raise ShortsPipelineError(
            "Final QC short'u reddetti; YAYINLANMADI. Başarısız kontroller: " + failed_names
            + f"\nİnceleme kopyası: {rejected}\nRapor: {rejected.with_suffix('.qc.json')}"
        )

    published_path = _publish_final(candidate, video_path)
    if not _valid_file(published_path, MIN_VIDEO_BYTES):
        raise ShortsPipelineError("Published final short oluşmadı.")
    _write_json_atomic(published_path.with_suffix(".qc.json"),
                       {"status": gate.get("status"), "gate": gate, "review": review_doc, "repairs": repairs,
                        "source": str(video_path), "published": str(published_path), "at": _now()})
    publish_status = "published" if gate.get("status") == v6_runtime.GATE_PASSED else "published_degraded"
    state["publish_status"] = publish_status
    _stage_done(published_path)
    book.record(
        "publish",
        "done",
        publish_sig,
        path=published_path,
        note=publish_status,
        elapsed=time.perf_counter() - started,
    )

    run_seconds = time.perf_counter() - run_started

    state["final_output"] = str(published_path)
    state["video_brain_source_report"] = (
        str(source_visual_report_path) if source_visual_report_path else None
    )
    state["video_brain_report"] = (
        str(video_brain_report_path) if video_brain_report_path else None
    )
    final_duration = _probe_video_duration(published_path)
    summary_stats = {
        "selected_duration_seconds": round(selected_duration, 3),
        "final_duration_seconds": round(final_duration, 3),
        "intro_duration_seconds": round(teaser_duration, 3),
        "meme_count": int(selected_meme_count),
        "anchor_count": int(anchor_count),
        "protected_range_count": int(protected_count),
        "covered_moment_count": int(selected_covered_count),
        "file_size_mb": round(_file_size_mb(published_path), 3),
    }

    profile = _build_profile(
        state=state,
        run_seconds=run_seconds,
        background_tasks=runtime_profiler.snapshot(),
    )
    state["last_run_seconds"] = round(run_seconds, 3)
    state["summary_stats"] = summary_stats
    state["profile"] = profile
    state["run_status"] = "success"
    book.save()

    removed: list[str] = []

    if not keep_temp:
        removed = _cleanup_temp_files(
            published_path,
            temp_candidates,
        )
        if removed:
            print(f"\n🧹 Temp cleanup: {len(removed)} dosya silindi.")

    result = {
        "success": True,
        "pipeline_version": PIPELINE_VERSION,
        "source_video": str(video_path),
        "selected_clip": {
            "clip_index": selected_clip_index,
            "title": selected_title,
            "score": round(
                selected_score,
                2,
            ),
            "score_policy": "ranking_only_no_threshold",
            "anchor_count": anchor_count,
            "protected_range_count": protected_count,
            "strongest_anchor": strongest_anchor,
            "clip_selection_owner": "Terra",
        },
        "final_output": str(published_path),
        "state_file": str(state_path),
        "video_brain_source_report": (
            str(source_visual_report_path) if source_visual_report_path else None
        ),
        "video_brain_report": (
            str(video_brain_report_path) if video_brain_report_path else None
        ),
        "warnings": list(state.get("warnings", [])),
        "cleanup_removed": removed,
        "run_seconds": round(run_seconds, 3),
        "stats": summary_stats,
        "profile": profile,
        "fast_resume": False,
        "status": state.get("publish_status", "published"),
        "final_qc": state.get("final_qc"),
    }
    if isinstance(state.get("pro_edit"), dict):
        result["pro_edit"] = dict(state["pro_edit"])
    if v6_summary is not None:
        result["v6"] = dict(v6_summary)

    print()
    print("╔════════════════════════════════════════════════════════════╗")
    if state.get("publish_status") == "published_degraded":
        print("║          SHORT HAZIR ⚠️  (QC geçti, DEGRADED uyarılarla)    ║")
    else:
        print("║                     SHORT HAZIR ✅                        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"🏆 {selected_title}")
    print(f"⭐ {selected_score:.1f}/10")
    print(f"⏱️ {run_seconds:.1f}s")
    if profile.get("slowest"):
        print("📈 Slowest stages/workers:")
        for item in profile["slowest"][:6]:
            print(f"   - {item['label']}: {float(item['seconds']):.1f}s")
    print(f"📂 {published_path}")

    if source_visual_report_path:
        print(
            f"👁️ Whole-VOD visual facts: {source_visual_report_path}"
        )

    if video_brain_report_path:
        print(
            f"🧠 Selected-clip visual support: {video_brain_report_path}"
        )

    if result["warnings"]:
        print(f"⚠️ Fallback/uyarı sayısı: {len(result['warnings'])}")

    return result


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bir video/VOD ver, MIMIR tek komutla doğrulanmış bir Short üretsin: hikaye seçimi, "
            "kilitli altyazı doğruluğu, cold open, kamera, efektler ve render edilmiş MP4 üzerinde final QC."
        )
    )
    parser.add_argument(
        "video",
        help="Video yolu veya Downloads/Desktop/Documents içindeki dosya adı.",
    )
    parser.add_argument("--creator", default=None, help="Doğrulanmış creator/streamer adı.")
    parser.add_argument(
        "--clip",
        type=int,
        default=None,
        help="Otomatik seçim yerine belirli clip index kullan.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Cache kullanmadan tüm pipeline'ı yeniden çalıştır.",
    )
    parser.add_argument(
        "--rerender",
        "--force-v6",
        dest="rerender",
        action="store_true",
        help="Yalnız sunum/render/QC/publish aşamalarını yeniden çalıştır; transkripsiyon, hikaye, "
             "caption ASR ve diğer ücretli upstream cache'ler korunur.",
    )
    parser.add_argument(
        "--no-memes",
        action="store_true",
        help="Meme analyzer/discovery/render kullanma.",
    )
    parser.add_argument(
        "--no-video-brain",
        action="store_true",
        help="Bu run için support-only Video Brain'i kapat.",
    )
    parser.add_argument(
        "--video-brain",
        action="store_true",
        help="Config kapalı olsa bile bu run için Video Brain'i aç.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Başarılı final sonrası ağır ara medya dosyalarını silme.",
    )
    parser.add_argument(
        "--terra-visual",
        action="store_true",
        help="Gemini'yi tamamen bypass et; tüm görsel analizleri Terra frame fallback ile yap.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Teknik stage/model çıktılarını göster. Varsayılan çıktı sade tutulur.",
    )

    args = parser.parse_args()

    if args.no_video_brain and args.video_brain:
        parser.error("--video-brain ve --no-video-brain birlikte kullanılamaz.")

    requested_vb: bool | None
    if args.video_brain:
        requested_vb = True
    elif args.no_video_brain:
        requested_vb = False
    else:
        requested_vb = None

    if args.terra_visual:
        if args.no_video_brain:
            parser.error("--terra-visual ve --no-video-brain birlikte kullanılamaz.")
        # video_analyzer is imported lazily, so this is guaranteed to take
        # effect before any Gemini request can be made.
        os.environ["MIMIR_VISUAL_BACKEND"] = "terra"
        requested_vb = True

    print()
    print("🎬 MIMIR videoyu işliyor...")
    print(f"   {args.video}")
    if args.terra_visual:
        print("   Görsel analiz: TERRA (Gemini tamamen bypass)")
    print("   Detaylı teknik çıktı için: --verbose")

    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    kwargs = dict(
        video_path=args.video,
        creator_name=args.creator,
        force=args.force,
        clip_index=args.clip,
        enable_memes=not args.no_memes,
        enable_video_brain=requested_vb,
        keep_temp=args.keep_temp,
        rerender=args.rerender,
    )

    try:
        if args.verbose:
            result = run_pipeline(**kwargs)
        else:
            with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
                result = run_pipeline(**kwargs)
    except NoStrongClipError as error:
        print()
        print("❌ Güçlü bir Short bulunamadı.")
        print(f"   {error}")
        raise SystemExit(2)
    except Exception as error:
        print()
        print("❌ MIMIR işlemi tamamlayamadı (hiçbir şey yayınlanmadı).")
        print(f"   {error}")
        print("   Daha fazla detay için aynı komutu --verbose ile çalıştır.")
        raise SystemExit(1)

    _print_friendly_result(result)
    if result.get("status") == "published_degraded":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
