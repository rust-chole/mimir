"""Render backend: ONE shared H.264 encoder choice for every real MIMIR video encode.

Hardware encoding is an optimization, never a correctness authority. MIMIR keeps
every filter where it is (trim/concat, perspective, libass subtitles, overlay,
scale stay CPU filters) and only swaps the ENCODER at the end of each graph:

    CPU filters -> h264_nvenc | h264_amf | h264_qsv | h264_videotoolbox | libx264

Detection asks the INSTALLED FFmpeg, never the GPU vendor name: an encoder must be
listed by ``ffmpeg -encoders`` AND pass a tiny real encode (lavfi test pattern ->
temporary MP4 -> ffprobe: H.264, yuv420p, every frame present). The result is
cached for the process; no render stage repeats discovery.

``MIMIR_RENDER_BACKEND`` (default ``auto``): auto | cpu | nvenc | amf | qsv |
videotoolbox. An unavailable explicit backend falls back to CPU (recorded).

At run time a hardware encode that fails (FFmpeg error, timeout, or the
renderer's own output contract) marks that backend unhealthy for the rest of
the run and the SAME render is retried once with libx264. Later stages never
retry the broken encoder. The final QC gates stay the only authority.

Render-stage cache signatures carry ``signature_payload`` / ``stage_payload``
(backend, encoder, profile, quality, pixel format, exact arguments, FFmpeg build
for hardware), so a libx264 render is never reused as an NVENC render. Bump
``RENDER_PROFILE_VERSION`` whenever a profile's meaning changes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

T = TypeVar("T")

ENV_VAR = "MIMIR_RENDER_BACKEND"
DEFAULT_MODE = "auto"
RENDER_PROFILE_VERSION = 1

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

CPU = "cpu"
NVENC = "nvenc"
AMF = "amf"
QSV = "qsv"
VIDEOTOOLBOX = "videotoolbox"

MODES = (DEFAULT_MODE, CPU, NVENC, AMF, QSV, VIDEOTOOLBOX)
# Discrete vendor encoders first; the first one that ACTUALLY encodes wins.
AUTO_PRIORITY = (NVENC, AMF, QSV, VIDEOTOOLBOX)

ENCODERS = {
    CPU: "libx264",
    NVENC: "h264_nvenc",
    AMF: "h264_amf",
    QSV: "h264_qsv",
    VIDEOTOOLBOX: "h264_videotoolbox",
}
LABELS = {
    CPU: "CPU",
    NVENC: "NVIDIA NVENC",
    AMF: "AMD AMF",
    QSV: "Intel Quick Sync",
    VIDEOTOOLBOX: "Apple VideoToolbox",
}
_ALIASES = {
    "": DEFAULT_MODE, "default": DEFAULT_MODE, "gpu": DEFAULT_MODE, "hw": DEFAULT_MODE, "hardware": DEFAULT_MODE,
    "software": CPU, "x264": CPU, "libx264": CPU, "none": CPU, "off": CPU,
    "nvidia": NVENC, "h264_nvenc": NVENC,
    "amd": AMF, "h264_amf": AMF,
    "intel": QSV, "quicksync": QSV, "h264_qsv": QSV,
    "apple": VIDEOTOOLBOX, "vt": VIDEOTOOLBOX, "h264_videotoolbox": VIDEOTOOLBOX,
}

# Smoke test: a few frames of a moving portrait test pattern (never user media).
SMOKE_WIDTH = 320
SMOKE_HEIGHT = 576
SMOKE_RATE = 30
SMOKE_FRAMES = 12
SMOKE_TIMEOUT_S = 30.0
PROBE_TIMEOUT_S = 30.0


# ============================================================
# ENCODER PROFILES
# ============================================================

@dataclass(frozen=True)
class EncoderProfile:
    """One concrete set of FFmpeg video-encoder arguments."""

    backend: str
    tier: str
    quality: str
    pix_fmt: str
    options: tuple[str, ...]

    @property
    def encoder(self) -> str:
        return ENCODERS[self.backend]

    @property
    def is_hardware(self) -> bool:
        return self.backend != CPU

    @property
    def label(self) -> str:
        return LABELS[self.backend]

    @property
    def profile_id(self) -> str:
        return f"{self.backend}_{self.tier}"

    def video_args(self) -> list[str]:
        """``-c:v <encoder> <options> -pix_fmt <fmt>``; audio arguments stay with the renderer."""
        return ["-c:v", self.encoder, *self.options, "-pix_fmt", self.pix_fmt]

    def signature_payload(self, ffmpeg_version: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {
            "backend": self.backend,
            "encoder": self.encoder,
            "class": "hardware" if self.is_hardware else "cpu",
            "profile": self.profile_id,
            "profile_version": RENDER_PROFILE_VERSION,
            "quality": self.quality,
            "pix_fmt": self.pix_fmt,
            "args": self.video_args(),
        }
        if self.is_hardware:
            # Hardware encoders depend on the FFmpeg build (SDK/driver glue); libx264
            # keeps the existing CPU render identity.
            payload["ffmpeg"] = ffmpeg_version
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "encoder": self.encoder, "label": self.label, "tier": self.tier,
                "quality": self.quality, "pix_fmt": self.pix_fmt, "args": self.video_args()}


# The universal fallback: exactly the encoder settings every MIMIR renderer used
# before this module (libx264, preset fast, CRF 18, yuv420p).
CPU_PROFILE = EncoderProfile(CPU, "x264", "crf18_preset_fast", "yuv420p", ("-preset", "fast", "-crf", "18"))

# Per backend: a high-quality profile, then ONE simpler compatible profile for older
# FFmpeg builds / drivers. Quality targets the libx264 CRF 18 class with each
# encoder's own rate control (hardware QP scales are not CRF scales).
HARDWARE_PROFILES: dict[str, tuple[EncoderProfile, ...]] = {
    NVENC: (
        EncoderProfile(NVENC, "hq", "vbr_cq19", "yuv420p", (
            "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
            "-spatial-aq", "1", "-rc-lookahead", "20", "-bf", "3", "-profile:v", "high")),
        EncoderProfile(NVENC, "compat", "vbr_cq19", "yuv420p", (
            "-rc", "vbr", "-cq", "19", "-b:v", "0", "-profile:v", "high")),
    ),
    AMF: (
        EncoderProfile(AMF, "hq", "cqp_i18_p19", "yuv420p", (
            "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "19", "-bf", "0", "-profile:v", "high")),
        EncoderProfile(AMF, "compat", "cqp_i18_p19", "yuv420p", (
            "-rc", "cqp", "-qp_i", "18", "-qp_p", "19")),
    ),
    QSV: (
        # QSV takes NV12 (4:2:0 8-bit, decoded as yuv420p); the scale is a cheap CPU conversion.
        EncoderProfile(QSV, "hq", "icq19", "nv12", (
            "-preset", "slow", "-global_quality", "19", "-profile:v", "high")),
        EncoderProfile(QSV, "compat", "cqp19", "nv12", ("-q:v", "19")),
    ),
    VIDEOTOOLBOX: (
        # Constant quality only (no fixed-bitrate guess): without it VideoToolbox is
        # unusable here and the CPU encoder is used.
        EncoderProfile(VIDEOTOOLBOX, "hq", "cq65", "yuv420p", ("-q:v", "65", "-profile:v", "high")),
        EncoderProfile(VIDEOTOOLBOX, "compat", "cq65", "yuv420p", ("-q:v", "65")),
    ),
}


# ============================================================
# FFMPEG CAPABILITIES
# ============================================================

@dataclass(frozen=True)
class RenderCapabilities:
    probed: bool
    ffmpeg_available: bool
    version: str
    h264_encoders: tuple[str, ...] = ()
    hwaccels: tuple[str, ...] = ()
    error: str = ""

    def lists(self, backend: str) -> bool:
        return ENCODERS[backend] in self.h264_encoders

    def to_dict(self) -> dict[str, Any]:
        return {"probed": self.probed, "ffmpeg_available": self.ffmpeg_available, "version": self.version,
                "h264_encoders": list(self.h264_encoders), "hwaccels": list(self.hwaccels), "error": self.error}


NOT_PROBED = RenderCapabilities(probed=False, ffmpeg_available=False, version="not probed")


@dataclass(frozen=True)
class SmokeResult:
    ok: bool
    detail: str
    seconds: float = 0.0


def _run(args: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """Argument array (never a shell), UTF-8 output; never raises."""
    try:
        completed = subprocess.run(
            list(args), stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except FileNotFoundError:
        return 127, "", f"{args[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f}s"
    except OSError as error:
        return 126, "", f"{type(error).__name__}: {error}"
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _tail(text: str, limit: int = 220) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    tail = lines[-1] if lines else ""
    return tail if len(tail) <= limit else tail[: limit - 3] + "..."


def _first_error(text: str, limit: int = 220) -> str:
    """FFmpeg's first stderr line names the cause (e.g. ``Cannot load libcuda.so.1``)."""
    lines = [re.sub(r" @ 0x[0-9a-fA-F]+", "", line.strip()) for line in str(text or "").splitlines() if line.strip()]
    first = lines[0] if lines else ""
    return first if len(first) <= limit else first[: limit - 3] + "..."


_CAUSE = re.compile(r"error|unknown|cannot|failed|invalid|unsupported|not supported|not found|no such|"
                    r"out of memory|no device|incompatible|mismatch|timed out", re.IGNORECASE)
_BANNER = re.compile(r"^(ffmpeg version|built with|configuration:|lib(av|sw|post)[a-z]*\s)", re.IGNORECASE)


def describe_error(error: BaseException | str, limit: int = 600) -> str:
    """The renderer's message plus the FFmpeg lines that name the cause (never the version banner)."""
    text = error if isinstance(error, str) else str(error)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    head = lines[0] if lines else ""
    causes: list[str] = []
    for line in lines[1:]:
        cleaned = re.sub(r" @ 0x[0-9a-fA-F]+", "", line)
        if _BANNER.match(cleaned) or not _CAUSE.search(cleaned) or cleaned in causes:
            continue
        causes.append(cleaned)
    prefix = "" if isinstance(error, str) else f"{type(error).__name__}: "
    summary = prefix + head + ("" if not causes else " | " + " | ".join(causes[:3]))
    return summary if len(summary) <= limit else summary[: limit - 3] + "..."


def parse_encoders(text: str) -> tuple[str, ...]:
    """Video encoder names relevant to H.264 from ``ffmpeg -encoders``."""
    names: list[str] = []
    for line in str(text or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0] or parts[0][0] != "V" or "=" in parts[0]:
            continue
        name = parts[1]
        if "264" in name and name not in names:
            names.append(name)
    return tuple(names)


def parse_hwaccels(text: str) -> tuple[str, ...]:
    names: list[str] = []
    started = False
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("hardware acceleration methods"):
            started = True
            continue
        if started and stripped not in names:
            names.append(stripped)
    return tuple(names)


def probe_capabilities() -> RenderCapabilities:
    code, out, err = _run([FFMPEG, "-hide_banner", "-version"], PROBE_TIMEOUT_S)
    if code != 0:
        return RenderCapabilities(probed=True, ffmpeg_available=False, version="unavailable",
                                  error=_tail(err or out) or f"exit {code}")
    version = out.strip().splitlines()[0].strip() if out.strip() else "unknown"
    code, enc_out, enc_err = _run([FFMPEG, "-hide_banner", "-encoders"], PROBE_TIMEOUT_S)
    encoders = parse_encoders(enc_out) if code == 0 else ()
    error = "" if code == 0 else f"-encoders failed: {_tail(enc_err)}"
    code, hw_out, _ = _run([FFMPEG, "-hide_banner", "-hwaccels"], PROBE_TIMEOUT_S)
    hwaccels = parse_hwaccels(hw_out) if code == 0 else ()
    return RenderCapabilities(probed=True, ffmpeg_available=True, version=version, h264_encoders=encoders,
                              hwaccels=hwaccels, error=error)


def smoke_test(profile: EncoderProfile) -> SmokeResult:
    """Prove the exact profile encodes: lavfi pattern -> temp MP4 -> ffprobe; nothing is left behind."""
    started = time.perf_counter()

    def done(ok: bool, detail: str) -> SmokeResult:
        return SmokeResult(ok, detail, round(time.perf_counter() - started, 3))

    with tempfile.TemporaryDirectory(prefix="mimir_render_probe_") as tmp:
        output = Path(tmp) / "probe.mp4"
        code, _, err = _run([
            FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={SMOKE_WIDTH}x{SMOKE_HEIGHT}:rate={SMOKE_RATE}",
            "-frames:v", str(SMOKE_FRAMES), "-an", *profile.video_args(),
            "-movflags", "+faststart", str(output),
        ], SMOKE_TIMEOUT_S)
        if code != 0:
            return done(False, _first_error(err) or f"exit {code}")
        if not output.is_file() or output.stat().st_size <= 0:
            return done(False, "encoder produced no output")
        code, out, err = _run([
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames", "-of", "json", str(output),
        ], PROBE_TIMEOUT_S)
        if code != 0:
            return done(False, f"ffprobe: {_first_error(err) or f'exit {code}'}")
        try:
            streams = json.loads(out or "{}").get("streams") or []
            stream = streams[0] if streams else {}
        except (ValueError, AttributeError, IndexError):
            return done(False, "ffprobe returned invalid JSON")
        problems = []
        if stream.get("codec_name") != "h264":
            problems.append(f"codec {stream.get('codec_name')!r}")
        if stream.get("pix_fmt") != "yuv420p":
            problems.append(f"pix_fmt {stream.get('pix_fmt')!r}")
        if (stream.get("width"), stream.get("height")) != (SMOKE_WIDTH, SMOKE_HEIGHT):
            problems.append(f"size {stream.get('width')}x{stream.get('height')}")
        frames = str(stream.get("nb_frames", "") or "")
        if frames.isdigit() and int(frames) != SMOKE_FRAMES:
            problems.append(f"{frames}/{SMOKE_FRAMES} frames")
        if problems:
            return done(False, "output contract: " + ", ".join(problems))
    return done(True, "passed")


# ============================================================
# SELECTION
# ============================================================

@dataclass(frozen=True)
class BackendSelection:
    requested: str
    mode: str
    profile: EncoderProfile
    reason: str
    hardware_test: str
    capabilities: RenderCapabilities
    candidates: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    detection_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested, "mode": self.mode, "selected": self.profile.to_dict(),
            "reason": self.reason, "hardware_test": self.hardware_test,
            "ffmpeg": self.capabilities.to_dict(), "candidates": [dict(row) for row in self.candidates],
            "notes": list(self.notes), "detection_seconds": round(self.detection_seconds, 3),
        }


def normalize_mode(requested: str | None) -> tuple[str, tuple[str, ...]]:
    raw = str(requested if requested is not None else DEFAULT_MODE).strip().lower()
    if raw in MODES:
        return raw, ()
    if raw in _ALIASES:
        return _ALIASES[raw], ()
    return DEFAULT_MODE, (f"unknown {ENV_VAR}={requested!r}; using auto",)


def choose_backend(
    requested: str | None,
    probe: Callable[[], RenderCapabilities],
    tester: Callable[[EncoderProfile], SmokeResult],
) -> BackendSelection:
    """Pure selection: what the installed FFmpeg can ACTUALLY encode wins over any vendor guess."""
    started = time.perf_counter()
    raw = str(requested if requested is not None else DEFAULT_MODE)
    mode, notes = normalize_mode(requested)

    def result(profile: EncoderProfile, reason: str, test: str, caps: RenderCapabilities,
               rows: Iterable[dict[str, Any]] = ()) -> BackendSelection:
        return BackendSelection(raw, mode, profile, reason, test, caps, tuple(rows), notes,
                                time.perf_counter() - started)

    if mode == CPU:
        return result(CPU_PROFILE, f"{ENV_VAR}=cpu (explicit)", "not_run", NOT_PROBED)
    caps = probe()
    if not caps.ffmpeg_available:
        return result(CPU_PROFILE, f"FFmpeg unavailable ({caps.error or 'not found'}); CPU encoder kept",
                      "not_run", caps)
    order = AUTO_PRIORITY if mode == DEFAULT_MODE else (mode,)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for backend in order:
        encoder = ENCODERS[backend]
        if not caps.lists(backend):
            rows.append({"backend": backend, "encoder": encoder, "listed": False,
                         "result": "not in this FFmpeg build"})
            if mode != DEFAULT_MODE:
                failures.append(f"{encoder} is not in this FFmpeg build")
            continue
        attempts: list[dict[str, Any]] = []
        for profile in HARDWARE_PROFILES[backend]:
            smoke = tester(profile)
            attempts.append({"tier": profile.tier, "ok": smoke.ok, "detail": smoke.detail, "seconds": smoke.seconds})
            if smoke.ok:
                rows.append({"backend": backend, "encoder": encoder, "listed": True, "result": "usable",
                             "tier": profile.tier, "attempts": attempts})
                for rest in order[order.index(backend) + 1:]:
                    if caps.lists(rest):
                        rows.append({"backend": rest, "encoder": ENCODERS[rest], "listed": True,
                                     "result": "not tested (a higher-priority encoder works)"})
                why = (f"{encoder} passed the hardware smoke test"
                       + ("" if profile.tier == "hq" else f" ({profile.tier} profile)"))
                return result(profile, why if mode == DEFAULT_MODE else f"{ENV_VAR}={mode}: {why}",
                              "passed", caps, rows)
        rows.append({"backend": backend, "encoder": encoder, "listed": True, "result": "smoke test failed",
                     "attempts": attempts})
        failures.append(f"{encoder}: {attempts[-1]['detail']}")
    tested = any(row.get("listed") for row in rows)
    detail = "; ".join(failures)
    if mode == DEFAULT_MODE:
        reason = ("no usable hardware H.264 encoder" + (f" ({detail})" if detail else "")) if tested \
            else "no hardware H.264 encoder in this FFmpeg build"
    else:
        reason = f"{ENV_VAR}={mode} unavailable ({detail or 'not usable'}); CPU fallback"
    return result(CPU_PROFILE, reason, "failed" if tested else "not_run", caps, rows)


# ============================================================
# PROCESS STATE (detection once per process; health + usage per run)
# ============================================================

@dataclass
class EncodeEvent:
    stage: str
    profile: EncoderProfile
    ok: bool
    seconds: float
    fallback_from: str = ""
    error: str = ""
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict[str, Any]:
        row = {"stage": self.stage, "backend": self.profile.backend, "encoder": self.profile.encoder,
               "tier": self.profile.tier, "ok": self.ok, "seconds": round(self.seconds, 3), "at": self.at}
        if self.fallback_from:
            row["fallback_from"] = self.fallback_from
        if self.error:
            row["error"] = self.error
        return row


_LOCK = threading.RLock()
_CAPABILITIES: RenderCapabilities | None = None
_SMOKE: dict[EncoderProfile, SmokeResult] = {}
_SELECTION: BackendSelection | None = None
_UNHEALTHY: dict[str, dict[str, Any]] = {}
_EVENTS: list[EncodeEvent] = []
_PRINTED: str | None = None


def _cached_capabilities() -> RenderCapabilities:
    global _CAPABILITIES
    with _LOCK:
        if _CAPABILITIES is None:
            _CAPABILITIES = probe_capabilities()
        return _CAPABILITIES


def _cached_smoke(profile: EncoderProfile) -> SmokeResult:
    with _LOCK:
        if profile not in _SMOKE:
            _SMOKE[profile] = smoke_test(profile)
        return _SMOKE[profile]


def get_selection() -> BackendSelection:
    """The selected backend for the current ``MIMIR_RENDER_BACKEND`` (detected once per process)."""
    global _SELECTION
    requested = os.environ.get(ENV_VAR, DEFAULT_MODE)
    with _LOCK:
        if _SELECTION is None or _SELECTION.requested != requested:
            _SELECTION = choose_backend(requested, _cached_capabilities, _cached_smoke)
        return _SELECTION


def reset(*, detection: bool = True) -> None:
    """Forget per-run health/usage (and, by default, the cached detection). Tests use this."""
    global _CAPABILITIES, _SELECTION, _PRINTED
    with _LOCK:
        _UNHEALTHY.clear()
        _EVENTS.clear()
        if detection:
            _CAPABILITIES = None
            _SELECTION = None
            _SMOKE.clear()
            _PRINTED = None


def begin_run() -> BackendSelection:
    """A new pipeline run: health marks and usage start empty; detection stays cached."""
    reset(detection=False)
    return get_selection()


def cpu_profile() -> EncoderProfile:
    return CPU_PROFILE


def active_profile() -> EncoderProfile:
    """The selected profile unless its backend failed earlier in this run (then libx264)."""
    selection = get_selection()
    with _LOCK:
        if selection.profile.is_hardware and selection.profile.backend in _UNHEALTHY:
            return CPU_PROFILE
    return selection.profile


def mark_unhealthy(backend: str, stage: str, error: BaseException | str) -> None:
    with _LOCK:
        _UNHEALTHY.setdefault(backend, {"stage": stage, "error": describe_error(error),
                                        "at": time.strftime("%Y-%m-%dT%H:%M:%S")})


def unhealthy() -> dict[str, dict[str, Any]]:
    with _LOCK:
        return {key: dict(value) for key, value in _UNHEALTHY.items()}


def _record(event: EncodeEvent) -> None:
    with _LOCK:
        _EVENTS.append(event)


class HardwareOutputError(RuntimeError):
    """A hardware encode finished but its file breaks the output contract."""


def check_output(path: str | Path, profile: EncoderProfile) -> None:
    """Hardware encodes must produce the same stream class as libx264: H.264, yuv420p.

    No-op for the CPU profile (its existing behavior is the reference)."""
    if not profile.is_hardware:
        return
    code, out, err = _run([
        FFPROBE, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(path),
    ], PROBE_TIMEOUT_S)
    if code != 0:
        raise HardwareOutputError(f"{profile.encoder} output unreadable: {_first_error(err) or f'exit {code}'}")
    try:
        streams = json.loads(out or "{}").get("streams") or []
        stream = streams[0] if streams else {}
    except (ValueError, AttributeError):
        stream = {}
    if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
        raise HardwareOutputError(
            f"{profile.encoder} output is {stream.get('codec_name')!r}/{stream.get('pix_fmt')!r}, "
            "expected 'h264'/'yuv420p'")


def run_encode(stage: str, attempt: Callable[[EncoderProfile], T]) -> T:
    """Run one real encode with the active profile; a hardware failure retries ONCE on libx264.

    ``attempt(profile)`` builds the FFmpeg command from ``profile.video_args()``,
    runs it and applies the renderer's own output checks, raising on failure.
    The failing hardware backend is disabled for the rest of the run, so later
    stages go straight to the CPU encoder. A CPU failure propagates unchanged
    (existing stage-level fallbacks keep their meaning)."""
    print_summary_once()
    profile = active_profile()
    started = time.perf_counter()
    try:
        result = attempt(profile)
    except Exception as error:
        _record(EncodeEvent(stage, profile, False, time.perf_counter() - started, error=describe_error(error)))
        if not profile.is_hardware:
            raise
        mark_unhealthy(profile.backend, stage, error)
        print(f"⚠️ Render backend: {profile.encoder} failed in {stage}; it is disabled for the rest of this run "
              f"and the render is retried once with {CPU_PROFILE.encoder}. Detail: {describe_error(error)}")
        retry_started = time.perf_counter()
        try:
            result = attempt(CPU_PROFILE)
        except Exception as cpu_error:
            _record(EncodeEvent(stage, CPU_PROFILE, False, time.perf_counter() - retry_started,
                                fallback_from=profile.backend, error=describe_error(cpu_error)))
            raise
        _record(EncodeEvent(stage, CPU_PROFILE, True, time.perf_counter() - retry_started,
                            fallback_from=profile.backend))
        return result
    _record(EncodeEvent(stage, profile, True, time.perf_counter() - started))
    return result


# ============================================================
# SIGNATURES
# ============================================================

def signature_payload(profile: EncoderProfile | None = None) -> dict[str, Any]:
    """Stable render identity of ``profile`` (default: the active profile)."""
    chosen = profile or active_profile()
    version = get_selection().capabilities.version if chosen.is_hardware else ""
    return chosen.signature_payload(version)


def stage_payload(*stages: str) -> dict[str, Any]:
    """Identity of what ACTUALLY encoded ``stages`` in this run.

    No encode recorded -> the active profile. One profile -> its payload. A stage
    finished by more than one encoder is ``mixed`` and never matches a later
    single-backend signature (its output is re-rendered next time)."""
    with _LOCK:
        used: dict[str, EncoderProfile] = {}
        for event in _EVENTS:
            if event.ok and event.stage in stages:
                used[event.profile.profile_id] = event.profile
    if not used:
        return signature_payload()
    if len(used) == 1:
        return signature_payload(next(iter(used.values())))
    return {"mixed": sorted(used), "profile_version": RENDER_PROFILE_VERSION}


def run_identity() -> dict[str, Any]:
    """Run-level identity for the fast-resume request signature."""
    selection = get_selection()
    return {"mode": selection.mode, "selected": signature_payload(selection.profile),
            "degraded": sorted(unhealthy())}


# ============================================================
# REPORTING
# ============================================================

def report() -> dict[str, Any]:
    selection = get_selection()
    active = active_profile()
    with _LOCK:
        events = list(_EVENTS)
        broken = {key: dict(value) for key, value in _UNHEALTHY.items()}
    stages: dict[str, dict[str, Any]] = {}
    for event in events:
        row = stages.setdefault(event.stage, {"backends": [], "encoders": [], "encodes": 0, "failed_attempts": 0,
                                              "seconds": 0.0, "fallback": False})
        row["seconds"] = round(row["seconds"] + event.seconds, 3)
        if event.ok:
            row["encodes"] += 1
            if event.profile.backend not in row["backends"]:
                row["backends"].append(event.profile.backend)
                row["encoders"].append(event.profile.encoder)
        else:
            row["failed_attempts"] += 1
        if event.fallback_from:
            row["fallback"] = True
            row["fallback_from"] = event.fallback_from
            failed = next((e for e in events if e.stage == event.stage and not e.ok
                           and e.profile.backend == event.fallback_from), None)
            if failed is not None:
                row["fallback_reason"] = failed.error
    for row in stages.values():
        backends = row["backends"]
        row["backend"] = backends[0] if len(backends) == 1 else ("mixed" if backends else "failed")
        row["hardware"] = bool(backends) and all(name != CPU for name in backends)
    fallback = bool(broken) or any(event.fallback_from for event in events)
    return {
        "requested": selection.requested,
        "mode": selection.mode,
        "selected": selection.profile.to_dict(),
        "selection_reason": selection.reason,
        "hardware_test": selection.hardware_test,
        "active": active.to_dict(),
        "cpu_fallback_encoder": CPU_PROFILE.encoder,
        "fallback_occurred": fallback,
        "fallback_reason": "; ".join(f"{name} failed in {row['stage']}: {row['error']}" for name, row in broken.items()),
        "unhealthy": broken,
        "stages": stages,
        "encodes": [event.to_dict() for event in events],
        "detection": selection.to_dict(),
        "profile_version": RENDER_PROFILE_VERSION,
    }


def summary_lines(selection: BackendSelection | None = None) -> list[str]:
    selection = selection or get_selection()
    profile = selection.profile
    lines = [f"Render backend : {profile.label}",
             f"Encoder        : {profile.encoder}" + (f" ({profile.tier} profile)" if profile.is_hardware else ""),
             f"Mode           : {selection.mode}"]
    if profile.is_hardware:
        lines.append(f"CPU fallback   : {CPU_PROFILE.encoder}")
        lines.append(f"Hardware test  : {selection.hardware_test}")
    else:
        lines.append(f"Reason         : {selection.reason}")
    lines.extend(f"Note           : {note}" for note in selection.notes)
    return lines


def print_summary_once(stream: Any = None) -> None:
    """One concise render summary per process (and per backend selection)."""
    global _PRINTED
    selection = get_selection()
    key = f"{selection.requested}|{selection.profile.profile_id}"
    with _LOCK:
        if _PRINTED == key:
            return
        _PRINTED = key
    out = stream if stream is not None else sys.stdout
    for line in summary_lines(selection):
        print(f"🎞️ {line}", file=out)


def describe_usage(data: Mapping[str, Any] | None) -> str:
    """One line for the final console result: selected backend plus any fallback."""
    if not isinstance(data, Mapping) or not data:
        return ""
    selected = data.get("selected") if isinstance(data.get("selected"), Mapping) else {}
    text = f"{selected.get('label', '?')} ({selected.get('encoder', '?')})"
    if data.get("fallback_occurred"):
        text += f" -> {data.get('cpu_fallback_encoder', CPU_PROFILE.encoder)} after a hardware failure"
    return text
