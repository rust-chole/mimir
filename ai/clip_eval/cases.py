"""Clip-selection cases: one source video, its frozen Stage 1 inputs and the blind human labels.

One directory per source video under ``<root>/cases/<case_id>/``:

    meta.json                 split (dev | holdout), the source video (path, SHA256, size, duration) and
                              where the frozen inputs came from
    transcript.json           frozen copy of the cached Stage 1 transcript (vod_output/transcripts/<stem>.json)
    visual_report.json        frozen copy of the whole-VOD visual-facts report Stage 2 receives in
                              production (vod_output/video_brain/<stem>_video_report_v6.json), when one exists
    labels/<annotator>.json   THE HUMAN LABELS: 2-6 good moments (great / ok / bad, with a note) and
                              optional never-pick ranges. Only finished labels are scored.
    proxy.mp4                 optional small browser-playable copy for the editor (``prepare --proxy``)

Labeling is blind: nothing in this module (or the editor and the sheet built on it) reads, copies
or shows MIMIR's clip picks (``vod_output/analysis``), its published shorts, harvested reviews or
any other model output. Inputs are frozen so that every run of a case sees the same bytes even
after the pipeline re-transcribes the video. ``prepare`` calls no model.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.caption_eval import cases as caption_cases

PROJECT_ROOT = caption_cases.PROJECT_ROOT
DEFAULT_ROOT = PROJECT_ROOT / "vod_output" / "clip_eval"
# Mirrors ai.vod_processor.TRANSCRIPT_DIR and ai.video_brain.config.VIDEO_BRAIN_OUTPUT_DIR (tests pin both)
# without importing them: importing those modules builds API clients and output folders.
PIPELINE_TRANSCRIPT_DIR = PROJECT_ROOT / "vod_output" / "transcripts"
PIPELINE_VIDEO_BRAIN_DIR = PROJECT_ROOT / "vod_output" / "video_brain"

SPLITS = ("dev", "holdout")
GRADES = ("great", "ok", "bad")
DEFAULT_ANNOTATOR = "primary"
MIN_MOMENTS, MAX_MOMENTS = 2, 6
LABELS_VERSION = 1
NOTE_MAX = 300

_write_json = caption_cases._write_json
_read_json = caption_cases._read_json
_now = caption_cases._now

# Letters NFKD does not decompose into ASCII (Turkish dotless i, German sharp s, ...).
_TRANSLIT = str.maketrans({"ı": "i", "İ": "I", "ß": "ss", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O",
                           "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "þ": "th", "Þ": "Th"})


def eval_root(root: str | Path | None = None) -> Path:
    return Path(root or os.getenv("MIMIR_CLIP_EVAL_DIR", "") or DEFAULT_ROOT).resolve()


def case_slug(text: str) -> str:
    """ASCII case id from a file name: 'Kai'nin şaka İzmir' -> 'kai_nin_saka_izmir'."""
    ascii_text = unicodedata.normalize("NFKD", str(text).translate(_TRANSLIT))
    ascii_text = "".join(ch for ch in ascii_text if not unicodedata.combining(ch))
    return caption_cases.slug(ascii_text)


def ps_quote(value: str | Path) -> str:
    """A path quoted for PowerShell (double quotes unless the text holds $, ` or ")."""
    text = str(value)
    if any(ch in text for ch in '$`"'):
        return "'" + text.replace("'", "''") + "'"
    return f'"{text}"'


def clock(seconds: float | None) -> str:
    """H:MM:SS.s (or M:SS.s under an hour) on the source video's clock."""
    if seconds is None:
        return "--:--"
    total = max(0.0, float(seconds))
    hours, rest = divmod(total, 3600.0)
    minutes, secs = divmod(rest, 60.0)
    if secs >= 59.95:                                   # 59.97 must not print as "60.0"
        minutes, secs = minutes + 1, 0.0
        if minutes >= 60:
            hours, minutes = hours + 1, 0
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{secs:04.1f}"
    return f"{int(minutes)}:{secs:04.1f}"


def parse_clock(text: str | float | int) -> float:
    """Seconds from '1:02:03.5', '2:03.5', '123.5' (or a number)."""
    if isinstance(text, (int, float)):
        return float(text)
    raw = str(text).strip().replace(",", ".")
    if not raw:
        raise ValueError("empty time")
    parts = raw.split(":")
    if len(parts) > 3:
        raise ValueError(f"not a time: {text!r}")
    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"not a time: {text!r}") from None
    if any(v < 0 for v in values) or any(v >= 60 for v in values[1:]):
        raise ValueError(f"not a time: {text!r}")
    seconds = 0.0
    for value in values:
        seconds = seconds * 60.0 + value
    return seconds


# ============================================================
# FINGERPRINTS
# ============================================================

def file_sha256(path: str | Path, *, root: str | Path | None = None) -> str:
    """SHA256 of a file, cached by (path, size, mtime) under <root>/.hash_cache.json (VODs are large)."""
    path = Path(path).resolve()
    stat = path.stat()
    cache_path = eval_root(root) / ".hash_cache.json"
    cache = _read_json(cache_path)
    cache = cache if isinstance(cache, dict) else {}
    key = str(path)
    entry = cache.get(key)
    if isinstance(entry, Mapping) and entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
        return str(entry["sha256"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": value}
    _write_json(cache_path, cache)
    return value


def bytes_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probe_duration(path: str | Path) -> float | None:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=60,
                             encoding="utf-8", errors="replace", check=False)
        return round(float(out.stdout.strip()), 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def default_transcript_path(video: str | Path) -> Path:
    """Where Stage 1 writes the transcript of this video (``vod_processor.save_transcript``)."""
    return (PIPELINE_TRANSCRIPT_DIR / f"{Path(video).stem}.json").resolve()


def default_visual_report_path(video: str | Path) -> Path:
    """Where the whole-VOD visual-facts report of this video lives (``video_analyzer.get_output_path``)."""
    stem = Path(video).stem
    for char in '<>:"/\\|?*':
        stem = stem.replace(char, "_")
    stem = " ".join(stem.split()).strip(" ._") or "video"
    return (PIPELINE_VIDEO_BRAIN_DIR / f"{stem}_video_report_v6.json").resolve()


def transcribe_command(video: str | Path) -> str:
    """The one-time (paid) Stage 1 command for a video, with its cost cap."""
    return f"python -m ai.clip_eval transcribe {ps_quote(Path(video).resolve())} --dry-run   # then --max-usd <cap>"


# ============================================================
# LABELS
# ============================================================

@dataclass(frozen=True)
class Moment:
    start: float
    end: float
    grade: str = "great"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "grade": self.grade, "note": self.note}


@dataclass(frozen=True)
class NeverPick:
    start: float
    end: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "note": self.note}


@dataclass
class Labels:
    annotator: str
    finished: bool = False
    moments: list[Moment] = field(default_factory=list)
    negatives: list[NeverPick] = field(default_factory=list)
    updated_at: str = ""

    @property
    def great(self) -> list[Moment]:
        return [m for m in self.moments if m.grade == "great"]

    def problems(self, duration: float | None = None) -> list[str]:
        """Errors that keep these labels from being scored."""
        out: list[str] = []
        limit = (float(duration) + 0.5) if duration else None
        for kind, rows in (("moment", self.moments), ("never-pick range", self.negatives)):
            for index, row in enumerate(rows, start=1):
                if not (row.end > row.start >= 0.0):
                    out.append(f"{kind} {index}: end must be after start ({clock(row.start)} - {clock(row.end)})")
                elif limit is not None and row.end > limit:
                    out.append(f"{kind} {index}: ends at {clock(row.end)}, after the video ({clock(duration)})")
        for index, row in enumerate(self.moments, start=1):
            if row.grade not in GRADES:
                out.append(f"moment {index}: grade must be one of {', '.join(GRADES)}, got {row.grade!r}")
        if self.finished and not self.moments:
            out.append("finished labels need at least one moment")
        return out

    def warnings(self) -> list[str]:
        """Allowed, but worth a second look before the case is scored."""
        out: list[str] = []
        if self.moments and not (MIN_MOMENTS <= len(self.moments) <= MAX_MOMENTS):
            out.append(f"{len(self.moments)} moments (the guideline is {MIN_MOMENTS}-{MAX_MOMENTS})")
        if self.moments and not self.great:
            out.append("no 'great' moment: HIT@1, Recall@K and mAP are undefined for this video")
        for index, row in enumerate(self.moments, start=1):
            length = row.end - row.start
            if length > 0 and (length < 8.0 or length > 90.0):
                out.append(f"moment {index} is {length:.0f} s long; mark the Short you would publish (~15-60 s)")
        return out

    def status(self, duration: float | None = None) -> str:
        """todo (not finished) | invalid | ready"""
        if self.problems(duration):
            return "invalid"
        return "ready" if self.finished else "todo"

    def to_dict(self, case_id: str) -> dict[str, Any]:
        return {"version": LABELS_VERSION, "case_id": case_id, "annotator": self.annotator,
                "finished": bool(self.finished), "updated_at": self.updated_at,
                "moments": [m.to_dict() for m in sorted(self.moments, key=lambda m: (m.start, m.end))],
                "negatives": [n.to_dict() for n in sorted(self.negatives, key=lambda n: (n.start, n.end))]}


def _note(value: Any) -> str:
    return " ".join(str(value or "").split())[:NOTE_MAX]


def labels_from_dict(data: Mapping[str, Any], annotator: str) -> Labels:
    """Parse a labels payload (file, editor POST or sheet). Times may be seconds or 'H:MM:SS.s'."""
    moments = [Moment(parse_clock(row["start"]), parse_clock(row["end"]),
                      str(row.get("grade") or "great").strip().lower(), _note(row.get("note")))
               for row in data.get("moments") or [] if isinstance(row, Mapping)]
    negatives = [NeverPick(parse_clock(row["start"]), parse_clock(row["end"]), _note(row.get("note")))
                 for row in data.get("negatives") or [] if isinstance(row, Mapping)]
    return Labels(annotator=str(data.get("annotator") or annotator), finished=bool(data.get("finished")),
                  moments=moments, negatives=negatives, updated_at=str(data.get("updated_at") or ""))


def clean_annotator(name: str | None) -> str:
    value = case_slug(name or "") if name else DEFAULT_ANNOTATOR
    if not name or value == "case":
        return DEFAULT_ANNOTATOR
    return value


def default_annotator() -> str:
    return clean_annotator(os.getenv("MIMIR_CLIP_EVAL_ANNOTATOR") or DEFAULT_ANNOTATOR)


# ============================================================
# CASES
# ============================================================

@dataclass
class Case:
    case_id: str
    directory: Path
    meta: dict[str, Any]

    @property
    def split(self) -> str:
        return str(self.meta.get("split") or "dev")

    @property
    def video_path(self) -> Path:
        return Path(str((self.meta.get("source") or {}).get("video", "")))

    @property
    def video_sha256(self) -> str:
        return str((self.meta.get("source") or {}).get("sha256", ""))

    @property
    def duration(self) -> float:
        return float((self.meta.get("source") or {}).get("duration") or 0.0)

    @property
    def transcript_path(self) -> Path:
        return self.directory / "transcript.json"

    @property
    def visual_report_path(self) -> Path | None:
        path = self.directory / "visual_report.json"
        return path if (self.meta.get("visual_report") and path.is_file()) else None

    @property
    def proxy_path(self) -> Path | None:
        path = self.directory / "proxy.mp4"
        return path if path.is_file() else None

    @property
    def gold_annotator(self) -> str:
        return clean_annotator(self.meta.get("gold_annotator") or DEFAULT_ANNOTATOR)

    def has_transcript(self) -> bool:
        return self.transcript_path.is_file()

    def transcript(self) -> dict[str, Any] | None:
        data = _read_json(self.transcript_path)
        return data if isinstance(data, dict) else None

    def words(self) -> list[tuple[float, float, str]]:
        """(start, end, word) of the frozen transcript, sorted by start."""
        data = self.transcript() or {}
        rows: list[tuple[float, float, str]] = []
        for word in data.get("words") or []:
            try:
                rows.append((float(word["start"]), float(word["end"]), str(word.get("word", "")).strip()))
            except (KeyError, TypeError, ValueError):
                continue
        rows.sort()
        return rows

    def labels_path(self, annotator: str) -> Path:
        return self.directory / "labels" / f"{clean_annotator(annotator)}.json"

    def annotators(self) -> list[str]:
        folder = self.directory / "labels"
        return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []

    def labels(self, annotator: str | None = None) -> Labels | None:
        name = clean_annotator(annotator) if annotator else self.gold_annotator
        data = _read_json(self.labels_path(name))
        if not isinstance(data, Mapping):
            return None
        try:
            return labels_from_dict(data, name)
        except (KeyError, TypeError, ValueError):
            return Labels(annotator=name, finished=False, moments=[Moment(-1.0, -1.0)])   # -> invalid

    def label_status(self, annotator: str | None = None) -> str:
        """missing | todo | invalid | ready"""
        labels = self.labels(annotator)
        return "missing" if labels is None else labels.status(self.duration)

    def gold(self) -> Labels | None:
        """The scored labels (the gold annotator's), only when finished and valid."""
        labels = self.labels()
        return labels if labels is not None and labels.status(self.duration) == "ready" else None

    def other_finished_labels(self) -> list[Labels]:
        out = []
        for name in self.annotators():
            if name == self.gold_annotator:
                continue
            labels = self.labels(name)
            if labels is not None and labels.status(self.duration) == "ready":
                out.append(labels)
        return out


def list_cases(root: str | Path | None = None, only: Sequence[str] | None = None,
               split: str | None = None) -> list[Case]:
    base = eval_root(root) / "cases"
    if not base.is_dir():
        return []
    wanted = {str(x) for x in only} if only else None
    cases: list[Case] = []
    for directory in sorted(p for p in base.iterdir() if p.is_dir()):
        meta = _read_json(directory / "meta.json")
        if not isinstance(meta, dict):
            continue
        case = Case(str(meta.get("case_id") or directory.name), directory, meta)
        if wanted is not None and case.case_id not in wanted:
            continue
        if split not in (None, "all") and case.split != split:
            continue
        cases.append(case)
    return cases


def save_labels(case: Case, payload: Mapping[str, Any], *, annotator: str) -> tuple[str, list[str], list[str]]:
    """Validate and write one annotator's labels -> (status, problems, warnings).

    Invalid labels are never written over valid ones: the problems are returned instead."""
    name = clean_annotator(annotator)
    labels = labels_from_dict({**payload, "annotator": name}, name)
    problems = labels.problems(case.duration)
    if problems:
        return "invalid", problems, labels.warnings()
    labels.updated_at = _now()
    _write_json(case.labels_path(name), labels.to_dict(case.case_id))
    if "gold_annotator" not in case.meta:                  # the first person to label owns the gold
        case.meta["gold_annotator"] = name
        _write_json(case.directory / "meta.json", case.meta)
    return labels.status(case.duration), [], labels.warnings()


def set_split(case: Case, split: str) -> None:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    case.meta["split"] = split
    _write_json(case.directory / "meta.json", case.meta)


# ============================================================
# TRANSCRIPT VIEWS (for the human: timestamped text, never MIMIR's picks)
# ============================================================

def transcript_windows(words: Sequence[tuple[float, float, str]], window: float = 30.0,
                       duration: float | None = None) -> list[tuple[float, float, str]]:
    """The transcript as (window start, window end, text) blocks of ``window`` seconds."""
    window = max(5.0, float(window))
    end_time = max([w[1] for w in words] + [float(duration or 0.0)])
    buckets: dict[int, list[str]] = {}
    for start, _end, word in words:
        if word:
            buckets.setdefault(int(start // window), []).append(word)
    count = int(end_time // window) + 1 if end_time > 0 else 0
    return [(i * window, min((i + 1) * window, end_time), " ".join(buckets.get(i, [])))
            for i in range(count)]


def text_between(words: Sequence[tuple[float, float, str]], start: float, end: float) -> str:
    """Words whose middle lies inside [start, end]."""
    return " ".join(w for s, e, w in words if w and start <= (s + e) / 2.0 <= end)


# ============================================================
# PREPARE
# ============================================================

def _transcript_summary(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, Mapping) or not isinstance(data.get("words"), list):
        raise ValueError(f"not a Stage 1 transcript (no 'words' list): {path}")
    if int(data.get("version", 0) or 0) != 2:
        raise ValueError(f"unsupported transcript version {data.get('version')!r} (expected 2): {path}")
    source = data.get("source") if isinstance(data.get("source"), Mapping) else {}
    return {"words": len(data["words"]), "segments": len(data.get("segments") or []),
            "duration": float(source.get("duration") or 0.0),
            "video_path_in_transcript": str(source.get("video_path") or ""),
            "model": str((data.get("transcription") or {}).get("model", "")) if isinstance(
                data.get("transcription"), Mapping) else ""}


def freeze_inputs(case: Case, *, transcript: str | Path | None = None, visual_report: str | Path | None = None,
                  use_visual_report: bool = True, log: Any = print) -> None:
    """Copy the Stage 1 transcript and the visual-facts report into the case (byte for byte)."""
    video = case.video_path
    source = Path(transcript).resolve() if transcript else default_transcript_path(video)
    if source.is_file():
        summary = _transcript_summary(source)
        shutil.copyfile(source, case.transcript_path)
        case.meta["transcript"] = {"source": str(source), "sha256": bytes_sha256(case.transcript_path),
                                   "frozen_at": _now(), **summary}
    else:
        case.meta["transcript"] = None
        if case.transcript_path.exists():
            case.transcript_path.unlink()
        log(f"  no Stage 1 transcript at {source}\n  one-time command: {transcribe_command(video)}")
    report_target = case.directory / "visual_report.json"
    report = Path(visual_report).resolve() if visual_report else default_visual_report_path(video)
    if use_visual_report and report.is_file():
        shutil.copyfile(report, report_target)
        case.meta["visual_report"] = {"source": str(report), "sha256": bytes_sha256(report_target),
                                      "frozen_at": _now()}
    else:
        case.meta["visual_report"] = None
        if report_target.exists():
            report_target.unlink()
    _write_json(case.directory / "meta.json", case.meta)


def make_proxy(case: Case, *, height: int = 360) -> Path:
    """A small H.264/AAC copy the browser can play and seek (for MKV/TS or very large VODs)."""
    target = case.directory / "proxy.mp4"
    temp = case.directory / "proxy.tmp.mp4"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(case.video_path),
                    "-vf", f"scale=-2:{int(height)}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(temp)], check=True)
    os.replace(temp, target)
    return target


def prepare(video: str | Path, *, root: str | Path | None = None, case_id: str | None = None, split: str = "dev",
            transcript: str | Path | None = None, visual_report: str | Path | None = None,
            use_visual_report: bool = True, proxy: bool = False, refresh: bool = False, log: Any = print) -> Case:
    """Create (or, with ``refresh``, re-freeze the inputs of) the case of one source video.

    Labels are never touched. Nothing here shows or copies MIMIR's output for the video."""
    if split not in SPLITS:
        raise ValueError(f"--split must be one of {', '.join(SPLITS)}")
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    sha = file_sha256(video, root=root)
    case_id = case_id or case_slug(video.stem)
    directory = eval_root(root) / "cases" / case_id
    existing = _read_json(directory / "meta.json")
    if isinstance(existing, dict):
        if (existing.get("source") or {}).get("sha256") != sha:
            raise FileExistsError(f"case {case_id!r} already holds another video "
                                  f"({(existing.get('source') or {}).get('video')}); pass --id to name this one")
        if not refresh:
            raise FileExistsError(f"case {case_id!r} already exists (use --refresh to re-freeze its transcript and "
                                  "visual report; labels are never touched)")
        meta = existing
        meta["source"]["video"] = str(video)                 # the file may have moved; the bytes did not
    else:
        meta = {"case_id": case_id, "created_at": _now(), "split": split, "notes": "",
                "source": {"video": str(video), "name": video.name, "sha256": sha, "size": video.stat().st_size,
                           "duration": None}}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "labels").mkdir(exist_ok=True)
    case = Case(case_id, directory, meta)
    meta["source"]["duration"] = probe_duration(video)
    freeze_inputs(case, transcript=transcript, visual_report=visual_report, use_visual_report=use_visual_report,
                  log=log)
    if not meta["source"]["duration"]:
        meta["source"]["duration"] = (meta.get("transcript") or {}).get("duration") or None
        _write_json(directory / "meta.json", meta)
    if proxy:
        make_proxy(case)
    return case


def find_case_for_video(video: str | Path, cases: Iterable[Case], *, size: int | None = None) -> Case | None:
    """The case of a source video, by path (case-insensitive, as the pipeline compares) or by size."""
    target = str(Path(video)).casefold()
    for case in cases:
        if str(case.video_path).casefold() == target:
            return case
    if size is not None:
        matches = [c for c in cases if int((c.meta.get("source") or {}).get("size") or -1) == int(size)]
        if len(matches) == 1:
            return matches[0]
    return None


def case_inputs(case: Case) -> dict[str, Any]:
    """The bytes a run of this case depends on (part of every run signature)."""
    return {"video_sha256": case.video_sha256,
            "transcript_sha256": (case.meta.get("transcript") or {}).get("sha256"),
            "visual_report_sha256": (case.meta.get("visual_report") or {}).get("sha256")}


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
