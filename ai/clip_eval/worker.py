"""One clip-eval job in its own process: ``python -m ai.clip_eval.worker <job.json>``.

A subprocess because model routing (``ai.model_config``) and the Stage 2 settings are read when
the modules are imported: a variant's environment has to be in place before the first import.
The parent (``runner``) builds the environment (process env + .env + the variant) and the job file;
this process writes ``result.json`` next to the job file. Modes:

    describe     import Stage 2 under the variant's environment and report what it would run:
                 routing, settings, Stage 2 constraints, code fingerprints, prompt sizes, and the
                 transcript sizes the cost estimate needs. No model call.
    clip         Stage 2 alone: the real ``clip_analyzer.create_clip_analysis`` on the case's frozen
                 transcript / visual report. ``clip_analyzer.ANALYSIS_DIR`` is pointed at this job's
                 folder first, and the production cache files are fingerprinted before and after:
                 a run that touched them is an error. Every OpenAI call goes through the budget meter.
    full         the unmodified ``run_pipeline`` inside an isolated copy of the code (the parent made
                 it), so it has its own vod_output and every stage really runs.
    transcribe   Stage 1 (``vod_processor.process_vod``) for a video that has no transcript yet.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from ai.clip_eval import pricing

TEST_ANALYZER_ENV = "MIMIR_CLIP_EVAL_TEST_ANALYZER"   # tests only: "package.module:function" replaces Stage 2
STAGE2_SETTINGS = ("CLIP_HIGH_ESCALATION_MARGIN", "BOUNDARY_POLISH_MAX_EXTEND", "BOUNDARY_POLISH_MAX_TRIM",
                   "STORY_VISUAL_ORIGIN_ENABLED", "SOURCE_PEAK_GUARD_MIN_AUDIO_SCORE",
                   "SOURCE_PEAK_GUARD_MIN_PEAK_DBFS", "SOURCE_PEAK_GUARD_MAX", "MAX_CLIPS", "MAX_MONEY_MOMENTS",
                   "SCORE_POLICY")
INSTRUCTIONS = {"money_moments": "MONEY_MOMENT_INSTRUCTIONS", "visual_audit_repair": "MONEY_MOMENT_INSTRUCTIONS",
                "clips": "CLIP_COMPOSER_INSTRUCTIONS", "final_judge": "FINAL_JUDGE_INSTRUCTIONS",
                "final_judge_escalation": "FINAL_JUDGE_INSTRUCTIONS",
                "story_expansion": "STORY_EXPANSION_INSTRUCTIONS", "boundary_polish": "BOUNDARY_POLISH_INSTRUCTIONS"}
SCHEMAS = {"money_moments": "MONEY_MOMENT_SCHEMA", "visual_audit_repair": "VISUAL_AUDIT_REPAIR_SCHEMA",
           "clips": "CLIP_SCHEMA", "final_judge": "FINAL_JUDGE_SCHEMA", "final_judge_escalation": "FINAL_JUDGE_SCHEMA",
           "story_expansion": "STORY_EXPANSION_SCHEMA", "boundary_polish": "BOUNDARY_POLISH_SCHEMA"}
EVAL_PACKAGES = ("clip_eval", "caption_eval")


def module_fingerprint(module: Any) -> dict[str, Any]:
    """Code identity by content: the same record ``shorts_pipeline._module_fingerprint`` builds."""
    raw = getattr(module, "__file__", None)
    name = str(getattr(module, "__name__", "<unknown>"))
    if not raw:
        return {"module": name, "sha256": None}
    try:
        return {"module": name, "sha256": hashlib.sha256(Path(raw).resolve().read_bytes()).hexdigest()}
    except OSError:
        return {"module": name, "sha256": None}


def pipeline_code_fingerprint(project_root: Path) -> dict[str, Any]:
    """Every pipeline source file under ai/ (the eval packages excluded): what --stage full runs."""
    base = project_root / "ai"
    digest = hashlib.sha256()
    files = 0
    for path in sorted(base.rglob("*.py")):
        relative = path.relative_to(base)
        if relative.parts and relative.parts[0] in EVAL_PACKAGES:
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
        files += 1
    return {"files": files, "sha256": digest.hexdigest()}


def _file_state(path: Path) -> list[Any]:
    try:
        stat = path.stat()
    except OSError:
        return [False]
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if stat.st_size <= 200 * 1024 * 1024 else None
    return [True, stat.st_size, stat.st_mtime_ns, digest]


# ============================================================
# DESCRIBE
# ============================================================

def stage2_configuration() -> dict[str, Any]:
    from ai import model_config, openai_client
    from ai.editor import clip_analyzer as ca

    from ai.clip_eval.variants import STAGE2_KEYS

    return {
        "routing": {"scout": {"model": ca.CLIP_SCOUT_MODEL, "effort": ca.CLIP_SCOUT_REASONING_EFFORT},
                    "judge": {"model": ca.CLIP_JUDGE_MODEL, "effort": ca.CLIP_JUDGE_REASONING_EFFORT},
                    "escalation": {"model": ca.CLIP_JUDGE_MODEL, "effort": ca.CLIP_JUDGE_ESCALATION_REASONING_EFFORT}},
        "settings": {name: getattr(ca, name, None) for name in STAGE2_SETTINGS},
        "constraints": {"min_duration": float(ca.MIN_CLIP_DURATION), "max_duration": float(ca.MAX_CLIP_DURATION),
                        "target_min": float(ca.TARGET_CLIP_MIN), "target_max": float(ca.TARGET_CLIP_MAX),
                        "analyzer_version": int(ca.ANALYZER_VERSION)},
        "analyzer": {"version": int(ca.ANALYZER_VERSION), "revision": int(getattr(ca, "ANALYZER_REVISION", 0))},
        "code": [module_fingerprint(m) for m in (ca, model_config, openai_client)],
        "instruction_chars": {call: len(str(getattr(ca, name, ""))) + len(json.dumps(getattr(ca, SCHEMAS[call], {})))
                              for call, name in INSTRUCTIONS.items()},
        "env": {key: os.environ[key] for key in sorted(STAGE2_KEYS) if key in os.environ},
    }


def full_configuration(project_root: Path) -> dict[str, Any]:
    from ai import model_config, vod_processor
    from ai.video_brain import config as vb_config

    return {"model_plan": model_config.model_plan(), "code": pipeline_code_fingerprint(project_root),
            "vod_text_model": str(vod_processor.ACCURATE_MODEL), "timing_model": str(vod_processor.TIMING_MODEL),
            "video_brain_enabled": bool(vb_config.VIDEO_BRAIN_ENABLED),
            "video_brain_model": str(vb_config.VIDEO_BRAIN_MODEL),
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("MIMIR_", "VIDEO_BRAIN", "VIDEO_UPLOAD", "CAPTION_"))}}


def transcript_stats(transcript_path: Path, visual_report: Path | None) -> dict[str, Any]:
    from ai.editor import clip_analyzer as ca

    transcript = ca.load_json(transcript_path)
    duration = ca._video_duration(transcript)
    report = ca.load_visual_report(visual_report) if visual_report else None
    events = ca.build_visual_events(report, video_duration=duration) if report else []
    return {"transcript_chars": len(ca.build_timestamped_transcript(transcript)),
            "visual_chars": len(ca.build_visual_evidence_text(events)) if events else 0,
            "has_visual_report": report is not None, "duration": duration}


def describe(job: dict[str, Any]) -> dict[str, Any]:
    from ai.editor import clip_analyzer as ca

    result: dict[str, Any] = {"status": "ok", "configuration": stage2_configuration()}
    if job.get("stage") == "full":
        result["configuration"]["full"] = full_configuration(Path(ca.PROJECT_ROOT))
    stats = {}
    for case_id, paths in (job.get("cases") or {}).items():
        try:
            stats[case_id] = transcript_stats(Path(paths["transcript"]),
                                              Path(paths["visual_report"]) if paths.get("visual_report") else None)
        except Exception as error:  # one unreadable case never hides the others
            stats[case_id] = {"error": f"{type(error).__name__}: {error}"}
    result["stats"] = stats
    return result


# ============================================================
# STAGE 2 ALONE
# ============================================================

def _analyzer() -> Callable[..., Any]:
    hook = os.getenv(TEST_ANALYZER_ENV, "").strip()
    if hook:
        module_name, _, attr = hook.partition(":")
        return getattr(importlib.import_module(module_name), attr)
    from ai.editor import clip_analyzer

    return clip_analyzer.create_clip_analysis


def run_clip(job: dict[str, Any], *, meter: pricing.Meter | None = None,
             analyzer: Callable[..., Any] | None = None) -> dict[str, Any]:
    from ai.editor import clip_analyzer

    meter = meter or pricing.Meter(pricing.load_prices(), budget_usd=job.get("budget_usd"),
                                   strict_unpriced=True).install()
    out_dir = Path(job["out_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = Path(job["transcript"]).resolve()
    production_dir = Path(clip_analyzer.ANALYSIS_DIR).resolve()
    guarded = {Path(p).resolve() for p in job.get("guard_paths") or []}
    guarded.add(Path(clip_analyzer.get_output_path(transcript)).resolve())
    before = {path: _file_state(path) for path in guarded}
    original_dir = clip_analyzer.ANALYSIS_DIR
    clip_analyzer.ANALYSIS_DIR = out_dir                       # runtime only: no file under ai/ changes
    status, error, analysis = "ok", "", None
    started = time.perf_counter()
    try:
        target = Path(clip_analyzer.get_output_path(transcript)).resolve()
        if target.parent != out_dir:
            raise RuntimeError(f"Stage 2 output would go to {target}, not the job folder; refusing to run")
        report = job.get("visual_report")
        returned = (analyzer or _analyzer())(transcript, video_report_path=Path(report) if report else None)
        saved = clip_analyzer.load_json(target) if target.is_file() else None
        analysis = saved if isinstance(saved, dict) else returned     # what the pipeline would read back
        if not isinstance(analysis, dict):
            status, error = "error", "Stage 2 returned no analysis object"
    except pricing.BudgetExceeded as exc:
        status, error = "budget_stopped", str(exc)
    except Exception as exc:  # recorded per job; one case never stops the run
        status, error = "error", f"{type(exc).__name__}: {str(exc)[:800]}"
    finally:
        clip_analyzer.ANALYSIS_DIR = original_dir
    elapsed = round(time.perf_counter() - started, 2)
    if meter.refused and status == "ok":                         # refused inside an `except Exception` fallback
        status, error, analysis = "budget_stopped", meter.refused, None
    touched = sorted(str(p) for p in guarded if _file_state(p) != before[p])
    if touched:
        status, error = "error", "production Stage 2 cache changed during the run: " + ", ".join(touched)
    return {"status": status, "error": error, "analysis": analysis if status == "ok" else None,
            "usage": meter.summary(), "elapsed_s": elapsed, "production_dir": str(production_dir),
            "production_cache_untouched": not touched, "guarded": sorted(str(p) for p in guarded)}


# ============================================================
# FULL PIPELINE / STAGE 1
# ============================================================

def run_full(job: dict[str, Any]) -> dict[str, Any]:
    meter = pricing.Meter(pricing.load_prices(), budget_usd=job.get("budget_usd"),
                          reserved_usd=float(job.get("reserved_usd") or 0.0), strict_unpriced=False).install()
    from ai import shorts_pipeline as sp

    workspace = Path(job["workspace"]).resolve()
    if Path(sp.PROJECT_ROOT).resolve() != workspace:
        return {"status": "error", "error": f"pipeline imported from {sp.PROJECT_ROOT}, not the workspace {workspace}"}
    video = Path(job["video"]).resolve()
    status, error, result = "published", "", {}
    started = time.perf_counter()
    try:
        result = sp.run_pipeline(video, force=True, enable_memes=bool(job.get("memes", True)), enable_video_brain=None)
        status = str(result.get("status") or "published")
    except pricing.BudgetExceeded as exc:
        status, error = "budget_stopped", str(exc)
    except sp.NoStrongClipError as exc:
        status, error = "no_story", str(exc)[:800]
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {str(exc)[:800]}"
    elapsed = round(time.perf_counter() - started, 2)
    if meter.refused and status not in ("budget_stopped",):
        status, error = "budget_stopped", meter.refused
    state = sp._load_state(sp._state_path(video)) or {}
    analysis_path = sp._analysis_path(sp._transcript_path(video))
    analysis = sp._load_json(analysis_path) if analysis_path.is_file() else None
    if status == "failed" and state.get("publish_status") == "rejected":
        status = "rejected"
    artifacts = {}
    final = state.get("final_output") or ""
    for key, path in (("short", Path(final) if final else None),
                      ("qc", Path(final).with_suffix(".qc.json") if final else None),
                      ("review", Path(final).with_suffix(".review.md") if final else None),
                      ("state", sp._state_path(video))):
        if path is not None and path.is_file():
            artifacts[key] = str(path)
    return {"status": "ok" if analysis is not None and status != "budget_stopped" else
            ("budget_stopped" if status == "budget_stopped" else "error"),
            "pipeline_status": status, "error": error, "analysis": analysis, "usage": meter.summary(),
            "elapsed_s": elapsed, "artifacts": artifacts, "selected_clip": state.get("selected_clip"),
            "warnings": state.get("warnings")}


def run_transcribe(job: dict[str, Any]) -> dict[str, Any]:
    meter = pricing.Meter(pricing.load_prices(), budget_usd=job.get("budget_usd"),
                          reserved_usd=float(job.get("reserved_usd") or 0.0), strict_unpriced=True).install()
    from ai import vod_processor

    video = Path(job["video"]).resolve()
    started = time.perf_counter()
    try:
        vod_processor.process_vod(video)
        status, error = "ok", ""
    except pricing.BudgetExceeded as exc:
        status, error = "budget_stopped", str(exc)
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {str(exc)[:800]}"
    target = Path(vod_processor.TRANSCRIPT_DIR) / f"{video.stem}.json"
    return {"status": status, "error": error, "transcript": str(target) if target.is_file() else None,
            "usage": meter.summary(), "elapsed_s": round(time.perf_counter() - started, 2)}


MODES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "describe": describe, "clip": run_clip, "full": run_full, "transcribe": run_transcribe}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m ai.clip_eval.worker <job.json>", file=sys.stderr)
        return 2
    job_path = Path(args[0]).resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        result = MODES[job["mode"]](job)
    except pricing.BudgetExceeded as exc:
        result = {"status": "budget_stopped", "error": str(exc)}
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:800]}",
                  "traceback": traceback.format_exc()[-4000:]}
    target = job_path.with_name("result.json")
    temp = target.with_name("result.json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temp, target)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
