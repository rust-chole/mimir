"""Run a variant over the cases (cached per case and repeat), score it against the labels, compare runs.

    <root>/runs/<run name>/
        run.json                           variant + its env overrides, stage, the configuration it ran
                                           (routing, Stage 2 settings and limits, code fingerprints),
                                           git commit, money spent
        hyp/<case_id>/r<k>.json            one Stage 2 result per case and repeat (cached by signature:
                                           configuration + the case's frozen input bytes + repeat)
        hyp/<case_id>/r<k>.analysis.json   the raw Stage 2 output, exactly what the pipeline would read
        hyp/<case_id>/r<k>.log             the worker's console output
        score.json, report.md              metrics against the CURRENT labels, dev and holdout apart
        feedback.dev.json                  AI-readable per-video feedback, dev only (``feedback``)

The default run name is ``<variant>__<configuration hash>``: changing a prompt, a model, a setting or
the variant file makes a NEW run, so the run you compare against is never overwritten. Re-scoring and
comparing never call a model. Paid jobs need ``--max-usd``; ``--dry-run`` prints the estimate only.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.caption_eval.runner import _git_commit
from ai.clip_eval import cases as case_store
from ai.clip_eval import metrics, pricing, report, variants

RUNNER_VERSION = 1
CLIP_TIMEOUT_S = 3 * 3600
FULL_TIMEOUT_S = 8 * 3600
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,120}")
Executor = Callable[[dict[str, Any], dict[str, str], Path, Path], dict[str, Any]]
_now = case_store._now


def run_dir(name: str, root: str | Path | None = None) -> Path:
    if not _RUN_NAME.fullmatch(str(name)):
        raise ValueError(f"run name {name!r}: use letters, digits, '.', '_' or '-'")
    return case_store.eval_root(root) / "runs" / name


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
                          ).hexdigest()


# ============================================================
# SUBPROCESS
# ============================================================

def child_env(variant: variants.Variant, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Process env, then .env for keys not set (python-dotenv ``override=False`` like the pipeline), then
    the variant on top. The copied code of ``--stage full`` has no .env, so it is passed in here."""
    env = dict(os.environ if base is None else base)
    env_file = case_store.PROJECT_ROOT / ".env"
    if env_file.is_file():
        from dotenv import dotenv_values

        for key, value in dotenv_values(env_file).items():
            if value is not None:
                env.setdefault(key, value)
    env.update(variant.env)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return env


def spawn(job: dict[str, Any], env: dict[str, str], cwd: Path, log_path: Path) -> dict[str, Any]:
    """Default executor: ``python -m ai.clip_eval.worker`` in ``cwd`` -> its result.json."""
    work = Path(job["work_dir"])
    work.mkdir(parents=True, exist_ok=True)
    job_path = work / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = FULL_TIMEOUT_S if job["mode"] == "full" else CLIP_TIMEOUT_S
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        try:
            completed = subprocess.run([sys.executable, "-m", "ai.clip_eval.worker", str(job_path)], cwd=str(cwd),
                                       env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"worker timed out after {timeout} s (log: {log_path})"}
    result = case_store._read_json(work / "result.json")
    if not isinstance(result, dict):
        return {"status": "error", "error": f"worker exited {completed.returncode} without a result (log: {log_path})"}
    return result


# ============================================================
# PLAN
# ============================================================

@dataclass
class Job:
    case: case_store.Case
    repeat: int
    signature: str
    target: Path
    cached: bool
    estimate: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    name: str
    directory: Path
    variant: variants.Variant
    stage: str
    split: str
    repeats: int
    configuration: dict[str, Any]
    configuration_hash: str
    jobs: list[Job]
    skipped: list[dict[str, str]]
    warnings: list[str]
    prices: dict[str, Any]

    @property
    def to_run(self) -> list[Job]:
        return [job for job in self.jobs if not job.cached]

    @property
    def expected_usd(self) -> float:
        return sum(float(job.estimate.get("expected") or 0.0) for job in self.to_run)

    @property
    def worst_usd(self) -> float:
        return sum(float(job.estimate.get("worst") or 0.0) for job in self.to_run)

    @property
    def unpriced(self) -> list[str]:
        return sorted({m for job in self.to_run for m in job.estimate.get("unpriced") or []})


def routing_key(configuration: Mapping[str, Any], stage: str) -> str:
    """What decides a job's price: the stage and the models + efforts it routes to."""
    blob: dict[str, Any] = {"stage": stage, "routing": configuration.get("routing")}
    if stage == "full":
        blob["plan"] = (configuration.get("full") or {}).get("model_plan")
    return _hash(blob)[:16]


def configuration_hash(configuration: Mapping[str, Any], variant: variants.Variant, stage: str) -> str:
    return _hash({"stage": stage, "variant_env": variant.env, "configuration": configuration,
                  "runner": RUNNER_VERSION})


def job_signature(config_hash: str, case: case_store.Case, repeat: int, stage: str) -> str:
    return _hash({"configuration": config_hash, "inputs": case_store.case_inputs(case), "repeat": repeat,
                  "stage": stage, "runner": RUNNER_VERSION})


def _hyp_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("r*.json") if re.fullmatch(r"r\d+\.json", p.name))


def measured_costs(root: str | Path | None = None) -> dict[tuple[str, str], list[float]]:
    """Real cost of finished jobs anywhere under root, by (case inputs, routing key)."""
    out: dict[tuple[str, str], list[float]] = {}
    for path in (case_store.eval_root(root) / "runs").glob("*/hyp/*/r*.json"):
        if not re.fullmatch(r"r\d+\.json", path.name):
            continue
        record = case_store._read_json(path)
        if not isinstance(record, Mapping) or record.get("status") != "ok" or record.get("cost_usd") is None:
            continue
        key = (str(record.get("inputs_key", "")), str(record.get("routing_key", "")))
        out.setdefault(key, []).append(float(record["cost_usd"]))
    return out


def describe(variant: variants.Variant, stage: str, cases: Sequence[case_store.Case], *, root: str | Path | None,
             executor: Executor | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask a worker (with the variant's environment) what it would run. No model call."""
    work = case_store.eval_root(root) / "tmp" / f"describe_{os.getpid()}_{time.time_ns()}"
    job = {"mode": "describe", "stage": stage, "work_dir": str(work),
           "cases": {c.case_id: {"transcript": str(c.transcript_path),
                                 "visual_report": str(c.visual_report_path) if c.visual_report_path else None}
                     for c in cases if c.has_transcript()}}
    try:
        result = (executor or spawn)(job, child_env(variant), case_store.PROJECT_ROOT, work / "describe.log")
        if result.get("status") != "ok":
            raise RuntimeError(f"could not load Stage 2 with variant {variant.name!r}: {result.get('error')}")
        return result["configuration"], result.get("stats") or {}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def plan_run(variant_name: str, *, split: str = "dev", only: Sequence[str] | None = None, repeats: int = 1,
             stage: str = "clip", name: str | None = None, force: bool = False, include_unlabeled: bool = False,
             root: str | Path | None = None, executor: Executor | None = None) -> Plan:
    if stage not in ("clip", "full"):
        raise ValueError("--stage must be clip or full")
    if split not in ("dev", "holdout", "all"):
        raise ValueError("--split must be dev, holdout or all")
    repeats = max(1, int(repeats))
    variant = variants.load(variant_name, root)
    prices = pricing.load_prices()
    skipped: list[dict[str, str]] = []
    warnings: list[str] = []
    chosen: list[case_store.Case] = []
    for case in case_store.list_cases(root, only, split):
        status = case.label_status()
        if status != "ready" and not include_unlabeled:
            skipped.append({"case_id": case.case_id, "reason": f"labels {status} (only finished labels are scored)"})
            continue
        if stage == "clip" and not case.has_transcript():
            skipped.append({"case_id": case.case_id, "reason": "no Stage 1 transcript; one-time command: "
                            + case_store.transcribe_command(case.video_path)})
            continue
        if stage == "full" and not case.video_path.is_file():
            skipped.append({"case_id": case.case_id, "reason": f"source video missing: {case.video_path}"})
            continue
        chosen.append(case)
    if only:
        known = {c.case_id: c for c in case_store.list_cases(root, only)}
        for case_id in sorted(set(only)):
            if case_id not in known:
                skipped.append({"case_id": case_id, "reason": "no such case"})
            elif split != "all" and known[case_id].split != split:
                skipped.append({"case_id": case_id, "reason": f"in the {known[case_id].split} split "
                                                              f"(add --split {known[case_id].split})"})
    ignored = variant.ignored_keys(stage)
    if ignored:
        warnings.append(f"variant keys with no effect on --stage {stage}: {', '.join(ignored)}")
    configuration, stats = describe(variant, stage, chosen, root=root, executor=executor)
    config_hash = configuration_hash(configuration, variant, stage)
    name = name or f"{variant.name}__{config_hash[:8]}"
    directory = run_dir(name, root)
    previous = case_store._read_json(directory / "run.json")
    if isinstance(previous, Mapping) and previous.get("configuration_hash") != config_hash:
        raise ValueError(f"run {name!r} already holds another configuration (variant {previous.get('variant')!r}, "
                         f"{str(previous.get('configuration_hash'))[:8]}); choose another --name")
    for key, value in (configuration.get("env") or {}).items():
        if key not in variant.env:
            warnings.append(f"{key}={value} comes from your environment / .env (recorded in the run)")
    key = routing_key(configuration, stage)
    measured = measured_costs(root)
    jobs: list[Job] = []
    for case in chosen:
        inputs_key = _hash(case_store.case_inputs(case))[:16]
        case_stats = stats.get(case.case_id) or {"duration": case.duration}
        if case_stats.get("error"):
            skipped.append({"case_id": case.case_id, "reason": f"transcript unreadable: {case_stats['error']}"})
            continue
        clip_estimate = pricing.estimate_clip_job(prices, configuration, case_stats)
        estimate = (pricing.estimate_full_job(prices, configuration.get("full") or {}, {
            "duration": case.duration or case_stats.get("duration", 0.0)}, clip_estimate)
            if stage == "full" else clip_estimate)
        seen = measured.get((inputs_key, key))
        if seen:
            mean = sum(seen) / len(seen)
            estimate = {**estimate, "expected": mean * 1.1, "worst": max(max(seen) * 1.5, mean * 1.5),
                        "basis": f"measured ({len(seen)} run(s))"}
        for repeat in range(1, repeats + 1):
            target = directory / "hyp" / case.case_id / f"r{repeat}.json"
            signature = job_signature(config_hash, case, repeat, stage)
            cached = case_store._read_json(target)
            is_cached = (not force and isinstance(cached, Mapping) and cached.get("signature") == signature
                         and cached.get("status") == "ok")
            jobs.append(Job(case, repeat, signature, target, is_cached, {**estimate, "inputs_key": inputs_key}))
    if split in ("holdout", "all") and any(job.case.split == "holdout" for job in jobs):
        warnings.append("this run includes HOLDOUT videos: its report is for people only, never for an AI or an "
                        "optimizer (use `feedback` for the dev-only AI-readable export)")
    return Plan(name, directory, variant, stage, split, repeats, configuration, config_hash, jobs, skipped,
                warnings, prices)


def format_plan(plan: Plan) -> str:
    routing = plan.configuration.get("routing") or {}
    scout, judge, escalation = (routing.get(k) or {} for k in ("scout", "judge", "escalation"))
    code = " ".join(f"{str(m.get('module', '')).rsplit('.', 1)[-1]}:{str(m.get('sha256') or '-')[:8]}"
                    for m in plan.configuration.get("code") or [])
    prices = plan.prices
    lines = [f"Run '{plan.name}': variant {plan.variant.name}, stage {plan.stage}, split {plan.split}, "
             f"{plan.repeats} repeat(s)",
             f"  Stage 2   scout {scout.get('model')} [{scout.get('effort')}], judge {judge.get('model')} "
             f"[{judge.get('effort')}] -> close calls [{escalation.get('effort')}]",
             f"  code      {code}"]
    if plan.variant.env:
        lines.append("  variant   " + ", ".join(f"{k}={v}" for k, v in sorted(plan.variant.env.items())))
    if plan.stage == "full":
        full = plan.configuration.get("full") or {}
        lines.append(f"  pipeline  {full.get('code', {}).get('files')} files, sha {str(full.get('code', {}).get('sha256'))[:8]}; "
                     f"Stage 1 {full.get('vod_text_model')}, Video Brain "
                     f"{full.get('video_brain_model') if full.get('video_brain_enabled') else 'off'}")
    lines.append(f"  prices    {pricing.PRICING_PATH.name}" + (f" + {prices.get('override')}" if prices.get("override")
                                                                  else "")
                 + ("" if prices.get("verified") else " (UNVERIFIED, set high on purpose; check "
                    + ", ".join(prices.get("check_against") or []) + ")"))
    for warning in plan.warnings:
        lines.append(f"  note      {warning}")
    lines += ["", f"  {'case':<34} {'split':<8} {'length':>8} {'runs':>5} {'cached':>7} {'per run':>9}  basis"]
    by_case: dict[str, list[Job]] = {}
    for job in plan.jobs:
        by_case.setdefault(job.case.case_id, []).append(job)
    for case_id, jobs in by_case.items():
        case = jobs[0].case
        estimate = jobs[0].estimate
        per_run = "unpriced" if estimate.get("unpriced") else f"${float(estimate.get('expected') or 0):.2f}"
        lines.append(f"  {case_id[:34]:<34} {case.split:<8} {case_store.clock(case.duration):>8} {len(jobs):>5} "
                     f"{sum(j.cached for j in jobs):>7} {per_run:>9}  {estimate.get('basis', '')}")
    if not plan.jobs:
        lines.append("  (no case to run)")
    lines.append("")
    lines.append(f"  {len(plan.to_run)} job(s) to run, {len(plan.jobs) - len(plan.to_run)} cached (free): "
                 f"expected ${plan.expected_usd:.2f}, worst case ${plan.worst_usd:.2f}")
    if plan.unpriced:
        lines.append(f"  UNPRICED models (cannot be capped): {', '.join(plan.unpriced)}; add them to "
                     f"{pricing.PRICING_PATH}")
    for row in plan.skipped:
        lines.append(f"  not run: {row['case_id']}: {row['reason']}")
    return "\n".join(lines)


# ============================================================
# EXECUTE
# ============================================================

def _job_transcript(case: case_store.Case, work: Path) -> tuple[Path, list[str]]:
    """The case's frozen transcript for Stage 2, pointing at the video where it is now.

    Stage 2 reads ``source.video_path`` for its local motion scan; when the VOD moved since Stage 1,
    the copy points at the case's (hash-checked) video instead, so Stage 2 sees it as production did."""
    data = case.transcript() or {}
    notes: list[str] = []
    source = data.get("source") if isinstance(data.get("source"), dict) else {}
    recorded = Path(str(source.get("video_path") or ""))
    if not (str(recorded) and recorded.is_file()):
        if case.video_path.is_file():
            source["video_path"] = str(case.video_path)
            data["source"] = source
            notes.append("transcript source.video_path rewritten to the case video (it moved since Stage 1)")
        else:
            notes.append("source video not on disk: Stage 2 runs without its local motion scan (soft fallback)")
    target = work / f"{case.case_id}.json"
    work.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return target, notes


def make_workspace(target: Path) -> Path:
    """A byte-identical copy of the code the pipeline runs (its own vod_output, no .env, no tests)."""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(case_store.PROJECT_ROOT / "ai", target / "ai", ignore=ignore)
    shutil.copy2(case_store.PROJECT_ROOT / "main.py", target / "main.py")
    library = case_store.PROJECT_ROOT / "meme_library"
    if library.is_dir():
        shutil.copytree(library, target / "meme_library", ignore=ignore)
    return target


def _execute(plan: Plan, job: Job, *, budget: float, root: str | Path | None, executor: Executor | None,
             keep_workspace: bool) -> dict[str, Any]:
    case = job.case
    folder = job.target.parent
    work = folder / f"r{job.repeat}.work"
    shutil.rmtree(work, ignore_errors=True)
    log_path = folder / f"r{job.repeat}.log"
    env = child_env(plan.variant)
    notes: list[str] = []
    if plan.stage == "clip":
        transcript, notes = _job_transcript(case, work)
        production = (case_store.PROJECT_ROOT / "vod_output" / "analysis" / f"{case.video_path.stem}_clips.json")
        spec = {"mode": "clip", "work_dir": str(work), "out_dir": str(work / "analysis"),
                "transcript": str(transcript), "budget_usd": budget,
                "visual_report": str(case.visual_report_path) if case.visual_report_path else None,
                "guard_paths": [str(production.resolve())]}
        result = (executor or spawn)(spec, env, case_store.PROJECT_ROOT, log_path)
    else:
        workspace = make_workspace(case_store.eval_root(root) / "workspaces" / plan.name / case.case_id /
                                   f"r{job.repeat}")
        unmetered = float(job.estimate.get("unmetered_usd") or 0.0)
        spec = {"mode": "full", "work_dir": str(work), "workspace": str(workspace), "video": str(case.video_path),
                "budget_usd": budget, "reserved_usd": unmetered}
        result = (executor or spawn)(spec, env, workspace, log_path)
        artifacts_dir = folder / f"r{job.repeat}.full"
        for key, raw in (result.get("artifacts") or {}).items():
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            source = Path(raw)
            shutil.copy2(source, artifacts_dir / source.name)
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
    usage = result.get("usage") or {}
    metered = float(usage.get("spent_usd") or 0.0)
    reserved = float(usage.get("reserved_usd") or 0.0)
    analysis = result.get("analysis") if isinstance(result.get("analysis"), dict) else None
    record: dict[str, Any] = {
        "case_id": case.case_id, "split": case.split, "repeat": job.repeat, "variant": plan.variant.name,
        "stage": plan.stage, "signature": job.signature, "configuration_hash": plan.configuration_hash,
        "routing_key": routing_key(plan.configuration, plan.stage), "inputs_key": job.estimate.get("inputs_key"),
        "inputs": case_store.case_inputs(case), "created_at": _now(), "status": result.get("status", "error"),
        "error": result.get("error", ""), "elapsed_s": result.get("elapsed_s"),
        "cost_usd": round(metered + reserved, 6), "cost_basis": "metered" + (" + reserved estimate (unmetered)"
                                                                               if reserved else ""),
        "usage": usage, "input_notes": notes, "log": log_path.name,
        "production_cache_untouched": result.get("production_cache_untouched"),
    }
    if analysis is not None:
        analysis_path = folder / f"r{job.repeat}.analysis.json"
        case_store._write_json(analysis_path, analysis)
        record["analysis_file"] = analysis_path.name
        record["clips"] = metrics.ranked_clips(analysis)
        record["schema_problems"] = metrics.schema_problems(analysis, plan.configuration.get("constraints"))
    if plan.stage == "full":
        record["full"] = {"pipeline_status": result.get("pipeline_status"), "selected_clip": result.get("selected_clip"),
                          "warnings": result.get("warnings"), "artifacts": sorted((result.get("artifacts") or {}))}
    case_store._write_json(job.target, record)
    shutil.rmtree(work, ignore_errors=True)
    return record


def run_variant(variant_name: str, *, split: str = "dev", only: Sequence[str] | None = None, repeats: int = 1,
                stage: str = "clip", max_usd: float | None = None, dry_run: bool = False, name: str | None = None,
                force: bool = False, include_unlabeled: bool = False, keep_workspace: bool = False,
                root: str | Path | None = None, executor: Executor | None = None,
                log: Callable[[str], None] = print) -> tuple[Plan, list[dict[str, Any]], int]:
    """Plan, price and (only with a cap) run -> (plan, job records, exit code)."""
    plan = plan_run(variant_name, split=split, only=only, repeats=repeats, stage=stage, name=name, force=force,
                    include_unlabeled=include_unlabeled, root=root, executor=executor)
    log(format_plan(plan))
    if dry_run:
        log("\n--dry-run: nothing was run, no model was called.")
        return plan, [], 0
    if plan.to_run:
        if max_usd is None:
            log("\nNo model was called. Paid jobs need a cap: add --max-usd <dollars> (see the estimate above).")
            return plan, [], 2
        if plan.unpriced:
            log("\nRefused: some models have no price, so the cap cannot be enforced. No model was called.")
            return plan, [], 2
        if plan.expected_usd > float(max_usd) + 1e-9:
            log(f"\nRefused: the expected cost ${plan.expected_usd:.2f} is above --max-usd {float(max_usd):.2f}. "
                "Raise the cap, or use fewer --cases / --repeats. No model was called.")
            return plan, [], 2
    plan.directory.mkdir(parents=True, exist_ok=True)
    info = case_store._read_json(plan.directory / "run.json")
    info = dict(info) if isinstance(info, Mapping) else {"created_at": _now(), "spent_usd": 0.0}
    info.update(name=plan.name, variant=plan.variant.name, variant_env=plan.variant.env,
                variant_file=str(plan.variant.path) if plan.variant.path else None, stage=plan.stage,
                configuration=plan.configuration, configuration_hash=plan.configuration_hash, updated_at=_now(),
                git_commit=_git_commit(), version=RUNNER_VERSION,
                prices={"file": str(pricing.PRICING_PATH), "override": plan.prices.get("override"),
                        "verified": bool(plan.prices.get("verified"))})
    case_store._write_json(plan.directory / "run.json", info)
    spent = 0.0
    records: list[dict[str, Any]] = []
    stopped = ""
    exit_code = 0
    for job in plan.to_run:
        remaining = float(max_usd or 0.0) - spent
        if float(job.estimate.get("expected") or 0.0) > remaining + 1e-9:
            stopped = (f"budget: {job.case.case_id} r{job.repeat} needs ~${float(job.estimate['expected']):.2f}, "
                       f"${remaining:.2f} left")
            break
        log(f"  {job.case.case_id} r{job.repeat} ...")
        record = _execute(plan, job, budget=remaining, root=root, executor=executor, keep_workspace=keep_workspace)
        spent += float(record.get("cost_usd") or 0.0)
        records.append(record)
        cost = f"${float(record.get('cost_usd') or 0):.3f}"
        if record["status"] != "ok":
            exit_code = 1
            log(f"    ! {record['status']}: {record.get('error', '')[:300]}  ({cost}; log {job.target.parent / record['log']})")
        else:
            log(f"    ok in {record.get('elapsed_s')} s, {cost}")
        if record["status"] == "budget_stopped":
            stopped = "budget: a model call was refused by the --max-usd cap"
            break
    info["spent_usd"] = round(float(info.get("spent_usd") or 0.0) + spent, 6)
    info["updated_at"] = _now()
    case_store._write_json(plan.directory / "run.json", info)
    if stopped:
        exit_code = 1
        log(f"\nStopped early ({stopped}). Finished jobs are cached; re-run with a higher --max-usd to continue.")
    log(f"\nspent ${spent:.3f} in this command (run total ${info['spent_usd']:.3f}).")
    return plan, records, exit_code


# ============================================================
# SCORE
# ============================================================

def load_run(name: str, root: str | Path | None = None) -> dict[str, Any]:
    info = case_store._read_json(run_dir(name, root) / "run.json")
    if not isinstance(info, Mapping):
        raise FileNotFoundError(f"no run named {name!r} under {case_store.eval_root(root) / 'runs'} "
                                "(see `python -m ai.clip_eval runs`)")
    return dict(info)


def resolve_run(name: str, root: str | Path | None = None) -> str:
    """An exact run name, or a unique prefix of one (e.g. 'baseline' when only one baseline run exists)."""
    base = case_store.eval_root(root) / "runs"
    if (base / name / "run.json").is_file():
        return name
    found = sorted(p.parent.name for p in base.glob("*/run.json") if p.parent.name.startswith(name))
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"no run named {name!r} (see `python -m ai.clip_eval runs`)")
    raise ValueError(f"{name!r} matches several runs: {', '.join(found)}")


def _repeat_records(directory: Path) -> list[dict[str, Any]]:
    rows = [case_store._read_json(p) for p in _hyp_files(directory)]
    rows = [r for r in rows if isinstance(r, dict)]
    return sorted(rows, key=lambda r: int(r.get("repeat") or 0))


def score_run(name: str, *, root: str | Path | None = None) -> dict[str, Any]:
    """Score every finished job against the current labels; writes score.json + report.md."""
    from ai.clip_eval import reviews

    info = load_run(name, root)
    directory = run_dir(name, root)
    constraints = (info.get("configuration") or {}).get("constraints") or {}
    splits: dict[str, dict[str, Any]] = {s: {"cases": [], "per_repeat": {}} for s in case_store.SPLITS}
    skipped: list[dict[str, str]] = []
    weak_rows: dict[str, list[dict[str, Any]]] = {s: [] for s in case_store.SPLITS}
    for case in case_store.list_cases(root):
        folder = directory / "hyp" / case.case_id
        if not folder.is_dir():
            continue
        records = _repeat_records(folder)
        good = [r for r in records if r.get("status") == "ok"]
        gold = case.gold()
        if gold is None:
            skipped.append({"case_id": case.case_id, "reason": f"labels {case.label_status()}"})
            continue
        if not good:
            reasons = "; ".join(f"r{r.get('repeat')}: {r.get('status')} {str(r.get('error', ''))[:120]}"
                                for r in records) or "not run"
            skipped.append({"case_id": case.case_id, "reason": reasons})
            continue
        words = metrics.WordIndex(case.words())
        rows = []
        for record in good:
            row = metrics.score_repeat(record.get("clips") or [], gold, words, duration=case.duration,
                                       constraints=constraints, schema=record.get("schema_problems") or [])
            row.update(repeat=record.get("repeat"), cost_usd=record.get("cost_usd"), elapsed_s=record.get("elapsed_s"),
                       input_notes=record.get("input_notes") or [])
            rows.append(row)
        summary = metrics.case_summary(case.case_id, case.split, rows)
        summary["repeat_rows"] = rows
        summary["labels"] = {"annotator": gold.annotator, "moments": len(gold.moments), "great": len(gold.great),
                             "negatives": len(gold.negatives)}
        summary["human"] = [metrics.human_agreement(gold, other) for other in case.other_finished_labels()]
        summary["failed_repeats"] = len(records) - len(good)
        weak = reviews.check_against(case, [r.get("top") for r in rows], root=root)
        if weak:
            summary["weak"] = weak
            weak_rows[case.split].append({"case_id": case.case_id, **weak})
        bucket = splits[case.split]
        bucket["cases"].append(summary)
        bucket["per_repeat"][case.case_id] = rows
    result_splits = {}
    for split, bucket in splits.items():
        humans = [h for c in bucket["cases"] for h in c.get("human") or []]
        result_splits[split] = {
            "aggregate": metrics.aggregate(bucket["cases"], bucket["per_repeat"]),
            "cases": bucket["cases"],
            "human_ceiling": {"pairs": len(humans),
                              "tiou_mean": metrics._mean(h.get("tiou_mean") for h in humans),
                              "hit_05": metrics._mean(h.get("hit_05") for h in humans)} if humans else None,
            "weak": reviews.summarize(weak_rows[split]),
        }
    result = {"run": name, "variant": info.get("variant"), "variant_env": info.get("variant_env") or {},
              "stage": info.get("stage"), "configuration": info.get("configuration") or {},
              "configuration_hash": info.get("configuration_hash"), "git_commit": info.get("git_commit", ""),
              "spent_usd": info.get("spent_usd"), "scored_at": _now(), "metrics_version": metrics.METRICS_VERSION,
              "splits": result_splits, "skipped": skipped, "prices": info.get("prices") or {}}
    case_store._write_json(directory / "score.json", result)
    (directory / "report.md").write_text(report.run_report(result), encoding="utf-8")
    return result


# ============================================================
# COMPARE / GATE
# ============================================================

def gate_check(result_a: Mapping[str, Any], result_b: Mapping[str, Any], *, threshold: float = 0.1,
               judge: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail when holdout HIT@1 or Recall@3 of B drops by more than ``threshold`` vs A (fail closed)."""
    checks = []
    failed = False
    holdout_a = {c["case_id"]: c for c in result_a["splits"]["holdout"]["cases"]}
    holdout_b = {c["case_id"]: c for c in result_b["splits"]["holdout"]["cases"]}
    common = sorted(set(holdout_a) & set(holdout_b))
    if not common:
        return {"passed": False, "checks": [], "reason": "no holdout video was scored by both runs: the gate "
                                                         "cannot pass without holdout evidence"}
    for name in metrics.GATE_METRICS:
        delta = metrics.paired_delta({k: holdout_a[k]["metrics"].get(name) for k in common},
                                     {k: holdout_b[k]["metrics"].get(name) for k in common})
        bad = delta["delta"] is not None and delta["delta"] < -float(threshold) - 1e-9
        failed |= bad
        checks.append({"metric": name, "label": metrics.METRICS[name][1], "videos": delta["videos"],
                       "delta": delta["delta"], "ci95": delta["ci95"], "failed": bad})
    judge_check = None
    if judge and judge.get("pairs"):
        if judge.get("trusted"):
            net = (judge.get("losses", 0) - judge.get("wins", 0)) / max(1, judge["pairs"])
            judge_check = {"net_loss_rate": net, "failed": net > float(threshold)}
            failed |= judge_check["failed"]
        else:
            judge_check = {"ignored": "judge not calibrated or kappa < 0.6 (untrusted)"}
    return {"passed": not failed, "checks": checks, "judge": judge_check, "threshold": threshold,
            "videos": len(common)}


def compare(name_a: str, name_b: str, *, root: str | Path | None = None, gate: bool = False,
            threshold: float = 0.1, judge_result: Mapping[str, Any] | None = None,
            out: str | Path | None = None) -> tuple[str, dict[str, Any] | None]:
    """Markdown comparison (A = baseline, B = candidate) and, with ``gate``, the gate decision."""
    result_a, result_b = score_run(name_a, root=root), score_run(name_b, root=root)
    contexts = {c.case_id: c for c in case_store.list_cases(root)}
    decision = gate_check(result_a, result_b, threshold=threshold, judge=judge_result) if gate else None
    text = report.compare_report(result_a, result_b, cases=contexts, gate=decision, judge=judge_result)
    target = Path(out) if out else case_store.eval_root(root) / "compare.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text, decision


def list_runs(root: str | Path | None = None) -> str:
    base = case_store.eval_root(root) / "runs"
    rows = []
    for path in sorted(base.glob("*/run.json")):
        info = case_store._read_json(path) or {}
        hyps = list(path.parent.glob("hyp/*/r*.json"))
        hyps = [h for h in hyps if re.fullmatch(r"r\d+\.json", h.name)]
        ok = sum(1 for h in hyps if (case_store._read_json(h) or {}).get("status") == "ok")
        rows.append(f"{path.parent.name[:44]:<44} {str(info.get('variant', ''))[:16]:<16} {info.get('stage', ''):<5} "
                    f"{ok:>4}/{len(hyps):<4} ${float(info.get('spent_usd') or 0):>7.2f}  {info.get('updated_at', '')}")
    if not rows:
        return f"no runs under {base} (use `run`)"
    return "\n".join([f"{'run':<44} {'variant':<16} {'stage':<5} {'ok/jobs':>9} {'spent':>8}  updated", *rows])


def status_table(root: str | Path | None = None) -> str:
    from ai.clip_eval import reviews

    cases = case_store.list_cases(root)
    if not cases:
        return f"no cases under {case_store.eval_root(root) / 'cases'} (use `prepare`)"
    lines = [f"{'case':<36} {'split':<8} {'length':>9} {'transcript':<10} {'visual':<6} {'labels':<9} moments"]
    for case in cases:
        labels = case.labels()
        moments = (f"{len(labels.moments)} ({len(labels.great)} great, {len(labels.negatives)} never)"
                   if labels else "-")
        others = [f"{n}:{case.label_status(n)}" for n in case.annotators() if n != case.gold_annotator]
        lines.append(f"{case.case_id[:36]:<36} {case.split:<8} {case_store.clock(case.duration):>9} "
                     f"{'ok' if case.has_transcript() else 'MISSING':<10} "
                     f"{'yes' if case.visual_report_path else 'no':<6} {case.label_status():<9} {moments}"
                     + (f"  [{', '.join(others)}]" if others else ""))
        if not case.has_transcript():
            lines.append(f"{'':<4}transcript: {case_store.transcribe_command(case.video_path)}")
    ready = [c for c in cases if c.label_status() == "ready"]
    holdout = [c for c in cases if c.split == "holdout"]
    lines.append(f"\n{len(ready)}/{len(cases)} cases have finished labels (only those are scored); "
                 f"{len(holdout)} holdout")
    if len(holdout) < 3:
        lines.append("note: reserve at least 3 videos for holdout (prepare --split holdout, or `split <case> holdout`)")
    weak = reviews.count(root)
    if weak:
        lines.append(f"{weak} weak label(s) harvested from review sheets (reported separately, never shown while labeling)")
    return "\n".join(lines)
