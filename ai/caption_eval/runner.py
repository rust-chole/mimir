"""Run a system over the cases (cached), score it against the gold, compare runs.

    <root>/runs/<run name>/
        run.json                  system, overrides, when
        hyp/<case_id>.json        one hypothesis per case (cached: same system + overrides + audio)
        score.json                per-case and aggregate metrics against the CURRENT gold
        report.md                 the human-readable report

Hypotheses are cached so that re-scoring after a gold edit, or comparing runs, never
calls a model again. ``--force`` re-runs.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ai.caption_eval import cases as case_store
from ai.caption_eval import metrics, report, systems

RUNNER_VERSION = 1


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_dir(name: str, root: str | Path | None = None) -> Path:
    return case_store.eval_root(root) / "runs" / name


def default_run_name(system: str, overrides: Mapping[str, str]) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", system).strip("-")
    if overrides:
        name += "__" + hashlib.sha256(json.dumps(dict(sorted(overrides.items()))).encode()).hexdigest()[:8]
    return name


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=str(case_store.PROJECT_ROOT), check=False)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


_FINGERPRINT_ENV = ("MIMIR_CAPTION_KEYWORDS", "MIMIR_CAPTION_CONTEXT", "MIMIR_CAPTION_KNOWN_NAME_SIMILARITY",
                    "MIMIR_CAPTION_JUDGE_MIN_CONFIDENCE")


def configuration_fingerprint(system: str) -> str:
    """What decides a hypothesis besides the case: caption settings, judge routing, the stack's code.

    Read inside the run's ``--set`` overrides, so a changed .env or override re-runs the
    case instead of reusing a hypothesis made under another configuration."""
    if system == "published":
        return "published"
    import os

    from ai import model_config
    from ai.caption_stack.config import load_settings

    sources = sorted((case_store.PROJECT_ROOT / "ai" / "caption_stack").glob("*.py"))
    sources.append(case_store.PROJECT_ROOT / "ai" / "editor" / "caption_judge.py")
    code = hashlib.sha256(b"".join(p.read_bytes() for p in sources if p.is_file())).hexdigest()
    blob = json.dumps({"settings": load_settings().public(), "code": code,
                       "judge": [model_config.CAPTION_JUDGE_MODEL, model_config.CAPTION_JUDGE_REASONING_EFFORT],
                       "env": {k: os.getenv(k, "") for k in _FINGERPRINT_ENV}}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _signature(system: str, overrides: Mapping[str, str], case: case_store.Case, configuration: str) -> str:
    blob = json.dumps({"system": system, "overrides": dict(sorted(overrides.items())), "audio": case.audio_sha256,
                       "language": case.language, "names": case.names, "entities": case.entities,
                       "configuration": configuration, "version": RUNNER_VERSION}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_system(system: str, *, name: str | None = None, overrides: Mapping[str, str] | None = None,
               only: Sequence[str] | None = None, jobs: int = 1, force: bool = False, root: str | Path | None = None,
               runner: Callable[[case_store.Case, str], dict[str, Any]] = systems.run,
               log: Callable[[str], None] = print) -> tuple[str, list[dict[str, Any]]]:
    """Hypotheses for every case (gold-ready or not: they are scored once the gold is ready)."""
    if not systems.known(system):
        raise ValueError(f"unknown system {system!r}; see `python -m ai.caption_eval systems`")
    overrides = dict(overrides or {})
    name = name or default_run_name(system, overrides)
    directory = run_dir(name, root)
    (directory / "hyp").mkdir(parents=True, exist_ok=True)
    previous = case_store._read_json(directory / "run.json")
    if isinstance(previous, Mapping) and (previous.get("system") != system or previous.get("overrides") != overrides):
        raise ValueError(f"run {name!r} already holds {previous.get('system')} {previous.get('overrides')}; "
                         "choose another --name")
    case_store._write_json(directory / "run.json", {"system": system, "overrides": overrides, "updated_at": _now(),
                                                    "git_commit": _git_commit(), "version": RUNNER_VERSION})
    cases = case_store.list_cases(root, only)
    results: list[dict[str, Any]] = []

    def one(case: case_store.Case) -> dict[str, Any]:
        target = directory / "hyp" / f"{case.case_id}.json"
        signature = _signature(system, overrides, case, configuration)
        cached = case_store._read_json(target)
        if not force and isinstance(cached, Mapping) and cached.get("signature") == signature \
                and cached.get("status") == "ok":
            return {"case_id": case.case_id, "status": "cached"}
        record: dict[str, Any] = {"case_id": case.case_id, "system": system, "overrides": overrides,
                                  "signature": signature, "configuration": configuration, "created_at": _now()}
        try:
            hypothesis = runner(case, system)
            record.update(status="ok", **hypothesis)
        except Exception as error:  # one case never stops the run
            record.update(status="error", error=f"{type(error).__name__}: {str(error)[:500]}")
        case_store._write_json(target, record)
        return {"case_id": case.case_id, "status": record["status"], "error": record.get("error", ""),
                "elapsed_s": record.get("elapsed_s")}

    with systems.env_overrides(overrides):
        configuration = configuration_fingerprint(system)
        if jobs > 1 and len(cases) > 1:
            with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="mimir-eval") as pool:
                results = list(pool.map(one, cases))
        else:
            for case in cases:
                log(f"  {case.case_id} ...")
                results.append(one(case))
    for row in results:
        if row["status"] == "error":
            log(f"  ! {row['case_id']}: {row['error']}")
    return name, results


def score_run(name: str, *, root: str | Path | None = None, only: Sequence[str] | None = None
              ) -> dict[str, Any]:
    """Score cached hypotheses against the current gold; writes score.json + report.md."""
    directory = run_dir(name, root)
    info = case_store._read_json(directory / "run.json")
    if not isinstance(info, Mapping):
        raise FileNotFoundError(f"no run named {name!r} under {directory.parent}")
    rows: list[dict[str, Any]] = []
    scores: list[metrics.CaseScore] = []
    skipped: list[dict[str, str]] = []
    configurations: set[str] = set()
    for case in case_store.list_cases(root, only):
        status = case.gold_status()
        if status != "ready":
            skipped.append({"case_id": case.case_id, "reason": f"gold {status}"})
            continue
        hyp = case_store._read_json(directory / "hyp" / f"{case.case_id}.json")
        if not isinstance(hyp, Mapping):
            skipped.append({"case_id": case.case_id, "reason": "not run"})
            continue
        if hyp.get("status") != "ok":
            skipped.append({"case_id": case.case_id, "reason": f"error: {hyp.get('error', '')}"})
            continue
        times, reference = case.gold_times()
        gold = metrics.gold_tokens(case.gold_text(), names=[*case.names, *case.entities], times=times)
        coverage = ((hyp.get("meta") or {}).get("quality") or {}).get("coverage") or {}
        score = metrics.score_case(case.case_id, gold, metrics.hypothesis_tokens(hyp.get("words") or []),
                                   coverage=coverage, timing_reference=reference)
        scores.append(score)
        configurations.add(str(hyp.get("configuration", "")))
        rows.append({**score.to_dict(), "elapsed_s": hyp.get("elapsed_s"),
                     "degradations": [f"{d.get('subsystem')} -> {d.get('level')}" for d in
                                      ((hyp.get("meta") or {}).get("quality") or {}).get("degradations", []) or []]})
    result = {"run": name, "system": info.get("system"), "overrides": info.get("overrides") or {},
              "git_commit": info.get("git_commit", ""), "scored_at": _now(), "metrics_version": metrics.METRICS_VERSION,
              "aggregate": metrics.aggregate(scores), "cases": rows, "skipped": skipped,
              "configurations": sorted(configurations)}
    case_store._write_json(directory / "score.json", result)
    (directory / "report.md").write_text(report.run_report(result), encoding="utf-8")
    return result


def compare(names: Sequence[str], *, root: str | Path | None = None) -> str:
    """Markdown comparison of scored runs (scores refreshed against the current gold)."""
    results = [score_run(name, root=root) for name in names]
    text = report.compare_report(results)
    target = case_store.eval_root(root) / "compare.md"
    target.write_text(text, encoding="utf-8")
    return text


def status_table(root: str | Path | None = None) -> str:
    cases = case_store.list_cases(root)
    if not cases:
        return f"no cases under {case_store.eval_root(root) / 'cases'} (use `prepare`)"
    lines = [f"{'case':<48} {'gold':<8} {'sec':>6}  source"]
    for case in cases:
        source = (case.meta.get("source") or {}).get("kind", "")
        lines.append(f"{case.case_id[:48]:<48} {case.gold_status():<8} {case.duration:>6.1f}  {source}")
    ready = sum(1 for c in cases if c.gold_status() == "ready")
    lines.append(f"\n{ready}/{len(cases)} cases have a finished gold transcript")
    return "\n".join(lines)


def align_gold(*, root: str | Path | None = None, only: Sequence[str] | None = None, force: bool = False,
               log: Callable[[str], None] = print) -> list[dict[str, str]]:
    """Reference word times for finished gold transcripts with the configured forced aligner.

    Machine timing, not human: the report labels it "aligner". A human ``gold_words.json``
    (same words as gold.txt) always wins."""
    from ai.caption_stack import alignment
    from ai.caption_stack import audio as analysis_audio
    from ai.caption_stack.config import load_settings

    settings = load_settings()
    provider, _fallback = alignment.build_providers(settings)
    out: list[dict[str, str]] = []
    for case in case_store.list_cases(root, only):
        target = case.directory / "gold_words.auto.json"
        if case.gold_status() != "ready":
            out.append({"case_id": case.case_id, "status": "skipped: gold not ready"})
            continue
        if target.exists() and not force and case.gold_times()[0] is not None:
            out.append({"case_id": case.case_id, "status": "up to date"})
            continue
        tokens = [t for t in metrics.gold_display_tokens(case.gold_text()) if t != metrics._WILDCARD_TOKEN]
        audio = analysis_audio.load_analysis_audio(case.audio_path)
        try:
            with systems.env_overrides({"CAPTION_LANGUAGE": case.language}), \
                    tempfile.TemporaryDirectory(prefix="mimir_eval_align_") as tmp:
                units = provider.align(audio, (0.0, audio.duration), tokens, case.language, Path(tmp))
        except Exception as error:
            out.append({"case_id": case.case_id, "status": f"failed: {type(error).__name__}: {str(error)[:200]}"})
            continue
        words = [{"word": tokens[u.token_start], "start": u.start, "end": u.end} for u in units]
        case_store._write_json(target, {"source": f"{provider.name}:{settings.aligner_model}", "created_at": _now(),
                                        "note": "machine reference timing (not human)", "words": words})
        out.append({"case_id": case.case_id, "status": "aligned"})
        log(f"  {case.case_id}: aligned {len(words)} words")
    return out


def env_from_pairs(pairs: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--set expects KEY=VALUE, got {pair!r}")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"--set key must be an environment variable name, got {key!r}")
        if re.search(r"(^|_)KEY$|SECRET|TOKEN|PASSWORD", key):
            raise ValueError(f"--set {key}: secrets belong in .env, not in a run's recorded overrides")
        values[key] = value
    return values

