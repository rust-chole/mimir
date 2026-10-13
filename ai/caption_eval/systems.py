"""Systems the caption evaluation can score on the same cases.

    published                  what MIMIR actually captioned (the pipeline's speaker profile at
                               ``prepare`` time; no model call, the baseline you already have)
    stack                      the full final-caption stack on the case audio, with the current
                               .env settings plus any ``--set KEY=VALUE`` overrides
    ear:qwen_omni              one Omni listen, primary instructions (no names)
    ear:qwen_omni:names        one Omni listen, precision instructions with the case's names
    ear:qwen3_asr              local Qwen3-ASR, generic vocabulary as context
    ear:qwen3_asr:names        ... plus the case's verified names
    ear:qwen3_asr_flash        DashScope qwen3-asr-flash, generic vocabulary as context
    ear:qwen3_asr_flash:names  ... plus the case's verified names
    ear:openai:<model>         one unprompted OpenAI transcription (gpt-4o-transcribe, whisper-1, ...)

An ear returns text only (WER, critical words, dropped / invented phrases); the stack
and ``published`` also carry word times and uncertain flags.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from ai.caption_eval.cases import Case

SYSTEM_HELP = __doc__


class SystemUnavailable(RuntimeError):
    """The system cannot produce a hypothesis for this case."""


@contextmanager
def env_overrides(values: Mapping[str, str]) -> Iterator[None]:
    """Temporarily set environment variables (``--set``); the caption stack reads them at call time."""
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def known(system: str) -> bool:
    if system in ("published", "stack"):
        return True
    parts = system.split(":")
    if parts[0] != "ear" or len(parts) < 2:
        return False
    if parts[1] == "openai":
        return len(parts) == 3 and bool(parts[2])
    return parts[1] in ("qwen_omni", "qwen3_asr", "qwen3_asr_flash") and parts[2:] in ([], ["names"])


def _words_from_profile(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"word": str(r.get("word", "")), "start": r.get("start"), "end": r.get("end"),
             "status": str(r.get("status", "") or ""), **({"uncertain": True} if r.get("uncertain") else {})}
            for r in rows]


def _run_published(case: Case) -> dict[str, Any]:
    rows = case.published_words()
    if rows is None:
        raise SystemUnavailable("this case has no published captions (prepare it from a pipeline profile)")
    return {"words": _words_from_profile(rows), "meta": {"source": (case.meta.get("source") or {}).get("profile")}}


def _run_stack(case: Case) -> dict[str, Any]:
    from ai.caption_stack import final_captions

    with env_overrides({"CAPTION_LANGUAGE": case.language}):
        words, _parity, source, quality = final_captions.transcribe_final_short(
            case.audio_path, case.duration, participant_names=case.names, verified_terms=case.entities)
    rows = [{"word": w["word"], "start": w.get("edited_start"), "end": w.get("edited_end"),
             "status": w.get("lexical_status", "")} for w in words]
    return {"words": rows, "meta": {"text_source": source, "quality": quality}}


def _chunks(case: Case, workdir: Path) -> list[Path]:
    from ai.caption_stack import audio as analysis_audio
    from ai.caption_stack.config import ALIGNER_MAX_INPUT_SECONDS

    audio = analysis_audio.load_analysis_audio(case.audio_path)
    windows = analysis_audio.plan_chunks(audio, ALIGNER_MAX_INPUT_SECONDS)
    if len(windows) == 1:
        return [case.audio_path]
    return [analysis_audio.cut_window(audio, a, b, workdir / f"chunk_{i:02d}.wav")[0]
            for i, (a, b) in enumerate(windows)]


def _run_ear(case: Case, system: str) -> dict[str, Any]:
    from dataclasses import replace

    from ai.caption_stack import audio as analysis_audio
    from ai.caption_stack import lexical, openai_ears, qwen3_asr, qwen_omni
    from ai.caption_stack.config import load_settings

    parts = system.split(":")
    kind = parts[1]
    with_names = parts[2:] == ["names"]
    names = lexical.clean_verified_terms([*case.names, *case.entities]) if with_names else []
    settings = replace(load_settings(), language=case.language)
    texts: list[str] = []
    meta: dict[str, Any] = {"ear": system, "names_given": names}
    with tempfile.TemporaryDirectory(prefix="mimir_eval_ear_") as tmp:
        for path in _chunks(case, Path(tmp)):
            if kind == "openai":
                texts.append(openai_ears.acoustic_listen(path, model=parts[2], language=case.language))
                meta["model"] = parts[2]
            elif kind == "qwen_omni":
                domain = openai_ears.domain_keywords()
                instructions = (qwen_omni.precision_instructions(case.language, domain, names) if with_names
                                else qwen_omni.primary_instructions(case.language, domain))
                evidence, info = qwen_omni.listen(analysis_audio.wav_base64(path), instructions=instructions,
                                                  settings=settings)
                texts.append(evidence.text)
                meta.update(model=info.get("model"), uncertain_spans=[s.to_dict() for s in evidence.uncertain_spans])
            else:
                context = qwen3_asr.context_text(openai_ears.domain_keywords(), names)
                text, info = qwen3_asr.listen(str(path), lambda p=path: analysis_audio.wav_base64(p), context=context,
                                              settings=settings, provider=kind)
                texts.append(text)
                meta.update(model=info.get("model"), context=context)
    words = [{"word": token} for token in lexical.display_tokens(" ".join(texts))]
    return {"words": words, "meta": meta}


def run(case: Case, system: str) -> dict[str, Any]:
    """One hypothesis: {"words": [...], "meta": {...}, "elapsed_s": float}."""
    if not known(system):
        raise SystemUnavailable(f"unknown system {system!r}")
    started = time.perf_counter()
    if system == "published":
        result = _run_published(case)
    elif system == "stack":
        result = _run_stack(case)
    else:
        result = _run_ear(case, system)
    result["elapsed_s"] = round(time.perf_counter() - started, 2)
    return result
