"""Final-caption truth for the selected short: WHAT (frozen) then WHEN (aligned).

    1. analysis audio   one mono 16 kHz PCM WAV from the exact final edited clip
    2. chunk plan       one window normally; low-energy boundaries past the aligner limit
    3. listen           Qwen pass A + pass B per window (OpenAI only as bounded fallback)
    4. resolve + freeze deterministic agreement / Astra over local evidence -> FrozenTranscript
    5. align            ONE provider times the frozen words (validated, locally recovered,
                        else the fallback provider re-times everything)
    6. words            canonical timed words for the speaker profile; the frozen
                        signature is re-checked so no step can have rewritten a word

The returned words carry per-word lexical provenance (``lexical_status``:
agreed | resolved | uncertain) that caption truth, the name lock and the human
review read directly. WHO (diarization) and HOW (presentation) happen elsewhere.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ai.caption_stack import alignment, lexical
from ai.caption_stack import audio as analysis_audio
from ai.caption_stack.config import ALIGNER_MAX_INPUT_SECONDS, CaptionStackSettings, load_settings

FINAL_CAPTIONS_VERSION = 1
CONTRACT = "mimir_caption_stack_v1"
TIMING_BASIS = "exact_final_short_audio"
_STATUS_RANK = {"agreed": 0, "resolved": 1, "uncertain": 2}


def _name_similarity() -> float:
    try:
        value = float(os.getenv("MIMIR_CAPTION_KNOWN_NAME_SIMILARITY", "0.64") or 0.64)
    except ValueError:
        value = 0.64
    return max(0.55, min(0.95, value))


def profile_words(frozen: lexical.FrozenTranscript, units: Sequence[alignment.AlignedUnit]) -> list[dict[str, Any]]:
    """Timed units -> speaker-profile words (text is the frozen tokens, byte for byte)."""
    words: list[dict[str, Any]] = []
    for unit in units:
        ids = range(unit.token_start, unit.token_end)
        status = max((frozen.status[i] for i in ids), key=lambda s: _STATUS_RANK.get(s, 0))
        word: dict[str, Any] = {
            "word": " ".join(frozen.tokens[unit.token_start:unit.token_end]),
            "edited_start": round(float(unit.start), 3),
            "edited_end": round(float(unit.end), 3),
            "timing_source": unit.source,
            "token_ids": [unit.token_start, unit.token_end],
            "lexical_status": status,
        }
        span = next((s for s in (frozen.span_of(i) for i in ids) if s is not None), None)
        if span is not None:
            word["lexical_span"] = str(span["span_id"])
            if unit.token_end - unit.token_start == 1 and span["alternatives"]:
                word["lexical_alternatives"] = [dict(a) for a in span["alternatives"]]
        words.append(word)
    return words


def transcribe_final_short(
    edited_clip_path: str | Path,
    duration: float,
    *,
    participant_names: Sequence[str] = (),
    verified_terms: Sequence[str] = (),
    settings: CaptionStackSettings | None = None,
    ears: lexical.Ears | None = None,
    providers: tuple[Any, Any] | None = None,
    judge: Any = "default",
) -> tuple[list[dict[str, Any]], float, str, dict[str, Any]]:
    """(words, lexical parity, text source, caption quality) for the final edited short.

    ``participant_names``: human-verified speaker names (spelling references and
    direct-address orthography). ``verified_terms``: other user-verified
    names/terms (creator, entities); spelling references only.
    """
    from ai.editor import caption_judge

    settings = settings or load_settings()
    audio = analysis_audio.extract_analysis_audio(edited_clip_path)
    analysis_audio.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="final_", dir=str(analysis_audio.TEMP_DIR)))
    try:
        terms = lexical.clean_verified_terms([*participant_names, *verified_terms])
        chunks = analysis_audio.plan_chunks(audio, ALIGNER_MAX_INPUT_SECONDS)
        ears = ears or lexical.Ears()
        passes, degradations = lexical.listen_passes(audio, chunks, settings=settings, verified_terms=terms,
                                                     ears=ears, workdir=workdir)
        primary, fallback = providers or alignment.build_providers(settings)
        cache = alignment.AlignmentCache()
        locate = alignment.locator(primary, audio, chunks, settings.language, workdir, cache)
        judge_fn = caption_judge.default_judge() if judge == "default" else judge
        frozen, lexical_report = lexical.resolve_and_freeze(
            audio, chunks, passes, settings=settings, verified_terms=terms, orthography_names=participant_names,
            locator=locate, judge=judge_fn, ears=ears, workdir=workdir, name_similarity=_name_similarity())
        outcome = alignment.align_frozen(frozen, audio, primary=primary, fallback=fallback,
                                         language=settings.language, workdir=workdir, cache=cache)
        if not frozen.intact():
            raise RuntimeError("frozen caption transcript changed after the freeze")
        words = profile_words(frozen, outcome.units)
        if " ".join(w["word"] for w in words).split() != list(frozen.tokens):
            raise RuntimeError("timed words differ from the frozen transcript")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
        try:
            audio.path.unlink(missing_ok=True)
        except OSError:
            pass

    if outcome.degraded:
        failed = next(a for a in outcome.attempts if a["status"] == "failed")
        degradations.append({"subsystem": "caption_word_alignment", "level": outcome.provider,
                             "reason": f"{failed['provider']}: {failed['reason']}"})
    passes_meta = [ear for pair in passes for ear in pair]
    text_provider = passes_meta[0]
    agreement_values = lexical_report.get("agreement") or [1.0]
    uncertain_spans = int(lexical_report.get("uncertain_spans", 0))
    quality: dict[str, Any] = {
        "contract": CONTRACT,
        "version": FINAL_CAPTIONS_VERSION,
        "authorities": {
            "lexical": f"{text_provider.provider}:{text_provider.model} ears -> deterministic agreement / "
                       "caption judge -> frozen transcript",
            "timing": f"{outcome.provider} over the frozen words (one authority per run)",
            "speaker": "diarization + human identity; never changes words or times",
            "display": "deterministic caption presentation",
        },
        "settings": settings.public(),
        "analysis_audio": audio.describe(),
        "chunks": [[round(a, 3), round(b, 3)] for a, b in chunks],
        "lexical": lexical_report,
        "lexical_signature": frozen.signature,
        "alignment": {"provider": outcome.provider, **outcome.describe, "attempts": outcome.attempts,
                      "local_recoveries": outcome.recoveries, "reused_locator_alignment": outcome.reused_locator,
                      "version": alignment.ALIGNMENT_VERSION},
        "degradations": degradations,
        # Summary keys read by the pipeline QA line, caption truth and the name lock.
        "text_model": f"{text_provider.provider}:{text_provider.model}",
        "timing_model": outcome.describe.get("model", outcome.provider),
        "timing_authorities": 1,
        "speaker_timing_authority": False,
        "quality_score": round(sum(agreement_values) / len(agreement_values), 4),
        "target_met": uncertain_spans == 0 and not degradations,
        "local_corrections": int(lexical_report.get("changed_spans", 0)),
        "unresolved_local_conflicts": uncertain_spans,
        "full_asr_passes": len(passes_meta) + int(lexical_report.get("fallback_calls", {}).get("full", 0)),
        "local_asr_passes": int(lexical_report.get("fallback_calls", {}).get("local", 0)),
        "lexical_judge": lexical_report.get("judge", {"status": "not_needed"}),
        "known_names": lexical.name_keys(participant_names),
        "known_name_flagged_spans": lexical.known_name_suspicions(frozen.tokens, participant_names,
                                                                  similarity=_name_similarity()),
    }
    return words, 1.0, f"{text_provider.provider}:{text_provider.model}", quality
