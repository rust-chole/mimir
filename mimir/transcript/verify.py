"""Stage ``caption_verify``: caption-grade lexical + timing truth for the selected story window.

Ported CLEAN V3 caption accuracy policy (accuracy over cost, but expensive
evidence only where it matters):

1. full-window ears: primary lexical ear (acoustic, domain keywords), the
   timing ear (word clock), a model-diverse cross-check ear and a precision
   ear (verified vocabulary hints);
2. suspect spans = local disagreements between the ears + near-miss
   spellings of verified entities (never a whole-transcript span);
3. micro verification per suspect span on 6-10 s of audio: three
   independent ears must agree 3/3 to rewrite, otherwise two more ears and
   4/5 are required; deletions are never automatic;
4. the corrected text is re-aligned to the SAME immutable clock (timing truth
   is never re-derived from the lexical correction), then the backward-only
   PCM guard fixes proven late phrase starts;
5. words the ears could not settle are flagged ``uncertain`` (shown, never
   emphasized) - nothing is guessed.
"""
from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Sequence

from mimir.config import Settings, routes_for, section
from mimir.core.stage import StageContext, StageOutput
from mimir.errors import StageError
from mimir.media.audio import extract_wav, read_wav_mono
from mimir.media.probe import MediaInfo
from mimir.models.provider import AudioMeta, Transcription
from mimir.transcript.align import AlignedWord, AlignmentError, align_text_to_clock, text_agreement
from mimir.transcript.clock_guard import apply_clock_guard
from mimir.transcript.prompts import TRANSCRIPTION_PROMPT, micro_prompt, names_context
from mimir.transcript.tokens import (
    canon_phrase,
    canonical_word,
    preserve_case_and_punctuation,
    tokenize,
)

WINDOW_PAD = 1.0
DIRECT_ADDRESS_CUES = {"you", "your", "hey", "girl", "bro", "look", "good", "damn", "yo"}


# ============================================================ suspect spans

def edit_proposals(base_tokens: Sequence[str], ref_tokens: Sequence[str]) -> list[tuple[int, int, tuple[str, ...]]]:
    a = [canonical_word(t) for t in base_tokens]
    b = [canonical_word(t) for t in ref_tokens]
    rows = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            rows.append((i1, i2, tuple(ref_tokens[j1:j2])))
    return rows


def disagreement_spans(base_tokens: Sequence[str], refs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    count = len(base_tokens)
    rows: list[dict[str, Any]] = []
    for ref_index, ref in enumerate(refs, start=1):
        for start, end, replacement in edit_proposals(base_tokens, ref):
            if start == end:  # insertion in the reference: check the neighbours
                start = max(0, min(count - 1, start - 1)) if count else 0
                end = min(count, start + 2)
            if end <= start or start >= count:
                continue
            rows.append({"start": start, "end": min(count, max(start + 1, end)), "severity": 1.0,
                         "source": "asr_disagreement", "reason": f"full-window ear {ref_index} disagrees",
                         "alternative": " ".join(replacement)[:120]})
    return rows


def known_name_spans(base_tokens: Sequence[str], names: Sequence[str], similarity: float) -> list[dict[str, Any]]:
    known = sorted({canonical_word(part) for name in names for part in str(name).split()
                    if len(canonical_word(part)) >= 3})
    rows: list[dict[str, Any]] = []
    for index, token in enumerate(base_tokens):
        value = canonical_word(token)
        if len(value) < 3 or value in known:
            continue
        for name in known:
            if value[:2] != name[:2] or abs(len(value) - len(name)) > 2:
                continue
            ratio = SequenceMatcher(None, value, name, autojunk=False).ratio()
            if ratio >= similarity:
                rows.append({"start": index, "end": index + 1, "severity": min(1.0, 0.82 + 0.18 * ratio),
                             "source": "known_name", "reason": f"possible known-name spelling: {token} vs {name}",
                             "name": name})
                break
    return rows


def merge_spans(rows: Sequence[dict[str, Any]], token_count: int, *, max_words: int, max_spans: int
                ) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        start = max(0, min(token_count, int(row["start"])))
        end = max(start + 1, min(token_count, int(row["end"])))
        if end > start:
            clean.append({"start": start, "end": end, "severity": float(row.get("severity", 0.0)),
                          "sources": [str(row.get("source", ""))], "reasons": [str(row.get("reason", ""))[:200]]})
    clean.sort(key=lambda r: (r["start"], r["end"]))
    merged: list[dict[str, Any]] = []
    for row in clean:
        # merge only genuinely overlapping doubts and never grow a transcript-sized span
        if merged and row["start"] < merged[-1]["end"] and \
                max(merged[-1]["end"], row["end"]) - merged[-1]["start"] <= max_words:
            last = merged[-1]
            last["end"] = max(last["end"], row["end"])
            last["severity"] = max(last["severity"], row["severity"])
            last["sources"] = list(dict.fromkeys(last["sources"] + row["sources"]))
            last["reasons"] = list(dict.fromkeys(last["reasons"] + row["reasons"]))[:4]
        else:
            merged.append(dict(row))
    ranked = sorted(merged, key=lambda r: ("asr_disagreement" not in r["sources"], -r["severity"], r["start"]))
    return sorted(ranked[:max_spans], key=lambda r: r["start"])


def phrase_for_core(window_tokens: Sequence[str], heard: Sequence[str], core: tuple[int, int]) -> tuple[str, ...]:
    """The part of an ear's transcript that corresponds to ``core`` of the window tokens."""
    c0, c1 = core
    a = [canonical_word(t) for t in window_tokens]
    b = [canonical_word(t) for t in heard]
    lo_b: int | None = None
    hi_b: int | None = None
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "insert":
            if c0 < i1 < c1:
                lo_b = j1 if lo_b is None else min(lo_b, j1)
                hi_b = j2 if hi_b is None else max(hi_b, j2)
            continue
        if i2 <= c0 or i1 >= c1:
            continue
        if tag == "equal":
            lo, hi = max(i1, c0), min(i2, c1)
            jb0, jb1 = j1 + (lo - i1), j1 + (hi - i1)
        else:
            jb0, jb1 = j1, j2
        lo_b = jb0 if lo_b is None else min(lo_b, jb0)
        hi_b = jb1 if hi_b is None else max(hi_b, jb1)
    if lo_b is None or hi_b is None:
        return ()
    return tuple(heard[lo_b:hi_b])


# ============================================================ micro verification

@dataclass
class Ear:
    label: str
    role: str
    view: str            # raw | enhanced
    prompted: bool       # received context / vocabulary hints


ROUND_ONE = (Ear("raw acoustic", "transcribe_primary", "raw", False),
             Ear("enhanced acoustic", "transcribe_primary", "enhanced", False),
             Ear("context diverse", "transcribe_crosscheck", "raw", True))
ROUND_TWO = (Ear("enhanced diverse", "transcribe_crosscheck", "enhanced", True),
             Ear("context precision", "transcribe_primary", "raw", True))


def vote(window_tokens: Sequence[str], transcripts: Sequence[tuple[Ear, Transcription]], core: tuple[int, int]
         ) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], dict[str, Any]] = {}
    for ear, result in transcripts:
        text = result.text.strip()
        if not text:
            continue
        phrase = phrase_for_core(window_tokens, tokenize(text), core)
        slot = buckets.setdefault(canon_phrase(phrase), {"phrase": phrase, "votes": 0, "ears": []})
        slot["votes"] += 1
        slot["ears"].append(ear.label)
    return sorted(buckets.values(), key=lambda r: (r["votes"], len(canon_phrase(r["phrase"]))), reverse=True)


def alias_resolution(current: Sequence[str], candidates: Sequence[dict[str, Any]], context: Sequence[str],
                     aliases: dict[str, str]) -> list[str] | None:
    """Orthographic alias only when an independent ear emitted the preferred spelling (never invents words)."""
    if not ({canonical_word(t) for t in context} & DIRECT_ADDRESS_CUES):
        return None
    heard = set()
    for row in candidates:
        heard |= set(canon_phrase(row["phrase"]))
    updated = list(current)
    changed = False
    for index, token in enumerate(current):
        preferred = aliases.get(canonical_word(token))
        if preferred and preferred in heard:
            updated[index] = preserve_case_and_punctuation(token, preferred)
            changed = True
    return updated if changed else None


# ============================================================ stage

class CaptionVerifyStage:
    name = "caption_verify"
    version = 1
    deps = ("source", "probe", "story")

    def params(self, settings: Settings) -> Any:
        return {
            "verify": section(settings, "caption_verify"),
            "language": settings.transcript.language,
            "keywords": list(settings.transcript.domain_keywords),
            "vocabulary": self.vocabulary(settings),
            "routes": routes_for(settings, "transcribe_primary", "transcribe_crosscheck", "timing"),
        }

    @staticmethod
    def vocabulary(settings: Settings) -> list[str]:
        names = [*settings.identity.entities, settings.identity.creator]
        return [n for n in dict.fromkeys(" ".join(str(n).split()) for n in names) if n][:12]

    def run(self, ctx: StageContext) -> StageOutput:
        info = MediaInfo.from_dict(ctx.dep("probe").json("media"))
        story = ctx.dep("story").json("story")
        cfg = ctx.settings.caption_verify
        language = ctx.settings.transcript.language
        vocabulary = self.vocabulary(ctx.settings)
        window_start = max(0.0, float(story["start"]) - WINDOW_PAD)
        window_end = min(info.duration, float(story["end"]) + WINDOW_PAD)
        duration = window_end - window_start
        temp = ctx.out_dir / "tmp"
        raw = extract_wav(ctx.source.path, temp / "window.wav", start=window_start, duration=duration)
        enhanced = extract_wav(ctx.source.path, temp / "window_enhanced.wav", start=window_start,
                               duration=duration, enhance=True)
        meta = AudioMeta(window_start, window_end, "raw")
        try:
            result = self._verify(ctx, raw, enhanced, window_start, duration, vocabulary, language, meta)
        finally:
            shutil.rmtree(temp, ignore_errors=True)
        return StageOutput(data={"caption_words": result})

    # ------------------------------------------------------------------

    def _ear(self, ctx: StageContext, role: str, path: Path, *, language: str, meta: AudioMeta,
             prompt: str | None = None, keywords: Sequence[str] = (), words: bool = False,
             logprobs: bool = False) -> Transcription:
        return ctx.provider.transcribe(role, ctx.settings.route(role), path, language=language, prompt=prompt,
                                       keywords=keywords, word_timestamps=words, logprobs=logprobs, meta=meta)

    def _verify(self, ctx: StageContext, raw: Path, enhanced: Path, offset: float, duration: float,
                vocabulary: list[str], language: str, meta: AudioMeta) -> dict[str, Any]:
        cfg = ctx.settings.caption_verify
        keywords = list(ctx.settings.transcript.domain_keywords)
        precision_keywords = [*keywords, *vocabulary]
        enhanced_meta = AudioMeta(meta.source_start, meta.source_end, "enhanced")
        with ThreadPoolExecutor(max_workers=4) as pool:
            primary_f = pool.submit(self._ear, ctx, "transcribe_primary", raw, language=language, meta=meta,
                                    keywords=keywords)
            timing_f = pool.submit(self._ear, ctx, "timing", raw, language=language, meta=meta, words=True)
            cross_f = pool.submit(self._ear, ctx, "transcribe_crosscheck", raw, language=language, meta=meta,
                                  logprobs=True)
            if vocabulary:
                precision_f = pool.submit(self._ear, ctx, "transcribe_primary", raw, language=language, meta=meta,
                                          prompt=names_context(vocabulary), keywords=precision_keywords)
            else:
                precision_f = pool.submit(self._ear, ctx, "transcribe_crosscheck", enhanced, language=language,
                                          meta=enhanced_meta, logprobs=True)
            primary, timing, cross, precision = (primary_f.result(), timing_f.result(), cross_f.result(),
                                                 precision_f.result())
        if not primary.text.strip():
            primary = self._ear(ctx, "transcribe_primary", raw, language=language, meta=meta,
                                prompt=TRANSCRIPTION_PROMPT, keywords=keywords)
        if not primary.text.strip() and not timing.words:
            ctx.ledger.info("no_speech_in_story", "the story window contains no transcribable speech")
            return {"window": [round(offset, 3), round(offset + duration, 3)], "words": [], "quality": {},
                    "micro": {"checked_spans": 0, "details": []}, "clock_guard": {"status": "empty"}}
        if not primary.text.strip():
            raise StageError(self.name, "primary caption ear returned no text for a window with speech")
        if not timing.words:
            raise StageError(self.name, "timing ear returned no word clock for the story window")

        base_tokens = tokenize(primary.text)
        try:
            base_aligned, _ = align_text_to_clock(primary.text, timing.words, duration)
        except AlignmentError as error:
            raise StageError(self.name, f"caption clock alignment failed: {error}") from error
        token_rows = token_to_row(base_aligned)
        refs = [tokenize(cross.text), tokenize(precision.text)]
        spans = merge_spans(
            disagreement_spans(base_tokens, [r for r in refs if r])
            + known_name_spans(base_tokens, vocabulary, cfg.known_name_similarity),
            len(base_tokens), max_words=cfg.suspect_max_words, max_spans=cfg.max_suspect_spans)

        aliases = dict(cfg.orthographic_aliases)
        # every span is judged against the same primary wording (spans never overlap)
        details = [self._micro_span(ctx, span, base_tokens, base_aligned, token_rows, raw, enhanced, offset,
                                    duration, vocabulary, language, aliases) for span in spans]
        tokens = list(base_tokens)
        for detail in sorted(details, key=lambda d: d["core"][0], reverse=True):  # right to left
            if detail["selected_phrase"] is None:
                continue
            c0, c1 = detail["core"]
            new_tokens = tokenize(detail["selected_phrase"])
            if canon_phrase(tokens[c0:c1]) != canon_phrase(new_tokens):
                tokens[c0:c1] = new_tokens
                detail["applied"] = True
        # final token positions of unresolved (disputed) spans after length-changing corrections
        disputed: dict[int, tuple[dict[str, Any], int]] = {}
        for detail in details:
            if detail["selected_phrase"] is not None:
                continue
            c0, c1 = detail["core"]
            shift = sum(len(tokenize(d["selected_phrase"])) - (d["core"][1] - d["core"][0])
                        for d in details if d.get("applied") and d["core"][1] <= c0)
            for position in range(c0, c1):
                disputed[position + shift] = (detail, position - c0)
        applied_positions: set[int] = set()
        for detail in details:
            if detail.get("applied"):
                c0 = detail["core"][0]
                shift = sum(len(tokenize(d["selected_phrase"])) - (d["core"][1] - d["core"][0])
                            for d in details if d.get("applied") and d["core"][1] <= c0)
                applied_positions |= {c0 + shift + k for k in range(len(tokenize(detail["selected_phrase"])))}

        final_text = " ".join(tokens)
        try:
            aligned, ratio = align_text_to_clock(final_text, timing.words, duration)
        except AlignmentError as error:
            raise StageError(self.name, f"caption clock re-alignment failed: {error}") from error
        final_rows = token_to_row(aligned)
        words: list[dict[str, Any]] = []
        for index, row in enumerate(aligned):
            words.append({"id": f"c{index:05d}", "text": row.text, "start": round(row.start + offset, 3),
                          "end": round(row.end + offset, 3), "timing_source": row.source,
                          "lexical_source": "primary", "uncertain": False, "alternatives": []})
        for position, row_index in final_rows.items():
            if position in applied_positions:
                words[row_index]["lexical_source"] = "micro_vote"
        # disputed tokens (unresolved) keep the primary spelling; token-level majority decides uncertainty
        for position, (detail, offset_in_core) in disputed.items():
            if position >= len(tokens) or position not in final_rows:
                continue
            row_index = final_rows[position]
            observations = detail["observations"].get(offset_in_core, [])
            words[row_index]["alternatives"] = observations
            if not self._majority_confirms(tokens[position], observations):
                words[row_index]["uncertain"] = True
        samples, rate = read_wav_mono(raw)
        clock_report: dict[str, Any] = {"status": "disabled"}
        if cfg.clock_guard:
            words, clock_report = apply_clock_guard(words, samples, rate, offset=offset, duration=duration)
        agreements = [text_agreement(final_text, ref.text) for ref in (cross, precision) if ref.text.strip()]
        best = max(agreements) if agreements else 0.0
        quality = {"reference_agreement": round(best, 4), "target": cfg.word_accuracy_target,
                   "target_met": best >= cfg.word_accuracy_target and not disputed,
                   "alignment_ratio": round(ratio, 4), "suspect_spans": len(spans),
                   "corrected_spans": sum(1 for d in details if d.get("applied")),
                   "unresolved_spans": sum(1 for d in details if d["selected_phrase"] is None),
                   "uncertain_words": sum(1 for w in words if w["uncertain"])}
        if not quality["target_met"]:
            ctx.ledger.warning("caption_consensus_below_target",
                               f"ASR consensus {best:.3f} (target {cfg.word_accuracy_target}); primary wording kept "
                               "where ears disagreed, disputed words flagged uncertain")
        return {"window": [round(offset, 3), round(offset + duration, 3)], "words": words, "quality": quality,
                "ears": {"primary": primary.text, "crosscheck": cross.text, "precision": precision.text},
                "micro": {"checked_spans": len(spans), "details": details},
                "clock_guard": clock_report}

    @staticmethod
    def _majority_confirms(token: str, observations: Sequence[dict[str, Any]]) -> bool:
        independent = [o for o in observations if not o.get("prompted")]
        if len(observations) < 3:
            return False
        agree = sum(1 for o in independent if canonical_word(o["token"]) == canonical_word(token))
        return agree * 2 > len(independent) and agree >= 2

    def _micro_span(self, ctx: StageContext, span: dict[str, Any], tokens: list[str],
                    aligned: Sequence[AlignedWord], token_rows: dict[int, int], raw: Path, enhanced: Path,
                    offset: float, duration: float, vocabulary: list[str], language: str,
                    aliases: dict[str, str]) -> dict[str, Any]:
        cfg = ctx.settings.caption_verify
        count = len(tokens)
        c0 = max(0, min(count - 1, int(span["start"])))
        c1 = max(c0 + 1, min(count, int(span["end"])))
        w0 = max(0, c0 - cfg.micro_context_words)
        w1 = min(count, c1 + cfg.micro_context_words)
        first_row = aligned[token_rows.get(w0, 0)]
        last_row = aligned[token_rows.get(w1 - 1, len(aligned) - 1)]
        t0 = max(0.0, first_row.start - cfg.micro_pad_seconds)
        t1 = min(duration, last_row.end + cfg.micro_pad_seconds)
        t0, t1 = expand_window(t0, t1, duration, cfg.micro_min_seconds, cfg.micro_max_seconds)
        window_tokens = tokens[w0:w1]
        core = (c0 - w0, c1 - w0)
        before = " ".join(tokens[max(0, w0 - 12):w0])
        after = " ".join(tokens[w1:w1 + 12])
        micro_dir = raw.parent / f"micro_{c0}_{c1}"
        views = {
            "raw": extract_wav(raw, micro_dir / "raw.wav", start=t0, duration=t1 - t0),
            "enhanced": extract_wav(enhanced, micro_dir / "enhanced.wav", start=t0, duration=t1 - t0),
        }
        transcripts: list[tuple[Ear, Transcription]] = []

        def listen(ears: Sequence[Ear]) -> None:
            def one(ear: Ear) -> Transcription:
                instruction = ("Context may disambiguate, but audio always wins." if ear.prompted
                               else "Acoustic-only listen; preserve tiny function words; no guessing.")
                prompt = micro_prompt(before, after, instruction, vocabulary) if ear.prompted else None
                meta = AudioMeta(offset + t0, offset + t1, ear.view, {"micro_core": [c0, c1]})
                return self._ear(ctx, ear.role, views[ear.view], language=language, meta=meta, prompt=prompt,
                                 logprobs=ear.role != "timing")
            with ThreadPoolExecutor(max_workers=min(cfg.parallel_workers, len(ears))) as pool:
                for ear, result in zip(ears, pool.map(one, ears)):
                    transcripts.append((ear, result))

        listen(ROUND_ONE)
        candidates = vote(window_tokens, transcripts, core)
        selected: str | None = None
        source = ""
        if candidates and candidates[0]["votes"] >= 3 and candidates[0]["phrase"]:
            selected, source = " ".join(candidates[0]["phrase"]), f"micro_{candidates[0]['votes']}_of_3"
        if selected is None:
            listen(ROUND_TWO)
            candidates = vote(window_tokens, transcripts, core)
            if candidates and candidates[0]["votes"] >= 4 and candidates[0]["phrase"]:
                selected = " ".join(candidates[0]["phrase"])
                source = f"micro_{candidates[0]['votes']}_of_{len(transcripts)}"
        if selected is None and candidates:
            resolved = alias_resolution(tokens[c0:c1], candidates, tokens[max(0, c0 - 6):c1 + 6], aliases)
            if resolved is not None:
                selected, source = " ".join(resolved), "orthographic_alias_with_ear_evidence"
        shutil.rmtree(micro_dir, ignore_errors=True)
        observations = token_observations(window_tokens, transcripts, core)
        return {"span": [c0, c1], "core": [c0, c1], "sources": span["sources"], "reasons": span["reasons"],
                "audio_window": [round(offset + t0, 3), round(offset + t1, 3)],
                "ears": len(transcripts),
                "candidates": [{"phrase": " ".join(c["phrase"]), "votes": c["votes"], "ears": c["ears"]}
                               for c in candidates[:6]],
                "selected_phrase": selected, "selected_source": source, "applied": False,
                "observations": observations}


def expand_window(t0: float, t1: float, duration: float, minimum: float, maximum: float) -> tuple[float, float]:
    t0, t1 = max(0.0, t0), min(duration, max(t0 + 0.2, t1))
    if t1 - t0 < minimum:
        missing = minimum - (t1 - t0)
        t0, t1 = max(0.0, t0 - missing / 2), min(duration, t1 + missing / 2)
        if t1 - t0 < minimum:
            if t0 <= 0.001:
                t1 = min(duration, minimum)
            else:
                t0 = max(0.0, duration - minimum)
    if t1 - t0 > maximum:
        center = (t0 + t1) / 2
        t0 = max(0.0, center - maximum / 2)
        t1 = min(duration, t0 + maximum)
    return t0, t1


def token_to_row(aligned: Sequence[AlignedWord]) -> dict[int, int]:
    """Lexical token index -> aligned row index (a coincident group row holds several tokens)."""
    mapping: dict[int, int] = {}
    position = 0
    for row_index, row in enumerate(aligned):
        for _ in tokenize(row.text):
            mapping[position] = row_index
            position += 1
    return mapping


def token_observations(window_tokens: Sequence[str], transcripts: Sequence[tuple[Ear, Transcription]],
                       core: tuple[int, int]) -> dict[int, list[dict[str, Any]]]:
    """Per core-token: what each ear heard at that position (with token probability when reported)."""
    result: dict[int, list[dict[str, Any]]] = {}
    size = core[1] - core[0]
    for ear, transcription in transcripts:
        phrase = phrase_for_core(window_tokens, tokenize(transcription.text), core)
        if len(phrase) != size:
            continue
        probs = {canonical_word(t): p for t, p in transcription.token_probs}
        for k, token in enumerate(phrase):
            result.setdefault(k, []).append({"token": token, "ear": ear.label, "prompted": ear.prompted,
                                             "probability": probs.get(canonical_word(token))})
    return result
