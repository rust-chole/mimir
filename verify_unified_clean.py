from __future__ import annotations

import ast
import compileall
import inspect
import math
import shutil
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AI = ROOT / "ai"
errors: list[str] = []

REQUIRED = [
    ROOT / "main.py",
    ROOT / "requirements.txt",
    AI / "shorts_pipeline.py",
    AI / "vod_processor.py",
    AI / "editor" / "captions.py",
    AI / "editor" / "caption_renderer.py",
    AI / "editor" / "speaker_caption_support.py",
    AI / "editor" / "speaker_naming.py",
    AI / "editor" / "teaser_analyzer.py",
    AI / "editor" / "intro_analyzer.py",
    AI / "editor" / "intro_renderer.py",
]
FORBIDDEN_DIR_NAMES = {"__pycache__", "mimir_*_patch", "cache"}
FORBIDDEN_SYMBOLS = {
    "_calibrate_caption_clock",
    "_snap_caption_runs_out_of_silence",
    "_assign_words_identity_locked",
    "_hybrid_segment_word_clock",
    "transcribe_caption_segment_timing",
}

for path in REQUIRED:
    if not path.is_file():
        errors.append(f"missing: {path.relative_to(ROOT)}")

if not compileall.compile_dir(str(AI), quiet=1):
    errors.append("compileall failed")

for path in AI.rglob("*.py"):
    try:
        # utf-8-sig: Windows editors may add a BOM; the interpreter accepts it,
        # ast.parse(str) does not.
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    except Exception as error:
        errors.append(f"AST parse failed {path.relative_to(ROOT)}: {error}")
        continue
    defs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    bad = sorted(defs & FORBIDDEN_SYMBOLS)
    if bad:
        errors.append(f"legacy calibration symbol(s) in {path.relative_to(ROOT)}: {', '.join(bad)}")

# Import checks: OpenAI client is lazy, so no API call/key is required.
sys.path.insert(0, str(ROOT))
try:
    from ai import vod_processor
    from ai.editor import captions, intro_renderer, speaker_caption_support, speaker_naming, teaser_analyzer
except Exception as error:
    errors.append(f"core import failed: {type(error).__name__}: {error}")
else:
    sig = inspect.signature(intro_renderer.run_renderer)
    for keyword in ("edited_clip_path", "captioned_preview_path"):
        if keyword not in sig.parameters:
            errors.append(f"intro_renderer.run_renderer missing keyword: {keyword}")

    if getattr(speaker_caption_support, "SPEAKER_PROFILE_VERSION", 0) < 25:
        errors.append("speaker profile cache version is stale")

    # Caption clock invariants.
    def timing(words: list[str], starts: list[float], ends: list[float]) -> dict:
        return {
            "words": [
                {"word": word, "start": start, "end": end}
                for word, start, end in zip(words, starts, ends)
            ],
            "text": " ".join(words),
        }

    try:
        data = timing(["You", "right"], [1.0, 1.45], [1.25, 1.75])
        rows, _ = vod_processor.align_text_to_fixed_whisper_clock("You are right", data, 3.0)
        right = next(row for row in rows if row["word"] == "right")
        if abs(float(right["start"]) - 1.45) > 1e-9:
            errors.append("later Whisper anchor moved during GPT-only insertion")
        if " ".join(row["word"] for row in rows).split() != ["You", "are", "right"]:
            errors.append("GPT wording was not fully preserved")
        for previous, current in zip(rows, rows[1:]):
            if float(previous["end"]) > float(current["start"]) + 0.011:
                errors.append("caption word clock overlap/monotonicity violation")
                break
    except Exception as error:
        errors.append(f"caption clock regression test failed: {error}")

    # V4 dense-clock regression: a previous word whose old 25 ms minimum would
    # cross a later immutable onset must be shortened, not crash the profile.
    try:
        dense = timing(["A", "B", "C"], [1.000, 1.012, 1.060], [1.020, 1.050, 1.100])
        dense_rows, _ = vod_processor.align_text_to_fixed_whisper_clock("A B C", dense, 2.0)
        if [row["word"] for row in dense_rows] != ["A", "B", "C"]:
            errors.append("dense clock wording changed")
        if abs(float(dense_rows[1]["start"]) - 1.012) > 1e-9:
            errors.append("dense clock moved the later Whisper onset")
        for previous, current in zip(dense_rows, dense_rows[1:]):
            if float(previous["end"]) > float(current["start"]) + 1e-9:
                errors.append("dense clock still overlaps after previous-tail trim")
                break
    except Exception as error:
        errors.append(f"dense caption clock regression failed: {error}")

    # V4 human identity is sticky DISPLAY metadata. A manual voice-confirmed
    # raw-id mapping cannot be erased merely because an optional model anchor
    # is unavailable or inconclusive.
    try:
        scan = {
            "participant_speakers": ["A", "B"],
            "display_labels": {"A": "KAI", "B": "TYLA"},
            "speaker_names": {"A": "KAI", "B": "TYLA", "C": "", "source": "manual_voice_calibrated_v28"},
            "identity_calibration": {
                "A": {"raw_speaker": "A", "name": "KAI", "human_verified": True},
                "B": {"raw_speaker": "B", "name": "TYLA", "human_verified": True},
            },
        }
        manual = speaker_caption_support._manual_identity_map(scan)
        sticky = speaker_caption_support._human_verified_identity_map(scan, manual)
        if sticky != {"A": "KAI", "B": "TYLA"}:
            errors.append(f"human-confirmed identity map was not preserved: {sticky}")
    except Exception as error:
        errors.append(f"human identity persistence regression failed: {error}")

    # V4 caption version must distinguish the surgical profile/ASS contract.
    if getattr(captions, "CAPTION_VERSION", 0) < 16:
        errors.append("caption output cache version is stale")

    # V7 verified participant-name orthography: a homophonic/near-homophonic
    # spelling may be normalized only in a conservative direct-address vocative.
    try:
        fixed, meta = speaker_caption_support._apply_verified_vocative_name_orthography(
            "My only option is um, Tyler, would you like to go on a date with me?",
            ["KAI", "TYLA"],
        )
        if "Tyla," not in fixed or "Tyler," in fixed:
            errors.append(f"V7 vocative verified-name spelling lock failed: {fixed}")
        if meta.get("status") != "corrected":
            errors.append("V7 vocative verified-name correction audit missing")

        untouched, untouched_meta = speaker_caption_support._apply_verified_vocative_name_orthography(
            "I watched Tyler play yesterday.",
            ["KAI", "TYLA"],
        )
        if untouched != "I watched Tyler play yesterday.":
            errors.append("V7 verified-name spelling lock rewrote third-party mention")
        if untouched_meta.get("status") == "corrected":
            errors.append("V7 third-party mention incorrectly audited as corrected")

        no_second_person, _ = speaker_caption_support._apply_verified_vocative_name_orthography(
            "Tyler, come here.",
            ["KAI", "TYLA"],
        )
        if no_second_person != "Tyler, come here.":
            errors.append("V7 verified-name spelling lock bypassed conservative second-person gate")
    except Exception as error:
        errors.append(f"V7 verified-name orthography regression failed: {error}")

    # No phrase-specific spelling rules: wording changes only through the strict
    # independent acoustic micro-vote, never through a per-word alias table.
    if hasattr(speaker_caption_support, "_streamer_slang_evidence_resolution"):
        errors.append("phrase-specific caption spelling resolver is back")

    # Exact-PCM clock guard: a proven late isolated phrase may move backward,
    # while a correctly timed earlier anchor remains byte-for-byte untouched.
    try:
        with tempfile.TemporaryDirectory(prefix="mimir_guard_verify_") as temp_dir:
            wav_path = Path(temp_dir) / "guard.wav"
            sample_rate = 48000
            duration_seconds = 4.0
            samples: list[int] = []
            for index in range(int(sample_rate * duration_seconds)):
                t = index / sample_rate
                value = 80.0 * math.sin(2.0 * math.pi * 120.0 * t)
                # Actual isolated speech-like burst is around 2.00-2.40s.
                if 2.00 <= t < 2.10:
                    value += 2200.0 * math.sin(2.0 * math.pi * 4200.0 * t)
                elif 2.10 <= t < 2.40:
                    value += 9000.0 * math.sin(2.0 * math.pi * 220.0 * t)
                samples.append(max(-32768, min(32767, int(value))))
            import array as _array
            payload = _array.array("h", samples)
            if sys.byteorder != "little":
                payload.byteswap()
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate); wav.writeframes(payload.tobytes())
            base_words = [
                {"word": "How", "edited_start": 1.00, "edited_end": 1.20, "timing_source": "whisper_exact_anchor"},
                {"word": "Sixteen.", "edited_start": 2.80, "edited_end": 3.20, "timing_source": "whisper_exact_anchor"},
            ]
            guarded, guard_meta = speaker_caption_support._apply_local_acoustic_clock_guard(
                audio_path=wav_path, words=base_words, duration=duration_seconds
            )
            if int(guard_meta.get("corrected_groups", 0) or 0) != 1:
                errors.append("exact-PCM clock guard did not correct proven late phrase")
            if abs(float(guarded[0]["edited_start"]) - 1.00) > 1e-9:
                errors.append("clock guard mutated a clean earlier anchor")
            if not (1.90 <= float(guarded[1]["edited_start"]) <= 2.12):
                errors.append(f"clock guard onset unexpected: {guarded[1]['edited_start']}")
            if "acoustic_guard" not in str(guarded[1].get("timing_source", "")):
                errors.append("clock guard correction provenance missing")
    except Exception as error:
        errors.append(f"local acoustic clock guard regression test failed: {error}")

    # Final exact-48k profile must be the renderer's word-clock source, not fallback.
    try:
        profile_rows = captions._profile_edited_words({
            "status": "ok",
            "timing_basis": "exact_final_48k_audio",
            "words": [{
                "word": "Amy.", "edited_start": 4.95, "edited_end": 5.03,
                "speaker_raw": "A", "speaker_role": "main", "speaker_label": "",
            }],
        }, 10.0)
        if len(profile_rows) != 1 or abs(float(profile_rows[0]["edited_start"]) - 4.95) > 1e-9:
            errors.append("exact_final_48k_audio profile was rejected by caption renderer input")
    except Exception as error:
        errors.append(f"exact final profile contract test failed: {error}")

    # Display readability is separate from the acoustic clock.
    try:
        amy = [{"word": "Amy.", "edited_start": 4.95, "edited_end": 5.03}]
        display_end = captions.get_event_end(
            amy, 0, 10.0, next_group_start=5.47
        )
        if abs(float(amy[0]["edited_start"]) - 4.95) > 1e-9 or abs(float(amy[0]["edited_end"]) - 5.03) > 1e-9:
            errors.append("caption display hold mutated acoustic word timestamps")
        if display_end - 4.95 < 0.319:
            errors.append("terminal single-word caption can still flash below readability floor")
        tight_end = captions.get_event_end(amy, 0, 10.0, next_group_start=5.10)
        if tight_end > 5.100001:
            errors.append("readability hold overlapped/delayed the following caption")
    except Exception as error:
        errors.append(f"caption display regression test failed: {error}")

    # Count confidence must not hide physically broken A/B boundaries.
    try:
        kai16_like = [
            {"start": 0.0, "end": 1.05, "duration": 1.05, "speaker": "A", "word_count": 8},
            {"start": 1.3, "end": 3.4, "duration": 2.1, "speaker": "A", "word_count": 6},
            {"start": 3.4, "end": 3.45, "duration": 0.05, "speaker": "B", "word_count": 1},
            {"start": 3.45, "end": 3.8, "duration": 0.35, "speaker": "A", "word_count": 1},
            {"start": 3.8, "end": 4.3, "duration": 0.5, "speaker": "B", "word_count": 3},
        ]
        boundary = speaker_caption_support._speaker_boundary_audit({"segments": kai16_like})
        if not boundary.get("suspicious") or not boundary.get("hard_failure"):
            errors.append("kai16-like A/B ping-pong did not trigger boundary retry gate")
        clean = [
            {"start": 0.0, "end": 1.2, "duration": 1.2, "speaker": "A", "word_count": 4},
            {"start": 1.35, "end": 2.5, "duration": 1.15, "speaker": "B", "word_count": 4},
            {"start": 2.7, "end": 4.0, "duration": 1.3, "speaker": "A", "word_count": 5},
        ]
        clean_boundary = speaker_caption_support._speaker_boundary_audit({"segments": clean})
        if clean_boundary.get("suspicious"):
            errors.append("clean alternating speaker turns falsely trigger retry")
    except Exception as error:
        errors.append(f"speaker boundary QA regression test failed: {error}")

    # Missing real-name calibration must be able to preserve a trusted raw A/B structure.
    try:
        probe = {"mode": "dual", "primary_speaker": "A", "secondary_speaker": "B"}
        probe = speaker_naming._mark_unlabeled_speaker_structure(probe, "verify")
        if probe.get("force_plain_captions") is not False or probe.get("display_labels") != {}:
            errors.append("unlabeled dual structure incorrectly collapses to force_plain")
    except Exception as error:
        errors.append(f"unlabeled speaker structure regression test failed: {error}")

    # Short real speakers must still be available for HUMAN A/B preview even
    # when they cannot satisfy the 2s OpenAI known-speaker anchor contract.
    try:
        short_profile = {
            "segments": [
                {"start": 0.0, "end": 0.55, "duration": 0.55, "speaker": "B", "text": "Thank you", "word_count": 2},
                {"start": 1.0, "end": 1.55, "duration": 0.55, "speaker": "B", "text": "Amy", "word_count": 1},
                {"start": 2.0, "end": 2.60, "duration": 0.60, "speaker": "B", "text": "Sixteen", "word_count": 1},
            ],
        }
        candidates = speaker_naming._candidate_manual_preview_segments(short_profile, "B")
        ranges = speaker_naming._take_ranges(
            candidates,
            target=speaker_naming.MANUAL_PREVIEW_TARGET_SECONDS,
            minimum=speaker_naming.MANUAL_PREVIEW_MIN_SECONDS,
            maximum=speaker_naming.MANUAL_PREVIEW_MAX_SECONDS,
        )
        if not ranges:
            errors.append("short real speaker cannot produce human audio preview")
    except Exception as error:
        errors.append(f"short speaker preview regression test failed: {error}")

    # Gold V7 speaker assignment must not mutate wording/timestamps.
    try:
        words = [
            {"word": "You", "edited_start": 0.10, "edited_end": 0.30},
            {"word": "right", "edited_start": 0.31, "edited_end": 0.55},
            {"word": "okay", "edited_start": 0.75, "edited_end": 1.00},
            {"word": "bro", "edited_start": 1.01, "edited_end": 1.20},
        ]
        segments = [
            {"id": "a", "start": 0.05, "end": 0.60, "speaker": "S0", "text": "You right", "word_count": 2, "duration": 0.55},
            {"id": "b", "start": 0.70, "end": 1.25, "speaker": "S1", "text": "okay bro", "word_count": 2, "duration": 0.55},
        ]
        out, meta = speaker_caption_support._assign_words(
            words, segments, "dual", "S0", "S1", [], ["S0", "S1"]
        )
        if [row["word"] for row in out] != [row["word"] for row in words]:
            errors.append("speaker assignment mutated caption wording")
        if [(row["edited_start"], row["edited_end"]) for row in out] != [
            (row["edited_start"], row["edited_end"]) for row in words
        ]:
            errors.append("speaker assignment mutated caption timestamps")
        if float(meta.get("meaning_preservation_ratio", 0.0)) < 0.999:
            errors.append("Gold V7 meaning-preservation regression")
    except Exception as error:
        errors.append(f"speaker regression test failed: {error}")


    # V5 identity-gap regression: words far outside every diarization segment
    # must NEVER inherit the previous person's real name. A close local boundary
    # may still use continuity to avoid one-token jitter.
    try:
        hole_words = [
            {"word": "Kai", "edited_start": 0.10, "edited_end": 0.35},
            {"word": "Go", "edited_start": 2.70, "edited_end": 2.95},
            {"word": "ahead.", "edited_start": 2.95, "edited_end": 3.20},
        ]
        hole_segments = [
            {"id": "a", "start": 0.05, "end": 0.45, "speaker": "A", "text": "Kai", "word_count": 1, "duration": 0.40},
        ]
        hole_out, hole_meta = speaker_caption_support._assign_words(
            hole_words, hole_segments, "dual", "A", "B", [], ["A", "B"]
        )
        if str(hole_out[1].get("speaker_raw", "")) or str(hole_out[2].get("speaker_raw", "")):
            errors.append("long diarization hole still inherits previous raw speaker")
        if float(hole_out[1].get("speaker_confidence", 1.0)) != 0.0:
            errors.append("long diarization hole still carries fake speaker confidence")

        local_words = [
            {"word": "hello", "edited_start": 0.10, "edited_end": 0.30},
            {"word": "there", "edited_start": 0.34, "edited_end": 0.48},
            {"word": "bro", "edited_start": 0.52, "edited_end": 0.70},
            {"word": "okay", "edited_start": 0.90, "edited_end": 1.15},
            {"word": "now", "edited_start": 1.16, "edited_end": 1.32},
        ]
        local_segments = [
            {"id": "a1", "start": 0.05, "end": 0.31, "speaker": "A", "text": "hello", "word_count": 1, "duration": 0.26},
            {"id": "a2", "start": 0.50, "end": 0.72, "speaker": "A", "text": "bro", "word_count": 1, "duration": 0.22},
            {"id": "b1", "start": 0.86, "end": 1.35, "speaker": "B", "text": "okay now", "word_count": 2, "duration": 0.49},
        ]
        local_out, _ = speaker_caption_support._assign_words(
            local_words, local_segments, "dual", "A", "B", [], ["A", "B"]
        )
        if str(local_out[1].get("speaker_raw", "")) != "A":
            errors.append("local continuity bridge stopped working")
    except Exception as error:
        errors.append(f"speaker identity-gap regression failed: {error}")


    # V6 known-voice ownership: human-confirmed known-speaker segments may fix
    # WHO metadata but never wording/timing. One phrase must never render as two
    # different named people; ambiguous direct evidence fails closed.
    try:
        identity_words = [
            {"word": "Don't", "edited_start": 1.00, "edited_end": 1.18,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.72, "speaker_evidence_source": "timing_only"},
            {"word": "do", "edited_start": 1.18, "edited_end": 1.34,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.72, "speaker_evidence_source": "timing_only"},
            {"word": "that.", "edited_start": 1.34, "edited_end": 1.62,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.72, "speaker_evidence_source": "timing_only"},
        ]
        identity_segments = [[
            {"id": "t", "start": 0.95, "end": 1.70, "speaker": "TYLA",
             "text": "Don't do that.", "word_count": 3, "duration": 0.75},
        ]]
        over, over_meta = speaker_caption_support._apply_human_known_voice_overlay(
            words=identity_words,
            segment_sets=identity_segments,
            raw_to_name={"A": "KAI", "B": "TYLA"},
        )
        locked, lock_meta = speaker_caption_support._lock_human_identity_by_phrase(
            words=over,
            raw_to_name={"A": "KAI", "B": "TYLA"},
        )
        if any(str(row.get("speaker_raw", "")) != "B" for row in locked):
            errors.append("V6 known-voice phrase lock failed to remap a proven TYLA phrase")
        if [row["word"] for row in locked] != [row["word"] for row in identity_words]:
            errors.append("V6 identity layer mutated caption wording")
        if [(row["edited_start"], row["edited_end"]) for row in locked] != [
            (row["edited_start"], row["edited_end"]) for row in identity_words
        ]:
            errors.append("V6 identity layer mutated caption timing")
        labels = {str(row.get("speaker_label", "")) for row in locked if str(row.get("speaker_label", ""))}
        if len(labels) > 1:
            errors.append("V6 one spoken phrase still mixes human identity labels")

        # Final invariant is stricter: a partially unresolved or mixed phrase
        # must become entirely UNKNOWN rather than changing visible name mid-line.
        mixed_phrase = [
            {"word": "Yeah,", "edited_start": 2.0, "edited_end": 2.2, "speaker_raw": "B", "speaker_label": "TYLA", "speaker_confidence": 0.9},
            {"word": "we", "edited_start": 2.2, "edited_end": 2.4, "speaker_raw": "A", "speaker_label": "KAI", "speaker_confidence": 0.8},
            {"word": "friends.", "edited_start": 2.4, "edited_end": 2.7, "speaker_raw": "A", "speaker_label": "KAI", "speaker_confidence": 0.8},
        ]
        guarded, guard_meta = speaker_caption_support._enforce_one_identity_per_phrase(
            words=mixed_phrase, raw_to_name={"A": "KAI", "B": "TYLA"}
        )
        if any(str(row.get("speaker_raw", "")) or str(row.get("speaker_label", "")) for row in guarded):
            errors.append("V6 final phrase guard did not fail closed on mixed KAI/TYLA phrase")
        if int(guard_meta.get("phrases_cleared", 0) or 0) != 1:
            errors.append("V6 final phrase guard audit did not record mixed phrase")
    except Exception as error:
        errors.append(f"V6 known-voice phrase lock regression failed: {error}")

    # V6 short human reference: a 1.25s reference + 0.75s human-verified
    # held-out utterance must stitch to a valid 2s known-speaker sample instead
    # of falling back to generic raw A/B diarization ranges.
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            source = td_path / "identity.wav"
            sample_rate = 48000
            frames = b"\x00\x00" * (sample_rate * 4)
            with wave.open(str(source), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate); wav.writeframes(frames)
            stitched = speaker_caption_support._render_confirmed_identity_reference(
                source,
                [(0.00, 1.25), (2.00, 2.75)],
                clip_index=99,
                raw_speaker="B",
            )
            if stitched is None:
                errors.append("V6 failed to stitch short human-confirmed identity reference")
            else:
                duration = speaker_caption_support._probe_duration(stitched)
                if not (1.95 <= duration <= 2.10):
                    errors.append(f"V6 stitched identity reference duration unexpected: {duration}")
            checkable = speaker_caption_support._disjoint_holdout_checkable_names(
                {"identity_calibration": {
                    "B": {"human_verified": True, "raw_speaker": "B", "name": "TYLA",
                          "known_speaker_reference_eligible": False,
                          "verification_ranges": [[2.0, 2.75]]}
                }},
                {"B": "TYLA"},
            )
            if "TYLA" in checkable:
                errors.append("V6 stitched reference incorrectly reused consumed verification audio as held-out evidence")
    except Exception as error:
        errors.append(f"V6 human reference stitch regression failed: {error}")

    # V6 risky phrase micro-consensus is local and fail-closed. Mock the API so
    # verification remains offline: two TYLA votes must override a stale KAI
    # phrase; disagreement must clear the label. Text/timing are immutable.
    try:
        base = [
            {"word": "All", "edited_start": 5.00, "edited_end": 5.20,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "KAI"},
            {"word": "right,", "edited_start": 5.20, "edited_end": 5.45,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "KAI"},
            {"word": "next", "edited_start": 5.45, "edited_end": 5.70,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "KAI"},
            {"word": "one.", "edited_start": 5.70, "edited_end": 6.00,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "KAI"},
        ]
        # Prepend an earlier B turn so gap-only verification is active after a
        # real speaker alternation.
        micro_words = [
            {"word": "Yeah.", "edited_start": 0.0, "edited_end": 0.4,
             "speaker_raw": "B", "speaker_role": "secondary", "speaker_label": "TYLA",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "TYLA"},
            {"word": "Okay.", "edited_start": 0.5, "edited_end": 0.9,
             "speaker_raw": "A", "speaker_role": "main", "speaker_label": "KAI",
             "speaker_confidence": 0.97, "speaker_evidence_source": "human_known_voice_phrase_lock",
             "identity_anchor_name": "KAI"},
            *base,
        ]
        original_bundle = speaker_caption_support._identity_reference_bundle
        original_render = speaker_caption_support._render_identity_micro_window
        original_run = speaker_caption_support._run_diarization
        try:
            speaker_caption_support._identity_reference_bundle = lambda **kwargs: (
                ["KAI", "TYLA"], ["ref-a", "ref-b"], ["a.wav", "b.wav"]
            )
            speaker_caption_support._render_identity_micro_window = lambda *args, **kwargs: Path("mock.wav")
            calls = {"n": 0}
            def fake_diarize(*args, **kwargs):
                calls["n"] += 1
                return {"segments": [
                    {"id": "m", "start": 0.0, "end": 9.0, "speaker": "TYLA", "text": "All right next one"}
                ]}
            speaker_caption_support._run_diarization = fake_diarize
            verified, audit = speaker_caption_support._micro_verify_risky_identity_phrases(
                audio_path=Path("mock.wav"), duration=8.0, scan={}, scan_segments=[],
                words=micro_words, raw_to_name={"A": "KAI", "B": "TYLA"}, clip_index=1,
            )
            tail = verified[-4:]
            if any(str(row.get("speaker_raw", "")) != "B" for row in tail):
                errors.append("V6 two-view micro identity consensus failed to override stale KAI phrase")
            if int(audit.get("api_calls_attempted", 0) or 0) < 2:
                errors.append("V6 risky identity phrase was not checked by two local views")
            if [row["word"] for row in verified] != [row["word"] for row in micro_words]:
                errors.append("V6 micro identity verifier mutated caption wording")
            if [(row["edited_start"], row["edited_end"]) for row in verified] != [
                (row["edited_start"], row["edited_end"]) for row in micro_words
            ]:
                errors.append("V6 micro identity verifier mutated caption timing")
        finally:
            speaker_caption_support._identity_reference_bundle = original_bundle
            speaker_caption_support._render_identity_micro_window = original_render
            speaker_caption_support._run_diarization = original_run
    except Exception as error:
        errors.append(f"V6 risky phrase micro identity regression failed: {error}")

    # Exact-final timing failure must not silently publish legacy-timed ASS.
    try:
        pipeline_source = (AI / "shorts_pipeline.py").read_text(encoding="utf-8")
        if "legacy timing fallback disabled" not in pipeline_source:
            errors.append("mandatory final-clock failure can still silently fall back")
    except Exception as error:
        errors.append(f"final-clock fail-closed regression failed: {error}")

    # Intro policy contract: the cold-open length comes from the EVENT (onset,
    # decay, phrases, shots), never from a score bucket or a fixed duration.
    try:
        from ai.editor import intro_bounds

        def env(loud):
            levels = [-50.0] * 1000
            for start, end, level in loud:
                for index in range(int(start / 0.02), int(end / 0.02)):
                    levels[index] = level
            return intro_bounds.Envelope(tuple(levels))

        tiny = intro_bounds.compute_intro_bounds(intro_bounds.EventEvidence(10.0, 10.3), clip_duration=20.0,
                                                 envelope=env([(10.0, 10.35, -8.0)]))
        reaction = intro_bounds.compute_intro_bounds(intro_bounds.EventEvidence(10.0, 10.3), clip_duration=20.0,
                                                     envelope=env([(10.0, 10.4, -8.0), (10.5, 12.6, -12.0)]))
        if not (tiny.duration < reaction.duration and reaction.end >= 12.6):
            errors.append(f"intro bounds ignore the event shape: tiny={tiny.duration}, reaction={reaction.to_dict()}")
        if "_adaptive_teaser_minimum" in dir(teaser_analyzer):
            errors.append("score-bucket intro minimum is back")
        schema = teaser_analyzer.TEASER_SCHEMA
        if sorted(schema["required"]) != sorted(schema["properties"]):
            errors.append("teaser strict schema: every property must be required")
    except Exception as error:
        errors.append(f"intro policy test failed: {error}")

# Single production path: Pro Edit presentation + caption truth + rendered-MP4
# QC + final gate always run. Unit / caption / render / QC suites run here;
# the multi-minute end-to-end run_pipeline suites run only with MIMIR_VERIFY_E2E=1.
pro_edit_summary = "not run"
try:
    import contextlib
    import io
    import os
    import unittest

    from ai import shorts_pipeline
    from ai.editor import pro_edit

    parameters = inspect.signature(shorts_pipeline.run_pipeline).parameters
    for keyword in ("rerender", "force_v6"):
        if keyword not in parameters:
            errors.append(f"run_pipeline has no {keyword} switch")
    for legacy in ("enable_pro_edit", "enable_v6"):
        if legacy in parameters:
            errors.append(f"run_pipeline still exposes the removed {legacy} switch (second path)")
    package, config = shorts_pipeline._load_pro_edit()
    if not (config.enabled and config.v6):
        errors.append("production path does not run the verified presentation layer")
    for module in pro_edit.MODULES:
        if not getattr(module, "__file__", None):
            errors.append(f"Pro Edit module not importable: {module}")

    tests_dir = ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    suites = ["test_pro_edit_timeline", "test_pro_edit_validator", "test_pro_edit_presets", "test_pro_edit_protection",
              "test_pro_edit_planner", "test_pro_edit_filters", "test_pro_edit_captions", "test_pro_edit_captions_v4",
              "test_pro_edit_v5", "test_pro_edit_render", "test_pro_edit_current_root",
              "test_v6_caption_truth", "test_v6_camera", "test_intro_bounds", "test_final_qc",
              "test_production_contracts"]
    if os.environ.get("MIMIR_VERIFY_E2E", "").strip() == "1":
        suites.extend(["test_pro_edit_pipeline", "test_v6_pipeline"])
    stream = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()):
        suite = unittest.defaultTestLoader.loadTestsFromNames(suites)
        outcome = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    pro_edit_summary = (f"{outcome.testsRun} run, {len(outcome.failures)} failed, {len(outcome.errors)} errors, "
                        f"{len(outcome.skipped)} skipped")
    if not outcome.wasSuccessful():
        errors.append(f"test suites failed ({pro_edit_summary}):\n{stream.getvalue()[-4000:]}")
except Exception as error:
    errors.append(f"production path verification failed: {type(error).__name__}: {error}")

if shutil.which("ffmpeg") is None:
    errors.append("ffmpeg not found on PATH")
if shutil.which("ffprobe") is None:
    errors.append("ffprobe not found on PATH")

if errors:
    print("\nUNIFIED CLEAN VERIFY: FAIL")
    for error in errors:
        print(f" - {error}")
    raise SystemExit(1)

print("UNIFIED CLEAN VERIFY: PASS")
print(" - compile/import contracts: OK")
print(" - legacy calibration/identity-word engines: absent")
print(" - GPT wording / Whisper anchor invariants: OK")
print(" - local exact-PCM caption clock outlier guard: OK")
print(" - no phrase-specific caption spelling rules: OK")
print(" - caption display readability without clock mutation: OK")
print(" - speaker boundary QA / retry gate: OK")
print(" - Gold V7 speaker word/time preservation: OK")
print(" - V5 diarization-gap identity fail-closed: OK")
print(" - V6 human-known-voice phrase identity lock: OK")
print(" - V6 short human reference stitch (2s contract): OK")
print(" - V6 risky phrase two-view identity consensus: OK")
print(" - V6 one-phrase-one-identity final fail-closed guard: OK")
print(" - V7 verified direct-address participant-name orthography: OK")
print(" - intro exact-path + evidence-derived cold-open bounds: OK")
print(f" - single production path + rendered-MP4 QC contracts + tests: OK ({pro_edit_summary})")
