"""Speaker colours: a caption colour names a VOICE (acoustic turns), never a person.

Covers the colour assignment, the final speaker profile, both caption renderers
(baseline ASS and the Pro Edit presentation), genuine overlap (lanes stay layout
only) and caption-truth integrity (colours are frozen and gate-checked).
"""
from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import test_pro_edit_captions as base
from ai.editor import caption_truth, captions, speaker_caption_support as scs, v6_runtime
from ai.editor.pro_edit import caption_presentation as cp

A_ACTIVE = cp.PALETTES["main"].active
B_ACTIVE = cp.PALETTES["secondary"].active
NEUTRAL_ACTIVE = cp.SPEAKER_PALETTES["neutral"].active
C_ACTIVE = cp.SPEAKER_PALETTES["C"].active


def turn(voice: str, count: int, start: float, confidence: float = 0.9, step: float = 0.3) -> list[dict]:
    return [{"word": f"{voice.lower()}t{int(round(start * 10))}w{i}", "edited_start": round(start + step * i, 3),
             "edited_end": round(start + step * i + 0.25, 3), "speaker_raw": voice, "speaker_role": "main",
             "speaker_label": "", "speaker_confidence": confidence} for i in range(count)]


def colour(rows: list[dict], mode: str = "dual", participants=("S0", "S1"), hard_failure: bool = False):
    return scs._assign_speaker_colors(rows, mode=mode, participants=list(participants), hard_failure=hard_failure)


def colours(rows: list[dict]) -> list[str]:
    return [row["speaker_color"] for row in rows]


def coloured_profile(rows: list[dict], mode: str = "dual", participants=("S0", "S1"), **extra) -> dict:
    rows, record = colour(rows, mode, participants)
    data = base.profile(rows)
    data.update({"mode": mode, "participant_speakers": list(participants), "speaker_colors": record, **extra})
    return data


def aba() -> list[dict]:
    # S1 speaks first, so S1 is voice A even though its raw id sorts second.
    return turn("S1", 4, 0.0) + turn("S0", 5, 1.5) + turn("S1", 4, 3.2)


def dialogues(ass_text: str) -> list[str]:
    return [line for line in ass_text.splitlines() if line.startswith("Dialogue:")]


def overlap_profile() -> dict:
    """Real-profile shape of an interruption: the forced-aligned word clock stays
    monotonic, the diarization turns overlap (S1 starts while S0 still talks)."""
    rows = turn("S0", 6, 0.0) + turn("S1", 3, 1.8) + turn("S1", 4, 3.4) + turn("S0", 4, 5.0)
    segments = [{"speaker": "S0", "start": 0.0, "end": 1.8}, {"speaker": "S1", "start": 0.9, "end": 2.7},
                {"speaker": "S1", "start": 3.4, "end": 4.5}, {"speaker": "S0", "start": 5.0, "end": 6.2}]
    return coloured_profile(rows, primary_speaker="S0", secondary_speaker="S1", segments=segments)


def active_word_colours(lines: list[str]) -> list[tuple[str, str, str]]:
    """(style, active word, active colour) of every baseline caption event."""
    rows = []
    for line in lines:
        found = re.search(r"\\b1\\c(&H[0-9A-F]+)[^}]*\}([a-z0-9]+)\{", line)
        if found:
            rows.append((line.split(",")[3], found.group(2), found.group(1)))
    return rows


class ColourAssignmentTests(unittest.TestCase):
    def test_a_single_voice_gets_no_speaker_colours(self) -> None:
        rows, record = colour(turn("S0", 8, 0.0), mode="single", participants=("S0",))
        self.assertEqual(set(colours(rows)), {""})
        self.assertEqual((record["engaged"], record["reason"]), (False, "mode_single"))

    def test_a_voice_keeps_its_colour_when_it_returns(self) -> None:
        rows, record = colour(aba())
        self.assertEqual(colours(rows), ["A"] * 4 + ["B"] * 5 + ["A"] * 4)
        self.assertEqual(record["slots"], {"S1": "A", "S0": "B"})
        self.assertTrue(record["engaged"])

    def test_three_voices_get_three_colours(self) -> None:
        rows, record = colour(turn("S0", 4, 0.0) + turn("S1", 4, 1.5) + turn("S2", 4, 3.0) + turn("S0", 3, 4.5),
                              mode="triple", participants=("S0", "S1", "S2"))
        self.assertEqual(colours(rows), ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["A"] * 3)
        self.assertEqual(len(set(record["slots"].values())), 3)

    def test_a_word_without_speaker_evidence_is_neutral(self) -> None:
        hole = dict(turn("S0", 1, 1.3)[0], speaker_raw="", speaker_confidence=0.0)
        rows, _ = colour(turn("S1", 4, 0.0) + [hole] + turn("S0", 5, 1.7) + turn("S1", 4, 3.4))
        self.assertEqual(colours(rows), ["A"] * 4 + ["neutral"] + ["B"] * 5 + ["A"] * 4)

    def test_a_weak_one_word_turn_never_flashes_a_colour(self) -> None:
        blip = turn("S1", 1, 1.6, confidence=0.7)
        rows, record = colour(turn("S0", 5, 0.0) + blip + turn("S0", 5, 2.0) + turn("S1", 5, 4.0))
        self.assertEqual(colours(rows), ["A"] * 5 + ["neutral"] + ["A"] * 5 + ["B"] * 5)
        self.assertEqual(record["demoted_turns"], 1)
        # A demoted opening blip does not claim colour A for its voice.
        rows, record = colour(turn("S1", 1, 0.0, confidence=0.6) + turn("S0", 6, 0.5) + turn("S1", 6, 2.5))
        self.assertEqual(record["slots"], {"S0": "A", "S1": "B"})
        self.assertEqual(colours(rows)[0], "neutral")

    def test_a_short_turn_with_strong_evidence_keeps_its_colour(self) -> None:
        interjection = turn("S1", 1, 1.6, confidence=0.95)
        rows, _ = colour(turn("S0", 5, 0.0) + interjection + turn("S0", 5, 2.0) + turn("S1", 5, 4.0))
        self.assertEqual(colours(rows)[5], "B")

    def test_unreliable_speaker_structure_gets_no_speaker_colours(self) -> None:
        cases = {
            "speaker_boundary_hard_failure": (aba(), "dual", True),
            "mode_unresolved": (aba(), "unresolved", False),
            "fewer_than_two_reliable_voices": (turn("S0", 6, 0.0) + turn("S1", 1, 2.0, confidence=0.5), "dual", False),
            "too_much_uncertain_speaker_evidence": (
                turn("S0", 4, 0.0) + [dict(r, speaker_raw="") for r in turn("S0", 5, 1.3)] + turn("S1", 4, 3.0),
                "dual", False),
            "speaker_color_flicker": (
                [row for k in range(8) for row in turn(("S0", "S1")[k % 2], 2, k * 0.7)], "dual", False),
        }
        for reason, (rows, mode, hard) in cases.items():
            with self.subTest(reason=reason):
                out, record = colour(rows, mode=mode, hard_failure=hard)
                self.assertEqual(set(colours(out)), {""})
                self.assertEqual((record["engaged"], record["reason"]), (False, reason))

    def test_colours_never_touch_words_times_or_speakers(self) -> None:
        rows = aba()
        before = json.dumps(rows, sort_keys=True)
        out, _ = colour(rows)
        self.assertEqual(json.dumps(rows, sort_keys=True), before)             # input untouched
        self.assertEqual([{k: v for k, v in r.items() if k != "speaker_color"} for r in out], rows)


class SpeakerProfileTests(unittest.TestCase):
    """The REAL create_speaker_profile; only the caption stack, ffprobe and audio extraction are faked."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir colours ")
        self.root = Path(self._tmp.name)
        self.clip = self.root / "clip" / "clip_01.mp4"
        self.clip.parent.mkdir()
        self.clip.write_bytes(b"0")
        self.audio = self.root / "clip.wav"
        self.audio.write_bytes(b"0")
        self.stack_calls: list[dict] = []
        self.words = [{"word": w, "edited_start": s, "edited_end": e, "timing_source": "qwen3_forced_aligner"}
                      for w, s, e in (("did", 0.1, 0.3), ("you", 0.32, 0.5), ("see", 0.52, 0.7), ("that", 0.72, 0.95),
                                      ("no", 1.4, 1.6), ("way", 1.62, 1.85), ("bro", 1.87, 2.1),
                                      ("told", 2.6, 2.8), ("you", 2.82, 3.0), ("so", 3.02, 3.3))]
        patches = [
            mock.patch.object(scs, "SPEAKER_OUTPUT_DIR", self.root / "out"),
            mock.patch.object(scs, "_probe_duration", return_value=4.0),
            mock.patch.object(scs, "_extract_audio", return_value=self.audio),
            mock.patch.object(scs.final_captions, "transcribe_final_short", side_effect=self.stack),
            mock.patch.object(scs, "_run_diarization", side_effect=AssertionError("no extra diarization call")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self._tmp.cleanup)

    def stack(self, clip, duration, **kwargs):
        self.stack_calls.append(kwargs)
        return [dict(w) for w in self.words], 1.0, "qwen_omni", {"target_met": True}

    def scan(self, mode: str, segments: list[dict], participants: list[str]) -> Path:
        path = self.root / "scan.json"
        path.write_text(json.dumps({
            "status": "ok", "mode": mode, "classification_source": "speaker_ensemble_v11",
            "primary_speaker": participants[0] if participants else None,
            "secondary_speaker": participants[1] if len(participants) > 1 else None,
            "participant_speakers": participants, "kept_speakers": participants, "background_speakers": [],
            "segments": segments, "speaker_stats": [{"speaker": p} for p in participants] or [{"speaker": "S0"}],
            "speaker_boundary_audit": {"hard_failure": False},
        }), encoding="utf-8")
        return path

    def build(self, scan: Path) -> dict:
        out = scs.create_speaker_profile(self.clip, 1, speaker_scan_path=scan, verified_terms=["Creator"])
        return json.loads(Path(out).read_text(encoding="utf-8"))

    def test_turn_taking_voices_get_colours_and_no_names(self) -> None:
        segments = [
            {"id": "a", "start": 0.05, "end": 1.0, "speaker": "S1", "text": "did you see that", "word_count": 4,
             "duration": 0.95},
            {"id": "b", "start": 1.35, "end": 2.2, "speaker": "S0", "text": "no way bro", "word_count": 3,
             "duration": 0.85},
            {"id": "c", "start": 2.55, "end": 3.35, "speaker": "S1", "text": "told you so", "word_count": 3,
             "duration": 0.8},
        ]
        profile = self.build(self.scan("dual", segments, ["S1", "S0"]))
        self.assertEqual(profile["status"], "ok", profile.get("error"))
        self.assertEqual(colours(profile["words"]), ["A"] * 4 + ["B"] * 3 + ["A"] * 3)
        self.assertEqual(profile["speaker_colors"]["slots"], {"S1": "A", "S0": "B"})
        self.assertTrue(all(w["speaker_label"] == "" for w in profile["words"]))
        self.assertEqual(profile["display_labels"], {})
        for key in ("identity_anchor", "identity_overlay", "identity_phrase_lock", "identity_micro_verify",
                    "identity_final_guard", "validated_identity_map", "speaker_names"):
            self.assertNotIn(key, profile)
        # WHAT and WHEN are the caption stack's, byte for byte.
        self.assertEqual([(w["word"], w["edited_start"], w["edited_end"]) for w in profile["words"]],
                         [(w["word"], w["edited_start"], w["edited_end"]) for w in self.words])
        # No human names reach the ears; only the user-verified terms.
        self.assertEqual(self.stack_calls, [{"verified_terms": ["Creator"]}])
        self.assertEqual(caption_truth.speaker_issues(profile), [])

    def test_a_single_voice_profile_is_plain(self) -> None:
        segments = [{"id": "a", "start": 0.05, "end": 3.35, "speaker": "S0",
                     "text": " ".join(w["word"] for w in self.words), "word_count": 10, "duration": 3.3}]
        profile = self.build(self.scan("single", segments, ["S0"]))
        self.assertEqual(set(colours(profile["words"])), {""})
        self.assertFalse(profile["speaker_colors"]["engaged"])
        self.assertEqual(caption_truth.speaker_issues(profile), [])

    def test_a_failed_diarization_scan_degrades_to_plain_instead_of_crashing(self) -> None:
        with mock.patch.object(scs, "_speaker_ensemble_scan", side_effect=RuntimeError("diarization offline")), \
                mock.patch.object(scs, "SPEAKER_OUTPUT_DIR", self.root / "scan out"):
            path = scs.create_speaker_scan_from_audio(self.audio, 1)
        scan = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual((scan["status"], scan["mode"], scan["display_labels"]), ("ok", "single", {}))
        self.assertIn("diarization offline", scan["diarization_error"])


class BaselineRendererTests(unittest.TestCase):
    def ass(self, profile: dict) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            path = captions.create_ass_for_clip({}, base.timeline(), Path(tmp) / "c.ass", profile)
            return path.read_text(encoding="utf-8-sig")

    def test_turns_change_colour_on_one_lane_and_a_caption_never_mixes_voices(self) -> None:
        rows = aba()
        lines = dialogues(self.ass(coloured_profile(rows, primary_speaker="S1", secondary_speaker="S0")))
        self.assertTrue(lines and all(",ViralMain," in line for line in lines))
        voice = {r["word"]: r["speaker_raw"] for r in rows}
        expected = {"S1": captions.ACTIVE_TEXT_COLOR, "S0": captions.SECONDARY_ACTIVE_TEXT_COLOR}
        events = active_word_colours(lines)
        self.assertEqual(len(events), len(rows))                          # one event per spoken word
        for _style, word, active in events:                               # A, then B, then A again
            self.assertEqual(active[-6:], expected[voice[word]][-6:], word)
        for line in lines:                                                # one caption == one voice
            self.assertEqual(len({voice[w] for w in re.findall(r"s[01]t\d+w\d", line)}), 1, line)
        self.assertFalse(any(re.search(r"\w+:\{\\r", line) for line in lines))  # no speaker name prefix

    def test_an_uncertain_word_renders_neutral(self) -> None:
        hole = dict(turn("S1", 1, 1.3)[0], word="huh", speaker_raw="", speaker_confidence=0.0)
        rows = turn("S1", 4, 0.0) + [hole] + turn("S0", 5, 1.7) + turn("S1", 4, 3.4)
        lines = dialogues(self.ass(coloured_profile(rows)))
        huh = [line for line in lines if "huh{" in line and "\\b1" in line]
        self.assertTrue(huh and all(captions.NEUTRAL_ACTIVE_TEXT_COLOR[-6:] in line for line in huh))

    def test_plain_profiles_render_exactly_as_before(self) -> None:
        rows = turn("S0", 8, 0.0)
        legacy = base.profile([{k: v for k, v in r.items()} for r in rows])
        plain = coloured_profile(rows, mode="single", participants=("S0",))
        self.assertEqual(self.ass(plain), self.ass(legacy))

    def test_a_boundary_hard_failure_shows_no_speaker_colours(self) -> None:
        profile = coloured_profile(aba())
        broken = dict(profile, speaker_boundary_audit={"hard_failure": True})
        self.assertNotIn(captions.SECONDARY_ACTIVE_TEXT_COLOR[-6:], self.ass(broken))

    def test_genuine_overlap_moves_the_lane_but_colours_follow_the_voice(self) -> None:
        profile = overlap_profile()
        words = captions._profile_edited_words(profile, 10.0)
        prepared, windows = captions._prepare_adaptive_render_words(words, speaker_profile=profile,
                                                                   trusted_display_map={})
        self.assertTrue(windows)
        lanes = {(w["speaker_raw"], w["speaker_role"], w["speaker_color"]) for w in prepared}
        self.assertIn(("S1", "secondary", "B"), lanes)                     # overlapping B: lane 2, colour B
        self.assertIn(("S1", "main", "B"), lanes)                          # later B turn: main lane, still B
        self.assertNotIn(("S0", "secondary", "A"), lanes)
        events = active_word_colours(dialogues(self.ass(profile)))
        self.assertTrue(any(style == "ViralSecondary" for style, _w, _c in events))
        for style, word, active in events:
            want = captions.SECONDARY_ACTIVE_TEXT_COLOR if word.startswith("s1") else captions.ACTIVE_TEXT_COLOR
            self.assertEqual(active[-6:], want[-6:], (style, word))


class PresentationTests(unittest.TestCase):
    def build(self, profile: dict) -> cp.CaptionPresentation:
        return cp.build_presentation(profile=profile, clip_timeline=base.timeline(), plan=None, width=1080,
                                     height=1920, metrics=base.BUILTIN)

    def test_pages_carry_their_voice_colour_and_stay_on_the_main_lane(self) -> None:
        built = self.build(coloured_profile(aba(), primary_speaker="S1", secondary_speaker="S0"))
        self.assertEqual({p.lane for p in built.pages}, {"main"})
        self.assertEqual({p.speaker_color for p in built.pages if p.tokens[0].text.startswith("s1")}, {"A"})
        for page in built.pages:
            self.assertEqual(len({t.speaker_color for t in page.tokens}), 1)
            self.assertEqual({t.speaker_raw for t in page.tokens}, {"S1" if page.speaker_color == "A" else "S0"})
            active = cp.page_style(page, built.geometry, built.settings.palettes, built.settings.speakers)
            self.assertEqual(active.active_colour, A_ACTIVE if page.speaker_color == "A" else B_ACTIVE)
            self.assertEqual(active.ass_style, "MimirMain")
        self.assertIn(B_ACTIVE, built.ass_text)
        self.assertEqual(built.metrics()["speaker_color_pages"].keys(), {"A", "B"})

    def test_presentation_matches_the_renderer_policy_and_keeps_the_style_lines(self) -> None:
        profile = coloured_profile(aba())
        tokens = cp.presentation_tokens(profile, 10.0)
        cp.verify_token_parity(tokens, profile, 10.0)                    # raises on any lane/label/colour drift
        forged = (dataclasses_replace(tokens[0], speaker_color="B"),) + tokens[1:]
        with self.assertRaises(cp.CaptionPresentationError):
            cp.verify_token_parity(forged, profile, 10.0)
        plain = self.build(base.profile(turn("S0", 8, 0.0)))
        styles = lambda text: [line for line in text.splitlines() if line.startswith("Style:")]
        self.assertEqual(styles(self.build(profile).ass_text), styles(plain.ass_text))

    def test_neutral_and_third_voice_pages_have_their_own_colours(self) -> None:
        hole = dict(turn("S1", 1, 1.3)[0], word="huh", speaker_raw="", speaker_confidence=0.0)
        rows = turn("S0", 4, 0.0) + [hole] + turn("S1", 5, 1.7) + turn("S2", 4, 3.4) + turn("S0", 3, 4.8)
        built = self.build(coloured_profile(rows, mode="triple", participants=("S0", "S1", "S2")))
        by_colour = {p.speaker_color: p for p in built.pages}
        self.assertEqual(set(by_colour), {"A", "B", "C", "neutral"})
        style = lambda key: cp.page_style(by_colour[key], built.geometry, built.settings.palettes,
                                          built.settings.speakers)
        self.assertEqual(style("neutral").active_colour, NEUTRAL_ACTIVE)
        self.assertEqual(style("C").active_colour, C_ACTIVE)
        self.assertEqual(len({style(k).active_colour for k in by_colour}), 4)
        self.assertEqual({p.lane for p in built.pages}, {"main"})

    def test_overlap_page_on_lane_two_keeps_the_voice_colour(self) -> None:
        built = self.build(overlap_profile())
        second = [p for p in built.pages if p.lane == "secondary"]
        self.assertTrue(second and all(p.speaker_color == "B" for p in second))
        self.assertTrue(any(p.lane == "main" and p.speaker_color == "B" for p in built.pages))


def dataclasses_replace(token, **changes):
    import dataclasses

    return dataclasses.replace(token, **changes)


class CaptionTruthIntegrityTests(unittest.TestCase):
    def test_the_frozen_truth_covers_the_colours(self) -> None:
        profile = coloured_profile(aba())
        self.assertEqual([row[6] for row in caption_truth.truth_rows(profile)], colours(profile["words"]))
        tampered = json.loads(json.dumps(profile))
        tampered["words"][0]["speaker_color"] = "B"
        self.assertNotEqual(caption_truth.truth_signature(tampered), caption_truth.truth_signature(profile))
        with self.assertRaisesRegex(ValueError, "speaker_color"):
            caption_truth.verify_truth_invariants(profile["words"], tampered["words"])

    def test_the_gate_accepts_only_the_acoustic_colour_record(self) -> None:
        profile = coloured_profile(aba())
        self.assertEqual(caption_truth.speaker_issues(profile), [])
        swapped = json.loads(json.dumps(profile))
        swapped["words"][0]["speaker_color"] = "B"
        self.assertTrue(any("not the colour of voice" in i for i in caption_truth.speaker_issues(swapped)))
        off = json.loads(json.dumps(profile))
        off["speaker_colors"]["engaged"] = False
        self.assertTrue(any("without reliable speaker colours" in i for i in caption_truth.speaker_issues(off)))
        duplicate = json.loads(json.dumps(profile))
        duplicate["speaker_colors"]["slots"] = {"S1": "A", "S0": "A"}
        self.assertEqual(caption_truth.speaker_issues(duplicate),
                         ["speaker colour record gives two voices the same colour"])
        neutral = json.loads(json.dumps(profile))
        neutral["words"][2]["speaker_color"] = "neutral"                 # an uncertain word may always be neutral
        self.assertEqual(caption_truth.speaker_issues(neutral), [])

    def test_the_final_gate_row_passes_for_a_clean_coloured_profile(self) -> None:
        profile = coloured_profile(aba())
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            truth_path = Path(tmp) / "truth.json"
            truth_path.write_text(json.dumps({"kind": caption_truth.TRUTH_KIND,
                                              "signature": caption_truth.truth_signature(profile)}), encoding="utf-8")
            rows = {row["check"]: row["status"] for row in v6_runtime.check_caption_truth(profile_path, truth_path)}
        self.assertEqual(rows, {"caption_truth_frozen": "pass", "word_timing": "pass", "speaker_ownership": "pass"})


class ProductionPathTests(unittest.TestCase):
    def test_the_identity_naming_path_is_gone(self) -> None:
        self.assertIsNone(importlib.util.find_spec("ai.editor.speaker_naming"))
        for name in ("_run_human_identity_anchor", "_apply_human_known_voice_overlay", "_lock_human_identity_by_phrase",
                     "_micro_verify_risky_identity_phrases", "_enforce_one_identity_per_phrase",
                     "_manual_identity_map", "_identity_reference_bundle"):
            self.assertFalse(hasattr(scs, name), name)
        source = (Path(scs.__file__).resolve().parent.parent / "shorts_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("speaker_naming", source)
        self.assertNotIn("input(", source)

    def test_the_review_sheet_asks_the_human_to_check_the_colours(self) -> None:
        from ai.editor import human_review

        doc = {"intro": {"duration": 0.0}, "restart": {"main_restart_paced": 0.0}}

        def sheet(colour_column: list[str]) -> tuple[dict, str]:
            truth = {"words": [[i, f"w{i}", 1.0 + i, 1.2 + i, "S0", "", c] for i, c in enumerate(colour_column)]}
            packet = human_review.build_review_packet(
                published=Path("x_short.mp4"), source=Path("x.mp4"), status="published", qc_rows=[],
                timeline_doc=doc, truth=truth, headline="", degradations=[])
            return packet, human_review.render_markdown(packet)

        packet, text = sheet(["A", "A", "B", "neutral", "A"])
        self.assertEqual(packet["speaker_colors"], {"A": 3, "B": 1, "neutral": 1})
        self.assertIn("- [ ] Each voice keeps one caption colour", text)
        self.assertNotIn("human verified the voice", text)
        _packet, text = sheet(["", ""])
        self.assertIn("Speaker colours off", text)


if __name__ == "__main__":
    unittest.main()
