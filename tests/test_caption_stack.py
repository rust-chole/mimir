"""The final-caption stack: Qwen ears (WHAT), frozen transcript, one word aligner (WHEN).

Hermetic: the DashScope client, the OpenAI ears, the official aligner model and
whisper-1 are fakes with the real interfaces; FFmpeg produces real audio.
"""
from __future__ import annotations

import array
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from ai.caption_stack import alignment, config, final_captions, lexical, qwen_omni
from ai.caption_stack import audio as analysis_audio

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
ENV_KEYS = ("DASHSCOPE_API_KEY", "DASHSCOPE_BASE_URL", "MIMIR_CAPTION_PRIMARY_PROVIDER", "MIMIR_QWEN_OMNI_MODEL",
            "MIMIR_QWEN_REASONING_EFFORT", "MIMIR_WORD_ALIGNMENT_PROVIDER", "MIMIR_WORD_ALIGNMENT_FALLBACK_PROVIDER",
            "MIMIR_QWEN_ALIGNER_MODEL", "MIMIR_QWEN_ALIGNER_DEVICE", "MIMIR_QWEN_ALIGNER_DTYPE",
            "MIMIR_CAPTION_TRANSCRIBE_FALLBACK_PROVIDER", "MIMIR_QWEN_TIMEOUT_SECONDS", "MIMIR_QWEN_MAX_RETRIES")


def settings(**overrides) -> config.CaptionStackSettings:
    values = dict(primary_provider="qwen_omni", qwen_model="fake-qwen", qwen_reasoning_effort="none",
                  qwen_timeout_s=30.0, qwen_max_retries=1, dashscope_base_url="https://example.invalid/v1",
                  transcribe_fallback_provider="openai_transcribe", transcribe_fallback_model="fake-diverse",
                  alignment_provider="qwen3_forced_aligner", alignment_fallback_provider="legacy_whisper",
                  aligner_model="fake/aligner", aligner_device="auto", aligner_dtype="auto", language="en",
                  dashscope_api_key="sk-secret-value")
    values.update(overrides)
    return config.CaptionStackSettings(**values)


def write_wav(path: Path, seconds: float, bursts: list[tuple[float, float]], rate: int = 16000) -> Path:
    """Mono PCM16: quiet floor + loud 'speech' bursts (a controllable low-energy structure)."""
    samples = array.array("h")
    for index in range(int(seconds * rate)):
        t = index / rate
        loud = any(a <= t < b for a, b in bursts)
        samples.append(int((6000 if loud else 40) * math.sin(2 * math.pi * 220 * t)))
    if sys.byteorder != "little":
        samples.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(samples.tobytes())
    return path


class FakeAlignerModel:
    """The official ``Qwen3ForcedAligner.align`` shape: cleaned words, start/end seconds."""

    def __init__(self, start: float = 0.3, step: float = 0.4, broken: dict[str, tuple[float, float]] | None = None):
        self.start, self.step, self.broken, self.calls = start, step, dict(broken or {}), []

    def align(self, audio, text, language):
        self.calls.append((Path(audio).name, text, language))
        words = [w for w in (alignment.aligner_clean(t) for t in text.split()) if w]
        base = self.start
        with wave.open(audio, "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        items = []
        for index, word in enumerate(words):
            s, e = base + index * self.step, base + index * self.step + self.step * 0.75
            if word in self.broken and len(self.calls) == 1:
                s, e = self.broken[word]
            items.append(SimpleNamespace(text=word, start_time=round(min(s, duration), 3),
                                         end_time=round(min(e, duration), 3)))
        return [SimpleNamespace(items=items)]


def loaded(model) -> alignment.Loader:
    return lambda _settings: alignment.LoadedAligner(model, "cpu", "float32", 180.0, [])


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {k: "" for k in ENV_KEYS})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()

    def test_defaults_are_model_ids_only(self) -> None:
        loaded_settings = config.load_settings()
        self.assertEqual((loaded_settings.primary_provider, loaded_settings.qwen_model,
                          loaded_settings.qwen_reasoning_effort), ("qwen_omni", "qwen3.8-omni-flash", "none"))
        self.assertEqual((loaded_settings.alignment_provider, loaded_settings.aligner_model),
                         ("qwen3_forced_aligner", "Qwen/Qwen3-ForcedAligner-0.6B"))
        self.assertEqual((loaded_settings.aligner_device, loaded_settings.aligner_dtype), ("auto", "auto"))
        self.assertEqual(loaded_settings.dashscope_base_url, "")               # no region is ever assumed
        self.assertFalse(loaded_settings.qwen_configured)

    def test_secrets_never_reach_the_public_view(self) -> None:
        os.environ["DASHSCOPE_API_KEY"] = "sk-very-secret"
        os.environ["DASHSCOPE_BASE_URL"] = "https://example.invalid/compatible-mode/v1/"
        loaded_settings = config.load_settings()
        self.assertTrue(loaded_settings.qwen_configured)
        self.assertEqual(loaded_settings.dashscope_base_url, "https://example.invalid/compatible-mode/v1")
        self.assertNotIn("sk-very-secret", json.dumps(loaded_settings.public()))
        self.assertNotIn("sk-very-secret", repr(loaded_settings))

    def test_unsupported_values_fail_loudly(self) -> None:
        for key, value in (("MIMIR_CAPTION_PRIMARY_PROVIDER", "whisper"), ("MIMIR_QWEN_ALIGNER_DEVICE", "tpu"),
                           ("MIMIR_QWEN_ALIGNER_DTYPE", "int4"), ("MIMIR_QWEN_REASONING_EFFORT", "extreme")):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: value}):
                with self.assertRaises(config.CaptionStackConfigError):
                    config.load_settings()
        with mock.patch.dict(os.environ, {"MIMIR_QWEN_ALIGNER_DEVICE": "cuda:1"}):
            self.assertEqual(config.load_settings().aligner_device, "cuda:1")   # a user choice, not a default
        with mock.patch.dict(os.environ, {"MIMIR_WORD_ALIGNMENT_PROVIDER": "legacy_whisper",
                                          "MIMIR_WORD_ALIGNMENT_FALLBACK_PROVIDER": "legacy_whisper"}):
            self.assertEqual(config.load_settings().alignment_fallback_provider, "none")

    def test_device_and_dtype_never_assume_a_gpu(self) -> None:
        no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False),
                                  float32="f32", float16="f16", bfloat16="bf16")
        cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                               float32="f32", float16="f16", bfloat16="bf16")
        self.assertEqual(config.resolve_device("auto", no_cuda)[0], "cpu")
        device, note = config.resolve_device("cuda", no_cuda)
        self.assertEqual(device, "cpu")
        self.assertIn("unavailable", note)
        self.assertEqual(config.resolve_device("auto", cuda)[0], "cuda")
        self.assertEqual(config.resolve_dtype("auto", "cpu", cuda), ("f32", "float32"))
        self.assertEqual(config.resolve_dtype("auto", "cuda", cuda), ("bf16", "bfloat16"))
        self.assertEqual(config.aligner_language("en-US"), "English")
        with self.assertRaises(config.CaptionStackConfigError):
            config.aligner_language("xx")


class QwenEvidenceTests(unittest.TestCase):
    def evidence(self, **fields) -> str:
        base = {"language": "en", "text": "no way bro, that's cooked", "utterances": [{"text": "no way bro,"},
                {"text": "that's cooked"}], "uncertain_spans": [{"heard": "cooked", "alternatives": ["cook"]}]}
        base.update(fields)
        return json.dumps(base)

    def test_valid_evidence_is_parsed_and_validated(self) -> None:
        parsed = qwen_omni.parse_caption_evidence("```json\n" + self.evidence() + "\n```")
        self.assertEqual(parsed.text, "no way bro, that's cooked")
        self.assertEqual(parsed.uncertain_spans[0].alternatives, ("cook",))

    def test_json_that_parses_is_not_trusted_blindly(self) -> None:
        bad = {"not json": "okay here you go",
               "no text": self.evidence(text=None),
               "text not a string": self.evidence(text=["no", "way"]),
               "utterance shape": self.evidence(utterances=["no way"]),
               "utterances drop a word": self.evidence(utterances=[{"text": "no way bro"}]),
               "no lexical content": self.evidence(text="...", utterances=[]),
               "markdown in text": self.evidence(text='{"text": "no way"}', utterances=[]),
               "alternatives type": self.evidence(uncertain_spans=[{"heard": "cooked", "alternatives": "cook"}])}
        for label, raw in bad.items():
            with self.subTest(label=label), self.assertRaises(qwen_omni.QwenInvalidResponse):
                qwen_omni.parse_caption_evidence(raw)
        self.assertRaises(qwen_omni.QwenInvalidResponse, qwen_omni.parse_caption_evidence, "[1, 2]")

    def test_non_speech_annotations_and_foreign_spans_are_dropped(self) -> None:
        parsed = qwen_omni.parse_caption_evidence(self.evidence(
            text="[laughs] no way bro, that's cooked", utterances=[],
            uncertain_spans=[{"heard": "never said", "alternatives": []}]))
        self.assertEqual(parsed.text, "no way bro, that's cooked")
        self.assertEqual(parsed.uncertain_spans, ())
        self.assertTrue(any("not found in text" in note for note in parsed.notes))

    def stream(self, *parts: str, reasoning: str = ""):
        chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None,
                                                                                 reasoning_content=reasoning))])]
        chunks += [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=p))]) for p in parts]
        chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 7})))
        return iter(chunks)

    def test_request_settings_and_streamed_text(self) -> None:
        calls = []
        raw = self.evidence()

        def create(**kwargs):
            calls.append(kwargs)
            return self.stream(raw[:20], raw[20:], reasoning="thinking that must be ignored")

        evidence, meta = qwen_omni.listen("QUJD", instructions="rules", settings=settings(), create=create)
        self.assertEqual(evidence.text, "no way bro, that's cooked")
        kwargs = calls[0]
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["modalities"], ["text"])
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertTrue(kwargs["stream"])
        audio_part = kwargs["messages"][1]["content"][0]
        self.assertEqual(audio_part["type"], "input_audio")
        self.assertTrue(audio_part["input_audio"]["data"].endswith("QUJD"))
        self.assertEqual(meta["usage"], {"total_tokens": 7})

    def test_invalid_evidence_is_retried_a_bounded_number_of_times(self) -> None:
        answers = iter(["not json", self.evidence()])
        evidence, meta = qwen_omni.listen("QUJD", instructions="x", settings=settings(qwen_max_retries=1),
                                          create=lambda **k: self.stream(next(answers)))
        self.assertEqual(meta["attempts"], 2)
        with self.assertRaises(qwen_omni.QwenInvalidResponse):
            qwen_omni.listen("QUJD", instructions="x", settings=settings(qwen_max_retries=1),
                             create=lambda **k: self.stream("still not json"))

    def test_provider_errors_never_leak_the_key(self) -> None:
        def create(**kwargs):
            raise RuntimeError("401 invalid key sk-secret-value for request")
        with self.assertRaises(qwen_omni.QwenUnavailable) as caught:
            qwen_omni.listen("QUJD", instructions="x", settings=settings(), create=create)
        self.assertNotIn("sk-secret-value", str(caught.exception))
        with self.assertRaises(qwen_omni.QwenUnavailable):
            qwen_omni.listen("QUJD", instructions="x", settings=settings(dashscope_api_key=""))

    def test_the_precision_ear_treats_names_as_spelling_references_only(self) -> None:
        primary = qwen_omni.primary_instructions("en", ["chat"])
        precision = qwen_omni.precision_instructions("en", ["chat"], ["Tyla", "KAI"])
        self.assertNotIn("Tyla", primary)
        self.assertIn("SPELLING REFERENCES ONLY", precision)
        self.assertIn("do NOT prove", precision)
        for text in (primary, precision):
            self.assertIn("Never improve grammar", text)
            self.assertNotIn("timestamp", text.split("Return ONLY")[0].lower())


class ListeningFallbackTests(unittest.TestCase):
    def test_qwen_unavailable_uses_the_bounded_fallback_ear_and_says_so(self) -> None:
        heard = []

        def qwen(path, instructions, settings_):
            raise qwen_omni.QwenUnavailable("DASHSCOPE_API_KEY is not set")

        def openai(path, *, prompted, model, language, verified_terms):
            heard.append((prompted, tuple(verified_terms)))
            return "hey Tyla come here" if prompted else "hey Tyler come here"

        with tempfile.TemporaryDirectory() as tmp:
            audio = analysis_audio.load_analysis_audio(write_wav(Path(tmp) / "a.wav", 2.0, [(0.2, 1.5)]))
            passes, degradations = lexical.listen_passes(
                audio, [(0.0, audio.duration)], settings=settings(), verified_terms=["Tyla", "speaker a"],
                ears=lexical.Ears(qwen=qwen, openai=openai), workdir=Path(tmp))
            self.assertEqual(sorted(heard), [(False, ("Tyla",)), (True, ("Tyla",))])
            self.assertEqual([e.provider for e in passes[0]], ["openai_transcribe", "openai_transcribe"])
            self.assertEqual({d["subsystem"] for d in degradations}, {"caption_lexical_primary"})
            with self.assertRaises(lexical.LexicalUnavailable):
                lexical.listen_passes(audio, [(0.0, audio.duration)],
                                      settings=settings(transcribe_fallback_provider="none"), verified_terms=[],
                                      ears=lexical.Ears(qwen=qwen, openai=openai), workdir=Path(tmp))


class AlignmentValidationTests(unittest.TestCase):
    def units(self, *times: tuple[float, float]) -> list[alignment.AlignedUnit]:
        return [alignment.AlignedUnit(i, i + 1, a, b, "t") for i, (a, b) in enumerate(times)]

    def test_a_clean_clock_passes(self) -> None:
        check = alignment.validate_units(self.units((0.1, 0.3), (0.3, 0.6), (0.7, 0.9)), 3, (0.0, 2.0), 2.0)
        self.assertTrue(check.ok, check.issues)

    def test_every_rejection_rule(self) -> None:
        cases = {
            "zero duration": ((0.1, 0.3), (0.5, 0.5), (0.7, 0.9)),
            "negative duration": ((0.1, 0.3), (0.6, 0.5), (0.7, 0.9)),
            "start moves backward": ((0.1, 0.3), (0.05, 0.2), (0.7, 0.9)),
            "overlap": ((0.1, 0.5), (0.3, 0.6), (0.7, 0.9)),
            "outside the audio": ((0.1, 0.3), (0.4, 0.6), (1.9, 2.4)),
            "collapsed at start": ((0.0, 0.2), (0.0, 0.3), (0.4, 0.6)),
            "collapsed region": ((0.1, 0.11), (0.11, 0.12), (0.12, 0.13)),
            "implausible duration": ((0.1, 0.3), (0.3, 1.9), (1.95, 1.99)),
        }
        for label, times in cases.items():
            with self.subTest(label=label):
                duration = 2.0 if label != "implausible duration" else 2.0
                check = alignment.validate_units(self.units(*times), 3, (0.0, duration),
                                                 duration if label != "implausible duration" else 2.0)
                if label == "implausible duration":
                    with mock.patch.object(alignment, "MAX_WORD_SECONDS", 1.0):
                        check = alignment.validate_units(self.units(*times), 3, (0.0, 2.0), 2.0)
                self.assertFalse(check.ok, label)

    def test_coverage_and_order_are_fatal(self) -> None:
        gap = [alignment.AlignedUnit(0, 1, 0.1, 0.2, "t"), alignment.AlignedUnit(2, 3, 0.3, 0.4, "t")]
        self.assertTrue(alignment.validate_units(gap, 3, (0.0, 1.0), 1.0).fatal)
        short = [alignment.AlignedUnit(0, 1, 0.1, 0.2, "t")]
        self.assertTrue(alignment.validate_units(short, 2, (0.0, 1.0), 1.0).fatal)

    def test_official_items_map_to_frozen_tokens_by_exact_parity(self) -> None:
        tokens = ["Wait—", "don't", "12", "东京"]
        items = [SimpleNamespace(text=t, start_time=s, end_time=e) for t, s, e in
                 (("Wait", 0.1, 0.3), ("don't", 0.4, 0.6), ("12", 0.7, 0.9), ("东", 1.0, 1.1), ("京", 1.1, 1.3))]
        units = alignment.map_items_to_tokens(tokens, items, 10.0, "qwen")
        self.assertEqual([(u.token_start, u.start, u.end) for u in units],
                         [(0, 10.1, 10.3), (1, 10.4, 10.6), (2, 10.7, 10.9), (3, 11.0, 11.3)])
        swapped = items[:1] + [SimpleNamespace(text="dont", start_time=0.4, end_time=0.6)] + items[2:]
        with self.assertRaises(alignment.AlignmentError):
            alignment.map_items_to_tokens(tokens, swapped, 0.0, "qwen")             # a replaced word
        with self.assertRaises(alignment.AlignmentError):
            alignment.map_items_to_tokens(tokens, items[:-1], 0.0, "qwen")          # a lost word


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class AlignmentAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir align ş ")
        self.dir = Path(self._tmp.name)
        self.audio = analysis_audio.load_analysis_audio(write_wav(self.dir / "a.wav", 6.0, [(0.3, 5.0)]))
        alignment.reset_shared_aligners()

    def tearDown(self) -> None:
        alignment.reset_shared_aligners()
        self._tmp.cleanup()

    def frozen(self, text: str) -> lexical.FrozenTranscript:
        tokens = tuple(lexical.display_tokens(text))
        chunks = ((0.0, self.audio.duration),)
        return lexical.FrozenTranscript(tokens, ("agreed",) * len(tokens), (0,) * len(tokens), chunks, (), "en",
                                        lexical.lexical_signature(tokens, chunks))

    def test_a_small_failed_region_is_recovered_locally_and_nothing_else_moves(self) -> None:
        model = FakeAlignerModel(broken={"told": (1.1, 1.1)})                     # zero-duration on the first pass
        provider = alignment.QwenForcedAlignmentProvider(settings(), loaded(model))
        frozen = self.frozen("I literally told you not to do that")
        outcome = alignment.align_frozen(frozen, self.audio, primary=provider, fallback=None, language="en",
                                         workdir=self.dir)
        self.assertEqual(outcome.provider, "qwen3_forced_aligner")
        self.assertEqual([r["status"] for r in outcome.recoveries], ["recovered"])
        self.assertEqual(len(model.calls), 2)
        self.assertNotEqual(model.calls[1][0], "a.wav")                            # the local audio window only
        self.assertEqual(model.calls[1][1].split(), ["I", "literally", "told", "you", "not", "to", "do"])
        told = outcome.units[2]
        self.assertGreater(told.end, told.start)
        for index in (0, 1, 3, 4, 5, 6, 7):                                          # trusted words untouched
            self.assertAlmostEqual(outcome.units[index].start, 0.3 + index * 0.4, places=3)

    def test_an_unrecoverable_clock_goes_to_the_fallback_for_the_whole_transcript(self) -> None:
        class Broken(FakeAlignerModel):
            def align(self, audio, text, language):
                result = super().align(audio, text, language)
                for item in result[0].items:
                    object.__setattr__(item, "end_time", item.start_time)           # every word collapses
                return result

        whisper_calls = []

        def whisper(path, known_names=None):
            whisper_calls.append(Path(path).name)
            words = "I literally told you not to do that".split()
            return {"words": [{"word": w, "start": 0.4 + i * 0.5, "end": 0.8 + i * 0.5} for i, w in enumerate(words)]}

        primary, fallback = alignment.build_providers(settings(), qwen_loader=loaded(Broken()),
                                                      whisper_transcribe=whisper)
        outcome = alignment.align_frozen(self.frozen("I literally told you not to do that"), self.audio,
                                         primary=primary, fallback=fallback, language="en", workdir=self.dir)
        self.assertEqual(outcome.provider, "legacy_whisper")
        self.assertTrue(outcome.degraded)
        self.assertEqual(whisper_calls, ["a.wav"])
        self.assertEqual({u.source.split(":")[0] for u in outcome.units}, {"legacy_whisper"})   # ONE authority

    def test_no_valid_clock_fails_closed(self) -> None:
        provider = alignment.QwenForcedAlignmentProvider(settings(), lambda s: (_ for _ in ()).throw(
            ImportError("No module named 'qwen_asr'")))
        with self.assertRaises(alignment.AlignmentError):
            alignment.align_frozen(self.frozen("hello there"), self.audio, primary=provider, fallback=None,
                                   language="en", workdir=self.dir)

    def test_the_aligner_loads_once_and_a_failure_is_remembered(self) -> None:
        loads = []

        def loader(s):
            loads.append(s.aligner_model)
            return alignment.LoadedAligner(FakeAlignerModel(), "cpu", "float32", 180.0, [])

        provider = alignment.QwenForcedAlignmentProvider(settings(), loader)
        self.assertEqual(provider.describe()["device"], "not_loaded")                # nothing loaded up front
        for _ in range(3):
            provider.align(self.audio, (0.0, self.audio.duration), ["hello", "there"], "en", self.dir)
        self.assertEqual(loads, ["fake/aligner"])
        alignment.reset_shared_aligners()
        failing = []

        def bad_loader(s):
            failing.append(1)
            raise OSError("model download blocked")

        other = alignment.QwenForcedAlignmentProvider(settings(aligner_model="other"), bad_loader)
        for _ in range(2):
            with self.assertRaises(alignment.AlignmentError):
                other.align(self.audio, (0.0, 1.0), ["x"], "en", self.dir)
        self.assertEqual(failing, [1])

    def test_long_audio_is_chunked_at_pauses_with_exact_offsets(self) -> None:
        bursts = [(0.2, 3.8), (4.4, 7.6), (8.4, 11.8)]                               # pauses near 4.1 s and 8.0 s
        audio = analysis_audio.load_analysis_audio(write_wav(self.dir / "long.wav", 12.0, bursts))
        with mock.patch.object(analysis_audio, "CHUNK_SEARCH_S", 3.0):
            chunks = analysis_audio.plan_chunks(audio, max_seconds=5.0)
        self.assertEqual(chunks[0][0], 0.0)
        self.assertEqual(chunks[-1][1], audio.duration)
        for (a, b), (c, _d) in zip(chunks, chunks[1:]):
            self.assertEqual(b, c)                                                  # contiguous, no overlap
        self.assertTrue(all(b - a <= 5.0 + 1e-9 for a, b in chunks))
        self.assertTrue(3.8 <= chunks[0][1] <= 4.4, chunks)                         # the pause, not a fixed time
        self.assertEqual(analysis_audio.plan_chunks(audio, max_seconds=20.0), [(0.0, audio.duration)])
        tokens = ("one", "two", "three", "four", "five", "six")
        chunk_of = (0, 0, 1, 1, 2, 2)
        frozen = lexical.FrozenTranscript(tokens, ("agreed",) * 6, chunk_of, tuple(chunks), (), "en",
                                          lexical.lexical_signature(tokens, chunks))
        model = FakeAlignerModel(start=0.5, step=1.0)
        outcome = alignment.align_frozen(frozen, audio, primary=alignment.QwenForcedAlignmentProvider(
            settings(), loaded(model)), fallback=None, language="en", workdir=self.dir)
        self.assertEqual([(u.token_start, u.token_end) for u in outcome.units], [(i, i + 1) for i in range(6)])
        for (chunk_start, _end), unit in zip([chunks[c] for c in chunk_of[::2]], outcome.units[::2]):
            self.assertAlmostEqual(unit.start, round(chunk_start + 0.5, 3), places=3)
        starts = [u.start for u in outcome.units]
        self.assertEqual(starts, sorted(starts))


class LazyLoadingTests(unittest.TestCase):
    def test_importing_the_pipeline_never_loads_the_aligner(self) -> None:
        code = ("import sys; import ai.shorts_pipeline, ai.caption_stack.final_captions; "
                "print(int('torch' in sys.modules), int('qwen_asr' in sys.modules), int('transformers' in sys.modules))")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                cwd=str(Path(__file__).resolve().parent.parent), check=True)
        self.assertEqual(result.stdout.split(), ["0", "0", "0"])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class FinalShortTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir final ş ")
        self.dir = Path(self._tmp.name)
        self.clip = self.dir / "final short.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=6",
                        "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000:duration=6", "-shortest",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(self.clip)], check=True)
        alignment.reset_shared_aligners()

    def tearDown(self) -> None:
        alignment.reset_shared_aligners()
        self._tmp.cleanup()

    A = "I literally told you not to do that, Tyler."
    B = "I literally told you not to do that, Tyla."

    def ears(self, a: str, b: str, fallback_heard: str, qwen_calls: list, fallback_calls: list) -> lexical.Ears:
        def qwen(path, instructions, settings_):
            qwen_calls.append(("precision" if "PRECISION" in instructions else "primary", Path(path).name))
            text = b if "PRECISION" in instructions else a
            return qwen_omni.parse_caption_evidence(json.dumps({"text": text, "utterances": [{"text": text}],
                                                                "uncertain_spans": []})), {"model": "fake-qwen"}

        def openai(path, *, prompted, model, language, verified_terms):
            fallback_calls.append((Path(path).name, prompted))
            return fallback_heard
        return lexical.Ears(qwen=qwen, openai=openai)

    def run_short(self, a: str, b: str, fallback_heard: str = "", **kwargs):
        qwen_calls: list = []
        fallback_calls: list = []
        model = FakeAlignerModel()
        providers = (alignment.QwenForcedAlignmentProvider(settings(), loaded(model)), None)
        result = final_captions.transcribe_final_short(
            self.clip, 6.0, participant_names=["KAI", "TYLA"], settings=settings(),
            ears=self.ears(a, b, fallback_heard, qwen_calls, fallback_calls), providers=providers, judge=None,
            **kwargs)
        return result, qwen_calls, fallback_calls, model

    def test_agreeing_ears_cost_two_qwen_calls_and_one_alignment(self) -> None:
        (words, parity, source, quality), qwen_calls, fallback_calls, model = self.run_short(self.A, self.A)
        self.assertEqual(sorted(k for k, _ in qwen_calls), ["precision", "primary"])
        self.assertEqual(fallback_calls, [])                                         # no OpenAI ear needed
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(" ".join(w["word"] for w in words), self.A)
        self.assertEqual(parity, 1.0)
        self.assertEqual(source, "qwen_omni:fake-qwen")
        self.assertEqual({w["timing_source"] for w in words}, {"qwen3_forced_aligner:fake/aligner"})
        self.assertEqual([w["token_ids"] for w in words], [[i, i + 1] for i in range(len(words))])
        self.assertTrue(quality["target_met"])
        self.assertEqual(quality["timing_authorities"], 1)
        self.assertNotIn("sk-secret-value", json.dumps(quality))
        self.assertEqual(quality["analysis_audio"]["sample_rate"], 16000)
        self.assertEqual(list(analysis_audio.TEMP_DIR.glob("analysis_*.wav")), [])   # the artifact is cleaned up

    def test_a_disputed_name_is_settled_by_evidence_and_timed_on_the_frozen_words(self) -> None:
        (words, _p, _s, quality), _q, fallback_calls, model = self.run_short(self.A, self.B, "do that, Tyla.")
        self.assertEqual(words[-1]["word"], "Tyla.")
        self.assertEqual(words[-1]["lexical_status"], "resolved")
        self.assertEqual(len(fallback_calls), 1)
        self.assertEqual(len(model.calls), 2)                                        # locator on A, final on frozen
        self.assertIn("Tyla.", model.calls[-1][1])
        self.assertEqual(quality["lexical"]["spans"][0]["decided_by"], "agreement_qwen_precision+fallback_local")
        alternatives = {a["source"]: a["token"] for a in words[-1]["lexical_alternatives"]}
        self.assertEqual(alternatives["qwen_primary"], "Tyler.")

    def test_unsettled_words_are_kept_and_marked_not_guessed(self) -> None:
        (words, _p, _s, quality), *_ = self.run_short(self.A, self.B, "do that, Tyson.")
        self.assertEqual(words[-1]["word"], "Tyler.")
        self.assertEqual(words[-1]["lexical_status"], "uncertain")
        self.assertFalse(quality["target_met"])
        self.assertEqual(quality["lexical_judge"]["status"], "unavailable")

    @mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "", "DASHSCOPE_BASE_URL": ""})
    def test_the_speaker_profile_uses_the_stack_and_keeps_its_contract(self) -> None:
        from ai.editor import captions, speaker_caption_support as scs

        scan = self.dir / "scan.json"
        scan.write_text(json.dumps({"segments": [], "speaker_names": {"A": "KAI", "B": "TYLA"}}), encoding="utf-8")
        seen = {}
        real_run = self.run_short(self.A, self.A)[0]              # the real stack with fake providers

        def fake_stack(clip, duration, *, participant_names, verified_terms):
            seen.update(clip=Path(clip), names=list(participant_names), terms=list(verified_terms))
            return real_run

        with mock.patch.object(scs.final_captions, "transcribe_final_short", side_effect=fake_stack), \
                mock.patch.object(scs, "SPEAKER_OUTPUT_DIR", self.dir / "speakers"):
            path = scs.create_speaker_profile(self.clip, 1, speaker_scan_path=scan, verified_terms=["Creator X"])
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "ok", data.get("error"))
        self.assertEqual(seen["clip"], self.clip.resolve())                       # the FINAL edited short itself
        self.assertEqual((seen["names"], seen["terms"]), (["KAI", "TYLA"], ["Creator X"]))
        self.assertEqual(data["timing_basis"], final_captions.TIMING_BASIS)
        self.assertEqual(" ".join(w["word"] for w in data["words"]), self.A)
        self.assertTrue(all("token_ids" in w for w in data["words"]))
        rows = captions._profile_edited_words(data, 6.0)
        self.assertEqual([r["word"] for r in rows], self.A.split())


if __name__ == "__main__":
    unittest.main()
