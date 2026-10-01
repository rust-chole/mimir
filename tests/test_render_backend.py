"""Shared render backend: detection, selection, encoder profiles, runtime fallback, cache identity.

Offline. No GPU is needed or assumed: hardware encoders are simulated with mocks
(capability listings, smoke results, failing encodes). The tests that use the
real FFmpeg only rely on libx264 and on encoders that are absent or unusable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pro_edit_fixtures as fx  # noqa: F401  (sys.path)
from ai import shorts_pipeline as sp
from ai.editor import caption_renderer, intro_renderer, meme_renderer, pacing_cutter
from ai.editor import render_backend as rb
from ai.editor.pro_edit import executor
from ai.editor.pro_edit.camera import IDENTITY, CameraPath, CameraSegment, CameraState, OutputProfile
from ai.editor.pro_edit.errors import EditRenderError
from ai.editor.pro_edit.ffmpeg_filters import FfmpegCapabilities
from ai.editor.pro_edit.presets import ResolvedPlan

ROOT = Path(__file__).resolve().parent.parent
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
LEGACY_CPU_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p"]

ENCODERS_NVENC = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 A....D aac                  AAC (Advanced Audio Coding)
"""
ENCODERS_AMF = """ V....D libx264              libx264 H.264 (codec h264)
 V....D h264_amf             AMD AMF H.264 Encoder (codec h264)
 V....D hevc_amf             AMD AMF HEVC encoder (codec hevc)
"""
ENCODERS_QSV = """ V....D libx264              libx264 H.264 (codec h264)
 V..... h264_qsv             H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (Intel Quick Sync Video acceleration) (codec h264)
"""
ENCODERS_VT = """ V....D libx264              libx264 H.264 (codec h264)
 V....D h264_videotoolbox    VideoToolbox H.264 Encoder (codec h264)
"""
ENCODERS_CPU = """ V....D libx264              libx264 H.264 (codec h264)
 V....D libx264rgb           libx264 H.264 RGB (codec h264)
 V....D mpeg4                MPEG-4 part 2
"""
HWACCELS = "Hardware acceleration methods:\ncuda\nd3d11va\nqsv\n\n"


def caps(*encoders: str) -> rb.RenderCapabilities:
    return rb.RenderCapabilities(probed=True, ffmpeg_available=True, version="ffmpeg version test",
                                 h264_encoders=("libx264", *encoders), hwaccels=("cuda",))


def fake_smoke(working: dict[str, set[str]] | None = None, calls: list | None = None):
    """Fake smoke test: ``working`` maps backend -> tiers that encode."""
    working = working or {}

    def run(profile: rb.EncoderProfile) -> rb.SmokeResult:
        if calls is not None:
            calls.append(profile.profile_id)
        ok = profile.tier in working.get(profile.backend, set())
        return rb.SmokeResult(ok, "passed" if ok else f"[{profile.encoder}] Cannot load driver", 0.01)

    return run


def selection(profile: rb.EncoderProfile, requested: str = "auto") -> rb.BackendSelection:
    return rb.BackendSelection(requested, rb.normalize_mode(requested)[0], profile, "test selection",
                               "passed" if profile.is_hardware else "not_run", caps(profile.encoder))


NVENC_HQ = rb.HARDWARE_PROFILES[rb.NVENC][0]
AMF_HQ = rb.HARDWARE_PROFILES[rb.AMF][0]


class BackendStateMixin:
    def setUp(self) -> None:  # noqa: D401
        rb.reset()
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(rb.ENV_VAR, None)

    def tearDown(self) -> None:
        self._env.stop()
        rb.reset()


# ============================================================
# 1. CAPABILITY PARSING
# ============================================================

class CapabilityParsingTests(unittest.TestCase):
    def test_each_vendor_listing_is_parsed(self) -> None:
        self.assertIn("h264_nvenc", rb.parse_encoders(ENCODERS_NVENC))
        self.assertNotIn("hevc_nvenc", rb.parse_encoders(ENCODERS_NVENC))      # H.264 only
        self.assertNotIn("aac", rb.parse_encoders(ENCODERS_NVENC))
        self.assertIn("h264_amf", rb.parse_encoders(ENCODERS_AMF))
        self.assertIn("h264_qsv", rb.parse_encoders(ENCODERS_QSV))
        self.assertIn("h264_videotoolbox", rb.parse_encoders(ENCODERS_VT))

    def test_cpu_only_build_lists_no_hardware_encoder(self) -> None:
        listed = rb.parse_encoders(ENCODERS_CPU)
        self.assertIn("libx264", listed)
        self.assertFalse({rb.ENCODERS[name] for name in rb.AUTO_PRIORITY} & set(listed))

    def test_legend_and_header_lines_are_ignored(self) -> None:
        self.assertEqual(rb.parse_encoders(" V..... = Video\n ------\n"), ())

    def test_hwaccels(self) -> None:
        self.assertEqual(rb.parse_hwaccels(HWACCELS), ("cuda", "d3d11va", "qsv"))
        self.assertEqual(rb.parse_hwaccels(""), ())

    def test_probe_reads_version_encoders_and_hwaccels(self) -> None:
        outputs = {"-version": (0, "ffmpeg version 7.1-full\nbuilt with gcc\n", ""),
                   "-encoders": (0, ENCODERS_NVENC + ENCODERS_AMF, ""), "-hwaccels": (0, HWACCELS, "")}
        with mock.patch.object(rb, "_run", side_effect=lambda args, timeout: outputs[args[-1]]):
            found = rb.probe_capabilities()
        self.assertTrue(found.ffmpeg_available)
        self.assertEqual(found.version, "ffmpeg version 7.1-full")
        self.assertTrue(found.lists(rb.NVENC) and found.lists(rb.AMF))
        self.assertFalse(found.lists(rb.QSV))
        self.assertEqual(found.hwaccels, ("cuda", "d3d11va", "qsv"))

    def test_missing_ffmpeg_is_reported_not_raised(self) -> None:
        with mock.patch.object(rb, "_run", return_value=(127, "", "ffmpeg not found")):
            found = rb.probe_capabilities()
        self.assertFalse(found.ffmpeg_available)
        self.assertIn("not found", found.error)


# ============================================================
# 2. SMOKE TEST ROUTING
# ============================================================

class SmokeTestTests(unittest.TestCase):
    def fake_run(self, *, encode_rc: int = 0, probe: dict | None = None, seen: list | None = None):
        probe = probe if probe is not None else {"codec_name": "h264", "pix_fmt": "yuv420p",
                                                 "width": rb.SMOKE_WIDTH, "height": rb.SMOKE_HEIGHT,
                                                 "nb_frames": str(rb.SMOKE_FRAMES)}

        def run(args, timeout):
            if seen is not None:
                seen.append(list(args))
            if args[0] == rb.FFMPEG:
                if encode_rc == 0:
                    Path(args[-1]).write_bytes(b"\0" * 2048)
                    return 0, "", ""
                return encode_rc, "", "[h264_nvenc @ 0x55d] Cannot load libcuda.so.1\nError while opening encoder\n"
            return 0, json.dumps({"streams": [probe]}), ""

        return run

    def test_success_encodes_a_tiny_generated_pattern_and_leaves_nothing(self) -> None:
        seen: list[list[str]] = []
        with mock.patch.object(rb, "_run", side_effect=self.fake_run(seen=seen)):
            result = rb.smoke_test(NVENC_HQ)
        self.assertTrue(result.ok, result.detail)
        encode, probe = seen
        self.assertIn("lavfi", encode)
        self.assertTrue(any(arg.startswith("testsrc2=") for arg in encode))       # never user media
        self.assertEqual(encode[encode.index("-frames:v") + 1], str(rb.SMOKE_FRAMES))
        joined = " ".join(encode)
        self.assertIn(" ".join(NVENC_HQ.video_args()), joined)                     # the EXACT profile
        self.assertFalse(Path(encode[-1]).exists())                                # temp dir removed
        self.assertFalse(Path(encode[-1]).parent.exists())
        self.assertEqual(probe[0], rb.FFPROBE)

    def test_encoder_failure_is_a_failed_smoke_with_the_first_error_line(self) -> None:
        with mock.patch.object(rb, "_run", side_effect=self.fake_run(encode_rc=1)):
            result = rb.smoke_test(NVENC_HQ)
        self.assertFalse(result.ok)
        self.assertEqual(result.detail, "[h264_nvenc] Cannot load libcuda.so.1")

    def test_output_contract_violations_fail(self) -> None:
        for change in ({"pix_fmt": "yuvj420p"}, {"codec_name": "hevc"}, {"nb_frames": "11"}, {"width": 160}):
            with self.subTest(change=change):
                probe = {"codec_name": "h264", "pix_fmt": "yuv420p", "width": rb.SMOKE_WIDTH,
                         "height": rb.SMOKE_HEIGHT, "nb_frames": str(rb.SMOKE_FRAMES), **change}
                with mock.patch.object(rb, "_run", side_effect=self.fake_run(probe=probe)):
                    result = rb.smoke_test(AMF_HQ)
                self.assertFalse(result.ok)
                self.assertIn("output contract", result.detail)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
    def test_real_ffmpeg_cpu_profile_passes_and_an_unusable_encoder_fails(self) -> None:
        self.assertTrue(rb.smoke_test(rb.CPU_PROFILE).ok)
        with mock.patch.dict(rb.ENCODERS, {rb.NVENC: "h264_mimir_missing_encoder"}):
            missing = rb.smoke_test(NVENC_HQ)
        self.assertFalse(missing.ok)
        self.assertTrue(missing.detail)
        leftovers = list(Path(tempfile.gettempdir()).glob("mimir_render_probe_*"))
        self.assertEqual(leftovers, [])


# ============================================================
# 3-5, 15. SELECTION
# ============================================================

class SelectionTests(BackendStateMixin, unittest.TestCase):
    def test_auto_prefers_a_working_encoder_in_deterministic_order(self) -> None:
        both = caps("h264_nvenc", "h264_amf", "h264_qsv")
        chosen = rb.choose_backend("auto", lambda: both, fake_smoke({rb.NVENC: {"hq"}, rb.AMF: {"hq"}, rb.QSV: {"hq"}}))
        self.assertEqual(chosen.profile, NVENC_HQ)
        self.assertEqual(chosen.hardware_test, "passed")
        rows = {row["backend"]: row["result"] for row in chosen.candidates}
        self.assertTrue(rows[rb.AMF].startswith("not tested"))                 # no needless smoke tests

    def test_actual_working_encoder_beats_the_vendor_guess(self) -> None:
        # NVENC is listed (e.g. a gyan.dev build) but has no NVIDIA driver; AMF works (RX 580 class).
        listed = caps("h264_nvenc", "h264_amf")
        chosen = rb.choose_backend("auto", lambda: listed, fake_smoke({rb.AMF: {"hq"}}))
        self.assertEqual(chosen.profile, AMF_HQ)
        nvenc = next(row for row in chosen.candidates if row["backend"] == rb.NVENC)
        self.assertEqual(nvenc["result"], "smoke test failed")
        self.assertEqual([a["tier"] for a in nvenc["attempts"]], ["hq", "compat"])   # bounded: two tiers

    def test_a_simpler_hardware_profile_is_tried_before_cpu(self) -> None:
        chosen = rb.choose_backend("auto", lambda: caps("h264_qsv"), fake_smoke({rb.QSV: {"compat"}}))
        self.assertEqual(chosen.profile, rb.HARDWARE_PROFILES[rb.QSV][1])
        self.assertIn("compat", chosen.reason)

    def test_cpu_only_machine(self) -> None:
        chosen = rb.choose_backend("auto", lambda: caps(), fake_smoke())
        self.assertEqual(chosen.profile, rb.CPU_PROFILE)
        self.assertEqual(chosen.hardware_test, "not_run")
        self.assertIn("no hardware H.264 encoder", chosen.reason)
        listed_but_broken = rb.choose_backend("auto", lambda: caps("h264_nvenc", "h264_qsv"), fake_smoke())
        self.assertEqual(listed_but_broken.profile, rb.CPU_PROFILE)
        self.assertEqual(listed_but_broken.hardware_test, "failed")
        self.assertIn("no usable hardware H.264 encoder", listed_but_broken.reason)

    def test_explicit_override_selects_that_backend_only(self) -> None:
        calls: list[str] = []
        chosen = rb.choose_backend("amf", lambda: caps("h264_nvenc", "h264_amf"),
                                   fake_smoke({rb.NVENC: {"hq"}, rb.AMF: {"hq"}}, calls))
        self.assertEqual(chosen.profile, AMF_HQ)
        self.assertEqual(calls, ["amf_hq"])                                     # nvenc never touched
        self.assertEqual(rb.choose_backend("NVIDIA", lambda: caps("h264_nvenc"), fake_smoke({rb.NVENC: {"hq"}})).profile,
                         NVENC_HQ)                                              # alias, case-insensitive

    def test_unavailable_explicit_backend_falls_back_to_cpu_without_crashing(self) -> None:
        missing = rb.choose_backend("videotoolbox", lambda: caps("h264_nvenc"), fake_smoke({rb.NVENC: {"hq"}}))
        self.assertEqual(missing.profile, rb.CPU_PROFILE)
        self.assertIn("unavailable", missing.reason)
        self.assertIn("h264_videotoolbox is not in this FFmpeg build", missing.reason)
        broken = rb.choose_backend("nvenc", lambda: caps("h264_nvenc"), fake_smoke())
        self.assertEqual(broken.profile, rb.CPU_PROFILE)
        self.assertEqual(broken.hardware_test, "failed")
        self.assertIn("Cannot load driver", broken.reason)
        no_ffmpeg = rb.choose_backend("nvenc", lambda: rb.RenderCapabilities(True, False, "unavailable", error="x"),
                                      fake_smoke())
        self.assertEqual(no_ffmpeg.profile, rb.CPU_PROFILE)

    def test_explicit_cpu_never_probes_hardware(self) -> None:
        probe = mock.Mock(side_effect=AssertionError("must not probe"))
        chosen = rb.choose_backend("cpu", probe, fake_smoke())
        self.assertEqual(chosen.profile, rb.CPU_PROFILE)
        probe.assert_not_called()

    def test_default_is_auto_and_unknown_values_fall_back_to_auto(self) -> None:
        self.assertEqual(rb.normalize_mode(None), ("auto", ()))
        self.assertEqual(rb.normalize_mode(""), ("auto", ()))
        mode, notes = rb.normalize_mode("rtx4090")
        self.assertEqual(mode, "auto")
        self.assertTrue(notes)
        self.assertEqual(rb.DEFAULT_MODE, "auto")

    def test_env_example_documents_auto_as_the_default(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        values = dict(line.split("=", 1) for line in text.splitlines() if line and not line.startswith("#")
                      and "=" in line)
        self.assertEqual(values.get("MIMIR_RENDER_BACKEND"), "auto")
        block = text[text.index("VIDEO RENDERING"):text.index("MIMIR_RENDER_BACKEND=")]
        for phrase in ("recommended", "No NVIDIA", "FFmpeg build", "CPU is always the safe fallback"):
            self.assertIn(phrase, block)
        for mode in rb.MODES:
            self.assertIn(mode, block)

    def test_detection_runs_once_per_process(self) -> None:
        probe = mock.Mock(return_value=caps("h264_nvenc"))
        smoke = mock.Mock(return_value=rb.SmokeResult(True, "passed"))
        with mock.patch.object(rb, "probe_capabilities", probe), mock.patch.object(rb, "smoke_test", smoke):
            for _ in range(5):
                rb.get_selection()
                rb.active_profile()
                rb.signature_payload()
            rb.begin_run()                                  # a new pipeline run keeps the detection
            rb.get_selection()
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(smoke.call_count, 1)

    def test_render_selection_is_independent_of_the_aligner_device(self) -> None:
        # Ryzen 7 5700X + RX 580: Qwen ForcedAligner on CPU, video on AMD AMF.
        with mock.patch.dict(os.environ, {"MIMIR_QWEN_ALIGNER_DEVICE": "cpu", rb.ENV_VAR: "auto"}), \
                mock.patch.object(rb, "probe_capabilities", return_value=caps("h264_amf")), \
                mock.patch.object(rb, "smoke_test", side_effect=fake_smoke({rb.AMF: {"hq"}})):
            self.assertEqual(rb.get_selection().profile, AMF_HQ)
        source = (ROOT / "ai" / "editor" / "render_backend.py").read_text(encoding="utf-8")
        for coupling in ("ALIGNER", "torch", "caption_stack"):
            self.assertNotIn(coupling, source)

    def test_summary_is_concise_and_printed_once(self) -> None:
        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)):
            lines = rb.summary_lines()
            self.assertEqual(lines[0], "Render backend : NVIDIA NVENC")
            self.assertIn("Encoder        : h264_nvenc", lines[1])
            self.assertIn("CPU fallback   : libx264", lines)
            self.assertIn("Hardware test  : passed", lines)
            out = mock.Mock()
            with mock.patch("builtins.print") as printed:
                rb.print_summary_once(out)
                rb.print_summary_once(out)
            self.assertEqual(printed.call_count, len(lines))
        cpu = rb.BackendSelection("auto", "auto", rb.CPU_PROFILE, "no usable hardware H.264 encoder", "failed", caps())
        self.assertEqual(rb.summary_lines(cpu)[:2], ["Render backend : CPU", "Encoder        : libx264"])
        self.assertIn("Reason         : no usable hardware H.264 encoder", rb.summary_lines(cpu))


# ============================================================
# 6, 11. ARGUMENTS
# ============================================================

class ArgumentTests(unittest.TestCase):
    def test_cpu_profile_is_exactly_the_previous_libx264_settings(self) -> None:
        self.assertEqual(rb.CPU_PROFILE.video_args(), LEGACY_CPU_VIDEO_ARGS)

    def test_exact_hardware_arguments(self) -> None:
        expected = {
            "nvenc_hq": ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                         "-spatial-aq", "1", "-rc-lookahead", "20", "-bf", "3", "-profile:v", "high",
                         "-pix_fmt", "yuv420p"],
            "nvenc_compat": ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", "19", "-b:v", "0", "-profile:v", "high",
                             "-pix_fmt", "yuv420p"],
            "amf_hq": ["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "19",
                       "-bf", "0", "-profile:v", "high", "-pix_fmt", "yuv420p"],
            "amf_compat": ["-c:v", "h264_amf", "-rc", "cqp", "-qp_i", "18", "-qp_p", "19", "-pix_fmt", "yuv420p"],
            "qsv_hq": ["-c:v", "h264_qsv", "-preset", "slow", "-global_quality", "19", "-profile:v", "high",
                       "-pix_fmt", "nv12"],
            "qsv_compat": ["-c:v", "h264_qsv", "-q:v", "19", "-pix_fmt", "nv12"],
            "videotoolbox_hq": ["-c:v", "h264_videotoolbox", "-q:v", "65", "-profile:v", "high",
                                "-pix_fmt", "yuv420p"],
            "videotoolbox_compat": ["-c:v", "h264_videotoolbox", "-q:v", "65", "-pix_fmt", "yuv420p"],
        }
        actual = {profile.profile_id: profile.video_args()
                  for profiles in rb.HARDWARE_PROFILES.values() for profile in profiles}
        self.assertEqual(actual, expected)

    def test_every_profile_stays_h264_and_never_crf_copies_a_qp_scale(self) -> None:
        for profiles in rb.HARDWARE_PROFILES.values():
            for profile in profiles:
                args = profile.video_args()
                self.assertTrue(args[1].startswith("h264_"), args)
                self.assertNotIn("-crf", args)                              # hardware has no CRF
                self.assertNotIn("hevc", " ".join(args))
                self.assertIn(profile.pix_fmt, ("yuv420p", "nv12"))         # both decode as yuv420p
                self.assertNotIn("-maxrate", args)
                if "-b:v" in args:
                    self.assertEqual(args[args.index("-b:v") + 1], "0")     # constant quality, no bitrate guess

    def renders(self, profile: rb.EncoderProfile) -> dict[str, list[str]]:
        src, out = Path("C:/in dir/src.mp4"), Path("C:/o/out.mp4")
        caps_ = FfmpegCapabilities("ffmpeg test", True, True, 1, 0, "-filter_script:v")
        with mock.patch.object(meme_renderer, "build_visual_filter", return_value=("VFC", "[out_v]")), \
                mock.patch.object(meme_renderer, "build_audio_filter", return_value=("AFC", "[aout]")):
            meme_visual = meme_renderer.build_render_command(
                base_video=src, asset_path=Path("m.png"), event={"media_type": "image"},
                base_info={"has_audio": True, "duration": 30.0}, output_path=out, profile=profile)
            meme_audio = meme_renderer.build_render_command(
                base_video=src, asset_path=Path("m.wav"), event={"media_type": "audio"},
                base_info={"has_audio": True, "duration": 30.0}, output_path=out, profile=profile)
        return {
            "pacing": pacing_cutter.build_render_command(
                source_video=src, filter_complex="FC", video_output="[vout]", audio_output="[aout]",
                output_path=out, profile=profile),
            "caption": caption_renderer.build_render_command(
                edited_clip=src, subtitle_filter="subtitles=filename='x.ass'", output_path=out, profile=profile),
            "pro_edit": executor.build_render_command(src, Path("C:/g/graph.txt"), out, caps_, profile),
            "intro": intro_renderer.build_render_command(
                edited_clip=src, captioned_preview=Path("C:/p/main.mp4"), filter_complex="IFC",
                maps=["-map", "[v]", "-map", "[a]"], fps=29.97003, reencode_audio=True, output_path=out,
                profile=profile),
            "meme_visual": meme_visual,
            "meme_audio": meme_audio,
        }

    def test_cpu_commands_are_identical_to_the_previous_renderers(self) -> None:
        src, out = str(Path("C:/in dir/src.mp4")), str(Path("C:/o/out.mp4"))
        x264 = LEGACY_CPU_VIDEO_ARGS
        legacy = {
            "pacing": ["ffmpeg", "-y", "-i", src, "-filter_complex", "FC", "-map", "[vout]", *x264,
                       "-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out],
            "caption": ["ffmpeg", "-y", "-i", src, "-vf", "subtitles=filename='x.ass'", *x264, "-c:a", "copy",
                        "-movflags", "+faststart", out],
            "pro_edit": ["ffmpeg", "-y", "-hide_banner", "-nostdin", "-loglevel", "error", "-i", src,
                         "-filter_script:v", str(Path("C:/g/graph.txt")), *x264, "-c:a", "copy",
                         "-movflags", "+faststart", out],
            "intro": ["ffmpeg", "-y", "-i", src, "-i", str(Path("C:/p/main.mp4")), "-filter_complex", "IFC",
                      "-map", "[v]", "-map", "[a]", "-r", "29.970030", *x264, "-c:a", "aac", "-b:a", "192k",
                      "-movflags", "+faststart", out],
            "meme_visual": ["ffmpeg", "-y", "-i", src, "-i", "m.png", "-filter_complex", "VFC", "-map", "[out_v]",
                            *x264, "-map", "0:a:0?", "-c:a", "copy", "-map_metadata", "0", "-movflags",
                            "+faststart", out],
            "meme_audio": ["ffmpeg", "-y", "-i", src, "-i", "m.wav", "-filter_complex", "AFC", "-map", "0:v:0",
                           "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map_metadata", "0",
                           "-movflags", "+faststart", out],
        }
        self.assertEqual(self.renders(rb.CPU_PROFILE), legacy)

    def test_hardware_swaps_only_the_encoder_arguments(self) -> None:
        cpu, hw = self.renders(rb.CPU_PROFILE), self.renders(NVENC_HQ)
        for name in cpu:
            with self.subTest(render=name):
                if name == "meme_audio":
                    self.assertEqual(hw[name], cpu[name])                   # stream copy: no encoder at all
                    continue
                start = cpu[name].index("-c:v")
                swapped = cpu[name][:start] + NVENC_HQ.video_args() + cpu[name][start + len(LEGACY_CPU_VIDEO_ARGS):]
                self.assertEqual(hw[name], swapped)                         # filters / maps / audio untouched


# ============================================================
# 12. AUDIO-ONLY MEME STAYS A VIDEO STREAM COPY
# ============================================================

class AudioMemeTests(unittest.TestCase):
    def test_audio_only_meme_copies_video_and_never_asks_the_backend(self) -> None:
        with mock.patch.object(meme_renderer, "build_audio_filter", return_value=("AFC", "[aout]")):
            command = meme_renderer.build_render_command(
                base_video=Path("b.mp4"), asset_path=Path("a.wav"), event={"media_type": "audio"},
                base_info={"has_audio": True, "duration": 10.0}, output_path=Path("o.mp4"), profile=None)
        self.assertEqual(command[command.index("-c:v") + 1], "copy")
        self.assertNotIn("-pix_fmt", command)
        self.assertFalse(any(arg in command for arg in ("libx264", "h264_nvenc", "h264_amf", "-crf")))
        with self.assertRaises(ValueError):                                 # a visual meme needs an encoder
            meme_renderer.build_render_command(
                base_video=Path("b.mp4"), asset_path=Path("a.png"), event={"media_type": "image"},
                base_info={"has_audio": True, "duration": 10.0}, output_path=Path("o.mp4"), profile=None)
        source = (ROOT / "ai" / "editor" / "meme_renderer.py").read_text(encoding="utf-8")
        audio_branch = source[source.index('# Audio-only effect: the video stream is copied'):]
        audio_branch = audio_branch[:audio_branch.index("    else:")]
        self.assertNotIn("run_encode", audio_branch)


# ============================================================
# 7, 8. RUNTIME FALLBACK
# ============================================================

class RuntimeFallbackTests(BackendStateMixin, unittest.TestCase):
    def test_hardware_failure_retries_once_on_cpu_and_disables_hardware(self) -> None:
        attempts: list[str] = []

        def encode(profile: rb.EncoderProfile) -> str:
            attempts.append(profile.encoder)
            if profile.is_hardware:
                raise RuntimeError("FFmpeg pacing render hatası: [h264_nvenc] OpenEncodeSessionEx failed: out of memory")
            return "ok"

        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)), \
                mock.patch("builtins.print"):
            self.assertEqual(rb.run_encode("pacing_cut", encode), "ok")
            self.assertEqual(attempts, ["h264_nvenc", "libx264"])           # exactly one CPU retry
            self.assertIn(rb.NVENC, rb.unhealthy())
            self.assertEqual(rb.active_profile(), rb.CPU_PROFILE)
            # Later stages never touch the broken encoder again in this run.
            rb.run_encode("caption_render", encode)
            rb.run_encode("intro_render", encode)
            rb.run_encode("meme_render", encode)
            self.assertEqual(attempts, ["h264_nvenc", "libx264", "libx264", "libx264", "libx264"])
            report = rb.report()
        self.assertTrue(report["fallback_occurred"])
        self.assertEqual(report["selected"]["encoder"], "h264_nvenc")
        self.assertEqual(report["active"]["encoder"], "libx264")
        self.assertIn("OpenEncodeSessionEx", report["fallback_reason"])
        self.assertEqual(report["unhealthy"][rb.NVENC]["stage"], "pacing_cut")
        self.assertEqual(report["stages"]["pacing_cut"]["backend"], rb.CPU)
        self.assertTrue(report["stages"]["pacing_cut"]["fallback"])
        self.assertEqual(report["stages"]["pacing_cut"]["failed_attempts"], 1)
        self.assertFalse(report["stages"]["caption_render"]["fallback"])
        self.assertEqual(sum(1 for row in report["encodes"] if row["backend"] == rb.NVENC), 1)

    def test_cpu_retry_failure_propagates_and_there_is_no_loop(self) -> None:
        calls: list[str] = []

        def encode(profile: rb.EncoderProfile) -> None:
            calls.append(profile.backend)
            raise RuntimeError(f"broken input ({profile.backend})")

        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)), \
                mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, r"broken input \(cpu\)"):
                rb.run_encode("caption_render", encode)
        self.assertEqual(calls, [rb.NVENC, rb.CPU])

    def test_cpu_failure_is_not_retried(self) -> None:
        calls: list[str] = []

        def encode(profile: rb.EncoderProfile) -> None:
            calls.append(profile.backend)
            raise RuntimeError("cpu failure")

        with mock.patch.object(rb, "get_selection", return_value=selection(rb.CPU_PROFILE)):
            with self.assertRaises(RuntimeError):
                rb.run_encode("caption_render", encode)
        self.assertEqual(calls, [rb.CPU])
        self.assertEqual(rb.unhealthy(), {})

    def test_interrupt_is_never_swallowed_or_retried(self) -> None:
        def encode(profile: rb.EncoderProfile) -> None:
            raise KeyboardInterrupt

        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)):
            with self.assertRaises(KeyboardInterrupt):
                rb.run_encode("pacing_cut", encode)
        self.assertEqual(rb.unhealthy(), {})

    def test_health_resets_for_a_new_run_but_not_mid_run(self) -> None:
        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)):
            rb.mark_unhealthy(rb.NVENC, "pacing_cut", "driver reset")
            self.assertEqual(rb.active_profile(), rb.CPU_PROFILE)
            rb.begin_run()
            self.assertEqual(rb.active_profile(), NVENC_HQ)

    def test_hardware_output_contract_check(self) -> None:
        with mock.patch.object(rb, "_run", side_effect=AssertionError("cpu output is never re-probed")):
            rb.check_output("x.mp4", rb.CPU_PROFILE)
        bad = json.dumps({"streams": [{"codec_name": "h264", "pix_fmt": "yuvj420p"}]})
        with mock.patch.object(rb, "_run", return_value=(0, bad, "")):
            with self.assertRaises(rb.HardwareOutputError):
                rb.check_output("x.mp4", NVENC_HQ)
        good = json.dumps({"streams": [{"codec_name": "h264", "pix_fmt": "yuv420p"}]})
        with mock.patch.object(rb, "_run", return_value=(0, good, "")):
            rb.check_output("x.mp4", rb.HARDWARE_PROFILES[rb.QSV][0])      # nv12 in -> yuv420p H.264 out


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class RealRendererFallbackTests(BackendStateMixin, unittest.TestCase):
    """A real caption render with a 'hardware' encoder that passed detection but fails on media."""

    ASS = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 180\n\n[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
           "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
           "MarginL, MarginR, MarginV, Encoding\n"
           "Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,"
           "10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           "Dialogue: 0,0:00:00.20,0:00:01.50,Default,,0,0,0,,hello world\n")

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir render backend ✓ ")
        self.root = Path(self._tmp.name)
        self.clip = self.root / "clip_01_demo_edited.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=30:duration=2", "-f", "lavfi", "-i",
                        "sine=frequency=300:sample_rate=48000:duration=2", "-shortest", "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(self.clip)],
                       check=True, capture_output=True)
        self.ass = self.root / "captions.ass"
        self.ass.write_text(self.ASS, encoding="utf-8")
        self.timeline = self.root / "demo_timeline_v3.json"
        self.timeline.write_text(json.dumps({"version": 3, "source": {"video_stem": "demo"}, "timelines": [
            {"clip_index": 1, "title": "demo", "edited": {"estimated_duration": 2.0}}]}), encoding="utf-8")
        self._preview = mock.patch.object(caption_renderer, "PREVIEW_DIR", self.root / "previews")
        self._preview.start()

    def tearDown(self) -> None:
        self._preview.stop()
        self._tmp.cleanup()
        super().tearDown()

    def test_broken_hardware_encode_is_retried_once_on_libx264_with_a_valid_output(self) -> None:
        broken = rb.EncoderProfile(rb.NVENC, "hq", "test", "yuv420p", ("-rc", "vbr", "-cq", "19"))
        with mock.patch.dict(rb.ENCODERS, {rb.NVENC: "h264_mimir_missing_encoder"}), \
                mock.patch.object(rb, "get_selection", return_value=selection(broken)), \
                mock.patch("builtins.print"):
            output = caption_renderer.render_captioned_clip(self.timeline, 1, caption_path=self.ass,
                                                            edited_clip_path=self.clip)
            caption_renderer.render_captioned_clip(self.timeline, 1, caption_path=self.ass,
                                                   edited_clip_path=self.clip)
            report = rb.report()
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "json", str(output)], capture_output=True, text=True, check=True).stdout)["streams"][0]
        self.assertEqual((probe["codec_name"], probe["pix_fmt"]), ("h264", "yuv420p"))
        encoders = [(row["encoder"], row["ok"]) for row in report["encodes"]]
        self.assertEqual(encoders, [("h264_mimir_missing_encoder", False), ("libx264", True), ("libx264", True)])
        self.assertTrue(report["stages"]["caption_render"]["fallback"])

    def test_a_working_hardware_class_profile_is_used_and_checked(self) -> None:
        # A libx264-backed stand-in for a GPU encoder: the hardware code path end to end on real media.
        stand_in = rb.EncoderProfile(rb.NVENC, "hq", "test", "yuv420p", ("-preset", "ultrafast", "-crf", "17"))
        with mock.patch.dict(rb.ENCODERS, {rb.NVENC: "libx264"}), \
                mock.patch.object(rb, "get_selection", return_value=selection(stand_in)):
            caption_renderer.render_captioned_clip(self.timeline, 1, caption_path=self.ass,
                                                   edited_clip_path=self.clip)
            report = rb.report()
        self.assertFalse(report["fallback_occurred"])
        self.assertEqual(report["stages"]["caption_render"]["backend"], rb.NVENC)
        self.assertTrue(report["stages"]["caption_render"]["hardware"])


# ============================================================
# 13. PRO EDIT: ONE GRAPH, ONE ENCODE
# ============================================================

class ProEditSingleEncodeTests(BackendStateMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir pro edit encode ")
        self.root = Path(self._tmp.name)
        self.media = fx.media(duration=4.0)
        peak = CameraState(1.2, 0.6, 0.4)
        frames = self.media.frame_count
        path = CameraPath(frames, (CameraSegment(10, 20, IDENTITY, peak, "ease_out_cubic", "e", "attack"),
                                   CameraSegment(20, 40, peak, IDENTITY, "smoothstep", "e", "release")))
        self.resolved = ResolvedPlan(1, "clip", "pro_stream_v1", 1, self.media.fps, frames, 1920, 1080,
                                     OutputProfile.PRESERVE, (0, 0, 1920, 1080), (), path)
        self.caps = FfmpegCapabilities("ffmpeg test", True, True, 1, 0, "-filter_script:v")
        self.commands: list[list[str]] = []

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def fake_run(self, fail_encoder: str | None = None):
        def run(command, *, timeout, what):
            self.commands.append(list(command))
            if fail_encoder and fail_encoder in command:
                raise EditRenderError("pro edit render: exit code 1", returncode=1, stderr="encoder died",
                                      command=command)
            Path(command[-1]).write_bytes(b"\0" * 4096)
        return run

    def render(self, profile: rb.EncoderProfile, fail_encoder: str | None = None):
        with mock.patch.object(rb, "get_selection", return_value=selection(profile)), \
                mock.patch.object(executor, "run_command", side_effect=self.fake_run(fail_encoder)), \
                mock.patch.object(executor, "probe_media", return_value=self.media), \
                mock.patch.object(rb, "check_output"), mock.patch("builtins.print"):
            return executor.render_camera_captions(
                edited_clip=self.root / "paced.mp4", caption_file=self.root / "caps.ass",
                output_path=self.root / "out.mp4", resolved=self.resolved, caps=self.caps,
                source_media=self.media, script_path=self.root / "graph.txt", render_stage="pro_edit_main")

    def test_camera_and_captions_stay_one_graph_and_one_encode(self) -> None:
        result = self.render(NVENC_HQ)
        self.assertEqual(len(self.commands), 1)
        command = self.commands[0]
        self.assertEqual(command.count("-i"), 1)                               # no intermediate file
        graph = (self.root / "graph.txt").read_text(encoding="utf-8")
        self.assertLess(graph.index("perspective="), graph.index("subtitles="))   # camera BEFORE captions
        self.assertEqual(command[command.index("-c:v") + 1], "h264_nvenc")
        self.assertEqual(command[command.index("-c:a") + 1], "copy")
        self.assertEqual(result.command, tuple(command))
        self.assertTrue((self.root / "out.mp4").is_file())
        self.assertFalse(list(self.root.glob("*.proedit-tmp*")))

    def test_hardware_failure_reruns_the_same_single_graph_on_cpu(self) -> None:
        result = self.render(NVENC_HQ, fail_encoder="h264_nvenc")
        self.assertEqual([c[c.index("-c:v") + 1] for c in self.commands], ["h264_nvenc", "libx264"])
        hw, cpu = self.commands
        strip = lambda c: [a for a in c if a not in NVENC_HQ.video_args() + LEGACY_CPU_VIDEO_ARGS  # noqa: E731
                           and ".proedit-tmp" not in a]
        self.assertEqual(strip(hw), strip(cpu))                                # identical graph and audio copy
        self.assertEqual(result.command[result.command.index("-c:v") + 1], "libx264")
        self.assertIn(rb.NVENC, rb.unhealthy())

    def test_post_render_validation_failure_on_hardware_falls_back(self) -> None:
        short = fx.media(duration=3.0)                                        # dropped frames
        outputs = iter([short, self.media, self.media])
        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)), \
                mock.patch.object(executor, "run_command", side_effect=self.fake_run()), \
                mock.patch.object(executor, "probe_media", side_effect=lambda path: next(outputs)), \
                mock.patch.object(rb, "check_output"), mock.patch("builtins.print"):
            executor.render_camera_captions(
                edited_clip=self.root / "paced.mp4", caption_file=None, output_path=self.root / "intro.mp4",
                resolved=self.resolved, caps=self.caps, source_media=self.media,
                script_path=self.root / "intro.txt", render_stage="pro_edit_intro")
        self.assertEqual([c[c.index("-c:v") + 1] for c in self.commands], ["h264_nvenc", "libx264"])
        self.assertEqual(rb.report()["stages"]["pro_edit_intro"]["backend"], rb.CPU)

    def test_stage_labels_main_and_intro_renders(self) -> None:
        source = (ROOT / "ai" / "editor" / "pro_edit" / "stage.py").read_text(encoding="utf-8")
        self.assertIn('render_stage="pro_edit_main"', source)
        self.assertIn('render_stage="pro_edit_intro"', source)


# ============================================================
# 14. EVERY RENDERER USES THE SHARED BACKEND
# ============================================================

class SharedBackendUsageTests(unittest.TestCase):
    RENDERERS = {
        "ai/editor/pacing_cutter.py": "pacing_cut",
        "ai/editor/caption_renderer.py": "caption_render",
        "ai/editor/intro_renderer.py": "intro_render",
        "ai/editor/meme_renderer.py": "meme_render",
        "ai/editor/pro_edit/executor.py": None,
    }

    def test_no_private_encoder_settings_remain(self) -> None:
        for relative, stage in self.RENDERERS.items():
            with self.subTest(module=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotRegex(source, r'["\']libx264["\']|"-crf"|"-preset"|VIDEO_CODEC|VIDEO_PRESET|VIDEO_CRF')
                self.assertIn("render_backend.run_encode(", source)
                self.assertIn(".video_args()", source)
                if stage:
                    self.assertRegex(source, r'run_encode\(\s*"%s"' % stage)
        for module in (pacing_cutter, caption_renderer, intro_renderer, meme_renderer):
            for name in ("VIDEO_CODEC", "VIDEO_PRESET", "VIDEO_CRF"):
                self.assertFalse(hasattr(module, name), (module.__name__, name))

    def test_libx264_is_owned_by_the_backend_only(self) -> None:
        offenders = []
        for path in (ROOT / "ai").rglob("*.py"):
            if path.name == "render_backend.py":
                continue
            if re.search(r'["\']libx264["\']', path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


# ============================================================
# 9, 10. CACHE / SIGNATURE IDENTITY
# ============================================================

class SignatureTests(BackendStateMixin, unittest.TestCase):
    def test_render_signature_differs_between_backends_and_is_stable(self) -> None:
        payloads = {}
        for profile in (rb.CPU_PROFILE, NVENC_HQ, AMF_HQ, rb.HARDWARE_PROFILES[rb.NVENC][1]):
            with mock.patch.object(rb, "get_selection", return_value=selection(profile)):
                payloads[profile.profile_id] = rb.signature_payload()
                self.assertEqual(rb.signature_payload(), payloads[profile.profile_id])   # stable
        self.assertEqual(len({json.dumps(p, sort_keys=True) for p in payloads.values()}), 4)
        nvenc = payloads["nvenc_hq"]
        for key in ("backend", "encoder", "class", "profile", "profile_version", "quality", "pix_fmt", "args",
                    "ffmpeg"):
            self.assertIn(key, nvenc)
        self.assertEqual(nvenc["class"], "hardware")
        self.assertEqual(payloads["cpu_x264"]["class"], "cpu")
        self.assertNotIn("ffmpeg", payloads["cpu_x264"])
        sigs = {name: sp._render_signature("caption_render", backend=payload, options={"clip_index": 1})
                for name, payload in payloads.items()}
        self.assertEqual(len(set(sigs.values())), 4)

    def test_stage_payload_names_the_encoder_that_actually_ran(self) -> None:
        def flaky(profile: rb.EncoderProfile) -> None:
            if profile.is_hardware:
                raise RuntimeError("driver reset")

        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)), \
                mock.patch("builtins.print"):
            self.assertEqual(rb.stage_payload("pacing_cut")["backend"], rb.NVENC)    # nothing ran yet: active
            rb.run_encode("pacing_cut", flaky)
            self.assertEqual(rb.stage_payload("pacing_cut")["backend"], rb.CPU)       # the CPU retry produced it
            rb.begin_run()
            rb.run_encode("intro_render", lambda profile: None)
            self.assertEqual(rb.stage_payload("intro_render")["backend"], rb.NVENC)
            rb.mark_unhealthy(rb.NVENC, "intro_render", "later failure")
            rb.run_encode("intro_render", lambda profile: None)
            self.assertIn("mixed", rb.stage_payload("intro_render"))                 # never matches later

    def test_semantic_stage_signatures_ignore_the_render_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "t.json"
            transcript.write_text("{}", encoding="utf-8")
            sigs = set()
            for profile in (rb.CPU_PROFILE, NVENC_HQ, AMF_HQ):
                with mock.patch.object(rb, "get_selection", return_value=selection(profile)):
                    sigs.add(sp._stage_signature("captions", inputs=[transcript], modules=[sp.captions],
                                                 options={"clip_index": 1}))
        self.assertEqual(len(sigs), 1)
        # The paced clip's EDIT identity is the pre-backend pacing signature formula.
        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)):
            edit = sp._stage_signature("pacing_cut", options={"clip_index": 1})
            render = sp._render_signature("pacing_cut", backend=rb.signature_payload(), options={"clip_index": 1})
        self.assertNotEqual(edit, render)
        self.assertEqual(edit, sp._hash({"pipeline_version": sp.PIPELINE_VERSION, "stage": "pacing_cut", "inputs": [],
                                         "modules": [], "options": {"clip_index": 1}}))

    def test_render_stages_carry_the_backend(self) -> None:
        source = (ROOT / "ai" / "shorts_pipeline.py").read_text(encoding="utf-8")
        for stage in ("pacing_cut", "caption_render", "intro_final_base", "meme_render"):
            self.assertRegex(source, r'_render_signature\("%s", backend=render_backend\.signature_payload\(\)' % stage)
        for stage, keys in (("pacing_cut", '"pacing_cut"'), ("caption_render", '"pro_edit_main", "caption_render"'),
                            ("intro_final_base", '"intro_render"'), ("meme_render", '"meme_render"')):
            self.assertRegex(source, r'"%s", backend=render_backend\.stage_payload\(%s\)' % (stage, keys))

    def test_request_signature_tracks_backend_and_runtime_degradation(self) -> None:
        src = {"path": "x.mp4", "size": 1, "mtime_ns": 2}
        kw = dict(creator_name="C", clip_index=1, enable_memes=True, enable_video_brain=False, video_brain_model="m",
                  presentation_signature="p")
        with mock.patch.object(rb, "get_selection", return_value=selection(NVENC_HQ)):
            healthy = sp._request_signature(src, render_identity=rb.run_identity(), **kw)
            rb.mark_unhealthy(rb.NVENC, "pacing_cut", "driver reset")
            degraded = sp._request_signature(src, render_identity=rb.run_identity(), **kw)
        with mock.patch.object(rb, "get_selection", return_value=selection(rb.CPU_PROFILE)):
            rb.reset(detection=False)
            cpu = sp._request_signature(src, render_identity=rb.run_identity(), **kw)
        self.assertEqual(len({healthy, degraded, cpu}), 3)


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not available on PATH")
class PacedClipPinTests(unittest.TestCase):
    """A paced clip re-encoded by another video encoder keeps upstream analysis caches valid."""

    FC = ("[0:v]trim=start=0:end=2,setpts=PTS-STARTPTS[v0];[0:a]atrim=start=0:end=2,asetpts=PTS-STARTPTS[a0];"
          "[0:v]trim=start=3:end=5.5,setpts=PTS-STARTPTS[v1];[0:a]atrim=start=3:end=5.5,asetpts=PTS-STARTPTS[a1];"
          "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="mimir paced pin ş'")
        cls.root = Path(cls._tmp.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=30000/1001:duration=6", "-f", "lavfi", "-i",
                        "anoisesrc=color=pink:sample_rate=48000:duration=6:amplitude=0.1", "-shortest", "-c:v",
                        "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source)],
                       check=True, capture_output=True)
        cls.renders = {}
        # The pacing command with three different video encoders (and one different audio edit).
        for name, video, fc in (("cpu", LEGACY_CPU_VIDEO_ARGS, cls.FC),
                                ("other_encoder", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                                                   "-pix_fmt", "yuv420p"], cls.FC),
                                ("no_bframes", ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"], cls.FC),
                                ("other_edit", LEGACY_CPU_VIDEO_ARGS, cls.FC.replace("start=3:end=5.5",
                                                                                     "start=3.2:end=5.7"))):
            out = cls.root / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(cls.source), "-filter_complex", fc,
                            "-map", "[v]", *video, "-map", "[a]", "-c:a", "aac", "-b:a", "192k", "-movflags",
                            "+faststart", str(out)], check=True, capture_output=True)
            cls.renders[name] = out

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def setUp(self) -> None:
        sp._ANALYSIS_PINS.clear()
        self.paced = self.root / "clip_01_edited ✓.mp4"

    def tearDown(self) -> None:
        sp._ANALYSIS_PINS.clear()

    def first_run(self) -> tuple[dict, str]:
        shutil.copy2(self.renders["cpu"], self.paced)
        media = sp._media_identity(self.paced)
        analysis, fingerprint = sp._settle_paced_analysis(self.paced, {}, "edit-sig", media, rendered=True)
        record = {"signature": "render-sig-cpu", "path": str(self.paced), "media": media, "analysis": analysis,
                  "analysis_fingerprint": fingerprint}
        captions_sig = sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1})
        sp._ANALYSIS_PINS.clear()
        return record, captions_sig

    def rerender(self, name: str, record: dict, edit: str = "edit-sig") -> dict:
        shutil.copy2(self.renders[name], self.paced)                         # a re-render: new bytes + clock
        os.utime(self.paced, ns=(self.paced.stat().st_atime_ns, record["media"]["mtime_ns"] + 7_000_000_000))
        media = sp._keep_render_clock(self.paced, record, "render-sig-nvenc")
        sp._settle_paced_analysis(self.paced, record, edit, media, rendered=True)
        return media

    def test_audio_packets_and_clock_do_not_depend_on_the_video_encoder(self) -> None:
        identities = {name: sp._paced_analysis_identity(self.renders[name], "e")
                      for name in ("cpu", "other_encoder", "no_bframes")}
        self.assertIsNotNone(identities["cpu"])
        self.assertEqual(identities["cpu"], identities["other_encoder"])
        self.assertEqual(identities["cpu"], identities["no_bframes"])
        self.assertNotEqual(identities["cpu"], sp._paced_analysis_identity(self.renders["other_edit"], "e"))

    def test_other_encoder_same_edit_keeps_semantic_signatures_and_changes_render_signatures(self) -> None:
        record, captions_before = self.first_run()
        render_before = sp._render_signature("caption_render", backend={"b": "cpu"}, inputs=[self.paced])
        media = self.rerender("other_encoder", record)
        self.assertEqual(media.get("clock"), "pinned_equivalent_render")
        self.assertEqual(sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1}),
                         captions_before)                                    # Qwen / captions cache stays valid
        self.assertNotEqual(sp._render_signature("caption_render", backend={"b": "cpu"}, inputs=[self.paced]),
                            render_before)                                   # renders see the real bytes
        truth = sp._caption_truth_clip_fingerprint(self.paced)
        self.assertEqual(truth, {"name": self.paced.name, "size": record["media"]["size"],
                                 "mtime_ns": record["media"]["mtime_ns"]})

    def test_a_different_edit_or_audio_is_never_pinned(self) -> None:
        record, captions_before = self.first_run()
        media = self.rerender("other_edit", record)
        self.assertNotIn("clock", media)
        self.assertNotEqual(sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1}),
                            captions_before)
        self.assertIsNone(sp._caption_truth_clip_fingerprint(self.paced))
        sp._ANALYSIS_PINS.clear()
        record, captions_before = self.first_run()
        self.rerender("other_encoder", record, edit="another-pacing-signature")   # cutter/timeline changed
        self.assertNotEqual(sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1}),
                            captions_before)

    def test_pin_survives_a_reused_clip_and_breaks_when_the_file_changes(self) -> None:
        record, captions_before = self.first_run()
        media = self.rerender("other_encoder", record)
        _, pinned = sp._settle_paced_analysis(self.paced, record, "edit-sig", media, rendered=True)
        second = {**record, "media": media, "analysis_fingerprint": pinned,
                  "analysis": sp._paced_analysis_identity(self.paced, "edit-sig")}
        sp._ANALYSIS_PINS.clear()                                            # next run, clip reused as is
        sp._settle_paced_analysis(self.paced, second, "edit-sig", dict(media), rendered=False)
        self.assertEqual(sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1}),
                         captions_before)
        os.utime(self.paced, ns=(self.paced.stat().st_atime_ns, self.paced.stat().st_mtime_ns + 1))
        self.assertNotEqual(sp._stage_signature("captions", inputs=[self.paced], options={"clip_index": 1}),
                            captions_before)                                 # touched file: pin no longer applies


class ProfileTests(unittest.TestCase):
    def test_render_profile_compares_existing_stage_times_with_the_encoder_used(self) -> None:
        state = {"stages": {"pacing_cut": {"status": "done", "duration_seconds": 12.0},
                            "caption_render": {"status": "done", "duration_seconds": 30.0},
                            "intro_final_base": {"status": "done", "duration_seconds": 9.0},
                            "meme_render": {"status": "skipped", "duration_seconds": 0.1}}}
        backend = {"stages": {"pacing_cut": {"backends": ["nvenc"], "seconds": 10.0, "fallback": False},
                              "pro_edit_main": {"backends": ["cpu"], "seconds": 25.0, "fallback": True}}}
        profile = sp._build_profile(state, 100.0, {"final_qc": 4.5, "pacing_encode_worker": 11.0}, backend)
        rows = {row["label"]: row for row in profile["render"]["rows"]}
        self.assertEqual(rows["Pacing encode"]["backend"], "nvenc")
        self.assertEqual(rows["Pacing encode"]["worker_seconds"], 11.0)
        self.assertEqual(rows["Caption / Pro Edit render"]["backend"], "cpu")
        self.assertTrue(rows["Caption / Pro Edit render"]["fallback"])
        self.assertEqual(rows["Intro render"]["stage_wall_seconds"], 9.0)
        self.assertEqual(rows["Final QC"]["worker_seconds"], 4.5)
        self.assertEqual(profile["render"]["total_seconds"], 100.0)
        self.assertIn("final_qc", sp._PROFILE_LABELS)


if __name__ == "__main__":
    unittest.main()
