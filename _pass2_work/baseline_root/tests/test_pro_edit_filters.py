from __future__ import annotations

import unittest
from dataclasses import replace

import pro_edit_fixtures as fx
from ai.editor.pro_edit.camera import IDENTITY, CameraPath, CameraSegment, CameraState, OutputProfile
from ai.editor.pro_edit.errors import EditRenderError
from ai.editor.pro_edit.executor import ExpectedOutput, build_render_command, subtitle_filter_for, validate_output
from ai.editor.pro_edit.ffmpeg_filters import (
    FfmpegCapabilities,
    build_perspective_filter,
    build_video_filter,
    channel_expression,
    fmt,
    validate_filter_graph,
)
from ai.editor.pro_edit.presets import resolve_plan
from ai.editor.pro_edit.style import PRO_STREAM_V1
from ai.editor.pro_edit.validator import validate_plan_payload

CAPS = FfmpegCapabilities("ffmpeg test", True, True, 1, 0, "-filter_script:v")


def punch_path(frames: int = 90) -> CameraPath:
    peak = CameraState(1.2, 0.6, 0.4)
    return CameraPath(frames, (CameraSegment(10, 16, IDENTITY, peak, "ease_out_cubic", "e", "attack"),
                               CameraSegment(16, 30, peak, peak, "linear", "e", "hold"),
                               CameraSegment(30, 38, peak, IDENTITY, "smoothstep", "e", "release")))


def evaluate(expr: str, n: int) -> float:
    """Recursive-descent evaluator for the generated FFmpeg expression subset
    (numbers, N, + - * /, parentheses, if/lt/clip/pow). No eval()."""
    import math
    import re as _re

    tokens = _re.findall(r"\d+\.?\d*|[A-Za-z_]+|[-+*/(),]", expr.replace("(in-1)", "N"))
    pos = 0

    def peek() -> str | None:
        return tokens[pos] if pos < len(tokens) else None

    def take(expected: str | None = None) -> str:
        nonlocal pos
        token = tokens[pos]
        if expected is not None and token != expected:
            raise AssertionError(f"expected {expected!r} got {token!r}")
        pos += 1
        return token

    def atom() -> float:
        token = take()
        if token == "(":
            value = add()
            take(")")
            return value
        if token == "-":
            return -atom()
        if token == "N":
            return float(n)
        if token in {"if", "lt", "clip", "pow", "max"}:
            take("(")
            args = [add()]
            while peek() == ",":
                take(",")
                args.append(add())
            take(")")
            if token == "if":
                return args[1] if args[0] else args[2]
            if token == "lt":
                return 1.0 if args[0] < args[1] else 0.0
            if token == "clip":
                return max(args[1], min(args[2], args[0]))
            if token == "max":
                return max(args)
            return math.pow(args[0], args[1])
        return float(token)

    def mul() -> float:
        value = atom()
        while peek() in {"*", "/"}:
            op = take()
            rhs = atom()
            value = value * rhs if op == "*" else value / rhs
        return value

    def add() -> float:
        value = mul()
        while peek() in {"+", "-"}:
            op = take()
            rhs = mul()
            value = value + rhs if op == "+" else value - rhs
        return value

    result = add()
    if pos != len(tokens):
        raise AssertionError("unparsed tail")
    return result


class FilterTests(unittest.TestCase):
    def test_channel_expression_matches_python_math(self) -> None:
        path = punch_path()
        for channel in ("zoom", "cx", "cy"):
            expr = channel_expression(path, channel, "(in-1)")
            for frame in range(90):
                self.assertAlmostEqual(evaluate(expr, frame), getattr(path.state_at(frame), channel), places=5,
                                       msg=f"{channel}@{frame}")

    def test_perspective_filter_structure(self) -> None:
        text = build_perspective_filter(punch_path(), CAPS, interpolation="cubic")
        self.assertTrue(text.startswith("perspective="))
        for key in ("x0=", "y0=", "x1=", "y3=", "interpolation=cubic", "sense=source", "eval=frame",
                    "enable='between(n,10,37)'"):
            self.assertIn(key, text)
        validate_filter_graph(text)
        self.assertEqual(text, build_perspective_filter(punch_path(), CAPS, interpolation="cubic"))  # deterministic
        self.assertEqual(fmt(1.0), "1")
        self.assertEqual(fmt(-0.0000000001), "0")

    def test_uncalibrated_counter_refused(self) -> None:
        with self.assertRaises(EditRenderError):
            build_perspective_filter(punch_path(), replace(CAPS, perspective_frame_base=None))

    def test_filter_graph_validation(self) -> None:
        for bad in ("", "[0:v]null", "drawtext=text=x", "perspective=x0='((1)'", "null,\nnull",
                    "perspective=x0='1';movie=a", "subtitles=filename='a"):
            with self.subTest(bad=bad):
                with self.assertRaises(EditRenderError):
                    validate_filter_graph(bad)
        validate_filter_graph("crop=100:100:0:0:exact=1,perspective=x0='1':y0='0',subtitles=filename='C\\:/a b/c.ass'")

    def test_full_chain_order_camera_before_captions(self) -> None:
        ws = fx.Workspace()
        try:
            ctx = fx.make_context(ws)
            report = validate_plan_payload(fx.plan(fx.event("e", 13.0, 14.2)), ctx, PRO_STREAM_V1)
            resolved = resolve_plan(report.plan, ctx, PRO_STREAM_V1)
            graph = build_video_filter(resolved, CAPS, subtitle_filter=subtitle_filter_for("C:/caps dir/x.ass"))
            self.assertLess(graph.index("perspective="), graph.index("subtitles="))
            portrait = resolve_plan(report.plan, ctx, PRO_STREAM_V1, output_profile=OutputProfile.PORTRAIT_9_16)
            self.assertTrue(build_video_filter(portrait, CAPS, subtitle_filter=None).startswith("crop="))
            self.assertEqual(resolve_plan(report.plan, ctx, PRO_STREAM_V1).output_size, (1920, 1080))
        finally:
            ws.cleanup()

    def test_render_command_is_argument_array_with_audio_copy(self) -> None:
        from pathlib import Path
        cmd = build_render_command(Path("C:/in dir/a b.mp4"), Path("C:/x/graph.txt"), Path("C:/o/out.mp4"), CAPS)
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[cmd.index("-c:a") + 1], "copy")
        self.assertIn("-filter_script:v", cmd)
        self.assertIn(str(Path("C:/in dir/a b.mp4")), cmd)
        self.assertNotIn("-vf", cmd)  # graph goes through a script file (Windows command-line limit)

    def test_post_render_validation(self) -> None:
        source = fx.media()
        validate_output(source, ExpectedOutput(1920, 1080, source))
        for change in ({"width": 1080}, {"frame_count": source.frame_count - 1}, {"has_audio": False},
                       {"audio_start_s": 0.2}, {"video_duration_s": source.video_duration_s + 0.2}):
            with self.subTest(change=change):
                with self.assertRaises(EditRenderError):
                    validate_output(replace(source, **change), ExpectedOutput(1920, 1080, source))


if __name__ == "__main__":
    unittest.main()
