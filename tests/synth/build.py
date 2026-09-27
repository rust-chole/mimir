"""Render a scenario into a real MP4 VOD + ground-truth JSON (cached by scenario content)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from tests.synth import scene, speech
from tests.synth.scenarios import Scenario

FPS = 30
BUILD_VERSION = 3


def _fingerprint(scenario: Scenario) -> str:
    data = {"v": BUILD_VERSION, "name": scenario.name, "kind": scenario.kind, "duration": scenario.duration,
            "lines": scenario.lines, "events": scenario.events,
            "speakers": {k: [s.voice, s.speed, s.pitch, repr(s.placement)] for k, s in scenario.speakers.items()},
            "extra": scenario.extra}
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[:12]


def ground_truth(scenario: Scenario) -> dict[str, Any]:
    words: list[dict[str, Any]] = []
    lines = []
    for index, (speaker_key, start, text) in enumerate(scenario.lines):
        spec = scenario.speakers[speaker_key]
        rows, end = speech.speak_line(text, spec.voice, spec.speed, spec.pitch, start)
        line_words = []
        for token, a, b, _ in rows:
            words.append({"id": f"g{len(words):04d}", "text": token, "start": a, "end": b, "speaker": speaker_key,
                          "line": index})
            line_words.append(words[-1]["id"])
        lines.append({"index": index, "speaker": speaker_key, "start": start, "end": round(end, 3), "text": text,
                      "words": line_words})
    words.sort(key=lambda w: (w["start"], w["id"]))
    events = [{"index": i, "t": t, "kind": kind} for i, (t, kind) in enumerate(scenario.events)]
    return {"name": scenario.name, "duration": scenario.duration, "words": words, "lines": lines, "events": events}


def build_audio(scenario: Scenario, truth: dict[str, Any]) -> np.ndarray:
    track = np.zeros(int(scenario.duration * speech.RATE), dtype=np.float32)
    rng = np.random.default_rng(7)
    track += rng.standard_normal(len(track)).astype(np.float32) * 0.003
    for speaker_key, start, text in scenario.lines:
        spec = scenario.speakers[speaker_key]
        rows, _ = speech.speak_line(text, spec.voice, spec.speed, spec.pitch, start)
        for _, a, _, samples in rows:
            speech.mix_into(track, samples, a)
    for event in truth["events"]:
        sound = {"impact": speech.impact, "cheer": speech.cheer, "scream": speech.scream,
                 "whoosh": speech.whoosh}[event["kind"]]()
        speech.mix_into(track, sound, event["t"], 0.9)
    return np.clip(track, -1.0, 1.0)


def draw_frame(scenario: Scenario, truth: dict[str, Any], t: float) -> np.ndarray:
    words = truth["words"]
    events = {e["index"]: e["t"] for e in truth["events"]}
    if scenario.kind == "gameplay":
        canvas = scene.gameplay_frame(t, events[scenario.extra["explosion_event"]])
        box = (int(0.80 * scene.W), int(0.66 * scene.H), scene.W - 20, scene.H - 20)
        cam = scene.room_background(3)[box[1]:box[3], box[0]:box[2]].copy()
        spec = scenario.speakers["A"]
        placement = scene.PersonPlacement(spec.placement.variant, 0.5, 0.02, 0.98, spec.placement.phase, 0.3)
        scene.draw_person(cam, placement, t, scene.openness_at(words, "A", t, spec.placement.phase))
        canvas[box[1]:box[3], box[0]:box[2]] = cam
        import cv2

        cv2.rectangle(canvas, (box[0] - 6, box[1] - 6), (box[2] + 6, box[3] + 6), (240, 240, 240), 6)
        return canvas
    if scenario.kind == "ui":
        rows = [f"INV-{1000 + k}   client {k % 7}   2024-03-{k % 28 + 1:02d}   total {137 * k % 9000:>6}.00   paid"
                for k in range(26)]
        return scene.ui_frame(t, events[scenario.extra["error_event"]], rows)
    canvas = scene.room_background(1 if scenario.kind == "talk" else 2).copy()
    if scenario.kind == "irl":
        scene.boxes_frame(canvas, t, events[scenario.extra["fall_event"]])
    for key, spec in scenario.speakers.items():
        if spec.placement is None:
            continue
        dx = 0.0
        if scenario.kind == "irl":
            a, b = scenario.extra["walk"]
            dx = 0.22 * min(1.0, max(0.0, (t - a) / (b - a)))
        scene.draw_person(canvas, spec.placement, t, scene.openness_at(words, key, t, spec.placement.phase), dx)
    return canvas


def build(scenario: Scenario, root: Path) -> tuple[Path, dict[str, Any]]:
    folder = root / f"{scenario.name}_{_fingerprint(scenario)}"
    video = folder / f"{scenario.name}.mp4"
    truth_path = folder / "truth.json"
    if video.is_file() and truth_path.is_file():
        return video, json.loads(truth_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    truth = ground_truth(scenario)
    audio_path = speech.write_wav(folder / "audio.wav", build_audio(scenario, truth))
    frames = int(round(scenario.duration * FPS))
    process = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{scene.W}x{scene.H}", "-r", str(FPS), "-i", "pipe:0", "-i", str(audio_path),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-b:a", "160k", "-shortest", str(video)], stdin=subprocess.PIPE)
    for index in range(frames):
        process.stdin.write(draw_frame(scenario, truth, index / FPS).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"synthetic VOD encode failed for {scenario.name}")
    audio_path.unlink()
    truth_path.write_text(json.dumps(truth, indent=1))
    return video, truth
