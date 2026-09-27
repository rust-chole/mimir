"""Bounded multimodal factual observer for the selected Short (one call, <= N keyframes)."""
from __future__ import annotations

from typing import Any, Sequence

from mimir.models.provider import ImageInput

GRID = ["top-left", "top-center", "top-right", "middle-left", "center", "middle-right", "bottom-left",
        "bottom-center", "bottom-right"]
KINDS = ["person", "face", "object", "action", "screen", "text", "gameplay", "other"]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "frame_id": {"type": "string"},
                    "summary": {"type": "string"},
                    "elements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string", "enum": KINDS},
                                "description": {"type": "string"},
                                "cells": {"type": "array", "items": {"type": "string", "enum": GRID}},
                                "importance": {"type": "string", "enum": ["low", "medium", "high"]},
                            },
                            "required": ["kind", "description", "cells", "importance"],
                        },
                    },
                },
                "required": ["frame_id", "summary", "elements"],
            },
        },
    },
    "required": ["frames"],
}

INSTRUCTIONS = """
You are MIMIR's factual visual observer for one selected Short. For each frame, state what is visibly happening and
list the visible elements that matter for understanding the moment (people, faces, objects being used or affected,
actions, gameplay, on-screen text/UI) with the 3x3 grid cells they occupy. importance=high only for elements the
viewer must see to follow the story at that moment (for example the object being broken, the gameplay action, the
person reacting). Facts only: never judge quality, never identify real people, never invent elements.
""".strip()


def cells_to_box(cells: Sequence[str]) -> list[float] | None:
    boxes = []
    for cell in cells:
        if cell not in GRID:
            continue
        index = GRID.index(cell)
        row, col = divmod(index, 3)
        boxes.append((col / 3, row / 3, (col + 1) / 3, (row + 1) / 3))
    if not boxes:
        return None
    return [round(min(b[0] for b in boxes), 4), round(min(b[1] for b in boxes), 4),
            round(max(b[2] for b in boxes), 4), round(max(b[3] for b in boxes), 4)]


def observe(provider, route, frames: Sequence[tuple[float, bytes]], context: str) -> list[dict[str, Any]]:
    images = [ImageInput(jpeg, f"frame f{index:02d} at source {t:.2f}s") for index, (t, jpeg) in enumerate(frames)]
    result = provider.json_task("visual_observer", route, instructions=INSTRUCTIONS,
                                input_text=f"STORY CONTEXT (for orientation only): {context}\n"
                                           f"Describe frames f00..f{len(frames) - 1:02d}.",
                                schema=SCHEMA, schema_name="mimir_visual_observer_v1", images=images)
    by_id = {f"f{index:02d}": t for index, (t, _) in enumerate(frames)}
    observations = []
    for row in result.get("frames", []):
        t = by_id.get(str(row.get("frame_id", "")))
        if t is None:
            continue
        elements = []
        for element in row.get("elements", []):
            box = cells_to_box(element.get("cells", []))
            if box is None:
                continue
            elements.append({"kind": element.get("kind", "other"),
                             "description": " ".join(str(element.get("description", "")).split())[:160],
                             "box": box, "importance": element.get("importance", "low")})
        observations.append({"t": t, "summary": " ".join(str(row.get("summary", "")).split())[:300],
                             "elements": elements})
    return sorted(observations, key=lambda o: o["t"])
