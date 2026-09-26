"""Optional bounded multimodal final reviewer (one call; advisory, mapped to deterministic repairs)."""
from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from mimir.models.provider import ImageInput

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail"]},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["cropped_subject", "missing_action", "caption_error",
                                                        "confusing_story", "weak_cold_open", "other"]},
                    "frame_id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "description": {"type": "string"},
                },
                "required": ["type", "frame_id", "severity", "description"],
            },
        },
    },
    "required": ["verdict", "issues"],
}

INSTRUCTIONS = """
You are the final reviewer of a vertical Short. You see frames from the SOURCE story and the matching frames of the
RENDERED Short, plus the story beats. Report only concrete, visible problems: a story-critical person or action cut
out of the frame, a face cut in half, captions covering what matters or wrong, a cold open that does not show the
peak. Do not rate taste. verdict=fail only for a high-severity problem.
""".strip()


def _jpeg(image: np.ndarray, width: int) -> bytes:
    h, w = image.shape[:2]
    image = cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def review(provider, route, pairs: Sequence[tuple[int, np.ndarray, np.ndarray]], story: dict[str, Any]
           ) -> dict[str, Any]:
    images = []
    for index, source, rendered in pairs:
        images.append(ImageInput(_jpeg(source, 480), f"f{index} SOURCE"))
        images.append(ImageInput(_jpeg(rendered, 270), f"f{index} RENDERED"))
    beats = "; ".join(f"{b['role']}: {b.get('note', '')}" for b in story["beats"])
    return provider.json_task("final_reviewer", route, instructions=INSTRUCTIONS,
                              input_text=f"STORY: {story['title']}\nBEATS: {beats}", schema=SCHEMA,
                              schema_name="mimir_final_review_v1", images=images)
