"""Action-region restraint: the highest-motion corner is not automatically the action; gameplay/screen owns the frame."""
from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace

import pro_edit_fixtures as fx
from ai.editor.pro_edit import caption_action as ca
from ai.editor.pro_edit import direction
from ai.editor.pro_edit.caption_occupancy import VisualOccupancyMap
from ai.editor.pro_edit.subjects import SubjectSample, SubjectTrack

COLS = ROWS = 8


def occupancy(active: dict[tuple[int, int], float], *, moving_samples: int = 10, total: int = 10) -> VisualOccupancyMap:
    """Samples at 1 fps over [0, total); ``active`` cells move in the first ``moving_samples`` samples."""
    cells = []
    for index in range(total):
        frame = bytearray(COLS * ROWS)
        if index < moving_samples:
            for (row, col), value in active.items():
                frame[row * COLS + col] = int(value * 255)
        cells.append(bytes(frame))
    return VisualOccupancyMap(COLS, ROWS, 1.0, tuple(float(i) for i in range(total)), tuple(cells), 0, "fp")


def span(start: float = 0.0, end: float = 9.0, role: str = "payoff"):
    return SimpleNamespace(span_id=f"{role}_01", start=start, end=end, role=SimpleNamespace(value=role))


class HotspotClassificationTests(unittest.TestCase):
    CORNER = {(0, 7): 0.9}                                       # one small cell, top-right corner
    CENTER = {(3, 3): 0.9, (3, 4): 0.9, (4, 3): 0.9, (4, 4): 0.9}

    def classify(self, active, **kwargs):
        occ = occupancy(active, **{k: kwargs.pop(k) for k in ("moving_samples",) if k in kwargs})
        box, _mean = ca.activity_hotspot(occ, 0.0, 9.0)
        return ca.classify_hotspot(box, occupancy=occ, start=0.0, end=9.0, **kwargs)[0], box

    def test_persistent_central_motion_is_an_action(self) -> None:
        self.assertEqual(self.classify(self.CENTER)[0], "action")

    def test_motion_inside_persistent_ui_text_is_ui_not_action(self) -> None:
        status, box = self.classify(self.CORNER, ui_boxes=[(0.75, 0.0, 1.0, 0.25)])      # e.g. a chat overlay
        self.assertEqual(status, "ui_motion")

    def test_small_corner_motion_without_support_is_ambiguous(self) -> None:
        self.assertEqual(self.classify(self.CORNER)[0], "ambiguous")

    def test_transient_motion_is_ambiguous(self) -> None:
        burst = {cell: 1.0 for cell in self.CENTER}                     # strong enough to be a hotspot ...
        self.assertEqual(self.classify(burst, moving_samples=3)[0], "ambiguous")   # ... but moves in 3 of 10

    def test_motion_of_a_tracked_person_is_that_person(self) -> None:
        face = (0.40, 0.28, 0.55, 0.40)                                                   # body region covers center
        self.assertEqual(self.classify(self.CENTER, faces=[face])[0], "person_motion")

    def test_derive_keeps_every_hotspot_as_evidence_with_its_status(self) -> None:
        occ = occupancy(self.CORNER)
        rows = ca.derive_action_regions([span(), span(role="setup")], occ)
        self.assertEqual([(r.story_span_id, r.status) for r in rows], [("payoff_01", "ambiguous")])
        self.assertLess(rows[0].confidence, 0.9)                                           # demoted, not deleted


class LayoutPriorityTests(unittest.TestCase):
    def context(self, layout):
        workspace = fx.Workspace()
        self.addCleanup(workspace.cleanup)
        return dataclasses.replace(fx.make_context(workspace), director_evidence={"layout": layout})

    def test_measured_gameplay_layout_owns_the_frame(self) -> None:
        mode, reason = direction.screen_priority(self.context({"class": "FULLSCREEN_GAMEPLAY", "confidence": 0.8}))
        self.assertEqual(mode, "GAMEPLAY_PRIORITY")
        self.assertIn("measured layout", reason)
        self.assertEqual(direction.screen_priority(self.context({"class": "SCREEN_SHARE", "confidence": 0.7}))[0],
                         "SCREEN_PRIORITY")

    def test_an_unsure_or_talking_layout_keeps_subject_direction(self) -> None:
        self.assertEqual(direction.screen_priority(self.context({"class": "FULLSCREEN_GAMEPLAY",
                                                                 "confidence": 0.4}))[0], "subjects")
        self.assertEqual(direction.screen_priority(self.context({"class": "TALKING_HEAD", "confidence": 0.9}))[0],
                         "subjects")

    def test_director_evidence_never_offers_ambiguous_motion_as_action(self) -> None:
        from ai.editor.pro_edit import stage

        workspace = fx.Workspace()
        self.addCleanup(workspace.cleanup)
        context = fx.make_context(workspace)
        face = SubjectTrack("face_1", "face", (SubjectSample(1.0, 0.5, 0.4, 0.1, 0.15, 0.9),))
        context = dataclasses.replace(context, subject_tracks=(face,))
        request = SimpleNamespace(config=SimpleNamespace(caption_activity=True, caption_ui=False,
                                                         caption_layout=False), force=False)
        occ = occupancy({(0, 7): 0.9}, total=int(context.clip.duration_s))
        original = stage.load_or_analyze
        stage.load_or_analyze = lambda *a, **k: (occ, "cached")
        self.addCleanup(setattr, stage, "load_or_analyze", original)
        evidence = stage.director_evidence(request, SimpleNamespace(artifacts=SimpleNamespace(caption_occupancy="x")),
                                           context, fx.media())
        self.assertEqual(evidence["action_regions"], [])
        if evidence["ambiguous_motion"]:
            self.assertEqual(evidence["ambiguous_motion"][0]["note"], "not a framing target; keep the context wide")


if __name__ == "__main__":
    unittest.main()
