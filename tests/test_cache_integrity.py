"""Cache integrity: content identity survives a move, edits never slip through, legacy records upgrade once."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai import shorts_pipeline as sp


class CacheIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="mimir cache ş ")
        self.root = Path(self._tmp.name)
        patcher = mock.patch.object(sp, "STATE_DIR", self.root / "state")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        sp._HASH_CACHE = None
        self.addCleanup(setattr, sp, "_HASH_CACHE", None)

    def file(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def book(self) -> sp.StageBook:
        return sp.StageBook(self.root / "state" / "run.json", {"stages": {}}, force=False)

    def test_a_moved_byte_identical_input_keeps_every_signature(self) -> None:
        original = self.file("project a/vod_output/transcript.json", b'{"words": []}')
        moved = self.file("project b/vod_output/transcript.json", b'{"words": []}')
        os.utime(moved, (1, 1))                                           # different mtime too
        sig = sp._stage_signature("clip_analysis", inputs=[original], options={"m": 1})
        self.assertEqual(sig, sp._stage_signature("clip_analysis", inputs=[moved], options={"m": 1}))
        moved.write_bytes(b'{"words": [1]}')
        self.assertNotEqual(sig, sp._stage_signature("clip_analysis", inputs=[moved], options={"m": 1}))

    def test_a_legacy_record_is_accepted_once_then_upgraded(self) -> None:
        source = self.file("in.json", b"x")
        output = self.file("out.json", b"y")
        sig = sp._stage_signature("timeline", inputs=[source])
        book = self.book()
        book.state["stages"]["timeline"] = {"status": "done", "signature": sp._LEGACY_SIGNATURES[sig],
                                            "path": str(output)}
        self.assertTrue(book.reusable("timeline", sig, lambda: True, output=output))
        self.assertEqual(book.state["stages"]["timeline"]["signature"], sig)
        book.state["stages"]["timeline"]["signature"] = "something else"
        self.assertFalse(book.reusable("timeline", sig, lambda: True, output=output))

    def test_an_edited_critical_artifact_is_never_silently_reused(self) -> None:
        output = self.file("clip_edited.mp4", b"paced clip bytes")
        book = self.book()
        book.record("pacing_cut", "done", "sig", path=output)
        self.assertEqual(book.state["stages"]["pacing_cut"]["sha256"], sp._content_digest(output))
        self.assertTrue(book.reusable("pacing_cut", "sig", lambda: True, output=output))
        output.write_bytes(b"edited by hand!!")
        self.assertFalse(book.reusable("pacing_cut", "sig", lambda: True, output=output))
        self.assertTrue(any("sha256" in w for w in book.state["warnings"]))

    def test_non_critical_outputs_are_not_hashed(self) -> None:
        book = self.book()
        book.record("speaker_preflight", "done", "sig", path=self.file("scan.json", b"{}"))
        self.assertNotIn("sha256", book.state["stages"]["speaker_preflight"])

    def test_source_identity_is_content_not_location(self) -> None:
        a = self.file("a/vod.mp4", b"video bytes")
        b = self.file("elsewhere/vod.mp4", b"video bytes")
        self.assertTrue(sp._same_source({"source": sp._source_fingerprint(a)}, sp._source_fingerprint(b)))
        b.write_bytes(b"other video")
        self.assertFalse(sp._same_source({"source": sp._source_fingerprint(a)}, sp._source_fingerprint(b)))
        legacy = {k: v for k, v in sp._source_fingerprint(a).items() if k != "sha256"}
        self.assertTrue(sp._same_source({"source": legacy}, sp._source_fingerprint(a)))

    def test_fast_resume_refuses_a_replaced_final(self) -> None:
        final = self.file("final/x_short.mp4", b"0" * (sp.MIN_VIDEO_BYTES + 10))
        state = {"run_status": "success", "request_signature": "r", "final_output": str(final),
                 "stages": {"publish": {"status": "done", "sha256": sp._content_digest(final)}}}
        self.assertIsNotNone(sp._fast_resume(state, "r", self.root / "state.json"))
        final.write_bytes(b"1" * (sp.MIN_VIDEO_BYTES + 10))
        self.assertIsNone(sp._fast_resume(state, "r", self.root / "state.json"))

    def test_digests_are_cached_by_path_size_and_mtime(self) -> None:
        path = self.file("big.bin", b"a" * 1024)
        first = sp._content_digest(path)
        cache = json.loads((self.root / "state" / "content_hashes.json").read_text(encoding="utf-8"))
        self.assertIn(first, cache.values())
        with mock.patch.object(Path, "open", side_effect=AssertionError("re-hashed an unchanged file")):
            self.assertEqual(sp._content_digest(path), first)


if __name__ == "__main__":
    unittest.main()
