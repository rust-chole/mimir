"""Batch N1: wire the verified participant name lock into ai/shorts_pipeline.py (stage 7b)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crlf_patch import apply  # noqa: E402

PIPELINE = Path(__file__).resolve().parents[2] / "ai" / "shorts_pipeline.py"

IMPORT_OLD = """    pacing,
    pacing_cutter,
    speaker_caption_support,
"""
IMPORT_NEW = """    pacing,
    pacing_cutter,
    participant_name_lock,
    speaker_caption_support,
"""

LABEL_OLD = """    "captions": "Caption accuracy",
"""
LABEL_NEW = """    "captions": "Caption accuracy",
    "caption_name_lock": "Participant name lock",
"""

CODESIG_OLD = """        captions,
        pacing_cutter,
        caption_renderer,
"""
CODESIG_NEW = """        captions,
        participant_name_lock,
        pacing_cutter,
        caption_renderer,
"""

STAGE_ANCHOR = """        error_message="Caption ASS dosyası oluşmadı.",
    )

    # V14: caption rendering is deterministic FFmpeg work and does not feed
"""
STAGE_NEW = """        error_message="Caption ASS dosyası oluşmadı.",
    )

    # --------------------------------------------------------
    # 7B. VERIFIED PARTICIPANT NAME LOCK (caption truth)
    # --------------------------------------------------------
    # Runs on the identity-resolved final profile, where WHO is known (the V7
    # text-level vocative lock runs before speaker assignment). Only word TEXT
    # may change; word ids, times and speakers are verified unchanged, and the
    # caption ASS is regenerated from the corrected truth. Local and cheap:
    # the paid caption stage above is never re-run for it.
    name_lock_profile = _speaker_profile_path(edited_clip_path, selected_clip_index)
    name_lock_started = time.perf_counter()
    print(f"\\n[7b/{TOTAL_STAGES}] Verified participant name lock")
    print("-" * 60)

    def _name_lock_signature() -> str:
        return _stage_signature(
            "caption_name_lock",
            inputs=[path for path in (name_lock_profile, caption_path, timeline_path, transcript_path) if path],
            modules=[participant_name_lock, captions],
            options={
                "clip_index": selected_clip_index,
                "version": int(getattr(participant_name_lock, "NAME_LOCK_VERSION", 1)),
            },
        )

    def _name_lock_state(audit: dict[str, Any] | None) -> None:
        if isinstance(audit, dict):
            state["caption_name_lock"] = {
                key: audit.get(key)
                for key in ("status", "corrections", "rejected", "new_corrections", "version")
                if key in audit
            }
            book.save()

    if name_lock_profile is None:
        print("⏭️ Final speaker profile yok; participant name lock atlandı.")
        book.record(
            "caption_name_lock",
            "skipped",
            None,
            note="final speaker profile not found",
            elapsed=time.perf_counter() - name_lock_started,
        )
    elif book.reusable(
        "caption_name_lock",
        _name_lock_signature(),
        lambda: _valid_file(caption_path, MIN_CAPTION_BYTES) and _valid_json(name_lock_profile),
        output=caption_path,
    ):
        _stage_skip(name_lock_profile)
        try:
            _name_lock_state((_load_json(name_lock_profile) or {}).get("participant_name_lock"))
        except Exception:
            pass
        book.record(
            "caption_name_lock",
            "skipped",
            _name_lock_signature(),
            path=name_lock_profile,
            elapsed=time.perf_counter() - name_lock_started,
        )
    else:
        try:
            name_lock_audit = participant_name_lock.apply_to_profile_file(name_lock_profile)
            if name_lock_audit.get("changed"):
                caption_path = Path(
                    captions.create_clip_captions(
                        transcript_path=transcript_path,
                        timeline_path=timeline_path,
                        clip_index=selected_clip_index,
                        speaker_profile_path=name_lock_profile,
                    )
                ).resolve()
            name_lock_rows = list(name_lock_audit.get("corrections") or [])
            for row in name_lock_rows:
                print(
                    f"🔤 Name lock: {row.get('from')} → {row.get('to')} "
                    f"(participant {row.get('participant')}; {', '.join(row.get('evidence') or [])})"
                )
            if not name_lock_rows:
                print(f"🔤 Name lock: düzeltme yok ({name_lock_audit.get('status')})")
            _name_lock_state(name_lock_audit)
            _stage_done(name_lock_profile)
            book.record(
                "caption_name_lock",
                "done",
                _name_lock_signature(),
                path=name_lock_profile,
                note=f"{len(name_lock_rows)} correction(s); status={name_lock_audit.get('status')}",
                elapsed=time.perf_counter() - name_lock_started,
            )
        except Exception as error:
            # Fail closed: the unmodified caption truth (and ASS) stay in use.
            book.warn(
                "Participant name lock uygulanamadı; ASR yazımı korundu. "
                f"Detay: {type(error).__name__}: {error}"
            )
            book.record(
                "caption_name_lock",
                "fallback",
                None,
                note=f"{type(error).__name__}: {error}",
                elapsed=time.perf_counter() - name_lock_started,
            )

    # V14: caption rendering is deterministic FFmpeg work and does not feed
"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    apply(PIPELINE, [(IMPORT_OLD, IMPORT_NEW), (LABEL_OLD, LABEL_NEW), (CODESIG_OLD, CODESIG_NEW),
                     (STAGE_ANCHOR, STAGE_NEW)])
    print("N1 pipeline: participant name lock stage 7b wired (import, label, code signature, stage)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
