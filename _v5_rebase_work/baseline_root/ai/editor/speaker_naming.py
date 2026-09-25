from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PREVIEW_DIR = PROJECT_ROOT / "vod_output" / "speaker_previews"

# Human identity calibration, not a decorative teaser.
# OpenAI known-speaker references must be 2-10 seconds. We deliberately stay
# comfortably inside that contract and use clean 48 kHz mono PCM.
REFERENCE_TARGET_SECONDS = 3.40
REFERENCE_MIN_SECONDS = 2.20
REFERENCE_MAX_SECONDS = 4.80
REFERENCE_GAP_SECONDS = 0.08
VERIFY_TARGET_SECONDS = 1.80
VERIFY_MIN_SECONDS = 0.95
VERIFY_MAX_SECONDS = 2.60
MANUAL_PREVIEW_TARGET_SECONDS = 1.25
MANUAL_PREVIEW_MIN_SECONDS = 0.65
MANUAL_PREVIEW_MAX_SECONDS = 1.75
MANUAL_PREVIEW_MIN_SEGMENT_SECONDS = 0.28
MANUAL_PREVIEW_BOUNDARY_TRIM_SECONDS = 0.06
PREVIEW_SAMPLE_RATE = 48000
BOUNDARY_TRIM_SECONDS = 0.16
MIN_SEGMENT_SECONDS = 0.95
IDENTITY_PROMPT_MIN_CONFIDENCE = 0.90
IDENTITY_PROMPT_SECONDARY_MIN_CONFIDENCE = 0.82
IDENTITY_PROMPT_MIN_BOUNDARY_QUALITY = 0.68


def _console_print(text: str = "") -> None:
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    stream.write(str(text) + "\n")
    stream.flush()


def _console_input(prompt: str) -> str:
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    stream.write(prompt)
    stream.flush()
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _interactive() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except Exception:
        return False


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)
    return path


def _safe_name(value: str) -> str:
    text = str(value).strip()
    for char in '<>:"/\\|?*':
        text = text.replace(char, "_")
    return " ".join(text.split()).strip(" ._") or "clip"


def _clean_display_name(value: str, fallback: str = "X") -> str:
    text = " ".join(str(value).replace("\n", " ").replace("\r", " ").split()).strip()
    text = text.replace("{", "(").replace("}", ")").replace("\\", "/")
    if not text:
        text = fallback
    return text[:32].strip() or fallback


def _ordered_identity_speakers(profile: dict[str, Any]) -> list[str]:
    """Return up to three plausible HUMAN speakers for the naming checkpoint.

    V7 originally returned speakers only when the automatic classifier was
    already confident enough to call the clip ``dual`` or ``triple``. That is
    backwards for a human checkpoint: the cases where the model is unsure are
    exactly the cases where asking the user is useful.

    This helper does *not* change diarization, words or timestamps. It only
    chooses which existing raw speaker clusters may be previewed to the user.
    Noisy/tiny/background clusters are filtered conservatively.
    """
    mode = str(profile.get("mode", "")).strip().casefold()
    background = {
        str(value or "").strip()
        for value in (profile.get("background_speakers", []) or [])
        if str(value or "").strip()
    }

    stats_by_speaker: dict[str, dict[str, Any]] = {}
    for row in profile.get("speaker_stats", []) or []:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("speaker", "")).strip()
        if raw:
            stats_by_speaker[raw] = row

    def _strong_enough(raw: str) -> bool:
        if not raw or raw in background:
            return False
        row = stats_by_speaker.get(raw)
        if not row:
            # Older profiles may not have speaker_stats. Preserve compatibility.
            return True
        try:
            seconds = float(row.get("speaking_seconds", 0.0) or 0.0)
            words = int(row.get("word_count", 0) or 0)
            share = float(row.get("share", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        return seconds >= 0.70 and words >= 2 and share >= 0.045

    ordered: list[str] = []
    for value in (
        profile.get("primary_speaker"),
        profile.get("secondary_speaker"),
        *(profile.get("participant_speakers", []) or []),
    ):
        raw = str(value or "").strip()
        if raw and raw not in ordered and _strong_enough(raw):
            ordered.append(raw)

    # When the automatic mode is unresolved, participant_speakers can be in a
    # noisy order. Prefer the speakers with the most real speech.
    if mode not in {"dual", "triple"} and stats_by_speaker:
        ranked = sorted(
            (raw for raw in stats_by_speaker if _strong_enough(raw)),
            key=lambda raw: (
                float(stats_by_speaker[raw].get("speaking_seconds", 0.0) or 0.0),
                int(stats_by_speaker[raw].get("word_count", 0) or 0),
            ),
            reverse=True,
        )
        for raw in ranked:
            if raw not in ordered:
                ordered.append(raw)

    if mode == "dual":
        wanted = 2
    elif mode == "triple":
        wanted = 3
    else:
        # Ambiguous multi-talk: asking two or at most three useful voices is
        # better than silently skipping the checkpoint or asking about every
        # tiny crowd/noise cluster.
        wanted = min(3, len(ordered))

    return ordered[:wanted] if wanted >= 2 and len(ordered) >= 2 else []


def _candidate_segments(profile: dict[str, Any], raw_speaker: str) -> list[dict[str, Any]]:
    """Rank clean same-speaker regions; aggressively penalize turn boundaries."""
    background = {str(value) for value in profile.get("background_speakers", []) or []}
    if raw_speaker in background:
        return []

    segments = [item for item in profile.get("segments", []) or [] if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if str(segment.get("speaker", "")) != str(raw_speaker):
            continue
        text = " ".join(str(segment.get("text", "")).split()).strip()
        try:
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        word_count = int(segment.get("word_count", len(text.split())) or 0)
        if word_count < 2 or end - start < MIN_SEGMENT_SECONDS:
            continue

        previous = segments[index - 1] if index > 0 else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        left_margin = 9.0
        right_margin = 9.0
        if previous and str(previous.get("speaker", "")) != raw_speaker:
            left_margin = max(0.0, start - float(previous.get("end", start) or start))
            start += BOUNDARY_TRIM_SECONDS
        if following and str(following.get("speaker", "")) != raw_speaker:
            right_margin = max(0.0, float(following.get("start", end) or end) - end)
            end -= BOUNDARY_TRIM_SECONDS

        duration = max(0.0, end - start)
        if duration < 0.78:
            continue
        boundary_penalty = 0.0
        if left_margin < 0.12:
            boundary_penalty += 1.1
        elif left_margin < 0.25:
            boundary_penalty += 0.45
        if right_margin < 0.12:
            boundary_penalty += 1.1
        elif right_margin < 0.25:
            boundary_penalty += 0.45
        score = min(duration, 4.0) * 2.2 + min(word_count, 12) * 0.28 - boundary_penalty
        if text.endswith((".", "!", "?")):
            score += 0.30
        candidates.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "word_count": word_count,
            "score": round(score, 4),
            "text": text[:180],
        })
    candidates.sort(key=lambda row: (float(row["score"]), float(row["duration"])), reverse=True)
    return candidates


def _candidate_manual_preview_segments(
    profile: dict[str, Any], raw_speaker: str
) -> list[dict[str, Any]]:
    """Short, interior-only utterances for HUMAN listening only.

    These may be shorter than OpenAI's known-speaker reference contract. They
    are never sent as a 2-10s biometric anchor unless the final combined WAV
    independently satisfies that contract.
    """
    segments = [item for item in profile.get("segments", []) or [] if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if str(segment.get("speaker", "")) != str(raw_speaker):
            continue
        text = " ".join(str(segment.get("text", "")).split()).strip()
        try:
            start = float(segment.get("start", 0.0) or 0.0)
            end = float(segment.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        words = max(1, int(segment.get("word_count", len(text.split())) or 0))
        if end - start < MANUAL_PREVIEW_MIN_SEGMENT_SECONDS:
            continue

        previous = segments[index - 1] if index > 0 else None
        following = segments[index + 1] if index + 1 < len(segments) else None
        if previous and str(previous.get("speaker", "")) != raw_speaker:
            start += MANUAL_PREVIEW_BOUNDARY_TRIM_SECONDS
        if following and str(following.get("speaker", "")) != raw_speaker:
            end -= MANUAL_PREVIEW_BOUNDARY_TRIM_SECONDS
        duration = max(0.0, end - start)
        if duration < MANUAL_PREVIEW_MIN_SEGMENT_SECONDS:
            continue
        # Prefer longer multi-word interiors, but allow short answers such as
        # "Amy" / "Sixteen" to be concatenated for the human A/B preview.
        score = min(duration, 2.0) * 2.0 + min(words, 8) * 0.20
        rows.append({
            "start": round(start, 3), "end": round(end, 3),
            "duration": round(duration, 3), "word_count": words,
            "score": round(score, 4), "text": text[:180],
        })
    rows.sort(key=lambda row: (float(row["score"]), float(row["duration"])), reverse=True)
    return rows


def _overlaps(a: tuple[float, float], b: tuple[float, float], pad: float = 0.20) -> bool:
    return min(a[1] + pad, b[1] + pad) > max(a[0] - pad, b[0] - pad)


def _take_ranges(
    candidates: list[dict[str, Any]],
    *,
    target: float,
    minimum: float,
    maximum: float,
    exclude: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    exclude = list(exclude or [])
    picked: list[tuple[float, float]] = []
    total = 0.0
    for row in candidates:
        start = float(row["start"])
        end = float(row["end"])
        if any(_overlaps((start, end), blocked) for blocked in exclude):
            continue
        remaining = min(maximum - total, target - total if total < target else 0.0)
        if remaining <= 0.04:
            break
        duration = end - start
        take = min(duration, remaining)
        if take < 0.55:
            continue
        # Prefer the middle of a long segment, away from speaker-turn boundaries.
        if duration > take + 0.04:
            center = (start + end) / 2.0
            start = center - take / 2.0
            end = center + take / 2.0
        picked.append((round(start, 3), round(end, 3)))
        total += end - start
        exclude.append((start, end))
        if total >= target - 0.02:
            break
    return picked if total + 1e-6 >= minimum else []


def _audio_source(profile: dict[str, Any], edited_clip_path: str | Path) -> Path | None:
    raw_audio = str(profile.get("speaker_audio_path") or "").strip()
    if raw_audio:
        candidate = Path(raw_audio).resolve()
        if candidate.is_file():
            return candidate
    candidate = Path(edited_clip_path).resolve()
    return candidate if candidate.is_file() else None


def _render_ranges_wav(source: Path, output: Path, ranges: list[tuple[float, float]]) -> bool:
    if not ranges:
        return False
    chains: list[str] = []
    concat_inputs: list[str] = []
    for idx, (start, end) in enumerate(ranges):
        tag = f"v{idx}"
        chains.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
            f"aresample={PREVIEW_SAMPLE_RATE},aformat=sample_fmts=s16:channel_layouts=mono[{tag}]"
        )
        concat_inputs.append(f"[{tag}]")
        if idx + 1 < len(ranges):
            gap = f"g{idx}"
            chains.append(
                f"anullsrc=r={PREVIEW_SAMPLE_RATE}:cl=mono:d={REFERENCE_GAP_SECONDS:.3f}[{gap}]"
            )
            concat_inputs.append(f"[{gap}]")
    chains.append("".join(concat_inputs) + f"concat=n={len(concat_inputs)}:v=0:a=1[aout]")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-filter_complex", ";".join(chains),
        "-map", "[aout]", "-vn", "-c:a", "pcm_s16le",
        "-ar", str(PREVIEW_SAMPLE_RATE), "-ac", "1", str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False
    )
    return completed.returncode == 0 and output.is_file() and output.stat().st_size >= 4096


def _build_identity_assets(
    *,
    profile: dict[str, Any],
    edited_clip_path: str | Path,
    clip_index: int,
) -> dict[str, dict[str, Any]]:
    source = _audio_source(profile, edited_clip_path)
    if source is None:
        return {}
    speakers = _ordered_identity_speakers(profile)
    out_dir = PREVIEW_DIR / _safe_name(Path(edited_clip_path).resolve().parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(speakers):
        label = chr(ord("A") + index)
        candidates = _candidate_segments(profile, raw)
        reference_ranges = _take_ranges(
            candidates,
            target=REFERENCE_TARGET_SECONDS,
            minimum=REFERENCE_MIN_SECONDS,
            maximum=REFERENCE_MAX_SECONDS,
        )
        reference_contract = "openai_known_speaker_2_10s"
        known_speaker_reference_eligible = bool(reference_ranges)
        preview_candidates = candidates
        if not reference_ranges:
            preview_candidates = _candidate_manual_preview_segments(profile, raw)
            reference_ranges = _take_ranges(
                preview_candidates,
                target=MANUAL_PREVIEW_TARGET_SECONDS,
                minimum=MANUAL_PREVIEW_MIN_SECONDS,
                maximum=MANUAL_PREVIEW_MAX_SECONDS,
            )
            reference_contract = "human_preview_short_audio"
            known_speaker_reference_eligible = False
        if not reference_ranges:
            continue

        verification_ranges = _take_ranges(
            preview_candidates,
            target=VERIFY_TARGET_SECONDS if known_speaker_reference_eligible else 0.75,
            minimum=VERIFY_MIN_SECONDS if known_speaker_reference_eligible else 0.35,
            maximum=VERIFY_MAX_SECONDS if known_speaker_reference_eligible else 1.20,
            exclude=reference_ranges,
        )
        ref_path = out_dir / f"clip_{int(clip_index):02d}_speaker_{label}_reference.wav"
        verify_path = out_dir / f"clip_{int(clip_index):02d}_speaker_{label}_verify.wav"
        if not _render_ranges_wav(source, ref_path, reference_ranges):
            continue
        verify_ok = bool(verification_ranges) and _render_ranges_wav(source, verify_path, verification_ranges)
        assets[label] = {
            "label": label,
            "raw_speaker": raw,
            "reference_contract": reference_contract,
            "known_speaker_reference_eligible": known_speaker_reference_eligible,
            "reference_path": str(ref_path.resolve()),
            "reference_ranges": [[round(a, 3), round(b, 3)] for a, b in reference_ranges],
            "reference_duration": round(
                sum(b - a for a, b in reference_ranges)
                + REFERENCE_GAP_SECONDS * max(0, len(reference_ranges) - 1), 3
            ),
            "verification_path": str(verify_path.resolve()) if verify_ok else "",
            "verification_ranges": [[round(a, 3), round(b, 3)] for a, b in verification_ranges] if verify_ok else [],
            "verification_available": bool(verify_ok),
        }
    return assets


def _open_preview(path: Path) -> None:
    if os.name != "nt":
        return
    try:
        os.startfile(str(path))  # type: ignore[attr-defined]
    except Exception:
        pass


def _mark_plain_captions(profile: dict[str, Any], source: str) -> dict[str, Any]:
    profile["display_labels"] = {}
    profile["speaker_names"] = {"A": "", "B": "", "C": "", "source": source}
    profile["force_plain_captions"] = True
    profile["identity_calibration"] = {}
    profile["speaker_preview_path"] = ""
    profile["speaker_preview_media"] = "none"
    return profile


def _mark_unlabeled_speaker_structure(profile: dict[str, Any], source: str) -> dict[str, Any]:
    """Keep stable raw A/B turns but print no real names.

    This is used only after speaker count AND boundary quality passed the prompt
    gate. Missing identity references must not collapse a trustworthy dual-turn
    structure into one generic lane.
    """
    profile["display_labels"] = {}
    profile["speaker_names"] = {"A": "", "B": "", "C": "", "source": source}
    profile["force_plain_captions"] = False
    profile["identity_calibration"] = {}
    profile["speaker_preview_path"] = ""
    profile["speaker_preview_media"] = "none"
    return profile


def _apply_calibrated_identities(
    profile: dict[str, Any],
    *,
    assets: dict[str, dict[str, Any]],
    names: dict[str, str],
) -> dict[str, Any]:
    raw_to_display: dict[str, str] = {}
    calibration: dict[str, Any] = {}
    for label, name in names.items():
        asset = assets.get(label) or {}
        raw = str(asset.get("raw_speaker", "")).strip()
        if not raw or not name:
            continue
        raw_to_display[raw] = name
        calibration[label] = {
            **asset,
            "name": name,
            "human_verified": True,
            "source": "manual_voice_calibrated_v28",
        }

    for raw in profile.get("words", []) or []:
        if not isinstance(raw, dict):
            continue
        speaker = str(raw.get("speaker_raw", ""))
        if speaker in raw_to_display:
            raw["speaker_label"] = raw_to_display[speaker]

    profile["display_labels"] = raw_to_display
    profile["speaker_names"] = {
        "A": names.get("A", ""),
        "B": names.get("B", ""),
        "C": names.get("C", ""),
        "source": "manual_voice_calibrated_v28",
    }
    profile["identity_calibration"] = calibration
    profile["force_plain_captions"] = not bool(raw_to_display)
    profile["speaker_preview_media"] = "separate_48k_wav_reference_plus_heldout_verify"
    profile["speaker_preview_order"] = sorted(assets)
    profile["speaker_preview_path"] = next(
        (str((assets[k] or {}).get("reference_path", "")) for k in sorted(assets) if (assets[k] or {}).get("reference_path")),
        "",
    )
    return profile


def _ask_name(label: str, asset: dict[str, Any], used_names: set[str]) -> str:
    reference = Path(str(asset["reference_path"])).resolve()
    _console_print()
    _console_print(f"🎧 SPEAKER {label} — temiz kimlik referansı açılıyor ({asset.get('reference_duration', '?')}s)")
    _console_print(f"   {reference}")
    _open_preview(reference)

    while True:
        raw = _console_input(f"   Bu kişi kim? (bilmiyorsan boş bırak): ")
        if not raw:
            return ""
        name = _clean_display_name(raw, fallback=label)
        folded = name.casefold()
        if folded in used_names:
            _console_print(f"   ⚠️ '{name}' başka bir speaker'a zaten verildi. Tekrar gir veya boş bırak.")
            continue
        break

    verify_raw = str(asset.get("verification_path", "")).strip()
    if verify_raw:
        verify = Path(verify_raw).resolve()
        while True:
            _console_print(f"   🔎 Şimdi AYNI cluster'dan, ilk referansta kullanılmayan başka bir cümle açılıyor:")
            _console_print(f"   {verify}")
            _open_preview(verify)
            answer = _console_input(
                f"   Bu ses de {name} mi? [Enter=evet / N=hayır / R=tekrar / X=isimsiz bırak]: "
            ).strip().casefold()
            if answer in {"", "e", "evet", "y", "yes"}:
                return name
            if answer in {"r", "tekrar", "replay"}:
                continue
            if answer in {"n", "hayır", "hayir", "no", "x"}:
                _console_print("   ↪ Bu speaker için isim kullanılmayacak.")
                return ""
            _console_print("   Enter, N, R veya X kullan.")
    else:
        answer = _console_input(
            f"   İkinci bağımsız örnek bulunamadı. '{name}' isminden eminsen Enter; isimsiz bırakmak için X: "
        ).strip().casefold()
        return "" if answer in {"x", "n", "hayır", "hayir", "no"} else name


def resolve_interactive_speaker_names(
    speaker_profile_path: str | Path,
    edited_clip_path: str | Path,
    clip_index: int,
    creator_name: str | None = None,
) -> dict[str, Any]:
    """Human-calibrated speaker identity checkpoint.

    Rules:
    - one plausible speaker -> no question, plain captions
    - two/three plausible speakers -> ask, especially when automatic identity is uncertain
    - each entered name is verified on a DISJOINT held-out utterance when available
    - partial identity is allowed (e.g. KAI known, second speaker unlabeled)
    - human naming may become a biometric anchor for final known-speaker diarization
    - caption text/timestamps are never touched here
    """
    path = Path(speaker_profile_path).resolve()
    profile = _load_json(path)
    if str(profile.get("status", "")) != "ok":
        return {"profile_path": path, "mode": "fallback", "preview_path": None, "labels": {}}

    mode = str(profile.get("mode", "single"))
    diarization_ok = str(profile.get("diarization_status", "")) == "ok"
    speakers = _ordered_identity_speakers(profile)
    role = profile.get("role_judge", {}) if isinstance(profile.get("role_judge"), dict) else {}
    role_confidence = float(role.get("confidence", 0.0) or 0.0)
    count_confidence = float(profile.get("speaker_count_confidence", 0.0) or 0.0)
    identity_signals = [count_confidence]
    if role:
        identity_signals.append(role_confidence)
    identity_confidence = max(identity_signals)
    identity_floor = min(identity_signals)
    boundary_quality = float(profile.get("speaker_boundary_quality", 0.0) or 0.0)

    # A true single-speaker clip should stay fully automatic. Do not trust the
    # mode string alone, though: low-confidence multi-talk can be labelled
    # "single"/"unresolved" by the automatic classifier. If we have two
    # substantial raw speaker clusters, the human checkpoint is useful.
    if len(speakers) < 2:
        source = (
            "single_auto_plain_no_prompt"
            if mode == "single"
            else "no_two_plausible_speakers_plain"
        )
        profile = _mark_plain_captions(profile, source)
        _write_json(path, profile)
        return {"profile_path": path, "mode": mode, "preview_path": None, "labels": {}}

    # Surgical V7.2 policy fix:
    # Confidence is diagnostic here, not a reason to suppress the HUMAN prompt.
    # If diarization produced 2/3 substantial clusters, ask the user even when
    # count/role/boundary confidence is low. The held-out confirmation inside
    # _ask_name() remains the safety net against mixed/bad clusters.
    eligible = diarization_ok and len(speakers) in {2, 3}
    if not eligible:
        profile = _mark_plain_captions(profile, "speaker_preview_not_safe_plain")
        _write_json(path, profile)
        return {"profile_path": path, "mode": mode, "preview_path": None, "labels": {}}

    prompt_reason = (
        "confident_multispeaker"
        if (
            identity_confidence >= IDENTITY_PROMPT_MIN_CONFIDENCE
            and identity_floor >= IDENTITY_PROMPT_SECONDARY_MIN_CONFIDENCE
            and boundary_quality >= IDENTITY_PROMPT_MIN_BOUNDARY_QUALITY
        )
        else "ambiguous_multispeaker_human_review"
    )

    if not _interactive():
        profile = _mark_plain_captions(profile, "noninteractive_plain_no_prompt")
        _write_json(path, profile)
        return {"profile_path": path, "mode": mode, "preview_path": None, "labels": {}}

    profile["speaker_identity_prompt"] = {
        "reason": prompt_reason,
        "candidate_speakers": list(speakers),
        "identity_confidence": round(identity_confidence, 4),
        "identity_floor": round(identity_floor, 4),
        "boundary_quality": round(boundary_quality, 4),
    }
    _write_json(path, profile)

    assets = _build_identity_assets(
        profile=profile,
        edited_clip_path=edited_clip_path,
        clip_index=clip_index,
    )
    if not assets:
        # Boundary separation was already good enough to reach this checkpoint.
        # Keep raw A/B structure even when a 2s+ known-speaker reference cannot
        # be built; simply omit real-name labels.
        profile = _mark_unlabeled_speaker_structure(
            profile, "identity_reference_unavailable_unlabeled"
        )
        _write_json(path, profile)
        return {"profile_path": path, "mode": mode, "preview_path": None, "labels": {}}

    _console_print()
    if prompt_reason == "ambiguous_multispeaker_human_review":
        _console_print(
            f"🎙️ MIMIR birden fazla konuşmacı duydu ({len(speakers)} aday) ama kimlikte tam emin değil — senden doğrulama istiyor."
        )
    else:
        _console_print(f"🎙️ MIMIR {len(speakers)} gerçek konuşmacıyı ayırdı — kimlik kalibrasyonu başlıyor.")
    _console_print("   Her speaker AYRI dinletilecek. İsim verirsen ikinci, farklı cümleyle doğrulanacak.")
    _console_print("   Emin olmadığın kişiyi boş bırakabilirsin; bildiğimiz diğer isim yine kullanılacak.")
    _console_print("   Bu aşama yalnız KİM konuşuyor bilgisini kilitler; caption zamanlarına dokunmaz.")

    names: dict[str, str] = {}
    used_names: set[str] = set()
    for index, raw_speaker in enumerate(speakers):
        label = chr(ord("A") + index)
        asset = assets.get(label)
        if not asset:
            _console_print(f"   ⚠️ Speaker {label} için güvenli insan-preview örneği çıkmadı; isimsiz kalacak.")
            continue
        name = _ask_name(label, asset, used_names)
        if name:
            names[label] = name
            used_names.add(name.casefold())

    if not names:
        profile = _mark_unlabeled_speaker_structure(profile, "manual_all_unknown_unlabeled")
        _write_json(path, profile)
        first = next((str((assets[k] or {}).get("reference_path", "")) for k in sorted(assets)), "")
        return {"profile_path": path, "mode": mode, "preview_path": first or None, "labels": {}}

    profile = _apply_calibrated_identities(profile, assets=assets, names=names)
    _write_json(path, profile)
    first = str(profile.get("speaker_preview_path", "")).strip()

    _console_print()
    mapped = ", ".join(f"{label}={name}" for label, name in sorted(names.items()))
    unknown = [chr(ord("A") + i) for i in range(len(speakers)) if chr(ord("A") + i) not in names]
    _console_print(f"   ✅ Kimlik kalibrasyonu: {mapped}")
    if unknown:
        _console_print(f"   🕶️ İsimsiz kalacak speaker: {', '.join(unknown)}")
    _console_print("   Finalde bu isimler gerçek voice-reference olarak diarization modeline geri verilecek.")

    return {
        "profile_path": path,
        "mode": mode,
        "preview_path": first or None,
        "labels": names,
        "identity_calibration": profile.get("identity_calibration", {}),
    }
