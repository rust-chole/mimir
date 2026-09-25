from pathlib import Path
import re
import shutil
import py_compile

TARGET = Path("ai/shorts_pipeline.py")

if not TARGET.is_file():
    raise SystemExit(
        "HATA: ai/shorts_pipeline.py bulunamadi.\n"
        "Bu scripti MIMIR proje kokunde calistir."
    )

text = TARGET.read_text(encoding="utf-8")

if "Mandatory intro structural guard" not in text:
    raise SystemExit(
        "HATA: Beklenen mandatory intro guard bu dosyada bulunamadi. "
        "Dosyaya dokunulmadi."
    )

pattern = re.compile(
    r"^    \# Structural contract:.*?(?=^    _stage_done\(final_preview_path\))",
    re.MULTILINE | re.DOTALL,
)

matches = list(pattern.finditer(text))
if len(matches) != 1:
    raise SystemExit(
        f"HATA: Guard blogu tekil bulunamadi (eslesme={len(matches)}). "
        "Dosyaya dokunulmadi."
    )

replacement = '''    # Structural contract: final base must be longer than the EFFECTIVE main
    # actually used by intro_renderer, proving a real cold-open was prepended.
    #
    # V7.1 can trim a long dead lead before the first caption. Comparing final
    # duration against the ORIGINAL untrimmed captioned main creates a false
    # "main-only" failure even when the intro is really present.
    #
    # Reuse intro_renderer's exact restart calculation here. If anything is
    # uncertain, fall back to restart=0.0 so the guard remains strict/safe.
    main_duration_check = _probe_video_duration(captioned_preview_path)
    final_duration_check = _probe_video_duration(final_preview_path)

    main_restart_check = 0.0
    if main_duration_check > 0.0:
        try:
            main_restart_check, _ = intro_renderer.calculate_main_restart_seconds(
                caption_path=caption_path,
                main_duration=main_duration_check,
            )
        except Exception:
            main_restart_check = 0.0

    effective_main_duration_check = max(
        0.0,
        main_duration_check - main_restart_check,
    )

    minimum_intro_delta = max(0.20, min(0.45, teaser_duration * 0.35))

    if (
        effective_main_duration_check > 0.0
        and final_duration_check > 0.0
        and final_duration_check
        < effective_main_duration_check + minimum_intro_delta
    ):
        raise ShortsPipelineError(
            "Mandatory intro structural guard: final video main-only gorunuyor "
            f"(main_raw={main_duration_check:.3f}s, "
            f"main_restart={main_restart_check:.3f}s, "
            f"main_effective={effective_main_duration_check:.3f}s, "
            f"final={final_duration_check:.3f}s). "
            "Introsuz cikti yayinlanmadi."
        )

'''

new_text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit("HATA: Guard degistirilemedi. Dosyaya dokunulmadi.")

backup = TARGET.with_suffix(".py.before_intro_guard_fix.bak")
if not backup.exists():
    shutil.copy2(TARGET, backup)

TARGET.write_text(new_text, encoding="utf-8")

try:
    py_compile.compile(str(TARGET), doraise=True)
except Exception as exc:
    shutil.copy2(backup, TARGET)
    raise SystemExit(
        "HATA: Syntax kontrolu gecmedi; eski dosya otomatik geri yuklendi.\n"
        f"{exc}"
    )

print("OK: Mandatory intro structural guard duzeltildi.")
print(f"Backup: {backup}")
print("Degisen tek dosya: ai/shorts_pipeline.py")
print("Caption / speaker / intro renderer koduna dokunulmadi.")
