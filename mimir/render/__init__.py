"""Deterministic rendering. Layers stay separate and are combined only here:

    base story timeline (render_base) -> camera/reframe (compositor) -> effects (flash)
    -> captions (libass burn) + audio (story audio + SFX mix, loudness) -> final MP4
"""
