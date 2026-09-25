"""OpenCV runtime: one import, with the exact reason when it cannot load.

``import cv2`` fails for causes that need different fixes, and a vague
"opencv not installed" hides the real one:

* not installed                     -> install the requirements;
* installed, native module refused  -> on Windows 11 Smart App Control (or a
  WDAC policy) blocks unsigned native modules it has no reputation for, so a
  fresh opencv-python-headless build can be installed yet never load
  (``DLL load failed ... Application Control policy``); older, widely used
  builds are trusted -> install the version range pinned in requirements.txt;
* other import errors (broken build / ABI mismatch) -> reinstall.

Every local-vision consumer (face detection/tracking, caption activity and
background evidence, V6 pixel proof) imports OpenCV through ``load_opencv`` so
the failure is classified once, cached (a refused DLL is not re-attempted for
every frame) and reported with its remedy. ``OpenCVUnavailable`` subclasses
ImportError, so existing ``except ImportError`` fallbacks keep working.
"""
from __future__ import annotations

import sys
from importlib import metadata
from typing import Any

INSTALL_HINT = "python -m pip install -r requirements.txt"
_DISTRIBUTIONS = ("opencv-python-headless", "opencv-python", "opencv-contrib-python-headless",
                  "opencv-contrib-python")
_SAC_KEY = r"SYSTEM\CurrentControlSet\Control\CI\Policy"
_SAC_STATES = {0: "off", 1: "on", 2: "evaluation"}


class OpenCVUnavailable(ImportError):
    """OpenCV cannot be used in this process (``kind`` + human ``reason`` with the remedy)."""

    def __init__(self, reason: str, kind: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


def installed_distribution() -> str | None:
    """'<distribution> <version>' of the installed OpenCV wheel, read without importing it."""
    for name in _DISTRIBUTIONS:
        try:
            return f"{name} {metadata.version(name)}"
        except metadata.PackageNotFoundError:
            continue
    return None


def smart_app_control_state() -> str | None:
    """'on' | 'evaluation' | 'off' on Windows 11 (None when unknown / not Windows)."""
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _SAC_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "VerifiedAndReputablePolicyState")
    except (OSError, ImportError, ValueError):
        return None
    try:
        return _SAC_STATES.get(int(value), str(value))
    except (TypeError, ValueError):
        return None


def classify_import_error(error: BaseException, *, installed: str | None = None,
                          sac_state: str | None = None) -> tuple[str, str]:
    """(kind, reason with remedy) for a failed ``import cv2``."""
    text = " ".join(str(error).split())[:240]
    if isinstance(error, ModuleNotFoundError) and not installed:
        return "not_installed", f"OpenCV is not installed; run: {INSTALL_HINT}"
    what = installed or "OpenCV"
    if "DLL load failed" in text:
        if sac_state == "on":       # evaluation mode only audits, it never blocks
            return ("blocked_by_windows_app_control",
                    f"{what} is installed but Windows Smart App Control (on) refused to load its native "
                    f"module cv2.pyd (a build without Windows reputation); install the trusted build range pinned "
                    f"in requirements.txt: {INSTALL_HINT} [{text}]")
        return ("dll_load_failed",
                f"{what} is installed but its native module did not load; reinstall it: {INSTALL_HINT} [{text}]")
    return "import_failed", f"{what} import failed ({type(error).__name__}: {text}); reinstall it: {INSTALL_HINT}"


_MISSING = object()
_failure: tuple[str, str] | None = None


def _probe() -> tuple[Any | None, str, str]:
    """(module | None, kind, reason). A genuine import failure is classified once and
    not retried in this process (a refused DLL would be re-attempted per frame)."""
    global _failure
    loaded = sys.modules.get("cv2", _MISSING)
    if loaded is None:       # import explicitly disabled in this process
        return None, "import_failed", "OpenCV import is disabled in this process (sys.modules['cv2'] is None)"
    if loaded is not _MISSING:
        return loaded, "ok", f"OpenCV {getattr(loaded, '__version__', '?')}"
    if _failure is not None:
        return None, *_failure
    try:
        import cv2
    except Exception as error:
        _failure = classify_import_error(error, installed=installed_distribution(),
                                         sac_state=smart_app_control_state())
        return None, *_failure
    return cv2, "ok", f"OpenCV {getattr(cv2, '__version__', '?')}"


def load_opencv() -> Any:
    """The cv2 module, or ``OpenCVUnavailable`` with the classified reason."""
    module, kind, reason = _probe()
    if module is None:
        raise OpenCVUnavailable(reason, kind)
    return module


def opencv_status() -> dict[str, str]:
    """{'status': 'ok' | <failure kind>, 'reason': ...} (never raises)."""
    module, kind, reason = _probe()
    return {"status": "ok" if module is not None else kind, "reason": reason}
