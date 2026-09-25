from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ai.video_brain.config import (
    GEMINI_API_KEY,
    VIDEO_BRAIN_MODEL,
    VIDEO_UPLOAD_POLL_SECONDS,
    VIDEO_UPLOAD_TIMEOUT_SECONDS,
    validate_config,
)


# ============================================================
# MIMIR VIDEO BRAIN CLIENT V5
# ============================================================
#
# SUPPORT-ONLY network layer.
#
# Uses Windows curl.exe + Gemini REST only.
# Existing MIMIR/OpenAI pipeline is untouched.
#
# Flow:
#   1) Files API resumable upload start
#   2) upload + finalize video bytes
#   3) poll until ACTIVE
#   4) models/<model>:generateContent with file_data
# ============================================================


BASE_URL = "https://generativelanguage.googleapis.com"
UPLOAD_URL = f"{BASE_URL}/upload/v1beta/files"

CONNECT_TIMEOUT = 30
REQUEST_TIMEOUT = 600
UPLOAD_TIMEOUT = 900
CURL_RETRIES = 2


class GeminiAPIError(RuntimeError):
    """Structured Gemini API failure."""


class GeminiQuotaError(GeminiAPIError):
    """Gemini quota/rate-limit exhaustion (HTTP 429 / RESOURCE_EXHAUSTED)."""



# ============================================================
# CURL HELPERS
# ============================================================

def _find_curl() -> str:

    curl_path = (
        shutil.which("curl.exe")
        or shutil.which("curl")
    )

    if not curl_path:
        raise RuntimeError(
            "curl.exe bulunamadı."
        )

    return curl_path


def _curl_base() -> list[str]:

    # HTTP 429 must NOT be retried blindly: it wastes quota attempts and can
    # concatenate multiple JSON error bodies. Transport-level failures are
    # retried manually in _run_curl instead.
    return [
        _find_curl(),
        "-4",
        "--http1.1",
        "--tlsv1.2",
        "--silent",
        "--show-error",
        "--connect-timeout",
        str(CONNECT_TIMEOUT),
    ]


def _api_key_header() -> list[str]:

    return [
        "-H",
        f"x-goog-api-key: {GEMINI_API_KEY}",
    ]


def _run_curl(
    args: list[str],
    *,
    timeout: float = REQUEST_TIMEOUT,
) -> str:
    """Execute curl without echoing secrets; retry transport failures only."""

    command = _curl_base() + args
    last_stdout = ""
    last_stderr = ""
    last_code = 0

    for attempt in range(CURL_RETRIES + 1):
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            if attempt < CURL_RETRIES:
                time.sleep(1.0)
                continue
            raise TimeoutError(
                f"Gemini curl timeout ({timeout:.0f}s)."
            ) from error

        last_stdout = result.stdout.strip() if result.stdout else ""
        last_stderr = result.stderr.strip() if result.stderr else ""
        last_code = int(result.returncode)

        # curl normally exits 0 for HTTP 4xx/5xx; those responses must be
        # parsed once by _parse_json, not retried here.
        if result.returncode == 0:
            return last_stdout

        if attempt < CURL_RETRIES:
            time.sleep(1.0)

    raise RuntimeError(
        "Gemini REST/curl isteği başarısız:\n"
        + (last_stdout or last_stderr or f"curl exit code={last_code}")[:5000]
    )


# ============================================================
# JSON HELPERS
# ============================================================

def _parse_json(
    text: str,
    *,
    context: str,
) -> dict[str, Any]:

    raw = str(
        text
    ).strip()

    if not raw:
        raise RuntimeError(
            f"{context}: boş response."
        )

    try:

        data = json.loads(
            raw
        )

    except json.JSONDecodeError as error:

        raise RuntimeError(
            f"{context}: geçersiz JSON.\n"
            + raw[:5000]
        ) from error

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"{context}: root object değil."
        )

    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        try:
            code = int(error_obj.get("code", 0))
        except (TypeError, ValueError):
            code = 0
        status = str(error_obj.get("status", "")).strip()
        message = str(error_obj.get("message", "")).strip()
        compact = (
            f"{context}: Gemini API error {code or '?'}"
            + (f" {status}" if status else "")
            + (f". {message}" if message else "")
        )
        if code == 429 or status.upper() == "RESOURCE_EXHAUSTED":
            raise GeminiQuotaError(compact[:5000])
        raise GeminiAPIError(compact[:5000])

    return data


def _extract_text(
    response: dict[str, Any],
) -> str:

    candidates = response.get(
        "candidates",
        [],
    )

    if not isinstance(
        candidates,
        list,
    ):
        return ""

    texts: list[str] = []

    for candidate in candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            continue

        content = candidate.get(
            "content",
            {},
        )

        if not isinstance(
            content,
            dict,
        ):
            continue

        parts = content.get(
            "parts",
            [],
        )

        if not isinstance(
            parts,
            list,
        ):
            continue

        for part in parts:

            if not isinstance(
                part,
                dict,
            ):
                continue

            text = str(
                part.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

            if text:
                texts.append(
                    text
                )

    return "\n".join(
        texts
    ).strip()


# ============================================================
# CONNECTION TEST
# ============================================================

def test_connection() -> str:

    validate_config()

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Reply exactly: "
                            "MIMIR_VIDEO_BRAIN_OK"
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="mimir_vbrain_test_"
    ) as temp_dir:

        request_file = (
            Path(temp_dir)
            / "request.json"
        )

        request_file.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        endpoint = (
            f"{BASE_URL}/v1beta/models/"
            f"{VIDEO_BRAIN_MODEL}:generateContent"
        )

        raw = _run_curl(
            [
                "-X",
                "POST",
                endpoint,
                *_api_key_header(),
                "-H",
                "Content-Type: application/json",
                "-H",
                "Connection: close",
                "--data-binary",
                f"@{request_file}",
            ],
            timeout=60,
        )

    response = _parse_json(
        raw,
        context="Gemini API test",
    )

    text = _extract_text(
        response
    )

    expected = "MIMIR_VIDEO_BRAIN_OK"
    cleaned = text.strip()

    if expected not in cleaned:

        usage = response.get(
            "usageMetadata",
            {},
        )

        raise RuntimeError(
            "Gemini bağlantısı cevap verdi fakat "
            "test cevabı tamamlanmadı.\n"
            f"Gelen: {cleaned or '<boş>'}\n"
            f"finishReason: "
            f"{response.get('candidates', [{}])[0].get('finishReason', '') if response.get('candidates') else ''}\n"
            f"usageMetadata: "
            f"{json.dumps(usage, ensure_ascii=False)}"
        )

    return expected


# ============================================================
# VIDEO HELPERS
# ============================================================

def _resolve_video(
    video_path: str | Path,
) -> Path:

    path = Path(
        str(
            video_path
        ).strip().strip('"')
    ).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Video bulunamadı:\n{path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Video yolu dosya değil:\n{path}"
        )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Video dosyası boş:\n{path}"
        )

    return path


def _mime_type(
    path: Path,
) -> str:

    guessed, _ = mimetypes.guess_type(
        path.name
    )

    if (
        guessed
        and guessed.startswith(
            "video/"
        )
    ):
        return guessed

    return "video/mp4"


# ============================================================
# RESUMABLE UPLOAD START
# ============================================================

def _extract_upload_url(
    headers_file: Path,
) -> str:

    if not headers_file.exists():
        raise RuntimeError(
            "Upload header dosyası oluşmadı."
        )

    upload_url = ""

    for line in headers_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():

        if ":" not in line:
            continue

        name, value = line.split(
            ":",
            1,
        )

        if (
            name.strip().lower()
            == "x-goog-upload-url"
        ):
            upload_url = (
                value.strip()
            )

    if not upload_url:
        raise RuntimeError(
            "X-Goog-Upload-URL alınamadı."
        )

    return upload_url


def start_resumable_upload(
    video_path: str | Path,
) -> dict[str, Any]:

    validate_config()

    path = _resolve_video(
        video_path
    )

    file_size = int(
        path.stat().st_size
    )

    mime_type = _mime_type(
        path
    )

    with tempfile.TemporaryDirectory(
        prefix="mimir_upload_start_"
    ) as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        metadata_file = (
            temp_dir
            / "metadata.json"
        )

        headers_file = (
            temp_dir
            / "headers.txt"
        )

        metadata_file.write_text(
            json.dumps(
                {
                    "file": {
                        "display_name": path.name,
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        _run_curl(
            [
                UPLOAD_URL,
                *_api_key_header(),
                "-D",
                str(
                    headers_file
                ),
                "-H",
                "X-Goog-Upload-Protocol: resumable",
                "-H",
                "X-Goog-Upload-Command: start",
                "-H",
                (
                    "X-Goog-Upload-Header-Content-Length: "
                    f"{file_size}"
                ),
                "-H",
                (
                    "X-Goog-Upload-Header-Content-Type: "
                    f"{mime_type}"
                ),
                "-H",
                "Content-Type: application/json",
                "-H",
                "Connection: close",
                "--data-binary",
                f"@{metadata_file}",
            ],
            timeout=90,
        )

        upload_url = _extract_upload_url(
            headers_file
        )

    return {
        "local_path": str(
            path
        ),
        "file_size": file_size,
        "mime_type": mime_type,
        "upload_url": upload_url,
    }


# ============================================================
# UPLOAD + FINALIZE
# ============================================================

def finalize_resumable_upload(
    upload_session: dict[str, Any],
) -> dict[str, Any]:

    path = _resolve_video(
        upload_session[
            "local_path"
        ]
    )

    upload_url = str(
        upload_session.get(
            "upload_url",
            "",
        )
    ).strip()

    file_size = int(
        upload_session.get(
            "file_size",
            path.stat().st_size,
        )
    )

    if not upload_url:
        raise RuntimeError(
            "upload_url yok."
        )

    raw = _run_curl(
        [
            upload_url,
            "-H",
            f"Content-Length: {file_size}",
            "-H",
            "X-Goog-Upload-Offset: 0",
            "-H",
            "X-Goog-Upload-Command: upload, finalize",
            "-H",
            "Connection: close",
            "--data-binary",
            f"@{path}",
        ],
        timeout=max(
            UPLOAD_TIMEOUT,
            float(
                VIDEO_UPLOAD_TIMEOUT_SECONDS
            ),
        ),
    )

    response = _parse_json(
        raw,
        context="Gemini video upload",
    )

    file_info = response.get(
        "file",
        {},
    )

    if not isinstance(
        file_info,
        dict,
    ):
        raise RuntimeError(
            "Upload response içinde file object yok."
        )

    name = str(
        file_info.get(
            "name",
            "",
        )
        or ""
    ).strip()

    if not name:
        raise RuntimeError(
            "Upload response file.name içermiyor."
        )

    return {
        "name": name,
        "uri": str(
            file_info.get(
                "uri",
                "",
            )
            or ""
        ).strip(),
        "state": str(
            file_info.get(
                "state",
                "",
            )
            or ""
        ).strip().upper(),
        "mime_type": str(
            file_info.get(
                "mimeType",
                upload_session.get(
                    "mime_type",
                    "video/mp4",
                ),
            )
            or upload_session.get(
                "mime_type",
                "video/mp4",
            )
        ).strip(),
        "display_name": str(
            file_info.get(
                "displayName",
                path.name,
            )
            or path.name
        ),
    }


# ============================================================
# FILE STATUS
# ============================================================

def _normalize_file_name(
    file_name: str,
) -> str:

    name = str(
        file_name
    ).strip().lstrip("/")

    if name.startswith(
        "v1beta/"
    ):
        name = name[
            len(
                "v1beta/"
            ):
        ]

    if not name.startswith(
        "files/"
    ):
        name = (
            "files/"
            + name
        )

    return name


def get_remote_file(
    file_name: str,
) -> dict[str, Any]:

    validate_config()

    name = _normalize_file_name(
        file_name
    )

    endpoint = (
        f"{BASE_URL}/v1beta/{name}"
    )

    raw = _run_curl(
        [
            endpoint,
            *_api_key_header(),
            "-H",
            "Connection: close",
        ],
        timeout=60,
    )

    return _parse_json(
        raw,
        context="Gemini file status",
    )


def wait_until_active(
    file_info: dict[str, Any],
) -> dict[str, Any]:

    name = str(
        file_info.get(
            "name",
            "",
        )
    ).strip()

    if not name:
        raise RuntimeError(
            "file name yok."
        )

    current = dict(
        file_info
    )

    started = time.monotonic()

    while True:

        state = str(
            current.get(
                "state",
                "",
            )
            or ""
        ).upper()

        if state == "ACTIVE":

            if not current.get(
                "uri"
            ):

                status = get_remote_file(
                    name
                )

                current[
                    "uri"
                ] = str(
                    status.get(
                        "uri",
                        "",
                    )
                    or ""
                ).strip()

            if not current.get(
                "uri"
            ):
                raise RuntimeError(
                    "ACTIVE file için URI alınamadı."
                )

            print(
                "✅ Gemini video ACTIVE."
            )

            return current

        if state == "FAILED":

            raise RuntimeError(
                "Gemini video processing FAILED."
            )

        elapsed = (
            time.monotonic()
            - started
        )

        if (
            elapsed
            >= float(
                VIDEO_UPLOAD_TIMEOUT_SECONDS
            )
        ):

            raise TimeoutError(
                "Gemini video processing timeout.\n"
                f"Limit: "
                f"{VIDEO_UPLOAD_TIMEOUT_SECONDS:.0f}s"
            )

        print(
            "   ⏳ Video işleniyor..."
            f" state={state or 'PROCESSING'}"
        )

        time.sleep(
            max(
                1.0,
                float(
                    VIDEO_UPLOAD_POLL_SECONDS
                ),
            )
        )

        status = get_remote_file(
            name
        )

        current = {
            "name": str(
                status.get(
                    "name",
                    name,
                )
                or name
            ).strip(),
            "uri": str(
                status.get(
                    "uri",
                    current.get(
                        "uri",
                        "",
                    ),
                )
                or current.get(
                    "uri",
                    "",
                )
            ).strip(),
            "state": str(
                status.get(
                    "state",
                    "",
                )
                or ""
            ).upper(),
            "mime_type": str(
                status.get(
                    "mimeType",
                    current.get(
                        "mime_type",
                        "video/mp4",
                    ),
                )
                or current.get(
                    "mime_type",
                    "video/mp4",
                )
            ).strip(),
            "display_name": str(
                status.get(
                    "displayName",
                    current.get(
                        "display_name",
                        "",
                    ),
                )
                or current.get(
                    "display_name",
                    "",
                )
            ).strip(),
        }


def upload_video(
    video_path: str | Path,
) -> dict[str, Any]:

    path = _resolve_video(
        video_path
    )

    print()
    print(
        "☁️ Gemini Files API upload"
    )
    print(
        f"   {path.name}"
    )

    upload_session = start_resumable_upload(
        path
    )

    print(
        "   📤 Upload session hazır."
    )

    file_info = finalize_resumable_upload(
        upload_session
    )

    print(
        "   ✅ Video bytes gönderildi."
    )
    print(
        f"   📄 {file_info['name']}"
    )

    return wait_until_active(
        file_info
    )


# ============================================================
# GENERATE CONTENT WITH VIDEO
# ============================================================

def analyze_uploaded_video(
    uploaded_file: dict[str, Any],
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
) -> dict[str, Any]:

    validate_config()

    selected_model = str(
        model
        or VIDEO_BRAIN_MODEL
    ).strip()

    file_uri = str(
        uploaded_file.get(
            "uri",
            "",
        )
        or ""
    ).strip()

    mime_type = str(
        uploaded_file.get(
            "mime_type",
            "video/mp4",
        )
        or "video/mp4"
    ).strip()

    if not file_uri:
        raise RuntimeError(
            "Uploaded file URI yok."
        )

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "file_data": {
                            "mime_type": mime_type,
                            "file_uri": file_uri,
                        }
                    },
                    {
                        "text": str(
                            prompt
                        )
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": float(
                temperature
            ),
            "maxOutputTokens": int(
                max_output_tokens
            ),
        },
    }

    with tempfile.TemporaryDirectory(
        prefix="mimir_video_generate_"
    ) as temp_dir:

        request_file = (
            Path(temp_dir)
            / "request.json"
        )

        request_file.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        endpoint = (
            f"{BASE_URL}/v1beta/models/"
            f"{selected_model}:generateContent"
        )

        raw = _run_curl(
            [
                "-X",
                "POST",
                endpoint,
                *_api_key_header(),
                "-H",
                "Content-Type: application/json",
                "-H",
                "Connection: close",
                "--data-binary",
                f"@{request_file}",
            ],
            timeout=REQUEST_TIMEOUT,
        )

    response = _parse_json(
        raw,
        context="Gemini video generateContent",
    )

    output_text = _extract_text(
        response
    )

    if not output_text:
        raise RuntimeError(
            "Gemini video analizi boş çıktı döndürdü.\n\n"
            "Raw response:\n"
            + json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            )[:5000]
        )

    return {
        "model": selected_model,
        "model_version": str(
            response.get(
                "modelVersion",
                "",
            )
            or ""
        ),
        "response_id": str(
            response.get(
                "responseId",
                "",
            )
            or ""
        ),
        "usage_metadata": response.get(
            "usageMetadata",
            {},
        ),
        "output_text": output_text,
        "remote_file": dict(
            uploaded_file
        ),
    }


def analyze_local_video(
    video_path: str | Path,
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int = 8192,
) -> dict[str, Any]:
    """Upload, analyze and always attempt to remove the remote Gemini file."""

    uploaded_file = upload_video(video_path)

    try:
        return analyze_uploaded_video(
            uploaded_file=uploaded_file,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    finally:
        remote_name = str(uploaded_file.get("name", "")).strip()
        if remote_name:
            deleted = delete_remote_file(remote_name)
            if not deleted:
                print("⚠️ Gemini remote video cleanup başarısız; analiz sonucu korunuyor.")


# ============================================================
# OPTIONAL REMOTE CLEANUP
# ============================================================

def delete_remote_file(
    file_name: str,
) -> bool:

    validate_config()

    name = _normalize_file_name(
        file_name
    )

    endpoint = (
        f"{BASE_URL}/v1beta/{name}"
    )

    try:

        _run_curl(
            [
                "-X",
                "DELETE",
                endpoint,
                *_api_key_header(),
                "-H",
                "Connection: close",
            ],
            timeout=60,
        )

        return True

    except Exception:

        return False


# ============================================================
# CLI
# ============================================================

def main() -> int:

    print()
    print(
        "MIMIR Video Brain Client V5"
    )
    print(
        "Transport: curl.exe + Gemini REST"
    )
    print(
        f"Model: {VIDEO_BRAIN_MODEL}"
    )
    print()

    try:

        result = test_connection()

        print(
            f"✅ API OK: {result}"
        )

        return 0

    except Exception as error:

        print(
            "❌ VIDEO BRAIN CLIENT HATASI:"
        )
        print(
            error
        )

        return 1


if __name__ == "__main__":

    raise SystemExit(
        main()
    )
