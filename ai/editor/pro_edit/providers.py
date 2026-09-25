"""Planner provider adapters. Model-vendor details stop here.

Every provider returns a ``ProviderResponse`` holding the raw text. Parsing,
validation, resolution and rendering after that point are identical for a
live model call and a replayed recording (same boundary, no fake editor).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ai.editor.pro_edit.errors import PlannerOutputError, PlannerUnavailableError
from ai.editor.pro_edit.request import PlannerRequest


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    provider: str
    model: str
    response_id: str | None = None
    status: str = "completed"
    usage: dict[str, Any] = field(default_factory=dict)

    def to_artifact(self) -> dict[str, Any]:
        # Raw model output + non-secret metadata only (no keys, no headers).
        return {"provider": self.provider, "model": self.model, "response_id": self.response_id,
                "status": self.status, "usage": dict(self.usage), "text": self.text}


class PlannerProvider(Protocol):
    name: str
    model: str

    def complete(self, request: PlannerRequest) -> ProviderResponse:
        ...


def _usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    result = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, (int, float)):
            result[key] = value
    return result


def _refusal(response: Any) -> str | None:
    for item in getattr(response, "output", None) or []:
        for content in getattr(item, "content", None) or []:
            if getattr(content, "type", "") == "refusal":
                return str(getattr(content, "refusal", "") or "refused")
    return None


class OpenAIResponsesProvider:
    """Structured-output call through MIMIR's shared lazy OpenAI client.

    Uses the same Responses API + strict json_schema pattern as the existing
    MIMIR judges (speaker_role_judge, clip_analyzer). Credentials come only
    from MIMIR's existing .env / OPENAI_API_KEY handling.
    """

    name = "openai_responses"

    def __init__(self, model: str, reasoning_effort: str, *, timeout_s: float = 90.0, client: Any | None = None) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_s = float(timeout_s)
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from ai.openai_client import get_client
            client = get_client()
        except Exception as error:  # missing key / package -> explicit unavailability
            raise PlannerUnavailableError(f"planner client unavailable: {type(error).__name__}: {error}") from error
        with_options = getattr(client, "with_options", None)
        return with_options(timeout=self.timeout_s) if callable(with_options) else client

    def complete(self, request: PlannerRequest) -> ProviderResponse:
        client = self._get_client()
        try:
            response = client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                instructions=request.instructions,
                input=request.input_text(),
                text={"format": {"type": "json_schema", "name": request.schema_name, "strict": True,
                                 "schema": dict(request.schema)}},
            )
        except Exception as error:  # network/auth/quota/model errors are availability failures
            raise PlannerUnavailableError(f"planner call failed: {type(error).__name__}: {error}") from error
        status = str(getattr(response, "status", "completed") or "completed")
        if status in {"failed", "cancelled", "incomplete"}:
            details = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
            raise PlannerOutputError(f"planner response status {status}: {details}")
        refusal = _refusal(response)
        if refusal is not None:
            raise PlannerOutputError(f"planner refused: {refusal[:300]}")
        return ProviderResponse(str(getattr(response, "output_text", "") or ""), self.name, self.model,
                                getattr(response, "id", None), status, _usage(response))


class ReplayProvider:
    """Replays a recorded raw response artifact (or a bare plan JSON file).

    Enters exactly where a live response enters; nothing downstream differs.
    A recording may hold ``repair`` for the repair round.
    """

    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise PlannerUnavailableError(f"replay recording unreadable: {self.path}: {error}") from error
        if isinstance(data, dict) and isinstance(data.get("text"), str):
            self._rounds = [data] + ([data["repair"]] if isinstance(data.get("repair"), dict) else [])
        elif isinstance(data, dict):
            self._rounds = [{"text": json.dumps(data), "model": "replay"}]
        else:
            raise PlannerUnavailableError("replay recording must be a JSON object")
        self.model = str(self._rounds[0].get("model", "replay"))
        self._index = 0

    def complete(self, request: PlannerRequest) -> ProviderResponse:
        if self._index >= len(self._rounds):
            raise PlannerOutputError(f"replay has no recorded response for a {request.kind} round")
        row = self._rounds[self._index]
        self._index += 1
        return ProviderResponse(str(row.get("text", "")), self.name, str(row.get("model", self.model)),
                                row.get("response_id"), "completed", dict(row.get("usage", {}) or {}))
