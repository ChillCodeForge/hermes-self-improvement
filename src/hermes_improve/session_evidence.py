"""Bounded, redacted tool-failure evidence derived from Hermes tool results (finding F11).

Hermes reports a failed tool call inside the tool result, not in a session
field: ``{"error": "..."}``, a non-zero ``exit_code`` from ``terminal``, or
``{"success": false, ...}``. Batch trajectories (Hermes v0.22.2) wrap each
result as ``<tool_response>{"tool_call_id", "name", "content"}</tool_response>``
inside a ``tool`` turn; persisted sessions (the plugin path) hold the result as
the JSON-string ``content`` of a ``role: tool`` message.

The derived strings are untrusted evidence like the rest of an export: at most
:data:`MAX_TOOL_ERRORS` per session, each ``"<tool>: <text>"``, secret-redacted
before it is truncated to :data:`MAX_TOOL_ERROR_CHARS`.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

from .ingestion import redact_text


MAX_TOOL_ERRORS = 20
MAX_TOOL_ERROR_CHARS = 300
# Redaction runs on this much of the raw text, so a secret straddling the
# final truncation point is still recognised and masked as a whole.
_SCAN_CHARS = 4000
_MAX_TOOL_NAME = 200
_TOOL_RESPONSE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_TEXT_FAILURE = re.compile(r"^\s*(?:error|exception|traceback)\b", re.IGNORECASE)
_FAILED_STATUS = frozenset({"error", "failed", "failure"})
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def tool_failure_text(content: Any) -> str | None:
    """Return the failure description of one tool result, or ``None`` if it succeeded."""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return tool_failure_text(parsed)
        return stripped if _TEXT_FAILURE.match(stripped) else None
    if not isinstance(content, Mapping):
        return None
    error = content.get("error")
    if isinstance(error, Mapping):
        error = error.get("message")
    if isinstance(error, str) and error.strip():
        return error.strip()
    exit_code = content.get("exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0:
        detail = next(
            (value.strip() for key in ("stderr", "output")
             if isinstance(value := content.get(key), str) and value.strip()),
            "",
        )
        return f"exit code {exit_code}" + (f": {detail}" if detail else "")
    status = content.get("status")
    if content.get("success") is False or (isinstance(status, str) and status.casefold() in _FAILED_STATUS):
        message = next(
            (value.strip() for key in ("message", "msg", "detail")
             if isinstance(value := content.get(key), str) and value.strip()),
            "",
        )
        return message or "tool reported failure"
    return None


def format_tool_error(tool_name: Any, text: str) -> str:
    name = tool_name.strip()[:_MAX_TOOL_NAME] if isinstance(tool_name, str) and tool_name.strip() else "tool"
    safe = redact_text(_CONTROL.sub(" ", f"{name}: {text[:_SCAN_CHARS]}"))
    return " ".join(safe.split())[:MAX_TOOL_ERROR_CHARS]


def collect_tool_errors(results: Iterable[tuple[Any, Any]]) -> list[str]:
    """Bounded error strings for ``(tool_name, result_content)`` pairs, in order."""
    errors: list[str] = []
    for tool_name, content in results:
        text = tool_failure_text(content)
        if text is None:
            continue
        errors.append(format_tool_error(tool_name, text))
        if len(errors) >= MAX_TOOL_ERRORS:
            break
    return errors


def _batch_results(turns: Iterable[Mapping[str, Any]]) -> Iterable[tuple[Any, Any]]:
    for turn in turns:
        if turn.get("from") != "tool" or not isinstance(turn.get("value"), str):
            continue
        for match in _TOOL_RESPONSE.finditer(turn["value"]):
            try:
                body = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(body, Mapping) and "content" in body:
                yield body.get("name"), body["content"]


def tool_errors_from_batch_turns(turns: Iterable[Mapping[str, Any]]) -> list[str]:
    """Tool errors from BatchRunner ``conversations`` (``from: tool`` turns)."""
    return collect_tool_errors(_batch_results(turns))


def _message_results(messages: Iterable[Mapping[str, Any]]) -> Iterable[tuple[Any, Any]]:
    names: dict[str, str] = {}
    for message in messages:
        calls = message.get("tool_calls")
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, Mapping) or not isinstance(call.get("id"), str):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, Mapping) else call.get("name")
            if isinstance(name, str):
                names[call["id"]] = name
        if message.get("role") != "tool":
            continue
        name = message.get("name")
        call_id = message.get("tool_call_id")
        if not isinstance(name, str):
            name = names.get(call_id, "tool") if isinstance(call_id, str) else "tool"
        content = message.get("content")
        if isinstance(content, list):
            content = "\n".join(
                item["text"] for item in content if isinstance(item, Mapping) and isinstance(item.get("text"), str)
            )
        yield name, content


def tool_errors_from_messages(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    """Tool errors from OpenAI-style persisted messages (``role: tool``)."""
    return collect_tool_errors(_message_results(messages))
