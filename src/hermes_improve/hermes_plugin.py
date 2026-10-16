from __future__ import annotations

import logging
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

from .ingestion import SessionQueue

logger = logging.getLogger("hermes_improve.hermes_plugin")


def register(context: Any) -> None:
    """Register the opt-in Hermes lifecycle observer."""
    context.register_hook(
        "on_session_finalize",
        _build_finalize_callback(context),
    )


def _build_finalize_callback(context: Any):
    def on_session_finalize(*, session_id: str = "", **event: Any) -> None:
        if context.get_config("auto_ingest", False) is not True:
            return
        if not isinstance(session_id, str) or not session_id.strip():
            logger.warning("Skipping Hermes session ingestion: session_id is missing")
            return
        try:
            messages, metadata = _load_session(session_id)
            if not messages:
                logger.info("Skipping Hermes session %s: no persisted transcript", session_id)
                return
            export = _session_export(session_id, messages, metadata, event)
            state_root = _state_root(context)
            destination = state_root / "sessions.sqlite3"
            digest = SessionQueue(destination).ingest(export)
            if os.name == "posix":
                destination.chmod(0o600)
            logger.info("Queued redacted Hermes session %s (sha256=%s)", session_id, digest)
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError, sqlite3.Error):
            logger.warning("Could not ingest finalized Hermes session %s", session_id, exc_info=True)

    return on_session_finalize


def _load_session(session_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agent.trace_upload import load_session_messages

    messages, metadata = load_session_messages(session_id)
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("Hermes returned an invalid session transcript")
    if not isinstance(metadata, dict):
        metadata = {}
    return messages, metadata


def _state_root(context: Any) -> Path:
    configured = context.get_config("state_dir")
    if configured is not None:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError("Hermes plugin state_dir must be an absolute path")
        state_root = path
    else:
        from hermes_constants import get_hermes_home

        state_root = Path(get_hermes_home()) / "self-improvement"

    state_root = state_root.resolve()
    if state_root in {Path(state_root.anchor), Path.home().resolve()}:
        raise ValueError("Hermes plugin state_dir must be a dedicated subdirectory")
    state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name == "posix":
        if stat.S_IMODE(state_root.stat().st_mode) & 0o077:
            raise PermissionError("Hermes plugin state_dir must not be accessible by group or other users")
    return state_root


def _session_export(
    session_id: str,
    messages: list[dict[str, Any]],
    metadata: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    task_prompt = next(
        (
            text for message in messages
            if message.get("role") == "user"
            if (text := _message_text(message.get("content")))
        ),
        "",
    )
    tool_calls = []
    for message in messages:
        for call in message.get("tool_calls", []) if isinstance(message.get("tool_calls"), list) else []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else call.get("name")
            if isinstance(name, str) and name.strip():
                tool_calls.append({"name": name[:200]})
    export: dict[str, Any] = {
        "session_id": session_id,
        "task": {"prompt": task_prompt[:10000]},
        "messages": messages,
        "tool_calls": tool_calls,
        "errors": [],
        "final_status": {
            key: event[key]
            for key in ("completed", "failed", "interrupted", "reason", "turn_exit_reason")
            if key in event and isinstance(event[key], (str, bool, int, float, type(None)))
        },
        "model": str(metadata.get("model") or event.get("model") or ""),
        "environment": {"type": "hermes-session"},
        "security": {"sanitized": True, "contains_secrets": False},
    }
    cwd = metadata.get("cwd")
    if isinstance(cwd, str):
        export["working_directory"] = cwd[:2000]
    return export


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return ""
