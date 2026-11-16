from __future__ import annotations

import logging
import os
import sqlite3
import stat
from pathlib import Path
from typing import Any

from .ingestion import SessionQueue, contains_secret_material
from .models import utc_now

logger = logging.getLogger("hermes_improve.hermes_plugin")


_SKILL_LIFECYCLE_ACTIONS = frozenset({
    "loaded", "patched", "edited", "created", "installed", "archived", "stale", "restored",
})
_SKILL_PROVENANCE = frozenset({"installed", "agent_created", "external", "local", "unknown"})
_HOOK_DB_TIMEOUT_SECONDS = 0.25


def register(context: Any) -> None:
    """Register the opt-in Hermes observers.

    ``on_session_finalize`` is always registered and checks ``auto_ingest`` per
    call. The skill-lifecycle and tool-call observers are registered only when
    ``observe_skill_lifecycle`` / ``observe_tool_calls`` is explicitly ``true``,
    so a disabled observer adds no work to Hermes' tool-call path. Both
    observers return ``None`` (never a ``pre_tool_call`` directive) and swallow
    their own failures: they can never block or alter a user turn.
    """
    context.register_hook(
        "on_session_finalize",
        _build_finalize_callback(context),
    )
    if context.get_config("observe_skill_lifecycle", False) is True:
        context.register_hook("on_skill_lifecycle", _build_skill_lifecycle_callback(context))
    if context.get_config("observe_tool_calls", False) is True:
        context.register_hook("pre_tool_call", _build_pre_tool_call_callback(context))


class HookSignalStore:
    """Bounded local store of skill-lifecycle facts and per-day tool-call counts.

    Tool-call arguments are never stored; only the tool name and a daily count.
    """

    def __init__(self, path: Path, *, timeout: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout = timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=self.timeout) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS skill_lifecycle_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    provenance TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    use_count INTEGER,
                    reused INTEGER,
                    reuse_after_patch INTEGER,
                    created_at TEXT NOT NULL
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS tool_call_counts (
                    day TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (day, tool_name)
                )
            """)
        if os.name == "posix":
            self.path.chmod(0o600)

    def record_skill_lifecycle(
        self, *, action: Any, skill_name: Any, provenance: Any, session_id: Any,
        use_count: Any, reused: Any, reuse_after_patch: Any,
    ) -> None:
        from .curator import validate_curator_skill_name

        if action not in _SKILL_LIFECYCLE_ACTIONS:
            raise ValueError("unknown skill lifecycle action")
        validate_curator_skill_name(skill_name)
        if provenance not in _SKILL_PROVENANCE:
            raise ValueError("unknown skill provenance label")
        if not isinstance(session_id, str) or len(session_id) > 256 or contains_secret_material(session_id):
            raise ValueError("invalid session_id")
        if use_count is not None and (
            isinstance(use_count, bool) or not isinstance(use_count, int) or not 0 <= use_count < (1 << 63)
        ):
            raise ValueError("use_count must be a non-negative integer or null")
        for flag in (reused, reuse_after_patch):
            if flag is not None and not isinstance(flag, bool):
                raise ValueError("lifecycle reuse flags must be boolean or null")
        with sqlite3.connect(self.path, timeout=self.timeout) as db:
            db.execute(
                """INSERT INTO skill_lifecycle_events
                   (action,skill_name,provenance,session_id,use_count,reused,reuse_after_patch,created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (action, skill_name, provenance, session_id, use_count,
                 None if reused is None else int(reused),
                 None if reuse_after_patch is None else int(reuse_after_patch), utc_now()),
            )

    def record_tool_call(self, tool_name: Any) -> None:
        if (
            not isinstance(tool_name, str) or not tool_name.strip() or len(tool_name) > 200
            or any(ord(character) < 32 for character in tool_name) or contains_secret_material(tool_name)
        ):
            raise ValueError("invalid tool name")
        with sqlite3.connect(self.path, timeout=self.timeout) as db:
            db.execute(
                """INSERT INTO tool_call_counts (day,tool_name,count) VALUES (?,?,1)
                   ON CONFLICT(day,tool_name) DO UPDATE SET count=count+1""",
                (utc_now()[:10], tool_name),
            )

    def summary(self) -> dict[str, Any]:
        with sqlite3.connect(self.path, timeout=self.timeout) as db:
            actions = dict(db.execute(
                "SELECT action, COUNT(*) FROM skill_lifecycle_events GROUP BY action ORDER BY action"
            ).fetchall())
            skills = [
                {"skill_name": name, "loaded": loaded, "patched": patched, "reuse_after_patch": reuse}
                for name, loaded, patched, reuse in db.execute("""
                    SELECT skill_name,
                           SUM(action='loaded'),
                           SUM(action IN ('patched','edited')),
                           SUM(COALESCE(reuse_after_patch,0))
                    FROM skill_lifecycle_events GROUP BY skill_name
                    ORDER BY COUNT(*) DESC, skill_name LIMIT 50
                """)
            ]
            tools = dict(db.execute(
                "SELECT tool_name, SUM(count) FROM tool_call_counts GROUP BY tool_name ORDER BY tool_name"
            ).fetchall())
        return {"skill_lifecycle_actions": actions, "skills": skills, "tool_calls": tools}


def _build_skill_lifecycle_callback(context: Any):
    def on_skill_lifecycle(
        *, action: Any = None, skill_name: Any = None, provenance: Any = None,
        task_id: Any = "", session_id: Any = "", use_count: Any = None,
        reused: Any = None, reuse_after_patch: Any = None, **_extra: Any,
    ) -> None:
        try:
            HookSignalStore(_state_root(context) / "hook_signals.sqlite3", timeout=_HOOK_DB_TIMEOUT_SECONDS).record_skill_lifecycle(
                action=action, skill_name=skill_name, provenance=provenance,
                session_id=session_id if isinstance(session_id, str) else "",
                use_count=use_count, reused=reused, reuse_after_patch=reuse_after_patch,
            )
        except Exception:
            logger.debug("Skipped Hermes skill lifecycle signal", exc_info=True)
        return None

    return on_skill_lifecycle


def _build_pre_tool_call_callback(context: Any):
    def pre_tool_call(*, tool_name: Any = None, **_ignored: Any) -> None:
        try:
            HookSignalStore(
                _state_root(context) / "hook_signals.sqlite3", timeout=_HOOK_DB_TIMEOUT_SECONDS,
            ).record_tool_call(tool_name)
        except Exception:
            logger.debug("Skipped Hermes tool-call signal", exc_info=True)
        return None

    return pre_tool_call


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
