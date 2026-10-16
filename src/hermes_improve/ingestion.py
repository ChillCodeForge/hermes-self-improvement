from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .models import canonical_json, utc_now


_SECRET_KEY = re.compile(
    r"^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|"
    r"client[_-]?secret|private[_-]?key|authorization|cookie|token)$",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@"),
)
_MAX_SESSION_BYTES = 10 * 1024 * 1024


def redact_text(value: str) -> str:
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def contains_secret_material(value: Any, *, key: str = "") -> bool:
    if _SECRET_KEY.search(key) and value != "[REDACTED]":
        return True
    if isinstance(value, str):
        return any(pattern.search(value) for pattern in _SECRET_PATTERNS)
    if isinstance(value, Mapping):
        return any(contains_secret_material(item, key=str(child_key)) for child_key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_secret_material(item) for item in value)
    return False


def redact(value: Any, *, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(child_key): redact(child_value, key=str(child_key)) for child_key, child_value in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ValueError(f"unsupported session data type: {type(value).__name__}")


class SessionQueue:
    """Durable local queue storing redacted session exports as JSON."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    ingested_at TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT ''
                )
            """)

    def ingest(self, export: Mapping[str, Any]) -> str:
        session_id = export.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session export requires a non-empty session_id")
        safe_export = redact(export)
        safe_export["_evidence_trust"] = "untrusted"
        payload = canonical_json(safe_export)
        if len(payload.encode("utf-8")) > _MAX_SESSION_BYTES:
            raise ValueError("redacted session export exceeds the 10 MiB ingestion limit")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.path) as db:
            existing = db.execute(
                "SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if existing and existing[0] != digest:
                raise ValueError(f"session {session_id!r} was already ingested with different content")
            if existing:
                return digest
            db.execute(
                "INSERT INTO sessions (session_id,content_hash,payload_json,ingested_at) VALUES (?,?,?,?)",
                (session_id, digest, payload, utc_now()),
            )
        return digest

    def get(self, session_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        payload = json.loads(row[0])
        if payload.get("_evidence_trust") != "untrusted":
            raise ValueError("session is missing the untrusted-evidence marker")
        return payload

    def queued(self, limit: int = 100) -> list[str]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with sqlite3.connect(self.path) as db:
            return [row[0] for row in db.execute(
                "SELECT session_id FROM sessions WHERE status='queued' ORDER BY ingested_at LIMIT ?", (limit,)
            )]

    def set_status(self, session_id: str, status: str, error: str = "") -> None:
        if status not in {"queued", "processed", "excluded", "failed"}:
            raise ValueError("invalid session queue status")
        with sqlite3.connect(self.path) as db:
            cursor = db.execute(
                "UPDATE sessions SET status=?,error=? WHERE session_id=?", (status, redact_text(error), session_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(session_id)
