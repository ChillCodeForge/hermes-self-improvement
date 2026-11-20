from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

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
_MAX_INGEST_BATCH_RECORDS = 10_000
_MAX_INGEST_BATCH_BYTES = 10 * 1024 * 1024


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


_DREAMING_STAGES = frozenset({"light", "rem"})
OUTCOME_LABELS = frozenset({"correct", "incorrect", "partial"})
OUTCOME_SOURCES = frozenset({"verifier", "user", "monitor"})
MAX_OUTCOME_NOTE_CHARS = 500
_LABEL_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _validated_label(label: Mapping[str, Any]) -> tuple[str, str, str, str]:
    session_id = label.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 256:
        raise ValueError("outcome label requires a session_id of at most 256 characters")
    outcome, source = label.get("outcome"), label.get("source")
    if outcome not in OUTCOME_LABELS:
        raise ValueError(f"outcome must be one of {sorted(OUTCOME_LABELS)}")
    if source not in OUTCOME_SOURCES:
        raise ValueError(f"outcome source must be one of {sorted(OUTCOME_SOURCES)}")
    note = label.get("note", "")
    if note is None:
        note = ""
    if not isinstance(note, str) or len(note) > MAX_OUTCOME_NOTE_CHARS:
        raise ValueError(f"outcome note must be text of at most {MAX_OUTCOME_NOTE_CHARS} characters")
    if _LABEL_CONTROL.search(note):
        raise ValueError("outcome note must not contain control characters")
    return session_id, outcome, source, redact_text(note.strip())


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
            # Per-stage processing marks (lab finding F7). LIGHT records its
            # mark here and leaves ``status`` alone, so a later REM run still
            # sees the session; REM marks the session ``processed``.
            db.execute("""
                CREATE TABLE IF NOT EXISTS session_stages (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    stage TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, stage)
                )
            """)
            # Operator-attached outcome evidence (lab finding F11). Kept apart
            # from the hashed export, so labelling never changes the stored
            # evidence and an export can never label itself.
            db.execute("""
                CREATE TABLE IF NOT EXISTS session_outcomes (
                    session_id TEXT PRIMARY KEY REFERENCES sessions(session_id),
                    outcome TEXT NOT NULL,
                    source TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    labelled_at TEXT NOT NULL
                )
            """)

    def ingest(self, export: Mapping[str, Any]) -> str:
        return self.ingest_many([export])[0]

    def ingest_many(self, exports: Iterable[Mapping[str, Any]]) -> list[str]:
        prepared: list[tuple[str, str, str]] = []
        seen: dict[str, str] = {}
        total_bytes = 0
        for export in exports:
            if len(prepared) >= _MAX_INGEST_BATCH_RECORDS:
                raise ValueError("session export batch exceeds the 10,000-record limit")
            session_id = export.get("session_id")
            if not isinstance(session_id, str) or not session_id.strip():
                raise ValueError("session export requires a non-empty session_id")
            safe_export = redact(export)
            safe_export["_evidence_trust"] = "untrusted"
            payload = canonical_json(safe_export)
            payload_bytes = len(payload.encode("utf-8"))
            if payload_bytes > _MAX_SESSION_BYTES:
                raise ValueError("redacted session export exceeds the 10 MiB ingestion limit")
            total_bytes += payload_bytes
            if total_bytes > _MAX_INGEST_BATCH_BYTES:
                raise ValueError("redacted session export batch exceeds the 10 MiB limit")
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if session_id in seen and seen[session_id] != digest:
                raise ValueError(f"session {session_id!r} was supplied with different content")
            seen[session_id] = digest
            prepared.append((session_id, digest, payload))
        if not prepared:
            raise ValueError("session export batch contains no records")

        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            for session_id, digest, payload in prepared:
                existing = db.execute(
                    "SELECT content_hash FROM sessions WHERE session_id=?", (session_id,)
                ).fetchone()
                if existing and existing[0] != digest:
                    raise ValueError(f"session {session_id!r} was already ingested with different content")
                if not existing:
                    db.execute(
                        "INSERT INTO sessions (session_id,content_hash,payload_json,ingested_at) VALUES (?,?,?,?)",
                        (session_id, digest, payload, utc_now()),
                    )
        return [digest for _session_id, digest, _payload in prepared]

    def get(self, session_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(session_id)
        payload = json.loads(row[0])
        if payload.get("_evidence_trust") != "untrusted":
            raise ValueError("session is missing the untrusted-evidence marker")
        # Only an operator label counts; an ``outcome`` claimed by the export is dropped.
        payload.pop("outcome", None)
        label = self.outcome(session_id)
        if label is not None:
            payload["outcome"] = label
        return payload

    def label_outcome(self, session_id: str, *, outcome: str, source: str, note: str = "") -> dict[str, str]:
        """Attach (or replace) the outcome label of one queued session."""
        self.label_outcomes_many([
            {"session_id": session_id, "outcome": outcome, "source": source, "note": note},
        ])
        label = self.outcome(session_id)
        assert label is not None
        return label

    def label_outcomes_many(self, labels: Iterable[Mapping[str, Any]]) -> int:
        """Validate every label, then store them in one transaction (all or nothing).

        Unknown sessions raise ``KeyError``; invalid labels raise ``ValueError``.
        The latest label of a session replaces an earlier one.
        """
        prepared: list[tuple[str, str, str, str]] = []
        for label in labels:
            if len(prepared) >= _MAX_INGEST_BATCH_RECORDS:
                raise ValueError("outcome label batch exceeds the 10,000-record limit")
            if not isinstance(label, Mapping):
                raise ValueError("outcome label must be a JSON object")
            unknown = set(label) - {"session_id", "outcome", "source", "note"}
            if unknown:
                raise ValueError(f"outcome label has unknown fields: {sorted(unknown)}")
            prepared.append(_validated_label(label))
        if not prepared:
            raise ValueError("outcome label batch contains no records")
        labelled_at = utc_now()
        with sqlite3.connect(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            for session_id, outcome, source, note in prepared:
                if db.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone() is None:
                    raise KeyError(session_id)
                db.execute(
                    """INSERT INTO session_outcomes (session_id,outcome,source,note,labelled_at)
                       VALUES (?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET
                       outcome=excluded.outcome, source=excluded.source, note=excluded.note,
                       labelled_at=excluded.labelled_at""",
                    (session_id, outcome, source, note, labelled_at),
                )
        return len(prepared)

    def outcome(self, session_id: str) -> dict[str, str] | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT outcome,source,note,labelled_at FROM session_outcomes WHERE session_id=?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return {"label": row[0], "source": row[1], "note": row[2], "labelled_at": row[3]}

    def queued(self, limit: int = 100) -> list[str]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with sqlite3.connect(self.path) as db:
            return [row[0] for row in db.execute(
                "SELECT session_id FROM sessions WHERE status='queued' ORDER BY ingested_at LIMIT ?", (limit,)
            )]

    def pending_for_stage(self, stage: str, limit: int = 100) -> list[str]:
        """Queued sessions that ``stage`` has not processed yet, oldest first."""
        if stage not in _DREAMING_STAGES:
            raise ValueError("unknown dreaming stage")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with sqlite3.connect(self.path) as db:
            return [row[0] for row in db.execute(
                """SELECT session_id FROM sessions WHERE status='queued' AND session_id NOT IN
                   (SELECT session_id FROM session_stages WHERE stage=?) ORDER BY ingested_at LIMIT ?""",
                (stage, limit),
            )]

    def mark_stage(self, session_id: str, stage: str) -> None:
        """Record that ``stage`` processed ``session_id`` (idempotent)."""
        if stage not in _DREAMING_STAGES:
            raise ValueError("unknown dreaming stage")
        with sqlite3.connect(self.path) as db:
            if db.execute("SELECT 1 FROM sessions WHERE session_id=?", (session_id,)).fetchone() is None:
                raise KeyError(session_id)
            db.execute(
                "INSERT OR IGNORE INTO session_stages (session_id,stage,processed_at) VALUES (?,?,?)",
                (session_id, stage, utc_now()),
            )

    def session_ids(self, limit: int = 1000) -> list[str]:
        """All recorded session IDs regardless of processing status, oldest first (read-only)."""
        if not 1 <= limit <= 10_000:
            raise ValueError("limit must be between 1 and 10000")
        with sqlite3.connect(self.path) as db:
            return [row[0] for row in db.execute(
                "SELECT session_id FROM sessions ORDER BY ingested_at, session_id LIMIT ?", (limit,)
            )]

    def status_counts(self) -> dict[str, int]:
        with sqlite3.connect(self.path) as db:
            return dict(db.execute(
                "SELECT status, COUNT(*) FROM sessions GROUP BY status ORDER BY status"
            ).fetchall())

    def set_status(self, session_id: str, status: str, error: str = "") -> None:
        if status not in {"queued", "processed", "excluded", "failed"}:
            raise ValueError("invalid session queue status")
        with sqlite3.connect(self.path) as db:
            cursor = db.execute(
                "UPDATE sessions SET status=?,error=? WHERE session_id=?", (status, redact_text(error), session_id)
            )
            if cursor.rowcount != 1:
                raise KeyError(session_id)
