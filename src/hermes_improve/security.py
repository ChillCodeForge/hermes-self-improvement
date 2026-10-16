from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import threading
from dataclasses import dataclass

from .errors import AccessDenied
from .models import Role, Split


@dataclass(frozen=True)
class HoldoutGrant:
    """Single-process evaluator capability; not a substitute for OS isolation."""

    _nonce: str


class AccessController:
    def __init__(self) -> None:
        self._nonce = secrets.token_urlsafe(32)

    def evaluator_holdout_grant(self, role: Role) -> HoldoutGrant:
        if role is not Role.EVALUATOR:
            raise AccessDenied("only the evaluator can receive a holdout grant")
        return HoldoutGrant(self._nonce)

    def authorize_benchmark_read(
        self, role: Role, split: Split, grant: HoldoutGrant | None = None
    ) -> None:
        if split is Split.SEALED_HOLDOUT:
            if role is not Role.EVALUATOR or grant is None or not hmac.compare_digest(grant._nonce, self._nonce):
                raise AccessDenied("sealed holdout is accessible only with an evaluator capability")
        elif role is Role.GENERATOR and split not in {Split.TRAIN, Split.VALIDATION}:
            raise AccessDenied(f"generator cannot access {split.value} benchmarks")


class SpentHoldoutRegistry:
    """Tracks consumed holdout generations in memory or durable SQLite."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path
        self._spent: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        if self._path is not None:
            with sqlite3.connect(self._path) as db:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS spent_holdouts (
                        generation TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        spent_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (generation, candidate_id)
                    )
                """)

    def mark_spent(self, generation: str, candidate_id: str) -> None:
        with self._lock:
            key = (generation, candidate_id)
            if self._path is not None:
                try:
                    with sqlite3.connect(self._path) as db:
                        db.execute(
                            "INSERT INTO spent_holdouts (generation,candidate_id) VALUES (?,?)",
                            key,
                        )
                except sqlite3.IntegrityError as exc:
                    raise AccessDenied("holdout feedback for this candidate was already consumed") from exc
                return
            if key in self._spent:
                raise AccessDenied("holdout feedback for this candidate was already consumed")
            self._spent.add(key)

    def is_spent(self, generation: str, candidate_id: str) -> bool:
        with self._lock:
            if self._path is not None:
                with sqlite3.connect(self._path) as db:
                    row = db.execute(
                        "SELECT 1 FROM spent_holdouts WHERE generation=? AND candidate_id=?",
                        (generation, candidate_id),
                    ).fetchone()
                return row is not None
            return (generation, candidate_id) in self._spent


def manifest_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
