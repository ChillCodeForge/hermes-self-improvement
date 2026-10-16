from __future__ import annotations

import hashlib
import hmac
import json
import secrets
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
    """Tracks consumed holdout generations without exposing task content."""

    def __init__(self) -> None:
        self._spent: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def mark_spent(self, generation: str, candidate_id: str) -> None:
        with self._lock:
            key = (generation, candidate_id)
            if key in self._spent:
                raise AccessDenied("holdout feedback for this candidate was already consumed")
            self._spent.add(key)

    def is_spent(self, generation: str, candidate_id: str) -> bool:
        with self._lock:
            return (generation, candidate_id) in self._spent


def manifest_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()
