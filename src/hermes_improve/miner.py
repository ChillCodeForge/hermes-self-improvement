from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .ingestion import contains_secret_material
from .models import Eligibility, canonical_json


@dataclass(frozen=True)
class BenchmarkCandidate:
    candidate_id: str
    task_fingerprint: str
    cluster_id: str
    eligibility: Eligibility
    reproducibility_score: float
    ground_truth_type: str
    reasons: tuple[str, ...]


def task_fingerprint(task: str) -> str:
    normalized = re.sub(r"\s+", " ", task.casefold()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def mine_session(session: Mapping[str, Any]) -> BenchmarkCandidate:
    if contains_secret_material(session):
        return _excluded(session, "credential-like material detected")
    task = session.get("task")
    if not isinstance(task, Mapping):
        return _excluded(session, "task statement unavailable")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return _excluded(session, "task prompt unavailable")
    environment = session.get("environment")
    environment = environment if isinstance(environment, Mapping) else {}
    verification = session.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    security = session.get("security")
    security = security if isinstance(security, Mapping) else {}
    checks = {
        "repository_commit_known": bool(environment.get("git_commit")),
        "dependency_state_known": bool(environment.get("dependency_digest")),
        "environment_known": bool(environment.get("container_image") or environment.get("type")),
        "input_available": task.get("input_available") is True,
        "external_services_controlled": bool(environment.get("network_policy") in {"disabled", "allowlist", "fixture"}),
        "tests_present": (
            isinstance(verification.get("deterministic"), list)
            and bool(verification.get("deterministic"))
        ),
        "ground_truth_strong": (
            isinstance(verification.get("deterministic"), list)
            and bool(verification.get("deterministic"))
        ) or isinstance(verification.get("structured_result"), Mapping),
        "secret_free": security.get("sanitized") is True and security.get("contains_secrets") is False,
        "replay_verified": session.get("replay_verified") is True,
    }
    score = sum(checks.values()) / len(checks)
    reasons = tuple(name for name, passed in checks.items() if not passed)
    if not checks["secret_free"]:
        eligibility = Eligibility.EXCLUDED
    elif checks["ground_truth_strong"] and checks["replay_verified"] and score >= 0.8:
        eligibility = Eligibility.HARD_BENCHMARK
    elif score >= 0.45:
        eligibility = Eligibility.SOFT_BENCHMARK
    else:
        eligibility = Eligibility.DREAMING_ONLY
    ground_truth = "deterministic" if checks["tests_present"] else "structured" if checks["ground_truth_strong"] else "agent_self_report"
    fingerprint = task_fingerprint(prompt)
    return BenchmarkCandidate(
        candidate_id=hashlib.sha256(f"{session.get('session_id', '')}:{fingerprint}".encode()).hexdigest()[:24],
        task_fingerprint=fingerprint,
        cluster_id=fingerprint,
        eligibility=eligibility,
        reproducibility_score=score,
        ground_truth_type=ground_truth,
        reasons=reasons,
    )


def _excluded(session: Mapping[str, Any], reason: str) -> BenchmarkCandidate:
    identity = str(session.get("session_id", "unknown"))
    digest = hashlib.sha256(identity.encode()).hexdigest()
    return BenchmarkCandidate(
        candidate_id=digest[:24], task_fingerprint=digest, cluster_id=digest,
        eligibility=Eligibility.EXCLUDED, reproducibility_score=0.0,
        ground_truth_type="unknown", reasons=(reason,),
    )


class DedupRegistry:
    """Exact and normalized task deduplication; near/semantic models are deferred."""

    def __init__(self) -> None:
        self._clusters: dict[str, str] = {}

    def assign(self, candidate: BenchmarkCandidate, split: str) -> tuple[str, bool]:
        existing = self._clusters.get(candidate.cluster_id)
        if existing is not None and existing != split:
            return existing, False
        self._clusters[candidate.cluster_id] = split
        return split, True
