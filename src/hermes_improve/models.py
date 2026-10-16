from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def candidate_digest(
    candidate_id: str,
    experiment_id: str,
    candidate_type: str,
    target: str,
    base_commit: str,
    commit: str,
    diff_hash: str,
) -> str:
    payload = canonical_json({
        "candidate_id": candidate_id,
        "experiment_id": experiment_id,
        "candidate_type": candidate_type,
        "target": target,
        "base_commit": base_commit,
        "commit": commit,
        "diff_hash": diff_hash,
    })
    return hashlib.sha256(payload.encode()).hexdigest()


class Split(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    REGRESSION = "REGRESSION"
    SEALED_HOLDOUT = "SEALED_HOLDOUT"
    SHADOW = "SHADOW"


class Eligibility(StrEnum):
    HARD_BENCHMARK = "HARD_BENCHMARK"
    SOFT_BENCHMARK = "SOFT_BENCHMARK"
    DREAMING_ONLY = "DREAMING_ONLY"
    EXCLUDED = "EXCLUDED"


class Role(StrEnum):
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    PROMOTER = "promoter"
    OPERATOR = "operator"


class CandidateType(StrEnum):
    SKILL_CREATE = "SKILL_CREATE"
    SKILL_PATCH = "SKILL_PATCH"
    SKILL_MERGE = "SKILL_MERGE"
    SKILL_SPLIT = "SKILL_SPLIT"
    TOOL_DESCRIPTION = "TOOL_DESCRIPTION"
    WORKFLOW = "WORKFLOW"
    ROUTING = "ROUTING"
    MCP_TOOL_FILTER = "MCP_TOOL_FILTER"
    MCP_CONFIG = "MCP_CONFIG"
    SYSTEM_PROMPT = "SYSTEM_PROMPT"
    TOOL_CREATE = "TOOL_CREATE"
    TOOL_PATCH = "TOOL_PATCH"


@dataclass(frozen=True)
class Benchmark:
    schema_version: int
    id: str
    version: int
    source: Mapping[str, Any]
    task: Mapping[str, Any]
    taxonomy: Mapping[str, Any]
    environment: Mapping[str, Any]
    eligibility: Mapping[str, Any]
    verification: Mapping[str, Any]
    ground_truth: Mapping[str, Any]
    benchmark: Mapping[str, Any]
    security: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Benchmark":
        required = {
            "schema_version", "id", "version", "source", "task", "taxonomy",
            "environment", "eligibility", "verification", "ground_truth",
            "benchmark", "security",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"benchmark missing required fields: {', '.join(sorted(missing))}")
        if isinstance(data["schema_version"], bool) or data["schema_version"] != 1:
            raise ValueError("unsupported benchmark schema_version")
        if not isinstance(data["id"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", data["id"]
        ):
            raise ValueError("benchmark id must be a safe 1-128 character identifier")
        if isinstance(data["version"], bool) or not isinstance(data["version"], int) or data["version"] < 1:
            raise ValueError("benchmark version must be a positive integer")
        for section in required - {"schema_version", "id", "version"}:
            if not isinstance(data[section], Mapping):
                raise ValueError(f"benchmark {section} must be an object")
        bench = data["benchmark"]
        try:
            Split(bench["split"])
            Eligibility(data["eligibility"]["class"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid benchmark split or eligibility: {exc}") from exc
        score = data["eligibility"].get("reproducibility_score")
        if (
            isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 1
        ):
            raise ValueError("reproducibility_score must be between 0 and 1")
        if (
            data["security"].get("sanitized") is not True
            or data["security"].get("contains_secrets") is not False
        ):
            raise ValueError("benchmark must be sanitized and contain no secrets")
        if not isinstance(data["task"].get("title"), str) or not isinstance(data["task"].get("prompt"), str):
            raise ValueError("benchmark task requires title and prompt strings")
        for section, key in (
            ("taxonomy", "categories"), ("taxonomy", "required_capabilities"),
            ("verification", "deterministic"), ("verification", "subjective"),
            ("benchmark", "suites"),
        ):
            if not isinstance(data[section].get(key), list):
                raise ValueError(f"benchmark {section}.{key} must be an array")
        from .ingestion import contains_secret_material
        if contains_secret_material(data):
            raise ValueError("benchmark contains unredacted credential-like material")
        if not isinstance(bench.get("cluster_id"), str) or not bench["cluster_id"]:
            raise ValueError("benchmark requires a non-empty cluster_id")
        return cls(**{key: data[key] for key in required})

    @property
    def split(self) -> Split:
        return Split(self.benchmark["split"])

    @property
    def cluster_id(self) -> str:
        return str(self.benchmark["cluster_id"])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    experiment_id: str
    candidate_type: CandidateType
    target: str
    base_commit: str
    commit: str
    diff: str
    created_at: str = field(default_factory=utc_now)

    @property
    def content_hash(self) -> str:
        return candidate_digest(
            self.candidate_id,
            self.experiment_id,
            self.candidate_type.value,
            self.target,
            self.base_commit,
            self.commit,
            hashlib.sha256(self.diff.encode()).hexdigest(),
        )


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    split: Split
    suite: str
    baseline_success: bool
    candidate_success: bool
    baseline_quality: float = 0.0
    candidate_quality: float = 0.0
    baseline_tokens: int = 0
    candidate_tokens: int = 0
    deterministic_pass: bool = True
    security_pass: bool = True
    critical_regressions: int = 0
    high_regressions: int = 0
    medium_regressions: int = 0
    categories: tuple[str, ...] = ()
    baseline_latency_ms: float = 0.0
    candidate_latency_ms: float = 0.0
    baseline_tool_calls: int = 0
    candidate_tool_calls: int = 0
    baseline_failed_tool_calls: int = 0
    candidate_failed_tool_calls: int = 0
    baseline_retries: int = 0
    candidate_retries: int = 0
    baseline_user_corrections: int = 0
    candidate_user_corrections: int = 0


@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    candidate_hash: str
    benchmark_version: str
    policy_version: str
    outcomes: tuple[TaskOutcome, ...]
    passed: bool
    reasons: tuple[str, ...]
    created_at: str = field(default_factory=utc_now)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    run_observations: tuple["RunObservation", ...] = ()


def evaluation_payload(report: EvaluationReport) -> dict[str, Any]:
    outcomes = []
    for outcome in sorted(report.outcomes, key=lambda row: row.task_id):
        item = asdict(outcome)
        item["split"] = outcome.split.value
        item["categories"] = list(outcome.categories)
        outcomes.append(item)
    return {
        "candidate_id": report.candidate_id,
        "candidate_hash": report.candidate_hash,
        "benchmark_version": report.benchmark_version,
        "policy_version": report.policy_version,
        "outcomes": outcomes,
        "passed": report.passed,
        "reasons": list(report.reasons),
        "created_at": report.created_at,
        "metrics": dict(report.metrics),
        "run_observations": [
            {
                **asdict(row),
                "artifact_digests": list(row.artifact_digests),
            }
            for row in sorted(report.run_observations, key=lambda item: (item.task_id, item.commit, item.seed))
        ],
    }


@dataclass(frozen=True)
class RunObservation:
    task_id: str
    commit: str
    seed: int
    environment_digest: str
    success: bool
    quality: float
    tokens: int
    latency_ms: float
    deterministic_pass: bool
    security_pass: bool
    critical_regressions: int = 0
    high_regressions: int = 0
    medium_regressions: int = 0
    tool_calls: int = 0
    failed_tool_calls: int = 0
    retries: int = 0
    user_corrections: int = 0
    context_tokens: int = 0
    exit_code: int = 0
    final_output: str = ""
    artifact_digests: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImprovementSignal:
    source: str
    summary: str
    issue: str
    impact_score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
