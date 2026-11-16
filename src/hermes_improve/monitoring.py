from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from .decisions import DecisionRecorder
from .ingestion import contains_secret_material
from .models import canonical_json, utc_now
from .store import ExperimentStore

if TYPE_CHECKING:
    from .promotion import GitPromoter


OBSERVATION_SOURCES = frozenset({"production", "shadow_replay", "historical_replay"})
SEVERITY_OPTIONS = ("none", "medium", "high", "critical")
_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MAX_OBSERVATION_SKILLS = 32
_MONITORING_MODEL_VERSION = "monitoring-policy-1"


@dataclass(frozen=True)
class ProductionObservation:
    release_id: str
    task_id: str
    success: bool
    tool_errors: int
    tokens: int
    tool_calls: int
    latency_ms: float
    user_corrections: int = 0
    retries: int = 0
    critical_regression: bool = False
    observation_id: str = ""
    skills: tuple[str, ...] = ()
    source: str = "production"


@dataclass(frozen=True)
class MonitoringDecision:
    status: str
    sample_count: int
    reasons: tuple[str, ...]
    duplicate: bool = False


@dataclass(frozen=True)
class MonitoringPolicy:
    min_observations: int = 20
    max_success_rate_drop: float = 0.05
    max_tool_error_rate_increase: float = 0.05
    max_tokens_per_success_increase: float = 0.15
    max_latency_increase: float = 0.15
    max_correction_rate_increase: float = 0.05
    max_tool_calls_per_success_increase: float = 0.15
    max_retry_rate_increase: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.min_observations, bool) or self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        thresholds = (
            self.max_success_rate_drop,
            self.max_tool_error_rate_increase,
            self.max_tokens_per_success_increase,
            self.max_latency_increase,
            self.max_correction_rate_increase,
            self.max_tool_calls_per_success_increase,
            self.max_retry_rate_increase,
        )
        if any(not math.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("monitoring thresholds must be finite and non-negative")


@dataclass(frozen=True)
class ReleaseMetrics:
    sample_count: int
    success_rate: float
    tool_error_rate: float
    tokens_per_success: float
    tool_calls_per_success: float
    latency_per_success_ms: float
    retry_rate: float
    correction_rate: float
    regression_signals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            key: (None if isinstance(value, float) and not math.isfinite(value) else value)
            for key, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class MonitoringAssessment:
    release_id: str
    release_status: str
    status: str
    severity: str
    reasons: tuple[str, ...]
    metrics: ReleaseMetrics | None
    baseline: Mapping[str, Any]
    affected_skills: tuple[str, ...]
    matched_observations: int
    unmatched_observations: int
    sources: Mapping[str, int]
    unavailable_baselines: tuple[str, ...] = ()
    decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "release_status": self.release_status,
            "status": self.status,
            "severity": self.severity,
            "reasons": list(self.reasons),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "baseline": dict(self.baseline),
            "affected_skills": list(self.affected_skills),
            "matched_observations": self.matched_observations,
            "unmatched_observations": self.unmatched_observations,
            "sources": dict(self.sources),
            "unavailable_baselines": list(self.unavailable_baselines),
            "decision_id": self.decision_id,
        }


@dataclass
class _Row:
    success: int
    tool_errors: int
    tokens: int
    tool_calls: int
    latency_ms: float
    user_corrections: int
    retries: int
    critical_regression: int
    skills: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class _Comparison:
    key: str
    reason: str
    severity: str
    value: float


class ReleaseMonitor:
    """Records post-release evidence, assesses it, and applies the lifecycle outcome.

    Observations are matched to the release's affected area: an observation
    that names the skills it exercised counts only if one of them was changed
    by the release; an observation without skill attribution counts for the
    whole release. Shadow/historical replay evidence is recorded through the
    same path with its own ``source`` label.
    """

    def __init__(
        self,
        path: Path,
        store: ExperimentStore,
        policy: MonitoringPolicy = MonitoringPolicy(),
        *,
        decisions: DecisionRecorder | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.policy = policy
        self.decisions = decisions
        with sqlite3.connect(self.path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS production_observations (
                    id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    tool_errors INTEGER NOT NULL,
                    tokens INTEGER NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    user_corrections INTEGER NOT NULL,
                    retries INTEGER NOT NULL,
                    critical_regression INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(production_observations)")}
            if "skills_json" not in columns:
                db.execute("ALTER TABLE production_observations ADD COLUMN skills_json TEXT NOT NULL DEFAULT '[]'")
            if "source" not in columns:
                db.execute("ALTER TABLE production_observations ADD COLUMN source TEXT NOT NULL DEFAULT 'production'")
            db.execute("""
                CREATE TABLE IF NOT EXISTS monitoring_actions (
                    id TEXT PRIMARY KEY,
                    release_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    target_release TEXT,
                    previous_commit TEXT,
                    reasons_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def observe(self, observation: ProductionObservation) -> MonitoringDecision:
        validate_observation(observation)
        record_id = observation.observation_id or uuid.uuid4().hex
        skills_json = canonical_json(sorted(set(observation.skills)))
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """SELECT release_id,task_id,success,tool_errors,tokens,tool_calls,latency_ms,
                          user_corrections,retries,critical_regression,skills_json,source
                   FROM production_observations WHERE id=?""",
                (record_id,),
            ).fetchone()
            if existing:
                supplied = (
                    observation.release_id, observation.task_id, int(observation.success),
                    observation.tool_errors, observation.tokens, observation.tool_calls,
                    float(observation.latency_ms), observation.user_corrections,
                    observation.retries, int(observation.critical_regression),
                    skills_json, observation.source,
                )
                if tuple(existing) != supplied:
                    raise ValueError("observation_id was already recorded with different content")
                count = db.execute(
                    "SELECT COUNT(*) FROM production_observations WHERE release_id=?",
                    (observation.release_id,),
                ).fetchone()[0]
                return MonitoringDecision("DUPLICATE", count, ("observation was already recorded",), True)
            release = self.store.get_release(observation.release_id)
            if release["status"] != "PROVISIONAL":
                raise ValueError("post-promotion observations are accepted only for provisional releases")
            baseline = release["manifest"].get("baseline_metrics")
            if not isinstance(baseline, dict):
                raise ValueError("release manifest does not include comparable baseline metrics")
            db.execute(
                """INSERT INTO production_observations
                (id,release_id,task_id,success,tool_errors,tokens,tool_calls,latency_ms,user_corrections,
                 retries,critical_regression,created_at,skills_json,source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (record_id, observation.release_id, observation.task_id, int(observation.success),
                 observation.tool_errors, observation.tokens, observation.tool_calls, observation.latency_ms,
                 observation.user_corrections, observation.retries, int(observation.critical_regression),
                 utc_now(), skills_json, observation.source),
            )
        assessment = self.evaluate(observation.release_id, record_decision=False)
        if assessment.status == "STABLE":
            self.store.set_release_status(observation.release_id, "STABLE")
        return MonitoringDecision(assessment.status, assessment.matched_observations, assessment.reasons)

    def evaluate(self, release_id: str, *, record_decision: bool = True) -> MonitoringAssessment:
        """Assess a release from matched evidence without changing its status."""
        release = self.store.get_release(release_id)
        manifest = release["manifest"]
        baseline = manifest.get("baseline_metrics")
        if not isinstance(baseline, dict):
            raise ValueError("release manifest does not include comparable baseline metrics")
        affected = release_affected_skills(manifest)
        rows = self._rows(release_id)
        matched = [row for row in rows if _matches(row.skills, affected)]
        sources: dict[str, int] = {}
        for row in matched:
            sources[row.source] = sources.get(row.source, 0) + 1
        metrics = compute_metrics(matched) if matched else None
        base = dict(
            release_id=release_id,
            release_status=release["status"],
            baseline=baseline,
            affected_skills=tuple(sorted(affected)),
            matched_observations=len(matched),
            unmatched_observations=len(rows) - len(matched),
            sources=sources,
            metrics=metrics,
        )
        if metrics is not None and metrics.regression_signals:
            assessment = MonitoringAssessment(
                status="ROLLBACK_RECOMMENDED", severity="critical",
                reasons=("critical regression observed",), **base,
            )
        elif len(matched) < self.policy.min_observations:
            return MonitoringAssessment(
                status="PROVISIONAL", severity="undetermined",
                reasons=("minimum observation sample not reached",), **base,
            )
        else:
            comparisons, unavailable = self._compare(metrics, baseline)
            if comparisons:
                severity = "high" if any(item.severity == "high" for item in comparisons) else "medium"
                assessment = MonitoringAssessment(
                    status="ROLLBACK_RECOMMENDED", severity=severity,
                    reasons=tuple(item.reason for item in comparisons),
                    unavailable_baselines=unavailable, **base,
                )
            else:
                assessment = MonitoringAssessment(
                    status="STABLE", severity="none", reasons=(),
                    unavailable_baselines=unavailable, **base,
                )
        if record_decision and self.decisions is not None:
            record = self.decisions.record(
                decision_type="regression_severity",
                input_features={
                    "release_id": release_id,
                    "affected_skills": list(assessment.affected_skills),
                    "metrics": assessment.metrics.to_dict() if assessment.metrics else {},
                    "baseline": {key: value for key, value in baseline.items() if isinstance(value, (int, float))},
                    "sources": dict(sources),
                },
                available_options=SEVERITY_OPTIONS,
                selected_option=assessment.severity,
                source="release-monitor",
                model_version=_MONITORING_MODEL_VERSION,
            )
            assessment = MonitoringAssessment(**{**assessment.__dict__, "decision_id": record.id})
        return assessment

    def apply(
        self,
        release_id: str,
        *,
        promoter: "GitPromoter | None",
        manifest_key: bytes | None,
        actor: str,
    ) -> dict[str, Any]:
        """Execute the assessed outcome: stabilize, or roll back through the promoter."""
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 128:
            raise ValueError("actor must identify who applied the monitoring decision")
        assessment = self.evaluate(release_id)
        if assessment.release_status != "PROVISIONAL":
            raise ValueError(f"only provisional releases can be applied; {release_id} is {assessment.release_status}")
        if assessment.status == "PROVISIONAL":
            return {"action": "NONE", "assessment": assessment.to_dict()}
        if assessment.status == "STABLE":
            self.store.set_release_status(release_id, "STABLE")
            audit_id = self._audit(release_id, "STABLE", None, None, assessment, actor)
            return {"action": "STABLE", "audit_id": audit_id, "assessment": assessment.to_dict()}
        if promoter is None or manifest_key is None:
            raise ValueError("rollback requires the promoter and the release manifest key")
        release = self.store.get_release(release_id)
        target = self.previous_release(release_id)
        if target is None:
            self._audit(release_id, "ROLLBACK_BLOCKED", None, None, assessment, actor)
            raise RuntimeError(
                f"{release_id} regressed but has no earlier active release to roll back to; operator action required"
            )
        active = _main_commit(promoter.repository)
        if active != release["manifest"].get("candidate_commit"):
            self._audit(release_id, "ROLLBACK_BLOCKED", target, None, assessment, actor)
            raise RuntimeError(f"{release_id} is not the active production release; refusing rollback")
        previous_commit = promoter.rollback(target, manifest_key=manifest_key)
        status = self.store.get_release(release_id)["status"]
        if status != "ROLLED_BACK":
            raise RuntimeError(f"promoter rollback did not mark {release_id} rolled back (status {status})")
        audit_id = self._audit(release_id, "ROLLBACK", target, previous_commit, assessment, actor)
        return {
            "action": "ROLLBACK",
            "audit_id": audit_id,
            "rolled_back_release": release_id,
            "active_release": target,
            "previous_commit": previous_commit,
            "assessment": assessment.to_dict(),
        }

    def previous_release(self, release_id: str) -> str | None:
        """The release whose commit was the baseline of ``release_id``, if still restorable."""
        release = self.store.get_release(release_id)
        baseline_commit = release["manifest"].get("baseline_commit")
        if not isinstance(baseline_commit, str) or not baseline_commit:
            return None
        matches = []
        for item in self.store.list_releases():
            if item["id"] == release_id or item["status"] == "ROLLED_BACK":
                continue
            candidate = self.store.get_release(item["id"])
            if candidate["manifest"].get("candidate_commit") == baseline_commit:
                matches.append(item["id"])
        if len(matches) > 1:
            raise ValueError("several releases claim the regressing release's baseline commit")
        return matches[0] if matches else None

    def actions(self, release_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM monitoring_actions"
        params: tuple[Any, ...] = ()
        if release_id is not None:
            query += " WHERE release_id=?"
            params = (release_id,)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(query + " ORDER BY created_at, id", params).fetchall()
        return [
            {**dict(row), "reasons": json.loads(row["reasons_json"]), "metrics": json.loads(row["metrics_json"])}
            for row in rows
        ]

    def _rows(self, release_id: str) -> list[_Row]:
        with sqlite3.connect(self.path, timeout=30) as db:
            rows = db.execute(
                """SELECT success,tool_errors,tokens,tool_calls,latency_ms,user_corrections,retries,
                          critical_regression,skills_json,source
                   FROM production_observations WHERE release_id=?""",
                (release_id,),
            ).fetchall()
        return [_Row(*row[:8], tuple(json.loads(row[8])), row[9]) for row in rows]

    def _compare(
        self, metrics: ReleaseMetrics, baseline: Mapping[str, Any],
    ) -> tuple[list[_Comparison], tuple[str, ...]]:
        policy = self.policy
        checks = (
            ("success_rate", "success rate regression", "high",
             lambda base: metrics.success_rate < base - policy.max_success_rate_drop),
            ("tool_error_rate", "tool error rate regression", "high",
             lambda base: metrics.tool_error_rate > base + policy.max_tool_error_rate_increase),
            ("tokens_per_success", "tokens per success regression", "medium",
             lambda base: metrics.tokens_per_success > base * (1 + policy.max_tokens_per_success_increase)),
            ("tool_calls_per_success", "tool calls per success regression", "medium",
             lambda base: metrics.tool_calls_per_success > base * (1 + policy.max_tool_calls_per_success_increase)),
            ("latency_per_success_ms", "latency regression", "medium",
             lambda base: metrics.latency_per_success_ms > base * (1 + policy.max_latency_increase)),
            ("retry_rate", "retry rate regression", "medium",
             lambda base: metrics.retry_rate > base + policy.max_retry_rate_increase),
            ("correction_rate", "user correction rate regression", "medium",
             lambda base: metrics.correction_rate > base + policy.max_correction_rate_increase),
        )
        failures, unavailable = [], []
        for key, reason, severity, regressed in checks:
            base = baseline.get(key)
            if isinstance(base, bool) or not isinstance(base, (int, float)) or not math.isfinite(base):
                unavailable.append(key)
                continue
            if regressed(float(base)):
                failures.append(_Comparison(key, reason, severity, getattr(metrics, key)))
        return failures, tuple(unavailable)

    def _audit(
        self, release_id: str, action: str, target: str | None, previous_commit: str | None,
        assessment: MonitoringAssessment, actor: str,
    ) -> str:
        audit_id = uuid.uuid4().hex
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute(
                """INSERT INTO monitoring_actions
                   (id,release_id,action,target_release,previous_commit,reasons_json,metrics_json,actor,created_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (audit_id, release_id, action, target, previous_commit,
                 canonical_json(list(assessment.reasons)),
                 canonical_json(assessment.metrics.to_dict() if assessment.metrics else {}),
                 actor.strip(), utc_now()),
            )
        return audit_id


def compute_metrics(rows: Iterable[_Row]) -> ReleaseMetrics:
    rows = list(rows)
    if not rows:
        raise ValueError("metrics require at least one observation")
    successes = sum(row.success for row in rows)

    def per_success(total: float) -> float:
        return total / successes if successes else float("inf")

    return ReleaseMetrics(
        sample_count=len(rows),
        success_rate=mean(row.success for row in rows),
        tool_error_rate=sum(row.tool_errors for row in rows) / max(1, sum(row.tool_calls for row in rows)),
        tokens_per_success=per_success(sum(row.tokens for row in rows)),
        tool_calls_per_success=per_success(sum(row.tool_calls for row in rows)),
        latency_per_success_ms=per_success(sum(row.latency_ms for row in rows)),
        retry_rate=mean(bool(row.retries) for row in rows),
        correction_rate=mean(bool(row.user_corrections) for row in rows),
        regression_signals=sum(bool(row.critical_regression) for row in rows),
    )


def release_affected_skills(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Skill names changed by a release, from its signed ``changed_paths``."""
    paths = manifest.get("changed_paths", [])
    if not isinstance(paths, list):
        raise ValueError("release manifest changed_paths must be a list")
    names = set()
    for path in paths:
        if not isinstance(path, str):
            raise ValueError("release manifest changed_paths must contain strings")
        parts = PurePosixPath(path).parts
        if len(parts) >= 3 and parts[:2] == ("brain", "skills"):
            names.add(parts[2])
    return frozenset(names)


def _matches(skills: tuple[str, ...], affected: frozenset[str]) -> bool:
    return not skills or not affected or bool(set(skills) & affected)


def validate_observation(observation: ProductionObservation) -> None:
    counters = (
        observation.tool_errors, observation.tokens, observation.tool_calls,
        observation.user_corrections, observation.retries,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value > (1 << 63) - 1
        for value in counters
    ):
        raise ValueError("monitoring counters must be non-negative integers")
    if (
        not isinstance(observation.release_id, str) or not observation.release_id.strip()
        or not isinstance(observation.task_id, str) or not observation.task_id.strip()
    ):
        raise ValueError("monitoring release_id and task_id must be non-empty strings")
    if not isinstance(observation.success, bool) or not isinstance(observation.critical_regression, bool):
        raise ValueError("monitoring status fields must be boolean")
    if (
        not isinstance(observation.observation_id, str)
        or len(observation.observation_id) > 200
        or (observation.observation_id and not observation.observation_id.strip())
    ):
        raise ValueError("observation_id must be empty or a non-blank string of at most 200 characters")
    if observation.source not in OBSERVATION_SOURCES:
        raise ValueError(f"observation source must be one of: {', '.join(sorted(OBSERVATION_SOURCES))}")
    if (
        not isinstance(observation.skills, tuple) or len(observation.skills) > _MAX_OBSERVATION_SKILLS
        or any(not isinstance(name, str) or not _SKILL_NAME.fullmatch(name) for name in observation.skills)
        or contains_secret_material(list(observation.skills))
    ):
        raise ValueError("observation skills must be a tuple of at most 32 brain skill directory names")
    try:
        valid_latency = (
            not isinstance(observation.latency_ms, bool)
            and isinstance(observation.latency_ms, (int, float))
            and math.isfinite(observation.latency_ms)
            and observation.latency_ms >= 0
        )
    except OverflowError:
        valid_latency = False
    if not valid_latency:
        raise ValueError("latency must be a finite non-negative number")


def _main_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), "rev-parse", "--verify", "refs/heads/main"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if result.returncode:
        raise RuntimeError(f"cannot read the production Git ref: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout.decode().strip()
