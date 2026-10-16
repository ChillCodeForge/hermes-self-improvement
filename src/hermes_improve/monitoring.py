from __future__ import annotations

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .models import utc_now
from .store import ExperimentStore


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


@dataclass(frozen=True)
class MonitoringDecision:
    status: str
    sample_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class MonitoringPolicy:
    min_observations: int = 20
    max_success_rate_drop: float = 0.05
    max_tool_error_rate_increase: float = 0.05
    max_tokens_per_success_increase: float = 0.15
    max_latency_increase: float = 0.15
    max_correction_rate_increase: float = 0.05

    def __post_init__(self) -> None:
        if isinstance(self.min_observations, bool) or self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        thresholds = (
            self.max_success_rate_drop,
            self.max_tool_error_rate_increase,
            self.max_tokens_per_success_increase,
            self.max_latency_increase,
            self.max_correction_rate_increase,
        )
        if any(not math.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("monitoring thresholds must be finite and non-negative")


class ReleaseMonitor:
    """Records post-release evidence and recommends stabilize or rollback."""

    def __init__(self, path: Path, store: ExperimentStore, policy: MonitoringPolicy = MonitoringPolicy()) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.policy = policy
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

    def observe(self, observation: ProductionObservation) -> MonitoringDecision:
        counters = (
            observation.tool_errors, observation.tokens, observation.tool_calls,
            observation.user_corrections, observation.retries,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise ValueError("monitoring counters must be non-negative integers")
        if not isinstance(observation.task_id, str) or not observation.task_id.strip():
            raise ValueError("monitoring task_id must be a non-empty string")
        if not isinstance(observation.success, bool) or not isinstance(observation.critical_regression, bool):
            raise ValueError("monitoring status fields must be boolean")
        if (
            isinstance(observation.latency_ms, bool)
            or not isinstance(observation.latency_ms, (int, float))
            or not math.isfinite(observation.latency_ms)
            or observation.latency_ms < 0
        ):
            raise ValueError("latency must be a finite non-negative number")
        release = self.store.get_release(observation.release_id)
        if release["status"] != "PROVISIONAL":
            raise ValueError("post-promotion observations are accepted only for provisional releases")
        baseline = release["manifest"].get("baseline_metrics")
        if not isinstance(baseline, dict):
            raise ValueError("release manifest does not include comparable baseline metrics")
        with sqlite3.connect(self.path) as db:
            db.execute(
                """INSERT INTO production_observations
                (id,release_id,task_id,success,tool_errors,tokens,tool_calls,latency_ms,user_corrections,retries,critical_regression,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uuid.uuid4().hex, observation.release_id, observation.task_id, int(observation.success),
                 observation.tool_errors, observation.tokens, observation.tool_calls, observation.latency_ms,
                 observation.user_corrections, observation.retries, int(observation.critical_regression), utc_now()),
            )
            rows = db.execute(
                """SELECT success,tool_errors,tokens,tool_calls,latency_ms,user_corrections,retries,critical_regression
                   FROM production_observations WHERE release_id=?""",
                (observation.release_id,),
            ).fetchall()
        sample_count = len(rows)
        if any(row[7] for row in rows):
            return MonitoringDecision("ROLLBACK_RECOMMENDED", sample_count, ("critical regression observed",))
        if sample_count < self.policy.min_observations:
            return MonitoringDecision("PROVISIONAL", sample_count, ("minimum observation sample not reached",))
        success_rate = mean(row[0] for row in rows)
        total_success = sum(row[0] for row in rows)
        tokens_per_success = sum(row[2] for row in rows) / total_success if total_success else float("inf")
        correction_rate = mean(bool(row[5]) for row in rows)
        error_rate = sum(row[1] for row in rows) / max(1, sum(row[3] for row in rows))
        latency = mean(row[4] for row in rows)
        reasons = []
        if success_rate < baseline.get("success_rate", 0) - self.policy.max_success_rate_drop:
            reasons.append("success rate regression")
        if error_rate > baseline.get("tool_error_rate", 0) + self.policy.max_tool_error_rate_increase:
            reasons.append("tool error rate regression")
        if tokens_per_success > baseline.get("tokens_per_success", float("inf")) * (1 + self.policy.max_tokens_per_success_increase):
            reasons.append("tokens per success regression")
        if latency > baseline.get("latency_per_success_ms", float("inf")) * (1 + self.policy.max_latency_increase):
            reasons.append("latency regression")
        if correction_rate > baseline.get("correction_rate", 0) + self.policy.max_correction_rate_increase:
            reasons.append("user correction rate regression")
        if reasons:
            return MonitoringDecision("ROLLBACK_RECOMMENDED", sample_count, tuple(reasons))
        self.store.update_release_status(observation.release_id, "STABLE")
        return MonitoringDecision("STABLE", sample_count, ())
