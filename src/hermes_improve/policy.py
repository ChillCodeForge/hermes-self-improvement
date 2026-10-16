from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
from statistics import mean
from typing import Iterable

from .models import CandidateType, EvaluationReport, Split, TaskOutcome
from .statistics import paired_mean_confidence_interval


@dataclass(frozen=True)
class PromotionPolicy:
    version: str = "1"
    max_medium_regression_rate: float = 0.01
    min_quality_delta: float = 3.0
    max_category_drop: float = 2.0
    max_token_increase_ratio: float = 0.10
    max_latency_increase_ratio: float = 0.15
    auto_promotable_types: frozenset[CandidateType] = frozenset({
        CandidateType.SKILL_CREATE,
        CandidateType.SKILL_PATCH,
        CandidateType.SKILL_MERGE,
        CandidateType.SKILL_SPLIT,
    })

    def assess(
        self,
        candidate_id: str,
        candidate_hash: str,
        benchmark_version: str,
        outcomes: Iterable[TaskOutcome],
    ) -> EvaluationReport:
        rows = tuple(outcomes)
        for row in rows:
            if (
                not isinstance(row.task_id, str) or not row.task_id.strip()
                or not isinstance(row.suite, str) or not row.suite.strip()
                or not isinstance(row.split, Split)
            ):
                raise ValueError("task outcomes require valid task, suite, and split identities")
            numeric_metrics = (
                row.baseline_quality, row.candidate_quality,
                row.baseline_latency_ms, row.candidate_latency_ms,
            )
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numeric_metrics
            ) or row.baseline_latency_ms < 0 or row.candidate_latency_ms < 0:
                raise ValueError("task outcomes require finite quality and latency metrics")
            counters = (
                row.baseline_tokens, row.candidate_tokens, row.critical_regressions,
                row.high_regressions, row.medium_regressions, row.baseline_tool_calls,
                row.candidate_tool_calls, row.baseline_failed_tool_calls,
                row.candidate_failed_tool_calls, row.baseline_retries, row.candidate_retries,
                row.baseline_user_corrections, row.candidate_user_corrections,
            )
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
                raise ValueError("task outcome counters must be non-negative integers")
            if (
                not isinstance(row.baseline_success, bool)
                or not isinstance(row.candidate_success, bool)
                or not isinstance(row.deterministic_pass, bool)
                or not isinstance(row.security_pass, bool)
            ):
                raise ValueError("task outcome statuses must be boolean")
            if not isinstance(row.categories, (tuple, list)) or not all(
                isinstance(category, str) for category in row.categories
            ):
                raise ValueError("task outcome categories must be strings")
        reasons: list[str] = []
        final_rows = tuple(
            row for row in rows
            if row.suite.upper() == "FULL" or row.split is Split.SEALED_HOLDOUT
        )
        if not rows:
            reasons.append("no task outcomes supplied")
        if any(not row.security_pass for row in rows):
            reasons.append("security verifier failed")
        if any(not row.deterministic_pass for row in rows):
            reasons.append("deterministic verifier failed")
        if any(row.critical_regressions for row in rows):
            reasons.append("critical regression detected")
        if any(row.high_regressions for row in rows):
            reasons.append("high regression detected")
        if final_rows:
            if not any(row.suite.upper() == "FULL" and row.split is not Split.SEALED_HOLDOUT for row in final_rows):
                reasons.append("full-suite evidence is missing")
            if not any(row.split is Split.SEALED_HOLDOUT for row in final_rows):
                reasons.append("sealed holdout evidence is missing")
            medium = sum(row.medium_regressions for row in final_rows)
            medium_rate = medium / len(final_rows)
            if medium_rate > self.max_medium_regression_rate:
                reasons.append(f"medium regression rate {medium_rate:.3%} exceeds policy")
            quality_delta = mean(row.candidate_quality - row.baseline_quality for row in final_rows)
            quality_interval = paired_mean_confidence_interval(
                [row.candidate_quality - row.baseline_quality for row in final_rows]
            )
            if quality_delta < self.min_quality_delta:
                reasons.append(f"quality delta {quality_delta:.3f} is below minimum")
            if quality_interval[0] < 0:
                reasons.append(f"95% quality delta interval includes regression: {quality_interval}")
            category_deltas: dict[str, list[float]] = defaultdict(list)
            for row in final_rows:
                for category in row.categories:
                    category_deltas[category].append(row.candidate_quality - row.baseline_quality)
            category_drops = {
                name: mean(deltas)
                for name, deltas in category_deltas.items()
                if mean(deltas) < -self.max_category_drop
            }
            if category_drops:
                reasons.append(f"category quality drop exceeds policy: {category_drops}")
            baseline_tokens = sum(row.baseline_tokens for row in final_rows)
            candidate_tokens = sum(row.candidate_tokens for row in final_rows)
            if baseline_tokens and candidate_tokens > baseline_tokens * (1 + self.max_token_increase_ratio):
                reasons.append("token increase exceeds policy")
            baseline_latency = sum(row.baseline_latency_ms for row in final_rows)
            candidate_latency = sum(row.candidate_latency_ms for row in final_rows)
            latency_excesses = [
                row.candidate_latency_ms
                - row.baseline_latency_ms * (1 + self.max_latency_increase_ratio)
                for row in final_rows
            ]
            latency_excess_ci = paired_mean_confidence_interval(latency_excesses)
            if baseline_latency and latency_excess_ci[0] > 0:
                reasons.append("latency increase exceeds policy")
            if not all(row.candidate_success for row in final_rows):
                reasons.append("candidate failed at least one final-suite task")
        passed = not reasons and bool(final_rows)
        return EvaluationReport(
            candidate_id=candidate_id,
            candidate_hash=candidate_hash,
            benchmark_version=benchmark_version,
            policy_version=self.version,
            outcomes=rows,
            passed=passed,
            reasons=tuple(reasons),
            metrics={
                "final_tasks": len(final_rows),
                "quality_delta_mean": mean(
                    row.candidate_quality - row.baseline_quality for row in final_rows
                ) if final_rows else None,
                "quality_delta_ci95": list(quality_interval) if final_rows else None,
                "baseline_success_rate": (
                    sum(row.baseline_success for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "candidate_success_rate": (
                    sum(row.candidate_success for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "tokens_per_task_baseline": (
                    sum(row.baseline_tokens for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "tokens_per_task_candidate": (
                    sum(row.candidate_tokens for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "latency_per_task_baseline_ms": (
                    sum(row.baseline_latency_ms for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "latency_per_task_candidate_ms": (
                    sum(row.candidate_latency_ms for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "latency_excess_ci95_ms": list(latency_excess_ci) if final_rows else None,
                "baseline_tool_error_rate": (
                    sum(row.baseline_failed_tool_calls for row in final_rows)
                    / max(1, sum(row.baseline_tool_calls for row in final_rows))
                    if final_rows else None
                ),
                "candidate_tool_error_rate": (
                    sum(row.candidate_failed_tool_calls for row in final_rows)
                    / max(1, sum(row.candidate_tool_calls for row in final_rows))
                    if final_rows else None
                ),
                "baseline_correction_rate": (
                    sum(bool(row.baseline_user_corrections) for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "candidate_correction_rate": (
                    sum(bool(row.candidate_user_corrections) for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "baseline_tokens_per_success": (
                    sum(row.baseline_tokens for row in final_rows)
                    / max(1, sum(row.baseline_success for row in final_rows))
                    if final_rows else None
                ),
                "candidate_tokens_per_success": (
                    sum(row.candidate_tokens for row in final_rows)
                    / max(1, sum(row.candidate_success for row in final_rows))
                    if final_rows else None
                ),
                "baseline_latency_per_success_ms": (
                    sum(row.baseline_latency_ms for row in final_rows)
                    / max(1, sum(row.baseline_success for row in final_rows))
                    if final_rows else None
                ),
                "candidate_latency_per_success_ms": (
                    sum(row.candidate_latency_ms for row in final_rows)
                    / max(1, sum(row.candidate_success for row in final_rows))
                    if final_rows else None
                ),
            },
        )

    def authorize_type(self, candidate_type: CandidateType) -> None:
        if candidate_type not in self.auto_promotable_types:
            raise ValueError(f"{candidate_type.value} requires a manual promotion gate")
