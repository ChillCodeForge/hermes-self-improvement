from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from statistics import mean
from typing import Iterable

from .models import CandidateType, EvaluationReport, Split, TaskOutcome


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
            if quality_delta < self.min_quality_delta:
                reasons.append(f"quality delta {quality_delta:.3f} is below minimum")
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
            if baseline_latency and candidate_latency > baseline_latency * (1 + self.max_latency_increase_ratio):
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
        )

    def authorize_type(self, candidate_type: CandidateType) -> None:
        if candidate_type not in self.auto_promotable_types:
            raise ValueError(f"{candidate_type.value} requires a manual promotion gate")
