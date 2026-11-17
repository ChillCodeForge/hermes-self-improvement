from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import math
from statistics import mean
from typing import TYPE_CHECKING, Iterable

from .models import CandidateType, EvaluationReport, Split, TaskOutcome
from .statistics import paired_mean_confidence_interval

if TYPE_CHECKING:
    from .config import PromotionConfig


#: Hard constraints in the order of spec §23. ``assess`` sorts its reasons by
#: this order and reports the first failed one as ``metrics["primary_reason"]``.
#: ``medium_regressions`` belongs to the regression-severity group after high
#: regressions. ``evidence`` (no outcomes, missing Full or sealed holdout rows)
#: is a precondition for the final-suite constraints; it follows the regression
#: constraints because security and regression findings apply to any suite.
CONSTRAINT_ORDER: tuple[str, ...] = (
    "security",
    "deterministic_correctness",
    "critical_regressions",
    "high_regressions",
    "medium_regressions",
    "evidence",
    "task_success",
    "generalization",
    "category_regressions",
    "quality",
    "token_efficiency",
    "latency",
)


@dataclass(frozen=True)
class PromotionPolicy:
    version: str = "1"
    max_critical_regressions: int = 0
    max_high_regressions: int = 0
    max_medium_regression_rate: float = 0.01
    min_quality_delta: float = 3.0
    max_category_drop: float = 2.0
    max_token_increase_ratio: float = 0.10
    max_latency_increase_ratio: float = 0.15
    # Generalization (§23): the sealed holdout alone must reach this mean
    # quality delta and must not lose task success; Full rows cannot carry a
    # holdout regression through the pooled quality mean.
    min_holdout_quality_delta: float = 0.0
    # §24: token efficiency is a hard constraint. When any final row lacks
    # reported token usage the constraint cannot be evaluated; ``True`` makes
    # that a blocking reason instead of a silent pass.
    require_token_measurement: bool = True
    # §23 exceptions: a token increase above ``max_token_increase_ratio`` or a
    # latency increase above ``max_latency_increase_ratio`` is allowed only with
    # a significant quality gain: at least two final tasks (so the interval has
    # a variance estimate) and a pooled Full+holdout 95% CI lower bound of the
    # quality delta strictly above this value. ``None`` disables the exception.
    token_exception_min_quality_ci_lower: float | None = 3.0
    latency_exception_min_quality_ci_lower: float | None = 3.0
    auto_promotable_types: frozenset[CandidateType] = frozenset({
        CandidateType.SKILL_CREATE,
        CandidateType.SKILL_PATCH,
        CandidateType.SKILL_MERGE,
        CandidateType.SKILL_SPLIT,
    })
    required_evidence_suites: tuple[str, ...] = ("SMOKE", "CORE", "FULL", "SEALED_HOLDOUT")

    def __post_init__(self) -> None:
        suites = self.required_evidence_suites
        if (
            not isinstance(suites, tuple)
            or any(item not in {"SMOKE", "CORE", "FULL", "SEALED_HOLDOUT"} for item in suites)
            or not {"FULL", "SEALED_HOLDOUT"} <= set(suites)
        ):
            raise ValueError("required evidence suites must include FULL and SEALED_HOLDOUT")
        if not isinstance(self.auto_promotable_types, frozenset) or not all(
            isinstance(item, CandidateType) for item in self.auto_promotable_types
        ):
            raise ValueError("auto-promotable types must be a frozenset of CandidateType")
        for name in ("max_critical_regressions", "max_high_regressions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.require_token_measurement, bool):
            raise ValueError("require_token_measurement must be a boolean")
        for name, optional in (
            ("min_holdout_quality_delta", False),
            ("token_exception_min_quality_ci_lower", True),
            ("latency_exception_min_quality_ci_lower", True),
        ):
            value = getattr(self, name)
            if value is None and optional:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")

    @classmethod
    def from_config(cls, config: "PromotionConfig") -> "PromotionPolicy":
        """Build the policy from protected configuration (``config/promotion.yaml``)."""
        return cls(
            version=config.policy_version,
            max_critical_regressions=config.max_critical_regressions,
            max_high_regressions=config.max_high_regressions,
            max_medium_regression_rate=config.max_medium_regression_rate,
            min_quality_delta=config.min_quality_delta,
            max_category_drop=config.max_category_drop,
            max_token_increase_ratio=config.max_token_increase_ratio,
            max_latency_increase_ratio=config.max_latency_increase_ratio,
            min_holdout_quality_delta=config.min_holdout_quality_delta,
            require_token_measurement=config.require_token_measurement,
            token_exception_min_quality_ci_lower=config.token_exception_min_quality_ci_lower,
            latency_exception_min_quality_ci_lower=config.latency_exception_min_quality_ci_lower,
            auto_promotable_types=config.auto_promotable_types,
            required_evidence_suites=tuple(config.required_evidence_suites),
        )

    def requires_manual_approval(
        self, candidate_type: CandidateType, *, removed_skills: Iterable[str] = (),
    ) -> bool:
        """True when promotion needs a signed operator approval.

        A candidate that removes a production skill (its diff deletes a
        ``brain/skills/<name>/SKILL.md``, see ``target_paths.removed_skill_units``)
        is never auto-promotable, whatever its type gate says: archive, delete,
        merge and split removals all go through the manual gate. The removal
        set is derived from the candidate diff, not from the declared type.
        """
        if not isinstance(candidate_type, CandidateType):
            raise ValueError("candidate type must be a CandidateType")
        if tuple(removed_skills):
            return True
        return candidate_type not in self.auto_promotable_types

    @staticmethod
    def _validate(rows: tuple[TaskOutcome, ...]) -> None:
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
                or not isinstance(row.token_usage_reported, bool)
            ):
                raise ValueError("task outcome statuses must be boolean")
            if not isinstance(row.categories, (tuple, list)) or not all(
                isinstance(category, str) for category in row.categories
            ):
                raise ValueError("task outcome categories must be strings")

    @staticmethod
    def _significant_gain(
        threshold: float | None, deltas: list[float], interval: tuple[float, float],
    ) -> bool:
        """Quality gain significant enough to justify a token or latency increase."""
        return threshold is not None and len(deltas) >= 2 and interval[0] > threshold

    def assess(
        self,
        candidate_id: str,
        candidate_hash: str,
        benchmark_version: str,
        outcomes: Iterable[TaskOutcome],
    ) -> EvaluationReport:
        rows = tuple(outcomes)
        self._validate(rows)
        failures: list[tuple[str, str]] = []

        def fail(constraint: str, reason: str) -> None:
            failures.append((constraint, reason))

        final_rows = tuple(
            row for row in rows
            if row.suite.upper() == "FULL" or row.split is Split.SEALED_HOLDOUT
        )
        holdout_rows = tuple(row for row in final_rows if row.split is Split.SEALED_HOLDOUT)
        deltas = [row.candidate_quality - row.baseline_quality for row in final_rows]
        quality_interval = paired_mean_confidence_interval(deltas) if final_rows else None
        tokens_reported = bool(final_rows) and all(row.token_usage_reported for row in final_rows)
        token_check = "not evaluated"
        token_exception_applied = False
        latency_exception_applied = False
        latency_excess_ci: tuple[float, float] | None = None
        holdout_delta: float | None = None

        # 1. Security, on every row of every suite.
        if any(not row.security_pass for row in rows):
            fail("security", "security verifier failed")
        # 2. Deterministic correctness.
        if any(not row.deterministic_pass for row in rows):
            fail("deterministic_correctness", "deterministic verifier failed")
        # 3./4. Critical and high regressions, counted over every row.
        if sum(row.critical_regressions for row in rows) > self.max_critical_regressions:
            fail("critical_regressions", "critical regression detected")
        if sum(row.high_regressions for row in rows) > self.max_high_regressions:
            fail("high_regressions", "high regression detected")
        if final_rows:
            medium_rate = sum(row.medium_regressions for row in final_rows) / len(final_rows)
            if medium_rate > self.max_medium_regression_rate:
                fail("medium_regressions", f"medium regression rate {medium_rate:.3%} exceeds policy")
        # Evidence preconditions for the final-suite constraints.
        if not rows:
            fail("evidence", "no task outcomes supplied")
        elif final_rows:
            if not any(row.suite.upper() == "FULL" and row.split is not Split.SEALED_HOLDOUT for row in final_rows):
                fail("evidence", "full-suite evidence is missing")
            if not holdout_rows:
                fail("evidence", "sealed holdout evidence is missing")
        # 5. Task success must not regress: no task the baseline solved may be
        #    lost, checked per task on every suite so a gain elsewhere cannot
        #    offset it. A task both arms fail is not a regression; it is left to
        #    the quality gate (and a cheaper failure is never an improvement).
        regressed = sorted({row.task_id for row in rows if row.baseline_success and not row.candidate_success})
        if regressed:
            fail("task_success", f"task success regressed on {len(regressed)} task(s): {', '.join(regressed)}")
        if final_rows:
            # 6. Generalization: the sealed holdout on its own.
            if holdout_rows:
                holdout_delta = mean(row.candidate_quality - row.baseline_quality for row in holdout_rows)
                if holdout_delta < self.min_holdout_quality_delta:
                    fail(
                        "generalization",
                        f"sealed holdout quality delta {holdout_delta:.3f} is below the generalization "
                        f"minimum {self.min_holdout_quality_delta:.3f}",
                    )
                if (
                    sum(row.candidate_success for row in holdout_rows)
                    < sum(row.baseline_success for row in holdout_rows)
                ):
                    fail("generalization", "sealed holdout task success rate regressed")
            # 7. Category regressions (pooled Full + holdout).
            category_deltas: dict[str, list[float]] = defaultdict(list)
            for row in final_rows:
                for category in row.categories:
                    category_deltas[category].append(row.candidate_quality - row.baseline_quality)
            category_drops = {
                name: mean(values)
                for name, values in category_deltas.items()
                if mean(values) < -self.max_category_drop
            }
            if category_drops:
                fail("category_regressions", f"category quality drop exceeds policy: {category_drops}")
            # 8. Quality (pooled Full + holdout).
            quality_delta = mean(deltas)
            if quality_delta < self.min_quality_delta:
                fail("quality", f"quality delta {quality_delta:.3f} is below minimum")
            if quality_interval[0] < 0:
                fail("quality", f"95% quality delta interval includes regression: {quality_interval}")
            # 9. Token efficiency.
            if not tokens_reported:
                if self.require_token_measurement:
                    token_check = "not measured"
                    fail("token_efficiency", "token efficiency not measured: runner reported no token usage")
                else:
                    token_check = "skipped: not measured and require_token_measurement is false"
            else:
                baseline_tokens = sum(row.baseline_tokens for row in final_rows)
                candidate_tokens = sum(row.candidate_tokens for row in final_rows)
                token_check = "passed"
                if candidate_tokens > baseline_tokens * (1 + self.max_token_increase_ratio):
                    if self._significant_gain(self.token_exception_min_quality_ci_lower, deltas, quality_interval):
                        token_check = "exceeded; allowed by significant quality gain"
                        token_exception_applied = True
                    else:
                        token_check = "failed"
                        fail("token_efficiency", "token increase exceeds policy")
            # 10. Latency: CI of the per-task excess over the allowed increase.
            baseline_latency = sum(row.baseline_latency_ms for row in final_rows)
            latency_excess_ci = paired_mean_confidence_interval([
                row.candidate_latency_ms - row.baseline_latency_ms * (1 + self.max_latency_increase_ratio)
                for row in final_rows
            ])
            if baseline_latency and latency_excess_ci[0] > 0:
                if self._significant_gain(self.latency_exception_min_quality_ci_lower, deltas, quality_interval):
                    latency_exception_applied = True
                else:
                    fail("latency", "latency increase exceeds policy")

        order = {name: index for index, name in enumerate(CONSTRAINT_ORDER)}
        failures.sort(key=lambda item: order[item[0]])  # stable within one constraint
        reasons = [reason for _, reason in failures]
        passed = not reasons and bool(final_rows)

        def per_task(total: float) -> float | None:
            return total / len(final_rows) if final_rows else None

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
                "constraint_order": list(CONSTRAINT_ORDER),
                "failed_constraints": list(dict.fromkeys(constraint for constraint, _ in failures)),
                "primary_reason": reasons[0] if reasons else None,
                "quality_delta_mean": mean(deltas) if final_rows else None,
                "quality_delta_ci95": list(quality_interval) if final_rows else None,
                "holdout_tasks": len(holdout_rows),
                "holdout_quality_delta_mean": holdout_delta,
                "baseline_success_rate": (
                    sum(row.baseline_success for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "candidate_success_rate": (
                    sum(row.candidate_success for row in final_rows) / len(final_rows)
                    if final_rows else None
                ),
                "token_usage_reported": tokens_reported,
                "token_check": token_check,
                "token_exception_applied": token_exception_applied,
                "latency_exception_applied": latency_exception_applied,
                "tokens_per_task_baseline": (
                    per_task(sum(row.baseline_tokens for row in final_rows)) if tokens_reported else None
                ),
                "tokens_per_task_candidate": (
                    per_task(sum(row.candidate_tokens for row in final_rows)) if tokens_reported else None
                ),
                "latency_per_task_baseline_ms": per_task(sum(row.baseline_latency_ms for row in final_rows)),
                "latency_per_task_candidate_ms": per_task(sum(row.candidate_latency_ms for row in final_rows)),
                "latency_excess_ci95_ms": list(latency_excess_ci) if latency_excess_ci is not None else None,
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
                    if tokens_reported else None
                ),
                "candidate_tokens_per_success": (
                    sum(row.candidate_tokens for row in final_rows)
                    / max(1, sum(row.candidate_success for row in final_rows))
                    if tokens_reported else None
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
