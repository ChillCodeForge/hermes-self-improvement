from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Any, Iterable, Mapping, Protocol

from .artifacts import ArtifactStore
from .benchmarks import BenchmarkStore
from .errors import IntegrityError
from .evaluation_signing import sign_report
from .deliverables import declared_deliverables, validate_deliverables
from .judge import JudgeVerdict, PairwiseJudge, aggregate_verdicts
from .metrics import efficiency_report
from .models import (
    EVALUATION_SCHEMA_VERSION,
    Benchmark,
    Candidate,
    Eligibility,
    EvaluationReport,
    Role,
    RunObservation,
    Split,
    TaskOutcome,
)
from .policy import PromotionPolicy
from .run_policy import RunPolicy
from .security import HoldoutGrant
from .skill_use import skill_use_report, target_loaded_rate, target_skill_aliases, valid_skill_use
from .statistics import paired_summary
from .store import ExperimentStore
from .verifier import has_deterministic_verification


class BenchmarkRunner(Protocol):
    """Runtime adapter contract; implementations must provide actual isolation.

    ``llm_influenced`` declares whether a model is in the loop. Runners that
    omit it are treated as LLM-influenced (more runs), never as deterministic.
    """

    llm_influenced: bool

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation: ...


def _subjective_criteria(benchmark: Benchmark) -> list[str]:
    criteria = benchmark.verification.get("subjective", [])
    if not isinstance(criteria, list):
        raise ValueError(f"benchmark {benchmark.id!r} verification.subjective must be an array")
    rendered = []
    for item in criteria:
        if isinstance(item, str) and item.strip():
            rendered.append(item)
        elif isinstance(item, dict) and isinstance(item.get("criterion"), str) and item["criterion"].strip():
            rendered.append(item["criterion"])
        else:
            raise ValueError(f"benchmark {benchmark.id!r} has an invalid subjective criterion")
    return rendered


JUDGE_ROUTES = ("deterministic_verification", "qwen_judge", "exit_code_only")
NEEDS_QWEN_RULES_VERSION = "needs-qwen-rules-1"


def judge_route(benchmark: Benchmark) -> str:
    """How a task is verified: deterministic checks win; the Qwen judge only for subjective-only tasks."""
    if has_deterministic_verification(benchmark):
        return "deterministic_verification"
    if _subjective_criteria(benchmark):
        return "qwen_judge"
    return "exit_code_only"


def needs_qwen_decision_fields(benchmark: Benchmark, *, suite: str, judge_configured: bool) -> dict[str, Any]:
    """Decision-history fields (spec §36) for the needs-Qwen decision on one task.

    Uses the same :func:`judge_route` the evaluator uses, so the recorded
    decision is the one that is applied. Sealed holdout tasks are recorded
    without their ID. Rule-based, so there is no calibrated confidence.
    """
    checks = benchmark.verification.get("deterministic", [])
    features: dict[str, Any] = {
        "split": benchmark.split.value,
        "suite": "SEALED_HOLDOUT" if benchmark.split is Split.SEALED_HOLDOUT else suite.upper(),
        "deterministic_checks": len(checks) if isinstance(checks, list) else 0,
        "subjective_criteria": len(_subjective_criteria(benchmark)),
        "judge_configured": judge_configured,
    }
    if benchmark.split is not Split.SEALED_HOLDOUT:
        features["task_id"] = benchmark.id
        features["task_version"] = benchmark.version
    return {
        "decision_type": "needs_qwen",
        "input_features": features,
        "available_options": JUDGE_ROUTES,
        "selected_option": judge_route(benchmark),
        "confidence": None,
        "source": "evaluator-verification-rules",
        "model_version": NEEDS_QWEN_RULES_VERSION,
    }


class PairedEvaluator:
    """Pairs baseline/candidate runs on identical tasks, seeds, environment and limits.

    The evaluator orchestrates an injected runner; it does not itself sandbox
    candidate execution. Deterministic verifier results always take precedence:
    the judge is only consulted for benchmarks without deterministic checks
    that declare subjective criteria. A `baseline` or `unreliable` verdict
    adds a blocking reason here; the policy leaves judged tasks out of the
    deterministic quality mean and lets a `candidate` verdict carry quality
    only into the signed manual approval gate, never into auto-promotion.
    """

    def __init__(
        self,
        benchmark_store: BenchmarkStore,
        runner: BenchmarkRunner,
        policy: PromotionPolicy,
        experiment_store: ExperimentStore | None = None,
        evaluator_signing_key: bytes | None = None,
        artifact_store: ArtifactStore | None = None,
        *,
        run_policy: RunPolicy | None = None,
        judge: PairwiseJudge | None = None,
    ) -> None:
        self.benchmarks = benchmark_store
        self.runner = runner
        self.policy = policy
        self.experiment_store = experiment_store
        self.evaluator_signing_key = evaluator_signing_key
        self.artifact_store = artifact_store
        self.run_policy = run_policy or RunPolicy()
        self.judge = judge
        if experiment_store is not None and evaluator_signing_key is None:
            raise ValueError("persisted evaluations require an evaluator Ed25519 signing key")

    @property
    def llm_influenced(self) -> bool:
        value = getattr(self.runner, "llm_influenced", True)
        if not isinstance(value, bool):
            raise TypeError("runner.llm_influenced must be a boolean")
        return value

    def evaluate(
        self,
        candidate: Candidate,
        *,
        baseline_commit: str,
        benchmark_ids: Iterable[str],
        suite: str,
        benchmark_version: str,
        seeds: Iterable[int] | None = None,
        holdout_grant: HoldoutGrant | None = None,
        evaluation_id: str | None = None,
        suite_selection: Mapping[str, Any] | None = None,
    ) -> EvaluationReport:
        if candidate.base_commit != baseline_commit:
            raise IntegrityError("candidate base commit does not match requested baseline")
        explicit_seeds: tuple[int, ...] | None = None
        if seeds is not None:
            supplied_seeds = tuple(seeds)
            if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in supplied_seeds):
                raise ValueError("evaluation seeds must be non-negative integers")
            explicit_seeds = tuple(dict.fromkeys(supplied_seeds))
            if not explicit_seeds:
                raise ValueError("at least one deterministic seed is required")
        if not suite.strip():
            raise ValueError("suite name cannot be empty")
        if benchmark_version != self.benchmarks.version:
            raise IntegrityError("requested benchmark version does not match the current validated benchmark manifest")
        if self.experiment_store is not None:
            # Refuse before any run: promotion only accepts evaluations under the
            # benchmark version the experiment was bound to at creation.
            try:
                experiment = self.experiment_store.show_experiment(candidate.experiment_id)
            except KeyError:
                experiment = None
            if experiment is not None:
                self.benchmarks.require_bound_version(candidate.experiment_id, experiment["benchmark_version"])
        requested_ids = tuple(benchmark_ids)
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("benchmark IDs must be unique within an evaluation")
        benchmarks = [
            self.benchmarks.get(item, role=Role.EVALUATOR, grant=holdout_grant)
            for item in requested_ids
        ]
        if not benchmarks:
            raise ValueError("at least one benchmark is required")
        ineligible = [
            benchmark.id for benchmark in benchmarks
            if Eligibility(benchmark.eligibility["class"]) is not Eligibility.HARD_BENCHMARK
        ]
        if ineligible:
            raise ValueError(
                "evaluation requires HARD_BENCHMARK tasks; ineligible tasks: "
                + ", ".join(ineligible)
            )
        llm_influenced = self.llm_influenced
        plan: dict[str, tuple[int, ...]] = {}
        for benchmark in benchmarks:
            allowed_suites = benchmark.benchmark.get("suites", [])
            if benchmark.split is not Split.SEALED_HOLDOUT and suite.upper() not in {
                str(item).upper() for item in allowed_suites
            }:
                raise ValueError(f"benchmark {benchmark.id!r} is not in suite {suite!r}")
            required = self.run_policy.runs_for(benchmark, suite, llm_influenced=llm_influenced)
            task_seeds = explicit_seeds if explicit_seeds is not None else self.run_policy.seeds(required)
            if len(task_seeds) < required:
                raise ValueError(
                    f"run policy requires at least {required} paired runs for {benchmark.id!r} "
                    f"in {suite.upper()}; got {len(task_seeds)} seeds"
                )
            plan[benchmark.id] = task_seeds
            if not has_deterministic_verification(benchmark) and _subjective_criteria(benchmark) and self.judge is None:
                raise ValueError(
                    f"benchmark {benchmark.id!r} relies on subjective verification and requires a configured judge"
                )

        observations: dict[tuple[str, str], list[RunObservation]] = defaultdict(list)
        all_observations: list[RunObservation] = []
        for benchmark in benchmarks:
            deterministic = has_deterministic_verification(benchmark)
            for seed in plan[benchmark.id]:
                baseline = self.runner.run(baseline_commit, benchmark, seed)
                candidate_run = self.runner.run(candidate.commit, benchmark, seed)
                self._validate_observation(baseline, benchmark, baseline_commit, seed)
                self._validate_observation(candidate_run, benchmark, candidate.commit, seed)
                if baseline.environment_digest != candidate_run.environment_digest:
                    raise IntegrityError("baseline and candidate ran in different environments")
                if baseline.model_config_digest != candidate_run.model_config_digest:
                    raise IntegrityError("baseline and candidate ran with different model configurations")
                if baseline.limits_digest != candidate_run.limits_digest:
                    raise IntegrityError("baseline and candidate ran with different sandbox limits")
                if deterministic and (not baseline.verifier_results or not candidate_run.verifier_results):
                    raise IntegrityError(
                        f"runner did not report deterministic verifier results for {benchmark.id!r}"
                    )
                declared = declared_deliverables(benchmark.verification)
                if any(
                    tuple(item["path"] for item in run.deliverables) != declared
                    for run in (baseline, candidate_run)
                ):
                    raise IntegrityError(
                        f"runner did not capture the deliverable files declared by {benchmark.id!r}"
                    )
                observations[(benchmark.id, "baseline")].append(baseline)
                observations[(benchmark.id, "candidate")].append(candidate_run)
                all_observations.extend((baseline, candidate_run))

        outcomes = []
        judgments: dict[str, JudgeVerdict] = {}
        skill_aliases = target_skill_aliases(candidate)
        for benchmark in benchmarks:
            baseline_runs = observations[(benchmark.id, "baseline")]
            candidate_runs = observations[(benchmark.id, "candidate")]
            if len(baseline_runs) != len(candidate_runs):
                raise IntegrityError(f"unpaired run count for benchmark {benchmark.id!r}")
            verdict = self._judge(benchmark, baseline_runs, candidate_runs)
            if verdict is not None:
                judgments[benchmark.id] = verdict
            outcomes.append(replace(
                self._outcome(benchmark, suite, baseline_runs, candidate_runs, verdict),
                baseline_skill_loaded_rate=target_loaded_rate(baseline_runs, skill_aliases),
                candidate_skill_loaded_rate=target_loaded_rate(candidate_runs, skill_aliases),
            ))
        report = self.policy.assess(candidate.candidate_id, candidate.content_hash, benchmark_version, outcomes)
        extra_reasons = []
        for task_id, verdict in sorted(judgments.items()):
            if verdict.result == "baseline":
                extra_reasons.append(f"judge preferred baseline on subjective task {task_id}")
            elif verdict.result == "unreliable":
                extra_reasons.append(f"subjective judgment unreliable on task {task_id}")
        pairs = [
            (baseline, candidate_run)
            for benchmark in benchmarks
            for baseline, candidate_run in zip(
                observations[(benchmark.id, "baseline")], observations[(benchmark.id, "candidate")]
            )
        ]
        metrics: dict[str, Any] = {
            **report.metrics,
            "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
            "suite": suite.upper(),
            "suite_selection": dict(suite_selection) if suite_selection is not None else None,
            "sealed_holdout_version": (
                self.benchmarks.holdout_version
                if any(benchmark.split is Split.SEALED_HOLDOUT for benchmark in benchmarks) else None
            ),
            "run_policy": {
                **self.run_policy.describe(),
                "llm_influenced": llm_influenced,
                "runs_per_task": {task_id: len(task_seeds) for task_id, task_seeds in sorted(plan.items())},
            },
            "paired_statistics": self._paired_statistics(benchmarks, observations),
            "efficiency": efficiency_report(pairs),
            "skill_use": skill_use_report(candidate, pairs),
            "judge": {
                "configured": self.judge.describe() if self.judge is not None else None,
                "verdicts": {task_id: verdict.to_dict() for task_id, verdict in sorted(judgments.items())},
            },
        }
        report = replace(
            report,
            passed=report.passed and not extra_reasons,
            reasons=report.reasons + tuple(extra_reasons),
            metrics=metrics,
            run_observations=tuple(sorted(
                all_observations, key=lambda item: (item.task_id, item.commit, item.seed)
            )),
        )
        if self.experiment_store is not None:
            if evaluation_id is None:
                raise ValueError("evaluation_id is required when persisting the evaluation")
            for observation in report.run_observations:
                for digest in observation.artifact_digests:
                    if self.artifact_store is None:
                        raise ValueError("persisted run artifacts require an ArtifactStore")
                    artifact_path = self.artifact_store.verify(digest)
                    self.experiment_store.record_artifact(digest, str(artifact_path))
            signature = sign_report(report, self.evaluator_signing_key)
            self.experiment_store.record_evaluation(
                evaluation_id, candidate.candidate_id, report, signature=signature,
                decides_experiment=(
                    suite.upper() in {"FULL", "SEALED_HOLDOUT"} or staged_regression(report.outcomes)
                ),
                regressions=detect_regressions(report.outcomes),
            )
        return report

    def _judge(
        self, benchmark: Benchmark, baseline_runs: list[RunObservation], candidate_runs: list[RunObservation]
    ) -> JudgeVerdict | None:
        if judge_route(benchmark) != "qwen_judge":
            return None
        criteria = _subjective_criteria(benchmark)
        if self.judge is None:
            raise ValueError(f"benchmark {benchmark.id!r} requires a configured judge")
        verdicts = [
            self.judge.compare(
                task_prompt=str(benchmark.task["prompt"]),
                criteria=criteria,
                baseline_output=baseline.final_output,
                candidate_output=candidate_run.final_output,
                baseline_files=baseline.deliverables,
                candidate_files=candidate_run.deliverables,
            )
            for baseline, candidate_run in zip(baseline_runs, candidate_runs)
        ]
        return aggregate_verdicts(verdicts, seeds=[baseline.seed for baseline in baseline_runs])

    @staticmethod
    def _outcome(
        benchmark: Benchmark,
        suite: str,
        baseline_runs: list[RunObservation],
        candidate_runs: list[RunObservation],
        verdict: JudgeVerdict | None,
    ) -> TaskOutcome:
        count = len(baseline_runs)
        return TaskOutcome(
            task_id=benchmark.id,
            split=benchmark.split,
            suite="SEALED_HOLDOUT" if benchmark.split is Split.SEALED_HOLDOUT else suite.upper(),
            baseline_success=all(run.success for run in baseline_runs),
            candidate_success=all(run.success for run in candidate_runs),
            baseline_quality=sum(run.quality for run in baseline_runs) / count,
            candidate_quality=sum(run.quality for run in candidate_runs) / count,
            baseline_tokens=sum(run.tokens for run in baseline_runs) // count,
            candidate_tokens=sum(run.tokens for run in candidate_runs) // count,
            deterministic_pass=all(run.deterministic_pass for run in candidate_runs),
            security_pass=all(run.security_pass for run in candidate_runs),
            critical_regressions=max(run.critical_regressions for run in candidate_runs),
            high_regressions=max(run.high_regressions for run in candidate_runs),
            medium_regressions=max(run.medium_regressions for run in candidate_runs),
            categories=tuple(str(value) for value in benchmark.taxonomy.get("categories", [])),
            baseline_latency_ms=sum(run.latency_ms for run in baseline_runs) / count,
            candidate_latency_ms=sum(run.latency_ms for run in candidate_runs) / count,
            baseline_tool_calls=sum(run.tool_calls for run in baseline_runs) // count,
            candidate_tool_calls=sum(run.tool_calls for run in candidate_runs) // count,
            baseline_failed_tool_calls=sum(run.failed_tool_calls for run in baseline_runs) // count,
            candidate_failed_tool_calls=sum(run.failed_tool_calls for run in candidate_runs) // count,
            baseline_retries=sum(run.retries for run in baseline_runs) // count,
            candidate_retries=sum(run.retries for run in candidate_runs) // count,
            baseline_user_corrections=sum(run.user_corrections for run in baseline_runs) // count,
            candidate_user_corrections=sum(run.user_corrections for run in candidate_runs) // count,
            run_count=count,
            judge_result=verdict.result if verdict is not None else "",
            judge_confidence=verdict.confidence if verdict is not None else None,
            judge_uncertainty=verdict.uncertainty if verdict is not None else None,
            token_usage_reported=all(run.token_usage_reported for run in (*baseline_runs, *candidate_runs)),
        )

    @staticmethod
    def _paired_statistics(
        benchmarks: list[Benchmark], observations: dict[tuple[str, str], list[RunObservation]]
    ) -> dict[str, Any]:
        """Per-task paired deltas (task means across runs), CI95 and Hedges' g_z."""

        def task_means(field: str, arm: str) -> list[float]:
            values = []
            for benchmark in benchmarks:
                runs = observations[(benchmark.id, arm)]
                values.append(sum(float(getattr(run, field)) for run in runs) / len(runs))
            return values

        summary = {
            name: paired_summary(task_means(field, "baseline"), task_means(field, "candidate"))
            for name, field in (
                ("quality", "quality"), ("success_rate", "success"), ("tokens", "tokens"),
                ("latency_ms", "latency_ms"), ("tool_calls", "tool_calls"),
            )
        }
        summary["unit"] = "per-task mean over paired runs; delta = candidate - baseline; no p-values"
        return summary

    @staticmethod
    def _validate_observation(
        observation: RunObservation, benchmark: Benchmark, commit: str, seed: int
    ) -> None:
        if observation.task_id != benchmark.id or observation.commit != commit or observation.seed != seed:
            raise IntegrityError("runner returned an observation for a different task, commit, or seed")
        if not observation.environment_digest:
            raise IntegrityError("runner did not identify the execution environment")
        counters = (
            observation.tokens, observation.tool_calls, observation.failed_tool_calls,
            observation.retries, observation.user_corrections, observation.context_tokens,
            observation.critical_regressions, observation.high_regressions,
            observation.medium_regressions, observation.seed, observation.input_tokens,
            observation.output_tokens, observation.context_chars, observation.turns,
            observation.api_calls,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise IntegrityError("runner returned invalid token or latency measurements")
        if observation.failed_tool_calls > observation.tool_calls:
            raise IntegrityError("runner reported more failed tool calls than tool calls")
        if (
            isinstance(observation.quality, bool) or not isinstance(observation.quality, (int, float))
            or not math.isfinite(observation.quality)
            or isinstance(observation.latency_ms, bool) or not isinstance(observation.latency_ms, (int, float))
            or not math.isfinite(observation.latency_ms) or observation.latency_ms < 0
        ):
            raise IntegrityError("runner returned invalid quality or latency measurements")
        if (
            not isinstance(observation.success, bool) or not isinstance(observation.deterministic_pass, bool)
            or not isinstance(observation.security_pass, bool)
            or not isinstance(observation.token_usage_reported, bool)
        ):
            raise IntegrityError("runner returned invalid verifier status")
        if not isinstance(observation.verifier_results, tuple) or not all(
            isinstance(item, dict) and isinstance(item.get("type"), str) and isinstance(item.get("passed"), bool)
            for item in observation.verifier_results
        ):
            raise IntegrityError("runner returned malformed verifier results")
        if not validate_deliverables(observation.deliverables):
            raise IntegrityError("runner returned malformed deliverable files")
        if not valid_skill_use(observation):
            raise IntegrityError("runner returned malformed skill-use measurements")
        if observation.verifier_results and observation.deterministic_pass != all(
            item["passed"] for item in observation.verifier_results
        ):
            raise IntegrityError("runner deterministic status contradicts its verifier results")
        if observation.success and not observation.deterministic_pass and observation.verifier_results:
            raise IntegrityError("runner reported success despite failed deterministic verifiers")


def detect_regressions(outcomes: Iterable[TaskOutcome]) -> list[dict[str, Any]]:
    """One record per regression detected on one task (spec section 26 Regression).

    ``reported``: the runner's verifiers counted critical/high/medium
    regressions in the candidate runs (count = the per-run maximum the outcome
    carries). ``task_failure``: the baseline solved the task and the candidate
    did not; it is recorded as ``high``, the weight ``staged_regression`` and
    the evolution screen already give it.
    """
    records = []
    for row in sorted(outcomes, key=lambda item: item.task_id):
        baseline = {
            "success": row.baseline_success, "quality": row.baseline_quality,
            "tokens": row.baseline_tokens, "latency_ms": row.baseline_latency_ms,
            "tool_calls": row.baseline_tool_calls, "failed_tool_calls": row.baseline_failed_tool_calls,
        }
        candidate = {
            "success": row.candidate_success, "quality": row.candidate_quality,
            "tokens": row.candidate_tokens, "latency_ms": row.candidate_latency_ms,
            "tool_calls": row.candidate_tool_calls, "failed_tool_calls": row.candidate_failed_tool_calls,
            "deterministic_pass": row.deterministic_pass, "security_pass": row.security_pass,
            "critical_regressions": row.critical_regressions, "high_regressions": row.high_regressions,
            "medium_regressions": row.medium_regressions, "run_count": row.run_count,
        }
        found = [
            ("reported", severity, count)
            for severity, count in (
                ("critical", row.critical_regressions), ("high", row.high_regressions),
                ("medium", row.medium_regressions),
            )
            if count > 0
        ]
        if row.baseline_success and not row.candidate_success:
            found.append(("task_failure", "high", 1))
        records.extend(
            {
                "benchmark_id": row.task_id, "suite": row.suite, "split": row.split.value,
                "kind": kind, "severity": severity, "count": count,
                "baseline_outcome": baseline, "candidate_outcome": candidate,
            }
            for kind, severity, count in found
        )
    return records


def staged_regression(outcomes: Iterable[TaskOutcome]) -> bool:
    """True when staged (SMOKE/CORE) evidence already proves a regression.

    A staged run never marks an experiment ``passed`` - only FULL with the
    sealed holdout can. But when screening shows that the candidate fails a
    task the baseline solves, fails a security verifier, or carries a critical
    or high regression, the experiment is recorded as ``rejected`` right away,
    so a non-finalist is rejected without ever consuming the sealed holdout.
    """
    return any(
        (row.baseline_success and not row.candidate_success)
        or not row.security_pass
        or row.critical_regressions > 0
        or row.high_regressions > 0
        for row in outcomes
    )
