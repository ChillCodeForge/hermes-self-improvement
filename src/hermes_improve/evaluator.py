from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Any, Iterable, Mapping, Protocol

from .artifacts import ArtifactStore
from .benchmarks import BenchmarkStore
from .errors import IntegrityError
from .evaluation_signing import sign_report
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


class PairedEvaluator:
    """Pairs baseline/candidate runs on identical tasks, seeds, environment and limits.

    The evaluator orchestrates an injected runner; it does not itself sandbox
    candidate execution. Deterministic verifier results always take precedence:
    the judge is only consulted for benchmarks without deterministic checks
    that declare subjective criteria, and its verdict can only add blocking
    reasons, never make a candidate pass.
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
                observations[(benchmark.id, "baseline")].append(baseline)
                observations[(benchmark.id, "candidate")].append(candidate_run)
                all_observations.extend((baseline, candidate_run))

        outcomes = []
        judgments: dict[str, JudgeVerdict] = {}
        for benchmark in benchmarks:
            baseline_runs = observations[(benchmark.id, "baseline")]
            candidate_runs = observations[(benchmark.id, "candidate")]
            if len(baseline_runs) != len(candidate_runs):
                raise IntegrityError(f"unpaired run count for benchmark {benchmark.id!r}")
            verdict = self._judge(benchmark, baseline_runs, candidate_runs)
            if verdict is not None:
                judgments[benchmark.id] = verdict
            outcomes.append(self._outcome(benchmark, suite, baseline_runs, candidate_runs, verdict))
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
            "run_policy": {
                **self.run_policy.describe(),
                "llm_influenced": llm_influenced,
                "runs_per_task": {task_id: len(task_seeds) for task_id, task_seeds in sorted(plan.items())},
            },
            "paired_statistics": self._paired_statistics(benchmarks, observations),
            "efficiency": efficiency_report(pairs),
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
                decides_experiment=suite.upper() in {"FULL", "SEALED_HOLDOUT"},
            )
        return report

    def _judge(
        self, benchmark: Benchmark, baseline_runs: list[RunObservation], candidate_runs: list[RunObservation]
    ) -> JudgeVerdict | None:
        if has_deterministic_verification(benchmark):
            return None
        criteria = _subjective_criteria(benchmark)
        if not criteria:
            return None
        if self.judge is None:
            raise ValueError(f"benchmark {benchmark.id!r} requires a configured judge")
        verdicts = [
            self.judge.compare(
                task_prompt=str(benchmark.task["prompt"]),
                criteria=criteria,
                baseline_output=baseline.final_output,
                candidate_output=candidate_run.final_output,
            )
            for baseline, candidate_run in zip(baseline_runs, candidate_runs)
        ]
        return aggregate_verdicts(verdicts)

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
        if observation.verifier_results and observation.deterministic_pass != all(
            item["passed"] for item in observation.verifier_results
        ):
            raise IntegrityError("runner deterministic status contradicts its verifier results")
        if observation.success and not observation.deterministic_pass and observation.verifier_results:
            raise IntegrityError("runner reported success despite failed deterministic verifiers")
