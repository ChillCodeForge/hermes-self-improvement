from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from typing import Iterable, Protocol

from .artifacts import ArtifactStore
from .benchmarks import BenchmarkStore
from .errors import IntegrityError
from .evaluation_signing import sign_report
from .models import (
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
from .security import HoldoutGrant
from .store import ExperimentStore


class BenchmarkRunner(Protocol):
    """Runtime adapter contract; implementations must provide actual isolation."""

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation: ...


class PairedEvaluator:
    """Pairs baseline/candidate runs on identical tasks, seeds and environment.

    This class orchestrates an injected runner; it does not itself sandbox
    candidate execution or imply that a concrete Hermes runner is integrated.
    """

    def __init__(
        self,
        benchmark_store: BenchmarkStore,
        runner: BenchmarkRunner,
        policy: PromotionPolicy,
        experiment_store: ExperimentStore | None = None,
        evaluator_signing_key: bytes | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.benchmarks = benchmark_store
        self.runner = runner
        self.policy = policy
        self.experiment_store = experiment_store
        self.evaluator_signing_key = evaluator_signing_key
        self.artifact_store = artifact_store
        if experiment_store is not None and evaluator_signing_key is None:
            raise ValueError("persisted evaluations require an evaluator Ed25519 signing key")

    def evaluate(
        self,
        candidate: Candidate,
        *,
        baseline_commit: str,
        benchmark_ids: Iterable[str],
        seeds: Iterable[int],
        suite: str,
        benchmark_version: str,
        holdout_grant: HoldoutGrant | None = None,
        evaluation_id: str | None = None,
    ) -> EvaluationReport:
        if candidate.base_commit != baseline_commit:
            raise IntegrityError("candidate base commit does not match requested baseline")
        supplied_seeds = tuple(seeds)
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in supplied_seeds):
            raise ValueError("evaluation seeds must be non-negative integers")
        seed_set = tuple(dict.fromkeys(supplied_seeds))
        if not seed_set:
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

        observations: dict[tuple[str, str], list[RunObservation]] = defaultdict(list)
        all_observations: list[RunObservation] = []
        for benchmark in benchmarks:
            allowed_suites = benchmark.benchmark.get("suites", [])
            if benchmark.split is not Split.SEALED_HOLDOUT and suite.upper() not in {
                str(item).upper() for item in allowed_suites
            }:
                raise ValueError(f"benchmark {benchmark.id!r} is not in suite {suite!r}")
            for seed in seed_set:
                baseline = self.runner.run(baseline_commit, benchmark, seed)
                candidate_run = self.runner.run(candidate.commit, benchmark, seed)
                self._validate_observation(baseline, benchmark, baseline_commit, seed)
                self._validate_observation(candidate_run, benchmark, candidate.commit, seed)
                if baseline.environment_digest != candidate_run.environment_digest:
                    raise IntegrityError("baseline and candidate ran in different environments")
                observations[(benchmark.id, "baseline")].append(baseline)
                observations[(benchmark.id, "candidate")].append(candidate_run)
                all_observations.extend((baseline, candidate_run))

        outcomes = []
        for benchmark in benchmarks:
            baseline_runs = observations[(benchmark.id, "baseline")]
            candidate_runs = observations[(benchmark.id, "candidate")]
            if len(baseline_runs) != len(candidate_runs):
                raise IntegrityError(f"unpaired run count for benchmark {benchmark.id!r}")
            outcomes.append(TaskOutcome(
                task_id=benchmark.id,
                split=benchmark.split,
                suite="SEALED_HOLDOUT" if benchmark.split is Split.SEALED_HOLDOUT else suite.upper(),
                baseline_success=all(run.success for run in baseline_runs),
                candidate_success=all(run.success for run in candidate_runs),
                baseline_quality=sum(run.quality for run in baseline_runs) / len(baseline_runs),
                candidate_quality=sum(run.quality for run in candidate_runs) / len(candidate_runs),
                baseline_tokens=sum(run.tokens for run in baseline_runs) // len(baseline_runs),
                candidate_tokens=sum(run.tokens for run in candidate_runs) // len(candidate_runs),
                deterministic_pass=all(run.deterministic_pass for run in candidate_runs),
                security_pass=all(run.security_pass for run in candidate_runs),
                critical_regressions=max(run.critical_regressions for run in candidate_runs),
                high_regressions=max(run.high_regressions for run in candidate_runs),
                medium_regressions=max(run.medium_regressions for run in candidate_runs),
                categories=tuple(str(value) for value in benchmark.taxonomy.get("categories", [])),
                baseline_latency_ms=sum(run.latency_ms for run in baseline_runs) / len(baseline_runs),
                candidate_latency_ms=sum(run.latency_ms for run in candidate_runs) / len(candidate_runs),
                baseline_tool_calls=sum(run.tool_calls for run in baseline_runs) // len(baseline_runs),
                candidate_tool_calls=sum(run.tool_calls for run in candidate_runs) // len(candidate_runs),
                baseline_failed_tool_calls=sum(run.failed_tool_calls for run in baseline_runs) // len(baseline_runs),
                candidate_failed_tool_calls=sum(run.failed_tool_calls for run in candidate_runs) // len(candidate_runs),
                baseline_retries=sum(run.retries for run in baseline_runs) // len(baseline_runs),
                candidate_retries=sum(run.retries for run in candidate_runs) // len(candidate_runs),
                baseline_user_corrections=sum(run.user_corrections for run in baseline_runs) // len(baseline_runs),
                candidate_user_corrections=sum(run.user_corrections for run in candidate_runs) // len(candidate_runs),
            ))
        report = self.policy.assess(candidate.candidate_id, candidate.content_hash, benchmark_version, outcomes)
        report = replace(
            report,
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
                evaluation_id, candidate.candidate_id, report, signature=signature
            )
        return report

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
            observation.medium_regressions, observation.seed,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters):
            raise IntegrityError("runner returned invalid token or latency measurements")
        if (
            isinstance(observation.quality, bool) or not isinstance(observation.quality, (int, float))
            or not math.isfinite(observation.quality)
            or isinstance(observation.latency_ms, bool) or not isinstance(observation.latency_ms, (int, float))
            or not math.isfinite(observation.latency_ms) or observation.latency_ms < 0
        ):
            raise IntegrityError("runner returned invalid quality or latency measurements")
        if not isinstance(observation.success, bool) or not isinstance(observation.deterministic_pass, bool) or not isinstance(observation.security_pass, bool):
            raise IntegrityError("runner returned invalid verifier status")
