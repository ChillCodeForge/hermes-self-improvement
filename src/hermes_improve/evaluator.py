from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol

from .benchmarks import BenchmarkStore
from .errors import IntegrityError
from .models import (
    Benchmark,
    Candidate,
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
    ) -> None:
        self.benchmarks = benchmark_store
        self.runner = runner
        self.policy = policy
        self.experiment_store = experiment_store

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
        seed_set = tuple(dict.fromkeys(seeds))
        if not seed_set:
            raise ValueError("at least one deterministic seed is required")
        if not suite.strip():
            raise ValueError("suite name cannot be empty")
        benchmarks = [
            self.benchmarks.get(item, role=Role.EVALUATOR, grant=holdout_grant)
            for item in benchmark_ids
        ]
        if not benchmarks:
            raise ValueError("at least one benchmark is required")

        observations: dict[tuple[str, str], list[RunObservation]] = defaultdict(list)
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
            ))
        report = self.policy.assess(candidate.candidate_id, candidate.content_hash, benchmark_version, outcomes)
        if self.experiment_store is not None:
            if evaluation_id is None:
                raise ValueError("evaluation_id is required when persisting the evaluation")
            self.experiment_store.record_evaluation(evaluation_id, candidate.candidate_id, report)
        return report

    @staticmethod
    def _validate_observation(
        observation: RunObservation, benchmark: Benchmark, commit: str, seed: int
    ) -> None:
        if observation.task_id != benchmark.id or observation.commit != commit or observation.seed != seed:
            raise IntegrityError("runner returned an observation for a different task, commit, or seed")
        if not observation.environment_digest:
            raise IntegrityError("runner did not identify the execution environment")
        if observation.tokens < 0 or observation.latency_ms < 0:
            raise IntegrityError("runner returned invalid token or latency measurements")
