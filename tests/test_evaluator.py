from __future__ import annotations

import json

import pytest

from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import AccessDenied, IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.models import Candidate, CandidateType, Role, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.security import AccessController


def benchmark(identifier, split, cluster, suites):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test"}, "task": {"title": identifier, "prompt": "Complete task"},
        "taxonomy": {"categories": ["test"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"}, "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0},
        "verification": {"deterministic": ["unit-test"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": suites, "cluster_id": cluster, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }


class FixtureRunner:
    def __init__(self, *, bad_commit=None):
        self.calls = []
        self.bad_commit = bad_commit

    def run(self, commit, task, seed):
        self.calls.append((commit, task.id, seed))
        is_candidate = commit != "base"
        return RunObservation(
            task_id=task.id,
            commit=commit,
            seed=seed,
            environment_digest="fixture-digest",
            success=True,
            quality=10 if is_candidate else 5,
            tokens=100,
            latency_ms=20,
            deterministic_pass=commit != self.bad_commit,
            security_pass=True,
        )


def setup_store(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "full.json").write_text(json.dumps(benchmark("full-task", "REGRESSION", "full-cluster", ["FULL"])))
    (root / "holdout.json").write_text(json.dumps(benchmark("holdout-task", "SEALED_HOLDOUT", "holdout-cluster", ["SEALED_HOLDOUT"])))
    access = AccessController()
    return BenchmarkStore(root, access), access


def candidate():
    return Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "skill/a", "base", "candidate", "diff")


def test_paired_evaluator_runs_same_tasks_and_seeds(tmp_path):
    store, access = setup_store(tmp_path)
    runner = FixtureRunner()
    evaluator = PairedEvaluator(store, runner, PromotionPolicy())
    holdout_grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    report = evaluator.evaluate(
        candidate(), baseline_commit="base", benchmark_ids=["full-task", "holdout-task"],
        seeds=[3, 7], suite="FULL", benchmark_version="bench-v1", holdout_grant=holdout_grant,
    )
    assert report.passed
    assert len(runner.calls) == 8
    for task_id in ("full-task", "holdout-task"):
        assert [call for call in runner.calls if call[1] == task_id] == [
            ("base", task_id, 3), ("candidate", task_id, 3),
            ("base", task_id, 7), ("candidate", task_id, 7),
        ]


def test_holdout_cannot_be_evaluated_without_grant(tmp_path):
    store, _ = setup_store(tmp_path)
    evaluator = PairedEvaluator(store, FixtureRunner(), PromotionPolicy())
    with pytest.raises(AccessDenied):
        evaluator.evaluate(
            candidate(), baseline_commit="base", benchmark_ids=["holdout-task"], seeds=[1],
            suite="SEALED_HOLDOUT", benchmark_version="bench-v1",
        )


def test_candidate_must_be_based_on_requested_baseline(tmp_path):
    store, _ = setup_store(tmp_path)
    evaluator = PairedEvaluator(store, FixtureRunner(), PromotionPolicy())
    with pytest.raises(IntegrityError, match="base commit"):
        evaluator.evaluate(
            candidate(), baseline_commit="other", benchmark_ids=["full-task"], seeds=[1],
            suite="FULL", benchmark_version="bench-v1",
        )


def test_deterministic_failure_blocks_candidate_even_when_quality_improves(tmp_path):
    store, access = setup_store(tmp_path)
    runner = FixtureRunner(bad_commit="candidate")
    evaluator = PairedEvaluator(store, runner, PromotionPolicy())
    holdout_grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    report = evaluator.evaluate(
        candidate(), baseline_commit="base", benchmark_ids=["full-task", "holdout-task"],
        seeds=[1], suite="FULL", benchmark_version="bench-v1", holdout_grant=holdout_grant,
    )
    assert not report.passed
    assert "deterministic verifier failed" in report.reasons
