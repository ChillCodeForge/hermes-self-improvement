from __future__ import annotations

import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve.evaluation_signing import report_payload, verify_report_signature
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import AccessDenied, IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.models import Candidate, CandidateType, Role, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.security import AccessController
from hermes_improve.store import ExperimentStore


def benchmark(identifier, split, cluster, suites):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": f"Complete task {identifier}"},
        "taxonomy": {"categories": ["test"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {
            "class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True,
        },
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
            tool_calls=2,
        )


def setup_store(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "full.json").write_text(json.dumps(benchmark("full-task", "REGRESSION", "full-cluster", ["FULL"])))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "holdout.json").write_text(json.dumps(benchmark("holdout-task", "SEALED_HOLDOUT", "holdout-cluster", ["SEALED_HOLDOUT"])))
    access = AccessController()
    return BenchmarkStore(root, access, holdout_root=holdout), access


def candidate():
    return Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "skill/a", "base", "candidate", "diff")


def test_paired_evaluator_runs_same_tasks_and_seeds(tmp_path):
    store, access = setup_store(tmp_path)
    runner = FixtureRunner()
    evaluator = PairedEvaluator(store, runner, PromotionPolicy())
    holdout_grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    report = evaluator.evaluate(
        candidate(), baseline_commit="base", benchmark_ids=["full-task", "holdout-task"],
        seeds=[3, 7], suite="FULL", benchmark_version=store.version, holdout_grant=holdout_grant,
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
            suite="SEALED_HOLDOUT", benchmark_version=store.version,
        )


def test_evaluator_rejects_soft_benchmarks_before_running_tasks(tmp_path):
    store, _ = setup_store(tmp_path)
    full_benchmark = tmp_path / "benchmarks" / "full.json"
    record = json.loads(full_benchmark.read_text())
    record["eligibility"] = {
        "class": "SOFT_BENCHMARK",
        "reproducibility_score": 0.6,
        "replay_verified": False,
    }
    full_benchmark.write_text(json.dumps(record))
    runner = FixtureRunner()
    evaluator = PairedEvaluator(store, runner, PromotionPolicy())

    with pytest.raises(ValueError, match="requires HARD_BENCHMARK"):
        evaluator.evaluate(
            candidate(), baseline_commit="base", benchmark_ids=["full-task"], seeds=[1],
            suite="FULL", benchmark_version=store.version,
        )
    assert runner.calls == []


def test_candidate_must_be_based_on_requested_baseline(tmp_path):
    store, _ = setup_store(tmp_path)
    evaluator = PairedEvaluator(store, FixtureRunner(), PromotionPolicy())
    with pytest.raises(IntegrityError, match="base commit"):
        evaluator.evaluate(
            candidate(), baseline_commit="other", benchmark_ids=["full-task"], seeds=[1],
            suite="FULL", benchmark_version=store.version,
        )


def test_deterministic_failure_blocks_candidate_even_when_quality_improves(tmp_path):
    store, access = setup_store(tmp_path)
    runner = FixtureRunner(bad_commit="candidate")
    evaluator = PairedEvaluator(store, runner, PromotionPolicy())
    holdout_grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    report = evaluator.evaluate(
        candidate(), baseline_commit="base", benchmark_ids=["full-task", "holdout-task"],
        seeds=[1], suite="FULL", benchmark_version=store.version, holdout_grant=holdout_grant,
    )
    assert not report.passed
    assert "deterministic verifier failed" in report.reasons


def test_paired_evaluator_rejects_stale_benchmark_manifest_version(tmp_path):
    store, _ = setup_store(tmp_path)
    evaluator = PairedEvaluator(store, FixtureRunner(), PromotionPolicy())
    with pytest.raises(IntegrityError, match="benchmark version"):
        evaluator.evaluate(
            candidate(), baseline_commit="base", benchmark_ids=["full-task"], seeds=[1],
            suite="FULL", benchmark_version="stale",
        )


def test_paired_evaluator_rejects_non_finite_run_metrics(tmp_path):
    store, _ = setup_store(tmp_path)

    class InvalidRunner(FixtureRunner):
        def run(self, commit, task, seed):
            return RunObservation(
                task_id=task.id, commit=commit, seed=seed,
                environment_digest="fixture-digest", success=True,
                quality=float("nan"), tokens=0, latency_ms=1,
                deterministic_pass=True, security_pass=True,
            )

    evaluator = PairedEvaluator(store, InvalidRunner(), PromotionPolicy())
    with pytest.raises(IntegrityError, match="quality or latency"):
        evaluator.evaluate(
            candidate(), baseline_commit="base", benchmark_ids=["full-task"],
            seeds=[1], suite="FULL", benchmark_version=store.version,
        )


def test_paired_evaluator_persists_signed_run_level_evidence(tmp_path):
    benchmarks, access = setup_store(tmp_path)
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="lineage-1", hypothesis="test evaluator",
        candidate_type="SKILL_PATCH", target="brain/skills/demo", risk="LOW",
        baseline_commit="base", benchmark_version=benchmarks.version, policy_version="1",
    )
    subject = candidate()
    store.add_candidate(subject)
    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    evaluator = PairedEvaluator(
        benchmarks, FixtureRunner(), PromotionPolicy(), store,
        evaluator_signing_key=private_pem,
    )
    grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    report = evaluator.evaluate(
        subject, baseline_commit="base", benchmark_ids=["full-task", "holdout-task"],
        seeds=[1], suite="FULL", benchmark_version=benchmarks.version,
        holdout_grant=grant, evaluation_id="eval-1",
    )
    evidence = store.promotion_evidence(subject.candidate_id)
    verify_report_signature(
        {
            "candidate_id": evidence["evaluation"]["candidate_id"],
            "candidate_hash": evidence["evaluation"]["candidate_hash"],
            "benchmark_version": evidence["evaluation"]["benchmark_version"],
            "policy_version": evidence["evaluation"]["policy_version"],
            "outcomes": [
                row["result"] for row in sorted(
                    evidence["task_results"], key=lambda item: item["task_id"]
                )
            ],
            "passed": bool(evidence["evaluation"]["passed"]),
            "reasons": json.loads(evidence["evaluation"]["reasons"]),
            "created_at": evidence["evaluation"]["created_at"],
            "metrics": json.loads(evidence["evaluation"]["metrics_json"]),
            "run_observations": evidence["run_observations"],
        },
        evidence["evaluation"]["signature"],
        public_pem,
    )
    assert report.passed
    assert len(evidence["run_observations"]) == 4
    assert report_payload(report)["run_observations"]
