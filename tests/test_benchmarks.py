from __future__ import annotations

import json

import pytest

from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import AccessDenied
from hermes_improve.models import Benchmark, Role, Split
from hermes_improve.security import AccessController


def data(identifier: str, split: str, cluster: str = "shared") -> dict:
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "manual"}, "task": {"title": identifier, "prompt": "Do a test"},
        "taxonomy": {"categories": [], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"}, "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 0.9},
        "verification": {"deterministic": ["exit_code"], "subjective": []},
        "ground_truth": {"type": "deterministic_test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": ["SMOKE"], "cluster_id": cluster, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }


def test_benchmark_schema_and_digest_are_stable():
    first = Benchmark.from_dict(data("task-a", "TRAIN"))
    second = Benchmark.from_dict(data("task-a", "TRAIN"))
    assert first.digest == second.digest


def test_cluster_split_leakage_is_rejected(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "a.json").write_text(json.dumps(data("task-a", "TRAIN")))
    (root / "b.json").write_text(json.dumps(data("task-b", "SEALED_HOLDOUT")))
    with pytest.raises(ValueError, match="leaks across splits"):
        BenchmarkStore(root).validate()


def test_generator_cannot_read_holdout_but_evaluator_capability_can(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    (root / "secret.json").write_text(json.dumps(data("secret-task", "SEALED_HOLDOUT", "holdout-only")))
    access = AccessController()
    store = BenchmarkStore(root, access)
    assert store.list(Role.GENERATOR) == []
    with pytest.raises(AccessDenied):
        store.get("secret-task", Role.GENERATOR)
    grant = access.evaluator_holdout_grant(Role.EVALUATOR)
    assert store.get("secret-task", Role.EVALUATOR, grant).split is Split.SEALED_HOLDOUT
    with pytest.raises(AccessDenied):
        store.get("secret-task", Role.EVALUATOR, None)


def test_unsanitized_or_secret_benchmark_is_rejected():
    bad = data("secret", "TRAIN")
    bad["security"]["contains_secrets"] = True
    with pytest.raises(ValueError, match="sanitized"):
        Benchmark.from_dict(bad)


def test_truthy_non_boolean_security_flag_is_rejected():
    bad = data("invalid-security", "TRAIN")
    bad["security"]["sanitized"] = "true"
    with pytest.raises(ValueError, match="sanitized"):
        Benchmark.from_dict(bad)
