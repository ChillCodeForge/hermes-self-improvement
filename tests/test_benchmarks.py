from __future__ import annotations

import json

import pytest

from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import AccessDenied
from hermes_improve.models import Benchmark, Role, Split
from hermes_improve.security import (
    AccessController,
    SpentHoldoutRegistry,
    consume_holdout_generation,
)


def data(identifier: str, split: str, cluster: str = "shared") -> dict:
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "manual", "provenance": "test-fixture"},
        "task": {"title": identifier, "prompt": "Do a test"},
        "taxonomy": {"categories": [], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {
            "class": "HARD_BENCHMARK", "reproducibility_score": 0.9, "replay_verified": True,
        },
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
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "b.json").write_text(json.dumps(data("task-b", "SEALED_HOLDOUT")))
    with pytest.raises(ValueError, match="leaks across splits"):
        BenchmarkStore(root, holdout_root=holdout).validate()


def test_near_duplicate_tasks_cannot_cross_benchmark_splits(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    train = data("task-train", "TRAIN", "cluster-train")
    train["task"]["prompt"] = (
        "Create a Python package that parses CSV files and validates malformed rows "
        "with clear useful error messages for users."
    )
    (root / "train.json").write_text(json.dumps(train))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    sealed = data("task-holdout", "SEALED_HOLDOUT", "cluster-holdout")
    sealed["task"]["prompt"] = (
        "Create a Python package that parses CSV files and reports malformed rows "
        "with clear useful error messages for users."
    )
    (holdout / "holdout.json").write_text(json.dumps(sealed))

    with pytest.raises(ValueError, match="near-duplicate benchmarks"):
        BenchmarkStore(root, holdout_root=holdout).validate()


def test_near_duplicate_tasks_must_share_cluster_even_within_one_split(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    first = data("task-a", "TRAIN", "cluster-a")
    first["task"]["prompt"] = (
        "Create a Python package that parses CSV files and validates malformed rows "
        "with clear useful error messages for users."
    )
    second = data("task-b", "TRAIN", "cluster-b")
    second["task"]["prompt"] = (
        "Create a Python package that parses CSV files and reports malformed rows "
        "with clear useful error messages for users."
    )
    (root / "a.json").write_text(json.dumps(first))
    (root / "b.json").write_text(json.dumps(second))

    with pytest.raises(ValueError, match="same split and cluster_id"):
        BenchmarkStore(root).validate()


def test_near_duplicate_tasks_share_cluster_and_split(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    first = data("task-a", "TRAIN", "shared-cluster")
    first["task"]["prompt"] = (
        "Create a Python package that parses CSV files and validates malformed rows "
        "with clear useful error messages for users."
    )
    second = data("task-b", "TRAIN", "shared-cluster")
    second["task"]["prompt"] = (
        "Create a Python package that parses CSV files and reports malformed rows "
        "with clear useful error messages for users."
    )
    (root / "a.json").write_text(json.dumps(first))
    (root / "b.json").write_text(json.dumps(second))

    assert len(BenchmarkStore(root).validate()) == 2


def test_generator_cannot_read_holdout_but_evaluator_capability_can(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "secret.json").write_text(json.dumps(data("secret-task", "SEALED_HOLDOUT", "holdout-only")))
    access = AccessController()
    store = BenchmarkStore(root, access, holdout_root=holdout)
    with pytest.raises(AccessDenied):
        store.list(Role.GENERATOR)
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


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("source", "provenance", "", "source.provenance"),
        ("taxonomy", "difficulty", "", "taxonomy.difficulty"),
        ("taxonomy", "categories", ["cli", "cli"], "unique"),
        ("taxonomy", "required_capabilities", ["local", ""], "unique"),
        ("ground_truth", "type", "", "ground_truth.type"),
        ("ground_truth", "confidence", True, "confidence"),
        ("ground_truth", "confidence", float("nan"), "confidence"),
        ("ground_truth", "confidence", 1.1, "confidence"),
        ("ground_truth", "confidence", -0.1, "confidence"),
        ("eligibility", "replay_verified", "true", "replay_verified"),
        ("eligibility", "reproducibility_score", 0.79, "at least 0.8"),
    ],
)
def test_benchmark_metadata_fields_are_validated(section, field, value, message):
    invalid = data("invalid-metadata", "TRAIN")
    invalid[section][field] = value
    with pytest.raises(ValueError, match=message):
        Benchmark.from_dict(invalid)


def test_hard_benchmark_requires_strong_ground_truth():
    invalid = data("weak-hard-benchmark", "TRAIN")
    invalid["verification"]["deterministic"] = []
    with pytest.raises(ValueError, match="deterministic verification or structured ground truth"):
        Benchmark.from_dict(invalid)


def test_manifest_version_changes_with_benchmark_content(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    path = root / "task.json"
    path.write_text(json.dumps(data("task", "TRAIN", "cluster")))
    first = BenchmarkStore(root).version
    changed = data("task", "TRAIN", "cluster")
    changed["task"]["prompt"] = "Different task"
    path.write_text(json.dumps(changed))
    assert BenchmarkStore(root).version != first


def test_benchmark_secret_scanner_rejects_declared_sanitized_but_secret_data():
    bad = data("secret-token", "TRAIN")
    bad["task"]["prompt"] = "Authorization: Bearer abcdefghijklmnop"
    with pytest.raises(ValueError, match="credential-like"):
        Benchmark.from_dict(bad)


def test_holdout_generation_is_durable_per_lineage_and_changes_on_rotation(tmp_path):
    registry = SpentHoldoutRegistry(str(tmp_path / "experiments.sqlite3"))
    holdout = Benchmark.from_dict(data("holdout-a", "SEALED_HOLDOUT", "holdout-cluster"))
    public = Benchmark.from_dict(data("train-a", "TRAIN", "train-cluster"))

    generation = consume_holdout_generation(registry, [holdout], "lineage-a")
    assert registry.is_spent(generation, "lineage:lineage-a")
    with pytest.raises(AccessDenied, match="lineage"):
        consume_holdout_generation(
            SpentHoldoutRegistry(str(tmp_path / "experiments.sqlite3")),
            [public, holdout],
            "lineage-a",
        )

    changed = data("holdout-a", "SEALED_HOLDOUT", "holdout-cluster")
    changed["task"]["prompt"] = "A rotated holdout task"
    next_generation = consume_holdout_generation(
        registry, [public, Benchmark.from_dict(changed)], "lineage-a"
    )
    assert next_generation != generation


@pytest.mark.parametrize("identifier", ["../outside", "nested/task", ""])
def test_benchmark_id_is_a_safe_identifier(identifier):
    with pytest.raises(ValueError, match="safe"):
        Benchmark.from_dict(data(identifier, "TRAIN"))
