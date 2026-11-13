from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier
import time

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
        ("ground_truth", "confidence", 0.79, "at least 0.8"),
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


def test_cli_admits_authored_public_benchmark_and_reports_new_manifest(
    tmp_path, monkeypatch, capsys,
):
    import sys

    from hermes_improve.cli import main

    source = tmp_path / "authored.json"
    source.write_text(json.dumps(data("authored-task", "TRAIN")), encoding="utf-8")
    root = tmp_path / "public"
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    assert main() == 0

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "admitted"
    assert result["path"] == str(root / "train" / "authored-task-v1.json")
    assert result["manifest_version"] == BenchmarkStore(root).version


def test_cli_rejects_sealed_holdout_benchmark_without_writing(tmp_path, monkeypatch):
    import sys

    from hermes_improve.cli import main

    source = tmp_path / "holdout.json"
    source.write_text(
        json.dumps(data("private-task", "SEALED_HOLDOUT")), encoding="utf-8",
    )
    root = tmp_path / "public"
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    with pytest.raises(ValueError, match="sealed holdout"):
        main()

    assert not root.exists()


def test_cli_rejects_corpus_leakage_without_admitting_record(tmp_path, monkeypatch):
    import sys

    from hermes_improve.cli import main

    root = tmp_path / "public"
    root.mkdir()
    existing = data("existing-task", "TRAIN", "existing-cluster")
    existing["task"]["prompt"] = (
        "Create a Python package that parses CSV files and validates malformed rows "
        "with clear useful error messages for users."
    )
    (root / "existing.json").write_text(json.dumps(existing), encoding="utf-8")
    authored = data("authored-task", "TRAIN", "different-cluster")
    authored["task"]["prompt"] = (
        "Create a Python package that parses CSV files and reports malformed rows "
        "with clear useful error messages for users."
    )
    source = tmp_path / "authored.json"
    source.write_text(json.dumps(authored), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    with pytest.raises(ValueError, match="same split and cluster_id"):
        main()

    assert sorted(path.name for path in root.rglob("*.json")) == ["existing.json"]


def test_cli_rejects_duplicate_id_version_without_overwriting_existing_record(
    tmp_path, monkeypatch,
):
    import sys

    from hermes_improve.cli import main

    root = tmp_path / "public"
    existing_path = root / "train" / "same-task-v1.json"
    existing_path.parent.mkdir(parents=True)
    original = data("same-task", "TRAIN")
    existing_path.write_text(json.dumps(original), encoding="utf-8")
    changed = data("same-task", "TRAIN")
    changed["task"]["prompt"] = "Different operator-authored task."
    source = tmp_path / "replacement.json"
    source.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    with pytest.raises(ValueError, match="duplicate benchmark id/version"):
        main()

    assert json.loads(existing_path.read_text(encoding="utf-8")) == original
    assert sorted(path.name for path in root.rglob("*.json")) == ["same-task-v1.json"]


def test_cli_does_not_overwrite_existing_destination_with_different_record(
    tmp_path, monkeypatch,
):
    import sys

    from hermes_improve.cli import main

    root = tmp_path / "public"
    existing_path = root / "train" / "authored-task-v1.json"
    existing_path.parent.mkdir(parents=True)
    original = data("different-id", "TRAIN", "different-cluster")
    original["task"]["prompt"] = "Build an unrelated calendar reminder application."
    existing_path.write_text(json.dumps(original), encoding="utf-8")
    source = tmp_path / "authored.json"
    source.write_text(json.dumps(data("authored-task", "TRAIN")), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    with pytest.raises(ValueError, match="destination already exists"):
        main()

    assert json.loads(existing_path.read_text(encoding="utf-8")) == original
    assert list(existing_path.parent.iterdir()) == [existing_path]


def test_cli_rejects_public_split_symlink_outside_root(tmp_path, monkeypatch):
    import sys

    from hermes_improve.cli import main

    root = tmp_path / "public"
    root.mkdir()
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (root / "train").symlink_to(holdout, target_is_directory=True)
    source = tmp_path / "authored.json"
    source.write_text(json.dumps(data("authored-task", "TRAIN")), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(tmp_path / "holdout"),
    ])

    with pytest.raises(ValueError, match="inside its configured root"):
        main()

    assert list(holdout.iterdir()) == []


def test_cli_rejects_near_duplicate_of_sealed_holdout(tmp_path, monkeypatch):
    import sys

    from hermes_improve.cli import main

    public = tmp_path / "public"
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    sealed = data("sealed-task", "SEALED_HOLDOUT", "sealed-cluster")
    sealed["task"]["prompt"] = (
        "Create a Python package that parses CSV files and validates malformed rows "
        "with clear useful error messages for users."
    )
    (holdout / "sealed.json").write_text(json.dumps(sealed), encoding="utf-8")
    authored = data("authored-task", "TRAIN", "public-cluster")
    authored["task"]["prompt"] = (
        "Create a Python package that parses CSV files and reports malformed rows "
        "with clear useful error messages for users."
    )
    source = tmp_path / "authored.json"
    source.write_text(json.dumps(authored), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source),
        "--holdout-root", str(holdout),
    ])

    with pytest.raises(ValueError, match="near-duplicate benchmarks"):
        main()

    assert not list(public.rglob("*.json"))


def test_concurrent_admissions_revalidate_after_acquiring_the_store_lock(tmp_path, monkeypatch):
    root = tmp_path / "public"
    store = BenchmarkStore(root, holdout_root=tmp_path / "holdout")
    first = Benchmark.from_dict(data("train-task", "TRAIN", "shared-cluster"))
    second = Benchmark.from_dict(data("validation-task", "VALIDATION", "shared-cluster"))
    start = Barrier(3)
    load_all = store.load_all

    def delayed_load_all():
        rows = load_all()
        time.sleep(0.05)
        return rows

    monkeypatch.setattr(store, "load_all", delayed_load_all)

    def admit(benchmark):
        start.wait()
        try:
            store.admit_public(benchmark)
        except ValueError as exc:
            return exc
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(admit, benchmark) for benchmark in (first, second)]
        start.wait()
        results = [future.result() for future in futures]

    assert sum(result is None for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 1
    assert len(list(root.rglob("*.json"))) == 1


def test_admission_rejects_split_directory_replaced_during_validation(tmp_path, monkeypatch):
    import os

    import hermes_improve.benchmarks as benchmarks_module

    root = tmp_path / "public"
    train = root / "train"
    train.mkdir(parents=True)
    original = train / "existing.json"
    existing = data("existing-task", "TRAIN")
    existing["task"]["prompt"] = "Create an existing arithmetic exercise."
    original.write_text(json.dumps(existing), encoding="utf-8")
    moved = tmp_path / "moved-train"
    store = BenchmarkStore(root, holdout_root=tmp_path / "holdout")
    authored = data("authored-task", "TRAIN", "other-cluster")
    authored["task"]["prompt"] = "Design a new geography challenge."
    benchmark = Benchmark.from_dict(authored)
    validate = benchmarks_module._validate_benchmark_set

    def replace_split_directory(rows):
        os.rename(train, moved)
        train.mkdir()
        validate(rows)

    monkeypatch.setattr(benchmarks_module, "_validate_benchmark_set", replace_split_directory)

    with pytest.raises(ValueError, match="split directory changed"):
        store.admit_public(benchmark)

    assert sorted(path.name for path in moved.iterdir()) == ["existing.json"]
    assert not list(train.rglob("*.json"))


def test_corpus_loader_reads_the_pinned_split_after_path_replacement(tmp_path):
    import os

    import hermes_improve.benchmarks as benchmarks_module

    root = tmp_path / "public"
    train = root / "train"
    train.mkdir(parents=True)
    original = data("existing-task", "TRAIN")
    (train / "existing.json").write_text(json.dumps(original), encoding="utf-8")
    root_fd, split_fd = benchmarks_module._open_benchmark_split(root, Split.TRAIN)
    moved = tmp_path / "moved-train"
    try:
        os.rename(train, moved)
        train.mkdir()
        replacement = data("replacement-task", "TRAIN")
        (train / "replacement.json").write_text(json.dumps(replacement), encoding="utf-8")

        rows = benchmarks_module._load_root_fd(
            root,
            root_fd,
            holdout=False,
            pinned_split_name="train",
            pinned_split_fd=split_fd,
        )
    finally:
        os.close(split_fd)
        os.close(root_fd)

    assert [row.id for row in rows] == ["existing-task"]
