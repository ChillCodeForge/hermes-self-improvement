from __future__ import annotations

import json
import sys

import pytest

from benchmark_factory import record
from hermes_improve.benchmarks import BenchmarkStore, validate_benchmark_set
from hermes_improve.clustering import (
    SimilarityIndex,
    cluster_texts,
    exact_content_hash,
    find_similarity_split_leaks,
    minhash_signature,
    shingles,
    similarity,
    task_fingerprint,
)
from hermes_improve.miner import DedupRegistry, mine_sessions
from hermes_improve.models import Benchmark

BASE = "Convert the inventory CSV export into a JSON array sorted by SKU with typed numeric fields"
REWORDED = "Convert the inventory CSV export to a JSON array, sorted by SKU, with typed numeric values"
UNRELATED = "Revert the faulty commit that broke the greeting script while keeping later history"


def test_exact_hash_and_fingerprint_layers_differ_in_normalization():
    assert exact_content_hash("Fix a bug") != exact_content_hash("fix  a BUG")
    assert task_fingerprint("Fix a bug") == task_fingerprint("  fix  a BUG ")


def test_minhash_is_deterministic_and_similarity_separates_paraphrase_from_unrelated():
    assert minhash_signature(shingles(BASE)) == minhash_signature(shingles(BASE))
    assert similarity(BASE, REWORDED) >= 0.5
    assert similarity(BASE, UNRELATED) < 0.2


def test_cluster_ids_are_deterministic_and_order_independent():
    texts = {"a": BASE, "b": UNRELATED, "c": REWORDED}
    forward = cluster_texts(texts)
    backward = cluster_texts(dict(reversed(list(texts.items()))))
    assert forward == backward
    assert forward["a"] == forward["c"] != forward["b"]
    assert forward["a"].startswith("sem-")


def test_similarity_index_joins_existing_cluster_or_starts_new_one():
    index = SimilarityIndex()
    assert index.add("corpus", BASE, cluster_id="inventory") == "inventory"
    assert index.add("session", REWORDED) == "inventory"
    assert index.add("other", UNRELATED).startswith("sem-")
    assert index.similar_keys(REWORDED) == ("session", "corpus")  # best match first
    with pytest.raises(ValueError, match="already present"):
        index.add("corpus", BASE)


def test_duplicate_cluster_leakage_is_reported_across_splits():
    rows = [
        Benchmark.from_dict(record("inv-train", "TRAIN", prompt=BASE)),
        Benchmark.from_dict(record("inv-validation", "VALIDATION", prompt=REWORDED)),
        Benchmark.from_dict(record("git-task", "VALIDATION", prompt=UNRELATED)),
    ]
    leaks = find_similarity_split_leaks(rows)
    assert len(leaks) == 1
    assert {member[0] for member in leaks[0].members} == {"inv-train", "inv-validation"}
    with pytest.raises(ValueError, match="similar benchmarks must share one split"):
        validate_benchmark_set(rows)


def test_similar_tasks_in_one_split_are_allowed_with_different_clusters():
    validate_benchmark_set([
        Benchmark.from_dict(record("inv-a", "TRAIN", prompt=BASE)),
        Benchmark.from_dict(record("inv-b", "TRAIN", prompt=REWORDED)),
    ])


def test_train_holdout_similarity_leakage_is_rejected_by_store_validation(tmp_path):
    public = tmp_path / "public" / "train"
    public.mkdir(parents=True)
    (public / "a.json").write_text(json.dumps(record("inv-train", "TRAIN", prompt=BASE)))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    (holdout / "h.json").write_text(json.dumps(record("inv-holdout", "SEALED_HOLDOUT", prompt=REWORDED)))
    with pytest.raises(ValueError, match="SEALED_HOLDOUT"):
        BenchmarkStore(tmp_path / "public", holdout_root=holdout).validate()


def test_admission_rejects_similar_task_in_another_split(tmp_path, monkeypatch):
    from hermes_improve.cli import main

    root = tmp_path / "public"
    (root / "train").mkdir(parents=True)
    (root / "train" / "a.json").write_text(json.dumps(record("inv-train", "TRAIN", prompt=BASE)))
    source = tmp_path / "new.json"
    source.write_text(json.dumps(record("inv-regression", "REGRESSION", prompt=REWORDED)))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "add", str(source), "--holdout-root", str(tmp_path / "holdout"),
    ])
    with pytest.raises(ValueError, match="similar benchmarks must share one split"):
        main()
    assert not (root / "regression").exists() or not any((root / "regression").iterdir())


def test_mined_sessions_inherit_corpus_cluster_and_registry_pins_split():
    corpus = [Benchmark.from_dict(record("inventory", "TRAIN", "inventory-cluster", prompt=BASE))]
    first, second, third = mine_sessions(
        [
            {"session_id": "s1", "task": {"prompt": REWORDED}},
            {"session_id": "s2", "task": {"prompt": REWORDED + " please"}},
            {"session_id": "s3", "task": {"prompt": UNRELATED}},
        ],
        corpus=corpus,
    )
    assert first.cluster_id == second.cluster_id == "inventory-cluster"
    assert third.cluster_id == third.task_fingerprint
    registry = DedupRegistry()
    assert registry.assign(first, "TRAIN") == ("TRAIN", True)
    assert registry.assign(second, "SEALED_HOLDOUT") == ("TRAIN", False)
