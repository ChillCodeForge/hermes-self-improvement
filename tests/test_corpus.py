"""The curated public corpus under benchmarks/ must stay valid, replay-backed and leak-free."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from hermes_improve.benchmark_schema import DIFFICULTIES, TAXONOMY_CATEGORIES
from hermes_improve.benchmarks import BenchmarkStore, lint_public_corpus
from hermes_improve.clustering import cluster_texts
from hermes_improve.docker_runner import is_skill_procedure_benchmark
from hermes_improve.models import Eligibility, Split
from hermes_improve.replay import fixture_tree_digest, is_command_benchmark
from hermes_improve.suites import Suite, corpus_manifest, select_suite, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def corpus():
    return BenchmarkStore(BENCHMARKS).validate()


def test_corpus_validates_and_passes_lint():
    rows = corpus()
    lint_public_corpus(rows)
    assert 20 <= len(rows) <= 40


def test_corpus_contains_only_public_splits_and_no_holdout_tree():
    rows = corpus()
    assert {item.split for item in rows} == {Split.TRAIN, Split.VALIDATION, Split.REGRESSION}
    assert not any(path.name.lower() in {"sealed_holdout", "holdout"} for path in BENCHMARKS.rglob("*"))


def test_corpus_covers_taxonomy_and_difficulties():
    rows = corpus()
    categories = Counter(category for item in rows for category in item.taxonomy["categories"])
    assert set(categories) == set(TAXONOMY_CATEGORIES)
    assert min(categories.values()) >= 3
    assert {item.taxonomy["difficulty"] for item in rows} == set(DIFFICULTIES)


def test_every_record_is_a_replay_verified_hard_command_benchmark_bound_to_its_fixture():
    for item in corpus():
        if is_skill_procedure_benchmark(item):
            continue  # brain-consuming records: see test_skill_procedure_corpus.py
        assert is_command_benchmark(item), item.id
        assert Eligibility(item.eligibility["class"]) is Eligibility.HARD_BENCHMARK, item.id
        evidence = item.eligibility["replay_evidence"]
        assert evidence["reproducible"] and evidence["passed"] and evidence["runs"] >= 3, item.id
        fixture = ROOT / item.environment["repository_fixture"]
        assert item.environment["cwd"] == item.environment["repository_fixture"]
        assert item.environment["input_data"] == fixture_tree_digest(fixture), item.id
        for name in ("solve.sh", "check.sh"):
            assert (fixture / name).is_file(), f"{item.id}: missing {name}"
        assert item.environment["network_policy"] == "disabled"
        assert "@sha256:" in item.environment["container_image"]


def test_every_record_has_its_own_cluster_and_no_similarity_clusters_span_splits():
    rows = corpus()
    assert len({item.cluster_id for item in rows}) == len(rows)
    clusters = cluster_texts({item.id: item.task["prompt"] for item in rows})
    by_cluster: dict[str, set] = {}
    for item in rows:
        by_cluster.setdefault(clusters[item.id], set()).add(item.split)
    assert all(len(splits) == 1 for splits in by_cluster.values())


def test_every_suite_is_non_empty():
    rows = corpus()
    for suite in (Suite.SMOKE, Suite.CORE, Suite.FULL):
        assert select_suite(rows, suite), suite
    assert len(select_suite(rows, Suite.FULL)) == len(rows)


def test_committed_corpus_manifest_matches_the_corpus():
    store = BenchmarkStore(BENCHMARKS)
    committed = json.loads((BENCHMARKS / "manifests" / "corpus.json").read_text(encoding="utf-8"))
    verify_manifest(committed)
    assert committed == corpus_manifest(store.validate(), benchmark_version=store.version)
