from __future__ import annotations

import json
import sys

import pytest

from benchmark_factory import record
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.models import Benchmark
from hermes_improve.suites import (
    ImpactScope,
    Suite,
    corpus_manifest,
    select_suite,
    suite_manifest,
    verify_manifest,
)

VERSION = "sha256:" + "0" * 64


def corpus() -> list[Benchmark]:
    rows = [
        record("docker-build", categories=["devops"], capabilities=["docker"],
               impact_areas=["skills/docker*"]),
        record("git-merge", categories=["git"], capabilities=["git-cli"]),
        record("research-summary", "VALIDATION", categories=["research"]),
        record("critical-shell", "REGRESSION", categories=["shell"], criticality="critical"),
        record("full-only", "REGRESSION", categories=["docs"], suites=["FULL"]),
        record("soft-task", categories=["devops"], eligibility="SOFT_BENCHMARK"),
        record("shadow-task", "SHADOW", categories=["devops"]),
        record("holdout-a", "SEALED_HOLDOUT", categories=["devops"]),
    ]
    rows[5]["eligibility"]["reproducibility_score"] = 0.6
    return [Benchmark.from_dict(item) for item in rows]


def ids(rows) -> list[str]:
    return [item.id for item in rows]


def test_core_selection_excludes_shadow_holdout_soft_and_undeclared():
    assert ids(select_suite(corpus(), Suite.CORE)) == [
        "critical-shell", "docker-build", "git-merge", "research-summary",
    ]


def test_impact_narrows_smoke_and_core_but_keeps_critical_regressions():
    by_category = select_suite(corpus(), "smoke", impact=ImpactScope.parse(["category:devops"]))
    assert ids(by_category) == ["critical-shell", "docker-build"]
    by_capability = select_suite(corpus(), Suite.CORE, impact=ImpactScope.parse(["capability:git-cli"]))
    assert ids(by_capability) == ["critical-shell", "git-merge"]
    by_target = select_suite(
        corpus(), Suite.CORE, impact=ImpactScope.parse(["target:brain/skills/docker-compose"]),
    )
    assert ids(by_target) == ["critical-shell", "docker-build"]


def test_full_and_holdout_ignore_impact():
    impact = ImpactScope.parse(["category:git"])
    assert ids(select_suite(corpus(), Suite.FULL, impact=impact)) == [
        "critical-shell", "docker-build", "full-only", "git-merge", "research-summary",
    ]
    assert ids(select_suite(corpus(), Suite.SEALED_HOLDOUT, impact=impact)) == ["holdout-a"]


def test_smoke_is_capped_and_selection_is_deterministic():
    rows = [Benchmark.from_dict(record(f"task-{index:02d}")) for index in range(60)]
    first = select_suite(rows, Suite.SMOKE)
    assert len(first) == 50
    assert ids(first) == ids(select_suite(list(reversed(rows)), Suite.SMOKE))


def test_only_latest_version_is_selected_and_duplicates_rejected():
    newer = record("git-merge", categories=["git"])
    newer["version"] = 2
    rows = corpus() + [Benchmark.from_dict(newer)]
    assert [(item.id, item.version) for item in select_suite(rows, Suite.CORE) if item.id == "git-merge"] == [
        ("git-merge", 2)
    ]
    with pytest.raises(ValueError, match="unique"):
        select_suite(corpus() + corpus(), Suite.CORE)


@pytest.mark.parametrize("spec", ["owner:me", "category:", "plain", "target:/etc/passwd", "target:../x"])
def test_invalid_impact_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        ImpactScope.parse([spec])


def test_suite_manifest_digest_pins_selection_and_detects_tampering():
    selected = select_suite(corpus(), Suite.CORE)
    manifest = suite_manifest(selected, suite=Suite.CORE, benchmark_version=VERSION)
    again = suite_manifest(select_suite(corpus(), Suite.CORE), suite="CORE", benchmark_version=VERSION)
    assert manifest == again
    verify_manifest(manifest)
    tampered = json.loads(json.dumps(manifest))
    tampered["tasks"].pop()
    with pytest.raises(ValueError, match="does not match"):
        verify_manifest(tampered)
    holdout = suite_manifest(
        select_suite(corpus(), Suite.SEALED_HOLDOUT), suite=Suite.SEALED_HOLDOUT,
        benchmark_version=VERSION, include_task_details=False,
    )
    assert set(holdout["tasks"][0]) == {"id", "version", "digest"}


def test_corpus_manifest_refuses_holdout_records():
    with pytest.raises(ValueError, match="sealed holdout"):
        corpus_manifest(corpus(), benchmark_version=VERSION)


def test_cli_lists_impact_aware_suite_manifest(tmp_path, monkeypatch, capsys):
    from hermes_improve.cli import main

    root = tmp_path / "public"
    for item in corpus():
        if item.split.value in {"SEALED_HOLDOUT", "SHADOW"}:
            continue
        directory = root / item.split.value.lower()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{item.id}.json").write_text(json.dumps(item.to_dict()))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(root))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "list", "--suite", "CORE", "--impact", "category:git",
    ])
    assert main() == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in manifest["tasks"]] == ["critical-shell", "git-merge"]
    assert manifest["benchmark_version"] == BenchmarkStore(root).version
    verify_manifest(manifest)

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "list", "--role", "generator", "--suite", "CORE",
    ])
    assert main() == 0
    generator_view = json.loads(capsys.readouterr().out)
    assert [task["id"] for task in generator_view["tasks"]] == ["docker-build", "git-merge", "research-summary"]

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "benchmarks", "list", "--impact", "category:git"])
    with pytest.raises(ValueError, match="requires --suite"):
        main()


def test_impact_scope_for_candidate_narrows_skills_only():
    from hermes_improve.benchmark_schema import TAXONOMY_CATEGORIES
    from hermes_improve.suites import impact_scope_for_candidate

    scope = impact_scope_for_candidate("SKILL_PATCH", "brain/skills/git-helper", TAXONOMY_CATEGORIES)
    assert scope.target_areas == frozenset({"skills/git-helper"})
    assert scope.categories == frozenset({"git"})
    data = impact_scope_for_candidate("SKILL_CREATE", "brain/skills/data-transform-csv", TAXONOMY_CATEGORIES)
    assert data.categories == frozenset({"data-transform"})
    for kind, target in (("SYSTEM_PROMPT", "brain/prompts/system.md"), ("ROUTING", "brain/routing/rules.yaml")):
        assert impact_scope_for_candidate(kind, target, TAXONOMY_CATEGORIES).empty


def test_evaluation_suite_falls_back_to_full_suite_when_impact_matches_nothing():
    from hermes_improve.suites import ImpactScope, select_evaluation_suite

    rows = [_cli_bench("smoke-a", areas=["skills/coding*"]), _cli_bench("full-a", suites=("FULL",)),
            _cli_bench("holdout-a", split="SEALED_HOLDOUT", suites=("SEALED_HOLDOUT",))]
    narrowed, _, applied = select_evaluation_suite(
        rows, "SMOKE", impact=ImpactScope(target_areas=frozenset({"skills/coding-helper"})),
    )
    assert [item.id for item in narrowed] == ["smoke-a"] and not applied.empty
    fallback, _, applied = select_evaluation_suite(
        rows, "SMOKE", impact=ImpactScope(target_areas=frozenset({"skills/unrelated"})),
    )
    assert [item.id for item in fallback] == ["smoke-a"] and applied.empty
    selected, holdout, _ = select_evaluation_suite(rows, "FULL")
    assert [item.id for item in selected] == ["full-a"] and [item.id for item in holdout] == ["holdout-a"]
    with pytest.raises(ValueError, match="only as part of FULL"):
        select_evaluation_suite(rows, "SEALED_HOLDOUT")


def _cli_bench(identifier, split="REGRESSION", suites=("SMOKE",), areas=None):
    from hermes_improve.models import Benchmark

    return Benchmark.from_dict({
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": f"task {identifier}"},
        "taxonomy": {"categories": ["coding"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": ["exit_code"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": list(suites), "cluster_id": identifier, "criticality": "normal",
                      **({"impact_areas": areas} if areas else {})},
        "security": {"sanitized": True, "contains_secrets": False},
    })
