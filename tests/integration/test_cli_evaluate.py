from __future__ import annotations

import os

import pytest

from fake_docker import install
from hermes_improve.artifacts import ArtifactStore
from hermes_improve.cli import build_parser, build_runner, select_suite_benchmarks
from hermes_improve.docker_runner import DockerCommandRunner
from hermes_improve.models import Benchmark


def bench(identifier, split="REGRESSION", suites=("SMOKE",), eligibility="HARD_BENCHMARK"):
    hard = eligibility == "HARD_BENCHMARK"
    return Benchmark.from_dict({
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": "p"},
        "taxonomy": {"categories": [], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": eligibility, "reproducibility_score": 1.0 if hard else 0.5,
                        "replay_verified": hard},
        "verification": {"deterministic": ["exit_code"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": list(suites), "cluster_id": identifier, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    })


def parse(*extra):
    return build_parser().parse_args(["evaluate", "cand-1", "--repository", ".", "--holdout-root", "/h", *extra])


def test_evaluate_accepts_suite_and_defaults_to_full_with_policy_seeds():
    args = parse()
    assert args.suite == "FULL" and args.seeds is None and args.runner == "docker"
    assert parse("--suite", "SMOKE").suite == "SMOKE"
    with pytest.raises(SystemExit):
        parse("--suite", "SEALED_HOLDOUT")


def test_suite_selection_includes_holdout_only_for_full():
    benchmarks = [
        bench("smoke-a"), bench("core-a", suites=("CORE", "FULL")), bench("full-a", suites=("FULL",)),
        bench("soft", suites=("SMOKE", "FULL"), eligibility="SOFT_BENCHMARK"),
        bench("holdout", split="SEALED_HOLDOUT", suites=("SEALED_HOLDOUT",)),
    ]
    assert select_suite_benchmarks(benchmarks, "SMOKE") == (["smoke-a"], [])
    assert select_suite_benchmarks(benchmarks, "CORE") == (["core-a"], [])
    assert select_suite_benchmarks(benchmarks, "FULL") == (["core-a", "full-a"], ["holdout"])
    with pytest.raises(ValueError, match="sealed holdout"):
        select_suite_benchmarks(benchmarks[:3], "FULL")
    with pytest.raises(ValueError, match="no HARD_BENCHMARK"):
        select_suite_benchmarks([bench("x", suites=("FULL",))], "SMOKE")


def test_build_runner_configures_docker_network_and_validates_hermes_options(tmp_path, monkeypatch):
    install(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    args = parse("--network", "hermes-eval-net", "--network-allow", "fixture-api:8080",
                 "--verifier-http-allow", "127.0.0.1:9000", "--timeout", "30")
    args.repository = tmp_path
    runner = build_runner(args, ArtifactStore(tmp_path / "artifacts"))
    assert isinstance(runner, DockerCommandRunner)
    assert runner.network.mode == "allowlist" and runner.network.network_name == "hermes-eval-net"
    assert runner.verifier_http_allowlist == frozenset({"127.0.0.1:9000"})
    assert runner.llm_influenced is False
    hermes = parse("--runner", "hermes")
    with pytest.raises(ValueError, match="requires --hermes-agent-dir"):
        build_runner(hermes, ArtifactStore(tmp_path / "artifacts"))
    networked = parse("--runner", "hermes", "--hermes-agent-dir", str(tmp_path), "--model", "m",
                      "--model-base-url", "http://127.0.0.1:1", "--network-allow", "a:1", "--network", "n")
    with pytest.raises(ValueError, match="networking disabled"):
        build_runner(networked, ArtifactStore(tmp_path / "artifacts"))


def test_staged_suite_evaluations_do_not_decide_the_experiment(tmp_path):
    from hermes_improve.models import Candidate, CandidateType
    from hermes_improve.policy import PromotionPolicy
    from hermes_improve.store import ExperimentStore

    store = ExperimentStore(tmp_path / "state.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="l", hypothesis="h", candidate_type="SKILL_PATCH", target="brain/skills/x",
        risk="LOW", baseline_commit="base", benchmark_version="v1", policy_version="1",
    )
    subject = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "brain/skills/x", "base", "c", "diff")
    store.add_candidate(subject)
    report = PromotionPolicy().assess("cand-1", subject.content_hash, "v1", [])
    store.record_evaluation("eval-smoke", "cand-1", report, decides_experiment=False)
    assert store.show_experiment("exp-1")["decision"] == "pending"
    store.record_evaluation("eval-full", "cand-1", report)
    assert store.show_experiment("exp-1")["decision"] == "rejected"
