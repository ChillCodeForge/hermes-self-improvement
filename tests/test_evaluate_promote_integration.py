"""End to end: staged ``evaluate --suite`` runs feed WP-A's GitPromoter evidence checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve import cli
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.candidate_builder import create_skill_candidate
from hermes_improve.config import default_config
from hermes_improve.errors import PolicyRejected
from hermes_improve.models import CandidateType, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.run_policy import RunPolicy
from hermes_improve.store import ExperimentStore


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def _benchmark(identifier, split, suites):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": f"Complete {identifier}"},
        "taxonomy": {"categories": ["skills"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": ["exit_code"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": suites, "cluster_id": identifier, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }


class FakeBenchmarkRunner:
    """Deterministic test double for the BenchmarkRunner protocol."""

    llm_influenced = False

    def __init__(self, baseline_commit):
        self.baseline_commit = baseline_commit
        self.calls = []

    def run(self, commit, benchmark, seed):
        self.calls.append((commit, benchmark.id, seed))
        candidate = commit != self.baseline_commit
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed,
            environment_digest="sha256:" + "e" * 64,
            success=True, quality=10 if candidate else 5, tokens=100, latency_ms=20,
            deterministic_pass=True, security_pass=True, tool_calls=2,
            verifier_results=({"type": "exit_code", "passed": True},),
        )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    public_root = tmp_path / "benchmarks"
    public_root.mkdir()
    holdout_root = tmp_path / "holdout"
    holdout_root.mkdir()
    for identifier, suites in (("smoke-task", ["SMOKE"]), ("core-task", ["CORE"]), ("full-task", ["FULL"])):
        (public_root / f"{identifier}.json").write_text(json.dumps(_benchmark(identifier, "REGRESSION", suites)))
    (holdout_root / "holdout-task.json").write_text(
        json.dumps(_benchmark("holdout-task", "SEALED_HOLDOUT", ["SEALED_HOLDOUT"]))
    )
    benchmark_version = BenchmarkStore(public_root, holdout_root=holdout_root).version

    repository = tmp_path / "brain-repo"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "main")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    skill = repository / "brain/skills/demo"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("original\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "baseline")
    baseline = _git(repository, "rev-parse", "HEAD")

    home = tmp_path / "home"
    home.mkdir()
    store = ExperimentStore(home / "experiments.sqlite3")
    config = default_config()
    candidate = create_skill_candidate(
        repository, store, experiment_id="exp-e2e", candidate_id="cand-e2e",
        candidate_type=CandidateType.SKILL_PATCH, target="brain/skills/demo",
        content="---\nname: demo\ndescription: improved\n---\n\nImproved demo skill.\n",
        hypothesis="improve the demo skill", benchmark_version=benchmark_version,
        policy_version=config.promotion.policy_version,
    )

    private = Ed25519PrivateKey.generate()
    key_file = tmp_path / "evaluator.pem"
    key_file.write_bytes(private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    runner = FakeBenchmarkRunner(baseline)
    monkeypatch.setattr(cli, "build_runner", lambda args, artifact_store: runner)
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public_root))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key_file))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))

    def evaluate(suite):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evaluate", "cand-e2e", "--repository", str(repository),
            "--holdout-root", str(holdout_root), "--suite", suite,
        ])
        assert cli.main() == 0

    def promoter():
        return GitPromoter(
            store, repository, tmp_path / "releases", tmp_path / "brain-current",
            PromotionPolicy.from_config(config.promotion), public,
        )

    return {"evaluate": evaluate, "promoter": promoter, "store": store, "candidate": candidate, "runner": runner}


def test_staged_cli_evaluations_satisfy_promotion_evidence(setup):
    for suite in ("SMOKE", "CORE"):
        setup["evaluate"](suite)
        assert setup["store"].show_experiment("exp-e2e")["decision"] == "pending"
    setup["evaluate"]("FULL")
    assert setup["store"].show_experiment("exp-e2e")["decision"] == "passed"

    runs = setup["store"].candidate_evaluations("cand-e2e")
    assert [json.loads(run["evaluation"]["metrics_json"])["suite"] for run in runs] == ["SMOKE", "CORE", "FULL"]
    expected = RunPolicy.from_config(default_config().scoring).describe()
    for run in runs:
        assert run["evaluation"]["policy_version"] == default_config().promotion.policy_version
        recorded = json.loads(run["evaluation"]["metrics_json"])["run_policy"]
        assert {key: recorded[key] for key in expected} == expected

    manifest = setup["promoter"]().promote("exp-e2e", release_name="brain-v1", manifest_key=b"m" * 32)
    assert manifest["candidate_commit"] == setup["candidate"].commit


def test_promotion_rejects_cli_evidence_without_core_run(setup):
    setup["evaluate"]("SMOKE")
    setup["evaluate"]("FULL")
    with pytest.raises(PolicyRejected, match="required CORE evidence is missing"):
        setup["promoter"]().promote("exp-e2e", release_name="brain-v1", manifest_key=b"m" * 32)


def test_run_policy_follows_scoring_config():
    scoring = default_config().scoring
    policy = RunPolicy.from_config(scoring)
    assert (policy.deterministic_runs, policy.smoke_runs, policy.core_runs, policy.full_runs, policy.holdout_runs) == (
        1, 2, 3, 5, 5,
    )
    assert policy == RunPolicy()
