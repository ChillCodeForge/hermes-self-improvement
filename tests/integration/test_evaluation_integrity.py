"""Regression tests for evaluation-integrity fixes found while building the WP-G end-to-end test."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from hermes_improve import cli
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.models import Benchmark, Candidate, CandidateType, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.role_handoff import export_candidate, import_candidate
from hermes_improve.sandbox import LocalSandboxExecutor
from hermes_improve.store import ExperimentStore
from skill_procedure_support import HOLDOUT_TASKS, skill_record

DIGEST = "sha256:" + "0" * 64


def git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        check=True, capture_output=True,
    ).stdout


def public_record(identifier="public-task", suites=("SMOKE", "FULL")):
    return skill_record(
        identifier, split="TRAIN", suites=list(suites), title="Commit a parser change",
        prompt="Commit the tokenizer edit in parser/lexer.py on its own with a conventional subject line.",
        input_data=DIGEST, harness_digest=DIGEST,
    )


def holdout_record(identifier="holdout-commit-cache-cleanup"):
    task = HOLDOUT_TASKS[identifier]
    return skill_record(
        identifier, split="SEALED_HOLDOUT", suites=["SEALED_HOLDOUT"], title=task["title"],
        prompt=task["prompt"], input_data=DIGEST, harness_digest=DIGEST,
    )


def test_benchmark_version_binds_the_public_corpus_and_holdout_has_its_own_version(tmp_path):
    """CLI-created candidates (generator view, no holdout) must match the evaluator's version."""
    public, holdout = tmp_path / "public", tmp_path / "holdout"
    public.mkdir()
    holdout.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    generator_view = BenchmarkStore(public).version
    assert BenchmarkStore(public, holdout_root=holdout).holdout_version is None
    (holdout / "a.json").write_text(json.dumps(holdout_record()))
    evaluator = BenchmarkStore(public, holdout_root=holdout)
    assert evaluator.version == generator_view
    first_holdout = evaluator.holdout_version
    assert first_holdout and first_holdout.startswith("sha256:")
    (holdout / "b.json").write_text(json.dumps(holdout_record("holdout-commit-billing-ticket")))
    assert BenchmarkStore(public, holdout_root=holdout).version == generator_view
    assert BenchmarkStore(public, holdout_root=holdout).holdout_version != first_holdout
    (public / "second.json").write_text(json.dumps(skill_record(
        "second-task", split="VALIDATION", suites=["CORE"], title="Commit a migration",
        prompt="Write one commit for the new database migration file under migrations/ only.",
        input_data=DIGEST, harness_digest=DIGEST,
    )))
    assert BenchmarkStore(public).version != generator_view


class SplitRunner:
    llm_influenced = False

    def __init__(self, baseline, candidate_success):
        self.baseline = baseline
        self.candidate_success = candidate_success

    def run(self, commit, benchmark, seed):
        success = True if commit == self.baseline else self.candidate_success
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed, environment_digest="e" * 64,
            success=success, quality=100.0 if success else 50.0, tokens=0, latency_ms=10.0,
            deterministic_pass=success, security_pass=True, verifier_results=({"type": "exit_code", "passed": success},),
        )


@pytest.mark.parametrize(("candidate_success", "decision"), [(False, "rejected"), (True, "pending")])
def test_staged_run_with_a_task_success_regression_rejects_the_experiment(tmp_path, candidate_success, decision):
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    benchmarks = BenchmarkStore(public)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "brain/skills/x", "a" * 40, "b" * 40, "diff")
    store.create_experiment_with_candidate(
        candidate, lineage_id="exp-1", hypothesis="h", risk="LOW",
        benchmark_version=benchmarks.version, policy_version="1",
    )
    key = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )
    report = PairedEvaluator(
        benchmarks, SplitRunner("a" * 40, candidate_success), PromotionPolicy(), store, key,
    ).evaluate(
        candidate, baseline_commit="a" * 40, benchmark_ids=["public-task"], suite="SMOKE",
        benchmark_version=benchmarks.version, evaluation_id="eval-smoke",
    )
    assert not report.passed  # a staged run alone never passes an experiment
    assert store.show_experiment("exp-1")["decision"] == decision
    assert report.metrics["sealed_holdout_version"] is None


@pytest.fixture
def tampering_candidate(tmp_path):
    """A candidate commit that edits its skill *and* the benchmark check it is graded with."""
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init", "-q", "-b", "main")
    (repository / "brain/skills/demo").mkdir(parents=True)
    (repository / "brain/skills/demo/SKILL.md").write_text("original\n")
    (repository / "benchmarks/fixtures/task").mkdir(parents=True)
    (repository / "benchmarks/fixtures/task/check.sh").write_text("exit 1\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "checkout", "-q", "-b", "candidate/exp-t")
    (repository / "brain/skills/demo/SKILL.md").write_text("improved\n")
    (repository / "benchmarks/fixtures/task/check.sh").write_text("exit 0\n")
    git(repository, "commit", "-q", "-am", "candidate")
    commit = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "checkout", "-q", "main")
    diff = git(repository, "diff", "--no-ext-diff", "--no-textconv", "--binary", base, commit)
    candidate = Candidate("cand-t", "exp-t", CandidateType.SKILL_PATCH, "brain/skills/demo", base, commit,
                          diff.decode())
    return repository, candidate, hashlib.sha256(diff).hexdigest()


def test_handoff_import_refuses_a_candidate_that_touches_protected_paths(tmp_path, tampering_candidate):
    repository, candidate, _ = tampering_candidate
    agent = ExperimentStore(tmp_path / "agent.sqlite3")
    agent.create_experiment_with_candidate(
        candidate, lineage_id="exp-t", hypothesis="h", risk="LOW", benchmark_version="v", policy_version="1",
    )
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    export_candidate(agent, repository, "cand-t", handoff)
    mirror = tmp_path / "mirror.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(mirror)], check=True)
    ledger = ExperimentStore(tmp_path / "ledger.sqlite3")
    with pytest.raises(IntegrityError, match="protected paths: benchmarks/fixtures/task/check.sh"):
        import_candidate(ledger, handoff, "cand-t", mirror)
    with pytest.raises(KeyError):
        ledger.get_candidate("cand-t")


def test_cli_evaluate_refuses_a_candidate_that_touches_protected_paths(tmp_path, monkeypatch, tampering_candidate):
    repository, candidate, _ = tampering_candidate
    home = tmp_path / "home"
    home.mkdir()
    ExperimentStore(home / "experiments.sqlite3").create_experiment_with_candidate(
        candidate, lineage_id="exp-t", hypothesis="h", risk="LOW", benchmark_version="v", policy_version="1",
    )
    (tmp_path / "public").mkdir()
    (tmp_path / "holdout").mkdir()
    key = tmp_path / "evaluator.pem"
    key.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    ran = []
    monkeypatch.setattr(cli, "build_runner", lambda args, artifacts: ran.append(args))
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "public"))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))
    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "evaluate", "cand-t", "--repository", str(repository),
        "--holdout-root", str(tmp_path / "holdout"), "--suite", "SMOKE",
    ])
    with pytest.raises(ValueError, match="protected paths: benchmarks/fixtures/task/check.sh"):
        cli.run()
    assert ran == []  # nothing was executed for the tampering candidate


def test_local_sandbox_executor_creates_the_snapshot_directory(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "a.txt").write_text("x")
    executor = LocalSandboxExecutor(acknowledge_no_isolation=True)
    result = executor.execute(["sh", "-c", "echo y > b.txt"], input_root=source, snapshot_dir=tmp_path / "out" / "new")
    assert result.exit_code == 0 and result.snapshot is not None and result.snapshot.is_file()


def test_benchmark_from_skill_record_is_valid():
    assert Benchmark.from_dict(public_record()).environment["type"] == "skill_procedure"


def shadow_record(identifier="shadow-task", prompt=None):
    return skill_record(
        identifier, split="SHADOW", suites=[], title="Tag a release",
        prompt=prompt or "Tag the current release commit with the version from CHANGELOG.md and push the tag.",
        input_data=DIGEST, harness_digest=DIGEST,
    )


def test_shadow_benchmarks_do_not_change_the_evaluation_benchmark_version(tmp_path):
    """SHADOW is post-promotion monitoring traffic, never selected for SMOKE/CORE/FULL."""
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    before = BenchmarkStore(public)
    assert before.shadow_version is None
    unchanged = before.version
    (public / "shadow.json").write_text(json.dumps(shadow_record()))
    after = BenchmarkStore(public)
    first_shadow = after.shadow_version
    assert after.version == unchanged
    assert first_shadow and first_shadow.startswith("sha256:")
    (public / "shadow2.json").write_text(json.dumps(shadow_record(
        "shadow-task-two", prompt="Rotate the staging database credentials and update the vault entry.",
    )))
    assert BenchmarkStore(public).version == unchanged
    assert BenchmarkStore(public).shadow_version != first_shadow


@pytest.fixture
def skill_candidate(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    git(repository, "init", "-q", "-b", "main")
    (repository / "brain/skills/demo").mkdir(parents=True)
    (repository / "brain/skills/demo/SKILL.md").write_text("original\n")
    git(repository, "add", "-A")
    git(repository, "commit", "-q", "-m", "base")
    base = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "checkout", "-q", "-b", "candidate/exp-v")
    (repository / "brain/skills/demo/SKILL.md").write_text("improved\n")
    git(repository, "commit", "-q", "-am", "candidate")
    commit = git(repository, "rev-parse", "HEAD").decode().strip()
    git(repository, "checkout", "-q", "main")
    diff = git(repository, "diff", "--no-ext-diff", "--no-textconv", "--binary", base, commit)
    candidate = Candidate("cand-v", "exp-v", CandidateType.SKILL_PATCH, "brain/skills/demo", base, commit,
                          diff.decode())
    return repository, candidate


def test_cli_evaluate_refuses_a_changed_benchmark_version_before_any_run(tmp_path, monkeypatch, skill_candidate):
    repository, candidate = skill_candidate
    home = tmp_path / "home"
    home.mkdir()
    public, holdout = tmp_path / "public", tmp_path / "holdout"
    public.mkdir()
    holdout.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    (holdout / "a.json").write_text(json.dumps(holdout_record()))
    bound = BenchmarkStore(public).version
    ExperimentStore(home / "experiments.sqlite3").create_experiment_with_candidate(
        candidate, lineage_id="exp-v", hypothesis="h", risk="LOW", benchmark_version=bound, policy_version="1",
    )
    (public / "second.json").write_text(json.dumps(skill_record(
        "second-task", split="VALIDATION", suites=["SMOKE", "FULL"], title="Commit a migration",
        prompt="Write one commit for the new database migration file under migrations/ only.",
        input_data=DIGEST, harness_digest=DIGEST,
    )))
    current = BenchmarkStore(public).version
    assert current != bound
    key = tmp_path / "evaluator.pem"
    key.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    ran = []
    monkeypatch.setattr(cli, "build_runner", lambda args, artifacts: ran.append(args))
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))
    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    for suite in ("SMOKE", "FULL"):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evaluate", "cand-v", "--repository", str(repository),
            "--holdout-root", str(holdout), "--suite", suite,
        ])
        with pytest.raises(IntegrityError, match="benchmark version changed since experiment exp-v was created") as error:
            cli.run()
        assert bound in str(error.value) and current in str(error.value)
    assert ran == []
    ledger = ExperimentStore(home / "experiments.sqlite3")
    assert ledger.candidate_evaluations("cand-v") == []
    import sqlite3
    with sqlite3.connect(home / "experiments.sqlite3") as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "decision_history" in tables:
            assert db.execute("SELECT COUNT(*) FROM decision_history").fetchone()[0] == 0


def test_evaluator_refuses_a_benchmark_version_other_than_the_experiment_binding(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    benchmarks = BenchmarkStore(public)
    store = ExperimentStore(tmp_path / "state.sqlite3")
    candidate = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "brain/skills/x", "a" * 40, "b" * 40, "diff")
    store.create_experiment_with_candidate(
        candidate, lineage_id="exp-1", hypothesis="h", risk="LOW",
        benchmark_version="sha256:" + "1" * 64, policy_version="1",
    )
    key = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )
    runner = SplitRunner("a" * 40, True)
    runner.calls = 0
    with pytest.raises(IntegrityError, match="benchmark version changed since experiment exp-1 was created"):
        PairedEvaluator(benchmarks, runner, PromotionPolicy(), store, key).evaluate(
            candidate, baseline_commit="a" * 40, benchmark_ids=["public-task"], suite="SMOKE",
            benchmark_version=benchmarks.version, evaluation_id="eval-smoke",
        )
    assert store.candidate_evaluations("cand-1") == []


def test_experiment_bound_before_the_shadow_exclusion_gets_an_explicit_refusal(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text(json.dumps(public_record()))
    plain = BenchmarkStore(public)
    assert plain.legacy_version == plain.version  # no SHADOW records: both formulas agree
    (public / "shadow.json").write_text(json.dumps(shadow_record()))
    store = BenchmarkStore(public)
    legacy = store.legacy_version
    assert legacy != store.version
    assert store.require_bound_version("exp-new", store.version) == store.version
    with pytest.raises(IntegrityError, match="created before SHADOW benchmarks were excluded"):
        store.require_bound_version("exp-old", legacy)
    with pytest.raises(IntegrityError, match="Benchmarks were added, changed or removed"):
        store.require_bound_version("exp-other", "sha256:" + "2" * 64)
