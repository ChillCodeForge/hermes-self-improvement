from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_factory import record
from hermes_improve.benchmark_schema import parse_benchmark_document
from hermes_improve.errors import IntegrityError
from hermes_improve.models import Benchmark, RunObservation
from hermes_improve.replay import fixture_tree_digest, replay_benchmark
from hermes_improve.store import ExperimentStore

COMMIT = "c" * 40
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def command_record(identifier: str = "cmd-task", split: str = "TRAIN") -> dict:
    data = record(identifier, split)
    data["environment"] |= {
        "container_image": "python:3.12-slim@sha256:" + "a" * 64,
        "command": ["sh", "check.sh"],
    }
    return data


class FakeRunner:
    """Implements the evaluator's BenchmarkRunner protocol with scripted outcomes."""

    def __init__(self, outcome=lambda seed: (0, True), environment=lambda seed: "e" * 64, task_id=None):
        self.outcome = outcome
        self.environment = environment
        self.task_id = task_id
        self.calls: list[tuple[str, str, int]] = []

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        self.calls.append((commit, benchmark.id, seed))
        exit_code, passed = self.outcome(seed)
        return RunObservation(
            task_id=self.task_id or benchmark.id, commit=commit, seed=seed,
            environment_digest=self.environment(seed), success=passed,
            quality=100.0 if passed else 50.0, tokens=0, latency_ms=10.0 + seed,
            deterministic_pass=passed, security_pass=True, exit_code=exit_code,
            final_output=f"run {seed}", artifact_digests=(f"{seed:064x}",),
        )


def benchmark() -> Benchmark:
    return Benchmark.from_dict(command_record())


def test_reproducible_passing_replay_is_verified_and_yields_valid_evidence():
    runner = FakeRunner()
    result = replay_benchmark(runner, benchmark(), commit=COMMIT, runs=3)
    assert runner.calls == [(COMMIT, "cmd-task", seed) for seed in range(3)]
    assert result.reproducible and result.passed and result.replay_verified
    assert len({run.outcome_digest for run in result.runs}) == 1
    assert result.outcome_digest == result.runs[0].outcome_digest
    data = command_record()
    data["eligibility"]["replay_evidence"] = result.evidence()
    parse_benchmark_document(data)


def test_seed_dependent_outcome_is_not_reproducible():
    result = replay_benchmark(FakeRunner(outcome=lambda seed: (seed % 2, seed % 2 == 0)), benchmark(),
                              commit=COMMIT, runs=3)
    assert not result.reproducible
    assert not result.replay_verified


def test_reproducible_failure_is_not_replay_verified():
    result = replay_benchmark(FakeRunner(outcome=lambda seed: (1, False)), benchmark(), commit=COMMIT, runs=2)
    assert result.reproducible and not result.passed and not result.replay_verified


def test_environment_drift_and_foreign_observations_are_integrity_errors():
    with pytest.raises(IntegrityError, match="different environments"):
        replay_benchmark(FakeRunner(environment=lambda seed: f"{seed:064x}"), benchmark(), commit=COMMIT, runs=2)
    with pytest.raises(IntegrityError, match="different task"):
        replay_benchmark(FakeRunner(task_id="other"), benchmark(), commit=COMMIT, runs=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"runs": 1}, "between 2 and 20"),
        ({"runs": 21}, "between 2 and 20"),
        ({"runs": True}, "between 2 and 20"),
        ({"commit": "HEAD"}, "full Git object ID"),
    ],
)
def test_replay_argument_validation(kwargs, message):
    arguments = {"commit": COMMIT, "runs": 2} | kwargs
    with pytest.raises(ValueError, match=message):
        replay_benchmark(FakeRunner(), benchmark(), **arguments)


def test_non_command_and_holdout_benchmarks_are_not_replayed():
    with pytest.raises(ValueError, match="not a command benchmark"):
        replay_benchmark(FakeRunner(), Benchmark.from_dict(record("plain")), commit=COMMIT, runs=2)
    with pytest.raises(ValueError, match="sealed holdout"):
        replay_benchmark(
            FakeRunner(), Benchmark.from_dict(command_record("h", "SEALED_HOLDOUT")), commit=COMMIT, runs=2,
        )


def test_replay_evidence_is_persisted_append_only(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    payload = replay_benchmark(FakeRunner(), benchmark(), commit=COMMIT, runs=2).to_dict()
    store.record_benchmark_replay(payload)
    store.record_benchmark_replay(payload)
    assert store.list_benchmark_replays("cmd-task") == [payload, payload]
    with pytest.raises(ValueError, match="malformed"):
        store.record_benchmark_replay(payload | {"commit": "HEAD"})


def test_cli_replay_uses_runner_reports_and_persists(tmp_path, monkeypatch, capsys):
    from hermes_improve import cli

    root = tmp_path / "public" / "train"
    root.mkdir(parents=True)
    (root / "a.json").write_text(json.dumps(command_record("cmd-a")))
    (root / "b.json").write_text(json.dumps(command_record("cmd-b")))
    (root / "plain.json").write_text(json.dumps(record("plain")))
    outcomes = {"cmd-a": lambda seed: (0, True), "cmd-b": lambda seed: (seed, seed == 0)}

    class CliRunner:
        def __init__(self, repository, artifacts):
            self.repository = repository

        def run(self, commit, benchmark, seed):
            return FakeRunner(outcome=outcomes[benchmark.id]).run(commit, benchmark, seed)

    monkeypatch.setattr(cli, "DockerCommandRunner", CliRunner)
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "public"))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "replay", "--commit", COMMIT,
        "--repository", str(tmp_path), "--runs", "2",
    ])
    assert cli.main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["replayed"] == 2 and report["replay_verified"] == 1
    assert {item["benchmark_id"]: item["reproducible"] for item in report["results"]} == {
        "cmd-a": True, "cmd-b": False,
    }
    assert "run 0" not in json.dumps(report)
    stored = ExperimentStore(tmp_path / "home" / "experiments.sqlite3").list_benchmark_replays()
    assert [item["benchmark_id"] for item in stored] == ["cmd-a", "cmd-b"]

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "replay", "--id", "cmd-a", "--commit", COMMIT,
        "--repository", str(tmp_path),
    ])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)["replayed"] == 1
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "replay", "--id", "plain", "--commit", COMMIT,
        "--repository", str(tmp_path),
    ])
    with pytest.raises(ValueError, match="no public command benchmark"):
        cli.main()


def test_fixture_tree_digest_tracks_content_and_rejects_symlinks(tmp_path):
    fixture = tmp_path / "fixture"
    (fixture / "task").mkdir(parents=True)
    (fixture / "task" / "a.txt").write_text("one")
    first = fixture_tree_digest(fixture)
    assert first == fixture_tree_digest(fixture)
    (fixture / "task" / "a.txt").write_text("two")
    assert fixture_tree_digest(fixture) != first
    (fixture / "link").symlink_to(fixture / "task" / "a.txt")
    with pytest.raises(ValueError, match="symlinks"):
        fixture_tree_digest(fixture)


DOCKER_IMAGE = "python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"


def _docker_image_available() -> bool:
    if shutil.which("docker") is None or shutil.which("git") is None:
        return False
    probe = subprocess.run(["docker", "image", "inspect", DOCKER_IMAGE], capture_output=True, check=False)
    return probe.returncode == 0


@pytest.mark.docker
@pytest.mark.skipif(not _docker_image_available(), reason="Docker and the pinned corpus image are required")
def test_corpus_benchmark_replays_reproducibly_in_docker(tmp_path):
    from hermes_improve.artifacts import ArtifactStore
    from hermes_improve.benchmark_schema import load_benchmark_document
    from hermes_improve.docker_runner import DockerCommandRunner

    repository = tmp_path / "repo"
    shutil.copytree(REPOSITORY_ROOT / "benchmarks" / "fixtures", repository / "benchmarks" / "fixtures")
    git = ["git", "-C", str(repository), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "fixtures"], check=True)
    commit = subprocess.run([*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    item = parse_benchmark_document(load_benchmark_document(
        REPOSITORY_ROOT / "benchmarks" / "validation" / "shell-count-error-codes-v1.json"
    ))
    runner = DockerCommandRunner(repository, ArtifactStore(tmp_path / "artifacts"))
    result = replay_benchmark(runner, item, commit=commit, runs=2)
    assert result.replay_verified
