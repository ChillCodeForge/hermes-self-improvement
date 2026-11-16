from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_improve.artifacts import ArtifactStore
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.hermes_runner import (
    HermesBatchRunner,
    HermesModelConfig,
    load_single_trajectory,
    parse_trajectory_record,
)
from hermes_improve.models import Benchmark, Candidate, CandidateType
from hermes_improve.policy import PromotionPolicy
from hermes_improve.sandbox import LocalSandboxExecutor, SandboxLimits, SandboxPolicyError
from hermes_improve.security import AccessController


FIXTURES = Path(__file__).parent / "fixtures" / "hermes"
IMAGE = "example/hermes-tools@sha256:" + "b" * 64
MODEL = HermesModelConfig(model="qwen3-coder", base_url="http://127.0.0.1:8000/v1")


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def brain_repo(tmp_path):
    repo = tmp_path / "brain-repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.invalid")
    skill = repo / "brain" / "skills" / "answer" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Answer skill\nANSWER=41\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "baseline")
    baseline = git(repo, "rev-parse", "HEAD")
    skill.write_text("# Answer skill\nANSWER=42\n")
    git(repo, "commit", "-q", "-am", "candidate")
    return repo, baseline, git(repo, "rev-parse", "HEAD")


def hermes_install(tmp_path, mode="normal"):
    agent = tmp_path / "hermes-agent"
    agent.mkdir(parents=True)
    shutil.copy(FIXTURES / "batch_runner.py", agent / "batch_runner.py")
    log = tmp_path / "batch-calls.jsonl"
    (agent / "fake_config.json").write_text(json.dumps({"log": str(log), "mode": mode}))
    return agent, log


def benchmark_record(identifier="hermes-task", checks=None, subjective=None, structured=None):
    record = {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "manual", "provenance": "unit-test"},
        "task": {"title": "Answer", "prompt": "Write the answer into answer.txt"},
        "taxonomy": {"categories": ["skills"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "hermes_agent", "container_image": IMAGE, "network_policy": "disabled",
                        "repository_fixture": "repo"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {
            "deterministic": checks if checks is not None else [
                {"type": "file_content", "path": "answer.txt", "equals": "42\n"},
                {"type": "unchanged_path", "path": "fixture.txt"},
            ],
            "subjective": subjective or [],
        },
        "ground_truth": {"type": "deterministic_test", "confidence": 1.0, **(
            {"structured_result": structured} if structured else {})},
        "benchmark": {"split": "TRAIN", "suites": ["SMOKE", "CORE"], "cluster_id": identifier,
                      "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }
    return record


def make_runner(tmp_path, mode="normal", **kwargs):
    repo, baseline, candidate = brain_repo(tmp_path)
    agent, log = hermes_install(tmp_path, mode)
    fixtures = tmp_path / "fixtures"
    (fixtures / "repo").mkdir(parents=True)
    (fixtures / "repo" / "fixture.txt").write_text("input\n")
    runner = HermesBatchRunner(
        repo, ArtifactStore(tmp_path / "artifacts"),
        hermes_agent_dir=agent, model=MODEL,
        executor_factory=lambda benchmark: LocalSandboxExecutor(acknowledge_no_isolation=True),
        python_executable=sys.executable, fixtures_root=fixtures,
        terminal_env="local", allow_local_terminal=True,
        limits=kwargs.pop("limits", SandboxLimits(timeout_seconds=60)), **kwargs,
    )
    return runner, baseline, candidate, log


def test_parse_real_schema_trajectory_fixture():
    trajectory = load_single_trajectory(FIXTURES / "trajectories.jsonl")
    assert trajectory.completed and not trajectory.partial
    assert trajectory.tool_calls == 3 and trajectory.failed_tool_calls == 1
    assert trajectory.assistant_turns == 3 and trajectory.api_calls == 4
    assert trajectory.final_output == "There are 2 Python files: src/a.py and src/b.py."
    assert trajectory.context_chars > 300


@pytest.mark.parametrize(
    "mutation",
    [
        {"completed": "yes"},
        {"api_calls": -1},
        {"conversations": []},
        {"conversations": [{"from": "robot", "value": "x"}]},
        {"tool_stats": {"terminal": {"count": 1, "success": 1, "failure": 1}}},
    ],
)
def test_parse_trajectory_rejects_malformed_rows(mutation):
    record = json.loads((FIXTURES / "trajectories.jsonl").read_text())
    record.update(mutation)
    with pytest.raises(IntegrityError):
        parse_trajectory_record(record)


def test_missing_or_ambiguous_trajectory_is_never_success(tmp_path):
    assert load_single_trajectory(tmp_path / "missing.jsonl") is None
    row = (FIXTURES / "trajectories.jsonl").read_text().strip()
    duplicated = tmp_path / "two.jsonl"
    duplicated.write_text(row + "\n" + row + "\n")
    with pytest.raises(IntegrityError, match="unexpected number"):
        load_single_trajectory(duplicated)


def test_hermes_runner_invokes_batch_runner_with_isolated_brain_and_parses_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCTION_DB_PASSWORD", "never-forward")
    runner, baseline, candidate, log = make_runner(tmp_path)
    task = Benchmark.from_dict(benchmark_record())

    good = runner.run(candidate, task, 3)
    bad = runner.run(baseline, task, 3)

    assert good.success and good.deterministic_pass and good.exit_state == "completed"
    assert not bad.success and not bad.deterministic_pass
    assert good.final_output == "The answer is 42."
    assert (good.tool_calls, good.failed_tool_calls, good.turns, good.api_calls, good.retries) == (1, 0, 2, 3, 1)
    assert good.token_usage_reported is False and good.tokens == 0
    assert good.filesystem_changes == ("A answer.txt",)
    assert good.environment_digest == bad.environment_digest
    assert good.model_config_digest == MODEL.digest == bad.model_config_digest
    trajectory = json.loads(runner.artifacts.get(good.trajectory_digest))
    assert trajectory["prompt_index"] == 0
    assert good.trajectory_digest in good.artifact_digests

    call = json.loads(log.read_text().splitlines()[0])
    flags = dict(arg[2:].split("=", 1) for arg in call["argv"])
    assert flags["batch_size"] == "1" and flags["num_workers"] == "1"
    assert flags["model"] == "qwen3-coder" and flags["max_turns"] == "30"
    env = call["env"]
    assert "PRODUCTION_DB_PASSWORD" not in env
    assert env["TERMINAL_DOCKER_NETWORK"] == "false"
    assert env["TERMINAL_DOCKER_IMAGE"] == IMAGE and env["TERMINAL_DOCKER_IMAGE_PINNED"] == "1"
    assert env["TERMINAL_DOCKER_FORWARD_ENV"] == "[]" and env["HERMES_EVAL_SEED"] == "3"
    assert env["HERMES_HOME"].endswith("hermes-home")
    assert Path(call["cwd"]).name == "run"


def test_hermes_runner_reports_agent_error_and_tool_limit_as_failures(tmp_path):
    task = Benchmark.from_dict(benchmark_record(checks=[{"type": "exit_code"}]))
    runner, _baseline, candidate, _log = make_runner(tmp_path / "a", mode="agent_error")
    failed = runner.run(candidate, task, 1)
    assert failed.exit_state == "agent_error" and not failed.success and failed.trajectory_digest == ""

    limited, _b, commit, _l = make_runner(tmp_path / "b", mode="tool_storm",
                                          limits=SandboxLimits(timeout_seconds=60, max_tool_calls=10))
    stormed = limited.run(commit, task, 1)
    assert stormed.exit_state == "tool_call_limit_exceeded" and not stormed.success


def test_hermes_runner_crash_is_infrastructure_error(tmp_path):
    runner, _baseline, candidate, _log = make_runner(tmp_path, mode="crash")
    with pytest.raises(RuntimeError, match="batch_runner failed"):
        runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)


def test_hermes_runner_refuses_unsandboxed_or_networked_configuration(tmp_path):
    repo, _baseline, candidate = brain_repo(tmp_path)
    agent, _log = hermes_install(tmp_path)
    factory = lambda benchmark: LocalSandboxExecutor(acknowledge_no_isolation=True)  # noqa: E731
    with pytest.raises(SandboxPolicyError, match="unsandboxed"):
        HermesBatchRunner(repo, ArtifactStore(tmp_path / "a"), hermes_agent_dir=agent, model=MODEL,
                          executor_factory=factory, terminal_env="local")
    runner = HermesBatchRunner(repo, ArtifactStore(tmp_path / "a"), hermes_agent_dir=agent, model=MODEL,
                               executor_factory=factory)
    record = benchmark_record()
    record["environment"]["network_policy"] = "allowlist"
    record["environment"]["network_allowlist"] = ["api:443"]
    with pytest.raises(SandboxPolicyError, match="networking disabled"):
        runner.run(candidate, Benchmark.from_dict(record), 1)
    with pytest.raises(FileNotFoundError):
        HermesBatchRunner(repo, ArtifactStore(tmp_path / "a"), hermes_agent_dir=tmp_path, model=MODEL,
                          executor_factory=factory)
    with pytest.raises(ValueError):
        HermesModelConfig(model="", base_url="http://x")


def store_with(tmp_path, records):
    root = tmp_path / "benchmarks"
    root.mkdir()
    for record in records:
        (root / f"{record['id']}.json").write_text(json.dumps(record))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    return BenchmarkStore(root, AccessController(), holdout_root=holdout)


def test_paired_evaluation_through_hermes_runner_uses_run_policy(tmp_path):
    runner, baseline, candidate, log = make_runner(tmp_path)
    store = store_with(tmp_path, [benchmark_record()])
    subject = Candidate("cand-h", "exp-h", CandidateType.SKILL_PATCH, "brain/skills/answer",
                        baseline, candidate, "diff")

    report = PairedEvaluator(store, runner, PromotionPolicy()).evaluate(
        subject, baseline_commit=baseline, benchmark_ids=["hermes-task"], suite="SMOKE",
        benchmark_version=store.version,
    )

    assert report.metrics["run_policy"]["runs_per_task"] == {"hermes-task": 2}
    assert report.metrics["run_policy"]["llm_influenced"] is True
    assert len(log.read_text().splitlines()) == 4
    outcome = report.outcomes[0]
    assert outcome.candidate_success and not outcome.baseline_success and outcome.run_count == 2
    efficiency = report.metrics["efficiency"]
    assert efficiency["candidate"]["tokens_per_success"] is None, "missing token usage must not look like zero"
    assert efficiency["candidate"]["token_usage_reported"] is False
    assert report.metrics["paired_statistics"]["quality"]["mean_delta"] > 0
    with pytest.raises(ValueError, match="run policy requires at least 3"):
        PairedEvaluator(store, runner, PromotionPolicy()).evaluate(
            subject, baseline_commit=baseline, benchmark_ids=["hermes-task"], suite="CORE",
            benchmark_version=store.version, seeds=[1],
        )
