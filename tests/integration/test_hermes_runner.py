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
from openai_stub import ScriptedModelStub, chat_completion_json, chat_completion_sse


FIXTURES = Path(__file__).parents[1] / "fixtures" / "hermes"
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


def hermes_install(tmp_path, mode="normal", **fake):
    agent = tmp_path / "hermes-agent"
    agent.mkdir(parents=True)
    shutil.copy(FIXTURES / "batch_runner.py", agent / "batch_runner.py")
    shutil.copy(FIXTURES / "run_agent.py", agent / "run_agent.py")
    log = tmp_path / "batch-calls.jsonl"
    (agent / "fake_config.json").write_text(json.dumps({"log": str(log), "mode": mode, **fake}))
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


def make_runner(tmp_path, mode="normal", fake=None, **kwargs):
    repo, baseline, candidate = brain_repo(tmp_path)
    agent, log = hermes_install(tmp_path, mode, **(fake or {}))
    fixtures = tmp_path / "fixtures"
    (fixtures / "repo").mkdir(parents=True)
    (fixtures / "repo" / "fixture.txt").write_text("input\n")
    runner = HermesBatchRunner(
        repo, ArtifactStore(tmp_path / "artifacts"),
        hermes_agent_dir=agent, model=kwargs.pop("model", MODEL),
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


def test_hermes_runner_invokes_the_driver_with_isolated_brain_and_parses_result(tmp_path, monkeypatch):
    monkeypatch.setenv("PRODUCTION_DB_PASSWORD", "never-forward")
    runner, baseline, candidate, log = make_runner(tmp_path)
    task = Benchmark.from_dict(benchmark_record())

    good = runner.run(candidate, task, 3)
    bad = runner.run(baseline, task, 3)

    assert good.success and good.deterministic_pass and good.exit_state == "completed"
    assert not bad.success and not bad.deterministic_pass
    assert good.final_output == "The answer is 42."
    assert (good.tool_calls, good.failed_tool_calls, good.turns, good.api_calls, good.retries) == (1, 0, 2, 3, 1)
    # The fake made no model calls although the trajectory reports 3: traffic that
    # bypassed the metering proxy must never count as measured.
    assert good.token_usage_reported is False and good.tokens == 0
    assert good.filesystem_changes == ("A answer.txt",)
    assert good.environment_digest == bad.environment_digest
    assert good.model_config_digest == MODEL.digest == bad.model_config_digest
    trajectory = json.loads(runner.artifacts.get(good.trajectory_digest))
    assert trajectory["prompt_index"] == 0
    assert good.trajectory_digest in good.artifact_digests

    call = json.loads(log.read_text().splitlines()[0])
    flags = dict(arg[2:].split("=", 1) for arg in call["argv"])
    assert flags["model"] == "qwen3-coder" and flags["max-turns"] == "30"
    assert flags["toolsets"] == "terminal,file,skills"
    assert call["agent"]["enabled_toolsets"] == ["terminal", "file", "skills"]
    assert call["agent"]["skip_context_files"] is True and call["agent"]["skip_memory"] is True
    assert call["agent"]["max_iterations"] == 30 and call["task_id"] == "task_0"
    record = json.loads(runner.artifacts.get(good.artifact_digests[0]))
    assert record["hermes_reported_usage"] == {"input_tokens": 0, "output_tokens": 0, "api_calls": 3}
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
    with pytest.raises(RuntimeError, match="Hermes driver failed"):
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


def flags_of(log, index=0):
    call = json.loads(log.read_text().splitlines()[index])
    return dict(arg[2:].split("=", 1) for arg in call["argv"])


def test_metering_proxy_measures_tokens_of_every_model_call(tmp_path, monkeypatch):
    monkeypatch.setenv("HSI_TEST_MODEL_KEY", "sk-test-only")
    replies = iter([chat_completion_json(100, 20), chat_completion_sse(140, 30), chat_completion_json(180, 10)])

    def reply(method, path, body):
        raw = next(replies)
        return 200, "text/event-stream" if raw.startswith(b"data:") else "application/json", raw

    with ScriptedModelStub(reply) as upstream:
        model = HermesModelConfig(model="qwen3-coder", base_url=upstream.base_url, api_key_env="HSI_TEST_MODEL_KEY")
        runner, _baseline, candidate, log = make_runner(tmp_path, fake={"model_calls": 3}, model=model)
        observation = runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)

    assert observation.success and observation.api_calls == 3
    assert observation.token_usage_reported is True
    assert (observation.input_tokens, observation.output_tokens, observation.tokens) == (420, 60, 480)
    assert observation.context_tokens == 180
    flags = flags_of(log)
    assert flags["base-url"].startswith("http://127.0.0.1:") and flags["base-url"] != upstream.base_url
    assert all("sk-test-only" not in arg for arg in json.loads(log.read_text().splitlines()[0])["argv"])
    assert [request[1] for request in upstream.requests] == ["/v1/chat/completions"] * 3
    assert all(request[2]["Authorization"] == "Bearer sk-test-only" for request in upstream.requests)
    record = json.loads(runner.artifacts.get(observation.artifact_digests[0]))
    assert record["token_usage"]["measured"] is True and record["token_usage"]["generation_calls"] == 3
    assert "sk-test-only" not in json.dumps(record)


def test_metering_marks_run_unmeasured_when_a_response_lacks_usage(tmp_path):
    replies = iter([chat_completion_json(100, 20), chat_completion_json(None, None), chat_completion_json(1, 1)])
    with ScriptedModelStub(lambda method, path, body: (200, "application/json", next(replies))) as upstream:
        model = HermesModelConfig(model="qwen3-coder", base_url=upstream.base_url)
        runner, _baseline, candidate, _log = make_runner(tmp_path, fake={"model_calls": 3}, model=model)
        observation = runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)
    assert observation.token_usage_reported is False
    assert (observation.tokens, observation.input_tokens, observation.output_tokens) == (0, 0, 0)


def test_metering_detects_model_traffic_that_bypassed_the_proxy(tmp_path):
    with ScriptedModelStub(lambda method, path, body: (200, "application/json", chat_completion_json(9, 9))) as upstream:
        model = HermesModelConfig(model="qwen3-coder", base_url=upstream.base_url)
        runner, _baseline, candidate, _log = make_runner(tmp_path, fake={"model_calls": 1}, model=model)
        observation = runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)
    assert observation.api_calls == 3 and len(upstream.requests) == 1
    assert observation.token_usage_reported is False and observation.tokens == 0


def test_token_metering_can_be_disabled_and_then_reports_unmeasured(tmp_path):
    runner, _baseline, candidate, log = make_runner(tmp_path, token_metering=False)
    observation = runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)
    assert flags_of(log)["base-url"] == MODEL.base_url
    assert observation.token_usage_reported is False and observation.tokens == 0


def test_measured_hermes_runs_reach_the_efficiency_report(tmp_path):
    def reply(method, path, body):
        return 200, "application/json", chat_completion_json(50, 10)

    with ScriptedModelStub(reply) as upstream:
        model = HermesModelConfig(model="qwen3-coder", base_url=upstream.base_url)
        runner, baseline, candidate, _log = make_runner(tmp_path, fake={"model_calls": 3}, model=model)
        store = store_with(tmp_path, [benchmark_record()])
        subject = Candidate("cand-m", "exp-m", CandidateType.SKILL_PATCH, "brain/skills/answer",
                            baseline, candidate, "diff")
        report = PairedEvaluator(store, runner, PromotionPolicy()).evaluate(
            subject, baseline_commit=baseline, benchmark_ids=["hermes-task"], suite="SMOKE",
            benchmark_version=store.version,
        )
    efficiency = report.metrics["efficiency"]
    assert efficiency["candidate"]["token_usage_reported"] is True
    assert efficiency["candidate"]["tokens_per_success"] == 180
    assert report.outcomes[0].token_usage_reported is True


def test_task_image_and_workspace_cwd_are_registered_for_the_docker_mount(tmp_path):
    # Hermes v0.22.2 bind-mounts only a task's ``cwd`` override (registered from the
    # task's ``image``/``cwd`` by batch_runner's helper); without it the agent edits
    # the container's private filesystem (found in the isolated lab).
    runner, _baseline, candidate, log = make_runner(tmp_path)
    observation = runner.run(candidate, Benchmark.from_dict(benchmark_record()), 1)
    call = json.loads(log.read_text().splitlines()[0])
    assert call["overrides"]["docker_image"] == IMAGE
    assert call["overrides"]["cwd"] == call["env"]["TERMINAL_CWD"]
    assert call["overrides"]["cwd"].endswith("workspace") and observation.success


def test_skills_reach_the_agent_only_through_the_skills_toolset(tmp_path):
    # Lab finding F10: batch_runner's distributions never enable ``skills``, so
    # brain skills could not influence any run. The driver enables it explicitly.
    task = Benchmark.from_dict(benchmark_record())
    with_skills, _baseline, candidate, _log = make_runner(tmp_path / "a")
    without, _b, commit, _l = make_runner(
        tmp_path / "b", model=HermesModelConfig(model="qwen3-coder", base_url=MODEL.base_url, toolsets="terminal,file"),
    )
    assert with_skills.run(candidate, task, 1).success
    blind = without.run(commit, task, 1)
    assert not blind.success and blind.final_output == "The answer is unknown."


def test_agent_failure_before_a_trajectory_is_agent_error_not_infrastructure(tmp_path):
    runner, _baseline, candidate, _log = make_runner(tmp_path, mode="agent_error")
    observation = runner.run(candidate, Benchmark.from_dict(benchmark_record(checks=[{"type": "exit_code"}])), 1)
    assert observation.exit_state == "agent_error" and not observation.success


def test_toolsets_are_validated_and_part_of_the_model_digest():
    assert HermesModelConfig(model="m", base_url="http://x").digest != HermesModelConfig(
        model="m", base_url="http://x", toolsets="terminal,file").digest
    for bad in ("", "terminal;rm", "Terminal", "a," * 20):
        with pytest.raises(ValueError, match="toolsets"):
            HermesModelConfig(model="m", base_url="http://x", toolsets=bad)
