from __future__ import annotations

import json
import subprocess

import pytest

from fake_docker import calls, install
from hermes_improve.artifacts import ArtifactStore
from hermes_improve.docker_runner import DockerCommandRunner
from hermes_improve.models import Benchmark
from hermes_improve.sandbox import (
    DockerSandboxExecutor,
    NetworkPolicy,
    SandboxLimits,
    SandboxPolicyError,
    _docker_env,
)


IMAGE = "example/python@sha256:" + "a" * 64


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit_tree(tmp_path, files):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if not (repo / ".git").exists():
        git(repo, "init", "-q")
        git(repo, "config", "user.name", "Test")
        git(repo, "config", "user.email", "test@example.invalid")
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixture")
    return repo, git(repo, "rev-parse", "HEAD")


def benchmark(checks, *, environment=None, expected_exit=0):
    env = {
        "container_image": IMAGE,
        "network_policy": "disabled",
        "command": ["python3", "candidate.py"],
    }
    env.update(environment or {})
    return Benchmark.from_dict({
        "schema_version": 1, "id": "docker-task", "version": 1,
        "source": {"type": "manual", "provenance": "unit-test"},
        "task": {"title": "Docker task", "prompt": "Write the answer"},
        "taxonomy": {"categories": ["code"], "difficulty": "easy", "required_capabilities": []},
        "environment": env,
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": checks, "subjective": []},
        "ground_truth": {"type": "deterministic_test", "confidence": 1.0, "expected_exit_code": expected_exit},
        "benchmark": {"split": "TRAIN", "suites": ["SMOKE"], "cluster_id": "docker", "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    })


CANDIDATE = """\
import json, pathlib
pathlib.Path("answer.txt").write_text("42\\n")
pathlib.Path("README.md").unlink()
print(json.dumps({"answer": ANSWER}))
"""


def runner_for(tmp_path, answer=42, **kwargs):
    repo, commit = commit_tree(tmp_path, {
        "candidate.py": CANDIDATE.replace("ANSWER", str(answer)),
        "README.md": "fixture\n",
        "protected/config.txt": "keep\n",
        "check.py": "import pathlib, sys\nsys.exit(0 if pathlib.Path('answer.txt').read_text().strip() == '42' else 3)\n",
    })
    docker = install(tmp_path, network=kwargs.pop("network_json", None))
    runner = DockerCommandRunner(repo, ArtifactStore(tmp_path / "artifacts"), docker_binary=str(docker), **kwargs)
    return runner, commit, docker


def test_docker_runner_captures_workspace_changes_and_runs_verifiers_in_sandbox(tmp_path):
    runner, commit, _docker = runner_for(tmp_path)
    task = benchmark([
        {"type": "json_path_equals", "path": ["answer"], "expected": 42},
        {"type": "file_content", "path": "answer.txt", "contains": "42"},
        {"type": "unchanged_path", "path": "protected"},
        {"type": "git_diff", "allowed_paths": ["answer.txt", "README.md"], "max_changed_files": 2},
        {"type": "unit_test", "argv": ["python3", "check.py"], "label": "answer check"},
    ])

    result = runner.run(commit, task, 7)

    assert result.success and result.deterministic_pass and result.quality == 100.0
    # No model runs and no token usage is measured: the zero must not pass as a measurement.
    assert result.tokens == 0 and result.token_usage_reported is False
    assert [row["type"] for row in result.verifier_results] == [
        "exit_code", "json_path_equals", "file_content", "unchanged_path", "git_diff", "unit_test",
    ]
    assert result.filesystem_changes == ("D README.md", "A answer.txt")
    assert result.exit_state == "exited"
    assert result.limits_digest == runner.limits.digest
    diff = runner.artifacts.get(result.diff_digest).decode()
    assert "+++ b/answer.txt" in diff and "--- a/README.md" in diff
    record = json.loads(runner.artifacts.get(result.artifact_digests[0]))
    assert record["verifier_results"][-1]["label"] == "answer check"
    runs = [call for call in calls(tmp_path) if call[0] == "run"]
    assert len(runs) == 2, "task and command verifier must each run in their own container"
    for argv in runs:
        for flag in ("--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL",
                     "--security-opt=no-new-privileges", "--ipc=none", "--pull=never"):
            assert flag in argv
    assert "HERMES_EVAL_SEED=7" in runs[0]


def test_docker_runner_applies_verifier_instead_of_only_exit_code(tmp_path):
    runner, commit, _docker = runner_for(tmp_path, answer=41)
    task = benchmark([{"type": "json_path_equals", "path": ["answer"], "expected": 42}])

    result = runner.run(commit, task, 7)

    assert not result.success
    assert not result.deterministic_pass
    assert result.quality == 50.0
    artifact = json.loads(runner.artifacts.get(result.artifact_digests[0]))
    assert [(row["type"], row["passed"]) for row in artifact["verifier_results"]] == [
        ("exit_code", True), ("json_path_equals", False),
    ]


def test_docker_runner_command_verifier_failure_blocks_success(tmp_path):
    runner, commit, _docker = runner_for(tmp_path)
    task = benchmark([{"type": "build", "argv": ["python3", "-c", "raise SystemExit(4)"]}])

    result = runner.run(commit, task, 1)

    assert not result.success
    assert result.verifier_results[-1] == {"type": "build", "passed": False}


def test_docker_runner_times_out_and_kills_container(tmp_path):
    runner, commit, _docker = runner_for(tmp_path, default_timeout=30)
    task = benchmark(
        [{"type": "file_exists", "path": "answer.txt"}],
        environment={"command": ["python3", "-c", "import time; time.sleep(30)"], "timeout_seconds": 1},
    )

    result = runner.run(commit, task, 1)

    assert result.exit_state == "timeout"
    assert result.exit_code == 124
    assert not result.success
    assert any(call[0] == "kill" for call in calls(tmp_path))


def test_docker_runner_rejects_network_when_sandbox_network_is_disabled(tmp_path):
    runner, commit, _docker = runner_for(tmp_path)
    task = benchmark(["exit_code"], environment={"network_policy": "allowlist", "network_allowlist": ["fixture-api:8080"]})
    with pytest.raises(SandboxPolicyError, match="networking disabled"):
        runner.run(commit, task, 1)


def internal_network(**overrides):
    network = {
        "Name": "hermes-eval-net", "Internal": True,
        "Containers": {"abc": {"Name": "fixture-api"}},
    }
    network.update(overrides)
    return network


def test_docker_runner_allowlist_network_uses_dedicated_internal_network(tmp_path):
    policy = NetworkPolicy("allowlist", "hermes-eval-net", frozenset({"fixture-api:8080"}))
    runner, commit, _docker = runner_for(tmp_path, network=policy, network_json=internal_network())
    task = benchmark(["exit_code"], environment={"network_policy": "allowlist", "network_allowlist": ["fixture-api:8080"]})

    result = runner.run(commit, task, 1)

    assert result.success
    run_argv = next(call for call in calls(tmp_path) if call[0] == "run")
    assert "--network=hermes-eval-net" in run_argv
    assert "--network=none" not in run_argv


@pytest.mark.parametrize(
    ("network_json", "requested", "message"),
    [
        (internal_network(Internal=False), ["fixture-api:8080"], "--internal"),
        (internal_network(Containers={"x": {"Name": "evil-proxy"}}), ["fixture-api:8080"], "non-allowlisted"),
        (internal_network(), ["other-host:443"], "outside the evaluator allowlist"),
        (None, ["fixture-api:8080"], "cannot be inspected"),
    ],
)
def test_docker_runner_allowlist_violations_fail_closed(tmp_path, network_json, requested, message):
    policy = NetworkPolicy("allowlist", "hermes-eval-net", frozenset({"fixture-api:8080"}))
    runner, commit, _docker = runner_for(tmp_path, network=policy, network_json=network_json)
    task = benchmark(["exit_code"], environment={"network_policy": "allowlist", "network_allowlist": requested})
    with pytest.raises(SandboxPolicyError, match=message):
        runner.run(commit, task, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "allowlist", "network_name": "bridge", "allowed_hosts": frozenset({"a:1"})},
        {"mode": "allowlist", "network_name": "evalnet", "allowed_hosts": frozenset()},
        {"mode": "disabled", "network_name": "evalnet"},
        {"mode": "open"},
    ],
)
def test_network_policy_rejects_unsafe_configuration(kwargs):
    with pytest.raises(ValueError):
        NetworkPolicy(**kwargs)


def test_docker_executor_requires_pinned_image_and_valid_limits(tmp_path):
    docker = install(tmp_path)
    with pytest.raises(ValueError, match="pinned"):
        DockerSandboxExecutor("python:3.12", docker_binary=str(docker))
    with pytest.raises(ValueError):
        SandboxLimits(memory="lots")
    with pytest.raises(ValueError):
        SandboxLimits(max_tool_calls=0)
    with pytest.raises(ValueError):
        SandboxLimits(timeout_seconds=float("inf"))


DAEMON_SELECTION = {
    "DOCKER_HOST": "tcp://remote-daemon.invalid:2376",
    "DOCKER_CONTEXT": "lab",
    "DOCKER_TLS_VERIFY": "1",
    "DOCKER_CERT_PATH": "/etc/hermes/docker-certs",
    "DOCKER_CONFIG": "/etc/hermes/docker",
}


def test_docker_client_env_forwards_daemon_selection_and_nothing_else(monkeypatch):
    for key, value in DAEMON_SELECTION.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "never-forward")
    monkeypatch.setenv("DOCKER_AUTH_CONFIG", "never-forward")
    env = _docker_env()
    assert {key: env[key] for key in DAEMON_SELECTION} == DAEMON_SELECTION
    assert set(env) == {"PATH", "HOME", *DAEMON_SELECTION}


def test_docker_client_env_omits_unset_daemon_selection(monkeypatch):
    for key in DAEMON_SELECTION:
        monkeypatch.delenv(key, raising=False)
    assert set(_docker_env()) == {"PATH", "HOME"}


def test_docker_executor_reports_unreachable_daemon_instead_of_missing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", DAEMON_SELECTION["DOCKER_HOST"])
    docker = install(tmp_path, daemon_unreachable=True)
    executor = DockerSandboxExecutor(IMAGE, docker_binary=str(docker))
    with pytest.raises(RuntimeError, match="daemon") as caught:
        executor.execute(["true"], input_root=tmp_path, snapshot_dir=tmp_path / "out")
    message = str(caught.value)
    assert "snapshot" not in message
    # The CLI saw the forwarded DOCKER_HOST, and its diagnosis reaches the operator.
    assert DAEMON_SELECTION["DOCKER_HOST"] in message


def test_docker_executor_rejects_non_evaluator_environment_variables(tmp_path):
    docker = install(tmp_path)
    executor = DockerSandboxExecutor(IMAGE, docker_binary=str(docker))
    with pytest.raises(SandboxPolicyError, match="HERMES_EVAL_"):
        executor.execute(["true"], input_root=tmp_path, env={"AWS_SECRET_ACCESS_KEY": "x"})
