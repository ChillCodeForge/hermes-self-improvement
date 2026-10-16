from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_improve.docker_runner import DockerCommandRunner
from hermes_improve.models import Benchmark
from hermes_improve.verifier import verify_deterministic_output


def benchmark(checks, *, expected_exit=0) -> Benchmark:
    return Benchmark.from_dict({
        "schema_version": 1,
        "id": "verifier-task",
        "version": 1,
        "source": {"type": "manual"},
        "task": {"title": "Verifier test", "prompt": "Return a structured result"},
        "taxonomy": {"categories": [], "required_capabilities": []},
        "environment": {
            "container_image": "example/image@sha256:" + "a" * 64,
            "network_policy": "disabled",
            "command": ["python", "candidate.py"],
        },
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0},
        "verification": {"deterministic": checks, "subjective": []},
        "ground_truth": {"expected_exit_code": expected_exit},
        "benchmark": {
            "split": "TRAIN", "suites": ["SMOKE"], "cluster_id": "verifier",
            "criticality": "normal",
        },
        "security": {"sanitized": True, "contains_secrets": False},
    })


def test_verifier_checks_exit_code_and_typed_json_path():
    task = benchmark([
        {"type": "exit_code"},
        {"type": "json_path_equals", "path": ["answer"], "expected": 42},
    ])
    passed, quality, results = verify_deterministic_output(
        task, exit_code=0, output='{"answer":42}\n'
    )
    assert passed
    assert quality == 100.0
    assert results == [
        {"type": "exit_code", "passed": True},
        {"type": "json_path_equals", "passed": True},
    ]


def test_verifier_rejects_wrong_output_even_when_command_exits_successfully():
    task = benchmark([
        {"type": "output_contains", "expected": "answer"},
        {"type": "json_path_equals", "path": ["answer"], "expected": 42},
    ])
    passed, quality, results = verify_deterministic_output(
        task, exit_code=0, output='{"answer":41}\n'
    )
    assert not passed
    assert quality == pytest.approx(200 / 3)
    assert results[0]["passed"] and results[1]["passed"] and not results[2]["passed"]


def test_verifier_rejects_nonstandard_json_numbers():
    task = benchmark([
        {"type": "json_path_equals", "path": ["answer"], "expected": 1},
    ])
    passed, _, _ = verify_deterministic_output(
        task, exit_code=0, output='{"answer":NaN}'
    )
    assert not passed


def test_verifier_does_not_treat_json_boolean_as_integer():
    task = benchmark([
        {"type": "json_path_equals", "path": ["answer"], "expected": 1},
    ])
    passed, _, _ = verify_deterministic_output(
        task, exit_code=0, output='{"answer":true}'
    )
    assert not passed


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        (["unknown-check"], "unsupported deterministic verifier"),
        ([{"type": "output_contains"}], "expected value must be a string"),
        ([{"type": "json_path_equals", "path": ["missing"]}], "requires an expected value"),
        ([{"type": "exit_code", "expected": 1}], "must match ground_truth"),
    ],
)
def test_verifier_fails_closed_on_invalid_specs(checks, message):
    with pytest.raises(ValueError, match=message):
        verify_deterministic_output(benchmark(checks), exit_code=0, output="{}")


def test_docker_runner_applies_verifier_instead_of_only_exit_code(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_improve.docker_runner.shutil.which", lambda _name: "/usr/bin/docker")

    class Artifacts:
        payload = None

        def put(self, payload):
            self.payload = payload
            return "b" * 64, tmp_path / "artifact.json"

    artifacts = Artifacts()
    runner = DockerCommandRunner(tmp_path, artifacts)
    runner._tree_digest = lambda _commit: "c" * 64
    runner._export_commit = lambda _commit, _destination: None
    monkeypatch.setattr(
        "hermes_improve.docker_runner.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout='{"answer":41}\n'),
    )
    task = benchmark([
        {"type": "json_path_equals", "path": ["answer"], "expected": 42},
    ])

    result = runner.run("a" * 40, task, 7)
    assert not result.success
    assert not result.deterministic_pass
    assert result.quality == 50.0
    artifact = json.loads(artifacts.payload)
    assert artifact["verifier_results"] == [
        {"type": "exit_code", "passed": True},
        {"type": "json_path_equals", "passed": False},
    ]
