"""Spec §35/§36: Qwen usage, GPU runtime and judge uncertainty in the report; decision recording."""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import threading

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openai_stub import OpenAIStub
from hermes_improve import cli
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.candidate_builder import create_skill_candidate
from hermes_improve.config import default_config
from hermes_improve.decisions import DecisionRecorder
from hermes_improve.evaluator import judge_route
from hermes_improve.ingestion import SessionQueue
from hermes_improve.models import CandidateType, RunObservation
from hermes_improve.observability import (
    GpuRuntimeProbe,
    ObservabilityLog,
    build_report,
    record_qwen_usage,
    sum_prometheus_counters,
)
from hermes_improve.store import ExperimentStore


class MetricsServer:
    """Loopback Prometheus endpoint whose seconds counters grow by ``step`` on every scrape."""

    def __init__(self, *, step: float = 0.25, start: float = 1.0, body: str | None = None) -> None:
        self.scrapes = 0
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                value = start + server.scrapes * step
                server.scrapes += 1
                text = body if body is not None else (
                    "# HELP llamacpp:prompt_seconds_total Prompt processing time\n"
                    "# TYPE llamacpp:prompt_seconds_total counter\n"
                    f"llamacpp:prompt_seconds_total {value}\n"
                    f'llamacpp:tokens_predicted_seconds_total{{slot="0"}} {value}\n'
                    f'llamacpp:tokens_predicted_seconds_total{{slot="1"}} {value}\n'
                    "llamacpp:requests_processing 1\n"
                )
                payload = text.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/metrics"

    def __enter__(self) -> "MetricsServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


_METRICS = "llamacpp:prompt_seconds_total,llamacpp:tokens_predicted_seconds_total"


class FakeModel:
    """TEST DOUBLE with the ``stats`` shape of ``OpenAICompatibleQwen``."""

    def __init__(self, role: str = "qwen_dreamer") -> None:
        self.role = role
        self.requests = 0

    def call(self) -> None:
        self.requests += 1

    @property
    def stats(self):
        if not self.requests:
            return {}
        return {self.role: {
            "requests": self.requests, "input_tokens": 10 * self.requests,
            "output_tokens": 4 * self.requests, "latency_ms": 5.0 * self.requests,
        }}


def test_prometheus_counters_sum_every_series_and_require_each_metric():
    text = (
        "# TYPE a counter\n"
        'a{x="1"} 1.5\n'
        'a{x="2",y="a b"} 2.5 1700000000\n'
        "b 3\n"
        "ab 100\n"
    )
    assert sum_prometheus_counters(text, ("a", "b")) == 7.0
    with pytest.raises(ValueError, match="does not expose: c"):
        sum_prometheus_counters(text, ("a", "c"))
    with pytest.raises(ValueError, match="finite"):
        sum_prometheus_counters("a NaN\n", ("a",))


def test_gpu_probe_is_configured_only_from_the_environment_and_loopback_by_default():
    assert GpuRuntimeProbe.from_environment({}) is None
    with pytest.raises(ValueError, match="set together"):
        GpuRuntimeProbe.from_environment({"HERMES_GPU_METRICS_URL": "http://127.0.0.1:9/metrics"})
    with pytest.raises(ValueError, match="loopback"):
        GpuRuntimeProbe.from_environment({
            "HERMES_GPU_METRICS_URL": "http://gpu.example.invalid/metrics", "HERMES_GPU_RUNTIME_METRICS": "a",
        })
    with pytest.raises(ValueError, match="metric names"):
        GpuRuntimeProbe.from_environment({
            "HERMES_GPU_METRICS_URL": "http://127.0.0.1:9/metrics", "HERMES_GPU_RUNTIME_METRICS": "bad name",
        })
    remote = GpuRuntimeProbe.from_environment({
        "HERMES_GPU_METRICS_URL": "http://gpu.example.invalid/metrics", "HERMES_GPU_RUNTIME_METRICS": "a,b",
        "HERMES_GPU_METRICS_ALLOW_REMOTE": "true",
    })
    assert remote is not None and remote.metrics == ("a", "b")


def test_unconfigured_gpu_runtime_stays_unavailable_with_its_reason(tmp_path):
    home = tmp_path / "home"
    model = FakeModel()
    with record_qwen_usage(home, model, command="dream-rem", environ={}):
        model.call()
    qwen = build_report(home, tmp_path / "absent")["qwen"]
    assert qwen["requests"] == 1 and qwen["by_role"]["qwen_dreamer"]["requests"] == 1
    assert qwen["gpu_runtime"] == "unavailable"
    assert "HERMES_GPU_METRICS_URL" in qwen["gpu_runtime_reason"]


def test_gpu_runtime_is_the_counter_delta_across_the_command(tmp_path):
    home = tmp_path / "home"
    model = FakeModel("qwen_optimizer")
    with MetricsServer(step=0.25) as server:
        environ = {"HERMES_GPU_METRICS_URL": server.url, "HERMES_GPU_RUNTIME_METRICS": _METRICS}
        with record_qwen_usage(home, model, command="candidate-generate", environ=environ):
            model.call()
        assert server.scrapes == 2
    qwen = build_report(home, tmp_path / "absent")["qwen"]
    # Three series, each +0.25 between the before and after scrape.
    assert qwen["gpu_runtime"] == {
        "seconds": 0.75, "measured_commands": 1, "unmeasured_commands": {},
        "seconds_by_command": {"candidate-generate": 0.75},
    }
    assert "gpu_runtime_reason" not in qwen


def test_failed_or_reset_scrapes_are_recorded_as_unmeasured_and_never_fail_the_command(tmp_path):
    home = tmp_path / "home"
    model = FakeModel()
    with MetricsServer(step=-1.0, start=5.0) as server:
        environ = {"HERMES_GPU_METRICS_URL": server.url, "HERMES_GPU_RUNTIME_METRICS": _METRICS}
        with record_qwen_usage(home, model, command="dream-rem", environ=environ):
            model.call()
    with MetricsServer(body="other_metric 1\n") as server:
        environ = {"HERMES_GPU_METRICS_URL": server.url, "HERMES_GPU_RUNTIME_METRICS": _METRICS}
        with record_qwen_usage(home, model, command="dream-rem", environ=environ):
            model.call()
    closed = MetricsServer()
    url = closed.url
    closed.httpd.server_close()
    with record_qwen_usage(home, model, command="dream-deep", environ={
        "HERMES_GPU_METRICS_URL": url, "HERMES_GPU_RUNTIME_METRICS": _METRICS,
    }):
        model.call()
    gpu = build_report(home, tmp_path / "absent")["qwen"]["gpu_runtime"]
    assert gpu["seconds"] == 0 and gpu["measured_commands"] == 0
    assert gpu["unmeasured_commands"] == {"counter_reset": 1, "metrics_missing": 1, "scrape_failed": 1}


def test_qwen_usage_is_recorded_even_when_the_command_fails(tmp_path):
    home = tmp_path / "home"
    model = FakeModel("qwen_judge")
    with pytest.raises(RuntimeError, match="boom"):
        with record_qwen_usage(home, model, command="evaluate-judge", environ={}):
            model.call()
            raise RuntimeError("boom")
    assert build_report(home, tmp_path / "absent")["qwen"]["by_role"]["qwen_judge"]["requests"] == 1
    with pytest.raises(ValueError, match="unknown Qwen-backed command"):
        with record_qwen_usage(home, model, command="promote", environ={}):
            pass
    with pytest.raises(ValueError, match="only an ok"):
        ObservabilityLog(home / "observability.sqlite3").record_gpu_runtime(
            command="evolve", metrics=("a",), seconds=1.0, status="scrape_failed",
        )


def test_cli_dream_deep_persists_qwen_dreamer_usage(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_GPU_METRICS_URL", raising=False)
    hypotheses = tmp_path / "hypotheses.json"
    hypotheses.write_text(json.dumps([{"summary": "s", "issue": "i"}]))
    with OpenAIStub(lambda body: json.dumps({"candidates": []})) as stub:
        monkeypatch.setenv("QWEN_BASE_URL", stub.base_url)
        monkeypatch.setenv("QWEN_MODEL", "dreamer")
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "dream", "--stage", "deep", "--hypotheses", str(hypotheses),
        ])
        assert cli.main() == 0
    capsys.readouterr()
    qwen = build_report(home, tmp_path / "absent")["qwen"]
    assert qwen["by_role"] == {"qwen_optimizer": {"requests": 1, "input_tokens": 50, "output_tokens": 10}}


def test_cli_benchmarks_mine_records_the_eligibility_decision(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "public"))
    monkeypatch.delenv("QWEN_EMBEDDING_BASE_URL", raising=False)
    SessionQueue(home / "sessions.sqlite3").ingest({
        "session_id": "mined-1",
        "task": {"prompt": "Build the package and run its tests", "input_available": True},
        "verification": {"deterministic": [{"type": "exit_code"}]},
        "security": {"sanitized": True, "contains_secrets": False},
    })
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "benchmarks", "mine", "mined-1"])
    assert cli.main() == 0
    mined = json.loads(capsys.readouterr().out)
    [record] = DecisionRecorder(home / "experiments.sqlite3").iter_records(decision_type="benchmark_eligibility")
    assert record.selected_option == mined["eligibility"]
    assert record.available_options == ("EXCLUDED", "DREAMING_ONLY", "SOFT_BENCHMARK", "HARD_BENCHMARK")
    assert record.input_features["task_fingerprint"] == mined["task_fingerprint"]
    assert record.input_features["reproducibility_score"] == mined["reproducibility_score"]
    assert record.input_features["ground_truth_type"] == mined["ground_truth_type"]
    assert record.input_features["reasons"] == mined["reasons"]
    assert record.source == "benchmark-miner" and record.confidence == 1.0  # deterministic-test ground truth


# --- evaluate: suite_selection, needs_qwen, judge usage and uncertainty --------------------------------

_PROMPTS = {
    "smoke-task": "Repair the broken quoting in the nightly backup shell script",
    "smoke-unrelated": "Translate the onboarding glossary into a sorted YAML mapping",
    "core-task": "Add pagination limits to the audit log query helper",
    "full-task": "Pin every container base image in the deployment manifests",
    "essay-task": "Explain the root cause of the failed certificate renewal",
    "holdout-task": "Rotate the expiring TLS certificate referenced by the proxy unit",
}


def _benchmark(identifier, split, suites, impact_areas=None, *, subjective=False):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": _PROMPTS[identifier]},
        "taxonomy": {"categories": ["skills"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": (
            {"deterministic": [], "subjective": ["names the root cause"]} if subjective
            else {"deterministic": ["exit_code"], "subjective": []}
        ),
        "ground_truth": {"type": "structured", "confidence": 1.0, "structured_result": {"task": identifier}},
        "benchmark": {
            "split": split, "suites": suites, "cluster_id": identifier, "criticality": "normal",
            **({"impact_areas": impact_areas} if impact_areas else {}),
        },
        "security": {"sanitized": True, "contains_secrets": False},
    }


class _Runner:
    llm_influenced = False

    def __init__(self, baseline_commit):
        self.baseline_commit = baseline_commit

    def run(self, commit, benchmark, seed):
        candidate = commit != self.baseline_commit
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed, environment_digest="sha256:" + "e" * 64,
            success=True, quality=10 if candidate else 5, tokens=100, latency_ms=20,
            deterministic_pass=True, security_pass=True, tool_calls=2,
            verifier_results=({"type": "exit_code", "passed": True},),
            final_output="candidate answer" if candidate else "baseline answer",
        )


def _git(repository, *args):
    return subprocess.run(
        ["git", "-C", str(repository), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_cli_evaluate_records_suite_selection_needs_qwen_and_judge_metrics(tmp_path, monkeypatch, capsys):
    public_root, holdout_root = tmp_path / "benchmarks", tmp_path / "holdout"
    public_root.mkdir()
    holdout_root.mkdir()
    for identifier, suites, areas, subjective in (
        ("smoke-task", ["SMOKE"], ["skills/demo*"], False),
        ("smoke-unrelated", ["SMOKE"], ["skills/unrelated*"], False),
        ("core-task", ["CORE"], None, False),
        ("full-task", ["FULL"], None, False),
        ("essay-task", ["FULL"], None, True),
    ):
        (public_root / f"{identifier}.json").write_text(
            json.dumps(_benchmark(identifier, "REGRESSION", suites, areas, subjective=subjective))
        )
    (holdout_root / "holdout-task.json").write_text(
        json.dumps(_benchmark("holdout-task", "SEALED_HOLDOUT", ["SEALED_HOLDOUT"]))
    )
    version = BenchmarkStore(public_root, holdout_root=holdout_root).version
    repository = tmp_path / "brain-repo"
    repository.mkdir()
    _git(repository, "init", "-q", "-b", "main")
    _git(repository, "config", "user.name", "Test")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "brain/skills/demo").mkdir(parents=True)
    (repository / "brain/skills/demo/SKILL.md").write_text("original\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-q", "-m", "baseline")
    baseline = _git(repository, "rev-parse", "HEAD")
    home = tmp_path / "home"
    home.mkdir()
    store = ExperimentStore(home / "experiments.sqlite3")
    create_skill_candidate(
        repository, store, experiment_id="exp-obs", candidate_id="cand-obs",
        candidate_type=CandidateType.SKILL_PATCH, target="brain/skills/demo",
        content="---\nname: demo\ndescription: improved\n---\n\nImproved demo skill.\n",
        hypothesis="improve the demo skill", benchmark_version=version,
        policy_version=default_config().promotion.policy_version,
    )
    key_file = tmp_path / "evaluator.pem"
    key_file.write_bytes(Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    monkeypatch.setattr(cli, "build_runner", lambda args, artifact_store: _Runner(baseline))
    for name in ("HERMES_IMPROVE_CONFIG", "HERMES_PROMOTER_UID", "QWEN_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "no-layout.json"))
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public_root))
    monkeypatch.setenv("HERMES_EVALUATOR_PRIVATE_KEY", str(key_file))
    monkeypatch.setenv("HERMES_EVALUATOR_UID", str(os.geteuid()))

    def evaluate(suite, *extra):
        monkeypatch.setattr(sys, "argv", [
            "hermes-improve", "evaluate", "cand-obs", "--repository", str(repository),
            "--holdout-root", str(holdout_root), "--suite", suite, *extra,
        ])
        assert cli.main() == 0
        return json.loads(capsys.readouterr().out)

    evaluate("SMOKE")
    evaluate("CORE")  # the sealed holdout in FULL is reserved for finalists

    def verdict(body):
        payload = json.loads(body["messages"][1]["content"])
        winner = "A" if payload["response_A"] == "candidate answer" else "B"
        return json.dumps({"winner": winner, "confidence": 0.8, "uncertainty": 0.2, "rationale": "r"})

    with OpenAIStub(verdict) as stub, MetricsServer(step=0.5) as metrics:
        monkeypatch.setenv("QWEN_BASE_URL", stub.base_url)
        monkeypatch.setenv("QWEN_MODEL", "judge-model")
        monkeypatch.setenv("HERMES_GPU_METRICS_URL", metrics.url)
        monkeypatch.setenv("HERMES_GPU_RUNTIME_METRICS", _METRICS)
        evaluate("FULL", "--judge")
        judge_requests = len(stub.requests)
    assert judge_requests >= 2, "the subjective FULL task is judged (A/B and swapped)"

    recorder = DecisionRecorder(home / "experiments.sqlite3")
    selections = {
        item.input_features["suite"]: item for item in recorder.iter_records(decision_type="suite_selection")
    }
    assert set(selections) == {"SMOKE", "CORE", "FULL"}
    smoke = selections["SMOKE"]
    assert smoke.selected_option == "impact_narrowed" and smoke.available_options == ("impact_narrowed", "unnarrowed")
    assert smoke.input_features["applied_impact"]["target_areas"] == ["skills/demo"]
    assert smoke.input_features["selected_tasks"] == 1 and smoke.input_features["holdout_tasks"] == 0
    assert smoke.source == "suite-selection-rules" and smoke.confidence is None
    full = selections["FULL"]
    assert full.selected_option == "unnarrowed"
    assert full.input_features["selected_tasks"] == 2 and full.input_features["holdout_tasks"] == 1
    runs = store.candidate_evaluations("cand-obs")
    stored_digests = {json.loads(run["evaluation"]["metrics_json"])["suite_selection"]["suite_manifest_digest"]
                      for run in runs}
    assert {item.input_features["suite_manifest_digest"] for item in selections.values()} == stored_digests

    routes = list(recorder.iter_records(decision_type="needs_qwen"))
    by_task = {item.input_features.get("task_id", "<holdout>"): item for item in routes}
    assert by_task["essay-task"].selected_option == "qwen_judge"
    assert by_task["essay-task"].input_features["judge_configured"] is True
    assert by_task["full-task"].selected_option == "deterministic_verification"
    assert by_task["smoke-task"].input_features["judge_configured"] is False
    holdout = by_task["<holdout>"]
    assert holdout.input_features["split"] == "SEALED_HOLDOUT" and "holdout-task" not in json.dumps(holdout.to_dict())
    for item in routes:
        assert item.available_options == ("deterministic_verification", "qwen_judge", "exit_code_only")
        assert item.source == "evaluator-verification-rules" and item.confidence is None
    # The recorded route is the one the evaluator applied.
    essay = BenchmarkStore(public_root).get("essay-task", role=cli.Role.EVALUATOR)
    assert judge_route(essay) == "qwen_judge"

    qwen = build_report(home, public_root)["qwen"]
    assert qwen["by_role"]["qwen_judge"]["requests"] == judge_requests
    assert qwen["judge_uncertainty"] == {"judged_tasks": 1, "mean": 0.2, "max": 0.2, "unreliable_tasks": 0}
    assert qwen["gpu_runtime"]["measured_commands"] == 1
    assert qwen["gpu_runtime"]["seconds_by_command"] == {"evaluate-judge": 1.5}
