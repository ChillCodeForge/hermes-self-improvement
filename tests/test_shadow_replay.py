from __future__ import annotations

import json
import sqlite3
import stat
import sys

import pytest

from hermes_improve.cli import main
from hermes_improve.ingestion import SessionQueue
from hermes_improve.monitoring import MonitoringPolicy, ReleaseMonitor
from hermes_improve.shadow_replay import CommandReplayRunner, ReplayOutcome, ShadowReplayer
from hermes_improve.store import ExperimentStore


def _checkout(tmp_path):
    checkout = tmp_path / "releases" / "brain-v1"
    (checkout / "brain" / "skills" / "demo").mkdir(parents=True)
    (checkout / "brain" / "skills" / "demo" / "SKILL.md").write_text("demo\n", encoding="utf-8")
    checkout.chmod(0o555)
    return checkout


def _setup(tmp_path, *, checkout=None):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="l", hypothesis="h", candidate_type="SKILL_PATCH", target="brain/skills/demo",
        risk="LOW", baseline_commit="base", benchmark_version="v1", policy_version="1",
    )
    store.record_release("brain-v1", "exp-1", "hash", "PROVISIONAL", {
        "baseline_metrics": {"success_rate": 0.9, "tool_error_rate": 0.1, "tokens_per_success": 100,
                             "latency_per_success_ms": 100, "correction_rate": 0.1},
        "changed_paths": ["brain/skills/demo/SKILL.md"],
        "deployment_path": str(checkout or _checkout(tmp_path)),
    })
    queue = SessionQueue(tmp_path / "sessions.sqlite3")
    queue.ingest_many([
        {"session_id": "s-demo", "task": {"prompt": "Use the demo skill"},
         "tool_calls": [{"name": "skill_view"}], "messages": [{"role": "assistant", "content": "loading demo"}]},
        {"session_id": "s-other", "task": {"prompt": "Unrelated work"}, "messages": [{"role": "user", "content": "demonstrate"}]},
        {"session_id": "s-empty", "task": {"prompt": ""}, "messages": [{"role": "user", "content": "demo"}]},
    ])
    queue.set_status("s-demo", "processed")
    return store, queue


class FakeRunner:
    def __init__(self, outcome=None, error=None):
        self.calls = []
        self.outcome = outcome or ReplayOutcome(True, 0, 80, 2, 50.0)
        self.error = error

    def run(self, task, release_checkout):
        self.calls.append((task, release_checkout))
        if self.error:
            raise self.error
        return self.outcome


def test_shadow_replay_selects_matching_sessions_and_records_evidence(tmp_path):
    store, queue = _setup(tmp_path)
    queue_before = sqlite3.connect(tmp_path / "sessions.sqlite3").execute(
        "SELECT session_id,status,content_hash FROM sessions ORDER BY session_id").fetchall()
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=1))
    runner = FakeRunner()

    report = ShadowReplayer(queue, monitor, runner).replay("brain-v1")

    assert [call[0].session_id for call in runner.calls] == ["s-demo"]
    assert runner.calls[0][0].evidence_trust == "untrusted"
    assert runner.calls[0][0].skills == ("demo",)
    assert report.recorded == 1 and report.last_status == "STABLE"
    assert store.get_release("brain-v1")["status"] == "STABLE"
    with sqlite3.connect(tmp_path / "monitor.sqlite3") as db:
        assert db.execute("SELECT source, skills_json FROM production_observations").fetchall() == [
            ("shadow_replay", '["demo"]'),
        ]
    queue_after = sqlite3.connect(tmp_path / "sessions.sqlite3").execute(
        "SELECT session_id,status,content_hash FROM sessions ORDER BY session_id").fetchall()
    assert queue_after == queue_before


def test_shadow_replay_is_idempotent_and_ignores_runner_failures_as_evidence(tmp_path):
    store, queue = _setup(tmp_path)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=5))
    failing = ShadowReplayer(queue, monitor, FakeRunner(error=RuntimeError("runner crashed")))
    report = failing.replay("brain-v1")
    assert report.recorded == 0 and "runner crashed" in report.failures[0]["error"]
    replayer = ShadowReplayer(queue, monitor, FakeRunner())
    assert replayer.replay("brain-v1").recorded == 1
    assert replayer.replay("brain-v1").duplicates == 1


def test_shadow_replay_requires_a_read_only_checkout(tmp_path):
    writable = tmp_path / "writable"
    writable.mkdir()
    writable.chmod(0o755)
    store, queue = _setup(tmp_path, checkout=writable)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store)
    with pytest.raises(PermissionError, match="read-only"):
        ShadowReplayer(queue, monitor, FakeRunner()).replay("brain-v1")


def test_command_runner_contract_and_cli(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    _setup(home)
    script = tmp_path / "replay.py"
    script.write_text(
        "import json, os, sys\n"
        "request = json.load(sys.stdin)\n"
        "assert request['evidence_trust'] == 'untrusted'\n"
        "assert os.environ['HERMES_BRAIN_CHECKOUT'] == request['release_checkout']\n"
        "assert 'HERMES_IMPROVE_HOME' not in os.environ\n"
        "print(json.dumps({'success': True, 'tool_errors': 0, 'tokens': 70, 'tool_calls': 2, 'latency_ms': 12.5}))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "monitor", "replay", "brain-v1",
        "--runner-command", json.dumps([sys.executable, str(script)]),
    ])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["recorded"] == 1 and report["failures"] == []

    with pytest.raises(ValueError, match="absolute path"):
        CommandReplayRunner(["python"])
    bad = tmp_path / "bad.py"
    bad.write_text("print('{\"success\": true, \"extra\": 1}')\n", encoding="utf-8")
    runner = CommandReplayRunner([sys.executable, str(bad)])
    task = ShadowReplayer(SessionQueue(home / "sessions.sqlite3"), ReleaseMonitor(
        home / "monitoring.sqlite3", ExperimentStore(home / "experiments.sqlite3")), runner).select_tasks("brain-v1")[0]
    with pytest.raises(ValueError, match="missing or unknown fields"):
        runner.run(task, tmp_path)
