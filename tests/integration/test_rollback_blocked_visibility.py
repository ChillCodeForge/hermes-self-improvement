"""A blocked monitoring rollback is visible outside monitoring.sqlite3 (lab finding F21)."""

from __future__ import annotations

import json
import sys

import pytest

from hermes_improve.cli import main
from hermes_improve.monitoring import MonitoringPolicy, ProductionObservation, ReleaseMonitor
from hermes_improve.observability import build_report, build_status
from hermes_improve.store import ExperimentStore

KEY = b"m" * 32


def _blocked_release(home):
    store = ExperimentStore(home / "experiments.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="lineage", hypothesis="monitor", candidate_type="SKILL_PATCH",
        target="brain/skills/demo", risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    store.record_release(
        "brain-v1", "exp-1", "hash", "PROVISIONAL",
        {"baseline_metrics": {"success_rate": 0.9}, "changed_paths": ["brain/skills/demo/SKILL.md"]},
    )
    monitor = ReleaseMonitor(home / "monitoring.sqlite3", store, MonitoringPolicy(min_observations=1))
    monitor.observe(ProductionObservation(
        "brain-v1", "task", False, 0, 100, 1, 100.0, critical_regression=True, observation_id="o1",
    ))
    with pytest.raises(RuntimeError, match="no earlier active release"):
        monitor.apply("brain-v1", promoter=object(), manifest_key=KEY, actor="uid:0")
    return store, monitor


def test_blocked_rollback_is_a_release_event(tmp_path):
    store, monitor = _blocked_release(tmp_path)
    events = [event for event in store.list_release_events("brain-v1") if event["event"] == "ROLLBACK_BLOCKED"]
    assert len(events) == 1
    event = events[0]
    assert event["from_status"] == event["to_status"] == "PROVISIONAL"
    action = monitor.actions("brain-v1")[0]
    assert event["detail"] == {
        "actor": "uid:0",
        "monitoring_action_id": action["id"],
        "reason": "no earlier active release to roll back to; operator action required",
        "assessment_reasons": ["critical regression observed"],
        "target_release": None,
    }
    # The release itself is unchanged and still provisional.
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"


def test_status_and_report_show_the_blocked_rollback_of_the_active_release(tmp_path):
    _blocked_release(tmp_path)
    status = build_status(tmp_path, tmp_path / "absent-benchmarks")
    blocked = status["active_release"]["rollback_blocked"]
    assert blocked["reason"] == "no earlier active release to roll back to; operator action required"
    assert blocked["assessment_reasons"] == ["critical regression observed"]
    assert status["counts"]["blocked_rollbacks"] == 1
    report = build_report(tmp_path, tmp_path / "absent-benchmarks")
    assert report["production"]["post_promotion_regressions"]["blocked_rollbacks"] == 1


def test_blocked_rollback_of_an_earlier_activation_is_not_reported_as_current(tmp_path):
    store, _ = _blocked_release(tmp_path)
    # brain-v1 is re-activated later (a rollback from another release): a new epoch.
    store.record_rollback("other-commit", "brain-v1", {"actor": "uid:0"})
    status = build_status(tmp_path, tmp_path / "absent-benchmarks")
    assert "rollback_blocked" not in status["active_release"]
    assert status["counts"]["blocked_rollbacks"] == 1


def test_releases_events_cli_lists_the_blocked_rollback(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    _blocked_release(home)
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "releases", "events", "--release", "brain-v1"])
    assert main() == 0
    events = json.loads(capsys.readouterr().out)
    assert [event["event"] for event in events] == ["ROLLBACK_BLOCKED"]
