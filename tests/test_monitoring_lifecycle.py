from __future__ import annotations

import json
import os
import sys

import pytest

from test_promotion import _add_candidate, _repository, git
from hermes_improve.cli import main
from hermes_improve.decisions import DecisionRecorder
from hermes_improve.monitoring import MonitoringPolicy, ProductionObservation, ReleaseMonitor
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.store import ExperimentStore

BASELINE = {
    "success_rate": 0.9,
    "tool_error_rate": 0.05,
    "tokens_per_success": 100,
    "latency_per_success_ms": 100,
    "correction_rate": 0.05,
}
KEY = b"m" * 32


def _release(store, release_id="brain-v1", *, status="PROVISIONAL", baseline=BASELINE, **manifest):
    experiment = f"exp-{release_id}"
    store.create_experiment(
        experiment, lineage_id="lineage", hypothesis="monitor", candidate_type="SKILL_PATCH",
        target="brain/skills/demo", risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    store.record_release(
        release_id, experiment, "hash", status,
        {"baseline_metrics": baseline, "changed_paths": ["brain/skills/demo/SKILL.md"], **manifest},
    )


def _healthy(release_id, task, **overrides):
    values = dict(release_id=release_id, task_id=task, success=True, tool_errors=0, tokens=90,
                  tool_calls=3, latency_ms=90.0, observation_id=f"{release_id}-{task}")
    values.update(overrides)
    return ProductionObservation(**values)


def test_release_status_transitions_are_validated(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    with pytest.raises(ValueError, match="PROVISIONAL -> CANDIDATE is not allowed"):
        store.set_release_status("brain-v1", "CANDIDATE")
    assert store.set_release_status("brain-v1", "STABLE") == "PROVISIONAL"
    assert store.set_release_status("brain-v1", "STABLE") == "STABLE"
    store.set_release_status("brain-v1", "ROLLED_BACK")
    with pytest.raises(ValueError, match="ROLLED_BACK -> STABLE is not allowed"):
        store.set_release_status("brain-v1", "STABLE")
    with pytest.raises(ValueError, match="invalid release status"):
        store.set_release_status("brain-v1", "LIVE")
    with pytest.raises(KeyError):
        store.set_release_status("brain-v9", "STABLE")


def test_observations_for_unrelated_skills_do_not_count(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2))
    for task in ("a", "b", "c"):
        monitor.observe(_healthy("brain-v1", task, success=False, skills=("unrelated",)))
    assessment = monitor.evaluate("brain-v1")
    assert assessment.matched_observations == 0
    assert assessment.unmatched_observations == 3
    assert assessment.status == "PROVISIONAL"
    decision = monitor.observe(_healthy("brain-v1", "d", skills=("demo", "unrelated")))
    assert decision.sample_count == 1
    decision = monitor.observe(_healthy("brain-v1", "e"))
    assert decision.status == "STABLE"
    assert store.get_release("brain-v1")["status"] == "STABLE"


def test_monitor_flags_every_spec_metric_with_a_baseline(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store, baseline=BASELINE | {"tool_calls_per_success": 3, "retry_rate": 0.0})
    monitor = ReleaseMonitor(
        tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2),
        decisions=DecisionRecorder(tmp_path / "db.sqlite3"),
    )
    for task in ("a", "b"):
        monitor.observe(_healthy("brain-v1", task, tool_calls=9, retries=1, tokens=95, latency_ms=95.0))
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "ROLLBACK_RECOMMENDED"
    assert assessment.severity == "medium"
    assert set(assessment.reasons) == {"tool calls per success regression", "retry rate regression"}
    assert assessment.metrics.tool_calls_per_success == 9
    assert assessment.metrics.retry_rate == 1.0
    assert assessment.unavailable_baselines == ()
    record = DecisionRecorder(tmp_path / "db.sqlite3").get(assessment.decision_id)
    assert record.decision_type == "regression_severity"
    assert record.selected_option == "medium"
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"


def test_missing_baselines_are_reported_not_invented(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store, baseline={"success_rate": 0.9})
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=1))
    assessment = monitor.evaluate("brain-v1")
    assert assessment.metrics is None
    monitor_rows = ReleaseMonitor(tmp_path / "monitor2.sqlite3", store, MonitoringPolicy(min_observations=5))
    monitor_rows.observe(_healthy("brain-v1", "a"))
    assessment = monitor_rows.evaluate("brain-v1")
    assert assessment.status == "PROVISIONAL"
    monitor_rows.policy = MonitoringPolicy(min_observations=1)
    assessment = monitor_rows.evaluate("brain-v1")
    assert assessment.status == "STABLE"
    assert "tokens_per_success" in assessment.unavailable_baselines


def test_apply_without_previous_release_blocks_and_audits(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=1))
    monitor.observe(_healthy("brain-v1", "a", critical_regression=True))
    with pytest.raises(RuntimeError, match="no earlier active release"):
        monitor.apply("brain-v1", promoter=object(), manifest_key=KEY, actor="operator")
    assert [row["action"] for row in monitor.actions("brain-v1")] == ["ROLLBACK_BLOCKED"]
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"


def test_apply_marks_stable_with_audit(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2))
    monitor.observe(_healthy("brain-v1", "a"))
    assert monitor.apply("brain-v1", promoter=None, manifest_key=None, actor="operator")["action"] == "NONE"
    monitor.policy = MonitoringPolicy(min_observations=1)
    result = monitor.apply("brain-v1", promoter=None, manifest_key=None, actor="operator")
    assert result["action"] == "STABLE"
    assert store.get_release("brain-v1")["status"] == "STABLE"
    assert monitor.actions("brain-v1")[0]["actor"] == "operator"


def test_regression_executes_rollback_through_git_promoter(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "version one\n")
    current = tmp_path / "brain-current"
    promoter = GitPromoter(store, repository, tmp_path / "releases", current, PromotionPolicy(), public)
    promoter.promote("cand-1", release_name="brain-v1", manifest_key=KEY)
    second, _, _ = _add_candidate(
        repository, store, "exp-2", "cand-2", first.commit, "version two\n", signing_keys=(private, public),
    )
    promoter.promote("cand-2", release_name="brain-v2", manifest_key=KEY)
    monitor = ReleaseMonitor(
        tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2),
        decisions=DecisionRecorder(tmp_path / "db.sqlite3"),
    )
    assert monitor.previous_release("brain-v2") == "brain-v1"
    for task in ("a", "b"):
        monitor.observe(_healthy("brain-v2", task, success=False, tool_errors=3, skills=("demo",)))

    result = monitor.apply("brain-v2", promoter=promoter, manifest_key=KEY, actor="uid:1001")

    assert result["action"] == "ROLLBACK"
    assert result["active_release"] == "brain-v1"
    assert result["previous_commit"] == second.commit
    assert git(repository, "rev-parse", "main") == first.commit
    assert (current / "brain/skills/demo/SKILL.md").read_text() == "version one\n"
    assert store.get_release("brain-v2")["status"] == "ROLLED_BACK"
    audit = monitor.actions("brain-v2")
    assert [row["action"] for row in audit] == ["ROLLBACK"]
    assert audit[0]["target_release"] == "brain-v1"
    assert "success rate regression" in audit[0]["reasons"]


def test_cli_monitor_evaluate_reports_and_apply_requires_promoter_identity(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    store = ExperimentStore(home / "experiments.sqlite3")
    _release(store)
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "monitor", "evaluate", "brain-v1"])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PROVISIONAL"
    assert report["affected_skills"] == ["demo"]

    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "monitor", "evaluate", "brain-v1", "--apply"])
    with pytest.raises(RuntimeError, match="HERMES_PROMOTER_UID"):
        main()
    monkeypatch.setenv("HERMES_PROMOTER_UID", str(os.geteuid()))
    monkeypatch.setenv("HERMES_IMPROVE_MANIFEST_KEY", "k" * 32)
    with pytest.raises(ValueError, match="--repository"):
        main()
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"


def test_cli_monitor_ingest_accepts_skill_attribution_and_rejects_bad_sources(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    _release(ExperimentStore(home / "experiments.sqlite3"))
    row = {
        "observation_id": "evt-1", "release_id": "brain-v1", "task_id": "t", "success": True,
        "tool_errors": 0, "tokens": 10, "tool_calls": 1, "latency_ms": 5, "skills": ["demo"],
        "source": "historical_replay",
    }
    path = tmp_path / "obs.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "monitor", "ingest", str(path)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["observations"][0]["sample_count"] == 1
    path.write_text(json.dumps(row | {"observation_id": "evt-2", "source": "guess"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: observation source"):
        main()
