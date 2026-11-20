"""Monitoring reference metrics (lab finding F19).

A release is monitored against how the promoted brain behaved in its own
evaluation (the candidate arm), in per-run units that stay meaningful when an
arm never succeeded. Old manifests without that reference stay readable and
never compare against per-success "baselines" that are really totals.
"""

from __future__ import annotations

import pytest

from test_promotion import ALL_SUITES, _add_candidate, _record_run, _repository
from hermes_improve.models import Split, TaskOutcome
from hermes_improve.monitoring import (
    MONITORING_REFERENCE_VERSION,
    MonitoringPolicy,
    ProductionObservation,
    ReleaseMonitor,
    monitoring_reference,
)
from hermes_improve.policy import PromotionPolicy
from hermes_improve.promotion import GitPromoter
from hermes_improve.store import ExperimentStore

KEY = b"m" * 32
# Evaluation of a knowledge-adding skill: the baseline arm never succeeded,
# the candidate always did (3 tasks, per-run values are means of the runs).
NEVER_SUCCEEDED = {
    suite: {
        "baseline_success": False, "candidate_success": True,
        "baseline_tokens": 20_000, "candidate_tokens": 1_000,
        "baseline_latency_ms": 160_000.0, "candidate_latency_ms": 10_000.0,
        "baseline_tool_calls": 7, "candidate_tool_calls": 4,
    }
    for suite in ALL_SUITES
}


def _outcome(**kwargs):
    values = dict(
        task_id="t", split=Split.REGRESSION, suite="FULL", baseline_success=False, candidate_success=True,
        baseline_quality=0, candidate_quality=10, baseline_tokens=20_000, candidate_tokens=1_000,
        token_usage_reported=True, categories=("skills",), baseline_latency_ms=160_000.0,
        candidate_latency_ms=10_000.0, baseline_tool_calls=7, candidate_tool_calls=4,
        baseline_retries=1, candidate_retries=0,
    )
    values.update(kwargs)
    return TaskOutcome(**values)


def test_policy_never_reports_totals_as_per_success_and_records_per_run_metrics():
    rows = [
        _outcome(task_id="a"), _outcome(task_id="b"),
        _outcome(task_id="h", split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT"),
    ]
    metrics = PromotionPolicy().assess("c1", "hash", "v1", rows).metrics
    # 0 baseline successes: there is no per-success value, not the 60,000-token sum.
    assert metrics["baseline_tokens_per_success"] is None
    assert metrics["baseline_latency_per_success_ms"] is None
    assert metrics["candidate_tokens_per_success"] == 1_000
    assert metrics["tokens_per_task_candidate"] == 1_000
    assert metrics["latency_per_task_candidate_ms"] == 10_000
    assert metrics["tool_calls_per_task_candidate"] == 4
    assert metrics["tool_calls_per_task_baseline"] == 7
    assert metrics["candidate_retry_rate"] == 0
    assert metrics["baseline_retry_rate"] == 1


def _promote_never_succeeded(tmp_path):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(tmp_path / "db.sqlite3")
    candidate, private, public = _add_candidate(
        repository, store, "exp-1", "cand-1", baseline, "knowledge\n", evaluate=False,
    )
    _record_run(store, candidate, private, "eval-cand-1", ALL_SUITES, **NEVER_SUCCEEDED)
    promoter = GitPromoter(store, repository, tmp_path / "releases", tmp_path / "current", PromotionPolicy(), public)
    return store, promoter.promote("cand-1", release_name="brain-v1", manifest_key=KEY)


def test_promotion_manifest_records_versioned_candidate_reference_in_per_run_units(tmp_path):
    store, manifest = _promote_never_succeeded(tmp_path)
    reference = manifest["monitoring_reference"]
    assert reference["version"] == MONITORING_REFERENCE_VERSION
    assert reference["arm"] == "candidate"
    assert reference["unit"] == "per_run"
    assert reference["metrics"] == {
        "success_rate": 1.0, "tool_error_rate": 0.0, "tokens_per_run": 1_000.0,
        "tool_calls_per_run": 4.0, "latency_per_run_ms": 10_000.0, "retry_rate": 0.0,
        "correction_rate": 0.0,
    }
    # The legacy field stays for old readers, without totals posing as per-success values.
    assert manifest["baseline_metrics"]["success_rate"] == 0.0
    assert manifest["baseline_metrics"]["tokens_per_success"] is None
    assert manifest["baseline_metrics"]["latency_per_success_ms"] is None


def test_release_regressing_against_its_own_evaluation_is_flagged_although_baseline_never_succeeded(tmp_path):
    store, _ = _promote_never_succeeded(tmp_path)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=4))
    for index in range(4):
        monitor.observe(ProductionObservation(
            "brain-v1", f"t{index}", index == 0, 0, 3_000, 4, 9_000.0, observation_id=f"o{index}",
        ))
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "ROLLBACK_RECOMMENDED"
    assert "success rate regression" in assessment.reasons
    assert "tokens per run regression" in assessment.reasons
    assert assessment.reference == {
        "source": "monitoring_reference", "version": MONITORING_REFERENCE_VERSION,
        "arm": "candidate", "unit": "per_run", "excluded": [],
    }
    assert assessment.metrics.tokens_per_run == 3_000
    assert assessment.metrics.latency_per_run_ms == 9_000


def test_release_matching_its_evaluation_becomes_stable(tmp_path):
    store, _ = _promote_never_succeeded(tmp_path)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=3))
    for index in range(3):
        decision = monitor.observe(ProductionObservation(
            "brain-v1", f"t{index}", True, 0, 1_050, 4, 9_500.0, observation_id=f"o{index}",
        ))
    assert decision.status == "STABLE"


def _legacy_release(store, baseline):
    store.create_experiment(
        "exp-legacy", lineage_id="lineage", hypothesis="monitor", candidate_type="SKILL_PATCH",
        target="brain/skills/demo", risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    store.record_release("brain-v1", "exp-legacy", "hash", "PROVISIONAL", {"baseline_metrics": baseline})


def test_legacy_manifest_per_success_totals_of_a_never_successful_baseline_are_not_compared(tmp_path):
    # The live brain-v2 manifest: success_rate 0.0 and per-success values that are sums.
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _legacy_release(store, {
        "success_rate": 0.0, "tool_error_rate": 0.0, "tokens_per_success": 59374.0,
        "latency_per_success_ms": 495434.8, "correction_rate": 0.0,
    })
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2))
    for index in range(2):
        monitor.observe(ProductionObservation(
            "brain-v1", f"t{index}", True, 0, 100, 1, 100.0, observation_id=f"o{index}",
        ))
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "STABLE"
    assert assessment.reference["source"] == "baseline_metrics"
    assert assessment.reference["unit"] == "per_success"
    assert assessment.reference["excluded"] == ["tokens_per_success", "latency_per_success_ms"]
    assert "tokens_per_success" in assessment.unavailable_baselines
    assert "latency_per_success_ms" in assessment.unavailable_baselines


def test_legacy_manifest_with_successful_baseline_keeps_per_success_comparison(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _legacy_release(store, {"success_rate": 0.5, "tokens_per_success": 100})
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2))
    for index in range(2):
        monitor.observe(ProductionObservation(
            "brain-v1", f"t{index}", True, 0, 500, 1, 100.0, observation_id=f"o{index}",
        ))
    assessment = monitor.evaluate("brain-v1")
    assert "tokens per success regression" in assessment.reasons
    assert assessment.reference["excluded"] == []


@pytest.mark.parametrize("reference", [
    {"version": 99, "arm": "candidate", "unit": "per_run", "metrics": {}},
    {"version": 1, "arm": "candidate", "unit": "per_fortnight", "metrics": {}},
    {"version": 1, "arm": "candidate", "unit": "per_run", "metrics": []},
    "not-an-object",
])
def test_unknown_or_malformed_reference_is_refused_not_guessed(reference):
    with pytest.raises(ValueError, match="monitoring_reference"):
        monitoring_reference({"monitoring_reference": reference, "baseline_metrics": {"success_rate": 1.0}})
