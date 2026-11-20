"""Latency is advisory in post-promotion monitoring (lab finding F20).

The reference latency was measured in the evaluation environment; production
or replay hosts under load are slower without the brain being worse. A slower
but otherwise healthy release must not be rolled back for latency alone.
"""

from __future__ import annotations

import pytest

from hermes_improve.monitoring import MonitoringPolicy, ProductionObservation, ReleaseMonitor
from hermes_improve.store import ExperimentStore

REFERENCE = {
    "version": 1, "arm": "candidate", "unit": "per_run",
    "metrics": {
        "success_rate": 1.0, "tool_error_rate": 0.0, "tokens_per_run": 1_000.0, "tool_calls_per_run": 4.0,
        "latency_per_run_ms": 170_000.0, "retry_rate": 0.0, "correction_rate": 0.0,
    },
}


def _release(store):
    store.create_experiment(
        "exp-1", lineage_id="lineage", hypothesis="monitor", candidate_type="SKILL_PATCH",
        target="brain/skills/demo", risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    store.record_release(
        "brain-v1", "exp-1", "hash", "PROVISIONAL",
        {"baseline_metrics": {"success_rate": 0.0}, "monitoring_reference": REFERENCE},
    )


def _observe(monitor, count, *, latency_ms, success=True):
    for index in range(count):
        monitor.observe(ProductionObservation(
            "brain-v1", f"t{index}", success, 0, 1_000, 4, latency_ms, observation_id=f"o{index}",
        ))


def test_healthy_but_slow_release_is_stable_with_a_latency_warning(tmp_path):
    # Live lab: healthy replays under GPU load took 392 s, 2.3x the evaluation latency.
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=20))
    _observe(monitor, 20, latency_ms=392_000.0)
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "STABLE"
    assert assessment.reasons == ()
    assert assessment.warnings == ("latency regression",)
    assert assessment.to_dict()["warnings"] == ["latency regression"]
    assert store.get_release("brain-v1")["status"] == "STABLE"


def test_latency_warning_does_not_mask_correctness_regressions(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=3))
    _observe(monitor, 3, latency_ms=392_000.0, success=False)
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "ROLLBACK_RECOMMENDED"
    assert assessment.severity == "high"
    assert assessment.reasons == ("success rate regression",)
    assert assessment.warnings == ("latency regression",)


def test_latency_can_be_made_a_rollback_signal_with_an_explicit_tolerance(tmp_path):
    store = ExperimentStore(tmp_path / "db.sqlite3")
    _release(store)
    policy = MonitoringPolicy(min_observations=3, latency_action="rollback", max_latency_increase=1.5)
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store, policy)
    _observe(monitor, 3, latency_ms=392_000.0)  # 2.3x: within +150 % tolerance
    assert monitor.evaluate("brain-v1").status == "STABLE"
    monitor.policy = MonitoringPolicy(min_observations=3, latency_action="rollback", max_latency_increase=0.15)
    assessment = monitor.evaluate("brain-v1")
    assert assessment.status == "ROLLBACK_RECOMMENDED"
    assert assessment.severity == "medium"
    assert assessment.reasons == ("latency regression",)


def test_latency_action_is_validated():
    with pytest.raises(ValueError, match="latency_action"):
        MonitoringPolicy(latency_action="ignore")
