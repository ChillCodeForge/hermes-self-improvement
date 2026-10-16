from __future__ import annotations

import json

import pytest

from hermes_improve.monitoring import (
    MonitoringPolicy,
    ProductionObservation,
    ReleaseMonitor,
)
from hermes_improve.optimizer import run_evolution
from hermes_improve.self_evolution import EvolutionLimits, SelfEvolutionAdapter
from hermes_improve.store import ExperimentStore


class Strategy:
    version = "1"

    def evolve(self, candidate, feedback, generation):
        return [f"{candidate}-{generation}-a", f"{candidate}-{generation}-b"]


def test_iterative_optimizer_selects_only_measured_improvement_and_stops():
    adapter = SelfEvolutionAdapter("1", Strategy())
    scores = {"base-0-a": 0.2, "base-0-b": 1.0}

    def evaluate(candidate):
        return scores.get(candidate, 1.0), "non-holdout feedback"

    result = run_evolution(
        adapter, "base", 0.0, evaluate=evaluate,
        limits=EvolutionLimits(
            max_generations=5, candidates_per_generation=2, max_candidates_total=6,
            no_improvement_candidates=4, no_improvement_generations=2,
            minimum_meaningful_delta=0.5,
        ),
    )
    assert result.best_candidate == "base-0-b"
    assert result.best_score == 1.0
    assert result.evaluated <= 6


def _add_provisional_release(store, candidate_hash="hash"):
    store.create_experiment(
        "exp-monitor", lineage_id="lineage", hypothesis="monitor",
        candidate_type="SKILL_PATCH", target="brain/skills/a",
        risk="LOW", baseline_commit="base", benchmark_version="v1",
        policy_version="1",
    )
    store.record_release(
        "brain-v1", "exp-monitor", candidate_hash, "PROVISIONAL",
        {
            "baseline_metrics": {
                "success_rate": 0.95,
                "tool_error_rate": 0.05,
                "tokens_per_success": 100,
                "latency_per_success_ms": 100,
                "correction_rate": 0.05,
            }
        },
    )


def test_monitor_marks_stable_after_minimum_healthy_observations(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    _add_provisional_release(store)
    monitor = ReleaseMonitor(
        tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2)
    )
    for task in ("t1", "t2"):
        decision = monitor.observe(ProductionObservation(
            "brain-v1", task, True, 0, 90, 4, 90,
        ))
    assert decision.status == "STABLE"
    assert store.get_release("brain-v1")["status"] == "STABLE"


def test_monitor_recommends_rollback_for_critical_or_metric_regression(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    _add_provisional_release(store)
    monitor = ReleaseMonitor(
        tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=2)
    )
    monitor.observe(ProductionObservation("brain-v1", "t1", False, 10, 300, 2, 300))
    decision = monitor.observe(ProductionObservation("brain-v1", "t2", False, 10, 300, 2, 300))
    assert decision.status == "ROLLBACK_RECOMMENDED"
    assert "success rate regression" in decision.reasons
    assert store.get_release("brain-v1")["status"] == "PROVISIONAL"


def test_critical_monitoring_regression_recommends_immediate_rollback(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    _add_provisional_release(store)
    monitor = ReleaseMonitor(
        tmp_path / "monitor.sqlite3", store, MonitoringPolicy(min_observations=100)
    )
    decision = monitor.observe(ProductionObservation(
        "brain-v1", "task", True, 0, 100, 1, 100, critical_regression=True,
    ))
    assert decision.status == "ROLLBACK_RECOMMENDED"
    assert decision.sample_count == 1


def test_monitoring_rejects_non_finite_policy_and_observation_metrics(tmp_path):
    store = ExperimentStore(tmp_path / "state.sqlite3")
    _add_provisional_release(store)
    with pytest.raises(ValueError, match="finite"):
        MonitoringPolicy(max_latency_increase=float("nan"))
    monitor = ReleaseMonitor(tmp_path / "monitor.sqlite3", store)
    observation = ProductionObservation(
        "brain-v1", "task", True, 0, 0, 0, float("nan"),
    )
    with pytest.raises(ValueError, match="finite"):
        monitor.observe(observation)
