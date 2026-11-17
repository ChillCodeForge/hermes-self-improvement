from __future__ import annotations

import json
import sys

import pytest

from hermes_improve.cli import main
from hermes_improve.decisions import DecisionRecorder, record_decision
from hermes_improve.ingestion import SessionQueue
from hermes_improve.monitoring import MonitoringPolicy, ProductionObservation, ReleaseMonitor
from hermes_improve.observability import ObservabilityLog, build_report, render_text
from hermes_improve.store import ExperimentStore


def _decision(**overrides):
    values = dict(
        decision_type="benchmark_eligibility",
        input_features={"reproducibility_score": 0.8, "has_tests": True},
        available_options=("HARD_BENCHMARK", "SOFT_BENCHMARK", "DREAMING_ONLY", "EXCLUDED"),
        selected_option="SOFT_BENCHMARK",
        confidence=0.7,
        source="miner-rules",
        model_version="miner-1",
    )
    values.update(overrides)
    return values


def test_decision_recording_is_idempotent_and_validated(tmp_path):
    database = tmp_path / "experiments.sqlite3"
    first = record_decision(database, **_decision())
    again = record_decision(database, **_decision())
    assert first.id == again.id and len(first.id) == 32
    assert DecisionRecorder(database).counts() == {"benchmark_eligibility": {"total": 1, "labeled": 0, "correct": 0}}
    recorder = DecisionRecorder(database)
    with pytest.raises(ValueError, match="different confidence"):
        recorder.record(**_decision(confidence=0.9))
    with pytest.raises(ValueError, match="decision_type"):
        recorder.record(**_decision(decision_type="promote"))
    with pytest.raises(ValueError, match="selected_option"):
        recorder.record(**_decision(selected_option="MAYBE"))
    with pytest.raises(ValueError, match="confidence"):
        recorder.record(**_decision(confidence=float("nan")))
    with pytest.raises(ValueError, match="finite JSON"):
        recorder.record(**_decision(input_features={"score": float("inf")}))
    with pytest.raises(ValueError, match="credential"):
        recorder.record(**_decision(input_features={"note": "Bearer abcdefghijklmnop"}))
    with pytest.raises(ValueError, match="duplicates"):
        recorder.record(**_decision(available_options=("A", "A")))


def test_ground_truth_backfill_is_immutable_and_scores_correctness(tmp_path):
    recorder = DecisionRecorder(tmp_path / "experiments.sqlite3")
    record = recorder.record(**_decision())
    labeled = recorder.backfill_ground_truth(record.id, ground_truth="HARD_BENCHMARK", ground_truth_source="operator-review")
    assert labeled.correct is False and labeled.ground_truth == "HARD_BENCHMARK"
    assert recorder.backfill_ground_truth(record.id, ground_truth="HARD_BENCHMARK", ground_truth_source="operator-review") == labeled
    with pytest.raises(ValueError, match="immutable"):
        recorder.backfill_ground_truth(record.id, ground_truth="SOFT_BENCHMARK", ground_truth_source="operator-review")
    with pytest.raises(ValueError, match="available_options"):
        recorder.backfill_ground_truth(recorder.record(**_decision(selected_option="EXCLUDED")).id,
                                       ground_truth="UNKNOWN", ground_truth_source="operator")
    with pytest.raises(KeyError):
        recorder.backfill_ground_truth("0" * 32, ground_truth="EXCLUDED", ground_truth_source="operator")


def test_cli_decisions_backfill_and_export_jsonl(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    recorder = DecisionRecorder(home / "experiments.sqlite3")
    record = recorder.record(**_decision())
    recorder.record(**_decision(decision_type="risk", available_options=("LOW", "HIGH"), selected_option="LOW"))

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "decisions", "backfill", record.id,
                                      "--ground-truth", "SOFT_BENCHMARK", "--source", "operator-review"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["correct"] is True

    output = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "decisions", "export", "--labeled-only", "--output", str(output)])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["exported"] == 1
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert set(rows[0]) == {
        "id", "decision_type", "input_features", "available_options", "selected_option", "confidence",
        "source", "ground_truth", "ground_truth_source", "correct", "model_version", "timestamp",
    }
    with pytest.raises(FileExistsError):
        main()

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "decisions", "export", "--type", "risk"])
    assert main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["decision_type"] for line in lines] == ["risk"]


def test_report_without_state_says_no_data_and_creates_nothing(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "no-benchmarks"))
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "report", "--json"])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"generated_at", "hermes", "improvement", "benchmarks", "dreaming", "qwen", "production"}
    for section in ("hermes", "improvement", "benchmarks", "dreaming", "production"):
        assert set(report[section].values()) == {"no data"}, section
    assert report["qwen"]["gpu_runtime"] == "unavailable"
    assert not home.exists()

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "report"])
    assert main() == 0
    text = capsys.readouterr().out
    assert "Production" in text and "gpu_runtime: unavailable" in text


def test_report_computes_sections_from_persisted_state(tmp_path):
    home = tmp_path / "home"
    store = ExperimentStore(home / "experiments.sqlite3")
    store.create_experiment(
        "exp-1", lineage_id="l", hypothesis="h", candidate_type="SKILL_PATCH", target="brain/skills/demo",
        risk="LOW", baseline_commit="base", benchmark_version="v1", policy_version="1",
    )
    store.create_experiment(
        "exp-2", lineage_id="l", hypothesis="h2", candidate_type="SKILL_PATCH", target="brain/skills/demo",
        risk="LOW", baseline_commit="base", benchmark_version="v1", policy_version="1", parent_id="exp-1",
    )
    store.record_release("brain-v1", "exp-1", "hash", "PROVISIONAL", {"baseline_metrics": {"success_rate": 0.5}})
    monitor = ReleaseMonitor(home / "monitoring.sqlite3", store, MonitoringPolicy(min_observations=50))
    monitor.observe(ProductionObservation("brain-v1", "t1", True, 1, 100, 4, 10.0, retries=1))
    monitor.observe(ProductionObservation("brain-v1", "t2", False, 0, 50, 2, 10.0, critical_regression=True))
    SessionQueue(home / "sessions.sqlite3").ingest({"session_id": "s1", "task": {"prompt": "p"}})
    log = ObservabilityLog(home / "observability.sqlite3")
    log.record_dreaming_run(stage="light", sessions_processed=3, clusters={"errors": 2}, signals=1, hypotheses=0)
    log.record_dreaming_run(stage="rem", sessions_processed=3, clusters={"errors": 1}, signals=1, hypotheses=4)
    log.record_proposal_conversion("cand-1")
    log.record_qwen_stats({"qwen_dreamer": {"requests": 2, "input_tokens": 30, "output_tokens": 10, "latency_ms": 400.0}})

    report = build_report(home, tmp_path.parent / "absent-benchmarks")

    assert report["hermes"]["success_rate"] == 0.5
    assert report["hermes"]["tokens_per_success"] == 150
    assert report["hermes"]["tool_calls_per_success"] == 6
    assert report["hermes"]["failed_tool_calls"] == 1
    assert report["hermes"]["retry_rate"] == 0.5
    assert report["hermes"]["queued_sessions"] == {"queued": 1}
    assert report["improvement"]["generations"] == 2
    assert report["improvement"]["candidates_promoted"] == 1
    assert report["dreaming"]["sessions_processed"] == 6
    assert report["dreaming"]["clusters"] == {"errors": 3}
    assert report["dreaming"]["candidate_conversion_rate"] == 0.25
    assert report["qwen"]["requests"] == 2 and report["qwen"]["average_latency_ms"] == 200
    assert report["qwen"]["judge_uncertainty"] == "no data"
    assert report["production"]["brain_version"] == "brain-v1"
    assert report["production"]["post_promotion_regressions"]["critical_observations"] == 1
    assert report["production"]["rollback_count"] == 0
    assert "Benchmarks" in render_text(report)


def test_report_reads_public_benchmarks_only(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "benchmarks"
    report = build_report(tmp_path / "home", root)
    # The curated public corpus: 26 WP-B command tasks + 6 WP-G skill_procedure tasks, no holdout.
    assert report["benchmarks"]["count"] == 32
    assert report["benchmarks"]["split_distribution"] == {"TRAIN": 12, "VALIDATION": 10, "REGRESSION": 10}
    assert report["benchmarks"]["ground_truth_types"] == {"deterministic_tests": 32}
    assert report["benchmarks"]["reproducibility"] == {"mean": 1.0, "min": 1.0, "max": 1.0}
    assert report["benchmarks"]["spent_holdouts"] == "no data"
