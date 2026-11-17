from __future__ import annotations

import json
import sys

from hermes_improve.cli import main
from hermes_improve.ingestion import SessionQueue
from hermes_improve.miner import mine_session
from hermes_improve.store import ExperimentStore


def test_cli_inspects_redacted_evidence_and_records_non_admitting_review(
    tmp_path, monkeypatch, capsys,
):
    home = tmp_path / "state"
    export = {
        "session_id": "review-session",
        "task": {
            "title": "Build a package",
            "prompt": "Build the package and run its tests.",
            "input_available": True,
        },
        "environment": {"type": "container", "network_policy": "disabled"},
        "verification": {"deterministic": [{"type": "exit_code"}]},
        "security": {"sanitized": True, "contains_secrets": False},
        "replay_verified": True,
    }
    SessionQueue(home / "sessions.sqlite3").ingest(export)
    proposal = mine_session(export)
    store = ExperimentStore(home / "experiments.sqlite3")
    store.save_benchmark_candidate(proposal, "review-session")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "benchmarks"))

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "inspect-candidate", proposal.task_fingerprint,
    ])
    assert main() == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["proposal"]["review_status"] == "PENDING"
    assert inspected["source_previews"][0]["session_id"] == "review-session"
    task_preview = inspected["source_previews"][0]["task"]["json"]
    assert "Build the package and run its tests." in task_preview

    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "review-candidate", proposal.task_fingerprint,
        "--decision", "accept", "--reviewer", "operator",
        "--rationale", "Reproducible task with deterministic verification.",
    ])
    assert main() == 0
    review = json.loads(capsys.readouterr().out)
    assert review["review_status"] == "ACCEPTED"
    assert review["benchmark_admitted"] is False

