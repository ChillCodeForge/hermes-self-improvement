from __future__ import annotations

import json
import sys

import pytest

from hermes_improve.batch_ingestion import load_batch_runner_export
from hermes_improve.cli import main
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.ingestion import SessionQueue
from hermes_improve.miner import mine_session
from hermes_improve.models import Eligibility


def trajectory_record(prompt_index=0, *, partial=False):
    return {
        "prompt_index": prompt_index,
        "conversations": [
            {"from": "system", "value": "You are an agent."},
            {"from": "human", "value": "Summarize the fixture."},
            {"from": "gpt", "value": "Bearer abcdefghijk123456789"},
        ],
        "metadata": {"batch_num": 2, "timestamp": "2025-01-01T00:00:00", "model": "test-model"},
        "completed": True,
        "partial": partial,
    }


def test_batch_runner_export_is_imported_redacted_and_untrusted(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(trajectory_record()) + "\n", encoding="utf-8")
    exports = load_batch_runner_export(source)
    queue = SessionQueue(tmp_path / "sessions.sqlite3")
    queue.ingest_many(exports)

    session_id = exports[0]["session_id"]
    session = queue.get(session_id)
    assert session["_evidence_trust"] == "untrusted"
    assert session["source"]["type"] == "hermes_batch_runner"
    assert session["task"]["prompt"] == "Summarize the fixture."
    assert session["messages"][-1]["content"] == "[REDACTED]"
    assert mine_session(session).eligibility is Eligibility.DREAMING_ONLY


def test_batch_runner_import_skips_discard_tombstones_and_is_idempotent(tmp_path):
    source = tmp_path / "trajectories.jsonl"
    records = [
        {"prompt_index": 1, "discarded": "no_reasoning"},
        trajectory_record(prompt_index=2),
        trajectory_record(prompt_index=2),
    ]
    source.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    exports = load_batch_runner_export(source)
    assert len(exports) == 1

    queue = SessionQueue(tmp_path / "sessions.sqlite3")
    queue.ingest_many(exports)
    queue.ingest_many(exports)
    assert queue.queued() == [exports[0]["session_id"]]


def test_batch_runner_tool_failures_feed_dreaming_failure_signals(tmp_path):
    record = trajectory_record()
    record["tool_stats"] = {
        "terminal": {"count": 3, "success": 1, "failure": 2},
    }
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    export = load_batch_runner_export(source)[0]

    report = DreamingEngine().light([export])

    assert report.clusters == {"errors": 1, "tools": 1}
    assert report.signals[0].issue == "tool terminal failed"
    assert report.signals[0].evidence["occurrences"] == 2


@pytest.mark.parametrize(
    "tool_stats",
    [
        {"terminal": {"count": True, "success": 0, "failure": 0}},
        {"terminal": {"count": 1, "success": 1, "failure": 1}},
    ],
)
def test_batch_runner_import_rejects_invalid_tool_counters(tmp_path, tool_stats):
    record = trajectory_record()
    record["tool_stats"] = tool_stats
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tool counters"):
        load_batch_runner_export(source)


def test_batch_runner_import_rejects_malformed_file_before_cli_ingest(tmp_path, monkeypatch, capsys):
    source = tmp_path / "trajectories.jsonl"
    source.write_text(
        json.dumps(trajectory_record()) + "\n"
        + '{"prompt_index": 2, "completed": true, "conversations": []}\n',
        encoding="utf-8",
    )
    home = tmp_path / "state"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "ingest-batch", str(source)])

    with pytest.raises(ValueError, match="no conversations"):
        main()

    queue = SessionQueue(home / "sessions.sqlite3")
    assert queue.queued() == []
    assert capsys.readouterr().out == ""


def test_ingest_many_rolls_back_all_rows_on_existing_id_conflict(tmp_path):
    queue = SessionQueue(tmp_path / "sessions.sqlite3")
    queue.ingest({"session_id": "existing", "summary": "original"})

    with pytest.raises(ValueError, match="different content"):
        queue.ingest_many([
            {"session_id": "new", "summary": "would otherwise be inserted"},
            {"session_id": "existing", "summary": "conflicting content"},
        ])

    assert queue.queued() == ["existing"]


def test_batch_import_rejects_unhashable_turn_role(tmp_path):
    record = trajectory_record()
    record["conversations"][1]["from"] = []
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid turn"):
        load_batch_runner_export(source)
