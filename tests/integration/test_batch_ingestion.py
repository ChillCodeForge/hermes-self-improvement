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


_ZERO = {"count": 0, "success": 0, "failure": 0}
_USED_TOOLS = {
    "terminal": {"count": 4, "success": 3, "failure": 1},
    "read_file": {"count": 2, "success": 2, "failure": 0},
    "web_search": {"count": 1, "success": 0, "failure": 0},
}


def hermes_v022_tool_stats():
    """Hermes v0.22 lists every registered tool; only three were actually used."""
    stats = {}
    for prefix in ("kanban", "feishu", "browser", "github", "calendar", "memory"):
        for index in range(15):
            stats[f"{prefix}_action_{index}"] = dict(_ZERO)
    stats.update({name: dict(counts) for name, counts in _USED_TOOLS.items()})
    return stats


def test_batch_import_drops_unused_zero_tool_stats_and_shrinks_rem_payload(tmp_path):
    from hermes_improve.dreaming import _untrusted_projection

    record = trajectory_record()
    record["tool_stats"] = hermes_v022_tool_stats()
    assert len(record["tool_stats"]) == 93
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    export = load_batch_runner_export(source)[0]

    assert export["tool_stats"] == _USED_TOOLS
    unfiltered = dict(export, tool_stats=record["tool_stats"])
    projected = len(json.dumps(_untrusted_projection(export)))
    assert projected < len(json.dumps(_untrusted_projection(unfiltered))) / 5
    report = DreamingEngine().light([export])
    assert report.clusters["tools"] == 3


def test_batch_import_keeps_empty_tool_stats_when_no_tool_was_used(tmp_path):
    record = trajectory_record()
    record["tool_stats"] = {"kanban_create": dict(_ZERO), "feishu_send": dict(_ZERO)}
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    assert load_batch_runner_export(source)[0]["tool_stats"] == {}


@pytest.mark.parametrize(
    "bad_row",
    [
        {"count": 0, "success": 0},
        {"count": 0, "success": 0, "failure": -1},
        {"count": False, "success": 0, "failure": 0},
    ],
)
def test_batch_import_validates_zero_like_rows_before_dropping(tmp_path, bad_row):
    record = trajectory_record()
    record["tool_stats"] = {**hermes_v022_tool_stats(), "kanban_broken": bad_row}
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="tool counters"):
        load_batch_runner_export(source)


def test_tool_stats_limit_applies_to_raw_rows_including_zero_rows(tmp_path):
    record = trajectory_record()
    record["tool_stats"] = {f"tool_{index}": dict(_ZERO) for index in range(513)}
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid tool statistics"):
        load_batch_runner_export(source)


def test_zero_tool_rows_do_not_change_evaluation_tool_metrics():
    from hermes_improve.hermes_runner import parse_trajectory_record

    base = {
        "prompt_index": 0,
        "conversations": trajectory_record()["conversations"],
        "completed": True,
        "partial": False,
        "api_calls": 2,
    }
    used_only = parse_trajectory_record(dict(base, tool_stats=dict(_USED_TOOLS)))
    with_zero_rows = parse_trajectory_record(dict(base, tool_stats=hermes_v022_tool_stats()))

    assert (used_only.tool_calls, used_only.failed_tool_calls) == (7, 1)
    assert (with_zero_rows.tool_calls, with_zero_rows.failed_tool_calls) == (7, 1)
