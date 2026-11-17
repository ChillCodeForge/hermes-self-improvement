"""LIGHT must not consume the session queue that REM reads (lab finding F7).

Before the fix ``dream --stage light`` marked every session ``processed``, so
the documented LIGHT-then-REM order left REM with ``sessions_processed: 0``
while it still sent a Qwen request.
"""

from __future__ import annotations

import json
import sys

import pytest

from openai_stub import OpenAIStub
from hermes_improve.cli import main
from hermes_improve.ingestion import SessionQueue


def run(monkeypatch, capsys, *args) -> dict:
    monkeypatch.setattr(sys, "argv", ["hermes-improve", *args])
    capsys.readouterr()
    assert main() == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def state(tmp_path, monkeypatch):
    home = tmp_path / "state"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    queue = SessionQueue(home / "sessions.sqlite3")
    queue.ingest({"session_id": "s-1", "task": "write the file", "final_status": "failed",
                  "errors": ["write_file was not called"]})
    return queue


def test_rem_after_light_still_sees_the_sessions_and_completes_them(state, monkeypatch, capsys):
    reply = json.dumps({"hypotheses": [
        {"summary": "tool unused", "issue": "write_file not called", "evidence_session_ids": ["s-1"]},
    ]})
    light = run(monkeypatch, capsys, "dream", "--stage", "light")
    assert light["sessions_processed"] == 1
    assert state.queued() == ["s-1"], "LIGHT must leave the session queued for REM"
    assert run(monkeypatch, capsys, "dream", "--stage", "light")["sessions_processed"] == 0

    with OpenAIStub(lambda body: reply) as stub:
        monkeypatch.setenv("QWEN_BASE_URL", stub.base_url)
        monkeypatch.setenv("QWEN_MODEL", "dreamer")
        rem = run(monkeypatch, capsys, "dream", "--stage", "rem")
        assert rem["sessions_processed"] == 1
        assert [item["summary"] for item in rem["hypotheses"]] == ["tool unused"]
        assert state.queued() == [] and state.status_counts() == {"processed": 1}

        again = run(monkeypatch, capsys, "dream", "--stage", "rem")
    assert again["sessions_processed"] == 0 and again["hypotheses"] == []
    assert len(stub.requests) == 1, "REM without queued sessions must not call the model"


def test_stage_marks_are_validated(state):
    with pytest.raises(ValueError, match="unknown dreaming stage"):
        state.pending_for_stage("deep")
    with pytest.raises(KeyError):
        state.mark_stage("missing", "light")
    state.mark_stage("s-1", "light")
    state.mark_stage("s-1", "light")  # idempotent
    assert state.pending_for_stage("light") == [] and state.pending_for_stage("rem") == ["s-1"]
