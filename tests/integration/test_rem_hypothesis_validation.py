"""REM hypotheses are checked against the sessions REM received (lab finding F31).

In the labelled S6 rerun qwen3:4b returned ``efficiency: ... unnecessary use
of terminal tool`` whose only evidence session was labelled ``incorrect``.
DEEP then folded it into a correctness proposal ("without relying on terminal
tools") and steered the agent away from computing, the opposite of what the
labelled failures needed. The prompt rule alone did not hold, so the engine
enforces it after parsing.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

from openai_stub import OpenAIStub
from hermes_improve.cli import main
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.ingestion import SessionQueue
from hermes_improve.prompting import load_prompt


LIVE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "dreaming" / "s6_labelled_rem.json").read_text(encoding="utf-8")
)
TERMINAL_SESSION = "hermes-batch-2b8ff771c196e92f6485578fd5e5485f0da15c774ece53451283107755b62ece"


class ReplyModel:
    def __init__(self, rem_reply, deep_reply=None):
        self.rem_reply = rem_reply
        self.deep_reply = deep_reply or {"candidates": []}
        self.calls = []

    def complete(self, *, role, system_prompt, user_payload, json_object=False):
        self.calls.append({"role": role, "system_prompt": system_prompt, "user_payload": user_payload})
        return json.dumps(self.rem_reply if role == "qwen_dreamer" else self.deep_reply)


def live_sessions():
    return [
        {
            "session_id": row["session_id"],
            "task": "compute a value from the data file",
            "final_status": "completed",
            "outcome": {"label": row["outcome"], "source": row["source"], "note": row["note"]},
        }
        for row in LIVE["labels"]
    ]


def session(session_id, label=None):
    row = {"session_id": session_id, "task": "t", "final_status": "completed"}
    if label is not None:
        row["outcome"] = {"label": label, "source": "verifier"}
    return row


def hypothesis(summary, ids):
    return {"summary": summary, "issue": "i", "evidence_session_ids": ids}


def rem(sessions, *hypotheses):
    return DreamingEngine(ReplyModel({"hypotheses": list(hypotheses)})).rem(sessions)


# --- live shapes ---------------------------------------------------------------


def test_live_efficiency_hypothesis_on_an_incorrect_session_is_rejected():
    report = DreamingEngine(ReplyModel(LIVE["rem_reply"])).rem(live_sessions())

    [kept] = report.hypotheses
    assert kept["kind"] == "correctness"
    assert kept["summary"].startswith("correctness: The agent fails to correctly parse")
    assert len(kept["evidence_session_ids"]) == 10

    [rejected] = report.rejected_hypotheses
    assert rejected["kind"] == "efficiency"
    assert rejected["reason"] == "efficiency_on_failed_sessions"
    assert rejected["evidence_session_ids"] == [TERMINAL_SESSION]
    assert rejected["failed_session_ids"] == [TERMINAL_SESSION]
    assert "terminal tool" in rejected["summary"]
    assert report.clusters["rejected_hypotheses"] == 1


def test_deep_receives_only_the_kept_hypothesis_with_its_kind():
    model = ReplyModel(LIVE["rem_reply"])
    engine = DreamingEngine(model)
    report = engine.rem(live_sessions())
    engine.deep(report.hypotheses)

    payload = model.calls[-1]["user_payload"]
    assert model.calls[-1]["role"] == "qwen_optimizer"
    assert '"kind": "correctness"' in payload
    assert "terminal tool" not in payload


# --- evidence ids ----------------------------------------------------------------


def test_unknown_evidence_ids_are_dropped_and_reported():
    report = rem([session("s1", "incorrect")], hypothesis("correctness: x", ["s1", "ghost", "s1"]))
    [kept] = report.hypotheses
    assert kept["evidence_session_ids"] == ["s1"]
    assert kept["unknown_evidence_session_ids"] == ["ghost"]
    assert report.rejected_hypotheses == ()
    assert "rejected_hypotheses" not in report.clusters


@pytest.mark.parametrize(("ids", "reason"), [(["ghost"], "unknown_evidence"), ([], "no_evidence")])
def test_hypotheses_without_known_evidence_are_rejected(ids, reason):
    report = rem([session("s1")], hypothesis("correctness: x", ids))
    assert report.hypotheses == ()
    [rejected] = report.rejected_hypotheses
    assert rejected["reason"] == reason
    assert rejected["evidence_session_ids"] == []


# --- efficiency rule ---------------------------------------------------------------


def test_efficiency_on_correct_sessions_is_kept_and_marked_verified():
    report = rem([session("a", "correct"), session("b")], hypothesis("efficiency: fewer calls", ["a", "b"]))
    [kept] = report.hypotheses
    assert kept["kind"] == "efficiency" and kept["outcome_verified"] is True


@pytest.mark.parametrize("label", ["incorrect", "partial"])
def test_efficiency_with_any_failed_session_is_rejected(label):
    report = rem([session("a", "correct"), session("b", label)], hypothesis("efficiency: x", ["a", "b"]))
    assert report.hypotheses == ()
    [rejected] = report.rejected_hypotheses
    assert rejected["reason"] == "efficiency_on_failed_sessions"
    assert rejected["failed_session_ids"] == ["b"]


def test_efficiency_on_unlabelled_sessions_is_kept_but_marked_unverified():
    report = rem([session("a"), session("b")], hypothesis("efficiency: x", ["a", "b"]))
    [kept] = report.hypotheses
    assert kept["outcome_verified"] is False


def test_correctness_hypotheses_on_failed_sessions_are_kept():
    report = rem([session("a", "incorrect")], hypothesis("correctness: wrong method", ["a"]))
    [kept] = report.hypotheses
    assert kept["kind"] == "correctness" and "outcome_verified" not in kept


@pytest.mark.parametrize(("summary", "kind"), [
    ("  Efficiency: x", "efficiency"),
    ("CORRECTNESS:x", "correctness"),
    ("tool unused", "unknown"),
    ("efficiency - x", "unknown"),
])
def test_kind_comes_from_the_summary_prefix(summary, kind):
    report = rem([session("a", "correct")], hypothesis(summary, ["a"]))
    assert report.hypotheses[0]["kind"] == kind


# --- DEEP prompt ---------------------------------------------------------------------


def test_deep_prompt_keeps_kinds_apart_and_the_untrusted_framing():
    prompt = load_prompt("dreaming/deep.md")
    assert "UNTRUSTED EVIDENCE" in prompt
    assert "kind" in prompt and "efficiency" in prompt and "correctness" in prompt


# --- CLI and observability -----------------------------------------------------------


def run(monkeypatch, capsys, *args) -> dict:
    monkeypatch.setattr(sys, "argv", ["hermes-improve", *args])
    capsys.readouterr()
    assert main() == 0
    return json.loads(capsys.readouterr().out)


def test_dream_rem_reports_rejected_hypotheses(tmp_path, monkeypatch, capsys):
    home = tmp_path / "state"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    queue = SessionQueue(home / "sessions.sqlite3")
    for row in live_sessions():
        queue.ingest({key: row[key] for key in ("session_id", "task", "final_status")})
        queue.label_outcome(row["session_id"], outcome=row["outcome"]["label"], source="verifier",
                            note=row["outcome"]["note"])

    reply = json.dumps(LIVE["rem_reply"])
    with OpenAIStub(lambda body: reply) as stub:
        monkeypatch.setenv("QWEN_BASE_URL", stub.base_url)
        monkeypatch.setenv("QWEN_MODEL", "dreamer")
        result = run(monkeypatch, capsys, "dream", "--stage", "rem")

    assert [item["kind"] for item in result["hypotheses"]] == ["correctness"]
    [rejected] = result["rejected_hypotheses"]
    assert rejected["reason"] == "efficiency_on_failed_sessions"
    assert result["clusters"]["rejected_hypotheses"] == 1
    with sqlite3.connect(home / "observability.sqlite3") as db:
        clusters, hypotheses = db.execute(
            "SELECT clusters_json, hypotheses FROM dreaming_runs WHERE stage = 'rem'"
        ).fetchone()
    assert json.loads(clusters)["rejected_hypotheses"] == 1 and hypotheses == 1
