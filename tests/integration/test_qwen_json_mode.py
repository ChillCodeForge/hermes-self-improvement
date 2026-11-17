"""JSON-object roles request ``response_format`` from the endpoint (lab finding F6).

Measured in the isolated lab: qwen3:1.7b answered the REM prompt with a
top-level JSON array 3 of 3 times without JSON mode, and with a valid object
3 of 3 times with ``response_format: {"type": "json_object"}``.
"""

from __future__ import annotations

import json

import pytest

from openai_stub import OpenAIStub
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.judge import PairwiseJudge
from hermes_improve.qwen import OpenAICompatibleQwen, QwenConfig


def client(stub):
    return OpenAICompatibleQwen(QwenConfig.from_env(environ={"QWEN_BASE_URL": stub.base_url, "QWEN_MODEL": "m"}))


def test_json_object_calls_send_response_format_and_plain_calls_do_not():
    with OpenAIStub(lambda body: '{"ok": true}') as stub:
        model = client(stub)
        model.complete(role="qwen_dreamer", system_prompt="s", user_payload="{}", json_object=True)
        model.complete(role="qwen_optimizer", system_prompt="s", user_payload="{}")
    assert stub.requests[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in stub.requests[1]


def test_endpoint_rejecting_response_format_is_retried_once_without_it_and_remembered():
    def reply(body):
        if "response_format" in body:
            return 400, json.dumps({"error": {"message": "unknown field: response_format"}})
        return '{"hypotheses": []}'

    with OpenAIStub(reply) as stub:
        model = client(stub)
        first = model.complete(role="qwen_dreamer", system_prompt="s", user_payload="{}", json_object=True)
        second = model.complete(role="qwen_dreamer", system_prompt="s", user_payload="{}", json_object=True)
    assert first == second == '{"hypotheses": []}'
    assert ["response_format" in body for body in stub.requests] == [True, False, False]
    assert model.stats["qwen_dreamer"]["requests"] == 2


def test_other_client_errors_are_not_mistaken_for_missing_json_mode():
    with OpenAIStub(lambda body: (400, '{"error": "context length exceeded"}')) as stub:
        model = client(stub)
        with pytest.raises(RuntimeError, match="local model request failed"):
            model.complete(role="qwen_dreamer", system_prompt="s", user_payload="{}", json_object=True)
    assert len(stub.requests) == 1


def test_dreaming_and_judge_request_json_mode_end_to_end():
    def reply(body):
        role = body["messages"][0]["content"].split("]", 1)[0]
        if "qwen_judge" in role:
            return json.dumps({"winner": "B", "confidence": 0.9, "uncertainty": 0.1, "rationale": "better"})
        if "hypotheses" in body["messages"][1]["content"]:
            return json.dumps({"candidates": []})
        return json.dumps({"hypotheses": []})

    with OpenAIStub(reply) as stub:
        model = client(stub)
        engine = DreamingEngine(model)
        engine.rem([{"session_id": "s1", "task": "t", "final_status": "completed"}])
        engine.deep([{"summary": "x", "issue": "y", "evidence_session_ids": ["s1"]}])
        PairwiseJudge(model).compare(task_prompt="t", criteria=["c"], baseline_output="a", candidate_output="b")
    assert len(stub.requests) >= 3
    assert all(body.get("response_format") == {"type": "json_object"} for body in stub.requests)
