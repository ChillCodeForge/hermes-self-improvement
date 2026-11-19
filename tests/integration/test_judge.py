from __future__ import annotations

import json

import pytest

from openai_stub import OpenAIStub
from hermes_improve.judge import (
    MAX_AGGREGATED_PRESENTATIONS, MAX_PRESENTATION_RATIONALE_CHARS, JudgeConfig, PairwiseJudge, aggregate_verdicts,
    parse_judgment,
)
from hermes_improve.qwen import OpenAICompatibleQwen, QwenConfig


def verdict(winner, confidence=0.9, uncertainty=0.1):
    return json.dumps({"winner": winner, "confidence": confidence, "uncertainty": uncertainty, "rationale": "r"})


def prefers(output):
    """Judge that consistently prefers the response equal to *output*, whatever its position."""
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        return verdict("A" if payload["response_A"] == output else "B")
    return reply


def client(stub, tmp_path=None):
    return OpenAICompatibleQwen(QwenConfig(
        stub.base_url, "solver-model", judge_model="judge-model",
        log_dir=str(tmp_path / "logs") if tmp_path else None,
    ))


def compare(judge):
    return judge.compare(
        task_prompt="Summarize the incident", criteria=["mentions root cause"],
        baseline_output="baseline answer", candidate_output="candidate answer",
    )


def test_judge_swaps_positions_over_real_http_and_maps_consistent_preference(tmp_path):
    with OpenAIStub(prefers("candidate answer")) as stub:
        result = compare(PairwiseJudge(client(stub, tmp_path)))
    assert result.result == "candidate"
    assert result.confidence == 0.9 and result.uncertainty == 0.1
    assert [request["model"] for request in stub.requests] == ["judge-model", "judge-model"]
    first, second = stub.user_payloads()
    assert (first["response_A"], first["response_B"]) == ("baseline answer", "candidate answer")
    assert (second["response_A"], second["response_B"]) == ("candidate answer", "baseline answer")
    assert set(first) == {"task", "criteria", "response_A", "response_B"}, "judge input must stay blind"
    system = stub.requests[0]["messages"][0]["content"]
    assert system.startswith("[logical_role=qwen_judge]") and "untrusted data" in system


def test_position_dependent_contradiction_is_unreliable():
    with OpenAIStub(lambda body: verdict("B")) as stub:
        result = compare(PairwiseJudge(client(stub)))
    assert result.result == "unreliable"
    assert "position-dependent" in result.reasons[0]


def test_consistent_tie_and_baseline_preference():
    with OpenAIStub(lambda body: verdict("TIE")) as stub:
        assert compare(PairwiseJudge(client(stub))).result == "tie"
    with OpenAIStub(prefers("baseline answer")) as stub:
        assert compare(PairwiseJudge(client(stub))).result == "baseline"


def test_invalid_json_is_retried_within_bound_then_unreliable():
    answers = iter(["not json", verdict("B"), "{broken", verdict("A")])
    with OpenAIStub(lambda body: next(answers)) as stub:
        recovered = compare(PairwiseJudge(client(stub)))
    assert recovered.result == "candidate"
    assert recovered.attempts == 4
    with OpenAIStub(lambda body: "I think B is better") as stub:
        failed = compare(PairwiseJudge(client(stub), JudgeConfig(max_attempts=2)))
    assert failed.result == "unreliable"
    assert len(stub.requests) == 4, "retries must be bounded per presentation"
    assert "no valid judgment" in failed.reasons[0]


def test_transport_errors_never_become_success():
    with OpenAIStub(lambda body: (500, "upstream exploded")) as stub:
        result = compare(PairwiseJudge(client(stub), JudgeConfig(max_attempts=1)))
    assert result.result == "unreliable"
    assert len(stub.requests) == 2


@pytest.mark.parametrize(
    ("confidence", "uncertainty", "reason"),
    [(0.3, 0.1, "confidence"), (0.9, 0.8, "uncertainty")],
)
def test_low_confidence_or_high_uncertainty_is_unreliable(confidence, uncertainty, reason):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        winner = "A" if payload["response_A"] == "candidate answer" else "B"
        return verdict(winner, confidence, uncertainty)

    with OpenAIStub(reply) as stub:
        result = compare(PairwiseJudge(client(stub)))
    assert result.result == "unreliable"
    assert reason in result.reasons[0]


@pytest.mark.parametrize(
    "answer",
    [
        '{"winner":"A","confidence":0.9}',
        '{"winner":"C","confidence":0.9,"uncertainty":0.1}',
        '{"winner":"A","confidence":2,"uncertainty":0.1}',
        '{"winner":"A","confidence":NaN,"uncertainty":0.1}',
        '{"winner":"A","confidence":0.9,"uncertainty":0.1,"promote":true}',
        '["A"]',
    ],
)
def test_parse_judgment_is_strict(answer):
    with pytest.raises(ValueError):
        parse_judgment(answer)


def test_judge_role_has_separate_log_without_content(tmp_path):
    with OpenAIStub(prefers("candidate answer")) as stub:
        model = client(stub, tmp_path)
        compare(PairwiseJudge(model))
        model.complete(role="qwen_solver", system_prompt="solve", user_payload=json.dumps({"response_A": "a", "response_B": "b"}))
    judge_log = (tmp_path / "logs" / "qwen_judge.jsonl").read_text().splitlines()
    solver_log = (tmp_path / "logs" / "qwen_solver.jsonl").read_text().splitlines()
    assert len(judge_log) == 2 and len(solver_log) == 1
    entry = json.loads(judge_log[0])
    assert entry["role"] == "qwen_judge" and entry["model"] == "judge-model" and entry["status"] == "ok"
    assert entry["input_tokens"] == 50
    assert "candidate answer" not in (tmp_path / "logs" / "qwen_judge.jsonl").read_text()
    assert model.stats["qwen_judge"]["requests"] == 2


def test_judge_requires_prompt_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        PairwiseJudge(object(), JudgeConfig(prompt_dir=tmp_path))


def test_aggregate_verdicts_requires_agreement_across_runs():
    with OpenAIStub(prefers("candidate answer")) as stub:
        judge = PairwiseJudge(client(stub))
        good = compare(judge)
    with OpenAIStub(lambda body: verdict("TIE")) as stub:
        tie = compare(PairwiseJudge(client(stub)))
    assert aggregate_verdicts([good, good]).result == "candidate"
    mixed = aggregate_verdicts([good, tie])
    assert mixed.result == "unreliable" and "disagree" in mixed.reasons[-1]


def test_aggregate_keeps_bounded_per_seed_presentations_for_position_bias_audit():
    with OpenAIStub(prefers("candidate answer")) as stub:
        good = compare(PairwiseJudge(client(stub)))
    with OpenAIStub(lambda body: verdict("B")) as stub:
        contradiction = compare(PairwiseJudge(client(stub)))
    combined = aggregate_verdicts([good, contradiction], seeds=[7, 8])
    assert combined.result == "unreliable"
    assert [(item["seed"], item["order"]) for item in combined.presentations] == [
        (7, "A=baseline,B=candidate"), (7, "A=candidate,B=baseline"),
        (8, "A=baseline,B=candidate"), (8, "A=candidate,B=baseline"),
    ]
    assert [item["judgment"]["winner"] for item in combined.presentations] == ["B", "A", "B", "B"]
    assert combined.presentations_omitted == 0
    assert json.loads(json.dumps(combined.to_dict()))["presentations"] == list(combined.presentations)
    assert aggregate_verdicts([good, good]).presentations[2]["run"] == 1, "run index without explicit seeds"


def test_aggregate_presentations_are_capped_in_count_and_rationale_length():
    long = json.dumps({"winner": "TIE", "confidence": 0.9, "uncertainty": 0.1, "rationale": "x" * 2000})
    with OpenAIStub(lambda body: long) as stub:
        tie = compare(PairwiseJudge(client(stub)))
    combined = aggregate_verdicts([tie] * 30)
    assert combined.result == "tie"
    assert len(combined.presentations) == MAX_AGGREGATED_PRESENTATIONS
    assert combined.presentations_omitted == 60 - MAX_AGGREGATED_PRESENTATIONS
    rationale = combined.presentations[0]["judgment"]["rationale"]
    assert len(rationale) <= MAX_PRESENTATION_RATIONALE_CHARS and rationale.startswith("xxx")
    assert len(tie.presentations[0]["judgment"]["rationale"]) == 2000, "per-seed verdict is not mutated"
    with pytest.raises(ValueError, match="seeds"):
        aggregate_verdicts([tie, tie], seeds=[1])
