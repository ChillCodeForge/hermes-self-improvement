"""F28: which ground truth lets a judge-routed (subjective-only) task be a HARD_BENCHMARK."""

from __future__ import annotations

import pytest

from benchmark_factory import record
from hermes_improve.benchmark_schema import parse_benchmark_document
from hermes_improve.eligibility import GroundTruthSource, assess_benchmark
from hermes_improve.models import Benchmark, Eligibility


def judge_record(identifier="judge-task", **ground_truth):
    data = record(identifier, categories=["docs"])
    data["environment"].update({"container_image": "image@sha256:" + "a" * 64, "repository_fixture": "repo"})
    data["environment"]["input_data"] = "sha256:" + "b" * 64
    data["environment"]["dependency_digest"] = "sha256:" + "c" * 64
    data["verification"] = {"deterministic": [], "subjective": ["States the root cause."]}
    data["ground_truth"] = {"type": "llm_judge", "confidence": 0.9, **ground_truth}
    return data


def test_hard_judge_benchmark_with_llm_judge_ground_truth_is_valid():
    benchmark = parse_benchmark_document(judge_record())
    assert benchmark.ground_truth["type"] == "llm_judge"


def test_structured_result_no_longer_stands_in_for_checks():
    data = judge_record()
    data["ground_truth"] = {"type": "structured_result", "confidence": 0.9,
                            "structured_result": {"rubric": "unchecked"}}
    with pytest.raises(ValueError, match="llm_judge"):
        Benchmark.from_dict(data)


def test_llm_judge_ground_truth_requires_subjective_criteria():
    data = judge_record()
    data["verification"]["subjective"] = []
    with pytest.raises(ValueError, match="subjective criteria"):
        Benchmark.from_dict(data)


def test_llm_judge_ground_truth_requires_high_confidence_for_hard():
    with pytest.raises(ValueError, match="confidence of at least 0.8"):
        Benchmark.from_dict(judge_record(confidence=0.5))


def test_soft_judge_benchmarks_keep_working_without_the_hard_rule():
    data = judge_record(confidence=0.4)
    data["eligibility"]["class"] = "SOFT_BENCHMARK"
    assert Benchmark.from_dict(data).ground_truth["confidence"] == 0.4


def test_deterministic_benchmarks_are_unaffected():
    assert Benchmark.from_dict(record("plain")).verification["deterministic"]


def test_classifier_accepts_an_authored_judge_rubric_as_hard():
    decision = assess_benchmark(Benchmark.from_dict(judge_record()))
    assert decision.ground_truth is GroundTruthSource.LLM_JUDGE
    assert decision.eligibility is Eligibility.HARD_BENCHMARK
    assert any("pairwise judge" in reason for reason in decision.reasons)


def test_classifier_keeps_llm_judge_soft_without_subjective_rubric_or_replay():
    data = judge_record()
    data["eligibility"]["class"] = "SOFT_BENCHMARK"
    data["eligibility"]["replay_verified"] = False
    assert assess_benchmark(Benchmark.from_dict(data)).eligibility is Eligibility.SOFT_BENCHMARK
