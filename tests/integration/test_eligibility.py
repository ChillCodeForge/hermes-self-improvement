from __future__ import annotations

import math

import pytest

from benchmark_factory import record
from hermes_improve.benchmarks import lint_public_corpus
from hermes_improve.eligibility import (
    GROUND_TRUTH_PRIORITY,
    REPRODUCIBILITY_WEIGHTS,
    GroundTruthSource,
    ReproducibilityFactors,
    assess_benchmark,
    assess_session,
    classify,
    rank_ground_truth,
    reproducibility_score,
)
from hermes_improve.miner import mine_session
from hermes_improve.models import Benchmark, Eligibility


def factors(**overrides) -> ReproducibilityFactors:
    values = {name: True for name in REPRODUCIBILITY_WEIGHTS}
    values.update(overrides)
    return ReproducibilityFactors(**values)


def full_session(**overrides) -> dict:
    session = {
        "session_id": "s",
        "task": {"prompt": "Build the package", "input_available": True},
        "environment": {
            "git_commit": "a" * 40, "dependency_digest": "sha256:deps",
            "container_image": "image@sha256:" + "b" * 64, "network_policy": "disabled",
        },
        "verification": {"deterministic": [{"type": "exit_code"}]},
        "security": {"sanitized": True, "contains_secrets": False},
        "replay_verified": True,
    }
    session.update(overrides)
    return session


def test_weights_cover_the_nine_spec_factors_and_sum_to_one():
    assert list(REPRODUCIBILITY_WEIGHTS) == [
        "git_commit_present", "input_present", "dependencies_reproducible", "container_present",
        "external_services_controllable", "tests_present", "ground_truth_present",
        "secrets_removed", "replay_successful",
    ]
    assert math.isclose(sum(REPRODUCIBILITY_WEIGHTS.values()), 1.0)
    assert reproducibility_score(factors()) == 1.0
    assert reproducibility_score(factors(replay_successful=False, container_present=False)) == 0.75


def test_factors_must_be_real_booleans():
    with pytest.raises(TypeError):
        factors(replay_successful="true")


def test_ground_truth_priority_order_matches_spec():
    assert [source for source, _ in sorted(GROUND_TRUTH_PRIORITY.items(), key=lambda item: item[1])] == [
        GroundTruthSource.DETERMINISTIC_TESTS,
        GroundTruthSource.INTEGRATION_TESTS,
        GroundTruthSource.STRUCTURED_RESULT,
        GroundTruthSource.HUMAN_ACCEPTED,
        GroundTruthSource.MULTIPLE_MACHINE_SIGNALS,
        GroundTruthSource.LLM_JUDGE,
        GroundTruthSource.AGENT_SELF_REPORT,
    ]
    assert rank_ground_truth([GroundTruthSource.LLM_JUDGE, GroundTruthSource.HUMAN_ACCEPTED]) is (
        GroundTruthSource.HUMAN_ACCEPTED
    )
    assert rank_ground_truth([]) is GroundTruthSource.AGENT_SELF_REPORT


@pytest.mark.parametrize(
    ("overrides", "ground_truth", "expected"),
    [
        ({}, GroundTruthSource.DETERMINISTIC_TESTS, Eligibility.HARD_BENCHMARK),
        ({}, GroundTruthSource.STRUCTURED_RESULT, Eligibility.HARD_BENCHMARK),
        ({"replay_successful": False}, GroundTruthSource.DETERMINISTIC_TESTS, Eligibility.SOFT_BENCHMARK),
        ({}, GroundTruthSource.HUMAN_ACCEPTED, Eligibility.SOFT_BENCHMARK),
        ({}, GroundTruthSource.LLM_JUDGE, Eligibility.SOFT_BENCHMARK),
        ({"secrets_removed": False}, GroundTruthSource.DETERMINISTIC_TESTS, Eligibility.EXCLUDED),
        (
            {name: False for name in REPRODUCIBILITY_WEIGHTS if name != "secrets_removed"},
            GroundTruthSource.AGENT_SELF_REPORT,
            Eligibility.DREAMING_ONLY,
        ),
    ],
)
def test_classifier_maps_evidence_to_eligibility(overrides, ground_truth, expected):
    if ground_truth is not GroundTruthSource.AGENT_SELF_REPORT:
        overrides.setdefault("ground_truth_present", True)
    assert classify(factors(**overrides), ground_truth).eligibility is expected


def test_agent_self_report_is_never_benchmark_ground_truth():
    decision = classify(factors(ground_truth_present=False), GroundTruthSource.AGENT_SELF_REPORT)
    assert decision.eligibility is Eligibility.DREAMING_ONLY
    assert decision.ground_truth_confidence == 0.0
    assert "agent self-report is never benchmark ground truth" in decision.reasons
    with pytest.raises(ValueError, match="must agree"):
        classify(factors(), GroundTruthSource.AGENT_SELF_REPORT)


def test_session_assessment_ranks_ground_truth_and_scores_factors():
    hard = assess_session(full_session())
    assert hard.eligibility is Eligibility.HARD_BENCHMARK
    assert hard.ground_truth is GroundTruthSource.DETERMINISTIC_TESTS
    judged = assess_session(full_session(
        verification={}, ground_truth={"llm_judge": {"verdict": "better"}, "human_accepted": True},
    ))
    assert judged.ground_truth is GroundTruthSource.HUMAN_ACCEPTED
    assert judged.eligibility is Eligibility.SOFT_BENCHMARK
    signals = assess_session(full_session(verification={"machine_signals": ["lint", "build"]}))
    assert signals.ground_truth is GroundTruthSource.MULTIPLE_MACHINE_SIGNALS


def test_miner_applies_classifier_and_ground_truth_type():
    candidate = mine_session(full_session(verification={"structured_result": {"rows": 3}}))
    assert candidate.eligibility is Eligibility.HARD_BENCHMARK
    assert candidate.ground_truth_type == "structured_result"
    assert candidate.reproducibility_score == 0.85


def test_benchmark_assessment_and_lint_reject_overclaimed_records():
    authored = record("overclaimed", categories=["git"])
    decision = assess_benchmark(Benchmark.from_dict(authored))
    assert decision.eligibility is Eligibility.SOFT_BENCHMARK
    with pytest.raises(ValueError, match="exceeds classifier result"):
        lint_public_corpus([Benchmark.from_dict(authored)])


def test_lint_enforces_taxonomy_difficulty_and_replay_evidence():
    data = record("lint-me", categories=["cli"], eligibility="SOFT_BENCHMARK")
    data["eligibility"]["reproducibility_score"] = 0.0
    with pytest.raises(ValueError) as error:
        lint_public_corpus([Benchmark.from_dict(data)])
    assert "unknown taxonomy categories ['cli']" in str(error.value)
    assert "replay_verified requires eligibility.replay_evidence" in str(error.value)
