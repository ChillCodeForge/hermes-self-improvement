from hermes_improve.models import CandidateType, Split, TaskOutcome
from hermes_improve.policy import PromotionPolicy


def outcome(**kwargs):
    values = dict(
        task_id="task-1", split=Split.REGRESSION, suite="FULL", baseline_success=True, candidate_success=True,
        baseline_quality=5, candidate_quality=10, baseline_tokens=100, candidate_tokens=105,
        deterministic_pass=True, security_pass=True, categories=("skills",),
        baseline_latency_ms=100, candidate_latency_ms=105,
    )
    values.update(kwargs)
    return TaskOutcome(**values)


def test_security_failure_cannot_be_offset_by_quality():
    report = PromotionPolicy().assess("c1", "hash", "v1", [outcome(security_pass=False, candidate_quality=100)])
    assert not report.passed
    assert "security verifier failed" in report.reasons


def test_critical_and_high_regressions_block():
    for field in ("critical_regressions", "high_regressions"):
        report = PromotionPolicy().assess("c1", "hash", "v1", [outcome(**{field: 1})])
        assert not report.passed


def test_promotion_requires_full_and_holdout_evidence():
    full_only = outcome()
    report = PromotionPolicy().assess("c1", "hash", "v1", [full_only])
    assert not report.passed
    assert "sealed holdout evidence is missing" in report.reasons


def test_pass_requires_full_and_holdout_and_hard_gates():
    full = outcome()
    holdout = outcome(task_id="holdout-1", split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT")
    report = PromotionPolicy().assess("c1", "hash", "v1", [full, holdout])
    assert report.passed


def test_non_skill_candidate_requires_manual_gate():
    try:
        PromotionPolicy().authorize_type(CandidateType.TOOL_PATCH)
    except ValueError as exc:
        assert "manual" in str(exc)
    else:
        raise AssertionError("tool code must not be auto-promotable")


def test_latency_and_category_regressions_block():
    latency_report = PromotionPolicy().assess(
        "c1", "hash", "v1",
        [
            outcome(candidate_latency_ms=200),
            outcome(task_id="h", split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT", candidate_latency_ms=200),
        ],
    )
    assert "latency increase exceeds policy" in latency_report.reasons
    category_report = PromotionPolicy().assess(
        "c1", "hash", "v1",
        [
            outcome(candidate_quality=0),
            outcome(task_id="h", split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT", candidate_quality=0),
        ],
    )
    assert any(reason.startswith("category quality drop") for reason in category_report.reasons)


def test_skills_are_initial_auto_promotable_types():
    PromotionPolicy().authorize_type(CandidateType.SKILL_PATCH)


def test_non_finite_task_metrics_cannot_become_a_policy_report():
    import pytest

    with pytest.raises(ValueError, match="finite"):
        PromotionPolicy().assess(
            "c1", "hash", "v1", [outcome(candidate_quality=float("nan"))],
        )
