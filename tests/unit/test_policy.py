import pytest

from hermes_improve.models import CandidateType, Split, TaskOutcome
from hermes_improve.policy import CONSTRAINT_ORDER, PromotionPolicy


def outcome(**kwargs):
    values = dict(
        task_id="task-1", split=Split.REGRESSION, suite="FULL", baseline_success=True, candidate_success=True,
        baseline_quality=5, candidate_quality=10, baseline_tokens=100, candidate_tokens=105,
        token_usage_reported=True,
        deterministic_pass=True, security_pass=True, categories=("skills",),
        baseline_latency_ms=100, candidate_latency_ms=105,
    )
    values.update(kwargs)
    return TaskOutcome(**values)


def holdout(**kwargs):
    return outcome(**{"task_id": "holdout-1", "split": Split.SEALED_HOLDOUT, "suite": "SEALED_HOLDOUT", **kwargs})


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
    report = PromotionPolicy().assess("c1", "hash", "v1", [outcome(), holdout()])
    assert report.passed, report.reasons
    assert report.metrics["primary_reason"] is None
    assert report.metrics["token_check"] == "passed"


def test_non_skill_candidate_requires_manual_gate():
    try:
        PromotionPolicy().authorize_type(CandidateType.TOOL_PATCH)
    except ValueError as exc:
        assert "manual" in str(exc)
    else:
        raise AssertionError("tool code must not be auto-promotable")


def test_latency_and_category_regressions_block():
    # Quality gain of exactly min_quality_delta (3.0): enough to pass the
    # quality gate, not a *significant* gain that could justify the latency.
    latency_report = PromotionPolicy().assess(
        "c1", "hash", "v1",
        [
            outcome(candidate_quality=8, candidate_latency_ms=200),
            holdout(candidate_quality=8, candidate_latency_ms=200),
        ],
    )
    assert latency_report.reasons == ("latency increase exceeds policy",)
    assert latency_report.metrics["latency_exception_applied"] is False
    category_report = PromotionPolicy().assess(
        "c1", "hash", "v1",
        [outcome(candidate_quality=0), holdout(candidate_quality=0)],
    )
    assert any(reason.startswith("category quality drop") for reason in category_report.reasons)


def test_skills_are_initial_auto_promotable_types():
    PromotionPolicy().authorize_type(CandidateType.SKILL_PATCH)


def test_non_finite_task_metrics_cannot_become_a_policy_report():
    with pytest.raises(ValueError, match="finite"):
        PromotionPolicy().assess(
            "c1", "hash", "v1", [outcome(candidate_quality=float("nan"))],
        )


def test_section_23_defaults_are_unchanged():
    policy = PromotionPolicy()
    assert (
        policy.max_critical_regressions, policy.max_high_regressions, policy.max_medium_regression_rate,
        policy.min_quality_delta, policy.max_category_drop, policy.max_token_increase_ratio,
        policy.max_latency_increase_ratio,
    ) == (0, 0, 0.01, 3.0, 2.0, 0.10, 0.15)
    assert policy.require_token_measurement is True
    assert policy.min_holdout_quality_delta == 0.0


# --------------------------------------------------------------- token measurement (§24)

def test_unmeasured_token_usage_blocks_by_default():
    rows = [outcome(token_usage_reported=False, baseline_tokens=0, candidate_tokens=0),
            holdout(token_usage_reported=False, baseline_tokens=0, candidate_tokens=0)]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert not report.passed
    assert report.reasons == ("token efficiency not measured: runner reported no token usage",)
    assert report.metrics["token_check"] == "not measured"
    assert report.metrics["token_usage_reported"] is False
    assert report.metrics["tokens_per_task_candidate"] is None


def test_one_unmeasured_final_row_is_enough_to_block():
    report = PromotionPolicy().assess("c1", "hash", "v1", [outcome(), holdout(token_usage_reported=False)])
    assert report.reasons == ("token efficiency not measured: runner reported no token usage",)


def test_rows_default_to_unmeasured_tokens():
    # A row that does not state measurement (e.g. stored before the field
    # existed) must not silently pass the token constraint.
    assert TaskOutcome("t", Split.REGRESSION, "FULL", True, True).token_usage_reported is False


def test_explicit_opt_out_records_the_skipped_token_check():
    rows = [outcome(token_usage_reported=False), holdout(token_usage_reported=False)]
    report = PromotionPolicy(require_token_measurement=False).assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons
    assert report.metrics["token_check"].startswith("skipped")


def test_token_increase_from_measured_zero_baseline_is_not_skipped():
    # Gain of exactly 3.0: passes the quality gate, no significant gain.
    rows = [
        outcome(baseline_tokens=0, candidate_tokens=500, candidate_quality=8),
        holdout(baseline_tokens=0, candidate_tokens=500, candidate_quality=8),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert "token increase exceeds policy" in report.reasons


# --------------------------------------------------------------- exceptions (§23)

def _gain_rows(gain_low: float, gain_high: float, **extra):
    return [
        outcome(task_id="full-1", candidate_quality=5 + gain_low, **extra),
        outcome(task_id="full-2", candidate_quality=5 + gain_high, **extra),
        holdout(task_id="holdout-1", candidate_quality=5 + gain_low, **extra),
        holdout(task_id="holdout-2", candidate_quality=5 + gain_high, **extra),
    ]


def test_token_increase_without_significant_gain_blocks():
    # Mean gain 4 passes the quality gate, but the CI lower bound is below 3.
    rows = _gain_rows(2, 6, candidate_tokens=200)
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.metrics["quality_delta_ci95"][0] <= 3.0
    assert report.reasons == ("token increase exceeds policy",)


def test_token_increase_is_allowed_with_significant_quality_gain():
    rows = _gain_rows(20, 21, candidate_tokens=200)
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.metrics["quality_delta_ci95"][0] > 3.0
    assert report.passed, report.reasons
    assert report.metrics["token_exception_applied"] is True
    # The exception is configurable and can be disabled.
    strict = PromotionPolicy(token_exception_min_quality_ci_lower=None).assess("c1", "hash", "v1", rows)
    assert strict.reasons == ("token increase exceeds policy",)
    higher = PromotionPolicy(token_exception_min_quality_ci_lower=50.0).assess("c1", "hash", "v1", rows)
    assert higher.reasons == ("token increase exceeds policy",)


def test_latency_increase_is_justified_only_by_significant_quality_gain():
    rows = _gain_rows(20, 21, candidate_latency_ms=300)
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons
    assert report.metrics["latency_exception_applied"] is True
    disabled = PromotionPolicy(latency_exception_min_quality_ci_lower=None).assess("c1", "hash", "v1", rows)
    assert disabled.reasons == ("latency increase exceeds policy",)


def test_single_final_task_cannot_establish_a_significant_gain():
    # A single final delta gives a zero-width interval: no variance estimate.
    rows = [outcome(candidate_quality=90, candidate_tokens=500)]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert "token increase exceeds policy" in report.reasons
    assert report.metrics["token_exception_applied"] is False


# --------------------------------------------------------------- generalization (§23)

def _full_rows(count: int = 5, candidate_quality: float = 25):
    return [outcome(task_id=f"full-{index}", candidate_quality=candidate_quality) for index in range(count)]


def test_full_suite_cannot_carry_a_holdout_quality_regression():
    rows = [*_full_rows(), holdout(candidate_quality=4)]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    # The pooled quality gate alone passes: mean and CI lower bound are fine.
    assert report.metrics["quality_delta_mean"] > 3.0
    assert report.metrics["quality_delta_ci95"][0] > 0
    assert report.metrics["holdout_quality_delta_mean"] == -1.0
    assert report.reasons == (
        "sealed holdout quality delta -1.000 is below the generalization minimum 0.000",
    )
    assert report.metrics["failed_constraints"] == ["generalization"]
    assert report.metrics["primary_reason"] == report.reasons[0]


def test_holdout_quality_minimum_is_configurable():
    rows = [*_full_rows(), holdout(candidate_quality=6)]
    assert PromotionPolicy().assess("c1", "hash", "v1", rows).passed
    report = PromotionPolicy(min_holdout_quality_delta=2.0).assess("c1", "hash", "v1", rows)
    assert report.metrics["failed_constraints"] == ["generalization"]


def test_holdout_task_success_regression_fails_generalization():
    rows = [*_full_rows(), holdout(candidate_success=False, candidate_quality=9)]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert "sealed holdout task success rate regressed" in report.reasons
    assert report.metrics["failed_constraints"] == ["task_success", "generalization"]


# --------------------------------------------------------------- task success and order (§23)

def test_task_success_must_not_regress_per_task():
    rows = [outcome(task_id="a", candidate_success=False), outcome(task_id="b"), holdout()]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.reasons == ("task success regressed on 1 task(s): a",)


def test_task_both_arms_fail_is_not_a_success_regression():
    rows = [outcome(task_id="hard", baseline_success=False, candidate_success=False), holdout()]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons


def test_task_success_regression_in_screening_suite_blocks():
    rows = [outcome(task_id="smoke-1", suite="SMOKE", candidate_success=False), outcome(), holdout()]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.reasons == ("task success regressed on 1 task(s): smoke-1",)


def test_reasons_follow_section_23_order_and_expose_the_primary_reason():
    rows = [
        outcome(
            task_id="full-1", candidate_quality=0, candidate_success=False, candidate_tokens=1000,
            candidate_latency_ms=1000, high_regressions=1, security_pass=False, deterministic_pass=False,
            critical_regressions=1, categories=("skills",),
        ),
        holdout(candidate_quality=0, candidate_success=False, candidate_tokens=1000, candidate_latency_ms=1000),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    expected = [
        "security", "deterministic_correctness", "critical_regressions", "high_regressions",
        "task_success", "generalization", "category_regressions", "quality", "token_efficiency", "latency",
    ]
    assert report.metrics["failed_constraints"] == expected
    positions = [CONSTRAINT_ORDER.index(name) for name in report.metrics["failed_constraints"]]
    assert positions == sorted(positions)
    assert report.reasons[0] == "security verifier failed" == report.metrics["primary_reason"]
    assert report.reasons[-1] == "latency increase exceeds policy"


# F27: judge-routed tasks and the pooled quality mean.

def judged(result, **kwargs):
    return outcome(**{"baseline_quality": 100, "candidate_quality": 100, "judge_result": result,
                      "judge_confidence": 0.9, "judge_uncertainty": 0.1, **kwargs})


def test_judged_tasks_no_longer_dilute_the_deterministic_quality_mean():
    # Deterministic delta +4 (above 3.0); two judged rows with delta 0 used to pull the mean to 2.0.
    rows = [
        outcome(task_id="det-1", baseline_quality=90, candidate_quality=94),
        judged("tie", task_id="essay-1"), judged("tie", task_id="essay-2"),
        holdout(baseline_quality=90, candidate_quality=94),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons
    assert report.metrics["quality_delta_mean"] == 4
    assert report.metrics["quality_tasks"] == 2 and report.metrics["judged_final_tasks"] == 2
    assert report.metrics["quality_evidence"] == "deterministic"
    assert report.metrics["manual_approval_reason"] is None


def test_gain_resting_on_judge_preferred_tasks_passes_only_for_manual_approval():
    rows = [
        outcome(task_id="det-1", baseline_quality=90, candidate_quality=90),
        judged("candidate", task_id="essay-1"),
        holdout(baseline_quality=90, candidate_quality=90),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons
    assert report.metrics["quality_evidence"] == "judge"
    assert "judge-preferred" in report.metrics["manual_approval_reason"]
    assert PromotionPolicy().requires_manual_approval(CandidateType.SKILL_PATCH, judge_only_gain=True)
    assert not PromotionPolicy().requires_manual_approval(CandidateType.SKILL_PATCH)


def test_subjective_only_final_evidence_routes_to_manual_approval():
    rows = [judged("candidate", task_id="essay-1"), judged("candidate", task_id="essay-holdout",
                                                           split=Split.SEALED_HOLDOUT, suite="SEALED_HOLDOUT")]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert report.passed, report.reasons
    assert report.metrics["quality_evidence"] == "judge"
    assert report.metrics["quality_delta_mean"] is None and report.metrics["quality_tasks"] == 0


@pytest.mark.parametrize("verdicts", [("tie",), ("baseline",), ("unreliable",), ("candidate", "unreliable")])
def test_no_judge_preference_or_a_blocking_verdict_never_carries_quality(verdicts):
    rows = [judged(result, task_id=f"essay-{index}") for index, result in enumerate(verdicts)]
    rows += [holdout(baseline_quality=90, candidate_quality=90)]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert not report.passed
    assert "quality" in report.metrics["failed_constraints"]
    assert report.metrics["quality_evidence"] is None


def test_judge_preference_cannot_offset_a_deterministic_quality_regression():
    rows = [
        outcome(task_id="det-1", baseline_quality=90, candidate_quality=80),
        judged("candidate", task_id="essay-1"),
        holdout(baseline_quality=90, candidate_quality=90),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert not report.passed
    assert report.metrics["quality_evidence"] is None
    assert any("quality delta" in reason for reason in report.reasons)


def test_judged_tasks_still_count_for_task_success_and_tokens():
    rows = [
        outcome(task_id="det-1", baseline_quality=90, candidate_quality=94),
        judged("candidate", task_id="essay-1", candidate_success=False),
        holdout(baseline_quality=90, candidate_quality=94),
    ]
    report = PromotionPolicy().assess("c1", "hash", "v1", rows)
    assert any("task success regressed" in reason for reason in report.reasons)
    unmeasured = PromotionPolicy().assess("c1", "hash", "v1", [
        outcome(task_id="det-1", baseline_quality=90, candidate_quality=94),
        judged("tie", task_id="essay-1", token_usage_reported=False),
        holdout(baseline_quality=90, candidate_quality=94),
    ])
    assert "token_efficiency" in unmeasured.metrics["failed_constraints"]
