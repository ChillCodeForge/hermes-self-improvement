"""Benchmark eligibility: reproducibility scoring and ground-truth ranking (spec §13/§14).

Reproducibility weights (sum to 1.0):

| Factor                          | Weight |
|---------------------------------|--------|
| git_commit_present              | 0.10   |
| input_present                   | 0.10   |
| dependencies_reproducible       | 0.10   |
| container_present               | 0.10   |
| external_services_controllable  | 0.10   |
| tests_present                   | 0.15   |
| ground_truth_present            | 0.15   |
| secrets_removed                 | 0.05   |
| replay_successful               | 0.15   |

`secrets_removed` carries a small weight because it is also a hard gate: a
record that is not sanitized is always EXCLUDED regardless of its score.

Ground-truth priority (1 = strongest): deterministic tests, integration tests,
structured API/file result, human accepted, multiple machine signals, LLM
judge, agent self-report. Only priorities 1-3 may back a HARD_BENCHMARK;
agent self-report never counts as ground truth at all.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .models import Benchmark, Eligibility


class GroundTruthSource(StrEnum):
    DETERMINISTIC_TESTS = "deterministic_tests"
    INTEGRATION_TESTS = "integration_tests"
    STRUCTURED_RESULT = "structured_result"
    HUMAN_ACCEPTED = "human_accepted"
    MULTIPLE_MACHINE_SIGNALS = "multiple_machine_signals"
    LLM_JUDGE = "llm_judge"
    AGENT_SELF_REPORT = "agent_self_report"


GROUND_TRUTH_PRIORITY: Mapping[GroundTruthSource, int] = MappingProxyType({
    source: rank for rank, source in enumerate(GroundTruthSource, start=1)
})
GROUND_TRUTH_CONFIDENCE: Mapping[GroundTruthSource, float] = MappingProxyType({
    GroundTruthSource.DETERMINISTIC_TESTS: 1.0,
    GroundTruthSource.INTEGRATION_TESTS: 0.95,
    GroundTruthSource.STRUCTURED_RESULT: 0.9,
    GroundTruthSource.HUMAN_ACCEPTED: 0.8,
    GroundTruthSource.MULTIPLE_MACHINE_SIGNALS: 0.6,
    GroundTruthSource.LLM_JUDGE: 0.4,
    GroundTruthSource.AGENT_SELF_REPORT: 0.0,
})
HARD_GROUND_TRUTH: frozenset[GroundTruthSource] = frozenset({
    GroundTruthSource.DETERMINISTIC_TESTS,
    GroundTruthSource.INTEGRATION_TESTS,
    GroundTruthSource.STRUCTURED_RESULT,
})
REPRODUCIBILITY_WEIGHTS: Mapping[str, float] = MappingProxyType({
    "git_commit_present": 0.10,
    "input_present": 0.10,
    "dependencies_reproducible": 0.10,
    "container_present": 0.10,
    "external_services_controllable": 0.10,
    "tests_present": 0.15,
    "ground_truth_present": 0.15,
    "secrets_removed": 0.05,
    "replay_successful": 0.15,
})
HARD_SCORE_THRESHOLD = 0.8
SOFT_SCORE_THRESHOLD = 0.5
CONTROLLED_NETWORK_POLICIES = frozenset({"disabled", "allowlist", "fixture"})
ELIGIBILITY_RANK: Mapping[Eligibility, int] = MappingProxyType({
    Eligibility.EXCLUDED: 0,
    Eligibility.DREAMING_ONLY: 1,
    Eligibility.SOFT_BENCHMARK: 2,
    Eligibility.HARD_BENCHMARK: 3,
})


@dataclass(frozen=True)
class ReproducibilityFactors:
    git_commit_present: bool
    input_present: bool
    dependencies_reproducible: bool
    container_present: bool
    external_services_controllable: bool
    tests_present: bool
    ground_truth_present: bool
    secrets_removed: bool
    replay_successful: bool

    def __post_init__(self) -> None:
        for item in fields(self):
            if not isinstance(getattr(self, item.name), bool):
                raise TypeError(f"reproducibility factor {item.name} must be a boolean")

    def as_dict(self) -> dict[str, bool]:
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def missing(self) -> tuple[str, ...]:
        return tuple(name for name, present in self.as_dict().items() if not present)


@dataclass(frozen=True)
class EligibilityDecision:
    eligibility: Eligibility
    reproducibility_score: float
    ground_truth: GroundTruthSource
    ground_truth_confidence: float
    reasons: tuple[str, ...]


def reproducibility_score(factors: ReproducibilityFactors) -> float:
    present = factors.as_dict()
    if set(present) != set(REPRODUCIBILITY_WEIGHTS):
        raise AssertionError("reproducibility weights do not cover every factor")
    return round(sum(weight for name, weight in REPRODUCIBILITY_WEIGHTS.items() if present[name]), 4)


def rank_ground_truth(signals: Iterable[GroundTruthSource]) -> GroundTruthSource:
    """Return the strongest available ground-truth source (self-report when none)."""
    ranked = sorted(set(signals), key=GROUND_TRUTH_PRIORITY.__getitem__)
    return ranked[0] if ranked else GroundTruthSource.AGENT_SELF_REPORT


def classify(factors: ReproducibilityFactors, ground_truth: GroundTruthSource) -> EligibilityDecision:
    """Classify evidence into HARD/SOFT/DREAMING_ONLY/EXCLUDED."""
    if factors.ground_truth_present != (ground_truth is not GroundTruthSource.AGENT_SELF_REPORT):
        raise ValueError("ground_truth_present must agree with the ranked ground-truth source")
    score = reproducibility_score(factors)
    reasons = list(factors.missing())
    if not factors.secrets_removed:
        eligibility = Eligibility.EXCLUDED
    elif (
        ground_truth in HARD_GROUND_TRUTH
        and factors.replay_successful
        and score >= HARD_SCORE_THRESHOLD
    ):
        eligibility = Eligibility.HARD_BENCHMARK
    elif ground_truth is not GroundTruthSource.AGENT_SELF_REPORT and score >= SOFT_SCORE_THRESHOLD:
        eligibility = Eligibility.SOFT_BENCHMARK
        if ground_truth not in HARD_GROUND_TRUTH:
            reasons.append(f"ground truth {ground_truth.value} cannot back a hard benchmark")
    else:
        eligibility = Eligibility.DREAMING_ONLY
        if ground_truth is GroundTruthSource.AGENT_SELF_REPORT:
            reasons.append("agent self-report is never benchmark ground truth")
    return EligibilityDecision(
        eligibility=eligibility,
        reproducibility_score=score,
        ground_truth=ground_truth,
        ground_truth_confidence=GROUND_TRUTH_CONFIDENCE[ground_truth],
        reasons=tuple(reasons),
    )


def session_ground_truth_signals(session: Mapping[str, Any]) -> tuple[GroundTruthSource, ...]:
    verification = _mapping(session.get("verification"))
    ground_truth = _mapping(session.get("ground_truth"))
    signals: list[GroundTruthSource] = []
    if _non_empty_list(verification.get("deterministic")):
        signals.append(GroundTruthSource.DETERMINISTIC_TESTS)
    if _non_empty_list(verification.get("integration")):
        signals.append(GroundTruthSource.INTEGRATION_TESTS)
    if _non_empty_mapping(verification.get("structured_result")) or _non_empty_mapping(
        ground_truth.get("structured_result")
    ):
        signals.append(GroundTruthSource.STRUCTURED_RESULT)
    if ground_truth.get("human_accepted") is True:
        signals.append(GroundTruthSource.HUMAN_ACCEPTED)
    machine_signals = verification.get("machine_signals")
    if isinstance(machine_signals, list) and len({
        item for item in machine_signals if isinstance(item, str) and item.strip()
    }) >= 2:
        signals.append(GroundTruthSource.MULTIPLE_MACHINE_SIGNALS)
    if _non_empty_mapping(ground_truth.get("llm_judge")):
        signals.append(GroundTruthSource.LLM_JUDGE)
    return tuple(signals)


def session_factors(session: Mapping[str, Any], ground_truth: GroundTruthSource) -> ReproducibilityFactors:
    task = _mapping(session.get("task"))
    environment = _mapping(session.get("environment"))
    verification = _mapping(session.get("verification"))
    security = _mapping(session.get("security"))
    return ReproducibilityFactors(
        git_commit_present=_non_empty_str(environment.get("git_commit")),
        input_present=task.get("input_available") is True or _non_empty_str(environment.get("input_data")),
        dependencies_reproducible=_non_empty_str(environment.get("dependency_digest")),
        container_present=_non_empty_str(environment.get("container_image")),
        external_services_controllable=environment.get("network_policy") in CONTROLLED_NETWORK_POLICIES,
        tests_present=(
            _non_empty_list(verification.get("deterministic"))
            or _non_empty_list(verification.get("integration"))
        ),
        ground_truth_present=ground_truth is not GroundTruthSource.AGENT_SELF_REPORT,
        secrets_removed=security.get("sanitized") is True and security.get("contains_secrets") is False,
        replay_successful=session.get("replay_verified") is True,
    )


def assess_session(session: Mapping[str, Any]) -> EligibilityDecision:
    ground_truth = rank_ground_truth(session_ground_truth_signals(session))
    return classify(session_factors(session, ground_truth), ground_truth)


def benchmark_factors(benchmark: Benchmark) -> tuple[ReproducibilityFactors, GroundTruthSource]:
    """Derive factors from an authored record; repository fixtures pin the commit at replay."""
    environment = benchmark.environment
    verification = benchmark.verification
    signals: list[GroundTruthSource] = []
    if _non_empty_list(verification.get("deterministic")):
        signals.append(GroundTruthSource.DETERMINISTIC_TESTS)
    if _non_empty_list(verification.get("integration")):
        signals.append(GroundTruthSource.INTEGRATION_TESTS)
    if _non_empty_mapping(benchmark.ground_truth.get("structured_result")):
        signals.append(GroundTruthSource.STRUCTURED_RESULT)
    if benchmark.ground_truth.get("human_accepted") is True:
        signals.append(GroundTruthSource.HUMAN_ACCEPTED)
    ground_truth = rank_ground_truth(signals)
    factors = ReproducibilityFactors(
        git_commit_present=(
            _non_empty_str(environment.get("git_commit"))
            or _non_empty_str(environment.get("repository_fixture"))
        ),
        input_present=_non_empty_str(environment.get("input_data")),
        dependencies_reproducible=_non_empty_str(environment.get("dependency_digest")),
        container_present=_non_empty_str(environment.get("container_image")),
        external_services_controllable=environment.get("network_policy") in CONTROLLED_NETWORK_POLICIES,
        tests_present=_non_empty_list(verification.get("deterministic")) or _non_empty_list(
            verification.get("integration")
        ),
        ground_truth_present=ground_truth is not GroundTruthSource.AGENT_SELF_REPORT,
        secrets_removed=(
            benchmark.security.get("sanitized") is True
            and benchmark.security.get("contains_secrets") is False
        ),
        replay_successful=benchmark.eligibility.get("replay_verified") is True,
    )
    return factors, ground_truth


def assess_benchmark(benchmark: Benchmark) -> EligibilityDecision:
    factors, ground_truth = benchmark_factors(benchmark)
    return classify(factors, ground_truth)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_empty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _non_empty_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and bool(value)


ELIGIBILITY_RULES_VERSION = "eligibility-rules-1"


def eligibility_decision_fields(
    *,
    eligibility: Eligibility,
    reproducibility_score: float,
    ground_truth_type: str,
    reasons: Iterable[str],
    subject: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    """Decision-history fields (spec §36) for one eligibility classification.

    Pure: it only describes a classification that was already made and is
    recorded after the fact; it never feeds back into ``classify``. The
    confidence is the ground-truth source's fixed confidence from
    ``GROUND_TRUTH_CONFIDENCE`` and ``None`` when no ground truth was ranked
    (pre-screen exclusions).
    """
    try:
        confidence: float | None = GROUND_TRUTH_CONFIDENCE[GroundTruthSource(ground_truth_type)]
    except ValueError:
        confidence = None
    return {
        "decision_type": "benchmark_eligibility",
        "input_features": {
            **dict(subject),
            "reproducibility_score": reproducibility_score,
            "ground_truth_type": ground_truth_type,
            "reasons": list(reasons),
        },
        "available_options": tuple(item.value for item in ELIGIBILITY_RANK),
        "selected_option": eligibility.value,
        "confidence": confidence,
        "source": source,
        "model_version": ELIGIBILITY_RULES_VERSION,
    }
