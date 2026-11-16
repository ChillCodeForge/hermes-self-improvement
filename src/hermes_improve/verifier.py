"""Deterministic verification of one benchmark run through the verifier registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import Benchmark
from .verifiers import (
    MAX_CHECKS,
    ExitCodeVerifier,
    VerificationContext,
    VerificationResult,
    VerifierRegistry,
    default_registry,
)


@dataclass(frozen=True)
class DeterministicVerification:
    passed: bool
    quality: float
    results: tuple[VerificationResult, ...]

    def summaries(self) -> list[dict[str, Any]]:
        return [result.summary() for result in self.results]

    def details(self) -> list[dict[str, Any]]:
        return [{**result.summary(), "detail": result.detail} for result in self.results]


def expected_exit_code(benchmark: Benchmark) -> int:
    expected = benchmark.ground_truth.get("expected_exit_code", 0)
    if isinstance(expected, bool) or not isinstance(expected, int):
        raise ValueError("ground_truth.expected_exit_code must be an integer")
    return expected


def has_deterministic_verification(benchmark: Benchmark) -> bool:
    checks = benchmark.verification.get("deterministic", [])
    return isinstance(checks, list) and bool(checks)


def build_verifiers(benchmark: Benchmark, registry: VerifierRegistry | None = None) -> tuple:
    """Create the benchmark's verifiers; an implicit exit-code check is always included."""
    registry = registry or default_registry()
    expected_exit = expected_exit_code(benchmark)
    verifiers = list(registry.build(benchmark.verification.get("deterministic", [])))
    exit_checks = [item for item in verifiers if isinstance(item, ExitCodeVerifier)]
    for check in exit_checks:
        if check.expected != expected_exit:
            raise ValueError("exit_code verifier must match ground_truth.expected_exit_code")
    if not exit_checks:
        if len(verifiers) >= MAX_CHECKS:
            raise ValueError(f"at most {MAX_CHECKS} effective deterministic checks are supported")
        verifiers.insert(0, ExitCodeVerifier(expected_exit))
    return tuple(verifiers)


def run_deterministic_verification(
    benchmark: Benchmark,
    context: VerificationContext,
    registry: VerifierRegistry | None = None,
) -> DeterministicVerification:
    verifiers = build_verifiers(benchmark, registry)
    results = tuple(verifier.verify(context) for verifier in verifiers)
    passed_count = sum(result.passed for result in results)
    return DeterministicVerification(
        passed=passed_count == len(results),
        quality=100.0 * passed_count / len(results),
        results=results,
    )


def verify_deterministic_output(
    benchmark: Benchmark,
    *,
    exit_code: int,
    output: str,
) -> tuple[bool, float, list[dict[str, Any]]]:
    """Run output-only checks; workspace or command checks raise a configuration error."""
    outcome = run_deterministic_verification(
        benchmark, VerificationContext(exit_code=exit_code, output=output)
    )
    return outcome.passed, outcome.quality, outcome.summaries()
