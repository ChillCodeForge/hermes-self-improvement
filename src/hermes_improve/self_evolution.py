from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EvolutionStrategy(Protocol):
    def evolve(self, candidate: str, feedback: str, generation: int) -> list[str]: ...


def verify_strategy_version(strategy: object, expected_version: str) -> None:
    """Refuse any external or built-in strategy whose version is not the pinned, tested one."""
    if not isinstance(expected_version, str) or not expected_version.strip():
        raise ValueError("a pinned strategy version is required")
    actual = getattr(strategy, "version", None)
    if actual != expected_version:
        raise RuntimeError(
            f"self-evolution adapter version mismatch: expected {expected_version!r}, got {actual!r}"
        )


@dataclass(frozen=True)
class SelfEvolutionAdapter:
    """Version gate around optional external candidate evolution."""

    expected_version: str
    strategy: EvolutionStrategy

    def __post_init__(self) -> None:
        verify_strategy_version(self.strategy, self.expected_version)

    def evolve(self, candidate: str, feedback: str, generation: int) -> list[str]:
        if generation < 0:
            raise ValueError("generation must be non-negative")
        variants = self.strategy.evolve(candidate, feedback, generation)
        if not isinstance(variants, list) or not all(isinstance(item, str) for item in variants):
            raise ValueError("evolution strategy must return a list of candidate strings")
        return variants


@dataclass(frozen=True)
class EvolutionLimits:
    max_generations: int = 10
    candidates_per_generation: int = 3
    max_candidates_total: int = 30
    no_improvement_candidates: int = 8
    no_improvement_generations: int = 3
    minimum_meaningful_delta: float = 0.5


def evolve_candidates(
    adapter: SelfEvolutionAdapter,
    initial: str,
    feedback: str,
    limits: EvolutionLimits = EvolutionLimits(),
) -> list[list[str]]:
    generations: list[list[str]] = []
    total = 0
    current = initial
    for generation in range(limits.max_generations):
        variants = adapter.evolve(current, feedback, generation)[:limits.candidates_per_generation]
        if not variants:
            break
        remaining = limits.max_candidates_total - total
        variants = variants[:remaining]
        if not variants:
            break
        generations.append(variants)
        total += len(variants)
        current = variants[0]
        if total >= limits.max_candidates_total:
            break
    return generations
