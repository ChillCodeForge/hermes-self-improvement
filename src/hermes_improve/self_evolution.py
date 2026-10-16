from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EvolutionStrategy(Protocol):
    def evolve(self, candidate: str, feedback: str, generation: int) -> list[str]: ...


@dataclass(frozen=True)
class SelfEvolutionAdapter:
    """Version gate around optional external candidate evolution."""

    expected_version: str
    strategy: EvolutionStrategy

    def __post_init__(self) -> None:
        actual = getattr(self.strategy, "version", None)
        if actual != self.expected_version:
            raise RuntimeError(
                f"self-evolution adapter version mismatch: expected {self.expected_version!r}, got {actual!r}"
            )

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
