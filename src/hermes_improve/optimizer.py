from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from .self_evolution import EvolutionLimits, SelfEvolutionAdapter


@dataclass(frozen=True)
class EvolutionResult:
    best_candidate: str
    best_score: float
    generations: tuple[tuple[str, ...], ...]
    evaluated: int
    stop_reason: str


def run_evolution(
    adapter: SelfEvolutionAdapter,
    baseline: str,
    baseline_score: float,
    *,
    evaluate: Callable[[str], tuple[float, str]],
    limits: EvolutionLimits = EvolutionLimits(),
) -> EvolutionResult:
    """Iteratively optimize against the caller's non-holdout evaluation feedback."""
    if min(
        limits.max_generations, limits.candidates_per_generation, limits.max_candidates_total,
        limits.no_improvement_candidates, limits.no_improvement_generations,
    ) < 1 or not math.isfinite(limits.minimum_meaningful_delta) or limits.minimum_meaningful_delta < 0:
        raise ValueError("evolution limits must be positive")
    if not math.isfinite(baseline_score):
        raise ValueError("baseline score must be finite")
    best = baseline
    best_score = baseline_score
    feedback = "Create candidates that measurably improve task success without regression."
    history: list[tuple[str, ...]] = []
    evaluated = 0
    no_improvement_candidates = 0
    no_improvement_generations = 0
    for generation in range(limits.max_generations):
        remaining = limits.max_candidates_total - evaluated
        if remaining <= 0:
            return EvolutionResult(best, best_score, tuple(history), evaluated, "max_candidates_total")
        variants = adapter.evolve(best, feedback, generation)[: min(limits.candidates_per_generation, remaining)]
        if not variants:
            return EvolutionResult(best, best_score, tuple(history), evaluated, "optimizer_exhausted")
        generation_candidates = []
        generation_improved = False
        for variant in variants:
            score, candidate_feedback = evaluate(variant)
            if (
                isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not isinstance(candidate_feedback, str)
            ):
                raise ValueError("evaluation callback must return a numeric score and text feedback")
            evaluated += 1
            generation_candidates.append(variant)
            delta = float(score) - best_score
            if delta >= limits.minimum_meaningful_delta:
                best = variant
                best_score = float(score)
                feedback = candidate_feedback
                generation_improved = True
                no_improvement_candidates = 0
            else:
                no_improvement_candidates += 1
        history.append(tuple(generation_candidates))
        no_improvement_generations = 0 if generation_improved else no_improvement_generations + 1
        if no_improvement_candidates >= limits.no_improvement_candidates:
            return EvolutionResult(best, best_score, tuple(history), evaluated, "no_improvement_candidates")
        if no_improvement_generations >= limits.no_improvement_generations:
            return EvolutionResult(best, best_score, tuple(history), evaluated, "no_improvement_generations")
        if evaluated >= limits.max_candidates_total:
            return EvolutionResult(best, best_score, tuple(history), evaluated, "max_candidates_total")
    return EvolutionResult(best, best_score, tuple(history), evaluated, "max_generations")
