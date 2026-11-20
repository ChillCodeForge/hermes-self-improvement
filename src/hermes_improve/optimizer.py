from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any, Callable, Protocol

from .models import CandidateType
from .prompting import MAX_SKILL_CONTENT_BYTES, load_prompt, wrap_untrusted_evidence
from .self_evolution import EvolutionLimits, SelfEvolutionAdapter


MAX_VARIANTS_PER_REQUEST = 10
_MAX_RATIONALE_CHARS = 2000


@dataclass(frozen=True)
class TrainTaskFeedback:
    """Detailed per-task feedback; only produced for TRAIN split tasks."""

    task_id: str
    title: str
    prompt: str
    baseline_success: bool
    candidate_success: bool
    quality_delta: float
    candidate_failed_tool_calls: int
    candidate_output_excerpt: str
    # Share of the candidate's runs on this task that loaded the evolved skill
    # (lab finding F30); None when the runner has no agent trajectory.
    skill_loaded_rate: float | None = None


@dataclass(frozen=True)
class AggregateFeedback:
    """Limited feedback for VALIDATION/REGRESSION: counts and means, no task identities."""

    split: str
    task_count: int
    candidate_success_rate: float
    mean_quality_delta: float
    candidate_skill_loaded_rate: float | None = None


@dataclass(frozen=True)
class StageFeedback:
    """Generator-visible result of one screening stage; never contains holdout data."""

    candidate_id: str
    stage: str
    score: float
    gates_passed: bool
    gate_failures: tuple[str, ...]
    train: tuple[TrainTaskFeedback, ...]
    aggregates: tuple[AggregateFeedback, ...]
    # Share of all candidate runs of the stage that loaded the evolved skill; 0.0
    # means the agent never opened it, so the content could not have mattered.
    skill_loaded_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PriorAttempt:
    content_sha256: str
    status: str
    generation: int
    scores: dict[str, Any]


@dataclass(frozen=True)
class MutationRequest:
    target: str
    candidate_type: CandidateType
    parent_candidate_id: str
    parent_content: str
    generation: int
    count: int
    feedback: tuple[StageFeedback, ...]
    prior_attempts: tuple[PriorAttempt, ...] = ()
    # Only a single pass/fail bit; set only after the lineage's holdout generation was spent.
    holdout_passed: bool | None = None


@dataclass(frozen=True)
class VariantProposal:
    content: str
    rationale: str


class OptimizerStrategy(Protocol):
    """Generator-side mutation strategy. It proposes text only; it never evaluates or promotes."""

    name: str
    version: str

    def propose(self, request: MutationRequest) -> list[VariantProposal]: ...


class OptimizerModel(Protocol):
    def complete(self, *, role: str, system_prompt: str, user_payload: str, json_object: bool = False) -> str: ...


def mutation_evidence(request: MutationRequest) -> dict[str, Any]:
    """Generator-visible request projection shared by built-in and external strategies."""
    evidence: dict[str, Any] = {
        "target": request.target,
        "candidate_type": request.candidate_type.value,
        "generation": request.generation,
        "requested_variants": request.count,
        "parent_skill_text": request.parent_content,
        "feedback": [item.to_dict() for item in request.feedback],
        "prior_attempts": [asdict(item) for item in request.prior_attempts],
    }
    if request.holdout_passed is not None:
        evidence["finalist_holdout_passed"] = request.holdout_passed
    return evidence


class ModelMutationStrategy:
    """Built-in strategy: asks the ``qwen_optimizer`` role for skill variants."""

    name = "qwen-optimizer"
    version = "1"

    def __init__(self, model: OptimizerModel) -> None:
        if not callable(getattr(model, "complete", None)):
            raise TypeError("optimizer model must provide complete(role=, system_prompt=, user_payload=)")
        self.model = model

    def propose(self, request: MutationRequest) -> list[VariantProposal]:
        validate_mutation_request(request)
        response = self.model.complete(
            role="qwen_optimizer",
            system_prompt=load_prompt("optimizer/mutate_skill.md"),
            user_payload=wrap_untrusted_evidence(mutation_evidence(request)),
            json_object=True,
        )
        return parse_variant_response(response, request.count)


def parse_variant_response(response: str, limit: int) -> list[VariantProposal]:
    if not isinstance(response, str):
        raise ValueError("optimizer model response must be text")
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"optimizer model response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("variants"), list):
        raise ValueError("optimizer model response must be a JSON object with a variants array")
    proposals = []
    for item in value["variants"][:limit]:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        rationale = item.get("rationale", "")
        if (
            not isinstance(content, str) or not content.strip()
            or len(content.encode("utf-8")) > MAX_SKILL_CONTENT_BYTES
            or not isinstance(rationale, str)
        ):
            continue
        proposals.append(VariantProposal(content=content, rationale=rationale[:_MAX_RATIONALE_CHARS]))
    return proposals


def validate_mutation_request(request: MutationRequest) -> None:
    if not isinstance(request, MutationRequest):
        raise TypeError("strategy requires a MutationRequest")
    if (
        isinstance(request.count, bool) or not isinstance(request.count, int)
        or not 1 <= request.count <= MAX_VARIANTS_PER_REQUEST
    ):
        raise ValueError(f"variant count must be between 1 and {MAX_VARIANTS_PER_REQUEST}")
    if not isinstance(request.parent_content, str) or not request.parent_content.strip():
        raise ValueError("mutation request requires the parent skill text")


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
