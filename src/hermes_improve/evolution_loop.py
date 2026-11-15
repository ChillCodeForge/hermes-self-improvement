"""Iterative skill evolution (spec section 25) with strict feedback filtering.

Roles stay separated inside one process:

* the :class:`OptimizerStrategy` (generator) only receives filtered
  :class:`StageFeedback` and never sees the evaluator, benchmark stores or
  holdout data;
* the :class:`EvaluationGateway` (evaluator side) owns the screening
  :class:`PairedEvaluator`, the only holdout capability and the spent-holdout
  registry, and converts reports into generator-visible feedback;
* nothing here promotes. Finalists are reported for the deterministic promotion
  policy and the separate promoter identity.

Flow per run: verify the root candidate, screen it on SMOKE and CORE, then for
each generation ask the strategy for variants, commit every new variant as an
immutable candidate, screen it on SMOKE, escalate survivors to CORE, keep the
best as the next parent, and after the search run FULL and (if configured) the
sealed holdout for finalists only.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import subprocess
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .benchmarks import BenchmarkStore
from .candidate_builder import create_skill_candidate
from .errors import AccessDenied, IntegrityError
from .evaluator import BenchmarkRunner, PairedEvaluator
from .ingestion import redact_text
from .models import Candidate, CandidateType, Eligibility, EvaluationReport, Role, Split
from .optimizer import (
    MAX_VARIANTS_PER_REQUEST,
    AggregateFeedback,
    MutationRequest,
    OptimizerStrategy,
    PriorAttempt,
    StageFeedback,
    TrainTaskFeedback,
    VariantProposal,
)
from .policy import PromotionPolicy
from .prompting import check_generated_skill_content, skill_content_digest
from .security import SpentHoldoutRegistry, consume_holdout_generation, manifest_digest
from .self_evolution import verify_strategy_version
from .skill_paths import validate_skill_target
from .store import ExperimentStore


SCREENING_STAGES = ("SMOKE", "CORE", "FULL")
_EVOLVABLE_TYPES = {CandidateType.SKILL_CREATE, CandidateType.SKILL_PATCH}
_OUTPUT_EXCERPT_CHARS = 1000
_PRIOR_ATTEMPTS = 20
# Holdout outcomes are stored for operators but never shown to the generator.
_GENERATOR_VISIBLE_STATUS = {
    "holdout_passed": "full_passed",
    "holdout_failed": "full_passed",
    "holdout_refused": "full_passed",
}


@dataclass(frozen=True)
class EvolutionConfig:
    """Loop limits; overridable from a mapping such as a protected YAML config section."""

    max_generations: int = 10
    candidates_per_generation: int = 3
    max_candidates_total: int = 30
    stop_after_no_improvement_candidates: int = 8
    stop_after_no_improvement_generations: int = 3
    minimum_meaningful_delta: float = 0.5
    max_finalists: int = 2
    smoke_seeds: tuple[int, ...] = (1,)
    core_seeds: tuple[int, ...] = (1, 2, 3)
    full_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    holdout_seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    allow_holdout_feedback: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_generations", "candidates_per_generation", "max_candidates_total",
            "stop_after_no_improvement_candidates", "stop_after_no_improvement_generations",
            "max_finalists",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"evolution setting {name} must be a positive integer")
        if self.candidates_per_generation > MAX_VARIANTS_PER_REQUEST:
            raise ValueError(
                f"evolution setting candidates_per_generation must be at most {MAX_VARIANTS_PER_REQUEST}"
            )
        delta = self.minimum_meaningful_delta
        if (
            isinstance(delta, bool) or not isinstance(delta, (int, float))
            or not math.isfinite(delta) or delta < 0
        ):
            raise ValueError("evolution setting minimum_meaningful_delta must be finite and non-negative")
        for name in ("smoke_seeds", "core_seeds", "full_seeds", "holdout_seeds"):
            seeds = getattr(self, name)
            if (
                not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds)
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
            ):
                raise ValueError(f"evolution setting {name} must be unique non-negative integers")
        if not isinstance(self.allow_holdout_feedback, bool):
            raise ValueError("evolution setting allow_holdout_feedback must be boolean")

    @classmethod
    def from_mapping(cls, overrides: Mapping[str, Any] | None = None) -> "EvolutionConfig":
        if overrides is None:
            return cls()
        if not isinstance(overrides, Mapping):
            raise ValueError("evolution overrides must be a mapping")
        known = {item.name for item in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise ValueError(f"unknown evolution settings: {', '.join(sorted(map(str, unknown)))}")
        values = dict(overrides)
        for name in ("smoke_seeds", "core_seeds", "full_seeds", "holdout_seeds"):
            if name in values:
                if not isinstance(values[name], (list, tuple)):
                    raise ValueError(f"evolution setting {name} must be a list of integers")
                values[name] = tuple(values[name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(self).items()}


@dataclass(frozen=True)
class StageResult:
    feedback: StageFeedback

    @property
    def score(self) -> float:
        return self.feedback.score

    @property
    def gates_passed(self) -> bool:
        return self.feedback.gates_passed


class EvaluationGateway:
    """Evaluator-side service: runs paired screening and holdout, returns filtered feedback."""

    def __init__(
        self,
        evaluator_store: BenchmarkStore,
        runner: BenchmarkRunner,
        policy: PromotionPolicy,
        *,
        generator_store: BenchmarkStore,
        holdout_registry: SpentHoldoutRegistry | None = None,
        experiment_store: ExperimentStore | None = None,
        evaluator_signing_key: bytes | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        if generator_store is evaluator_store:
            raise ValueError("generator and evaluator require separate benchmark views")
        if generator_store.holdout_root is not None:
            raise AccessDenied("generator benchmark view must not be configured with the sealed holdout root")
        self._generator_store = generator_store
        self._screening = PairedEvaluator(evaluator_store, runner, policy)
        self._final = (
            PairedEvaluator(
                evaluator_store, runner, policy,
                experiment_store=experiment_store,
                evaluator_signing_key=evaluator_signing_key,
                artifact_store=artifact_store,
            )
            if experiment_store is not None else self._screening
        )
        self._persist_final = experiment_store is not None
        benchmarks = evaluator_store.validate()
        self._benchmark_version = evaluator_store.version
        hard = [
            item for item in benchmarks
            if Eligibility(item.eligibility["class"]) is Eligibility.HARD_BENCHMARK
        ]
        self._suite_ids = {
            stage: [
                item.id for item in hard
                if item.split is not Split.SEALED_HOLDOUT
                and stage in {str(suite).upper() for suite in item.benchmark.get("suites", [])}
            ]
            for stage in SCREENING_STAGES
        }
        self._holdout_manifest = [item for item in benchmarks if item.split is Split.SEALED_HOLDOUT]
        self._holdout_ids = [item.id for item in hard if item.split is Split.SEALED_HOLDOUT]
        self._registry = holdout_registry
        self._grant = None
        if self._holdout_ids:
            if holdout_registry is None:
                raise ValueError("sealed holdout evaluation requires a SpentHoldoutRegistry")
            self._grant = evaluator_store.access.evaluator_holdout_grant(Role.EVALUATOR)

    @property
    def benchmark_version(self) -> str:
        return self._benchmark_version

    @property
    def holdout_available(self) -> bool:
        return bool(self._holdout_ids)

    def require_screening_suites(self) -> None:
        missing = [stage for stage in SCREENING_STAGES if not self._suite_ids[stage]]
        if missing:
            raise ValueError(
                "evolution requires HARD_BENCHMARK tasks in every screening suite; missing: "
                + ", ".join(missing)
            )

    def screen(self, candidate: Candidate, stage: str, seeds: tuple[int, ...]) -> StageResult:
        if stage not in SCREENING_STAGES:
            raise ValueError(f"unknown screening stage {stage!r}")
        report = self._screening.evaluate(
            candidate,
            baseline_commit=candidate.base_commit,
            benchmark_ids=self._suite_ids[stage],
            seeds=seeds,
            suite=stage,
            benchmark_version=self._benchmark_version,
        )
        return StageResult(self._filter(candidate, stage, report))

    def holdout_generation(self) -> str:
        if not self._holdout_manifest:
            raise ValueError("no sealed holdout benchmarks are configured")
        manifest = sorted(
            ({"id": item.id, "version": item.version, "digest": item.digest} for item in self._holdout_manifest),
            key=lambda item: (item["id"], item["version"]),
        )
        return manifest_digest({"sealed_holdout": manifest})

    def lineage_holdout_spent(self, lineage_id: str) -> bool:
        if self._registry is None or not self._holdout_manifest:
            return False
        return self._registry.is_spent(self.holdout_generation(), f"lineage:{lineage_id}")

    def spend_lineage_holdout(self, lineage_id: str) -> str:
        """Mark the holdout generation spent before its outcome may influence a mutation."""
        if self._registry is None or not self._holdout_manifest:
            raise ValueError("no sealed holdout benchmarks are configured")
        return consume_holdout_generation(self._registry, self._holdout_manifest, lineage_id)

    def evaluate_finalist_with_holdout(
        self, candidate: Candidate, *, lineage_id: str, seeds: tuple[int, ...],
    ) -> bool:
        """Run FULL + sealed holdout through the policy; only the pass/fail bit leaves."""
        if not self._holdout_ids or self._registry is None:
            raise ValueError("no sealed holdout benchmarks are configured")
        if self.lineage_holdout_spent(lineage_id):
            raise AccessDenied(
                "the sealed holdout generation was already spent for this lineage; "
                "rotate the holdout before evaluating further finalists"
            )
        consume_holdout_generation(
            self._registry, self._holdout_manifest, f"{lineage_id}/finalist/{candidate.candidate_id}",
        )
        report = self._final.evaluate(
            candidate,
            baseline_commit=candidate.base_commit,
            benchmark_ids=self._suite_ids["FULL"] + self._holdout_ids,
            seeds=seeds,
            suite="FULL",
            benchmark_version=self._benchmark_version,
            holdout_grant=self._grant,
            evaluation_id=(
                f"eval-{candidate.candidate_id}-{secrets.token_hex(8)}" if self._persist_final else None
            ),
        )
        return bool(report.passed)

    def _filter(self, candidate: Candidate, stage: str, report: EvaluationReport) -> StageFeedback:
        rows = report.outcomes
        if not rows:
            raise IntegrityError("screening evaluation returned no task outcomes")
        if any(row.split is Split.SEALED_HOLDOUT for row in rows):
            raise IntegrityError("screening feedback must never contain sealed holdout outcomes")
        failures = []
        if any(not row.security_pass for row in rows):
            failures.append("security verifier failed")
        if any(not row.deterministic_pass for row in rows):
            failures.append("deterministic verifier failed")
        if any(row.critical_regressions for row in rows):
            failures.append("critical regression detected")
        if any(row.high_regressions for row in rows):
            failures.append("high regression detected")
        if sum(row.candidate_success for row in rows) < sum(row.baseline_success for row in rows):
            failures.append("candidate solved fewer tasks than the production baseline")
        train = []
        for row in sorted(rows, key=lambda item: item.task_id):
            if row.split is not Split.TRAIN:
                continue
            task = self._generator_store.get(row.task_id, role=Role.GENERATOR)
            outputs = [
                observation.final_output for observation in report.run_observations
                if observation.task_id == row.task_id and observation.commit == candidate.commit
            ]
            train.append(TrainTaskFeedback(
                task_id=task.id,
                title=str(task.task.get("title", ""))[:200],
                prompt=str(task.task.get("prompt", ""))[:4000],
                baseline_success=row.baseline_success,
                candidate_success=row.candidate_success,
                quality_delta=row.candidate_quality - row.baseline_quality,
                candidate_failed_tool_calls=row.candidate_failed_tool_calls,
                candidate_output_excerpt=redact_text(outputs[0])[:_OUTPUT_EXCERPT_CHARS] if outputs else "",
            ))
        aggregates = []
        for split in sorted({row.split for row in rows if row.split is not Split.TRAIN}):
            members = [row for row in rows if row.split is split]
            aggregates.append(AggregateFeedback(
                split=split.value,
                task_count=len(members),
                candidate_success_rate=sum(row.candidate_success for row in members) / len(members),
                mean_quality_delta=mean(row.candidate_quality - row.baseline_quality for row in members),
            ))
        return StageFeedback(
            candidate_id=candidate.candidate_id,
            stage=stage,
            score=mean(row.candidate_quality - row.baseline_quality for row in rows),
            gates_passed=not failures,
            gate_failures=tuple(failures),
            train=tuple(train),
            aggregates=tuple(aggregates),
        )


@dataclass(frozen=True)
class GenerationReport:
    generation: int
    parent_candidate_id: str
    proposed: int
    duplicates: int
    rejected: int
    created: int
    smoke_passed: int
    core_evaluated: int
    improved: int
    best_candidate_id: str
    best_core_score: float


@dataclass(frozen=True)
class FinalistReport:
    candidate_id: str
    experiment_id: str
    core_score: float
    full_score: float
    full_passed: bool
    holdout: str


@dataclass(frozen=True)
class EvolutionRunReport:
    run_id: str
    root_candidate_id: str
    lineage_id: str
    strategy: str
    strategy_version: str
    stop_reason: str
    root_core_score: float
    best_candidate_id: str
    best_core_score: float
    candidates_created: int
    holdout_feedback_used: bool
    generations: tuple[GenerationReport, ...]
    finalists: tuple[FinalistReport, ...]
    promotion: str = "not performed: the loop only proposes; promotion requires the policy and promoter"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Variant:
    candidate: Candidate
    content: str
    digest: str
    smoke_score: float
    core_score: float
    core_feedback: StageFeedback


@dataclass
class _RunState:
    run_id: str
    root: Candidate
    lineage_id: str
    source_sessions: tuple[str, ...]
    parent: _Variant
    best: _Variant
    feedback: list[StageFeedback]
    generation: int = 0
    created: int = 0
    no_improvement_candidates: int = 0
    no_improvement_generations: int = 0
    improved: list[_Variant] = field(default_factory=list)
    finalized: set[str] = field(default_factory=set)
    generations: list[GenerationReport] = field(default_factory=list)
    finalists: list[FinalistReport] = field(default_factory=list)


class EvolutionLoop:
    def __init__(
        self,
        *,
        repository: Path,
        store: ExperimentStore,
        gateway: EvaluationGateway,
        strategy: OptimizerStrategy,
        expected_strategy_version: str,
        config: EvolutionConfig | None = None,
    ) -> None:
        verify_strategy_version(strategy, expected_strategy_version)
        name = getattr(strategy, "name", None)
        if not isinstance(name, str) or not name.strip() or not callable(getattr(strategy, "propose", None)):
            raise TypeError("optimizer strategy must define name, version and propose()")
        self.repository = Path(repository).resolve()
        self.store = store
        self.gateway = gateway
        self.strategy = strategy
        self.config = config or EvolutionConfig()

    def run(self, root_candidate_id: str) -> EvolutionRunReport:
        root = load_verified_candidate(self.repository, self.store, root_candidate_id)
        if root.candidate_type not in _EVOLVABLE_TYPES:
            raise ValueError("evolution currently supports SKILL_CREATE and SKILL_PATCH candidates")
        validate_skill_target(root.target)
        self.gateway.require_screening_suites()
        experiment = self.store.show_experiment(root.experiment_id)
        run_id = f"evo-{secrets.token_hex(6)}"
        self.store.create_evolution_run(
            run_id,
            root_candidate_id=root.candidate_id,
            lineage_id=experiment["lineage_id"],
            strategy=self.strategy.name,
            strategy_version=self.strategy.version,
            config=self.config.to_dict(),
        )
        try:
            report = self._run(run_id, root, experiment)
        except BaseException as exc:
            self.store.finish_evolution_run(
                run_id, status="failed", stop_reason=f"{type(exc).__name__}: {exc}"[:500], report={},
            )
            raise
        self.store.finish_evolution_run(
            run_id, status="completed", stop_reason=report.stop_reason, report=report.to_dict(),
        )
        return report

    def _run(self, run_id: str, root: Candidate, experiment: Mapping[str, Any]) -> EvolutionRunReport:
        content = read_skill_content(self.repository, root.commit, root.target)
        digest = skill_content_digest(content)
        if self.store.find_generated_variant(root.target, digest) is None:
            self.store.record_generated_variant(
                target=root.target, content_sha256=digest, origin=f"evolution:{run_id}",
                generation=0, status="root", candidate_id=root.candidate_id,
            )
        smoke = self.gateway.screen(root, "SMOKE", self.config.smoke_seeds)
        core = self.gateway.screen(root, "CORE", self.config.core_seeds)
        root_variant = _Variant(root, content, digest, smoke.score, core.score, core.feedback)
        state = _RunState(
            run_id=run_id,
            root=root,
            lineage_id=experiment["lineage_id"],
            source_sessions=tuple(experiment.get("source_sessions", ())),
            parent=root_variant,
            best=root_variant,
            feedback=[smoke.feedback, core.feedback],
        )
        stop_reason = self._search(state, holdout_passed=None)
        self._finalize(state)
        holdout_feedback_used = False
        holdout_bits = [item.holdout for item in state.finalists]
        if (
            self.config.allow_holdout_feedback
            and "failed" in holdout_bits and "passed" not in holdout_bits
            and state.generation < self.config.max_generations
            and state.created < self.config.max_candidates_total
        ):
            try:
                self.gateway.spend_lineage_holdout(state.lineage_id)
            except AccessDenied:
                stop_reason = "holdout_generation_spent"
            else:
                holdout_feedback_used = True
                state.no_improvement_candidates = 0
                state.no_improvement_generations = 0
                stop_reason = self._search(state, holdout_passed=False)
                self._finalize(state)
        return EvolutionRunReport(
            run_id=run_id,
            root_candidate_id=root.candidate_id,
            lineage_id=state.lineage_id,
            strategy=self.strategy.name,
            strategy_version=self.strategy.version,
            stop_reason=stop_reason,
            root_core_score=root_variant.core_score,
            best_candidate_id=state.best.candidate.candidate_id,
            best_core_score=state.best.core_score,
            candidates_created=state.created,
            holdout_feedback_used=holdout_feedback_used,
            generations=tuple(state.generations),
            finalists=tuple(state.finalists),
        )

    def _search(self, state: _RunState, *, holdout_passed: bool | None) -> str:
        config = self.config
        while state.generation < config.max_generations:
            remaining = config.max_candidates_total - state.created
            if remaining <= 0:
                return "max_candidates_total"
            state.generation += 1
            parent = state.parent
            count = min(config.candidates_per_generation, remaining)
            proposals = self.strategy.propose(MutationRequest(
                target=state.root.target,
                candidate_type=state.root.candidate_type,
                parent_candidate_id=parent.candidate.candidate_id,
                parent_content=parent.content,
                generation=state.generation,
                count=count,
                feedback=tuple(state.feedback),
                prior_attempts=self._prior_attempts(state.root.target),
                holdout_passed=holdout_passed,
            ))
            if not isinstance(proposals, list) or not all(isinstance(item, VariantProposal) for item in proposals):
                raise ValueError("optimizer strategy must return a list of VariantProposal objects")
            counters = {key: 0 for key in (
                "proposed", "duplicates", "rejected", "created", "smoke_passed", "core_evaluated", "improved",
            )}
            generation_feedback: list[StageFeedback] = []
            for index, proposal in enumerate(proposals[:count]):
                counters["proposed"] += 1
                self._try_variant(state, parent, proposal, index, counters, generation_feedback)
            state.generations.append(GenerationReport(
                generation=state.generation,
                parent_candidate_id=parent.candidate.candidate_id,
                best_candidate_id=state.best.candidate.candidate_id,
                best_core_score=state.best.core_score,
                **counters,
            ))
            if not proposals:
                return "optimizer_exhausted"
            if generation_feedback:
                # Next prompt: the best candidate's CORE feedback plus everything measured this
                # generation (failures included), so the strategy learns from rejected variants too.
                best_feedback = state.best.core_feedback
                state.feedback = [best_feedback] + [
                    item for item in generation_feedback if item is not best_feedback
                ]
            state.parent = state.best
            if counters["improved"]:
                state.no_improvement_generations = 0
            else:
                state.no_improvement_generations += 1
            if state.no_improvement_candidates >= config.stop_after_no_improvement_candidates:
                return "no_improvement_candidates"
            if state.no_improvement_generations >= config.stop_after_no_improvement_generations:
                return "no_improvement_generations"
            if state.created >= config.max_candidates_total:
                return "max_candidates_total"
        return "max_generations"

    def _try_variant(
        self, state: _RunState, parent: _Variant, proposal: VariantProposal, index: int,
        counters: dict[str, int], generation_feedback: list[StageFeedback],
    ) -> None:
        target = state.root.target
        digest = skill_content_digest(proposal.content)
        if self.store.find_generated_variant(target, digest) is not None:
            counters["duplicates"] += 1
            state.no_improvement_candidates += 1
            return
        experiment_id = f"{state.run_id}-g{state.generation:02d}-v{index}"
        candidate_id = f"{experiment_id}-c"
        origin = f"evolution:{state.run_id}"
        try:
            check_generated_skill_content(proposal.content)
            candidate = create_skill_candidate(
                self.repository,
                self.store,
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                candidate_type=state.root.candidate_type,
                target=target,
                content=proposal.content,
                hypothesis=(
                    f"Evolution {state.run_id} generation {state.generation} variant {index} "
                    f"of {parent.candidate.candidate_id}: {' '.join(proposal.rationale.split())}"
                )[:4000],
                benchmark_version=self.gateway.benchmark_version,
                source_sessions=state.source_sessions,
            )
        except ValueError as exc:
            self.store.record_generated_variant(
                target=target, content_sha256=digest, origin=origin, generation=state.generation,
                status="rejected", parent_candidate_id=parent.candidate.candidate_id, reason=str(exc),
            )
            counters["rejected"] += 1
            state.no_improvement_candidates += 1
            return
        self.store.link_experiment_lineage(
            experiment_id, parent_id=parent.candidate.experiment_id, lineage_id=state.lineage_id,
        )
        self.store.record_generated_variant(
            target=target, content_sha256=digest, origin=origin, generation=state.generation,
            status="created", parent_candidate_id=parent.candidate.candidate_id,
            candidate_id=candidate.candidate_id,
        )
        state.created += 1
        counters["created"] += 1
        smoke = self.gateway.screen(candidate, "SMOKE", self.config.smoke_seeds)
        generation_feedback.append(smoke.feedback)
        if not smoke.gates_passed or smoke.score < parent.smoke_score:
            self.store.update_generated_variant(
                target, digest, status="smoke_rejected",
                reason="; ".join(smoke.feedback.gate_failures) or "smoke score below parent",
                scores={"smoke": smoke.score},
            )
            state.no_improvement_candidates += 1
            return
        counters["smoke_passed"] += 1
        core = self.gateway.screen(candidate, "CORE", self.config.core_seeds)
        counters["core_evaluated"] += 1
        generation_feedback.append(core.feedback)
        scores = {"smoke": smoke.score, "core": core.score}
        if not core.gates_passed:
            self.store.update_generated_variant(
                target, digest, status="core_rejected",
                reason="; ".join(core.feedback.gate_failures), scores=scores,
            )
            state.no_improvement_candidates += 1
            return
        if core.score - state.best.core_score < self.config.minimum_meaningful_delta:
            self.store.update_generated_variant(
                target, digest, status="core_no_improvement",
                reason="core score did not beat the best by the minimum meaningful delta", scores=scores,
            )
            state.no_improvement_candidates += 1
            return
        variant = _Variant(candidate, proposal.content, digest, smoke.score, core.score, core.feedback)
        self.store.update_generated_variant(target, digest, status="improved", scores=scores)
        state.best = variant
        state.improved.append(variant)
        state.no_improvement_candidates = 0
        counters["improved"] += 1

    def _finalize(self, state: _RunState) -> None:
        pending = [item for item in state.improved if item.candidate.candidate_id not in state.finalized]
        ranked = sorted(pending, key=lambda item: item.core_score, reverse=True)[: self.config.max_finalists]
        for variant in ranked:
            state.finalized.add(variant.candidate.candidate_id)
            target = state.root.target
            full = self.gateway.screen(variant.candidate, "FULL", self.config.full_seeds)
            state.feedback.append(full.feedback)
            full_passed = full.gates_passed and full.score >= self.config.minimum_meaningful_delta
            scores = {"smoke": variant.smoke_score, "core": variant.core_score, "full": full.score}
            holdout = "not_run"
            if not full_passed:
                self.store.update_generated_variant(
                    target, variant.digest, status="full_rejected",
                    reason="; ".join(full.feedback.gate_failures) or "full-suite gain below minimum", scores=scores,
                )
            elif not self.gateway.holdout_available:
                holdout = "awaiting_evaluator"
                self.store.update_generated_variant(target, variant.digest, status="full_passed", scores=scores)
            else:
                try:
                    passed = self.gateway.evaluate_finalist_with_holdout(
                        variant.candidate, lineage_id=state.lineage_id, seeds=self.config.holdout_seeds,
                    )
                except AccessDenied as exc:
                    holdout = "refused_spent"
                    self.store.update_generated_variant(
                        target, variant.digest, status="holdout_refused", reason=str(exc), scores=scores,
                    )
                else:
                    holdout = "passed" if passed else "failed"
                    self.store.update_generated_variant(
                        target, variant.digest, status="holdout_passed" if passed else "holdout_failed",
                        scores=scores,
                    )
            state.finalists.append(FinalistReport(
                candidate_id=variant.candidate.candidate_id,
                experiment_id=variant.candidate.experiment_id,
                core_score=variant.core_score,
                full_score=full.score,
                full_passed=full_passed,
                holdout=holdout,
            ))

    def _prior_attempts(self, target: str) -> tuple[PriorAttempt, ...]:
        return tuple(
            PriorAttempt(
                content_sha256=row["content_sha256"],
                status=_GENERATOR_VISIBLE_STATUS.get(row["status"], row["status"]),
                generation=row["generation"],
                scores={key: row["scores"][key] for key in ("smoke", "core", "full") if key in row["scores"]},
            )
            for row in self.store.list_generated_variants(target, _PRIOR_ATTEMPTS)
        )


def load_verified_candidate(repository: Path, store: ExperimentStore, candidate_id: str) -> Candidate:
    """Rebuild a stored candidate from Git and verify its immutable diff and content hash."""
    stored = store.get_candidate(candidate_id)
    diff = git_output(
        Path(repository), "diff", "--no-ext-diff", "--no-textconv", "--binary",
        stored["base_commit"], stored["commit_id"],
    )
    if hashlib.sha256(diff).hexdigest() != stored["diff_hash"]:
        raise IntegrityError("Git candidate diff does not match its stored immutable hash")
    candidate = Candidate(
        stored["id"], stored["experiment_id"], CandidateType(stored["candidate_type"]),
        stored["target"], stored["base_commit"], stored["commit_id"], diff.decode("utf-8"),
        stored["created_at"],
    )
    if candidate.content_hash != stored["content_hash"]:
        raise IntegrityError("candidate metadata differs from its stored immutable content hash")
    return candidate


def read_skill_content(repository: Path, commit: str, target: str) -> str:
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit):
        raise ValueError("candidate commit must be a full Git commit ID")
    path = validate_skill_target(target) / "SKILL.md"
    return git_output(Path(repository), "show", f"{commit}:{path.as_posix()}").decode("utf-8")


def git_output(repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if result.returncode:
        raise RuntimeError(f"Git {arguments[0]} failed: {result.stderr.decode(errors='replace')}")
    return result.stdout
