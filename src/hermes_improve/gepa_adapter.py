"""Optional GEPA strategies (``pip install '.[gepa]'``, pinned ``gepa==0.1.4``).

* :class:`GepaOptimizeStrategy` runs ``gepa.optimize`` through a project
  ``GEPAAdapter`` whose metric is this project's paired evaluator (CORE
  screening on TRAIN tasks plus a VALIDATION aggregate) and whose reflection LM
  is the ``qwen_optimizer`` role.
* :class:`GepaStrategy` reuses only GEPA's reflective instruction proposer for
  one rewrite per requested variant.

In both, the sealed holdout, final selection and promotion stay outside GEPA,
and any GEPA version other than the pinned, tested one is refused.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any, Callable, Protocol

from .optimizer import (
    AggregateFeedback,
    MutationRequest,
    OptimizerModel,
    StageFeedback,
    TrainTaskFeedback,
    VariantProposal,
    validate_mutation_request,
)
from .prompting import (
    MAX_SKILL_CONTENT_BYTES,
    check_payload_size,
    escape_untrusted_json,
    load_prompt,
    new_envelope_nonce,
)


SUPPORTED_GEPA_VERSION = "0.1.4"
_SIGNATURE_MODULE = "gepa.strategies.instruction_proposal"
_PLACEHOLDERS = ("<curr_param>", "<side_info>")


class GepaUnavailableError(RuntimeError):
    """GEPA is not installed in this environment."""


class GepaVersionError(RuntimeError):
    """Installed GEPA differs from the pinned, tested version."""


class GepaStrategy:
    """OptimizerStrategy that delegates one reflective rewrite per variant to GEPA."""

    name = "gepa-reflective-mutation"

    def __init__(
        self,
        model: OptimizerModel,
        *,
        signature: Any | None = None,
        installed_version: str | None = None,
    ) -> None:
        _require_model(model)
        installed_version = _pinned_gepa_version(installed_version, injected=signature is not None)
        if signature is None:
            signature = importlib.import_module(_SIGNATURE_MODULE).InstructionProposalSignature
        if not callable(getattr(signature, "run", None)):
            raise TypeError("GEPA instruction proposal signature must provide run(lm, input_dict)")
        self.model = model
        self.signature = signature
        self.version = f"gepa-{installed_version}"

    def propose(self, request: MutationRequest) -> list[VariantProposal]:
        validate_mutation_request(request)
        _check_parent_text(request.parent_content)
        records = _feedback_records(request)
        if not records:
            raise ValueError("GEPA reflective mutation requires screening feedback")
        system_prompt = load_prompt("optimizer/gepa_system.md")
        template = load_prompt("optimizer/gepa_reflection.md")
        proposals = []
        for index in range(request.count):
            nonce = new_envelope_nonce()
            rotated = records[index % len(records):] + records[: index % len(records)]

            def reflection_lm(prompt: Any, *, _nonce: str = nonce) -> str:
                if not isinstance(prompt, str):
                    raise ValueError("multimodal GEPA prompts are not supported")
                if prompt.count(_nonce) != 2:
                    raise ValueError("GEPA prompt lost the untrusted evidence envelope")
                check_payload_size(prompt)
                return self.model.complete(
                    role="qwen_optimizer", system_prompt=system_prompt, user_payload=prompt,
                )

            result = self.signature.run(reflection_lm, {
                "current_instruction_doc": request.parent_content,
                "dataset_with_feedback": rotated,
                "prompt_template": template.replace("<nonce>", nonce),
            })
            text = result.get("new_instruction") if isinstance(result, dict) else None
            if (
                not isinstance(text, str) or not text.strip()
                or len(text.encode("utf-8")) > MAX_SKILL_CONTENT_BYTES
            ):
                continue
            proposals.append(VariantProposal(
                content=text, rationale=f"GEPA {self.version} reflective mutation over feedback rotation {index}",
            ))
        return proposals


def _feedback_records(request: MutationRequest) -> list[dict[str, str]]:
    """GEPA ``dataset_with_feedback`` rows; every value is escaped JSON, never raw evidence."""
    records = []
    for stage in request.feedback:
        for task in stage.train:
            records.append(_train_record(stage.stage, task, stage.gate_failures))
        if stage.aggregates:
            records.append(_aggregate_record(stage.stage, stage.aggregates))
    return records


_MISSING_SCORE = -1000.0


class ScreeningOracleProtocol(Protocol):
    def evaluate_content(self, content: str) -> StageFeedback | None: ...


class GepaOptimizeStrategy:
    """Runs ``gepa.optimize`` with this project's evaluator as the metric.

    GEPA's inner search sees TRAIN tasks (per-task scores and filtered detail)
    and, when present, one VALIDATION aggregate instance; it never sees holdout.
    Every text GEPA wants scored becomes an immutable candidate through the
    loop's :class:`~hermes_improve.evolution_loop.ScreeningOracle`, which runs
    the paired CORE screening. The best GEPA candidates are returned as
    proposals and still pass the loop's SMOKE gate, improvement threshold, FULL
    and holdout; GEPA never decides promotion. ``qwen_optimizer`` is the
    reflection LM.
    """

    name = "gepa-optimize"

    def __init__(
        self,
        model: OptimizerModel,
        *,
        max_metric_calls: int = 24,
        optimize: Callable[..., Any] | None = None,
        evaluation_batch: Callable[..., Any] | None = None,
        installed_version: str | None = None,
    ) -> None:
        _require_model(model)
        injected = optimize is not None or evaluation_batch is not None
        if injected and (optimize is None or evaluation_batch is None):
            raise ValueError("inject both optimize and evaluation_batch, or neither")
        installed_version = _pinned_gepa_version(installed_version, injected=injected)
        if not injected:
            optimize = importlib.import_module("gepa").optimize
            evaluation_batch = importlib.import_module("gepa.core.adapter").EvaluationBatch
        if (
            isinstance(max_metric_calls, bool) or not isinstance(max_metric_calls, int)
            or not 1 <= max_metric_calls <= 1000
        ):
            raise ValueError("GEPA max_metric_calls must be between 1 and 1000")
        self.model = model
        self.max_metric_calls = max_metric_calls
        self._optimize = optimize
        self._evaluation_batch = evaluation_batch
        self._oracle: ScreeningOracleProtocol | None = None
        self.version = f"gepa-{installed_version}"

    def bind_screening_oracle(self, oracle: ScreeningOracleProtocol) -> None:
        if not callable(getattr(oracle, "evaluate_content", None)):
            raise TypeError("screening oracle must provide evaluate_content(content)")
        self._oracle = oracle

    def propose(self, request: MutationRequest) -> list[VariantProposal]:
        validate_mutation_request(request)
        if self._oracle is None:
            raise RuntimeError("GEPA optimization requires the evolution loop's screening oracle")
        _check_parent_text(request.parent_content)
        task_ids = sorted({task.task_id for stage in request.feedback for task in stage.train})
        if not task_ids:
            raise ValueError("GEPA optimization requires TRAIN feedback")
        trainset = [{"kind": "train", "task_id": task_id} for task_id in task_ids]
        has_validation = any(
            item.split == "VALIDATION" for stage in request.feedback for item in stage.aggregates
        )
        valset = [{"kind": "validation_aggregate"}] if has_validation else list(trainset)
        nonce = new_envelope_nonce()
        system_prompt = load_prompt("optimizer/gepa_system.md")
        template = load_prompt("optimizer/gepa_reflection.md").replace("<nonce>", nonce)
        result = self._optimize(
            seed_candidate={"skill": request.parent_content},
            trainset=trainset,
            valset=valset,
            adapter=_HermesGepaAdapter(self._oracle, self._evaluation_batch),
            reflection_lm=_reflection_lm(self.model, system_prompt, nonce),
            reflection_prompt_template=template,
            max_metric_calls=self.max_metric_calls,
            reflection_minibatch_size=min(3, len(trainset)),
            skip_perfect_score=False,
            seed=request.generation,
            raise_on_exception=True,
            display_progress_bar=False,
        )
        ranked = sorted(
            zip(result.candidates, result.val_aggregate_scores), key=lambda item: item[1], reverse=True,
        )
        proposals: list[VariantProposal] = []
        seen = {request.parent_content}
        for candidate, score in ranked:
            text = candidate.get("skill") if isinstance(candidate, dict) else None
            if not isinstance(text, str) or text in seen or not text.strip():
                continue
            if len(text.encode("utf-8")) > MAX_SKILL_CONTENT_BYTES:
                continue
            seen.add(text)
            proposals.append(VariantProposal(
                content=text,
                rationale=f"GEPA {self.version} optimize candidate; inner validation score {float(score):.3f}",
            ))
            if len(proposals) == request.count:
                break
        return proposals


class _HermesGepaAdapter:
    """GEPAAdapter implementation whose metric is the loop's filtered CORE screening."""

    propose_new_texts = None

    def __init__(self, oracle: ScreeningOracleProtocol, evaluation_batch: Callable[..., Any]) -> None:
        self._oracle = oracle
        self._evaluation_batch = evaluation_batch

    def evaluate(self, batch: list[dict[str, str]], candidate: dict[str, str], capture_traces: bool = False) -> Any:
        feedback = self._oracle.evaluate_content(candidate["skill"])
        outputs, scores, traces = [], [], []
        train = {task.task_id: task for task in feedback.train} if feedback is not None else {}
        validation = (
            next((item for item in feedback.aggregates if item.split == "VALIDATION"), None)
            if feedback is not None else None
        )
        for instance in batch:
            if feedback is None:
                score, output, record = _MISSING_SCORE, "", _unscored_record(
                    "not scored: duplicate of an earlier experiment, rejected by content checks, or over budget"
                )
            elif not feedback.gates_passed:
                score, output, record = _MISSING_SCORE, "", _unscored_record(
                    "hard gate failed: " + "; ".join(feedback.gate_failures)
                )
            elif instance.get("kind") == "train" and instance.get("task_id") in train:
                task = train[instance["task_id"]]
                score, output = task.quality_delta, task.candidate_output_excerpt
                record = _train_record(feedback.stage, task, feedback.gate_failures)
            elif instance.get("kind") == "validation_aggregate" and validation is not None:
                score, output = validation.mean_quality_delta, ""
                record = _aggregate_record(feedback.stage, feedback.aggregates)
            else:
                score, output, record = _MISSING_SCORE, "", _unscored_record("task missing from screening feedback")
            outputs.append(output)
            scores.append(float(score))
            traces.append({"record": record})
        return self._evaluation_batch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None,
        )

    def make_reflective_dataset(
        self, candidate: dict[str, str], eval_batch: Any, components_to_update: list[str],
    ) -> dict[str, list[dict[str, str]]]:
        records = [trace["record"] for trace in (eval_batch.trajectories or [])]
        return {name: records for name in components_to_update}


def _reflection_lm(model: OptimizerModel, system_prompt: str, nonce: str) -> Callable[[Any], str]:
    def reflection_lm(prompt: Any) -> str:
        if not isinstance(prompt, str):
            raise ValueError("multimodal GEPA prompts are not supported")
        if prompt.count(nonce) != 2:
            raise ValueError("GEPA prompt lost the untrusted evidence envelope")
        check_payload_size(prompt)
        return model.complete(role="qwen_optimizer", system_prompt=system_prompt, user_payload=prompt)

    return reflection_lm


def _require_model(model: Any) -> None:
    if not callable(getattr(model, "complete", None)):
        raise TypeError("GEPA reflection model must provide complete(role=, system_prompt=, user_payload=)")


def _pinned_gepa_version(installed_version: str | None, *, injected: bool) -> str:
    if not injected:
        if installed_version is not None:
            raise ValueError("installed_version may only accompany injected GEPA components")
        try:
            installed_version = importlib.metadata.version("gepa")
        except importlib.metadata.PackageNotFoundError as exc:
            raise GepaUnavailableError(
                "GEPA is not installed; install the optional extra with pip install '.[gepa]' "
                f"(pinned gepa=={SUPPORTED_GEPA_VERSION})"
            ) from exc
    if installed_version != SUPPORTED_GEPA_VERSION:
        raise GepaVersionError(
            f"GEPA version {installed_version!r} is not supported; this adapter is pinned and "
            f"tested against gepa=={SUPPORTED_GEPA_VERSION}"
        )
    return installed_version


def _check_parent_text(text: str) -> None:
    if any(token in text for token in _PLACEHOLDERS) or "<<<UNTRUSTED" in text:
        raise ValueError("parent skill text contains GEPA template placeholders or evidence delimiters")


def _train_record(stage: str, task: TrainTaskFeedback, gate_failures: tuple[str, ...]) -> dict[str, str]:
    return {
        "Inputs": escape_untrusted_json({"title": task.title, "prompt": task.prompt}),
        "Generated Outputs": escape_untrusted_json(task.candidate_output_excerpt),
        "Feedback": escape_untrusted_json({
            "stage": stage,
            "baseline_success": task.baseline_success,
            "candidate_success": task.candidate_success,
            "quality_delta": task.quality_delta,
            "failed_tool_calls": task.candidate_failed_tool_calls,
            # 0.0: the agent never opened the skill on this task (F30); null: unknown.
            "skill_loaded_rate": task.skill_loaded_rate,
            "gate_failures": list(gate_failures),
        }),
    }


def _aggregate_record(stage: str, aggregates: tuple[AggregateFeedback, ...]) -> dict[str, str]:
    return {
        "Inputs": escape_untrusted_json({"stage": stage, "held_back_task_summary": True}),
        "Generated Outputs": escape_untrusted_json(""),
        "Feedback": escape_untrusted_json([
            {
                "split": item.split, "task_count": item.task_count,
                "candidate_success_rate": item.candidate_success_rate,
                "mean_quality_delta": item.mean_quality_delta,
                "candidate_skill_loaded_rate": item.candidate_skill_loaded_rate,
            }
            for item in aggregates
        ]),
    }


def _unscored_record(reason: str) -> dict[str, str]:
    return {
        "Inputs": escape_untrusted_json({"not_scored": True}),
        "Generated Outputs": escape_untrusted_json(""),
        "Feedback": escape_untrusted_json(reason),
    }
