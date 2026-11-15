"""Optional GEPA reflective-mutation strategy (``pip install '.[gepa]'``).

Only GEPA's reflective instruction proposer
(``gepa.strategies.instruction_proposal.InstructionProposalSignature``) is
reused. GEPA's own optimization loop, candidate selection and evaluation are
deliberately not used: in this system the evaluator, the sealed holdout and the
promotion policy stay outside the generator. The adapter refuses any GEPA
version other than the pinned, tested one.
"""
from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any

from .optimizer import (
    MutationRequest,
    OptimizerModel,
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
        if not callable(getattr(model, "complete", None)):
            raise TypeError("GEPA reflection model must provide complete(role=, system_prompt=, user_payload=)")
        if signature is None:
            if installed_version is not None:
                raise ValueError("installed_version may only accompany an injected signature")
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
        if signature is None:
            signature = importlib.import_module(_SIGNATURE_MODULE).InstructionProposalSignature
        if not callable(getattr(signature, "run", None)):
            raise TypeError("GEPA instruction proposal signature must provide run(lm, input_dict)")
        self.model = model
        self.signature = signature
        self.version = f"gepa-{installed_version}"

    def propose(self, request: MutationRequest) -> list[VariantProposal]:
        validate_mutation_request(request)
        if any(token in request.parent_content for token in _PLACEHOLDERS) or "<<<UNTRUSTED" in request.parent_content:
            raise ValueError("parent skill text contains GEPA template placeholders or evidence delimiters")
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
            records.append({
                "Inputs": escape_untrusted_json({"title": task.title, "prompt": task.prompt}),
                "Generated Outputs": escape_untrusted_json(task.candidate_output_excerpt),
                "Feedback": escape_untrusted_json({
                    "stage": stage.stage,
                    "baseline_success": task.baseline_success,
                    "candidate_success": task.candidate_success,
                    "quality_delta": task.quality_delta,
                    "failed_tool_calls": task.candidate_failed_tool_calls,
                    "gate_failures": list(stage.gate_failures),
                }),
            })
        if stage.aggregates:
            records.append({
                "Inputs": escape_untrusted_json({"stage": stage.stage, "held_back_task_summary": True}),
                "Generated Outputs": escape_untrusted_json(""),
                "Feedback": escape_untrusted_json([
                    {
                        "split": item.split, "task_count": item.task_count,
                        "candidate_success_rate": item.candidate_success_rate,
                        "mean_quality_delta": item.mean_quality_delta,
                    }
                    for item in stage.aggregates
                ]),
            })
    return records
