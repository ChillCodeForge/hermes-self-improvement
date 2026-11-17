"""Turn selected DEEP Dreaming proposals into immutable skill candidates (spec section 27).

The ``qwen_optimizer`` role writes the SKILL.md text; target, candidate type and
hypothesis come only from the validated proposal, never from the model reply.
Generated text must pass secret and protected-path checks and is committed only
through :func:`create_skill_candidate` on a fresh candidate branch. Nothing here
writes to production, evaluates or promotes.
"""
from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .candidate_builder import create_skill_candidate
from .evolution_loop import git_output
from .ingestion import contains_secret_material
from .models import CandidateType
from .optimizer import OptimizerModel
from .prompting import check_generated_skill_content, load_prompt, skill_content_digest, wrap_untrusted_evidence
from .skill_paths import validate_skill_target
from .store import ExperimentStore


MAX_CANDIDATES_PER_RUN = 5
_PROPOSAL_TYPES = {CandidateType.SKILL_CREATE.value, CandidateType.SKILL_PATCH.value}


@dataclass(frozen=True)
class GeneratedCandidateOutcome:
    proposal_index: int
    target: str
    status: str
    reason: str = ""
    candidate_id: str | None = None
    experiment_id: str | None = None
    commit: str | None = None
    candidate_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_candidates_from_proposals(
    proposals: Sequence[Mapping[str, Any]],
    *,
    model: OptimizerModel,
    repository: Path,
    store: ExperimentStore,
    benchmark_version: str,
    source_sessions: tuple[str, ...] = (),
    policy_version: str = "1",
) -> list[GeneratedCandidateOutcome]:
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise ValueError("selected proposals must be a sequence of objects")
    if not proposals:
        raise ValueError("select at least one DEEP Dreaming proposal")
    if len(proposals) > MAX_CANDIDATES_PER_RUN:
        raise ValueError(f"at most {MAX_CANDIDATES_PER_RUN} candidates may be generated per Dreaming run")
    if not callable(getattr(model, "complete", None)):
        raise TypeError("candidate author model must provide complete(role=, system_prompt=, user_payload=)")
    repository = Path(repository).resolve()
    run_id = f"dream-{secrets.token_hex(6)}"
    system_prompt = load_prompt("dreaming/candidate_content.md")
    outcomes = []
    for index, proposal in enumerate(proposals):
        outcomes.append(_generate_one(
            index, proposal, model=model, system_prompt=system_prompt, repository=repository,
            store=store, benchmark_version=benchmark_version, run_id=run_id,
            source_sessions=source_sessions, policy_version=policy_version,
        ))
    return outcomes


def _generate_one(
    index: int, proposal: Mapping[str, Any], *, model: OptimizerModel, system_prompt: str,
    repository: Path, store: ExperimentStore, benchmark_version: str, run_id: str,
    source_sessions: tuple[str, ...], policy_version: str,
) -> GeneratedCandidateOutcome:
    target = proposal.get("target") if isinstance(proposal, Mapping) else None
    shown_target = target if isinstance(target, str) else ""
    try:
        candidate_type, target, hypothesis, summary = _validate_proposal(proposal)
    except ValueError as exc:
        return GeneratedCandidateOutcome(index, shown_target, "rejected", f"invalid proposal: {exc}")
    existing = _main_skill(repository, target)
    if candidate_type is CandidateType.SKILL_PATCH and existing is None:
        return GeneratedCandidateOutcome(index, target, "rejected", "skill to patch does not exist on main")
    if candidate_type is CandidateType.SKILL_CREATE and existing is not None:
        return GeneratedCandidateOutcome(index, target, "rejected", "skill to create already exists on main")
    evidence: dict[str, Any] = {
        "proposal": {
            "summary": summary, "hypothesis": hypothesis,
            "target": target, "candidate_type": candidate_type.value,
        },
    }
    if existing is not None:
        evidence["current_skill_text"] = existing
    response = model.complete(
        role="qwen_optimizer", system_prompt=system_prompt, user_payload=wrap_untrusted_evidence(evidence),
        json_object=True,
    )
    try:
        content = _parse_content(response)
        check_generated_skill_content(content)
    except ValueError as exc:
        return GeneratedCandidateOutcome(index, target, "rejected", str(exc))
    digest = skill_content_digest(content)
    previous = store.find_generated_variant(target, digest)
    if previous is not None:
        return GeneratedCandidateOutcome(
            index, target, "duplicate",
            f"identical content already generated ({previous['status']})",
            candidate_id=previous["candidate_id"],
        )
    experiment_id = f"{run_id}-p{index}"
    origin = f"dreaming:{run_id}"
    try:
        candidate = create_skill_candidate(
            repository, store,
            experiment_id=experiment_id, candidate_id=f"{experiment_id}-c",
            candidate_type=candidate_type, target=target, content=content,
            hypothesis=hypothesis, benchmark_version=benchmark_version,
            source_sessions=source_sessions, policy_version=policy_version,
        )
    except ValueError as exc:
        store.record_generated_variant(
            target=target, content_sha256=digest, origin=origin, generation=0,
            status="rejected", reason=str(exc),
        )
        return GeneratedCandidateOutcome(index, target, "rejected", str(exc))
    store.record_generated_variant(
        target=target, content_sha256=digest, origin=origin, generation=0,
        status="created", candidate_id=candidate.candidate_id,
    )
    return GeneratedCandidateOutcome(
        index, target, "created", "awaiting evaluation",
        candidate_id=candidate.candidate_id, experiment_id=candidate.experiment_id,
        commit=candidate.commit, candidate_hash=candidate.content_hash,
    )


def _validate_proposal(proposal: Any) -> tuple[CandidateType, str, str, str]:
    if not isinstance(proposal, Mapping):
        raise ValueError("proposal must be an object")
    candidate_type = proposal.get("candidate_type")
    if candidate_type not in _PROPOSAL_TYPES:
        raise ValueError("only SKILL_CREATE and SKILL_PATCH proposals can be generated")
    target = proposal.get("target")
    validate_skill_target(target)
    hypothesis = proposal.get("hypothesis")
    summary = proposal.get("summary", "")
    if (
        not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 2000
        or not isinstance(summary, str) or len(summary) > 500
    ):
        raise ValueError("proposal requires a hypothesis up to 2000 and a summary up to 500 characters")
    if contains_secret_material({"hypothesis": hypothesis, "summary": summary}):
        raise ValueError("proposal contains credential-like material")
    return CandidateType(candidate_type), target, hypothesis, summary


def _parse_content(response: Any) -> str:
    if not isinstance(response, str):
        raise ValueError("candidate author response must be text")
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate author response is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("content"), str):
        raise ValueError("candidate author response must be a JSON object with a content string")
    return value["content"]


def _main_skill(repository: Path, target: str) -> str | None:
    path = f"{validate_skill_target(target).as_posix()}/SKILL.md"
    listed = git_output(repository, "ls-tree", "--name-only", "refs/heads/main", "--", path)
    if not listed.strip():
        return None
    return git_output(repository, "show", f"refs/heads/main:{path}").decode("utf-8")
