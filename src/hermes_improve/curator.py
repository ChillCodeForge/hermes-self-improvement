from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .ingestion import contains_secret_material
from .models import Candidate, CandidateType, ImprovementSignal

if TYPE_CHECKING:
    from .curator_connector import PendingSkillWrite
    from .store import ExperimentStore


_MAX_COUNTER = (1 << 63) - 1
_MAX_SKILL_BYTES = 256 * 1024
_BRAIN_SKILL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
MUTATION_KINDS = frozenset({"create", "patch", "consolidate", "archive"})


class CuratorMutationRefused(ValueError):
    """A Curator mutation that cannot be expressed as an evaluable candidate."""


def validate_curator_skill_name(name: Any) -> str:
    if (
        not isinstance(name, str) or not name.strip() or name != name.strip()
        or len(name) > 128 or any(ord(character) < 32 for character in name)
        or contains_secret_material(name)
    ):
        raise ValueError("Curator skill names must be bounded, clean, and credential-free")
    return name


@dataclass(frozen=True)
class CuratorMutation:
    """A mutation the Curator wants to apply to a skill.

    ``content`` is the complete new ``SKILL.md`` text (create, full-rewrite
    patch, or the umbrella skill of a consolidation). A targeted patch instead
    supplies ``old_string``/``new_string`` which are applied to the current
    production ``SKILL.md`` read from the candidate repository's ``main``.
    """

    kind: str
    skill: str
    rationale: str
    content: str | None = None
    old_string: str | None = None
    new_string: str | None = None
    replace_all: bool = False
    into: str | None = None
    source_sessions: tuple[str, ...] = ()


@dataclass(frozen=True)
class CuratorConversion:
    candidate: Candidate
    mutation_kind: str
    unapplied: tuple[str, ...] = field(default_factory=tuple)


class CuratorAdapter:
    """Bridges the Hermes Curator to the gated candidate path.

    ``signals`` turns read-only usage data into improvement signals. Mutation
    conversion never writes production: every desired Curator change becomes an
    immutable candidate commit on a ``candidate/<experiment-id>`` branch that
    must still pass evaluation and promotion.
    """

    def __init__(
        self,
        usage_report: Callable[[], Iterable[Mapping[str, Any]]] | None = None,
        *,
        repository: Path | None = None,
        store: "ExperimentStore | None" = None,
        benchmark_version: str | None = None,
    ) -> None:
        self._usage_report = usage_report
        self._repository = Path(repository).resolve() if repository is not None else None
        self._store = store
        self._benchmark_version = benchmark_version

    def signals(self) -> list[ImprovementSignal]:
        if self._usage_report is None:
            raise RuntimeError("CuratorAdapter was created without a usage report source")
        result = []
        for row in self._usage_report():
            if not isinstance(row, Mapping):
                raise ValueError("Curator usage rows must be objects")
            name = validate_curator_skill_name(row.get("name"))
            use_count = _counter(row.get("use_count", 0), "use_count")
            patch_count = _counter(row.get("patch_count", 0), "patch_count")
            activity = _counter(row.get("activity_count", 0), "activity_count")
            if use_count == patch_count == activity == 0:
                continue
            result.append(ImprovementSignal(
                source="curator",
                summary=f"Review usage and maintenance signals for {name[:100]}",
                issue=f"Assess whether {name[:100]} needs a proposal, consolidation, or documentation improvement",
                impact_score=min(1.0, max(0.0, max(use_count, patch_count) / max(1, activity + 1))),
                evidence={
                    "name": name,
                    "use_count": use_count,
                    "patch_count": patch_count,
                    "activity_count": activity,
                },
            ))
        return result

    def to_candidate(
        self, mutation: CuratorMutation, *, experiment_id: str, candidate_id: str,
    ) -> CuratorConversion:
        """Convert a desired Curator mutation into an immutable candidate.

        ``create``/``patch`` become ``SKILL_CREATE``/``SKILL_PATCH``.
        ``consolidate`` becomes a ``SKILL_MERGE`` that writes the umbrella
        ``SKILL.md`` and deletes every tracked file of the absorbed skill; when
        no new umbrella content is supplied (or it equals production) only the
        removal remains, as for ``archive``: a ``SKILL_PATCH`` that deletes the
        whole skill directory. Every candidate that removes a skill is manually
        gated at promotion (``PromotionPolicy.requires_manual_approval``).
        """
        from .candidate_builder import create_candidate, create_skill_candidate, production_skill_files

        if self._repository is None or self._store is None or not self._benchmark_version:
            raise RuntimeError("mutation conversion requires a candidate repository, experiment store, and benchmark version")
        if not isinstance(mutation, CuratorMutation) or mutation.kind not in MUTATION_KINDS:
            raise ValueError(f"Curator mutation kind must be one of: {', '.join(sorted(MUTATION_KINDS))}")
        skill = _brain_skill_name(mutation.skill)
        if not isinstance(mutation.rationale, str) or not mutation.rationale.strip():
            raise ValueError("Curator mutations require a rationale")
        if not isinstance(mutation.replace_all, bool):
            raise ValueError("replace_all must be boolean")
        rationale = mutation.rationale.strip()
        if mutation.kind in {"create", "patch"}:
            if mutation.kind == "create":
                candidate_type = CandidateType.SKILL_CREATE
                content = _require_content(mutation, "create")
            else:
                candidate_type = CandidateType.SKILL_PATCH
                content = self._patched_content(skill, mutation)
            candidate = create_skill_candidate(
                self._repository,
                self._store,
                experiment_id=experiment_id,
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                target=f"brain/skills/{skill}",
                content=content,
                hypothesis=f"Curator {mutation.kind} {skill}: {rationale}"[:4000],
                benchmark_version=self._benchmark_version,
                source_sessions=tuple(mutation.source_sessions),
            )
            return CuratorConversion(candidate=candidate, mutation_kind=mutation.kind)

        files: dict[str, str] = {}
        if mutation.kind == "archive":
            if mutation.into is not None or mutation.content is not None:
                raise ValueError("archive takes neither 'into' nor content; use consolidate")
            hypothesis = f"Curator archive {skill}: {rationale}"
        else:
            if mutation.into is None:
                raise ValueError("consolidation requires the umbrella skill in 'into'")
            umbrella = _brain_skill_name(mutation.into)
            if umbrella == skill:
                raise ValueError("a skill cannot be consolidated into itself")
            current = _production_skill(self._repository, umbrella)
            if mutation.content is not None:
                content = _require_content(mutation, "consolidate")
                # The builder normalizes to one trailing newline; an unchanged
                # umbrella leaves a pure removal, which SKILL_MERGE cannot express.
                if content.rstrip() + "\n" != current:
                    files[f"brain/skills/{umbrella}/SKILL.md"] = content
            hypothesis = f"Curator consolidate {skill} into {umbrella}: {rationale}"
        try:
            removed = production_skill_files(self._repository, f"brain/skills/{skill}")
        except ValueError as exc:
            raise CuratorMutationRefused(f"cannot remove {skill}: {exc}") from exc
        candidate = create_candidate(
            self._repository,
            self._store,
            experiment_id=experiment_id,
            candidate_id=candidate_id,
            candidate_type=CandidateType.SKILL_MERGE if files else CandidateType.SKILL_PATCH,
            files=files,
            deletions=removed,
            hypothesis=hypothesis[:4000],
            benchmark_version=self._benchmark_version,
            source_sessions=tuple(mutation.source_sessions),
            risk="HIGH",
        )
        return CuratorConversion(candidate=candidate, mutation_kind=mutation.kind)

    def convert_pending_write(
        self, pending: "PendingSkillWrite", *, experiment_id: str, candidate_id: str,
    ) -> CuratorConversion:
        """Convert a skill write staged by Hermes' ``skills.write_approval`` gate."""
        payload = pending.payload
        rationale = pending.summary or f"staged {pending.action} of {pending.skill} ({pending.origin})"
        if pending.action in {"write_file", "remove_file"} or (
            pending.action == "patch" and payload.get("file_path") not in (None, "", "SKILL.md")
        ):
            raise CuratorMutationRefused(
                "staged supporting-file writes cannot be converted: candidates currently cover SKILL.md only"
            )
        if pending.action == "delete":
            absorbed_into = payload.get("absorbed_into")
            if absorbed_into in (None, ""):
                mutation = CuratorMutation("archive", pending.skill, rationale)
            else:
                mutation = CuratorMutation(
                    "consolidate", pending.skill, rationale,
                    content=payload.get("content") or None, into=absorbed_into,
                )
        elif pending.action == "create":
            mutation = CuratorMutation("create", pending.skill, rationale, content=payload.get("content"))
        elif pending.action == "edit" or (pending.action == "patch" and payload.get("content")):
            mutation = CuratorMutation("patch", pending.skill, rationale, content=payload.get("content"))
        else:
            mutation = CuratorMutation(
                "patch", pending.skill, rationale,
                old_string=payload.get("old_string"), new_string=payload.get("new_string"),
                replace_all=payload.get("replace_all", False),
            )
        return self.to_candidate(mutation, experiment_id=experiment_id, candidate_id=candidate_id)

    def _patched_content(self, skill: str, mutation: CuratorMutation) -> str:
        targeted = mutation.old_string is not None or mutation.new_string is not None
        if mutation.content is not None and targeted:
            raise ValueError("a patch supplies either full content or old_string/new_string, not both")
        if not targeted:
            return _require_content(mutation, "patch")
        if not isinstance(mutation.old_string, str) or not mutation.old_string:
            raise ValueError("targeted patch requires a non-empty old_string")
        if not isinstance(mutation.new_string, str):
            raise ValueError("targeted patch requires new_string")
        if self._repository is None:
            raise RuntimeError("targeted patches require the candidate repository")
        current = _production_skill(self._repository, skill)
        occurrences = current.count(mutation.old_string)
        if occurrences == 0:
            raise CuratorMutationRefused(f"old_string does not occur in the production SKILL.md of {skill}")
        if occurrences > 1 and not mutation.replace_all:
            raise CuratorMutationRefused(
                f"old_string occurs {occurrences} times in {skill}; set replace_all for a non-unique patch"
            )
        return current.replace(mutation.old_string, mutation.new_string)


def _production_skill(repository: Path, skill: str) -> str:
    """Read the current production SKILL.md from ``main`` without touching any checkout."""
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository),
         "show", f"refs/heads/main:brain/skills/{skill}/SKILL.md"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        },
    )
    if result.returncode:
        raise CuratorMutationRefused(f"production SKILL.md for {skill} does not exist on main")
    if len(result.stdout) > _MAX_SKILL_BYTES:
        raise ValueError("production SKILL.md exceeds the 256 KiB limit")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("production SKILL.md is not UTF-8") from exc


def _brain_skill_name(name: Any) -> str:
    validate_curator_skill_name(name)
    if not _BRAIN_SKILL_NAME.fullmatch(name):
        raise CuratorMutationRefused(
            f"skill name {name!r} cannot be mapped to a brain/skills/ directory component"
        )
    return name


def _require_content(mutation: CuratorMutation, kind: str) -> str:
    if not isinstance(mutation.content, str) or not mutation.content.strip():
        raise ValueError(f"Curator {kind} mutations require complete SKILL.md content")
    return mutation.content


def _counter(value: Any, field_name: str) -> int:
    if value is None:
        return 0
    if (
        isinstance(value, bool) or not isinstance(value, int)
        or value < 0 or value > _MAX_COUNTER
    ):
        raise ValueError(f"Curator {field_name} must be a non-negative bounded integer")
    return value
