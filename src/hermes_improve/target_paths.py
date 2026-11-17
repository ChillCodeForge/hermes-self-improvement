"""Allowed brain target paths for every candidate type.

Candidates may only change files under ``brain/``. Each candidate type owns
one area of the brain and a fixed path shape inside it; the canonical
``Candidate.target`` string names the touched *units* (a skill or tool
directory, or a single file) joined by commas in sorted order. Path
components are restricted to a safe ASCII alphabet, which rules out traversal,
hidden files, separators, and the comma used as unit delimiter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from .models import CandidateType


BRAIN_ROOT = "brain"
BRAIN_AREAS = ("skills", "prompts", "workflows", "routing", "tool-descriptions", "mcp", "metadata", "tools")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PROTECTED_COMPONENTS = frozenset({
    ".git", "evaluator", "verifiers", "benchmarks", "holdout", "security",
    "promotion", "ci", ".github", "tests", "judge",
})
PROTECTED_REPOSITORY_PREFIXES = (
    "config/", "benchmarks/", "src/", "tests/", "scripts/", ".github/", "prompts/judge",
)
TEXT_SUFFIXES = frozenset({".md", ".txt", ".yaml", ".yml", ".json"})
TOOL_SUFFIXES = TEXT_SUFFIXES | {".py"}
_MAX_UNIT_DEPTH = 3


@dataclass(frozen=True)
class TargetRule:
    """Path shape for one candidate type.

    ``area`` is the directory under ``brain/``. A ``directory`` unit is
    ``brain/<area>/<name>/`` with files up to ``_MAX_UNIT_DEPTH`` levels below
    it; a ``file`` unit is exactly ``brain/<area>/<name><suffix>``.
    """

    area: tuple[str, ...]
    unit_kind: str
    suffixes: frozenset[str]
    min_units: int
    max_units: int | None


TARGET_RULES: dict[CandidateType, TargetRule] = {
    CandidateType.SKILL_CREATE: TargetRule(("skills",), "directory", TEXT_SUFFIXES, 1, 1),
    CandidateType.SKILL_PATCH: TargetRule(("skills",), "directory", TEXT_SUFFIXES, 1, 1),
    CandidateType.SKILL_MERGE: TargetRule(("skills",), "directory", TEXT_SUFFIXES, 2, None),
    CandidateType.SKILL_SPLIT: TargetRule(("skills",), "directory", TEXT_SUFFIXES, 2, None),
    CandidateType.TOOL_DESCRIPTION: TargetRule(("tool-descriptions",), "file", frozenset({".md"}), 1, 1),
    CandidateType.WORKFLOW: TargetRule(("workflows",), "file", frozenset({".yaml"}), 1, 1),
    CandidateType.ROUTING: TargetRule(("routing",), "file", frozenset({".yaml"}), 1, None),
    CandidateType.MCP_TOOL_FILTER: TargetRule(("mcp", "tool-filters"), "file", frozenset({".yaml"}), 1, None),
    CandidateType.MCP_CONFIG: TargetRule(("mcp", "servers"), "file", frozenset({".yaml"}), 1, None),
    CandidateType.SYSTEM_PROMPT: TargetRule(("prompts",), "file", frozenset({".md"}), 1, 1),
    CandidateType.TOOL_CREATE: TargetRule(("tools",), "directory", TOOL_SUFFIXES, 1, 1),
    CandidateType.TOOL_PATCH: TargetRule(("tools",), "directory", TOOL_SUFFIXES, 1, 1),
}


def is_protected_repository_path(path: str) -> bool:
    """True for repository paths no candidate may ever change."""
    folded = path.casefold()
    return any(
        folded == prefix.rstrip("/").casefold() or folded.startswith(prefix.casefold())
        for prefix in PROTECTED_REPOSITORY_PREFIXES
    )


def validate_brain_file(candidate_type: CandidateType, path: str) -> str:
    """Validate one changed file path for ``candidate_type`` and return its unit."""
    if not isinstance(candidate_type, CandidateType):
        raise ValueError("candidate type must be a CandidateType")
    if not isinstance(path, str) or not path or "\\" in path or "\x00" in path:
        raise ValueError("candidate file path must be a POSIX path under brain/")
    if is_protected_repository_path(path):
        raise ValueError(f"candidate may not change protected path: {path}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or pure.as_posix() != path or pure.parts[:1] != (BRAIN_ROOT,):
        raise ValueError(f"candidate file path must be a normalized path under brain/: {path}")
    rule = TARGET_RULES[candidate_type]
    area_length = 1 + len(rule.area)
    if pure.parts[1:area_length] != rule.area:
        raise ValueError(
            f"{candidate_type.value} may only change brain/{'/'.join(rule.area)}/: {path}"
        )
    below = pure.parts[area_length:]
    if not below:
        raise ValueError(f"candidate file path names no file: {path}")
    for part in below:
        if not SAFE_COMPONENT.fullmatch(part) or part.casefold() in PROTECTED_COMPONENTS:
            raise ValueError(f"candidate file path contains an unsafe or protected component: {path}")
    suffix = PurePosixPath(below[-1]).suffix
    if suffix not in rule.suffixes:
        raise ValueError(
            f"{candidate_type.value} files must end in one of {', '.join(sorted(rule.suffixes))}: {path}"
        )
    if rule.unit_kind == "file":
        if len(below) != 1:
            raise ValueError(f"{candidate_type.value} files must sit directly in brain/{'/'.join(rule.area)}/")
        if below[0].casefold() == "readme.md":
            raise ValueError(f"area README files are operator-maintained: {path}")
        return path
    if len(below) < 2 or len(below) > 1 + _MAX_UNIT_DEPTH:
        raise ValueError(
            f"{candidate_type.value} files must be inside brain/{'/'.join(rule.area)}/<name>/ "
            f"at most {_MAX_UNIT_DEPTH} levels deep: {path}"
        )
    if below[0].casefold() == "skill.md":
        raise ValueError(f"unit directory may not be named SKILL.md: {path}")
    if below[-1].casefold() == "skill.md" and below[-1] != "SKILL.md":
        raise ValueError(f"skill entry file must be named exactly SKILL.md: {path}")
    return "/".join(pure.parts[:area_length + 1])


def units_for_paths(candidate_type: CandidateType, paths: Iterable[str]) -> tuple[str, ...]:
    """Validate ``paths`` for the type and return the sorted touched units."""
    paths = tuple(paths)
    if not paths:
        raise ValueError("candidate must change at least one brain file")
    folded: dict[str, str] = {}
    for path in paths:
        key = path.casefold()
        if key in folded and folded[key] != path:
            raise ValueError(f"candidate paths collide case-insensitively: {folded[key]} / {path}")
        folded[key] = path
    units = sorted({validate_brain_file(candidate_type, path) for path in paths})
    rule = TARGET_RULES[candidate_type]
    if len(units) < rule.min_units or (rule.max_units is not None and len(units) > rule.max_units):
        bound = rule.min_units if rule.max_units == rule.min_units else (
            f"{rule.min_units}+" if rule.max_units is None else f"{rule.min_units}-{rule.max_units}"
        )
        raise ValueError(
            f"{candidate_type.value} must touch {bound} target unit(s); got {len(units)}: {', '.join(units)}"
        )
    return tuple(units)


def target_for_units(units: Iterable[str]) -> str:
    return ",".join(sorted(units))


def parse_target(candidate_type: CandidateType, target: str) -> tuple[str, ...]:
    """Split and validate a stored ``Candidate.target`` for its type."""
    if not isinstance(target, str) or not target or target != target.strip():
        raise ValueError("candidate target must be a non-empty unit list")
    units = tuple(target.split(","))
    if list(units) != sorted(set(units)):
        raise ValueError("candidate target units must be unique and sorted")
    rule = TARGET_RULES[candidate_type]
    # A directory unit is validated through a representative file inside it.
    probes = [f"{unit}/README.md" if rule.unit_kind == "directory" else unit for unit in units]
    for unit, probe in zip(units, probes):
        if validate_brain_file(candidate_type, probe) != unit:
            raise ValueError(f"candidate target unit is not a {candidate_type.value} unit: {unit}")
    units_for_paths(candidate_type, probes)
    return units


def assert_changed_paths_match_target(
    candidate_type: CandidateType, target: str, changed_paths: Iterable[str],
) -> tuple[str, ...]:
    """Ensure every changed path lies inside the declared target units and
    every declared unit is actually touched."""
    declared = parse_target(candidate_type, target)
    touched = units_for_paths(candidate_type, changed_paths)
    if set(touched) != set(declared):
        raise ValueError(
            "changed paths do not match the declared candidate target "
            f"(declared: {', '.join(declared)}; touched: {', '.join(touched)})"
        )
    return touched


def removed_skill_units(deleted_paths: Iterable[str]) -> tuple[str, ...]:
    """Skill units whose ``SKILL.md`` entry file a diff deletes.

    Hermes only loads a skill directory through its ``SKILL.md``; deleting
    that file removes the skill from production. Callers pass the paths that
    ``git diff --no-renames --diff-filter=D`` reports (rename detection would
    hide a deletion behind a rename). The file name is compared
    case-insensitively so an odd spelling can never dodge the removal gate.
    """
    units = set()
    for path in deleted_paths:
        pure = PurePosixPath(path)
        if (
            len(pure.parts) >= 4 and pure.parts[:2] == (BRAIN_ROOT, "skills")
            and pure.name.casefold() == "skill.md"
        ):
            units.add("/".join(pure.parts[:3]))
    return tuple(sorted(units))


def find_case_collisions(paths: Iterable[str]) -> list[tuple[str, str]]:
    """Return pairs of distinct paths (or path prefixes) equal under casefold."""
    seen: dict[str, str] = {}
    collisions = []
    for path in paths:
        parts = PurePosixPath(path).parts
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            key = prefix.casefold()
            existing = seen.setdefault(key, prefix)
            if existing != prefix:
                collisions.append((existing, prefix))
                break
    return collisions


def assert_evaluable_candidate_paths(
    candidate_type: CandidateType, target: str, changed_paths: Iterable[str],
) -> tuple[str, ...]:
    """Evaluator-side guard: a candidate may change only its declared brain units.

    Benchmarks read fixtures, harnesses and checks from the evaluated tree (or
    from protected roots next to it); a candidate commit that also touched
    ``benchmarks/``, ``config/``, evaluator code or any path outside its
    declared target could otherwise change the yardstick it is measured with.
    The promoter applies the same rules again before release.
    """
    paths = [path for path in changed_paths if path]
    protected = sorted(path for path in paths if is_protected_repository_path(path))
    if protected:
        raise ValueError(f"candidate diff touches protected paths: {', '.join(protected[:20])}")
    outside = sorted(path for path in paths if not path.startswith(f"{BRAIN_ROOT}/"))
    if outside:
        raise ValueError(f"candidate diff touches paths outside brain/: {', '.join(outside[:20])}")
    return assert_changed_paths_match_target(candidate_type, target, paths)
