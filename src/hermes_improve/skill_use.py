"""Skill use per run: which skills the agent actually loaded (lab finding F30).

Hermes lists the installed skills in its system prompt, but the agent has to
call ``skill_view(name)`` to read one. A skill candidate that is never loaded
cannot change behaviour, so its delta is 0 whatever its content. To tell
"useless content" from "never read", the runner records per run which skills
were requested through ``skill_view`` and how often ``skills_list`` was called,
and the evaluator reports per arm the share of runs that loaded the
candidate's target skill.

Source: the assistant (``gpt``) turns of the Hermes trajectory, where each tool
call is a ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` block
(``_convert_to_trajectory_format``). A call counts as a load when it was
issued, whether or not Hermes found the skill. Malformed blocks are ignored;
they never invalidate a trajectory. Names are bounded in length and number.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .models import Candidate, CandidateType, RunObservation


MAX_SKILLS_PER_RUN = 32
MAX_SKILL_NAME_CHARS = 200
SKILL_CANDIDATE_TYPES = frozenset({
    CandidateType.SKILL_CREATE, CandidateType.SKILL_PATCH, CandidateType.SKILL_MERGE, CandidateType.SKILL_SPLIT,
})
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, Mapping) else {}


def skill_loads(conversations: Iterable[Mapping[str, Any]]) -> tuple[tuple[str, ...], int]:
    """``(skills requested via skill_view in first-call order, number of skills_list calls)``."""
    viewed: list[str] = []
    listed = 0
    for turn in conversations:
        if not isinstance(turn, Mapping) or turn.get("from") != "gpt" or not isinstance(turn.get("value"), str):
            continue
        for block in _TOOL_CALL.findall(turn["value"]):
            try:
                call = json.loads(block)
            except json.JSONDecodeError:
                continue
            if not isinstance(call, Mapping):
                continue
            if call.get("name") == "skills_list":
                listed += 1
            elif call.get("name") == "skill_view":
                name = _arguments(call).get("name")
                if not isinstance(name, str):
                    continue
                name = name.strip()
                if (
                    name and len(name) <= MAX_SKILL_NAME_CHARS and name not in viewed
                    and len(viewed) < MAX_SKILLS_PER_RUN
                ):
                    viewed.append(name)
    return tuple(viewed), listed


def target_skill_aliases(candidate: Candidate) -> frozenset[str] | None:
    """Names under which the agent can load the candidate's target skill(s); ``None`` if not a skill candidate.

    Hermes' ``skill_view`` accepts the directory name or the path below the
    skills directory (``ops/deploy``). Generated skills carry a front-matter
    ``name`` equal to the directory name (``prompting.check_generated_skill_content``);
    a hand-written skill whose front-matter name differs from its directory is
    not matched under that name.
    """
    units = _skill_units(candidate)
    if not units:
        return None
    return frozenset(alias for unit in units for alias in (unit.as_posix(), unit.name))


def target_skill_names(candidate: Candidate) -> list[str] | None:
    """The target skills as paths below ``brain/skills`` (for reports); ``None`` if not a skill candidate."""
    units = _skill_units(candidate)
    return sorted(unit.as_posix() for unit in units) if units else None


def _skill_units(candidate: Candidate) -> list[PurePosixPath]:
    if candidate.candidate_type not in SKILL_CANDIDATE_TYPES:
        return []
    units = []
    for unit in candidate.target.split(","):
        path = PurePosixPath(unit.strip())
        if len(path.parts) >= 3 and path.parts[:2] == ("brain", "skills"):
            units.append(PurePosixPath(*path.parts[2:]))
    return units


def loaded_target(run: RunObservation, aliases: frozenset[str]) -> bool:
    return any(name in aliases for name in run.skills_viewed)


def target_loaded_rate(runs: Sequence[RunObservation], aliases: frozenset[str] | None) -> float | None:
    """Share of runs with an observed trajectory that loaded a target skill; ``None`` when unknown."""
    observed = [run for run in runs if run.skill_use_observed]
    if aliases is None or not observed:
        return None
    return sum(loaded_target(run, aliases) for run in observed) / len(observed)


def arm_skill_use(runs: Sequence[RunObservation], aliases: frozenset[str] | None) -> dict[str, Any]:
    observed = [run for run in runs if run.skill_use_observed]
    loaded = sum(loaded_target(run, aliases) for run in observed) if aliases is not None else None
    viewed = Counter(name for run in observed for name in run.skills_viewed)
    return {
        "runs": len(runs),
        "observed_runs": len(observed),
        "runs_with_skill_view": sum(bool(run.skills_viewed) for run in observed),
        "runs_with_skills_list": sum(run.skills_list_calls > 0 for run in observed),
        "target_skill_loaded_runs": loaded if observed else None,
        "target_skill_loaded_rate": target_loaded_rate(runs, aliases),
        # Runs per skill name, at most MAX_SKILLS_PER_RUN names (most frequent first, then by name).
        "skills_viewed": dict(sorted(viewed.items(), key=lambda item: (-item[1], item[0]))[:MAX_SKILLS_PER_RUN]),
    }


def skill_use_report(
    candidate: Candidate, pairs: Sequence[tuple[RunObservation, RunObservation]],
) -> dict[str, Any]:
    """Per-arm skill use of paired (baseline, candidate) runs for the signed evaluation metrics."""
    aliases = target_skill_aliases(candidate)
    return {
        "target_skill": target_skill_names(candidate),
        "baseline": arm_skill_use([pair[0] for pair in pairs], aliases),
        "candidate": arm_skill_use([pair[1] for pair in pairs], aliases),
        "note": (
            "target_skill_loaded_rate = share of runs with an observed trajectory that called skill_view "
            "for the candidate's target skill; null = no skill candidate or no trajectory (unknown)"
        ),
    }


def valid_skill_use(observation: RunObservation) -> bool:
    viewed = observation.skills_viewed
    listed = observation.skills_list_calls
    return (
        isinstance(observation.skill_use_observed, bool)
        and isinstance(viewed, tuple) and len(viewed) <= MAX_SKILLS_PER_RUN
        and all(isinstance(name, str) and 0 < len(name) <= MAX_SKILL_NAME_CHARS for name in viewed)
        and len(set(viewed)) == len(viewed)
        and not isinstance(listed, bool) and isinstance(listed, int) and listed >= 0
    )
