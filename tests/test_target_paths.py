from __future__ import annotations

from pathlib import Path

import pytest

from hermes_improve.config import strict_yaml_load
from hermes_improve.models import CandidateType
from hermes_improve.target_paths import (
    BRAIN_AREAS,
    TARGET_RULES,
    assert_changed_paths_match_target,
    find_case_collisions,
    is_protected_repository_path,
    parse_target,
    target_for_units,
    units_for_paths,
    validate_brain_file,
)


REPO = Path(__file__).parents[1]


def test_every_candidate_type_maps_to_a_brain_area():
    assert set(TARGET_RULES) == set(CandidateType)
    for rule in TARGET_RULES.values():
        assert rule.area[0] in BRAIN_AREAS
        assert (REPO / "brain" / rule.area[0] / "README.md").is_file()


@pytest.mark.parametrize(
    ("candidate_type", "path", "unit"),
    [
        (CandidateType.SKILL_CREATE, "brain/skills/docker-debug/SKILL.md", "brain/skills/docker-debug"),
        (CandidateType.SKILL_PATCH, "brain/skills/docker-debug/references/flags.md", "brain/skills/docker-debug"),
        (CandidateType.TOOL_DESCRIPTION, "brain/tool-descriptions/terminal.md", "brain/tool-descriptions/terminal.md"),
        (CandidateType.WORKFLOW, "brain/workflows/release.yaml", "brain/workflows/release.yaml"),
        (CandidateType.ROUTING, "brain/routing/models.yaml", "brain/routing/models.yaml"),
        (CandidateType.MCP_TOOL_FILTER, "brain/mcp/tool-filters/github.yaml", "brain/mcp/tool-filters/github.yaml"),
        (CandidateType.MCP_CONFIG, "brain/mcp/servers/github.yaml", "brain/mcp/servers/github.yaml"),
        (CandidateType.SYSTEM_PROMPT, "brain/prompts/default.md", "brain/prompts/default.md"),
        (CandidateType.TOOL_CREATE, "brain/tools/jsonfmt/tool.py", "brain/tools/jsonfmt"),
        (CandidateType.TOOL_PATCH, "brain/tools/jsonfmt/README.md", "brain/tools/jsonfmt"),
    ],
)
def test_valid_paths_resolve_to_their_unit(candidate_type, path, unit):
    assert validate_brain_file(candidate_type, path) == unit


@pytest.mark.parametrize(
    ("candidate_type", "path", "message"),
    [
        (CandidateType.SKILL_CREATE, "brain/skills/../config/SKILL.md", "unsafe"),
        (CandidateType.SKILL_CREATE, "brain/skills/x/../../config/promotion.yaml", "normalized|unsafe"),
        (CandidateType.SKILL_CREATE, "/brain/skills/x/SKILL.md", "normalized"),
        (CandidateType.SKILL_CREATE, "Brain/skills/x/SKILL.md", "normalized"),
        (CandidateType.SKILL_CREATE, "brain/Skills/x/SKILL.md", "may only change"),
        (CandidateType.SKILL_CREATE, "brain/skills/x/skill.md", "exactly SKILL.md"),
        (CandidateType.SKILL_CREATE, "brain/skills/x/.hidden.md", "unsafe"),
        (CandidateType.SKILL_CREATE, "brain/skills/x/run.sh", "must end in"),
        (CandidateType.SKILL_CREATE, "brain/skills/SKILL.md", "inside"),
        (CandidateType.SKILL_CREATE, "brain/skills/judge/SKILL.md", "protected"),
        (CandidateType.SKILL_CREATE, "brain/skills/a/b/c/d/e.md", "levels deep"),
        (CandidateType.SKILL_CREATE, "brain\\skills\\x\\SKILL.md", "POSIX"),
        (CandidateType.SKILL_CREATE, "config/promotion.yaml", "protected"),
        (CandidateType.SKILL_PATCH, "src/hermes_improve/policy.py", "protected"),
        (CandidateType.SKILL_PATCH, "tests/test_policy.py", "protected"),
        (CandidateType.SKILL_PATCH, "benchmarks/manifest.json", "protected"),
        (CandidateType.SKILL_PATCH, "scripts/docker-test.sh", "protected"),
        (CandidateType.SKILL_PATCH, ".github/workflows/ci.yml", "protected"),
        (CandidateType.SYSTEM_PROMPT, "prompts/judge/pairwise.md", "protected"),
        (CandidateType.TOOL_DESCRIPTION, "brain/skills/x/SKILL.md", "may only change"),
        (CandidateType.TOOL_DESCRIPTION, "brain/tool-descriptions/README.md", "operator-maintained"),
        (CandidateType.WORKFLOW, "brain/workflows/deploy.yml", "must end in"),
        (CandidateType.WORKFLOW, "brain/workflows/nested/deploy.yaml", "directly"),
        (CandidateType.MCP_CONFIG, "brain/mcp/tool-filters/github.yaml", "may only change"),
        (CandidateType.SYSTEM_PROMPT, "brain/metadata/owners.md", "may only change"),
    ],
)
def test_invalid_paths_are_rejected(candidate_type, path, message):
    with pytest.raises(ValueError, match=message):
        validate_brain_file(candidate_type, path)


def test_unit_counts_per_type():
    assert units_for_paths(CandidateType.SKILL_MERGE, [
        "brain/skills/a/SKILL.md", "brain/skills/b/SKILL.md",
    ]) == ("brain/skills/a", "brain/skills/b")
    with pytest.raises(ValueError, match="2\\+ target unit"):
        units_for_paths(CandidateType.SKILL_SPLIT, ["brain/skills/a/SKILL.md"])
    with pytest.raises(ValueError, match="must touch 1 target unit"):
        units_for_paths(CandidateType.SKILL_PATCH, ["brain/skills/a/SKILL.md", "brain/skills/b/SKILL.md"])
    with pytest.raises(ValueError, match="collide case-insensitively"):
        units_for_paths(CandidateType.ROUTING, ["brain/routing/a.yaml", "brain/routing/A.yaml"])
    with pytest.raises(ValueError, match="at least one"):
        units_for_paths(CandidateType.ROUTING, [])


def test_target_round_trip_and_changed_path_matching():
    target = target_for_units(["brain/skills/b", "brain/skills/a"])
    assert target == "brain/skills/a,brain/skills/b"
    assert parse_target(CandidateType.SKILL_MERGE, target) == ("brain/skills/a", "brain/skills/b")
    assert assert_changed_paths_match_target(
        CandidateType.SKILL_MERGE, target, ["brain/skills/a/SKILL.md", "brain/skills/b/SKILL.md"],
    ) == ("brain/skills/a", "brain/skills/b")
    with pytest.raises(ValueError, match="do not match"):
        assert_changed_paths_match_target(
            CandidateType.SKILL_MERGE, target, ["brain/skills/a/SKILL.md", "brain/skills/c/SKILL.md"],
        )
    with pytest.raises(ValueError, match="sorted"):
        parse_target(CandidateType.SKILL_MERGE, "brain/skills/b,brain/skills/a")
    with pytest.raises(ValueError, match="unsafe|protected"):
        parse_target(CandidateType.SKILL_PATCH, "brain/skills/.git")
    assert parse_target(CandidateType.TOOL_PATCH, "brain/tools/jsonfmt") == ("brain/tools/jsonfmt",)


def test_protected_prefixes_and_case_collisions():
    assert is_protected_repository_path("CONFIG/promotion.yaml")
    assert is_protected_repository_path("prompts/judge")
    assert not is_protected_repository_path("brain/prompts/judge-free.md")
    assert find_case_collisions(["brain/skills/demo/SKILL.md", "brain/skills/Demo/SKILL.md"]) == [
        ("brain/skills/demo", "brain/skills/Demo"),
    ]
    assert find_case_collisions(["brain/skills/a/SKILL.md", "brain/skills/b/SKILL.md"]) == []


def test_baseline_skill_is_a_valid_skill_file():
    skill = REPO / "brain/skills/git-commit-hygiene/SKILL.md"
    validate_brain_file(CandidateType.SKILL_PATCH, "brain/skills/git-commit-hygiene/SKILL.md")
    text = skill.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = strict_yaml_load(text.split("---\n")[1])
    assert front["name"] == "git-commit-hygiene"
    assert len(front["description"]) > 20
