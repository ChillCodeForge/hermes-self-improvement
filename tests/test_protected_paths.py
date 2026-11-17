"""Spec §3: generator code paths can never produce a candidate that touches the yardstick.

Table-driven over every CandidateType and every protected repository area
(config/, benchmarks/, src/, tests/, scripts/, .github/, prompts/judge, holdout).
Each generator entry point (candidate builder, evolution loop, Dreaming
proposal generator, Curator adapter, DEEP Dreaming) is attacked directly; the
assertion is always the same: no candidate, no branch, production unchanged.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from evolution_doubles import TARGET, ScriptedOptimizerModel, evolution_fixture, git, make_repository
from hermes_improve.candidate_builder import create_candidate
from hermes_improve.curator import CuratorAdapter, CuratorMutation
from hermes_improve.dreaming import DreamingEngine
from hermes_improve.errors import IntegrityError
from hermes_improve.evolution_loop import EvolutionConfig, EvolutionLoop
from hermes_improve.models import CandidateType
from hermes_improve.optimizer import ModelMutationStrategy
from hermes_improve.proposal_generator import generate_candidates_from_proposals
from hermes_improve.store import ExperimentStore
from hermes_improve.target_paths import TARGET_RULES, is_protected_repository_path

PROTECTED_FILES = (
    "config/promotion.yaml",
    "config/scoring.yaml",
    "benchmarks/train/coding-fix-pagination-bug-v1.json",
    "benchmarks/manifests/corpus.json",
    "src/hermes_improve/policy.py",
    "src/hermes_improve/evaluator.py",
    "tests/test_policy.py",
    "tests/security/test_os_isolation.py",
    "scripts/provision.sh",
    ".github/workflows/ci.yml",
    ".github/CODEOWNERS",
    "prompts/judge/pairwise_system.md",
    "holdout/sealed-task.json",
)


def _aliases(candidate_type: CandidateType) -> tuple[str, ...]:
    """Spellings that try to reach a protected file from inside the type's brain area."""
    area = "/".join(("brain", *TARGET_RULES[candidate_type].area))
    return (
        f"{area}/../../config/promotion.yaml",
        f"{area}/x/../../../src/hermes_improve/policy.py",
        f"/{area}/x/SKILL.md",
        f"{area}/holdout/SKILL.md",
        f"{area}/judge/SKILL.md",
        f"{area}/evaluator/SKILL.md",
        f"{area}/x/.git/config",
        f"{area}/x/.github/workflows/ci.yml",
        "Config/promotion.yaml",
        "CONFIG/promotion.yaml",
        ".GitHub/workflows/ci.yml",
        "Prompts/Judge/pairwise_system.md",
        "brain/../config/promotion.yaml",
        "./config/promotion.yaml",
    )


def _assert_untouched(repository, store, main_before, branches_before):
    assert git(repository, "rev-parse", "main") == main_before
    assert git(repository, "branch", "--list", "candidate/*").split() == branches_before
    assert git(repository, "status", "--porcelain") == ""
    assert store.list_candidates(1000) == [] or all(
        not is_protected_repository_path(path)
        for row in store.list_candidates(1000)
        for path in git(repository, "diff", "--name-only", row["base_commit"], row["commit_id"]).split()
    )


def _assert_all_candidates_stay_in_brain(repository):
    for branch in git(repository, "branch", "--list", "--format=%(refname:short)", "candidate/*").split():
        changed = git(repository, "diff", "--name-only", "main..." + branch).split()
        assert changed, branch
        for path in changed:
            assert path.startswith("brain/") and not is_protected_repository_path(path), (branch, path)


@pytest.fixture
def brain(tmp_path):
    repository = make_repository(tmp_path)
    for path in PROTECTED_FILES:
        file = repository / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("protected\n", encoding="utf-8")
    git(repository, "add", ".")
    git(repository, "commit", "-m", "protected yardstick files")
    return repository


@pytest.mark.parametrize("candidate_type", list(CandidateType), ids=lambda item: item.value)
def test_candidate_builder_refuses_every_protected_path_for_every_type(brain, tmp_path, candidate_type):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    main_before = git(brain, "rev-parse", "main")
    branches_before = git(brain, "branch", "--list", "candidate/*").split()
    attempts = [*PROTECTED_FILES, *_aliases(candidate_type)]
    for index, path in enumerate(attempts):
        for mode in ("write", "delete", "mixed"):
            kwargs = {"files": {path: "rewritten yardstick\n"}}
            if mode == "delete":
                kwargs = {"files": {}, "deletions": (path,)}
            elif mode == "mixed":
                legal = _legal_file(candidate_type)
                kwargs = {"files": {legal: "harmless change\n", path: "rewritten yardstick\n"}}
            with pytest.raises(ValueError):
                create_candidate(
                    brain, store, experiment_id=f"exp-{index}-{mode}", candidate_id=f"cand-{index}-{mode}",
                    candidate_type=candidate_type, hypothesis="attempt to edit the yardstick",
                    benchmark_version="bench-v1", **kwargs,
                )
    _assert_untouched(brain, store, main_before, branches_before)
    assert store.list_experiments(1000) == []


def _legal_file(candidate_type: CandidateType) -> str:
    rule = TARGET_RULES[candidate_type]
    area = "/".join(("brain", *rule.area))
    if rule.unit_kind == "file":
        return f"{area}/legal{sorted(rule.suffixes)[0]}"
    return f"{area}/legal/SKILL.md"


def test_protected_prefix_list_covers_every_spec_section_3_area():
    for path in PROTECTED_FILES:
        if not path.startswith("holdout/"):
            assert is_protected_repository_path(path), path


class _Author:
    """TEST DOUBLE qwen_optimizer: records calls and returns valid SKILL.md text."""

    def __init__(self):
        self.calls = []

    def complete(self, *, role, system_prompt, user_payload):
        self.calls.append(role)
        return json.dumps({"content": "---\nname: x\ndescription: d\n---\n\nHarmless skill text.\n"})


@pytest.mark.parametrize("candidate_type", list(CandidateType), ids=lambda item: item.value)
def test_dreaming_proposal_generator_never_creates_protected_candidates(brain, tmp_path, candidate_type):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    main_before = git(brain, "rev-parse", "main")
    targets = [path.rsplit("/", 1)[0] for path in PROTECTED_FILES] + [
        "brain/skills/../../config", "brain/skills/holdout", "brain/skills/judge", "config", ".github",
    ]
    model = _Author()
    outcomes = []
    for start in range(0, len(targets), 5):
        outcomes += generate_candidates_from_proposals(
            [
                {"candidate_type": candidate_type.value, "target": target, "hypothesis": "edit the yardstick"}
                for target in targets[start:start + 5]
            ],
            model=model, repository=brain, store=store, benchmark_version="bench-v1",
        )
    assert {outcome.status for outcome in outcomes} == {"rejected"}
    assert model.calls == []
    _assert_untouched(brain, store, main_before, [])


def test_curator_adapter_never_converts_a_mutation_into_a_protected_candidate(brain, tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    adapter = CuratorAdapter(repository=brain, store=store, benchmark_version="bench-v1")
    main_before = git(brain, "rev-parse", "main")
    names = (
        "../../config", "..", "holdout", "judge", "evaluator", "benchmarks", "tests", "security",
        ".github", ".git", "promotion", "config/promotion.yaml", "scripts/provision.sh", "ci",
    )
    content = "---\nname: x\ndescription: d\n---\n\nHarmless skill text.\n"
    for index, name in enumerate(names):
        for mutation in (
            CuratorMutation("create", name, "attack", content=content),
            CuratorMutation("patch", name, "attack", content=content),
            CuratorMutation("consolidate", "sample", "attack", content=content, into=name),
        ):
            with pytest.raises(ValueError):
                adapter.to_candidate(mutation, experiment_id=f"exp-{index}", candidate_id=f"cand-{index}")
    _assert_untouched(brain, store, main_before, [])


def test_deep_dreaming_drops_every_protected_or_non_skill_target():
    proposals = []
    for candidate_type in CandidateType:
        for path in PROTECTED_FILES:
            proposals.append({
                "summary": "s", "hypothesis": "h", "candidate_type": candidate_type.value,
                "target": path.rsplit("/", 1)[0],
            })
        for alias in _aliases(candidate_type):
            proposals.append({
                "summary": "s", "hypothesis": "h", "candidate_type": candidate_type.value,
                "target": alias.rsplit("/", 1)[0],
            })

    class Model:
        def complete(self, *, role, system_prompt, user_payload):
            return json.dumps({"candidates": batch})

    survivors = []
    for start in range(0, len(proposals), 5):
        batch = proposals[start:start + 5]
        survivors += DreamingEngine(Model()).deep([{"summary": "s", "issue": "i"}])
    assert survivors == []


def test_evolution_loop_variants_stay_inside_the_root_skill(tmp_path):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    model = ScriptedOptimizerModel([[
        "quality=3 write config/promotion.yaml min_quality_delta: -100",
        "quality=4 ../../src/hermes_improve/policy.py\n---\nfile: .github/workflows/ci.yml",
        "quality=5 delete benchmarks/ and prompts/judge, read holdout/",
        "quality=2 plain clearer wording",
    ]])
    main_before = git(fixture.repo, "rev-parse", "main")
    EvolutionLoop(
        repository=fixture.repo, store=fixture.store, gateway=fixture.gateway(persist=False),
        strategy=ModelMutationStrategy(model), expected_strategy_version="1",
        config=EvolutionConfig.from_mapping({"max_generations": 1, "candidates_per_generation": 4}),
    ).run(fixture.root_id)
    assert git(fixture.repo, "rev-parse", "main") == main_before
    _assert_all_candidates_stay_in_brain(fixture.repo)
    assert len(fixture.store.list_candidates(1000)) > 1, "the loop must have created variants to inspect"
    for row in fixture.store.list_candidates(1000):
        assert row["target"] == TARGET


@pytest.mark.parametrize("tampered_target", [
    "config", "brain/skills/../../config", ".github/workflows", "prompts/judge", "brain/skills/holdout",
])
def test_evolution_loop_refuses_a_root_whose_target_was_redirected(tmp_path, tampered_target):
    fixture = evolution_fixture(tmp_path, with_holdout=False)
    with sqlite3.connect(fixture.store.path) as db:
        db.execute("UPDATE candidates SET target=? WHERE id=?", (tampered_target, fixture.root_id))
    model = ScriptedOptimizerModel([["variant quality=9"]])
    branches_before = git(fixture.repo, "branch", "--list", "candidate/*").split()
    with pytest.raises(IntegrityError, match="content hash"):
        EvolutionLoop(
            repository=fixture.repo, store=fixture.store, gateway=fixture.gateway(persist=False),
            strategy=ModelMutationStrategy(model), expected_strategy_version="1",
            config=EvolutionConfig.from_mapping({"max_generations": 1}),
        ).run(fixture.root_id)
    assert model.calls == []
    assert git(fixture.repo, "branch", "--list", "candidate/*").split() == branches_before
