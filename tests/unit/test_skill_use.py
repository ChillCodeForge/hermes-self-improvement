"""Skill-use measurement (lab finding F30): which skills the agent loaded per run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.hermes_runner import parse_trajectory_record
from hermes_improve.models import Candidate, CandidateType, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.security import AccessController
from hermes_improve.skill_use import (
    MAX_SKILL_NAME_CHARS,
    MAX_SKILLS_PER_RUN,
    skill_loads,
    target_skill_aliases,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "hermes"


def gpt_turn(*calls):
    return {"from": "gpt", "value": "<think>\n</think>\n" + "\n".join(
        "<tool_call>\n" + (call if isinstance(call, str) else json.dumps(call)) + "\n</tool_call>" for call in calls
    )}


def record_with(*turns):
    record = json.loads((FIXTURES / "trajectories.jsonl").read_text())
    record["conversations"][2:2] = list(turns)
    return record


def test_trajectory_records_skill_view_names_and_skills_list_calls():
    trajectory = parse_trajectory_record(record_with(
        gpt_turn({"name": "skills_list", "arguments": {}}, {"name": "skill_view", "arguments": {"name": "answer"}}),
        gpt_turn(
            {"name": "skill_view", "arguments": {"name": "answer", "file_path": "references/api.md"}},
            # Some models emit the arguments as a JSON string.
            {"name": "skill_view", "arguments": json.dumps({"name": "ops/deploy"})},
            {"name": "skills_list", "arguments": {"category": "ops"}},
        ),
    ))
    assert trajectory.skills_viewed == ("answer", "ops/deploy")
    assert trajectory.skills_list_calls == 2


def test_trajectory_without_skill_calls_records_no_skill_use():
    trajectory = parse_trajectory_record(json.loads((FIXTURES / "trajectories.jsonl").read_text()))
    assert trajectory.skills_viewed == () and trajectory.skills_list_calls == 0


def test_lab_trajectories_of_f30_show_no_skill_use():
    # The stored S6 trajectories (qwen3:4b-instruct) never call skill_view or skills_list.
    for line in (FIXTURES / "s6_lab_trajectories.jsonl").read_text().splitlines():
        trajectory = parse_trajectory_record(json.loads(line))
        assert trajectory.skills_viewed == () and trajectory.skills_list_calls == 0


def test_skill_use_is_bounded_and_malformed_calls_are_ignored_not_fatal():
    names = [f"skill-{index:02d}" for index in range(MAX_SKILLS_PER_RUN + 8)]
    conversations = [
        gpt_turn(*({"name": "skill_view", "arguments": {"name": name}} for name in names)),
        gpt_turn(
            "{not json",
            {"name": "skill_view", "arguments": {"name": "x" * (MAX_SKILL_NAME_CHARS + 1)}},
            {"name": "skill_view", "arguments": {"name": 7}},
            {"name": "skill_view", "arguments": {"name": "  "}},
            {"name": "skill_view", "arguments": "[1, 2]"},
            {"name": "skill_view"},
            ["not", "an", "object"],
        ),
        # Tool calls are only read from assistant turns.
        {"from": "tool", "value": "<tool_call>\n" + json.dumps({"name": "skill_view", "arguments": {"name": "tool"}})
         + "\n</tool_call>"},
    ]
    viewed, listed = skill_loads(conversations)
    assert viewed == tuple(names[:MAX_SKILLS_PER_RUN])
    assert listed == 0


@pytest.mark.parametrize(
    ("candidate_type", "target", "aliases"),
    [
        (CandidateType.SKILL_CREATE, "brain/skills/answer", {"answer"}),
        (CandidateType.SKILL_PATCH, "brain/skills/ops/deploy", {"ops/deploy", "deploy"}),
        (CandidateType.SKILL_MERGE, "brain/skills/a,brain/skills/b", {"a", "b"}),
        (CandidateType.SKILL_PATCH, "skill/a", None),
        (CandidateType.ROUTING, "brain/routing/main.yaml", None),
    ],
)
def test_target_skill_aliases(candidate_type, target, aliases):
    subject = Candidate("c", "e", candidate_type, target, "base", "cand", "diff")
    result = target_skill_aliases(subject)
    assert (None if result is None else set(result)) == aliases


# --- evaluator plumbing -------------------------------------------------------------------------


def benchmark(identifier, split="TRAIN"):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": f"Complete task {identifier}"},
        "taxonomy": {"categories": ["test"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": ["unit-test"], "subjective": []},
        "ground_truth": {"type": "test", "confidence": 1.0},
        "benchmark": {"split": split, "suites": ["SMOKE"], "cluster_id": identifier, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }


class SkillRunner:
    """Returns runs whose loaded skills come from a per-(commit, task, seed) script."""

    llm_influenced = False

    def __init__(self, loads, *, observed=True, listed=0):
        self.loads = loads
        self.observed = observed
        self.listed = listed

    def run(self, commit, task, seed):
        return RunObservation(
            task_id=task.id, commit=commit, seed=seed, environment_digest="env",
            success=True, quality=1.0, tokens=10, latency_ms=1.0, deterministic_pass=True, security_pass=True,
            tool_calls=3, verifier_results=({"type": "unit-test", "passed": True},),
            skills_viewed=tuple(self.loads.get((commit, task.id, seed), ())),
            skills_list_calls=self.listed,
            skill_use_observed=self.observed,
        )


def store(tmp_path, *identifiers):
    root = tmp_path / "benchmarks"
    root.mkdir()
    for identifier in identifiers:
        (root / f"{identifier}.json").write_text(json.dumps(benchmark(identifier)))
    return BenchmarkStore(root, AccessController())


def evaluate(tmp_path, runner, target="brain/skills/answer", candidate_type=CandidateType.SKILL_PATCH):
    benchmarks = store(tmp_path, "task-a", "task-b")
    subject = Candidate("cand-1", "exp-1", candidate_type, target, "base", "cand", "diff")
    return PairedEvaluator(benchmarks, runner, PromotionPolicy()).evaluate(
        subject, baseline_commit="base", benchmark_ids=["task-a", "task-b"], suite="SMOKE",
        benchmark_version=benchmarks.version, seeds=[1, 2],
    )


def test_evaluation_reports_per_arm_target_skill_loaded_rate(tmp_path):
    runner = SkillRunner({
        ("cand", "task-a", 1): ["answer"],
        ("cand", "task-a", 2): ["answer", "other"],
        ("cand", "task-b", 1): ["other"],
        ("base", "task-a", 1): ["answer"],
    }, listed=1)
    report = evaluate(tmp_path, runner)

    use = report.metrics["skill_use"]
    assert use["target_skill"] == ["answer"]
    assert use["candidate"]["runs"] == 4 and use["candidate"]["observed_runs"] == 4
    assert use["candidate"]["target_skill_loaded_runs"] == 2
    assert use["candidate"]["target_skill_loaded_rate"] == 0.5
    assert use["candidate"]["runs_with_skill_view"] == 3
    assert use["candidate"]["runs_with_skills_list"] == 4
    assert use["candidate"]["skills_viewed"] == {"answer": 2, "other": 2}
    assert use["baseline"]["target_skill_loaded_rate"] == 0.25
    rows = {row.task_id: row for row in report.outcomes}
    assert rows["task-a"].candidate_skill_loaded_rate == 1.0
    assert rows["task-a"].baseline_skill_loaded_rate == 0.5
    assert rows["task-b"].candidate_skill_loaded_rate == 0.0
    assert rows["task-b"].baseline_skill_loaded_rate == 0.0


def test_never_loaded_candidate_skill_reports_zero_not_unknown(tmp_path):
    report = evaluate(tmp_path, SkillRunner({}))
    assert report.metrics["skill_use"]["candidate"]["target_skill_loaded_rate"] == 0.0
    assert all(row.candidate_skill_loaded_rate == 0.0 for row in report.outcomes)


def test_skill_use_is_unknown_for_runners_without_trajectories(tmp_path):
    # Command runners (DockerCommandRunner) have no agent trajectory: unknown, never "0 % loaded".
    report = evaluate(tmp_path, SkillRunner({("cand", "task-a", 1): ["answer"]}, observed=False))
    use = report.metrics["skill_use"]
    assert use["candidate"]["observed_runs"] == 0
    assert use["candidate"]["target_skill_loaded_rate"] is None
    assert all(row.candidate_skill_loaded_rate is None for row in report.outcomes)


def test_non_skill_candidates_have_no_target_rate_but_still_report_skill_views(tmp_path):
    runner = SkillRunner({("cand", "task-a", 1): ["answer"]})
    report = evaluate(tmp_path, runner, target="brain/routing/main.yaml", candidate_type=CandidateType.ROUTING)
    use = report.metrics["skill_use"]
    assert use["target_skill"] is None
    assert use["candidate"]["target_skill_loaded_rate"] is None
    assert use["candidate"]["runs_with_skill_view"] == 1
    assert all(row.candidate_skill_loaded_rate is None for row in report.outcomes)


@pytest.mark.parametrize(
    "fields",
    [
        {"skills_viewed": ["answer"]},
        {"skills_viewed": ("answer", 3)},
        {"skills_viewed": tuple(f"s{index}" for index in range(MAX_SKILLS_PER_RUN + 1))},
        {"skills_list_calls": -1},
        {"skills_list_calls": True},
        {"skill_use_observed": "yes"},
    ],
)
def test_evaluator_rejects_malformed_skill_use(tmp_path, fields):
    class Broken(SkillRunner):
        def run(self, commit, task, seed):
            observation = super().run(commit, task, seed)
            return RunObservation(**{**observation.__dict__, **fields})

    with pytest.raises(IntegrityError, match="skill"):
        evaluate(tmp_path, Broken({}))


# --- operator views -------------------------------------------------------------------------------


def persisted_evaluation(tmp_path, runner, suite_seeds=(1, 2)):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from hermes_improve.store import ExperimentStore

    benchmarks = store(tmp_path, "task-a", "task-b")
    home = tmp_path / "home"
    home.mkdir()
    ledger = ExperimentStore(home / "experiments.sqlite3")
    ledger.create_experiment(
        "exp-1", lineage_id="lineage-1", hypothesis="skill use", candidate_type="SKILL_PATCH",
        target="brain/skills/answer", risk="LOW", baseline_commit="base",
        benchmark_version=benchmarks.version, policy_version="1",
    )
    subject = Candidate("cand-1", "exp-1", CandidateType.SKILL_PATCH, "brain/skills/answer", "base", "cand", "diff")
    ledger.add_candidate(subject)
    key = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )
    PairedEvaluator(benchmarks, runner, PromotionPolicy(), ledger, evaluator_signing_key=key).evaluate(
        subject, baseline_commit="base", benchmark_ids=["task-a", "task-b"], suite="SMOKE",
        benchmark_version=benchmarks.version, seeds=list(suite_seeds), evaluation_id="eval-smoke",
    )
    return home, ledger


def test_experiments_show_lists_skill_use_of_every_evaluation(tmp_path):
    home, ledger = persisted_evaluation(tmp_path, SkillRunner({("cand", "task-a", 1): ["answer"]}))
    shown = ledger.show_experiment("exp-1")
    assert shown["skill_use"] == [{
        "evaluation_id": "eval-smoke", "candidate_id": "cand-1", "suite": "SMOKE",
        "target_skill": ["answer"],
        "baseline_target_skill_loaded_rate": 0.0,
        "candidate_target_skill_loaded_rate": 0.25,
        "candidate_observed_runs": 4,
    }]


def test_report_counts_skill_evaluations_whose_skill_was_never_loaded(tmp_path):
    from hermes_improve.observability import build_report, render_text

    home, _ledger = persisted_evaluation(tmp_path, SkillRunner({}))
    improvement = build_report(home, tmp_path / "absent-benchmarks")["improvement"]
    assert improvement["skill_load"] == {
        "evaluations_with_target_skill": 1,
        "candidate_skill_never_loaded": 1,
        "candidate_target_skill_loaded_rate_mean": 0.0,
    }
    assert "skill_load" in render_text(build_report(home, tmp_path / "absent-benchmarks"))
