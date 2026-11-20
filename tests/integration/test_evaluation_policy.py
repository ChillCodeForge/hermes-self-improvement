from __future__ import annotations

import json
import math

import pytest

from openai_stub import OpenAIStub
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.errors import IntegrityError
from hermes_improve.evaluator import PairedEvaluator
from hermes_improve.judge import PairwiseJudge
from hermes_improve.metrics import arm_metrics, efficiency_report
from hermes_improve.models import Benchmark, Candidate, CandidateType, Role, RunObservation
from hermes_improve.policy import PromotionPolicy
from hermes_improve.qwen import OpenAICompatibleQwen, QwenConfig
from hermes_improve.run_policy import RunPolicy
from hermes_improve.security import AccessController
from hermes_improve.statistics import cohens_dz, hedges_gz, paired_summary


def record(identifier, *, split="REGRESSION", suites=("SMOKE", "CORE", "FULL"), deterministic=True, subjective=()):
    return {
        "schema_version": 1, "id": identifier, "version": 1,
        "source": {"type": "test", "provenance": "unit-test"},
        "task": {"title": identifier, "prompt": f"Explain incident {identifier}"},
        "taxonomy": {"categories": ["writing"], "difficulty": "easy", "required_capabilities": []},
        "environment": {"type": "fixture"},
        "eligibility": {"class": "HARD_BENCHMARK", "reproducibility_score": 1.0, "replay_verified": True},
        "verification": {"deterministic": ["exit_code"] if deterministic else [], "subjective": list(subjective)},
        "ground_truth": (
            {"type": "structured", "confidence": 1.0, "structured_result": {"incident": identifier}}
            if deterministic else {"type": "llm_judge", "confidence": 0.9}
        ),
        "benchmark": {"split": split, "suites": list(suites), "cluster_id": identifier, "criticality": "normal"},
        "security": {"sanitized": True, "contains_secrets": False},
    }


def store_with(tmp_path, records, holdout_records=()):
    root = tmp_path / "benchmarks"
    root.mkdir()
    for item in records:
        (root / f"{item['id']}.json").write_text(json.dumps(item))
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    for item in holdout_records:
        (holdout / f"{item['id']}.json").write_text(json.dumps(item))
    access = AccessController()
    return BenchmarkStore(root, access, holdout_root=holdout), access


class Runner:
    def __init__(self, *, llm=True, candidate_pass=True, outputs=None, **overrides):
        self.llm_influenced = llm
        self.candidate_pass = candidate_pass
        self.outputs = outputs or {}
        self.overrides = overrides
        self.calls = []

    def run(self, commit, task, seed):
        self.calls.append((commit, task.id, seed))
        is_candidate = commit == "candidate"
        passed = self.candidate_pass or not is_candidate
        values = dict(
            task_id=task.id, commit=commit, seed=seed, environment_digest="env",
            success=passed, quality=(80.0 + seed) if is_candidate else 70.0,
            tokens=90 if is_candidate else 100, latency_ms=10.0, deterministic_pass=passed,
            security_pass=True, tool_calls=3, final_output=self.outputs.get(commit, f"{commit} output"),
            verifier_results=(({"type": "exit_code", "passed": passed},) if task.verification["deterministic"] else ()),
            model_config_digest="model", limits_digest="limits",
        )
        if is_candidate:
            values.update(self.overrides)
        return RunObservation(**values)


def candidate():
    return Candidate("cand", "exp", CandidateType.SKILL_PATCH, "brain/skills/x", "base", "candidate", "diff")


def evaluate(store, runner, suite="SMOKE", ids=("task",), judge=None, **kwargs):
    return PairedEvaluator(store, runner, PromotionPolicy(), judge=judge).evaluate(
        candidate(), baseline_commit="base", benchmark_ids=list(ids), suite=suite,
        benchmark_version=store.version, **kwargs,
    )


@pytest.mark.parametrize(
    ("suite", "llm", "split", "expected"),
    [
        ("SMOKE", False, "REGRESSION", 1), ("FULL", False, "REGRESSION", 1),
        ("SMOKE", True, "REGRESSION", 2), ("CORE", True, "REGRESSION", 3),
        ("FULL", True, "REGRESSION", 5), ("FULL", True, "SEALED_HOLDOUT", 5),
    ],
)
def test_run_policy_counts(suite, llm, split, expected):
    benchmark = Benchmark.from_dict(record("t", split=split))
    assert RunPolicy().runs_for(benchmark, suite, llm_influenced=llm) == expected


@pytest.mark.parametrize(
    "kwargs", [{"smoke_runs": 3}, {"core_runs": 2}, {"full_runs": 4}, {"holdout_runs": 1}, {"deterministic_runs": 2}],
)
def test_run_policy_rejects_counts_below_spec(kwargs):
    with pytest.raises(ValueError):
        RunPolicy(**kwargs)


def test_evaluator_derives_identical_seed_sets_for_both_arms(tmp_path):
    store, _ = store_with(tmp_path, [record("task")])
    runner = Runner()
    report = evaluate(store, runner, suite="CORE")
    assert report.metrics["run_policy"]["runs_per_task"] == {"task": 3}
    assert runner.calls == [
        (commit, "task", seed) for seed in (0, 1, 2) for commit in ("base", "candidate")
    ]
    assert report.outcomes[0].run_count == 3
    assert report.metrics["evaluation_schema_version"] == 2


def test_runner_without_declaration_is_treated_as_llm_influenced(tmp_path):
    store, _ = store_with(tmp_path, [record("task")])

    class Undeclared(Runner):
        pass

    runner = Undeclared()
    del runner.llm_influenced
    with pytest.raises(ValueError, match="at least 5"):
        evaluate(store, runner, suite="FULL", seeds=[1])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_config_digest": "other"}, "model configurations"),
        ({"limits_digest": "other"}, "sandbox limits"),
        ({"verifier_results": ()}, "did not report deterministic verifier results"),
        ({"deterministic_pass": False}, "contradicts"),
        ({"failed_tool_calls": 9}, "more failed tool calls"),
    ],
)
def test_evaluator_rejects_unpaired_conditions_and_inconsistent_observations(tmp_path, override, message):
    store, _ = store_with(tmp_path, [record("task")])
    with pytest.raises(IntegrityError, match=message):
        evaluate(store, Runner(**override))


def test_report_contains_paired_deltas_ci_and_effect_size(tmp_path):
    store, _ = store_with(tmp_path, [record("a"), record("b"), record("c")])
    report = evaluate(store, Runner(llm=False), ids=("a", "b", "c"), seeds=[1])
    quality = report.metrics["paired_statistics"]["quality"]
    assert quality["n"] == 3 and quality["mean_delta"] == pytest.approx(11.0)
    assert quality["delta_ci95"] == [pytest.approx(11.0), pytest.approx(11.0)]
    assert quality["magnitude"] == "constant_shift"
    assert "p_value" not in quality


def test_effect_sizes():
    deltas = [1.0, 2.0, 3.0, 4.0]
    dz = cohens_dz(deltas)
    assert dz == pytest.approx(2.5 / 1.2909944, rel=1e-6)
    assert hedges_gz(deltas) == pytest.approx(dz * (1 - 3 / 11))
    assert cohens_dz([1.0]) is None and cohens_dz([2.0, 2.0]) is None
    summary = paired_summary([1.0, 1.0, 1.0], [1.5, 3.0, 2.0])
    assert summary["magnitude"] == "large" and summary["delta_ci95"][0] < summary["mean_delta"]
    with pytest.raises(ValueError):
        paired_summary([1.0], [math.nan])


def observation(commit, seed, *, success, tokens, reported=True):
    return RunObservation(
        task_id="t", commit=commit, seed=seed, environment_digest="e", success=success, quality=1.0,
        tokens=tokens, latency_ms=10.0, deterministic_pass=success, security_pass=True, tool_calls=2,
        token_usage_reported=reported,
    )


def test_cheaper_failure_is_never_reported_as_efficiency_gain():
    pairs = [
        (observation("b", 1, success=True, tokens=100), observation("c", 1, success=False, tokens=10)),
        (observation("b", 2, success=True, tokens=100), observation("c", 2, success=True, tokens=90)),
    ]
    report = efficiency_report(pairs)
    assert report["cheaper_failures"] == [{"task_id": "t", "seed": 1}]
    assert report["success_not_lower"] is False
    assert report["efficiency_deltas"]["tokens_per_success_ratio"] is None
    assert report["candidate"]["tokens_per_success"] == 100.0
    assert arm_metrics([observation("b", 1, success=False, tokens=5)])["tokens_per_success"] is None
    assert arm_metrics([observation("b", 1, success=True, tokens=5, reported=False)])["tokens_per_task"] is None


def judge_verdict(winner):
    return json.dumps({"winner": winner, "confidence": 0.9, "uncertainty": 0.1, "rationale": "r"})


def judge_preferring(output):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        return judge_verdict("A" if payload["response_A"] == output else "B")
    return reply


def judge_for(stub):
    return PairwiseJudge(OpenAICompatibleQwen(QwenConfig(stub.base_url, "m", judge_model="judge")))


def test_deterministic_verifiers_take_precedence_over_judge(tmp_path):
    store, _ = store_with(tmp_path, [record("task", subjective=["clear explanation"])])
    with OpenAIStub(judge_preferring("candidate output")) as stub:
        report = evaluate(store, Runner(llm=False, candidate_pass=False), judge=judge_for(stub), seeds=[1])
    assert stub.requests == [], "judge must not be consulted when deterministic verifiers exist"
    assert report.outcomes[0].judge_result == ""
    assert not report.outcomes[0].deterministic_pass


def test_subjective_task_uses_blind_judge_and_baseline_preference_blocks(tmp_path):
    store, _ = store_with(tmp_path, [record("essay", deterministic=False, subjective=["names the root cause"])])
    with OpenAIStub(judge_preferring("base output")) as stub:
        report = evaluate(store, Runner(), ids=("essay",), judge=judge_for(stub))
    assert len(stub.requests) == 4, "two seeds x two swapped presentations"
    assert report.outcomes[0].judge_result == "baseline"
    assert "judge preferred baseline on subjective task essay" in report.reasons
    assert not report.passed


def test_judge_preference_alone_passes_only_into_the_manual_approval_gate(tmp_path):
    holdout = record("essay-holdout", split="SEALED_HOLDOUT", suites=("SEALED_HOLDOUT",), deterministic=False,
                     subjective=["names the root cause"])
    # A distinct prompt: near-duplicate tasks across splits are rejected as leakage.
    holdout["task"]["prompt"] = "Summarize why the nightly backup job silently skipped encrypted volumes"
    store, access = store_with(
        tmp_path, [record("essay", deterministic=False, subjective=["names the root cause"])], [holdout],
    )

    class EqualQuality(Runner):
        def run(self, commit, task, seed):
            result = super().run(commit, task, seed)
            from dataclasses import replace
            return replace(result, quality=70.0)

    with OpenAIStub(judge_preferring("candidate output")) as stub:
        report = evaluate(
            store, EqualQuality(), suite="FULL", ids=("essay", "essay-holdout"), judge=judge_for(stub),
            holdout_grant=access.evaluator_holdout_grant(Role.EVALUATOR),
        )
    assert {outcome.judge_result for outcome in report.outcomes} == {"candidate"}
    # Judged tasks no longer count as quality delta 0; the gain rests on the
    # judge alone, so the evaluation passes but the promoter requires a signed
    # operator approval (policy metrics, F27).
    assert report.passed, report.reasons
    assert report.metrics["quality_evidence"] == "judge"
    assert report.metrics["quality_tasks"] == 0 and report.metrics["judged_final_tasks"] == 2
    assert "operator approval" in report.metrics["manual_approval_reason"]


def test_unreliable_judge_blocks_and_missing_judge_fails_closed(tmp_path):
    store, _ = store_with(tmp_path, [record("essay", deterministic=False, subjective=["accurate"])])
    with OpenAIStub(lambda body: judge_verdict("B")) as stub:
        report = evaluate(store, Runner(), ids=("essay",), judge=judge_for(stub))
    assert report.outcomes[0].judge_result == "unreliable"
    assert "subjective judgment unreliable on task essay" in report.reasons
    runner = Runner()
    with pytest.raises(ValueError, match="requires a configured judge"):
        evaluate(store, runner, ids=("essay",))
    assert runner.calls == []


def test_outcome_records_whether_both_arms_reported_token_usage(tmp_path):
    store, _ = store_with(tmp_path, [record("task")])
    measured = evaluate(store, Runner(), suite="CORE")
    assert measured.outcomes[0].token_usage_reported is True
    # One arm without token usage makes the whole task row unmeasured.
    unmeasured = evaluate(store, Runner(token_usage_reported=False), suite="CORE")
    assert unmeasured.outcomes[0].token_usage_reported is False


def test_unmeasured_token_usage_reaches_the_policy_as_a_blocking_reason(tmp_path):
    store, access = store_with(
        tmp_path, [record("full-task", suites=("FULL",))], [record("holdout-task", split="SEALED_HOLDOUT")],
    )
    report = evaluate(
        store, Runner(token_usage_reported=False, quality=90.0), suite="FULL", ids=("full-task", "holdout-task"),
        holdout_grant=access.evaluator_holdout_grant(Role.EVALUATOR),
    )
    assert not report.passed
    assert "token efficiency not measured: runner reported no token usage" in report.reasons
    assert report.metrics["token_check"] == "not measured"


def test_cheaper_failure_ignores_unmeasured_token_counts():
    pairs = [(
        observation("b", 1, success=True, tokens=0, reported=False),
        observation("c", 1, success=False, tokens=0, reported=False),
    )]
    assert efficiency_report(pairs)["cheaper_failures"] == []


def file_task():
    return record("incident-file", deterministic=False, subjective=[
        {"criterion": "Starts with an INCIDENT headline.", "files": ["incident.md"]},
    ])


class FileRunner(Runner):
    """Both arms reply "done"; only the deliverable file differs (lab finding F24)."""

    def __init__(self, files=None, **kwargs):
        super().__init__(outputs={"base": "done", "candidate": "done"}, **kwargs)
        self.files = files

    def run(self, commit, task, seed):
        from dataclasses import replace
        content = "INCIDENT: api - 503" if commit == "candidate" else "something broke"
        files = self.files if self.files is not None else (
            {"path": "incident.md", "status": "ok", "bytes": len(content), "sha256": "", "content": content},
        )
        return replace(super().run(commit, task, seed), deliverables=files)


def judge_preferring_file(content):
    def reply(body):
        payload = json.loads(body["messages"][1]["content"])
        return judge_verdict("A" if payload["files_A"][0]["content"] == content else "B")
    return reply


def test_judge_grades_declared_deliverable_files_not_only_the_chat_reply(tmp_path):
    store, _ = store_with(tmp_path, [file_task()])
    with OpenAIStub(judge_preferring_file("INCIDENT: api - 503")) as stub:
        report = evaluate(store, FileRunner(), ids=("incident-file",), judge=judge_for(stub))
    assert report.outcomes[0].judge_result == "candidate"
    assert not any("unreliable" in reason for reason in report.reasons)
    payload = json.loads(stub.requests[0]["messages"][1]["content"])
    assert payload["response_A"] == payload["response_B"] == "done"
    assert {payload["files_A"][0]["content"], payload["files_B"][0]["content"]} == {
        "INCIDENT: api - 503", "something broke",
    }
    stored = [row.deliverables for row in report.run_observations]
    assert all(row and row[0]["path"] == "incident.md" for row in stored)


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ((), "did not capture the deliverable files"),
        (({"path": "incident.md", "status": "ok", "bytes": 1, "sha256": "", "content": "x", "x": 1},),
         "malformed deliverable files"),
    ],
)
def test_runner_must_report_exactly_the_declared_deliverables(tmp_path, files, message):
    store, _ = store_with(tmp_path, [file_task()])
    with OpenAIStub(lambda body: judge_verdict("A")) as stub:
        with pytest.raises(IntegrityError, match=message):
            evaluate(store, FileRunner(files=files), ids=("incident-file",), judge=judge_for(stub))
    assert stub.requests == []
