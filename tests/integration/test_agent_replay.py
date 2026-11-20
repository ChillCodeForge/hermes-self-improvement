"""Replay evidence for LLM-driven `hermes_agent` benchmarks (lab finding F14).

`HARD_BENCHMARK` needs `replay_verified`, and `benchmarks validate` needs
`eligibility.replay_evidence` for it; before F14 only command benchmarks could
be replayed, so no `hermes_agent` record could ever be lint-clean.
"""

from __future__ import annotations

import dataclasses
import json
import sys

import pytest

from hermes_improve.benchmark_schema import parse_benchmark_document
from hermes_improve.benchmarks import lint_public_corpus
from hermes_improve.errors import IntegrityError
from hermes_improve.models import Benchmark, RunObservation
from hermes_improve.replay import replay_benchmark, replay_kind
from hermes_improve.store import ExperimentStore
from hermes_improve.sandbox import LocalSandboxExecutor
from test_hermes_runner import benchmark_record, brain_repo, hermes_install

COMMIT = "c" * 40
MODEL_DIGEST = "d" * 64


class FakeAgentRunner:
    """LLM-influenced `BenchmarkRunner` with scripted verdicts per seed."""

    llm_influenced = True

    def __init__(self, verdict=lambda seed: (True, "completed"), model_digest=lambda seed: MODEL_DIGEST):
        self.verdict = verdict
        self.model_digest = model_digest
        self.calls: list[int] = []

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        self.calls.append(seed)
        verifiers_pass, exit_state = self.verdict(seed)
        success = verifiers_pass and exit_state == "completed"
        return RunObservation(
            task_id=benchmark.id, commit=commit, seed=seed, environment_digest="e" * 64,
            success=success, quality=100.0 if verifiers_pass else 0.0, tokens=100 + seed,
            latency_ms=1000.0 + seed, deterministic_pass=verifiers_pass, security_pass=True,
            exit_code=0 if exit_state == "completed" else 1, final_output=f"different text {seed}",
            exit_state=exit_state, model_config_digest=self.model_digest(seed), tool_calls=seed,
        )


def agent_record() -> dict:
    """A `hermes_agent` record shaped for the curated corpus (taxonomy and score lint)."""
    record = benchmark_record("agent-task")
    record["taxonomy"]["categories"] = ["coding"]
    record["eligibility"]["reproducibility_score"] = 0.8
    return record


def agent_benchmark(split: str = "TRAIN") -> Benchmark:
    record = agent_record()
    record["benchmark"]["split"] = split
    return Benchmark.from_dict(record)


def test_hermes_agent_benchmarks_are_replayable_and_typed():
    assert replay_kind(agent_benchmark()) == "hermes_agent"
    plain = benchmark_record("plain")
    plain["environment"] = {"container_image": plain["environment"]["container_image"]}
    assert replay_kind(Benchmark.from_dict(plain)) is None


def test_agent_replay_requires_every_run_to_pass_and_records_the_runner():
    runner = FakeAgentRunner()
    result = replay_benchmark(runner, agent_benchmark(), commit=COMMIT, runs=3, runner_name="hermes",
                              model="qwen3-coder")
    assert runner.calls == [0, 1, 2]
    # Token counts, latency, tool calls and wording differ between LLM runs; the verdict does not.
    assert result.reproducible and result.passed and result.replay_verified
    evidence = result.evidence()
    assert evidence["runner"] == "hermes"
    assert evidence["llm_influenced"] is True
    assert evidence["criterion"] == "all_runs_pass"
    assert evidence["model"] == "qwen3-coder"
    assert evidence["model_config_digest"] == MODEL_DIGEST
    assert result.to_dict()["benchmark_kind"] == "hermes_agent"

    record = agent_record()
    record["eligibility"]["replay_evidence"] = evidence
    parsed = parse_benchmark_document(record)
    assert parsed.eligibility["replay_evidence"]["runner"] == "hermes"
    lint_public_corpus([parsed])  # F14: the record is now lint-clean


@pytest.mark.parametrize(
    "verdict",
    [
        lambda seed: (True, "timeout" if seed == 1 else "completed"),  # workspace right, run not finished
        lambda seed: (seed != 2, "completed"),  # one of three runs fails the verifiers
        lambda seed: (False, "completed"),  # consistently failing
    ],
)
def test_agent_replay_is_not_verified_unless_every_run_completes_and_passes(verdict):
    result = replay_benchmark(FakeAgentRunner(verdict=verdict), agent_benchmark(), commit=COMMIT, runs=3)
    assert not result.passed and not result.replay_verified


def test_agent_replay_rejects_a_model_change_between_runs():
    runner = FakeAgentRunner(model_digest=lambda seed: f"{seed:064x}")
    with pytest.raises(IntegrityError, match="model configuration"):
        replay_benchmark(runner, agent_benchmark(), commit=COMMIT, runs=2)


def test_agent_replay_never_touches_the_sealed_holdout():
    runner = FakeAgentRunner()
    with pytest.raises(ValueError, match="sealed holdout"):
        replay_benchmark(runner, agent_benchmark("SEALED_HOLDOUT"), commit=COMMIT, runs=2)
    assert runner.calls == []


def test_replay_evidence_schema_stays_compatible_and_typed():
    record = agent_record()
    base = replay_benchmark(FakeAgentRunner(), agent_benchmark(), commit=COMMIT, runs=2).evidence()
    legacy = {key: base[key] for key in (
        "commit", "runs", "reproducible", "passed", "outcome_digest", "environment_digest", "replayed_at",
    )}
    record["eligibility"]["replay_evidence"] = legacy
    parse_benchmark_document(record)
    record["eligibility"]["replay_evidence"] = legacy | {"model_config_digest": "not-a-digest"}
    with pytest.raises(ValueError):
        parse_benchmark_document(record)


def _cli_setup(tmp_path, monkeypatch, fake=None):
    repo, _baseline, candidate = brain_repo(tmp_path)
    agent, log = hermes_install(tmp_path, **(fake or {}))
    fixtures = tmp_path / "fixtures"
    (fixtures / "repo").mkdir(parents=True)
    (fixtures / "repo" / "fixture.txt").write_text("input\n")
    public = tmp_path / "public" / "train"
    public.mkdir(parents=True)
    (public / "agent.json").write_text(json.dumps(agent_record()))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(tmp_path / "public"))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    argv = [
        "hermes-improve", "benchmarks", "replay", "--commit", candidate, "--repository", str(repo),
        "--runs", "2", "--runner", "hermes", "--hermes-agent-dir", str(agent),
        "--hermes-python", sys.executable, "--model", "qwen3-coder",
        "--model-base-url", "http://127.0.0.1:9/v1", "--fixtures-root", str(fixtures),
    ]
    return argv, log


def test_cli_replays_hermes_agent_benchmarks_through_the_selected_runner(tmp_path, monkeypatch, capsys):
    from hermes_improve import cli, hermes_runner

    argv, log = _cli_setup(tmp_path, monkeypatch)
    # The fake checkout writes into the task cwd itself; no Docker terminal or verifier sandbox is involved.
    original = hermes_runner.HermesBatchRunner.__init__

    def local_terminal(self, *args, **kwargs):
        kwargs["executor_factory"] = lambda benchmark: LocalSandboxExecutor(acknowledge_no_isolation=True)
        original(self, *args, **kwargs, terminal_env="local", allow_local_terminal=True)

    monkeypatch.setattr(hermes_runner.HermesBatchRunner, "__init__", local_terminal)
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["replayed"] == 1 and report["replay_verified"] == 1
    (result,) = report["results"]
    assert result["benchmark_id"] == "agent-task" and result["benchmark_kind"] == "hermes_agent"
    assert result["runner"] == "hermes" and result["model"] == "qwen3-coder"
    assert len(result["runs"]) == 2
    assert len(log.read_text().splitlines()) == 2  # two real driver invocations
    stored = ExperimentStore(tmp_path / "home" / "experiments.sqlite3").list_benchmark_replays("agent-task")
    assert stored == [result]

    record = agent_record()
    record["eligibility"]["replay_evidence"] = {
        key: result[key] for key in (
            "commit", "reproducible", "passed", "outcome_digest", "environment_digest", "replayed_at",
            "runner", "llm_influenced", "criterion", "model", "model_config_digest",
        )
    } | {"runs": len(result["runs"])}
    lint_public_corpus([parse_benchmark_document(record)])


def test_cli_replay_selects_benchmarks_matching_the_runner(tmp_path, monkeypatch):
    from hermes_improve import cli

    argv, _log = _cli_setup(tmp_path, monkeypatch)
    calls = []

    def fake_build_runner(args, artifact_store):
        calls.append((args.runner, args.benchmark_root, args.holdout_root))
        return FakeAgentRunner()

    monkeypatch.setattr(cli, "build_runner", fake_build_runner)
    # The default docker runner has no command benchmark to replay in this corpus.
    docker_argv = [item for item in argv if item not in {"--runner", "hermes"}]
    monkeypatch.setattr(sys, "argv", docker_argv)
    with pytest.raises(ValueError, match="no public command benchmarks"):
        cli.main()
    monkeypatch.setattr(sys, "argv", [*argv[:3], "--id", "unknown-task", *argv[3:]])
    with pytest.raises(ValueError, match="no public hermes_agent benchmark"):
        cli.main()
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    # The replay never configures the sealed holdout root.
    assert calls[-1] == ("hermes", tmp_path / "public", None)


def test_benchmark_kind_is_part_of_the_result_payload():
    result = replay_benchmark(FakeAgentRunner(), agent_benchmark(), commit=COMMIT, runs=2)
    payload = result.to_dict()
    assert payload["llm_influenced"] is True and payload["criterion"] == "all_runs_pass"
    assert dataclasses.replace(result, runner="x").evidence()["runner"] == "x"
