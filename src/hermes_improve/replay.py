"""Benchmark replay: reproducibility evidence for authored benchmarks (spec §13, §34).

A replay re-runs one benchmark several times at a fixed commit through a
`BenchmarkRunner`. The CLI selects the runner the way `evaluate` does: the
digest-pinned `DockerCommandRunner` for command benchmarks, the
`LocalSkillProcedureRunner` for `skill_procedure` benchmarks, or the
LLM-driven `HermesBatchRunner` for `hermes_agent` benchmarks.

The benchmark is reproducible when every run yields the identical verifier
outcome (exit code, deterministic pass/fail, success, quality, security pass,
environment digest), independent of the seed. A replay counts as verified only
when it is reproducible *and* it passed.

What "passed" means depends on whether a model is in the loop
(`runner.llm_influenced`, which defaults to true as in the evaluator):

* deterministic runners (`criterion: identical_outcome`): every run passed the
  deterministic verifiers, i.e. the fixture's reference solution satisfies its
  own ground truth;
* LLM-driven runners (`criterion: all_runs_pass`): every run passed the
  deterministic verifiers *and* finished as a successful agent run
  (`success`; a timeout or a partial trajectory never counts). Token counts,
  latency, tool calls and wording vary between LLM runs and are not part of
  the outcome; the verdict must not vary. All runs must use the same model
  configuration (`model_config_digest`), which the evidence records together
  with the runner name and model.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from .evaluator import BenchmarkRunner
from .errors import IntegrityError
from .models import Benchmark, RunObservation, Split, canonical_json, utc_now

MIN_REPLAY_RUNS = 2
MAX_REPLAY_RUNS = 20
_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

#: Replayable benchmark kinds (see `replay_kind`).
COMMAND = "command"
SKILL_PROCEDURE = "skill_procedure"
HERMES_AGENT = "hermes_agent"
#: Replay criteria recorded in the evidence.
CRITERION_DETERMINISTIC = "identical_outcome"
CRITERION_LLM = "all_runs_pass"


@dataclass(frozen=True)
class ReplayRun:
    seed: int
    exit_code: int
    deterministic_pass: bool
    quality: float
    outcome_digest: str
    artifact_digests: tuple[str, ...]
    success: bool = True
    exit_state: str = ""


@dataclass(frozen=True)
class ReplayResult:
    benchmark_id: str
    benchmark_version: int
    benchmark_digest: str
    commit: str
    environment_digest: str
    runs: tuple[ReplayRun, ...]
    reproducible: bool
    passed: bool
    replayed_at: str
    benchmark_kind: str = COMMAND
    runner: str = ""
    llm_influenced: bool = False
    model: str = ""
    model_config_digest: str = ""

    @property
    def criterion(self) -> str:
        return CRITERION_LLM if self.llm_influenced else CRITERION_DETERMINISTIC

    @property
    def replay_verified(self) -> bool:
        return self.reproducible and self.passed

    @property
    def outcome_digest(self) -> str:
        """Digest over every run's outcome; equals each run's digest when reproducible."""
        digests = sorted({run.outcome_digest for run in self.runs})
        if len(digests) == 1:
            return digests[0]
        return hashlib.sha256(canonical_json(digests).encode()).hexdigest()

    def _runner_identity(self) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "runner": self.runner, "llm_influenced": self.llm_influenced, "criterion": self.criterion,
        }
        if self.model:
            identity["model"] = self.model
        if self.model_config_digest:
            identity["model_config_digest"] = self.model_config_digest
        return identity

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_digest": self.benchmark_digest,
            "benchmark_kind": self.benchmark_kind,
            "commit": self.commit,
            "environment_digest": self.environment_digest,
            **self._runner_identity(),
            "runs": [
                {
                    "seed": run.seed,
                    "exit_code": run.exit_code,
                    "exit_state": run.exit_state,
                    "success": run.success,
                    "deterministic_pass": run.deterministic_pass,
                    "quality": run.quality,
                    "outcome_digest": run.outcome_digest,
                    "artifact_digests": list(run.artifact_digests),
                }
                for run in self.runs
            ],
            "reproducible": self.reproducible,
            "passed": self.passed,
            "replay_verified": self.replay_verified,
            "outcome_digest": self.outcome_digest,
            "replayed_at": self.replayed_at,
        }

    def evidence(self) -> dict[str, Any]:
        """`eligibility.replay_evidence` block for an authored benchmark record."""
        return {
            "commit": self.commit,
            "runs": len(self.runs),
            "reproducible": self.reproducible,
            "passed": self.passed,
            "outcome_digest": self.outcome_digest,
            "environment_digest": self.environment_digest,
            "replayed_at": self.replayed_at,
            **self._runner_identity(),
        }


def is_command_benchmark(benchmark: Benchmark) -> bool:
    command = benchmark.environment.get("command")
    return isinstance(command, list) and bool(command)


def replay_kind(benchmark: Benchmark) -> str | None:
    """`hermes_agent`, `skill_procedure`, `command`, or None when the record cannot be replayed."""
    environment_type = benchmark.environment.get("type")
    if environment_type == HERMES_AGENT:
        return HERMES_AGENT
    if environment_type == SKILL_PROCEDURE:
        return SKILL_PROCEDURE
    if is_command_benchmark(benchmark):
        return COMMAND
    return None


def observation_outcome_digest(observation: RunObservation) -> str:
    return hashlib.sha256(canonical_json({
        "exit_code": observation.exit_code,
        "deterministic_pass": observation.deterministic_pass,
        "success": observation.success,
        "quality": observation.quality,
        "security_pass": observation.security_pass,
        "environment_digest": observation.environment_digest,
    }).encode()).hexdigest()


def replay_benchmark(
    runner: BenchmarkRunner, benchmark: Benchmark, *, commit: str, runs: int,
    runner_name: str | None = None, model: str | None = None,
) -> ReplayResult:
    """Run `benchmark` `runs` times (seeds 0..runs-1) and compare verifier outcomes.

    `runner_name` and `model` are recorded in the evidence (defaults: the
    runner's class name, no model).
    """
    if benchmark.split is Split.SEALED_HOLDOUT:
        raise ValueError("sealed holdout benchmarks are not replayed through the public replay path")
    kind = replay_kind(benchmark)
    if kind is None:
        raise ValueError(
            f"benchmark {benchmark.id!r} is not a command, skill_procedure or hermes_agent benchmark"
        )
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("replay requires a full Git object ID")
    if isinstance(runs, bool) or not isinstance(runs, int) or not MIN_REPLAY_RUNS <= runs <= MAX_REPLAY_RUNS:
        raise ValueError(f"replay runs must be an integer between {MIN_REPLAY_RUNS} and {MAX_REPLAY_RUNS}")
    llm_influenced = getattr(runner, "llm_influenced", True)
    if not isinstance(llm_influenced, bool):
        raise TypeError("runner.llm_influenced must be a boolean")
    results: list[ReplayRun] = []
    environment_digests: set[str] = set()
    model_digests: set[str] = set()
    for seed in range(runs):
        observation = runner.run(commit, benchmark, seed)
        if (observation.task_id, observation.commit, observation.seed) != (benchmark.id, commit, seed):
            raise IntegrityError("runner returned an observation for a different task, commit or seed")
        environment_digests.add(observation.environment_digest)
        model_digests.add(observation.model_config_digest)
        results.append(ReplayRun(
            seed=seed,
            exit_code=observation.exit_code,
            deterministic_pass=observation.deterministic_pass,
            quality=observation.quality,
            outcome_digest=observation_outcome_digest(observation),
            artifact_digests=tuple(observation.artifact_digests),
            success=observation.success,
            exit_state=observation.exit_state,
        ))
    if len(environment_digests) != 1:
        raise IntegrityError("replay runs executed in different environments")
    if len(model_digests) != 1:
        raise IntegrityError("replay runs used different model configurations")
    reproducible = len({run.outcome_digest for run in results}) == 1
    if llm_influenced:
        passed = all(run.deterministic_pass and run.success for run in results)
    else:
        passed = all(run.deterministic_pass for run in results)
    return ReplayResult(
        benchmark_id=benchmark.id,
        benchmark_version=benchmark.version,
        benchmark_digest=benchmark.digest,
        commit=commit,
        environment_digest=environment_digests.pop(),
        runs=tuple(results),
        reproducible=reproducible,
        passed=passed,
        replayed_at=utc_now(),
        benchmark_kind=kind,
        runner=runner_name or type(runner).__name__,
        llm_influenced=llm_influenced,
        model=model or "",
        model_config_digest=model_digests.pop(),
    )


def fixture_tree_digest(directory: Path) -> str:
    """`sha256:` digest over a fixture tree (relative POSIX paths and file content digests).

    Corpus records store it in `environment.input_data`, binding replay
    evidence to the exact fixture bytes. Symlinks and special files are rejected.
    """
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"fixture directory does not exist: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"fixture must not contain symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"fixture contains a special file: {relative}")
        entries.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    if not entries:
        raise ValueError(f"fixture directory is empty: {root}")
    return "sha256:" + hashlib.sha256(canonical_json(entries).encode()).hexdigest()
