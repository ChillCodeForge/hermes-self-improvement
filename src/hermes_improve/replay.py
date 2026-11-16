"""Benchmark replay: reproducibility evidence for command benchmarks (spec §13, §34).

A replay re-runs one command benchmark several times at a fixed commit through
a `BenchmarkRunner` (the CLI uses the digest-pinned `DockerCommandRunner`).
The benchmark is reproducible when every run yields the identical verifier
outcome (exit code, deterministic pass/fail, quality, environment digest),
independent of the seed. A replay counts as verified only when it is
reproducible *and* the deterministic verification passed, i.e. the fixture's
reference solution satisfies its own ground truth.
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


@dataclass(frozen=True)
class ReplayRun:
    seed: int
    exit_code: int
    deterministic_pass: bool
    quality: float
    outcome_digest: str
    artifact_digests: tuple[str, ...]


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "benchmark_digest": self.benchmark_digest,
            "commit": self.commit,
            "environment_digest": self.environment_digest,
            "runs": [
                {
                    "seed": run.seed,
                    "exit_code": run.exit_code,
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
        }


def is_command_benchmark(benchmark: Benchmark) -> bool:
    command = benchmark.environment.get("command")
    return isinstance(command, list) and bool(command)


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
) -> ReplayResult:
    """Run `benchmark` `runs` times (seeds 0..runs-1) and compare verifier outcomes."""
    if benchmark.split is Split.SEALED_HOLDOUT:
        raise ValueError("sealed holdout benchmarks are not replayed through the public replay path")
    if not is_command_benchmark(benchmark):
        raise ValueError(f"benchmark {benchmark.id!r} is not a command benchmark")
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        raise ValueError("replay requires a full Git object ID")
    if isinstance(runs, bool) or not isinstance(runs, int) or not MIN_REPLAY_RUNS <= runs <= MAX_REPLAY_RUNS:
        raise ValueError(f"replay runs must be an integer between {MIN_REPLAY_RUNS} and {MAX_REPLAY_RUNS}")
    results: list[ReplayRun] = []
    environment_digests: set[str] = set()
    for seed in range(runs):
        observation = runner.run(commit, benchmark, seed)
        if (observation.task_id, observation.commit, observation.seed) != (benchmark.id, commit, seed):
            raise IntegrityError("runner returned an observation for a different task, commit or seed")
        environment_digests.add(observation.environment_digest)
        results.append(ReplayRun(
            seed=seed,
            exit_code=observation.exit_code,
            deterministic_pass=observation.deterministic_pass,
            quality=observation.quality,
            outcome_digest=observation_outcome_digest(observation),
            artifact_digests=tuple(observation.artifact_digests),
        ))
    if len(environment_digests) != 1:
        raise IntegrityError("replay runs executed in different environments")
    reproducible = len({run.outcome_digest for run in results}) == 1
    return ReplayResult(
        benchmark_id=benchmark.id,
        benchmark_version=benchmark.version,
        benchmark_digest=benchmark.digest,
        commit=commit,
        environment_digest=environment_digests.pop(),
        runs=tuple(results),
        reproducible=reproducible,
        passed=all(run.deterministic_pass for run in results),
        replayed_at=utc_now(),
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
