"""BenchmarkRunner adapter that executes tasks through Hermes' ``batch_runner.py``.

Each run gets a private temporary directory containing:

* ``brain/``       - ``git archive`` export of the evaluated brain commit;
* ``hermes-home/`` - ``HERMES_HOME`` whose ``skills/`` is copied from
  ``brain/brain/skills`` (or empty when the commit has no skills);
* ``source/``      - the benchmark's repository fixture (pre-run state);
* ``workspace/``   - a copy of ``source`` used as the agent's ``TERMINAL_CWD``;
* ``dataset.jsonl`` with the single task prompt.

``batch_runner.py`` is invoked unmodified (fire CLI flags) with one worker and
batch size one, writes ``data/<run_name>/trajectories.jsonl`` below the run
directory, and is parsed into a ``RunObservation``. Hermes tool execution is
configured through Hermes' own ``TERMINAL_*`` variables: Docker backend,
pinned image, networking disabled, no forwarded environment or volumes,
non-persistent containers. Live execution needs a Hermes installation and a
reachable model endpoint; neither is exercised in CI.

Hermes provides no token usage on this path, so observations from this runner
set ``token_usage_reported=False``; token-efficiency metrics are then reported
as unavailable rather than as zero, and the promotion policy blocks with
"token efficiency not measured" unless ``require_token_measurement`` is false.
Verified in Hermes v0.22.2: ``run_conversation`` returns token counters, but
``batch_runner`` keeps only completion flags, ``api_calls``, tool stats and the
trajectory, and builds ``AIAgent`` without a ``session_db``, so no SessionDB
session row (with its token columns) is written either. ``hermes -z
--usage-file`` reports tokens but produces no trajectory, so it cannot replace
batch_runner for this runner (see ``docs/evaluation.md``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .artifacts import ArtifactStore
from .errors import IntegrityError
from .ingestion import redact_text
from .models import Benchmark, RunObservation, canonical_json
from .sandbox import (
    SandboxExecutor,
    SandboxLimits,
    SandboxPolicyError,
    compute_changes,
    export_git_commit,
    extract_workspace_snapshot,
    git_tree_digest,
    parse_size,
    unified_diff,
    validate_relative_path,
)
from .verifier import run_deterministic_verification
from .verifiers import VerificationContext, VerifierRegistry, default_registry


_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_MAX_TRAJECTORY_BYTES = 10 * 1024 * 1024
_TURN_ROLES = {"system", "human", "gpt", "tool"}
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}


@dataclass(frozen=True)
class HermesModelConfig:
    """Model settings passed to batch_runner; identical for baseline and candidate."""

    model: str
    base_url: str
    distribution: str = "default"
    reasoning_effort: str | None = None
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 200:
            raise ValueError("Hermes model name must be a non-empty string")
        if not isinstance(self.base_url, str) or not re.fullmatch(r"https?://[^\s]+", self.base_url):
            raise ValueError("Hermes model base_url must be an http(s) URL")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.distribution):
            raise ValueError("Hermes toolset distribution must be a simple name")
        if self.reasoning_effort is not None and self.reasoning_effort not in _REASONING_EFFORTS:
            raise ValueError("unsupported Hermes reasoning effort")
        if self.api_key_env is not None and not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", self.api_key_env):
            raise ValueError("api_key_env must name an environment variable")

    @property
    def digest(self) -> str:
        """Digest of model settings; the key's *name* is included, never its value."""
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class HermesTrajectory:
    completed: bool
    partial: bool
    api_calls: int
    tool_calls: int
    failed_tool_calls: int
    assistant_turns: int
    final_output: str
    context_chars: int
    raw: bytes


def parse_trajectory_record(record: Mapping[str, Any]) -> HermesTrajectory:
    """Validate one ``trajectories.jsonl`` entry written by Hermes batch_runner."""
    if not isinstance(record, Mapping):
        raise IntegrityError("Hermes trajectory record must be an object")
    completed, partial = record.get("completed"), record.get("partial", False)
    if not isinstance(completed, bool) or not isinstance(partial, bool):
        raise IntegrityError("Hermes trajectory has invalid completion status")
    api_calls = record.get("api_calls")
    if isinstance(api_calls, bool) or not isinstance(api_calls, int) or api_calls < 0:
        raise IntegrityError("Hermes trajectory has an invalid api_calls counter")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise IntegrityError("Hermes trajectory has no conversations")
    assistant_turns = 0
    final_output = ""
    context_chars = 0
    for turn in conversations:
        if not isinstance(turn, Mapping) or turn.get("from") not in _TURN_ROLES or not isinstance(turn.get("value"), str):
            raise IntegrityError("Hermes trajectory contains an invalid turn")
        context_chars += len(turn["value"])
        if turn["from"] == "gpt":
            assistant_turns += 1
            if "<tool_call>" not in turn["value"]:
                final_output = _THINK_BLOCK.sub("", turn["value"]).strip()
    tool_stats = record.get("tool_stats", {})
    if not isinstance(tool_stats, Mapping):
        raise IntegrityError("Hermes trajectory has invalid tool statistics")
    tool_calls = failed = 0
    for name, counts in tool_stats.items():
        if not isinstance(name, str) or not isinstance(counts, Mapping):
            raise IntegrityError("Hermes trajectory has invalid tool statistics")
        values = []
        for key in ("count", "success", "failure"):
            value = counts.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise IntegrityError("Hermes trajectory has invalid tool counters")
            values.append(value)
        if values[1] + values[2] > values[0]:
            raise IntegrityError("Hermes trajectory has inconsistent tool counters")
        tool_calls += values[0]
        failed += values[2]
    return HermesTrajectory(
        completed=completed,
        partial=partial,
        api_calls=api_calls,
        tool_calls=tool_calls,
        failed_tool_calls=failed,
        assistant_turns=assistant_turns,
        final_output=final_output,
        context_chars=context_chars,
        raw=canonical_json(record).encode("utf-8"),
    )


def load_single_trajectory(path: Path) -> HermesTrajectory | None:
    """Read the merged trajectories file of a one-prompt run; ``None`` if the prompt failed."""
    if not path.is_file():
        return None
    with path.open("rb") as stream:
        raw = stream.read(_MAX_TRAJECTORY_BYTES + 1)
    if len(raw) > _MAX_TRAJECTORY_BYTES:
        raise IntegrityError("Hermes trajectory file exceeds the 10 MiB limit")
    records = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"Hermes trajectory file contains invalid JSON: {exc}") from exc
        if isinstance(record, dict) and record.get("discarded"):
            continue
        records.append(record)
    if not records:
        return None
    if len(records) != 1 or records[0].get("prompt_index") != 0:
        raise IntegrityError("single-task Hermes run produced an unexpected number of trajectories")
    return parse_trajectory_record(records[0])


class HermesBatchRunner:
    """Runs one benchmark task for one brain commit through Hermes' batch_runner."""

    llm_influenced = True

    def __init__(
        self,
        repository: Path,
        artifacts: ArtifactStore,
        *,
        hermes_agent_dir: Path,
        model: HermesModelConfig,
        executor_factory: Callable[[Benchmark], SandboxExecutor],
        python_executable: str = sys.executable,
        limits: SandboxLimits | None = None,
        fixtures_root: Path | None = None,
        terminal_env: str = "docker",
        allow_local_terminal: bool = False,
        verifier_http_allowlist: frozenset[str] = frozenset(),
        registry: VerifierRegistry | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.artifacts = artifacts
        self.hermes_agent_dir = Path(hermes_agent_dir).resolve()
        self.batch_runner = self.hermes_agent_dir / "batch_runner.py"
        if not self.batch_runner.is_file():
            raise FileNotFoundError(f"Hermes batch_runner.py not found in {self.hermes_agent_dir}")
        self.batch_runner_sha256 = hashlib.sha256(self.batch_runner.read_bytes()).hexdigest()
        resolved_python = shutil.which(python_executable)
        if resolved_python is None:
            raise FileNotFoundError(f"Python interpreter for Hermes not found: {python_executable}")
        self.python = resolved_python
        self.model = model
        self.executor_factory = executor_factory
        self.limits = limits or SandboxLimits()
        self.fixtures_root = Path(fixtures_root).resolve() if fixtures_root is not None else None
        if terminal_env not in {"docker", "local"}:
            raise ValueError("Hermes terminal_env must be 'docker' or 'local'")
        if terminal_env == "local" and allow_local_terminal is not True:
            raise SandboxPolicyError(
                "Hermes local terminal execution is unsandboxed; set allow_local_terminal only for tests"
            )
        self.terminal_env = terminal_env
        self.verifier_http_allowlist = frozenset(verifier_http_allowlist)
        self.registry = registry or default_registry()
        if model.api_key_env is not None and not os.environ.get(model.api_key_env):
            raise ValueError(f"model API key variable {model.api_key_env} is not set")

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ValueError("benchmark runner requires a full Git object ID")
        environment = benchmark.environment
        image = environment.get("container_image")
        if not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("Hermes benchmarks require a container image pinned by sha256 digest")
        if environment.get("network_policy", "disabled") != "disabled":
            raise SandboxPolicyError("Hermes runner executes agent tools with networking disabled only")
        prompt = benchmark.task.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Hermes benchmarks require a non-empty task prompt")
        fixture = self._fixture_path(environment.get("repository_fixture"))
        workspace_digest = git_tree_digest(self.repository, commit, self.limits.max_workspace_bytes)
        environment_digest = hashlib.sha256(canonical_json({
            "runner": "hermes_batch_runner",
            "batch_runner_sha256": self.batch_runner_sha256,
            "image": image,
            "dependency_digest": environment.get("dependency_digest"),
            "fixture": environment.get("repository_fixture"),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "terminal_env": self.terminal_env,
            "network": "disabled",
        }).encode()).hexdigest()
        executor = self.executor_factory(benchmark)
        with tempfile.TemporaryDirectory(prefix="hermes-batch-") as temporary:
            root = Path(temporary)
            brain = root / "brain"
            brain.mkdir()
            export_git_commit(self.repository, commit, brain)
            hermes_home = root / "hermes-home"
            skills = brain / "brain" / "skills"
            if skills.is_dir():
                shutil.copytree(skills, hermes_home / "skills")
            else:
                (hermes_home / "skills").mkdir(parents=True)
            source = root / "source"
            if fixture is not None:
                shutil.copytree(fixture, source, symlinks=False)
            else:
                source.mkdir()
            workspace = root / "workspace"
            shutil.copytree(source, workspace)
            run_dir = root / "run"
            run_dir.mkdir()
            dataset = root / "dataset.jsonl"
            dataset.write_text(json.dumps({"prompt": prompt}) + "\n", encoding="utf-8")
            run_name = "eval-" + hashlib.sha256(f"{benchmark.id}:{commit}:{seed}".encode()).hexdigest()[:16]
            argv, env = self._command(dataset, run_name, workspace, hermes_home, root, image, seed)
            started = time.monotonic()
            exit_code, output, timed_out = self._invoke(argv, env, run_dir)
            latency_ms = (time.monotonic() - started) * 1000
            if exit_code != 0 and not timed_out:
                raise RuntimeError(f"Hermes batch_runner failed (exit {exit_code}): {redact_text(output[-2000:])}")
            trajectory = None if timed_out else load_single_trajectory(
                run_dir / "data" / run_name / "trajectories.jsonl"
            )
            exit_state = self._exit_state(trajectory, timed_out)
            result_dir = root / "result"
            snapshot = root / "workspace.tar"
            with tarfile.open(snapshot, "w") as archive:
                archive.add(workspace, arcname=".")
            special = extract_workspace_snapshot(
                snapshot, result_dir,
                max_bytes=self.limits.max_workspace_bytes, max_entries=self.limits.max_snapshot_entries,
            )
            final_output = trajectory.final_output if trajectory is not None else ""
            context = VerificationContext(
                exit_code=0 if exit_state == "completed" else 1,
                output=final_output,
                workspace=result_dir,
                source=source,
                executor=executor,
                http_allowlist=self.verifier_http_allowlist,
                special_entries=special,
            )
            verification = run_deterministic_verification(benchmark, context, self.registry)
            changes = compute_changes(source, result_dir, special)
            diff_digest, _ = self.artifacts.put(redact_text(unified_diff(source, result_dir, changes)).encode())
            trajectory_digest = ""
            if trajectory is not None:
                trajectory_digest, _ = self.artifacts.put(redact_text(trajectory.raw.decode("utf-8")).encode())
            redacted_output = redact_text(final_output)
            record_digest, _ = self.artifacts.put(json.dumps({
                "task_id": benchmark.id,
                "commit": commit,
                "seed": seed,
                "exit_state": exit_state,
                "runner_output_tail": redact_text(output[-4000:]),
                "verifier_results": verification.details(),
                "filesystem_changes": list(changes.entries()),
                "trajectory_sha256": trajectory_digest or None,
                "diff_sha256": diff_digest,
                "workspace_digest": workspace_digest,
                "environment_digest": environment_digest,
                "model_config_digest": self.model.digest,
                "limits": asdict(self.limits),
                "seed_note": "batch_runner exposes no sampling seed; seed is passed as HERMES_EVAL_SEED only",
            }, sort_keys=True).encode())
        success = verification.passed and exit_state == "completed"
        digests = tuple(item for item in (record_digest, trajectory_digest, diff_digest) if item)
        return RunObservation(
            task_id=benchmark.id,
            commit=commit,
            seed=seed,
            environment_digest=environment_digest,
            success=success,
            quality=verification.quality,
            tokens=0,
            latency_ms=latency_ms,
            deterministic_pass=verification.passed,
            security_pass=True,
            tool_calls=trajectory.tool_calls if trajectory else 0,
            failed_tool_calls=trajectory.failed_tool_calls if trajectory else 0,
            retries=max(0, trajectory.api_calls - trajectory.assistant_turns) if trajectory else 0,
            exit_code=context.exit_code,
            final_output=redacted_output,
            artifact_digests=digests,
            token_usage_reported=False,
            context_chars=trajectory.context_chars if trajectory else 0,
            turns=trajectory.assistant_turns if trajectory else 0,
            api_calls=trajectory.api_calls if trajectory else 0,
            exit_state=exit_state,
            model_config_digest=self.model.digest,
            limits_digest=self.limits.digest,
            verifier_results=tuple(verification.summaries()),
            filesystem_changes=changes.entries(),
            trajectory_digest=trajectory_digest,
            diff_digest=diff_digest,
        )

    def _exit_state(self, trajectory: HermesTrajectory | None, timed_out: bool) -> str:
        if timed_out:
            return "timeout"
        if trajectory is None:
            return "agent_error"
        if trajectory.tool_calls > self.limits.max_tool_calls:
            return "tool_call_limit_exceeded"
        if trajectory.assistant_turns > self.limits.max_turns:
            return "turn_limit_exceeded"
        if trajectory.partial:
            return "partial"
        return "completed" if trajectory.completed else "incomplete"

    def _fixture_path(self, value: object) -> Path | None:
        if value is None:
            return None
        if self.fixtures_root is None:
            raise ValueError("benchmark declares a repository_fixture but no fixtures_root is configured")
        relative = validate_relative_path(value, label="repository_fixture", allow_root=False)
        path = self.fixtures_root.joinpath(*PurePosixPath(relative).parts).resolve()
        if self.fixtures_root not in path.parents or not path.is_dir():
            raise ValueError("repository_fixture must be a directory inside fixtures_root")
        return path

    def _command(
        self, dataset: Path, run_name: str, workspace: Path, hermes_home: Path, root: Path, image: str, seed: int
    ) -> tuple[list[str], dict[str, str]]:
        argv = [
            self.python, str(self.batch_runner),
            f"--dataset_file={dataset}",
            "--batch_size=1",
            f"--run_name={run_name}",
            f"--distribution={self.model.distribution}",
            f"--model={self.model.model}",
            f"--base_url={self.model.base_url}",
            f"--max_turns={self.limits.max_turns}",
            "--num_workers=1",
        ]
        if self.model.reasoning_effort is not None:
            argv.append(f"--reasoning_effort={self.model.reasoning_effort}")
        if self.model.api_key_env is not None:
            argv.append(f"--api_key={os.environ[self.model.api_key_env]}")
        home = root / "home"
        home.mkdir(exist_ok=True)
        memory_mb = max(1, parse_size(self.limits.memory) // (1024 * 1024))
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HERMES_HOME": str(hermes_home),
            "HERMES_EVAL_SEED": str(seed),
            "TERMINAL_ENV": self.terminal_env,
            "TERMINAL_CWD": str(workspace),
            "TERMINAL_TIMEOUT": str(max(1, math.floor(self.limits.timeout_seconds))),
            "TERMINAL_DOCKER_IMAGE": image,
            "TERMINAL_DOCKER_IMAGE_PINNED": "1",
            "TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE": "true",
            "TERMINAL_DOCKER_NETWORK": "false",
            "TERMINAL_DOCKER_FORWARD_ENV": "[]",
            "TERMINAL_DOCKER_VOLUMES": "[]",
            "TERMINAL_DOCKER_ENV": "{}",
            "TERMINAL_DOCKER_EXTRA_ARGS": "[]",
            "TERMINAL_DOCKER_RUN_AS_HOST_USER": "false",
            "TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES": "false",
            "TERMINAL_CONTAINER_PERSISTENT": "false",
            "TERMINAL_CONTAINER_CPU": self.limits.cpus,
            "TERMINAL_CONTAINER_MEMORY": str(memory_mb),
            "TERMINAL_LOCAL_PERSISTENT": "false",
        }
        for name in ("DOCKER_HOST", "DOCKER_CONFIG"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        return argv, env

    def _invoke(self, argv: list[str], env: dict[str, str], cwd: Path) -> tuple[int, str, bool]:
        log_path = cwd / "batch_runner.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            timed_out = False
            try:
                process.wait(timeout=self.limits.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        size = log_path.stat().st_size
        with log_path.open("rb") as stream:
            stream.seek(max(0, size - 64 * 1024))
            output = stream.read().decode("utf-8", errors="replace")
        return (124 if timed_out else process.returncode), output, timed_out
