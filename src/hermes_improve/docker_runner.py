from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

from .artifacts import ArtifactStore
from .deliverables import collect_deliverables, declared_deliverables
from .ingestion import redact_text
from .models import Benchmark, RunObservation, Split, canonical_json
from .replay import fixture_tree_digest
from .sandbox import (
    DockerSandboxExecutor,
    LocalSandboxExecutor,
    NetworkPolicy,
    SandboxExecutor,
    SandboxLimits,
    SandboxPolicyError,
    compute_changes,
    export_git_commit,
    extract_workspace_snapshot,
    git_env,
    git_tree_digest,
    unified_diff,
    validate_argv,
    validate_relative_path,
)
from .verifier import run_deterministic_verification
from .verifiers import VerificationContext, VerifierRegistry, default_registry


_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

#: ``environment.type`` of brain-consuming benchmarks (see ``SandboxCommandRunner``).
SKILL_PROCEDURE = "skill_procedure"
_FIXTURES_DIRECTORY = "fixtures"


def is_skill_procedure_benchmark(benchmark: Benchmark) -> bool:
    return benchmark.environment.get("type") == SKILL_PROCEDURE


class SandboxCommandRunner:
    """Runs deterministic command benchmarks through a ``SandboxExecutor``.

    The task command and every command-style verifier run in a fresh sandbox
    copy of an input tree. The post-task workspace is exported as a tar
    snapshot, extracted on the host as plain files only, and inspected
    read-only by file/diff verifiers.

    Two input layouts exist:

    * command benchmarks (default): the whole evaluated commit, with
      ``environment.cwd`` inside it;
    * ``environment.type == "skill_procedure"`` (brain-consuming benchmarks):
      only ``brain/`` comes from the evaluated commit. The executor harness
      (``environment.harness``) and the task fixture
      (``environment.repository_fixture``) are copied from the evaluator's own
      benchmark roots - the public root, or the sealed holdout root for
      SEALED_HOLDOUT tasks - and must match the digests pinned in the record
      (``environment.harness_digest`` and ``environment.input_data``). A
      candidate commit therefore cannot change the yardstick it is measured
      with; only its brain content varies between baseline and candidate.
    """

    llm_influenced = False

    def __init__(
        self,
        repository: Path,
        artifacts: ArtifactStore,
        *,
        executor_factory: Callable[[str, NetworkPolicy], SandboxExecutor],
        limits: SandboxLimits,
        default_timeout: float,
        network: NetworkPolicy | None = None,
        verifier_http_allowlist: frozenset[str] = frozenset(),
        registry: VerifierRegistry | None = None,
        benchmark_root: Path | None = None,
        holdout_root: Path | None = None,
    ) -> None:
        if not math.isfinite(default_timeout) or default_timeout <= 0:
            raise ValueError("timeout and workspace limit must be positive")
        self.repository = Path(repository).resolve()
        self.artifacts = artifacts
        self.limits = limits
        self.default_timeout = default_timeout
        self.network = network or NetworkPolicy()
        self.verifier_http_allowlist = frozenset(verifier_http_allowlist)
        self.registry = registry or default_registry()
        self.executor_factory = executor_factory
        self.benchmark_root = Path(benchmark_root).resolve() if benchmark_root is not None else None
        self.holdout_root = Path(holdout_root).resolve() if holdout_root is not None else None

    # ------------------------------------------------------------------ inputs

    def _trusted_directory(self, root: Path | None, value: object, label: str) -> Path:
        """Resolve a record path below ``<root>/fixtures/``; refuse anything else."""
        if root is None:
            raise ValueError(f"skill_procedure benchmarks need the evaluator's {label} root")
        relative = validate_relative_path(value, label=label, allow_root=False)
        if relative.parts[0] != _FIXTURES_DIRECTORY or len(relative.parts) < 2:
            raise ValueError(f"{label} must name a directory below {_FIXTURES_DIRECTORY}/")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"{label} is not a directory inside the evaluator benchmark root")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"{label} escapes the evaluator benchmark root")
        return resolved

    def _skill_procedure_inputs(self, benchmark: Benchmark) -> tuple[Path, Path, dict[str, str]]:
        environment = benchmark.environment
        fixture_root = self.holdout_root if benchmark.split is Split.SEALED_HOLDOUT else self.benchmark_root
        fixture = self._trusted_directory(fixture_root, environment.get("repository_fixture"), "repository_fixture")
        harness = self._trusted_directory(self.benchmark_root, environment.get("harness"), "harness")
        fixture_digest = fixture_tree_digest(fixture)
        harness_digest = fixture_tree_digest(harness)
        for key, actual in (("input_data", fixture_digest), ("harness_digest", harness_digest)):
            declared = environment.get(key)
            if not isinstance(declared, str) or not _DIGEST.fullmatch(declared):
                raise ValueError(f"skill_procedure benchmarks must pin environment.{key} as a sha256 digest")
            if declared != actual:
                raise SandboxPolicyError(
                    f"benchmark {benchmark.id!r}: environment.{key} does not match the evaluator's fixture tree"
                )
        return fixture, harness, {"fixture": fixture_digest, "harness": harness_digest}

    def _export_brain(self, commit: str, destination: Path) -> None:
        listing = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.repository), "ls-tree", "--name-only",
             commit, "brain"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=git_env(),
        )
        if listing.returncode:
            raise RuntimeError(f"cannot inspect Git commit: {listing.stderr.decode(errors='replace')}")
        if listing.stdout.strip():
            export_git_commit(self.repository, commit, destination, paths=("brain",))
        (destination / "brain").mkdir(exist_ok=True)

    # --------------------------------------------------------------------- run

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ValueError("benchmark runner requires a full Git object ID")
        environment = benchmark.environment
        image = environment.get("container_image")
        command = environment.get("command")
        if not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("Docker command benchmarks require an image pinned by sha256 digest")
        network = self.network.for_benchmark(environment)
        if not isinstance(command, list):
            raise ValueError("benchmark environment.command must be a non-empty argv array")
        try:
            argv = validate_argv(command)
        except ValueError as exc:
            raise ValueError("benchmark environment.command must be a non-empty argv array") from exc
        cwd = environment.get("cwd", ".")
        if not isinstance(cwd, str):
            raise ValueError("benchmark environment.cwd must be a relative path")
        try:
            cwd_path = validate_relative_path(cwd, label="benchmark cwd")
        except ValueError as exc:
            raise ValueError("benchmark cwd must not escape the candidate workspace") from exc
        timeout = float(environment.get("timeout_seconds", self.default_timeout))
        if not math.isfinite(timeout) or timeout <= 0 or timeout > self.default_timeout:
            raise ValueError("benchmark timeout must be positive and within the evaluator limit")
        procedure = is_skill_procedure_benchmark(benchmark)
        if procedure:
            fixture, harness, trusted_digests = self._skill_procedure_inputs(benchmark)
        else:
            self._check_command_benchmark_allowed(benchmark)
            fixture = harness = None
            trusted_digests = {}
        workspace_digest = git_tree_digest(self.repository, commit, self.limits.max_workspace_bytes)
        identity: dict[str, object] = {
            "image": image,
            "dependency_digest": environment.get("dependency_digest"),
            "command": command,
            "cwd": cwd,
            "network": "disabled" if network.mode == "disabled" else network.describe(),
        }
        if procedure:
            identity.update({"type": SKILL_PROCEDURE, **trusted_digests})
        environment_digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        executor = self.executor_factory(image, network)
        with tempfile.TemporaryDirectory(prefix="hermes-eval-") as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir(mode=0o755)
            if procedure:
                self._export_brain(commit, source)
                shutil.copytree(harness, source / "harness", symlinks=False)
                shutil.copytree(fixture, source / "fixture", symlinks=False)
            else:
                export_git_commit(self.repository, commit, source)
            result = executor.execute(
                argv, input_root=source, cwd=str(cwd_path), timeout=timeout,
                env={"HERMES_EVAL_SEED": str(seed)}, snapshot_dir=root / "output",
            )
            workspace = root / "result"
            special: tuple[str, ...] = ()
            if result.snapshot is not None:
                special = extract_workspace_snapshot(
                    result.snapshot, workspace,
                    max_bytes=self.limits.max_workspace_bytes,
                    max_entries=self.limits.max_snapshot_entries,
                )
            else:
                workspace.mkdir()
            context = VerificationContext(
                exit_code=result.exit_code,
                output=result.output,
                workspace=workspace,
                source=source,
                executor=executor,
                http_allowlist=self.verifier_http_allowlist,
                special_entries=special,
                command_timeout=timeout,
            )
            verification = run_deterministic_verification(benchmark, context, self.registry)
            deliverables = collect_deliverables(declared_deliverables(benchmark.verification), workspace)
            changes = compute_changes(source, workspace, special)
            diff = unified_diff(source, workspace, changes) if result.snapshot is not None else ""
            diff_digest, _ = self.artifacts.put(redact_text(diff).encode("utf-8"))
            output = redact_text(result.output)
            payload = json.dumps({
                "task_id": benchmark.id,
                "commit": commit,
                "seed": seed,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "deterministic_pass": verification.passed,
                "quality": verification.quality,
                "verifier_results": verification.details(),
                "filesystem_changes": list(changes.entries()),
                "diff_sha256": diff_digest,
                "output": output,
                "workspace_digest": workspace_digest,
                "environment_digest": environment_digest,
                "limits": self.limits.digest,
                "isolation": executor.isolation,
                **({"trusted_inputs": trusted_digests} if procedure else {}),
            }, sort_keys=True).encode()
            artifact_digest, _ = self.artifacts.put(payload)
        return RunObservation(
            task_id=benchmark.id,
            commit=commit,
            seed=seed,
            environment_digest=environment_digest,
            success=verification.passed,
            quality=verification.quality,
            # No model runs here and nothing measures token usage, so the zero
            # is not a measurement: report it as unmeasured (see PromotionPolicy
            # ``require_token_measurement``).
            tokens=0,
            token_usage_reported=False,
            latency_ms=result.duration_ms,
            deterministic_pass=verification.passed,
            security_pass=True,
            exit_code=result.exit_code,
            final_output=output,
            artifact_digests=(artifact_digest, diff_digest),
            exit_state="timeout" if result.timed_out else "exited",
            limits_digest=self.limits.digest,
            verifier_results=tuple(verification.summaries()),
            filesystem_changes=changes.entries(),
            diff_digest=diff_digest,
            deliverables=deliverables,
        )

    def _check_command_benchmark_allowed(self, benchmark: Benchmark) -> None:
        """Hook for executors that must not run candidate-provided commands."""


class DockerCommandRunner(SandboxCommandRunner):
    """Runs deterministic command benchmarks in disposable isolated containers.

    See ``DockerSandboxExecutor`` for the container hardening. Both input
    layouts of ``SandboxCommandRunner`` are supported.
    """

    def __init__(
        self,
        repository: Path,
        artifacts: ArtifactStore,
        *,
        docker_binary: str = "docker",
        default_timeout: float = 120,
        memory_limit: str = "2g",
        cpu_limit: str = "1",
        max_workspace_bytes: int = 100 * 1024 * 1024,
        network: NetworkPolicy | None = None,
        verifier_http_allowlist: frozenset[str] = frozenset(),
        registry: VerifierRegistry | None = None,
        benchmark_root: Path | None = None,
        holdout_root: Path | None = None,
    ) -> None:
        if not math.isfinite(default_timeout) or default_timeout <= 0 or max_workspace_bytes < 1:
            raise ValueError("timeout and workspace limit must be positive")
        try:
            limits = SandboxLimits(
                timeout_seconds=default_timeout, memory=memory_limit, cpus=cpu_limit,
                max_workspace_bytes=max_workspace_bytes,
            )
        except ValueError as exc:
            raise ValueError("Docker memory and CPU limits must be finite positive values") from exc
        self.docker_binary = docker_binary
        if shutil.which(docker_binary) is None:
            raise RuntimeError(f"Docker executable was not found: {docker_binary}")

        def executor_factory(image: str, network_policy: NetworkPolicy) -> SandboxExecutor:
            return DockerSandboxExecutor(image, limits, docker_binary=docker_binary, network=network_policy)

        super().__init__(
            repository, artifacts, executor_factory=executor_factory, limits=limits,
            default_timeout=default_timeout, network=network,
            verifier_http_allowlist=verifier_http_allowlist, registry=registry,
            benchmark_root=benchmark_root, holdout_root=holdout_root,
        )


class LocalSkillProcedureRunner(SandboxCommandRunner):
    """Runs only ``skill_procedure`` benchmarks through ``LocalSandboxExecutor``.

    For tests, development and hosts without Docker. The executed program is
    the evaluator-owned harness; the candidate contributes brain *data* (the
    skill's declared procedure), which the harness interprets with a fixed
    operation set and never evaluates as code. Command benchmarks, whose
    commands come from the candidate commit, are refused. The executor has no
    kernel isolation (scrubbed environment, private copy, POSIX rlimits only),
    so construction requires an explicit acknowledgement.
    """

    def __init__(
        self,
        repository: Path,
        artifacts: ArtifactStore,
        *,
        benchmark_root: Path,
        holdout_root: Path | None = None,
        default_timeout: float = 120,
        acknowledge_no_isolation: bool = False,
        registry: VerifierRegistry | None = None,
    ) -> None:
        limits = SandboxLimits(timeout_seconds=default_timeout)
        executor = LocalSandboxExecutor(limits, acknowledge_no_isolation=acknowledge_no_isolation)

        def executor_factory(image: str, network_policy: NetworkPolicy) -> SandboxExecutor:
            if network_policy.mode != "disabled":
                raise SandboxPolicyError("the local skill-procedure runner supports disabled networking only")
            return executor

        super().__init__(
            repository, artifacts, executor_factory=executor_factory, limits=limits,
            default_timeout=default_timeout, registry=registry,
            benchmark_root=benchmark_root, holdout_root=holdout_root,
        )

    def _check_command_benchmark_allowed(self, benchmark: Benchmark) -> None:
        raise SandboxPolicyError(
            f"benchmark {benchmark.id!r} runs commands from the evaluated commit; "
            "only skill_procedure benchmarks may run without container isolation"
        )


__all__ = [
    "DockerCommandRunner",
    "LocalSkillProcedureRunner",
    "SKILL_PROCEDURE",
    "SandboxCommandRunner",
    "is_skill_procedure_benchmark",
]

