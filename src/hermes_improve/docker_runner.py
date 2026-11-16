from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from .artifacts import ArtifactStore
from .ingestion import redact_text
from .models import Benchmark, RunObservation, canonical_json
from .sandbox import (
    DockerSandboxExecutor,
    NetworkPolicy,
    SandboxLimits,
    compute_changes,
    export_git_commit,
    extract_workspace_snapshot,
    git_tree_digest,
    unified_diff,
    validate_argv,
    validate_relative_path,
)
from .verifier import run_deterministic_verification
from .verifiers import VerificationContext, VerifierRegistry, default_registry


_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class DockerCommandRunner:
    """Runs deterministic command benchmarks in disposable isolated containers.

    The task command and every command-style verifier run in separate hardened
    containers (see ``DockerSandboxExecutor``). The post-task workspace is
    exported as a tar snapshot, extracted on the host as plain files only, and
    inspected read-only by file/diff verifiers.
    """

    llm_influenced = False

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
    ) -> None:
        if not math.isfinite(default_timeout) or default_timeout <= 0 or max_workspace_bytes < 1:
            raise ValueError("timeout and workspace limit must be positive")
        try:
            self.limits = SandboxLimits(
                timeout_seconds=default_timeout, memory=memory_limit, cpus=cpu_limit,
                max_workspace_bytes=max_workspace_bytes,
            )
        except ValueError as exc:
            raise ValueError("Docker memory and CPU limits must be finite positive values") from exc
        self.docker_binary = docker_binary
        self.repository = Path(repository).resolve()
        self.artifacts = artifacts
        self.default_timeout = default_timeout
        self.network = network or NetworkPolicy()
        self.verifier_http_allowlist = frozenset(verifier_http_allowlist)
        self.registry = registry or default_registry()
        if shutil.which(docker_binary) is None:
            raise RuntimeError(f"Docker executable was not found: {docker_binary}")

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
        workspace_digest = git_tree_digest(self.repository, commit, self.limits.max_workspace_bytes)
        environment_digest = hashlib.sha256(canonical_json({
            "image": image,
            "dependency_digest": environment.get("dependency_digest"),
            "command": command,
            "cwd": cwd,
            "network": "disabled" if network.mode == "disabled" else network.describe(),
        }).encode()).hexdigest()
        executor = DockerSandboxExecutor(image, self.limits, docker_binary=self.docker_binary, network=network)
        with tempfile.TemporaryDirectory(prefix="hermes-eval-") as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir(mode=0o755)
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
            }, sort_keys=True).encode()
            artifact_digest, _ = self.artifacts.put(payload)
        return RunObservation(
            task_id=benchmark.id,
            commit=commit,
            seed=seed,
            environment_digest=environment_digest,
            success=verification.passed,
            quality=verification.quality,
            tokens=0,
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
        )
