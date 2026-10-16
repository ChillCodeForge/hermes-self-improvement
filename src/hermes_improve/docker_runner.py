from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

from .artifacts import ArtifactStore
from .errors import IntegrityError
from .ingestion import redact_text
from .models import Benchmark, RunObservation, canonical_json
from .verifier import verify_deterministic_output


_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")


class DockerCommandRunner:
    """Runs deterministic command benchmarks in a disposable isolated container."""

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
    ) -> None:
        resolved = shutil.which(docker_binary)
        if resolved is None:
            raise RuntimeError(f"Docker executable was not found: {docker_binary}")
        self.docker_binary = resolved
        self.repository = Path(repository).resolve()
        self.artifacts = artifacts
        self.default_timeout = default_timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.max_workspace_bytes = max_workspace_bytes
        if not math.isfinite(default_timeout) or default_timeout <= 0 or max_workspace_bytes < 1:
            raise ValueError("timeout and workspace limit must be positive")
        if (
            not re.fullmatch(r"[1-9][0-9]*[kKmMgG]?", memory_limit)
            or not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", cpu_limit)
            or not math.isfinite(float(cpu_limit)) or float(cpu_limit) <= 0
        ):
            raise ValueError("Docker memory and CPU limits must be finite positive values")

    def run(self, commit: str, benchmark: Benchmark, seed: int) -> RunObservation:
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            raise ValueError("benchmark runner requires a full Git object ID")
        environment = benchmark.environment
        image = environment.get("container_image")
        command = environment.get("command")
        if not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("Docker command benchmarks require an image pinned by sha256 digest")
        if environment.get("network_policy") != "disabled":
            raise ValueError("Docker runner only supports network_policy=disabled")
        if not isinstance(command, list) or not command or not all(
            isinstance(arg, str) and arg and "\x00" not in arg for arg in command
        ):
            raise ValueError("benchmark environment.command must be a non-empty argv array")
        cwd = environment.get("cwd", ".")
        if not isinstance(cwd, str):
            raise ValueError("benchmark environment.cwd must be a relative path")
        cwd_path = PurePosixPath(cwd)
        if cwd_path.is_absolute() or any(part in {"..", ""} for part in cwd_path.parts):
            raise ValueError("benchmark cwd must not escape the candidate workspace")
        timeout = float(environment.get("timeout_seconds", self.default_timeout))
        if not math.isfinite(timeout) or timeout <= 0 or timeout > self.default_timeout:
            raise ValueError("benchmark timeout must be positive and within the evaluator limit")
        expected_exit = benchmark.ground_truth.get("expected_exit_code", 0)
        if isinstance(expected_exit, bool) or not isinstance(expected_exit, int):
            raise ValueError("ground_truth.expected_exit_code must be an integer")
        workspace_digest = self._tree_digest(commit)
        environment_digest = hashlib.sha256(canonical_json({
            "image": image,
            "dependency_digest": environment.get("dependency_digest"),
            "command": command,
            "cwd": cwd,
            "network": "disabled",
        }).encode()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="hermes-eval-") as temporary:
            root = Path(temporary)
            source = root / "candidate"
            source.mkdir(mode=0o755)
            self._export_commit(commit, source)
            cidfile = root / "container.id"
            argv = [
                self.docker_binary, "run", "--rm", "--pull=never",
                "--network=none", "--read-only", "--user=65534:65534",
                "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--pids-limit=128", "--memory", self.memory_limit,
                "--cpus", self.cpu_limit, "--ipc=none",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
                "--tmpfs", "/workspace:rw,nosuid,nodev,size=128m,mode=1777",
                "--mount", f"type=bind,source={source},target=/input,readonly",
                "--cidfile", str(cidfile),
                "--env", f"HERMES_EVAL_SEED={seed}",
                "--workdir", "/workspace",
                "--entrypoint", "/bin/sh",
                image, "-c",
                'cp -R /input/. /workspace/ && cd "$1" && shift && exec "$@"',
                "hermes-eval", str(cwd_path), *command,
            ]
            started = time.monotonic()
            try:
                result = subprocess.run(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout,
                    check=False,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "HOME": os.environ.get("HOME", ""),
                        "DOCKER_CONFIG": os.environ.get("DOCKER_CONFIG", ""),
                    },
                )
                exit_code = result.returncode
                output = result.stdout[-16000:]
            except subprocess.TimeoutExpired as exc:
                self._kill_timed_out_container(cidfile)
                exit_code = 124
                output = str(exc.stdout or "")[-16000:] + "\nbenchmark timed out"
            latency_ms = (time.monotonic() - started) * 1000
            if exit_code in {125, 126, 127}:
                raise RuntimeError(f"Docker failed to execute benchmark: {redact_text(output)}")
            passed, quality, verifier_results = verify_deterministic_output(
                benchmark, exit_code=exit_code, output=output
            )
            output = redact_text(output)
            payload = json.dumps({
                "task_id": benchmark.id,
                "commit": commit,
                "seed": seed,
                "exit_code": exit_code,
                "deterministic_pass": passed,
                "quality": quality,
                "verifier_results": verifier_results,
                "output": output,
                "workspace_digest": workspace_digest,
                "environment_digest": environment_digest,
            }, sort_keys=True).encode()
            artifact_digest, _ = self.artifacts.put(payload)
        return RunObservation(
            task_id=benchmark.id,
            commit=commit,
            seed=seed,
            environment_digest=environment_digest,
            success=passed,
            quality=quality,
            tokens=0,
            latency_ms=latency_ms,
            deterministic_pass=passed,
            security_pass=True,
            exit_code=exit_code,
            final_output=output,
            artifact_digests=(artifact_digest,),
        )

    def _tree_digest(self, commit: str) -> str:
        output = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.repository), "ls-tree", "-r", "-l", commit],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=self._git_env(),
        )
        if output.returncode:
            raise RuntimeError(f"cannot inspect Git commit: {output.stderr.decode(errors='replace')}")
        total = 0
        for line in output.stdout.decode().splitlines():
            metadata, separator, _path = line.partition("\t")
            fields = metadata.split()
            if not separator or len(fields) < 3:
                raise IntegrityError("Git returned an invalid tree entry")
            if fields[0] == "120000":
                raise IntegrityError("candidate source archives may not contain symbolic links")
            if len(fields) >= 4 and fields[3].isdigit():
                total += int(fields[3])
        if total > self.max_workspace_bytes:
            raise IntegrityError("candidate source exceeds the configured workspace size limit")
        return hashlib.sha256(output.stdout).hexdigest()

    def _export_commit(self, commit: str, destination: Path) -> None:
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(self.repository), "archive", "--format=tar", commit],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=self._git_env(),
        )
        if result.returncode:
            raise RuntimeError(f"cannot export Git commit: {result.stderr.decode(errors='replace')}")
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            root = destination.resolve()
            for member in archive:
                relative = PurePosixPath(member.name)
                if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                    raise IntegrityError(f"unsafe path in candidate archive: {member.name!r}")
                target = root.joinpath(*relative.parts)
                if root not in target.resolve().parents and target.resolve() != root:
                    raise IntegrityError(f"archive path escapes candidate workspace: {member.name!r}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o755)
                    continue
                if not member.isfile():
                    raise IntegrityError(f"unsupported archive object: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
                source = archive.extractfile(member)
                if source is None:
                    raise IntegrityError(f"archive object has no data: {member.name!r}")
                mode = 0o755 if member.mode & 0o111 else 0o644
                with target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(mode)

    def _kill_timed_out_container(self, cidfile: Path) -> None:
        if not cidfile.exists():
            return
        container_id = cidfile.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise IntegrityError("Docker wrote an invalid container ID file")
        subprocess.run(
            [self.docker_binary, "kill", container_id],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False,
        )

    @staticmethod
    def _git_env() -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
        }
