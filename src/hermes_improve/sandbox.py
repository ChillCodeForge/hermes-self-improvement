"""Sandboxed command execution, workspace snapshots and change capture.

Every command that touches candidate-produced state (benchmark task commands
and command-style verifiers) runs through a ``SandboxExecutor``. Executors
always copy a read-only input tree into a fresh workspace, run the command,
and optionally return a tar snapshot of the resulting workspace. The host
inspects snapshots read-only; it never executes candidate output itself.
"""

from __future__ import annotations

import difflib
import hashlib
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol, Sequence

from .errors import IntegrityError
from .models import canonical_json


_PINNED_IMAGE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
_SIZE = re.compile(r"^[1-9][0-9]*[kKmMgG]?$")
_HOST_PORT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,252})(?::[0-9]{1,5})?$")
_DOCKER_NETWORK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_MAX_OUTPUT_CHARS = 16000
_SANDBOX_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_CONTAINER_SCRIPT = (
    'in=$1; ws=$2; out=$3; cwd=$4; shift 4; '
    'cp -R "$in"/. "$ws"/ || exit 125; '
    'cd "$ws/$cwd" || exit 125; '
    '"$@"; code=$?; '
    'if [ "$out" != - ]; then tar -C "$ws" -cf "$out/workspace.tar" . || exit 125; fi; '
    'exit $code'
)


class SandboxPolicyError(ValueError):
    """A request would violate the evaluator sandbox policy."""


def parse_size(value: str) -> int:
    if not isinstance(value, str) or not _SIZE.fullmatch(value):
        raise ValueError(f"invalid size value: {value!r}")
    multiplier = {"k": 1024, "m": 1024 ** 2, "g": 1024 ** 3}.get(value[-1].lower(), 1)
    digits = value[:-1] if value[-1].lower() in "kmg" else value
    return int(digits) * multiplier


@dataclass(frozen=True)
class SandboxLimits:
    """Resource and agent-interaction limits applied identically to both arms."""

    timeout_seconds: float = 120
    memory: str = "2g"
    cpus: str = "1"
    pids: int = 128
    workspace_size: str = "128m"
    tmp_size: str = "64m"
    max_workspace_bytes: int = 100 * 1024 * 1024
    max_snapshot_entries: int = 20_000
    max_turns: int = 30
    max_tool_calls: int = 200

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("sandbox timeout must be a finite positive number")
        for name in ("memory", "workspace_size", "tmp_size"):
            parse_size(getattr(self, name))
        if (
            not re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)", self.cpus)
            or not math.isfinite(float(self.cpus)) or float(self.cpus) <= 0
        ):
            raise ValueError("sandbox CPU limit must be a finite positive value")
        for name in ("pids", "max_workspace_bytes", "max_snapshot_entries", "max_turns", "max_tool_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"sandbox {name} must be a positive integer")

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class NetworkPolicy:
    """Network access for sandboxed commands.

    ``disabled`` (the default) runs containers with ``--network=none``.
    ``allowlist`` attaches containers to one operator-created Docker network
    that must be ``--internal`` (no route to the outside world) and whose
    attached containers must all be named in ``allowed_hosts``; benchmarks may
    only request hosts from that allowlist.
    """

    mode: str = "disabled"
    network_name: str | None = None
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "allowlist"}:
            raise ValueError("network mode must be 'disabled' or 'allowlist'")
        if self.mode == "disabled":
            if self.network_name is not None or self.allowed_hosts:
                raise ValueError("disabled network policy cannot name a network or hosts")
            return
        if not isinstance(self.network_name, str) or not _DOCKER_NETWORK.fullmatch(self.network_name):
            raise ValueError("allowlist network policy requires a valid dedicated Docker network name")
        if self.network_name in {"bridge", "host", "none", "default"}:
            raise ValueError("allowlist network policy requires a dedicated network, not a Docker default")
        if not self.allowed_hosts or not all(
            isinstance(host, str) and _HOST_PORT.fullmatch(host) for host in self.allowed_hosts
        ):
            raise ValueError("allowlist network policy requires valid host[:port] entries")

    def for_benchmark(self, environment: Mapping[str, object]) -> "NetworkPolicy":
        """Return the narrowest policy satisfying a benchmark's declared needs."""
        requested = environment.get("network_policy", "disabled")
        if requested == "disabled":
            return NetworkPolicy()
        if requested != "allowlist":
            raise SandboxPolicyError(f"unsupported benchmark network_policy: {requested!r}")
        if self.mode != "allowlist":
            raise SandboxPolicyError("benchmark requests network access but the evaluator sandbox has networking disabled")
        hosts = environment.get("network_allowlist")
        if (
            not isinstance(hosts, list) or not hosts
            or not all(isinstance(host, str) and _HOST_PORT.fullmatch(host) for host in hosts)
        ):
            raise SandboxPolicyError("benchmark network_allowlist must be a non-empty host[:port] array")
        outside = sorted(set(hosts) - self.allowed_hosts)
        if outside:
            raise SandboxPolicyError(f"benchmark requests hosts outside the evaluator allowlist: {outside}")
        return NetworkPolicy("allowlist", self.network_name, frozenset(hosts))

    def docker_argument(self) -> str:
        return "--network=none" if self.mode == "disabled" else f"--network={self.network_name}"

    def describe(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "network": self.network_name,
            "allowed_hosts": sorted(self.allowed_hosts),
        }


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    output: str
    timed_out: bool
    duration_ms: float
    snapshot: Path | None = None


class SandboxExecutor(Protocol):
    """Runs one command against a fresh copy of ``input_root``."""

    isolation: str

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_root: Path,
        cwd: str = ".",
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
        snapshot_dir: Path | None = None,
    ) -> CommandResult: ...


def validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(argv, (str, bytes)) or not argv
        or not all(isinstance(arg, str) and arg and "\x00" not in arg for arg in argv)
    ):
        raise ValueError("sandbox commands must be non-empty argv arrays of non-empty strings")
    if len(argv) > 256 or sum(len(arg) for arg in argv) > 64 * 1024:
        raise ValueError("sandbox command exceeds argv size limits")
    return tuple(argv)


def validate_relative_path(value: object, *, label: str = "path", allow_root: bool = True) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        raise ValueError(f"{label} must not escape the workspace")
    if not allow_root and path == PurePosixPath("."):
        raise ValueError(f"{label} must name a path inside the workspace")
    return path


def _validate_env(env: Mapping[str, str] | None) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in (env or {}).items():
        if not isinstance(key, str) or not _SAFE_ENV_NAME.fullmatch(key) or not key.startswith("HERMES_EVAL_"):
            raise SandboxPolicyError("sandbox environment variables must be HERMES_EVAL_* names")
        if not isinstance(value, str) or "\x00" in value or len(value) > 4096:
            raise SandboxPolicyError("sandbox environment values must be short strings")
        values[key] = value
    return values


def _timeout(limits: SandboxLimits, requested: float | None) -> float:
    if requested is None:
        return float(limits.timeout_seconds)
    if (
        isinstance(requested, bool) or not isinstance(requested, (int, float))
        or not math.isfinite(requested) or requested <= 0 or requested > limits.timeout_seconds
    ):
        raise SandboxPolicyError("command timeout must be positive and within the sandbox limit")
    return float(requested)


class LocalSandboxExecutor:
    """Process-level executor for tests and operator development only.

    Commands run as the current user in a private temporary copy with a
    scrubbed environment (no inherited credentials), a fixed PATH, their own
    session/process group and POSIX resource limits. This is *not* kernel-level
    isolation; production evaluation uses ``DockerSandboxExecutor``.
    """

    isolation = "local-process"

    def __init__(self, limits: SandboxLimits | None = None, *, acknowledge_no_isolation: bool = False) -> None:
        if acknowledge_no_isolation is not True:
            raise SandboxPolicyError(
                "LocalSandboxExecutor provides no container isolation; pass acknowledge_no_isolation=True "
                "only for tests or trusted development"
            )
        self.limits = limits or SandboxLimits()

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_root: Path,
        cwd: str = ".",
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
        snapshot_dir: Path | None = None,
    ) -> CommandResult:
        command = validate_argv(argv)
        cwd_path = validate_relative_path(cwd, label="command cwd")
        limit = _timeout(self.limits, timeout)
        extra_env = _validate_env(env)
        source = Path(input_root)
        if not source.is_dir():
            raise ValueError("sandbox input root must be a directory")
        with tempfile.TemporaryDirectory(prefix="hermes-sandbox-") as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            shutil.copytree(source, workspace, symlinks=True)
            (root / "home").mkdir()
            (root / "tmp").mkdir()
            run_dir = workspace.joinpath(*cwd_path.parts)
            if not run_dir.is_dir():
                return CommandResult(125, "command cwd does not exist in workspace", False, 0.0)
            output_path = root / "output.log"
            process_env = {
                "PATH": _SANDBOX_PATH,
                "HOME": str(root / "home"),
                "TMPDIR": str(root / "tmp"),
                "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                **extra_env,
            }
            limits = self.limits
            output_cap = parse_size(limits.workspace_size)

            def apply_limits() -> None:
                import resource

                resource.setrlimit(resource.RLIMIT_FSIZE, (output_cap, output_cap))
                memory = parse_size(limits.memory)
                resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                cpu_seconds = max(1, math.ceil(limit))
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))

            started = time.monotonic()
            timed_out = False
            with output_path.open("wb") as output_stream:
                process = subprocess.Popen(
                    command,
                    cwd=run_dir,
                    stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                    stdout=output_stream,
                    stderr=subprocess.STDOUT,
                    env=process_env,
                    start_new_session=True,
                    preexec_fn=apply_limits,
                )
                try:
                    process.communicate(input=stdin, timeout=limit)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _kill_group(process)
                    process.communicate()
            duration_ms = (time.monotonic() - started) * 1000
            exit_code = 124 if timed_out else process.returncode
            if exit_code < 0:
                exit_code = 128 + (-exit_code)
            output = _read_tail(output_path)
            if timed_out:
                output += "\ncommand timed out"
            snapshot = None
            if snapshot_dir is not None and not timed_out:
                Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
                snapshot = Path(snapshot_dir) / "workspace.tar"
                with tarfile.open(snapshot, "w") as archive:
                    archive.add(workspace, arcname=".")
            return CommandResult(exit_code, output, timed_out, duration_ms, snapshot)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def _read_tail(path: Path, limit: int = _MAX_OUTPUT_CHARS * 4) -> str:
    size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, size - limit))
        data = stream.read()
    return data.decode("utf-8", errors="replace")[-_MAX_OUTPUT_CHARS:]


class DockerSandboxExecutor:
    """Runs commands in disposable, digest-pinned, hardened containers.

    Hardening: ``--network=none`` unless an internal allowlist network is
    configured, read-only root filesystem, non-root user 65534, all
    capabilities dropped, no-new-privileges, PID/memory/CPU limits, private
    IPC, size-bounded tmpfs for ``/tmp`` and ``/workspace``, read-only input
    bind mount, and no host environment beyond what the Docker client needs.
    """

    isolation = "docker"

    def __init__(
        self,
        image: str,
        limits: SandboxLimits | None = None,
        *,
        docker_binary: str = "docker",
        network: NetworkPolicy | None = None,
    ) -> None:
        resolved = shutil.which(docker_binary)
        if resolved is None:
            raise RuntimeError(f"Docker executable was not found: {docker_binary}")
        if not isinstance(image, str) or not _PINNED_IMAGE.fullmatch(image):
            raise ValueError("sandbox images must be pinned by sha256 digest")
        self.docker_binary = resolved
        self.image = image
        self.limits = limits or SandboxLimits()
        self.network = network or NetworkPolicy()
        if self.network.mode == "allowlist":
            self._verify_allowlist_network()

    def _verify_allowlist_network(self) -> None:
        result = subprocess.run(
            [self.docker_binary, "network", "inspect", self.network.network_name or ""],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30, check=False, env=_docker_env(),
        )
        if result.returncode:
            raise SandboxPolicyError(f"allowlist network cannot be inspected: {result.stderr.strip()[:500]}")
        try:
            described = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise SandboxPolicyError("docker network inspect returned invalid JSON") from exc
        if not isinstance(described, list) or len(described) != 1 or not isinstance(described[0], dict):
            raise SandboxPolicyError("docker network inspect must describe exactly one network")
        network = described[0]
        if network.get("Internal") is not True:
            raise SandboxPolicyError("allowlist network must be created with --internal (no external route)")
        containers = network.get("Containers") or {}
        if not isinstance(containers, dict):
            raise SandboxPolicyError("docker network inspect returned invalid container attachments")
        allowed_names = {host.split(":", 1)[0] for host in self.network.allowed_hosts}
        attached = {
            str(item.get("Name", "")) for item in containers.values() if isinstance(item, dict)
        }
        unexpected = sorted(name for name in attached if name not in allowed_names)
        if unexpected:
            raise SandboxPolicyError(f"allowlist network has non-allowlisted attached containers: {unexpected}")

    def execute(
        self,
        argv: Sequence[str],
        *,
        input_root: Path,
        cwd: str = ".",
        timeout: float | None = None,
        stdin: bytes | None = None,
        env: Mapping[str, str] | None = None,
        snapshot_dir: Path | None = None,
    ) -> CommandResult:
        command = validate_argv(argv)
        cwd_path = validate_relative_path(cwd, label="command cwd")
        limit = _timeout(self.limits, timeout)
        extra_env = _validate_env(env)
        source = Path(input_root).resolve()
        if not source.is_dir():
            raise ValueError("sandbox input root must be a directory")
        with tempfile.TemporaryDirectory(prefix="hermes-docker-") as temporary:
            root = Path(temporary)
            cidfile = root / "container.id"
            output_target = "-"
            mounts = ["--mount", f"type=bind,source={source},target=/input,readonly"]
            if snapshot_dir is not None:
                out = Path(snapshot_dir).resolve()
                out.mkdir(parents=True, exist_ok=True)
                out.chmod(0o777)
                mounts += ["--mount", f"type=bind,source={out},target=/output"]
                output_target = "/output"
            argv_docker = [
                self.docker_binary, "run", "--rm", "--pull=never",
                self.network.docker_argument(), "--read-only", "--user=65534:65534",
                "--cap-drop=ALL", "--security-opt=no-new-privileges",
                f"--pids-limit={self.limits.pids}", "--memory", self.limits.memory,
                "--cpus", self.limits.cpus, "--ipc=none",
                "--tmpfs", f"/tmp:rw,nosuid,nodev,size={self.limits.tmp_size}",
                "--tmpfs", f"/workspace:rw,nosuid,nodev,size={self.limits.workspace_size},mode=1777",
                *mounts,
                "--cidfile", str(cidfile),
            ]
            for key, value in sorted(extra_env.items()):
                argv_docker += ["--env", f"{key}={value}"]
            if stdin is not None:
                argv_docker.append("--interactive")
            argv_docker += [
                "--workdir", "/workspace", "--entrypoint", "/bin/sh", self.image,
                "-c", _CONTAINER_SCRIPT, "hermes-eval",
                "/input", "/workspace", output_target, str(cwd_path), *command,
            ]
            started = time.monotonic()
            timed_out = False
            try:
                result = subprocess.run(
                    argv_docker,
                    input=stdin,
                    stdin=None if stdin is not None else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=limit,
                    check=False,
                    env=_docker_env(),
                )
                exit_code = result.returncode
                output = result.stdout.decode("utf-8", errors="replace")[-_MAX_OUTPUT_CHARS:]
            except subprocess.TimeoutExpired as exc:
                self._kill_timed_out_container(cidfile)
                timed_out = True
                exit_code = 124
                raw = exc.stdout or b""
                output = (raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw))
                output = output[-_MAX_OUTPUT_CHARS:] + "\ncommand timed out"
            duration_ms = (time.monotonic() - started) * 1000
            if exit_code in {125, 126, 127}:
                raise RuntimeError(f"sandbox failed to execute command (exit {exit_code}): {output[-2000:]}")
            if exit_code != 0 and not timed_out and not cidfile.exists():
                # Docker writes the cidfile when it creates the container, so a
                # failure without one never reached a container: usually the
                # daemon selected by DOCKER_HOST/DOCKER_CONTEXT is unreachable.
                raise RuntimeError(
                    f"Docker could not create the sandbox container (exit {exit_code}); "
                    "check that the Docker daemon is running and reachable via "
                    f"DOCKER_HOST/DOCKER_CONTEXT: {output[-2000:]}"
                )
            snapshot = None
            if snapshot_dir is not None and not timed_out:
                snapshot = Path(snapshot_dir).resolve() / "workspace.tar"
                if not snapshot.is_file():
                    raise RuntimeError("sandbox did not produce a workspace snapshot")
            return CommandResult(exit_code, output, timed_out, duration_ms, snapshot)

    def _kill_timed_out_container(self, cidfile: Path) -> None:
        if not cidfile.exists():
            return
        container_id = cidfile.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            raise IntegrityError("Docker wrote an invalid container ID file")
        subprocess.run(
            [self.docker_binary, "kill", container_id],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=10, check=False, env=_docker_env(),
        )


# Variables that select and authenticate the Docker daemon the CLI talks to
# (remote, rootless or DinD daemons). They come from the operator's trusted
# runner environment and are forwarded only to the Docker client, never into
# containers. Everything else in the host environment is dropped.
DOCKER_CLIENT_ENV = ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "DOCKER_CONFIG")


def docker_client_selection() -> dict[str, str]:
    """Return the set (non-empty) Docker daemon-selection variables of this process."""
    return {name: os.environ[name] for name in DOCKER_CLIENT_ENV if os.environ.get(name)}


def _docker_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        **docker_client_selection(),
    }


def git_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_OPTIONAL_LOCKS": "0",
    }


def git_tree_digest(repository: Path, commit: str, max_bytes: int) -> str:
    output = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), "ls-tree", "-r", "-l", commit],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, env=git_env(),
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
    if total > max_bytes:
        raise IntegrityError("candidate source exceeds the configured workspace size limit")
    return hashlib.sha256(output.stdout).hexdigest()


def export_git_commit(
    repository: Path, commit: str, destination: Path, *, paths: Sequence[str] = (),
) -> None:
    """Extract ``commit`` (optionally only the tree below ``paths``) into ``destination``."""
    pathspec = [validate_relative_path(path, label="export path", allow_root=False).as_posix() for path in paths]
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository), "archive", "--format=tar", commit,
         *(["--", *pathspec] if pathspec else [])],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, env=git_env(),
    )
    if result.returncode:
        raise RuntimeError(f"cannot export Git commit: {result.stderr.decode(errors='replace')}")
    _extract_tar(io.BytesIO(result.stdout), destination, max_bytes=None, max_entries=None, strict=True)


def extract_workspace_snapshot(
    snapshot: Path, destination: Path, *, max_bytes: int, max_entries: int
) -> tuple[str, ...]:
    """Extract only regular files and directories; return skipped special entries.

    Symlinks, devices and other special objects produced inside the sandbox
    are never materialized on the host; they are reported so verifiers and
    reports can account for them.
    """
    with Path(snapshot).open("rb") as stream:
        return _extract_tar(stream, destination, max_bytes=max_bytes, max_entries=max_entries, strict=False)


def _extract_tar(
    stream, destination: Path, *, max_bytes: int | None, max_entries: int | None, strict: bool
) -> tuple[str, ...]:
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    total = 0
    count = 0
    try:
        archive = tarfile.open(fileobj=stream, mode="r:")
    except tarfile.TarError as exc:
        raise IntegrityError(f"workspace archive is not a valid tar file: {exc}") from exc
    with archive:
        for member in archive:
            count += 1
            if max_entries is not None and count > max_entries:
                raise IntegrityError("workspace snapshot exceeds the entry limit")
            name = member.name
            while name.startswith("./"):
                name = name[2:]
            if name in {"", "."}:
                continue
            relative = PurePosixPath(name)
            if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
                raise IntegrityError(f"unsafe path in workspace archive: {member.name!r}")
            target = root.joinpath(*relative.parts)
            parent = target.parent.resolve()
            if parent != root and root not in parent.parents:
                raise IntegrityError(f"archive path escapes workspace: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True, mode=0o755)
                continue
            if not member.isfile():
                if strict:
                    raise IntegrityError(f"unsupported archive object: {member.name!r}")
                skipped.append(relative.as_posix())
                continue
            total += member.size
            if max_bytes is not None and total > max_bytes:
                raise IntegrityError("workspace snapshot exceeds the size limit")
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            if target.is_symlink() or target.exists():
                raise IntegrityError(f"duplicate path in workspace archive: {member.name!r}")
            source = archive.extractfile(member)
            if source is None:
                raise IntegrityError(f"archive object has no data: {member.name!r}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(0o755 if member.mode & 0o111 else 0o644)
    return tuple(sorted(skipped))


def _file_digests(root: Path) -> dict[str, str]:
    digests: dict[str, str] = {}
    base = Path(root)
    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames.sort()
        for filename in sorted(filenames):
            path = Path(directory) / filename
            relative = path.relative_to(base).as_posix()
            if path.is_symlink() or not path.is_file():
                digests[relative] = "special"
                continue
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digests[relative] = digest.hexdigest()
    return digests


@dataclass(frozen=True)
class WorkspaceChanges:
    added: tuple[str, ...]
    modified: tuple[str, ...]
    deleted: tuple[str, ...]
    special: tuple[str, ...] = ()

    @property
    def changed_paths(self) -> tuple[str, ...]:
        return tuple(sorted({*self.added, *self.modified, *self.deleted, *self.special}))

    def entries(self, limit: int = 1000) -> tuple[str, ...]:
        rows = (
            [f"A {path}" for path in self.added]
            + [f"M {path}" for path in self.modified]
            + [f"D {path}" for path in self.deleted]
            + [f"S {path}" for path in self.special]
        )
        rows.sort(key=lambda row: (row[2:], row[0]))
        return tuple(rows[:limit])


def compute_changes(source: Path, result: Path, special: Sequence[str] = ()) -> WorkspaceChanges:
    before = _file_digests(source)
    after = _file_digests(result)
    return WorkspaceChanges(
        added=tuple(sorted(set(after) - set(before))),
        modified=tuple(sorted(path for path in set(before) & set(after) if before[path] != after[path])),
        deleted=tuple(sorted(set(before) - set(after))),
        special=tuple(sorted(special)),
    )


def unified_diff(source: Path, result: Path, changes: WorkspaceChanges, *, max_bytes: int = 1024 * 1024) -> str:
    """Git-style unified diff of text changes; binary files are summarized."""
    chunks: list[str] = []
    size = 0
    for path in sorted({*changes.added, *changes.modified, *changes.deleted}):
        before = _read_text(source / path)
        after = _read_text(result / path)
        if before is None and (source / path).exists() or after is None and (result / path).exists():
            text = f"diff --git a/{path} b/{path}\nBinary files differ\n"
        else:
            text = f"diff --git a/{path} b/{path}\n" + "".join(difflib.unified_diff(
                (before or "").splitlines(keepends=True),
                (after or "").splitlines(keepends=True),
                fromfile=f"a/{path}" if before is not None else "/dev/null",
                tofile=f"b/{path}" if after is not None else "/dev/null",
            ))
            if not text.endswith("\n"):
                text += "\n"
        size += len(text.encode())
        if size > max_bytes:
            chunks.append("diff truncated: size limit reached\n")
            break
        chunks.append(text)
    for path in changes.special:
        chunks.append(f"special file not materialized: {path}\n")
    return "".join(chunks)


def _read_text(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def glob_match(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(path, pattern) or path.startswith(pattern.rstrip("/") + "/") for pattern in patterns)
