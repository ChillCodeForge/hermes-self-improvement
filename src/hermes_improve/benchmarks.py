from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from .errors import AccessDenied
from .models import Benchmark, Role, Split
from .security import AccessController, HoldoutGrant
from .task_similarity import are_near_duplicate_tasks


class BenchmarkStore:
    def __init__(
        self,
        root: Path,
        access: AccessController | None = None,
        *,
        holdout_root: Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.access = access or AccessController()
        self.holdout_root = Path(holdout_root) if holdout_root is not None else None

    def _load_root(self, root: Path, *, holdout: bool) -> list[Benchmark]:
        benchmarks: list[Benchmark] = []
        if not root.exists():
            return benchmarks
        for path in sorted(root.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path}: benchmark document must be an object")
            benchmark = Benchmark.from_dict(data)
            if holdout != (benchmark.split is Split.SEALED_HOLDOUT):
                raise ValueError(
                    f"{path}: sealed holdout tasks must be separated from public benchmarks"
                )
            benchmarks.append(benchmark)
        return benchmarks

    def load_public(self) -> list[Benchmark]:
        return self._load_root(self.root, holdout=False)

    def load_holdout(self) -> list[Benchmark]:
        if self.holdout_root is None:
            return []
        return self._load_root(self.holdout_root, holdout=True)

    def load_all(self) -> list[Benchmark]:
        return self.load_public() + self.load_holdout()

    def validate(self) -> list[Benchmark]:
        benchmarks = self.load_all()
        _validate_benchmark_set(benchmarks)
        return benchmarks

    def admit_public(self, benchmark: Benchmark) -> Path:
        if benchmark.split is Split.SEALED_HOLDOUT:
            raise ValueError("sealed holdout benchmarks cannot be admitted to the public store")
        if self.holdout_root is None:
            raise ValueError("public benchmark admission requires the sealed holdout root")
        public_root = self.root.resolve()
        holdout_root = self.holdout_root.resolve()
        if (
            public_root == holdout_root
            or public_root.is_relative_to(holdout_root)
            or holdout_root.is_relative_to(public_root)
        ):
            raise ValueError("sealed holdout root must be separate from the public benchmark tree")
        with _admission_lock(self.root):
            _validate_benchmark_set([*self.load_all(), benchmark])
            return write_benchmark(self.root, benchmark)

    @property
    def version(self) -> str:
        entries = [
            {"id": benchmark.id, "version": benchmark.version, "digest": benchmark.digest}
            for benchmark in self.validate()
        ]
        body = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(body).hexdigest()}"

    def list(self, role: Role = Role.OPERATOR, grant: HoldoutGrant | None = None) -> list[dict[str, str]]:
        visible = []
        if role is Role.GENERATOR and self.holdout_root is not None:
            raise AccessDenied("generator benchmark stores must not be configured with the sealed holdout root")
        benchmarks = self.load_public() if role is Role.GENERATOR else self.validate()
        for benchmark in benchmarks:
            try:
                self.access.authorize_benchmark_read(role, benchmark.split, grant)
            except AccessDenied:
                continue
            visible.append({"id": benchmark.id, "split": benchmark.split.value, "cluster_id": benchmark.cluster_id})
        return visible

    def get(
        self, benchmark_id: str, role: Role, grant: HoldoutGrant | None = None,
        version: int | None = None,
    ) -> Benchmark:
        if role is Role.GENERATOR:
            if self.holdout_root is not None:
                raise AccessDenied("generator benchmark stores must not be configured with the sealed holdout root")
            matches = [
                item for item in self.load_public()
                if item.id == benchmark_id and (version is None or item.version == version)
            ]
        else:
            matches = [
                item for item in self.validate()
                if item.id == benchmark_id and (version is None or item.version == version)
            ]
        if not matches:
            raise KeyError(f"benchmark {benchmark_id!r} version {version!r} was not found")
        if version is None:
            matches.sort(key=lambda item: item.version, reverse=True)
        benchmark = matches[0]
        self.access.authorize_benchmark_read(role, benchmark.split, grant)
        return benchmark


def write_benchmark(
    root: Path, benchmark: Benchmark, *, holdout_root: Path | None = None
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", benchmark.id):
        raise ValueError("benchmark id must be a safe 1-128 character identifier")
    if benchmark.split is Split.SEALED_HOLDOUT:
        if holdout_root is None:
            raise ValueError("sealed holdout benchmarks require a separate holdout_root")
        base = Path(holdout_root)
    else:
        base = Path(root)
    split_name = benchmark.split.value.lower()
    base.mkdir(parents=True, exist_ok=True)
    resolved_base = base.resolve()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(resolved_base, directory_flags)
    split_fd = -1
    temporary_name = f".benchmark-{secrets.token_hex(16)}"
    destination_name = f"{benchmark.id}-v{benchmark.version}.json"
    payload = json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n"
    try:
        try:
            os.mkdir(split_name, dir_fd=root_fd)
        except FileExistsError:
            pass
        try:
            split_fd = os.open(split_name, directory_flags, dir_fd=root_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    "benchmark split directory must remain inside its configured root"
                ) from exc
            raise
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=split_fd,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write(payload)
                destination.flush()
                os.fchmod(destination.fileno(), 0o644)
                os.fsync(destination.fileno())
            try:
                os.link(
                    temporary_name,
                    destination_name,
                    src_dir_fd=split_fd,
                    dst_dir_fd=split_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise ValueError(
                    f"benchmark destination already exists: {destination_name}"
                ) from exc
            os.fsync(split_fd)
        finally:
            os.unlink(temporary_name, dir_fd=split_fd)
    finally:
        if split_fd >= 0:
            os.close(split_fd)
        os.close(root_fd)
    return base / split_name / destination_name


@contextmanager
def _admission_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve()
    root_fd = os.open(resolved_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        descriptor = os.open(
            ".benchmark-admission.lock",
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    finally:
        os.close(root_fd)


def _validate_benchmark_set(benchmarks: Iterable[Benchmark]) -> None:
    rows = list(benchmarks)
    ids: set[tuple[str, int]] = set()
    cluster_splits: dict[str, Split] = {}
    for index, benchmark in enumerate(rows):
        identity = (benchmark.id, benchmark.version)
        if identity in ids:
            raise ValueError(f"duplicate benchmark id/version: {identity}")
        ids.add(identity)
        existing = cluster_splits.setdefault(benchmark.cluster_id, benchmark.split)
        if existing is not benchmark.split:
            raise ValueError(
                f"cluster {benchmark.cluster_id!r} leaks across splits "
                f"({existing.value}, {benchmark.split.value})"
            )
        for previous in rows[:index]:
            if (
                benchmark.split is not previous.split
                or benchmark.cluster_id != previous.cluster_id
            ) and are_near_duplicate_tasks(
                str(benchmark.task["prompt"]), str(previous.task["prompt"]),
            ):
                raise ValueError(
                    f"near-duplicate benchmarks {previous.id!r} and {benchmark.id!r} "
                    "must share the same split and cluster_id"
                )


def validate_benchmarks(root: Path, *, holdout_root: Path | None = None) -> list[Benchmark]:
    return BenchmarkStore(root, holdout_root=holdout_root).validate()
