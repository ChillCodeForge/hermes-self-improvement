"""Operator-managed sealed-holdout generation rotation (spec §16, issue #7).

Managed store layout::

    STORE_ROOT/
      .rotation.lock              # flock; serializes rotations
      generations/<digest>/...    # immutable sealed generations (read-only modes)
      active -> generations/<digest>

`<digest>` is the sealed-manifest digest computed exactly like
`security.consume_holdout_generation`, so spent-holdout registry rows and
generation directories share one identifier. The evaluator consumes the active
generation through `--holdout-root STORE_ROOT/active`.

Rotation snapshots the operator staging directory (regular files only, no
symlinks), validates every record with the published schema plus semantic
validation, requires SEALED_HOLDOUT splits, runs the full public/holdout
leakage checks, copies the snapshot into a private temporary directory,
re-reads and re-verifies it, removes write permission bits, renames it into
place and only then atomically replaces the `active` symlink. Any failure
leaves the previous active generation untouched. Generations are never
overwritten, re-activated or deleted; the spent-holdout registry is never
touched. File modes are only ever tightened, never relaxed.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import secrets
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .benchmark_schema import (
    BENCHMARK_FILE_SUFFIXES,
    MAX_BENCHMARK_FILE_BYTES,
    decode_benchmark_document,
    parse_benchmark_document,
)
from .benchmarks import BenchmarkStore, validate_benchmark_set
from .clustering import SemanticReport
from .embeddings import SemanticLayer
from .models import Benchmark, Split
from .security import manifest_digest

GENERATIONS_DIRECTORY = "generations"
ACTIVE_LINK = "active"
LOCK_FILE = ".rotation.lock"
MAX_RECORDS = 10_000
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_GENERATION = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


@dataclass(frozen=True)
class RotationResult:
    generation: str
    record_count: int
    status: str
    # Layer-5 status of the leakage check (counts and model id only, no content).
    semantic_layer: SemanticReport | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "generation": self.generation, "record_count": self.record_count, "status": self.status,
        }
        if self.semantic_layer is not None:
            result["semantic_layer"] = self.semantic_layer.to_dict()
        return result


def holdout_generation_digest(benchmarks: Iterable[Benchmark]) -> str:
    """Sealed-manifest digest; identical to `consume_holdout_generation`'s generation."""
    manifest = sorted(
        (
            {"id": item.id, "version": item.version, "digest": item.digest}
            for item in benchmarks
            if item.split is Split.SEALED_HOLDOUT
        ),
        key=lambda item: (item["id"], item["version"]),
    )
    if not manifest:
        raise ValueError("at least one sealed holdout benchmark is required")
    return manifest_digest({"sealed_holdout": manifest})


def active_generation(store_root: Path) -> str | None:
    """Return the active generation digest, or None before the first rotation."""
    link = Path(store_root) / ACTIVE_LINK
    try:
        info = os.lstat(link)
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(info.st_mode):
        raise ValueError("holdout store 'active' entry must be a symlink")
    return _generation_from_target(os.readlink(link))


def rotate_holdout(
    staging: Path, store_root: Path, *, public_root: Path, semantic: SemanticLayer | None = None,
) -> RotationResult:
    """Validate STAGING and activate it as a new immutable holdout generation."""
    staging = Path(staging)
    store_root = Path(store_root)
    _require_real_directory(staging, "holdout staging directory")
    store_exists = os.path.lexists(store_root)
    if store_exists:
        _require_real_directory(store_root, "holdout store root")
    staging_resolved = staging.resolve(strict=True)
    store_resolved = store_root.resolve()
    public_resolved = Path(public_root).resolve()
    for left, right, message in (
        (staging_resolved, store_resolved, "staging directory and holdout store must be separate"),
        (staging_resolved, public_resolved, "staging directory must be outside the public benchmark tree"),
        (store_resolved, public_resolved, "holdout store must be outside the public benchmark tree"),
    ):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError(message)
    if not store_exists:
        try:
            os.mkdir(store_resolved, 0o700)
        except FileExistsError:
            _require_real_directory(store_resolved, "holdout store root")

    store_fd = os.open(store_resolved, _DIRECTORY_FLAGS)
    try:
        with _rotation_lock(store_fd):
            previous = _read_active(store_fd)
            snapshot = _snapshot_tree(staging_resolved)
            records = _parse_snapshot(snapshot)
            public = BenchmarkStore(public_resolved).load_public()
            leakage = validate_benchmark_set([*public, *records], semantic=semantic)
            generation = holdout_generation_digest(records)
            generations_fd = _open_or_create_directory(store_fd, GENERATIONS_DIRECTORY)
            try:
                if previous == generation:
                    raise ValueError("holdout manifest is unchanged; generation is already active")
                if _exists(generations_fd, generation):
                    raise ValueError(
                        "holdout generation already exists; generations are never overwritten "
                        "or re-activated"
                    )
                incoming = f".incoming-{secrets.token_hex(16)}"
                os.mkdir(incoming, 0o700, dir_fd=generations_fd)
                try:
                    incoming_fd = os.open(incoming, _DIRECTORY_FLAGS, dir_fd=generations_fd)
                    try:
                        _write_snapshot(incoming_fd, snapshot)
                        copied = _snapshot_fd(incoming_fd, "copied generation")
                        if copied != snapshot:
                            raise ValueError("copied holdout generation differs from the validated staging snapshot")
                        if holdout_generation_digest(_parse_snapshot(copied)) != generation:
                            raise ValueError("copied holdout generation digest mismatch")
                        _seal_tree(incoming_fd)
                    finally:
                        os.close(incoming_fd)
                    if _exists(generations_fd, generation):
                        raise ValueError("holdout generation appeared during rotation")
                    os.rename(incoming, generation, src_dir_fd=generations_fd, dst_dir_fd=generations_fd)
                    os.fsync(generations_fd)
                except BaseException:
                    _remove_incoming(generations_fd, incoming)
                    raise
            finally:
                os.close(generations_fd)
            temporary_link = f".active-{secrets.token_hex(16)}"
            os.symlink(f"{GENERATIONS_DIRECTORY}/{generation}", temporary_link, dir_fd=store_fd)
            try:
                os.replace(temporary_link, ACTIVE_LINK, src_dir_fd=store_fd, dst_dir_fd=store_fd)
            except BaseException:
                os.unlink(temporary_link, dir_fd=store_fd)
                raise
            os.fsync(store_fd)
            return RotationResult(generation, len(records), "activated", leakage.semantic)
    finally:
        os.close(store_fd)


def _require_real_directory(path: Path, label: str) -> None:
    try:
        info = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} must be a directory")


@contextmanager
def _rotation_lock(store_fd: int) -> Iterator[None]:
    descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=store_fd)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _read_active(store_fd: int) -> str | None:
    try:
        info = os.stat(ACTIVE_LINK, dir_fd=store_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISLNK(info.st_mode):
        raise ValueError("holdout store 'active' entry must be a symlink")
    return _generation_from_target(os.readlink(ACTIVE_LINK, dir_fd=store_fd))


def _generation_from_target(target: str) -> str:
    prefix = f"{GENERATIONS_DIRECTORY}/"
    generation = target[len(prefix):] if target.startswith(prefix) else ""
    if not _GENERATION.fullmatch(generation):
        raise ValueError("holdout store 'active' symlink does not point at a managed generation")
    return generation


def _open_or_create_directory(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"holdout store {name!r} must be a real directory") from exc
        raise


def _exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


Snapshot = tuple[tuple[str, bytes], ...]


def _snapshot_tree(root: Path) -> Snapshot:
    root_fd = os.open(root, _DIRECTORY_FLAGS)
    try:
        return _snapshot_fd(root_fd, "staging")
    finally:
        os.close(root_fd)


def _snapshot_fd(root_fd: int, label: str) -> Snapshot:
    files: list[tuple[str, bytes]] = []

    def visit(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            relative = f"{prefix}{name}"
            if not _SAFE_NAME.fullmatch(name):
                raise ValueError(f"{label} entry {relative!r} has an unsafe or hidden name")
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                try:
                    visit(child_fd, f"{relative}/")
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode):
                if Path(name).suffix.lower() not in BENCHMARK_FILE_SUFFIXES:
                    raise ValueError(f"{label} entry {relative!r} is not a benchmark record file")
                if info.st_nlink != 1:
                    raise ValueError(f"{label} entry {relative!r} must not be hard-linked")
                if info.st_size > MAX_BENCHMARK_FILE_BYTES:
                    raise ValueError(f"{label} entry {relative!r} exceeds the record size limit")
                descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                with os.fdopen(descriptor, "rb") as source:
                    data = source.read(MAX_BENCHMARK_FILE_BYTES + 1)
                if len(data) > MAX_BENCHMARK_FILE_BYTES:
                    raise ValueError(f"{label} entry {relative!r} exceeds the record size limit")
                files.append((relative, data))
                if len(files) > MAX_RECORDS:
                    raise ValueError(f"{label} contains more than {MAX_RECORDS} records")
            else:
                raise ValueError(f"{label} entry {relative!r} must be a regular file or directory")

    visit(root_fd, "")
    return tuple(files)


def _parse_snapshot(snapshot: Snapshot) -> list[Benchmark]:
    if not snapshot:
        raise ValueError("holdout staging directory contains no benchmark records")
    records = []
    for relative, data in snapshot:
        benchmark = parse_benchmark_document(
            decode_benchmark_document(relative, data), source=relative,
        )
        if benchmark.split is not Split.SEALED_HOLDOUT:
            raise ValueError(f"{relative}: holdout generations may contain only SEALED_HOLDOUT records")
        records.append(benchmark)
    return records


def _write_snapshot(root_fd: int, snapshot: Snapshot) -> None:
    opened: dict[str, int] = {"": root_fd}
    try:
        for relative, data in snapshot:
            parts = relative.split("/")
            parent = ""
            for part in parts[:-1]:
                child = f"{parent}{part}/"
                if child not in opened:
                    os.mkdir(part, 0o750, dir_fd=opened[parent])
                    opened[child] = os.open(part, _DIRECTORY_FLAGS, dir_fd=opened[parent])
                parent = child
            descriptor = os.open(
                parts[-1],
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o640,
                dir_fd=opened[parent],
            )
            with os.fdopen(descriptor, "wb") as destination:
                destination.write(data)
                destination.flush()
                os.fsync(destination.fileno())
        for descriptor in opened.values():
            os.fsync(descriptor)
    finally:
        for key, descriptor in opened.items():
            if key:
                os.close(descriptor)


def _seal_tree(directory_fd: int) -> None:
    """Remove write bits bottom-up; never adds any permission."""
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                _seal_tree(child_fd)
            finally:
                os.close(child_fd)
        else:
            descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            try:
                os.fchmod(descriptor, stat.S_IMODE(info.st_mode) & ~_WRITE_BITS)
            finally:
                os.close(descriptor)
    os.fchmod(directory_fd, stat.S_IMODE(os.fstat(directory_fd).st_mode) & ~_WRITE_BITS)


def _remove_incoming(generations_fd: int, name: str) -> None:
    """Delete a failed private temporary copy (never a published generation)."""
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=generations_fd)
    except FileNotFoundError:
        return
    try:
        _remove_contents(directory_fd)
    finally:
        os.close(directory_fd)
    os.rmdir(name, dir_fd=generations_fd)


def _remove_contents(directory_fd: int) -> None:
    os.fchmod(directory_fd, stat.S_IMODE(os.fstat(directory_fd).st_mode) | stat.S_IWUSR | stat.S_IXUSR)
    for name in os.listdir(directory_fd):
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            try:
                _remove_contents(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)
