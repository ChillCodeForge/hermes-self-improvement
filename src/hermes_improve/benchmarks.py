from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from .benchmark_schema import (
    BENCHMARK_FILE_SUFFIXES,
    CRITICALITIES,
    DIFFICULTIES,
    MAX_BENCHMARK_FILE_BYTES,
    TAXONOMY_CATEGORIES,
    decode_benchmark_document,
    load_benchmark_document,
    parse_benchmark_document,
)
from .clustering import find_similarity_split_leaks
from .eligibility import ELIGIBILITY_RANK, assess_benchmark
from .errors import AccessDenied
from .models import Benchmark, Eligibility, Role, Split
from .security import AccessController, HoldoutGrant
from .task_similarity import are_near_duplicate_tasks

# Top-level directories of the public benchmark root that hold task fixtures and
# manifests rather than benchmark records; the record loader skips them.
RESERVED_PUBLIC_DIRECTORIES = frozenset({"fixtures", "manifests"})


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
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not holdout and relative.parts[0] in RESERVED_PUBLIC_DIRECTORIES:
                continue
            if path.suffix.lower() not in BENCHMARK_FILE_SUFFIXES or path.is_dir():
                continue
            data = load_benchmark_document(path)
            benchmarks.append(_parse_benchmark(path, data, holdout=holdout))
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
            root_fd, split_fd = _open_benchmark_split(self.root, benchmark.split)
            try:
                split_name = benchmark.split.value.lower()
                public_benchmarks = _load_root_fd(
                    self.root,
                    root_fd,
                    holdout=False,
                    pinned_split_name=split_name,
                    pinned_split_fd=split_fd,
                )
                holdout_benchmarks = self.load_holdout()
                original_directory = os.fstat(split_fd)
                _validate_benchmark_set([
                    *public_benchmarks, *holdout_benchmarks, benchmark,
                ])
                current_directory = os.stat(
                    benchmark.split.value.lower(),
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(current_directory.st_mode)
                    or (current_directory.st_dev, current_directory.st_ino)
                    != (original_directory.st_dev, original_directory.st_ino)
                ):
                    raise ValueError("benchmark split directory changed during admission")
                return _write_benchmark_in_directory(
                    self.root, benchmark.split, split_fd, benchmark,
                )
            finally:
                os.close(split_fd)
                os.close(root_fd)

    @property
    def version(self) -> str:
        """Version of the *public* benchmark manifest (the experiment/evaluation binding).

        Experiments are created by the generator, which can never read the
        sealed holdout, so the shared identity covers public records only. The
        whole set (including the holdout) is still validated for leakage; the
        holdout's own identity is ``holdout_version`` and is signed into FULL
        evaluation reports separately.
        """
        public = [benchmark for benchmark in self.validate() if benchmark.split is not Split.SEALED_HOLDOUT]
        return _manifest_version(public)

    @property
    def holdout_version(self) -> str | None:
        """Version of the sealed holdout records, or ``None`` without a holdout."""
        holdout = [benchmark for benchmark in self.validate() if benchmark.split is Split.SEALED_HOLDOUT]
        return _manifest_version(holdout) if holdout else None

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
    root_fd, split_fd = _open_benchmark_split(base, benchmark.split)
    try:
        return _write_benchmark_in_directory(base, benchmark.split, split_fd, benchmark)
    finally:
        os.close(split_fd)
        os.close(root_fd)


def _open_benchmark_split(base: Path, split: Split) -> tuple[int, int]:
    split_name = split.value.lower()
    base.mkdir(parents=True, exist_ok=True)
    resolved_base = base.resolve()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(resolved_base, directory_flags)
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
    except Exception:
        os.close(root_fd)
        raise
    return root_fd, split_fd


def _write_benchmark_in_directory(
    base: Path, split: Split, split_fd: int, benchmark: Benchmark,
) -> Path:
    split_name = split.value.lower()
    temporary_name = f".benchmark-{secrets.token_hex(16)}"
    destination_name = f"{benchmark.id}-v{benchmark.version}.json"
    payload = json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n"
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
    return base / split_name / destination_name


def _load_root_fd(
    root: Path,
    root_fd: int,
    *,
    holdout: bool,
    pinned_split_name: str,
    pinned_split_fd: int,
) -> list[Benchmark]:
    benchmarks: list[Benchmark] = []
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    def visit(directory_fd: int, relative_path: Path) -> None:
        for name in sorted(os.listdir(directory_fd)):
            path = root / relative_path / name
            if not relative_path.parts and name == pinned_split_name:
                continue
            if not holdout and not relative_path.parts and name in RESERVED_PUBLIC_DIRECTORIES:
                continue
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    visit(child_fd, relative_path / name)
                finally:
                    os.close(child_fd)
            elif Path(name).suffix.lower() in BENCHMARK_FILE_SUFFIXES:
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError(f"{path}: benchmark file must be a regular file")
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
                with os.fdopen(descriptor, "rb") as source:
                    raw = source.read(MAX_BENCHMARK_FILE_BYTES + 1)
                data = decode_benchmark_document(name, raw)
                benchmarks.append(_parse_benchmark(path, data, holdout=holdout))

    visit(root_fd, Path())
    visit(pinned_split_fd, Path(pinned_split_name))
    return benchmarks


def _parse_benchmark(path: Path, data: object, *, holdout: bool) -> Benchmark:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: benchmark document must be an object")
    benchmark = parse_benchmark_document(data, source=str(path))
    if holdout != (benchmark.split is Split.SEALED_HOLDOUT):
        raise ValueError(
            f"{path}: sealed holdout tasks must be separated from public benchmarks"
        )
    return benchmark


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
    leaks = find_similarity_split_leaks(rows)
    if leaks:
        raise ValueError(
            "similar benchmarks must share one split: "
            + "; ".join(leak.describe() for leak in leaks)
        )


def validate_benchmark_set(benchmarks: Iterable[Benchmark]) -> None:
    """Public entry point for full corpus leakage validation (ids, clusters, similarity)."""
    _validate_benchmark_set(benchmarks)


def lint_public_corpus(benchmarks: Iterable[Benchmark]) -> None:
    """Curated-corpus rules applied by `benchmarks validate` on top of record validation.

    Records must use the shared category taxonomy and difficulty scale, must
    not claim a stronger eligibility class or higher reproducibility score
    than the classifier derives from their own fields, and a claimed
    successful replay must carry replay evidence.
    """
    errors: list[str] = []
    for benchmark in benchmarks:
        label = f"{benchmark.id} v{benchmark.version}"
        categories = benchmark.taxonomy.get("categories", [])
        if not categories:
            errors.append(f"{label}: at least one taxonomy category is required")
        unknown = sorted(set(categories) - set(TAXONOMY_CATEGORIES))
        if unknown:
            errors.append(f"{label}: unknown taxonomy categories {unknown}")
        if benchmark.taxonomy.get("difficulty") not in DIFFICULTIES:
            errors.append(f"{label}: difficulty must be one of {list(DIFFICULTIES)}")
        if benchmark.benchmark.get("criticality") not in CRITICALITIES:
            errors.append(f"{label}: criticality must be one of {list(CRITICALITIES)}")
        decision = assess_benchmark(benchmark)
        declared = Eligibility(benchmark.eligibility["class"])
        if ELIGIBILITY_RANK[declared] > ELIGIBILITY_RANK[decision.eligibility]:
            errors.append(
                f"{label}: declared {declared.value} exceeds classifier result "
                f"{decision.eligibility.value} (missing: {', '.join(decision.reasons) or 'none'})"
            )
        if float(benchmark.eligibility["reproducibility_score"]) > decision.reproducibility_score + 1e-9:
            errors.append(
                f"{label}: declared reproducibility_score exceeds computed "
                f"{decision.reproducibility_score}"
            )
        if (
            benchmark.eligibility.get("replay_verified") is True
            and not isinstance(benchmark.eligibility.get("replay_evidence"), dict)
        ):
            errors.append(f"{label}: replay_verified requires eligibility.replay_evidence")
    if errors:
        raise ValueError("benchmark corpus lint failed: " + "; ".join(errors))


def validate_benchmarks(root: Path, *, holdout_root: Path | None = None) -> list[Benchmark]:
    return BenchmarkStore(root, holdout_root=holdout_root).validate()


def _manifest_version(benchmarks: Iterable[Benchmark]) -> str:
    entries = [
        {"id": benchmark.id, "version": benchmark.version, "digest": benchmark.digest}
        for benchmark in benchmarks
    ]
    body = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(body).hexdigest()}"
