from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

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
        ids: set[tuple[str, int]] = set()
        cluster_splits: dict[str, Split] = {}
        for index, benchmark in enumerate(benchmarks):
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
            for previous in benchmarks[:index]:
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
        return benchmarks

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
    split_dir = base / benchmark.split.value.lower()
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{benchmark.id}-v{benchmark.version}.json"
    path.write_text(json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_benchmarks(root: Path, *, holdout_root: Path | None = None) -> list[Benchmark]:
    return BenchmarkStore(root, holdout_root=holdout_root).validate()
