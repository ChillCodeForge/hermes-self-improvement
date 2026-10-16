from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .errors import AccessDenied
from .models import Benchmark, Role, Split
from .security import AccessController, HoldoutGrant


class BenchmarkStore:
    def __init__(self, root: Path, access: AccessController | None = None) -> None:
        self.root = Path(root)
        self.access = access or AccessController()

    def load_all(self) -> list[Benchmark]:
        benchmarks: list[Benchmark] = []
        for path in sorted(self.root.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path}: benchmark document must be an object")
            benchmarks.append(Benchmark.from_dict(data))
        return benchmarks

    def validate(self) -> list[Benchmark]:
        benchmarks = self.load_all()
        ids: set[tuple[str, int]] = set()
        cluster_splits: dict[str, Split] = {}
        for benchmark in benchmarks:
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
        return benchmarks

    def list(self, role: Role = Role.OPERATOR, grant: HoldoutGrant | None = None) -> list[dict[str, str]]:
        visible = []
        for benchmark in self.validate():
            try:
                self.access.authorize_benchmark_read(role, benchmark.split, grant)
            except AccessDenied:
                continue
            visible.append({"id": benchmark.id, "split": benchmark.split.value, "cluster_id": benchmark.cluster_id})
        return visible

    def get(self, benchmark_id: str, role: Role, grant: HoldoutGrant | None = None) -> Benchmark:
        matches = [item for item in self.validate() if item.id == benchmark_id]
        if len(matches) != 1:
            raise KeyError(f"expected one benchmark named {benchmark_id!r}, found {len(matches)}")
        benchmark = matches[0]
        self.access.authorize_benchmark_read(role, benchmark.split, grant)
        return benchmark


def write_benchmark(root: Path, benchmark: Benchmark) -> Path:
    split_dir = Path(root) / benchmark.split.value.lower()
    split_dir.mkdir(parents=True, exist_ok=True)
    path = split_dir / f"{benchmark.id}-v{benchmark.version}.json"
    path.write_text(json.dumps(benchmark.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_benchmarks(root: Path) -> list[Benchmark]:
    return BenchmarkStore(root).validate()
