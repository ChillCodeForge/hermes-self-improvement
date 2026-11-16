"""How many paired runs each benchmark needs (spec §21)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import Benchmark, Split


SUITES = ("SMOKE", "CORE", "FULL", "SEALED_HOLDOUT")


@dataclass(frozen=True)
class RunPolicy:
    """Paired-run counts per suite.

    Deterministic runners (no model in the loop) need exactly one run per
    task. LLM-influenced runners need SMOKE 1-2, CORE >= 3 and
    FULL/SEALED_HOLDOUT >= 5 runs. Baseline and candidate always receive the
    identical seed list.
    """

    version: str = "1"
    deterministic_runs: int = 1
    smoke_runs: int = 2
    core_runs: int = 3
    full_runs: int = 5
    holdout_runs: int = 5
    base_seed: int = 0

    def __post_init__(self) -> None:
        counts = (self.deterministic_runs, self.smoke_runs, self.core_runs, self.full_runs, self.holdout_runs)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("run counts must be integers")
        if self.deterministic_runs != 1:
            raise ValueError("deterministic tasks run exactly once")
        if not 1 <= self.smoke_runs <= 2:
            raise ValueError("SMOKE runs must be between 1 and 2")
        if self.core_runs < 3:
            raise ValueError("CORE requires at least 3 runs for LLM-influenced tasks")
        if self.full_runs < 5 or self.holdout_runs < 5:
            raise ValueError("FULL and SEALED_HOLDOUT require at least 5 runs for LLM-influenced tasks")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int) or self.base_seed < 0:
            raise ValueError("base_seed must be a non-negative integer")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("run policy version must be a non-empty string")

    def runs_for(self, benchmark: Benchmark, suite: str, *, llm_influenced: bool) -> int:
        if not isinstance(llm_influenced, bool):
            raise TypeError("llm_influenced must be a boolean")
        name = suite.upper()
        if name not in SUITES:
            raise ValueError(f"unknown evaluation suite: {suite!r}")
        if not llm_influenced:
            return self.deterministic_runs
        if benchmark.split is Split.SEALED_HOLDOUT or name == "SEALED_HOLDOUT":
            return self.holdout_runs
        return {"SMOKE": self.smoke_runs, "CORE": self.core_runs, "FULL": self.full_runs}[name]

    def seeds(self, count: int) -> tuple[int, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("seed count must be a positive integer")
        return tuple(self.base_seed + index for index in range(count))

    def describe(self) -> dict[str, object]:
        return asdict(self)
