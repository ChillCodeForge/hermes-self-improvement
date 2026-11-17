"""Unit tests for spend-on-execution of the sealed holdout (spec §16)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from hermes_improve.errors import AccessDenied
from hermes_improve.holdout_gate import HoldoutSpendingRunner
from hermes_improve.models import Benchmark
from hermes_improve.security import (
    SpentHoldoutRegistry,
    consume_holdout_generation,
    holdout_generation,
    holdout_generation_spent,
)
from test_benchmarks import data


class RecordingRunner:
    llm_influenced = False

    def __init__(self):
        self.calls = []

    def run(self, commit, benchmark, seed):
        self.calls.append(benchmark.id)
        return ("observation", benchmark.id)


@pytest.fixture
def tasks():
    return (
        Benchmark.from_dict(data("public-a", "REGRESSION", "public-cluster")),
        Benchmark.from_dict(data("holdout-a", "SEALED_HOLDOUT", "holdout-a")),
        Benchmark.from_dict(data("holdout-b", "SEALED_HOLDOUT", "holdout-b")),
    )


def test_holdout_is_spent_once_right_before_the_first_holdout_task(tmp_path, tasks):
    public, holdout_a, holdout_b = tasks
    registry = SpentHoldoutRegistry(str(tmp_path / "ledger.sqlite3"))
    inner = RecordingRunner()
    spent_before_run = []

    def spend():
        spent_before_run.append(list(inner.calls))
        consume_holdout_generation(registry, [holdout_a, holdout_b], "lineage-1")

    runner = HoldoutSpendingRunner(inner, spend)
    assert runner.llm_influenced is False  # delegated to the wrapped runner
    runner.run("c", public, 0)
    assert not runner.holdout_spent
    assert not holdout_generation_spent(registry, [holdout_a, holdout_b], "lineage-1")
    runner.run("c", holdout_a, 0)
    runner.run("c", holdout_b, 0)
    runner.run("c", holdout_a, 1)
    assert runner.holdout_spent
    assert spent_before_run == [["public-a"]]  # one spend, committed before any holdout task ran
    assert inner.calls == ["public-a", "holdout-a", "holdout-b", "holdout-a"]
    # The spend is durable: a fresh registry on the same ledger sees it.
    reopened = SpentHoldoutRegistry(str(tmp_path / "ledger.sqlite3"))
    assert reopened.is_spent(holdout_generation([holdout_a, holdout_b]), "lineage:lineage-1")


def test_already_spent_generation_blocks_holdout_execution(tmp_path, tasks):
    public, holdout_a, holdout_b = tasks
    path = str(tmp_path / "ledger.sqlite3")
    consume_holdout_generation(SpentHoldoutRegistry(path), [holdout_a, holdout_b], "lineage-1")
    inner = RecordingRunner()
    runner = HoldoutSpendingRunner(
        inner, lambda: consume_holdout_generation(SpentHoldoutRegistry(path), [holdout_a, holdout_b], "lineage-1"),
    )
    runner.run("c", public, 0)
    with pytest.raises(AccessDenied, match="already consumed"):
        runner.run("c", holdout_a, 0)
    assert inner.calls == ["public-a"]
    assert not runner.holdout_spent


def test_concurrent_runs_of_one_lineage_execute_the_holdout_at_most_once(tmp_path, tasks):
    _, holdout_a, holdout_b = tasks
    path = str(tmp_path / "ledger.sqlite3")
    barrier = Barrier(4)
    inners = [RecordingRunner() for _ in range(4)]

    def attempt(inner):
        registry = SpentHoldoutRegistry(path)  # separate connection per "process"
        runner = HoldoutSpendingRunner(
            inner, lambda: consume_holdout_generation(registry, [holdout_a, holdout_b], "lineage-1"),
        )
        barrier.wait()
        try:
            runner.run("c", holdout_a, 0)
            return True
        except AccessDenied:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, inners))
    assert results.count(True) == 1
    assert sum(len(inner.calls) for inner in inners) == 1


def test_spend_precheck_does_not_consume(tmp_path, tasks):
    _, holdout_a, _ = tasks
    registry = SpentHoldoutRegistry(str(tmp_path / "ledger.sqlite3"))
    assert not holdout_generation_spent(registry, [holdout_a], "lineage-1")
    assert not holdout_generation_spent(registry, [holdout_a], "lineage-1")
    consume_holdout_generation(registry, [holdout_a], "lineage-1")
    assert holdout_generation_spent(registry, [holdout_a], "lineage-1")
    assert not holdout_generation_spent(registry, [holdout_a], "lineage-2")
    with pytest.raises(ValueError, match="lineage"):
        holdout_generation_spent(registry, [holdout_a], " ")
