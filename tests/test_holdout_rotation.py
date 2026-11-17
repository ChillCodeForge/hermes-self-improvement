from __future__ import annotations

import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from benchmark_factory import record
from hermes_improve import holdout_rotation
from hermes_improve.benchmarks import BenchmarkStore
from hermes_improve.holdout_rotation import active_generation, rotate_holdout
from hermes_improve.security import SpentHoldoutRegistry, consume_holdout_generation


def staging_dir(base: Path, name: str, *identifiers: str) -> Path:
    directory = base / name
    directory.mkdir()
    for identifier in identifiers:
        (directory / f"{identifier}.json").write_text(
            json.dumps(record(identifier, "SEALED_HOLDOUT")), encoding="utf-8",
        )
    return directory


@pytest.fixture
def layout(tmp_path):
    public = tmp_path / "public"
    (public / "train").mkdir(parents=True)
    (public / "train" / "p.json").write_text(json.dumps(record("public-task", "TRAIN")))
    return tmp_path, public, tmp_path / "store"


def generations(store: Path) -> list[str]:
    root = store / "generations"
    return sorted(os.listdir(root)) if root.exists() else []


def test_initial_rotation_activates_immutable_generation_usable_by_evaluator(layout):
    tmp_path, public, store = layout
    staging = staging_dir(tmp_path, "stage-1", "holdout-a", "holdout-b")
    result = rotate_holdout(staging, store, public_root=public)
    payload = result.to_dict()
    assert payload.pop("semantic_layer")["status"] == "not_applied"
    assert payload == {"generation": result.generation, "record_count": 2, "status": "activated"}
    assert os.readlink(store / "active") == f"generations/{result.generation}"
    assert active_generation(store) == result.generation
    evaluator_view = BenchmarkStore(public, holdout_root=store / "active")
    holdout = evaluator_view.load_holdout()
    assert sorted(item.id for item in holdout) == ["holdout-a", "holdout-b"]
    evaluator_view.validate()
    registry = SpentHoldoutRegistry(str(tmp_path / "spent.sqlite3"))
    assert consume_holdout_generation(registry, holdout, "lineage-1") == result.generation
    generation_dir = store / "generations" / result.generation
    for path in [generation_dir, *generation_dir.iterdir()]:
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


def test_subsequent_rotation_switches_pointer_and_retains_history(layout):
    tmp_path, public, store = layout
    first = rotate_holdout(staging_dir(tmp_path, "s1", "holdout-a"), store, public_root=public)
    registry = SpentHoldoutRegistry(str(tmp_path / "spent.sqlite3"))
    registry.mark_spent(first.generation, "lineage:lineage-1")
    second = rotate_holdout(staging_dir(tmp_path, "s2", "holdout-b"), store, public_root=public)
    assert second.generation != first.generation
    assert active_generation(store) == second.generation
    assert generations(store) == sorted([first.generation, second.generation])
    assert registry.is_spent(first.generation, "lineage:lineage-1")
    assert (store / "generations" / first.generation / "holdout-a.json").is_file()


def test_unchanged_manifest_and_reactivation_are_rejected(layout):
    tmp_path, public, store = layout
    first = rotate_holdout(staging_dir(tmp_path, "s1", "holdout-a"), store, public_root=public)
    with pytest.raises(ValueError, match="unchanged"):
        rotate_holdout(staging_dir(tmp_path, "s1-again", "holdout-a"), store, public_root=public)
    rotate_holdout(staging_dir(tmp_path, "s2", "holdout-b"), store, public_root=public)
    with pytest.raises(ValueError, match="never overwritten or re-activated"):
        rotate_holdout(staging_dir(tmp_path, "s1-third", "holdout-a"), store, public_root=public)
    assert active_generation(store) != first.generation


def assert_preserved(store: Path, before: str | None, before_generations: list[str]) -> None:
    assert active_generation(store) == before
    assert generations(store) == before_generations


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda data: data["security"].__setitem__("sanitized", False), "sanitized"),
        (lambda data: data["benchmark"].__setitem__("split", "TRAIN"), "only SEALED_HOLDOUT"),
        (lambda data: data.__setitem__("version", "1"), "schema validation failed"),
        (lambda data: data["benchmark"].__setitem__("cluster_id", "cluster-public-task"), "leaks across splits"),
    ],
)
def test_invalid_staging_leaves_previous_generation_active(layout, mutate, message):
    tmp_path, public, store = layout
    previous = rotate_holdout(staging_dir(tmp_path, "good", "holdout-a"), store, public_root=public)
    bad = tmp_path / "bad"
    bad.mkdir()
    data = record("holdout-new", "SEALED_HOLDOUT")
    mutate(data)
    (bad / "x.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        rotate_holdout(bad, store, public_root=public)
    assert_preserved(store, previous.generation, [previous.generation])


def test_similarity_leak_with_public_corpus_is_rejected(layout):
    tmp_path, public, store = layout
    leaked = tmp_path / "leaked"
    leaked.mkdir()
    prompt = BenchmarkStore(public).load_public()[0].task["prompt"]
    (leaked / "x.json").write_text(json.dumps(record("holdout-copy", "SEALED_HOLDOUT", prompt=prompt + " now")))
    with pytest.raises(ValueError, match="near-duplicate|similar benchmarks"):
        rotate_holdout(leaked, store, public_root=public)
    assert active_generation(store) is None


def test_staging_path_and_entry_safety(layout):
    tmp_path, public, store = layout
    good = staging_dir(tmp_path, "good", "holdout-a")
    link = tmp_path / "link"
    link.symlink_to(good)
    with pytest.raises(ValueError, match="must not be a symlink"):
        rotate_holdout(link, store, public_root=public)
    with pytest.raises(ValueError, match="does not exist"):
        rotate_holdout(tmp_path / "missing", store, public_root=public)
    with pytest.raises(ValueError, match="outside the public"):
        rotate_holdout(good, public / "holdout-store", public_root=public)
    inside = staging_dir(store.parent, "store-inner", "holdout-z")
    with pytest.raises(ValueError, match="separate"):
        rotate_holdout(inside, inside / "nested-store", public_root=public)

    with_symlink = staging_dir(tmp_path, "with-symlink", "holdout-a")
    (with_symlink / "other.json").symlink_to(good / "holdout-a.json")
    with pytest.raises(ValueError, match="regular file"):
        rotate_holdout(with_symlink, store, public_root=public)
    hidden = staging_dir(tmp_path, "hidden", "holdout-a")
    (hidden / ".notes.json").write_text("{}")
    with pytest.raises(ValueError, match="unsafe or hidden"):
        rotate_holdout(hidden, store, public_root=public)
    stray = staging_dir(tmp_path, "stray", "holdout-a")
    (stray / "README.txt").write_text("notes")
    with pytest.raises(ValueError, match="not a benchmark record"):
        rotate_holdout(stray, store, public_root=public)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no benchmark records"):
        rotate_holdout(empty, store, public_root=public)
    assert active_generation(store) is None


def test_active_entry_must_be_managed_symlink(layout):
    tmp_path, public, store = layout
    store.mkdir()
    (store / "active").write_text("not a link")
    with pytest.raises(ValueError, match="must be a symlink"):
        rotate_holdout(staging_dir(tmp_path, "s", "holdout-a"), store, public_root=public)
    (store / "active").unlink()
    (store / "active").symlink_to("/etc")
    with pytest.raises(ValueError, match="managed generation"):
        rotate_holdout(staging_dir(tmp_path, "s2", "holdout-a"), store, public_root=public)


def test_existing_generation_directory_is_never_overwritten(layout, monkeypatch):
    tmp_path, public, store = layout
    staging = staging_dir(tmp_path, "s", "holdout-a")
    digest = holdout_rotation.holdout_generation_digest(
        [holdout_rotation.parse_benchmark_document(json.loads((staging / "holdout-a.json").read_text()))]
    )
    (store / "generations" / digest).mkdir(parents=True)
    (store / "generations" / digest / "planted.json").write_text("{}")
    with pytest.raises(ValueError, match="never overwritten"):
        rotate_holdout(staging, store, public_root=public)
    assert os.listdir(store / "generations" / digest) == ["planted.json"]
    assert active_generation(store) is None


def test_copy_failure_preserves_active_generation_and_cleans_up(layout, monkeypatch):
    tmp_path, public, store = layout
    previous = rotate_holdout(staging_dir(tmp_path, "s1", "holdout-a"), store, public_root=public)

    def failing_write(directory_fd, snapshot):
        os.close(os.open("partial.json", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=directory_fd))
        raise OSError("disk full")

    monkeypatch.setattr(holdout_rotation, "_write_snapshot", failing_write)
    with pytest.raises(OSError, match="disk full"):
        rotate_holdout(staging_dir(tmp_path, "s2", "holdout-b"), store, public_root=public)
    assert_preserved(store, previous.generation, [previous.generation])


def test_verification_mismatch_after_copy_is_rejected(layout, monkeypatch):
    tmp_path, public, store = layout
    original = holdout_rotation._write_snapshot

    def corrupting_write(directory_fd, snapshot):
        original(directory_fd, tuple((name, data.replace(b"Handle", b"Mangle")) for name, data in snapshot))

    monkeypatch.setattr(holdout_rotation, "_write_snapshot", corrupting_write)
    with pytest.raises(ValueError, match="differs from the validated staging snapshot"):
        rotate_holdout(staging_dir(tmp_path, "s", "holdout-a"), store, public_root=public)
    assert active_generation(store) is None and generations(store) == []


def test_rotation_never_relaxes_store_permissions(layout):
    tmp_path, public, store = layout
    store.mkdir(mode=0o700)
    os.chmod(store, 0o700)
    rotate_holdout(staging_dir(tmp_path, "s", "holdout-a"), store, public_root=public)
    assert stat.S_IMODE(store.stat().st_mode) == 0o700
    assert stat.S_IMODE((store / "generations").stat().st_mode) & 0o077 == 0


def test_concurrent_rotations_are_serialized(layout):
    tmp_path, public, store = layout
    stagings = [staging_dir(tmp_path, f"s{index}", f"holdout-{index}") for index in range(4)]
    barrier = Barrier(len(stagings))

    def rotate(staging):
        barrier.wait()
        return rotate_holdout(staging, store, public_root=public).generation

    with ThreadPoolExecutor(max_workers=len(stagings)) as pool:
        results = list(pool.map(rotate, stagings))
    assert len(set(results)) == 4
    assert generations(store) == sorted(results)
    assert active_generation(store) in results
    assert not [name for name in os.listdir(store) if name.startswith(".active-")]
    assert not [name for name in os.listdir(store / "generations") if name.startswith(".incoming-")]


def test_concurrent_rotation_of_same_manifest_activates_once(layout):
    tmp_path, public, store = layout
    stagings = [staging_dir(tmp_path, f"same{index}", "holdout-a") for index in range(3)]
    barrier = Barrier(len(stagings))

    def rotate(staging):
        barrier.wait()
        try:
            return rotate_holdout(staging, store, public_root=public).status
        except ValueError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(rotate, stagings))
    assert outcomes.count("activated") == 1
    assert all("unchanged" in item for item in outcomes if item != "activated")


def test_cli_rotation_prints_only_digest_count_and_status(layout, monkeypatch, capsys):
    from hermes_improve.cli import main

    tmp_path, public, store = layout
    staging = staging_dir(tmp_path, "s", "holdout-a")
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(public))
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "benchmarks", "holdout", "rotate", str(staging), "--store-root", str(store),
    ])
    assert main() == 0
    captured = capsys.readouterr()
    output = captured.out
    payload = json.loads(output)
    assert set(payload) == {"generation", "record_count", "status", "semantic_layer"}
    # The layer-5 status carries counts and the model id only, never content.
    assert payload["semantic_layer"]["status"] == "not_applied"
    assert "semantic clustering layer NOT applied" in captured.err
    assert "Handle" not in output and "holdout-a" not in output
