"""Spec §34 CLI completeness: every required command exists and the read-only ones run for real."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_curator_fixture import build_hermes_home
from hermes_improve import cli
from hermes_improve.promotion import GitPromoter
from hermes_improve.policy import PromotionPolicy
from hermes_improve.store import ExperimentStore
from test_promotion import _add_candidate, _repository, git

REPO = Path(__file__).resolve().parents[1]

# Spec §34, in spec order, with the minimal arguments each command requires.
SPEC_COMMANDS = (
    ("status",),
    ("ingest",),
    ("dream",),
    ("candidates", "list"),
    ("evaluate",),
    ("evolve",),
    ("experiments", "list"),
    ("experiments", "show"),
    ("benchmarks", "list"),
    ("benchmarks", "validate"),
    ("benchmarks", "replay"),
    ("releases", "list"),
    ("promote",),
    ("rollback",),
    ("curator", "status"),
)


@pytest.mark.parametrize("command", SPEC_COMMANDS, ids=" ".join)
def test_every_spec_command_exists_and_prints_help(command, capsys):
    # ``candidates``/``releases`` take their sub-action as a positional choice.
    argv = [command[0], "--help"] if command[0] in {"candidates", "releases"} else [*command, "--help"]
    with pytest.raises(SystemExit) as raised:
        cli.build_parser().parse_args(argv)
    assert raised.value.code == 0
    usage = capsys.readouterr().out
    assert usage.startswith("usage: hermes-improve " + command[0])
    if command[0] in {"candidates", "releases"}:
        assert command[1] in usage


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["hermes-improve", *argv])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    home = tmp_path / "state"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(home))
    monkeypatch.setenv("HERMES_IMPROVE_BENCHMARKS", str(REPO / "benchmarks"))
    monkeypatch.setenv("HERMES_IMPROVE_LAYOUT", str(tmp_path / "absent-layout.json"))
    monkeypatch.delenv("HERMES_IMPROVE_CONFIG", raising=False)
    return home


def test_status_on_an_empty_installation_reports_no_data_and_creates_nothing(cli_env, monkeypatch, capsys):
    status = _run(monkeypatch, capsys, "status")
    assert status["version"]
    assert status["brain_version"] == "no data" and status["active_release"] == "no data"
    assert status["counts"]["public_benchmarks"] > 0
    assert status["counts"]["experiments_by_decision"] == "no data"
    assert not cli_env.exists()


def test_read_only_commands_run_against_real_state(tmp_path, cli_env, monkeypatch, capsys):
    repository, baseline = _repository(tmp_path)
    store = ExperimentStore(cli_env / "experiments.sqlite3")
    first, private, public = _add_candidate(repository, store, "exp-1", "cand-1", baseline, "one\n")
    link = tmp_path / "brain-current"
    promoter = GitPromoter(store, repository, tmp_path / "releases", link, PromotionPolicy(), public)
    promoter.promote("exp-1", manifest_key=b"m" * 32)
    git(repository, "checkout", "-q", "main")
    _add_candidate(
        repository, store, "exp-2", "cand-2", first.commit, "two\n", signing_keys=(private, public),
    )
    promoter.promote("exp-2", manifest_key=b"m" * 32)
    _add_candidate(
        repository, store, "exp-3", "cand-3", first.commit, "three\n",
        signing_keys=(private, public), evaluate=False,
    )

    status = _run(monkeypatch, capsys, "status", "--deployment-link", str(link))
    assert status["brain_version"] == "brain-v2"
    assert status["brain_version_source"] == "deployment link"
    assert status["active_release"]["id"] == "brain-v2"
    assert status["active_release"]["status"] == "PROVISIONAL"
    assert status["deployment"] == {"link": str(link), "release": "brain-v2"}
    assert status["counts"]["experiments_by_decision"] == {"pending": 1, "promoted": 2}
    assert status["counts"]["candidates"] == 3 and status["counts"]["evaluation_runs"] == 2
    assert status["counts"]["releases_by_status"] == {"PROVISIONAL": 2}

    promoter.rollback("brain-v1", manifest_key=b"m" * 32)
    status = _run(monkeypatch, capsys, "status", "--deployment-link", str(link))
    assert status["brain_version"] == "brain-v1" and status["active_release"]["id"] == "brain-v1"
    assert status["counts"]["rollbacks"] == 1
    assert "mismatch" not in status["deployment"]
    # A pointer that disagrees with the ledger is reported, never silently trusted.
    link.unlink()
    link.symlink_to(tmp_path / "releases" / "brain-v2", target_is_directory=True)
    status = _run(monkeypatch, capsys, "status", "--deployment-link", str(link))
    assert status["brain_version"] == "brain-v2"
    assert "ledger's active release is brain-v1" in status["deployment"]["mismatch"]

    assert {row["id"] for row in _run(monkeypatch, capsys, "candidates", "list")} == {"cand-1", "cand-2", "cand-3"}
    assert {row["id"] for row in _run(monkeypatch, capsys, "experiments", "list")} == {"exp-1", "exp-2", "exp-3"}
    assert _run(monkeypatch, capsys, "experiments", "show", "exp-2")["id"] == "exp-2"
    releases = _run(monkeypatch, capsys, "releases", "list")
    assert {row["id"]: row["status"] for row in releases} == {"brain-v1": "PROVISIONAL", "brain-v2": "ROLLED_BACK"}
    assert _run(monkeypatch, capsys, "benchmarks", "validate")["valid_benchmarks"] > 0
    listed = _run(monkeypatch, capsys, "benchmarks", "list")
    assert listed and all(row["split"] != "SEALED_HOLDOUT" for row in listed)
    hermes_home = build_hermes_home(tmp_path)
    curator = _run(monkeypatch, capsys, "curator", "status", "--hermes-home", str(hermes_home))
    assert curator["curator"]["skills"]["total"] == 4
