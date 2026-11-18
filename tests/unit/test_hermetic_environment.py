"""Regression for finding F4: tests must not pick up a host's provisioned layout.

``tests/conftest.py`` pins ``HERMES_IMPROVE_LAYOUT`` and the state homes to
absent temporary paths and clears every ``HERMES_*``/``QWEN_*`` variable. These
tests simulate a provisioned default layout and check the CLI still runs in
unprovisioned development mode instead of verifying the host layout.
"""

import json
import os
import sys
from pathlib import Path

import pytest

from hermes_improve import isolation
from hermes_improve.isolation import enforce_isolation, resolve_layout_path


@pytest.fixture
def provisioned_default_layout(tmp_path, monkeypatch):
    """A well-formed layout at the (patched) default path, as on a provisioned host."""
    path = tmp_path / "srv" / "config" / "layout.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "root": str(tmp_path / "srv"),
        "users": {"agent": "hsi-f4-agent", "evaluator": "hsi-f4-eval", "promoter": "hsi-f4-prom"},
        "ledger_group": "hsi-f4-ledger",
        "cli": "/usr/bin/hermes-improve",
    }))
    path.chmod(0o644)
    monkeypatch.setattr(isolation, "DEFAULT_LAYOUT_PATH", path)
    return path


def test_layout_and_state_homes_point_at_absent_temporary_paths(provisioned_default_layout):
    layout = resolve_layout_path()
    assert layout != provisioned_default_layout
    assert not layout.exists()
    assert layout.parent == Path(os.environ["HERMES_IMPROVE_HOME"]).parent
    for name in ("HERMES_IMPROVE_HOME", "HERMES_HOME"):
        home = Path(os.environ[name])
        assert not home.exists()
        assert Path.home() not in home.parents


def test_role_and_operator_variables_are_cleared():
    leaked = sorted(
        name for name in os.environ
        if name.startswith(("HERMES_", "QWEN_"))
        and name not in {"HERMES_IMPROVE_LAYOUT", "HERMES_IMPROVE_HOME", "HERMES_HOME"}
    )
    assert leaked == []


def test_unprovisioned_mode_ignores_a_provisioned_default_layout(provisioned_default_layout, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_IMPROVE_ISOLATION", "unprovisioned")
    assert enforce_isolation("promoter") is None
    assert "without a provisioned isolation layout" in capsys.readouterr().err
    # Control: without the pin the simulated host layout is verified, which is the F4 leak.
    monkeypatch.delenv("HERMES_IMPROVE_LAYOUT")
    assert resolve_layout_path() == provisioned_default_layout
    with pytest.raises(PermissionError):
        enforce_isolation("promoter")


def test_cli_releases_command_stays_hermetic_with_a_provisioned_default_layout(
    provisioned_default_layout, monkeypatch, capsys,
):
    from hermes_improve.cli import main

    monkeypatch.setattr(sys, "argv", ["hermes-improve", "releases", "list"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out) == []
    assert (Path(os.environ["HERMES_IMPROVE_HOME"]) / "experiments.sqlite3").exists()
