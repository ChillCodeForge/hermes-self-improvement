"""Expected refusals end with a one-line error, not a traceback (lab finding F23)."""

from __future__ import annotations

import json
import sys

import pytest

from hermes_improve import cli
from hermes_improve.errors import AccessDenied, IntegrityError, PolicyRejected
from hermes_improve.monitoring import RollbackBlocked


def _fail_status_with(monkeypatch, error):
    def raise_error(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "build_status", raise_error)
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "status"])


@pytest.mark.parametrize("error", [
    ValueError("requested release is already active"),
    RollbackBlocked("brain-v1 regressed but has no earlier active release to roll back to"),
    RuntimeError("HERMES_IMPROVE_MANIFEST_KEY is required"),
    IntegrityError("release manifest signature or digest verification failed"),
    AccessDenied("holdout access denied"),
    PolicyRejected("candidate tree contains paths that collide case-insensitively"),
    PermissionError("process uid 0 is not the promoter user"),
    FileNotFoundError("observations.jsonl"),
])
def test_expected_refusals_print_one_line_and_exit_non_zero(monkeypatch, capsys, error):
    _fail_status_with(monkeypatch, error)
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"


def test_unknown_record_names_what_was_not_found(monkeypatch, capsys):
    _fail_status_with(monkeypatch, KeyError("brain-v9"))
    assert cli.main() == 1
    assert capsys.readouterr().err == "error: not found: 'brain-v9'\n"


def test_unexpected_errors_keep_their_traceback(monkeypatch):
    _fail_status_with(monkeypatch, TypeError("bug"))
    with pytest.raises(TypeError, match="bug"):
        cli.main()


def test_debug_environment_shows_the_traceback_of_expected_errors(monkeypatch):
    _fail_status_with(monkeypatch, ValueError("requested release is already active"))
    monkeypatch.setenv("HERMES_IMPROVE_DEBUG", "1")
    with pytest.raises(ValueError, match="already active"):
        cli.main()


def test_run_raises_for_callers_that_handle_errors_themselves(monkeypatch):
    _fail_status_with(monkeypatch, ValueError("refused"))
    with pytest.raises(ValueError, match="refused"):
        cli.run()


def test_real_operator_refusal_from_the_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "home"))
    observations = tmp_path / "bad.jsonl"
    observations.write_text(json.dumps({"release_id": "brain-v1"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "monitor", "ingest", str(observations)])
    assert cli.main() == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert err.count("\n") == 1
    assert "Traceback" not in err
