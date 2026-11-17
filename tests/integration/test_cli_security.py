import pytest

from hermes_improve.cli import _require_role_uid


def test_role_uid_requires_current_process_to_match_configured_identity(monkeypatch):
    monkeypatch.setattr("hermes_improve.cli.os.geteuid", lambda: 1001)
    monkeypatch.setenv("HERMES_EVALUATOR_UID", "1001")
    monkeypatch.delenv("HERMES_PROMOTER_UID", raising=False)
    _require_role_uid("evaluator")

    monkeypatch.setenv("HERMES_PROMOTER_UID", "1002")
    _require_role_uid("evaluator")


def test_evaluator_and_promoter_must_not_share_configured_uid(monkeypatch):
    monkeypatch.setattr("hermes_improve.cli.os.geteuid", lambda: 1001)
    monkeypatch.setenv("HERMES_EVALUATOR_UID", "1001")
    monkeypatch.setenv("HERMES_PROMOTER_UID", "1001")
    with pytest.raises(PermissionError, match="distinct Unix UIDs"):
        _require_role_uid("evaluator")


def test_role_uid_requires_numeric_identity(monkeypatch):
    monkeypatch.setenv("HERMES_PROMOTER_UID", "operator")
    with pytest.raises(ValueError, match="numeric"):
        _require_role_uid("promoter")
