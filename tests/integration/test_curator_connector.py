from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from hermes_curator_fixture import build_hermes_home, usage_record, write_pending
from hermes_improve.cli import main, run
from hermes_improve.curator_connector import CuratorConnector
from hermes_improve.decisions import DecisionRecorder


def _tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        stat = path.lstat()
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_connector_parses_real_curator_state_shapes(tmp_path):
    home = build_hermes_home(tmp_path)
    status = CuratorConnector(home).status()

    assert status["skills"]["total"] == 4
    assert status["skills"]["curator_managed"] == 3
    assert status["skills"]["by_state"] == {"active": 2, "archived": 1, "stale": 1}
    assert status["skills"]["never_used"] == 1
    assert status["scheduler"]["run_count"] == 8
    assert status["runs"]["latest"]["run_id"] == "20261109-102239"
    assert status["runs"]["consolidations"] == 1
    assert status["ledger"]["by_action"] == {"patch": 1, "pre-rollback": 1, "rollback": 1}
    assert status["ledger"]["rollbacks"] == 1
    assert status["backups"]["count"] == 1
    assert status["pending_skill_writes"][0]["pending_id"] == "1a2b3c4d"
    assert status["rejected_records"] == []


def test_connector_signals_cover_lifecycle_churn_overlap_rollback_and_pending(tmp_path):
    connector = CuratorConnector(build_hermes_home(tmp_path))
    kinds = {signal.evidence["kind"]: signal for signal in connector.signals()}

    assert set(kinds) == {
        "lifecycle_stale", "lifecycle_archived_used", "patch_churn",
        "overlap_consolidation", "ledger_rollback", "pending_write",
    }
    assert kinds["patch_churn"].evidence["name"] == "claude-code"
    assert kinds["overlap_consolidation"].evidence["into"] == "claude-code"
    assert all(signal.source == "curator" for signal in kinds.values())
    assert all(0.0 <= signal.impact_score <= 1.0 for signal in kinds.values())


def test_connector_never_writes_the_hermes_home(tmp_path):
    home = build_hermes_home(tmp_path)
    before = _tree_digest(home)
    connector = CuratorConnector(home)
    connector.status()
    connector.signals()
    connector.usage_rows()
    assert _tree_digest(home) == before


def test_connector_reports_malformed_records_instead_of_trusting_them(tmp_path):
    home = build_hermes_home(tmp_path)
    usage = json.loads((home / "skills" / ".usage.json").read_text())
    usage["bad-state"] = usage_record(state="deleted")
    usage["bad-counter"] = usage_record(use_count=-3)
    usage["leaky ghp_abcdefghijklmnopqrstuvwxyz1234"] = usage_record()
    (home / "skills" / ".usage.json").write_text(json.dumps(usage), encoding="utf-8")
    write_pending(home / "pending" / "skills", "badid001", {"action": "explode", "name": "demo"})

    status = CuratorConnector(home).status()
    assert status["skills"]["total"] == 4
    reasons = " ".join(item["reason"] for item in status["rejected_records"])
    assert "unknown lifecycle state" in reasons
    assert "use_count" in reasons
    assert "credential-free" in reasons
    assert "unknown or inconsistent skill_manage action" in reasons
    assert "ghp_" not in json.dumps(status)


def test_connector_refuses_symlinked_state_files(tmp_path):
    home = build_hermes_home(tmp_path)
    usage = home / "skills" / ".usage.json"
    secret = tmp_path / "elsewhere.json"
    secret.write_text(usage.read_text(), encoding="utf-8")
    usage.unlink()
    os.symlink(secret, usage)
    with pytest.raises(ValueError, match="symbolic link"):
        CuratorConnector(home).status()


def test_connector_bounds_the_ledger_to_its_tail(tmp_path, monkeypatch):
    home = build_hermes_home(tmp_path)
    monkeypatch.setattr("hermes_improve.curator_connector._MAX_LEDGER_TAIL_BYTES", 600)
    status = CuratorConnector(home).status()
    assert status["ledger"]["truncated_to_tail"] is True
    assert status["ledger"]["entries_read"] < 3


def test_connector_requires_an_existing_absolute_home(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        CuratorConnector("relative/home")
    with pytest.raises(FileNotFoundError):
        CuratorConnector(tmp_path / "missing")


def test_cli_curator_status_reports_live_state_and_records_signal_decisions(tmp_path, monkeypatch, capsys):
    home = build_hermes_home(tmp_path)
    state = tmp_path / "state"
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(state))
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "curator", "status", "--hermes-home", str(home)])

    assert main() == 0
    first = json.loads(capsys.readouterr().out)
    assert first["curator"]["skills"]["total"] == 4
    assert first["hook_signals"] == "no data"
    assert len(first["signals"]) == 6
    assert all(len(signal["decision_id"]) == 32 for signal in first["signals"])

    assert main() == 0
    capsys.readouterr()
    counts = DecisionRecorder(state / "experiments.sqlite3").counts()
    assert counts["curator_signal"]["total"] == 6


def test_cli_curator_status_uses_hermes_home_environment(tmp_path, monkeypatch, capsys):
    home = build_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(sys, "argv", ["hermes-improve", "curator", "status"])
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["curator"]["hermes_home"] == str(home)


def test_dream_light_reads_live_curator_usage(tmp_path, monkeypatch, capsys):
    home = build_hermes_home(tmp_path)
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "dream", "--stage", "light", "--curator-hermes-home", str(home),
    ])
    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["clusters"]["curator"] == 3
    assert {signal["evidence"]["name"] for signal in result["signals"]} == {
        "claude-code", "old-notes", "retired-helper",
    }


def test_dream_rejects_two_curator_sources(tmp_path, monkeypatch):
    home = build_hermes_home(tmp_path)
    report = tmp_path / "report.json"
    report.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "dream", "--curator-report", str(report), "--curator-hermes-home", str(home),
    ])
    with pytest.raises(ValueError, match="not both"):
        run()
