from __future__ import annotations

import json
import sys

import pytest

from hermes_improve.cli import main


def test_dream_light_accepts_bounded_curator_json_report(tmp_path, monkeypatch, capsys):
    report_path = tmp_path / "curator.json"
    report_path.write_text(json.dumps([
        {"name": "archive-skill", "use_count": 0, "patch_count": 0, "activity_count": 0},
        {"name": "active-skill", "use_count": 5, "patch_count": 1, "activity_count": 4},
    ]), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "dream", "--stage", "light",
        "--curator-report", str(report_path),
    ])

    assert main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["sessions_processed"] == 0
    assert result["clusters"]["curator"] == 1
    assert result["signals"][0]["evidence"]["name"] == "active-skill"


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": []},
        [{"name": "skill", "token": "not-accepted"}],
        [None],
    ],
)
def test_dream_light_rejects_invalid_curator_reports(tmp_path, monkeypatch, payload):
    report_path = tmp_path / "curator.json"
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "dream", "--stage", "light",
        "--curator-report", str(report_path),
    ])

    with pytest.raises(ValueError, match="Curator usage report"):
        main()


def test_dream_deep_rejects_curator_usage_report(tmp_path, monkeypatch):
    report_path = tmp_path / "curator.json"
    report_path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv("HERMES_IMPROVE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(sys, "argv", [
        "hermes-improve", "dream", "--stage", "deep",
        "--curator-report", str(report_path),
    ])

    with pytest.raises(ValueError, match="only by LIGHT and REM"):
        main()
