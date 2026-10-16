from __future__ import annotations

import json
from importlib.metadata import entry_points

import pytest

from hermes_improve.hermes_plugin import register
from hermes_improve.ingestion import SessionQueue


class PluginContext:
    def __init__(self, settings):
        self.settings = settings
        self.hooks = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def test_hermes_plugin_is_published_as_entry_point():
    entries = entry_points().select(group="hermes_agent.plugins")
    plugin = next(item for item in entries if item.name == "hermes-self-improvement")
    assert plugin.load() is register


def test_hermes_session_finalize_ingestion_is_opt_in(monkeypatch, tmp_path):
    def unexpected_load(_session_id):
        raise AssertionError("disabled observer must not read session data")

    monkeypatch.setattr("hermes_improve.hermes_plugin._load_session", unexpected_load)
    context = PluginContext({"state_dir": str(tmp_path)})
    register(context)
    context.hooks["on_session_finalize"](session_id="session-1")
    assert not (tmp_path / "sessions.sqlite3").exists()


def test_hermes_finalize_hook_loads_redacts_and_queues_persisted_transcript(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hermes_improve.hermes_plugin._load_session",
        lambda _session_id: (
            [
                {"role": "user", "content": "Build the package"},
                {
                    "role": "assistant",
                    "content": "Using a credential now",
                    "tool_calls": [{"function": {"name": "terminal", "arguments": "{}"}}],
                },
                {"role": "tool", "content": "Authorization: Bearer abcdefghijklmnop"},
            ],
            {"model": "local-model", "cwd": "/work/project"},
        ),
    )
    context = PluginContext({"auto_ingest": True, "state_dir": str(tmp_path)})
    register(context)
    context.hooks["on_session_finalize"](
        session_id="session-2", completed=True, failed=False,
    )

    queued = SessionQueue(tmp_path / "sessions.sqlite3").get("session-2")
    encoded = json.dumps(queued)
    assert queued["task"]["prompt"] == "Build the package"
    assert queued["tool_calls"] == [{"name": "terminal"}]
    assert queued["_evidence_trust"] == "untrusted"
    assert "abcdefghijklmnop" not in encoded
    assert "session-2" in encoded
    assert (tmp_path / "sessions.sqlite3").stat().st_mode & 0o077 == 0


def test_hermes_plugin_refuses_shared_state_directory(monkeypatch, tmp_path):
    from hermes_improve.hermes_plugin import _state_root

    tmp_path.chmod(0o755)
    context = PluginContext({"state_dir": str(tmp_path)})
    with pytest.raises(PermissionError, match="group or other"):
        _state_root(context)


def test_hermes_finalize_hook_logs_session_store_errors(monkeypatch, tmp_path, caplog):
    def fail(_session_id):
        raise OSError("session database unavailable")

    monkeypatch.setattr("hermes_improve.hermes_plugin._load_session", fail)
    context = PluginContext({"auto_ingest": True, "state_dir": str(tmp_path)})
    register(context)
    context.hooks["on_session_finalize"](session_id="session-3")
    assert "Could not ingest finalized Hermes session" in caplog.text
