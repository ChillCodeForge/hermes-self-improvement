from __future__ import annotations

import sqlite3

from hermes_improve.hermes_plugin import HookSignalStore, register


class PluginContext:
    def __init__(self, settings):
        self.settings = settings
        self.hooks = {}

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def register_hook(self, name, callback):
        self.hooks[name] = callback


def _state_dir(tmp_path):
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    state.chmod(0o700)
    return state


def test_observer_hooks_are_not_registered_unless_opted_in(tmp_path):
    context = PluginContext({"state_dir": str(_state_dir(tmp_path)), "observe_skill_lifecycle": "true"})
    register(context)
    assert set(context.hooks) == {"on_session_finalize"}


def test_skill_lifecycle_hook_records_hermes_kwargs(tmp_path):
    state = _state_dir(tmp_path)
    context = PluginContext({"state_dir": str(state), "observe_skill_lifecycle": True})
    register(context)
    hook = context.hooks["on_skill_lifecycle"]
    # Kwargs exactly as tools/skill_usage.py::_emit_skill_lifecycle passes them.
    assert hook(action="loaded", skill_name="demo", provenance="agent_created", task_id="t1",
                session_id="s1", use_count=3, reused=True, reuse_after_patch=True) is None
    assert hook(action="patched", skill_name="demo", provenance="agent_created", task_id="",
                session_id="s1", use_count=None, reused=None, reuse_after_patch=None) is None

    summary = HookSignalStore(state / "hook_signals.sqlite3").summary()
    assert summary["skill_lifecycle_actions"] == {"loaded": 1, "patched": 1}
    assert summary["skills"] == [{"skill_name": "demo", "loaded": 1, "patched": 1, "reuse_after_patch": 1}]
    assert (state / "hook_signals.sqlite3").stat().st_mode & 0o077 == 0


def test_pre_tool_call_hook_counts_names_only_and_never_blocks(tmp_path):
    state = _state_dir(tmp_path)
    context = PluginContext({"state_dir": str(state), "observe_tool_calls": True})
    register(context)
    hook = context.hooks["pre_tool_call"]
    # Kwargs as hermes_cli/plugins.py::_get_pre_tool_call_directive_details passes them.
    for _ in range(2):
        result = hook(tool_name="terminal", args={"command": "cat secrets.env"}, task_id="t", session_id="s",
                      tool_call_id="c", turn_id="u", api_request_id="a", middleware_trace=[])
        assert result is None
    with sqlite3.connect(state / "hook_signals.sqlite3") as db:
        rows = db.execute("SELECT tool_name, count FROM tool_call_counts").fetchall()
        dumped = "\n".join(db.iterdump())
    assert rows == [("terminal", 2)]
    assert "secrets.env" not in dumped


def test_observer_failures_never_reach_hermes(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    context = PluginContext({
        "state_dir": str(shared), "observe_tool_calls": True, "observe_skill_lifecycle": True,
    })
    register(context)
    assert context.hooks["pre_tool_call"](tool_name="terminal") is None
    assert context.hooks["on_skill_lifecycle"](action="loaded", skill_name="demo", provenance="local") is None
    assert not (shared / "hook_signals.sqlite3").exists()

    state = _state_dir(tmp_path)
    context = PluginContext({"state_dir": str(state), "observe_skill_lifecycle": True})
    register(context)
    assert context.hooks["on_skill_lifecycle"](action="exploded", skill_name="demo", provenance="local") is None
    assert context.hooks["on_skill_lifecycle"](
        action="loaded", skill_name="ghp_abcdefghijklmnopqrstuvwxyz123456", provenance="local",
    ) is None
    assert HookSignalStore(state / "hook_signals.sqlite3").summary()["skill_lifecycle_actions"] == {}
