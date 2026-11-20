"""Stand-in for Hermes ``run_agent.AIAgent`` (v0.22.2) used by runner tests.

The driver constructs ``AIAgent(**kwargs)``, calls
``run_conversation(prompt, task_id=...)``, ``_convert_to_trajectory_format``
and ``close``, and reads ``session_input_tokens``/``session_output_tokens``/
``session_api_calls``. Instead of calling a model the fake reads the evaluated
skill from ``$HERMES_HOME/skills`` (only when the ``skills`` toolset is
enabled, like the real agent) and writes the answer into the task's ``cwd``
override, so brain commits change its behaviour. Behaviour switches come from
``fake_config.json`` next to this file; every construction is logged.

Modes ``slow`` and ``stuck`` emulate a run that outlives the evaluation
timeout: the agent keeps calling tools until ``interrupt()`` (Hermes' API to
stop a running conversation) is requested, then returns its messages with
``completed: False`` like the real loop. ``stuck`` ignores the interrupt, so
only the driver's SIGTERM fallback record survives.
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import batch_runner

_CONFIG = json.loads(Path(__file__).with_name("fake_config.json").read_text())


def _call_model(base_url, model, count, stream):
    """POST ``count`` chat completions like Hermes' OpenAI client (``include_usage`` when streaming)."""
    for index in range(count):
        body = {"model": model, "messages": [{"role": "user", "content": f"turn {index}"}]}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        headers = {"Content-Type": "application/json"}
        if os.environ.get("HERMES_EVAL_API_KEY"):
            headers["Authorization"] = "Bearer " + os.environ["HERMES_EVAL_API_KEY"]
        request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()


class AIAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_api_calls = 0
        self._interrupt_requested = False
        mode = _CONFIG.get("mode", "normal")
        if mode == "crash":
            print("Fatal error: interpreter crashed")
            os._exit(1)
        if mode == "agent_error":
            raise RuntimeError("model returned 500")

    def run_conversation(self, prompt, task_id):
        overrides = batch_runner.TASK_OVERRIDES.get(task_id, {})
        logged = {key: value for key, value in self.kwargs.items() if key != "api_key"}
        logged["api_key_set"] = bool(self.kwargs.get("api_key"))
        with open(_CONFIG["log"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd(),
                                  "overrides": overrides, "agent": logged, "task_id": task_id}) + "\n")
        answer = "unknown"
        if "skills" in self.kwargs["enabled_toolsets"]:
            skill = Path(os.environ["HERMES_HOME"]) / "skills" / "answer" / "SKILL.md"
            found = re.search(r"ANSWER=(\d+)", skill.read_text()) if skill.exists() else None
            answer = found.group(1) if found else "unknown"
        workspace = Path(overrides.get("cwd") or "/nonexistent-container-only")
        if workspace.is_dir():
            (workspace / "answer.txt").write_text(answer + "\n")
        if _CONFIG.get("mode") in ("slow", "stuck"):
            return self._outlive_the_timeout(prompt, honor_interrupt=_CONFIG["mode"] == "slow")
        _call_model(self.kwargs["base_url"], self.kwargs["model"], int(_CONFIG.get("model_calls", 0)),
                    bool(_CONFIG.get("stream", False)))
        calls = 500 if _CONFIG.get("mode") == "tool_storm" else 1
        self.session_input_tokens, self.session_output_tokens = _CONFIG.get("hermes_tokens", [0, 0])
        self.session_api_calls = 3
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "write_file", "repeat": calls}]},
            {"role": "tool", "content": json.dumps({"success": True})},
            {"role": "assistant", "content": f"The answer is {answer}."},
        ]
        return {"messages": messages, "completed": True, "partial": False, "api_calls": 3, "answer": answer}

    def _outlive_the_timeout(self, prompt, honor_interrupt):
        self.session_input_tokens, self.session_output_tokens = _CONFIG.get("hermes_tokens", [0, 0])
        messages = [{"role": "user", "content": prompt}]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"name": "terminal"}]})
            messages.append({"role": "tool", "content": json.dumps({"success": True})})
            self.session_api_calls += 1
            if honor_interrupt and self._interrupt_requested:
                messages.append({"role": "assistant", "content": "Interrupted before finishing."})
                return {"messages": messages, "completed": False, "interrupted": True,
                        "api_calls": self.session_api_calls}
            time.sleep(0.05)
        raise RuntimeError("the fake agent was never stopped")

    def interrupt(self, message=None):
        self._interrupt_requested = True

    def _convert_to_trajectory_format(self, messages, prompt, completed):
        answer = messages[-1]["content"]
        skill_turns = []
        if "skills" in self.kwargs["enabled_toolsets"]:
            # ``skill_view``/``skills_list`` in fake_config: emit those tool calls like the real agent.
            calls = [{"name": "skills_list", "arguments": {}}] if _CONFIG.get("skills_list") else []
            calls += [{"name": "skill_view", "arguments": {"name": name}} for name in _CONFIG.get("skill_view", [])]
            if calls:
                skill_turns.append({"from": "gpt", "value": "<think>\n</think>\n" + "\n".join(
                    "<tool_call>\n" + json.dumps(call) + "\n</tool_call>" for call in calls)})
        return [
            {"from": "system", "value": "You are a function calling AI model. <tools>[]</tools>"},
            {"from": "human", "value": prompt},
            *skill_turns,
            {"from": "gpt", "value": "<think>\nNeed to write the file.\n</think>\n<tool_call>\n"
                                     + json.dumps({"name": "write_file", "arguments": {"path": "answer.txt"}})
                                     + "\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n" + json.dumps(
                {"tool_call_id": "call_1", "name": "write_file", "content": {"success": True}}) + "\n</tool_response>"},
            {"from": "gpt", "value": f"<think>\n</think>\n{answer}"},
        ]

    def close(self):
        return None
