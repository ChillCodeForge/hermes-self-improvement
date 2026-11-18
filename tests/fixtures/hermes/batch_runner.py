"""Stand-in for Hermes ``batch_runner.py`` (v0.22.2) used by adapter tests.

It accepts exactly the fire CLI flags of the real ``main()`` and writes
``data/<run_name>/batch_0.jsonl`` plus the merged ``trajectories.jsonl`` with
the real row schema (prompt_index, conversations from/value with
``<tool_call>``/``<tool_response>`` tags, metadata, completed, partial,
api_calls, toolsets_used, tool_stats, tool_error_counts). Instead of calling
a model it reads the evaluated skill from ``$HERMES_HOME/skills`` and writes
the answer into ``$TERMINAL_CWD``, so brain commits change its behaviour.
Behaviour switches come from ``fake_config.json`` next to this file.
"""

import json
import os
import re
import sys
from pathlib import Path

REAL_FLAGS = {
    "dataset_file", "batch_size", "run_name", "distribution", "model", "api_key", "base_url",
    "max_turns", "num_workers", "resume", "verbose", "list_distributions",
    "ephemeral_system_prompt", "log_prefix_chars", "providers_allowed", "providers_ignored",
    "providers_order", "provider_sort", "reasoning_effort", "reasoning_disabled",
    "prefill_messages_file", "max_samples",
}


def call_model(flags, count, stream):
    """POST ``count`` chat completions to ``--base_url`` like Hermes' OpenAI client.

    Streaming requests set ``stream_options.include_usage`` exactly as Hermes
    v0.22.2 does (``agent/chat_completion_helpers.py``); the body is read
    to the end so the server sees a complete exchange.
    """
    import urllib.request

    for index in range(count):
        body = {"model": flags.get("model"), "messages": [{"role": "user", "content": f"turn {index}"}]}
        if stream:
            body.update(stream=True, stream_options={"include_usage": True})
        headers = {"Content-Type": "application/json"}
        if flags.get("api_key"):
            headers["Authorization"] = "Bearer " + flags["api_key"]
        request = urllib.request.Request(
            flags["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), headers=headers, method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()


def main():
    config = json.loads(Path(__file__).with_name("fake_config.json").read_text())
    flags = {}
    for arg in sys.argv[1:]:
        match = re.fullmatch(r"--([a-z_]+)=(.*)", arg, re.DOTALL)
        if not match or match.group(1) not in REAL_FLAGS:
            print(f"unknown flag {arg}")
            return 2
        flags[match.group(1)] = match.group(2)
    for required in ("dataset_file", "batch_size", "run_name"):
        if required not in flags:
            print(f"Error: --{required} is required")
            return 1
    with open(config["log"], "a", encoding="utf-8") as log:
        row = json.loads(Path(flags["dataset_file"]).read_text().splitlines()[0])
        log.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd(), "row": row}) + "\n")
    mode = config.get("mode", "normal")
    if mode == "crash":
        print("Fatal error: provider unreachable")
        return 1
    prompt = json.loads(Path(flags["dataset_file"]).read_text().splitlines()[0])["prompt"]
    output_dir = Path("data") / flags["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "no_reasoning":
        # Real v0.22.2 behaviour: a completed prompt without reasoning is dropped,
        # a tombstone stays in batch_0.jsonl, trajectories.jsonl stays empty.
        (output_dir / "batch_0.jsonl").write_text(json.dumps(
            {"prompt_index": 0, "discarded": "no_reasoning", "prompt": prompt}) + "\n")
        (output_dir / "trajectories.jsonl").write_text("")
        return 0
    if mode == "agent_error":
        print("Error processing prompt 0: model returned 500")
        (output_dir / "trajectories.jsonl").write_text("")
        return 0
    skill = Path(os.environ["HERMES_HOME"]) / "skills" / "answer" / "SKILL.md"
    found = re.search(r"ANSWER=(\d+)", skill.read_text()) if skill.exists() else None
    answer = found.group(1) if found else "unknown"
    workspace = Path(os.environ["TERMINAL_CWD"])
    (workspace / "answer.txt").write_text(answer + "\n")
    call_model(flags, int(config.get("model_calls", 0)), bool(config.get("stream", False)))
    calls = 500 if mode == "tool_storm" else 1
    record = {
        "prompt_index": 0,
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model. <tools>[]</tools>"},
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": "<think>\nNeed to write the file.\n</think>\n<tool_call>\n"
                                     + json.dumps({"name": "write_file", "arguments": {"path": "answer.txt"}})
                                     + "\n</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n" + json.dumps(
                {"tool_call_id": "call_1", "name": "write_file", "content": {"success": True}}) + "\n</tool_response>"},
            {"from": "gpt", "value": f"<think>\n</think>\nThe answer is {answer}."},
        ],
        "metadata": {"batch_num": 0, "timestamp": "2026-11-15T12:00:00", "model": flags.get("model")},
        "completed": True,
        "partial": False,
        "api_calls": 3,
        "toolsets_used": ["file"],
        "tool_stats": {"write_file": {"count": calls, "success": calls, "failure": 0},
                       "terminal": {"count": 0, "success": 0, "failure": 0}},
        "tool_error_counts": {"write_file": 0, "terminal": 0},
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    (output_dir / "batch_0.jsonl").write_text(line)
    (output_dir / "trajectories.jsonl").write_text(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
