"""Single-task Hermes evaluation driver, executed with the Hermes interpreter.

This file is run as a script by ``HermesBatchRunner`` with the Python of the
Hermes installation (``--hermes-python``). It must not import
``hermes_improve``: the two packages live in separate environments.

Why not ``batch_runner.py``: Hermes v0.22.2's batch_runner selects toolsets only
from the fixed ``toolset_distributions.DISTRIBUTIONS``, and none of them
contains the ``skills`` toolset. The evaluated brain's skills therefore never
reached the agent, so skill candidates could not change any outcome (isolated
lab finding F10). batch_runner also drops completed prompts without reasoning
(a training-data filter) and receives the API key on its command line.

The driver uses the unmodified Hermes checkout the way batch_runner does
(``AIAgent`` with ``skip_context_files``/``skip_memory``, one task id, the
row's ``image``/``cwd`` registered as task overrides so the workspace is
bind-mounted into the Docker sandbox, ``_convert_to_trajectory_format``), but:

* ``enabled_toolsets`` is explicit (default ``terminal,file,skills``);
* no reasoning filter;
* the API key is read from ``HERMES_EVAL_API_KEY``, never from argv;
* Hermes' own token counters (``session_input_tokens``/``session_output_tokens``)
  are written next to the trajectory.

Output: ``<out>/trajectories.jsonl`` with one row in batch_runner's schema
(``prompt_index``, ``conversations``, ``metadata``, ``completed``, ``partial``,
``api_calls``, ``toolsets_used``, ``tool_stats``, ``tool_error_counts``) and
``<out>/usage.json``. Exit 0 = conversation finished (row written), 3 = the
agent failed before producing a trajectory (``<out>/error.txt``), 2 = usage
error.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes_driver")
    parser.add_argument("--hermes-agent-dir", required=True, type=Path)
    parser.add_argument("--task-file", required=True, type=Path, help="JSON object: prompt, image, cwd")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--max-turns", required=True, type=int)
    parser.add_argument("--toolsets", default="terminal,file,skills")
    parser.add_argument("--reasoning-effort")
    return parser.parse_args(argv)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    args = _parse(argv)
    hermes = args.hermes_agent_dir.resolve()
    if not (hermes / "run_agent.py").is_file():
        print(f"not a Hermes checkout: {hermes}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(hermes))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    row = json.loads(args.task_file.read_text(encoding="utf-8"))
    toolsets = [name for name in args.toolsets.split(",") if name]

    import batch_runner as hermes_batch  # helpers of the same checkout, imported unmodified
    from run_agent import AIAgent

    task_id = "task_0"
    failure = hermes_batch._prepare_container_image(0, row, 0, task_id, {"verbose": False})
    if failure is not None:
        (out / "error.txt").write_text(str(failure.get("error", "container image unavailable")), encoding="utf-8")
        return 3
    reasoning = {"enabled": True, "effort": args.reasoning_effort} if args.reasoning_effort else None
    try:
        agent = AIAgent(
            model=args.model,
            max_iterations=args.max_turns,
            enabled_toolsets=toolsets,
            save_trajectories=False,
            skip_context_files=True,
            skip_memory=True,
            base_url=args.base_url,
            api_key=os.environ.get("HERMES_EVAL_API_KEY") or None,
            reasoning_config=reasoning,
            platform="batch",
            log_prefix="[eval]",
        )
        try:
            result = agent.run_conversation(row["prompt"], task_id=task_id)
            raw_stats = hermes_batch._extract_tool_stats(result["messages"])
            trajectory = agent._convert_to_trajectory_format(result["messages"], row["prompt"], result["completed"])
            usage = {
                "input_tokens": int(getattr(agent, "session_input_tokens", 0) or 0),
                "output_tokens": int(getattr(agent, "session_output_tokens", 0) or 0),
                "api_calls": int(getattr(agent, "session_api_calls", 0) or 0),
            }
        finally:
            with contextlib.suppress(Exception):
                agent.close()
    except Exception as exc:  # the agent failed before producing a trajectory
        (out / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-4000:]}",
                                       encoding="utf-8")
        print(f"agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    record = {
        "prompt_index": 0,
        "conversations": trajectory,
        "metadata": {"timestamp": datetime.now(timezone.utc).isoformat(), "model": args.model,
                     "driver": "hermes_improve.hermes_driver"},
        "completed": result["completed"],
        "partial": result.get("partial", False),
        "api_calls": result["api_calls"],
        "toolsets_used": toolsets,
        "tool_stats": hermes_batch._normalize_tool_stats(raw_stats),
        "tool_error_counts": {name: stats.get("failure", 0) for name, stats in raw_stats.items()},
    }
    _write_json(out / "trajectories.jsonl", record)
    _write_json(out / "usage.json", usage)
    print(f"conversation finished: completed={result['completed']} api_calls={result['api_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
