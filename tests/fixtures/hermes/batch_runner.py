"""Stand-in for the helpers of Hermes ``batch_runner.py`` (v0.22.2) that the driver imports.

``hermes_improve/hermes_driver.py`` uses three helpers of the real module:
``_prepare_container_image`` (registers the row's ``image``/``cwd`` as task
overrides), ``_extract_tool_stats`` and ``_normalize_tool_stats``. This fake
keeps the same signatures and records the registered overrides so the fake
``run_agent.AIAgent`` can see them.
"""

from typing import Any

TASK_OVERRIDES: dict[str, dict[str, Any]] = {}


def _prepare_container_image(prompt_index, prompt_data, batch_num, task_id, config):
    image = prompt_data.get("image") or prompt_data.get("docker_image")
    if not image:
        return None
    overrides = {"docker_image": image}
    if prompt_data.get("cwd"):
        overrides["cwd"] = prompt_data["cwd"]
    TASK_OVERRIDES[task_id] = overrides
    return None


def _extract_tool_stats(messages: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for message in messages:
        for call in message.get("tool_calls", []) or []:
            entry = stats.setdefault(call["name"], {"count": 0, "success": 0, "failure": 0})
            entry["count"] += call.get("repeat", 1)
            entry["success"] += call.get("repeat", 1)
    return stats


def _normalize_tool_stats(tool_stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    normalized = {"terminal": {"count": 0, "success": 0, "failure": 0}}
    normalized.update(tool_stats)
    return normalized
