from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import canonical_json


_MAX_BATCH_FILE_BYTES = 10 * 1024 * 1024
_MAX_BATCH_RECORDS = 10_000
_MAX_TOOL_STATS = 512
_MAX_COUNTER = (1 << 63) - 1
_TURN_ROLES = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}


def load_batch_runner_export(path: Path) -> list[dict[str, Any]]:
    """Convert Hermes BatchRunner trajectories into untrusted session exports."""
    with Path(path).open("rb") as source:
        raw = source.read(_MAX_BATCH_FILE_BYTES + 1)
    if len(raw) > _MAX_BATCH_FILE_BYTES:
        raise ValueError("batch trajectory file exceeds the 10 MiB limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("batch trajectory file must be UTF-8") from exc

    exports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    record_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        record_count += 1
        if record_count > _MAX_BATCH_RECORDS:
            raise ValueError("batch trajectory file exceeds the 10,000-record limit")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"batch trajectory line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"batch trajectory line {line_number} must be a JSON object")
        if record.get("discarded"):
            continue

        prompt_index = record.get("prompt_index")
        if isinstance(prompt_index, bool) or not isinstance(prompt_index, int) or prompt_index < 0:
            raise ValueError(f"batch trajectory line {line_number} has an invalid prompt_index")
        completed = record.get("completed")
        partial = record.get("partial", False)
        if not isinstance(completed, bool) or not isinstance(partial, bool):
            raise ValueError(f"batch trajectory line {line_number} has invalid completion status")
        trajectory = record.get("conversations")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"batch trajectory line {line_number} has no conversations")
        messages: list[dict[str, str]] = []
        prompt: str | None = None
        for turn in trajectory:
            if not isinstance(turn, dict):
                raise ValueError(f"batch trajectory line {line_number} contains an invalid turn")
            source_role = turn.get("from")
            if not isinstance(source_role, str) or source_role not in _TURN_ROLES:
                raise ValueError(f"batch trajectory line {line_number} contains an invalid turn")
            value = turn.get("value")
            if not isinstance(value, str):
                raise ValueError(f"batch trajectory line {line_number} contains a non-text turn")
            role = _TURN_ROLES[source_role]
            if role == "user" and prompt is None:
                prompt = value
            messages.append({"role": role, "content": value})
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"batch trajectory line {line_number} has no non-empty user prompt")

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"batch trajectory line {line_number} metadata must be an object")
        batch_num = metadata.get("batch_num")
        if batch_num is not None and (
            isinstance(batch_num, bool) or not isinstance(batch_num, int) or batch_num < 0
        ):
            raise ValueError(f"batch trajectory line {line_number} has an invalid batch number")
        timestamp, model = metadata.get("timestamp"), metadata.get("model")
        if timestamp is not None and not isinstance(timestamp, str):
            raise ValueError(f"batch trajectory line {line_number} has an invalid timestamp")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"batch trajectory line {line_number} has an invalid model")
        raw_tool_stats = record.get("tool_stats", {})
        if not isinstance(raw_tool_stats, dict) or len(raw_tool_stats) > _MAX_TOOL_STATS:
            raise ValueError(f"batch trajectory line {line_number} has invalid tool statistics")
        tool_stats = {}
        for name, counts in raw_tool_stats.items():
            if not isinstance(name, str) or not name.strip() or len(name) > 200 or not isinstance(counts, dict):
                raise ValueError(f"batch trajectory line {line_number} has invalid tool statistics")
            normalized = {}
            for key in ("count", "success", "failure"):
                value = counts.get(key)
                if (
                    isinstance(value, bool) or not isinstance(value, int)
                    or value < 0 or value > _MAX_COUNTER
                ):
                    raise ValueError(f"batch trajectory line {line_number} has invalid tool counters")
                normalized[key] = value
            if normalized["success"] + normalized["failure"] > normalized["count"]:
                raise ValueError(f"batch trajectory line {line_number} has inconsistent tool counters")
            tool_stats[name] = normalized
        identity = hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
        session_id = f"hermes-batch-{identity}"
        if session_id in seen_ids:
            continue
        seen_ids.add(session_id)
        exports.append({
            "session_id": session_id,
            "source": {
                "type": "hermes_batch_runner",
                "prompt_index": prompt_index,
                "batch_num": batch_num,
                "timestamp": timestamp,
                "model": model,
            },
            "task": {"prompt": prompt, "input_available": True},
            "messages": messages,
            "tool_stats": tool_stats,
            "final_status": "partial" if partial else "completed" if completed else "incomplete",
            "security": {"sanitized": True, "contains_secrets": False},
        })
    if not exports:
        raise ValueError("batch trajectory file contains no usable conversation records")
    return exports
