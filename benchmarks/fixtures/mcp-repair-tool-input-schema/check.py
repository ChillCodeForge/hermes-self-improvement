import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

VALID_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def all_types_valid(node):
    if isinstance(node, dict):
        kind = node.get("type")
        if kind is not None and kind not in VALID_TYPES:
            return False
        return all(all_types_valid(value) for value in node.values())
    if isinstance(node, list):
        return all(all_types_valid(value) for value in node)
    return True


def body(checks):
    tool = json.loads(read("tools/create_ticket.json"))
    schema = tool.get("inputSchema", {})
    properties = schema.get("properties", {})
    checks.check("identity_unchanged", tool.get("name") == "create_ticket"
                 and tool.get("description") == "Create a support ticket in the helpdesk.")
    checks.check("required_defined", set(schema.get("required", [])) <= set(properties))
    checks.check("required_kept", schema.get("required") == ["title", "priority"])
    checks.check("types_valid", all_types_valid(schema))
    checks.check("priority_enum", properties.get("priority") == {"type": "string", "enum": ["low", "normal", "high"]})
    checks.check("labels_array_of_strings", properties.get("labels") == {"type": "array", "items": {"type": "string"}})
    checks.check("rejects_unknown_arguments", schema.get("additionalProperties") is False)
    checks.check("title_unchanged", properties.get("title") == {"type": "string", "minLength": 1})


run(body)
