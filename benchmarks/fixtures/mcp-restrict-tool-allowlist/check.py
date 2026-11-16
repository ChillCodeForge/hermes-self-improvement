import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED = {
    "mcpServers": {
        "files": {
            "command": "files-mcp",
            "args": ["--root", "/srv/shared"],
            "tools": {"include": ["list_directory", "read_file", "search_files"]},
        },
        "tracker": {
            "command": "tracker-mcp",
            "args": ["--project", "OPS"],
            "tools": {"include": ["get_issue", "search_issues"]},
        },
    }
}


def body(checks):
    config = json.loads(read("mcp.json"))
    files = config.get("mcpServers", {}).get("files", {})
    checks.check("files_allowlist", files.get("tools") == {"include": ["list_directory", "read_file", "search_files"]})
    checks.check("tracker_unchanged", config.get("mcpServers", {}).get("tracker") == EXPECTED["mcpServers"]["tracker"])
    checks.check("config_exact", config == EXPECTED)


run(body)
