import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

import json

EXPECTED = {
    "calendar": {"command": "npx", "args": ["-y", "@acme/calendar-mcp@2.4.1"]},
    "browser": {"command": "npx", "args": ["-y", "@acme/browser-mcp"], "disabled": True},
    "notes": {"command": "npx", "args": ["-y", "notes-mcp@latest", "--readonly"], "disabled": True},
    "search": {"command": "npx", "args": ["-y", "search-mcp@1.0.0"]},
    "metrics": {"command": "npx", "args": ["-y", "metrics-mcp@^3.1.0"], "disabled": True},
    "local": {"command": "/opt/mcp/local-server", "args": []},
}


def body(checks):
    servers = json.loads(read("mcp.json")).get("mcpServers", {})
    for name, expected in EXPECTED.items():
        checks.check(f"server:{name}", servers.get(name) == expected)
    checks.check("no_servers_removed_or_added", sorted(servers) == sorted(EXPECTED))


run(body)
