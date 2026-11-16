"""Tiny MCP-style JSON-RPC 2.0 stdio server used as a health-check fixture.

Reads newline-delimited requests until EOF and answers ``initialize``,
``tools/list`` and ``ping``. ``--broken`` makes ``tools/list`` return an error.
"""

import json
import sys

TOOLS = [
    {"name": "read_file", "description": "Read a file", "inputSchema": {"type": "object"}},
    {"name": "search", "description": "Search text", "inputSchema": {"type": "object"}},
]


def respond(message_id, result=None, error=None):
    payload = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main():
    broken = "--broken" in sys.argv
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        method = request.get("method")
        if "id" not in request:
            continue
        if method == "initialize":
            respond(request["id"], {
                "protocolVersion": request["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture-mcp", "version": "1.0"},
            })
        elif method == "tools/list" and not broken:
            respond(request["id"], {"tools": TOOLS})
        elif method == "ping":
            respond(request["id"], {})
        else:
            respond(request["id"], error={"code": -32601, "message": "method not found"})


if __name__ == "__main__":
    main()
