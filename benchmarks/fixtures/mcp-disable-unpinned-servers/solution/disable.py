import json
import re
from pathlib import Path

EXACT = re.compile(r"^(@[^/@]+/)?[^@/]+@\d+\.\d+\.\d+$")
config = json.loads(Path("mcp.json").read_text())
for server in config["mcpServers"].values():
    if server.get("command") != "npx":
        continue
    package = next(arg for arg in server["args"] if not arg.startswith("-"))
    if not EXACT.match(package):
        server["disabled"] = True
Path("mcp.json").write_text(json.dumps(config, indent=2) + "\n")
