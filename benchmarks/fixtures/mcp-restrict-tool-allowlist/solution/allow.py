import json
from pathlib import Path

config = json.loads(Path("mcp.json").read_text())
config["mcpServers"]["files"]["tools"] = {"include": sorted(["read_file", "search_files", "list_directory"])}
Path("mcp.json").write_text(json.dumps(config, indent=2) + "\n")
