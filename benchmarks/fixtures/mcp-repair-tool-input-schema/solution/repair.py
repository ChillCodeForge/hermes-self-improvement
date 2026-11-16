import json
from pathlib import Path

path = Path("tools/create_ticket.json")
tool = json.loads(path.read_text())
schema = tool["inputSchema"]
schema["properties"]["labels"] = {"type": "array", "items": {"type": "string"}}
schema["properties"]["priority"] = {"type": "string", "enum": ["low", "normal", "high"]}
schema["additionalProperties"] = False
path.write_text(json.dumps(tool, indent=2) + "\n")
