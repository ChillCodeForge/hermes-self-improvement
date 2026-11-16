import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from syncer import settings  # noqa: E402

key = "sync.request_timeout"
variable = next(name for name, target in settings.ENVIRONMENT_OVERRIDES.items() if target == key)
answer = {"setting": key, "default_seconds": settings.DEFAULTS[key], "env_var": variable}
Path("answer.json").write_text(json.dumps(answer, indent=2) + "\n")
