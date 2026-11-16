import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

EXPECTED = """# Production deployment for the shop backend.
# Image tags are bumped by the release manager only.
services:
  api:
    image: registry.example.com/shop/api:1.5.0  # current API release
    replicas: 3
    env:
      LOG_LEVEL: info
  worker:
    image: registry.example.com/shop/worker:1.4.2
    replicas: 2
"""


def body(checks):
    text = read("deploy.yaml")
    checks.check("api_bumped", "registry.example.com/shop/api:1.5.0" in text)
    checks.check("worker_unchanged", "registry.example.com/shop/worker:1.4.2" in text)
    checks.check("comments_and_layout_kept", text == EXPECTED)


run(body)
