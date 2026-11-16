import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent
sys.path.insert(0, str(FIXTURE.parent / "harness"))
from checklib import read, run  # noqa: E402

D1 = "sha256:1111111111111111111111111111111111111111111111111111111111111111"
D2 = "sha256:2222222222222222222222222222222222222222222222222222222222222222"
API = f"""FROM python:3.12-slim@{D1}
WORKDIR /app
COPY . .
CMD ["python", "-m", "api"]
"""
WORKER = f"""FROM python:3.12-slim@{D1} AS build
WORKDIR /src
COPY . .
RUN pip install --target /deps .

FROM gcr.io/distroless/python3-debian12@{D2}
COPY --from=build /deps /deps
CMD ["/deps/bin/worker"]
"""


def body(checks):
    checks.check("api_pinned", read("services/api/Dockerfile") == API)
    checks.check("worker_pinned_and_stage_kept", read("services/worker/Dockerfile") == WORKER)
    checks.check("pins_unchanged", read("pins.txt").count("sha256:") == 2)


run(body)
