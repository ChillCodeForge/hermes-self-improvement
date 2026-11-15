#!/usr/bin/env bash
# Run the test suite in a Linux container (project targets POSIX: fcntl, Unix users, symlinks).
# Usage: scripts/docker-test.sh [pytest args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HSI_TEST_IMAGE:-hsi-test:py312}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  (cd "$ROOT/scripts" && docker build -q -t "$IMAGE" -f test.Dockerfile . >/dev/null)
fi
MSYS_NO_PATHCONV=1 docker run --rm -v "$ROOT:/src:ro" "$IMAGE" bash -c '
  set -e; mkdir /work && cd /src && git ls-files -co --exclude-standard | grep -v "^\.venv/" | tar -cf - -T - | tar -xf - -C /work
  cd /work && uv venv -q && uv pip install -q -e ".[test]" && .venv/bin/pytest -q -p no:cacheprovider "$@"' _ "$@"
