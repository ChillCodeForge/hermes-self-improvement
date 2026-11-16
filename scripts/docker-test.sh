#!/usr/bin/env bash
# Run the test suite in a Linux container (project targets POSIX: fcntl, Unix users, symlinks).
# The container runs as root, so OS-isolation tests marked `root_required` execute and must
# not be skipped (HSI_REQUIRE_ROOT_TESTS=1 turns a skip into a failure).
# The file list is built on the host and streamed in as a tar, so this also works from a
# git worktree (whose .git file points at a host path the container cannot resolve).
# Usage: scripts/docker-test.sh [pytest args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Tag the image by Dockerfile content so a changed Dockerfile is never served from a stale cache.
DOCKERFILE_DIGEST="$(sha256sum "$ROOT/scripts/test.Dockerfile" | cut -c1-12)"
IMAGE="${HSI_TEST_IMAGE:-hsi-test:py312-$DOCKERFILE_DIGEST}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  (cd "$ROOT/scripts" && docker build -q -t "$IMAGE" -f test.Dockerfile . >/dev/null)
fi
(cd "$ROOT" && git ls-files -co --exclude-standard -z | grep -zv '^\.venv/' | tar --null -cf - -T -) |
MSYS_NO_PATHCONV=1 docker run --rm -i -e HSI_REQUIRE_ROOT_TESTS=1 "$IMAGE" bash -c '
  set -e; mkdir /work && tar -xf - -C /work --no-same-owner && chmod -R go-w /work
  cd /work && uv venv -q && uv pip install -q -e ".[test]" && .venv/bin/pytest -q -p no:cacheprovider "$@"' _ "$@"
