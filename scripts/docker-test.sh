#!/usr/bin/env bash
# Run the test suite in a Linux container (project targets POSIX: fcntl, Unix users, symlinks).
# Usage: scripts/docker-test.sh [pytest args...]
#
# The file list is built on the host and streamed into the container as a tar,
# so the script works from the main checkout and from git worktrees (whose .git
# is a pointer file that is meaningless inside the container). The test image is
# tagged by the hash of test.Dockerfile, so editing the Dockerfile rebuilds it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$ROOT/scripts/test.Dockerfile"
if command -v sha256sum >/dev/null 2>&1; then
  DOCKERFILE_HASH="$(sha256sum "$DOCKERFILE" | cut -c1-12)"
else
  DOCKERFILE_HASH="$(shasum -a 256 "$DOCKERFILE" | cut -c1-12)"
fi
IMAGE="${HSI_TEST_IMAGE:-hsi-test:py312-$DOCKERFILE_HASH}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  (cd "$ROOT/scripts" && docker build -q -t "$IMAGE" -f test.Dockerfile . >/dev/null)
fi
git -C "$ROOT" ls-files -z -co --exclude-standard \
  | grep -zv '^\.venv/' \
  | (cd "$ROOT" && tar --null --ignore-failed-read -cf - -T - 2>/dev/null) \
  | MSYS_NO_PATHCONV=1 docker run --rm -i "$IMAGE" bash -c '
      set -e; mkdir /work && tar -xf - -C /work
      cd /work && uv venv -q && uv pip install -q -e ".[test]" && .venv/bin/pytest -q -p no:cacheprovider "$@"' _ "$@"
