#!/bin/sh
# Reference replay harness for corpus command benchmarks.
#
# Invoked from a fixture directory (benchmarks/fixtures/<id>). Copies task/ into
# a scratch workspace, runs the optional setup.sh and the reference solve.sh
# there, then runs check.sh. Only check.sh's one-line JSON report is printed.
# Agent runs use the same contract with the agent's work replacing solve.sh.
set -eu
fixture=$(pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/bench-work.XXXXXX")
home=$(mktemp -d "${TMPDIR:-/tmp}/bench-home.XXXXXX")
trap 'rm -rf "$work" "$home"' EXIT
if [ -d task ]; then cp -R task/. "$work/"; fi
export HOME="$home" LC_ALL=C TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
export GIT_AUTHOR_NAME="Bench Author" GIT_AUTHOR_EMAIL="author@example.invalid"
export GIT_COMMITTER_NAME="Bench Author" GIT_COMMITTER_EMAIL="author@example.invalid"
export GIT_AUTHOR_DATE="2024-01-01T00:00:00+0000" GIT_COMMITTER_DATE="2024-01-01T00:00:00+0000"
cd "$work"
if [ -f "$fixture/setup.sh" ]; then
  if ! sh "$fixture/setup.sh" >/dev/null 2>&1; then
    printf '{"checks": 0, "failed": ["setup"], "status": "error"}\n'
    exit 3
  fi
fi
if ! sh "$fixture/solve.sh" >/dev/null 2>&1; then
  printf '{"checks": 0, "failed": ["solve"], "status": "error"}\n'
  exit 4
fi
set +e
sh "$fixture/check.sh"
status=$?
exit "$status"
