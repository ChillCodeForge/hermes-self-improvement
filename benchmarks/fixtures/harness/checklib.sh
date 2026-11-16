# Shared helpers for corpus check.sh scripts (POSIX sh).
# Usage: . checklib.sh; check NAME COMMAND...; finish
_checks=0
_failed=""
check() {
  _name=$1
  shift
  _checks=$((_checks + 1))
  if ! "$@" >/dev/null 2>&1; then
    if [ -z "$_failed" ]; then _failed="\"$_name\""; else _failed="$_failed, \"$_name\""; fi
  fi
}
finish() {
  if [ -z "$_failed" ]; then
    printf '{"checks": %d, "failed": [], "status": "pass"}\n' "$_checks"
    exit 0
  fi
  printf '{"checks": %d, "failed": [%s], "status": "fail"}\n' "$_checks" "$_failed"
  exit 1
}
