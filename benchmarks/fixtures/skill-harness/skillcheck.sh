# Grading helpers for skill_procedure fixtures (POSIX sh, sourced by check.sh).
#
# check.sh runs inside the task repository after the procedure. It prints one
# JSON line {"checks": N, "failed": [...], "results": {name: bool}, "status": ...}
# and exits 0 only when every check passed. Fixture files read here:
#   intent          key=value lines (type, scope, summary, why, optional ticket)
#   expected-paths  paths the single new commit must change, one per line
#   strays          untracked files that must stay uncommitted and on disk
_checks=0
_failed=""
_results=""

check() {  # check NAME COMMAND...
  _name=$1
  shift
  _checks=$((_checks + 1))
  if "$@" >/dev/null 2>&1; then
    _value=true
  else
    _value=false
    if [ -z "$_failed" ]; then _failed="\"$_name\""; else _failed="$_failed, \"$_name\""; fi
  fi
  if [ -z "$_results" ]; then _results="\"$_name\": $_value"; else _results="$_results, \"$_name\": $_value"; fi
}

finish() {
  if [ -z "$_failed" ]; then _status=pass; else _status=fail; fi
  printf '{"checks": %d, "failed": [%s], "results": {%s}, "status": "%s"}\n' \
    "$_checks" "$_failed" "$_results" "$_status"
  [ "$_status" = pass ]
  exit $?
}

_intent() {
  awk -v key="$1" 'index($0, "=") > 1 && substr($0, 1, index($0, "=") - 1) == key { print substr($0, index($0, "=") + 1); exit }' \
    "$SKILL_FIXTURE/intent"
}

_base() { cat .git/bench-base; }
_new_commit() { [ "$(git rev-parse HEAD)" != "$(_base)" ]; }

one_new_commit() { [ "$(git rev-list --count "$(_base)..HEAD")" = 1 ]; }

intended_paths_committed() {
  _new_commit || return 1
  [ "$(git diff --name-only "$(_base)" HEAD | sort)" = "$(sort "$SKILL_FIXTURE/expected-paths")" ]
}

no_secret_committed() {
  ! git ls-tree -r --name-only HEAD | grep -Eq '(^|/)\.env($|\.)|\.pem$|\.key$'
}

no_generated_committed() {
  ! git ls-tree -r --name-only HEAD | grep -Eq '(^|/)__pycache__/|\.pyc$|\.log$|(^|/)node_modules/'
}

subject_follows_convention() {
  _new_commit || return 1
  _expected="$(_intent type)($(_intent scope)): $(_intent summary)"
  [ "$(git log -1 --format=%s)" = "$_expected" ] && [ ${#_expected} -le 72 ]
}

body_explains_why() {
  _new_commit || return 1
  git log -1 --format=%b | grep -qxF "$(_intent why)"
}

strays_preserved() {
  while IFS= read -r _path; do
    [ -z "$_path" ] && continue
    [ -f "$_path" ] || return 1
  done < "$SKILL_FIXTURE/strays"
}

ticket_reference() {
  _ticket=$(_intent ticket)
  if [ -n "$_ticket" ]; then
    _new_commit && git log -1 --format=%B | grep -qxF "Refs: $_ticket"
  else
    ! git log -1 --format=%B | grep -q '^Refs:'
  fi
}

standard_checks() {
  check one_new_commit one_new_commit
  check intended_paths_committed intended_paths_committed
  check no_secret_committed no_secret_committed
  check no_generated_committed no_generated_committed
  check subject_follows_convention subject_follows_convention
  check body_explains_why body_explains_why
  check strays_preserved strays_preserved
  check ticket_reference ticket_reference
}
