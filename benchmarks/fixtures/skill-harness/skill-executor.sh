#!/bin/sh
# Deterministic skill executor (evaluator-owned yardstick; never candidate-modifiable).
#
# Usage (from the composed sandbox input root): sh harness/skill-executor.sh <skill-slug>
#
# Input layout prepared by the evaluator's runner:
#   brain/    the evaluated brain commit (brain/ subtree only) - the variable under test
#   harness/  this directory, copied from the evaluator's public benchmark root
#   fixture/  the task fixture (setup.sh, check.sh, intent, ...) from the evaluator's root
#
# The skill declares its procedure in exactly one fenced block in SKILL.md:
#
#   ```hermes-procedure
#   <operation> [arguments...]
#   ```
#
# Each line is one operation from the fixed set below. Arguments are data only:
# glob patterns are matched with `case`, message templates are filled from the
# fixture's intent file; nothing from the skill is ever evaluated as shell code.
# The first failing or unknown operation stops the procedure (like an agent that
# gave up); the fixture's check.sh then grades the resulting repository state.
# Only check.sh's one-line JSON verdict is printed; the operation trace is
# written to ./trace.log (the only workspace path a run may change).
set -u
root=$(pwd)
skill=${1:-}
case $skill in
  ''|*[!a-z0-9-]*|-*) printf '{"checks": 0, "failed": ["skill-name"], "results": {}, "status": "error"}\n'; exit 2 ;;
esac
skill_file="$root/brain/skills/$skill/SKILL.md"
fixture="$root/fixture"
trace="$root/trace.log"
: > "$trace"
work=$(mktemp -d "${TMPDIR:-/tmp}/skill-work.XXXXXX")
home=$(mktemp -d "${TMPDIR:-/tmp}/skill-home.XXXXXX")
trap 'rm -rf "$work" "$home"' EXIT
export HOME="$home" LC_ALL=C TZ=UTC SKILL_HARNESS="$root/harness" SKILL_FIXTURE="$fixture"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
export GIT_AUTHOR_NAME="Bench Author" GIT_AUTHOR_EMAIL="author@example.invalid"
export GIT_COMMITTER_NAME="Bench Author" GIT_COMMITTER_EMAIL="author@example.invalid"
export GIT_AUTHOR_DATE="2024-01-01T00:00:00+0000" GIT_COMMITTER_DATE="2024-01-01T00:00:00+0000"

log() { printf '%s\n' "$*" >> "$trace"; }

cd "$work" || exit 3
if [ -d "$fixture/task" ]; then cp -R "$fixture/task/." "$work/"; fi
if ! sh "$fixture/setup.sh" >> "$trace" 2>&1; then
  printf '{"checks": 0, "failed": ["setup"], "results": {}, "status": "error"}\n'
  exit 3
fi

intent_value() {  # intent_value KEY -> value of KEY=... in the fixture intent file
  awk -v key="$1" 'index($0, "=") > 1 && substr($0, 1, index($0, "=") - 1) == key { print substr($0, index($0, "=") + 1); exit }' \
    "$fixture/intent"
}

render() {  # render TEMPLATE -> TEMPLATE with {key} placeholders filled from the intent file
  awk -v f="$fixture/intent" -v tpl="$1" 'BEGIN {
    while ((getline line < f) > 0) { i = index(line, "="); if (i > 1) v[substr(line, 1, i - 1)] = substr(line, i + 1) }
    out = ""; s = tpl
    while ((i = index(s, "{")) > 0) {
      out = out substr(s, 1, i - 1); s = substr(s, i + 1); j = index(s, "}")
      if (j == 0) { bad = 1; break }
      key = substr(s, 1, j - 1)
      if (!(key in v)) { bad = 1; break }
      out = out v[key]; s = substr(s, j + 1)
    }
    if (bad) exit 3
    print out s
  }'
}

procedure="$home/procedure"
if [ ! -f "$skill_file" ]; then
  log "procedure: skill $skill not found in the evaluated brain"
  : > "$procedure"
else
  # Exactly one closed ```hermes-procedure block is accepted.
  awk 'BEGIN { blocks = 0; open = 0 }
       /^```hermes-procedure[ \t]*$/ { if (open) exit 2; blocks++; open = 1; next }
       open && /^```[ \t]*$/ { open = 0; next }
       open { print }
       END { if (open || blocks != 1) exit 2 }' "$skill_file" > "$procedure"
  if [ $? -ne 0 ]; then
    log "procedure: SKILL.md must contain exactly one closed hermes-procedure block"
    : > "$procedure"
  fi
fi

subject="" body="" trailers="" steps=0 failed=""

do_commit() {  # do_commit [--no-verify]
  [ -n "$subject" ] || { log "commit: no subject set"; return 1; }
  message="$home/message"
  printf '%s\n' "$subject" > "$message"
  if [ -n "$body" ]; then printf '\n%s\n' "$body" >> "$message"; fi
  if [ -n "$trailers" ]; then printf '\n%s\n' "$trailers" >> "$message"; fi
  git commit -q "$@" -F "$message" >> "$trace" 2>&1
}

run_step() {  # run_step OPERATION [ARGS...]
  op=$1
  shift
  case $op in
    inspect) git status --short >> "$trace" 2>&1 && git diff --stat >> "$trace" 2>&1 ;;
    stage-all) [ $# -eq 0 ] && git add -A >> "$trace" 2>&1 ;;
    stage-tracked) [ $# -eq 0 ] && git add -u >> "$trace" 2>&1 ;;
    stage-untracked)
      [ $# -ge 1 ] || return 1
      git ls-files --others --exclude-standard > "$home/paths" || return 1
      while IFS= read -r path; do
        for pattern in "$@"; do
          case $path in $pattern) git add -- "$path" >> "$trace" 2>&1 || return 1; break ;; esac
        done
      done < "$home/paths" ;;
    unstage)
      [ $# -ge 1 ] || return 1
      git diff --cached --name-only > "$home/paths" || return 1
      while IFS= read -r path; do
        for pattern in "$@"; do
          case $path in $pattern) git reset -q -- "$path" >> "$trace" 2>&1 || return 1; break ;; esac
        done
      done < "$home/paths" ;;
    review-staged) [ $# -eq 0 ] && git diff --cached --stat >> "$trace" 2>&1 ;;
    subject) [ $# -ge 1 ] && subject=$(render "$*") ;;
    body)
      [ $# -ge 1 ] || return 1
      paragraph=$(render "$*") || return 1
      if [ -n "$body" ]; then body="$body

$paragraph"; else body=$paragraph; fi ;;
    trailer)  # trailer KEY TEMPLATE: added only when the intent defines KEY
      [ $# -ge 2 ] || return 1
      key=$1
      shift
      [ -n "$(intent_value "$key")" ] || return 0
      line=$(render "$*") || return 1
      if [ -n "$trailers" ]; then trailers="$trailers
$line"; else trailers=$line; fi ;;
    commit) [ $# -eq 0 ] && do_commit ;;
    commit-no-verify) [ $# -eq 0 ] && do_commit --no-verify ;;
    verify) [ $# -eq 0 ] && git log --oneline -1 >> "$trace" 2>&1 && git status --short >> "$trace" 2>&1 ;;
    *) log "unknown operation"; return 1 ;;
  esac
}

set -f  # arguments are patterns and templates, never expanded by the shell
while IFS= read -r line || [ -n "$line" ]; do
  set -- $line
  [ $# -eq 0 ] && continue
  case $1 in \#*) continue ;; esac
  steps=$((steps + 1))
  if [ "$steps" -gt 64 ] || [ ${#line} -gt 400 ]; then
    log "procedure: step or line limit exceeded"
    failed=limit
    break
  fi
  log "step $steps: $1"
  if ! run_step "$@"; then
    log "step $steps failed: $1"
    failed=$1
    break
  fi
done < "$procedure"
set +f
log "procedure finished: steps=$steps failed=${failed:-none}"

sh "$fixture/check.sh"
exit $?
