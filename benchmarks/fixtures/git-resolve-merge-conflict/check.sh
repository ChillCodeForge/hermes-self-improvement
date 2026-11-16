#!/bin/sh
. "$(dirname "$0")/../harness/checklib.sh"
expected='[server]
host = localhost
port = 9090
timeout = 30

[logging]
level = debug'
check on_main [ "$(git symbolic-ref --short HEAD 2>/dev/null)" = main ]
check merge_commit [ "$(git rev-list --parents -n 1 HEAD | wc -w)" -eq 3 ]
check first_parent_is_old_main [ "$(git rev-parse HEAD^1 2>/dev/null)" = "$(cat .git/bench-main-before)" ]
check second_parent_is_feature [ "$(git rev-parse HEAD^2 2>/dev/null)" = "$(cat .git/bench-feature-tip)" ]
check resolved_content [ "$(git show HEAD:settings.ini 2>/dev/null)" = "$expected" ]
check no_conflict_markers sh -c '! git grep -q -E "^(<<<<<<<|=======|>>>>>>>)" HEAD'
check clean_tree [ -z "$(git status --porcelain)" ]
finish
