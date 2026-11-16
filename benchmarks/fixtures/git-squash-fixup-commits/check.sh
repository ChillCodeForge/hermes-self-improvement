#!/bin/sh
. "$(dirname "$0")/../harness/checklib.sh"
check on_feature_branch [ "$(git symbolic-ref --short HEAD 2>/dev/null)" = feature/parser ]
check main_unchanged [ "$(git rev-parse main)" = "$(cat .git/bench-main-before)" ]
check two_commits_on_main [ "$(git rev-list --count main..HEAD)" -eq 2 ]
check subjects_in_order [ "$(git log --reverse --format=%s main..HEAD | tr '\n' '|')" = "Add parser module|Add parser tests|" ]
check final_tree_unchanged git diff --quiet "$(cat .git/bench-tip-before)" HEAD
check first_commit_has_fix [ "$(git show HEAD~1:parser.py | tail -n 1)" = "    return [part.strip() for part in text.split(\",\")]" ]
check no_fixup_left sh -c '! git log --format=%s main..HEAD | grep -q "^fixup!"'
check clean_tree [ -z "$(git status --porcelain)" ]
finish
