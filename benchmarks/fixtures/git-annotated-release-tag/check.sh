#!/bin/sh
. "$(dirname "$0")/../harness/checklib.sh"
check tag_exists git rev-parse -q --verify refs/tags/v1.2.0
check tag_is_annotated [ "$(git cat-file -t v1.2.0 2>/dev/null)" = tag ]
check tag_points_to_release [ "$(git show v1.2.0:VERSION 2>/dev/null)" = 1.2.0 ]
check tag_on_bump_commit [ "$(git log -1 --format=%s 'v1.2.0^{commit}' 2>/dev/null)" = "Bump version to 1.2.0" ]
check tag_message [ "$(git tag -l --format='%(contents:subject)' v1.2.0)" = "Release 1.2.0" ]
check head_unchanged [ "$(git rev-parse HEAD)" = "$(cat .git/bench-head-before)" ]
check single_tag [ "$(git tag | wc -l)" -eq 1 ]
finish
