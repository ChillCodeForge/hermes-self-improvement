#!/bin/sh
. "$(dirname "$0")/../harness/checklib.sh"
check history_kept [ "$(git rev-parse HEAD~1 2>/dev/null)" = "$(cat .git/bench-head-before)" ]
check five_commits [ "$(git rev-list --count HEAD)" -eq 5 ]
check revert_subject [ "$(git log -1 --format=%s)" = 'Revert "Break greeting output"' ]
check greeting_restored [ "$(cat greet.sh)" = "$(printf '#!/bin/sh\necho "Hello, $1."')" ]
check readme_kept [ -f README.md ]
check farewell_kept [ "$(cat farewell.txt)" = "Goodbye for now." ]
check clean_tree [ -z "$(git status --porcelain)" ]
finish
