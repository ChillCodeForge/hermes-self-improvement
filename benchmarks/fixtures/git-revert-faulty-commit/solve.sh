set -eu
git revert --no-edit "$(git log --format=%H --grep='^Break greeting output$')" >/dev/null
