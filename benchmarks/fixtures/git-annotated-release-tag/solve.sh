set -eu
git tag -a v1.2.0 -m "Release 1.2.0" "$(git log --format=%H --grep='^Bump version to 1.2.0$')"
