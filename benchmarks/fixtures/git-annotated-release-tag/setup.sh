set -eu
git init -q -b main .
printf '1.1.0\n' > VERSION
git add VERSION && git commit -q -m "Initial project"
printf '1.2.0\n' > VERSION
git commit -q -am "Bump version to 1.2.0"
printf '1.3.0-dev\n' > VERSION
git commit -q -am "Start 1.3.0 development"
git rev-parse HEAD > .git/bench-head-before
