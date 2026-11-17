set -eu
git init -q -b main .
mkdir -p cli
printf 'def main(argv):\n    return 0\n' > cli/main.py
printf 'Reference the operations ticket in every commit (Refs trailer).\n' > CONTRIBUTING.md
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf 'def main(argv):\n    if "--version" in argv:\n        print("1.4.0")\n    return 0\n' > cli/main.py
printf 'server started on port 8080\n' > server.log
