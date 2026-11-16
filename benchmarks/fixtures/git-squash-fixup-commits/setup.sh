set -eu
git init -q -b main .
printf 'Parser project\n' > README.md
git add README.md && git commit -q -m "Initial commit"
git checkout -q -b feature/parser
printf 'def parse(text):\n    return text.split(",")\n' > parser.py
git add parser.py && git commit -q -m "Add parser module"
printf 'from parser import parse\n\nassert parse("a,b") == ["a", "b"]\n' > test_parser.py
git add test_parser.py && git commit -q -m "Add parser tests"
printf 'def parse(text):\n    return [part.strip() for part in text.split(",")]\n' > parser.py
git commit -q -am "fixup! Add parser module"
printf 'from parser import parse\n\nassert parse("a, b") == ["a", "b"]\n' > test_parser.py
git commit -q -am "fixup! Add parser tests"
git rev-parse HEAD > .git/bench-tip-before
git rev-parse main > .git/bench-main-before
