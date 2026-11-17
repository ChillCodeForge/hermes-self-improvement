set -eu
git init -q -b main .
mkdir -p src
printf 'import os\n\n\ndef load(path):\n    return dict(line.split("=", 1) for line in open(path))\n' > src/app.py
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf 'import os\n\nfrom .config import parse\n\n\ndef load(path):\n    return parse(open(path))\n' > src/app.py
printf 'def parse(lines):\n    pairs = (line.split("=", 1) for line in lines if line.strip())\n    return {key: value.strip() for key, value in pairs if value.strip()}\n' > src/config.py
mkdir -p src/__pycache__
printf 'compiled-bytecode-placeholder\n' > src/__pycache__/app.cpython-312.pyc
printf 'APP_MODE=local-development\n' > .env
