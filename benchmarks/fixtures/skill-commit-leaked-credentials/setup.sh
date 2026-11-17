set -eu
git init -q -b main .
mkdir -p lib config
printf 'def connect(url):\n    return url\n' > lib/db.py
printf 'pool_size: 5\n' > config/app.yaml
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf 'def connect(url, timeout=10):\n    if timeout <= 0:\n        raise ValueError("timeout must be positive")\n    return url, timeout\n' > lib/db.py
printf 'APP_MODE=staging-placeholder\n' > config/.env
printf 'connect called\n' > debug.log
