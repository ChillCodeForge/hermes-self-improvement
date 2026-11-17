set -eu
git init -q -b main .
mkdir -p auth
printf 'def expired(token, now):\n    return token["exp"] < now\n' > auth/token.py
printf 'Security fixes must cite their SEC ticket as a Refs trailer.\n' > CONTRIBUTING.md
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf 'def expired(token, now):\n    return token["exp"] <= now\n' > auth/token.py
mkdir -p keys
printf 'signing-key-placeholder\n' > keys/dev.key
printf 'token check\n' > tmp.log
