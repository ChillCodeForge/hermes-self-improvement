set -eu
git init -q -b main .
mkdir -p api
printf 'ROUTES = ["/health"]\n' > api/routes.py
printf 'Commit messages reference the tracker issue in a Refs trailer.\n' > CONTRIBUTING.md
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf 'from .schemas import Order\n\nROUTES = ["/health", "/orders"]\n' > api/routes.py
printf 'class Order:\n    fields = ("id", "quantity")\n' > api/schemas.py
mkdir -p api/__pycache__
printf 'compiled-bytecode-placeholder\n' > api/__pycache__/routes.cpython-312.pyc
printf 'APP_MODE=local-development\n' > .env
