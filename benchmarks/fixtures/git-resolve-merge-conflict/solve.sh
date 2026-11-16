set -eu
git merge -q feature/port-change >/dev/null 2>&1 || true
printf '[server]\nhost = localhost\nport = 9090\ntimeout = 30\n\n[logging]\nlevel = debug\n' > settings.ini
git add settings.ini
git commit -q --no-edit
