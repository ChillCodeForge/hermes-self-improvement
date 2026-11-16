set -eu
git init -q -b main .
printf '[server]\nhost = localhost\nport = 8080\n\n[logging]\nlevel = info\n' > settings.ini
git add settings.ini && git commit -q -m "Add service settings"
git checkout -q -b feature/port-change
printf '[server]\nhost = localhost\nport = 9090\ntimeout = 30\n\n[logging]\nlevel = info\n' > settings.ini
git commit -q -am "Move service to port 9090 with timeout"
git checkout -q main
printf '[server]\nhost = localhost\nport = 8081\n\n[logging]\nlevel = debug\n' > settings.ini
git commit -q -am "Temporary port bump and debug logging"
git rev-parse main > .git/bench-main-before
git rev-parse feature/port-change > .git/bench-feature-tip
