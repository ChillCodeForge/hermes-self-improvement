set -eu
git init -q -b main .
printf '#!/bin/sh\necho "Hello, $1."\n' > greet.sh
git add greet.sh && git commit -q -m "Add greeting script"
printf 'Goodbye for now.\n' > farewell.txt
git add farewell.txt && git commit -q -m "Add farewell message"
printf '#!/bin/sh\necho "HELLO $1!!!" | tr a-z A-Z\n' > greet.sh
git commit -q -am "Break greeting output"
printf '# Greeter\n\nRun ./greet.sh NAME to greet someone.\n' > README.md
git add README.md && git commit -q -m "Document greeting usage"
git rev-parse HEAD > .git/bench-head-before
