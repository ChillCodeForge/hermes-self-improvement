set -eu
git init -q -b main .
mkdir -p docs
printf '# Guide\n\nInstall the tool.\n' > docs/guide.md
git add -A
git commit -q -m "Initial commit"
git rev-parse HEAD > .git/bench-base
printf '# Guide\n\nInstall the tool with pipx.\n\nSee the FAQ for upgrade notes.\n' > docs/guide.md
printf '# FAQ\n\n## How do I upgrade?\n\nRun pipx upgrade.\n' > docs/faq.md
printf 'render started\nrender finished\n' > build.log
mkdir -p node_modules/left-pad certs
printf 'module.exports = function pad() {};\n' > node_modules/left-pad/index.js
printf 'certificate-placeholder\n' > certs/dev.pem
