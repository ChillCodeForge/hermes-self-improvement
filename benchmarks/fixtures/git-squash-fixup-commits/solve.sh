set -eu
GIT_SEQUENCE_EDITOR=: git rebase -q -i --autosquash main
