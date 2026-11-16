---
name: git-commit-hygiene
description: Create small, reviewable Git commits with accurate messages and no stray files or secrets. Use whenever you are about to run git commit.
---

# Git commit hygiene

Use this skill before every `git commit`.

## Steps

1. Inspect state first: `git status --short` and `git diff --stat`. Know
   exactly which files changed before staging anything.
2. Stage deliberately. Prefer `git add <path>` for the files that belong to
   one logical change. Avoid `git add -A` / `git add .` unless you have just
   confirmed every listed file belongs in the commit.
3. Review what is staged: `git diff --cached`. Check for:
   - credentials, tokens, `.env` files, private keys, connection strings;
   - generated output (build directories, `__pycache__`, `node_modules`,
     coverage reports) and large binaries;
   - debug leftovers (print statements, commented-out code, focused tests).
   If anything is wrong, unstage it with `git restore --staged <path>`.
4. One commit = one logical change. If the staged diff mixes a refactor with
   a behavior change, split it into separate commits.
5. Run the project's fast checks (tests, linter) for the touched area before
   committing. Do not commit a known-red state without saying so.
6. Write the message:
   - subject line in the imperative mood, at most ~72 characters, following
     the repository's convention (for example `fix(parser): ...`);
   - a blank line, then a body explaining why the change was made and any
     behavior it affects;
   - reference the issue or ticket if the repository uses them.
7. Never bypass hooks (`--no-verify`) or rewrite published history
   (`commit --amend`, `rebase`, `push --force`) unless the user explicitly
   asked for it.
8. After committing, confirm with `git log --oneline -1` and
   `git status --short` that the tree is in the expected state.

## Failure handling

- A pre-commit hook fails: read the hook output, fix the cause, re-stage, and
  create the commit again. Do not disable the hook.
- You committed a secret: stop, tell the user, and do not push. Rotating the
  credential is required even after removing it from history.
