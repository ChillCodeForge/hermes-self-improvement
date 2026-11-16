# Git workflow

This document describes the Git behavior that is implemented in
`candidate_builder.py`, `target_paths.py` and `promotion.py`.

## Production brain

Git is the only source of truth for the production brain. `main` is the
current production brain. Brain artifacts live under `brain/`:

| Area | Candidate types | Allowed paths |
|---|---|---|
| `brain/skills/` | `SKILL_CREATE`, `SKILL_PATCH`, `SKILL_MERGE`, `SKILL_SPLIT` | `brain/skills/<slug>/SKILL.md` plus `.md/.txt/.yaml/.yml/.json` files up to 3 levels below the slug |
| `brain/tool-descriptions/` | `TOOL_DESCRIPTION` | `brain/tool-descriptions/<tool>.md` (exactly one) |
| `brain/workflows/` | `WORKFLOW` | `brain/workflows/<name>.yaml` (exactly one) |
| `brain/routing/` | `ROUTING` | `brain/routing/<name>.yaml` (one or more) |
| `brain/mcp/tool-filters/` | `MCP_TOOL_FILTER` | `brain/mcp/tool-filters/<name>.yaml` |
| `brain/mcp/servers/` | `MCP_CONFIG` | `brain/mcp/servers/<name>.yaml` |
| `brain/prompts/` | `SYSTEM_PROMPT` | `brain/prompts/<name>.md` (exactly one) |
| `brain/tools/` | `TOOL_CREATE`, `TOOL_PATCH` | `brain/tools/<name>/...` (`.py` and text files) |
| `brain/metadata/` | none | operator-maintained only |

Area `README.md` files are operator-maintained; no candidate may change them.
The repository ships one baseline skill, `brain/skills/git-commit-hygiene`.

The skill layout is flat (`brain/skills/<slug>`). The earlier skills-only
promoter already used `brain/skills/`, so no `skills/` migration is needed;
nested skill targets such as `brain/skills/a/b` are no longer accepted.

## Path safety

Every changed path is checked against its type's rule before a candidate is
committed and again at promotion:

- Paths must be normalized POSIX paths under `brain/`. Absolute paths,
  backslashes, NUL, `..`, hidden components and anything outside the
  `[A-Za-z0-9][A-Za-z0-9._-]` alphabet are rejected.
- Repository paths no candidate may touch are rejected explicitly:
  `config/`, `benchmarks/`, `src/`, `tests/`, `scripts/`, `.github/`,
  `prompts/judge` (case-insensitive), plus the protected components
  `evaluator`, `verifiers`, `holdout`, `security`, `promotion`, `ci`, `judge`, ...
- Case tricks: the skill entry file must be exactly `SKILL.md`; two paths of a
  candidate may not collide case-insensitively; a new path may not collide
  case-insensitively with a path tracked on `main`; the promoter rejects
  candidate trees with case-colliding paths.
- Symlinks: the builder refuses to write through a symlink tracked in the
  worktree; the promoter refuses release archives containing symlinks.
- Content: UTF-8 text, at most 256 KiB per file, 1 MiB and 32 files per
  candidate, no credential-like material. YAML must parse with the safe loader
  (duplicate keys rejected) to a mapping or list, JSON must parse, Python must
  compile. Candidate code is never executed.

## Candidates

```text
hermes-improve candidate create --repository REPO \
  --experiment-id exp-0001 --candidate-id cand-0001 \
  --type WORKFLOW --hypothesis "..." \
  --file brain/workflows/release.yaml=./release.yaml \
  [--delete brain/...] [--worktree-root worktrees] [--parent-experiment exp-0000]
```

The original skill form (`--type SKILL_CREATE|SKILL_PATCH --target
brain/skills/<slug> --content SKILL.md`, or `--proposal-file/--proposal-index`)
still works. Python API: `create_candidate(...)`, `create_skill_candidate(...)`.

For every candidate the builder:

1. resolves `main` to the parent baseline commit;
2. checks type semantics against the baseline tree: `*_CREATE` units must not
   exist, `*_PATCH` units must exist, `SKILL_MERGE` must remove at least one
   existing skill and keep or create one, `SKILL_SPLIT` must touch an existing
   skill and create a new one, every remaining skill keeps a `SKILL.md`;
3. creates branch `candidate/<experiment-id>` in a worktree (kept at
   `<worktree-root>/<experiment-id>` when `--worktree-root` is given, which must
   be outside the repository; otherwise a temporary worktree that is removed);
4. commits the files and deletions, computes the binary diff and confirms the
   changed paths equal the declared target units;
5. stores the experiment and candidate in one SQLite transaction: experiment
   ID, candidate type, target (sorted, comma-joined units), risk (from
   `config/taxonomy.yaml`), parent baseline, commit SHA, SHA-256 diff hash,
   content hash, policy version, and lineage (`parent_id`/`lineage_id`).

Any failure removes the branch and worktree; nothing is recorded.

## Immutability

An existing `candidate/<experiment-id>` branch is never reused, so changing an
evaluated candidate requires a new experiment ID, which yields a new commit,
diff hash and content hash and needs a new evaluation. `--parent-experiment`
records the revision lineage. The promoter refuses a candidate whose branch
no longer points to the evaluated commit, whose diff hash differs, or whose
stored metadata no longer matches its content hash.

## Release flow

`promote` fast-forwards `main` to the candidate commit, creates tag `brain-vN`,
extracts a read-only release checkout and switches the deployment symlink
atomically; see `docs/promotion.md`. `rollback` moves `main` back to an earlier
release commit and switches the symlink; release tags and manifests are not
rewritten.
