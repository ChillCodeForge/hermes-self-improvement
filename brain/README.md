# Production brain

Git is the only source of truth for the production brain. `main` is the
current production brain; releases are tagged `brain-v1`, `brain-v2`, ...
Hermes reads a read-only deployment checkout selected by an atomic symlink
switch (see `docs/git-workflow.md` and `docs/promotion.md`).

Nothing in this tree is edited in place. Every change is a candidate commit on
`candidate/<experiment-id>`, evaluated, and promoted by the promoter.

| Area | Candidate types | Allowed paths |
|---|---|---|
| `skills/` | `SKILL_CREATE`, `SKILL_PATCH`, `SKILL_MERGE`, `SKILL_SPLIT` | `brain/skills/<slug>/SKILL.md` plus `.md/.txt/.yaml/.yml/.json` files up to 3 levels below the slug |
| `tool-descriptions/` | `TOOL_DESCRIPTION` | `brain/tool-descriptions/<tool>.md` |
| `workflows/` | `WORKFLOW` | `brain/workflows/<name>.yaml` |
| `routing/` | `ROUTING` | `brain/routing/<name>.yaml` |
| `mcp/` | `MCP_TOOL_FILTER`, `MCP_CONFIG` | `brain/mcp/tool-filters/<name>.yaml`, `brain/mcp/servers/<name>.yaml` |
| `prompts/` | `SYSTEM_PROMPT` | `brain/prompts/<name>.md` |
| `tools/` | `TOOL_CREATE`, `TOOL_PATCH` | `brain/tools/<name>/...` (`.py` and text files) |
| `metadata/` | none | operator-maintained only; no candidate type may write here |

The authoritative rules are in `src/hermes_improve/target_paths.py`.
