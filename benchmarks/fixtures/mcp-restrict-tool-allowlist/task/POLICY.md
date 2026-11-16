# MCP tool policy

The `files` server may expose only its read-only tools: `read_file`,
`search_files` and `list_directory`. Restrict it with a `tools.include`
allowlist (sorted alphabetically). Do not change any other server.
