# Tool schema rejected by the MCP client

The client refuses `tools/create_ticket.json`. Make `inputSchema` a valid JSON
Schema without changing the tool's name or description:

- every name in `required` must be defined in `properties`; `priority` is a
  string that must be one of `low`, `normal`, `high`;
- `labels` is an array of strings;
- unknown arguments must be rejected (`additionalProperties: false`).
