# Supply-chain policy for MCP servers

Servers started through `npx` must reference their package with an exact
version (`name@1.2.3` or `@scope/name@1.2.3`). Set `"disabled": true` on every
`npx` server that does not, and leave every other server exactly as it is.
