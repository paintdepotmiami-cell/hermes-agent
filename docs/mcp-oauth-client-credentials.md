# MCP OAuth client credentials

Autonomous MCP agents can authenticate with their own OAuth identity without
borrowing a human user's authorization code or refresh token.

```yaml
mcp_servers:
  dps-work:
    url: https://dps-work-mcp.onrender.com/mcp
    auth: oauth
    oauth:
      grant_type: client_credentials
      token_endpoint: https://dev-g2ivzaysaodctyda.us.auth0.com/oauth/token
      client_id: ${env:DPS_WORK_HERMES_CLIENT_ID}
      client_secret: ${env:DPS_WORK_HERMES_CLIENT_SECRET}
      audience: https://dps-work-mcp.onrender.com/mcp
      scope: >-
        dps.read dps.action.internal dps.action.communications dps.action.calls
```

`token_endpoint`, `client_id`, `client_secret`, `audience`, and `scope` are
required. The token endpoint must use HTTPS and cannot contain URL credentials
or a fragment. Configuration errors and token failures are fatal for that MCP
connection; Hermes never falls back to browser OAuth.

Access tokens are cached only in memory, refreshed before expiry, and renewed
once after an HTTP 401. Hermes does not write client-credentials access tokens
or secrets under `mcp-tokens`. Keep the client secret in the Hermes `.env` file
and reference it from `config.yaml` as shown above.

For auditable workloads, map the OAuth machine subject to an agent-specific
operator. Authorization role and audit actor are separate concerns. For
example, a machine may receive Marcial's allowed business role while retaining
`operator_id=hermes`, so actions are recorded as `work:hermes` rather than
`work:marcial`.
