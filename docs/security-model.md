# Security Model

OmniFusion is an OpenAI-compatible inference endpoint. Clients own filesystem,
shell, and local tool execution; OmniFusion only owns server-side orchestration,
provider calls, and optional server-side web tools.

## Trust Boundaries

- API clients are untrusted and must authenticate with configured API keys.
- Provider credentials are encrypted at rest with `OMNIFUSION_SECRET_KEY`.
- Provider `base_url` values are SSRF-validated unless the operator explicitly
  opts into private egress.
- `web_fetch` content is untrusted data, never instructions.

## Web Egress

`web_fetch` allows only `http` and `https`, blocks metadata/private/local ranges
by default, revalidates redirects, caps content, strips active HTML markup, and
stores only URL, metadata, content hash, bounded excerpt, cache-hit status, and
truncation status in traces. `OMNIFUSION_ALLOW_PRIVATE_EGRESS=1` permits private
egress for operator-controlled deployments, but cloud metadata endpoints remain
blocked.

Operators are responsible for respecting robots policies, source terms, and
their own network boundaries.

## Benchmark Claims

No benchmark advantage claim may be made from mocked tests, smoke tests, or
undated local runs. Tier C claims require real-provider runs, provenance,
confidence intervals, cost normalization, and ablation artifacts that compare
against both the best single configured model and judge-selected best-of-N.

