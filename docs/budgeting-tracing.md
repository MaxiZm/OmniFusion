# Budgeting And Tracing

OmniFusion budgets every model call through `BudgetedExecutor`. Unary and
streaming calls reserve budget before provider execution and reconcile exactly
once after usage is known or the stream closes.

## Budget Controls

- `OMNIFUSION_GLOBAL_DAILY_BUDGET_USD` caps total instance spend.
- `OMNIFUSION_REQUEST_BUDGET_USD` caps a single request by default.
- Preset `cost_ceiling` can lower the request cap for that preset.
- Preset stage budgets bound `max_tokens` and `timeout` for panel, judge, and
  final synthesis stages.

Budget failures return an OpenAI-compatible error envelope with a budget-specific
status instead of silently downgrading to a generic panel failure.

## Trace Contents

Traces include the run id, preset, wall time, aggregate cost, panel results,
judge analysis, final answer, and metadata. Role prompts are redacted by
metadata flag rather than persisted verbatim.

Web-fetch traces store source URL, bounded metadata, content hash, excerpt,
cache-hit status, and truncation state by default. Full fetched pages are not
persisted unless an operator explicitly enables that retention.

### Stage events

Each trace additionally carries an optional `stage_events` list: a bounded,
per-stage timeline of the run. Every event records its `stage` (`web`, `panel`,
`judge`, `synthesis`, or `completion`), `role`, `provider_id`, `model`, `status`,
`tokens`, `cost_usd`, `wall_ms`, an optional `error_code`, and bounded metadata —
never prompt or response bodies. The field is additive: traces stored before it
existed simply load with an empty list. The admin run-history view renders this
timeline with a raw-JSON toggle.

## Admin Visibility

The admin console surfaces two read-only operator views backed by JSON routes:

- **Diagnostics** (`/admin/diagnostics`, `/admin/diagnostics.json`) — startup
  readiness, DB/WAL health, configured-key and provider counts, default-preset
  existence, web-search configuration, and deployment-hardening warnings. No
  secrets are shown — counts, flags, and identifiers only.
- **Budget** (`/admin/budget`, `/admin/budget.json`) — the global daily window,
  recent per-request windows, and recent reservations with their reconcile state.

## Access Pattern

Use the trace id returned in response headers to retrieve the run through the
trace API with the same API key hash. Traces are stored in the configured SQLite
database and can be purged with:

```bash
uv run omnifusion purge
```
