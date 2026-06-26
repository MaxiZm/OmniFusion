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

## Access Pattern

Use the trace id returned in response headers to retrieve the run through the
trace API with the same API key hash. Traces are stored in the configured SQLite
database and can be purged with:

```bash
uv run omnifusion purge
```
