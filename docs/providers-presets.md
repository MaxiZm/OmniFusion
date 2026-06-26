# Providers, Presets, And Role Prompts

OmniFusion routes model calls through registered providers and versioned presets.
Provider registration is explicit; request-level plugin overrides cannot create
implicit providers.

## Providers

Use the web UI or CLI import/export flow to configure providers. Each provider
declares:

- provider id
- provider type
- optional base URL
- API key or environment variable reference
- allowed model names

Provider API keys are encrypted with `OMNIFUSION_SECRET_KEY` before storage.
Exports contain decrypted keys and are written with restrictive file permissions;
treat export files as secrets and delete them after import.

## Presets

PresetV2 stores the model pool, role prompts, strategy, and stage budgets. The
classic `B` strategy uses panel, judge, and final roles. The conductor strategy
uses the same budget model with additional explicit stages.

Minimal preset shape:

```json
{
  "name": "general",
  "version": 2,
  "strategy": "B",
  "models": [
    {"provider_id": "default", "role": "panel", "model": "model-a"},
    {"provider_id": "default", "role": "judge", "model": "model-b"},
    {"provider_id": "default", "role": "final", "model": "model-c"}
  ],
  "budgets": {
    "panel": {"max_tokens": 1024, "timeout": 30},
    "judge": {"max_tokens": 1024, "timeout": 30},
    "final": {"max_tokens": 1024, "timeout": 30},
    "min_panel_success": 1
  }
}
```

Legacy fields are upgraded on load so existing presets continue to work.

## Role Prompts

Use global and per-role prompts to steer the panel, judge, and final synthesis
without changing request content. Role prompts are consumed by the runtime and
redacted in trace metadata.

## Search And Web Tools

`web_fetch` is built in and hardened by default. `web_search` is selected through
the search provider adapter settings (`OMNIFUSION_WEB_SEARCH_PROVIDER`) and
supports SearXNG (self-host default), Tavily, Brave, or a custom operator-provided
endpoint.

Web grounding is opt-in. Set `web_enabled: true` on a preset to make every request
to that preset run a server-side search before the panel ("web on"); callers can
override per request with the `plugins.web` flag. When enabled, search results
(and bounded fetched excerpts for the top results) are injected into the panel
context as untrusted, fenced, attributed reference data, and each web call is
budgeted as its own stage. Tune breadth with
`OMNIFUSION_WEB_GROUNDING_MAX_RESULTS` and `OMNIFUSION_WEB_GROUNDING_FETCH_TOP`.
