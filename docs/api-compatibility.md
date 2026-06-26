# API Compatibility

OmniFusion exposes OpenAI-compatible routes under both `/v1` and `/api/v1`.

## Chat

- `/v1/chat/completions` and `/api/v1/chat/completions` accept chat requests.
- `developer` messages normalize to `system`.
- `max_completion_tokens` normalizes to `max_tokens`.
- Legacy `functions` and `function_call` normalize to `tools` and `tool_choice`.
- Text content-part arrays are accepted; non-text parts are rejected.

### Judge Determinism

For OpenRouter parity the internal judge call is deterministic: it always runs at
temperature 0 regardless of the caller's `temperature` (which still flows to the
panel and synthesis). The documented, off-by-default
`OMNIFUSION_EXPERIMENTAL_JUDGE_TEMPERATURE` knob overrides this; it is flagged
**unsafe** because a nonzero judge temperature makes fusion non-deterministic and
exists only for experimentation.

## Model Aliases

- `openrouter/fusion` aliases to `fusion/general` by default.
- `fugu` aliases to `fusion/fugu`.
- `fugu-ultra` aliases to `fusion/fugu-ultra`.
- `fugu` and `fugu-ultra` remain placeholder presets until ablation-proven
  transparent Fugu-compatible configs replace them. Their model entries and
  traces self-label with `compat_placeholder - not conductor-backed yet`.

## OpenRouter Plugins Mapping

Chat requests may include an OpenRouter-style `plugins` object. OmniFusion accepts
only these fields:

- `analysis_models`: replaces the preset panel models for this request.
- `synthesis_model`: replaces the preset final synthesis model for this request.
- `web`: accepted as the request-level web-tools enable flag.
- `max_panel`: caps the panel size for this request.

Request `plugins` override the stored preset only for the current request; the
preset row is not mutated. `analysis_models` and `synthesis_model` must resolve to
registered providers or models. Unknown plugin fields are rejected, and
unregistered plugin models return HTTP 400 with `plugin_model_not_registered`.
Fusion model references such as `openrouter/fusion`, `openrouter:fusion`, and
`fusion/*` are blocked for internal panel/judge/synthesis model calls to prevent
recursive fusion.

## Client Compatibility Matrix

`docs/client-contract-matrix.json` pins each client version and classifies every
cell honestly via `coverage_type`: `sdk` (driven by the actual pinned client SDK —
currently openai-python), `wire_contract` (verified by HTTP-contract tests of the
OpenAI wire protocol the client speaks, without driving the client binary in CI),
or `manual` (a reproducible checklist, e.g. Cursor, not a CI gate). Pinning a client
version does not by itself imply SDK-level testing — read `coverage_type`.

## Web Tools

Server-side web grounding ("panel with web on") is opt-in: enable it per preset
with `web_enabled: true`, or per request with `plugins.web` (which overrides the
preset for that request only). When enabled, OmniFusion runs a bounded
`web_search` for the latest user turn before the panel, optionally fetches the
top results with `web_fetch`, and folds the untrusted, fenced, attributed results
into the panel context as a system turn. Each web call is accounted as its own
budget stage (`web_search`, `web_fetch/<n>`), and web grounding is strictly
additive — any web failure degrades to no/partial grounding rather than failing
the run. Bounded source attribution (URL, title, content hash, excerpt) is
recorded in the trace under `web_sources`.

`web_fetch` accepts only `http` and `https` URLs. It blocks cloud metadata,
loopback, private, link-local, multicast, reserved, and unspecified egress unless
private egress is explicitly enabled; cloud metadata remains blocked. Redirect
targets are revalidated on every hop.

Fetched content is treated as untrusted source data. It is byte-capped,
MIME-allowlisted, stripped of active HTML markup, and fenced in nonce-delimited
blocks before it can be included in prompts. By default traces store URL,
metadata, content hash, a bounded excerpt, cache-hit status, and truncation
status, not the full fetched page. `web_fetch` also applies an instance-level TTL
cache and per-domain fetch interval to reduce repeated origin hits.

OmniFusion does not bypass site access controls, robots policies, or terms of
service. Operators are responsible for configuring `web_fetch` and `web_search`
only for sources they are permitted to access.

## Experimental Conductor

Preset `strategy: "conductor"` enables an explicit, off-default transparent
approximation path with budgeted `plan`, `worker/<model>`, `verify`,
`repair/<n>`, and `merge` stages. It is marked in trace metadata as
experimental and `ablation_required`; it is not enabled by the `fugu` or
`fugu-ultra` aliases and does not imply benchmark advantage.

## Responses

`/v1/responses` and `/api/v1/responses` implement a minimal text-compatible
subset.

Supported request mapping:

- `input` string becomes one user message.
- `input` array supports strings, `input_text`/`text` items, and message-shaped
  objects.
- `instructions` becomes a system message.
- `tools` and `tool_choice` pass through to chat.
- `stream` maps to chat streaming.
- `max_output_tokens` maps to `max_tokens`.
- `metadata` is accepted and carried on the internal request.

Response object:

- `object: "response"`
- `status: "completed"`
- `output[].content[]` uses `{ "type": "output_text", "text": "..." }`.
- Usage maps chat `prompt_tokens` to `input_tokens` and `completion_tokens` to
  `output_tokens`.

Streaming event set:

- `response.output_text.delta`
- `response.completed`

Unsupported in this subset:

- `previous_response_id`
- stored or stateful conversations
- reasoning items
- refusal items
- tool output items in the full typed event taxonomy
- `response.output_item.*`
