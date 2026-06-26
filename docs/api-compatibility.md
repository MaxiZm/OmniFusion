# API Compatibility

OmniFusion exposes OpenAI-compatible routes under both `/v1` and `/api/v1`.

## Chat

- `/v1/chat/completions` and `/api/v1/chat/completions` accept chat requests.
- `developer` messages normalize to `system`.
- `max_completion_tokens` normalizes to `max_tokens`.
- Legacy `functions` and `function_call` normalize to `tools` and `tool_choice`.
- Text content-part arrays are accepted; non-text parts are rejected.

## Model Aliases

- `openrouter/fusion` aliases to `fusion/general` by default.
- `fugu` aliases to `fusion/fugu`.
- `fugu-ultra` aliases to `fusion/fugu-ultra`.
- `fugu` and `fugu-ultra` are placeholder presets until the conductor work lands.
  Their model entries and traces self-label with
  `compat_placeholder - not conductor-backed yet`.

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

## Web Tools

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
