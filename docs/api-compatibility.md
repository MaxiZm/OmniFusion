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
