# Client Compatibility Matrix

OmniFusion speaks the OpenAI Chat Completions wire shape on `/v1` and `/api/v1`,
plus a minimal text-compatible Responses subset. Any client that lets you set a
**base URL** and a **bearer key** works by pointing it at your instance and
calling `fusion/<preset>` (or the `openrouter/fusion` alias).

The authoritative per-endpoint contract is
[`client-contract-matrix.json`](client-contract-matrix.json); the wire-level
details are in [`api-compatibility.md`](api-compatibility.md). This page is the
reproducible operator checklist.

## Support matrix

| Client | Mechanism | Status | Notes |
|--------|-----------|--------|-------|
| OpenAI Python SDK | `base_url` + `api_key` | Supported | `make compat-openai-python` |
| OpenAI Node SDK | `baseURL` + `apiKey` | Supported | `make compat-openai-node` |
| `curl` / raw HTTP | `Authorization: Bearer` | Supported | see `api-compatibility.md` |
| Aider | OpenAI-compatible provider | Supported (opt-in live) | checklist below |
| OpenCode | custom provider config | Supported (opt-in live) | `opencode.example.json` |
| Cursor | OpenAI override base URL | Supported (opt-in live) | checklist below |

"Opt-in live" means the check talks to a running OmniFusion instance with real
providers; it is **not** run in CI. None of these checks assert a benchmark
advantage — they assert wire compatibility only.

## Runnable smokes

Both scripts skip cleanly (exit 0) when the endpoint env vars are unset, so they
are safe to invoke from a Makefile. Set the two variables to run them live:

```bash
export OMNIFUSION_BASE_URL=http://127.0.0.1:8000/v1
export OMNIFUSION_API_KEY=your-omnifusion-client-key

make compat-openai-python   # uses the openai Python SDK
make compat-openai-node     # uses the openai npm package (npm i openai)
```

Each prints the model, a content snippet, and usage, then asserts the response
carries the canonical OpenAI shape and an `X-OmniFusion-Run-Id` header.

## Aider checklist (opt-in, live)

1. Start OmniFusion and register at least one provider with a real key.
2. Confirm a `fusion/<preset>` preset exists (`uv run omnifusion preset list`).
3. Configure Aider to use the OpenAI-compatible endpoint:
   ```bash
   export OPENAI_API_BASE=http://127.0.0.1:8000/v1
   export OPENAI_API_KEY=your-omnifusion-client-key
   aider --model openai/fusion/general
   ```
4. Make a one-line edit request; confirm Aider applies a diff and that the run
   appears under **Admin → History** with a stage timeline.

## OpenCode checklist (opt-in, live)

1. Copy `opencode.example.json` to `opencode.json` and set the base URL to your
   instance and the key via the `OMNIFUSION_API_KEY` env var (never inline).
2. Run an OpenCode task and confirm streamed output and a returned run ID.
3. Cross-check the run under **Admin → Budget** for the reserved/spent ledger.

## Cursor checklist (opt-in, live)

1. In Cursor settings, set the OpenAI base URL override to
   `http://127.0.0.1:8000/v1` and the API key to an OmniFusion client key.
2. Set the model to `fusion/general`.
3. Send a chat message; confirm a normal completion and that streaming works.

## What is intentionally NOT claimed

- **Not full Responses parity** — only a minimal text-compatible subset.
- **No benchmark advantage** — see [benchmark-reproduction.md](benchmark-reproduction.md).
- The experimental conductor strategy stays **off by default**.
