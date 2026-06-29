# OmniFusion

[![CI](https://github.com/MaxiZm/OmniFusion/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxiZm/OmniFusion/actions/workflows/ci.yml)

OmniFusion is a self-hostable, OpenAI-compatible inference endpoint that runs a
"council of models" behind one API. A request goes to a panel of models, a
deterministic judge summarizes agreement and disagreement, and a final synthesis
model produces the answer.

It is designed for people who want an OpenRouter Fusion-like workflow they can
operate themselves: explicit provider configuration, local SQLite state,
encrypted provider keys, request budgets, traces, and optional server-side web
grounding.

OmniFusion also exposes an `openrouter/fusion` alias that resolves to the default
fusion preset. The project does not claim to be a complete Sakana Fugu / Fugu
Ultra replacement; the experimental conductor strategy stays off by default until
real provider ablation evidence proves a stronger preset.

## What It Is

- An OpenAI-compatible `/v1` and `/api/v1` HTTP service.
- A model-fusion runtime with panel, judge, and synthesis stages.
- A web UI for providers, presets, traces, and playground testing.
- A self-hosted operator tool with SQLite storage and Docker Compose support.
- A place to run reproducible fusion experiments without hiding the benchmark
  and cost assumptions.

## What It Is Not

- It is not a coding-agent runtime. Aider, OpenCode, Cursor, or your own client
  still owns filesystem edits, shell commands, and local tool execution.
- It is not a benchmark-backed Fugu replacement yet. Any Fugu-style quality claim
  needs Tier C ablation evidence first.
- It is not a hosted service. You bring provider accounts, API keys, network
  policy, and deployment hardening.

## How Fusion Works

1. The client sends one OpenAI-compatible request.
2. OmniFusion resolves the requested `fusion/<preset>` model.
3. The panel stage sends the prompt to several configured models in parallel.
4. The judge stage runs at temperature 0 by default and analyzes the panel
   answers for consensus, contradictions, and likely errors.
5. The final stage synthesizes the response.
6. OmniFusion records bounded trace metadata, usage, cost, and stage status.

The default strategy is the classic `B` council flow. An experimental
`conductor` strategy exists, but it is off by default and marked as requiring
ablation evidence before any benchmark or default-strategy claim.

## Features

- **OpenAI-compatible chat API**: `/v1/chat/completions` and
  `/api/v1/chat/completions`.
- **Responses subset**: `/v1/responses` for simple text-compatible Responses API
  workloads.
- **Streaming**: canonical OpenAI-style SSE chunks, terminal usage chunks when
  requested, and trace IDs.
- **Tool call compatibility**: per-step fusion judges the panel's proposed tool
  calls and emits a judge-authored final tool call (falling back to a panel
  proposal if the judge output is unusable), all while preserving OpenAI wire shape.
- **Provider registry**: configure provider type, base URL, API key or API key
  env reference, and allowed models.
- **Preset registry**: define panel, judge, final, prompts, strategy, stage
  budgets, and cost ceilings.
- **OpenRouter-style plugins**: request-level overrides for analysis models,
  synthesis model, panel cap, and web enablement.
- **Server-side web grounding**: optional `web_search` plus hardened `web_fetch`
  with SSRF protections and bounded trace attribution.
- **Budget ledger**: integer-microdollar reserve-and-reconcile accounting in
  SQLite.
- **Admin UI**: configure providers and presets, test in a playground, inspect
  runs, and delete invalid presets.
- **Launch hygiene**: CI, tests, Docker build, dependency audit, security policy,
  contribution guide, and benchmark-evidence rules.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Provider API keys for the models you want to call
- Optional: Docker / Docker Compose
- Optional: SearXNG, Tavily, or Brave for web search

## QuickStart

The fastest path from a fresh checkout to a running server is one command:

```bash
git clone https://github.com/MaxiZm/OmniFusion.git
cd OmniFusion
make quickstart
```

`make quickstart` installs dependencies, then runs `omnifusion quickstart --serve`,
which:

1. creates `.env` from `.env.example` if it does not exist,
2. generates any **missing** secrets — `OMNIFUSION_SECRET_KEY` (Fernet),
   `OMNIFUSION_ADMIN_PASSWORD`, and an `OMNIFUSION_API_KEYS` client key — writing
   `.env` with `0o600` permissions,
3. initializes the SQLite database and key-verification token,
4. prints the generated admin password and client API key, then starts the dev
   server on `http://127.0.0.1:8000`.

It is **idempotent and safe to re-run**: real secrets you have already set are
never overwritten — only placeholder values are filled in. Provision without
booting the server by dropping `--serve`:

```bash
uv run omnifusion quickstart          # provision .env + DB, print next steps
uv run omnifusion quickstart --serve  # ...and start the dev server
```

After it boots, open `http://127.0.0.1:8000/admin` to register a provider and
preset (next section), then call `fusion/<preset>`.

`OMNIFUSION_SECRET_KEY` encrypts provider API keys stored in SQLite. Keep it safe:
losing it means existing encrypted provider keys cannot be decrypted.

## Manual Setup

Prefer to wire things up by hand? The steps below are what `quickstart` automates.

### 1. Clone And Install

```bash
git clone https://github.com/MaxiZm/OmniFusion.git
cd OmniFusion
uv sync --group dev
```

For a runtime-only local install:

```bash
uv sync
```

### 2. Configure Local Secrets

```bash
cp .env.example .env
uv run omnifusion genkey
```

Copy only the generated key line into `OMNIFUSION_SECRET_KEY` in `.env`, then set:

```bash
OMNIFUSION_ADMIN_PASSWORD=<strong admin password>
OMNIFUSION_API_KEYS=<client key one>,<client key two>
```

`OMNIFUSION_SECRET_KEY` encrypts provider API keys stored in SQLite. Keep it safe:
losing it means existing encrypted provider keys cannot be decrypted.

### 3. Start The Server

```bash
make dev
```

The development server listens on `http://127.0.0.1:8000`.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

List models:

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OMNIFUSION_API_KEY"
```

### 4. Register Providers

Open the admin UI:

```text
http://127.0.0.1:8000/admin
```

Add at least one provider with:

- provider id, commonly `default`
- provider type, for example `openai`, `openrouter`, `custom_openai`, or
  `custom_anthropic`
- API key or API key environment variable reference
- allowed model names

Provider keys entered directly in the UI are encrypted at rest with
`OMNIFUSION_SECRET_KEY`.

You can also import a YAML export:

```bash
uv run omnifusion import providers.yaml
```

Treat exports as secrets. They contain decrypted provider API keys.

### 5. Create Or Confirm A Preset

Use the admin UI or CLI:

```bash
uv run omnifusion preset list
```

Create a real preset such as `general` with panel, judge, and final models
registered against your providers.

### 6. Call Chat Completions

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OMNIFUSION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fusion/general",
    "messages": [
      {"role": "user", "content": "Explain transformers in one paragraph."}
    ],
    "max_tokens": 512
  }'
```

Use any of these model names depending on your preset:

- `fusion/<preset>`
- `openrouter/fusion`, aliasing `fusion/general`

## Docker Compose

```bash
cp .env.example .env
uv run omnifusion genkey
# edit .env
docker compose -f deploy/docker-compose.yml up -d
```

The Compose file binds `127.0.0.1:8000:8000` by default. For public access, put
OmniFusion behind a reverse proxy you control, terminate TLS there, and keep the
application container private.

Stop it with:

```bash
docker compose -f deploy/docker-compose.yml down
```

## Python SDK Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="your-omnifusion-client-key",
)

response = client.chat.completions.create(
    model="fusion/general",
    messages=[{"role": "user", "content": "Give me a concise launch checklist."}],
    max_tokens=500,
)

print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="fusion/general",
    messages=[{"role": "user", "content": "Write a short haiku about SQLite."}],
    stream=True,
    stream_options={"include_usage": True},
)

for event in stream:
    if event.choices:
        print(event.choices[0].delta.content or "", end="")
```

## Responses API Subset

OmniFusion implements a minimal text-compatible `/v1/responses` subset:

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H "Authorization: Bearer $OMNIFUSION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fusion/general",
    "instructions": "Answer tersely.",
    "input": "What does OmniFusion do?",
    "max_output_tokens": 300
  }'
```

Supported mappings include `input`, `instructions`, `tools`, `tool_choice`,
`stream`, `max_output_tokens`, and `metadata`. Stateful `previous_response_id`
is intentionally unsupported.

## Client Configuration

### Generic OpenAI-Compatible Clients

Set:

```text
base_url = http://127.0.0.1:8000/v1
api_key  = one value from OMNIFUSION_API_KEYS
model    = fusion/general
```

### Aider

The coding eval harness uses Aider in OpenAI-compatible mode:

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=$OMNIFUSION_API_KEY
aider --model openai/fusion/general
```

### OpenCode

Use [opencode.example.json](opencode.example.json) as a safe starting point. It
expects the real API key in `OMNIFUSION_API_KEY`; do not commit local
`opencode.json` files with inline keys.

## Providers

Providers are explicit. A request-level plugin cannot silently create a provider
or call an unregistered model.

Each provider stores:

- id
- type
- optional base URL
- encrypted inline API key or environment variable reference
- allowed model names

Supported provider routing includes OpenAI-compatible providers, OpenRouter,
custom OpenAI-compatible endpoints, and custom Anthropic-compatible endpoints.

## Presets

A preset is the model contract exposed as `fusion/<name>`.

Minimal PresetV2 shape:

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

Legacy preset fields are upgraded on load. If a hand-edited or stale preset no
longer validates against current limits, listing endpoints skip the bad row and
the admin UI surfaces it for deletion instead of crashing the entire listing.

## OpenRouter-Style Plugins

Chat requests may include an OpenRouter-style `plugins` object. OmniFusion
currently supports:

- `analysis_models`: request-level panel model override
- `synthesis_model`: request-level final model override
- `web`: request-level web-grounding enable flag
- `max_panel`: request-level panel size cap

Unknown plugin fields are rejected. Plugin model references must resolve to
registered providers or models. Recursive fusion references are blocked for
internal panel, judge, and synthesis calls.

## Web Grounding

Web grounding is off by default. Enable it per preset with `web_enabled: true`,
or per request with `plugins.web`.

When enabled, OmniFusion:

1. runs bounded web search for the latest user turn,
2. fetches a small number of top results,
3. sanitizes and byte-caps fetched content,
4. injects the content as nonce-fenced untrusted reference data,
5. records bounded source attribution in traces.

`web_fetch` only accepts `http` and `https`. It blocks cloud metadata, loopback,
private, link-local, multicast, reserved, and unspecified egress by default.
Private egress is allowed only when `OMNIFUSION_ALLOW_PRIVATE_EGRESS=1`, and
cloud metadata remains blocked.

Search adapters:

- `searxng`
- `tavily`
- `brave`
- custom operator-provided endpoint

Important settings:

```bash
OMNIFUSION_WEB_SEARCH_PROVIDER=searxng
OMNIFUSION_SEARXNG_BASE_URL=http://localhost:8080
OMNIFUSION_TAVILY_API_KEY=<optional>
OMNIFUSION_BRAVE_API_KEY=<optional>
OMNIFUSION_WEB_GROUNDING_MAX_RESULTS=5
OMNIFUSION_WEB_GROUNDING_FETCH_TOP=2
```

Operators are responsible for site terms, robots policies, and network
boundaries.

## Security Model

Security-sensitive defaults:

- API requests require bearer tokens from `OMNIFUSION_API_KEYS`.
- Provider API keys are encrypted at rest with `OMNIFUSION_SECRET_KEY`.
- Placeholder secrets are rejected at startup.
- Docker Compose binds localhost by default.
- CORS is disabled by default.
- Admin sessions use secure cookies by default; set
  `OMNIFUSION_SECURE_COOKIE=0` only for local HTTP development.
- `OMNIFUSION_ALLOW_PRIVATE_EGRESS=0` blocks private-network `web_fetch` egress.
- `OMNIFUSION_UNSAFE_ALLOW_MULTIWORKER=0` keeps process-local limiters honest.

Production guidance:

- terminate TLS at a reverse proxy,
- keep the app bound to localhost or a private network,
- use strong admin and API keys,
- keep `OMNIFUSION_SECRET_KEY` backed up securely,
- avoid public admin exposure,
- keep private egress disabled unless reviewed,
- treat exports and local `.env` files as secrets.

Report security issues privately. See [SECURITY.md](SECURITY.md).

## Budgets And Tracing

OmniFusion uses a SQLite budget ledger with reserve-and-reconcile accounting.
Costs are stored as integer microdollars to avoid float drift in the ledger.

Common settings:

```bash
GLOBAL_DAILY_BUDGET_USD=100.0
REQUEST_BUDGET_USD=10.0
OMNIFUSION_MAX_CONCURRENT_PER_KEY=5
OMNIFUSION_WALL_TIMEOUT=90
OMNIFUSION_MAX_REQUEST_BODY_BYTES=1000000
OMNIFUSION_MAX_TOKENS_LIMIT=1000000
```

Every stored run has an `X-OmniFusion-Run-Id` response header. Use that run ID
to inspect traces through the API or admin UI when `store` is enabled.

## Evaluation And Claims

Mock evals validate harness contracts only:

```bash
EVAL_MOCK=1 make eval-coding-smoke
EVAL_MOCK=1 make eval-tool-smoke
```

Real provider evidence requires configured providers and live runs:

```bash
make eval-coding-full
```

No benchmark or advantage claim should be made from mocked tests. Any default
strategy or Fugu-compatible claim needs Tier C evidence comparing against:

- the best single configured model,
- judge-selected best-of-N under the same judge/verifier budget.

Current status: no Tier C benchmark advantage is claimed by this repository.

## Development

Install development dependencies:

```bash
uv sync --group dev
```

Run checks:

```bash
make lint
make test
make install-smoke
make security-audit
EVAL_MOCK=1 make eval-coding-smoke
EVAL_MOCK=1 make eval-tool-smoke
docker build -f deploy/Dockerfile .
```

Format:

```bash
make fmt
```

Useful CLI commands:

```bash
uv run omnifusion quickstart          # provision .env + init DB
uv run omnifusion quickstart --serve  # ...and start the dev server
uv run omnifusion genkey
uv run omnifusion preset list
uv run omnifusion preset get general
uv run omnifusion preset save preset.yaml
uv run omnifusion preset delete general
uv run omnifusion export omnifusion.yaml
uv run omnifusion import omnifusion.yaml
uv run omnifusion purge
```

## Troubleshooting

### Startup Rejects `OMNIFUSION_SECRET_KEY`

Use only the generated key line from `uv run omnifusion genkey`, not the label or
warning text.

### `/v1/models` Returns 401

Set `OMNIFUSION_API_KEYS` in `.env`, restart the server, and send one of those
values as `Authorization: Bearer <key>`.

### The Admin UI Does Not Accept The Password

Set `OMNIFUSION_ADMIN_PASSWORD` in `.env` and restart the server. Placeholder
values are rejected.

### Provider Calls Fail

Confirm the provider id, provider type, base URL, API key, and allowed model
names in `/admin/providers`. Request-level plugin models must be registered.

### Web Grounding Produces No Sources

Check `OMNIFUSION_WEB_SEARCH_PROVIDER`, the relevant search API key or SearXNG
URL, and the egress policy. Web failures degrade to partial or no grounding
instead of failing the whole answer.

### Mock Coding Smoke Prints `0/6 Passed`

That is expected for the mock coding harness: it does not call a model or edit
files. Use it as a contract check only, not as performance evidence.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Providers, presets, and role prompts](docs/providers-presets.md)
- [API compatibility](docs/api-compatibility.md)
- [Fugu approximation architecture](docs/fugu-architecture.md)
- [Security model](docs/security-model.md)
- [Budgeting and tracing](docs/budgeting-tracing.md)
- [Benchmark reproduction](docs/benchmark-reproduction.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## License

Apache-2.0. See [LICENSE](LICENSE).
