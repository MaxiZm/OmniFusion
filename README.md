# OmniFusion

[![CI](https://github.com/MaxiZm/OmniFusion/actions/workflows/ci.yml/badge.svg)](https://github.com/MaxiZm/OmniFusion/actions/workflows/ci.yml)

An open-source, self-hostable, OpenAI-compatible **"council of models"** inference
endpoint. It stands in for OpenRouter's "Fusion" feature and **transparently
approximates** Sakana Fugu / Fugu Ultra (never a complete replacement — see
[docs/fugu-architecture.md](docs/fugu-architecture.md)).

## Overview

OmniFusion is a single OpenAI-compatible API that fuses several models:
1. **Panel**: sends the prompt to a panel of models in parallel (optionally with
   server-side web grounding).
2. **Judge**: analyzes the answers for consensus, contradictions, and likely errors
   (deterministic, temperature 0).
3. **Synthesis**: a final model synthesizes the answer.

It is an **inference endpoint**, not a coding-agent runtime: the client (Aider,
OpenCode, Cursor) owns the filesystem/shell/tool-execution loop; the only tools
OmniFusion owns are server-side `web_search` / `web_fetch`.

## Quickstart (fresh clone → call `fugu-ultra`)

```bash
# 1. Install (Python 3.12+, uv)
git clone https://github.com/MaxiZm/OmniFusion.git && cd OmniFusion
uv sync                       # or: pip install -e .

# 2. Configure secrets
cp .env.example .env
uv run omnifusion genkey      # paste the value into OMNIFUSION_SECRET_KEY in .env
#   also set OMNIFUSION_ADMIN_PASSWORD and OMNIFUSION_API_KEYS in .env

# 3. Start the server (localhost:8000)
make dev                      # or: docker compose -f deploy/docker-compose.yml up -d

# 4. Register a provider (web UI or YAML import)
#   - Web UI:   open http://localhost:8000/admin and add a provider + models, or
#   - YAML:     uv run omnifusion import providers.yaml

# 5. Confirm the presets (fugu / fugu-ultra self-create as compat placeholders
#    until ablation-proven; upgrade them to real PresetV2 configs in the admin UI)
uv run omnifusion preset list

# 6. Call it — fugu-ultra, OpenAI-compatible
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $OMNIFUSION_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "fugu-ultra", "messages": [{"role": "user", "content": "Explain transformers in one paragraph."}]}'
```

Point any OpenAI SDK or client at `http://localhost:8000/v1` with your API key and
use a model name of `fusion/<preset>`, `openrouter/fusion`, `fugu`, or `fugu-ultra`.

## Features

- **OpenAI-Compatible API** under `/v1` and `/api/v1` (chat, streaming, tools,
  a minimal `/v1/responses` subset). Works with standard OpenAI SDKs.
- **Server-side web grounding** (opt-in per preset / `plugins.web`): hardened,
  SSRF-guarded `web_fetch` + pluggable `web_search` (SearXNG / Tavily / Brave).
- **Budget Limits**: integer-microdollar SQLite ledger with reserve-and-reconcile.
- **Transparent Fugu approximation**: an experimental, ablation-gated conductor
  strategy (off by default — the default strategy is the classic council `B`).
- **Web UI**: configure providers, presets, and test via the Playground.

## Documentation

- [Quickstart](docs/quickstart.md)
- [Providers, presets & role prompts](docs/providers-presets.md)
- [API compatibility + Responses subset + plugins mapping](docs/api-compatibility.md)
- [Fugu-approximation architecture](docs/fugu-architecture.md)
- [Security model (web-fetch hardening, search adapters)](docs/security-model.md)
- [Budgeting & tracing](docs/budgeting-tracing.md)
- [Benchmark reproduction (evidence tiers)](docs/benchmark-reproduction.md)
- [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Security policy](SECURITY.md)

## Evidence & claims

No benchmark or advantage claim rests on mocked tests. The default strategy stays
`B` and `fugu`/`fugu-ultra` stay compat placeholders until a real-provider (Tier C)
ablation beats both the best single model and judge-selected best-of-N at equal
budget. See [docs/benchmark-reproduction.md](docs/benchmark-reproduction.md).
