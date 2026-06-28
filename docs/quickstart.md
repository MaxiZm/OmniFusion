# Quickstart

## One Command

```bash
make quickstart
```

This installs dependencies and runs `omnifusion quickstart --serve`, which
creates `.env` (from `.env.example`), generates any missing secrets
(`OMNIFUSION_SECRET_KEY`, `OMNIFUSION_ADMIN_PASSWORD`, and an
`OMNIFUSION_API_KEYS` client key), initializes the SQLite database, prints the
generated admin password and client key, then boots the dev server on
`http://127.0.0.1:8000`.

It is idempotent — re-running never overwrites secrets you have already set, only
placeholder values. Provision without serving by dropping `--serve`:

```bash
uv run omnifusion quickstart          # provision .env + DB, print next steps
uv run omnifusion quickstart --serve  # ...and start the dev server
```

## Manual Local Development

```bash
uv sync --group dev
cp .env.example .env
uv run omnifusion genkey
make dev
```

Copy only the generated key line into `OMNIFUSION_SECRET_KEY` in `.env`, then set
`OMNIFUSION_ADMIN_PASSWORD` and `OMNIFUSION_API_KEYS` before serving requests.

## Docker

```bash
docker compose -f deploy/docker-compose.yml up -d
```

The Compose file binds `127.0.0.1:8000` by default. For public access, terminate
TLS and authentication controls at a reverse proxy and forward to localhost.

## Call The API

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $OMNIFUSION_API_KEY"
```

Use `fusion/<preset>` or `openrouter/fusion` as model IDs.

## More Operator Docs

- [Providers, presets, and role prompts](providers-presets.md)
- [API compatibility](api-compatibility.md)
- [Budgeting and tracing](budgeting-tracing.md)
- [Fugu-compatible architecture](fugu-architecture.md)
- [Benchmark reproduction](benchmark-reproduction.md)
- [Security model](security-model.md)
