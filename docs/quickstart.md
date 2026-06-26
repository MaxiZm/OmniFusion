# Quickstart

## Local Development

```bash
uv sync --group dev
uv run omnifusion genkey
cp .env.example .env
make dev
```

Set `OMNIFUSION_SECRET_KEY`, `OMNIFUSION_ADMIN_PASSWORD`, and
`OMNIFUSION_API_KEYS` in `.env` before serving requests.

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

Use `fusion/<preset>`, `openrouter/fusion`, `fugu`, or `fugu-ultra` as model IDs.
The `fugu` aliases are transparent compatibility placeholders unless explicitly
reconfigured.
