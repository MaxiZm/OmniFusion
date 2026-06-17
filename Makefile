.PHONY: dev test test-int lint fmt compose-up compose-down purge export import

dev:
	uv run uvicorn src.omnifusion.main:app --reload

test:
	uv run pytest tests/

test-int:
	uv run pytest -m integration tests/

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

compose-up:
	docker compose -f deploy/docker-compose.yml up -d

compose-down:
	docker compose -f deploy/docker-compose.yml down

purge:
	uv run python -m src.omnifusion.cli purge

export:
	uv run python -m src.omnifusion.cli export

import:
	uv run python -m src.omnifusion.cli import

