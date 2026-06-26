.PHONY: dev test test-int lint fmt compose-up compose-down purge export import eval-coding-smoke eval-coding-full eval-ablation-validate

EVAL_CODING_FLAGS :=
ifeq ($(EVAL_MOCK),1)
EVAL_CODING_FLAGS += --mock
endif
ABLATION_ARTIFACT :=

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

eval-coding-smoke:
	uv run python -m omnifusion.evals.coding smoke \
		--config evals/coding/aider_config.json \
		--tasks evals/coding/smoke_tasks.json \
		--output evals/coding/runs/smoke-latest.json \
		$(EVAL_CODING_FLAGS)

eval-coding-full:
	uv run python -m omnifusion.evals.coding full \
		--config evals/coding/aider_config.json \
		--tasks evals/coding/full_tasks.json \
		--output evals/coding/runs/full-latest.json \
		$(EVAL_CODING_FLAGS)

eval-ablation-validate:
	@test -n "$(ABLATION_ARTIFACT)" || (echo "Set ABLATION_ARTIFACT=path/to/artifact.json" && exit 2)
	uv run python -m omnifusion.evals.ablations $(ABLATION_ARTIFACT)
