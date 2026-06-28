.PHONY: quickstart dev test test-int lint fmt install-smoke security-audit compose-up compose-down purge export import eval-coding-smoke eval-coding-full eval-tool-smoke eval-ablation-validate verify-claims compat-openai-python compat-openai-node compat-aider compat-opencode compat-cursor

EVAL_CODING_FLAGS :=
ifeq ($(EVAL_MOCK),1)
EVAL_CODING_FLAGS += --mock
endif
ABLATION_ARTIFACT :=

# One command from a fresh checkout to a running server: install deps, provision
# .env secrets (generating any that are still placeholders), init the DB, then
# boot the dev server. Re-running is safe — real secrets are never overwritten.
quickstart:
	uv sync --group dev
	uv run omnifusion quickstart --serve

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

install-smoke:
	uv build
	uv run omnifusion genkey >/dev/null

security-audit:
	uvx --from pip-audit pip-audit --strict --progress-spinner off

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

eval-tool-smoke:
	uv run python -m omnifusion.evals.tools \
		--config evals/coding/aider_config.json \
		--tasks evals/coding/tool_tasks.json \
		--output evals/coding/runs/tool-smoke-latest.json \
		$(EVAL_CODING_FLAGS)

eval-ablation-validate:
	@test -n "$(ABLATION_ARTIFACT)" || (echo "Set ABLATION_ARTIFACT=path/to/artifact.json" && exit 2)
	uv run python -m omnifusion.evals.ablations $(ABLATION_ARTIFACT)

# Offline gate: the advertised-claims ledger must back every covered claim with
# real evidence, match the website's claim IDs, and stay honest about benchmarks.
verify-claims:
	uv run python scripts/verify_claims.py

# Client compatibility smokes. Opt-in / live: set OMNIFUSION_BASE_URL and
# OMNIFUSION_API_KEY to exercise a running instance; otherwise they skip (exit 0).
compat-openai-python:
	uv run python scripts/compat/openai_python.py

compat-openai-node:
	node scripts/compat/openai_node.mjs

# Aider / OpenCode / Cursor are documented, reproducible operator checklists —
# they require interactive clients and real providers, so they print the steps
# rather than driving the third-party tool from CI.
compat-aider compat-opencode compat-cursor:
	@echo "See docs/compatibility-matrix.md for the $(@:compat-%=%) checklist (opt-in, live)."
