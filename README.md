# OmniFusion

An open-source, self-hostable clone of OpenRouter's "Fusion" feature.

## Overview

OmniFusion acts as a single OpenAI-compatible API that routes requests to a "fusion" of models:
1. **Panel**: Sends the prompt to a panel of models in parallel.
2. **Judge**: Analyzes the answers for consensus, contradictions, and likely errors.
3. **Synthesis**: A final model synthesizes the answer.

## Setup

1. Copy `.env.example` to `.env` and configure keys.
2. Run `docker compose -f deploy/docker-compose.yml up -d` or use `make dev` locally.

## Features

- **OpenAI-Compatible API**: Works with standard OpenAI SDKs.
- **Budget Limits**: Integer-microdollar SQLite ledger with reserve-and-reconcile logic.
- **Provider Capability Mapping**: Automatic dropping of unsupported optional parameters.
- **Web UI**: Configure providers, presets, and test via the Playground.
- **Strict Parsing**: Defensive JSON parsing for Judge strategy.
