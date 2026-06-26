# Changelog

All notable changes to OmniFusion are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

Initial eval-first roadmap implementation (Step 0 → M9).

### Added
- **Step 0 / M1a** — streaming budget reconciliation ownership fix, settings-wired
  provider circuit breaker (single half-open probe), keyed HMAC API-key hashes,
  secure-cookie default, startup secret validation, session rotation, run-id logging
  context, `/health`; cancellation-safe rate-limiter `Slot`; packaging (`pip install
  -e .`, `omnifusion` console script, `importlib.resources` templates).
- **M1b** — Aider-driven coding eval harness (`eval-coding-smoke` / `-full`) and a
  thin internal tool-calling micro-bench (`eval-tool-smoke`).
- **M1c / M2** — bounded request/preset schemas; request normalization; model
  aliases (`openrouter/fusion`, self-labeling `fugu` / `fugu-ultra` placeholders);
  `/api/v1` mirror; minimal `/v1/responses` subset; uniform error envelopes;
  client-contract matrix.
- **M3a–c** — `BudgetedExecutor` (one reconcile shield per model call); a single
  canonical SSE/response adapter (`StreamingAdapter` / `ResponseShaper`); strategy
  registry with the `StrategyResult` contract.
- **M4 / M5** — versioned `PresetV2`; structured judge JSON; deterministic judge
  (temperature 0 with an off-by-default experimental override); pluggable
  `web_search` adapters; hardened `web_fetch` (SSRF, redirect re-validation, MIME
  allowlist incl. PDF text, prompt-injection fencing, bounded persistence with an
  opt-in full-page flag); server-side web grounding wired into the panel; OpenRouter
  `plugins` mapping; recursion guard.
- **M6 / M7** — experimental, off-by-default conductor strategy (bounded repair
  loop) with dated-artifact ablation validation; off-by-default bandit selector;
  cost-normalized full eval reports; external Tier C suite registry.
- **M8 / M9** — OpenCode-style multi-step tool loop coverage; CI (lint/test/build/
  install-smoke/docker/security-audit); operator docs; localhost-bound Docker
  Compose; contributing/security/changelog and issue/PR templates.

### Notes
- No benchmark/advantage claim rests on mocked tests. The default strategy remains
  the classic council (`B`); `fugu`/`fugu-ultra` remain transparent compat
  placeholders until a real-provider (Tier C) ablation clears both baselines.

[Unreleased]: https://github.com/MaxiZm/OmniFusion/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MaxiZm/OmniFusion/releases/tag/v0.1.0
