# Advertised Claims Ledger

This ledger is the contract between the marketing site
([`OmniFusionWEB/index.html`](https://github.com/MaxiZm/OmniFusion)) and the
backend. Every advertised capability has a stable **claim ID** and an evidence
row pointing at the implementation, the operator-facing docs, and the tests that
hold it. The site tags each feature card with a matching `data-claim-id`, so a
claim can never appear on the page without a backing row here.

The gate is enforced by [`scripts/verify_claims.py`](../scripts/verify_claims.py)
(run `make verify-claims`) and by `tests/test_advertised_claims.py`:

1. Every `covered` claim must name a real implementation file, docs file, and
   test file — and each referenced path must exist.
2. Every `data-claim-id` used on the website must exist in this ledger.
3. No affirmative benchmark/Fugu *advantage* wording may appear in the docs,
   README, or website unless an accepted **Tier C** artifact exists under
   `evals/coding/baselines/`. None is claimed today (see
   [benchmark-reproduction.md](benchmark-reproduction.md)).

Status values: `covered` (implemented + tested), `partial` (implemented, narrow
contract), `planned` (not yet built — must not be advertised).

## Ledger

| Claim ID | Claim (as advertised) | Implementation | Docs | Tests | Status |
|----------|-----------------------|----------------|------|-------|--------|
| `api-chat` | OpenAI-compatible `/v1` & `/api/v1` chat completions | `src/omnifusion/api/chat.py` | `docs/api-compatibility.md` | `tests/test_integration.py`, `tests/test_m2_client_matrix.py` | covered |
| `responses-subset` | Minimal, text-compatible Responses API subset | `src/omnifusion/api/responses.py` | `docs/api-compatibility.md` | `tests/test_m2_responses.py`, `tests/test_m3b_streaming_response.py` | covered |
| `streaming` | Canonical OpenAI-style SSE streaming with terminal usage chunks | `src/omnifusion/fusion/runtime/streaming.py`, `src/omnifusion/api/sse.py` | `docs/api-compatibility.md` | `tests/test_streaming.py`, `tests/test_stream_usage.py`, `tests/test_m3b_canonical_sse.py` | covered |
| `tool-passthrough` | Tool-call passthrough preserving OpenAI shape | `src/omnifusion/api/chat.py`, `src/omnifusion/fusion/tool_orchestrator.py` | `docs/api-compatibility.md` | `tests/test_tool_fusion.py` | covered |
| `openrouter-fusion-alias` | `openrouter/fusion` alias resolves to the default preset | `src/omnifusion/api/model_names.py` | `docs/api-compatibility.md` | `tests/test_m2_aliases.py`, `tests/test_m5_openrouter_parity.py` | covered |
| `plugins` | Per-request plugins override panel/synthesis/cap/web | `src/omnifusion/fusion/plugins.py` | `docs/providers-presets.md` | `tests/test_m5_plugins_search.py` | covered |
| `providers-presets` | Explicit providers and presets, managed via API and UI | `src/omnifusion/store/providers.py`, `src/omnifusion/api/providers.py`, `src/omnifusion/store/presets.py`, `src/omnifusion/api/presets.py` | `docs/providers-presets.md` | `tests/test_m4_preset_crud.py`, `tests/test_providers_api.py` | covered |
| `budget-ledger` | Reserve-and-reconcile microdollar budget ledger | `src/omnifusion/budget/ledger.py` | `docs/budgeting-tracing.md` | `tests/test_budget.py`, `tests/test_admin_budget.py` | covered |
| `encrypted-keys` | Provider keys encrypted at rest; placeholders rejected | `src/omnifusion/secrets/crypto.py`, `src/omnifusion/settings.py` | `docs/security-model.md` | `tests/test_crypto.py`, `tests/test_security.py` | covered |
| `web-grounding` | Optional search + SSRF-hardened `web_fetch` with bounded attribution | `src/omnifusion/fusion/web_grounding.py`, `src/omnifusion/tools/web.py` | `docs/security-model.md` | `tests/test_m5_web_grounding.py`, `tests/test_m5_web_fetch.py` | covered |
| `admin-ui` | Admin console for providers, presets, playground, runs, diagnostics | `src/omnifusion/admin/routes.py`, `src/omnifusion/admin/diagnostics.py` | `docs/providers-presets.md` | `tests/test_m4_preset_crud.py`, `tests/test_admin_diagnostics.py` | covered |
| `traces-run-id` | Per-run `X-OmniFusion-Run-Id` header and replayable bounded stage trace | `src/omnifusion/api/traces.py`, `src/omnifusion/store/runs.py` | `docs/budgeting-tracing.md` | `tests/test_trace_stage_events.py`, `tests/test_contract.py` | covered |
| `docker-self-host` | One container, localhost-bound, no hosted dependency | `deploy/docker-compose.yml`, `deploy/Dockerfile` | `docs/quickstart.md` | `tests/test_m9_launch_readiness.py` | covered |
| `security-defaults` | Bearer auth, CORS off, SSRF guard, localhost binding by default | `src/omnifusion/api/auth.py`, `src/omnifusion/providers/validation.py` | `docs/security-model.md` | `tests/test_security.py`, `tests/test_security_advanced.py` | covered |
| `benchmark-honesty` | No benchmark advantage claimed from mocks; conductor off by default | `docs/benchmark-reproduction.md`, `src/omnifusion/evals/coding.py` | `docs/benchmark-reproduction.md` | `tests/test_m9_launch_readiness.py`, `tests/test_m7_eval_reporting.py` | covered |

## Notes on bounded claims

- **`responses-subset`** is deliberately a *subset*. We do not claim full OpenAI
  Responses parity — only a text-compatible surface sufficient for common
  clients. See [api-compatibility.md](api-compatibility.md).
- **`benchmark-honesty`** is a non-claim: the conductor strategy stays off by
  default and no Tier C advantage has been measured. The forbidden-wording gate
  exists to keep it that way.
- **`traces-run-id`** stage events are additive and bounded — identifiers,
  status, tokens, cost, and timing only; never prompt or response bodies.
