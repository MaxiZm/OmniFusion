# Contributing

OmniFusion changes should preserve OpenAI-compatible API behavior and the roadmap
invariants in `docs/api-compatibility.md`.

Before opening a pull request:

- Run `make lint`.
- Run `make test`.
- Run `EVAL_MOCK=1 make eval-coding-smoke`.
- Do not claim benchmark advantage from mocked tests.
- Keep generated local eval runs under `evals/coding/runs/` out of commits unless
  deliberately promoting a reviewed baseline or ablation artifact.

Conductor or routing changes that might become defaults need Tier C ablation
artifacts under `evals/coding/ablations/` and must beat both required baselines.

