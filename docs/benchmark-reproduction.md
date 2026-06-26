# Benchmark Reproduction

OmniFusion benchmark claims are separated by evidence tier. Mock runs are useful
for API and harness compatibility only; they are not provider benchmark evidence.

## Local Smoke

Use the mock smoke command to check that the Aider-compatible harness still
emits the expected output shape:

```bash
EVAL_MOCK=1 make eval-coding-smoke
```

This command should print raw pass/fail, cost, and wall time. A zero-pass mock
run is acceptable because the mock path does not call a model or edit files.

## Real Provider Full Run

Tier C evidence requires live providers, a configured OmniFusion server, and an
API key accepted by the local instance:

```bash
make eval-coding-full
```

The full run writes JSON, JSONL, and Markdown reports under `evals/coding/runs/`.
Promote only intentional, dated provider evidence into
`evals/coding/baselines/`.

## Required Baselines

No default strategy or Fugu-compatible preset may be enabled from mocked output.
Every advantage claim must compare against:

- best single configured model
- judge-selected best-of-N under the same judge/verifier budget

The accepted artifact must include provider/model identifiers, OmniFusion commit
SHA, Aider version, task-suite checksum, raw pass/fail, wall time, cost,
confidence intervals, and cost-normalized solve rates.

## Current Status

No Tier C benchmark advantage has been measured in this repository yet. The
checked-in baseline template is intentionally not a claim.
