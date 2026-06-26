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

### Tool-Calling Micro-Bench

Aider owns the coding-edit loop but does not isolate the tool-selection decision
OmniFusion fuses on the agentic path. The thin internal tool micro-bench covers
that case (no off-the-shelf client fits). It scores tool-name and argument
correctness against the live endpoint:

```bash
EVAL_MOCK=1 make eval-tool-smoke   # Tier A contract check (deterministic)
make eval-tool-smoke               # Tier C probe against a running instance
```

Like the coding smoke, output is raw pass/fail + cost + wall-time with no CIs.

## Real Provider Full Run

Tier C evidence requires live providers, a configured OmniFusion server, and an
API key accepted by the local instance:

```bash
make eval-coding-full
```

The full run writes JSON, JSONL, and Markdown reports under `evals/coding/runs/`.
Promote only intentional, dated provider evidence into
`evals/coding/baselines/`.

Mocked runs are machine-labeled `"tier": "mock"` (never `"C"`) so automation cannot
mistake them for real-provider evidence. To use a real run as a CI quality gate,
pass `--fail-under <rate>`; the command then exits non-zero when the pass rate is
below the threshold (the flag is ignored for mock runs, which legitimately pass 0
tasks).

## Required Baselines

No default strategy or Fugu-compatible preset may be enabled from mocked output.
Every advantage claim must compare against:

- best single configured model
- judge-selected best-of-N under the same judge/verifier budget

The accepted artifact must include provider/model identifiers, OmniFusion commit
SHA, Aider version, task-suite checksum, raw pass/fail, wall time, cost,
confidence intervals, and cost-normalized solve rates.

## External Suites

The external Tier C suite registry lives at
`evals/coding/external_suites.json`. It names the required external harnesses for
SWE-bench Lite, SWE-bench Verified, Terminal-Bench, and the internal tool bench.
These suites are live-provider-only; `mock_allowed` is false for every entry.

## Current Status

No Tier C benchmark advantage has been measured in this repository yet. The
checked-in baseline template is intentionally not a claim.
