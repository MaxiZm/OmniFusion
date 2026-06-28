# OmniFusion Coding Evals

This harness runs Aider in OpenAI-compatible mode against OmniFusion.

Pinned driver:

- `aider-chat==0.86.2`
- `OPENAI_API_BASE=http://localhost:8000/v1`
- `OPENAI_API_KEY=<omnifusion api key>`
- `--model openai/fusion/general`

Commands:

- `make eval-coding-smoke` runs the checked-in smoke subset.
- `make eval-coding-full` runs the larger Tier C subset with confidence interval
  and cost-normalized summary output. Full runs emit JSON, task JSONL, and
  Markdown report files.
- `EVAL_MOCK=1 make eval-coding-smoke` exercises the harness contract without
  calling Aider or a provider. Mock outputs are not benchmark evidence.
