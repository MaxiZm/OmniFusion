# Coding Ablation Artifacts

This directory holds dated Tier C ablation artifacts for orchestration changes.
Templates are not benchmark evidence and must not be used for advantage claims.

An artifact can justify enabling a strategy or component by default only when it
passes `python -m omnifusion.evals.ablations <artifact>` and its 95% confidence
interval beats both required baselines for solve-per-dollar and
solve-per-wall-second:

- best single configured model
- judge-selected best-of-N under the same judge/verifier budget

Artifacts must use bootstrap confidence intervals, include at least three
real-provider runs, and include a per-failure-mode breakdown.
