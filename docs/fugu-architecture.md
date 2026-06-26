# Fugu-Compatible Architecture

OmniFusion provides a transparent approximation of Fugu-style orchestration. It
does not claim to reproduce proprietary Sakana internals, learned policies, or
Fugu Ultra quality.

## Alias Status

The `fugu` and `fugu-ultra` model aliases resolve to compatibility presets so
OpenAI-compatible clients never see a missing model. Those presets remain
placeholder presets until ablation-proven configs replace them.

Their model entries and traces self-label with:

```text
compat_placeholder - not conductor-backed yet
```

## Conductor Strategy

Preset `strategy: "conductor"` enables the experimental conductor path. It is
off by default and records trace metadata with `experimental: true` and
`ablation_required: true`.

The stages are:

- `plan`
- `worker/<model>`
- `verify`
- `repair/<n>`
- `merge`

Repairs are bounded by settings and budgeted through the same executor as the
classic fusion strategy.

## Promotion Rule

A conductor-backed or Fugu-compatible config can become a default only after
Tier C runs show a non-overlapping confidence-interval win against both required
baselines at equal budget. Until then it is available only as an explicit,
experimental preset.
