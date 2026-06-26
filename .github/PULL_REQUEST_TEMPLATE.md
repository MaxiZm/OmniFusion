## Summary

What does this change and why?

## Related

Closes #

## Checklist

- [ ] `make test` passes (Tier A + replay)
- [ ] `make lint` passes
- [ ] Tests added/updated for the change
- [ ] Docs updated if behavior/config changed
- [ ] No benchmark/advantage claim rests on mocked tests (Invariant 1)
- [ ] No new default strategy/stage shipped on without clearing the ablation rule
      (Invariant 2); experimental features stay behind off-by-default flags
- [ ] Any Fugu reference is described as a "transparent approximation" (Invariant 5)

See [CONTRIBUTING.md](../CONTRIBUTING.md).
