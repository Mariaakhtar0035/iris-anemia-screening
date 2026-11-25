## What this changes

<!-- One or two sentences. -->

## Checklist

- [ ] `make test` passes
- [ ] `git ls-files data results graphify-out` prints nothing
- [ ] Any new number has a bootstrap 95% CI (2000 resamples)
- [ ] Any new model has a dummy baseline and a simple rule baseline next to it
- [ ] The anemia cutoff is stated in the config and in the README (adjusted Hb < 12.0 g/dL, < 11.0 if pregnant; moderate or severe: Hb < 10.0)
- [ ] No locked test was scored (they are spent: cross-validation only)
- [ ] New settings are in `configs/`, seeds are set, and README / CHANGELOG are updated
