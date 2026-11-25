# Contributing

This is a student research project. The goal is a reproducible baseline we can explain, not a high score.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test
```

DHS data cannot be shared and is never committed. Place raw files in `data/raw/dhs/<country>/` yourself (see
[DATA.md](DATA.md)). `make test` needs no data.

## Rules

- **Never push to `main` directly.** Open a pull request and get a review.
- **Never commit `data/`, `results/`, `graphify-out/`, `.env`, `.venv` or model weights.** Before every push,
  `git ls-files data results graphify-out` must print nothing (CI checks this too).
- **Plan before a new stage.** Describe the question, models and comparisons in `configs/` *before* running them, with
  the date, and do not change them after seeing results.
- **Locked tests are spent.** Never score new models on them; use cross-validation.
- **Every reported number gets a bootstrap 95% CI** (2000 resamples, resampling villages).
- **Report AUROC, PR-AUC and sensitivity at 90% specificity**, and for the survey model recall in the top 30%, Brier
  score and a calibration plot. Never accuracy alone.
- **Put a dummy baseline and a simple rule baseline next to every model.**
- **State the anemia cutoff** in the config and the README: adjusted Hb < 12.0 g/dL (< 11.0 if pregnant); moderate or
  severe is Hb < 10.0.
- **Split by village**, set seeds, and keep settings in `configs/*.yaml`, not in code.
- **Never invent column names or variable codes.** Read the real files and the data dictionary first.

## Style

```
ruff check .
```

Keep the code simple and readable. Each script's first lines say what it does and where it fits.

## Naming

Files are named for what they contain (`zimbabwe_models.md`, `three_country_comparison.json`), not by stage number.
History and the old stage names are in [CHANGELOG.md](CHANGELOG.md). Add an entry there for every new analysis.
