# Project

Non-invasive anemia risk screening. We're building two baseline models and an honest evaluation of how they generalize.

1. An eyelid photo model, trained on CP-AnemiC (Ghana children) and then tested on Eyes-Defy-Anemia (Italy and India adults), and the reverse.
2. A survey model on DHS Individual Recode data. Train on Zimbabwe, then test on Malawi and Zambia with no changes.

This is a student research project. The goal is a reproducible baseline we can explain, not a high score.

# Ground rules

- Python 3.11 or newer. Use scikit-learn first and PyTorch only when it's needed. Keep the code simple and readable.
- Never commit raw data. data/, results/, .venv and model weights go in .gitignore. DHS data can't be shared, so the repo stays private.
- Work on the `jaineel` branch only. Never push to `main`.
- The GitHub repo is public until the owner makes it private. Before every push, `git ls-files data results graphify-out` must print nothing.
- Don't download anything from the internet without asking me first. I place raw data in data/raw myself.
- Never invent column names or variable codes. Inspect the real files and the data dictionary first, then tell me what you found.
- Set random seeds everywhere. Settings live in configs/*.yaml and not in the code.
- Every reported number gets a bootstrap 95 percent confidence interval with 2000 resamples.
- Never report accuracy alone. Always report AUROC, PR-AUC and sensitivity at 90 percent specificity. For the survey model also report recall when only the top 30 percent by risk score get tested, plus Brier score and a calibration plot.
- Put a dummy baseline and a simple rule baseline next to every model.
- Avoid leakage. Split survey data by cluster. Split images by patient and, where a site column exists, also run leave-one-site-out.
- Save every run's metrics as JSON in results/ with the config used and the git commit hash.
- Before writing code for a new stage, show me a short plan and wait for my go-ahead. When a stage is done, run the tests or smoke checks and tell me exactly what you ran and what passed.
- State the anemia cutoff used in the config and in the README every time. Don't mix cutoffs between datasets without saying so.

# Layout

```
anemia-screening/
  CLAUDE.md
  README.md
  requirements.txt
  configs/
  data/raw/cp_anemic/
  data/raw/eyes_defy/
  data/raw/dhs/zimbabwe/, malawi/, zambia/
  data/processed/
  src/common/   metrics.py, splits.py, io.py
  src/eyelid/   audit.py, roi.py, color_features.py, train_color.py, embed.py, train_probe.py, cross_dataset.py
  src/survey/   audit.py, build_dataset.py, train_baselines.py, transfer.py
  tests/
  results/
```

# Anemia definitions

- Children 6 to 59 months, anemic if hemoglobin is below 11.0 g/dL.
- Adults, use the WHO cutoff for the person's sex and pregnancy status. When a dataset gives its own anemia label or an altitude and smoking adjusted hemoglobin, say which one we use.
- If a dataset has no sex column, stop and ask me which single adult cutoff to use.
