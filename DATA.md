# Data

All data comes from the DHS Program (https://dhsprogram.com). It is **restricted**: you must register and be approved
for each survey, and the files cannot be shared or committed. This repository holds code, configs and aggregate reports
only; `data/` and `results/` are git-ignored, and CI fails if anything under them is tracked.

## What is needed

| Survey | Individual Recode (women) | Geospatial Covariates (optional, for the covariates analysis) |
|---|---|---|
| Zimbabwe 2015 | `ZWIR72FL` (Stata) | `ZWGC72FL.csv` |
| Malawi 2024 | `MWIR81FL` (Stata) | `MWGC81FL.csv` |
| Zambia 2024 | `ZMIR81FL` (Stata) | `ZMGC81FL.csv` |

The Geospatial Covariates files are requested on the DHS covariates page; each comes with the *DHS Geospatial
Covariate Datasets Manual*, which explains the columns. Values are averages over a 2 km (urban) or 10 km (rural) circle
around each village's displaced GPS point.

## Where to put it

```
data/raw/dhs/zimbabwe/ZWIR72DT/ZWIR72FL.DTA        data/raw/dhs/zimbabwe/ZWGC72FL/ZWGC72FL.csv
data/raw/dhs/malawi/MWIR81DT/MWIR81FL.dta          data/raw/dhs/malawi/MWGC81FL/MWGC81FL.csv
data/raw/dhs/zambia/ZMIR81DT/ZMIR81FL.dta          data/raw/dhs/zambia/ZMGC81FL/ZMGC81FL.csv
```

Keep the data dictionary files (`.DCT`, `.MAP`, `.DOC`) next to each file. Column names and variable codes are taken
from them, never guessed.

## What the code derives (all local, all git-ignored)

- `data/processed/`: cleaned tables (`<country>_clean.parquet`) and cached out-of-fold predictions (`*_oof/`).
- `results/`: one JSON per run, with its metrics, config and git commit.
- `reports/`: aggregate tables and figures with bootstrap 95% CIs. These contain no individual records and are the only
  outputs that are committed.

## Key definitions

- **Anemia** is the DHS adjusted-hemoglobin classification (`v457` 1-3): adjusted Hb < 12.0 g/dL, or < 11.0 g/dL if
  pregnant. **Moderate or severe** is adjusted Hb < 10.0 g/dL for everyone.
- **Village** is the DHS cluster (`v001`). All splits are by village, so no village appears in both training and test.
- Missing covariate values are `-9999` in Zimbabwe's file and blank cells in Malawi's and Zambia's.

## Other optional inputs

TabPFN needs a `TABPFN_TOKEN` in a git-ignored `.env` at the repo root.

## Ethics

Do not attempt to identify any person, household or village, and do not publish individual-level records or
predictions. Follow each survey's DHS data-use agreement.
