"""DHS survey models for anemia risk (Zimbabwe 2015, Malawi 2024, Zambia 2024).

Run each module from the repo root as python -m src.survey.<name>, in this order; later ones read what earlier ones save.

Stages 7-8, the first baselines (their own cleaning in build_dataset):
  audit              setup check, no reported number
  build_dataset      data
  train_baselines    Stage 7 result
  transfer           Stage 8 result

Stage 29, Zimbabwe feature selection from scratch:
  select_features    data: cleaning and the locked test split
  screen_features    data: single-question screen and the repeat rule
  select_models      result (Step 5)
  final_test         result, locked test (Step 6; already used, never score new models on it)
  tabpfn_cv          supporting check (Step 7; needs TABPFN_TOKEN in .env)
  repeated_cv        headline CV numbers (Step 8a)
  severity           second target, moderate or severe (Step 9)
  area_check         supporting check (Step 8b)
  blank_sensitivity  supporting check (Step 8c)
  hb_sources         supporting check (Step 8d)
  hb_vs_class        supporting check (Step 8e)
  exploratory        exploratory variants (Step 10)

Stage 30, each country on its own:
  select_features malawi, screen_features malawi, select_features zambia, screen_features zambia    data
  countries          result

Stage 31, village geospatial covariates (Zimbabwe; needs data/raw/dhs/zimbabwe/ZWGC72FL/ZWGC72FL.csv):
  covariates         result (cross-validation only; reads Stage 30's saved predictions)
"""
