# Changelog

The repo used to name everything by stage (stage29_models.json, make stage30). On 2026-11-24 the names became
descriptive and the stage history moved here. Old READMEs, commit messages and notes that say "Stage 29" mean
the Zimbabwe work; "Stage 30" the three-country work; "Stage 31" the covariates.

## Stage history

| Stage | Date | What it did | Now in |
|---|---|---|---|
| 7 | 2026-10-14 | 6 basic questions in Zimbabwe | 3.2 |
| 8 | 2026-10-14 | the Zimbabwe model unchanged in Malawi and Zambia | 3.5a |
| 29, Steps 1-4 | 2026-11-16 | Zimbabwe cleaning from every column; locked test | 3.1 |
| 29, Steps 5-6, 8a | 2026-11-16/17 | selection methods, locked test, repeated cross-validation | 3.3 |
| 29, Step 9 | 2026-11-16/17 | moderate or severe target | 3.4 |
| 29, Steps 7, 8b-8e, 10 | 2026-11-16/17 | TabPFN, reviewer checks, exploratory variants | 3.7 |
| Trisha | 2026-10-26/27 | data funnel; within-country overfit test; rules vs models across countries | 1, 3.5d, 3.7 |
| 30 | 2026-11-23/24 | Malawi and Zambia cleaning; each country on its own; random vs village split; calibration | 3.1, 3.4, 3.5b-c, 3.6 |
| severity curve | 2026-11-24 | AUROC against stricter and stricter Hb cutoffs | 3.4 |
| 31 | 2026-11-24 | village geospatial covariates (malaria, bed nets, rainfall, travel time, night lights), Zimbabwe, Malawi and Zambia | 3.7 |

## Old name to new name (file stems; the extension is unchanged)

| Old | New |
|---|---|
| `stage29_area_check` | `zimbabwe_area_check` |
| `stage29_blank_sensitivity` | `zimbabwe_blank_sensitivity` |
| `stage29_budget_any` | `zimbabwe_budget_any` |
| `stage29_budget_modsev` | `zimbabwe_budget_modsev` |
| `stage29_calibration` | `zimbabwe_calibration` |
| `stage29_data_cleaning` | `zimbabwe_data_cleaning` |
| `stage29_exploratory` | `zimbabwe_exploratory` |
| `stage29_feature_selection` | `zimbabwe_feature_selection` |
| `stage29_feature_selection_methods` | `zimbabwe_feature_selection_methods` |
| `stage29_final_test` | `zimbabwe_final_test` |
| `stage29_hb_sources` | `zimbabwe_hb_sources` |
| `stage29_hb_vs_class` | `zimbabwe_hb_vs_class` |
| `stage29_inventory` | `zimbabwe_inventory` |
| `stage29_models` | `zimbabwe_models` |
| `stage29_oof` | `zimbabwe_oof` |
| `stage29_repeated_cv` | `zimbabwe_repeated_cv` |
| `stage29_screening` | `zimbabwe_screening` |
| `stage29_severity` | `zimbabwe_severity` |
| `stage29_summary` | `zimbabwe_summary` |
| `stage29_tabpfn` | `zimbabwe_tabpfn` |
| `stage29_univariate` | `zimbabwe_univariate` |
| `stage30_calibration` | `three_country_calibration` |
| `stage30_countries` | `three_country_comparison` |
| `stage30_malawi_cleaning` | `malawi_cleaning` |
| `stage30_malawi_inventory` | `malawi_inventory` |
| `stage30_malawi_screening` | `malawi_screening` |
| `stage30_malawi_univariate` | `malawi_univariate` |
| `stage30_mentor_summary` | `three_country_summary` |
| `stage30_oof` | `three_country_oof` |
| `stage30_zambia_cleaning` | `zambia_cleaning` |
| `stage30_zambia_inventory` | `zambia_inventory` |
| `stage30_zambia_screening` | `zambia_screening` |
| `stage30_zambia_univariate` | `zambia_univariate` |
| `stage31_covariates` | `covariates` |
| `stage7_survey_baselines` | `six_question_baseline` |
| `stage8_survey_transfer` | `zimbabwe_transfer` |
| `survey_budget` | `six_question_budget` |
| `survey_calibration` | `six_question_calibration` |
| `survey_transfer_calibration` | `zimbabwe_transfer_calibration` |
| `stage29_zimbabwe.parquet` | `zimbabwe_clean.parquet` |
| `stage30_malawi.parquet`, `stage30_zambia.parquet` | `malawi_clean.parquet`, `zambia_clean.parquet` |

Make targets:

| Old | New |
|---|---|
| `make stage7` | `make baseline` |
| `make stage8` | `make transfer` |
| `make stage29` | `make zimbabwe` |
| `make stage30` | `make countries` |
| `make stage31` | `make covariates` |
