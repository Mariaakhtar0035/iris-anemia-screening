# Survey reports: reading order

Every report is written by the script named at its top, and every number in it comes from that script's
`results/<name>.json`. Start at the top; each row says what you get from reading it.

## Start here

| Report | What it answers |
|---|---|
| [../README.md](../README.md) | the whole survey project on one page: data, cutoffs, results, limits |
| [stage30_mentor_summary.md](stage30_mentor_summary.md) | Stage 30 in plain language for the mentors, and why it uses no feature selection |
| [stage30_countries.md](stage30_countries.md) | the latest results: random vs village split, each country on its own, transfer, calibration |
| [stage29_summary.md](stage29_summary.md) | Stage 29 in one page, with links to every step |

## How it was done, in plain language

| Report | What it answers |
|---|---|
| [stage29_data_cleaning.md](stage29_data_cleaning.md) | how 4,944 columns became 482 questions, and how blank answers are handled |
| [stage29_feature_selection_methods.md](stage29_feature_selection_methods.md) | exactly how the computer chose questions, and what it picked |

## Stages 7-8: the first baselines

| Report | What it answers |
|---|---|
| [stage7_survey.md](stage7_survey.md) | six questions in Zimbabwe, next to the dummy and simple rules |
| [stage8_transfer.md](stage8_transfer.md) | the same model unchanged in Malawi and Zambia, and recalibration on 200 local women |

## Stage 29: Zimbabwe, every step

| Step | Report | What it answers |
|---|---|---|
| 1-2 | [stage29_feature_selection.md](stage29_feature_selection.md) | the cleaning rules and the locked village split |
| 3-4 | [stage29_screening.md](stage29_screening.md) | each question on its own; which questions repeat each other |
| 5 | [stage29_models.md](stage29_models.md) | three ways of choosing questions vs. baselines (cross-validation) |
| 6 | [stage29_final_test.md](stage29_final_test.md) | the one-time locked-test score |
| 7 | [stage29_tabpfn.md](stage29_tabpfn.md) | TabPFN on the same folds |
| 8a | [stage29_repeated_cv.md](stage29_repeated_cv.md) | the headline cross-validation numbers, with region baselines |
| 8b | [stage29_area_check.md](stage29_area_check.md) | does it rank villages, or only repeat region? |
| 8c | [stage29_blank_sensitivity.md](stage29_blank_sensitivity.md) | did the 70%-blank rule throw away signal? |
| 8d | [stage29_hb_sources.md](stage29_hb_sources.md) | where the Hb ranking's gain comes from; tests saved per 100 women |
| 8e | [stage29_hb_vs_class.md](stage29_hb_vs_class.md) | predicting Hb vs. yes/no; what the model leans on; budget curves |
| 9 | [stage29_severity.md](stage29_severity.md) | the second target, moderate or severe anemia |
| 10 | [stage29_exploratory.md](stage29_exploratory.md) | four pre-registered exploratory variants (none is a headline) |

Column-by-column detail: [stage29_inventory.csv](stage29_inventory.csv) (every column, why it was kept or dropped) and
[stage29_univariate.csv](stage29_univariate.csv) (each question's strength on its own). Both hold labels and summary
statistics only.

## Stage 30: Malawi and Zambia cleaning

| Report | What it answers |
|---|---|
| [stage30_malawi_cleaning.md](stage30_malawi_cleaning.md), [stage30_zambia_cleaning.md](stage30_zambia_cleaning.md) | the Stage 29 cleaning rules applied to each 2024 survey |
| [stage30_malawi_screening.md](stage30_malawi_screening.md), [stage30_zambia_screening.md](stage30_zambia_screening.md) | each question on its own, and the repeat rule, per country |

The matching `stage30_{malawi,zambia}_{inventory,univariate}.csv` files hold the same column-by-column detail.
