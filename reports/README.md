# Survey reports: reading order

The reports are grouped by the question they answer, in the same order as Section 3 of the
[main README](../README.md). File names say what each report covers. The old stage numbers (7, 8, 29, 30, 31) and the
old file names are in [CHANGELOG.md](../CHANGELOG.md).

Every report is written by the script named at its top, and every number in it comes from that script's
`results/<name>.json`.

## Start here

| Report | What it answers |
|---|---|
| [../README.md](../README.md) | the whole survey project on one page: data, cutoffs, results, limits |
| [three_country_summary.md](three_country_summary.md) | the latest results in plain language |

## 3.1 Data and cleaning

| Report | What it answers |
|---|---|
| [zimbabwe_data_cleaning.md](zimbabwe_data_cleaning.md) | plain language: how 4,944 Zimbabwe columns became 482 questions, and how blanks are handled |
| [zimbabwe_feature_selection.md](zimbabwe_feature_selection.md) | every cleaning rule, and the locked village split |
| [zimbabwe_screening.md](zimbabwe_screening.md) | each question on its own, and which questions repeat each other |
| [malawi_cleaning.md](malawi_cleaning.md), [zambia_cleaning.md](zambia_cleaning.md) | the same cleaning rules on the 2024 surveys |
| [figures/funnel.png](figures/funnel.png) | Trisha: women interviewed, tested and complete, per country |
| [malawi_screening.md](malawi_screening.md), [zambia_screening.md](zambia_screening.md) | the same single-question screen and repeat rule, per country |

Column-by-column detail: `zimbabwe_inventory.csv`, `malawi_inventory.csv` and `zambia_inventory.csv`
(every column, kept or dropped and why); `zimbabwe_univariate.csv` and the matching Malawi and Zambia files (each
question's strength on its own). They hold labels and summary statistics only.

## 3.2 How far do six routine questions get?

| Report | What it answers |
|---|---|
| [six_question_baseline.md](six_question_baseline.md) | 6 basic questions in Zimbabwe, next to the dummy and simple rules |

## 3.3 Can more questions, or better models, do better?

| Report | What it answers |
|---|---|
| [zimbabwe_feature_selection_methods.md](zimbabwe_feature_selection_methods.md) | plain language: exactly how the computer chose questions, and what it picked |
| [zimbabwe_summary.md](zimbabwe_summary.md) | the whole Zimbabwe analysis on one page, with links to every part |
| [zimbabwe_models.md](zimbabwe_models.md) | three ways of choosing questions vs. baselines (cross-validation) |
| [zimbabwe_final_test.md](zimbabwe_final_test.md) | the one-time locked-test score |
| [zimbabwe_repeated_cv.md](zimbabwe_repeated_cv.md) | the headline cross-validation numbers, with region baselines |

## 3.4 Is it better for the sickest women?

| Report | What it answers |
|---|---|
| [zimbabwe_severity.md](zimbabwe_severity.md) | moderate or severe anemia in Zimbabwe, pre-registered |
| [severity_curve.md](severity_curve.md) | how well the model picks out women below stricter and stricter Hb cutoffs, three countries |

The other countries' moderate-or-severe results are in [three_country_comparison.md](three_country_comparison.md).

## 3.5 Does it work in other countries? and 3.6 Is the village split too harsh?

| Report | What it answers |
|---|---|
| [zimbabwe_transfer.md](zimbabwe_transfer.md) | the Zimbabwe model unchanged in Malawi and Zambia; recalibration on 200 local women |
| [three_country_comparison.md](three_country_comparison.md) | each country on its own, locked tests, home vs Zimbabwe-trained, random vs village split, calibration |
| [figures/rules_transfer.png](figures/rules_transfer.png) | Trisha: one-question rules vs the model in each country (numbers in README 3.5d) |

## 3.7 Supporting checks and exploratory variants

| Report | What it answers |
|---|---|
| [zimbabwe_tabpfn.md](zimbabwe_tabpfn.md) | TabPFN on the same folds |
| [zimbabwe_area_check.md](zimbabwe_area_check.md) | does it rank villages, or only repeat region? |
| [zimbabwe_blank_sensitivity.md](zimbabwe_blank_sensitivity.md) | did the 70%-blank rule throw away signal? |
| [zimbabwe_hb_sources.md](zimbabwe_hb_sources.md) | where the Hb ranking's gain comes from; tests saved per 100 women |
| [zimbabwe_hb_vs_class.md](zimbabwe_hb_vs_class.md) | predicting Hb vs. yes/no; what the model leans on; budget curves |
| [zimbabwe_exploratory.md](zimbabwe_exploratory.md) | four pre-registered exploratory variants (none is a headline) |
| [covariates.md](covariates.md) | do village covariates (malaria, bed nets, rainfall, travel time, night lights) add anything? Zimbabwe, Malawi, Zambia |
