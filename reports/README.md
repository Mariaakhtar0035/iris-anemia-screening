# Survey reports: reading order

The reports are grouped by the question they answer, in the same order as Section 3 of the
[main README](../README.md). File names still carry the stage in which each report was written (7, 8, 29, 30). The
[stage history](../README.md#38-stage-history) maps stages to sections.

Every report is written by the script named at its top, and every number in it comes from that script's
`results/<name>.json`.

## Start here

| Report | What it answers |
|---|---|
| [../README.md](../README.md) | the whole survey project on one page: data, cutoffs, results, limits |
| [stage30_mentor_summary.md](stage30_mentor_summary.md) | the latest results in plain language, for the mentors |

## 3.1 Data and cleaning

| Report | What it answers |
|---|---|
| [stage29_data_cleaning.md](stage29_data_cleaning.md) | plain language: how 4,944 Zimbabwe columns became 482 questions, and how blanks are handled |
| [stage29_feature_selection.md](stage29_feature_selection.md) | Zimbabwe Steps 1-2: every cleaning rule, and the locked village split |
| [stage29_screening.md](stage29_screening.md) | Zimbabwe Steps 3-4: each question on its own, and which questions repeat each other |
| [stage30_malawi_cleaning.md](stage30_malawi_cleaning.md), [stage30_zambia_cleaning.md](stage30_zambia_cleaning.md) | the same cleaning rules on the 2024 surveys |
| [figures/funnel.png](figures/funnel.png) | Trisha: women interviewed, tested and complete, per country |
| [stage30_malawi_screening.md](stage30_malawi_screening.md), [stage30_zambia_screening.md](stage30_zambia_screening.md) | the same single-question screen and repeat rule, per country |

Column-by-column detail: `stage29_inventory.csv`, `stage30_malawi_inventory.csv` and `stage30_zambia_inventory.csv`
(every column, kept or dropped and why); `stage29_univariate.csv` and the matching Malawi and Zambia files (each
question's strength on its own). They hold labels and summary statistics only.

## 3.2 How far do six routine questions get?

| Report | What it answers |
|---|---|
| [stage7_survey.md](stage7_survey.md) | 6 basic questions in Zimbabwe, next to the dummy and simple rules |

## 3.3 Can more questions, or better models, do better?

| Report | What it answers |
|---|---|
| [stage29_feature_selection_methods.md](stage29_feature_selection_methods.md) | plain language: exactly how the computer chose questions, and what it picked |
| [stage29_summary.md](stage29_summary.md) | all of Stage 29 on one page, with links to every step |
| [stage29_models.md](stage29_models.md) | Step 5: three ways of choosing questions vs. baselines (cross-validation) |
| [stage29_final_test.md](stage29_final_test.md) | Step 6: the one-time locked-test score |
| [stage29_repeated_cv.md](stage29_repeated_cv.md) | Step 8a: the headline cross-validation numbers, with region baselines |

## 3.4 Is it better for the sickest women?

| Report | What it answers |
|---|---|
| [stage29_severity.md](stage29_severity.md) | Step 9: moderate or severe anemia in Zimbabwe, pre-registered |

| [severity_curve.md](severity_curve.md) | how well the model picks out women below stricter and stricter Hb cutoffs, three countries |

The other countries' moderate-or-severe results are in [stage30_countries.md](stage30_countries.md).

## 3.5 Does it work in other countries? and 3.6 Is the village split too harsh?

| Report | What it answers |
|---|---|
| [stage8_transfer.md](stage8_transfer.md) | the Zimbabwe model unchanged in Malawi and Zambia; recalibration on 200 local women |
| [stage30_countries.md](stage30_countries.md) | each country on its own, locked tests, home vs Zimbabwe-trained, random vs village split, calibration |
| [figures/rules_transfer.png](figures/rules_transfer.png) | Trisha: one-question rules vs the model in each country (numbers in README 3.5d) |

## 3.7 Supporting checks and exploratory variants

| Step | Report | What it answers |
|---|---|---|
| 7 | [stage29_tabpfn.md](stage29_tabpfn.md) | TabPFN on the same folds |
| 8b | [stage29_area_check.md](stage29_area_check.md) | does it rank villages, or only repeat region? |
| 8c | [stage29_blank_sensitivity.md](stage29_blank_sensitivity.md) | did the 70%-blank rule throw away signal? |
| 8d | [stage29_hb_sources.md](stage29_hb_sources.md) | where the Hb ranking's gain comes from; tests saved per 100 women |
| 8e | [stage29_hb_vs_class.md](stage29_hb_vs_class.md) | predicting Hb vs. yes/no; what the model leans on; budget curves |
| 10 | [stage29_exploratory.md](stage29_exploratory.md) | four pre-registered exploratory variants (none is a headline) |
