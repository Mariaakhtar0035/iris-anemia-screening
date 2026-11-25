# Stage 29: survey feature selection on Zimbabwe, and the review follow-ups

This stage uses only the Zimbabwe 2015 DHS, with no transfer to other countries. Every number comes from a saved run.
Each run's results, config and git commit are in `results/zimbabwe_*.json`. Each number has a 95% bootstrap CI from
2,000 resamples of whole villages. The detailed reports are linked at each step.

## Definitions

- **Primary target, any anemia:** DHS `v457` codes 1-3.
  - The cutoff is WHO's for women, applied to Hb adjusted for altitude and smoking (`v456`): **< 12.0 g/dL, or
    < 11.0 g/dL if pregnant.**
  - 9,265 women have an Hb result; 27.3% are anemic.
- **Second target, moderate or severe:** DHS `v457` codes 1-2.
  - The cutoff is adjusted Hb **< 10.0 g/dL for every woman, pregnant or not**. This is the DHS convention: moderate is
    7.0-9.9 and severe is < 7.0.
  - For pregnant women this matches WHO's moderate cutoff. For non-pregnant women it is stricter than WHO's < 11.0.
  - 653 women (7.0%) are moderately or severely anemic.
  - This target was pre-registered in the config before any model was run on it. It is a second analysis; any anemia
    stays the primary result.
- **Locked test:** 20% of villages (79 villages, 1,849 women), set aside before anything else. Everything was built on
  the other 321 villages (7,416 women). Every split keeps a village whole.
- **Hb ranking:** a gradient boosting regressor predicts adjusted Hb. Women are ranked by her cutoff minus the
  prediction, so women predicted furthest below their own cutoff come first. It gives a ranking, not a probability.
- **Clinic number:** testing women from the highest score down, how many must be tested to catch 90% of cases. Random
  testing needs 90 per 100 women. The saving is shown as **fewer tests per 100 women** (90 minus the number tested),
  with its CI.

## What we did

| step | what | report |
|---|---|---|
| 1-2 | 4,944 columns cut to 552 usable questions, with a written reason for every column dropped: blood test, not asked, bookkeeping, older-child copies, text, unit-coded, constant, or > 70% blank | [feature_selection](zimbabwe_feature_selection.md) |
| 3 | Each question on its own: 219 of 552 beat chance. The best scored 0.563 AUROC, and none reached 0.60 | [screening](zimbabwe_screening.md) |
| 4 | 70 repeats dropped (one-way Goodman-Kruskal lambda >= 0.95), leaving 482 | [screening](zimbabwe_screening.md) |
| 5 | Models and three selection methods (L1, top-k by importance, forward by DHS section), 5-fold village CV | [models](zimbabwe_models.md) |
| 6 | Locked test scored once: base 6, the CV winner (top 10), and the Hb ranking | [final_test](zimbabwe_final_test.md) |
| 7 | TabPFN 3.5 on the same folds | [tabpfn](zimbabwe_tabpfn.md) |
| 8a | 5 x 5 repeated village CV, with no-model region baselines | [repeated_cv](zimbabwe_repeated_cv.md) |
| 8b | Ranking villages against region, with a noise ceiling | [area_check](zimbabwe_area_check.md) |
| 8c | Rerun with no limit on blank answers | [blank_sensitivity](zimbabwe_blank_sensitivity.md) |
| 8d | Where the Hb ranking's gain comes from, and the clinic number | [hb_sources](zimbabwe_hb_sources.md) |
| 8e | Hb regression vs. classification, grouped importance, budget curve | [hb_vs_class](zimbabwe_hb_vs_class.md) |
| 9 | Moderate or severe target (pre-registered) | [severity](zimbabwe_severity.md) |
| 10 | Five exploratory variants (pre-registered, CV only) | [exploratory](zimbabwe_exploratory.md) |

## Results

### 1. Individual women, any anemia (primary)

Results come from 5 x 5 repeated village CV on the building villages (Step 8a) and from the locked test (Step 6).

| model | questions | CV AUROC | locked-test AUROC |
|---|---|---|---|
| dummy | 0 | 0.493 [0.475, 0.512] | 0.500 |
| region share (no model) | 1 | 0.557 [0.538, 0.576] | 0.538 [0.505, 0.572] |
| base 6, logistic regression | 6 | 0.580 [0.566, 0.594] | 0.590 [0.555, 0.623] |
| base 6 + region, logistic regression (Step 10, CV only) | 7 | 0.606 [0.590, 0.621] | - |
| top 10 by importance (Step 5's CV winner) | 10 | 0.609 [0.593, 0.625] | 0.596 [0.563, 0.626] |
| TabPFN classifier, all questions | 482 | 0.618 [0.603, 0.634] | 0.607 [0.577, 0.634] |
| **Hb ranking, all questions** | 482 | **0.617 [0.601, 0.633]** | **0.614 [0.585, 0.643]** |
| TabPFN predicting Hb (Step 10, CV only) | 482 | 0.628 [0.613, 0.643] | - |

The base 6 questions are age, pregnant, wealth quintile, education, contraceptive method and urban/rural.

- **Every method reaches the same ceiling, about 0.60-0.63.** This holds for 6, 10, 482 and 713 questions, and for
  linear models, boosted trees and a pretrained transformer.
- **The Hb ranking beats base 6.**
  - In repeated CV the gain is +0.037 [+0.022, +0.054].
  - On the locked test it is +0.024 [-0.001, +0.052]. That is borderline, because the test is small.
- **The top 10's CV number is optimistic.** It was chosen as Step 5's winner using these same villages.
  - On the locked test it does not beat base 6: +0.006 [-0.022, +0.036].
  - Its questions change from fold to fold.
  - Its calibration slope is 0.74, against 1.02 for base 6.
- **A short 7-question model gets most of the way.** Base 6 + region reaches 0.606, a gain of +0.026 [+0.014, +0.040]
  over base 6. This is Step 10, cross-validation only.

**Where the gain comes from (Steps 8d and 8e).** All rows below use gradient boosting.

| comparison | any anemia | moderate or severe |
|---|---|---|
| Hb regressor, base 6 + region minus base 6 | +0.026 [+0.014, +0.039] | +0.021 [+0.002, +0.040] |
| Hb regressor, all questions minus base 6 + region | +0.023 [+0.011, +0.035] | +0.028 [+0.009, +0.047] |
| all questions: Hb regressor minus yes/no classifier (same learner, same folds) | +0.022 [+0.014, +0.030] | +0.034 [+0.013, +0.054] |
| all questions: Hb regressor minus TabPFN yes/no classifier | -0.002 [-0.010, +0.007] | - |

- **About half of the gain over base 6 comes from region.** The other half comes from the other questions, used by a
  flexible model.
- **Predicting the Hb number helps gradient boosting**, compared with predicting yes/no on the same questions and folds.
- **It is not the only route.** A stronger classifier (TabPFN) reaches the same score without the Hb number.
- With only base 6, the Hb regressor is slightly worse than logistic regression: -0.012 [-0.022, -0.003].

**Which blocks of questions the models rely on (Step 8e).** This is grouped permutation importance: each DHS section's
questions were shuffled together, with region as its own group. Results are the AUROC drop on held-out folds, as a mean
over 5 folds.

- **Hb regressor:**
  - contraception +0.035
  - region +0.029
  - fertility wishes, including the pregnancy/postpartum "exposure" question, +0.021
  - birth history +0.013
  - her health, breastfeeding and body measures +0.010
- **Classifier:**
  - region +0.031
  - contraception +0.020
  - partner and work +0.008
  - her health and body measures +0.005
- Domestic violence, siblings, delivery care and child body measures each contribute about 0.

This is **consistent with** the models using proxies for place, blood loss and pregnancy (contraception, menstruation,
postpartum stage) and nutrition (body measures). It shows which blocks the models rely on, not why. Region may also
carry differences between survey teams, their Hb devices or their fieldwork, not only biology. We cannot separate these
in this data.

### 2. Ranking villages (Step 8b)

A village's score is the mean score of its women, compared with its measured anemia share.

- **Noise ceiling:** each village has about 23 tested women, so its measured share is noisy.
  - Across all villages, the split-half reliability (Spearman-Brown corrected) is 0.39. So a perfect model could
    correlate at most **sqrt(0.39) = 0.62** with the measured shares.
  - Within regions, the reliability is 0.22, so the most a perfect model could reach is 0.47.
  - Spearman-Brown is defined for ordinary correlations, so applying it to Spearman correlations is an approximation.

| model (CV, 321 villages) | Spearman, all villages | top-20% overlap |
|---|---|---|
| region share (no model) | 0.30 [0.19, 0.40] | 0.44 [0.31, 0.53] |
| base 6 | 0.23 [0.13, 0.34] | 0.30 [0.20, 0.41] |
| Hb ranking, base 6 + region | 0.42 [0.32, 0.50] | 0.45 [0.33, 0.55] |
| **Hb ranking, all questions** | **0.42 [0.32, 0.51]** | **0.53 [0.42, 0.62]** |

Top-20% overlap is the share of the measured worst 20% of villages that the model also puts in its top 20%. Chance is 0.20.

- **The Hb ranking beats region.**
  - Spearman is higher by +0.12 [+0.05, +0.20].
  - Top-20% overlap is higher by +0.09 [+0.00, +0.19], which is borderline.
  - 0.42 out of a possible 0.62 is 68% of the maximum correlation, or about 46% of the variance that can be explained.
- **For village ranking, base 6 + region does as well as all 482 questions** (+0.005 [-0.041, +0.047]). Adding village
  averages of wealth, education and contraception does not help (Step 10).
- **Within regions,** base 6 (which has no region question) reaches 0.15 [0.03, 0.26] in CV. Region alone is 0 by
  construction.
  - We found a cross-validation artifact after the first run; only how it is read changed. In CV a village is left
    out of its own region's training average, which pulls any model that uses region downward within regions. So
    within-region CV numbers are not interpretable for models that use region.
  - On the locked test (descriptive only), which has no artifact, the top 10 and the Hb ranking reach 0.32 within
    regions ([0.04, 0.54] and [0.02, 0.55]). The ceiling there is 0.58.
  - The locked test has only 79 villages and 10 regions, so it cannot confirm this.

### 3. Moderate or severe (Step 9, pre-registered)

| model | CV AUROC | locked-test AUROC |
|---|---|---|
| region share (no model) | 0.566 [0.537, 0.593] | 0.550 [0.495, 0.607] |
| base 6 | 0.630 [0.604, 0.656] | 0.634 [0.580, 0.688] |
| base 6 + region, logistic / ordinal (Step 10, CV only) | 0.646 [0.622, 0.671] / 0.651 [0.626, 0.676] | - |
| **Hb ranking, all questions** | **0.666 [0.643, 0.688]** | **0.684 [0.639, 0.726]** |

- **The Hb ranking beats base 6.**
  - In CV the gain is +0.036 [+0.011, +0.059].
  - On the locked test it is +0.050 [+0.009, +0.091]. This was the first time the locked test was scored for this
    target.
- **PR-AUC is 0.13, against 0.07 by chance.** Of the women flagged for testing, only about 8-9% are moderately or
  severely anemic.

### 4. What a clinic would see

**Fewer blood tests per 100 women, to still catch 90% of cases.** Random testing needs 90 per 100.

| target | model | CV | locked test |
|---|---|---|---|
| any anemia | region share | 1 [-0, 3] | none (ties within regions: everyone is tested) |
| any anemia | base 6 | 4 [3, 6] | 3 [0, 7] |
| any anemia | **Hb ranking** | **6 [4, 8]** | **8 [2, 11]** |
| moderate or severe | region share | 0 [-4, 5] | none (ties) |
| moderate or severe | base 6 | 6 [2, 13] | 10 [-1, 17] |
| moderate or severe | **Hb ranking** | **17 [11, 23]** | **17 [11, 28]** |

**Share of cases found at a fixed testing budget** (testing from the highest score down). Random testing finds the same
share as it tests.

| target, model (CV) | test 10% | test 20% | test 30% | test 50% | test 70% |
|---|---|---|---|---|---|
| any anemia, base 6 | 13% [12, 14] | 25% [24, 27] | 38% [36, 39] | 59% [57, 61] | 77% [75, 78] |
| any anemia, Hb ranking | 16% [14, 17] | 29% [28, 31] | 42% [40, 44] | 62% [60, 64] | 80% [78, 81] |
| moderate or severe, base 6 | 20% [16, 22] | 33% [29, 36] | 46% [41, 50] | 70% [66, 74] | 84% [80, 87] |
| moderate or severe, Hb ranking | 21% [18, 24] | 38% [34, 41] | 49% [45, 53] | 72% [68, 76] | 89% [87, 92] |

Locked-test values (descriptive), for moderate or severe with the Hb ranking:

| test | 10% | 20% | 30% | 50% | 70% |
|---|---|---|---|---|---|
| cases found | 25% [17, 33] | 42% [35, 51] | 53% [44, 62] | 74% [67, 81] | 89% [83, 93] |

Every value is in [hb_vs_class](zimbabwe_hb_vs_class.md), with figures for both targets.

**This is a weak individual screen.** At small budgets it finds about twice the random share of badly anemic women: 38%
of cases when testing 20%. But to catch 90% of cases, it saves only about 6-8 tests per 100 women for any anemia, and
about 17 per 100 for moderate or severe. The Hb finger-prick test is cheap and quick, so survey answers alone are not
an acceptable individual screen.

### 5. Exploratory variants (Step 10)

These were pre-registered before running, are CV only, and the locked test was not used. Success means a paired gain of
at least 0.01 with the CI above 0.

| variant | reference | AUROC gain | success |
|---|---|---|---|
| A. Hb ranking + TabPFN, ranks averaged (any anemia) | Hb ranking | +0.007 [+0.002, +0.011] | no (below the bar) |
| B. P(Hb below her cutoff), modelled or constant spread | Hb ranking | -0.005 to +0.000 | no |
| C. base 6 + region, logistic (any anemia) | base 6 | +0.026 [+0.014, +0.040] | **yes** |
| C. base 6 + region, logistic (moderate or severe) | base 6 | +0.016 [+0.000, +0.032] | **yes**, borderline |
| C. base 6 + region, ordinal over 4 DHS levels (moderate or severe) | base 6 | +0.021 [-0.000, +0.041] | no, borderline |
| D. village averages added to the Hb ranking | Hb ranking | +0.000 to +0.002 | no |
| E. TabPFN predicting Hb (any anemia) | Hb ranking | +0.011 [+0.003, +0.019] | **yes**, just |
| E. TabPFN predicting Hb (moderate or severe) | Hb ranking | +0.011 [-0.002, +0.025] | no |

- **B gives probabilities, but they are not better than base 6.**
  - Its Brier score is the same (0.196-0.197).
  - With a constant spread, calibration is reasonable in slope (about 0.9), but the intercept is below 0, so it
    overestimates risk.
  - With a modelled spread, the predictions are too extreme: slope 0.60-0.75.
- **The ordinal model is the best calibrated model** (slope 1.11).
- **E is the highest CV score in Stage 29:** 0.628 [0.613, 0.643] for any anemia and 0.677 [0.654, 0.698] for moderate
  or severe. Treat it with caution:
  - the gain is about 0.01;
  - it is one of several variants tried on the same villages, so it is likely a little optimistic;
  - its run condition (from Step 8e) was stated in chat before 8e's results, but written into the config after them;
  - it was never scored on the locked test.

  Only a new survey can confirm it.

### 6. Checks

- **The blank rule (Step 8c).** We put back the 329 questions dropped for being more than 70% blank; 231 of them
  survive the repeat rule.
  - Gradient boosting changes by +0.005 [+0.000, +0.009], the Hb ranking by +0.002 [-0.001, +0.006], and the Hb error by
    -0.001 g/dL.
  - The returning questions are mostly vaccination dates and pill brand.
  - Iron in pregnancy, antenatal visits and months since last birth were kept all along: they are 51%, 51% and 27%
    blank.
- **How the locked test was used:**
  - Step 6 scored the main models.
  - Step 7 scored TabPFN, under the rule in force when that run started.
  - Step 9 scored the second target, its first look.
  - Steps 8b, 8d and 8e add descriptive metrics on the same frozen models. The refits reproduce the earlier AUROCs
    exactly.
  - An amended TabPFN rule was written before Step 7's output was opened. TabPFN passed it on the repeated CV, so Step
    7's locked-test score was reused rather than taking a third look.
  - Step 10 did not use the locked test.
  - **The locked test is no longer fresh.** Confirming any further change needs a new survey, such as the 2010-11
    Zimbabwe DHS.

## What it means

1. **Survey answers alone give about 0.61 AUROC for any anemia** (locked test 0.614 [0.585, 0.643]). Choosing questions
   carefully, adding more, or using a stronger model all reach the same ceiling. The limit is the information in the
   survey, not the method.
2. **A short version keeps most of it.** Base 6 + region (7 questions) reaches 0.606, against 0.617 with all 482
   questions (CV).
3. **What the models rely on is consistent with proxies for the causes:** place, blood loss and pregnancy, and
   nutrition.
   - None of these measure a cause directly.
   - Region may also carry differences between survey teams or Hb devices.
   - Village-level malaria and climate data is therefore the natural next addition.
4. **As an individual screen it is weak.** To catch 90% of anemic women it saves about 6-8 blood tests per 100 women
   (about 17 per 100 for moderate or severe).
5. **For ranking villages it adds value beyond region.** The village correlation is 0.42 against 0.30 for region, out
   of a possible 0.62, and base 6 + region already reaches that.

## Possible next steps

- **Village malaria and climate data** (the DHS geospatial covariates). These need the file from the DHS website.
  - The test is whether base 6 + covariates can replace base 6 + region. That would make the model portable.
- **Nearby villages' anemia rates,** for village ranking only. This needs the DHS GPS file and must be computed inside
  each fold. It only helps within surveyed areas.
- **The 2010-11 Zimbabwe DHS, as an untouched second test.** First check which waves tested women's Hb.

## Reproducing

From the repo root, after placing the Zimbabwe file in `data/raw/dhs/zimbabwe/`, run these in order:

```
python -m src.survey.select_features     # Steps 1-2 and the locked split
python -m src.survey.screen_features     # Steps 3-4
python -m src.survey.select_models       # Step 5
python -m src.survey.final_test          # Step 6
python -m src.survey.tabpfn_cv           # Step 7 (needs TABPFN_TOKEN in .env)
python -m src.survey.repeated_cv         # Step 8a (several hours; cached in data/processed/zimbabwe_oof/)
python -m src.survey.area_check          # Step 8b
python -m src.survey.blank_sensitivity   # Step 8c
python -m src.survey.severity            # Step 9
python -m src.survey.hb_sources          # Step 8d
python -m src.survey.hb_vs_class         # Step 8e
python -m src.survey.exploratory         # Step 10 (variant E: about 1 h of TabPFN on a GPU, saved per fold)
```

TabPFN 3.5 needs its licence accepted once at ux.priorlabs.ai, and the API key in `.env` as `TABPFN_TOKEN`. The `.env`
file is git-ignored.
