# Stage 29: how the computer chose the questions

This page explains exactly how feature selection was done. It covers what the computer was given, the three methods it
used, how we kept the scores honest, and what each method picked. The cleaning that came before (4,944 columns down to
482 questions) is in [zimbabwe_data_cleaning.md](zimbabwe_data_cleaning.md). Every number is from the saved results:
`results/zimbabwe_models.json`, `results/zimbabwe_final_test.json` and `results/zimbabwe_repeated_cv.json`.

## What the computer started with

- **482 questions:** what was left after cleaning (Steps 1-2) and after removing repeats (Step 4). With the
  `_special` side columns for codes like "don't know", they make 613 model columns.
- **7,416 women in 321 building villages.** The 79 locked-test villages were not used at all.
- **The target: anemic yes/no** (DHS `v457`: adjusted Hb < 12.0 g/dL, or < 11.0 if pregnant).
- **The 6 base questions** from the team's earlier baseline (Stage 7), used as a yardstick:
  - `v012` age
  - `v213` pregnant
  - `v190` wealth quintile
  - `v106` education
  - `v312` contraceptive method
  - `v025` urban/rural
- **Random seed 42** everywhere, so every run gives the same result.

## The rule that keeps it honest: choose inside the training villages only

The building villages were split into **5 groups of villages** (5-fold cross-validation, `StratifiedGroupKFold`: whole
villages per group, same anemia share in each). For each group in turn:

1. **The other 4 groups are the training villages.** The selection method chooses its questions **using these villages
   only**.
2. A model is trained on those chosen questions, still on the training villages only.
3. It is scored on the held-out group, which it has never seen. Nor did the step that chose the questions.

So each method chooses its questions 5 times, once per fold, and the choice can differ between folds. This matters:
choosing questions on all villages and then testing on the same villages makes any method look better than it is.

**How a question is fed to the models:**

- **Logistic regression:**
  - Number questions are standardised, with blanks filled by the training median plus a "was blank" yes/no column.
  - Category questions become one yes/no column per answer. "Blank" is an answer of its own, and answers given by
    under 1% of training women share one column.
- **Gradient boosting:**
  - Numbers are used as they are, blanks included.
  - Categories are given to it as categories.
  - Settings: 200 trees, learning rate 0.05, up to 15 leaves, at least 50 women per leaf.

All of this is learned on the training villages only.

## Method (a): L1 logistic regression, "let the model zero out what it doesn't need"

1. Fit a logistic regression on **all 482 questions** with an **L1 penalty**. The penalty pushes the weight of
   unhelpful answers to exactly zero (solver `liblinear`).
2. **A question counts as "used" if any of its columns** (any answer, its "was blank" column, or its `_special` column)
   **has a non-zero weight.**
3. The strength of the penalty is set by **C**. A smaller C means a stronger penalty and fewer questions. We tried four
   fixed values. Each is its own row; C was **not** tuned on the test folds.

| C | questions used (average over folds) | AUROC (5-fold CV) |
|---|---|---|
| 0.003 | 12 | 0.508 [0.491, 0.525] |
| 0.01 | 24 | 0.570 [0.555, 0.585] |
| **0.03** | **65** | **0.599 [0.583, 0.615]** |
| 0.1 | 179 | 0.592 [0.576, 0.609] |

## Method (b): top-k by importance, "rank every question, keep the best k"

Inside **each training fold**:

1. **Split the training villages again**, 80% / 20%, by village.
2. **Train gradient boosting on all 482 questions** using the 80%.
3. **Measure each question's importance on the 20%** (permutation importance):
   - shuffle that question's values across women, breaking its link to anemia;
   - see how much the model's AUROC drops;
   - repeat 3 times and average.

   A question's importance is the sum over its columns, including its `_special` column. A big drop means the model
   relied on that question.
4. **Rank the questions** from most to least important and keep the **top k** (k = 10, 20, 50 or 100).
5. **Train the final model on those k questions** using the whole training fold, then score it on the held-out fold. The
   final model is either logistic regression or gradient boosting.

| k | logistic regression AUROC | gradient boosting AUROC |
|---|---|---|
| **10** | **0.595 [0.579, 0.610]** | 0.582 [0.565, 0.597] |
| 20 | 0.594 [0.578, 0.610] | 0.583 [0.567, 0.599] |
| 50 | 0.592 [0.577, 0.608] | 0.589 [0.573, 0.605] |
| 100 | 0.580 [0.564, 0.596] | 0.590 [0.576, 0.606] |

## Method (c): forward selection by DHS section, "add whole survey sections one at a time"

The 482 questions (apart from the base 6) were grouped into **15 DHS sections** by their code. The patterns are fixed in
the config, for example `v3...` is contraception and `ml...` is bed nets. The sections are background and household,
births, contraception, her health and body measures, marriage and sex, fertility wishes, partner and work, HIV, child
body measures, child health, bed nets, siblings, delivery care, domestic violence, and Zimbabwe-only questions.

Inside **each training fold**:

1. **Start with the 6 base questions.**
2. **Try adding each remaining section, one at a time.** Score each candidate set with logistic regression in a **3-fold
   cross-validation by village inside the training villages**.
3. **Keep the section that raises AUROC the most**, if it raises it by **at least 0.002**.
4. **Repeat** with the sections left, until no section adds 0.002.
5. Train the final logistic regression on the chosen questions and score it on the held-out fold.

**Result:** on average 48 questions, AUROC **0.586 [0.572, 0.600]**.

Sections added, by number of folds (out of 5):

| section | folds |
|---|---|
| marriage and sex | 3 |
| her health, breastfeeding, body measures | 2 |
| contraception | 1 |
| background and household | 1 |
| siblings | 1 |
| fertility wishes | 1 |

## Next to them: no selection at all

| model | questions | AUROC (5-fold CV) |
|---|---|---|
| dummy (everyone the same risk) | 0 | 0.496 [0.479, 0.516] |
| rule: pregnant women first | 1 | 0.505 [0.488, 0.520] |
| base 6, logistic regression | 6 | 0.577 [0.563, 0.591] |
| all 482, logistic regression | 482 | 0.561 [0.547, 0.577] |
| all 482, gradient boosting | 482 | 0.593 [0.578, 0.608] |

**Logistic regression on all 482 is worse than on 6.** It overfits, which is why selection was worth trying.

## How the winner was picked

The rule was **written in the config before running:**

- take the selection method with the **highest CV AUROC**;
- **but** a method with **fewer questions wins if it is within 0.005** of the best, because it is simpler to explain
  and less fitted to these villages.

The best was L1 at C = 0.03 (0.599, about 65 questions). **Top 10 with logistic regression** (0.595, 10 questions) is
within 0.005 of it, so **top 10 won**.

## What the winner actually picked

The top 10 is chosen again in every fold, so it changes. Questions picked across the 5 folds:

| question | folds |
|---|---|
| `v139` de jure region | 5 |
| `v704` partner's occupation | 5 |
| `v024` region | 4 |
| `v379` knows a source of contraception | 4 |
| `v312` contraceptive method | 3 |
| `v604` preferred waiting time for next child | 3 |
| `v128` wall material, `v116` toilet type, `v836` lifetime sex partners, `v442` weight-for-height, `m5_1` months of breastfeeding | 2 each |
| 16 other questions | 1 each |

For the final model (Step 6), the selection was run once on **all** building villages. It chose:

1. `v024` region
2. `v312` contraceptive method
3. `v716` her occupation
4. `d103c` ever insulted by partner
5. `v139` de jure region
6. `v704` partner's occupation
7. `v379` knows a source of contraception
8. `v394` visited a health facility in the last 12 months
9. `v315` month she started her method
10. `v836` lifetime sex partners

## What happened to it afterwards

| check | top 10 | base 6 |
|---|---|---|
| Step 5, 5-fold CV | 0.595 | 0.577 |
| Step 8a, 5 x 5 repeated CV | 0.609 [0.593, 0.625] | 0.580 [0.566, 0.594] |
| **Step 6, locked test** | **0.596 [0.563, 0.626]** | **0.590 [0.555, 0.623]** |

- **On the locked test it does not beat base 6:** +0.006 [-0.022, +0.036].
- **Its CV numbers are a bit optimistic.** The questions are chosen inside each training fold, but the decision to use
  top 10 was made using these same villages.
- **Its picks are unstable.** Only region and partner's occupation are picked every time.
- **Some picks look like luck.** "Month she started her method" is one.
- **Its probabilities are too extreme.** Its calibration slope is 0.74, against 1.02 for base 6.

## What feature selection taught us

1. **Choosing questions carefully doesn't break the ceiling.** All three methods land at 0.586-0.599, close to gradient
   boosting on all 482 (0.593) and only a little above base 6.
2. **The computer keeps choosing the same kinds of question:** place (region), contraception, partner's work, and
   pregnancy and fertility.
3. **A simple, hand-chosen short list does about as well.** In a later pre-registered check (Step 10), base 6 + region,
   7 questions, reached **0.606 [0.590, 0.621]**.
4. **The limit is the survey's information, not the selection method.** That is why the next steps are about new data
   (malaria and climate, a new survey wave) rather than more selection.

## Reproducing

```
python -m src.survey.select_features   # cleaning and the locked split
python -m src.survey.screen_features   # each question alone, and the repeat rule
python -m src.survey.select_models     # this page: the three methods, baselines and the winner (Step 5)
python -m src.survey.final_test        # the winner on the locked test (Step 6)
```

Settings are in `configs/feature_selection.yaml`, under `models`:

- the C values;
- the k values;
- importance repeats (3);
- the inner split for forward selection (3-fold), with its minimum gain (0.002);
- the section patterns;
- the winner tolerance (0.005).

Full results are in [zimbabwe_models.md](zimbabwe_models.md).
