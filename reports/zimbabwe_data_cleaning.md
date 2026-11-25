# Stage 29: how we cleaned the Zimbabwe survey data

This page explains how we went from the raw DHS file to the questions the models use. It covers what we started with,
what we removed and why, and how blank answers are handled. The numbers come from `results/zimbabwe_inventory.json` and
`results/zimbabwe_screening.json`. The full column-by-column list is in
[zimbabwe_inventory.csv](zimbabwe_inventory.csv): every column, its label, how blank it is, and whether it was kept or why
it was dropped.

## What we started with

- **File:** the Zimbabwe 2015 DHS Individual Recode (`ZWIR72FL.DTA`), one row per woman aged 15-49.
- **9,955 women** were interviewed, and the file has **4,944 columns**.
- **We kept the 9,265 women with an Hb (blood test) result.** The other 690 have no anemia label, so they can't be
  used.
- **Anemia label:** DHS's own `v457`. A woman counts as anemic if her Hb, adjusted for altitude and smoking, is
  **< 12.0 g/dL, or < 11.0 g/dL if pregnant.** 27.3% are anemic.

## Step 0: lock away a final test

Before looking at anything, we set aside 20% of villages as a final test. Whole villages go to one side, because
neighbours are alike.

| part | women | villages | anemic |
|---|---|---|---|
| building set (used for everything below) | 7,416 | 321 | 27.3% |
| locked test (kept for the end) | 1,849 | 79 | 27.3% |

## Steps 1-2: from 4,944 columns to 552 questions

Every column was checked against the rules below, in order, and dropped for the **first** reason that applies. None of
these rules looks at how a column relates to anemia. So the cleaning can't "peek" at the answer.

| # | reason dropped | columns | what it means | example |
|---|---|---|---|---|
| 1 | the anemia label itself | 1 | the answer we are predicting | `v457` anemia level |
| 2 | **blood test (leakage)** | 42 | anything from the Hb test: her Hb, her children's Hb, consent, referral. Using it would be cheating. | `v456` adjusted Hb, `hw53_1` child's Hb |
| 3 | not asked in Zimbabwe | 1,609 | DHS labels these "na - ": the question wasn't in this country's questionnaire, so the column is empty | - |
| 4 | bookkeeping | 234 | IDs, sampling weights, interviewer, dates, interview length, data-quality flags. None of these describes the woman. | `v001` village number, `v005` sample weight |
| 5 | repeated copies | 2,102 | DHS repeats some questions for every birth (`_01`...`_20`) or every recent child (`_1`...`_6`). We kept only the most recent birth or child. | `b4_02`, the sex of her second-to-last child |
| 6 | free text | 2 | text, such as the month-by-month calendar | - |
| 7 | time written as a code | 14 | a time stored as "unit digit + number" (e.g. 101 = 1 day, 301 = 1 month), which is meaningless as a number. DHS usually also gives a clean version, which we kept. | `v215` time since last period (kept instead: `v226`, in months) |
| 8 | constant | 59 | every tested woman gave the same answer, or the column is entirely blank | `h42_1` child taking iron pills (blank for everyone) |
| 9 | **mostly blank** | 329 | **more than 70%** blank among tested women (see below) | `v214` duration of current pregnancy (94% blank) |
| | **kept** | **552** | 440 categories (e.g. education) and 112 numbers (e.g. age, BMI) | |

4,392 columns were dropped in total, leaving **552 questions**.

## Step 4: removing questions that repeat each other

Some kept questions say almost the same thing as another one. A question was dropped when a question already kept
predicts its answer almost perfectly: **at least 95% of the guessing mistakes removed**, using Goodman-Kruskal lambda,
among women who answered both.

The rule is **one-way on purpose.** For example, "pregnant" is fully contained in "exposure", which has a pregnant
answer, so "pregnant" goes. But urban/rural is *not* fully contained in wealth, so both stay.

**70 questions dropped, leaving 482.** Some examples:

| dropped | because this kept question already says it |
|---|---|
| `v213` currently pregnant | `v625` exposure (one of its answers is "pregnant") |
| `v012` age | `v447a`, the woman's age from the household questionnaire |
| `v101` region | `v024` region (the same thing coded twice) |
| `v102` urban/rural | `v025` urban/rural |
| `v135` usual resident or visitor | `v139` de jure region of residence |

## How we handle blank answers

DHS has a lot of blanks. Most are not "missing data" but **skip patterns**: a woman with no recent birth is never asked
about antenatal visits. So a blank often *carries information*, and we never just throw those women out.

### 1. Which questions we keep at all

A question is dropped only if it is **more than 70% blank**. How the cutoff changes the count:

| at most this share blank | questions kept |
|---|---|
| 10% | 278 |
| 30% | 334 |
| 50% | 409 |
| **70% (our rule)** | **552** |

Of the 552 kept questions:

- 203 have no blanks;
- 131 are 1-30% blank;
- 75 are 30-50% blank;
- 143 are 50-70% blank.

### 2. Inside the questions we keep

| kind of question | example | what a blank becomes |
|---|---|---|
| **category** | education, contraceptive method | its own answer, **"blank"**, next to "pill", "none" and so on. The model can learn that women who were skipped differ. |
| **number**, logistic regression | BMI, age at first birth | filled with the typical (median) value, **plus** a yes/no column "was this blank?", so the model still knows it was missing |
| **number**, gradient boosting or TabPFN | same | left blank. These models handle blanks themselves and learn where blank women belong. |

**Safety rule:** fill-in values such as the median are learned from the **training villages only**, inside every fold.
The test villages never influence how gaps are filled.

### 3. Codes hiding inside number questions

Some number questions also contain special codes. For example, `v115`, time to the water source in minutes, uses
**996 = "on premises"**, **997 = "not a de jure resident"** and **998 = "don't know"**. Reading 996 as 996 minutes would
be wrong.

For each such question:

- the code is removed from the number, which becomes blank there;
- the code's meaning is kept as text in a side column, `<question>_special` (e.g. `v115_special` = "on premises").

So the number stays a real number and the "don't know" or "on premises" is kept as its own answer. **61 questions** have
such a side column, which is why the 552 questions become **613 model columns**.

### 4. Did the 70% rule throw away anything useful? (Step 8c)

A reviewer asked whether the rule removed useful questions, such as iron tablets in pregnancy or antenatal visits.

- **Those questions were kept all along.** They are 51%, 51% and 27% blank:
  - `m45_1` iron tablets in pregnancy;
  - `m14_1` antenatal visits;
  - `v222` months since last birth.
- **Putting back all 329 mostly-blank questions** (231 survive the repeat rule) changed AUROC by at most
  **+0.005 [+0.000, +0.009]**, and the Hb error by -0.001 g/dL.
- **The returning questions are mostly vaccination dates and pill brand.**

So the rule cost nothing, and we kept it.

### A limit

"Blank" mixes two situations: "not asked, because the question didn't apply to her" (the usual case) and "asked but no
answer". We can't always tell them apart. DHS marks most of the second kind with its own "missing" or "don't know"
codes, and we keep those as their own answers.

## In one line

**4,944 columns → 552 questions → 482 questions (613 model columns), for 9,265 women.** Every removal has a written
reason, and blanks are kept as information rather than thrown away.

## Reproducing

```
python -m src.survey.select_features   # Steps 0-2: locked split, inventory, the 552 questions
python -m src.survey.screen_features   # Steps 3-4: each question alone, then the repeat rule (482 left)
python -m src.survey.blank_sensitivity # Step 8c: rerun with no blank limit
```

Settings are in `configs/feature_selection.yaml`: the drop rules, the 70% limit, and the repeat-rule threshold.
