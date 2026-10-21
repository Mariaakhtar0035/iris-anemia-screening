# Eyelid follow-up to the mentor meeting

This is the working log for the eyelid side of the project after the mentor meeting. Notes from the meeting arrived on 2026-10-20.

- Part 1 lists what the mentors suggested for the eyelid models.
- Part 2 logs each step we ran: why we ran it, what we ran, and what came out.
- The answers for the mentors are in [`reports/mentor_followup_eyelid.md`](../reports/mentor_followup_eyelid.md).

The mentors also gave survey advice: dealing with missing values, wealth scaling across countries, training on two countries and testing on the third, and TabPFN for tables. That belongs to the survey part of the project and is **not** covered here.

## Part 1. What the mentors suggested for the eyelids

Paraphrased from the meeting, without names.

| # | Suggestion | What we do | Status |
|---|---|---|---|
| M1 | Exhaust each dataset before anything else. Use the biggest model you can and see whether it can overfit. If a complex model can't fit data that all comes from one distribution, the signal isn't there. Only then worry about transfer and bias. | Step 1 (Stage 9): model ladder, training vs held-out AUROC, learning curves | planned |
| M2 | Train on each population separately and see how well that does. | Step 1: CP-AnemiC, India and Italy each on their own | planned |
| M3 | Leakage must be prevented. Duplicate photos are fine as augmentation if they never reach the test set and don't hurt transfer. | Step 1: train with the duplicate copies kept inside the training fold, vs without them | planned |
| M4 | When transfer fails, compare the populations to find out why (brightness, color, age, label balance). | Step 2 (Stage 10): population comparison and a "which country is this?" classifier | planned |
| M5 | The country acts as a shortcut. Look at recent debiasing and fairness work: define parity metrics, then regularize the model so its score distribution doesn't depend on the population. | Step 3 (Stage 11): fairness metrics, reweighing, fairness-penalized model | planned |
| M6 | Edit the images so they look more alike, for example by changing the brightness or shifting the color distributions. | Step 3: brightness and color normalization, color-jitter augmentation | planned |
| M7 | Don't drop India from the data. Use fairness metrics instead. | Step 3: per-country and worst-country AUROC | planned |
| M8 | Separate models per population are an acceptable fallback. Big models trained on diverse data can learn every population, as Google's skin model eventually did. | Step 3: per-country models vs one pooled model | planned |
| M9 | More countries make the data more diverse, but each new country brings its own shortcut, so debiasing is still needed. | Step 7: data search, plus the write-up | later |
| M10 | Look for vision models pretrained on eyelid images and reuse their pretraining. | Step 5: search, then ask Jaineel before downloading anything | planned |
| M11 | VLMs don't fit this problem, because the images come with no text or metadata. | Recorded, not pursued | done |
| M12 | Look for fingernail or palm datasets that include metadata (age, sex). | Step 7: data search | later |
| M13 | AUROC alone isn't enough. Use PR-AUC when the classes are imbalanced, and look at precision and recall at specific thresholds. The cost of a false positive vs a false negative has to come from clinicians, who might want recall near 1. Put that discussion in the paper. | Step 4: "share tested to catch X% of anemic people" table and PR curves | planned |
| M14 | The mentors will email a list of models to try. | Add them here when the email arrives | waiting |

## Part 2. Step log

### Step 0: record the suggestions (done)
- **Why:** to keep a record of what was asked, so next week's answers can be checked against it.
- **What:** this file.

### Data problem found on the way: Italian masks and Italian batches (fixed / reported)
While running Step 2 we found two problems in Eyes-Defy Italy. Neither was reported before.

1. **31 mask files had no transparency.** In Italy folders 1-31, the conjunctiva cut-out sits on a pure-white
   (255, 255, 255) background with no alpha channel, while all other masks use a transparent background. Our region of
   interest is "the opaque pixels", so for these 31 photos we were measuring a mostly white frame.
   - **Fix:** `src/eyelid/roi.py` `read_rgba` now makes pure white transparent when a mask has no transparency at all.
     Test: `tests/test_eyelid.py::test_white_background_mask_becomes_transparent`.
   - CP-AnemiC is not affected: every one of its 710 images has transparency.
   - **Effect on earlier results:** we reran Stage 5, the fusion simulation and the Eyes-Defy DINOv2 cache. The
     conclusions held. Redness rule CP-AnemiC -> Eyes-Defy: 0.808 -> 0.835 (Italy 0.935 -> 0.924, India unchanged
     at 0.671). Learned models across datasets: 0.25-0.58 -> 0.32-0.62.
2. **Italy is two batches, and almost all Italian anemia is in the first one.** Folders 1-31 (photos named `N.jpg`) are
   21/31 anemic. Folders 32-123 (photos named `T_*`) are 2/91 anemic.
   - Non-anemic people in the early batch already look paler (a* 32.9 vs 40.1). The batch alone predicts Italian
     anemia with AUROC 0.91, and inside the early batch the redness rule drops to about 0.75 (n = 31).
   - **Meaning:** the "redness rule works great in Italy (0.93)" result is mostly the batch, not pallor. From now on,
     Italy is also reported on the early batch alone. Details: `reports/stage10_populations.md`.

### Data problem 2: near-duplicates still leak in CP-AnemiC's "distinct photos" (avoided)
- **What we saw:** the first Stage 9 run used all 476 distinct CP-AnemiC photos, and gradient boosting scored 0.868.
  But on that subset a random forest that **never sees color** (only image size and cut-out shape) still scores 0.60
  (Stage 3). That means some re-cropped copies our duplicate detector missed sit on both sides of the split.
- **What we did:** Stage 9 now uses the 388 photos with no copy (the no-color control is 0.537 there, near chance),
  the same subset as Stages 3-5. The no-color control is now a row for every population, so any remaining leak shows up.

### Step 1 (Stage 9): how far can each population be fit on its own? (M1, M2, M3)
- **Why:** the mentors said to exhaust each dataset first: can a big model overfit, and is held-out performance still
  rising with more photos?
- **Ran:** `python -m src.eyelid.ceiling` and `python -m src.eyelid.finetune_vit`. Settings in `configs/ceiling.yaml`.
  Report: `reports/stage9_ceiling.md`.
- **Setup:**
  - Populations: CP-AnemiC (388, Ghana children), India (95), Italy (122) and Eyes-Defy pooled (217).
  - Model ladder: redness rule -> logistic regression -> random forest -> gradient boosting -> DINOv2 probe ->
    fine-tuned ViT-B/16.
  - For each model: training AUROC next to held-out AUROC (5x5 CV by photo), plus learning curves at 25-100% of
    each training fold.
- **Results (held-out AUROC [95% CI]):**

| Population | Redness rule | Best color model | DINOv2 probe | No-color control | Can the models overfit? | More photos would help? |
|---|---|---|---|---|---|---|
| CP-AnemiC (Ghana) | 0.650 | **gradient boosting 0.846 [0.807, 0.882]** | 0.672 | 0.537 (clean) | yes, training AUROC 1.00 | **yes**: gradient boosting 0.72 at 78 photos -> 0.85 at 310, random forest 0.71 -> 0.80 |
| India | 0.671 | random forest 0.673 [0.542, 0.791] | 0.658 | 0.548 | yes, training 0.95-1.00 | can't tell: too few non-anemic photos (27) for a curve |
| Italy | 0.924 | logistic regression 0.920 | 0.879 | **0.751** (batch) | yes | mostly batch, see Step 2 |
| Eyes-Defy pooled | 0.835 | random forest 0.892 | 0.868 | **0.742** (country/batch) | yes | **flat**: 0.86-0.89 from 43 photos up, so the country shortcut is learned from a handful of photos |

  - **Mask region (India, within-country):** the forniceal region (the deeper fold of the eyelid) beats the palpebral
    region we used: redness rule 0.755 vs 0.671, logistic regression 0.739 vs 0.612. Gray-world lowers almost
    everything, as in Stage 3.
  - **Duplicate copies as augmentation (M3):** training with all 566 rows instead of one per photo, scored on the
    photos with no copy: random forest 0.824 -> 0.831, logistic regression 0.680 -> 0.662, DINOv2 0.680 -> 0.675. **No
    real gain.** Copies are harmless in training, as long as no copy of a test photo is there, but they don't help.
- **Answer to M1:**
  - On CP-AnemiC there is a real color signal. A flexible model fits it, held-out AUROC reaches 0.85, and it is still
    rising, so more Ghana photos would likely help.
  - In India nothing beats the untrained redness rule (about 0.67): the models memorize the 76 training photos
    without generalizing.
  - Italy and pooled Eyes-Defy can't be judged on their own scores, because even the no-color control gets 0.74-0.75.

### Step 2 (Stage 10): why do the populations differ? (M4)
- **Why:** the mentors said that when transfer fails, compare the populations.
- **Ran:** `python -m src.eyelid.populations`. Settings in `configs/populations.yaml`. Report:
  `reports/stage10_populations.md`.
- **Results:**
  - **The population is obvious from the pixels.** A classifier on the same 29 color numbers tells India from Italy
    at AUROC 0.998, and still 0.999 among non-anemic people only. Ghana children vs Eyes-Defy adults: 0.989 (color),
    1.000 (DINOv2). Anemia within a population is much harder: 0.61-0.68 in Ghana and India.
  - **Between-population color gaps are bigger than anemia gaps.** In a*, anemic minus non-anemic is -3.5 (Ghana)
    and -2.8 (India). Non-anemic Ghana children minus non-anemic Italians is -12.0; non-anemic India minus Italy is
    -2.5. Brightness (L*) differs by 7-17 points between populations.
  - **Who is in each set:**
    - Ghana: children, average age 2-3 years, anemic = Hb < 11.
    - India: average age about 33, 72% anemic.
    - Italy: average age about 49, 19% anemic, and anemic Italians are older (65 vs 46).
  - **Italy batches:** see the data-problem entry above. The batch alone gives AUROC 0.91.
- **Answer to M4:** transfer fails because each dataset has its own camera, lighting and age group, and because
  prevalence differs, so any model can score well by recognizing the dataset or batch instead of pallor.

### Step 3 (Stage 11): debiasing the India/Italy shortcut (M5, M6, M7, M8)
- **Why:** the mentors suggested fairness metrics and regularization, editing images to look more alike, keeping
  India, and per-population models as a fallback.
- **Ran:** `python -m src.eyelid.debias`. Settings in `configs/debias.yaml`. Report: `reports/stage11_debias.md`.
- **Fairness metrics, fixed before running:**
  - AUROC within India and within Italy;
  - Italy early batch alone (the Italy check the batch can't pass);
  - the worse of India and Italy ("worst-country AUROC");
  - the rank gap between countries at the same anemia status;
  - the TPR and FPR gaps at the threshold that catches 90% of anemic people.
- **Methods tried:**
  - gray-world;
  - brightness scaling;
  - per-country standardization;
  - brightness/contrast jitter;
  - reweighing (Kamiran and Calders);
  - a fairness-penalized logistic regression with lambda 1, 10 and 100;
  - one model per country.
- **Results:**

| Method | Pooled AUROC | India | Italy early batch | Rank gap, non-anemic | FPR gap | Share tested for 90% sensitivity |
|---|---|---|---|---|---|---|
| Redness rule | 0.835 | 0.671 | 0.752 | 0.19 | 0.39 | 0.63 |
| Random forest (no debiasing) | 0.890 | 0.687 | 0.752 | 0.39 | 0.75 | 0.53 |
| Random forest, brightness scaled | 0.897 | **0.739** | 0.771 | 0.37 | 0.74 | 0.54 |
| Fairness penalty, lambda=100 | 0.756 | 0.675 | 0.667 | **0.10** | **0.11** | 0.76 |
| Per-country standardized, logistic regression | 0.720 | 0.621 | 0.638 | **0.05** | **0.04** | 0.86 |
| One model per country, random forest | 0.883 | 0.676 | 0.710 | 0.41 | 0.91 | 0.54 |

- **Answer to M5-M8:**
  - **The fairness methods do what they claim.** The penalty and per-country standardization shrink the country gaps
    from about 0.4 to about 0.05-0.10 in rank, and the FPR gap from 0.75 to about 0.1.
  - **But they don't make the model better inside a country.** India stays at 0.57-0.74 and the Italy early batch at
    0.52-0.77 (CIs overlap everywhere), while pooled AUROC drops and more people have to be tested. Taking the shortcut
    away doesn't add pallor signal that isn't there.
  - **Brightness scaling** gives the best within-country numbers (India 0.739) at no pooled cost, but the gain is
    inside the CI.
  - **Separate per-country models (M8)** are no better than one pooled model within each country.
  - **Transfer to CP-AnemiC:** no debiasing method beats the untrained redness rule (0.650). The best trained model is
    the random forest with jitter, 0.606.
