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
