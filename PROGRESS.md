# Progress: every strategy we tried, why, and what happened

*Last updated 2026-10-24, branch `jaineel`. Written to present from: plain English, one block per strategy, each with
why we tried it, what happened, and whether it worked. The full paper-style write-up is the [README](README.md); the
answers for the mentors are in [`reports/mentor_followup_eyelid.md`](reports/mentor_followup_eyelid.md).*

## Where we are

We checked the two published eyelid-photo results, built our own honest baselines, tested whether anything works on a
new dataset or country, built a survey model and tested it in two new countries, simulated combining the two, and
then followed up every piece of eyelid advice from the mentor meeting. All of that is done and pushed.

Since then: fingernails under fixed lighting (F13, F15), the eye-pretrained model VisionFM, frozen and fine-tuned
(F11), a proper check of the deeper eyelid fold (F2), TabPFN, the model the mentors named (F14), and palm and
fingernail photos of Ghanaian children (F16). See
[What's next](#whats-next) for what's left.

## The story in six points

1. **Published eyelid scores are inflated.** Both papers test on photos that are also in training (copies, or flipped
   versions). A clean split costs 7-8 accuracy points.
2. **On Eyes-Defy the model mostly learns the country.** Indian photos are mostly anemic, Italian ones mostly not.
   Inside India the published model is a coin flip.
3. **Color numbers beat big image models.** 29 color numbers beat every image model we tried, even one pretrained
   on eye photos (0.85 vs 0.79). The best so far is TabPFN, a model built for small tables, on those same 29 numbers:
   **0.92** in Ghana children, and it holds with whole hospitals left out (0.91). The untrained "less red = anemic" rule is still the
   only thing that keeps working on a new dataset.
4. **Survey answers are a weak but portable signal.** AUROC 0.58 in Zimbabwe, and it holds unchanged in Malawi and
   Zambia.
5. **After the mentors: the limit is the data, not the model.** A big ViT memorizes the training photos, debiasing
   makes scores fair across countries but not more accurate, and in a clinic the best model saves about a quarter of
   blood tests but can't rule anyone out.
6. **Under fixed lighting the color signal is clean.** On fingernails photographed in one light box, a simple model
   reaches 0.82 with no shortcut, the same on photo sessions it never saw.

## How to read the numbers

- **AUROC**: the chance a random anemic person gets a higher score than a random non-anemic one. 0.5 = coin flip,
  1.0 = perfect. Roughly: 0.6-0.7 weak, 0.8+ good.
- **[a, b]** after a number is its 95% range (bootstrap, 2000 resamples). If two ranges overlap a lot, we can't say
  one is better.
- **No-color control**: a model that sees only image size and cut-out shape, so it can't see pallor. If it scores
  above 0.5, the photos carry a shortcut (copies, batches, cameras).
- **Share tested to catch 90%**: rank people by score, test from the top, stop when 90% of anemic people are found.
  Testing at random needs 90%; lower is better.
- **Anemia cutoffs**:
  - Ghana children (CP-AnemiC): Hb < 11.0 g/dL.
  - Eyes-Defy adults (Italy, India): WHO by sex, women < 12.0 and men < 13.0. Exception: Part A uses the ViT paper's
    own cutoffs, to copy the paper.
  - DHS women: the survey's own `v457` label (WHO for women, adjusted for altitude and smoking).

## Scoreboard

| # | Strategy | Why | Result | Verdict |
|---|---|---|---|---|
| A1 | Reproduce the CP-AnemiC ResNet50 | our proposal builds on it | 85.1% vs published 84.79% | **worked** |
| A2 | Reproduce the Eyes-Defy ViT | same | 95.6% vs published 98.47% | **close, not exact** |
| A3 | Rerun both with no photo on both sides | the audit found copies and flips across the split | 77.5% and 88.5% | **found the inflation** |
| A4 | Score the ViT inside each country | India is mostly anemic, Italy mostly not | India AUROC 0.53 | **found the country shortcut** |
| B1 | Duplicate audit of CP-AnemiC | rows looked repeated | 710 rows = 477 photos | **found leakage** |
| B2 | No-color control | detect leaks we can't see | 0.83 with copies, 0.54 on clean photos | **worked** |
| B3 | Untrained "less red = anemic" rule | simplest baseline | 0.650 | baseline |
| B4 | Color numbers + random forest | does pallor color carry signal? | 0.796 | **worked** |
| B5 | Frozen DINOv2 probe | does a big image model do better? | 0.672 | **didn't work** |
| B6 | Gray-world color correction | remove lighting differences | -0.06 to -0.08 | **didn't work** |
| C1 | Train on one dataset, test on the other | does a model travel? | 0.32-0.62 | **didn't work** |
| C2 | The rule on the other dataset | needs no training, so nothing to overfit | 0.835 on Eyes-Defy | **worked best**, partly a batch effect |
| D1 | Survey model in Zimbabwe | cheap screen with no photo | 0.580 | **weak but real** |
| D2 | Same model, unchanged, in Malawi and Zambia | does it travel? | 0.568, 0.565 | **worked** |
| D3 | Recalibrate on 200 local women | fix the risk levels | average fixed, Brier not | **mixed** |
| E1 | Simulate survey + eyelid | no paired data exists | survey adds 0.01-0.02 | **informative only** |
| F1 | Push each eyelid dataset to its limit, up to a fine-tuned ViT (M1, M2) | the mentors' "exhaust the data first" | Ghana 0.846 with color numbers, ViT 0.715 | **ViT didn't help; diagnosis worked** |
| F2 | Deeper eyelid fold (forniceal mask) | a different region may show pallor better | India rule 0.671 -> 0.755, paired change +0.08 | **real but borderline lead** |
| F3 | Duplicate copies as extra training data (M3) | mentors said that's fine if kept out of test | 0.824 -> 0.831 | **harmless, no gain** |
| F4 | Compare the populations (M4) | find why transfer fails | dataset obvious from pixels (0.99+) | **worked: explains it** |
| F5 | Make images look alike (M6) | remove camera and lighting gaps | brightness scaling best, India 0.739, within the range | **mixed** |
| F6 | Fairness methods: reweighing, fairness penalty (M5) | stop scores depending on country | gaps shrink 0.39 -> 0.10, accuracy doesn't rise | **fair, not more accurate** |
| F7 | Keep India, use fairness metrics (M7) | don't hide the hard case | kept in every run | done |
| F8 | One model per country (M8) | fallback the mentors offered | no better, bigger gaps | **didn't work** |
| F9 | Debiased models on the other dataset | does fairness help transfer? | 0.606 vs rule 0.650 | **didn't work** |
| F10 | Clinic view: share tested, precision, PR-AUC (M13) | AUROC alone doesn't say what a clinic gains | 64% tested to catch 90% (TabPFN: 50%) | **worked: clearer story** |
| F11 | Eye-pretrained model, VisionFM (M10) | start from a model that has seen eyes | frozen 0.765, fine-tuned 0.785 in Ghana children: beats DINOv2 and our ViT, loses to color numbers (0.846) | **helps vs general models, not enough** |
| F12 | Vision-language models (M11) | mentors said no | not pursued | skipped |
| F13 | More data: fingernails under fixed lighting (M9, M12) | a new country and body site, without camera and batch differences | logistic regression 0.818, no shortcut, flat learning curve | **worked** |
| F14 | TabPFN, the tabular model the mentors named (M14) | a model pretrained for small tables, on our 29 color numbers | Ghana 0.919 vs 0.846 (0.913 with hospitals held out); India 0.648; nails 0.823 | **best in Ghana, passes the shortcut checks; no help elsewhere** |
| F16 | Palm and fingernail photos of Ghana children (M9, M12) | other body sites, same country and age group as the eyelids | palm 0.912, nails 0.798 (TabPFN); no shortcut in palm | **palm as good as eyelid; nails weaker** |
| F15 | Correct the nail colors: white board, or the skin next to the nail | cancel lighting and skin tone (M6) | white board: rule +0.025, models +0.00-0.03 (within noise); minus skin: -0.11 to -0.31 | **white board: small gain; minus skin: hurt** |
| F17 | One model for Ghana, India and Italy, with shortcuts taken out (M5, M8, M9) | can mixed training learn every country without guessing the country? | mixed TabPFN Ghana 0.90, India 0.64, Italy 0.91 (own-country models 0.92, 0.66, 0.92); standardizing per country halves the country gap (0.52 -> 0.27); train on two, test on the third: no better than the rule | **fair and as good as separate models; no new signal** |
| F18 | A few labelled local photos for a new country (M8, M9) | how many photos must a new clinic label before a model beats the rule? | 10-40 photos: no model beats the rule anywhere; Ghana needs ~80 (0.92 at 310); India and Italy never beat it, even with all their photos; the other two countries only keep small-k models at the rule's level | **a few photos aren't enough** |

## Part A: checking the published eyelid results

### A1. Reproduce the CP-AnemiC ResNet50 (paper A). Worked.
- **Why:** our proposal starts from this result (84.79% accuracy on Ghana children), so first we check that it's real.
- **What we did:** the paper's model, loss, optimizer, learning rate, batch size, augmentation and 5-fold split. Two
  changes: ImageNet starting weights and 30 epochs instead of 150 from scratch.
- **What happened:** 85.1% accuracy (published 84.79%), Hb error 1.46 g/dL (published 1.54).

### A2. Reproduce the Eyes-Defy ViT (paper B). Close, not exact.
- **Why:** the 98.47% headline is the strongest claim we build on.
- **What we did:** ViT-B/16, the paper's optimizer and settings, each photo flipped twice, then a random 80/20 split,
  5 seeds.
- **What happened:** 95.6% (range over splits 93.9-97.0%). Picking the best epoch on the test set adds about 1 point
  (best run 96.9%), still short of 98.47%. With 131 test images, 98.47% means 2 errors; our best run makes 4.

### A3. Rerun both with no photo on both sides. Found the inflation.
- **Why:** CP-AnemiC repeats photos, and the ViT paper flips each photo *before* splitting, so a photo's twin can be
  in training while the photo is in test.
- **What we did:** one change only: no photo, copy or flipped twin on both sides of the split.
- **What happened:**
  - CP-AnemiC: 85.1% -> **77.5%** (AUROC 0.921 -> 0.852).
  - Eyes-Defy: 95.6% -> **88.5%** (AUROC 0.991 -> 0.932).
- **Takeaway:** 7-8 points of each headline come from testing on photos the model already saw.

### A4. Score the Eyes-Defy ViT inside each country. Found the country shortcut.
- **Why:** under the paper's cutoffs 83% of Indian photos are anemic and 10% of Italian ones, so the rule "Indian
  photo = anemic" already scores 88.6%, the same as the clean ViT (88.5%).
- **What happened:** inside India the ViT's AUROC is **0.53 [0.29, 0.80]**, a coin flip. Italy is 0.93, but on only
  7 anemic test photos (and Part F shows Italy has its own batch problem).
- **Also checked:** test accuracy after every epoch. With a clean split it sits near 88% from the first epoch; only
  the leaky split keeps climbing.

## Part B: our own baselines on Ghana children (CP-AnemiC)

### B1. Duplicate audit. Found leakage.
- **Why:** many rows looked identical.
- **What happened:** the 710 rows hold only **477 distinct photos**. 212 rows are byte-identical copies under another
  ID, and 29 are re-saved or re-cropped. The copies often carry a *different* Hb, age, sex and hospital (one photo is
  recorded with Hb 10.2 and 5.8). Only **388 photos have no copy**; all our own CP-AnemiC results use those.

### B2. No-color control. Worked, now in every table.
- **Why:** to catch leaks our duplicate detector misses. A model that can't see color can only score well by
  recognizing photos.
- **What happened:**
  - With copies leaking across the split: **0.83**, so it was recognizing photos.
  - Split by photo group: still 0.68, so some re-cropped copies escape the detector.
  - On the 388 photos with no copy: **0.537**, chance. That's why we use the 388.

### B3-B5. Rule vs color numbers vs a big pretrained model.
- **Why:** to see whether pallor color carries signal, and whether a large image model sees more than simple numbers.
- **What we did:** 29 color numbers from the cut-out conjunctiva (mean, median and spread of the RGB, HSV and Lab
  channels), then logistic regression or random forest; and a linear probe on frozen DINOv2 image features.
- **What happened** (5x5 cross-validation, 388 photos):

| Model | AUROC | Leave one hospital out |
|---|---|---|
| Dummy | 0.493 | 0.381 |
| Rule: less red = anemic (B3) | 0.650 [0.597, 0.705] | 0.650 |
| Color logistic regression | 0.680 [0.627, 0.730] | 0.672 |
| **Color random forest (B4)** | **0.796 [0.752, 0.839]** | **0.786** |
| DINOv2 probe (B5) | 0.672 [0.616, 0.723] | 0.652 |

- **Takeaway:** color numbers work, and still work on a hospital the model never saw. The big pretrained model does
  worse than 29 numbers.

### B6. Gray-world color correction. Didn't work.
- **Why:** to remove lighting and camera color casts.
- **What happened:** every model dropped by 0.06-0.08. It removes the redness, which *is* the signal.

## Part C: does an eyelid model work on another dataset?

### C1. Train on one dataset, test on the other. Didn't work.
- **Why:** a screening tool has to work on people it wasn't built on.
- **What we did:** the same input on both sides (the hand-segmented conjunctiva). Fit on all of one dataset, apply
  unchanged to the other.
- **What happened:** learned models fall to **0.32-0.62**. The color logistic regression scores 0.315 on Eyes-Defy,
  *worse than chance*: the color patterns it learned are tied to one camera and reverse under the other.

### C2. The untrained redness rule on the other dataset. Worked best.
- **Why:** it has nothing to overfit.
- **What happened:** 0.835 on all of Eyes-Defy (better than "Indian photo = anemic", 0.766), but only **0.671 inside
  India**, and Italy's 0.924 is mostly the photo batch (F4).
- **Takeaway:** "less red = anemic" carries over; the learned details of it don't.

## Part D: the survey model (DHS; done earlier, the survey side belongs to teammates)

### D1. Survey answers in Zimbabwe. Weak but real.
- **Why:** a screen that needs no photo, only routine questions (age, pregnancy, wealth, education, contraception,
  urban/rural).
- **What happened** (9,265 women, cross-validated by village/neighborhood cluster):
  - Logistic regression **0.580 [0.566, 0.593]**; testing the top 30% finds 38% of anemic women vs 29% at random.
  - "Test pregnant women first": 0.504. Pregnant women are more often anemic, but only 6% of women.
  - Gradient boosting 0.570, and overconfident. Survey weights changed nothing.

### D2. Same model, unchanged, in Malawi and Zambia. Worked.
- **Why:** does a model built in one country work in another, 9 years later?
- **What happened:** 0.568 and 0.565, calibration close. It overestimates Malawi's anemia a little (29.0% predicted
  vs 26.1%).

### D3. Recalibrate on 200 local women. Mixed.
- **What happened:** fixes the average (26.2%), but with such a weak signal the fit is noisy and the Brier score
  doesn't improve.

## Part E: combining survey and eyelid (simulation only)

### E1. Simulated fusion. Informative, not proof.
- **Why:** no dataset has a photo and survey answers for the same woman.
- **What we did:** each Malawi and Zambia woman keeps her real survey score and gets an eyelid score drawn from
  Eyes-Defy photos with her anemia status. Weak scenario: Indian photos; strong: Italian photos.
- **What happened (Malawi):** survey alone 0.568; weak eyelid alone 0.674, plus survey 0.687; strong eyelid 0.927,
  plus survey 0.929.
- **Takeaway:** the eyelid score would do most of the work; the survey adds 0.01-0.02. The strong scenario borrows
  Italy's batch effect, so it's an upper bound. This argues for collecting paired data, not that fusion works.

## Part F: following up the mentors' eyelid advice (M1-M14)

The suggestions are recorded as the mentors gave them in [`docs/eyelid_followup.md`](docs/eyelid_followup.md), with a
log of every step.

### F1. Push each dataset to its limit (M1, M2). The ViT didn't help; the diagnosis worked.
- **Why:** the mentors said to exhaust each dataset first: if a big model can't even overfit one population, the
  signal isn't there. Only then worry about transfer.
- **What we did:** a ladder from the untrained rule up to a fine-tuned ViT-B/16 (about 2 hours on the M1), trained on
  each population alone. Training vs held-out AUROC, plus learning curves at 25-100% of the training data.
- **What happened:**

| Population | Rule | Best color model | DINOv2 | Fine-tuned ViT | VisionFM (F11) | TabPFN (F14) | No-color control |
|---|---|---|---|---|---|---|---|
| Ghana children (388) | 0.650 | **gradient boosting 0.846 [0.807, 0.882]** | 0.672 | 0.715 [0.661, 0.765] | 0.765 [0.716, 0.815] | **0.919 [0.893, 0.945]** | 0.537 |
| India (95) | **0.671** | random forest 0.673 | 0.658 | 0.654 | 0.585 | 0.648 | 0.548 |
| Italy (122) | 0.924 | logistic regression 0.920 | 0.879 | 0.919 | 0.927 | 0.909 | **0.751** |

- **Takeaways:**
  - **The models are big enough.** Gradient boosting and the ViT fit every training set perfectly (training AUROC
    1.00).
  - **Ghana has a real signal that more photos would improve:** gradient boosting goes 0.72 -> 0.78 -> 0.83 -> 0.85
    with 78, 155, 233 and 310 training photos, still rising.
  - **The ViT did worse than 29 color numbers** (0.715 vs 0.846), so a bigger model isn't what's missing. VisionFM,
    pretrained on eye photos, closes part of the gap (0.765) but not all of it (F11). TabPFN on the same color
    numbers goes further (0.919, F14).
  - **India** is stuck at the rule's level (0.65-0.67): 76 training photos get memorized.
  - **Italy's** scores can't be trusted: the no-color control alone scores 0.75 (see F4).

### F2. The deeper eyelid fold (forniceal mask). Real but borderline lead.
- **Question:** does the deeper fold of the eyelid show pallor better than the front part we used?
- **Answer:** in India, yes for the simple models, but only just. **Rule 0.671 -> 0.755.**
- **Why:** Eyes-Defy ships two masks; we had used the front part (palpebral).
- **How we checked (2026-10-24):** a paired test: both regions scored on the same resampled people, so the range is
  on the change itself.

| India (95 photos) | Front part | Deeper fold | Change [95% range] |
|---|---|---|---|
| Rule | 0.671 | **0.755** | +0.084 [+0.001, +0.175] |
| Logistic regression | 0.612 | **0.739** | +0.127 [+0.029, +0.228] |
| Random forest | 0.673 | 0.639 | -0.034, no change |
| Both regions together, rule | 0.671 | 0.746 | +0.075 [+0.022, +0.138] |

- **Italy:** no change.
- **Worked?** Real for the rule and logistic regression, but the lower end of the range is near 0, there are only
  95 photos and we ran many comparisons. A lead to confirm on new Indian photos, not a result.

### F3. Duplicate copies as extra training data (M3). Harmless, no gain.
- **Why:** the mentors said copies are fine as augmentation if no copy of a test photo is in training.
- **What happened:** random forest 0.824 -> 0.831, logistic regression 0.680 -> 0.662, DINOv2 0.680 -> 0.675. All
  within the ranges.

### F4. Compare the populations (M4). Worked: explains why transfer fails.
- **Why:** the mentors said to look at brightness, color, age and label balance when transfer fails.
- **What happened:**
  - **The dataset is obvious from the pixels.** The same color numbers tell India from Italy at AUROC 0.998 (0.999
    among healthy people only), and Ghana from Eyes-Defy at 0.989 (DINOv2: 1.000). Anemia inside a population is only
    0.61-0.68.
  - **Population gaps are bigger than anemia gaps.** Redness (a*) drops 3.5 with anemia in Ghana and 2.8 in India,
    but healthy Ghana children are 12.0 below healthy Italians. Brightness differs by 7-17 points.
  - **The people differ:** Ghana children aged 2-3; India adults around 33, 72% anemic; Italy adults around 49, 19%
    anemic, and the anemic Italians are older (65 vs 46).
  - **Italy is two photo batches.** Folders 1-31: 21 of 31 anemic. Folders 32-123: 2 of 91. Healthy people in the
    early batch already look paler (a* 32.9 vs 40.1). The batch alone predicts Italian anemia at 0.91; inside the
    early batch the rule drops to **0.752 [0.517, 0.937]**.

### F5. Make the images look alike (M6). Mixed.
- **Why:** if camera and lighting differences are the shortcut, editing them out should force the model to look at
  pallor.
- **What we did** (pooled Eyes-Defy, random forest unless noted):

| Edit | Pooled | Inside India | Italy early batch | Verdict |
|---|---|---|---|---|
| None | 0.890 | 0.687 [0.565, 0.799] | 0.752 | baseline |
| Gray-world | 0.831 | 0.572 | 0.571 | **hurt**: removes the redness |
| Brightness scaling | 0.897 | **0.739** [0.631, 0.838] | 0.771 | **best**, but inside the range |
| Per-country standardization (logistic regression) | 0.720 | 0.621 | 0.638 | closes the gaps, costs accuracy |
| Brightness/contrast jitter in training | 0.881 | 0.669 | 0.762 | no change |

### F6. Fairness methods (M5). Fair, but not more accurate.
- **Why:** the mentors pointed to the debiasing literature: define parity metrics, then regularize the model so its
  scores don't depend on the population.
- **What we did:** fixed the fairness metrics *before* running: AUROC inside each country and in the Italy early
  batch; the rank gap between countries among people with the same anemia status; the true- and false-positive rate
  gaps at the threshold that catches 90%. Then:
  - **reweighing** (Kamiran and Calders), so country and anemia are independent in training;
  - **a fairness penalty** on a logistic regression, punishing score gaps between countries (lambda 1, 10, 100).
- **What happened:**

| Method | Pooled | Inside India | Rank gap, healthy | FPR gap | Share tested to catch 90% |
|---|---|---|---|---|---|
| Random forest, no fix | 0.890 | 0.687 | 0.39 | 0.75 | 0.53 |
| Reweighing (random forest) | 0.860 | 0.724 | 0.24 | 0.48 | 0.55 |
| Fairness penalty, lambda=100 | 0.756 | 0.675 | **0.10** | **0.11** | 0.76 |

- **Takeaway:** the methods do what they claim and make the scores fair between countries. But no method makes the
  model better *inside* a country beyond its range, and the fairest one sends more people for testing. Removing a
  shortcut can't add pallor signal that small datasets don't contain.

### F7. Keep India (M7). Done.
India stays in every run, and is reported with the fairness metrics instead of being dropped.

### F8. One model per country (M8). Didn't work.
- **What happened:** inside India 0.676 and Italy early batch 0.710, vs 0.687 and 0.752 for one pooled model. And the
  gaps get bigger (FPR gap 0.91).

### F9. Debiased models on the other dataset. Didn't work.
- **What happened:** trained on Eyes-Defy and tested on Ghana children, the best debiased model scores 0.606 and the
  untrained rule 0.650. India -> Italy early batch: rule 0.752, random forest with jitter 0.781, within the range.

### F10. The clinic view (M13). Worked: a clearer story than AUROC.
- **Why:** the mentors said AUROC alone doesn't tell a clinic anything; use precision, recall and PR-AUC, and ask
  clinicians what miss rate is acceptable.
- **What we did:** in each population, the share of people that must be tested to catch 80, 90, 95 or 100% of the
  anemic ones, and the share of those tested who are actually anemic.
- **What happened:**

| Population (share anemic) | Model | Tested to catch 90% | Anemic among those tested | Tested to catch 100% |
|---|---|---|---|---|
| Ghana children (42.5%) | rule | 0.807 | 0.476 | 1.000 |
| | gradient boosting | 0.644 [0.552, 0.711] | 0.596 | 0.979 |
| | **TabPFN (F14)** | **0.500 [0.443, 0.570]** | **0.768** | 0.827 |
| | fine-tuned ViT | 0.776 | 0.495 | 0.979 |
| India (71.6%) | random forest | 0.832 | 0.797 | 1.000 |
| Italy early batch (67.7%) | rule | 0.871 | 0.778 | 0.871 |

- **Takeaway:** in Ghana children gradient boosting saves about a quarter of blood tests at 90% sensitivity, and
  TabPFN about 40% (test half the children instead of 90%). To catch
  everyone, nearly everyone gets tested. An eyelid photo can decide **who to test first**; it can't rule anyone out.

### F11. A model pretrained on eye photos (M10). Helped a little, not enough.
- **Why:** starting from a model that has seen eyes might need fewer of our photos.
- **What we found:**
  - The one Hugging Face anemia model (`dawahealth/anemia_screeniing`) was trained on images from Ghana, Italy and
    India, almost certainly our own datasets, so it has seen our test photos. Rejected.
  - **VisionFM's external-eye encoder** (`github.com/ABILab-CUHK/VisionFM`) is the only real candidate. Research use
    only; one 1.4 GB weights file.
- **What we did:** Jaineel downloaded it on 2026-10-23. The file is a pickle, so we loaded it in PyTorch's safe
  tensors-only mode (never the unsafe loader), and all 85.8M weights matched a standard ViT-B/16. Frozen features
  with the same probe as DINOv2, as one more rung of the F1 ladder; inputs normalized with VisionFM's own eye-photo
  statistics.
- **What happened:**

| Population | Rule | Best color model | DINOv2 | Fine-tuned ViT | **VisionFM** |
|---|---|---|---|---|---|
| Ghana children (388) | 0.650 | **gradient boosting 0.846** [0.807, 0.882] | 0.672 | 0.715 | **0.765 [0.716, 0.815]** |
| India (95) | **0.671** | random forest 0.673 | 0.658 | 0.654 | 0.585 [0.452, 0.722] |
| Italy (122) | 0.924 | logistic regression 0.920 | 0.879 | 0.919 | 0.927 |
| Italy + India pooled (217) | 0.835 | random forest 0.892 | 0.868 | 0.892 | 0.872 |

- **Takeaways:**
  - **Eye pretraining helps compared with general image models:** in Ghana children VisionFM beats DINOv2 (+0.09) and
    the ViT we fine-tuned ourselves (+0.05), with no fine-tuning at all.
  - **But 29 color numbers still win** (0.846 vs 0.765), and VisionFM improves more slowly with more photos (0.719
    with 78 training photos, 0.765 with 310).
  - **India doesn't improve** (0.585, below the untrained rule's 0.671). Italy and pooled sit where the batch and
    country shortcuts put every model.
  - **Clinic view:** to catch 90% of anemic Ghana children, VisionFM tests 74% of children, vs 64% for gradient
    boosting, 78% for the fine-tuned ViT and 81% for the rule.
  - **Fine-tuned (2026-10-24):** we also trained the whole VisionFM model on Ghana photos, with the same settings as
    our ViT (30 epochs, 5 folds, 1 seed). It reaches **0.785 [0.739, 0.829]**: a bit above frozen (0.765) and our
    ViT (0.715), still below color numbers (0.846). It memorizes the training photos (training AUROC 1.00). Clinic
    view: tests 73% to catch 90%, vs 78% for our ViT.
  - **Answer to the mentors:** a better starting model narrows the gap but doesn't close it. The limit is the data,
    not the model.

### F12. Vision-language models (M11). Skipped.
The mentors said VLMs don't fit: the photos come with no text or metadata. We agree.

### F13. More data: fingernails under fixed lighting (M9, M12). Worked.
- **Why:** more countries make the data more diverse (each bringing its own shortcut to test for), and the mentors
  suggested fingernail or palm sets with age and sex.
- **What we found:**

| Dataset | Body site | People | Access | Status |
|---|---|---|---|---|
| Yakimov et al. 2024, *Scientific Data* | fingernails and skin, fixed LED lighting box | 250 adults, Moscow; 69 below Hb 12 | open, CC0 on figshare (92 MB) | **done: Stage 12** |
| Girija et al. 2026, IEEE DataPort | conjunctiva, palms, nail beds | 1,485 blood-camp volunteers, India | paid IEEE DataPort subscription | **skipped**: no access |
| Asare et al. 2022, Mendeley | palms; fingernails | 514 and 526 distinct photos of children, Ghana | open, CC BY 4.0 | **done: Stage 13 (F16)**; labels only, no link to the eyelid photos |
| Suner et al. 2021, Zenodo | conjunctiva | 344 adults, USA | open, but no photos | tables only |
| Valles-Coral et al. 2025 | nails, palms | 909 young adults, Peru | not released | watch |

- **Why the fingernail set first:** open, a new country and body site, and photographed under fixed lighting. That
  makes it a direct test of our main suspicion: is the eyelid signal weak because the photos are uncontrolled?
- **What we did (Stage 12):** anemic = Hb < 12.0 g/dL for everyone (no sex column, so Jaineel picked one adult
  cutoff; 69 of 250 anemic). The same 29 color numbers from the three nail boxes in each photo; the rule, logistic
  regression, random forest, gradient boosting, a dummy, a no-color control (nail box sizes only), and a model that
  sees only the photo session (20 sessions) to catch a batch shortcut like Italy's. Scored with 5x5 CV by patient and
  on whole held-out sessions.
- **What happened:**

| Model | AUROC, CV by patient | AUROC, held-out sessions | Share tested to catch 90% |
|---|---|---|---|
| Rule: less red = anemic | 0.703 [0.632, 0.770] | 0.703 | 0.732 |
| **Logistic regression** | **0.818 [0.755, 0.872]** | **0.815** | 0.608 |
| Random forest | 0.809 [0.746, 0.865] | 0.802 | **0.576** |
| Gradient boosting | 0.791 [0.730, 0.848] | 0.795 | 0.612 |
| No-color control | 0.492 | 0.491 | 0.948 |
| Session only | 0.438 | - | 0.968 |

- **Takeaways:**
  - **A real color signal and no shortcut:** the no-color control and the session-only model sit at chance, and the
    scores hold on sessions the model never saw. That's the clean result the eyelid data never gave us.
  - **Simple is enough:** logistic regression wins; gradient boosting memorizes (training 1.00) and scores lower.
  - **Flat learning curve:** 0.811 with 50 training photos, 0.818 with 200. With fixed lighting these color numbers
    hit their limit fast, so more of the same photos won't help much.
  - **The untrained rule does better than on eyelids** (0.70 vs 0.65 in Ghana and 0.67 in India), consistent with
    controlled lighting helping the raw color.
  - **Clinic view:** catching 90% of anemic people means testing 58-61% of everyone, vs 90% at random.
- **Caveats:** one cutoff for both sexes (some anemic men count as healthy), mostly light skin, one hospital. A nail
  model can't be applied to eyelids, so this compares conditions, not transfer. Report:
  [`reports/stage12_nails.md`](reports/stage12_nails.md).

### F14. TabPFN, the tabular model the mentors named (M14). Best in Ghana; no help elsewhere.
- **Question:** does a "foundation model for tables" get more out of our 29 color numbers?
- **Answer:** in Ghana children, yes, a lot. In India and on fingernails, no. **Ghana 0.919 vs 0.846.**
- **Why:** the mentors' email with a model list never came, so we used the models named in the meeting. The eye
  models (VisionFM) and the biggest model (ViT) were done in F1 and F11. The one left was TabPFN: a model pretrained
  on millions of made-up small tables, which classifies a new table in one pass with no tuning.
- **What we did:** TabPFN version 2 (open weights; the newer versions need a license sign-up), on the same 29 color
  numbers, same splits, as one more rung of the F1 ladder. Weights loaded in PyTorch's safe tensors-only mode.
- **What happened:**

| Data | Best before | TabPFN |
|---|---|---|
| Ghana children (388) | gradient boosting 0.846 | **0.919 [0.893, 0.945]** |
| India (95) | rule 0.671 | 0.648 |
| Italy (122) | rule 0.924 | 0.909 |
| Fingernails (250) | logistic regression 0.818 | 0.823, same on held-out sessions |

- **Takeaways:**
  - **Ghana is the best eyelid result so far**, and still rising with more photos (0.72 -> 0.80 -> 0.90 -> 0.92 with
    78, 155, 233 and 310 training photos).
  - **Clinic view:** to catch 90% of anemic children, test 50% of children, vs 64% for gradient boosting.
  - **India doesn't move** (0.648, at the rule's level): too few photos for any model.
  - **Fingernails: same as logistic regression.** Under fixed lighting a simple model is already at the limit.
- **Shortcut checks (run 2026-10-24, Jaineel's go-ahead):**

| Check | Result | What it means |
|---|---|---|
| Train on 9 hospitals, score the 10th | **0.913** (vs 0.919 normal) | doesn't lean on the hospital |
| Same check, gradient boosting | 0.819 (vs 0.846) | TabPFN's lead holds |
| TabPFN on size and shape only, no color | 0.565 [0.51, 0.62] | a tiny shortcut, far below 0.92 |
| Knowing only the hospital | 0.549 | the hospital alone says little |
| Palm / nails, size and shape only | 0.438 / 0.572 | palm none; nails the same small one as before |

  - **Verdict: the 0.92 is real color signal.** It is our headline eyelid number.
  - Report: [`reports/tabpfn_checks.md`](reports/tabpfn_checks.md).
- **Across datasets (run 2026-10-24, Jaineel's go-ahead): it doesn't hold.** Trained on all Ghana eyelids and
  applied unchanged to the Eyes-Defy adults, and the reverse (Stage 5 setup):

| Train -> test | TabPFN | Untrained rule |
|---|---|---|
| Ghana -> Eyes-Defy (217) | 0.602 [0.53, 0.68] | **0.835** |
| Ghana -> India / Italy / Italy early batch | 0.658 / 0.651 / 0.629 | 0.671 / **0.924** / 0.752 |
| Eyes-Defy -> Ghana (388) | 0.595 [0.54, 0.65] | 0.650 |

  - **Verdict:** 0.92 within Ghana drops to about 0.60 on a new dataset and never beats the rule. The model learns
    Ghana's photos, not pallor in general; only the rule carries over. Report:
    [`reports/tabpfn_transfer.md`](reports/tabpfn_transfer.md).

### F15. Correct the nail colors for lighting and skin tone (Stage 12b). White board: small gain. Minus skin: hurt.
- **Why:** each nail photo has a white board behind the hand, and the data marks a patch of skin just behind each
  nail. That lets us try the mentors' "make the images alike" idea (M6) properly: cancel the lighting with a real white
  reference, or cancel skin tone (the bias they warned about) by comparing the nail with the person's own skin.
- **What we did:** three new inputs for the same rule and models, same splits as F13:
  - **white-balanced:** each photo divided by its own white board (the median of its bright, colorless pixels), so
    the board reads pure white;
  - **nail minus skin:** the nail's 29 color numbers minus those of the skin next to it;
  - **skin only:** the skin patches alone, to see how much signal the skin carries.
  - Each compared with the nails as photographed using a paired bootstrap (both scored on the same resampled
    people), so the CI is on the change itself.
- **What happened** (AUROC, CV by patient; the change vs as photographed in brackets when its CI excludes 0):

| Input | Rule | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|---|
| Nails as photographed (F13) | 0.703 | 0.818 | 0.809 | 0.791 |
| **White-balanced** | **0.727** (+0.025 [0.002, 0.049]) | 0.820 | **0.825** | 0.814 |
| Nail minus skin | 0.413 (-0.290) | 0.705 (-0.113) | 0.509 (-0.301) | 0.485 (-0.306) |
| Skin only | 0.730 | 0.763 | 0.728 (-0.081) | 0.732 |

- **Takeaways:**
  - **The white board helps a little.** The untrained rule gains 0.025 and that gain is real (CI above 0). The models
    gain 0.00-0.03, within noise; on held-out photo sessions random forest reaches 0.834 (+0.032 [-0.005, 0.068]).
    The light box already keeps the lighting steady, so there was little to correct.
  - **Why this worked when gray-world (B6) didn't:** gray-world assumes the average color should be gray, and we
    applied it to the eyelid cut-out, whose average is the eyelid itself, so it erased the redness. Here the white
    comes from a real white object.
  - **Subtracting the skin backfires:** the skin pales with anemia too (skin alone 0.73-0.76), so subtracting it
    throws the signal away along with the lighting. The rule even drops below chance: in anemic people the skin
    loses more redness than the nail.
  - **Lesson:** a color reference only helps if it doesn't share the signal. A white board is a good reference; the
    person's own skin is not.
- **Not tried:** the nail and skin numbers side by side as one input, letting the model weigh both. Report:
  [`reports/stage12_nails.md`](reports/stage12_nails.md), section "Stage 12b".

### F16. Palm and fingernail photos of Ghanaian children (M9, M12). Palm: as good as the eyelid. Nails: weaker.
- **Question:** in Ghana children, does the palm or the fingernail show anemia as well as the eyelid?
- **Answer:** the palm does, about the same. The nails less so. **Palm 0.91, nails 0.80, eyelid 0.92** (all TabPFN).
- **Why:** the mentors suggested palm and fingernail sets (M12). This pair comes from the same research group as the
  eyelid set, in Ghanaian children.
- **What we found in the files** (Jaineel downloaded them on 2026-10-24):
  - Each photo is saved 6 or 12 times (exact copies and rotations): 4,260 files per set are **514 palm photos and 526
    nail photos**. We merged them and dropped 1 palm photo filed under both labels.
  - **Only an Anemic / Non-anemic label in the file name.** No Hb, age, sex or hospital, and nothing links a palm or
    nail photo to an eyelid photo, so **we can't combine body sites per child**.
- **What we did (Stage 13):** the same 29 color numbers, rule, models (TabPFN included), no-color control, and a
  model that sees only the file-naming style (`AnemicP-`, `Anemic-Pa-`, ...), in case the styles are batches like
  Italy's. 5x5 CV by photo.
- **What happened:**

| Model | Palm (514) | Fingernails (526) | Eyelid, for comparison (388, F1/F14) |
|---|---|---|---|
| Rule: less red = anemic | 0.620 | 0.497 (chance) | 0.650 |
| Logistic regression | 0.703 | 0.648 | 0.680 |
| Random forest | 0.815 | 0.711 | 0.796 |
| Gradient boosting | 0.855 | 0.682 | 0.846 |
| **TabPFN** | **0.912 [0.886, 0.935]** | **0.798 [0.757, 0.836]** | **0.919** |
| No-color control | 0.454 | 0.566 | 0.537 |
| Naming style only | 0.497 | 0.431 | - |
| Tested to catch 90% (TabPFN) | 62.5% | 74.9% | 50.0% |

- **Takeaways:**
  - **Palm works as well as the eyelid**, with no shortcut: the no-color control and the naming style sit at
    chance, and inside the main naming style TabPFN still scores 0.900.
  - **Still rising with more photos** in the palm (gradient boosting 0.73 -> 0.86 from 103 to 411 training photos).
  - **Nails are weaker**, and the untrained rule is at chance there. Nail cut-out size alone scores 0.57 [0.52, 0.62],
    a small shortcut we can't explain without age data, far below the color models.
  - **TabPFN wins on every Ghana set** (eyelid, palm, nails), by 0.06-0.12 over gradient boosting.
- **Caveats:** the label's cutoff isn't stated in the files; the eyelid comparison uses different photos and maybe
  different children. Report: [`reports/stage13_ghana_sites.md`](reports/stage13_ghana_sites.md).

### F17. One model for Ghana, India and Italy (Stage 14). As good as separate models, and fair; no new signal.
- **Question:** if we train one model on all three countries and take the cheating out (guessing the country, the
  lighting, the photo batch), does it find anemia inside each country?
- **Answer:** it matches a separate model per country, and per-country standardization makes it treat the three
  countries alike. But it still doesn't beat the simple rule in India or Italy. **TabPFN mixed: Ghana 0.90, India
  0.64, Italy 0.91; rule 0.65, 0.67, 0.92.**
- **What we did:** 605 photos, 29 color numbers of the inner eyelid only (no skin). Cutoffs mixed by design: Ghana
  children Hb < 11.0 g/dL, adults WHO by sex (women < 12.0, men < 13.0). 5x5 CV by photo, stratified by country x
  anemia, scored inside each country. Controls: country only, no color, the rule, the Italy early batch. Fixes:
  per-country standardization, reweighing, both, brightness scaling, lighting jitter. Then train on two countries,
  test on the third.
- **What happened:**

| Model | Ghana | India | Italy | Italy early batch | Healthy rank spread |
|---|---|---|---|---|---|
| Rule (no training) | 0.650 | 0.671 | 0.924 | 0.752 | 0.43 |
| TabPFN, one model per country | 0.919 | 0.660 | 0.917 | 0.695 | 0.54 |
| TabPFN, one mixed model | 0.903 | 0.644 | 0.906 | 0.800 | 0.52 |
| TabPFN, mixed + per-country standardized | 0.906 | 0.668 | 0.922 | 0.705 | **0.27** |
| Random forest, mixed + standardized + reweighing | 0.767 | 0.693 | 0.920 | 0.624 | **0.02** |
| Country only (control) | 0.49 | 0.45 | 0.43 | 0.31 | 0.82 |
| No color (control) | 0.58 | 0.50 | **0.69** | 0.50 | 0.29 |

- **Takeaways:**
  - **Mixing doesn't hurt** inside any country. The pooled AUROC (0.895) is inflated: the country alone gives 0.651.
  - **Per-country standardization is the fix that works:** the country gap among healthy people halves for TabPFN
    and almost vanishes for a reweighed random forest, with no loss inside Ghana, India or Italy. Only the Italy
    early batch drops (0.80 -> 0.70, wide ranges).
  - **Lighting fixes (brightness, jitter) change nothing.**
  - **No-color check:** image size and shape give away Italy's early batch (0.69); inside the batch it's 0.50.
  - **Train on two, test on the third** is still no better than the rule: Ghana 0.61 (rule 0.65), India 0.70 (0.67),
    Italy 0.90 (0.92).
- **Caveats:** cutoffs differ between children and adults; India and Italy have few photos, so their ranges are wide
  (India about +-0.12). Report: [`reports/stage14_three_countries.md`](reports/stage14_three_countries.md).

### F18. A few local photos for a new country (Stage 15). Not enough; Ghana needs about 80.
- **Question:** a model trained on two countries doesn't work on the third (F17). If a clinic in the new country
  blood-tests a few people and labels their photos, how many does it take to beat the simple rule? Do the other two
  countries' photos help?
- **Answer:** 10 to 40 local photos are not enough anywhere. Ghana needs about 80 to beat the rule and 200+ for 0.90.
  India and Italy never beat it, even with all their photos. **Best model at 40 photos: Ghana 0.68, India 0.68,
  Italy 0.93; rule 0.65, 0.67, 0.92.**
- **What we did:** same 605 photos and mixed cutoffs as F17. For each country, 5x5 CV by photo; k = 0, 10, 20, 40 or
  all labelled local photos drawn from the training part. Trained on the other two countries + the k photos (each
  country standardized), or on the k photos alone; tested on that country's held-out photos.
- **What happened** (best AUROC at each k):

| Local photos | Ghana | India | Italy | Italy early batch |
|---|---|---|---|---|
| 0 | 0.61 | 0.67 | 0.89 | 0.68 |
| 10 | 0.63 | 0.68 | 0.93 | 0.71 |
| 20 | 0.65 | 0.69 | 0.93 | 0.73 |
| 40 | 0.68 | 0.68 | 0.93 | 0.74 |
| all (310 / 76 / 98) | **0.92** | 0.70 | 0.93 | 0.71 |
| Rule | 0.65 | 0.67 | 0.92 | 0.75 |

- **Takeaways:**
  - **No k from 10 to 40 beats the rule** (no paired change has its CI above 0).
  - **Ghana:** Stage 9's learning curve fills the gap between 40 and 310: TabPFN 0.72 at 78, 0.80 at 155, 0.90 at 233.
  - **The other two countries are a safety net, not extra signal:** with 20 Indian photos a model reaches 0.69 with
    them and 0.59 without. For TabPFN no gain is clear; with all Ghana photos they cost a little (-0.016).
  - The large gains for tree models at 10 photos (up to +0.34) only mean they can't learn from 10 photos alone.
- **Caveats:** India and Italy have few photos (ranges about +-0.11); the Ghana "about 80" comes from Stage 9's folds,
  not a paired test. Report: [`reports/stage15_few_local.md`](reports/stage15_few_local.md).

## Data problems we found along the way

| Problem | Where | What we did |
|---|---|---|
| 212 byte-identical copies + 29 near-copies, often with different labels | CP-AnemiC | use the 388 photos with no copy; split by photo group |
| Re-cropped copies escape the detector in the "476 distinct photos" set (no-color control 0.60) | CP-AnemiC | use the 388; no-color control in every table |
| 187 mask files with a corrupt color-profile block | Eyes-Defy | read with OpenCV |
| The ViT paper's counts only come out with Italy Hb <= 10.5, though it says "< 10.5" | Eyes-Defy | noted; Part A copies the paper, everything else uses WHO cutoffs |
| 31 Italian masks on a white background instead of transparent | Eyes-Defy Italy | fixed in `src/eyelid/roi.py`, reran Parts C and E; conclusions held (rule 0.808 -> 0.835) |
| Italy's anemic patients almost all in the first photo batch (21/31 vs 2/91) | Eyes-Defy Italy | report the early batch alone as the fair Italy check |
| Each palm and nail photo stored 6-12 times (copies and rotations); no Hb, age or link to the eyelid photos | Ghana palm and nails | one row per photo; body sites compared, not combined |

## What worked and what didn't, in one list

**Worked**
- Honest evaluation: clean splits, a duplicate audit, and the no-color control in every table.
- 29 color numbers with random forest or gradient boosting in Ghana children (0.80-0.85), still improving with more
  photos.
- The untrained "less red = anemic" rule for new datasets.
- The survey model carrying over to new countries.
- Fairness methods, for closing the gaps between countries.
- The clinic view (share tested), for explaining what a model is worth.
- Fingernails under fixed lighting: 0.82 with no shortcut, holding on unseen photo sessions.
- A white board in the photo as a color reference (small: the nail rule +0.025, models within noise).
- TabPFN on the 29 color numbers in Ghana children (0.92, the best eyelid result; 0.91 with hospitals held out).
- Palm photos of Ghana children (0.91 with TabPFN, no shortcut found).
- One model for all three countries: as good as a model per country, and fair after per-country standardization.

**Didn't work**
- Bigger image models (DINOv2, a fine-tuned ViT, the eye-pretrained VisionFM, frozen or fine-tuned) instead of
  color numbers.
- Any model, TabPFN included, inside India (stuck at the rule's 0.65-0.67).
- The untrained rule on Ghana fingernail cut-outs (chance).
- Learned eyelid models on a new dataset, also when trained on two countries and tested on the third.
- A few (10-40) labelled photos from a new country: not enough to beat the rule anywhere.
- Gray-world color correction (it erases the signal).
- Subtracting the skin next to the nail (the skin pales too, so it erases the signal).
- Debiasing, for raising accuracy inside a country.
- One model per country.
- Duplicate copies as extra training data (harmless, no gain).

**Still open**
- The deeper eyelid fold in India (F2): real but borderline, needs new Indian photos.

## What's next

| Item | Status | Who |
|---|---|---|
| Eyelid + palm + nail for the same child | can't: the Ghana palm and nail files have no IDs or Hb linking them to the eyelids | ask the dataset authors for the linking sheet |
| Fingernails: nail and skin numbers side by side as one input | idea, not started (F15 showed the skin carries signal of its own) | team |
| Indian IEEE DataPort set | skipped: paid subscription | - |
| Comparison with Chemhaka et al. (survey) | reproduced by the survey side (see their meeting notes) | survey side |
| Paired photo + survey data, to test fusion for real | future | team |

## Where to find things

| What | File |
|---|---|
| Full paper-style write-up | [`README.md`](README.md) |
| Answers for the mentors | [`reports/mentor_followup_eyelid.md`](reports/mentor_followup_eyelid.md) |
| The mentors' suggestions and the step log | [`docs/eyelid_followup.md`](docs/eyelid_followup.md) |
| Stage reports with every number and range | [`reports/`](reports/) |
| Settings, seeds and cutoffs | [`configs/`](configs/) |
