# Progress: every strategy we tried, why, and what happened

*Last updated 2026-11-18, branch `jaineel`. Written to present from: plain English, one block per strategy, each with
why we tried it, what happened, and whether it worked. The full paper-style write-up is the [README](README.md); the
answers for the mentors are in [`reports/mentor_followup_eyelid.md`](reports/mentor_followup_eyelid.md).*

## Where we are

We checked the two published eyelid-photo results, built our own honest baselines, tested whether anything works on a
new dataset or country, built a survey model and tested it in two new countries, simulated combining the two, and
then followed up every piece of eyelid advice from the mentor meeting. All of that is done and pushed.

Since then: fingernails under fixed lighting (F13, F15), the eye-pretrained model VisionFM, frozen and fine-tuned
(F11), a proper check of the deeper eyelid fold (F2), TabPFN, the model the mentors named (F14), and palm and
fingernail photos of Ghanaian children (F16). Then (Stages 16-26): the Hb number, metadata, photo quality, automatic
cropping, one child's eyelid, nail and palm together with cheating and fairness checks (F24, F25), and DINOv2
fine-tuned (F26), a severity check that raised a warning (F27), fingernail + skin together (F28), a full audit of the TabPFN eyelid result (F29), a check of whether the Ghana scores read anemia or how the photos were handled (F30: partly the handling), an attempt at a Ghana number without that trace (F31: not possible with this data, and the Hb-number models carry it too), the same small photo edits on every dataset (F32: TabPFN breaks only on the Ghana sets), measuring the step at the cutoff and planting a fake folder trace in clean data (F33: the step confirms the Ghana jump but can't be taken out; a mild one-folder edit adds 0.06-0.27, and the jump test catches fewer than half), and where in the photo the Ghana trace sits (F34: all through the cut-out, not its edge), TabPFN on Ghana with redness only (F35: no input set is free of the trace), and a model where less red can only mean more anemic (F36: it matches the rule; what the flexible models add comes with the trace), then the mentors' three asks from 2026-11-17 (F37: an MLP on the same 29 numbers gets 0.87 and TabPFN adds 0.05 on top; age and sex add nothing; the numbers that shouldn't carry paleness give TabPFN 0.90), and the numbers the mentors asked the whole group for (F38: no score gets better for the sickest children; with 20 tests per 100 children, TabPFN catches 43% of the anemic ones), and the same check on every Ghana body site against two reference lines (F39: on eyelids, nails and palms TabPFN's score follows a score that knows only the anemic/healthy folder, far below one that follows Hb). From 2026-11-15 new findings are Ghana only; India and Italy are kept for audits. See
[What's next](#whats-next) for what's left.

## Best results so far

The best honest number for each question, kept up to date after every stage. Every one is a held-out score with its
95% range in the stage it links to.

| Question | Best result | Strategy |
|---|---|---|
| Eyelid photo, Ghana children (Hb < 11.0) | TabPFN on 29 color numbers: 0.919 [0.893, 0.945], **but not a clean anemia number**: the score jumps at the Hb 11 cutoff instead of following Hb, so part of it reads how the anemic and healthy photos were handled (F30); no leak (F29); 0.59 on 61 differently processed photos; a small color edit costs it 0.21, which happens only on the Ghana sets (F32); the trace can't be taken out to give a clean number (F31) | F14, F29-F32 |
| Palm photo, Ghana children | TabPFN: 0.912; also jumps at the cutoff (F30), though the palm carries some Hb inside each group (0.32 anemic, 0.44 healthy) | F16, F30 |
| Eyelid + nail + palm of the same Ghana child | TabPFN 0.931 (eyelid alone 0.913; the gain isn't clear); same jump caveat (F30) | F24, F25, F30 |
| Fingernails, Moscow adults in a light box (Hb < 12.0) | **logistic regression: 0.818**, no shortcut, holds on unseen sessions; TabPFN 0.823 passes the jump test and follows Hb inside both groups (F30): our cleanest result | F13, F30 |
| Fingernails, Ghana children | TabPFN 0.798; jumps at the cutoff while the redness rule is at chance (F30) | F16, F30 |
| Eyelid, India adults (WHO by sex) | the untrained rule, 0.671; no model beats it | F1 |
| Eyelid, Italy adults | 0.92, but mostly the photo batch | F1, F4 |
| A new dataset with no training | the "less red = anemic" rule, 0.835 on Eyes-Defy | C2 |
| Survey answers only (DHS women; survey side) | 0.58 in Zimbabwe, unchanged in Malawi and Zambia (0.57) | D1, D2 |
| The Hb number, Ghana children | error 1.38 g/dL, vs 1.71 for guessing the mean, **but mostly the folder**: no Ghana Hb model follows Hb inside the anemic group (four models, every range crosses 0), and the best one jumps at the cutoff (F31) | F19, F31 |
| Does the Ghana score track how low the Hb is? | **No**: caught 81% severe, 73% moderate, 79% mild; inside the anemic group the score doesn't follow Hb (+0.11) | F27 |
| Do the scores read anemia, or how the photos were handled? | **Ghana: partly the handling.** Every model's Ghana score (TabPFN, gradient boosting, the fine-tuned vision models) jumps at Hb 11.0; India, Italy and Moscow follow Hb steadily | F30 |
| Can the trace be taken out to give a clean Ghana number? | **Not with this data.** A model trained only on differences inside each folder loses 0.15-0.41 on the clean datasets (F31); measuring the step at the cutoff and taking it out removes too much from a planted trace (F33). Neither gives a Ghana number we can trust | F31, F33 |
| Does a small lighting change break the model? | TabPFN: **only on the Ghana sets** (4-7 of 7 edits break it, average cost 0.08-0.21); India, Italy, Moscow 0 of 7 (cost 0.00-0.02). Training on color-edited copies cuts the Ghana eyelid cost from 0.21 to 0.05 for 0.04 of AUROC | F32 |
| How much can a handling trace add, and does the jump test catch it? | a mild edit of the anemic photos only (gamma 0.9, 10% less saturation or a 4% white-balance shift) adds **0.06-0.27** AUROC on India and Moscow; the jump test catches **10 of 24** such plants (Moscow 7 of 11, India 3 of 11), so a pass on a small set is weak evidence | F33 |
| Where in the photo is the Ghana trace? | **all through the cut-out**: TabPFN on the inner half alone (0.90 on eyelids) or the outer ring alone (0.89) still jumps; not the cut-out edge | F34 |
| Is there a Ghana input set where TabPFN carries no trace? | **No.** Redness only (0.88 eyelids, 0.70 nails, 0.78 palms) still jumps, has a step and breaks under photo edits; with mean redness alone TabPFN only matches the rule on eyelids (0.640 vs 0.650). Only logistic regression passes: nails 0.637 (+0.14 over the rule) | F35 |
| Does a model that can only say "less red = more anemic" beat the rule on Ghana? | **No: it matches the rule** (eyelids 0.62, nails 0.45-0.50, palms 0.58) and shows no trace on nails and palms; the free model on the same numbers gains 0.10-0.13 and that gain comes with the trace | F36 |
| How much of TabPFN's Ghana eyelid score is TabPFN, and how much the 29 numbers? | **mostly the 29 numbers:** an MLP on them gets 0.870, TabPFN adds **0.049 [0.023, 0.078]** (0.084 with one hospital held out); logistic regression only 0.680 (rule 0.650). Every flexible model jumps at the cutoff, so the extra over the rule comes with the trace | F37 |
| Does age and sex help, next to the 29 numbers? | **No:** TabPFN -0.001 [-0.009, 0.006], MLP -0.060; age and sex alone 0.57-0.59, no trace (no BMI in the data) | F37 |
| Do the numbers that shouldn't carry paleness (hue, green, blue, yellow-blue) score? | **Yes, almost as well:** TabPFN 0.903 vs 0.917 on the paleness numbers, MLP 0.835 vs 0.863, both jump; "weird" by the rule fixed before running | F37 |
| Does the Ghana score pick out the sickest children better? | **No:** for Hb < 8 (the sickest 9%) TabPFN falls from 0.919 to 0.749 (-0.170 [-0.235, -0.108]), about as much as a score that knows only the folder (1.00 -> 0.82); the rule and logistic regression don't clearly change | F38 |
| Does any Ghana body site's score read how sick a child is? | **No, on any site.** At Hb < 8 TabPFN sits on the line of a score that knows only the folder (eyelids 0.749 vs 0.764, palms 0.713 vs 0.734) and far below one that follows Hb (0.958, 0.928); nails fall below even the folder line | F39 |
| 100 Ghana children, blood tests for 20 | TabPFN catches **43%** of the anemic ones [38%, 48%] (at most 47% possible), the rule 29%, random 20%; same trace caveat | F38 |
| What it saves a clinic (Ghana) | test 50% of children to catch 90% of the anemic ones (TabPFN), vs 90% at random | F10 |

## The story in six points

1. **Published eyelid scores are inflated.** Both papers test on photos that are also in training (copies, or flipped
   versions). A clean split costs 7-8 accuracy points.
2. **On Eyes-Defy the model mostly learns the country.** Indian photos are mostly anemic, Italian ones mostly not.
   Inside India the published model is a coin flip.
3. **Color numbers beat big image models only on the Ghana photos, and partly for the wrong reason.** There TabPFN,
   a model built for small tables, reaches **0.92** on 29 color numbers against 0.67-0.79 for vision models. But every
   model's Ghana score jumps at the Hb 11 cutoff instead of following Hb (F30): part of what they read is how the
   anemic and healthy photos were handled, and TabPFN reads it best. The Ghana sets are also the only place where small color edits break TabPFN (F32), and a mild edit planted in one folder of a clean dataset adds up to 0.27, as much as TabPFN's Ghana lead over the rule (F33). Where there is no such trace, TabPFN matches the
   untrained rule (India 0.65, Italy 0.91) or adds a little (Moscow nails 0.82 vs 0.70). The untrained "less red =
   anemic" rule is still the only thing that keeps working on a new dataset.
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
| F19 | The Hb number inside and across datasets (Stages 16-19, Maria) | a number is more useful than yes/no | Ghana error 1.38 vs 1.71 g/dL; across datasets the scale shifts (Italy 2.9 g/dL low) | **works inside a dataset only** |
| F20 | Age and sex next to the photo (M17) | where does the metadata go? | yes/no: Ghana 0.919 -> 0.918; Hb number in India 1.45 -> 1.29 g/dL | **no gain for yes/no; helps the Hb number in adults** |
| F21 | Photo quality, glare, retake gate (M16) | screen out bad photos | no gain; glare alone flags Italy's batch (0.89) | **no gain; a good shortcut check** |
| F22 | TabPFN for the Hb number | the best yes/no model, as a regressor | Moscow nails 1.65 -> 1.59 g/dL; level elsewhere | **small gain on nails only** |
| F23 | A vision model crops the eyelid (M16) | full photos, not hand cut-outs | Dice 0.75 Italy, 0.66 India; Italy TabPFN 0.902 vs 0.909 | **close in Italy, looser in India** |
| F24 | One child's eyelid, nail and palm together (M18) | the mentors' "same person" idea | TabPFN 0.913 -> 0.931 (CI crosses 0); random forest, boosting +0.06 to +0.09 | **helps weaker models, not TabPFN** |
| F25 | Cheating and fairness checks on those children | asked 2026-11-07 | fair by sex and age; eyelid cut-out shape alone 0.69, not used by the color models | **fair; one shortcut found, not used** |
| F26 | DINOv2 fine-tuned, and its features next to the color numbers (M1, M10) | the mentors' DINOv2 question | fine-tuned 0.690 (frozen 0.672); color + DINOv2 0.897 vs color 0.919 | **didn't work** |
| F27 | Catch rate by Hb severity band, Ghana children | does the model catch the sickest children? | severe 81%, moderate 73%, mild 79%; score vs Hb inside the anemic group +0.11 | **warning: separates the groups, doesn't rank by Hb** |
| F28 | Fingernail and skin numbers together, Moscow | the idea left open in F15 | TabPFN nail 0.823 -> nail + skin 0.825; rule 0.703 -> 0.748 | **no gain for models**; small gain for the rule |
| F29 | Full audit of TabPFN on Ghana eyelids (7 checks) | is the 0.92 cheating, overfitting or our choices? | shuffles at chance, redness alone 0.88, model choice 0.92; 61 differently processed photos 0.59 (rule 0.67) | **no leak; holds within the batch, not across batches** |
| F30 | Does the score read anemia or photo handling? (jump at the cutoff, Hb inside each group, no-color numbers, edited photos) | the mentors' surprise at TabPFN; Jaineel: make sure it is seeing only anemia | Ghana eyelid TabPFN near the cutoff 0.938 vs 0.72 for a steady signal; nails, palms and the vision models the same; India, Italy, Moscow no jump; a 15% brightness change costs TabPFN 0.21 | **Ghana scores are partly a handling trace; "table beats vision" doesn't hold as a claim about anemia** |
| F31 | A Ghana number without the trace (a within-folder model); do the Hb-number models carry the trace? | F30: how much of the Ghana score is real? | within-folder model 0.50 on Ghana but loses 0.15-0.41 on clean data; no Ghana Hb model follows Hb inside the anemic group; the boosting and TabPFN Hb estimates jump; 6 eyelid color numbers separate best near the cutoff | **no trace-free number; the Ghana Hb results carry the trace too** |
| F32 | The same small photo edits on every dataset; TabPFN trained on color-edited copies | is TabPFN fragile everywhere, or only on Ghana? | fragile on 6, 4, 7 of 7 edits on Ghana eyelids, nails, palms; 0 of 7 on India, Italy, Moscow; copies cut the Ghana eyelid cost 0.21 -> 0.05, the jump stays | **the fragility is a sign of the Ghana trace; edited copies help robustness, not the trace** |
| F33 | Measure the step in the score at the cutoff and take it out; plant a fake folder trace in clean data | F31: can the trace be measured instead of avoided? | Ghana steps 0.9-1.8 score SDs, clean sets none; taking the step out removes too much from planted traces (0-1 of each model's plants back); a mild one-folder edit adds 0.06-0.27; the jump test catches 10 of 24 | **the step confirms the Ghana jump; no trusted trace-free number; a small-set pass is weak** |
| F34 | Color numbers on the inner half and the outer ring of each cut-out; each no-color number through the jump test | where in the photo is the trace? | Ghana: core and ring both jump and keep the AUROC (eyelids 0.90 / 0.89); no no-color number jumps except Ghana palm height | **all through the cut-out, not its edge** |
| F35 | TabPFN and logistic regression on Ghana with mean redness, the redness numbers, all 29 minus the jumping numbers, all 29; every score through the jump test, the step and the photo edits | is there an input set where TabPFN carries no trace? | no TabPFN set passes on eyelids, nails or palms; redness-only TabPFN 0.876 / 0.697 / 0.779 all jump; logistic regression on nails 0.637 passes | **no clean TabPFN number on Ghana; the gain over the rule comes with the trace** |
| F36 | Gradient boosting where less red can only mean more anemic, same checks | does a pallor-only model beat the rule? | forced model = rule (eyelids 0.62, nails 0.45-0.50, palms 0.58), no trace on nails and palms; free model on the same numbers +0.10-0.13, with the trace | **what the flexible models add on Ghana is not "less red = more anemic"** |
| F37 | Logistic regression, an MLP, gradient boosting and TabPFN on the same 29 numbers; + age and sex; the numbers that shouldn't carry paleness; redness, brightness and saturation combinations (Ghana eyelids), same checks | mentors 2026-11-17: how much is TabPFN, how much the features? add demographics; Jaineel: give it what shouldn't be useful | MLP 0.870, TabPFN 0.919 (+0.049); logistic regression 0.680; age + sex 0.000 to -0.060; not-useful numbers TabPFN 0.903 vs paleness 0.917 | **the 29 numbers do most of it, TabPFN adds 0.05; the lift over the rule is in every color number and comes with the trace** |
| F38 | The same held-out score against Hb < 11, 10, 9, 8, 7; the share of anemic children caught when 20% or 30% get a blood test (Ghana eyelids) | mentors 2026-11-17, advice to the whole group: severity on the x axis, AUROC on the y; "capacity to test 20" | TabPFN 0.919 -> 0.749 at Hb < 8; 20 tests: TabPFN 43%, rule 29%, random 20% | **no score is better for the sickest; TabPFN fills the tests well, with the trace** |
| F39 | F38's severity curve on the Ghana eyelids (388) and the linked eyelids, nails and palms (302), each score next to a folder-only line and a follows-Hb line matched to its AUROC at Hb < 11 | F38: is TabPFN's drop just the folder? Is the palm different? | at Hb < 8 TabPFN vs folder line: eyelids -0.015, linked eyelids +0.036, palms -0.020, nails -0.107 (CI below 0); follows-Hb line 0.16-0.28 higher | **no Ghana score reads severity beyond the folder, on any body site** |

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

### F19. The Hb number (Stages 16-19, Maria). Works inside a dataset; doesn't carry over.
- Inside Ghana the photo estimates Hb better than guessing the mean: error 1.38 g/dL vs 1.71.
- Across datasets the Hb scale shifts: Italy's estimates come out 2.9 g/dL too low.
- A few local Hb values fix that average shift, but not the ranking.
- Fingernail models don't carry between Ghana and Moscow (0.38-0.58).
- Explained in [`HB_EXPERIMENTS_EXPLAINED.md`](HB_EXPERIMENTS_EXPLAINED.md). Reports regenerated at a clean commit:
  [`stage16`](reports/stage16_hb_regression.md) to [`stage19`](reports/stage19_nails_cross.md).

### F20. Age and sex next to the photo (Stage 20). Yes/no: no gain. Hb number in adults: helps.
- **Question:** the mentors asked (2026-11-03) where the metadata goes into the model. Until now it went nowhere. Does
  adding age and sex help inside each location?
- **Answer:** not for anemic yes/no in any country. For the Hb number in the adults (India, Italy) it does: men's Hb
  runs higher. **Ghana TabPFN 0.919 -> 0.918; India Hb error 1.45 -> 1.29 g/dL.**
- **What we did:**
  - Data: Ghana 388 photos (Hb < 11.0), India 95 and Italy 122 (WHO by sex: women < 12.0, men < 13.0 g/dL).
  - Inputs: photo only, age + sex only, or both.
  - Models: the color models, TabPFN, the DINOv2 probe, and Stage 16's Hb regressors.
  - Splits: 5x5 CV by photo, plus leave one hospital out in Ghana. Changes are paired, same model. The hospital is
    never an input.
- **Takeaways:**
  - No yes/no change has its CI above 0. The one exception is the weak DINOv2 probe with a hospital held out
    (0.652 -> 0.674).
  - Hb number: India ridge -0.157 [-0.310, -0.005] g/dL; Italy ridge -0.082 [-0.154, -0.008]; Ghana no change.
  - In India the photo adds nothing to the Hb number: age + sex alone do as well as both together (1.28 vs 1.29).
  - Italy's 0.81 from age + sex alone means the anemic patients are older (median 65 vs 46) and more often women.
- **Caveats:** in India and Italy sex also sets the cutoff, so a yes/no gain from sex would be partly definition.
  Report: [`reports/stage20_metadata.md`](reports/stage20_metadata.md).

### F21. Photo quality and glare (Stage 21). No gain; it caught Italy's batch again.
- **Question:** the mentors advised (2026-10-31) screening out bad photos in the pipeline. Does a quality filter or
  glare removal help?
- **Answer:** no, these sets are already curated cut-outs. But the quality numbers are a useful shortcut check:
  **in Italy, glare alone separates anemic from healthy at 0.89.** Inside the early batch it is 0.53, so glare marks
  the batch.
- **What we did:**
  - Data: Ghana eyelids, palms and nails; India and Italy eyelids.
  - Per photo, from the pixels inside the cut-out: glare share, dark share, blown-out share, blur and size.
    Thresholds fixed before running.
  - (a) Each quality number on its own as a score.
  - (b) Glare removed before the 29 color numbers.
  - (c) Retake gate: drop the worst 10% or 20% at test time. No labels are used to pick them.
- **Takeaways:**
  - Retake gate: the best model changes by -0.003 to +0.017, no CI above 0. Weaker models gain about +0.03 on Ghana
    nails at 20%.
  - Glare removal: no change. The median glare share is 0% in every set except Italy's early batch (2%).
  - The gate is not neutral in Italy: 31-40% of the photos it rejects are anemic, vs 19% overall.
- **Caveats:** on palms and nails near-white can be the nail tip or pale skin, not only glare. Report:
  [`reports/stage21_quality.md`](reports/stage21_quality.md).

### F22. TabPFN for the Hb number (Stage 22). Closer only on Moscow nails.
- **Question:** TabPFN is the best yes/no model in Ghana (0.92). Does its regressor also estimate the Hb number better
  than Stage 16's models?
- **Answer:** only on the Moscow fingernails. Elsewhere it is level with the best Stage 16 regressor.
  **Moscow error 1.65 -> 1.59 g/dL; Ghana 1.44 vs 1.38 for gradient boosting.**
- **What we did:**
  - Data and cutoffs: Ghana eyelids, India and Italy eyelids, and the Moscow nails (Hb < 12.0 for everyone).
  - Inputs: the 29 color numbers, plus age + sex for the eyelid sets.
  - Splits: 5x5 CV; Ghana also leave one hospital out; Moscow also leave one session out.
  - Compared, paired, with the best of ridge, random forest and gradient boosting.
- **Takeaways:**
  - Moscow: -0.065 [-0.125, -0.004] g/dL, and about the same with a photo session held out (-0.072, CI just
    crosses 0).
  - Ghana: TabPFN ranks a little better (0.821 vs 0.802) but its error isn't lower. Ranking by its Hb estimate is still
    far below TabPFN's yes/no model (0.919).
  - India and Italy: changes from -0.03 to +0.03, every CI crosses 0.
- **Caveats:** the weights are an extra 44 MB download (approved 2026-11-04). Report:
  [`reports/stage22_hb_tabpfn.md`](reports/stage22_hb_tabpfn.md).

### F23. A vision model that crops the eyelid (Stage 23). Close in Italy, looser in India.
- **Question:** the mentors suggested (2026-10-31) a vision model that crops the region and one that flags bad photos.
  Can a model find the eyelid in the full photo, and does anemia scoring hold on its crop?
- **Answer:** yes in Italy, roughly in India. **Overlap with the hand mask (Dice) 0.75 Italy, 0.66 India; Italy
  TabPFN 0.902 on its crop vs 0.909 on the hand mask.**
- **What we did:**
  - Data: Eyes-Defy only, the one set with the full photos (India 94, Italy 121; Italy_15 and India_2 left out, broken
    masks).
  - Model: frozen DINOv2-small patch tokens plus a logistic regression that learns from the hand masks which patches
    are eyelid. 5-fold CV by photo, so no photo is cropped by a model that saw its mask. No anemia labels.
  - Scoring: the same 29 color numbers and models on its crop vs the hand mask.
  - Confidence (mean probability inside the crop) used as a retake flag.
- **Takeaways:**
  - It always found an eyelid. 67% of Italy's crops and 38% of India's reach Dice 0.7.
  - Anemia: only gradient boosting in Italy is clearly worse on the automatic crop (-0.046). The rule and TabPFN lose
    at most 0.015 in Italy. TabPFN in India drops 0.680 -> 0.603 (CI crosses 0).
  - Confidence tracks crop quality a little in Italy (Spearman 0.26 [0.08, 0.43]), not in India (0.07).
  - Retake gate: in India it lifts the learned models (+0.05). But all 10 photos it drops at 10% belong to anemic
    patients, so it isn't neutral. No gain in Italy.
- **Caveats:** the masks it learns from are the same hand masks it is judged against. Raw phone photos would be
  harder. Report: [`reports/stage23_auto_crop.md`](reports/stage23_auto_crop.md).

### F24. One child's eyelid, nail and palm together (Stage 24). Probable link; helps the weaker models, not TabPFN.
- **Question:** the mentors (2026-11-03): put each child's nail and eyelid photos together, as if from the same person.
  Does that help?
- **Answer:** a little. **TabPFN 0.913 on the eyelid -> 0.931 with nail and palm (+0.018, CI crosses 0).** Random forest
  and gradient boosting gain +0.06 to +0.09 (CI above 0).
- **What we did:**
  - New data (Jaineel's OK, 2026-11-05): Ghana conjunctiva set, Mendeley 10.17632/nt7r8hv2pz.1. 414 of its 514 photos
    are CP-AnemiC photos pixel for pixel, so they carry CP-AnemiC's Hb. Its file names follow the nail and palm sets
    (`Anemic-105` / `Anemic-FN-105` / `AnemicP-105`).
  - Link test, fixed before any model: same-number photos agree in color more than neighbouring numbers. Nail-palm
    0.38 vs 0.27 (p = 0.0005), eyelid-palm 0.18 vs 0.12 (p = 0.003), eyelid-nail 0.09 vs 0.06 (p = 0.07, not shown).
  - 302 children with all three photos and an agreed Hb; **Hb < 11.0 from the CP-AnemiC sheet** for all three sites.
    5x5 CV by child plus leave one hospital out; early and late fusion; 5 draws of pretend children (random same-label
    nail and palm).
- **Takeaways:**
  - The palm adds most: eyelid + palm late TabPFN 0.933 (+0.020 [-0.001, 0.043]).
  - Pretend children score 0.967 (+0.054, CI above 0 in every draw), about 3x the real gain. One child's photos
    probably share the same mistakes; random partners don't. So pretending overstates what combining gives.
  - Hb number: no gain (gradient boosting 1.32 -> 1.33 g/dL).
- **Caveats:** the link comes from file names and color agreement, not from the authors. Korle Bu colorimetry (PLOS ONE
  2023) was checked and not used: 43 children, 25 under 6 months. Report:
  [`reports/stage24_ghana_link.md`](reports/stage24_ghana_link.md).

### F25. Cheating and fairness checks on the linked children (Stage 25). Fair; one shortcut, not used.
- **Question:** Jaineel asked (2026-11-07) whether F24 had the usual cheating and fairness tests. It had the split by
  child, hospitals held out and pretend children, but not these four.
- **Answer:** adding nail and palm is fair by sex and age. **But the eyelid cut-outs' size and shape alone find anemia
  (0.693), a shortcut; the color models don't use it** (TabPFN still 0.895-0.929 inside each third of the shape score).
- **What we did:**
  - Same 302 children, folds and label as F24 (Hb < 11.0 from the CP-AnemiC sheet); age and sex from the same sheet.
  - Answers fixed before running in `configs/ghana_checks.yaml`.
  - (1) Fairness: AUROC, sensitivity and false alarms at one shared threshold, for girls vs boys and 6-23 vs 24-59
    months, eyelid alone vs all three.
  - (2) No color: TabPFN and boosting on the size and shape of the three photos.
  - (3) Same mistakes: inside each label, how much the eyelid and palm/nail scores move together, for the same
    number, the neighbouring number and random partners.
  - (4) Wrong matches: give a share of children a random same-label nail and palm and watch the score.
- **Takeaways:**
  - **Fair:** no gap widens (every CI of the change crosses 0). TabPFN with all three: girls 0.946, boys 0.919; 6-23
    months 0.880, 24-59 months 0.956.
  - **Shortcut:** it's the eyelid (0.699). Nail 0.555, palm 0.498. Anemic children's cut-outs are a little smaller,
    with more soft-edged pixels (edge share alone 0.684). Over all 388 CP-AnemiC photos the same check gave 0.565.
  - **Same mistakes:** eyelid and palm scores of one child move together (0.318), more than a neighbour's (0.195) and
    far more than a random partner's (0.00). That explains why pretend children score higher in F24, and backs the
    link.
  - **Wrong matches:** every wrong match pushes the score up (a quarter wrong: +0.026 instead of +0.018), because
    matches are made inside the Anemic / Non-anemic folders. So the real gain is at most +0.018.
- **Caveats:** the shape follow-up (which numbers, the thirds) was added after the check failed, not fixed
  beforehand. Report: [`reports/stage25_ghana_checks.md`](reports/stage25_ghana_checks.md).

### F26. DINOv2 fine-tuned, and DINOv2 features next to the color numbers (Stage 26). Didn't work.
- **Question:** the mentors asked about DINOv2. We had only used it frozen (B5). Does training it, or giving its
  features to TabPFN next to the color numbers, help?
- **Answer:** no. **Fine-tuned DINOv2 0.690 (frozen 0.672); TabPFN on color + DINOv2 0.897, worse than color alone
  (0.919, -0.022 [-0.033, -0.011]).**
- **What we did:**
  - DINOv2-small, already on disk (nothing downloaded), trained end to end on the 388 Ghana photos with the same
    settings as our ViT and VisionFM (30 epochs, 5 folds, 1 seed, 23 min on the M1).
  - TabPFN on the 29 color numbers plus DINOv2 or VisionFM features, squeezed to 16 PCA numbers inside each training
    fold. Ghana (also leave one hospital out), India and Italy.
  - Answers fixed before running: (a) beats frozen DINOv2; (b) color + image beats color alone (paired CI above 0).
- **Takeaways:**
  - Fine-tuning learns the training photos (training AUROC 0.997-1.00), and the held-out score barely moves (+0.02,
    ranges overlap almost completely).
  - The image features add noise: -0.022 (DINOv2) and -0.014 (VisionFM) in Ghana, also with a hospital held out. No
    change in India or Italy.
  - Same lesson as F1 and F11: with about 310 training photos the limit is the data, not the model.
- **Run it yourself:** `python -m src.eyelid.finetune_vit --model dinov2`. Report:
  [`reports/stage26_tabpfn_image.md`](reports/stage26_tabpfn_image.md).

### F27. Does the model catch the sickest children? (Stage 27). A warning: it doesn't rank by Hb.
- **Question:** one AUROC treats a child at Hb 10.9 and a child at Hb 6 the same. Does the Ghana eyelid model catch the
  severe cases and miss mostly the mild ones?
- **Answer:** no. **TabPFN catches about the same share in every band: severe 81%, moderate 73%, mild 79%** (90% of
  healthy children cleared). Moderate + severe minus mild: -0.04 [-0.17, +0.13].
- **What we did:** the 388 Ghana children, **Hb < 11.0 g/dL**, WHO severity bands (severe < 7.0, moderate 7.0-9.9, mild
  10.0-10.9), Stage 9's folds, the rule, gradient boosting and TabPFN. The answer rule was fixed before running.
- **Follow-up (added after seeing the flat rates):** rank correlation between Hb and the TabPFN score: -0.58 over all
  children, but **+0.11 [-0.04, +0.26] inside the anemic group and +0.10 [-0.03, +0.23] inside the healthy group**.
  Gradient boosting and the rule: the same.
- **Takeaways:**
  - The score separates the two groups (0.92) but doesn't rank children inside them. A child at Hb 6 doesn't score
    higher than one at 10.5, and a child at 11.2 isn't flagged more than one at 14 (8% vs 10% false alarms).
  - Paleness should grow as Hb falls, so a score that reads paleness should keep tracking Hb inside the anemic group.
    Ours doesn't. Either eyelid color saturates, or the score reads something tied to the Anemic / Non-anemic label
    and not to Hb itself, such as how each group's photos were taken or cut out (the same worry as the shape shortcut
    in F25).
  - TabPFN misses 40 of 165 anemic children, 3 of them severe (of 16).
  - It also explains F19: the Hb number estimate is mostly a two-group guess.
- **What would settle it:** photos with Hb collected in one session by one protocol.
- **Run it yourself:** `python -m src.eyelid.severity` (~5 min). Report:
  [`reports/stage27_severity.md`](reports/stage27_severity.md).

### F28. Fingernail and skin color numbers together (Stage 28). No gain for the models; a small one for the rule.
- **Question:** F15 left one idea open: the skin next to the nail pales with anemia too (0.73-0.76 alone), so does
  giving the model both the nail and the skin numbers beat the nail alone?
- **Answer:** not for any trained model. **TabPFN: nail 0.823, nail + skin in one model 0.825 (+0.002 [-0.019,
  +0.024]).** The untrained redness rule does gain: 0.703 -> 0.748 when the nail and skin redness ranks are averaged
  (+0.046 [+0.003, +0.084]).
- **What we did:** the 250 Moscow adults, **Hb < 12.0 g/dL for everyone**, Stage 12's splits (5x5 CV by patient, leave
  one session out). Inputs: nail, skin, both in one model, and the two models' scores averaged. Logistic regression,
  random forest, gradient boosting, TabPFN, the rule, a dummy and a no-color control (box sizes: 0.52). Answer fixed
  before running.
- **Takeaways:**
  - Every model's change is between -0.006 and +0.017 and every CI crosses 0, also on held-out sessions.
  - The skin carries the same signal as the nail, only weaker (skin alone 0.05-0.09 below the nail), so it adds
    nothing new to a model that already has the nail.
  - A rule with no training does gain from the second reading, but it stays below the models (0.75 vs 0.82).
- **Run it yourself:** `python -m src.nails_skin` (~15 min). Report:
  [`reports/stage28_nails_skin.md`](reports/stage28_nails_skin.md).

### F29. A full audit of the TabPFN eyelid result (Stage 29). No leak; redness carries it; but it doesn't hold on a differently processed batch.
- **Question:** a table model on 29 color numbers (0.92) beats every vision model (0.67-0.79). Is that cheating,
  overfitting or a fluke of our choices, and does it read paleness?
- **What we did:** seven checks on the 388 Ghana children (**Hb < 11.0 g/dL**), every pass rule fixed in
  `configs/tabpfn_audit.yaml` before running.
- **Answers:**

| Check | Result | Verdict |
|---|---|---|
| 1. Shuffled labels, 100 times | real 0.883 (one 5-fold run); shuffles mean 0.499, max 0.576; 0 of 100 reach it | pass |
| 2. Hidden copies kept together (looser photo match) | 0.884 vs 0.919 | fail (but see the follow-up) |
| 3. File metadata only | dpi and gamma identical in every photo | pass |
| 4. What it reads | redness alone (5 numbers) 0.876; brightness alone 0.785; hue 0.574 | reads redness |
| 5. Right direction | score vs mean a*: -0.47 [-0.55, -0.39] (paler = riskier) | pass |
| 6. Model chosen without the test children | 0.919; TabPFN picked in all 25 splits | pass |
| 7. 61 never-used Ghana eyelid photos (51 anemic) | TabPFN 0.586 [0.404, 0.759]; rule 0.673; gradient boosting 0.692 | fail |

- **Follow-up (added after checks 2 and 7 failed, not fixed beforehand; `reports/stage29_followup.md`):**
  - Check 2: the pre-set 30-bit match was too loose (the typical gap to the nearest unrelated photo is 34 bits), so
    look-alike photos chained into a group of 110 forced into one fold. With only the close pairs kept together the
    score barely moves: 10 bits 0.919 (2 pairs), 20 bits 0.910 (17 pairs; -0.009 [-0.020, +0.001]). Random groups of
    the same sizes drop 0.003-0.017. So there is no copy leak; look-alike photos (76% share a label, against 51% for
    random pairs) help TabPFN a little.
  - Check 7: the never-used photos are a differently processed batch: portrait cut-outs (median 69 x 172 px against
    211 x 81), darker (L* 44-53 against 57-58) and less red (a* 19-22 against 24-27). (The 37 "near-copies" of CP-AnemiC photos in that set were matched at the same too-loose 30 bits, so they are
    look-alikes, not proven copies; see F30.) So check 7 is really a small cross-batch test: TabPFN drops to
    0.59 while the plain rule holds (0.67), the same pattern as across datasets (TabPFN transfer 0.60, rule 0.84).
- **Takeaways:**
  - No sign of cheating: shuffled labels sit at chance, metadata is identical, the model choice adds no optimism, and
    the score follows paleness in the right direction.
  - Redness alone gives 0.88. (F30 later showed this redness is partly a trace of photo handling, not only paleness.)
  - TabPFN's edge over simpler models is within one consistent batch of photos. When colors shift, the plain redness
    rule holds better. Only 10 healthy photos in check 7, so that range is wide.
  - Paper wording, superseded by F30: "within CP-AnemiC, a table model on color numbers beat fine-tuned vision
    models", not a general claim.
- **Note:** the main report's near-copy pair count (75,690) is a counting bug; the right number is 224 (fixed in the
  code at `ef11c9a`; the follow-up report has it). No other number in the main report depends on it.
- **Run it yourself:** `python -m src.eyelid.tabpfn_audit` (~2 h, mostly the 100 shuffles) and
  `python -m src.eyelid.tabpfn_audit_followup` (~25 min). Reports:
  [`reports/stage29_tabpfn_audit.md`](reports/stage29_tabpfn_audit.md),
  [`reports/stage29_followup.md`](reports/stage29_followup.md).

### F30. Does the score read anemia, or how the photos were handled? (Stage 30). Ghana: partly the handling.
- **Question:** the Ghana scores flag 79% of children at Hb 10-11 but 8% at Hb 11-12 (F27), and TabPFN's big lead over
  the rule shows up only on the three Ghana sets. Paleness changes gradually with Hb, so a score that reads it can't
  jump at a cutoff that is a convention. Is the score reading anemia, or a trace of how the anemic and healthy photos
  were handled (the Ghana sets store them in separate folders)?
- **What we did:** four checks on every dataset with Hb, same 29 color numbers, every answer fixed in
  `configs/handling_check.yaml` before running. Cutoffs: Ghana **Hb < 11.0 g/dL** (children); India and Italy **WHO by
  sex, < 12.0 women, < 13.0 men**; Moscow **< 12.0 g/dL** for everyone.
  - A1, jump at the cutoff: AUROC among people within 1 g/dL of their cutoff, against a steady reference (a score that
    falls steadily with Hb, with the model's own overall AUROC, 2000 draws, no Hb error: the steepest a steady signal
    can be).
  - A3, Hb inside each group: a TabPFN regressor trained and tested inside the anemic group only, then the healthy one.
  - A2, no color: size, cut-out shape, soft edge, sharpness and file bytes only.
  - B, edits: the Ghana test photos brightened, darkened, re-toned or re-balanced, models trained on the originals.
- **Answers (TabPFN; gradient boosting the same):**

| Dataset | AUROC, everyone | AUROC near the cutoff | A steady signal would give (top 2.5%) | Jumps? |
|---|---|---|---|---|
| Ghana eyelids (388) | 0.919 | **0.938** [0.894, 0.974] | 0.72 (0.81) | **yes** |
| Ghana nails (302 linked children) | 0.754 | **0.760** [0.658, 0.855] | 0.60 (0.71) | **yes** |
| Ghana palms (302 linked children) | 0.866 | **0.888** [0.817, 0.951] | 0.66 (0.76) | **yes** |
| India eyelids (95) | 0.648 | 0.510 [0.289, 0.722] | 0.55 (0.75) | no |
| Italy eyelids (122) | 0.909 | 0.771 [0.511, 0.977] | 0.68 (0.88) | no |
| Moscow nails (250) | 0.823 | 0.569 [0.432, 0.700] | 0.58 (0.72) | no |

  - The fine-tuned vision models jump too (ViT 0.727 vs 0.58, DINOv2 0.679 vs 0.57, VisionFM 0.781 vs 0.61;
    `reports/stage30_vision.md`). The plain redness rule jumps a little on the eyelids (0.669 vs 0.55), not on nails
    or palms.
  - Score vs Hb inside the anemic group: Ghana eyelids +0.11 (a steady signal: -0.69); Moscow -0.51 and India -0.32
    follow Hb.
  - A3: Ghana eyelids anemic +0.21 [0.05, 0.36], healthy +0.12 (CI crosses 0); palms +0.32 / +0.44; nails +0.07 /
    +0.21; Moscow +0.55 / +0.24; India anemic +0.53. The Ghana eyelids carry only a weak Hb signal; the palms more.
  - A2: Ghana eyelids 0.606 [0.549, 0.662], palms 0.622, nails 0.573, so the two folders also differ in how the photos
    were cut out or saved. Italy 0.864 (its known early batch, F4), India 0.52.
  - B: a 15% brightness change, gamma 0.8 or 1.25, 20% less saturation or an 8% white-balance shift costs TabPFN
    0.20-0.23 (0.919 -> 0.69-0.72) and gradient boosting 0.15-0.19; the rule moves less than 0.03. A shift the size of
    the 61-photo batch takes TabPFN to 0.657, close to the 0.59 it got on those photos (F29).
- **Takeaways:**
  - The Ghana scores are not clean anemia numbers. A child at Hb 10.9 and one at 11.1 have the same eyelid, yet every
    Ghana model separates children near the cutoff better than it separates everyone. A trace tied to the folder can
    do that; paleness can't.
  - It is a mix, not all trace: the Ghana photos still carry some Hb (weakly for eyelids, more for palms). We can't
    say how much of the 0.92 is which.
  - "A table model beats vision models" doesn't hold as a claim about anemia. On Ghana every model reads the trace and
    TabPFN reads it best; where there is no trace, TabPFN matches the rule (India, Italy) or adds a little (Moscow).
  - TabPFN's numbers are tied to the exact colors of one batch: everyday color changes cost it 0.2.
  - Not cheating or a leak on our side: the held-out scores are real, but the trace is in every published photo,
    training and test alike, so the leak checks of F29 can't see it.
  - The jump test is a cheap shortcut check anyone can run on a dataset with Hb.
  - Unknown: what exactly differs between the folders (camera, session, cut-out, saving), and whether the 61 photos of
    F29 are new children or edited copies (two quick scratch matchers, keypoints and cut-out outline, failed their own
    controls).
- **Run it yourself:** `python -m src.eyelid.handling_check` (~1 h on CPU) and `python -m src.eyelid.handling_check_vision`
  (a minute). Reports: [`reports/stage30_handling_check.md`](reports/stage30_handling_check.md),
  [`reports/stage30_vision.md`](reports/stage30_vision.md).

### F31. A Ghana number without the folder trace? (Stage 31). Not with this data; the Hb-number models carry the trace too.
- **Question:** F30 showed part of every Ghana score is a trace of how the folders were handled. How much real anemia
  signal is left? And do the Hb-number models (F19, F22) carry the trace too?
- **What we did:** every answer fixed in `configs/trace_free.yaml` before running; same six datasets, cutoffs (Ghana
  **Hb < 11.0 g/dL**; India and Italy **WHO by sex, < 12.0 women, < 13.0 men**; Moscow **< 12.0 g/dL**) and folds as F30.
  - Within-folder model: a trace shared by a whole folder drops out of differences inside the folder. In each training
    fold we took each label group's average out of the 29 numbers and out of Hb, fit ridge on what was left, and scored
    the test photos as they are (their label is never used). Trusted only if it keeps the signal on at least 2 of the
    3 clean datasets.
  - The jump test (F30) on the Hb-number models: ridge, random forest and gradient boosting (Stage 16), and the TabPFN
    regressor (Stage 22).
  - The jump test on each of the 29 color numbers alone.
  - Transfer: Ghana eyelids to India and Italy, Ghana nails to Moscow.
  - A scratch look at the files: both Ghana eyelid folders have the same PNG layout, resolution tag and black under the
    transparent pixels, so the trace is in the colors, not in how the files were saved.
- **Answers:**
  - **No trace-free Ghana number.** The within-folder model doesn't jump anywhere, but it also loses a lot on the clean
    datasets (Italy -0.41, Moscow -0.25, India -0.15 against ordinary ridge on Hb), so by our rule it can't be trusted.
    Its Ghana scores: eyelids 0.50, nails 0.50, palms 0.53. Taking out each folder's average throws away real pallor
    too, because inside one folder Hb only spans a few g/dL.
  - **The Ghana Hb number is mostly the folder.** None of the four Hb models follows Hb inside the Ghana anemic group
    (eyelids -0.15 to 0.00, every range crosses 0), while in India (-0.26 to -0.32) and Moscow (-0.45 to -0.63) they
    do. Gradient boosting, Stage 16's best (error 1.378 g/dL), jumps at the cutoff on the eyelids (0.72 near the cutoff
    vs 0.62 for a steady signal, p = 0.024), and so does the TabPFN regressor (0.80 vs 0.63; also on the palms). So the
    1.38 vs 1.71 of F19 comes mostly from getting the folder right, not from reading how low the Hb is.
  - **Where the trace sits:** 6 of the 29 numbers jump on their own on the Ghana eyelids: saturation (mean, median),
    blue (mean, median), green (median) and R/G. They separate the groups better near the cutoff (0.67-0.71) than on
    everyone (0.52-0.62), which a steady signal can't do. One on the nails (hue), none on the palms, none on India,
    Italy or Moscow.
  - **Not one hospital:** with one of the 10 hospitals held out at a time, TabPFN still jumps (0.922 near the cutoff vs
    0.71).
  - **Transfer:** TabPFN trained on Ghana nails scores 0.30 on Moscow nails, worse than a coin flip; the within-folder
    model 0.67, level with the rule (0.70). On the eyelids, no gain (India 0.51, Italy 0.47).
- **Takeaways:**
  - We can't split the Ghana 0.92 into anemia and trace with these photos; the honest statement stays "a mix".
  - Tell Maria: the Ghana Hb results (F19, F22) carry the same trace. The Hb numbers in Moscow and India are the clean
    ones.
  - What TabPFN learns from Ghana nails runs backwards on another dataset (0.30 on Moscow).
- **Run it yourself:** `python -m src.eyelid.trace_free` (~25 min on CPU) and `python -m src.eyelid.trace_free_hb_models`
  (~8 min). Reports: [`reports/stage31_trace_free.md`](reports/stage31_trace_free.md),
  [`reports/stage31_hb_models.md`](reports/stage31_hb_models.md).

### F32. The same small photo edits on every dataset, and training on edited copies (Stage 32). Fragile only on Ghana.
- **Question:** in F30 a small color edit cost TabPFN 0.2 on the Ghana eyelids, and we read that as part of the trace.
  If TabPFN is just as fragile on the clean datasets, that reading is too strong. And does training on color-edited
  copies make it robust?
- **What we did:** answers fixed in `configs/edit_robustness.yaml` before running; same datasets and cutoffs as F31.
  - F30's seven edits (brightness x0.85 and x1.15, gamma 0.8 and 1.25, 20% less saturation, two 8% white-balance
    shifts) on the test photos of every dataset; models trained on the photos as they are.
  - A second TabPFN trained on each training photo plus 3 copies with a random white balance, brightness, gamma and
    saturation change (copies stay in their photo's fold). The test edits lie inside those ranges.
  - Fragile edit: AUROC drops by at least 0.05 (range below 0) while the rule moves less than 0.02. Fragile dataset: at
    least 4 of the 7 edits.
- **Answers:**

| Dataset | TabPFN | Edits that break it | Average cost | With edited copies | Edits that break it | Average cost | Still jumps? |
|---|---|---|---|---|---|---|---|
| Ghana eyelids | 0.919 | **6 of 7** | 0.21 | 0.878 | 4 | 0.05 | yes |
| Ghana nails | 0.754 | **4 of 7** | 0.08 | 0.740 | 3 | 0.04 | yes |
| Ghana palms | 0.866 | **7 of 7** | 0.12 | 0.865 | 4 | 0.06 | yes |
| India eyelids | 0.648 | 0 | 0.02 | 0.575 | 0 | +0.03 | - |
| Italy eyelids | 0.909 | 0 | 0.00 | 0.901 | 0 | 0.00 | - |
| Moscow nails | 0.823 | 0 | 0.02 | 0.797 | 0 | 0.01 | - |

  - Logistic regression and the rule hold everywhere (at most 1 edit breaks logistic regression, on India).
- **Takeaways:**
  - The fragility is a sign of the Ghana trace, not of TabPFN: the same model on the same numbers shrugs off the same
    edits on India, Italy and Moscow. This was a test the F30 reading could have failed.
  - Edited copies make TabPFN much steadier on Ghana for a small price, but the jump stays, so the trace isn't a plain
    color cast over the whole photo.
  - On clean data the copies aren't needed and cost a little (India -0.07, range crosses 0).
- **Run it yourself:** `python -m src.eyelid.edit_robustness` (~65 min on CPU). Report:
  [`reports/stage32_edit_robustness.md`](reports/stage32_edit_robustness.md).

### F33. Measure the step at the cutoff and take it out? Plant a fake trace to test it (Stage 33). The step confirms the Ghana jump; taking it out isn't trustworthy.
- **Question:** F31 couldn't build a model that avoids the Ghana trace. Can we measure it instead (the step in the score
  exactly at the Hb cutoff, which pallor can't make) and take it out? And would our checks catch a trace we put in
  ourselves?
- **What we did:** answers fixed in `configs/trace_split.yaml` before running; same datasets and cutoffs as F31.
  - Each score on one scale (ranks mapped to a normal curve), fit as a straight line in Hb plus a step at the cutoff,
    on everyone within 2 g/dL of it. Trace-free score = the score minus the step for every anemic person.
  - Planted trace: on India, Italy and Moscow the anemic photos only get one mild edit (gamma 0.9, 10% less
    saturation, or a 4% white-balance shift), as if that folder had been saved differently; models retrained.
  - Trusted for a model if taking the step out leaves the clean AUROC within 0.05 on 2 of 3 clean sets and gets back
    the unplanted AUROC (within 0.05) for at least 2 in 3 plants.
  - Changed after a smoke run, before the real run: one slope for both sides, a two-sided clean check (stricter), and
    the dummy not split (its tied scores).
- **Answers:**

| Dataset | TabPFN | Step at the cutoff (score SDs) | Trace-free AUROC |
|---|---|---|---|
| Ghana eyelids | 0.919 | **1.55** [1.15, 1.95] | 0.41 |
| Ghana nails | 0.754 | **0.94** [0.27, 1.56] | 0.48 |
| Ghana palms | 0.866 | **1.80** [1.22, 2.35] | 0.29 |
| India eyelids | 0.648 | -0.27 [-1.44, 0.85] | 0.72 |
| Italy eyelids | 0.909 | 0.09 [-1.22, 1.36] | 0.90 |
| Moscow nails | 0.823 | -0.01 [-0.48, 0.46] | 0.82 |

  - Gradient boosting has the same Ghana steps (1.08, 0.78, 1.75); no clean-set step differs from 0.
  - Planted trace: the edit lifts the trained models by **0.06-0.27** on India and Moscow (Italy, already at 0.91:
    0.01-0.06). The jump test catches **10 of 24** (Moscow 7 of 11, India 3 of 11, Italy 0 of 2).
  - Taking the step out gets the unplanted AUROC back for 0-1 of each model's 4-7 plants; on Moscow it lands 0.01-0.10
    too low. **Trusted for no model.**
- **Takeaways:**
  - The step is a second, simpler jump check, and it agrees with F30: a big step on all three Ghana sets, none on the
    clean ones.
  - Still no trace-free Ghana number. Taking the step out removes too much (it would put TabPFN on Ghana below
    chance), so the share of the Ghana score that is real stays unknown.
  - A small difference in how one folder was saved can add as much as TabPFN's whole Ghana lead over the rule (0.27).
  - The jump test misses most planted traces on the small sets, so India's and Italy's pass in F30 is weak evidence;
    Moscow (250 people) is the best-tested clean set.
- **Run it yourself:** `python -m src.eyelid.trace_split` (~30 min on CPU). Report:
  [`reports/stage33_trace_split.md`](reports/stage33_trace_split.md).

### F34. Where in the photo is the Ghana trace? (Stage 34). All through the cut-out, not at its edge.
- **Question:** the Ghana cut-outs give the label away with no color (0.57-0.62, F30) and by their shape (0.69, F25).
  If each folder was cut out by a different hand or tool, the trace would sit at the edge of the cut-out. Does it?
- **What we did:** answers fixed in `configs/trace_location.yaml` before running.
  - Core = the half of each cut-out's pixels farthest from its edge; ring = the other half. The 29 color numbers on
    each, models retrained, F30's jump test on each.
  - F30's no-color numbers (size, shape, soft edge, sharpness, file bytes), each alone through the jump test.
  - Moscow left out: its nail boxes sit inside one full photo, so there is no cut-out edge.
- **Answers:**

| Dataset | TabPFN, whole cut-out | Core only | Ring only | Core jumps? | Ring jumps? |
|---|---|---|---|---|---|
| Ghana eyelids | 0.919 | 0.900 | 0.893 | yes | yes |
| Ghana nails | 0.754 | 0.724 | 0.743 | yes | yes |
| Ghana palms | 0.868 | 0.873 | 0.836 | yes | yes |
| India eyelids | 0.642 | 0.636 | 0.585 | no | no |
| Italy eyelids | 0.910 | 0.903 | 0.913 | no | no |

  - The core alone keeps the clean signal on India and Italy for every model (changes within 0.01).
  - No single no-color number jumps on the Ghana eyelids or nails; one does on the palms (photo height: 0.66 near the
    cutoff, 0.50 on everyone), a lone flag with no clear story.
- **Takeaways:**
  - The trace isn't who drew the cut-out: it is in the colors all through the region.
  - With F31 (saturation, blue and green) and F32 (not a plain color cast), it fits a difference in how or when each
    folder's photos were taken or processed, which the data doesn't record.
- **Run it yourself:** `python -m src.eyelid.trace_location` (~22 min on CPU). Report:
  [`reports/stage34_trace_location.md`](reports/stage34_trace_location.md).

### F35. TabPFN on Ghana with redness only, through every trace check (Stage 35). No input set passes.
- **Question:** TabPFN only ever sees color numbers (no hospital, file name, child ID or photo size). Is there a set of
  color numbers, redness only for example, where it carries no folder trace? And is TabPFN on redness better than the
  plain redness rule?
- **What we did:** answers fixed in `configs/redness_check.yaml` before running. Ghana only (eyelids 388 photos, nails
  and palms 302 linked children), **Hb < 11.0 g/dL** everywhere. Logistic regression and TabPFN on: mean redness alone;
  4 redness numbers (no red/green ratio); 5 redness numbers; all 29 minus the numbers that jump on their own; all 29.
  Every score through F30's jump test, F33's step and F32's seven photo edits. "No trace we can find" = passes all three.
- **Answers:**

| Ghana set (Hb < 11.0 g/dL) | Rule | Forced model, mean redness | Free gradient boosting, mean redness | TabPFN, 5 redness numbers | TabPFN, all 29 |
|---|---|---|---|---|---|
| Eyelids | 0.650 (trace) | 0.621 (trace) | 0.606 (trace) | 0.876 (trace) | 0.919 (trace) |
| Nails | 0.493 | 0.453 | 0.542 | 0.697 (trace) | 0.754 (trace) |
| Palms | 0.597 | 0.580 | 0.697 (trace) | 0.779 (trace) | 0.866 (trace) |

  - **No TabPFN input set passes, on any Ghana set.** Each set that lifts TabPFN above the rule also jumps at the cutoff,
    has a step (0.65-1.80 score SDs) and breaks under 3-7 of the 7 edits (palms: 6-7 of 7 even with mean redness alone).
  - Given only mean redness, TabPFN matches the rule on eyelids (0.640 vs 0.650).
  - Taking out the numbers that jump on their own doesn't help: eyelids 0.917, still jumps.
  - On eyelids even the rule fails (jump p = 0.009, step 0.55 [0.04, 1.09]): the trace is in redness itself there.
  - Logistic regression passes on nails and palms. Nails, all 29: **0.637 [0.574, 0.699]**, +0.144 [0.049, 0.237] over
    the rule, the one Ghana score that passes every check and beats the rule. Palms 0.651, not clearly above the rule.
  - With one hospital held out the eyelid picture is the same (TabPFN redness 0.877, jumps).
- **Takeaways:** there is no clean TabPFN number on Ghana, with any input. TabPFN's lead over the rule comes together
  with the trace. Passing is not proof (the jump test caught 10 of 24 planted traces, F33).
- **Run it yourself:** `python -m src.eyelid.redness_check` (~50 min on CPU). Report:
  [`reports/stage35_redness_check.md`](reports/stage35_redness_check.md).

### F36. A model where less red can only mean more anemic (Stage 36). It matches the rule; the extra is the trace.
- **Question:** if a model is only allowed to say "less red = more anemic", does it beat the rule on Ghana, and does
  it carry the trace?
- **What we did:** answers fixed in `configs/forced_redness.yaml` before running; same Ghana sets, cutoff (**Hb < 11.0
  g/dL**) and checks as F35. Gradient boosting, free and with a monotonic constraint on mean a*, median a* and R/G, on
  1, 2 and 3 redness numbers and on all 29 (the other 26 free).
- **Answers:**
  - **The forced model matches the rule, no more:** eyelids 0.621-0.624 (rule 0.650), nails 0.453-0.495 (0.493), palms
    0.575-0.580 (0.597). On nails and palms it shows no trace (no jump, no step, 0 of 7 edits break it).
  - **The free model on the same redness numbers gains 0.10-0.13 on nails and palms, and that gain comes with the
    trace:** palms 0.693-0.714 (jumps, step 1.1-1.4, 6-7 of 7 edits break it); nails 0.607-0.624 (jumps).
  - With all 29 numbers the constraint changes nothing (eyelids 0.849 vs 0.846, both jump): the free numbers carry it.
  - On eyelids nothing passes, the forced model included (the rule itself jumps there).
- **Takeaways:** what flexible models add over the rule on Ghana is not "paler = more anemic". The honest Ghana
  numbers are the rule and the simple models: about 0.60-0.65. Nail redness carries no signal in these photos (0.49).
- **Run it yourself:** `python -m src.eyelid.forced_redness` (~3 min). Report:
  [`reports/stage36_forced_redness.md`](reports/stage36_forced_redness.md).

### F37. TabPFN or the 29 numbers? Age and sex, and the numbers that shouldn't carry paleness (Stage 37). Mostly the numbers; age and sex add nothing; the "not useful" numbers score.
- **Question (mentor meeting 2026-11-17):** how much of TabPFN's score comes from TabPFN and how much from the 29
  color numbers ("train a logistic regression or like an MLP on these extracted 29 features")? Does adding age and sex
  help? Jaineel: if it only gets the numbers that shouldn't be useful (green, blue, anything that isn't brightness or
  redness), how does it do, and do any combinations of redness, brightness and saturation behave oddly?
- **What we did:** answers fixed in `configs/model_vs_features.yaml` before running. Ghana eyelids only (388 photos,
  165 anemic), **Hb < 11.0 g/dL**. Logistic regression, an MLP (size and L2 penalty picked by an inner CV inside each
  training fold), gradient boosting and TabPFN on the same inputs and folds; 5-fold CV x 5 repeats, and one hospital
  held out. Age (months) and sex (0/1) next to the 29 numbers; the data has no BMI. Every score through the jump test,
  the step and the seven photo edits (F35).
- **Answers:**

| Inputs (Ghana eyelids, Hb < 11.0) | Logistic regression | MLP | Gradient boosting | TabPFN |
|---|---|---|---|---|
| All 29 numbers | 0.680 | 0.870 (trace) | 0.846 (trace) | 0.919 (trace) |
| 29 + age + sex | 0.686 | 0.810 (trace) | 0.833 (trace) | 0.918 (trace) |
| Age + sex alone | 0.591 | 0.576 | 0.571 | 0.576 |
| Not useful: hue, green, blue, yellow-blue (12) | 0.641 | 0.835 (trace) | 0.788 (trace) | 0.903 (trace) |
| Paleness: redness, brightness, saturation (14) | 0.667 | 0.863 (trace) | 0.846 (trace) | 0.917 (trace) |
| Brightness + saturation, no redness (9) | 0.628 | 0.812 (trace) | 0.797 (trace) | 0.895 (trace) |

  - **Mostly the 29 numbers.** An MLP on them gets 0.870; TabPFN adds **0.049 [0.023, 0.078]** on top (0.084 with one
    hospital held out). Logistic regression stays at the rule's level (0.680 vs 0.650, +0.030 [-0.019, 0.074]). So the
    lift over the rule needs a flexible model, and every flexible-model score carries the trace (5-6 of the 7 edits
    break each one; all but one jump at the cutoff).
  - **Age and sex help no model:** TabPFN -0.001 [-0.009, 0.006]; the MLP gets worse, -0.060 [-0.089, -0.031]. Alone
    they score 0.57-0.59 (rule 0.650) with no trace, and inside the anemic group that score follows Hb (rho -0.25
    [-0.39, -0.10]).
  - **The numbers that shouldn't carry paleness nearly match the ones that should:** TabPFN 0.903 vs 0.917
    (-0.014 [-0.030, 0.001]), MLP 0.835 vs 0.863. "Weird" for both by the rule fixed before running.
  - **Combinations:** TabPFN on redness alone 0.876, brightness alone 0.785, saturation alone 0.698, but brightness +
    saturation (no redness) 0.895. Saturation alone makes even logistic regression jump (step 0.73 [0.26, 1.22]).
  - Logistic regression passes every check on brightness, the not-useful set, paleness and three of the combinations,
    but none of them beats the rule (best: redness + brightness, +0.018 [-0.014, 0.051]).
- **Takeaways:** TabPFN is the best of four models at picking up what the 29 numbers carry, by about 0.05. What lifts
  Ghana above the rule is in every color number, including the ones that shouldn't carry paleness, so it fits the
  folder trace (F30) better than paleness. Age and sex don't change that.
- **Run it yourself:** `python -m src.eyelid.model_vs_features` (~2.5 h on CPU). Report:
  [`reports/stage37_model_vs_features.md`](reports/stage37_model_vs_features.md).

### F38. Severity on the x axis, and a fixed number of blood tests (Stage 38). No score is better for the sickest; TabPFN fills 20 tests well.
- **Question (mentors' advice to the whole group, 2026-11-17):** "On the X axis severity, on the Y axis AUROC ... maybe
  top 10% we get something surprisingly high", and "100 people come in. You only have capacity to test 20."
- **What we did:** answers fixed in `configs/presentation_numbers.yaml` before running. Ghana eyelids (388, 165
  anemic), models trained on **Hb < 11.0 g/dL**, 5x5 CV; the same held-out score checked against Hb < 10, 9, 8 and 7
  (each against everyone else), and the share of anemic children among the top 20% and 30% by score.
- **Answers:**

| Score | Hb < 11 (165) | Hb < 9 (68) | Hb < 8 (34) | Hb < 7 (16) | 20 tests per 100: anemic caught |
|---|---|---|---|---|---|
| Rule (untrained) | 0.650 | 0.546 | 0.572 | 0.570 | 29% |
| Logistic regression | 0.680 | 0.585 | 0.669 | 0.662 | 29% |
| TabPFN | 0.919 (trace) | 0.744 | 0.749 | 0.772 | 43% |
| Random / perfect | 0.5 / 1.0 | | | | 20% / 47% |

  - **No score picks out the sickest 9% better.** TabPFN falls the most: Hb < 8 minus Hb < 11 = -0.170 [-0.235,
    -0.108], about as much as a score that knows only the anemic and healthy folder falls on these children (1.00 ->
    0.82, -0.19). The rule
    (-0.079) and logistic regression (-0.011) don't clearly change. A score that follows Hb should rise there, since
    far fewer children sit near 8 g/dL than near 11.
  - **20 tests per 100 children:** TabPFN catches 43% [38%, 48%] of the anemic ones, +0.139 [0.073, 0.189] over the
    rule; 30 tests: 62% vs 42%. Logistic regression ties the rule (29%).
  - Of the 34 sickest children, 20 tests reach 38% with TabPFN and 56% with logistic regression (ranges overlap).
- **Takeaways:** TabPFN fills a fixed number of tests well, but it doesn't rank the sickest first, the opposite of
  what paleness would do. With F30 and F37, the Ghana TabPFN score is best read as "which folder", not "how anemic".
- **Run it yourself:** `python -m src.eyelid.presentation_numbers` (~10 min on CPU). Report:
  [`reports/stage38_presentation_numbers.md`](reports/stage38_presentation_numbers.md); chart
  [`reports/figures/stage38_severity_auroc.png`](reports/figures/stage38_severity_auroc.png); the slide outline that
  uses it: [`docs/talk_outline_image_side.md`](docs/talk_outline_image_side.md).

### F39. Does any Ghana body site's score read how sick a child is, beyond the folder? (Stage 39). No, on any site.
- **Question:** F38 showed TabPFN's eyelid score falling at lower cutoffs about as much as a score that knows only
  the anemic or healthy folder. Is that exact, and is any body site different? F30 found the palm score follows Hb a
  little inside each group, so the palm was the best chance.
- **What we did:** answer fixed in `configs/severity_reference.yaml` before running. Ghana eyelids (388) and the 302
  children with a linked eyelid, nail and palm photo (Hb from the CP-AnemiC sheet; the link is probable), models
  trained on **Hb < 11.0 g/dL**, 5x5 CV. Every score next to two lines matched to its own AUROC at Hb < 11: a score
  that knows only the folder (falls at lower cutoffs) and one that follows Hb (the jump test's steady score; rises).
  "Reads severity" = AUROC at Hb < 8 minus the folder line has its CI above 0.
- **Answers (TabPFN, Hb < 8):**

| Ghana set | Hb < 11 | Hb < 8 | knows only the folder | follows Hb | minus the folder line |
|---|---|---|---|---|---|
| Eyelids (388) | 0.919 | 0.749 | 0.764 | 0.958 | -0.015 [-0.076, 0.043] |
| Eyelids, linked (302) | 0.913 | 0.800 | 0.764 | 0.961 | +0.036 [-0.030, 0.101] |
| Nails, linked (302) | 0.754 | 0.556 | 0.663 | 0.835 | -0.107 [-0.213, -0.009] |
| Palms, linked (302) | 0.866 | 0.713 | 0.734 | 0.928 | -0.020 [-0.099, 0.057] |

  - **No score reads severity beyond the folder, on any site** (rule, logistic regression and TabPFN alike). The
    closest is logistic regression on the linked eyelids: +0.107 [-0.009, 0.215].
  - TabPFN stays within 0.05 of the folder line down to Hb < 8 on eyelids and palms (chart), about 0.21 below a score
    that follows Hb with the same AUROC. The palm is no different.
  - On the nails TabPFN falls below even the folder line: the sickest children score lower than the folder alone
    would give them.
- **Takeaways:** on every Ghana body site, the scores carry the anemic or healthy label and nothing about how low Hb
  is. A clean, simple check to add to the paper next to the jump test: one line per score, no simulation needed for
  the folder line.
- **Run it yourself:** `python -m src.eyelid.severity_reference` (~35 min on CPU). Report:
  [`reports/stage39_severity_reference.md`](reports/stage39_severity_reference.md); chart
  [`reports/figures/stage39_severity_reference.png`](reports/figures/stage39_severity_reference.png).

## Data problems we found along the way

| Problem | Where | What we did |
|---|---|---|
| 212 byte-identical copies + 29 near-copies, often with different labels | CP-AnemiC | use the 388 photos with no copy; split by photo group |
| Re-cropped copies escape the detector in the "476 distinct photos" set (no-color control 0.60) | CP-AnemiC | use the 388; no-color control in every table |
| 187 mask files with a corrupt color-profile block | Eyes-Defy | read with OpenCV |
| The ViT paper's counts only come out with Italy Hb <= 10.5, though it says "< 10.5" | Eyes-Defy | noted; Part A copies the paper, everything else uses WHO cutoffs |
| 31 Italian masks on a white background instead of transparent | Eyes-Defy Italy | fixed in `src/eyelid/roi.py`, reran Parts C and E; conclusions held (rule 0.808 -> 0.835) |
| Italy's anemic patients almost all in the first photo batch (21/31 vs 2/91) | Eyes-Defy Italy | report the early batch alone as the fair Italy check |
| Each palm and nail photo stored 6-12 times (copies and rotations); no Hb, age or link to the eyelid photos | Ghana palm and nails | one row per photo; linked to the eyelids in F24 through a third set |
| The conjunctiva set re-uses 414 CP-AnemiC photos under nail- and palm-style names, undocumented | Ghana conjunctiva | matched by pixels to get Hb; the file-number link tested by color agreement (F24) |
| In the 302 linked children, the eyelid cut-outs' size and edge softness alone predict anemia (0.69) | Ghana conjunctiva / CP-AnemiC | checked that the color models don't use it (F25); probably part of the handling trace (F30) |
| Every model's score jumps at Hb 11.0 instead of following Hb; no-color numbers alone predict the label (0.57-0.62) | CP-AnemiC and the Ghana nail and palm sets (photos stored in Anemic / Non-anemic folders) | reported as a handling trace (F30); Ghana scores are not clean anemia numbers. It sits in saturation, blue and green, not in the file format; it holds with a hospital held out, survives color-edited training copies, and runs all through the cut-out, not at its edge (F31, F32, F34) |

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
- TabPFN on the 29 color numbers in Ghana children (0.92 within the dataset; 0.91 with hospitals held out), though
  partly a handling trace (F30).
- Palm photos of Ghana children (0.91 with TabPFN; it carries some Hb inside each group, but also jumps at the cutoff, F30).
- One model for all three countries: as good as a model per country, and fair after per-country standardization.
- The Hb number inside a dataset for adults (Moscow and India: it follows Hb inside each group), with age and sex
  helping; in Ghana the 1.38 vs 1.71 g/dL is mostly the folder (F31).
- Photo-quality numbers as a shortcut check (glare flags Italy's anemic batch at 0.89).
- A vision model (DINOv2) cropping the eyelid from the full photo: nearly as good as the hand mask in Italy.
- TabPFN for the Hb number on the Moscow nails (error 1.65 -> 1.59 g/dL).
- Adding the same child's nail and palm for the weaker models in Ghana (+0.06 to +0.09; F24).
- Fairness by sex and age when combining the body sites: no gap widens (F25).
- The no-color control on combined photos: it caught a new eyelid shortcut that the color models turned out not to use
  (F25).
- The jump test at the cutoff as a shortcut check: it flags all three Ghana sets and passes India, Italy and Moscow (F30).
- Small photo edits as a second check: they break TabPFN only where the jump test flags a trace (F32).
- Training TabPFN on color-edited copies: the Ghana eyelid edit cost falls from 0.21 to 0.05, for 0.04 (F32).
- The step in the score at the cutoff as a second jump check: big on all three Ghana sets, none on the clean sets (F33).
- Planting a fake folder trace in clean data, to test our own checks (F33).
- Putting other models on TabPFN's exact inputs, to split the model from the features: an MLP gets 0.87 of TabPFN's
  0.92 (F37).
- The numbers that shouldn't carry paleness as a check: they nearly match the paleness numbers on Ghana, a sign of
  the trace (F37).
- A fixed number of blood tests as the clinic number: TabPFN 43% of the anemic children with 20 tests per 100, vs
  29% for the rule (F38).
- A folder-only reference line on the severity curve, as a trace check: it needs no simulation and shows at a
  glance that the Ghana scores carry only the label (F39).

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
- Age and sex next to the photo, for the anemic yes/no.
- A retake gate on photo quality, or removing glare (no gain on these curated cut-outs).
- The Hb number across datasets (the scale shifts).
- Adding nail and palm for TabPFN (+0.02, CI crosses 0), or for the Hb number (F24).
- Pretend children (random same-label partners) as a stand-in for real ones: they overstate the gain 3x (F24),
  because one child's real photos make the same mistakes (F25).
- Fine-tuning DINOv2 (0.690 vs 0.672 frozen; F26).
- Fingernail and skin numbers together for a trained model (TabPFN +0.002; F28).
- TabPFN on a differently processed batch of Ghana eyelid photos: 0.59, below the plain rule (0.67; F29).
- DINOv2 or VisionFM features next to the color numbers: TabPFN gets 0.01-0.02 worse (F26).
- TabPFN and gradient boosting under small color edits: 0.15-0.26 lost, while the rule holds (F30); only on the
  Ghana sets (F32).
- A within-folder model, to get a Ghana number without the trace: it loses 0.15-0.41 on clean data too (F31).
- The Hb number in Ghana: no model follows Hb inside the anemic group, and the best one jumps at the cutoff (F31).
- Taking the step at the cutoff out of the score, for a Ghana number without the trace: it removes too much from a
  planted trace (F33).
- The jump test on small sets: it misses most planted traces on India (3 of 11) and Italy (0 of 2) (F33).
- TabPFN on redness only, or without the numbers that jump on their own: still carries the trace (F35).
- Forcing "less red = more anemic": removes the trace on nails and palms but only matches the rule (F36).
- Age and sex next to the 29 numbers on the Ghana eyelids, for any of four models (TabPFN -0.001, MLP -0.060; F37).
- Logistic regression above the rule on the Ghana eyelids: it passes the checks on six color sets, but those score
  0.63-0.67, never clearly above the rule (F37).
- Any Ghana score picking out the sickest children better: TabPFN 0.92 -> 0.75 at Hb < 8 (F38).
- Any Ghana body site reading severity beyond the folder, the palm included (F39).

**Still open**
- Why the linked Ghana eyelid cut-outs' shape predicts anemia (0.69; F25).
- How many Ghana file-number matches are wrong (only the authors' sheet can say; wrong ones inflate the gain, F25).
- What exactly differs between the Ghana anemic and healthy photo folders (F30 answered F27: the score reads
  something tied to the label, not only Hb). F31-F32 narrowed it: saturation, blue and green, not the file format,
  not one hospital, not a plain color cast over the whole photo (F32), and not the cut-out edge (F34).
- The deeper eyelid fold in India (F2): real but borderline, needs new Indian photos.

## What's next

| Item | Status | Who |
|---|---|---|
| Eyelid + palm + nail for the same child | done (F24) on a probable file-number link; checks done (F25) | - |
| DINOv2 fine-tuned, and next to the color numbers | done (F26): no gain | - |
| Catch rate by Hb severity | done (F27): flat across bands; the score doesn't rank by Hb | - |
| Full audit of the TabPFN result | done (F29): no leak, reads redness; doesn't hold on a differently processed batch | - |
| Does the score read anemia or photo handling? | done (F30): Ghana partly handling; India, Italy, Moscow pass | - |
| Re-plan the image side after F30 | proposed, waiting for the team (F31-F32 done meanwhile) | team |
| A Ghana number without the trace | tried (F31): not possible with this data | - |
| The same photo edits on every dataset; edited training copies | done (F32): fragile only on Ghana; copies help | - |
| Measure the step at the cutoff; plant a fake trace in clean data | done (F33): the step confirms Ghana; taking it out not trusted; the jump test is weak on small sets | - |
| Where in the photo is the trace? | done (F34): all through the cut-out | - |
| TabPFN on Ghana with redness only, through every trace check | done (F35): no input set passes | - |
| A model where less red can only mean more anemic | done (F36): matches the rule; the free models' extra comes with the trace | - |
| The mentors' asks from 2026-11-17: an MLP on the 29 numbers, age and sex, the numbers that shouldn't be useful | done (F37): mostly the numbers, TabPFN +0.05; age and sex nothing; the not-useful numbers score 0.90 | - |
| The mentors' advice to the whole group: severity curve, a fixed number of tests, both splits, transfer as a negative result, a slide outline | done (F38): no score better for the sickest; TabPFN 43% with 20 tests; outline in `docs/talk_outline_image_side.md` | - |
| Does any Ghana body site read severity beyond the folder? | done (F39): no, on any site; TabPFN on the folder-only line | - |
| More Ghana modelling | stop: the honest Ghana numbers are the rule and simple models, 0.60-0.65 (F35-F36); from 2026-11-15 new findings are Ghana only, India and Italy for audits | - |
| Tell Maria the Ghana Hb results carry the trace (F31) | to do | Jaineel |
| A paper section on the trace and the checks (jump test, step, edit test, planted trace) | drafted: [`docs/paper_section_trace.md`](docs/paper_section_trace.md), seven checks in one table; for the team to review | team |
| Fingernails: nail and skin numbers side by side as one input | done (F28): no gain for the models, +0.05 for the untrained rule | - |
| Indian IEEE DataPort set | skipped: paid subscription | - |
| Comparison with Chemhaka et al. (survey) | reproduced by the survey side (see their meeting notes) | survey side |
| Paired photo + survey data, to test fusion for real | future | team |
| TabPFN for the Hb number (extends Stage 16) | done (F22) | - |
| Automatic cut-out from the full photo | done on Eyes-Defy (F23); raw phone photos would be the real test | team |

## Where to find things

| What | File |
|---|---|
| Full paper-style write-up | [`README.md`](README.md) |
| Answers for the mentors | [`reports/mentor_followup_eyelid.md`](reports/mentor_followup_eyelid.md) |
| The mentors' suggestions and the step log | [`docs/eyelid_followup.md`](docs/eyelid_followup.md) |
| Draft paper section on the folder trace and the checks | [`docs/paper_section_trace.md`](docs/paper_section_trace.md) |
| Stage reports with every number and range | [`reports/`](reports/) |
| Settings, seeds and cutoffs | [`configs/`](configs/) |
