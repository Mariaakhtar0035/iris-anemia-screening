# Eyelid models: what we did with the mentors' suggestions

Prepared for the next mentor meeting. This covers only the eyelid-photo advice; the survey advice belongs to the survey
side of the project. The suggestions are recorded, and every step is logged, in
[`docs/eyelid_followup.md`](../docs/eyelid_followup.md). The detailed reports are linked from each section.

**Setup used everywhere below.**
- Anemic means:
  - CP-AnemiC (Ghana, children 6-59 months): Hb < 11.0 g/dL. We use the 388 photos that have no duplicate copy.
  - Eyes-Defy (Italy and India, adults): WHO cutoffs by sex, women < 12.0 and men < 13.0 g/dL (pregnancy isn't
    recorded).
- Scores are held out with 5-fold cross-validation repeated 5 times, split by photo (one photo per person).
- Numbers are AUROC (0.5 = coin flip) with bootstrap 95% CIs from 2000 resamples.
- Every table has an untrained baseline: the "less red = anemic" rule.
- Every population also has a **no-color control**: a model that sees only image size and cut-out shape. If it scores
  well above 0.5, the photos carry a shortcut (copies, batches, cameras) and the other scores in that population can't
  be trusted on their own.

## The short version

| Question | Simple answer | Number |
|---|---|---|
| Is there a real pallor signal in eyelid photos? | **Yes, in Ghana children** | 0.85 with gradient boosting, 0.92 with TabPFN; a model blind to color: 0.54 |
| Would more photos help? | **Yes, in Ghana** | still rising: 0.72 -> 0.92 from 78 to 310 training photos |
| Does a bigger image model help? (M1, M10) | **No** | fine-tuned ViT 0.72, eye-pretrained VisionFM 0.79, vs 0.85 for 29 color numbers |
| Does TabPFN, the model you named, help? (M14) | **Yes in Ghana, not elsewhere** | Ghana 0.92 vs 0.85; India 0.65; fingernails 0.82 |
| Does TabPFN still work on a new dataset? | **No**, only the simple rule carries over | Ghana -> Eyes-Defy 0.60 (rule 0.84); reverse 0.60 (rule 0.65) |
| Can one model learn all three countries? (M8) | **Yes, as well as separate models, and it can be made fair**; still no better than the rule in India or Italy | Ghana 0.90, India 0.64, Italy 0.91 (rule 0.65, 0.67, 0.92); country gap 0.52 -> 0.27 |
| Can a new country get by with a few labelled photos? | **No**: 10-40 photos never beat the rule; Ghana needs about 80 | best at 40 photos: Ghana 0.68, India 0.68, Italy 0.93 (rule 0.65, 0.67, 0.92) |
| Can any model learn more than the simple rule in India? | **No**, too few photos | rule 0.67, best model 0.67 |
| Is the deeper eyelid fold better in India? | **Maybe**, borderline | rule 0.67 -> 0.76 (change +0.08 [+0.00, +0.18]) |
| Can Italy's score be trusted? | **No**, one early photo batch holds the anemic patients | the batch alone predicts anemia at 0.91 |
| Does debiasing help? (M5-M8) | **Fairer, not more accurate** | country gap 0.39 -> 0.10; no gain inside a country |
| What would it save a clinic? (M13) | **Who to test first, not a rule-out** | test 50% of children to catch 90% (random: 90%) |
| Does fixed lighting help? (M9, M12) | **Yes**: a clean signal on fingernails | 0.82, no shortcut, same on unseen sessions |
| Do palm or fingernail photos work in Ghana children? (M12) | **Palm yes, as well as the eyelid; nails weaker** | palm 0.91, nails 0.80, eyelid 0.92 |

**Checked for shortcuts:** trained on 9 hospitals and scored on the 10th, TabPFN still scores 0.91. Given only image
size and shape (no color) it scores 0.57. So the 0.92 is color signal, not the hospital or a leak.

## M1 and M2: exhaust each dataset, one population at a time

**You suggested:** use the biggest model we can on each population alone and see whether it can overfit. If it can't
fit one distribution, the signal isn't there.

**What we did:** a ladder of models from the untrained rule up to a fine-tuned ViT-B/16, trained on each population
alone. We report training AUROC next to held-out AUROC, and learning curves at 25-100% of the training data. Report:
[`stage9_ceiling.md`](stage9_ceiling.md).

| Population | n (anemic) | Redness rule | Best color model | DINOv2 probe | Fine-tuned ViT | VisionFM probe (eye-pretrained) | No-color control |
|---|---|---|---|---|---|---|---|
| Ghana children (CP-AnemiC) | 388 (42%) | 0.650 [0.597, 0.705] | **gradient boosting 0.846 [0.807, 0.882]** | 0.672 [0.616, 0.723] | 0.715 [0.661, 0.765] | 0.765 [0.716, 0.815] | 0.537 [0.474, 0.595] |
| India | 95 (72%) | **0.671 [0.552, 0.787]** | random forest 0.673 [0.542, 0.791] | 0.658 [0.543, 0.778] | 0.654 [0.522, 0.783] | 0.585 [0.452, 0.722] | 0.548 [0.410, 0.677] |
| Italy | 122 (19%) | 0.924 [0.856, 0.977] | logistic regression 0.920 [0.822, 0.983] | 0.879 [0.765, 0.963] | 0.919 [0.830, 0.980] | 0.927 [0.838, 0.986] | **0.751** [0.646, 0.837] |
| Italy + India pooled | 217 (42%) | 0.835 [0.779, 0.884] | random forest 0.892 [0.849, 0.935] | 0.868 [0.818, 0.916] | 0.892 [0.848, 0.930] | 0.872 [0.823, 0.916] | **0.742** [0.675, 0.804] |

**What we found:**
- **Every flexible model can overfit.** Gradient boosting and the ViT reach training AUROC 1.00 in every population.
  So the models are big enough; what limits them is the data.
- **Ghana children:**
  - The color signal is real (the no-color control is near chance).
  - Gradient boosting improves steadily with more photos: 0.72 with 78 training photos, 0.78 with 155, 0.83 with 233
    and 0.85 with 310. More photos from the same clinics would likely help.
  - The ViT does worse than gradient boosting on 29 color numbers, so a bigger model is not what's missing.
  - TabPFN on the same 29 numbers does better still: 0.919 [0.893, 0.945] (see M14).
- **India:** every model sits at about 0.65-0.67, the same as the untrained rule, with wide CIs.
  - Training AUROC is 0.95-1.00, so the models memorize the 76 training photos.
  - One lead: the deeper fold of the eyelid (the forniceal mask) works better than the palpebral one we used. The
    rule goes from 0.671 to 0.755 (paired change +0.084 [+0.001, +0.175]) and logistic regression from 0.612 to
    0.739 (+0.127 [+0.029, +0.228]); random forest doesn't change. Borderline with 95 photos.
- **Italy and pooled:** the no-color control already scores 0.74-0.75, so these numbers are inflated by a shortcut.
  The country gives it away in the pooled data and the batch does in Italy (see M4).

**Answer:** Ghana children have a real signal that a flexible model can learn, and it isn't saturated yet. India is too
small to learn more than the simple rule, and the Italy scores can't be taken at face value.

## M3: duplicate photos as augmentation

**You suggested:** copies are fine as augmentation if no copy of a test photo is in training, and if they don't hurt
transfer.

**What we did:** CP-AnemiC has 233 extra copies of its photos. We split by photo group, kept every copy of the
training photos in training, and tested on photos that have no copy. We compared that with training on one copy per
photo.

**What we found:**

| Model | One copy per photo (381 training rows) | All copies (566 training rows) |
|---|---|---|
| Random forest | 0.824 [0.783, 0.863] | 0.831 [0.791, 0.869] |
| Logistic regression | 0.680 [0.624, 0.733] | 0.662 [0.608, 0.717] |
| DINOv2 probe | 0.680 [0.624, 0.732] | 0.675 [0.620, 0.727] |

**Answer:** copies are harmless once they stay on the training side, but they add nothing.

## M4: why do the populations differ?

**You suggested:** when transfer fails, compare the populations: brightness, color, age, label balance.

**What we found** (report: [`stage10_populations.md`](stage10_populations.md)):
- **The population is obvious from the pixels.**
  - A classifier on the same 29 color numbers tells India from Italy at AUROC 0.998, and still 0.999 among non-anemic
    people only.
  - Ghana children vs Eyes-Defy adults: 0.989 from color numbers, 1.000 from DINOv2.
  - Anemia within a population is much harder: 0.61-0.68.
- **The gaps between populations are bigger than the gap from anemia.**
  - Redness (a*), anemic minus non-anemic: -3.5 in Ghana, -2.8 in India.
  - Between populations: non-anemic Ghana children are 12.0 lower than non-anemic Italians.
  - Brightness (L*) differs by 7-17 points between populations.
- **The people differ too.**
  - Ghana: children aged 2-3 on average.
  - India: adults around 33, 72% anemic.
  - Italy: adults around 49, 19% anemic, and the anemic Italians are older (65 vs 46).
- **Italy is two batches.**
  - Folders 1-31 are 21/31 anemic, while folders 32-123 are 2/91 anemic.
  - Non-anemic people in the early batch already look paler (a* 32.9 vs 40.1).
  - The batch alone predicts Italian anemia at 0.91. Inside the early batch the redness rule drops to 0.752 [0.517,
    0.937].
  - On the way we also found and fixed 31 Italian mask files with a white background.

**Answer:** each dataset has its own camera, lighting, age group and prevalence. A model can score well by recognizing
the dataset or the batch, which is far easier than seeing pallor.

## M5, M6 and M7: fairness metrics, debiasing, and making images look alike (keeping India)

**You suggested:** define parity metrics and regularize the model so its scores don't depend on the population; edit
the images to look more alike; keep India and use fairness metrics instead of dropping it.

**What we did** (report: [`stage11_debias.md`](stage11_debias.md)):
- **Fairness metrics, fixed before running:**
  - AUROC inside India and inside Italy;
  - the Italy early batch alone;
  - the worse of the two countries;
  - the rank gap between countries among people with the same anemia status;
  - the false-positive and true-positive rate gaps at the threshold that catches 90% of anemic people.
- **Methods:**
  - gray-world color correction;
  - brightness scaling;
  - per-country standardization of the color numbers;
  - brightness and contrast jitter during training;
  - reweighing so that country and anemia are independent (Kamiran and Calders);
  - a logistic regression with a fairness penalty on the score gap between countries (lambda 1, 10 and 100).
- India stays in every run.

| Method | Pooled | Inside India | Italy early batch | Rank gap, non-anemic | FPR gap | Share tested to catch 90% |
|---|---|---|---|---|---|---|
| Redness rule | 0.835 | 0.671 | 0.752 | 0.19 | 0.39 | 0.63 |
| Random forest | 0.890 | 0.687 [0.565, 0.799] | 0.752 | 0.39 | 0.75 | 0.53 |
| Random forest, brightness scaled | 0.897 | **0.739** [0.631, 0.838] | 0.771 | 0.37 | 0.74 | 0.54 |
| Fairness penalty, lambda=100 | 0.756 | 0.675 [0.557, 0.788] | 0.667 | **0.10** | **0.11** | 0.76 |
| Per-country standardized, logistic regression | 0.720 | 0.621 [0.502, 0.737] | 0.638 | **0.05** | **0.04** | 0.86 |

The CIs for every cell are in the report.

**What we found:**
- **The methods do what they claim.** The penalty and per-country standardization shrink the rank gap from about 0.4
  to 0.05-0.10, and the FPR gap from 0.75 to about 0.1.
- **But no method makes the model better inside a country.** All the within-country CIs overlap. Pooled AUROC drops,
  and more people need testing.
- **Brightness scaling is the most promising image edit:** India 0.739, at no pooled cost. The gain is inside the CI,
  though.
- **Across datasets nothing beats the untrained rule.**
  - Trained on Eyes-Defy, tested on Ghana children: rule 0.650, best debiased model 0.606.
  - Trained on India, tested on the Italy early batch: rule 0.752, random forest with jitter 0.781 [0.593, 0.943].

**Answer:** removing the shortcut makes the scores fair across countries, but it doesn't add pallor signal that the
small within-country data doesn't contain.

## M8: one model per population

**What we found:** a separate random forest per country scores 0.676 inside India and 0.710 in the Italy early batch.
One pooled model scores 0.687 and 0.752. Separate models are no better, and they widen the gaps (FPR gap 0.91).

**Update, all three countries (Stage 14, report: [`stage14_three_countries.md`](stage14_three_countries.md)):** one
TabPFN trained on Ghana, India and Italy together (cutoffs mixed by design: children Hb < 11.0 g/dL, adults WHO by
sex) scores 0.903 [0.874, 0.931] in Ghana, 0.644 in India and 0.906 in Italy, against 0.919, 0.660 and 0.917 for a
model per country on the same folds. Standardizing each country's color numbers on its own mean and SD halves the
gap between countries' healthy people (rank spread 0.52 -> 0.27) at no cost inside a country. Trained on two
countries and tested on the third, it is no better than the rule (Ghana 0.61 vs 0.65, India 0.70 vs 0.67, Italy
0.90 vs 0.92).

**Answer:** with this much data, per-country models aren't worth it. One mixed, standardized model is as good and
fairer, but diverse training data doesn't add pallor signal at our size.

**And a new country with a few labelled photos? (Stage 15, report: [`stage15_few_local.md`](stage15_few_local.md))**
Same photos and mixed cutoffs. Given 10, 20 or 40 labelled photos from the new country, trained with or without the
other two countries, no model beats the rule in any country (best at 40: Ghana 0.680 [0.629, 0.731] vs 0.650, India
0.682 vs 0.671, Italy 0.929 vs 0.924). Ghana passes the rule at about 80 photos (Stage 9 curve: TabPFN 0.72 at 78)
and reaches 0.92 at 310; India (76) and Italy (98) never pass it. The other two countries keep a small-k model near
the rule's level (India, 20 photos: 0.69 with them, 0.59 without) but add no signal of their own.

## M9 and M12: more data, including fingernails and palms

**You suggested:** more countries, keeping in mind that each one brings its own shortcut; fingernail or palm datasets
with age and sex.

**What we found** (full table in
[`docs/eyelid_followup.md`](../docs/eyelid_followup.md#step-7-more-data-m9-m12)):

| Dataset | Body site | People | Where | Access | Fit |
|---|---|---|---|---|---|
| Yakimov et al. 2024, *Scientific Data* | fingernails and skin | 250 adults | Moscow, Russia | open (figshare) | best fit for M12: new country, adults, fixed lighting. Age and sex are only in summary tables, not per patient |
| Girija et al. 2026, IEEE DataPort | conjunctiva, palms, nail beds | 1,485 blood-camp volunteers | Faridabad, India | paid subscription (we don't have one) | the largest conjunctiva set by far, but no age or sex, and 11% anemic |
| Asare et al. 2022, Mendeley | palms; fingernails | 514 and 526 distinct photos of children | Ghana | open, CC BY 4.0 | same group as CP-AnemiC; labels only, no IDs, so it can't be paired with the eyelids. We ran it (below) |
| Suner et al. 2021, Zenodo | conjunctiva | 344 emergency-department adults | Providence, USA | open, but only two CSV tables, no photos | might allow testing the redness rule on a fourth population |
| Valles-Coral et al. 2025 | nails, palm and fingertip videos | 909 young adults | Peru | not released yet | watch for release |

**Answer:**
- Our first choice was the Russian fingernail set: open, a new country, a new body site, and taken under fixed
  lighting. We ran it (below).
- Our second choice was the large Indian conjunctiva set, but it needs a paid IEEE DataPort subscription. So we ran
  the Ghana palm and fingernail sets next (below).
- Any new set would get the same duplicate audit and no-color control first, because Step 2 shows each new source
  brings its own shortcut.

**What we did with the fingernail set** (report: [`stage12_nails.md`](stage12_nails.md)):
- Anemic = Hb < 12.0 g/dL for everyone (69 of 250): the file has no sex column, so we picked one adult cutoff.
- The same 29 color numbers from the three nail boxes per photo, the same models, a no-color control (nail box
  sizes), and a model that sees only the photo session (20 sessions), to catch a batch shortcut like Italy's.

| Model | AUROC, CV by patient | AUROC, held-out sessions | Share tested to catch 90% |
|---|---|---|---|
| Rule: less red = anemic | 0.703 [0.632, 0.770] | 0.703 | 0.732 |
| Logistic regression | **0.818 [0.755, 0.872]** | **0.815 [0.752, 0.871]** | 0.608 |
| Random forest | 0.809 [0.746, 0.865] | 0.802 | 0.576 |
| Gradient boosting | 0.791 [0.730, 0.848] | 0.795 | 0.612 |
| No-color control | 0.492 | 0.491 | 0.948 |
| Session only | 0.438 | - | 0.968 |

**What we found:**
- No shortcut: the no-color control and the session-only model sit at chance.
- The signal holds on photo sessions the model never saw.
- A simple model is enough, and the learning curve is flat (0.811 with 50 training photos, 0.818 with 200).
- The untrained rule does better than on eyelids (0.70 vs 0.65-0.67).

**Answer:** when the lighting is fixed, a pallor color signal shows up cleanly and needs only a simple model. That
supports our reading of the eyelid results: the problem there is uncontrolled photos, not too small a model.

**Follow-up: making the nail photos alike (your M6 idea).** Each photo has a white board behind the hand, and the data
marks a patch of skin just behind each nail. Changes are paired (same resampled people):
- **White board as a color reference:** the untrained rule rises from 0.703 to 0.727 (+0.025 [0.002, 0.049]); the
  models gain 0.00-0.03, within noise (random forest 0.825, and 0.834 on held-out sessions). The light box already
  keeps the lighting steady, so there is little left to correct.
- **Nail minus the person's own skin hurts** (-0.11 to -0.31), because the skin pales with anemia too (skin alone
  scores 0.73-0.76). A reference only helps if it doesn't share the signal.

**What we did with the Ghana palm and fingernail sets** (report:
[`stage13_ghana_sites.md`](stage13_ghana_sites.md)):
- Each photo is stored 6-12 times (copies and rotations): merged, 514 palm and 526 nail photos.
- The only label is Anemic / Non-anemic in the file name (cutoff not stated). No Hb, age, sex or hospital, and no IDs
  linking to the eyelid photos, so we compare body sites but can't combine them per child.

| Model | Palm | Fingernails | Eyelid (for comparison) |
|---|---|---|---|
| Redness rule | 0.620 | 0.497 | 0.650 |
| Gradient boosting | 0.855 | 0.682 | 0.846 |
| **TabPFN** | **0.912 [0.886, 0.935]** | **0.798 [0.757, 0.836]** | **0.919** |
| No-color control | 0.454 | 0.566 | 0.537 |
| Naming style only (batch check) | 0.497 | 0.431 | - |

**Answer:** in Ghana children the palm carries as much signal as the eyelid, with no shortcut found; fingernail
cut-outs carry less. Combining them per child would need the authors' linking sheet. Do you think it's worth asking
them?

## M10: models pretrained on eyelid images

**What we found:** no public model is pretrained on conjunctiva photos that we could use without leakage.
- A Hugging Face anemia model (`dawahealth/anemia_screeniing`) was trained on 864 images from Ghana, Italy and India.
  Those are almost certainly our own two datasets, so it has seen our test photos. Its own validation accuracy is
  57.7%.
- The closest real candidate is VisionFM's external-eye encoder: research use only, and its pretraining photos aren't
  described. We ran it.

**What we did:** frozen VisionFM features with the same probe as DINOv2, as one more rung of the M1 ladder (table
above). The weights are a pickle file, so we loaded them in PyTorch's safe tensors-only mode.

**What we found:**
- **Eye pretraining helps compared with general image models.** In Ghana children VisionFM scores 0.765 [0.716,
  0.815], above DINOv2 (0.672) and the ViT we fine-tuned ourselves (0.715), with no fine-tuning at all.
- **But 29 color numbers still win** (gradient boosting 0.846), and VisionFM improves more slowly with more photos
  (0.719 with 78 training photos, 0.765 with 310).
- **India doesn't improve:** 0.585 [0.452, 0.722], below the untrained rule (0.671).
- **Clinic view:** to catch 90% of anemic Ghana children it tests 74% of children, vs 64% for gradient boosting.
- **Fine-tuned:** training all of VisionFM on Ghana photos, with the same recipe as our ViT, gives 0.785 [0.739,
  0.829], a little above frozen. It memorizes the training photos (training AUROC 1.00) and tests 73% to catch 90%.

**Answer:** a better starting model narrows the gap but doesn't close it, which fits M1: the limit is the data, not
the model.

## M11: vision-language models

We agree: the photos carry no text or metadata, so we didn't pursue VLMs.

## M13: beyond AUROC, what it would mean in a clinic

**You suggested:**
- use PR-AUC when the classes are imbalanced, and precision and recall at fixed thresholds;
- get the cost of a missed case vs an unneeded test from clinicians, who may want recall near 1.

**What we did:** in each population we ranked people by score and asked two questions:
- To catch 80, 90, 95 or 100% of the anemic people, what share of everyone must get a blood test?
- What share of those tested are actually anemic?

Testing in random order needs 90% of people to catch 90%. Report: [`clinical_eyelid.md`](clinical_eyelid.md), which
also has precision-recall curves against the prevalence line.

| Population (share anemic) | Model | Tested to catch 90% | Anemic among those tested | Tested to catch 100% | PR-AUC |
|---|---|---|---|---|---|
| Ghana children (42.5%) | redness rule | 0.807 [0.760, 0.897] | 0.476 [0.406, 0.534] | 1.000 [0.985, 1.000] | 0.555 [0.483, 0.645] |
| | gradient boosting | 0.644 [0.552, 0.711] | 0.596 [0.525, 0.704] | 0.979 [0.851, 0.990] | 0.780 [0.710, 0.845] |
| | fine-tuned ViT | 0.776 [0.665, 0.838] | 0.495 [0.436, 0.587] | 0.979 [0.954, 0.992] | 0.637 [0.559, 0.707] |
| | VisionFM probe | 0.742 [0.662, 0.843] | 0.517 [0.436, 0.589] | 0.982 [0.923, 0.992] | 0.693 [0.613, 0.770] |
| | VisionFM fine-tuned | 0.729 [0.639, 0.796] | 0.530 [0.464, 0.615] | 0.982 [0.956, 0.992] | 0.729 [0.658, 0.802] |
| | **TabPFN** | **0.500 [0.443, 0.570]** | **0.768 [0.673, 0.847]** | 0.827 [0.649, 0.861] | **0.874 [0.822, 0.923]** |
| India (71.6%) | redness rule | 0.874 [0.800, 0.926] | 0.759 [0.663, 0.845] | 0.947 [0.895, 0.989] | 0.846 [0.759, 0.919] |
| | random forest | 0.832 [0.758, 0.916] | 0.797 [0.678, 0.877] | 1.000 [0.947, 1.000] | 0.840 [0.751, 0.917] |
| Italy early batch (67.7%, n = 31) | redness rule | 0.871 [0.677, 0.968] | 0.778 [0.615, 0.929] | 0.871 [0.742, 0.968] | 0.810 [0.636, 0.979] |
| | random forest | 0.839 [0.677, 1.000] | 0.731 [0.548, 0.913] | 1.000 [0.839, 1.000] | 0.800 [0.613, 0.961] |

**What we found:**
- **Ghana children:**
  - TabPFN catches 90% of anemic children while testing half of all children, about 40% fewer tests than testing
    at random (gradient boosting: 64% tested, about a quarter fewer).
  - 77% of the children TabPFN sends for testing are anemic, against 42.5% overall.
- **At recall near 1, most of the saving disappears.** To catch every anemic child, most models test 96-100% of
  children; TabPFN tests 83%. The last few anemic children look no paler than healthy ones.
- **India and the Italy early batch are barely better than random.** Catching 90% means testing 83-87% of people, and
  precision sits only a little above prevalence.
- **Pooled Eyes-Defy looks better** (random forest: 51% tested to catch 90%), but most of that is the country shortcut
  from M4.

**Answer:**
- An eyelid photo can help decide who to test first when tests are rationed: in Ghana children it saves about a quarter
  to 40% of the tests at 90% sensitivity.
- It can't work as a rule-out test with recall near 1.
- Which operating point matters is a clinical question we'd like your help with (Question 1).

## M14: the models you named

**You suggested:** a list of models by email. It didn't arrive, so we used the ones named in the meeting: vision
models trained on eye images (VisionFM, M10), the biggest model we can run (ViT, M1), and tabular foundation models
such as TabPFN.

**What we did:** TabPFN v2 (open weights; newer versions need a license sign-up) on the same 29 color numbers, same
splits, as one more rung of the M1 ladder. It is pretrained on millions of synthetic small tables and classifies a new
table with no tuning.

| Data | Best before | TabPFN |
|---|---|---|
| Ghana children (388) | gradient boosting 0.846 | **0.919 [0.893, 0.945]** |
| India (95) | rule 0.671 | 0.648 [0.514, 0.766] |
| Italy (122) | rule 0.924 | 0.909 [0.811, 0.975] |
| Fingernails (250) | logistic regression 0.818 | 0.823 [0.764, 0.876] |

**What we found:**
- **Ghana: the best eyelid model so far**, still rising with more photos (0.72 with 78 training photos, 0.92 with 310).
- **India: no gain**, still at the rule's level.
- **Fingernails: no gain** over logistic regression; under fixed lighting a simple model is already at the limit.
- **Shortcut checks pass** (report: [`tabpfn_checks.md`](tabpfn_checks.md)): leave one hospital out 0.913 [0.885,
  0.940] (gradient boosting 0.819); size and shape only, no color, 0.565 [0.507, 0.622]; hospital alone 0.549.
- **Across datasets it doesn't hold** (report: [`tabpfn_transfer.md`](tabpfn_transfer.md)): trained on Ghana
  children and applied unchanged to the Eyes-Defy adults it scores 0.602 [0.528, 0.676], against 0.835 for the
  untrained rule; Italy 0.651 (rule 0.924), India 0.658 (rule 0.671). The reverse gives 0.595 [0.538, 0.649] (rule
  0.650).

**Answer:** the model matters for the color numbers in Ghana, where there are enough photos. It can't create signal
where the data is too small (India), and what it learns in one dataset doesn't carry over to another.

## Questions for you

1. **Clinical target.** Which miss rate would a clinic accept: 10%, 5% or none? That decides which row of the M13
   table matters. Do you know a clinician we could ask?
2. **Controlled photos.** Fingernails in a light box give a clean 0.82, and the white board in each photo adds a
   little more. Our eyelid photos have no such reference. Is it worth asking for eyelid photos with a white or color
   card in the frame, or finding a way into the larger Indian conjunctiva set (1,485 people, paid subscription)?
3. **Models.** TabPFN on color numbers is now our best (0.92 in Ghana); eye-pretrained VisionFM reaches 0.79 even
   fine-tuned. Were there other models you meant for the list?
4. **Eyelid region.** The deeper fold (forniceal) did better in India for the rule and logistic regression, but only
   just (95 photos). Is that consistent with what you've seen?