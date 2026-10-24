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
| M1 | Exhaust each dataset before anything else. Use the biggest model you can and see whether it can overfit. If a complex model can't fit data that all comes from one distribution, the signal isn't there. Only then worry about transfer and bias. | Step 1 (Stage 9): model ladder, training vs held-out AUROC, learning curves | done |
| M2 | Train on each population separately and see how well that does. | Step 1: CP-AnemiC, India and Italy each on their own | done |
| M3 | Leakage must be prevented. Duplicate photos are fine as augmentation if they never reach the test set and don't hurt transfer. | Step 1: train with the duplicate copies kept inside the training fold, vs without them | done |
| M4 | When transfer fails, compare the populations to find out why (brightness, color, age, label balance). | Step 2 (Stage 10): population comparison and a "which country is this?" classifier | done |
| M5 | The country acts as a shortcut. Look at recent debiasing and fairness work: define parity metrics, then regularize the model so its score distribution doesn't depend on the population. | Step 3 (Stage 11): fairness metrics, reweighing, fairness-penalized model | done |
| M6 | Edit the images so they look more alike, for example by changing the brightness or shifting the color distributions. | Step 3: brightness and color normalization, color-jitter augmentation | done |
| M7 | Don't drop India from the data. Use fairness metrics instead. | Step 3: per-country and worst-country AUROC | done |
| M8 | Separate models per population are an acceptable fallback. Big models trained on diverse data can learn every population, as Google's skin model eventually did. | Step 3: per-country models vs one pooled model | done |
| M9 | More countries make the data more diverse, but each new country brings its own shortcut, so debiasing is still needed. | Step 7: data search, plus the write-up | searched (Step 7); fingernail set downloaded (Step 8), waiting to be unzipped |
| M10 | Look for vision models pretrained on eyelid images and reuse their pretraining. | Step 5: search, then ask Jaineel before downloading anything | approved (Step 8); weights download pending |
| M11 | VLMs don't fit this problem, because the images come with no text or metadata. | Recorded, not pursued | done |
| M12 | Look for fingernail or palm datasets that include metadata (age, sex). | Step 7: data search | searched (Step 7); fingernail set downloaded (Step 8), waiting to be unzipped |
| M13 | AUROC alone isn't enough. Use PR-AUC when the classes are imbalanced, and look at precision and recall at specific thresholds. The cost of a false positive vs a false negative has to come from clinicians, who might want recall near 1. Put that discussion in the paper. | Step 4: "share tested to catch X% of anemic people" table and PR curves | done |
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
  - **Fine-tuned ViT-B/16 (1 seed, 30 epochs, flips plus brightness/contrast jitter):** training AUROC 1.00 everywhere.
    Held out: CP-AnemiC 0.715 [0.661, 0.765], India 0.654 [0.522, 0.783], Italy 0.919 [0.830, 0.980], pooled 0.892
    [0.848, 0.930]. Without jitter, pooled is 0.898 [0.853, 0.937], so the jitter makes no difference. On CP-AnemiC the
    ViT is clearly worse than gradient boosting on 29 color numbers (CIs don't overlap). It took about 2 h on an M1.
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

### Step 4: what the models would mean in a clinic (M13)
- **Why:** the mentors said AUROC alone isn't enough. They asked for PR-AUC, precision and recall at thresholds, and the
  cost of a miss vs an unneeded test, which clinicians have to set.
- **Ran:** `python -m src.eyelid.clinical`. Settings in `configs/clinical.yaml`. Report: `reports/clinical_eyelid.md`,
  figure `reports/figures/clinical_pr_curves.png`. New metrics in `src/common/metrics.py`:
  `share_tested_at_sensitivity` and `precision_at_sensitivity` (test:
  `tests/test_metrics.py::test_share_tested_and_precision_at_sensitivity`).
- **Setup:**
  - Stage 9's held-out scores.
  - For each population: what share of everyone has to be tested to catch 80, 90, 95 or 100% of the anemic people,
    and what share of those tested are anemic.
  - Testing in random order needs 90% of people to catch 90%.
- **Results:**

| Population (anemic) | Model | Tested to catch 90% | Anemic among tested | Tested to catch 100% | PR-AUC |
|---|---|---|---|---|---|
| CP-AnemiC (42.5%) | redness rule | 0.807 [0.760, 0.897] | 0.476 | 1.000 | 0.555 |
| | gradient boosting | **0.644 [0.552, 0.711]** | 0.596 | 0.979 | 0.780 |
| | fine-tuned ViT | 0.776 [0.665, 0.838] | 0.495 | 0.979 | 0.637 |
| India (71.6%) | redness rule | 0.874 [0.800, 0.926] | 0.759 | 0.947 | 0.846 |
| | random forest | 0.832 [0.758, 0.916] | 0.797 | 1.000 | 0.840 |
| Italy early batch (67.7%, n = 31) | redness rule | 0.871 [0.677, 0.968] | 0.778 | 0.871 | 0.810 |
| | random forest | 0.839 [0.677, 1.000] | 0.731 | 1.000 | 0.800 |

- **Answer to M13:**
  - In Ghana children the best model saves about a quarter of blood tests at 90% sensitivity.
  - At recall near 1, every model tests 96-100% of people, so there's no saving.
  - India and the Italy early batch are close to random.
  - So an eyelid photo can prioritize who to test, but it can't rule anyone out. The clinical operating point is a
    question for the mentors and for clinicians.

### Step 5: eyelid-pretrained models (M10)
- **Why:** the mentors suggested reusing a model that was already pretrained on eyelid images.
- **What we searched:** Hugging Face, GitHub and papers, for "conjunctiva", "eyelid", "anemia" and "external eye"
  checkpoints. Nothing was downloaded.
- **Found:**

| Model | What it is | License, access | Verdict |
|---|---|---|---|
| `dawahealth/anemia_screeniing` (Hugging Face) | an anemia classifier trained on 864 conjunctiva images from Ghana, Italy and India | Apache-2.0, open | **rejected**: those are almost certainly CP-AnemiC and Eyes-Defy, so it has already seen our test photos, and its own validation accuracy is 57.7% |
| VisionFM "External Eye" encoder (`github.com/ABILab-CUHK/VisionFM`) | an ophthalmic foundation model; one of its encoders is pretrained on external-eye photos | research use only, Google Drive download, pretraining images not described | **the only candidate**; waiting for Jaineel's go-ahead before downloading |

- **Answer to M10:** no public model is pretrained on conjunctiva photos that we could use without leakage. The closest
  is VisionFM's external-eye encoder. Given Step 1 (a fine-tuned ViT already reaches training AUROC 1.00 and does
  worse than gradient boosting on color numbers), a better starting point is unlikely to fix what is limiting us, which
  is the data rather than the model. It is still worth one run if Jaineel approves the download.

### Step 6: write-up for the mentors
- `reports/mentor_followup_eyelid.md`: one section per suggestion, with what we did, the numbers and a plain answer,
  plus our questions for the mentors.
- README Part III (Section 6) summarizes Stages 9-11 and the clinical view.

### Step 7: more data (M9, M12)
- **Why:** the mentors suggested more countries (each with its own shortcut) and fingernail or palm datasets with
  age and sex.
- **What we searched:** Mendeley Data, IEEE DataPort, Zenodo, figshare, Kaggle and papers that release data. Nothing
  was downloaded.
- **Found:**

| Dataset | Body site | People | Where | Labels and metadata | Access | Fit for us |
|---|---|---|---|---|---|---|
| Yakimov et al. 2024, *Scientific Data*; figshare `10.6084/m9.figshare.c.6760179` | skin and fingernails, one photo each | 250 adults, 18-95, about half women | Moscow, Russia, one hospital | venous Hb per patient (69/250 below 12 g/dL); nail and skin boxes. **Age and sex are not in the per-patient file** | open; article CC BY-NC-ND | **Best fit for M12.** New country and adults, fixed lighting box. Needs one adult cutoff (no sex column), and skin tones are mostly light (Fitzpatrick I-III) |
| Girija, Kaur, Sekhar 2026, IEEE DataPort `10.21227/n7km-nr64` | conjunctiva (both eyes), palms, up to 10 nail beds | 1,485 blood-camp volunteers | Faridabad, India | lab Hb 6.0-18.7; WHO < 12: 166 anemic. **No age or sex** | IEEE DataPort login, 1.0 GB, license not stated | **Biggest conjunctiva set by far** and a second Indian site. Blood-camp volunteers, 11% anemic. Needs one adult cutoff |
| Asare, Appiahene, Donkoh 2022, Mendeley `10.17632/ccr8cm22vz.1` (palm) and `10.17632/2xx4j3kjg2.1` (fingernails) | palms; fingernails | 710 images each, children under 6 | 10 hospitals in Ghana | lab Hb, age, sex | open, CC BY 4.0 | Same authors and same 710 count as CP-AnemiC, so very likely **the same children**. No new population, but if the IDs link, it shows whether palm or nails carry more signal than the eyelid in the same children |
| Suner et al. 2021, *PLoS ONE*; Zenodo `10.5281/zenodo.4926651` | conjunctiva | 344 emergency-department adults, 19-96 | Providence, USA | lab Hb 4.7-19.6; the paper reports age, sex and a skin-tone score | open, but only two CSV files (270 kB), no photos | Photos aren't shared. If the CSVs hold color values, they could test the redness rule on a fourth population |
| Valles-Coral et al. 2025 | fingertip and palm videos, nail photos | 909 records, young adults | Peru | Hb, age, sex | **not released yet** | watch for release |

  Also checked: the "710-image online dataset" in a 2025 Korean journal paper is CP-AnemiC again.
- **Our pick, if Jaineel approves:** (1) Yakimov fingernails, for a new country and body site with open access;
  (2) the Indian IEEE DataPort set, for a large third conjunctiva source; (3) the Ghana palm and nail sets, only if
  their IDs match CP-AnemiC's. Each would need a duplicate audit and a no-color control first, as in Steps 1-3.

### Step 8: downloads after Jaineel's go-ahead (2026-10-23, in progress)
- **Why:** Jaineel approved the VisionFM external-eye encoder (M10) and the first-choice dataset (M12).
- **Yakimov fingernails:** downloaded from figshare (`data.zip`, 91.6 MB, 253 files including `photo/*.jpg`; figshare
  lists the dataset as CC0) to `data/raw/yakimov_nails/data.zip`. Claude's permission check blocked unzipping it
  (medical photos), so Jaineel unzips it. The per-patient file has no sex column (Step 7), so Jaineel picks one adult
  cutoff before any labels are made.
- **VisionFM external-eye encoder:** `VFM_External_weights.pth`, 1.5 GB, Google Drive link from the VisionFM README,
  research use only. Claude's permission check blocked downloading the weights file, so Jaineel downloads it to
  `data/weights/visionfm/` (gitignored). It is a pickle file, so it will be loaded with `torch.load(...,
  weights_only=True)`.
- **Indian IEEE DataPort set:** needs Jaineel's IEEE login; not downloaded.
- **Next, once the files are in place:** VisionFM as one more probe row in Step 1's ladder and Step 4's clinic table;
  the fingernails as a new Stage 12 (audit, nail-box color numbers, rule, dummy, no-color control, 5x5 CV by
  patient). A short plan goes to Jaineel before the Stage 12 code.
