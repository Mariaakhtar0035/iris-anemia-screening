# Can eyelid photos and survey answers flag anemia across populations?

*Student research project for RBC Borealis's Let's SOLVE it mentorship program. Working branch: `jaineel`.*

> **Key findings at a glance**
>
> **Part I: the published eyelid results**
>
> | | Published | We reproduce it? | Photos kept out of training |
> |---|---|---|---|
> | CP-AnemiC, ResNet50 [1] | 84.79% | **yes**, 85.1% | 77.5% |
> | Eyes-Defy, ViT [2] | 98.47% | **close**, 95.6% (5 splits) | 88.5%, same as "Indian photo = anemic" (88.6%) |
>
> - Both papers test on photos the model has already seen (copies, or flipped versions). Removing that costs 7-8 accuracy points.
> - On Eyes-Defy the published model mostly learns *which country* a photo is from. Within India it is at chance (AUROC 0.53).
>
> **Part II: does anything carry over to new populations?**
>
> | Question | Answer (AUROC; 0.5 = coin flip) |
> |---|---|
> | Eyelid model trained on Ghana children, tested on Italian/Indian adults | learned models break (0.25-0.58); the untrained "less red = anemic" rule holds up (**0.81**) |
> | Survey answers (age, pregnancy, wealth, education, contraception, urban/rural) in Zimbabwe | weak signal, **0.58**; testing the top 30% finds 38% of anemic women vs 29% at random |
> | Same survey model, unchanged, in Malawi and Zambia | keeps its small signal (**0.57**, 0.57), calibration stays close |
> | Survey + eyelid score together (**simulation**, no paired data exists) | the eyelid score would do most of the work; the survey adds about 0.01-0.02 |
>
> *Research code only. This is not a medical device and must not be used to diagnose anyone.*

## Abstract

Anemia can be screened without a blood test by looking at how pale the inside of the lower eyelid (the conjunctiva) is, or by asking a few routine questions. Our goal is a tool that tells health workers who to test first, and an honest test of whether it works outside the place it was built. **Part I** re-implements the two published eyelid-photo results our proposal starts from: a ResNet50 on CP-AnemiC (Ghanaian children, 84.79% accuracy) and a Vision Transformer on Eyes-Defy-Anemia (Italian and Indian adults, 98.47%). We reproduce both protocols (85.1%, and 95.6% over 5 splits), but both let the same photo appear in training and test; with a clean split accuracy falls to 77.5% and 88.5%, and on Eyes-Defy 88.5% equals predicting "anemic" for every Indian photo. **Part II** tests transfer. Eyelid models trained on one dataset collapse on the other (AUROC 0.25-0.58), while the untrained rule "less red = anemic" holds up (0.81 on Eyes-Defy). A survey model trained on Zimbabwe DHS data carries a weak but real signal (AUROC 0.58) and keeps it, with close calibration, in Malawi and Zambia (0.57). A clearly labelled simulation suggests that if an eyelid score worked on these women as it does on Eyes-Defy, it would add far more than the survey. The headline numbers in this field overstate what a model does on new people; leakage, site confounding and camera differences matter more than model choice.

## 1. Introduction

About a quarter of the world's population is anemic, and confirming it needs a blood test that many clinics cannot run. Two cheaper signals exist: the conjunctiva turns pale when hemoglobin (Hb) is low, and routine intake details (age, pregnancy, wealth) shift a woman's risk. Our proposal asks whether models built on either signal **generalize**: to a new dataset, a new country, a new survey round.

Our starting point is three published results:

| | Paper | Dataset | Reported result |
|---|---|---|---|
| **A** | Appiahene et al., 2023 [1] | CP-AnemiC: 710 eyelid images of children in Ghana | joint ResNet50, 84.79% accuracy, Hb error 1.54 g/dL |
| **B** | Ramos-Soto et al., 2025 [2] | Eyes-Defy-Anemia: 218 eye photos of adults in Italy and India [3] | ViT-B/16, **98.47% accuracy** |
| **C** | Chemhaka et al., 2025 [4] | DHS survey, women in Zimbabwe | machine-learning model on socioeconomic and demographic factors |

We ask, in order:

1. **Do the published eyelid numbers reproduce, and how much of each is real?** (Part I)
2. **Does an eyelid model trained on one dataset work on the other?** (Part II, Section 5.1)
3. **How well do survey answers predict anemia, and does a Zimbabwe model work unchanged in Malawi and Zambia?** (Sections 5.2-5.3)
4. **What could combining both signals add?** (Section 5.4, simulation)

### How the pipeline fits together

```
 data/raw/ (never committed)
     │
     ▼
 1. AUDIT      eyelid/audit*.py, survey/audit.py     duplicates, labels, variable codes, missing data
     │         -> data/processed/ (one row per photo or per woman)
     ▼
 2. SPLIT      common/splits.py                      by photo / by cluster; never the same photo or cluster on both sides
     │
     ▼
 3. TRAIN      eyelid/replicate_*.py, train_color.py, train_probe.py, cross_dataset.py
     │         survey/train_baselines.py, transfer.py, fusion_sim.py
     │         every model next to a dummy and a simple rule
     ▼
 4. EVALUATE   common/metrics.py                     AUROC, PR-AUC, sensitivity, recall at a testing budget,
     │                                               calibration, 2000-resample bootstrap CIs
     │         -> results/*.json (metrics + config + git commit)
     ▼
 5. REPORT     summary.py, figures.py                -> reports/*.md, reports/figures/*.png
```

## 2. Data

| | CP-AnemiC [1] | Eyes-Defy-Anemia [3] | DHS Individual Recode [5] |
|---|---|---|---|
| People | children 6-60 months, 4 hospitals in Ghana | adults: Bari, Italy and Chennai, India | women 15-49: Zimbabwe 2015, Malawi 2024, Zambia 2024 |
| Size | 710 rows, only 477 distinct photos | 218 eye photos + hand-drawn masks | 9,265 / 10,000 / 6,829 women with an Hb test |
| Anemia label | Hb < 11.0 g/dL (WHO, children) | replication: the ViT paper's cutoffs; Part II: WHO by sex (women < 12, men < 13 g/dL) | DHS `v457`: WHO for women (< 12, < 11 if pregnant), Hb adjusted for altitude and smoking |
| Anemic | 424 / 710 rows | paper's cutoffs 91 / 217; WHO 91 / 217 (India 72%, Italy 19%) | 27.3% / 26.1% / 31.1% |

**Anemia cutoffs differ between datasets by design** (children vs adults, and the ViT paper's own choice); every table says which one it uses, and the settings live in `configs/*.yaml`.

We audited every dataset before training (`src/eyelid/audit.py`, `src/eyelid/audit_eyes_defy.py`, `src/survey/audit.py`). The findings that shape everything else:

- **CP-AnemiC repeats photos.** The 710 rows hold only **477 distinct photos**: 212 rows are byte-identical copies under a different ID, and 29 more are the same photo re-saved or slightly re-cropped. The copies usually carry a *different* Hb, hospital, age and sex. Only 388 photos have no copy at all.
- **In Eyes-Defy, country almost decides the label.** Under the ViT paper's cutoffs 83% of Indian photos are anemic and 10% of Italian ones. The paper's counts (126/91) only come out with Italy Hb <= 10.5, although it writes "< 10.5". Eyes-Defy has no duplicate photos, but 187 of its mask files carry a corrupt color-profile block that common image libraries reject (we read them with OpenCV).
- **DHS.** Every variable the analysis uses exists in all three surveys with the same codes, but Zimbabwe's survey is 9 years older than the other two. Half the 2024 households were never selected for the Hb test; Malawi's wealth index is missing for 408 tested women, all in the Central region; two contraception codes appear only in 2024. The DHS wealth index is built separately for each country and survey, so "richest" does not mean the same wealth across the three.

## 3. Methods

**Replications (Part I).** *A* (`src/eyelid/replicate_cnn.py`): ResNet50 with an anemia head and an Hb head, the paper's loss, optimizer, learning rate, batch size, augmentation and 5-fold CV; ImageNet weights and 30 epochs instead of 150 from scratch (the scores matched). *B* (`src/eyelid/replicate_vit.py`): ViT-B/16 (ImageNet-21k), NAdam, learning rate 1e-5, weight decay 1e-4, batch size 8, 30 epochs, 224x224; each photo flipped twice, then the 651 images split 80/20 at random, 5 seeds. Each is rerun with **one change**: no photo (or copy, or flipped twin) on both sides. A best-epoch check records test accuracy after every epoch, since the ViT paper's only held-out data is its test set.

**Cross-dataset (Part II, `src/eyelid/cross_dataset.py`).** The same input on both sides: the hand-segmented palpebral conjunctiva. Color statistics (mean, median, spread of RGB, HSV and Lab channels) with logistic regression or a random forest, and a linear probe on frozen DINOv2 features, fit on all of one dataset and applied unchanged to the other.

**Survey model (`src/survey/`).** Inputs: age, pregnancy, wealth quintile (ordered), education (ordered), contraceptive method and urban/rural (one-hot). Logistic regression and gradient boosting, with all preprocessing inside the training folds. Zimbabwe is evaluated with cross-validation **by cluster** (whole villages or neighborhoods held out), 5 folds x 5 seeds, with and without survey weights. For transfer, models are fit on all of Zimbabwe and scored unchanged on Malawi and Zambia; then Platt scaling on 200 random local women (20 repeats) tests whether a little local data fixes calibration.

**Fusion simulation (`src/fusion_sim.py`).** No dataset has a photo and survey answers for the same woman, so nothing is trained jointly. Each Malawi and Zambia woman keeps her real survey score and gets an eyelid redness score drawn from Eyes-Defy photos with her anemia status; the two are added on the log-odds scale, assuming they are independent given anemia. Two scenarios: scores from Indian photos (weak signal) and from Italian photos (strong).

**Baselines and metrics.** Every model sits next to a dummy (most common class) and a simple rule: "less red = anemic", "Indian photo = anemic", "test pregnant women first", or random testing. The main metric is **AUROC**; we also report PR-AUC, sensitivity at 90% specificity, and for the survey model **recall when only the top 30% by risk get tested**, Brier score and calibration. Every number has a 95% bootstrap CI (2000 resamples of whole photos or whole clusters).

## 4. Results, Part I: the published eyelid results

### 4.1 Both protocols reproduce; much of each score does not survive a clean split

![Published vs replicated accuracy](reports/figures/replication.png)

*Figure 1. Test accuracy as published, as replicated with the paper's split, and with no photo in both training and test. Eyes-Defy values are the mean of 5 random splits.*

| | Published | Ours, paper's split | Ours, no photo in both | Simple rule |
|---|---|---|---|---|
| **A. CP-AnemiC, joint ResNet50** | 84.79% | **85.1%** | 77.5% | - |
| AUROC | 0.835* | 0.921 [0.896, 0.944] | 0.852 [0.809, 0.889] | 0.650 (less red = anemic)** |
| Hb error (MAE, g/dL) | 1.54 | 1.46 | 1.61 | |
| **B. Eyes-Defy, ViT-B/16** | 98.47% | **95.6%** (93.9-97.0%) | 88.5% (85.6-90.2%) | 88.6% (India = anemic) |
| AUROC | not reported | 0.991 | 0.932 | 0.885 |

\* CP-AnemiC's reported AUC tracks its accuracy for all five of its models, which is what AUC computed from yes/no predictions gives, so it is not comparable with ours. \*\* On the 388 photos with no copy, Section 4.4.

- **Replication A matches**: 85.1% vs 84.79%, Hb error 1.46 vs 1.54.
- **Replication B lands 1.5-4.5 points under 98.47%.** Choosing the best epoch on the test set adds about 1 point (96.5% mean, best run 96.9%; Figure 2), and no run reaches 98.47%. With 131 test images, 98.47% means 2 errors and our best run makes 4, so a favourable split or an unreported detail is the likeliest explanation. We call this replication *close, not exact*.
- **Removing photo overlap costs 7-8 points** in both, with every other setting unchanged.

### 4.2 The Eyes-Defy model mostly detects country

![ViT test accuracy by epoch](reports/figures/vit_epochs.png)

*Figure 2. ViT test accuracy after each training epoch, 5 seeds per split (thin) and their mean (thick). Dashed: the published 98.47%. With no photo in both sets (green), accuracy sits near 88%, the country rule's level, from the first epoch. Only the paper's split (orange) keeps climbing, as the model learns the photos it will be tested on.*

| Country (no photo in both, 5 seeds pooled) | ViT AUROC [95% CI] | Anemic test photos |
|---|---|---|
| India | **0.53 [0.29, 0.80]**, chance | 53 of 64 |
| Italy | 0.93 [0.86, 0.98] | 7 of 86 |

Among Indian patients, where nearly all of the anemia is, the model cannot tell anemic from non-anemic. The Italian result rests on 7 anemic test photos with fairly severe anemia, so it is a hint, not a finding.

### 4.3 Duplicate photos inflate CP-AnemiC scores

| Model (all 710 rows) | AUROC, copies may leak | AUROC, no photo in both |
|---|---|---|
| Color random forest | 0.937 [0.915, 0.954] | 0.885 [0.847, 0.916] |
| DINOv2 probe | 0.857 [0.820, 0.888] | 0.769 [0.715, 0.815] |
| Joint ResNet50 (Replication A) | 0.921 [0.896, 0.944] | 0.852 [0.809, 0.889] |
| **Control: random forest that never sees color** | **0.827 [0.780, 0.864]** | 0.681 [0.620, 0.738] |

The control only sees image size, sharpness and similar properties, so it cannot see pallor; it scores 0.83 by recognizing photos it trained on. It stays above chance even after grouping (0.68), so some re-cropped copies escape our detector, which is why our own baselines use only the 388 photos with no detected copy.

### 4.4 Our own baselines on CP-AnemiC (388 photos with no copy)

| Model | AUROC, 5x5 cross-validation | AUROC, leave one hospital out |
|---|---|---|
| Dummy | 0.493 [0.435, 0.549] | 0.381 [0.326, 0.436] |
| Rule: less red = anemic | 0.650 [0.597, 0.705] | 0.650 [0.597, 0.705] |
| Control: no color | 0.537 [0.474, 0.595] | 0.517 [0.456, 0.575] |
| Color logistic regression | 0.680 [0.627, 0.730] | 0.672 [0.619, 0.723] |
| **Color random forest** | **0.796 [0.752, 0.839]** | **0.786 [0.740, 0.830]** |
| DINOv2 probe | 0.672 [0.616, 0.723] | 0.652 [0.595, 0.708] |

On clean data the control drops to chance, simple color statistics (0.80) beat a large pretrained image model (0.67), and the color model holds up on hospitals it never saw (0.79). Normalizing each image's color balance (gray-world) lowers every model by 0.06-0.08, because it removes the redness signal itself.

## 5. Results, Part II: does anything carry over?

### 5.1 Eyelid models across datasets

Input is the palpebral conjunctiva on both sides. CP-AnemiC: 388 photos with no copy, Hb < 11.0. Eyes-Defy: 217 photos, WHO cutoffs by sex. AUROC [95% CI]:

| Model | CP-AnemiC, within (CV) | CP-AnemiC -> Eyes-Defy | -> India only | -> Italy only | Eyes-Defy -> CP-AnemiC |
|---|---|---|---|---|---|
| Color logistic regression | 0.680 [0.627, 0.730] | **0.251** [0.189, 0.318] | 0.491 [0.359, 0.625] | 0.105 [0.039, 0.190] | 0.562 [0.504, 0.619] |
| Color random forest | 0.796 [0.752, 0.839] | 0.512 [0.435, 0.589] | 0.706 [0.573, 0.826] | 0.372 [0.270, 0.476] | 0.581 [0.529, 0.634] |
| DINOv2 probe | 0.672 [0.616, 0.723] | 0.580 [0.502, 0.658] | 0.568 [0.443, 0.692] | 0.535 [0.400, 0.671] | 0.557 [0.499, 0.612] |
| **Rule: less red = anemic (no training)** | 0.650 [0.597, 0.705] | **0.808** [0.749, 0.863] | 0.671 [0.552, 0.787] | 0.935 [0.865, 0.982] | **0.650** [0.597, 0.705] |
| Rule: Indian photo = anemic | | 0.766 [0.709, 0.823] | | | |

Every learned model loses most of its signal on the other dataset, and the color logistic regression ranks Eyes-Defy patients *worse than chance*: the color patterns it learned are tied to CP-AnemiC's camera and lighting, and they reverse under Eyes-Defy's. The untrained redness rule keeps working in both directions and even beats the country rule on Eyes-Defy. Redness carries over; the learned details of it do not.

### 5.2 Survey answers in Zimbabwe

9,265 women, cross-validated by cluster; bootstrap over clusters.

| Model | AUROC | PR-AUC | Recall, top 30% tested | Brier |
|---|---|---|---|---|
| Random testing | 0.493 [0.480, 0.507] | 0.270 [0.256, 0.285] | 0.294 [0.279, 0.309] | - |
| Pregnant women first, then random | 0.504 [0.490, 0.517] | 0.280 [0.266, 0.296] | 0.303 [0.289, 0.319] | - |
| Dummy | 0.490 [0.474, 0.507] | 0.273 [0.259, 0.289] | 0.292 [0.275, 0.314] | 0.199 [0.193, 0.204] |
| **Logistic regression** | **0.580 [0.566, 0.593]** | 0.323 [0.307, 0.341] | **0.379 [0.363, 0.394]** | 0.196 [0.190, 0.201] |
| Gradient boosting | 0.570 [0.556, 0.583] | 0.320 [0.303, 0.338] | 0.358 [0.341, 0.375] | 0.197 [0.192, 0.203] |

![Recall vs testing budget](reports/figures/survey_budget.png)

*Figure 3. Share of anemic women found as more women are tested, highest predicted risk first (Zimbabwe, out-of-fold).*

The survey answers carry a **weak but real** signal: testing the 30% of women the model ranks highest finds 38% of anemic women instead of 29%. Pregnant women are more often anemic (32% vs 27%) but are only 6% of women, so testing them first barely helps. Survey weights change nothing; the simpler logistic regression beats gradient boosting, which also turns out overconfident. Sensitivity at 90% specificity and the weighted runs are in [reports/stage7_survey.md](reports/stage7_survey.md).

### 5.3 The same survey model in Malawi and Zambia

Fit on all of Zimbabwe 2015 and applied **unchanged** to the 2024 surveys. Logistic regression:

| Country | Anemic | AUROC | Recall, top 30% tested | Brier | Calibration slope (ideal 1) | Calibration intercept (ideal 0) |
|---|---|---|---|---|---|---|
| Zimbabwe (cross-validated) | 27.3% | 0.580 [0.566, 0.593] | 0.379 [0.363, 0.394] | 0.196 | 0.93 [0.77, 1.10] | -0.07 [-0.23, 0.10] |
| Malawi | 26.1% | 0.568 [0.555, 0.581] | 0.351 [0.335, 0.367] | 0.192 | 0.90 [0.74, 1.07] | -0.24 [-0.40, -0.08] |
| Zambia | 31.1% | 0.565 [0.549, 0.580] | 0.349 [0.331, 0.365] | 0.212 | 1.00 [0.78, 1.22] | 0.09 [-0.11, 0.29] |

![Transfer calibration](reports/figures/survey_transfer_calibration.png)

*Figure 4. Predicted vs observed anemia, model trained on Zimbabwe. Left: logistic regression stays near the diagonal in all three countries. Right: gradient boosting is overconfident.*

The model keeps most of its small signal across countries and 9 years, and stays reasonably calibrated; it overestimates Malawi's anemia by about 3 points (29.0% predicted vs 26.1%). Recalibrating on 200 local women fixes that average (26.2%), but with a signal this weak its fitted slope is noisy, so the Brier score does not improve. A model that barely separates women has little to lose in transfer.

### 5.4 Combining both signals (simulation)

**This is a simulation, not a trained model.**

| Malawi | AUROC | Recall, top 30% tested |
|---|---|---|
| Survey only | 0.568 [0.555, 0.581] | 0.351 [0.335, 0.367] |
| Eyelid only, weak (Indian photos) | 0.674 [0.662, 0.687] | 0.512 [0.483, 0.524] |
| Survey + eyelid, weak | 0.687 [0.676, 0.700] | 0.487 [0.471, 0.504] |
| Eyelid only, strong (Italian photos) | 0.937 [0.932, 0.943] | 0.896 [0.858, 0.923] |
| Survey + eyelid, strong | 0.935 [0.930, 0.941] | 0.887 [0.863, 0.912] |

![Fusion simulation](reports/figures/fusion_simulation.png)

*Figure 5. Simulated share of anemic women found by testing budget. Zambia (right) looks the same; full table in [reports/fusion_simulation.md](reports/fusion_simulation.md).*

If an eyelid score worked on these women as well as the redness rule does on Eyes-Defy, it would carry most of the signal; the survey adds about 0.01-0.02 AUROC on top of a weak eyelid score and nothing on top of a strong one. Whether an eyelid score works on African women photographed in the field is exactly what Section 5.1 leaves open, so this is a reason to collect paired data, not evidence that fusion works.

## 6. Discussion

**Published numbers are real but not what they seem.** Both published eyelid methods reproduce, but their test sets share photos with training, and on Eyes-Defy the label is almost the same variable as the country. A 98% score there says little about a new patient.

**What survives is simple.** On clean CP-AnemiC data, color statistics beat a large pretrained model; across datasets, only the untrained "less red = anemic" rule keeps working. The learned models fit the camera, not the conjunctiva. That points to fixing the camera gap (a color reference card in each photo, or per-device color calibration) before training anything bigger.

**Survey answers are a weak but portable screen.** Six routine questions lift the share of anemic women found at a 30% testing budget from 29% to 38%, and that small gain carries to two other countries and a survey 9 years newer with little loss. It is a modest tool on its own; the simulation suggests it is most useful as a cheap add-on to a better signal.

**Practical lessons:**

1. Check for duplicate photos before splitting (CP-AnemiC has 233 extra copies).
2. Split by photo, patient or cluster *before* augmenting.
3. Report a site or country baseline next to any model trained on pooled sites.
4. Never choose epochs or thresholds on the test set.
5. Test on a different dataset or country before trusting a number.

## 7. Ethics and data use

- **No individual data in this repository.** Photos, survey records, labels and per-person results stay on our machines (`data/` and `results/` are gitignored); only code, configs and aggregate reports are committed.
- **We follow each dataset's terms.** CP-AnemiC is for academic use; Eyes-Defy asks users to cite its papers [3]; DHS data is used under the DHS Program's registration terms and cannot be shared.
- **Skin tone, camera and site.** Conjunctiva color varies with skin tone, camera and lighting. Our results show how easily models latch onto site and camera differences, which could make a deployed tool fail unevenly across populations.
- **Not a diagnostic tool.** Nothing here is clinically validated. A screening tool would need prospective testing on new patients from new sites.

## 8. Limitations

- **No paired data.** No dataset links an eyelid photo to survey answers for the same person, so fusion is simulated and assumes the two signals are independent given anemia.
- **Different populations.** CP-AnemiC is young children, Eyes-Defy is adults including men, the survey is women 15-49.
- **Cutoffs differ.** CP-AnemiC: Hb < 11.0 (children). Eyes-Defy: the ViT paper's cutoffs in Part I, WHO by sex in Part II (pregnancy not recorded; assumed not pregnant). DHS: `v457` (WHO for women, adjusted for altitude and smoking).
- **Survey rounds differ.** Zimbabwe 2015 vs Malawi and Zambia 2024, so transfer mixes country with time. Malawi's missing wealth index is imputed inside the model.
- **Small image test sets.** Eyes-Defy test sets hold 44 photos when split by photo; Italy has few anemic adults.
- **Replications are not identical recipes.** Replication A uses ImageNet weights and 30 epochs; we do not have either paper's exact split or code; GPU differences (Apple M1 vs NVIDIA) can shift results slightly.
- **Duplicate detection is not perfect**, and labels on copied CP-AnemiC photos conflict (one photo is recorded with Hb 10.2 and 5.8); there is no patient ID.
- **The Chemhaka et al. comparison is not done yet**: the publisher blocks scripted downloads, and we compare only against numbers taken from the paper itself.

## 9. Reproducing

```bash
python3.12 -m venv .venv          # Python 3.11 or newer
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Raw data is **never committed** (`data/` and `results/` are gitignored). Download each dataset and place it here:

| Dataset | Folder | Source |
|---|---|---|
| CP-AnemiC | `data/raw/cp_anemic/` | Mendeley Data (academic use) |
| Eyes-Defy-Anemia | `data/raw/eyes_defy/` (the `India/`, `Italy/` folders and the `.docx`) | IEEE DataPort, or the Kaggle mirror `harshwardhanfartale/eyes-defy-anemia` |
| DHS Individual Recode | `data/raw/dhs/{zimbabwe,malawi,zambia}/` (the `*IR*DT` folders: ZWIR72, MWIR81, ZMIR81) | DHS Program (registration required) |

Then run, in order (Section 10 lists what each file does):

```bash
python -m src.eyelid.audit                 # CP-AnemiC audit and labels
python -m src.eyelid.audit_eyes_defy       # Eyes-Defy audit and the ViT paper's labels
python -m src.eyelid.train_color           # color baselines (4.4)
python -m src.eyelid.train_probe           # DINOv2 probe (4.4)
python -m src.eyelid.replicate_cnn         # Replication A, ~1 h on an M1
python -m src.eyelid.replicate_vit --seeds 0 1 2 3 4                 # Replication B, ~2.5 h
python -m src.eyelid.replicate_vit --seeds 0 1 2 3 4 --track-epochs  # best-epoch check, ~2.5 h
python -m src.eyelid.cross_dataset         # 5.1, ~2 min
python -m src.survey.build_dataset         # DHS -> data/processed/{country}.parquet
python -m src.survey.audit                 # DHS audit
python -m src.survey.train_baselines       # 5.2, ~2 min
python -m src.survey.transfer              # 5.3, ~2 min
python -m src.fusion_sim                   # 5.4, ~2 min
python -m src.summary                      # reports/summary.md
python -m src.figures                      # reports/figures/replication.png, vit_epochs.png
```

Settings (seeds, cutoffs, hyperparameters) live in `configs/*.yaml`. Every run saves its metrics, config and git commit to `results/*.json`. Detailed reports: [summary](reports/summary.md), [Replication A](reports/replicate_cnn.md), [Replication B](reports/replicate_vit.md), [color baselines](reports/stage3_color.md), [DINOv2 probe](reports/stage4_probe.md), [cross-dataset](reports/stage5_cross_dataset.md), [survey baselines](reports/stage7_survey.md), [survey transfer](reports/stage8_transfer.md), [fusion simulation](reports/fusion_simulation.md).

## 10. Key files

| File | What it does |
|---|---|
| `src/eyelid/audit.py`, `audit_eyes_defy.py` | Image audits: duplicate photos, label checks against the papers |
| `src/eyelid/replicate_cnn.py`, `replicate_vit.py` | Replications A and B: paper's split vs no photo in both, best-epoch check |
| `src/eyelid/train_color.py`, `train_probe.py` | Color-statistics baselines, no-color control, DINOv2 probe |
| `src/eyelid/cross_dataset.py` | Train on one image dataset, test on the other |
| `src/eyelid/roi.py` | Reads the conjunctiva cut-outs (with an OpenCV fallback for Eyes-Defy's damaged masks) |
| `src/survey/build_dataset.py`, `audit.py` | DHS: verified variables -> clean table per country; audit and coding checks |
| `src/survey/train_baselines.py`, `transfer.py` | Survey baselines by cluster; transfer and recalibration |
| `src/fusion_sim.py` | Fusion simulation |
| `src/common/metrics.py`, `splits.py`, `evaluate.py` | Metrics, calibration, bootstrap CIs, grouped splits |
| `src/summary.py`, `src/figures.py` | Build `reports/summary.md` and the figures from `results/*.json` |
| `configs/*.yaml` | Every setting: seeds, cutoffs, hyperparameters, DHS variable codes |
| `tests/` | Metric, feature, split, label and pipeline tests (`python -m pytest -q`) |

## 11. Glossary

- **Leakage**: when the test set contains information the model saw in training (here, the same photo), so the test score overstates how the model does on new people.
- **AUROC**: the chance that a model scores a random anemic person higher than a random non-anemic one. 0.5 is a coin flip, 1.0 is perfect.
- **PR-AUC**: like AUROC but focused on the anemic class; its chance level equals the share of anemic people.
- **Sensitivity at 90% specificity**: the share of anemic people caught when only 10% of healthy people are flagged.
- **Recall at a 30% testing budget**: the share of all anemic women found if only the 30% with the highest predicted risk get a blood test. Random testing finds 30%.
- **Calibration**: whether predicted risks match reality (of women given 30%, are 30% anemic?). Slope 1 and intercept 0 are perfect.
- **Cluster**: a DHS sampling area (a village or neighborhood). Splitting by cluster keeps neighbors out of each other's test sets.
- **Bootstrap 95% CI**: we resample the test set 2000 times and report the range that holds the middle 95% of scores.
- **Seed**: the number that fixes the random split and training order, so a run can be repeated.

## 12. Status and next steps

| Step | Status |
|---|---|
| Audits of CP-AnemiC, Eyes-Defy and DHS | done |
| Replications A and B, with clean splits and the best-epoch check | done |
| Color and DINOv2 baselines on CP-AnemiC | done |
| Cross-dataset eyelid test | done |
| Survey model on Zimbabwe, transfer to Malawi and Zambia | done |
| Fusion simulation | done |
| Comparison with Chemhaka et al. [4] | waiting for the paper's numbers |

Next, with more time: (1) collect paired data, an eyelid photo and survey answers from the same women, to test fusion for real; (2) fix the camera gap with a color reference card or per-device calibration, then retest the learned eyelid models across datasets; (3) compare the survey baseline with Chemhaka et al. and ask the CP-AnemiC authors about the duplicate photos.

## References

1. P. Appiahene, K. Chaturvedi, J. W. Asare, E. T. Donkoh, M. Prasad. CP-AnemiC: A conjunctival pallor dataset and benchmark for anemia detection in children. *Medicine in Novel Technology and Devices*, 2023. doi:10.1016/j.medntd.2023.100244
2. O. Ramos-Soto, J. Ramos-Frutos, E. Perez-Zarate, D. Oliva, S. J. Mousavirad, S. E. Balderas-Mata. Non-invasive anemia detection from conjunctiva and sclera images using vision transformer with attention map explainability. *Scientific Reports* 15, 44142, 2025. doi:10.1038/s41598-025-32343-w
3. R. Maglietta, M. E. Griseta, F. Clemente, A. Guarini, G. Dimauro. Eyes-defy-anemia dataset, described in: Advances in the automated diagnosis of anemia: a shared Eyes-defy-anemia dataset and a novel non-invasive system with machine learning.
4. Chemhaka et al. Socioeconomic and demographic factors associated with anaemia among women of reproductive age in Zimbabwe: a supervised machine learning approach. 2025. doi:10.1186/s12982-025-00524-7
5. The DHS Program. Demographic and Health Surveys: Zimbabwe 2015 (ZWIR72), Malawi 2024 (MWIR81), Zambia 2024 (ZMIR81), Individual Recode. ICF.
