# Do eyelid-photo anemia models hold up? Replicating two published results

*Student research project for RBC Borealis's Let's SOLVE it mentorship program. Working branch: `jaineel`.*

> **Key findings at a glance**
>
> | | Published | We reproduce it? | Photos kept out of training |
> |---|---|---|---|
> | CP-AnemiC, ResNet50 [1] | 84.79% | **yes**, 85.1% | 77.5% |
> | Eyes-Defy, ViT [2] | 98.47% | **close**, 95.6% (5 splits) | 88.5%, same as "Indian photo = anemic" (88.6%) |
>
> - Both papers test on photos the model has already seen (copies, or flipped versions). Removing that costs 7-8 accuracy points.
> - On Eyes-Defy the model mostly learns *which country* a photo is from. Within India it is at chance (AUROC 0.53).
> - On clean CP-AnemiC data, simple color statistics (AUROC 0.80) beat a large pretrained image model (0.67).
>
> *Research code only. This is not a medical device and must not be used to diagnose anyone.*

## Abstract

Anemia can be screened without a blood test by looking at how pale the inside of the lower eyelid (the conjunctiva) is. Two published deep-learning results are the starting point of our project: a ResNet50 on the CP-AnemiC dataset of Ghanaian children (84.79% accuracy) and a Vision Transformer on the Eyes-Defy-Anemia dataset of Italian and Indian adults (98.47% accuracy). We re-implemented both and **reproduced the published protocols**: 85.1% for the ResNet50 and 95.6% (mean of 5 random splits) for the ViT. We then asked how much of each score survives a split in which no photo appears in both training and test. Both papers let that happen: CP-AnemiC contains the same photo under several IDs, and the ViT paper flips each photo before splitting. With a clean split, accuracy falls to 77.5% and 88.5%. On Eyes-Defy, 88.5% is no better than predicting "anemic" for every Indian photo (88.6%), because 83% of the Indian patients are anemic and 10% of the Italian ones are. Within India alone the ViT ranks patients at chance (AUROC 0.53). Choosing the best epoch on the test set, which the paper may have done, raises the ViT only to 96.5% (best run 96.9%), so the last 2 points of the published score remain unexplained. Eyelid-photo anemia scores are therefore much weaker than the headline numbers suggest, and the main problems we found are data leakage and site confounding rather than model choice.

## 1. Introduction

About a quarter of the world's population is anemic. The standard test needs a blood sample. A cheaper screen looks at the conjunctiva, which turns pale when hemoglobin (Hb) is low, and several papers train neural networks to read this from a phone photo.

Our project proposal builds on two of them:

| | Paper | Dataset | Reported result |
|---|---|---|---|
| **A** | Appiahene et al., 2023 [1] | CP-AnemiC: 710 eyelid images of children in Ghana | joint ResNet50, 84.79% accuracy, Hb error 1.54 g/dL |
| **B** | Ramos-Soto et al., 2025 [2] | Eyes-Defy-Anemia: 218 eye photos of adults in Italy and India [3] | ViT-B/16, **98.47% accuracy** |

Before building anything new, we wanted to know two things:

1. **Do the published numbers reproduce** when we follow each paper's method?
2. **How much of each number is real?** A test score only means something if the test photos are new to the model. We rerun each experiment with one change: no photo may appear in both the training and the test set.

### How the pipeline fits together

```
 data/raw/ (never committed)
     │
     ▼
 1. AUDIT      audit.py, audit_eyes_defy.py   find duplicate photos, check labels against the papers
     │         -> data/processed/*_labels.csv (one row per photo, with photo_group)
     ▼
 2. SPLIT      common/splits.py               paper's split  vs  no photo in both train and test
     │
     ▼
 3. TRAIN      replicate_cnn.py, replicate_vit.py, train_color.py, train_probe.py
     │         every model next to a dummy and a simple rule
     ▼
 4. EVALUATE   common/metrics.py              AUROC, PR-AUC, sensitivity, 2000-resample bootstrap CIs
     │         -> results/*.json (metrics + config + git commit)
     ▼
 5. REPORT     summary.py, figures.py         -> reports/*.md, reports/figures/*.png
```

## 2. Data

| | CP-AnemiC [1] | Eyes-Defy-Anemia [3] |
|---|---|---|
| People | children 6-60 months, 4 hospitals in Ghana | adults: Bari, Italy and Chennai, India |
| Images | 710 rows, cut-out conjunctiva (PNG) | 218 full eye photos (JPG) + hand-drawn masks |
| Anemia label | Hb < 11.0 g/dL (WHO, children) | the ViT paper's cutoffs: India women < 12, men < 14; Italy <= 10.5 g/dL |
| Anemic | 424 / 710 rows | 91 / 217 photos (India 79/95, Italy 12/122) |

We audited both datasets before training (`src/eyelid/audit.py`, `src/eyelid/audit_eyes_defy.py`). Two findings shape everything that follows.

**CP-AnemiC repeats photos.** The 710 rows hold only **477 distinct photos**. 212 rows are byte-identical copies under a different ID, and 29 more are the same photo re-saved or slightly re-cropped (found with a perceptual hash and checked by eye). The copies usually carry a *different* Hb, hospital, age and sex, so their labels cannot all be right. Only 388 photos have no copy at all.

**In Eyes-Defy, country almost decides the label.** Under the paper's cutoffs 83% of Indian photos are anemic and 10% of Italian ones. The two countries also differ in skin tone, camera handling and lighting, so a model can score well by telling the countries apart. Eyes-Defy has no duplicate photos.

The ViT paper's label counts (126 not anemic, 91 anemic) only come out if Italy's cutoff is Hb <= 10.5, although the paper writes "< 10.5"; one Italian patient sits exactly on 10.5. We follow the counts.

## 3. Methods

**Replication A (CP-AnemiC, `src/eyelid/replicate_cnn.py`).** A ResNet50 with two heads, one predicting anemia and one predicting Hb, trained with the paper's loss, optimizer, learning rate, batch size, augmentation and 5-fold cross-validation. To save time we start from ImageNet weights and train 30 epochs instead of training from scratch for 150; the scores matched, so we did not run the longer version.

**Replication B (Eyes-Defy, `src/eyelid/replicate_vit.py`).** The paper's ViT-B/16 pretrained on ImageNet-21k, with its optimizer (NAdam, learning rate 1e-5, weight decay 1e-4), batch size 8, 30 epochs and 224x224 input. As in the paper, each photo is flipped horizontally and vertically (217 -> 651 images) and the 651 images are split 80/20 at random. The paper reports one split; we run 5 (seeds 0-4).

**The clean split.** We rerun each experiment changing only the split:

- *Paper's split*: CP-AnemiC rows are split at random, so copies of one photo can land in train and test. Eyes-Defy images are split after flipping, so a test photo's flipped twin is almost always in training (in 101-113 of the 104-113 distinct test photos, depending on the seed).
- *No photo in both*: copies (CP-AnemiC) or flipped versions (Eyes-Defy) of a photo always stay on the same side.

**Baselines.** Every model sits next to a *dummy* (always predict the most common class) and a *simple rule*: for CP-AnemiC "less red = anemic", for Eyes-Defy "Indian photo = anemic".

**Best-epoch check.** The ViT paper mentions monitoring validation curves, but its only held-out data is the test set. Picking the best epoch on the test set inflates the score. We record test accuracy after every epoch to see how much.

**Metrics.** The papers report accuracy, so we compare on accuracy. Our main metric is **AUROC**: the chance that the model ranks a random anemic person above a random non-anemic one (0.5 = coin flip, 1.0 = perfect). It does not depend on a threshold or on how many people are anemic. Every AUROC comes with a 95% bootstrap confidence interval (2000 resamples, resampling whole photos).

**Our own baselines on CP-AnemiC.** Separately from the replications, we built two simple models on the 388 photos with no copy: color statistics of the conjunctiva fed to a random forest (`src/eyelid/train_color.py`), and a linear probe on frozen DINOv2 image features (`src/eyelid/train_probe.py`). Both use 5x5 repeated cross-validation and leave-one-hospital-out.

## 4. Results

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

\* CP-AnemiC's reported AUC tracks its accuracy for all five of its models, which is what AUC computed from yes/no predictions gives, so it is not comparable with ours. \*\* Measured on the 388 photos with no copy, Section 4.4.

- **Replication A matches**: 85.1% vs 84.79%, Hb error 1.46 vs 1.54.
- **Replication B lands 1.5-4.5 points under 98.47%** with the final epoch. Choosing the best epoch on the test set adds about 1 point (96.5% mean, best run 96.9%; Figure 2), and none of 5 runs reaches 98.47%. With 131 test images, 98.47% means 2 errors and our best run makes 4, so a favourable split or a detail the paper does not report is the likeliest explanation. We call this replication *close, not exact*.
- **Removing photo overlap costs 7-8 points** in both cases, with every other setting unchanged.

### 4.2 The Eyes-Defy model mostly detects country

![ViT test accuracy by epoch](reports/figures/vit_epochs.png)

*Figure 2. ViT test accuracy after each training epoch, 5 seeds per split (thin lines) and their mean (thick). Dashed: the published 98.47%. With no photo in both sets (green), accuracy sits near 88%, the country rule's level, from the first epoch and barely moves. Only the paper's split (orange) keeps climbing, as the model learns the photos it will be tested on.*

With no photo in both sets, the ViT is exactly as accurate as the "Indian photo = anemic" rule (88.5% vs 88.6%). Looking inside each country, where that rule cannot help (test sets of the 5 seeds pooled):

| Country | ViT AUROC [95% CI] | Anemic test photos |
|---|---|---|
| India | **0.53 [0.29, 0.80]**, chance | 53 of 64 |
| Italy | 0.93 [0.86, 0.98] | 7 of 86 |

Among Indian patients, where nearly all of the anemia is, the model cannot tell anemic from non-anemic. The Italian result rests on only 7 anemic test photos with fairly severe anemia (Hb <= 10.5), so it is a hint, not a finding.

### 4.3 Duplicate photos inflate CP-AnemiC scores

All 710 rows, with only the split changed:

| Model | AUROC, copies may leak | AUROC, no photo in both |
|---|---|---|
| Color random forest | 0.937 [0.915, 0.954] | 0.885 [0.847, 0.916] |
| DINOv2 probe | 0.857 [0.820, 0.888] | 0.769 [0.715, 0.815] |
| Joint ResNet50 (Replication A) | 0.921 [0.896, 0.944] | 0.852 [0.809, 0.889] |
| **Control: random forest that never sees color** | **0.827 [0.780, 0.864]** | 0.681 [0.620, 0.738] |

The control model only gets image size, sharpness and similar properties, so it cannot see pallor. It still scores 0.83 when copies leak, by recognizing photos it trained on. Even after grouping it stays above chance (0.68), which suggests some re-cropped copies escape our duplicate detector. That is why our own baselines (next section) use only the 388 photos with no detected copy.

### 4.4 Our own baselines on CP-AnemiC (388 photos with no copy)

| Model | AUROC, 5x5 cross-validation | AUROC, leave one hospital out |
|---|---|---|
| Dummy | 0.493 [0.435, 0.549] | 0.381 [0.326, 0.436] |
| Rule: less red = anemic | 0.650 [0.597, 0.705] | 0.650 [0.597, 0.705] |
| Control: no color | 0.537 [0.474, 0.595] | 0.517 [0.456, 0.575] |
| Color logistic regression | 0.680 [0.627, 0.730] | 0.672 [0.619, 0.723] |
| **Color random forest** | **0.796 [0.752, 0.839]** | **0.786 [0.740, 0.830]** |
| DINOv2 probe | 0.672 [0.616, 0.723] | 0.652 [0.595, 0.708] |

On clean data the control drops to chance, and simple color statistics (AUROC 0.80) beat a large pretrained image model (0.67). The color model holds up on hospitals it never saw (0.79). Normalizing the color balance of each image (gray-world) lowers every model by 0.06-0.08 AUROC, because the crops are almost all conjunctiva and normalizing removes the redness signal itself.

## 5. Discussion

Both published methods reproduce, so the published numbers are not errors in the usual sense. But in both, the test set shares photos with the training set, and the model can score by recognizing photos rather than reading pallor. Splitting by photo costs 7-8 accuracy points in each case.

The Eyes-Defy result has a second problem: anemia and country are almost the same variable. A model that only recognizes "this is an Indian photo" reaches 88.6%, and within India the ViT does no better than chance. A 98% score on this dataset says little about detecting anemia in a new patient.

On CP-AnemiC's photos with no copy, color still carries a real signal (AUROC about 0.80, also on unseen hospitals). That is the honest starting point for our project, and well short of the published headline numbers.

**Practical lessons for anyone working with these datasets:**

1. Check for duplicate photos before splitting (CP-AnemiC has 233 extra copies).
2. Split by photo or patient *before* augmenting.
3. Report a country or site baseline next to any model trained on pooled sites.
4. Never choose epochs or thresholds on the test set.

## 6. Ethics and data use

- **No patient data in this repository.** Photos, labels and results stay on our machines (`data/` and `results/` are gitignored); only code, configs and aggregate reports are committed.
- **We follow each dataset's terms.** CP-AnemiC is for academic use; Eyes-Defy asks users to cite its papers [3]; DHS data, when it arrives, cannot be shared at all.
- **Skin tone and site.** Conjunctiva color varies with skin tone, camera and lighting. Our Eyes-Defy result shows how easily a model latches onto site differences instead of anemia, which could fail unevenly across populations if deployed.
- **Not a diagnostic tool.** Nothing here is clinically validated. A screening tool would need prospective testing on new patients from new sites.

## 7. Limitations

- **One split per seed, small test sets.** Eyes-Defy test sets hold 44 photos when split by photo, so accuracy moves several points between seeds; we report the range over 5 seeds.
- **Replication A uses ImageNet weights and 30 epochs** instead of training from scratch for 150 epochs. The scores matched, but this is not the identical recipe.
- **We do not have the papers' exact splits or code.** We matched every setting the papers state; GPU differences (our Apple M1 vs their NVIDIA) can shift results slightly.
- **Duplicate detection is not perfect.** The no-color control stays above chance on the 477 distinct CP-AnemiC photos, so some heavily re-cropped copies probably remain. Our own baselines use only the 388 photos with no detected copy.
- **Labels on copied CP-AnemiC photos conflict** (one photo is recorded with Hb 10.2 and 5.8), and there is no patient ID.
- **Cutoffs differ between datasets.** CP-AnemiC uses the WHO child cutoff (< 11.0 g/dL); the Eyes-Defy replication uses the ViT paper's cutoffs, which are not WHO's for Indian men (< 14 instead of < 13) or for Italy (one cutoff for both sexes).
- **Different people.** CP-AnemiC is young children, Eyes-Defy is adults, so neither result transfers directly to the other population.

## 8. Reproducing

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
| DHS Individual Recode | `data/raw/dhs/{zimbabwe,malawi,zambia}/` | DHS Program (registration required; not used yet) |

Then run, in order (or see Section 9 for what each file does):

```bash
python -m src.eyelid.audit                 # CP-AnemiC audit and labels
python -m src.eyelid.audit_eyes_defy       # Eyes-Defy audit and the ViT paper's labels
python -m src.eyelid.train_color           # color baselines (Section 4.4)
python -m src.eyelid.train_probe           # DINOv2 probe (Section 4.4)
python -m src.eyelid.replicate_cnn         # Replication A, ~1 h on an M1
python -m src.eyelid.replicate_vit --seeds 0 1 2 3 4                 # Replication B, ~2.5 h
python -m src.eyelid.replicate_vit --seeds 0 1 2 3 4 --track-epochs  # best-epoch check, ~2.5 h
python -m src.summary                      # reports/summary.md
python -m src.figures                      # reports/figures/*.png
```

Settings (seeds, cutoffs, hyperparameters) live in `configs/*.yaml`. Every run saves its metrics, config and git commit to `results/*.json`. Detailed reports: [summary](reports/summary.md), [Replication A](reports/replicate_cnn.md), [Replication B](reports/replicate_vit.md), [color baselines](reports/stage3_color.md), [DINOv2 probe](reports/stage4_probe.md).

## 9. Key files

| File | What it does |
|---|---|
| `src/eyelid/audit.py` | CP-AnemiC audit: duplicate and near-duplicate photos, label checks, `photo_group` |
| `src/eyelid/audit_eyes_defy.py` | Eyes-Defy audit and the ViT paper's labels |
| `src/eyelid/replicate_cnn.py` | Replication A: joint ResNet50, paper's split vs grouped by photo |
| `src/eyelid/replicate_vit.py` | Replication B: ViT-B/16, paper's split vs by photo, best-epoch check |
| `src/eyelid/train_color.py` | Color-statistics baselines, no-color control, leakage check |
| `src/eyelid/train_probe.py` | Linear probe on frozen DINOv2 features |
| `src/common/metrics.py` | AUROC, PR-AUC, sensitivity at 90% specificity, bootstrap CIs |
| `src/common/splits.py` | Repeated, grouped and leave-one-site-out splits |
| `src/summary.py`, `src/figures.py` | Build `reports/summary.md` and the figures from `results/*.json` |
| `configs/*.yaml` | Every setting: seeds, cutoffs, hyperparameters |
| `tests/` | Metric, feature and split tests (`python -m pytest -q`) |

## 10. Glossary

- **Leakage**: when the test set contains information the model saw in training (here, the same photo), so the test score overstates how the model does on new people.
- **AUROC**: the chance that a model scores a random anemic person higher than a random non-anemic one. 0.5 is a coin flip, 1.0 is perfect.
- **PR-AUC**: like AUROC but focused on the anemic class; its chance level equals the share of anemic people.
- **Sensitivity at 90% specificity**: the share of anemic people caught when only 10% of healthy people are flagged.
- **Bootstrap 95% CI**: we resample the test set 2000 times and report the range that holds the middle 95% of scores.
- **Seed**: the number that fixes the random split and training order, so a run can be repeated.

## 11. Status and next steps

| Step | Status |
|---|---|
| Audits of CP-AnemiC and Eyes-Defy | done |
| Color and DINOv2 baselines on CP-AnemiC | done |
| Replication A (CP-AnemiC ResNet50) | done |
| Replication B (Eyes-Defy ViT) | done |
| Cross-dataset test: train on CP-AnemiC, test on Eyes-Defy and back | next |
| Survey model on DHS data (Zimbabwe -> Malawi, Zambia) | waiting for DHS access |

## References

1. P. Appiahene, K. Chaturvedi, J. W. Asare, E. T. Donkoh, M. Prasad. CP-AnemiC: A conjunctival pallor dataset and benchmark for anemia detection in children. *Medicine in Novel Technology and Devices*, 2023. doi:10.1016/j.medntd.2023.100244
2. O. Ramos-Soto, J. Ramos-Frutos, E. Perez-Zarate, D. Oliva, S. J. Mousavirad, S. E. Balderas-Mata. Non-invasive anemia detection from conjunctiva and sclera images using vision transformer with attention map explainability. *Scientific Reports* 15, 44142, 2025. doi:10.1038/s41598-025-32343-w
3. R. Maglietta, M. E. Griseta, F. Clemente, A. Guarini, G. Dimauro. Eyes-defy-anemia dataset, described in: Advances in the automated diagnosis of anemia: a shared Eyes-defy-anemia dataset and a novel non-invasive system with machine learning.
