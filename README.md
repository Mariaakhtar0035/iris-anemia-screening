# iris-anemia-screening
Cross-country ML for anemia risk screening. Tests whether models trained on DHS survey data and conjunctival (inner eyelid) images generalize across countries and datasets. Built for RBC Borealis’s Let’s SOLVE it mentorship program.

## Overview

Two baseline models, each with an honest look at how well it transfers:

1. **Eyelid photo model.** Trained on CP-AnemiC (children in Ghana), tested on Eyes-Defy-Anemia (adults in Italy and India), and the reverse. A color-feature baseline first, then a linear probe on frozen DINOv2 features.
2. **Survey model.** Trained on the DHS Individual Recode for Zimbabwe, then scored unchanged on Malawi and Zambia.

Every number is reported with a bootstrap 95% confidence interval (2000 resamples) next to a dummy baseline and a simple rule baseline. Survey data is split by cluster and images by patient, to avoid leakage. The aim is a reproducible baseline we can explain, not a high score.

## Setup

```bash
python3.12 -m venv .venv          # Python 3.11 or newer
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Settings (seeds, bootstrap size, anemia cutoffs) live in `configs/default.yaml`.

## Data

Raw data is **never committed**. `data/` and `results/` are gitignored. DHS terms don't allow sharing survey data, and CP-AnemiC is licensed for academic use only. Download each dataset yourself and place it here:

| Dataset | Folder | Source |
|---|---|---|
| CP-AnemiC | `data/raw/cp_anemic/` | Mendeley Data |
| Eyes-Defy-Anemia | `data/raw/eyes_defy/` (the `India/`, `Italy/` folders and the description `.docx`) | IEEE DataPort, or the Kaggle mirror `harshwardhanfartale/eyes-defy-anemia` |
| DHS Individual Recode, Zimbabwe | `data/raw/dhs/zimbabwe/` | DHS Program (registration and approval required) |
| DHS Individual Recode, Malawi | `data/raw/dhs/malawi/` | same |
| DHS Individual Recode, Zambia | `data/raw/dhs/zambia/` | same |

For each DHS country, put the data file and its dictionary files in the country folder. Create the folders with:

```bash
mkdir -p data/raw/cp_anemic data/raw/eyes_defy data/raw/dhs/{zimbabwe,malawi,zambia} data/processed results
```

## Anemia cutoff

- **Children 6 to 59 months:** anemic if hemoglobin is below **11.0 g/dL** (`anemia_cutoff_g_dl.child_6_59_months` in the config). This is the cutoff for CP-AnemiC.
- **Eyes-Defy-Anemia, ViT paper replication only:** the paper's cutoffs, not WHO's: India women < 12, men < 14 g/dL; Italy <= 10.5 g/dL for everyone (`paper_cutoffs_g_dl` in `configs/replicate_vit.yaml`). The paper writes "< 10.5" for Italy, but its 126/91 counts need <= 10.5.
- **Adults otherwise:** WHO cutoffs by sex and pregnancy status, set when the cross-dataset and DHS stages are built.

## CP-AnemiC data notes

From `python -m src.eyelid.audit`:

- **Labels agree.** Hb < 11.0 g/dL matches both the `REMARK` column and the image's folder on all 710 rows: 424 anemic, 286 not. 9 rows sit at exactly 11.0 and count as not anemic.
- **Duplicate images.** The 710 rows hold only **498 byte-distinct image files**: 91 groups of byte-identical images appear under different IDs (212 extra copies). The copies usually carry a different Hb, hospital, age and sex. On top of that, 29 links join images that are the **same photo re-saved or slightly re-cropped** (difference hash within 20 of 256 bits and size within 5%, checked by eye; see `configs/default.yaml`). That leaves **477 distinct photos**. One photo's copies disagree on the anemia label (`Image_188` / `Image_310`).
  - `data/processed/cp_anemic_labels.csv` has `image_group` (byte-identical) and `photo_group` (same photo). Every split keeps a `photo_group` in one fold.
  - Metadata on any photo with copies is unreliable (e.g. one photo is recorded as both Hb 10.2 and Hb 5.8), so hospital-level analysis uses only the **388 photos with no copy**.
- **No patient ID.** We assume one child per distinct photo.
- **Age.** Ages run 6 to 60 months, mostly in whole years. 90 rows are 60 months, outside the 6 to 59 month definition; they are kept for now.
- **Images** are RGBA with a transparent background, so the conjunctiva appears to be cut out already.

## Eyes-Defy-Anemia data notes

From `python -m src.eyelid.audit_eyes_defy`:

- 218 eye photos (Italy 123, India 95), one folder per patient with the full eye `.jpg` and hand-segmented conjunctiva masks. Italy patient 93 has no Hb, leaving 217 labelled photos.
- Both spreadsheets have `Number`, `Hgb`, `Gender`, `Age`, `Note`. Italy writes 15 Hb values with a decimal comma.
- No duplicate photos.
- Country nearly decides the label under the paper's cutoffs: India 79/95 anemic, Italy 12/122.

## Results so far

| Stage | Report |
|---|---|
| 3. Color-feature baseline | [reports/stage3_color.md](reports/stage3_color.md) |
| 4. Frozen DINOv2 probe vs color baseline | [reports/stage4_probe.md](reports/stage4_probe.md) |
| Replication of the CP-AnemiC paper (joint ResNet50) | [reports/replicate_cnn.md](reports/replicate_cnn.md) |
| Replication of the 2025 ViT paper on Eyes-Defy-Anemia | [reports/replicate_vit.md](reports/replicate_vit.md) |
| Summary of everything | [reports/summary.md](reports/summary.md) |

## Layout

```
configs/        settings, one YAML per experiment
src/common/     metrics.py (AUROC, PR-AUC, sensitivity at specificity, recall at top fraction, Brier, bootstrap CI)
src/eyelid/     eyelid photo pipeline
src/survey/     DHS survey pipeline
tests/          pytest
```
