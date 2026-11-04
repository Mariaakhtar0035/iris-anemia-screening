# Reports: within one site vs across sites

The mentors asked (2026-10-31) to keep the experiments that stay inside one site apart from the ones that move a model
between sites, so the two stories can be followed separately. On 2026-11-03 they added: don't set the goal at
transfer for the image data (small, very different datasets; real transfer needs domain-adaptation methods), focus
per location. So **within one site** is the main line now, and **across sites** is a closed chapter that explains why.

AUROC: 0.5 = guessing, 1.0 = perfect. Every number has a bootstrap 95% CI in its report. Cutoffs: Ghana children
Hb < 11.0 g/dL; Eyes-Defy adults WHO by sex (women < 12.0, men < 13.0); Moscow adults Hb < 12.0 for everyone (no sex
column); Ghana palms and nails: the dataset's own label (no Hb).

## Within one site (main line)

| Stage | Question | Answer | Number | Report |
|---|---|---|---|---|
| 1-2 | Can the CP-AnemiC and Eyes-Defy papers be reproduced? | Close on the papers' splits; lower once no photo is in both training and test | ResNet50 85.1% (paper 84.79%), 77.5% by photo; ViT 95.6% (paper 98.47%), 88.5% by photo | [CNN](replicate_cnn.md), [ViT](replicate_vit.md) |
| 3-4 | Ghana: how well do simple models do? | Color numbers beat a general image model | random forest 0.796, DINOv2 0.672, rule 0.650 | [color](stage3_color.md), [probe](stage4_probe.md) |
| 9 | How far can each population be fit on its own? | Ghana keeps rising with more photos; India stays at the rule; Italy's 0.92 is mostly a photo batch | Ghana gradient boosting 0.846; India about 0.67 | [ceiling](stage9_ceiling.md) |
| 9 | Do bigger models help in Ghana? | TabPFN yes; big image models no | TabPFN 0.919; ViT 0.715; VisionFM 0.765-0.785 | [ceiling](stage9_ceiling.md) |
| checks | Is Ghana's 0.92 real? | Yes: it needs color and holds on an unseen hospital | no color 0.565; leave one hospital out 0.913 | [TabPFN checks](tabpfn_checks.md) |
| clinic | How many blood tests does it save? | To catch 90% of anemic children in Ghana, test half of them | 50% tested | [clinical](clinical_eyelid.md) |
| 12 | Fingernails in Moscow (fixed light box)? | Work inside the site | logistic regression 0.818, rule 0.703 | [nails](stage12_nails.md) |
| 13 | Palms and nails in Ghana? | Palms work as well as eyelids; nails less | palm 0.912, nails 0.798 | [Ghana sites](stage13_ghana_sites.md) |
| 16 | Can the photo estimate the Hb number? | Yes, better than guessing the mean | Ghana error 1.38 g/dL (mean guess 1.71); Moscow nails 1.65 (1.98) | [Hb number](stage16_hb_regression.md) |
| 20 | Does adding age and sex to the photo help? | Not for yes/no anywhere; yes for the Hb number in adults (India, Italy) | Ghana TabPFN 0.919 -> 0.918; India Hb error 1.45 -> 1.29 g/dL | [age and sex](stage20_metadata.md) |
| 21 | Does a photo-quality filter or glare removal help? | No gain; but glare alone flags Italy's anemic photo batch | best-model change +0.00 to +0.02; Italy glare alone 0.89 | [quality](stage21_quality.md) |

## Across sites (closed: explains why per-location models)

| Stage | Question | Answer | Number | Report |
|---|---|---|---|---|
| 5 | Does a Ghana model work on Eyes-Defy, and back? | No; only the untrained redness rule carries over | random forest 0.587, rule 0.835 | [cross-dataset](stage5_cross_dataset.md) |
| 10 | Why do the datasets differ? | The color gap between datasets is about 3x the anemia signal | healthy Ghana vs Italy 12.0 a* vs anemia 3.5 inside Ghana; dataset detector 0.99 | [populations](stage10_populations.md) |
| 11 | Can debiasing remove the country shortcut? | Fairer scores, not more accurate ones | no within-country gain | [debias](stage11_debias.md) |
| TabPFN | Does the best model (TabPFN) carry over? | No | Ghana -> Eyes-Defy 0.602 (rule 0.835) | [TabPFN transfer](tabpfn_transfer.md) |
| 14 | One model for all three countries? | Fine in Ghana, never beats the rule in India or Italy | Ghana 0.903, India 0.644, Italy 0.906 | [three countries](stage14_three_countries.md) |
| 15 | How many labelled local photos does a new country need? | 10-40 are not enough; Ghana needs about 80 | best at 40: Ghana 0.68 vs rule 0.65 | [few local](stage15_few_local.md) |
| 17 | Does an Hb-number model carry over (no cutoff problem)? | No; the Hb scale shifts between datasets | ranking India 0.66, Italy 0.67; Italy estimates 2.9 g/dL too low | [Hb across](stage17_hb_cross.md) |
| 18 | Can a few local Hb values correct it? | Fixes the average shift, not the ranking | see report | [Hb local](stage18_hb_local.md) |
| 19 | Do fingernail models carry between Ghana and Moscow? | No, in both directions | 0.38-0.58 | [nails across](stage19_nails_cross.md) |

## Survey (Part II, teammates' side)

| Stage | Question | Number | Report |
|---|---|---|---|
| 7 | Survey model inside Zimbabwe | logistic regression 0.580 | [survey](stage7_survey.md) |
| 8 | The same model in Malawi and Zambia | 0.568, 0.565 | [transfer](stage8_transfer.md) |
| fusion | What could an eyelid score add? (simulation, no paired data) | see report | [fusion](fusion_simulation.md) |
