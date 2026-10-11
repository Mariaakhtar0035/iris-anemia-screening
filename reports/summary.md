# Summary: anemia screening baselines

Built from `results/*.json` by `python -m src.summary`. Anemia = Hb < 11.0 g/dL (children 6-59 months, CP-AnemiC). Every estimate has a bootstrap 95% CI from 2000 resamples. Commits that produced each file: `stage2_audit` 7b721ca, `stage3_color` bb9a083, `stage4_probe` 945b76d, `replicate_cnn` 1fd7b65, `stage_eyes_defy_audit` 2239613, `replicate_vit` 4125217.

## Status

| Stage | Status |
|---|---|
| 1. Skeleton and metrics | done |
| 2. CP-AnemiC audit | done |
| 3. Color-feature baseline | done |
| 4. Frozen DINOv2 probe | done |
| Replication of the CP-AnemiC paper's CNN | done |
| Replication of the 2025 ViT paper on Eyes-Defy-Anemia | done |
| 5. Cross-dataset test with Eyes-Defy-Anemia | not started (data in hand; parked until replication is done) |
| 6-8. DHS survey model and transfer | **blocked**: needs DHS registration and approval |

## 1. The CP-AnemiC data has far fewer distinct photos than rows

From `results/stage2_audit.json`: the 710 rows hold 498 byte-distinct images (91 groups of byte-identical files, 212 extra copies). 29 more links join the same photo re-saved or slightly re-cropped, leaving **477 distinct photos**, of which 388 have no copy at all. 1 photo's copies disagree on the anemia label. The Hb < 11.0 label disagrees with the lab remark or the image folder on 0 rows. 90 rows are 60 months old, just outside the 6-59 month definition.

## 2. Headline results: photos with no copy

`cv_5x5_no_copies` is repeated stratified 5-fold CV (5 repeats) on the 388 photos with no copy; `leave_one_hospital_out` holds out one hospital at a time on the same photos. Color rows are gray-world off.

| experiment | n | model | AUROC | PR-AUC | Sens @ 90% spec |
|---|---|---|---|---|---|
| cv_5x5_no_copies | 388 | dummy | 0.493 [0.435, 0.549] | 0.426 [0.367, 0.491] | 0.109 [0.000, 0.155] |
| cv_5x5_no_copies | 388 | rule_low_redness | 0.650 [0.597, 0.705] | 0.555 [0.483, 0.645] | 0.273 [0.135, 0.354] |
| cv_5x5_no_copies | 388 | negative_control_rf_no_color | 0.537 [0.474, 0.595] | 0.497 [0.425, 0.578] | 0.164 [0.098, 0.252] |
| cv_5x5_no_copies | 388 | logistic_regression | 0.680 [0.627, 0.730] | 0.580 [0.503, 0.663] | 0.267 [0.130, 0.345] |
| cv_5x5_no_copies | 388 | random_forest | 0.796 [0.752, 0.839] | 0.736 [0.666, 0.799] | 0.370 [0.256, 0.491] |
| cv_5x5_no_copies | 388 | dinov2_probe[cls_and_patch_mean] | 0.672 [0.616, 0.723] | 0.583 [0.507, 0.665] | 0.273 [0.156, 0.389] |
| leave_one_hospital_out | 388 | dummy | 0.381 [0.326, 0.436] | 0.358 [0.308, 0.413] | 0.000 [0.000, 0.000] |
| leave_one_hospital_out | 388 | rule_low_redness | 0.650 [0.597, 0.705] | 0.555 [0.483, 0.645] | 0.273 [0.135, 0.354] |
| leave_one_hospital_out | 388 | negative_control_rf_no_color | 0.517 [0.456, 0.575] | 0.473 [0.403, 0.552] | 0.133 [0.067, 0.205] |
| leave_one_hospital_out | 388 | logistic_regression | 0.672 [0.619, 0.723] | 0.562 [0.486, 0.646] | 0.224 [0.100, 0.288] |
| leave_one_hospital_out | 388 | random_forest | 0.786 [0.740, 0.830] | 0.702 [0.628, 0.773] | 0.333 [0.237, 0.454] |
| leave_one_hospital_out | 388 | dinov2_probe[cls_and_patch_mean] | 0.652 [0.595, 0.708] | 0.595 [0.516, 0.671] | 0.218 [0.155, 0.308] |

- The color random forest is the best baseline: AUROC 0.796 [0.752, 0.839] in CV and 0.786 [0.740, 0.830] on held-out hospitals.
- The frozen DINOv2 probe reaches 0.672 [0.616, 0.723], level with color logistic regression and below the random forest.
- The no-color negative control sits at 0.537 [0.474, 0.595], close to chance, so these splits do not leak duplicates.

## 3. Gray-world normalization hurts

| experiment | model | AUROC off | AUROC on | change |
|---|---|---|---|---|
| cv_5x5_no_copies | rule_low_redness | 0.650 [0.597, 0.705] | 0.577 [0.521, 0.633] | -0.073 |
| cv_5x5_no_copies | logistic_regression | 0.680 [0.627, 0.730] | 0.621 [0.565, 0.674] | -0.059 |
| cv_5x5_no_copies | random_forest | 0.796 [0.752, 0.839] | 0.733 [0.683, 0.779] | -0.063 |
| leave_one_hospital_out | rule_low_redness | 0.650 [0.597, 0.705] | 0.577 [0.521, 0.633] | -0.073 |
| leave_one_hospital_out | logistic_regression | 0.672 [0.619, 0.723] | 0.592 [0.536, 0.646] | -0.080 |
| leave_one_hospital_out | random_forest | 0.786 [0.740, 0.830] | 0.716 [0.667, 0.765] | -0.070 |

The crops contain only conjunctiva, so equalizing the channel means removes part of the redness signal the models rely on.

## 4. Duplicate photos inflate scores under a random split

All 710 rows. The only difference between each pair of rows is whether copies of a photo may sit in both train and test. Bootstrap resamples whole photo groups.

| source | split | model | AUROC | PR-AUC | Sens @ 90% spec |
|---|---|---|---|---|---|
| Stage 3 | random rows (copies can leak) | logistic_regression | 0.726 [0.670, 0.777] | 0.787 [0.714, 0.845] | 0.351 [0.183, 0.514] |
| Stage 3 | random rows (copies can leak) | random_forest | 0.937 [0.915, 0.954] | 0.959 [0.940, 0.973] | 0.811 [0.744, 0.859] |
| Stage 3 | random rows (copies can leak) | negative_control_rf_no_color | 0.827 [0.780, 0.864] | 0.885 [0.837, 0.920] | 0.604 [0.504, 0.690] |
| Stage 3 | grouped by photo (no leak) | logistic_regression | 0.693 [0.633, 0.749] | 0.762 [0.688, 0.823] | 0.278 [0.176, 0.445] |
| Stage 3 | grouped by photo (no leak) | random_forest | 0.885 [0.847, 0.916] | 0.923 [0.892, 0.947] | 0.675 [0.585, 0.774] |
| Stage 3 | grouped by photo (no leak) | negative_control_rf_no_color | 0.681 [0.620, 0.738] | 0.753 [0.688, 0.816] | 0.304 [0.219, 0.431] |
| Stage 4 | random rows (copies can leak) | dinov2_probe[cls_and_patch_mean] | 0.857 [0.820, 0.888] | 0.901 [0.854, 0.932] | 0.677 [0.558, 0.750] |
| Stage 4 | grouped by photo (no leak) | dinov2_probe[cls_and_patch_mean] | 0.769 [0.715, 0.815] | 0.843 [0.784, 0.888] | 0.502 [0.409, 0.596] |
| Replication | paper_random_rows | joint_resnet50 | 0.921 [0.896, 0.944] | 0.946 [0.922, 0.965] | 0.726 [0.649, 0.854] |
| Replication | grouped_by_photo | joint_resnet50 | 0.852 [0.809, 0.889] | 0.870 [0.824, 0.910] | 0.571 [0.419, 0.697] |

- A random forest that never sees color reaches AUROC 0.827 [0.780, 0.864] under the random split, from duplicates alone.
- Stage 3 `random_forest`: AUROC 0.937 with copies leaking vs 0.885 grouped by photo (+0.052).
- Stage 4 `dinov2_probe[cls_and_patch_mean]`: AUROC 0.857 with copies leaking vs 0.769 grouped by photo (+0.088).
- Replication `joint_resnet50`: AUROC 0.921 with copies leaking vs 0.852 grouped by photo (+0.069).

## 5. Replicating the CP-AnemiC paper's benchmark

Joint ResNet50 (anemia classification + Hb regression) with the paper's loss, optimizer, learning rate, batch size, augmentation and 5-fold CV; ImageNet-pretrained and 30 epochs instead of random init and 150 (see `configs/replicate.yaml`). Paper, for reference (Appiahene et al. 2023, Tables 3-4): ResNet50 AUC 0.835, accuracy 84.79%, Hb MAE 1.54.

| split | AUROC | accuracy at 0.5 | Hb MAE (g/dL) |
|---|---|---|---|
| paper-style random rows | 0.921 [0.896, 0.944] | 85.1% | 1.46 |
| grouped by photo | 0.852 [0.809, 0.889] | 77.5% | 1.61 |

- Under the paper's split, accuracy (85.1%) and Hb MAE (1.46) land close to the paper's ResNet50 (84.79%, 1.54), so the protocol reproduces.
- The paper's AUC sits within about 0.01 of its accuracy for all five of its models (Table 3), which is what an AUC computed from hard 0/1 predictions gives. Ours comes from probabilities, so compare on accuracy and Hb error rather than AUC.
- Keeping each photo's copies in one fold drops accuracy to 77.5% (AUROC 0.921 to 0.852) and raises Hb MAE to 1.61, with every other setting identical.

## 6. Replicating the 2025 ViT paper's 98.47% on Eyes-Defy-Anemia

From `results/stage_eyes_defy_audit.json`: 218 photos ({'India': 95, 'Italy': 123}), 217 with Hb, labelled with the paper's cutoffs (India women < 12, men < 14 g/dL; Italy <= 10.5 g/dL): 126 no anemia / 91 anemia, matching the paper. The paper writes Italy "< 10.5", but its counts need <= 10.5 (Italy_13 sits exactly on 10.5). No duplicate photos (0 near-duplicate links).

ViT-B/16 (ImageNet-21k), the paper's optimizer, learning rate, batch size and 30 epochs, one random 80/20 split per seed, 5 seeds. `paper` flips every photo and then splits the 3x images, so flipped copies of a test photo sit in training; `by_photo` splits photos first. Baselines use the same test images.

| split | model | mean accuracy | mean AUROC |
|---|---|---|---|
| paper | ViT-B/16 in21k | 95.6% | 0.991 |
| paper | country rule (India = anemic) | 86.7% | 0.866 |
| paper | dummy (always no anemia) | 56.9% | 0.500 |
| by_photo | ViT-B/16 in21k | 88.5% | 0.932 |
| by_photo | country rule (India = anemic) | 88.6% | 0.885 |
| by_photo | dummy (always no anemia) | 58.6% | 0.500 |

- Under the paper's protocol, accuracy is 95.6% (93.9%-96.9% across seeds) against the paper's single run of 98.47%. The protocol roughly reproduces; with a 131-image test set, one lucky split can add a few points.
- Keeping all versions of a photo on one side drops accuracy to 88.5%, level with the country rule (88.6%). India is mostly anemic and Italy mostly not, so where a photo was taken already predicts most labels. The ViT still ranks better (AUROC 0.932 vs 0.885), so it learns something beyond country.
- Within one country, split by photo (all seeds pooled): India AUROC 0.530 [0.293, 0.799], chance level (53 of 64 test photos anemic); Italy 0.934 [0.857, 0.983], but only 7 anemic test photos, and Italy's cutoff (Hb <= 10.5) picks out more severe anemia. Inside India, where most of the anemia is, the ViT does not separate anemic from not.

## Limitations

- **Different populations.** The eyelid images are children under five; the planned survey model is for women 15-49. The two models are not about the same people, and the survey model is not built yet.
- **Not population prevalence.** 59.7% of CP-AnemiC rows are anemic. The sample leaned toward children suspected of anemia, so this share is not the prevalence in Ghana.
- **Cutoffs differ.** Children use Hb < 11.0 g/dL; adults (Eyes-Defy, DHS) will use WHO cutoffs by sex and pregnancy. Results across datasets must say which cutoff each uses.
- **Lighting, camera and cropping differ between image datasets.** CP-AnemiC comes cut out on a transparent background; Eyes-Defy ships separate masks. Cross-dataset numbers will mix these differences with real transfer, and that test has not run yet.
- **Duplicates.** Near-duplicates are found by a perceptual hash with a threshold checked by eye; more aggressively re-cropped copies can slip through (the no-color control stays above chance on the 476-photo set). The headline uses only photos with no detected copy, which is the most conservative choice.
- **Metadata on copied photos is unreliable.** Copies carry different Hb, hospital, age and sex, so we cannot tell which record a copied photo belongs to. Hospital-level analysis uses only photos with no copy.
- **No patient ID.** We assume one child per distinct photo.
- **Small hospitals.** Two hospitals contribute very few photos, so leave-one-hospital-out pools predictions across hospitals rather than reporting each one.

## Next three steps with two more weeks

1. **Run the cross-dataset test (Stage 5)**: train on CP-AnemiC photos with no copy, test on Eyes-Defy, and the reverse. Also test Eyes-Defy within each country, to separate pallor from country.
2. **Build the DHS survey model (Stages 6-8)** once DHS access is approved.
3. **Ask the CP-AnemiC authors about the duplicates.**
