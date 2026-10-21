# Summary: anemia screening baselines

Built from `results/*.json` by `python -m src.summary`. Anemia = Hb < 11.0 g/dL (children 6-59 months, CP-AnemiC). Every estimate has a bootstrap 95% CI from 2000 resamples. Commits that produced each file: `stage2_audit` 7b721ca, `stage3_color` bb9a083, `stage4_probe` 945b76d, `replicate_cnn` 1fd7b65, `stage_eyes_defy_audit` 2239613, `replicate_vit` 07ef8ab, `stage5_cross_dataset` 30c6633, `stage6_survey_audit` 64ec683, `stage7_survey_baselines` dfd3d1f, `stage8_survey_transfer` cf16772, `fusion_simulation` 30c6633.

## Status

| Stage | Status |
|---|---|
| 1. Skeleton and metrics | done |
| 2. CP-AnemiC audit | done |
| 3. Color-feature baseline | done |
| 4. Frozen DINOv2 probe | done |
| Replication of the CP-AnemiC paper's CNN | done |
| Replication of the 2025 ViT paper on Eyes-Defy-Anemia | done |
| 5. Cross-dataset test with Eyes-Defy-Anemia | done |
| 6. DHS survey audit and datasets | done |
| 7. Survey baselines on Zimbabwe | done |
| 8. Survey transfer to Malawi and Zambia | done |
| Fusion simulation (survey + eyelid) | done |

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
- Choosing the best epoch on the test set, which the paper may have done, lifts the paper protocol from 95.6% to 96.5% (best single run 96.9%); 0 of 5 runs reach 98.47%. That does not explain the gap; the paper's particular split or an unreported detail remains the likeliest cause.
- Within one country, split by photo (all seeds pooled): India AUROC 0.530 [0.293, 0.799], chance level (53 of 64 test photos anemic); Italy 0.934 [0.857, 0.983], but only 7 anemic test photos, and Italy's cutoff (Hb <= 10.5) picks out more severe anemia. Inside India, where most of the anemia is, the ViT does not separate anemic from not.

## 7. Eyelid models across datasets (Stage 5)

Train on one image dataset, test on the other unchanged. Input on both sides is the palpebral conjunctiva. CP-AnemiC: the 388 photos with no copy, Hb < 11.0 g/dL (children). Eyes-Defy: 217 photos, WHO adult cutoffs by sex (women < 12.0, men < 13.0 g/dL), not the ViT paper's labels. Full table: `reports/stage5_cross_dataset.md`.

| model | CP-AnemiC within (CV) | CP-AnemiC -> Eyes-Defy | -> India only | -> Italy only | Eyes-Defy -> CP-AnemiC |
|---|---|---|---|---|---|
| logistic_regression | 0.680 [0.627, 0.730] | 0.315 [0.245, 0.389] | 0.491 [0.359, 0.625] | 0.441 [0.313, 0.567] | 0.506 [0.451, 0.562] |
| random_forest | 0.796 [0.752, 0.839] | 0.587 [0.511, 0.662] | 0.706 [0.573, 0.826] | 0.790 [0.692, 0.876] | 0.580 [0.521, 0.635] |
| dinov2_probe | 0.672 [0.616, 0.723] | 0.624 [0.546, 0.697] | 0.568 [0.443, 0.692] | 0.715 [0.598, 0.827] | 0.561 [0.502, 0.618] |
| rule_low_redness | 0.650 [0.597, 0.705] | 0.835 [0.779, 0.884] | 0.671 [0.552, 0.787] | 0.924 [0.856, 0.977] | 0.650 [0.597, 0.705] |

- Learned models do not survive the change of dataset: the color logistic regression ranks Eyes-Defy patients worse than chance, and the random forest and DINOv2 probe fall to near chance. They fit color patterns tied to CP-AnemiC's camera and lighting.
- The untrained redness rule (less red = anemic) holds up best in both directions, so redness itself carries across datasets even when learned color models do not.

## 8. Survey model on Zimbabwe (Stages 6-7)

DHS Individual Recode, women 15-49 with an Hb result. Anemic = DHS `v457` at any severity (WHO for women: Hb adjusted for altitude and smoking < 12.0 g/dL, < 11.0 if pregnant). Inputs: age, pregnancy, wealth quintile, education, contraceptive method, urban/rural.

| country | survey year | tested women | anemic | anemic if pregnant | clusters |
|---|---|---|---|---|---|
| zimbabwe | 2015 | 9265 | 27.3% | 32.5% | 400 |
| malawi | 2024 | 10000 | 26.1% | 36.2% | 793 |
| zambia | 2024 | 6829 | 31.1% | 37.5% | 545 |

Zimbabwe, StratifiedGroupKFold by cluster (5 folds x 5 seeds), bootstrap over clusters:

| model | weights | AUROC | PR-AUC | Sens @ 90% spec | Recall, top 30% tested | Brier |
|---|---|---|---|---|---|---|
| rule: random | - | 0.493 [0.480, 0.507] | 0.270 [0.256, 0.285] | 0.101 [0.088, 0.117] | 0.294 [0.279, 0.309] | - |
| rule: pregnant_first | - | 0.504 [0.490, 0.517] | 0.280 [0.266, 0.296] | 0.116 [0.100, 0.129] | 0.303 [0.289, 0.319] | - |
| dummy | none | 0.490 [0.474, 0.507] | 0.273 [0.259, 0.289] | 0.104 [0.087, 0.122] | 0.292 [0.275, 0.314] | 0.199 [0.193, 0.204] |
| dummy | survey | 0.498 [0.480, 0.516] | 0.271 [0.255, 0.288] | 0.100 [0.079, 0.115] | 0.296 [0.276, 0.319] | 0.199 [0.193, 0.204] |
| logistic_regression | none | 0.580 [0.566, 0.593] | 0.323 [0.307, 0.341] | 0.146 [0.123, 0.163] | 0.379 [0.363, 0.394] | 0.196 [0.190, 0.201] |
| logistic_regression | survey | 0.579 [0.566, 0.593] | 0.324 [0.307, 0.343] | 0.148 [0.132, 0.166] | 0.376 [0.359, 0.393] | 0.196 [0.190, 0.201] |
| hist_gradient_boosting | none | 0.570 [0.556, 0.583] | 0.320 [0.303, 0.338] | 0.146 [0.129, 0.163] | 0.358 [0.341, 0.375] | 0.197 [0.192, 0.203] |
| hist_gradient_boosting | survey | 0.566 [0.552, 0.579] | 0.315 [0.299, 0.332] | 0.145 [0.126, 0.162] | 0.357 [0.339, 0.373] | 0.199 [0.193, 0.204] |

- The survey answers carry a weak signal: logistic regression AUROC 0.580 [0.566, 0.593]; testing the top 30% by risk finds 37.9% of anemic women, vs 29.4% for random testing.
- Testing pregnant women first barely helps (they are about 6% of women). Survey weights change nothing.

## 9. Survey model transferred to Malawi and Zambia (Stage 8)

Fit on all of Zimbabwe 2015, applied unchanged to Malawi 2024 and Zambia 2024. The DHS wealth index is built separately for each country and survey, so the same quintile does not mean the same wealth across the three.

| country | n | prevalence | AUROC | PR-AUC | Sens @ 90% spec | Recall, top 30% tested | Brier | Cal. slope | Cal. intercept |
|---|---|---|---|---|---|---|---|---|---|
| zimbabwe (cross-validated) | 9265 | 27.3% | 0.580 [0.566, 0.593] | 0.323 [0.307, 0.341] | 0.146 [0.123, 0.163] | 0.379 [0.363, 0.394] | 0.196 [0.190, 0.201] | 0.932 [0.769, 1.095] | -0.066 [-0.234, 0.099] |
| malawi (untouched) | 10000 | 26.1% | 0.568 [0.555, 0.581] | 0.295 [0.281, 0.310] | 0.122 [0.105, 0.135] | 0.351 [0.335, 0.367] | 0.192 [0.187, 0.196] | 0.903 [0.738, 1.069] | -0.236 [-0.395, -0.077] |
| zambia (untouched) | 6829 | 31.1% | 0.565 [0.549, 0.580] | 0.356 [0.336, 0.376] | 0.137 [0.117, 0.157] | 0.349 [0.331, 0.365] | 0.212 [0.207, 0.217] | 0.997 [0.783, 1.217] | 0.087 [-0.114, 0.287] |

- The logistic regression keeps most of its small signal in both countries; calibration stays near ideal, with Malawi's risk overestimated by a few points.
- Platt scaling on 200 local women brings the mean predicted risk to the local prevalence (Malawi 29.0% -> 26.2% vs 26.1%), but with so weak a signal its fitted slope is noisy and the Brier score does not improve. Gradient boosting is overconfident everywhere and gains more from recalibration (see `reports/stage8_transfer.md`).

## 10. Fusion SIMULATION: survey score plus an eyelid score

No dataset has an eyelid photo and survey answers for the same woman. Each Malawi and Zambia woman keeps her real survey score and gets an eyelid redness score drawn from Eyes-Defy photos with her anemia status, assuming the two signals are independent given anemia. This shows what fusion could add if an eyelid score worked as well on these women as on Eyes-Defy; it is not evidence that it does.

| country | scores | AUROC | PR-AUC | Sens @ 90% spec | Recall, top 30% tested |
|---|---|---|---|---|---|
| malawi | survey only | 0.568 [0.555, 0.581] | 0.295 [0.281, 0.310] | 0.122 [0.105, 0.135] | 0.351 [0.335, 0.367] |
| malawi | image only, weak (India photos) | 0.674 [0.662, 0.687] | 0.507 [0.487, 0.526] | 0.255 [0.237, 0.273] | 0.512 [0.483, 0.524] |
| malawi | survey + image, weak (India photos) | 0.687 [0.676, 0.700] | 0.495 [0.473, 0.516] | 0.305 [0.284, 0.328] | 0.487 [0.471, 0.504] |
| malawi | image only, strong (Italy photos) | 0.927 [0.921, 0.932] | 0.801 [0.782, 0.818] | 0.818 [0.755, 0.831] | 0.818 [0.803, 0.833] |
| malawi | survey + image, strong (Italy photos) | 0.929 [0.924, 0.935] | 0.789 [0.770, 0.807] | 0.789 [0.767, 0.814] | 0.821 [0.807, 0.836] |
| zambia | survey only | 0.565 [0.549, 0.580] | 0.356 [0.336, 0.376] | 0.137 [0.117, 0.157] | 0.349 [0.331, 0.365] |
| zambia | image only, weak (India photos) | 0.675 [0.660, 0.689] | 0.559 [0.538, 0.578] | 0.258 [0.240, 0.277] | 0.466 [0.438, 0.496] |
| zambia | survey + image, weak (India photos) | 0.689 [0.674, 0.703] | 0.555 [0.532, 0.576] | 0.314 [0.292, 0.340] | 0.485 [0.467, 0.503] |
| zambia | image only, strong (Italy photos) | 0.923 [0.915, 0.930] | 0.830 [0.812, 0.849] | 0.779 [0.764, 0.836] | 0.766 [0.733, 0.786] |
| zambia | survey + image, strong (Italy photos) | 0.925 [0.919, 0.933] | 0.818 [0.799, 0.838] | 0.784 [0.765, 0.815] | 0.771 [0.753, 0.786] |

- The eyelid score would carry most of the signal; the survey score adds about 0.01-0.02 AUROC on top of a weak eyelid score and nothing on top of a strong one.

## Limitations

- **Different populations.** CP-AnemiC is children under five; Eyes-Defy is adults; the survey model is women 15-49. No dataset links an eyelid photo to survey answers, so fusion is only simulated.
- **Not population prevalence.** 59.7% of CP-AnemiC rows are anemic. The sample leaned toward children suspected of anemia, so this share is not the prevalence in Ghana.
- **Cutoffs differ.** CP-AnemiC uses Hb < 11.0 g/dL (children). Eyes-Defy uses WHO adult cutoffs by sex (pregnancy not recorded, assumed not pregnant), except the ViT replication, which uses that paper's cutoffs. DHS uses its own `v457` (WHO for women, adjusted for altitude and smoking).
- **Lighting, camera and cropping differ between image datasets.** Cross-dataset numbers mix these differences with real transfer; the collapse of learned color models suggests the camera difference dominates.
- **Survey rounds differ.** Zimbabwe is 2015, Malawi and Zambia 2024, so transfer mixes country with time. Malawi's wealth index is missing for 408 tested women, all in the Central region (imputed inside the model).
- **Duplicates.** Near-duplicates are found by a perceptual hash with a threshold checked by eye; more aggressively re-cropped copies can slip through (the no-color control stays above chance on the 476-photo set). The headline uses only photos with no detected copy, which is the most conservative choice.
- **Metadata on copied photos is unreliable.** Copies carry different Hb, hospital, age and sex, so we cannot tell which record a copied photo belongs to. Hospital-level analysis uses only photos with no copy.
- **No patient ID.** We assume one child per distinct photo.
- **Small hospitals.** Two hospitals contribute very few photos, so leave-one-hospital-out pools predictions across hospitals rather than reporting each one.

## Next three steps with two more weeks

1. **Collect paired data**: an eyelid photo and the survey answers for the same women, to test fusion for real instead of by simulation.
2. **Fix the camera gap before learning colors**: photograph a reference color card with each eyelid, or normalize colors per device, then retest the cross-dataset models; the redness rule suggests the signal survives.
3. **Compare the survey baseline with Chemhaka et al.** once their numbers are in hand, and ask the CP-AnemiC authors about the duplicate photos.
