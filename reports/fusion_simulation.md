# Fusion simulation: survey score plus an eyelid score

**This is a simulation, not a trained model.** No dataset has an eyelid photo and survey answers for the same woman. Each Malawi and Zambia woman keeps her real survey risk score (logistic regression fit on all of Zimbabwe, Stage 8) and gets an eyelid score drawn at random from Eyes-Defy photos with her anemia status. The eyelid score is the untrained redness rule (minus mean a*), the image signal that held up best across datasets (Stage 5). Combined score = survey logit + slope x eyelid score, with the slope fit on the Eyes-Defy subset; this assumes the two signals are independent given anemia status, which is unproven. Two scenarios: eyelid scores from Indian photos (a weak signal) and from Italian photos (a strong one, but Italy's anemic adults are few and fairly severe). Eyes-Defy labels use WHO adult cutoffs by sex; the survey labels use DHS `v457` (WHO for women, altitude- and smoking-adjusted). Bootstrap 95% CIs (2000 resamples) resample whole clusters, for one random image draw.

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

Spread of the point estimate over 20 different random image draws:

| scores | AUROC range | recall, top 30% range |
|---|---|---|
| malawi / survey + image, weak (India photos) | 0.673-0.697 | 0.468-0.504 |
| malawi / survey + image, strong (Italy photos) | 0.920-0.932 | 0.809-0.834 |
| zambia / survey + image, weak (India photos) | 0.668-0.698 | 0.468-0.500 |
| zambia / survey + image, strong (Italy photos) | 0.921-0.932 | 0.761-0.787 |

![Fusion simulation](figures/fusion_simulation.png)
