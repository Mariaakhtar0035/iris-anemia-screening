# Predicting the Hb number instead of "anemic or not"

*Follow-up to the mentor meeting of 2026-10-31. Written for students new to machine learning; terms are explained as they come up. Research notes only, not a medical tool.*

---

## 1. The idea we tested

So far our models answered a yes/no question: **is this person anemic?** That creates a problem when mixing datasets, because "anemic" means different things in each one:

| Dataset | Who | Anemic means |
|---|---|---|
| CP-AnemiC (Ghana) | children 6-59 months | hemoglobin (Hb) below **11.0** g/dL |
| Eyes-Defy (India and Italy) | adults | Hb below **12.0** (women) or **13.0** (men) |
| Moscow fingernails | adults | Hb below **12.0** for everyone (the file doesn't record sex) |

The mentors suggested: **if you have the Hb value, predict the number itself** and report how well you estimate it. Then each place can apply its own cutoff afterwards. In principle, one model could then serve datasets with different definitions.

We tested this in three steps:
- **Stage 16:** within one dataset at a time (Ghana eyelids, Moscow nails). Can a model estimate Hb at all?
- **Stage 17:** across datasets (Ghana eyelids ↔ Eyes-Defy eyelids). Does the Hb estimate carry over?
- **Stage 18:** if it doesn't, can a few local Hb measurements at the new site correct it?

Everything uses the same 29 color numbers measured on the eyelid (or nail), scikit-learn models only, and the usual checks: held-out data, a dummy baseline, a no-color control, and 95% ranges from 2,000 bootstrap resamples.

**Two new terms:**
- **Regression:** a model that predicts a number (here, Hb in g/dL) instead of a yes/no label.
- **MAE (mean absolute error):** on average, how far the estimate is from the lab Hb. Lower is better. We compare it to a "dummy" that guesses everyone's average Hb.

---

## 2. Stage 16: estimating Hb within one dataset

| | Ghana eyelids (388 children) | Moscow nails (250 adults, fixed lighting) |
|---|---|---|
| Hb error if you just guess the average | 1.71 g/dL | 1.98 g/dL |
| **Hb error, best model** | **1.38** g/dL (gradient boosting) | **1.65** g/dL (random forest) |
| Estimates within 1 g/dL of the lab value (best vs guessing) | 47% vs 31% | 40% vs 37% |
| Still works on a hospital / photo session it never saw? | yes (1.42) | yes (1.68) |
| Model that can't see color | no better than guessing | no better than guessing |

**So yes, eyelid and nail color estimate Hb**, about 17-20% more accurately than guessing. It's rough, but it's real signal, not a shortcut. For comparison, the published CP-AnemiC deep-learning model (ResNet50) reported 1.54 g/dL on a leaky test, and our fair re-run of it gave 1.61.

### Does ranking by the estimated Hb find anemic people?

**AUROC** = the chance a random anemic person gets a higher risk score than a random healthy one (0.5 is a coin flip, 1.0 is perfect).

| | Ghana eyelids | Moscow nails |
|---|---|---|
| Rank by estimated Hb (lowest first) | 0.80 | 0.81 |
| Model trained directly on yes/no | 0.85 | 0.82 |
| Untrained "less red = anemic" rule | 0.65 | 0.70 |
| Share of people tested to catch 90% of anemic ones (ranking by estimated Hb) | 65% | 56% |

Estimating Hb costs a little ranking ability in Ghana and almost none in Moscow. **Predicting Hb is a reasonable common target.**

### A warning: the estimates are pulled toward the average

Models hedge toward the middle, so **low Hb values come out too high**. This matters if you apply the cutoff to the estimate:

| Apply the cutoff to the estimate | Ghana | Moscow |
|---|---|---|
| Anemic people correctly flagged | 74% | **59%** |
| Healthy people correctly cleared | 71% | 81% |

In Moscow the average Hb (12.8) sits above the cutoff (12.0), so the estimates rarely drop below it and **about 4 in 10 anemic adults are missed**. If a clinician were shown an estimated "12.3 g/dL", that person could easily be anemic. **An Hb estimate should always be shown with its uncertainty (about ±1.5 g/dL), or the referral threshold set lower on purpose.** Ranking people is the reliable part; reading off the number is not.

---

## 3. Stage 17: does the Hb estimate cross datasets?

We trained on all of one dataset and applied the model **unchanged** to the other, each with its own cutoff.

### Ghana children → Eyes-Defy adults

| | All | India | Italy* |
|---|---|---|---|
| Hb error, best model | 2.5 g/dL (no better than guessing Ghana's average) | 1.6 | 3.2 |
| Average (estimate − lab Hb) | −1.6 | about 0 | **−2.9** |
| AUROC, ranking by estimated Hb (best) | 0.47 | 0.66 | 0.67 |
| AUROC, yes/no model (best) | 0.59 | 0.71 | 0.79 |
| **AUROC, untrained redness rule** | **0.84** | **0.67** | **0.92** |

\*Italy can't be trusted on its own: 21 of its 23 anemic patients come from one early photo batch, and that batch alone predicts anemia at 0.91.

**Reverse direction (Eyes-Defy adults → Ghana children):** estimates come out 1.7 to 3.2 g/dL too high, and ranking reaches 0.61 to 0.63, against the rule's 0.65.

**Scaling each dataset's colors separately** (a fairness fix the branch used in Stage 14) didn't help the Hb models.

### What this shows

1. **The Hb scale doesn't carry over.** A model trained on Ghana children reads Italian adults as having child-level Hb, almost 3 g/dL too low. Applying the adults' cutoffs then calls **nearly everyone anemic** (catches 90-100% of anemic people but clears almost none of the healthy). Two differences are mixed together here, children vs adults and one camera vs another, and our data can't separate them.
2. **Ranking doesn't beat the rule.** The best cross-dataset Hb ranking (0.66 in India) only ties the untrained rule (0.67).
3. **So predicting Hb solves the "different cutoffs" problem on paper, but not the "different cameras and populations" problem.** This matches the branch's finding that the region shifts eyelid redness about four times more than anemia does.

### A trap worth knowing about

When each person is judged against their own cutoff (12 for women, 13 for men), a weak model's ranking can end up driven by **sex** rather than by the photo. In Eyes-Defy, women are more often anemic but have the lower cutoff, so "being a man" alone scores 0.34, which is backwards. Our first version of the table hid this. We now also report the ranking by the estimate alone, so a model can't look better or worse just because of the cutoff rule.

---

## 4. Take-aways

1. **Eyelid and nail color do estimate Hb** within one dataset, with an error of about 1.4-1.7 g/dL, and it holds on unseen hospitals and photo sessions.
2. **Ranking by the estimate works almost as well as a yes/no model,** so Hb is a fine shared target.
3. **Don't read the Hb number at face value.** Estimates shrink toward the average, so low values are overestimated and anemic people get missed at the cutoff.
4. **Across datasets, the Hb estimate fails** in the same way the yes/no models did: the colors map to Hb differently in each population and camera. The untrained redness rule is still the best cross-dataset option.
5. **Watch for hidden shortcuts in the scoring rule itself,** not only in the data (the sex-cutoff trap).
6. **A few local Hb measurements fix the average but not the decisions** (Stage 18, section 6). A model that can't tell a new site's people apart can't be rescued by recalibrating its number. For a clinic, **rank people and refer the lowest-ranked group** rather than reading off an estimated Hb value.

---

## 5. Limits

- **Small datasets:** 388 Ghana photos, 250 Moscow, 217 Eyes-Defy (95 India, 122 Italy).
- **Children vs adults:** the cross-dataset test mixes age groups with cameras, so we can't tell how much of the failure is each.
- **Italy's photo batch** makes its numbers unreliable.
- **Moscow uses one cutoff for everyone**, so men with Hb between 12.0 and 13.0 count as not anemic.
- **scikit-learn only.** We didn't use TabPFN (the branch's best model, 0.92 on Ghana yes/no), because it needs PyTorch.

---

## 6. Stage 18: can a few local Hb measurements fix it?

Since the Ghana model's estimates were *shifted* at the new site (almost 3 g/dL too low in Italy), we tried correcting them. Suppose the new site measures Hb on **k** people (5, 10, 20 or 40). We fit a simple correction on those k people and score everyone else at the site. We repeated this with 200 random draws of the k people and report the average, with the range across draws.

- **Shift:** add the k people's average error to every estimate.
- **Shift and scale:** fit a straight line from estimate to lab Hb on the k people.

**Italy** (average Hb 13.8, 19% anemic), random forest trained on Ghana:

| | Hb error (g/dL) | Average offset | Anemic caught at own cutoff | Healthy cleared | AUROC |
|---|---|---|---|---|---|
| No correction | 3.21 | −2.91 | 100% | 1% | 0.67 |
| Shift on 10 local people | 1.56 | about 0 | **10%** | 90% | 0.67 |
| Shift on 40 local people | 1.50 | about 0 | **4%** | 94% | 0.67 |
| Model trained on the 40 local people only | 1.20 | about 0 | 60% | 91% | 0.91 |
| Untrained redness rule | | | | | **0.92** |

**India** (average Hb 11.5, 72% anemic):

| | Hb error | Average offset | Anemic caught | Healthy cleared | AUROC |
|---|---|---|---|---|---|
| No correction | 1.61 | about 0 | 97% | **0%** | 0.66 |
| Shift on 40 local people | 1.65 | about 0 | 95% | **1%** | 0.66 |
| Model trained on the 40 local people only | 1.49 | about 0 | 88% | 14% | 0.69 |
| Untrained redness rule | | | | | **0.67** |

### What this shows

- **The correction fixes the average, as intended.** Italy's −2.9 g/dL offset disappears and the error halves.
- **But the cutoff decisions don't become usable.** They swing from "everyone is anemic" to "almost no one is". The Ghana model gives nearly everyone at the new site a *similar* estimate. Once the average is corrected, everyone sits near the site's average Hb. In Italy that average is above the cutoffs, so only 4-10% of anemic people get flagged. In India it is below them, so nearly everyone stays flagged. **The site's average decides the result, not the photo.**
- **The ranking (AUROC) can't change with a shift,** because adding the same amount to everyone keeps their order. Fitting a line on 5-10 people is unstable and sometimes even reverses the order.
- **A model trained only on the local people doesn't beat the rule either:** 0.69 vs 0.67 in India; 0.91 vs 0.92 in Italy, where the early photo batch can explain the score. This matches the branch's Stage 15 finding that a trained model needs about 80 local labelled photos before it beats the rule.

**The lesson:** the real obstacle at a new site is not calibration but **discrimination**. The Ghana model barely orders the new site's people, so no correction to its Hb number can make the cutoff work. We expected the shift to fix the cutoff decisions; it didn't.

---

## 7. Where things are

| What | Where |
|---|---|
| Stage 16 code, settings, report | `src/eyelid/hb_regression.py`, `configs/hb_regression.yaml`, `reports/stage16_hb_regression.md` |
| Stage 17 code, settings, report | `src/eyelid/hb_cross.py`, `configs/hb_cross.yaml`, `reports/stage17_hb_cross.md` |
| Stage 18 code, settings, report | `src/eyelid/hb_local.py`, `configs/hb_local.yaml`, `reports/stage18_hb_local.md` |
| Tests | `tests/test_hb_regression.py` |
| Raw results with every number and range | `results/stage16_hb_regression.json`, `stage17_hb_cross.json`, `stage18_hb_local.json` (not committed) |
