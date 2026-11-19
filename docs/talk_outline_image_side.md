# Group talk, image side: rough outline (draft for the team)

From the 2026-11-17 mentor meeting: the talk is 10-15 minutes for the whole group, and the mentors will review a rough
draft. This is the image part, about 4-5 minutes, 7 slides. Ghana eyelids, children 6-59 months, anemic = **Hb < 11.0
g/dL**. Every number has its 95% range in the linked report.

**Where it sits in the group story** (the triage framing the mentors agreed with): the survey ranks areas and flags
likely severe cases; the eyelid photo is the last, cheap check before a blood test.

| # | Slide | One message | Numbers | Source |
|---|---|---|---|---|
| 1 | The photos | A phone photo of the inner eyelid; paler means more likely anemic | 388 Ghana children, 165 anemic, 10 hospitals | [Stage 3](../reports/stage3_color.md) |
| 2 | What the model sees | 29 color numbers from the cut-out (redness, brightness, saturation, hue); models from an untrained rule to TabPFN | rule 0.65; vision models 0.67-0.79 | [Stage 4](../reports/stage4_probe.md), [Stage 9](../reports/stage9_ceiling.md) |
| 3 | Headline, both splits | A table model on 29 numbers beats the vision models, and holds on a hospital it never saw | TabPFN 0.919 random split / 0.913 one hospital held out; MLP 0.870 / 0.829; logistic regression 0.680 / 0.672; rule 0.650 | [Stage 37](../reports/stage37_model_vs_features.md) |
| 4 | 100 children, 20 tests | Testing the 20 the photo flags highest catches twice as many anemic children as a random pick | random 20%, rule 29%, TabPFN 43% (at most 47% possible) | [Stage 38](../reports/stage38_presentation_numbers.md) |
| 5 | The catch | Part of the Ghana score is how the anemic and healthy photos were handled, not paleness: the score jumps at the cutoff, numbers that shouldn't carry paleness score 0.90, and it gets worse for the sickest children | near the cutoff 0.94 vs 0.72 for a steady signal; hue/green/blue 0.90; Hb < 8: 0.75 (chart) | [Stage 30](../reports/stage30_handling_check.md), [37](../reports/stage37_model_vs_features.md), [38](../reports/stage38_presentation_numbers.md) |
| 6 | It doesn't transfer (negative result) | Trained on Ghana, tested on adults in Italy and India, TabPFN falls below the untrained rule | 0.60 vs rule 0.835 | [TabPFN transfer](../reports/tabpfn_transfer.md) |
| 7 | Takeaways | Per-site models; the checks are the contribution; the honest Ghana number is about 0.60-0.65; what new data would need (anemic and healthy photos taken the same way, with Hb) | - | [Stage 35](../reports/stage35_redness_check.md), [36](../reports/stage36_forced_redness.md) |

Chart for slide 5: [`reports/figures/stage38_severity_auroc.png`](../reports/figures/stage38_severity_auroc.png).

**For the mentors' "top 10% surprisingly high":** no. On the sickest 9% (Hb < 8) no score does better than on all
anemic children. TabPFN clearly drops (0.92 to 0.75, -0.17 [-0.24, -0.11]), about as much as a score that knows only
the anemic and healthy folder falls on these children (1.00 to 0.82); the rule and logistic regression don't clearly change (Stage 38).
