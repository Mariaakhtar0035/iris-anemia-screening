# Draft paper section: a photo-handling trace, and the checks that find it

Draft for the team (2026-11-20). It pulls Stages 29-39 into one section; every number and its 95% range is in the
linked README section and stage report. Cutoffs: Ghana children 6-59 months, anemic = **Hb < 11.0 g/dL**; India and
Italy adults, WHO by sex (**women < 12.0, men < 13.0 g/dL**, pregnancy not recorded); Moscow adults **Hb < 12.0 g/dL**
for everyone (no sex column). TabPFN on the 29 color numbers unless said otherwise.

## The problem

The three Ghana sets (CP-AnemiC eyelids, and the palm and nail sets) store anemic and healthy photos in separate
folders. Anything that differed in how each folder was photographed, processed or saved becomes a second label that a
model can read with no anemia in it. Clean splits and leak checks can't see it: it is in every photo, training and test
alike. TabPFN passed all our leak checks on the Ghana eyelids (shuffled labels, hidden copies, file metadata, model
chosen without the test children; [6.14](../README.md#614-a-full-audit-of-the-tabpfn-eyelid-result-stage-29)) and
still failed on 61 differently processed Ghana photos (0.586 vs the rule's 0.673).

The checks below rest on one fact: pallor changes smoothly with Hb, while the folder changes in one step at a cutoff
that is a convention.

## The checks

| Check | A score that reads pallor | Ghana eyelids (388) | Clean sets (India, Italy, Moscow) | Needs |
|---|---|---|---|---|
| **Jump test**: AUROC within 1 g/dL of the cutoff vs a steady score with the same overall AUROC ([6.15](../README.md#615-does-the-score-read-anemia-or-how-the-photos-were-handled-stage-30)) | at or below the steady score | 0.938 vs 0.72 (top 2.5%: 0.81); nails and palms jump too | below the steady score | Hb |
| **Hb inside the anemic group** ([6.15](../README.md#615-does-the-score-read-anemia-or-how-the-photos-were-handled-stage-30)) | score falls as Hb rises | +0.11 (a steady score: -0.69); nails and palms don't follow Hb either | India -0.32, Moscow -0.51 (Italy -0.28, CI crosses 0) | Hb |
| **Step at the cutoff** in a line-plus-step fit ([6.18](../README.md#618-can-the-trace-be-measured-and-taken-out-would-our-checks-catch-a-trace-we-plant-stage-33)) | about 0 | 1.55 score SDs [1.15, 1.95]; palms 1.80, nails 0.94 | -0.27 to 0.09, all CIs cross 0 | Hb |
| **Severity against a folder-only line**: AUROC at Hb < 8 vs (H·A + 0.5·M)/(H + M) ([6.23](../README.md#623-does-any-ghana-body-site-read-severity-beyond-the-folder-stage-39)) | above the line, toward a score that follows Hb | 0.749 vs 0.764 (follows Hb: 0.958); no score passes on any Ghana site | not run | Hb |
| **Small photo edits** (brightness, gamma, saturation, white balance) on the test photos ([6.17](../README.md#617-is-tabpfn-fragile-to-small-photo-edits-everywhere-stage-32)) | holds, like the rule | 6 of 7 edits break it (average cost 0.21); nails 4, palms 7 | 0 of 7 on each | photos only |
| **Numbers with no color** (size, shape, sharpness, file bytes) ([6.15](../README.md#615-does-the-score-read-anemia-or-how-the-photos-were-handled-stage-30)) | chance | 0.57-0.62 across the Ghana sets | Italy 0.86 (its known early batch) | photos only |
| **Numbers that shouldn't carry paleness** (hue, green, blue, yellow-blue) ([6.21](../README.md#621-tabpfn-or-the-29-numbers-age-and-sex-and-the-numbers-that-shouldnt-carry-paleness-stage-37)) | well below the paleness numbers | 0.903 vs 0.917 | not run | photos only |

Every Ghana set fails every check we ran on it. The clean sets pass, except Italy on the no-color numbers, which
picks up its known early photo batch. The checks also flag
gradient boosting, the MLP and the fine-tuned vision models on Ghana (near the cutoff: ViT 0.727, DINOv2 0.679,
VisionFM 0.781, against steady scores of 0.57-0.61). So the trace comes from the data, not from TabPFN.

**How strong is a pass?** We planted a fake folder trace in the clean sets: one mild edit (gamma 0.9, 10% less
saturation, or a 4% white-balance shift) on the anemic photos only
([6.18](../README.md#618-can-the-trace-be-measured-and-taken-out-would-our-checks-catch-a-trace-we-plant-stage-33)). It
lifted AUROC by 0.06-0.27 on India and Moscow, as much as TabPFN's Ghana lead over the rule (0.27). The jump test caught 10 of 24 plants:
Moscow (250 adults) 7 of 11, India (95) 3 of 11, Italy (122) 0 of 2. On a small set, a pass is weak evidence.

## What the trace explains

- **Where it sits.** The trace is in the colors, mostly saturation, blue and green
  ([6.16](../README.md#616-can-the-trace-be-taken-out-and-does-it-reach-the-hb-number-stage-31)). It runs through the
  whole cut-out, centre and edge alike
  ([6.19](../README.md#619-where-in-the-photo-is-the-ghana-trace-stage-34)). It doesn't come from the file format or
  from one hospital (TabPFN still jumps with one held out: 0.922 vs 0.71). It isn't a plain color cast either: training on edited copies
  makes TabPFN steadier, but the jump stays.
- **It can't be taken out.** Two methods were tried. Removing each folder's average costs as much on clean data as on
  Ghana (Italy -0.41). Removing the step gets the clean AUROC back for at most 1 of each model's 4-7 planted traces, and would put TabPFN
  below chance on Ghana. So
  there is no trace-free Ghana number.
- **What Ghana results we can defend.** Only the rule and simple models (about 0.60-0.65) hold up. Every input set
  that lifts TabPFN above the rule also carries the trace
  ([6.20](../README.md#620-tabpfn-on-ghana-with-redness-only-and-a-model-forced-to-read-redness-one-way-stages-35-36)).
  A model forced to read "less red = more anemic" only matches the rule. One score passes every check and beats the
  rule: logistic regression on the nails, 0.637 [0.574, 0.699].
- **Claims it overturns.** First, "a table model beats vision models" does not hold as a claim about anemia. Second,
  the Ghana Hb-number gain (error 1.38 vs 1.71 g/dL) mostly comes from getting the folder right. Third, a model
  trained on Ghana nails scores 0.30 on Moscow nails.

## What we recommend

1. If Hb is measured, run the jump test and the folder-only line. Both are cheap, and the folder line needs no
   simulation.
2. If only photos exist, run the edit test and the no-color numbers.
3. Plant a trace at your own sample size, to learn what a pass is worth.
4. For new data: photograph anemic and healthy people the same way and in mixed order, record Hb, and don't sort the
   files by label.

**Limits.** The checks give evidence, not proof. The Ghana photos still carry some real Hb signal, and it can't be
separated from the trace. The nail and palm results rest on a probable file-number link (Stage 24). The folder line
and the "not useful" numbers were run only on Ghana.
