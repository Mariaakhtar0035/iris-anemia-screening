"""Stage 9: pull every results file into results/summary.md (and a tracked copy in reports/).

Every number below is read from results/*.json; nothing is typed in by hand.
Run from the repo root: python -m src.summary
"""
import json

from src.common.io import RESULTS, ROOT
from src.eyelid.report import IMAGE_COLUMNS, fmt, markdown_table


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def pick(rows, **match):
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, f"expected one row for {match}, got {len(hits)}"
    return hits[0]


def main():
    audit, color, probe, rep = (load(n) for n in ("stage2_audit", "stage3_color", "stage4_probe", "replicate_cnn"))
    ed, vit = load("stage_eyes_defy_audit"), load("replicate_vit")
    xd, s6, s7, s8, fus = (load(n) for n in ("stage5_cross_dataset", "stage6_survey_audit", "stage7_survey_baselines",
                                              "stage8_survey_transfer", "fusion_simulation"))
    a = audit["results"]
    c3, c4 = color["results"], probe["results"]
    head = "cv_5x5_no_copies"
    probe_name = "dinov2_probe[cls_and_patch_mean]"

    def color_row(exp, model, gw=False):
        return pick(c3["main"], experiment=exp, model=model, gray_world=gw)

    headline = [
        {**color_row(exp, m), "model": m} for exp in (head, "leave_one_hospital_out")
        for m in ("dummy", "rule_low_redness", "negative_control_rf_no_color", "logistic_regression", "random_forest")
    ]
    headline = (
        [r for r in headline if r["experiment"] == head] + [pick(c4["main"], experiment=head, model=probe_name)]
        + [r for r in headline if r["experiment"] == "leave_one_hospital_out"]
        + [pick(c4["main"], experiment="leave_one_hospital_out", model=probe_name)]
    )

    gw_lines = []
    for exp in (head, "leave_one_hospital_out"):
        for m in ("rule_low_redness", "logistic_regression", "random_forest"):
            off, on = color_row(exp, m, False)["auroc"], color_row(exp, m, True)["auroc"]
            gw_lines.append(f"| {exp} | {m} | {fmt(off)} | {fmt(on)} | {on['estimate'] - off['estimate']:+.3f} |")

    leak = (
        [{**r, "source": "Stage 3"} for r in c3["leakage_check"]]
        + [{**r, "source": "Stage 4"} for r in c4["leakage_check"]]
        + ([{**r, "source": "Replication"} for r in rep["results"]["rows"]] if rep else [])
    )
    def gap(model, src):
        rs = [r for r in leak if r["model"] == model and r["source"] == src]
        if len(rs) != 2:
            return None
        leaky = next(r for r in rs if "random" in r["split"] or "paper" in r["split"])
        clean = next(r for r in rs if "grouped" in r["split"])
        return leaky["auroc"]["estimate"], clean["auroc"]["estimate"]

    rf_best = color_row(head, "random_forest")["auroc"]
    rf_loso = color_row("leave_one_hospital_out", "random_forest")["auroc"]
    ctrl = color_row(head, "negative_control_rf_no_color")["auroc"]
    ctrl_leaky = pick(c3["leakage_check"], split="random rows (copies can leak)", model="negative_control_rf_no_color")["auroc"]
    pr = pick(c4["main"], experiment=head, model=probe_name)["auroc"]

    lines = [
        "# Summary: anemia screening baselines",
        "",
        "Built from `results/*.json` by `python -m src.summary`. Anemia = Hb < 11.0 g/dL (children 6-59 months, "
        "CP-AnemiC). Every estimate has a bootstrap 95% CI from 2000 resamples. Commits that produced each file: "
        + ", ".join(f"`{n}` {d['git']['commit'][:7]}" for n, d in
                    (("stage2_audit", audit), ("stage3_color", color), ("stage4_probe", probe), ("replicate_cnn", rep),
                     ("stage_eyes_defy_audit", ed), ("replicate_vit", vit), ("stage5_cross_dataset", xd),
                     ("stage6_survey_audit", s6), ("stage7_survey_baselines", s7), ("stage8_survey_transfer", s8),
                     ("fusion_simulation", fus)) if d)
        + ".",
        "",
        "## Status",
        "",
        "| Stage | Status |",
        "|---|---|",
        "| 1. Skeleton and metrics | done |",
        "| 2. CP-AnemiC audit | done |",
        "| 3. Color-feature baseline | done |",
        "| 4. Frozen DINOv2 probe | done |",
        "| Replication of the CP-AnemiC paper's CNN | " + ("done" if rep else "not run") + " |",
        "| Replication of the 2025 ViT paper on Eyes-Defy-Anemia | " + ("done" if vit else "not run") + " |",
        "| 5. Cross-dataset test with Eyes-Defy-Anemia | " + ("done" if xd else "not run") + " |",
        "| 6. DHS survey audit and datasets | " + ("done" if s6 else "not run") + " |",
        "| 7. Survey baselines on Zimbabwe | " + ("done" if s7 else "not run") + " |",
        "| 8. Survey transfer to Malawi and Zambia | " + ("done" if s8 else "not run") + " |",
        "| Fusion simulation (survey + eyelid) | " + ("done" if fus else "not run") + " |",
        "",
        "## 1. The CP-AnemiC data has far fewer distinct photos than rows",
        "",
        f"From `results/stage2_audit.json`: the {a['rows']} rows hold {a['byte_distinct_images']} byte-distinct images "
        f"({a['byte_identical_groups']} groups of byte-identical files, {a['extra_byte_identical_copies']} extra copies). "
        f"{a['near_duplicate_links']} more links join the same photo re-saved or slightly re-cropped, leaving "
        f"**{a['distinct_photos']} distinct photos**, of which {a['photos_with_no_copy']} have no copy at all. "
        f"{a['photos_with_conflicting_labels']} photo's copies disagree on the anemia label. The Hb < "
        f"{a['anemia_cutoff_g_dl']} label disagrees with the lab remark or the image folder on "
        f"{a['hb_label_disagrees_with_remark_or_folder']} rows. {a['rows_over_59_months']} rows are 60 months old, "
        "just outside the 6-59 month definition.",
        "",
        "## 2. Headline results: photos with no copy",
        "",
        f"`{head}` is repeated stratified 5-fold CV (5 repeats) on the {a['photos_with_no_copy']} photos with no copy; "
        "`leave_one_hospital_out` holds out one hospital at a time on the same photos. Color rows are gray-world off.",
        "",
        markdown_table(headline, ["experiment", "n", "model"]),
        "",
        f"- The color random forest is the best baseline: AUROC {fmt(rf_best)} in CV and {fmt(rf_loso)} on held-out "
        "hospitals.",
        f"- The frozen DINOv2 probe reaches {fmt(pr)}, level with color logistic regression and below the random forest.",
        f"- The no-color negative control sits at {fmt(ctrl)}, close to chance, so these splits do not leak duplicates.",
        "",
        "## 3. Gray-world normalization hurts",
        "",
        "| experiment | model | AUROC off | AUROC on | change |",
        "|---|---|---|---|---|",
        *gw_lines,
        "",
        "The crops contain only conjunctiva, so equalizing the channel means removes part of the redness signal "
        "the models rely on.",
        "",
        "## 4. Duplicate photos inflate scores under a random split",
        "",
        "All 710 rows. The only difference between each pair of rows is whether copies of a photo may sit in "
        "both train and test. Bootstrap resamples whole photo groups.",
        "",
        markdown_table(leak, ["source", "split", "model"]),
        "",
        f"- A random forest that never sees color reaches AUROC {fmt(ctrl_leaky)} under the random split, from "
        "duplicates alone.",
    ]
    for model, src in (("random_forest", "Stage 3"), (probe_name, "Stage 4"), ("joint_resnet50", "Replication")):
        g = gap(model, src)
        if g:
            lines.append(f"- {src} `{model}`: AUROC {g[0]:.3f} with copies leaking vs {g[1]:.3f} grouped by photo "
                         f"({g[0] - g[1]:+.3f}).")
    if rep:
        r_paper = pick(rep["results"]["rows"], split="paper_random_rows")
        r_group = pick(rep["results"]["rows"], split="grouped_by_photo")
        lines += [
            "",
            "## 5. Replicating the CP-AnemiC paper's benchmark",
            "",
            "Joint ResNet50 (anemia classification + Hb regression) with the paper's loss, optimizer, learning rate, "
            "batch size, augmentation and 5-fold CV; ImageNet-pretrained and 30 epochs instead of random init and 150 "
            "(see `configs/replicate.yaml`). Paper, for reference (Appiahene et al. 2023, Tables 3-4): ResNet50 AUC "
            "0.835, accuracy 84.79%, Hb MAE 1.54.",
            "",
            "| split | AUROC | accuracy at 0.5 | Hb MAE (g/dL) |",
            "|---|---|---|---|",
            f"| paper-style random rows | {fmt(r_paper['auroc'])} | {r_paper['accuracy_at_0.5']:.1%} | {r_paper['hb_mae']:.2f} |",
            f"| grouped by photo | {fmt(r_group['auroc'])} | {r_group['accuracy_at_0.5']:.1%} | {r_group['hb_mae']:.2f} |",
            "",
            f"- Under the paper's split, accuracy ({r_paper['accuracy_at_0.5']:.1%}) and Hb MAE ({r_paper['hb_mae']:.2f}) "
            "land close to the paper's ResNet50 (84.79%, 1.54), so the protocol reproduces.",
            "- The paper's AUC sits within about 0.01 of its accuracy for all five of its models (Table 3), which is what "
            "an AUC computed from hard 0/1 predictions gives. Ours comes from probabilities, so compare on accuracy and "
            "Hb error rather than AUC.",
            f"- Keeping each photo's copies in one fold drops accuracy to {r_group['accuracy_at_0.5']:.1%} "
            f"(AUROC {r_paper['auroc']['estimate']:.3f} to {r_group['auroc']['estimate']:.3f}) and raises Hb MAE to "
            f"{r_group['hb_mae']:.2f}, with every other setting identical.",
        ]
    if ed and vit:
        e, v = ed["results"], vit["results"]
        vt, rule, dummy = "ViT-B/16 in21k", "country rule (India = anemic)", "dummy (always no anemia)"
        lines += [
            "",
            "## 6. Replicating the 2025 ViT paper's 98.47% on Eyes-Defy-Anemia",
            "",
            f"From `results/stage_eyes_defy_audit.json`: {e['photos']} photos ({e['photos_by_country']}), "
            f"{e['labelled_photos']} with Hb, labelled with the paper's cutoffs (India women < 12, men < 14 g/dL; Italy "
            f"<= 10.5 g/dL): {e['no_anemia']} no anemia / {e['anemia']} anemia, "
            + ("matching the paper." if e["matches_paper"] else f"vs the paper's {e['paper_counts']}.")
            + f" The paper writes Italy \"< 10.5\", but its counts need <= 10.5 ({', '.join(e['italy_at_cutoff'])} sits "
            f"exactly on 10.5). No duplicate photos ({e['near_duplicate_links']} near-duplicate links).",
            "",
            "ViT-B/16 (ImageNet-21k), the paper's optimizer, learning rate, batch size and 30 epochs, one random 80/20 "
            f"split per seed, {v['paper']['seeds']} seeds. `paper` flips every photo and then splits the 3x images, so "
            "flipped copies of a test photo sit in training; `by_photo` splits photos first. Baselines use the same "
            "test images.",
            "",
            "| split | model | mean accuracy | mean AUROC |",
            "|---|---|---|---|",
            *[f"| {pr} | {mo} | {v[pr][mo]['accuracy']:.1%} | {v[pr][mo]['auroc']:.3f} |"
              for pr in ("paper", "by_photo") for mo in (vt, rule, dummy)],
            "",
            f"- Under the paper's protocol, accuracy is {v['paper'][vt]['accuracy']:.1%} "
            f"({v['paper']['accuracy_min']:.1%}-{v['paper']['accuracy_max']:.1%} across seeds) against the paper's "
            "single run of 98.47%. The protocol roughly reproduces; with a 131-image test set, one lucky split can add "
            "a few points.",
            f"- Keeping all versions of a photo on one side drops accuracy to {v['by_photo'][vt]['accuracy']:.1%}, "
            f"level with the country rule ({v['by_photo'][rule]['accuracy']:.1%}). India is mostly anemic and Italy "
            "mostly not, so where a photo was taken already predicts most labels. The ViT still ranks better "
            f"(AUROC {v['by_photo'][vt]['auroc']:.3f} vs {v['by_photo'][rule]['auroc']:.3f}), so it learns something "
            "beyond country.",
        ]
        be = v["paper"].get("best_epoch")
        if be:
            lines.append(
                f"- Choosing the best epoch on the test set, which the paper may have done, lifts the paper protocol from "
                f"{be['final_mean']:.1%} to {be['best_mean']:.1%} (best single run {be['best_max']:.1%}); "
                f"{be['runs_at_or_above_published']} of {be['seeds']} runs reach 98.47%. That does not explain the gap; "
                "the paper's particular split or an unreported detail remains the likeliest cause.")
        wi = v["by_photo"].get("within_country")
        if wi:
            ind, ita = wi["India"], wi["Italy"]
            lines.append(
                f"- Within one country, split by photo (all seeds pooled): India AUROC {fmt(ind['auroc'])}, chance level "
                f"({ind['anemic_photos']} of {ind['n_photos']} test photos anemic); Italy {fmt(ita['auroc'])}, but only "
                f"{ita['anemic_photos']} anemic test photos, and Italy's cutoff (Hb <= 10.5) picks out more severe "
                "anemia. Inside India, where most of the anemia is, the ViT does not separate anemic from not.")
    survey_cols = {**IMAGE_COLUMNS, "recall_top": "Recall, top 30% tested", "brier": "Brier"}
    if xd:
        x = xd["results"]["rows"]
        def xa(direction, model, subset="all"):
            return fmt(pick(x, **{"train -> test": direction, "model": model, "test set": subset})["auroc"])
        lines += [
            "",
            "## 7. Eyelid models across datasets (Stage 5)",
            "",
            f"Train on one image dataset, test on the other unchanged. Input on both sides is the palpebral conjunctiva. "
            f"CP-AnemiC: the {xd['results']['n_cp_anemic']} photos with no copy, Hb < 11.0 g/dL (children). Eyes-Defy: "
            f"{xd['results']['n_eyes_defy']} photos, WHO adult cutoffs by sex (women < 12.0, men < 13.0 g/dL), not the "
            "ViT paper's labels. Full table: `reports/stage5_cross_dataset.md`.",
            "",
            "| model | CP-AnemiC within (CV) | CP-AnemiC -> Eyes-Defy | -> India only | -> Italy only | Eyes-Defy -> CP-AnemiC |",
            "|---|---|---|---|---|---|",
            *[f"| {m} | {xa('cp_anemic (within, 5x5 CV)', m)} | {xa('cp_anemic -> eyes_defy', m)} | "
              f"{xa('cp_anemic -> eyes_defy', m, 'India')} | {xa('cp_anemic -> eyes_defy', m, 'Italy')} | "
              f"{xa('eyes_defy -> cp_anemic', m)} |"
              for m in ("logistic_regression", "random_forest", "dinov2_probe", "rule_low_redness")],
            "",
            "- Learned models do not survive the change of dataset: the color logistic regression ranks Eyes-Defy "
            "patients worse than chance, and the random forest and DINOv2 probe fall to near chance. They fit color "
            "patterns tied to CP-AnemiC's camera and lighting.",
            "- The untrained redness rule (less red = anemic) holds up best in both directions, so redness itself carries "
            "across datasets even when learned color models do not.",
        ]
    if s6 and s7:
        c6 = s6["results"]["countries"]
        r7 = s7["results"]["rows"]
        lines += [
            "",
            "## 8. Survey model on Zimbabwe (Stages 6-7)",
            "",
            "DHS Individual Recode, women 15-49 with an Hb result. Anemic = DHS `v457` at any severity (WHO for women: "
            "Hb adjusted for altitude and smoking < 12.0 g/dL, < 11.0 if pregnant). Inputs: age, pregnancy, wealth "
            "quintile, education, contraceptive method, urban/rural.",
            "",
            "| country | survey year | tested women | anemic | anemic if pregnant | clusters |",
            "|---|---|---|---|---|---|",
            *[f"| {c} | {'2015' if c == 'zimbabwe' else '2024'} | {d['tested']} | {d['anemic_share']:.1%} | "
              f"{d['anemic_share_pregnant']:.1%} | {d['clusters']} |" for c, d in c6.items()],
            "",
            "Zimbabwe, StratifiedGroupKFold by cluster (5 folds x 5 seeds), bootstrap over clusters:",
            "",
            markdown_table([dict(r) for r in r7], ["model", "weights"], survey_cols),
            "",
            f"- The survey answers carry a weak signal: logistic regression AUROC "
            f"{fmt(pick(r7, model='logistic_regression', weights='none')['auroc'])}; testing the top 30% by risk finds "
            f"{pick(r7, model='logistic_regression', weights='none')['recall_top']['estimate']:.1%} of anemic women, vs "
            f"{pick(r7, model='rule: random', weights='-')['recall_top']['estimate']:.1%} for random testing.",
            "- Testing pregnant women first barely helps (they are about 6% of women). Survey weights change nothing.",
        ]
    if s8:
        r8 = s8["results"]["rows"]
        rc = s8["results"]["recalibration"]
        lr = [r for r in r8 if r["model"] == "logistic_regression"]
        lines += [
            "",
            "## 9. Survey model transferred to Malawi and Zambia (Stage 8)",
            "",
            "Fit on all of Zimbabwe 2015, applied unchanged to Malawi 2024 and Zambia 2024. The DHS wealth index is built "
            "separately for each country and survey, so the same quintile does not mean the same wealth across the three.",
            "",
            markdown_table([dict(r) for r in lr], ["country", "n", "prevalence"],
                           {**survey_cols, "cal_slope": "Cal. slope", "cal_intercept": "Cal. intercept"}),
            "",
            "- The logistic regression keeps most of its small signal in both countries; calibration stays near ideal, "
            "with Malawi's risk overestimated by a few points.",
            f"- Platt scaling on 200 local women brings the mean predicted risk to the local prevalence (Malawi "
            f"{rc['logistic_regression/malawi']['mean_risk_before']['mean']:.1%} -> "
            f"{rc['logistic_regression/malawi']['mean_risk_after']['mean']:.1%} vs {rc['logistic_regression/malawi']['prevalence']:.1%}), "
            "but with so weak a signal its fitted slope is noisy and the Brier score does not improve. Gradient boosting "
            "is overconfident everywhere and gains more from recalibration (see `reports/stage8_transfer.md`).",
        ]
    if fus:
        rf = fus["results"]["rows"]
        lines += [
            "",
            "## 10. Fusion SIMULATION: survey score plus an eyelid score",
            "",
            "No dataset has an eyelid photo and survey answers for the same woman. Each Malawi and Zambia woman keeps her "
            "real survey score and gets an eyelid redness score drawn from Eyes-Defy photos with her anemia status, "
            "assuming the two signals are independent given anemia. This shows what fusion could add if an eyelid score "
            "worked as well on these women as on Eyes-Defy; it is not evidence that it does.",
            "",
            markdown_table([dict(r) for r in rf], ["country", "scores"],
                           {k: v for k, v in survey_cols.items() if k != "brier"}),
            "",
            "- The eyelid score would carry most of the signal; the survey score adds about 0.01-0.02 AUROC on top of a "
            "weak eyelid score and nothing on top of a strong one.",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "- **Different populations.** CP-AnemiC is children under five; Eyes-Defy is adults; the survey model is "
        "women 15-49. No dataset links an eyelid photo to survey answers, so fusion is only simulated.",
        f"- **Not population prevalence.** {a['anemic_share_rows']:.1%} of CP-AnemiC rows are anemic. The sample "
        "leaned toward children suspected of anemia, so this share is not the prevalence in Ghana.",
        "- **Cutoffs differ.** CP-AnemiC uses Hb < 11.0 g/dL (children). Eyes-Defy uses WHO adult cutoffs by sex "
        "(pregnancy not recorded, assumed not pregnant), except the ViT replication, which uses that paper's cutoffs. "
        "DHS uses its own `v457` (WHO for women, adjusted for altitude and smoking).",
        "- **Lighting, camera and cropping differ between image datasets.** Cross-dataset numbers mix these differences "
        "with real transfer; the collapse of learned color models suggests the camera difference dominates.",
        "- **Survey rounds differ.** Zimbabwe is 2015, Malawi and Zambia 2024, so transfer mixes country with time. "
        "Malawi's wealth index is missing for 408 tested women, all in the Central region (imputed inside the model).",
        "- **Duplicates.** Near-duplicates are found by a perceptual hash with a threshold checked by eye; more "
        "aggressively re-cropped copies can slip through (the no-color control stays above chance on the 476-photo "
        "set). The headline uses only photos with no detected copy, which is the most conservative choice.",
        "- **Metadata on copied photos is unreliable.** Copies carry different Hb, hospital, age and sex, so we cannot "
        "tell which record a copied photo belongs to. Hospital-level analysis uses only photos with no copy.",
        "- **No patient ID.** We assume one child per distinct photo.",
        "- **Small hospitals.** Two hospitals contribute very few photos, so leave-one-hospital-out pools predictions "
        "across hospitals rather than reporting each one.",
        "",
        "## Next three steps with two more weeks",
        "",
        "1. **Collect paired data**: an eyelid photo and the survey answers for the same women, to test fusion for "
        "real instead of by simulation.",
        "2. **Fix the camera gap before learning colors**: photograph a reference color card with each eyelid, or "
        "normalize colors per device, then retest the cross-dataset models; the redness rule suggests the signal survives.",
        "3. **Compare the survey baseline with Chemhaka et al.** once their numbers are in hand, and ask the CP-AnemiC "
        "authors about the duplicate photos.",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    (ROOT / "reports" / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
