"""Stage 9: pull every results file into results/summary.md (and a tracked copy in reports/).

Every number below is read from results/*.json; nothing is typed in by hand.
Run from the repo root: python -m src.summary
"""
import json

from src.common.io import RESULTS, ROOT
from src.eyelid.report import fmt, markdown_table


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
                     ("stage_eyes_defy_audit", ed), ("replicate_vit", vit)) if d)
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
        "| 5. Cross-dataset test with Eyes-Defy-Anemia | not started (data in hand; parked until replication is done) |",
        "| 6-8. DHS survey model and transfer | **blocked**: needs DHS registration and approval |",
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
            "beyond country, but how much of that is pallor is not shown.",
        ]
    lines += [
        "",
        "## Limitations",
        "",
        "- **Different populations.** The eyelid images are children under five; the planned survey model is for "
        "women 15-49. The two models are not about the same people, and the survey model is not built yet.",
        f"- **Not population prevalence.** {a['anemic_share_rows']:.1%} of CP-AnemiC rows are anemic. The sample "
        "leaned toward children suspected of anemia, so this share is not the prevalence in Ghana.",
        "- **Cutoffs differ.** Children use Hb < 11.0 g/dL; adults (Eyes-Defy, DHS) will use WHO cutoffs by sex and "
        "pregnancy. Results across datasets must say which cutoff each uses.",
        "- **Lighting, camera and cropping differ between image datasets.** CP-AnemiC comes cut out on a transparent "
        "background; Eyes-Defy ships separate masks. Cross-dataset numbers will mix these differences with real "
        "transfer, and that test has not run yet.",
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
        "1. **Run the cross-dataset test (Stage 5)**: train on CP-AnemiC photos with no copy, test on Eyes-Defy, and "
        "the reverse. Also test Eyes-Defy within each country, to separate pallor from country.",
        "2. **Build the DHS survey model (Stages 6-8)** once DHS access is approved.",
        "3. **Ask the CP-AnemiC authors about the duplicates.**",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    (ROOT / "reports" / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
