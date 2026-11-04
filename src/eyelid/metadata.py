"""Stage 20: does adding age and sex to the eyelid photo help inside each location?

At the 2026-11-03 meeting the mentors asked where the metadata goes: DINOv2 turns the photo into a representation, but
age and sex never reached any model. They also said to stop aiming for transfer between these small, very different
datasets and to work per location. So inside Ghana, India and Italy separately, the same models see the photo alone,
age and sex alone, or both; photo + age + sex is compared with the photo alone and with age + sex alone for the same
model (paired bootstrap). The Hb number (Stage 16) is estimated from the same three inputs.

Run from the repo root: python -m src.eyelid.metadata
"""
import numpy as np
from sklearn.dummy import DummyRegressor

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import bootstrap_ci
from src.common.splits import leave_one_group_out, repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames, population_inputs
from src.eyelid.hb_regression import mae, oof_regression, regressors
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

PHOTO, META, BOTH = "photo", "age + sex", "photo + age + sex"
NAMES = {"cp_anemic": "Ghana", "eyes_defy_india": "India", "eyes_defy_italy": "Italy"}
SEX = {"Male": 1, "Female": 0, "M": 1, "F": 0}


def age_sex(df):
    """Age (CP-AnemiC in months, Eyes-Defy in years) and sex (1 = male) as two columns."""
    age = df["age_months"] if "age_months" in df else df["age"]
    male = df.sex.map(SEX)
    if age.isna().any() or male.isna().any():
        raise ValueError("missing age, or a sex value other than Male/Female/M/F")
    return np.column_stack([age.to_numpy(float), male.to_numpy(float)])


def mae_change(hb, pred, reference, n_boot, ci, seed):
    """MAE of pred minus MAE of reference on the same resampled people; below 0 means pred is closer to the lab Hb."""
    packed = np.column_stack([reference, pred])
    return bootstrap_ci(lambda h, p: mae(h, p[:, 1]) - mae(h, p[:, 0]), hb, packed, n_boot, ci, seed=seed)


def main():
    base, cfg, color_cfg, probe_cfg = (load_config(n) for n in ("default", "metadata", "color", "probe"))
    ceiling_cfg, debias_cfg, hb_cfg = load_config("ceiling"), load_config("debias"), load_config("hb_regression")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    models = {n: make for n, (kind, make) in ladder.items() if kind == "color" and n != "dummy"}
    regs = {"dummy: everyone gets the mean Hb": DummyRegressor} | {
        n: make for n, make in regressors(hb_cfg, seed).items() if n in cfg["hb_regressors"]}
    data = population_inputs(population_frames(ceiling_cfg), color_cfg, probe_cfg, ceiling_cfg)

    rows, hb_rows = [], []
    for pop in cfg["populations"]:
        df, inp = data[pop]
        y, hb, meta = df.anemic.to_numpy(), df.hb.to_numpy(float), age_sex(df)
        X = {PHOTO: inp["color"], META: meta, BOTH: np.hstack([inp["color"], meta])}
        E = {PHOTO: inp["embedding"], BOTH: np.hstack([inp["embedding"], meta])}
        splits = {"5x5 CV by photo": repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)}
        if pop == "cp_anemic":
            splits["leave one hospital out"] = leave_one_group_out(df.site.to_numpy())
        test_sets = {NAMES[pop]: np.ones(len(y), bool)}
        if pop == "eyes_defy_italy":
            folder = df.id.str.split("_").str[1].astype(int).to_numpy()
            test_sets["Italy early batch"] = folder <= debias_cfg["italy_early_batch_last_folder"]

        for split, folds in splits.items():
            s = {(name, how): fit_and_score(make, Xi, y, folds)[0] for name, make in models.items()
                 for how, Xi in X.items()}
            s |= {("dinov2_probe", how): fit_and_score(ladder["dinov2_probe"][1], Ei, y, folds)[0]
                  for how, Ei in E.items()}
            print(f"done: {pop}, {split}", flush=True)
            for subset, m in test_sets.items():
                yy = y[m]
                head = {"location": NAMES[pop], "test set": subset, "split": split, "n": int(m.sum()),
                        "anemic": f"{yy.mean():.1%}"}
                rows.append(head | {"model": "dummy (prior)", "input": "-",
                                    **summarize(yy, np.full(len(yy), yy.mean()), n_boot, ci, seed, metrics=METRICS)})
                rows.append(head | {"model": "rule_low_redness", "input": "-",
                                    **summarize(yy, inp["rule"][m], n_boot, ci, seed, metrics=METRICS)})
                for (name, how), sc in s.items():
                    row = head | {"model": name, "input": how, **summarize(yy, sc[:, m], n_boot, ci, seed, metrics=METRICS)}
                    if how == BOTH:
                        row["vs_photo"] = auroc_change(yy, sc[:, m], s[(name, PHOTO)][:, m], n_boot, ci, seed)
                        if (name, META) in s:
                            row["vs_meta"] = auroc_change(yy, sc[:, m], s[(name, META)][:, m], n_boot, ci, seed)
                    rows.append(row)

        folds = splits["5x5 CV by photo"]
        preds = {(name, how): oof_regression(make, Xi, hb, folds).mean(0) for name, make in regs.items()
                 for how, Xi in X.items() if not (name.startswith("dummy") and how != PHOTO)}
        for subset, m in test_sets.items():
            for (name, how), p in preds.items():
                row = {"location": NAMES[pop], "test set": subset, "n": int(m.sum()), "model": name,
                       "input": "-" if name.startswith("dummy") else how,
                       "mae": bootstrap_ci(mae, hb[m], p[m], n_boot, ci, seed=seed)}
                if how == BOTH:
                    row["vs_photo"] = mae_change(hb[m], p[m], preds[(name, PHOTO)][m], n_boot, ci, seed)
                hb_rows.append(row)

    path = save_results("stage20_metadata", {"rows": rows, "hb_rows": hb_rows},
                        {"default": base, "metadata": cfg, "color": color_cfg, "probe": probe_cfg,
                         "ceiling": ceiling_cfg, "debias": debias_cfg, "hb_regression": hb_cfg})

    names = list(models) + ["dinov2_probe"]
    groups = list(dict.fromkeys((r["test set"], r["split"]) for r in rows))
    answer, gains = [], []
    for subset, split in groups:
        g = [r for r in rows if r["test set"] == subset and r["split"] == split]
        best = {how: max((r for r in g if r["input"] == how), key=lambda r: r["auroc"]["estimate"]) for how in (PHOTO, META, BOTH)}
        rule = next(r for r in g if r["model"] == "rule_low_redness")
        helps = [r["model"] for r in g if "vs_photo" in r and r["vs_photo"]["lo"] > 0
                 and r.get("vs_meta", {"lo": 1})["lo"] > 0]
        answer.append(f"| {subset} | {split} | " + " | ".join(f"{fmt(best[h]['auroc'])} ({best[h]['model']})"
                                                           for h in (PHOTO, META, BOTH))
                      + f" | {fmt(rule['auroc'])} | " + (", ".join(helps) or "none") + " |")
        gains.append(f"| {subset} | {split} | " + " | ".join(
            fmt(next(r["vs_photo"] for r in g if r["model"] == n and r["input"] == BOTH)) for n in names) + " |")

    hb_table = []
    for subset in dict.fromkeys(r["test set"] for r in hb_rows):
        h = {(r["model"], r["input"]): r for r in hb_rows if r["test set"] == subset}
        dummy = next(r for (n, _), r in h.items() if n.startswith("dummy"))
        for name in cfg["hb_regressors"]:
            hb_table.append(f"| {subset} | {name} | {fmt(dummy['mae'])} | " + " | ".join(
                fmt(h[(name, how)]["mae"]) for how in (PHOTO, META, BOTH)) + f" | {fmt(h[(name, BOTH)]['vs_photo'])} |")

    cut = load_config("cross_dataset")["eyes_defy_cutoff_g_dl"]
    report = (
        "# Stage 20: does adding age and sex help inside each location?\n\n"
        f"Generated by `python -m src.eyelid.metadata` at commit `{_short_commit(path)}`; settings in "
        "`configs/metadata.yaml`. Mentor question (2026-11-03): DINOv2 turns the photo into a representation, so where "
        "does the metadata go? Until now nowhere: every model saw the eyelid only. Per location only (mentor advice "
        "2026-11-03: don't aim for transfer between these small, very different datasets).\n\n"
        "- **Photos:** Ghana, the 388 CP-AnemiC photos with no copy (children 6-59 months, **anemic = Hb < "
        f"{base['anemia_cutoff_g_dl']['child_6_59_months']} g/dL** for both sexes); India (95) and Italy (122) from "
        f"Eyes-Defy (adults, **WHO by sex: women < {cut['F']}, men < {cut['M']} g/dL**; pregnancy not recorded). One "
        "photo per person. The cutoffs differ by dataset, but no model is carried between datasets.\n"
        "- **Inputs:** `photo` = the 29 color numbers of the palpebral conjunctiva (or the DINOv2 embedding for the "
        "probe); `age + sex` = age (Ghana in months, Eyes-Defy in years) and sex (1 = male) alone; `photo + age + sex` "
        "= both side by side. The hospital is never an input.\n"
        "- **Sex and the cutoff:** in India and Italy sex also sets the anemia cutoff, so part of any gain from sex there "
        "is the definition, not biology. Ghana has one cutoff for both sexes.\n"
        f"- **Split:** {cfg['cv']['n_splits']}-fold CV x {cfg['cv']['n_repeats']} repeats by photo, stratified by "
        "anemia; Ghana also leaving one hospital out (10 hospitals, pooled held-out scores). Bootstrap 95% CIs, "
        f"{n_boot} resamples. The AUROC change is a paired bootstrap (same resampled photos), same model.\n"
        "- **Italy early batch:** folders 1-"
        f"{debias_cfg['italy_early_batch_last_folder']}, where 21 of the 23 anemic patients are (Stage 10); scored "
        "from the same predictions.\n"
        "- **Answer, fixed before running:** age and sex help in a location if, for the same model, photo + age + sex "
        "beats the photo alone (CI of the change above 0) and also beats age + sex alone.\n\n"
        "## Best AUROC for each input\n\n"
        f"| test set | split | best, {PHOTO} | best, {META} only | best, {BOTH} | rule | age and sex help (both CIs "
        "above 0) |\n|---|---|---|---|---|---|---|\n" + "\n".join(answer) + "\n\n"
        f"## What adding age and sex changes (AUROC of `{BOTH}` minus `{PHOTO}`, same model)\n\n"
        "| test set | split | " + " | ".join(names) + " |\n|" + "---|" * (2 + len(names)) + "\n" + "\n".join(gains)
        + "\n\n"
        "## The Hb number (MAE against the lab Hb, g/dL; lower is better)\n\n"
        f"Stage 16's regressors, 5x5 CV by photo. The last column is the paired change in MAE from adding age and sex "
        "(below 0 = closer to the lab value).\n\n"
        f"| test set | regressor | dummy (mean Hb) | {PHOTO} | {META} only | {BOTH} | change from adding age + sex |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(hb_table) + "\n\n"
        "## All results\n\n"
        + markdown_table(rows, ["test set", "split", "n", "anemic", "model", "input"],
                         COLUMNS | {"vs_photo": f"AUROC change vs {PHOTO}", "vs_meta": f"AUROC change vs {META}"})
        + "\n"
    )
    (ROOT / "reports" / "stage20_metadata.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
