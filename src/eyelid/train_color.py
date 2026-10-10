"""Stage 3: color-feature baseline on CP-AnemiC.

Run from the repo root: python -m src.eyelid.train_color
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.evaluate import oof_scores, summarize
from src.common.io import RESULTS, ROOT, load_config, save_results
from src.common.splits import leave_one_group_out, repeated_stratified, repeated_stratified_grouped
from src.eyelid.color_features import feature_table, image_properties
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.report import markdown_table

REPORTS = ROOT / "reports"


def model_factories(cfg, seed):
    lr, rf = cfg["models"]["logistic_regression"], cfg["models"]["random_forest"]
    return {
        "dummy": lambda: DummyClassifier(strategy="prior"),
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], random_state=seed)
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=rf["n_estimators"], min_samples_leaf=rf["min_samples_leaf"],
            max_features=rf["max_features"], random_state=seed, n_jobs=-1,
        ),
    }


def main():
    base, cfg = load_config("default"), load_config("color")
    seed, boot = base["seed"], base["bootstrap"]
    n_splits, n_repeats = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    factories = model_factories(cfg, seed)

    everything = load_cp_anemic("all")
    features = {
        gw: feature_table(everything.image_path.map(lambda p: ROOT / p), everything.id, gw, cfg["roi"]["alpha_threshold"])
        for gw in cfg["gray_world"]
    }

    props = image_properties(everything.image_path.map(lambda p: ROOT / p), everything.id)

    def score_all(data, folds, gw, models):
        X = features[gw].loc[data.id].to_numpy()
        y = data.anemic.to_numpy()
        scores = {name: oof_scores(factories[name], X, y, folds) for name in models}
        scores["rule_low_redness"] = -features[gw].loc[data.id, cfg["rule_feature"]].to_numpy()
        scores["negative_control_rf_no_color"] = oof_scores(
            factories["random_forest"], props.loc[data.id].to_numpy(), y, folds
        )
        return y, scores

    rows, roc_inputs = [], {}
    unique, singletons = load_cp_anemic("unique"), load_cp_anemic("singletons")
    experiments = {
        "cv_5x5_no_copies": (singletons, repeated_stratified(singletons.anemic, n_splits, n_repeats, seed)),
        "leave_one_hospital_out": (singletons, leave_one_group_out(singletons.site)),
        "cv_5x5_distinct_photos": (unique, repeated_stratified(unique.anemic, n_splits, n_repeats, seed)),
    }
    for exp, (data, folds) in experiments.items():
        for gw in cfg["gray_world"]:
            y, scores = score_all(data, folds, gw, ["dummy", "logistic_regression", "random_forest"])
            for name, s in scores.items():
                rows.append({"experiment": exp, "n": len(y), "model": name, "gray_world": gw,
                             **summarize(y, s, boot["n_resamples"], boot["ci"], seed)})
            if exp == "cv_5x5_no_copies":
                roc_inputs[gw] = (y, {k: np.atleast_2d(v).mean(axis=0) for k, v in scores.items()})
            print(f"done: {exp}, gray_world={gw}")

    # Leakage check: all 710 rows, folds that let copies of an image split across train and test vs folds that don't.
    leak_rows = []
    y_all, groups = everything.anemic.to_numpy(), everything.photo_group.to_numpy()
    leak_splits = {
        "random rows (copies can leak)": repeated_stratified(y_all, n_splits, n_repeats, seed),
        "grouped by photo (no leak)": repeated_stratified_grouped(y_all, groups, n_splits, n_repeats, seed),
    }
    for split_name, folds in leak_splits.items():
        _, scores = score_all(everything, folds, False, ["logistic_regression", "random_forest"])
        for name in ["logistic_regression", "random_forest", "negative_control_rf_no_color"]:
            leak_rows.append({"split": split_name, "n": len(y_all), "model": name,
                              **summarize(y_all, scores[name], boot["n_resamples"], boot["ci"], seed, groups=groups)})
        print(f"done: leakage check, {split_name}")

    path = save_results("stage3_color", {"main": rows, "leakage_check": leak_rows},
                        {"default": base, "color": cfg})

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    styles = {False: "-", True: "--"}
    for gw, (y, scores) in roc_inputs.items():
        for name, color in [("logistic_regression", "C0"), ("random_forest", "C1"), ("rule_low_redness", "C2")]:
            fpr, tpr, _ = roc_curve(y, scores[name])
            ax.plot(fpr, tpr, styles[gw], color=color, label=f"{name}, gray-world {'on' if gw else 'off'}")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="dummy (chance)")
    ax.set(xlabel="False positive rate", ylabel="Sensitivity", title="CP-AnemiC color baseline, 5x5 CV, photos with no copy")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    for out in (RESULTS / "stage3_roc.png", REPORTS / "stage3_roc.png"):
        fig.savefig(out, dpi=150)

    main_table = markdown_table(rows, ["experiment", "n", "model", "gray_world"])
    leak_table = markdown_table(leak_rows, ["split", "n", "model"])
    report = (
        "# Stage 3: color-feature baseline (CP-AnemiC)\n\n"
        f"Generated by `python -m src.eyelid.train_color` at commit `{_short_commit(path)}`. "
        "Anemia = Hb < 11.0 g/dL. Estimates with bootstrap 95% CIs (2000 resamples).\n\n"
        f"- `cv_5x5_no_copies` (**headline**): repeated stratified 5-fold CV, 5 repeats, on the {len(singletons)} photos "
        "with no byte-identical copy and no detected near-duplicate. Scores averaged over repeats.\n"
        f"- `leave_one_hospital_out`: the same {len(singletons)} photos (the only ones whose hospital is trustworthy); "
        "pooled out-of-fold predictions across the held-out hospitals. The dummy scores below 0.5 here because "
        "each fold's prior comes from the other hospitals, so high-prevalence hospitals get lower scores.\n"
        f"- `cv_5x5_distinct_photos`: one row per distinct photo (n={len(unique)}), copies merged. The no-color control "
        "stays above chance here, which points to near-duplicates the audit rule missed, so treat it as optimistic.\n"
        "- `rule_low_redness`: no training, score = minus mean a* (paler = more likely anemic).\n"
        "- `negative_control_rf_no_color`: random forest on image size and cut-out shape only, no color. "
        "Near 0.5 means the split is not leaking duplicate photos.\n\n"
        f"{main_table}\n\n"
        "## Leakage check: all 710 rows\n\n"
        "Same features and models; the only difference is whether copies of the same photo may sit "
        "in both train and test. Bootstrap resamples whole photo groups.\n\n"
        f"{leak_table}\n\n![ROC](stage3_roc.png)\n"
    )
    (REPORTS / "stage3_color.md").write_text(report)
    print(f"\n{main_table}\n\nLeakage check:\n{leak_table}\n\nsaved {path.relative_to(ROOT)} and reports/stage3_color.md")


def _short_commit(path):
    import json
    git = json.loads(path.read_text())["git"]
    return git["commit"][:7] + (" + uncommitted changes" if git["dirty"] else "")


if __name__ == "__main__":
    main()
