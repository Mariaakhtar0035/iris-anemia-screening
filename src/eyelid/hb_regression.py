"""Stage 16: predict the Hb value from eyelid color, then apply the anemia cutoff (mentor advice, 2026-10-31).

Each dataset uses its own anemia cutoff (children < 11.0, adults < 12.0 or 13.0 g/dL), so a yes/no model can't be
carried from one dataset to another. A model that estimates Hb can: the local cutoff is applied afterwards. This stage
starts with Ghana (CP-AnemiC, the 388 photos with no copy), the one population whose eyelid signal passed the shortcut
checks, and asks two things: how far off is the Hb estimate, and does ranking by estimated Hb find anemic people as
well as models trained to say yes/no directly? The same is then run on the Moscow fingernails (Stage 12), the other
dataset with lab Hb and the one photographed under fixed lighting. scikit-learn only.

Run from the repo root: python -m src.eyelid.hb_regression
"""
import json
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.evaluate import IMAGE_METRICS, oof_scores
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import bootstrap_ci, share_tested_at_sensitivity
from src.common.splits import leave_one_group_out, repeated_stratified
from src.eyelid.color_features import color_features, feature_table, image_properties
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit

CLASS_METRICS = IMAGE_METRICS | {"share_tested_90": partial(share_tested_at_sensitivity, sens=0.90)}
COLUMNS = {"mae": "Hb error (MAE, g/dL)", "within_1": "Within 1 g/dL", "auroc": "AUROC", "pr_auc": "PR-AUC",
           "sens_at_90_spec": "Sens @ 90% spec", "share_tested_90": "Share tested to catch 90%",
           "sens_at_cutoff": "Sens at the cutoff", "spec_at_cutoff": "Spec at the cutoff"}


def mae(hb, pred):
    return float(np.mean(np.abs(np.asarray(hb) - np.asarray(pred))))


def within_1(hb, pred):
    """Share of children whose Hb estimate is within 1 g/dL of the lab value."""
    return float(np.mean(np.abs(np.asarray(hb) - np.asarray(pred)) <= 1.0))


def sens_at_cutoff(anemic, pred_hb, cutoff):
    """Share of anemic children whose *estimated* Hb is below the cutoff: the cutoff applied to the estimate."""
    anemic, pred_hb = np.asarray(anemic), np.asarray(pred_hb)
    if anemic.sum() == 0:
        raise ValueError("needs at least one anemic child")
    return float(np.mean(pred_hb[anemic == 1] < cutoff))


def spec_at_cutoff(anemic, pred_hb, cutoff):
    anemic, pred_hb = np.asarray(anemic), np.asarray(pred_hb)
    if anemic.sum() == len(anemic):
        raise ValueError("needs at least one child who is not anemic")
    return float(np.mean(pred_hb[anemic == 0] >= cutoff))


def oof_regression(make_model, X, y, folds):
    """Out-of-fold predictions for every row, shape (n_repeats, n_rows); like oof_scores, for regressors."""
    preds = np.full((max(r for r, _, _ in folds) + 1, len(y)), np.nan)
    for r, train, test in folds:
        preds[r, test] = make_model().fit(X[train], y[train]).predict(X[test])
    assert not np.isnan(preds).any(), "some rows were never in a test fold"
    return preds


def regressors(cfg, seed):
    r = cfg["regressors"]
    return {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(**r["ridge"])),
        "random_forest": lambda: RandomForestRegressor(**r["random_forest"], random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(**r["hist_gradient_boosting"], random_state=seed),
    }


def classifiers(cfg, seed):
    c = cfg["classifiers"]
    return {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(**c["logistic_regression"],
                                                                                          random_state=seed)),
        "random_forest": lambda: RandomForestClassifier(**c["random_forest"], random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(**c["hist_gradient_boosting"],
                                                                         random_state=seed),
    }


def load_ghana(cfg, color_cfg, base, seed):
    """CP-AnemiC eyelids: the photos with no copy, children, Hb < 11.0 g/dL."""
    cutoff = base["anemia_cutoff_g_dl"]["child_6_59_months"]
    cp = load_cp_anemic(cfg["cp_anemic_subset"])
    hb = cp.hb.to_numpy(dtype=float)
    assert ((hb < cutoff) == cp.anemic.astype(bool).to_numpy()).all(), "audit label disagrees with Hb < cutoff"
    paths = [ROOT / p for p in cp.image_path]
    feats = feature_table(paths, cp.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    y = (hb < cutoff).astype(int)
    return {"name": "Ghana eyelids (CP-AnemiC)", "hb": hb, "cutoff": cutoff, "color": feats.to_numpy(),
            "redness": feats[[color_cfg["rule_feature"]]].to_numpy(), "no_color": image_properties(paths, cp.id).to_numpy(),
            "splits": {"5x5 CV by photo": repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed),
                       "leave one hospital out": leave_one_group_out(cp.site.to_numpy())},
            "who": f"{len(y)} children 6-59 months, photos with no copy; anemic = Hb < {cutoff} g/dL"}


def box_pixels(rgb, boxes):
    """Pixels inside the nail boxes. Boxes are [row1, col1, row2, col2] (same reading as src/nails.py, which can't be
    imported here because it loads TabPFN)."""
    return np.concatenate([rgb[r1:r2, c1:c2].reshape(-1, 3) for r1, c1, r2, c2 in boxes])


def load_moscow(cfg, color_cfg, base, seed):
    """Moscow fingernails: one photo per adult, Hb in g/L converted to g/dL, one cutoff for everyone (nails.yaml)."""
    nails = load_config("nails")
    root = ROOT / nails["data_dir"]
    d = pd.read_csv(root / "metadata.csv")
    hb = d.HB_LEVEL_GperL.to_numpy(dtype=float) / 10.0
    cutoff = nails["anemia_cutoff_g_per_l"] / 10.0
    boxes = d.NAIL_BOUNDING_BOXES.map(json.loads)
    feats = pd.DataFrame([color_features(box_pixels(np.asarray(Image.open(root / "photo" / f"{i}.jpg").convert("RGB"),
                                                               dtype=np.float32) / 255.0, b))
                          for i, b in zip(d.PATIENT_ID, boxes)])
    sizes = np.array([[(b[:, 2] - b[:, 0]).mean(), (b[:, 3] - b[:, 1]).mean(),
                       ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).mean()] for b in map(np.asarray, boxes)], dtype=float)
    y = (hb < cutoff).astype(int)
    return {"name": "Moscow fingernails (Yakimov et al. 2024)", "hb": hb, "cutoff": cutoff, "color": feats.to_numpy(),
            "redness": feats[[color_cfg["rule_feature"]]].to_numpy(), "no_color": sizes,
            "splits": {"5x5 CV by patient": repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed),
                       "leave one photo session out": leave_one_group_out(d.MEASUREMENT_DATE.to_numpy())},
            "who": f"{len(y)} adults, one photo each, fixed light box; anemic = Hb < {cutoff} g/dL for everyone "
                   "(no sex column)"}


LOADERS = {"cp_anemic": load_ghana, "moscow_nails": load_moscow}


def evaluate(ds, cfg, boot, seed):
    """Regressors (estimate Hb, then rank by it and apply the cutoff) next to yes/no classifiers, on every split."""
    hb, cutoff, color, redness = ds["hb"], ds["cutoff"], ds["color"], ds["redness"]
    y = (hb < cutoff).astype(int)

    def ci(metric, truth, score):
        return bootstrap_ci(metric, truth, score, boot["n_resamples"], boot["ci"], None, seed)

    def regression_stats(pred):
        """Hb error against the lab value, then the cutoff applied to the estimate. Lower Hb = more likely anemic."""
        return {"mae": ci(mae, hb, pred), "within_1": ci(within_1, hb, pred),
                **{k: ci(f, y, -pred) for k, f in CLASS_METRICS.items()},
                "sens_at_cutoff": ci(partial(sens_at_cutoff, cutoff=cutoff), y, pred),
                "spec_at_cutoff": ci(partial(spec_at_cutoff, cutoff=cutoff), y, pred)}

    def class_stats(score):
        return {k: ci(f, y, score) for k, f in CLASS_METRICS.items()}

    rows = []
    for split, folds in ds["splits"].items():
        info = {"split": split}
        reg = regressors(cfg, seed)
        reg_runs = {"dummy: everyone gets the mean Hb": (color, DummyRegressor),
                    "redness only (ridge on mean a*)": (redness, reg["ridge"]),
                    **{name: (color, make) for name, make in reg.items()},
                    "control: random forest, no color": (ds["no_color"], reg["random_forest"])}
        for name, (X, make) in reg_runs.items():
            pred = oof_regression(make, X, hb, folds).mean(0)
            rows.append({**info, "target": "Hb value", "model": name, **regression_stats(pred)})
            print(f"{ds['name']} | {split} | Hb | {name}: MAE {rows[-1]['mae']['estimate']:.2f}, "
                  f"AUROC {rows[-1]['auroc']['estimate']:.3f}", flush=True)
        rows.append({**info, "target": "anemic yes/no", "model": "rule: less red = anemic (no training)",
                     **class_stats(-redness[:, 0])})
        for name, make in {"dummy": lambda: DummyClassifier(strategy="prior"), **classifiers(cfg, seed)}.items():
            score = oof_scores(make, color, y, folds).mean(0)
            rows.append({**info, "target": "anemic yes/no", "model": name, **class_stats(score)})
            print(f"{ds['name']} | {split} | yes/no | {name}: AUROC {rows[-1]['auroc']['estimate']:.3f}", flush=True)
    return rows


def main():
    base, cfg, color_cfg = load_config("default"), load_config("hb_regression"), load_config("color")
    seed, boot = base["seed"], base["bootstrap"]
    results, sections = {}, []
    for key in cfg["datasets"]:
        ds = LOADERS[key](cfg, color_cfg, base, seed)
        hb, y = ds["hb"], (ds["hb"] < ds["cutoff"]).astype(int)
        print(f"{ds['name']}: {len(hb)} photos, Hb {hb.min():.1f}-{hb.max():.1f} g/dL (mean {hb.mean():.2f}), "
              f"{y.mean():.1%} anemic", flush=True)
        rows = evaluate(ds, cfg, boot, seed)
        results[key] = {"n": len(hb), "hb_mean": float(hb.mean()), "cutoff_g_dl": ds["cutoff"], "rows": rows}
        sections.append(f"## {ds['name']}\n\n{ds['who']}. Lab Hb {hb.min():.1f}-{hb.max():.1f} g/dL, mean "
                        f"{hb.mean():.2f}; {y.mean():.1%} anemic.\n\n"
                        + markdown_table(rows, ["split", "target", "model"], COLUMNS) + "\n")

    path = save_results("stage16_hb_regression", results, {"default": base, "hb_regression": cfg, "color": color_cfg,
                                                            "nails": load_config("nails")})
    text = (
        "# Stage 16: estimate the Hb value, then apply the cutoff\n\n"
        f"Generated by `python -m src.eyelid.hb_regression` at commit `{_short_commit(path)}`; settings in "
        "`configs/hb_regression.yaml`. Mentor advice from the 2026-10-31 meeting: if you have the Hb, measure both the "
        "classification and how well Hb is estimated. Each dataset keeps its own anemia cutoff (stated per section).\n\n"
        "- **Input:** the 29 color numbers of the region (eyelid cut-out, or the three nail boxes as photographed). "
        "`redness only` uses the mean a* alone, which is what the rule ranks by. The no-color control sees only image "
        "size and shape (eyelids) or nail box sizes (nails).\n"
        "- **`Hb value` rows** are regressors. Their estimate is scored against the lab Hb (MAE, share within 1 g/dL), "
        "then used as a risk score (lower estimate = higher risk) for AUROC, PR-AUC, sensitivity at 90% specificity and "
        "the share tested to catch 90%. `Sens/Spec at the cutoff` apply the dataset's cutoff to the estimate itself.\n"
        "- **`anemic yes/no` rows** are the same model families trained on the label directly, for comparison.\n"
        "- One photo per person. 5x5 CV stratified on the label; the second split holds out a whole hospital (Ghana) or "
        "photo session (Moscow) and pools the held-out predictions. Bootstrap 95% CIs, 2000 resamples. For reference, "
        "the CP-AnemiC paper's ResNet50 reported an Hb error of 1.54 g/dL on a split with copies on both sides; our "
        "grouped replication gave 1.61 (`reports/replicate_cnn.md`).\n\n"
        + "\n".join(sections)
    )
    (ROOT / "reports" / "stage16_hb_regression.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
