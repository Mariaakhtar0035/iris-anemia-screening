"""Stage 12: is there a pallor color signal in fingernail photos taken under fixed lighting? (mentor advice M9, M12)

Yakimov et al. 2024: 250 adults in Moscow, one photo per patient in a fixed LED box, three nail boxes per photo. Same
color numbers, redness rule, models, no-color control and metrics as the eyelid stages. The hashed MEASUREMENT_DATE
marks 20 photo sessions, so after the Italy batch problem (Stage 10) we check whether the session alone predicts
anemia, and also score on whole held-out sessions. Stage 12b (approved by Jaineel 2026-10-24) reruns the rule and models
on white-balanced nails, on nail minus the skin next to it, and on that skin alone.

Run from the repo root: python -m src.nails
"""
import json
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.evaluate import IMAGE_METRICS, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci, precision_at_sensitivity, share_tested_at_sensitivity
from src.common.splits import leave_one_group_out, repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, train_text
from src.eyelid.color_features import color_features
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit

METRICS = IMAGE_METRICS | {"share_tested_90": partial(share_tested_at_sensitivity, sens=0.90),
                           "precision_90": partial(precision_at_sensitivity, sens=0.90)}
COLUMNS = {"auroc": "AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec",
           "share_tested_90": "Share tested to catch 90%", "precision_90": "Anemic among those tested"}


def load_nails(cfg):
    """One row per patient: id, photo path, Hb (g/L), session, nail boxes, anemic."""
    root = ROOT / cfg["data_dir"]
    d = pd.read_csv(root / "metadata.csv")
    return pd.DataFrame({
        "id": d.PATIENT_ID,
        "image_path": [root / "photo" / f"{i}.jpg" for i in d.PATIENT_ID],
        "hb_g_per_l": d.HB_LEVEL_GperL,
        "session": d.MEASUREMENT_DATE,
        "nail_boxes": d.NAIL_BOUNDING_BOXES.map(json.loads),
        "skin_boxes": d.SKIN_BOUNDING_BOXES.map(json.loads),
        "anemic": (d.HB_LEVEL_GperL < cfg["anemia_cutoff_g_per_l"]).astype(int),
    })


def read_photo(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def box_pixels(rgb, boxes):
    """RGB values of every pixel inside the boxes, shape (n_pixels, 3).
    Boxes are [row1, col1, row2, col2]: read as x/y they fall off the 800x600 photos (checked by drawing them)."""
    return np.concatenate([rgb[r1:r2, c1:c2].reshape(-1, 3) for r1, c1, r2, c2 in boxes])


def nail_pixels(path, boxes):
    return box_pixels(read_photo(path), boxes)


def white_balance(rgb):
    """Divide by the photo's own white board, so the board reads (1, 1, 1). The board = bright pixels with little color
    (HSV value > 0.5, saturation < 0.15), 54-74% of a photo; pale nails pass too but are too few to move the median."""
    hi, lo = rgb.max(-1), rgb.min(-1)
    white = np.median(rgb[(hi > 0.5) & (hi - lo < 0.15 * hi)], axis=0)
    return np.clip(rgb / white, 0.0, 1.0)


def photo_features(path, nail_boxes, skin_boxes):
    """Color numbers of the nails as photographed, of the nails after white balance, and of the skin next to them."""
    rgb = read_photo(path)
    return (color_features(box_pixels(rgb, nail_boxes)), color_features(box_pixels(white_balance(rgb), nail_boxes)),
            color_features(box_pixels(rgb, skin_boxes)))


def box_sizes(boxes):
    """No-color control: mean nail box height, width and area. Can't see pallor, but can see hand size."""
    b = np.asarray(boxes, dtype=float)
    h, w = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return [h.mean(), w.mean(), (h * w).mean()]


def main():
    base, cfg = load_config("default"), load_config("nails")
    color_cfg, ceiling_cfg = load_config("color"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_splits, n_repeats = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]

    df = load_nails(cfg)
    y, sessions = df.anemic.to_numpy(), df.session.to_numpy()
    nail, nail_wb, skin = (pd.DataFrame(list(f), index=df.id) for f in zip(*[
        photo_features(*row) for row in zip(df.image_path, df.nail_boxes, df.skin_boxes)]))
    color, rule = nail.to_numpy(), -nail[color_cfg["rule_feature"]].to_numpy()
    models = {name: make for name, (kind, make) in
              ladder_models(color_cfg, load_config("probe"), ceiling_cfg, seed).items() if kind == "color"}
    session_only = lambda: make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                         LogisticRegression(max_iter=5000, random_state=seed))

    def stats(scores, metrics=METRICS):
        return summarize(y, scores, boot["n_resamples"], boot["ci"], seed, metrics=metrics)

    info = {"n": len(y), "anemic": f"{y.mean():.1%}"}
    cv = repeated_stratified(y, n_splits, n_repeats, seed)
    splits = {"5x5 CV by patient": cv, "leave one session out": leave_one_group_out(sessions)}
    rows = []
    for split, folds in splits.items():
        rows.append({"split": split, **info, "model": "rule_low_redness (no training)", "train": "-", **stats(rule)})
        runs = {**{name: (color, make) for name, make in models.items()},
                "negative_control_rf_no_color": (np.array([box_sizes(b) for b in df.nail_boxes]), models["random_forest"])}
        if split == "5x5 CV by patient":  # a held-out session has no history, so this only runs within sessions
            runs["session_only"] = (sessions.reshape(-1, 1), session_only)
        for name, (X, make) in runs.items():
            held_out, train_auroc = fit_and_score(make, X, y, folds)
            rows.append({"split": split, **info, "model": name, "train": train_text(train_auroc), **stats(held_out)})
        print(f"done: {split}", flush=True)

    # Stage 12b: cancel lighting (white board) or skin tone (skin next to the nail) before the color numbers.
    def auroc_gain(scores, reference):
        """Paired bootstrap: AUROC of scores minus AUROC of reference, both on the same resampled photos."""
        packed = np.column_stack([np.atleast_2d(reference).mean(0), np.atleast_2d(scores).mean(0)])
        return bootstrap_ci(lambda yy, ss: auroc(yy, ss[:, 1]) - auroc(yy, ss[:, 0]), y, packed,
                            boot["n_resamples"], boot["ci"], seed=seed)

    inputs = {"nail (as photographed)": nail, "nail, white-balanced": nail_wb, "nail minus skin": nail - skin,
              "skin only": skin}
    corrections, reference = [], {}
    for split, folds in splits.items():
        for input_name, table in inputs.items():
            runs = {"rule_low_redness (no training)": -table[color_cfg["rule_feature"]].to_numpy()}
            for name, make in models.items():
                runs[name] = fit_and_score(make, table.to_numpy(), y, folds)[0]
            for name, scores in runs.items():
                row = {"split": split, "input": input_name, "model": name, **stats(scores, {"auroc": auroc})}
                if (split, name) in reference:
                    row["gain"] = auroc_gain(scores, reference[(split, name)])
                else:
                    reference[(split, name)] = scores  # the first input, as photographed
                corrections.append(row)
    print("done: corrections", flush=True)

    curve = []
    for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        for frac in ceiling_cfg["learning_curve"]["fractions"]:
            held_out, train_auroc = fit_and_score(models[name], color, y, cv, frac, seed)
            curve.append({"model": name, "fraction": frac,
                          "n_train": int(round(len(y) * (n_splits - 1) / n_splits * frac)),
                          "train AUROC": f"{np.nanmean(train_auroc):.3f}", **stats(held_out, {"auroc": auroc})})
    print("done: learning curve", flush=True)

    per_session = df.groupby("session").anemic.agg(["size", "mean"])
    session_info = {"sessions": len(per_session), "size_min": int(per_session["size"].min()),
                    "size_max": int(per_session["size"].max()), "anemic_share_min": float(per_session["mean"].min()),
                    "anemic_share_max": float(per_session["mean"].max()),
                    "sessions_with_no_anemic": int((per_session["mean"] == 0).sum())}

    path = save_results("stage12_nails", {"main": rows, "corrections": corrections, "learning_curve": curve,
                                          "sessions": session_info},
                        {"default": base, "nails": cfg, "color": color_cfg, "ceiling": ceiling_cfg})
    s = session_info
    report = (
        "# Stage 12: fingernail photos under fixed lighting\n\n"
        f"Generated by `python -m src.nails` at commit `{_short_commit(path)}`; settings in `configs/nails.yaml`. "
        "Mentor advice M9, M12 (`docs/eyelid_followup.md`). Data: Yakimov et al. 2024, *Scientific Data* (figshare "
        "10.6084/m9.figshare.c.6760179): adults in Moscow, one photo each, taken in a fixed LED box.\n\n"
        f"- **Anemic = Hb < 12.0 g/dL (120 g/L) for everyone**, {y.sum()} of {len(y)}. The per-patient file has no sex "
        "column, so one adult cutoff is used (the WHO cutoff for women, chosen by Jaineel): men between 12.0 and 13.0 "
        "count as not anemic.\n"
        "- **Input:** the pixels inside the three nail boxes of each photo, pooled; the same 29 color numbers as the "
        "eyelid stages.\n"
        "- `rule_low_redness`: no training, score = minus mean a* (paler nail = more likely anemic).\n"
        "- `negative_control_rf_no_color`: random forest on the nail box sizes only. Above 0.5 means either a leak or "
        "that hand size (and so sex, which shifts Hb) carries the label.\n"
        "- `session_only`: logistic regression on the photo session alone (hashed `MEASUREMENT_DATE`, "
        f"{s['sessions']} sessions of {s['size_min']}-{s['size_max']} people, anemic share "
        f"{s['anemic_share_min']:.0%}-{s['anemic_share_max']:.0%}, {s['sessions_with_no_anemic']} with no anemic "
        "person). Well above 0.5 means sessions differ in who was photographed, a shortcut a color model could use.\n"
        f"- **Splits:** repeated stratified 5-fold CV by patient ({n_repeats} repeats, scores averaged), and leave one "
        "session out (pooled out-of-fold scores; the dummy can fall below 0.5 there because each fold's prior comes "
        "from the other sessions). **train** = AUROC of each fold's model on its own training photos, mean (min-max).\n"
        f"- Bootstrap 95% CIs, {boot['n_resamples']} resamples. Share tested to catch 90%: testing at random needs "
        "90%; lower is better.\n\n"
        + markdown_table(rows, ["split", "n", "anemic", "model", "train"], COLUMNS)
        + "\n\n## Stage 12b: cancelling lighting or skin tone\n\nThe same rule and models on other color numbers:\n\n"
        "- **nail, white-balanced:** each photo divided by its own white board first (the median of its bright, "
        "colorless pixels), so the board reads pure white. Cancels lighting drift between sessions.\n"
        "- **nail minus skin:** the nail's color numbers minus those of the skin boxes just behind each nail in the "
        "same photo. Cancels lighting and skin tone, but also any pallor the skin shares with the nail bed.\n"
        "- **skin only:** the skin boxes alone, to show how much of the signal the skin carries.\n\n"
        "**Change** = held-out AUROC minus the same model's on the nails as photographed, with a paired bootstrap "
        "95% CI (both scored on the same resampled photos). A CI that crosses 0 = no clear change.\n\n"
        + markdown_table(corrections, ["split", "input", "model"],
                         {"auroc": "Held-out AUROC", "gain": "Change vs as photographed"})
        + "\n\n## Learning curve (5x5 CV by patient)\n\nHeld-out AUROC when each training fold is cut to a stratified "
        "share. Still rising at 1.0 = more photos would help.\n\n"
        + markdown_table(curve, ["model", "fraction", "n_train", "train AUROC"], {"auroc": "Held-out AUROC"}) + "\n"
    )
    (ROOT / "reports" / "stage12_nails.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
