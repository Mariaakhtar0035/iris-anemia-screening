"""Stage 13: palm and fingernail photos of Ghanaian children, each body site on its own (mentor advice M9, M12).

Asare et al. 2022 (Mendeley, approved by Jaineel 2026-10-24). Every photo is saved 6 or 12 times (exact copies and
rotations) and labelled only by its file name: no Hb, age, sex or hospital, and no link to the CP-AnemiC eyelid photos.
So we merge copies and rotations, then near-copies with the CP-AnemiC audit's rule, keep one row per photo, and run the
same rule, models, no-color control and metrics as the nails stage. The names follow a few styles (`AnemicP-`,
`Anemic-Pa-`, `Anemic-FN-`, ...), probably collection batches, so a model that sees only the naming style checks for a
batch shortcut like Italy's (Stage 10).

Run from the repo root: python -m src.ghana_sites
"""
import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc
from src.common.splits import repeated_stratified
from src.eyelid.audit import photo_groups
from src.eyelid.ceiling import fit_and_score, ladder_models, train_text
from src.eyelid.color_features import feature_table, image_properties
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

LABEL = re.compile(r"^(non-?)?(?:anemic|anrmic|anmeic)(.*)$", re.IGNORECASE)  # the typos are in the file names


def parse_name(stem):
    """File stem -> (photo name, anemic, naming style). 'Non-AnemicP-010 (3)' -> ('Non-AnemicP-010', 0, 'p')."""
    name = re.sub(r" \(\d+\)$", "", stem)
    m = LABEL.match(name)
    assert m, f"unexpected file name: {stem}"
    return name, int(m.group(1) is None), re.match(r"-?([a-z]*)", m.group(2).lower()).group(1) or "none"


def rotation_key(path):
    """The same key for a photo and its exact copies rotated by 90, 180 or 270 degrees."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    return min(hashlib.sha256(r.tobytes() + str(r.shape).encode()).hexdigest()
               for r in (np.ascontiguousarray(np.rot90(a, k)) for k in range(4)))


def load_site(folder, near):
    """One row per distinct photo (path, name, anemic, naming style), and the audit counts."""
    files = sorted((ROOT / folder).glob("*.png"))
    df = pd.DataFrame([(f, *parse_name(f.stem)) for f in files], columns=["path", "name", "anemic", "style"])
    # un-numbered file first, so each photo keeps its stored orientation for the near-copy check
    df["copy"] = [f.stem != n for f, n in zip(df.path, df.name)]
    df = df.sort_values(["name", "copy"], kind="stable").reset_index(drop=True)
    df["image_group"] = df.path.map(rotation_key)
    # shortcut: near-copies are matched in stored orientation only; a rotated re-crop would be missed. Upgrade to a
    # rotation-aware dHash if the no-color control ever sits well above chance.
    df["photo_group"], links = photo_groups(df, near["dhash_max_bits"], near["max_size_diff"])
    n_labels = df.groupby("photo_group").anemic.transform("nunique")
    photos = df[n_labels == 1].drop_duplicates("photo_group").reset_index(drop=True)
    audit = {"files": len(df), "exact_photos": int(df.image_group.nunique()), "photos": int(df.photo_group.nunique()),
             "near_copy_links": len(links),
             "photos_under_several_names": int((df.groupby("photo_group").name.nunique() > 1).sum()),
             "dropped_conflicting_labels": int(df[n_labels > 1].photo_group.nunique()), "kept": len(photos)}
    return photos, audit


def main():
    base, cfg = load_config("default"), load_config("ghana_sites")
    color_cfg, ceiling_cfg = load_config("color"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_splits, n_repeats = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    models = {name: make for name, (kind, make) in
              ladder_models(color_cfg, load_config("probe"), ceiling_cfg, seed).items() if kind == "color"}
    style_only = lambda: make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                                       LogisticRegression(max_iter=5000, random_state=seed))

    rows, curve, audits, styles = [], [], {}, {}
    for site, folder in cfg["sites"].items():
        photos, audits[site] = load_site(folder, base["cp_anemic_near_duplicate"])
        y, style = photos.anemic.to_numpy(), photos["style"].to_numpy()
        main_style = photos["style"].mode()[0]
        styles[site] = photos.groupby("style").anemic.agg(n="size", anemic="sum").to_dict("index")
        feats = feature_table(photos.path, photos.index, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
        color = feats.to_numpy()
        folds = repeated_stratified(y, n_splits, n_repeats, seed)

        def stats(scores, metrics=METRICS):
            out = summarize(y, scores, boot["n_resamples"], boot["ci"], seed, metrics=metrics)
            inside = style == main_style  # can't lean on the naming style here
            out["auroc_main_style"] = summarize(y[inside], np.atleast_2d(scores)[:, inside], boot["n_resamples"],
                                                boot["ci"], seed, metrics={"auroc": auroc})["auroc"]
            return out

        info = {"site": site, "n": len(y), "anemic": f"{y.mean():.1%}"}
        rows.append({**info, "model": "rule_low_redness (no training)", "train": "-",
                     **stats(-feats[color_cfg["rule_feature"]].to_numpy())})
        runs = {**{name: (color, make) for name, make in models.items()},
                "negative_control_rf_no_color": (image_properties(photos.path, photos.index).to_numpy(),
                                                 models["random_forest"]),
                "naming_style_only": (style.reshape(-1, 1), style_only)}
        for name, (X, make) in runs.items():
            held_out, train_auroc = fit_and_score(make, X, y, folds)
            rows.append({**info, "model": name, "train": train_text(train_auroc), **stats(held_out)})
            print(f"done: {site} {name}", flush=True)

        for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
            for frac in ceiling_cfg["learning_curve"]["fractions"]:
                held_out, train_auroc = fit_and_score(models[name], color, y, folds, frac, seed)
                curve.append({"site": site, "model": name, "fraction": frac,
                              "n_train": int(round(len(y) * (n_splits - 1) / n_splits * frac)),
                              "train AUROC": f"{np.nanmean(train_auroc):.3f}",
                              **summarize(y, held_out, boot["n_resamples"], boot["ci"], seed, metrics={"auroc": auroc})})

    path = save_results("stage13_ghana_sites", {"main": rows, "learning_curve": curve, "audit": audits,
                                                "naming_styles": styles},
                        {"default": base, "ghana_sites": cfg, "color": color_cfg, "ceiling": ceiling_cfg})
    audit_rows = [{"site": s, **a} for s, a in audits.items()]
    style_rows = [{"site": s, "style": k, "photos": v["n"], "anemic": f"{v['anemic'] / v['n']:.0%}"}
                  for s, t in styles.items() for k, v in t.items()]
    plain = lambda rs: "\n".join(["| " + " | ".join(rs[0]) + " |", "|" + "---|" * len(rs[0])]
                                 + ["| " + " | ".join(map(str, r.values())) + " |" for r in rs])
    report = (
        "# Stage 13: palm and fingernail photos of Ghanaian children\n\n"
        f"Generated by `python -m src.ghana_sites` at commit `{_short_commit(path)}`; settings in "
        "`configs/ghana_sites.yaml`. Mentor advice M9, M12 (`docs/eyelid_followup.md`). Data: Asare, Appiahene, Donkoh "
        "2022, Mendeley Data 10.17632/ccr8cm22vz.1 (palm) and 10.17632/2xx4j3kjg2.1 (fingernails), CC BY 4.0.\n\n"
        "- **Anemic = the dataset's own label**, read from the file name. The archives have no Hb, age, sex or "
        "hospital, so the cutoff behind the label isn't stated (the group's eyelid set uses Hb < 11.0 g/dL for "
        "children). Nothing links a palm or nail photo to an eyelid photo, so the body sites can't be combined per "
        "child.\n"
        "- **One row per photo.** Each photo is stored 6 or 12 times (exact copies plus 90/180/270-degree rotations); "
        "those are merged by a rotation-aware hash, then near-copies by the CP-AnemiC audit's rule (16x16 difference "
        "hash within 20 bits, size within 5%). Photos filed under both labels are dropped. Rotations don't change "
        "color numbers, so the copies add nothing as augmentation.\n\n"
        + plain(audit_rows) + "\n\n"
        "- **Input:** the 29 color numbers of the eyelid stages. Palm: every pixel of the patch (no transparency). "
        "Nails: the opaque pixels of the cut-out.\n"
        "- `rule_low_redness`: no training, score = minus mean a* (paler = more likely anemic).\n"
        "- `negative_control_rf_no_color`: random forest on image size and cut-out shape only.\n"
        "- `naming_style_only`: logistic regression on the file-naming style alone (below). Well above 0.5 means "
        "the styles differ in who was photographed, a batch shortcut.\n"
        "- **AUROC inside the main style** re-scores the same held-out scores on the largest naming style only, where "
        "the style can't help.\n"
        f"- **Split:** repeated stratified 5-fold CV by photo ({n_repeats} repeats, scores averaged). **train** = AUROC "
        "of each fold's model on its own training photos, mean (min-max). Bootstrap 95% CIs, "
        f"{boot['n_resamples']} resamples. Share tested to catch 90%: testing at random needs 90%; lower is better.\n\n"
        "## Naming styles\n\n" + plain(style_rows) + "\n\n## Results\n\n"
        + markdown_table(rows, ["site", "n", "anemic", "model", "train"],
                         COLUMNS | {"auroc_main_style": "AUROC inside the main style"})
        + "\n\n## Learning curve\n\nHeld-out AUROC when each training fold is cut to a stratified share. Still rising "
        "at 1.0 = more photos would help.\n\n"
        + markdown_table(curve, ["site", "model", "fraction", "n_train", "train AUROC"], {"auroc": "Held-out AUROC"})
        + "\n"
    )
    (ROOT / "reports" / "stage13_ghana_sites.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
