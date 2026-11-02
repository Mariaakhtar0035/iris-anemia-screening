"""Stage 19: fingernail models across sites, Ghana children <-> Moscow adults.

Every eyelid model so far failed on a new dataset, and only the untrained redness rule carried over. The Moscow nails
were photographed in one fixed light box (Stage 12) and the Ghana nails were not (Stage 13). Here each model is fit on
all of one nail dataset and applied unchanged to the other, as photographed and with each dataset's color numbers
standardized on its own photos (no labels), next to the rule and to the same models scored within each dataset.
Each dataset keeps its own anemia definition (see configs/nails_cross.yaml). scikit-learn only.

Run from the repo root: python -m src.nails_cross
"""
import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.dummy import DummyClassifier

from src.common.evaluate import oof_scores, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import repeated_stratified
from src.eyelid.audit import photo_groups
from src.eyelid.color_features import feature_table
from src.eyelid.hb_cross import standardize
from src.eyelid.hb_regression import CLASS_METRICS, classifiers, load_moscow
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit

COLUMNS = {"auroc": "AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec",
           "share_tested_90": "Share tested to catch 90%"}

# The next three pieces repeat src/ghana_sites.py (Stage 13), which can't be imported here because it loads TabPFN.
LABEL = re.compile(r"^(non-?)?(?:anemic|anrmic|anmeic)(.*)$", re.IGNORECASE)  # the typos are in the file names


def parse_name(stem):
    """File stem -> (photo name, anemic). 'Non-AnemicP-010 (3)' -> ('Non-AnemicP-010', 0)."""
    name = re.sub(r" \(\d+\)$", "", stem)
    m = LABEL.match(name)
    assert m, f"unexpected file name: {stem}"
    return name, int(m.group(1) is None)


def rotation_key(path):
    """The same key for a photo and its exact copies rotated by 90, 180 or 270 degrees."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    return min(hashlib.sha256(r.tobytes() + str(r.shape).encode()).hexdigest()
               for r in (np.ascontiguousarray(np.rot90(a, k)) for k in range(4)))


def load_ghana_nails(folder, near):
    """One row per distinct photo: exact copies and rotations merged, then near-copies; conflicting labels dropped."""
    files = sorted((ROOT / folder).glob("*.png"))
    df = pd.DataFrame([(f, *parse_name(f.stem)) for f in files], columns=["path", "name", "anemic"])
    df["copy"] = [f.stem != n for f, n in zip(df.path, df.name)]
    df = df.sort_values(["name", "copy"], kind="stable").reset_index(drop=True)
    df["image_group"] = df.path.map(rotation_key)
    df["photo_group"], _ = photo_groups(df, near["dhash_max_bits"], near["max_size_diff"])
    one_label = df.groupby("photo_group").anemic.transform("nunique") == 1
    return df[one_label].drop_duplicates("photo_group").reset_index(drop=True)


def main():
    base, cfg, color_cfg = load_config("default"), load_config("nails_cross"), load_config("color")
    reg_cfg = load_config("hb_regression")
    seed, boot = base["seed"], base["bootstrap"]

    ghana = load_ghana_nails(cfg["ghana_nails_dir"], base["cp_anemic_near_duplicate"])
    gfeats = feature_table(ghana.path, ghana.index, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    moscow = load_moscow(reg_cfg, color_cfg, base, seed)
    data = {
        "ghana_nails": {"X": gfeats.to_numpy(), "y": ghana.anemic.to_numpy(),
                        "rule": -gfeats[color_cfg["rule_feature"]].to_numpy()},
        "moscow_nails": {"X": moscow["color"], "y": (moscow["hb"] < moscow["cutoff"]).astype(int),
                         "rule": -moscow["redness"][:, 0]},
    }
    for name, d in data.items():
        print(f"{name}: {len(d['y'])} photos, {d['y'].mean():.1%} anemic", flush=True)

    def stats(y, score):
        return summarize(y, score, boot["n_resamples"], boot["ci"], seed, metrics=CLASS_METRICS)

    models = {"dummy": lambda: DummyClassifier(strategy="prior"), **classifiers(reg_cfg, seed)}
    rows = []
    # Reference: each dataset on its own, 5x5 CV by photo.
    for name, d in data.items():
        folds = repeated_stratified(d["y"], cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        info = {"train -> test": f"{name} (within, 5x5 CV)", "features": "as photographed"}
        rows.append({**info, "model": "rule: less red = anemic (no training)", **stats(d["y"], d["rule"])})
        for m, make in models.items():
            rows.append({**info, "model": m, **stats(d["y"], oof_scores(make, d["X"], d["y"], folds))})
    # Across sites: fit on all of one dataset, score all of the other unchanged.
    for src, dst in cfg["directions"]:
        s, t = data[src], data[dst]
        for version in cfg["feature_versions"]:
            Xs, Xt = (standardize(s["X"]), standardize(t["X"])) if version == "standardized per dataset" else (s["X"], t["X"])
            info = {"train -> test": f"{src} -> {dst}", "features": version}
            if version == "as photographed":  # the rule needs no training, so it is the same in both versions
                rows.append({**info, "model": "rule: less red = anemic (no training)", **stats(t["y"], t["rule"])})
            for m, make in models.items():
                rows.append({**info, "model": m, **stats(t["y"], make().fit(Xs, s["y"]).predict_proba(Xt)[:, 1])})
            print(f"{src} -> {dst} | {version}: " + ", ".join(
                f"{r['model'].split(':')[0]} {r['auroc']['estimate']:.3f}" for r in rows
                if r["train -> test"] == f"{src} -> {dst}" and r["features"] == version), flush=True)

    path = save_results("stage19_nails_cross", {"n_ghana": len(ghana), "n_moscow": len(data["moscow_nails"]["y"]),
                                                "rows": rows},
                        {"default": base, "nails_cross": cfg, "hb_regression": reg_cfg, "color": color_cfg,
                         "nails": load_config("nails")})
    text = (
        "# Stage 19: fingernail models across sites (Ghana children <-> Moscow adults)\n\n"
        f"Generated by `python -m src.nails_cross` at commit `{_short_commit(path)}`; settings in "
        "`configs/nails_cross.yaml`. Mentor advice (2026-10-31): try both cross-country and per-region.\n\n"
        f"- **Ghana fingernails** (Asare et al. 2022): {len(ghana)} distinct photos of children under 6 after merging "
        "copies and rotations (as in Stage 13). **Anemic = the dataset's own label from the file name**; there is no "
        "Hb, so the cutoff behind it is not stated.\n"
        f"- **Moscow fingernails** (Yakimov et al. 2024): {len(data['moscow_nails']['y'])} adults in one fixed light box. "
        "**Anemic = Hb < 12.0 g/dL for everyone** (no sex column). Color numbers of the three nail boxes as photographed.\n"
        "- **The two definitions of anemia differ** (children vs adults, a file-name label vs a lab cutoff), so a model "
        "trained on one is judged against the other's definition.\n"
        "- **Within** rows: each dataset alone, 5x5 CV by photo. **Across** rows: fit on all of one dataset, applied "
        "unchanged to the other. `standardized per dataset` scales each dataset's 29 color numbers to mean 0, sd 1 on "
        "its own photos (no labels). The rule is minus mean a*, no training.\n"
        "- Bootstrap 95% CIs, 2000 resamples, one photo per person.\n\n"
        + markdown_table(rows, ["train -> test", "features", "model"], COLUMNS) + "\n"
    )
    (ROOT / "reports" / "stage19_nails_cross.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
