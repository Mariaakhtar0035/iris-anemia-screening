"""Stage 23: a vision model that crops the eyelid and flags photos to retake, per location.

The mentors (2026-10-31) suggested vision models in the data pipeline: one that crops the region being analyzed and one
that separates good from bad photos. Every result so far used the dataset's hand-drawn cut-outs. Eyes-Defy also ships
the full photo, so here a frozen DINOv2 learns from the hand masks which 14 px patches are palpebral conjunctiva, crops
each held-out photo itself, and scores how sure it is. Then: how close is its crop to the hand mask, does anemia
scoring hold on its crop, and does its confidence work as a retake flag?

Run from the repo root: python -m src.eyelid.auto_crop
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # DINOv2 is already on disk; never download here

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from transformers import AutoModel

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames
from src.eyelid.color_features import color_features
from src.eyelid.cross_dataset import ED_LABELS
from src.eyelid.embed import pick_device
from src.eyelid.metadata import NAMES
from src.eyelid.quality import kept_minus_all
from src.eyelid.report import fmt, markdown_table
from src.eyelid.roi import read_rgba
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])   # DINOv2's ImageNet normalization
HAND, AUTO = "hand-drawn mask", "automatic crop"


def aligned(photo_path, mask_path, long_side, alpha_threshold):
    """The full photo (RGB in [0, 1]) and the hand mask, upright, same size, long side `long_side`."""
    mask = read_rgba(mask_path)
    photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB").resize(mask.size, Image.BILINEAR)
    w, h = mask.size
    size = (long_side, round(long_side * h / w)) if w >= h else (round(long_side * w / h), long_side)
    rgb = np.asarray(photo.resize(size, Image.BILINEAR), np.float32) / 255.0
    alpha = np.asarray(mask.resize(size, Image.NEAREST))[..., 3]
    return rgb, alpha >= alpha_threshold


def grid(shape, long_side, patch):
    """Patch rows and columns for a photo of `shape` whose long side becomes `long_side` (a multiple of `patch`)."""
    h, w = shape
    n = long_side // patch
    return (round(n * h / w), n) if w >= h else (n, round(n * w / h))


def patch_tokens(model, rgb, rows, cols, patch, device):
    x = cv2.resize(rgb, (cols * patch, rows * patch), interpolation=cv2.INTER_AREA)
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
    with torch.no_grad():
        out = model(pixel_values=torch.tensor(x, dtype=torch.float32, device=device)).last_hidden_state
    return out[0, 1:].cpu().numpy()   # one token per patch, row by row (the first token is the summary token)


def patch_share(mask, rows, cols):
    """Share of each patch covered by the mask, row by row."""
    return cv2.resize(mask.astype(np.float32), (cols, rows), interpolation=cv2.INTER_AREA).reshape(-1)


def to_mask(prob, shape, threshold):
    """Patch probabilities (rows x cols) -> pixel mask of `shape`: cut at `threshold`, keep the largest piece.
    Returns (mask, found, confidence); if nothing passes, the most likely patch is kept and found is False."""
    up = cv2.resize(prob.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    found = bool((up >= threshold).any())
    keep = up >= (threshold if found else up.max())
    n, labels, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8))
    mask = labels == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax()) if n > 1 else keep
    return mask, found, float(up[mask].mean())


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))


def crop_location(df, photos, cfg, alpha_threshold, model, device, seed):
    """Out-of-fold automatic crops for one location: color numbers from the hand mask and the crop, Dice, confidence."""
    imgs = [aligned(ROOT / photos[i], ROOT / m, cfg["work_long_side"], alpha_threshold)
            for i, m in zip(df.id, df.image_path)]
    shapes = [grid(rgb.shape[:2], cfg["encoder_long_side"], cfg["patch_size"]) for rgb, _ in imgs]
    tokens = [patch_tokens(model, rgb, r, c, cfg["patch_size"], device) for (rgb, _), (r, c) in zip(imgs, shapes)]
    target = [patch_share(hand, r, c) >= cfg["patch_target_share"] for (_, hand), (r, c) in zip(imgs, shapes)]
    out = [None] * len(imgs)
    for train, test in KFold(cfg["segmenter_folds"], shuffle=True, random_state=seed).split(imgs):
        seg = LogisticRegression(**cfg["segmenter"], class_weight="balanced", random_state=seed).fit(
            np.concatenate([tokens[k] for k in train]), np.concatenate([target[k] for k in train]))
        for k in test:
            rgb, hand = imgs[k]
            prob = seg.predict_proba(tokens[k])[:, 1].reshape(shapes[k])
            mask, found, conf = to_mask(prob, hand.shape, cfg["threshold"])
            out[k] = {"hand": color_features(rgb[hand]), "auto": color_features(rgb[mask]), "dice": dice(mask, hand),
                      "found": found, "confidence": conf}
    return out


def main():
    base, cfg, color_cfg, probe_cfg = (load_config(n) for n in ("default", "auto_crop", "color", "probe"))
    ceiling_cfg, debias_cfg = load_config("ceiling"), load_config("debias")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    rule_col = color_cfg["rule_feature"]
    models = {n: make for n, (kind, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if kind == "color" and n != "dummy"}
    device = pick_device()
    encoder = AutoModel.from_pretrained(probe_cfg["model_id"], revision=probe_cfg["revision"]).to(device).eval()
    photos = pd.read_csv(ED_LABELS).set_index("id").image_path
    frames = population_frames(ceiling_cfg)

    def interval(metric, a, b):
        return bootstrap_ci(metric, a, b, n_boot, ci, seed=seed)

    crop_rows, rows, gate_rows = [], [], []
    for key in cfg["locations"]:
        df = frames[key][~frames[key].id.isin(cfg["exclude"])].reset_index(drop=True)
        y = df.anemic.to_numpy()
        out = crop_location(df, photos, cfg, color_cfg["roi"]["alpha_threshold"], encoder, device, seed)
        d, conf = np.array([o["dice"] for o in out]), np.array([o["confidence"] for o in out])
        crop_rows.append({"location": NAMES[key], "n": len(df), "not found": int(sum(not o["found"] for o in out)),
                          "dice_median": interval(lambda a, _: np.median(a), d, d),
                          "dice_good": interval(lambda a, _: np.mean(a >= cfg["dice_good"]), d, d),
                          "spearman": interval(lambda a, b: spearmanr(a, b).statistic, conf, d)})
        X = {HAND: pd.DataFrame([o["hand"] for o in out]), AUTO: pd.DataFrame([o["auto"] for o in out])}
        rule = {how: -x[rule_col].to_numpy() for how, x in X.items()}
        folds = repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        s = {(n, how): fit_and_score(make, x.to_numpy(), y, folds)[0] for n, make in models.items() for how, x in X.items()}
        print(f"done: {key}", flush=True)

        test_sets = {NAMES[key]: np.ones(len(y), bool)}
        if key == "eyes_defy_italy":
            folder = df.id.str.split("_").str[1].astype(int).to_numpy()
            test_sets["Italy early batch"] = folder <= debias_cfg["italy_early_batch_last_folder"]
        for subset, m in test_sets.items():
            yy = y[m]
            head = {"test set": subset, "n": int(m.sum()), "anemic": f"{yy.mean():.1%}"}
            rows.append(head | {"model": "confidence alone (shortcut check)", "crop": AUTO,
                                **summarize(yy, conf[m], n_boot, ci, seed, metrics=METRICS)})
            scores = {("rule_low_redness", how): rule[how][m] for how in X} | {k: v[:, m] for k, v in s.items()}
            for (name, how), sc in scores.items():
                row = head | {"model": name, "crop": how, **summarize(yy, sc, n_boot, ci, seed, metrics=METRICS)}
                if how == AUTO:
                    row["vs_hand"] = auroc_change(yy, sc, scores[(name, HAND)], n_boot, ci, seed)
                rows.append(row)
            for share in cfg["reject_share"]:
                keep = conf[m] >= np.quantile(conf[m], share)
                for name in ["rule_low_redness", *models]:
                    sc = scores[(name, AUTO)]
                    sc = sc.mean(0) if sc.ndim == 2 else sc
                    gate_rows.append(head | {"reject": f"{share:.0%}", "rejected": int((~keep).sum()),
                                             "anemic among rejected": f"{yy[~keep].mean():.1%}",
                                             "median Dice, rejected": f"{np.median(d[m][~keep]):.2f}",
                                             "median Dice, kept": f"{np.median(d[m][keep]):.2f}", "model": name,
                                             "auroc_all": interval(auroc, yy, sc),
                                             "auroc_kept": interval(auroc, yy[keep], sc[keep]),
                                             "change": kept_minus_all(yy, sc, keep, n_boot, ci, seed)})

    path = save_results("stage23_auto_crop", {"crop": crop_rows, "rows": rows, "gate": gate_rows},
                        {"default": base, "auto_crop": cfg, "color": color_cfg, "probe": probe_cfg,
                         "ceiling": ceiling_cfg, "debias": debias_cfg})

    worse = [f"{r['test set']}: {r['model']} {fmt(r['vs_hand'])}" for r in rows if "vs_hand" in r and r["vs_hand"]["hi"] < 0]
    better = [f"{r['test set']}: {r['model']} {fmt(r['vs_hand'])}" for r in rows if "vs_hand" in r and r["vs_hand"]["lo"] > 0]
    gate_help = [f"{r['test set']}, reject {r['reject']}: {r['model']} {fmt(r['change'])}" for r in gate_rows
                 if r["change"]["lo"] > 0]
    shortcut = [f"{r['test set']} {fmt(r['auroc'])}" for r in rows if r["model"].startswith("confidence")
                and (r["auroc"]["lo"] > 0.5 or r["auroc"]["hi"] < 0.5)]
    crop_table = "\n".join(f"| {r['location']} | {r['n']} | {fmt(r['dice_median'])} | {fmt(r['dice_good'])} | "
                           f"{r['not found']} | {fmt(r['spearman'])} |" for r in crop_rows)
    main_rows = [r for r in rows if r["model"] in ("rule_low_redness", "tabpfn") or r["model"].startswith("confidence")]
    main_gate = [r for r in gate_rows if r["model"] in ("rule_low_redness", "tabpfn")]
    cut = load_config("cross_dataset")["eyes_defy_cutoff_g_dl"]
    gate_cols = ["test set", "reject", "rejected", "anemic", "anemic among rejected", "median Dice, rejected",
                 "median Dice, kept", "model"]
    report = (
        "# Stage 23: a vision model that crops the eyelid and flags photos to retake\n\n"
        f"Generated by `python -m src.eyelid.auto_crop` at commit `{_short_commit(path)}`; settings in "
        "`configs/auto_crop.yaml`. Mentor advice (2026-10-31): put vision models in the data pipeline, one that crops "
        "the region being analyzed and one that separates good from bad photos. Per location only.\n\n"
        "- **Photos:** Eyes-Defy, the only set with the full photo next to a hand-drawn palpebral mask: India 94, Italy "
        f"121 (Italy_15 and India_2 left out: broken mask). Adults, **WHO by sex: women < {cut['F']}, men < {cut['M']} "
        "g/dL** (pregnancy not recorded).\n"
        f"- **Vision model:** frozen DINOv2-small; photo resized to a long side of {cfg['encoder_long_side']} px, one "
        f"token per {cfg['patch_size']} px patch. A logistic regression learns from the hand masks which patches are "
        f"eyelid, {cfg['segmenter_folds']}-fold CV by photo inside each location, so no photo is cropped by a model that "
        f"saw its mask. Probabilities cut at {cfg['threshold']}, largest piece kept. No anemia label is used. "
        "**Confidence** = mean probability inside its crop.\n"
        "- **Anemia:** the same 29 color numbers and models as before, from the hand mask or the automatic crop on the "
        "same photo pixels; 5x5 CV by photo; bootstrap 95% CIs, "
        f"{n_boot} resamples; changes are paired.\n"
        "- **Retake gate:** the least confident 10% or 20% of photos are dropped at test time (no labels, no mask).\n\n"
        "## Answers (fixed before running)\n\n"
        "- **(a) How close is the crop?** See the table: Dice 1 = the same pixels as the hand mask.\n"
        "- **(b) Does the automatic crop cost accuracy (AUROC change CI below 0)?** "
        + ("; ".join(worse) if worse else "No model or rule, in any test set.") + "\n"
        + ("  Better than the hand mask (CI above 0): " + "; ".join(better) + "\n" if better else "")
        + "- **(c) Is confidence a usable quality check (Spearman with Dice, CI above 0)?** See the table.\n"
        "- **(d) Does the retake gate help (kept minus all, CI above 0)?** "
        + ("; ".join(gate_help) if gate_help else "No.") + "\n"
        "- **Does confidence alone separate anemic photos (CI excludes 0.5)?** "
        + ("; ".join(shortcut) if shortcut else "No.") + "\n\n"
        "## (a, c) Crop quality\n\n"
        f"| location | n | median Dice | share with Dice >= {cfg['dice_good']} | not found | Spearman, confidence vs "
        "Dice |\n|---|---|---|---|---|---|\n" + crop_table + "\n\n"
        "## (b) Anemia from the automatic crop vs the hand mask (rule and TabPFN; every model below)\n\n"
        + markdown_table(main_rows, ["test set", "n", "anemic", "model", "crop"],
                         {"auroc": "AUROC", "vs_hand": "AUROC change vs hand mask"}) + "\n\n"
        "## (d) Retake gate on the vision model's confidence (rule and TabPFN; every model in the JSON)\n\n"
        + markdown_table(main_gate, gate_cols, {"auroc_all": "AUROC, all", "auroc_kept": "AUROC, kept",
                                                "change": "change"}) + "\n\n"
        "## All anemia results\n\n"
        + markdown_table(rows, ["test set", "n", "anemic", "model", "crop"], COLUMNS | {"vs_hand": "AUROC change vs hand mask"})
        + "\n"
    )
    (ROOT / "reports" / "stage23_auto_crop.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
