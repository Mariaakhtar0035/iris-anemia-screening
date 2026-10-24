"""Stage 9: how far can each population be fit on its own? (mentor advice M1-M3, docs/eyelid_followup.md)

For CP-AnemiC, Eyes-Defy India, Italy and the two pooled, run a ladder of models from the no-training redness rule
up to frozen DINOv2 and VisionFM (eye-pretrained) probes. For each one, report training-set AUROC next to held-out AUROC, plus a learning curve.
The fine-tuned ViT rung comes from src.eyelid.finetune_vit and is read from its results files.
Also covered: the three Eyes-Defy mask regions with gray-world on and off, and CP-AnemiC duplicates used as
training-only augmentation.

Run from the repo root: python -m src.eyelid.ceiling
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.common.evaluate import auroc_change, summarize
from src.common.io import RESULTS, ROOT, load_config, save_results
from src.common.metrics import auroc
from src.common.splits import repeated_stratified
from src.eyelid.color_features import feature_table, image_properties
from src.eyelid.cross_dataset import eyes_defy_embeddings, inputs, load_eyes_defy
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.embed import load_embeddings
from src.eyelid.report import markdown_table
from src.eyelid.train_color import model_factories
from src.eyelid.train_probe import probe_factory
from src.eyelid import visionfm

FIGURES = ROOT / "reports" / "figures"
PROBE_SET = "cls_and_patch_mean"
AUROC_ONLY = {"auroc": auroc}


def population_frames(cfg):
    """name -> one row per photo with id, image_path (the conjunctiva cut-out or palpebral mask) and anemic."""
    cp = load_cp_anemic(cfg["cp_anemic_subset"])
    ed = load_eyes_defy(load_config("cross_dataset"))
    return {"cp_anemic": cp,
            "eyes_defy_india": ed[ed.country == "India"].reset_index(drop=True),
            "eyes_defy_italy": ed[ed.country == "Italy"].reset_index(drop=True),
            "eyes_defy_pooled": ed}


def has_visionfm(cfg):
    return (ROOT / cfg["visionfm"]["weights"]).exists()


def population_inputs(frames, color_cfg, probe_cfg, cfg):
    """name -> (frame, {"color", "rule", "embedding", "visionfm"?}), reusing the Stage 4 and Stage 5 embedding caches.
    VisionFM features only when its weights are present; cut-outs prepared as for DINOv2 (cropped for Eyes-Defy)."""
    emb = load_embeddings(probe_cfg)
    pos = {i: k for k, i in enumerate(emb["ids"])}
    ed = frames["eyes_defy_pooled"]
    ed_emb = pd.DataFrame(eyes_defy_embeddings(ed, probe_cfg), index=ed.id)
    out = {}
    for name, df in frames.items():
        e = emb[PROBE_SET][[pos[i] for i in df.id]] if name == "cp_anemic" else ed_emb.loc[df.id].to_numpy()
        out[name] = (df, inputs(df, e, color_cfg))
        if has_visionfm(cfg):
            out[name][1]["visionfm"] = visionfm.embeddings(df, cfg["visionfm"], name, crop=name != "cp_anemic")
    return out


def ladder_models(color_cfg, probe_cfg, cfg, seed):
    """name -> (input kind, model factory), smallest to biggest. The rule and the fine-tuned ViT are added separately."""
    color = model_factories(color_cfg, seed)
    hgb = cfg["hist_gradient_boosting"]
    out = {"dummy": ("color", color["dummy"]),
           "logistic_regression": ("color", color["logistic_regression"]),
           "random_forest": ("color", color["random_forest"]),
           "hist_gradient_boosting": ("color", lambda: HistGradientBoostingClassifier(**hgb, random_state=seed)),
           "dinov2_probe": ("embedding", probe_factory(probe_cfg, seed))}
    if has_visionfm(cfg):
        out["visionfm_probe"] = ("visionfm", probe_factory(probe_cfg, seed))
    return out


def subsample(train, y, frac, seed):
    """A stratified share `frac` of the training rows (all of them when frac is 1)."""
    if frac >= 1:
        return train
    keep, _ = train_test_split(train, train_size=frac, stratify=y[train], random_state=seed)
    return keep


def fit_and_score(make, X, y, folds, frac=1.0, seed=0):
    """Held-out P(anemic) of shape (n_repeats, n), and the AUROC each fold's model gets on its own training rows."""
    held_out = np.full((max(r for r, _, _ in folds) + 1, len(y)), np.nan)
    train_auroc = []
    for i, (r, train, test) in enumerate(folds):
        train = subsample(train, y, frac, seed + i)
        model = make().fit(X[train], y[train])
        held_out[r, test] = model.predict_proba(X[test])[:, 1]
        train_auroc.append(auroc(y[train], model.predict_proba(X[train])[:, 1]) if len(set(y[train])) == 2 else np.nan)
    assert not np.isnan(held_out).any()
    return held_out, train_auroc


def training_rows(copy_groups, train_groups, every_copy):
    """Rows of the full CP-AnemiC table that a training fold may use: every copy of each training photo, or only the
    first copy (the row the `unique` subset keeps). Copies of test photos are never included."""
    rows = np.flatnonzero(np.isin(copy_groups, train_groups))
    return rows if every_copy else rows[~pd.Series(copy_groups[rows]).duplicated().to_numpy()]


def train_text(t):
    t = np.asarray(t, dtype=float)
    return f"{np.nanmean(t):.3f} ({np.nanmin(t):.2f}-{np.nanmax(t):.2f})"


def vit_rows(pops, boot, seed):
    """Fine-tuned ViT rows from results/stage9_vit_*.json (written by src.eyelid.finetune_vit), when present."""
    rows = []
    for path in sorted(RESULTS.glob("stage9_vit_*.json")):
        r = json.loads(path.read_text())["results"]
        df = pops[r["population"]][0]
        assert list(df.id) == r["ids"], f"{path.name}: photos differ from the current population"
        y, s = df.anemic.to_numpy(), np.array(r["held_out"])
        m = summarize(y, s, boot["n_resamples"], boot["ci"], seed)
        rows.append({"population": r["population"], "n": len(y), "anemic": f"{y.mean():.1%}", "model": f"{r.get('model', 'vit_b16')}_finetuned ({'jitter' if r['jitter'] else 'no jitter'}, 1 seed)",
                     "train": train_text(r["train_auroc"]), "gap": f"{np.mean(r['train_auroc']) - m['auroc']['estimate']:+.3f}",
                     "minutes": round(r["seconds"] / 60), **m})
    return rows


def main():
    base, cfg = load_config("default"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_splits, n_repeats = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    pops = population_inputs(population_frames(cfg), color_cfg, probe_cfg, cfg)
    models = ladder_models(color_cfg, probe_cfg, cfg, seed)

    # 1. The ladder and learning curves, each population on its own.
    ladder, curve = [], []
    for name, (df, x) in pops.items():
        y = df.anemic.to_numpy()
        folds = repeated_stratified(y, n_splits, n_repeats, seed)
        info = {"population": name, "n": len(y), "anemic": f"{y.mean():.1%}"}
        ladder.append({**info, "model": "rule_low_redness (no training)", "train": "-", "gap": "-",
                       **summarize(y, x["rule"], boot["n_resamples"], boot["ci"], seed)})
        # Negative control: image size and cut-out shape only, no color. Near 0.5 unless photos leak or batches differ.
        props = image_properties([ROOT / p for p in df.image_path], df.id).to_numpy()
        held_out, train_auroc = fit_and_score(models["random_forest"][1], props, y, folds)
        m = summarize(y, held_out, boot["n_resamples"], boot["ci"], seed)
        ladder.append({**info, "model": "negative_control_rf_no_color", "train": train_text(train_auroc),
                       "gap": f"{np.nanmean(train_auroc) - m['auroc']['estimate']:+.3f}", **m})
        for model, (kind, make) in models.items():
            for frac in cfg["learning_curve"]["fractions"]:
                n_train = int(round(len(y) * (n_splits - 1) / n_splits * frac))
                if min(np.bincount(y)) * (n_splits - 1) / n_splits * frac < cfg["learning_curve"]["min_class_train"]:
                    continue
                held_out, train_auroc = fit_and_score(make, x[kind], y, folds, frac, seed)
                if frac == 1:
                    m = summarize(y, held_out, boot["n_resamples"], boot["ci"], seed)
                    ladder.append({**info, "model": model, "train": train_text(train_auroc),
                                   "gap": f"{np.nanmean(train_auroc) - m['auroc']['estimate']:+.3f}", **m})
                else:
                    m = summarize(y, held_out, boot["n_resamples"], boot["ci"], seed, metrics=AUROC_ONLY)
                if model != "dummy":
                    curve.append({"population": name, "model": model, "fraction": frac, "n_train": n_train,
                                  "train_auroc": float(np.nanmean(train_auroc)), **m})
            print(f"{name}: {model} done", flush=True)
    ladder += vit_rows(pops, boot, seed)
    ladder.sort(key=lambda r: list(pops).index(r["population"]))  # ViT rows next to their population

    # 2. Eyes-Defy: which mask region, and gray-world on or off. Only photos that have all three masks.
    region_rows = []
    ed = pops["eyes_defy_pooled"][0]
    paths = {}
    for region, pattern in cfg["regions"].items():
        found = [[m for m in (ROOT / p).parent.glob(pattern) if region != "palpebral" or "forniceal" not in m.name]
                 for p in ed.image_path]
        assert all(len(f) <= 1 for f in found), f"{region}: more than one mask in a folder"
        paths[region] = [f[0] if f else None for f in found]
    complete = np.array([all(paths[r][i] for r in paths) for i in range(len(ed))])
    color = model_factories(color_cfg, seed)
    for country in ("India", "Italy"):
        keep = complete & (ed.country == country).to_numpy()
        y = ed.anemic.to_numpy()[keep]
        folds = repeated_stratified(y, n_splits, n_repeats, seed)
        palpebral = {}  # (gray_world, model) -> scores, the reference for the paired change of the other regions
        for region in cfg["regions"]:
            for gw in cfg["gray_world"]:
                feats = feature_table([p for p, k in zip(paths[region], keep) if k], ed.id[keep], gw,
                                      color_cfg["roi"]["alpha_threshold"])
                info = {"country": country, "n": len(y), "anemic": f"{y.mean():.1%}", "region": region, "gray_world": gw}
                runs = {"rule_low_redness": -feats[color_cfg["rule_feature"]].to_numpy()}
                for model in ("logistic_regression", "random_forest"):
                    runs[model] = fit_and_score(color[model], feats.to_numpy(), y, folds)[0]
                for model, scores in runs.items():
                    row = {**info, "model": model,
                           **summarize(y, scores, boot["n_resamples"], boot["ci"], seed, metrics=AUROC_ONLY)}
                    if region == "palpebral":
                        palpebral[(gw, model)] = scores
                    else:
                        row["change"] = auroc_change(y, scores, palpebral[(gw, model)], boot["n_resamples"], boot["ci"], seed)
                    region_rows.append(row)
        print(f"regions: {country} done", flush=True)

    # 3. CP-AnemiC duplicates as training augmentation: same held-out photos, training with or without the copies.
    copy_rows = []
    everything = load_cp_anemic("all")
    everything = everything[everything.groupby("photo_group").anemic.transform("nunique") == 1].reset_index(drop=True)
    first = np.flatnonzero(~everything.photo_group.duplicated().to_numpy())
    assert list(everything.id[first]) == list(load_cp_anemic("unique").id), "first copies should be the `unique` subset"
    # Folds run over all distinct photos (so photos with copies can be trained on), but scores are only judged on
    # photos with no copy: the `unique` subset still holds near-duplicates the audit missed (see configs/ceiling.yaml).
    single = (everything.groupby("photo_group").id.transform("size") == 1).to_numpy()[first]
    emb = load_embeddings(probe_cfg)
    pos = {i: k for k, i in enumerate(emb["ids"])}
    x_all = {"color": feature_table([ROOT / p for p in everything.image_path], everything.id, False,
                                    color_cfg["roi"]["alpha_threshold"]).to_numpy(),
             "embedding": emb[PROBE_SET][[pos[i] for i in everything.id]]}
    y_all, groups = everything.anemic.to_numpy(), everything.photo_group.to_numpy()
    y = y_all[first]
    folds = repeated_stratified(y, n_splits, n_repeats, seed)
    for model in ("logistic_regression", "random_forest", "dinov2_probe"):
        kind, make = models[model]
        for every_copy in (False, True):
            held_out, n_rows = np.full((n_repeats, len(y)), np.nan), []
            for r, train, test in folds:
                rows = training_rows(groups, groups[first[train]], every_copy)
                assert not np.isin(groups[rows], groups[first[test]]).any()
                n_rows.append(len(rows))
                held_out[r, test] = make().fit(x_all[kind][rows], y_all[rows]).predict_proba(x_all[kind][first[test]])[:, 1]
            copy_rows.append({"model": model, "training rows": "every copy" if every_copy else "one per photo",
                              "mean training rows": round(float(np.mean(n_rows))), "n": int(single.sum()),
                              **summarize(y[single], held_out[:, single], boot["n_resamples"], boot["ci"], seed)})
        print(f"copies: {model} done", flush=True)

    # Save before writing the tracked figure and report, so the results record a clean commit on the first run.
    save_results("stage9_ceiling", {"ladder": ladder, "learning_curve": curve, "regions": region_rows,
                                    "copies_as_augmentation": copy_rows},
                 {"default": base, "ceiling": cfg, "color": color_cfg, "probe": probe_cfg})

    # Learning-curve figure: held-out AUROC (solid) and training AUROC (dotted) against training photos.
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame([{**c, "held_out": c["auroc"]["estimate"]} for c in curve])
    fig, axes = plt.subplots(1, len(pops), figsize=(4 * len(pops), 3.8), sharey=True)
    for ax, name in zip(axes, pops):
        for i, (model, g) in enumerate(cv[cv.population == name].groupby("model", sort=False)):
            ax.plot(g.n_train, g.held_out, "o-", color=f"C{i}", ms=3, label=f"{model}")
            ax.plot(g.n_train, g.train_auroc, ":", color=f"C{i}", lw=1)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel("training photos per fold")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC (solid: held-out, dotted: training)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage9_learning_curves.png", dpi=150)

    extra = {"auroc": "Held-out AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec"}
    ladder_table = markdown_table(ladder, ["population", "n", "anemic", "model", "train", "gap"], extra)
    curve_table = markdown_table([{**c, "train AUROC": f"{c['train_auroc']:.3f}"} for c in curve],
                                 ["population", "model", "fraction", "n_train", "train AUROC"], {"auroc": "Held-out AUROC"})
    region_table = markdown_table(region_rows, ["country", "n", "anemic", "region", "gray_world", "model"],
                                  {"auroc": "AUROC", "change": "Change vs palpebral"})
    copy_table = markdown_table(copy_rows, ["model", "training rows", "mean training rows", "n"])
    text = (
        "# Stage 9: how far can each population be fit on its own?\n\n"
        "Generated by `python -m src.eyelid.ceiling`; settings in `configs/ceiling.yaml`. Mentor advice M1-M3 "
        "(`docs/eyelid_followup.md`). Anemic: CP-AnemiC Hb < 11.0 g/dL (children); Eyes-Defy WHO adult cutoffs by sex "
        "(women < 12.0, men < 13.0). Input is the hand-segmented conjunctiva in every case. Held-out = repeated stratified "
        f"5-fold CV by photo ({n_repeats} repeats, scores averaged), bootstrap 95% CIs ({boot['n_resamples']} resamples). "
        "**train** = AUROC of each fold's model on its own training photos, mean (min-max) over folds: a diagnostic of "
        "how well the model can fit, so no bootstrap. **gap** = train minus held-out; a large gap means the model "
        "memorizes rather than generalizes. The fine-tuned ViT uses one seed and 5 folds. **visionfm_probe** = the "
        "frozen VisionFM external-eye encoder (pretrained on eye photos, mentor advice M10) with the same probe as "
        "DINOv2.\n\n"
        "## Model ladder\n\n" + ladder_table
        + "\n\n## Learning curves\n\nHeld-out AUROC when each training fold is cut to a stratified share. Still rising at "
        "1.0 = more photos would help; flat = the population's ceiling for that model. Points with fewer than "
        f"{cfg['learning_curve']['min_class_train']} training photos of the rarer class are skipped.\n\n"
        + curve_table + "\n\n![Learning curves](figures/stage9_learning_curves.png)\n\n"
        "## Eyes-Defy: mask region and gray-world\n\nOnly photos with all three masks. Color models and the rule, "
        "within one country at a time. **Change vs palpebral** = AUROC minus the same model's (same gray-world setting) on "
        "the palpebral mask, with a paired bootstrap 95% CI on the same photos; a CI that crosses 0 = no clear change."
        "\n\n" + region_table
        + "\n\n## CP-AnemiC: duplicate copies as training augmentation\n\nFolds over all 476 distinct photos; the "
        "training side either keeps every copy of its photos or only one, and copies of a test photo never enter "
        "training. Scored only on the held-out photos that have no copy (n below), the subset where the no-color "
        "control sits near chance.\n\n" + copy_table + "\n"
    )
    (ROOT / "reports" / "stage9_ceiling.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
