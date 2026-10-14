"""Stage 5: train eyelid models on CP-AnemiC and test on Eyes-Defy-Anemia unchanged, and the reverse.

Run from the repo root: python -m src.eyelid.cross_dataset
"""
import numpy as np
import pandas as pd

from src.common.evaluate import oof_scores, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import repeated_stratified
from src.eyelid.color_features import feature_table
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.embed import compute, load_embeddings
from src.eyelid.report import markdown_table
from src.eyelid.train_color import model_factories
from src.eyelid.train_probe import probe_factory

ED_LABELS = ROOT / "data" / "processed" / "eyes_defy_labels.csv"
ED_CACHE = ROOT / "data" / "processed" / "embeddings" / "dinov2_small_eyes_defy_palpebral.npz"
PROBE_SET = "cls_and_patch_mean"


def load_eyes_defy(cfg):
    """Eyes-Defy photos with Hb, their palpebral mask, and the WHO adult label by sex."""
    df = pd.read_csv(ED_LABELS).dropna(subset=["hb"]).reset_index(drop=True)
    masks = []
    for p in df.image_path:
        found = [m for m in (ROOT / p).parent.glob(cfg["eyes_defy_mask_glob"]) if "forniceal" not in m.name]
        assert len(found) == 1, f"{p}: expected one palpebral mask, found {found}"
        masks.append(str(found[0].relative_to(ROOT)))
    cut = df.sex.map(cfg["eyes_defy_cutoff_g_dl"])
    assert cut.notna().all(), "a sex value with no cutoff"
    return df.assign(image_path=masks, anemic=(df.hb < cut).astype(int))


def eyes_defy_embeddings(ed, probe_cfg):
    if ED_CACHE.exists():
        cached = np.load(ED_CACHE, allow_pickle=False)
        if str(cached["revision"]) == probe_cfg["revision"] and list(cached["ids"]) == list(ed.id):
            return cached[PROBE_SET]
    return compute(ed, probe_cfg, cache=ED_CACHE, crop=True)[PROBE_SET]


def inputs(data, emb, color_cfg):
    """Color features and DINOv2 embedding for each row, in row order."""
    paths = [ROOT / p for p in data.image_path]
    colors = feature_table(paths, data.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    return {"color": colors.to_numpy(), "rule": -colors[color_cfg["rule_feature"]].to_numpy(), "embedding": emb}


def models(color_cfg, probe_cfg, seed):
    color = model_factories(color_cfg, seed)
    return {"dummy": ("color", color["dummy"]), "logistic_regression": ("color", color["logistic_regression"]),
            "random_forest": ("color", color["random_forest"]), "dinov2_probe": ("embedding", probe_factory(probe_cfg, seed))}


def main():
    base, cfg = load_config("default"), load_config("cross_dataset")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    child_cut = base["anemia_cutoff_g_dl"]["child_6_59_months"]

    cp = load_cp_anemic(cfg["cp_anemic_subset"])
    all_emb = load_embeddings(probe_cfg)
    pos = {i: k for k, i in enumerate(all_emb["ids"])}
    cp_emb = all_emb[PROBE_SET][[pos[i] for i in cp.id]]
    ed = load_eyes_defy(cfg)
    data = {"cp_anemic": (cp, inputs(cp, cp_emb, color_cfg)),
            "eyes_defy": (ed, inputs(ed, eyes_defy_embeddings(ed, probe_cfg), color_cfg))}
    print(f"cp_anemic: {len(cp)} photos, {cp.anemic.mean():.1%} anemic (Hb < {child_cut}); eyes_defy: {len(ed)} photos, "
          f"{ed.anemic.mean():.1%} anemic (WHO by sex), India {ed[ed.country == 'India'].anemic.mean():.1%}, "
          f"Italy {ed[ed.country == 'Italy'].anemic.mean():.1%}", flush=True)

    rows = []

    def add(label, test_name, subset, y, s, n_train):
        rows.append({"train -> test": label, "test set": subset, "n": int(len(y)), "anemic": f"{y.mean():.1%}",
                     "model": test_name, "n_train": n_train,
                     **summarize(y, s, boot["n_resamples"], boot["ci"], seed)})

    # Within each dataset, for reference: repeated stratified 5x5 CV (one photo per patient in both subsets).
    for name, (df, x) in data.items():
        y = df.anemic.to_numpy()
        folds = repeated_stratified(y, cfg["within_cv"]["n_splits"], cfg["within_cv"]["n_repeats"], seed)
        for model, (kind, make) in models(color_cfg, probe_cfg, seed).items():
            add(f"{name} (within, 5x5 CV)", model, "all", y, oof_scores(make, x[kind], y, folds), len(y))
        add(f"{name} (within, 5x5 CV)", "rule_low_redness", "all", y, x["rule"], 0)

    # Across datasets: fit on all of one, score all of the other unchanged.
    for src, dst in (("cp_anemic", "eyes_defy"), ("eyes_defy", "cp_anemic")):
        (sdf, sx), (ddf, dx) = data[src], data[dst]
        ys, yd = sdf.anemic.to_numpy(), ddf.anemic.to_numpy()
        scores = {m: make().fit(sx[kind], ys).predict_proba(dx[kind])[:, 1]
                  for m, (kind, make) in models(color_cfg, probe_cfg, seed).items()}
        scores["rule_low_redness"] = dx["rule"]
        if dst == "eyes_defy":
            scores["rule_country (India = anemic)"] = (ddf.country == "India").astype(float).to_numpy()
        subsets = {"all": np.ones(len(ddf), bool)}
        if dst == "eyes_defy":
            subsets |= {c: (ddf.country == c).to_numpy() for c in ("India", "Italy")}
        for subset, m in subsets.items():
            for model, s in scores.items():
                if model.startswith("rule_country") and subset != "all":
                    continue  # constant within one country
                add(f"{src} -> {dst}", model, subset, yd[m], s[m], len(ys))
        print(f"{src} -> {dst}: " + ", ".join(f"{r['model']} {r['auroc']['estimate']:.3f}" for r in rows
                                             if r["train -> test"] == f"{src} -> {dst}" and r["test set"] == "all"), flush=True)

    table = markdown_table(rows, ["train -> test", "test set", "n", "anemic", "model"])
    text = (
        "# Stage 5: eyelid models across datasets\n\n"
        "Generated by `python -m src.eyelid.cross_dataset`. Settings in `configs/cross_dataset.yaml`. Input on both sides is "
        "the hand-segmented palpebral conjunctiva (CP-AnemiC cut-outs; Eyes-Defy `*_palpebral.png` masks cropped to the "
        f"conjunctiva). CP-AnemiC: the {len(cp)} photos with no copy, anemic = Hb < {child_cut} g/dL (children). "
        f"Eyes-Defy: {len(ed)} photos with Hb, anemic = WHO adult cutoffs by sex (women < 12.0, men < 13.0 g/dL; "
        "pregnancy not recorded), which is **not** the ViT paper's labelling used in the replication. The cutoffs differ "
        "between the datasets by design. Across datasets, models are fit on all of one dataset and applied unchanged. "
        f"Bootstrap 95% CIs ({boot['n_resamples']} resamples, one photo per patient). `rule_low_redness`: minus mean a*, "
        "no training. `rule_country`: India = anemic.\n\n" + table + "\n"
    )
    (ROOT / "reports" / "stage5_cross_dataset.md").write_text(text)
    print(text)
    save_results("stage5_cross_dataset", {"n_cp_anemic": len(cp), "n_eyes_defy": len(ed), "rows": rows},
                 {"default": base, "cross_dataset": cfg, "color": color_cfg, "probe": probe_cfg})


if __name__ == "__main__":
    main()
