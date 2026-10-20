"""Stage 10: why do the eyelid populations differ? (mentor advice M4, docs/eyelid_followup.md)

1. Who is in each population, and what their conjunctiva looks like, split by anemia status.
2. How easily the populations can be told apart from the same inputs the anemia models use (a shortcut
   detector), compared with how easily anemic and non-anemic people can be told apart within one population.
3. Italy's two acquisition batches, and how much of Italy's anemia signal the batch alone explains.

Run from the repo root: python -m src.eyelid.populations
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.evaluate import oof_scores, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import FIGURES, population_frames
from src.eyelid.color_features import feature_table
from src.eyelid.cross_dataset import eyes_defy_embeddings
from src.eyelid.embed import load_embeddings
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import model_factories
from src.eyelid.train_probe import probe_factory

NAMES = {"cp_anemic": "Ghana children", "eyes_defy_india": "India adults", "eyes_defy_italy": "Italy adults"}
PROBE_SET = "cls_and_patch_mean"


def mean_ci(values, boot, seed):
    v = np.asarray(values, dtype=float)
    return bootstrap_ci(lambda _, s: float(s.mean()), np.zeros(len(v)), v, boot["n_resamples"], boot["ci"], seed=seed)


def difference_ci(y, values, boot, seed):
    """Mean of `values` where y == 1 minus where y == 0, with a bootstrap CI."""
    return bootstrap_ci(lambda yy, s: float(s[yy == 1].mean() - s[yy == 0].mean()), np.asarray(y), np.asarray(values, float),
                        boot["n_resamples"], boot["ci"], seed=seed)


def main():
    base, cfg, color_cfg, probe_cfg = (load_config(n) for n in ("default", "populations", "color", "probe"))
    seed, boot = base["seed"], base["bootstrap"]
    frames = population_frames(load_config("ceiling"))
    pops = {k: frames[k] for k in NAMES}

    # One table across all three populations: people, color features, DINOv2 embedding.
    df = pd.concat([f.assign(population=k) for k, f in pops.items()], ignore_index=True)
    df["age_years"] = np.where(df.population == "cp_anemic", df.age_months / 12, df.age)
    df["female"] = df.sex.isin(["Female", "F"]).astype(float)
    feats = feature_table([ROOT / p for p in df.image_path], df.id, False, color_cfg["roi"]["alpha_threshold"])
    df = df.join(feats[cfg["measures"]].reset_index(drop=True))
    emb = load_embeddings(probe_cfg)
    pos = {i: k for k, i in enumerate(emb["ids"])}
    ed = frames["eyes_defy_pooled"]
    ed_emb = pd.DataFrame(eyes_defy_embeddings(ed, probe_cfg), index=ed.id)
    embedding = np.vstack([emb[PROBE_SET][[pos[i] for i in pops["cp_anemic"].id]], ed_emb.loc[pops["eyes_defy_india"].id],
                           ed_emb.loc[pops["eyes_defy_italy"].id]])
    color = feats.to_numpy()

    # 1. Descriptive table, by population and anemia status.
    desc = []
    for (pop, a), g in df.groupby(["population", "anemic"], sort=False):
        row = {"population": NAMES[pop], "anemic": "yes" if a else "no", "n": len(g), "female": f"{g.female.mean():.0%}"}
        for col in ["hb", "age_years"] + cfg["measures"]:
            row[col] = fmt(mean_ci(g[col], boot, seed))
        desc.append(row)
    desc_cols = ["population", "anemic", "n", "female", "hb", "age_years"] + cfg["measures"]
    desc_table = "| " + " | ".join(desc_cols) + " |\n|" + "---|" * len(desc_cols) + "\n" + "\n".join(
        "| " + " | ".join(str(r[c]) for c in desc_cols) + " |" for r in desc)

    # 2. a* gaps: anemic minus non-anemic inside each population, vs population minus population among the non-anemic.
    gaps = []
    for pop in NAMES:
        g = df[df.population == pop]
        gaps.append({"comparison": f"{NAMES[pop]}: anemic minus non-anemic", "n": len(g),
                     **{m: difference_ci(g.anemic, g[m], boot, seed) for m in ("a_mean", "L_mean")}})
    for p1, p2 in (("eyes_defy_india", "eyes_defy_italy"), ("cp_anemic", "eyes_defy_italy"), ("cp_anemic", "eyes_defy_india")):
        g = df[df.population.isin([p1, p2]) & (df.anemic == 0)]
        gaps.append({"comparison": f"non-anemic only: {NAMES[p1]} minus {NAMES[p2]}", "n": len(g),
                     **{m: difference_ci(g.population == p1, g[m], boot, seed) for m in ("a_mean", "L_mean")}})
    gap_table = markdown_table(gaps, ["comparison", "n"], {"a_mean": "a* difference", "L_mean": "L* difference"})

    # 3. Shortcut detectors: the anemia models' own inputs, asked "which population is this photo from?" instead.
    lr, probe = model_factories(color_cfg, seed)["logistic_regression"], probe_factory(probe_cfg, seed)
    tasks = {
        "India vs Italy, all photos": (df.population.isin(["eyes_defy_india", "eyes_defy_italy"]), df.population == "eyes_defy_india"),
        "India vs Italy, non-anemic only": (df.population.isin(["eyes_defy_india", "eyes_defy_italy"]) & (df.anemic == 0),
                                            df.population == "eyes_defy_india"),
        "India vs Italy, anemic only": (df.population.isin(["eyes_defy_india", "eyes_defy_italy"]) & (df.anemic == 1),
                                        df.population == "eyes_defy_india"),
        "Ghana children vs Eyes-Defy adults": (pd.Series(True, index=df.index), df.population == "cp_anemic"),
    }
    shortcut = []
    for task, (rows, target) in tasks.items():
        m, y = rows.to_numpy(), target.to_numpy()[rows.to_numpy()].astype(int)
        folds = repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        for model, (make, X) in {"logistic_regression (color)": (lr, color), "dinov2_probe": (probe, embedding)}.items():
            shortcut.append({"task": task, "n": int(m.sum()), "model": model,
                             **summarize(y, oof_scores(make, X[m], y, folds), boot["n_resamples"], boot["ci"], seed)})
        print(f"shortcut: {task} done", flush=True)
    # The anemia signal inside each population, with the same two models, for comparison.
    for pop in NAMES:
        m = (df.population == pop).to_numpy()
        y = df.anemic.to_numpy()[m]
        folds = repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        for model, (make, X) in {"logistic_regression (color)": (lr, color), "dinov2_probe": (probe, embedding)}.items():
            shortcut.append({"task": f"anemic vs not, {NAMES[pop]} only", "n": int(m.sum()), "model": model,
                             **summarize(y, oof_scores(make, X[m], y, folds), boot["n_resamples"], boot["ci"], seed)})
    shortcut_table = markdown_table(shortcut, ["task", "n", "model"])

    # Italy's two batches: the batch alone as a "score", and the redness rule with and without the batch mixed in.
    it = df[df.population == "eyes_defy_italy"].copy()
    early = (it.id.str.split("_").str[1].astype(int) <= cfg["italy_early_batch_last_folder"]).to_numpy()
    batch_desc = []
    for (b, a), g in it.groupby([np.where(early, "early (folders 1-31)", "later (folders 32-123)"), "anemic"]):
        batch_desc.append({"batch": b, "anemic": "yes" if a else "no", "n": len(g),
                           **{c: mean_ci(g[c], boot, seed) for c in ("hb", "L_mean", "a_mean")}})
    y_it, rule = it.anemic.to_numpy(), -it.a_mean.to_numpy()
    non = y_it == 0
    batch_rows = [{"score": s_name, "rows": rows_name, "n": int(m.sum()), "positives": int(yy[m].sum()),
                   **summarize(yy[m], s[m], boot["n_resamples"], boot["ci"], seed, metrics={"auroc": auroc})}
                  for s_name, rows_name, yy, s, m in [
                      ("batch only (early = 1), predicting anemia", "all Italy", y_it, early.astype(float), np.ones(len(it), bool)),
                      ("redness rule, predicting anemia", "all Italy", y_it, rule, np.ones(len(it), bool)),
                      ("redness rule, predicting anemia", "early batch only", y_it, rule, early),
                      ("redness rule, predicting the early batch", "non-anemic only", early.astype(int), rule, non)]]
    batch_desc_table = markdown_table(batch_desc, ["batch", "anemic", "n"], {"hb": "Hb", "L_mean": "L*", "a_mean": "a*"})
    batch_table = markdown_table(batch_rows, ["score", "rows", "n", "positives"], {"auroc": "AUROC"})

    # Figure: a* and L* by population and anemia status.
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (col, label) in zip(axes, [("a_mean", "a* (redness)"), ("L_mean", "L* (lightness)")]):
        data, ticks = [], []
        for pop in NAMES:
            for a in (0, 1):
                data.append(df[(df.population == pop) & (df.anemic == a)][col])
                ticks.append(f"{NAMES[pop]}\n{'anemic' if a else 'not anemic'}")
        ax.boxplot(data, tick_labels=ticks, showfliers=False)
        for i, d in enumerate(data):
            ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(seed).uniform(-0.15, 0.15, len(d)), d, s=5,
                       alpha=0.4, color="C3" if i % 2 else "C0")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Conjunctiva color by population and anemia status")
    fig.tight_layout()
    fig.savefig(FIGURES / "stage10_populations.png", dpi=150)

    text = (
        "# Stage 10: why do the eyelid populations differ?\n\n"
        "Generated by `python -m src.eyelid.populations`; settings in `configs/populations.yaml`. Mentor advice M4 "
        "(`docs/eyelid_followup.md`). Populations as in Stage 9: CP-AnemiC distinct photos (Ghana children, anemic = "
        "Hb < 11.0 g/dL) and Eyes-Defy India and Italy (adults, WHO by sex: women < 12.0, men < 13.0). Means with "
        f"bootstrap 95% CIs ({boot['n_resamples']} resamples). Color measured on the hand-segmented conjunctiva in "
        "CIELAB: L* lightness, a* red (+) to green (-), b* yellow (+) to blue (-); S = HSV saturation.\n\n"
        "## Who is in each population\n\n" + desc_table
        + "\n\n## Redness gaps: anemia vs population\n\nHow much a* (and L*) moves with anemia inside a population, "
        "compared with how much it moves between populations among people who are **not** anemic.\n\n" + gap_table
        + "\n\n![Populations](figures/stage10_populations.png)\n\n"
        "## Shortcut detectors\n\nThe anemia models' own inputs (29 color features, or the DINOv2 embedding), with a "
        "different question: which population is this photo from? 5x5 stratified CV. An AUROC near 1 means the "
        "population is obvious from the pixels, so a model can use it as a shortcut whenever prevalence differs. The "
        "last rows show the anemia signal inside each population, with the same models, for comparison.\n\n"
        + shortcut_table
        + "\n\n## Italy: two acquisition batches\n\nEyes-Defy Italy came in two batches: folders 1-31 (photos named "
        "`N.jpg`, masks on a white background) and folders 32-123 (photos named `T_*`). Anemia sits almost entirely in "
        "the first. If the early batch was also photographed differently, an Italian model can score well by "
        "recognizing the batch. The last row asks whether non-anemic people in the early batch already look paler.\n\n"
        + batch_desc_table + "\n\n" + batch_table + "\n"
    )
    (ROOT / "reports" / "stage10_populations.md").write_text(text)
    print(text)
    save_results("stage10_populations", {"descriptive": desc, "gaps": gaps, "shortcut": shortcut,
                                         "italy_batches": {"descriptive": batch_desc, "auroc": batch_rows}},
                 {"default": base, "populations": cfg, "color": color_cfg, "probe": probe_cfg})


if __name__ == "__main__":
    main()
