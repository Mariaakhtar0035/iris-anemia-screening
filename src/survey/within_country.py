"""Within-country models and the overfit test: train and test inside each country, cross-validated by cluster.

For every model, the train AUROC (on the rows it was fit on, averaged over folds) sits next to the out-of-fold test
AUROC. The ceiling is the best train AUROC any model could reach: women with identical answers get the same score, so
it is the AUROC of each answer combination's own anemia rate on a fold's training rows, averaged over the same folds.
Writes results/within_country.csv, results/within_country.json and reports/figures/within_country.png.

Run from the repo root: python -m src.survey.within_country
"""
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.cluster_split import cluster_folds
from src.common.evaluate import summarize
from src.common.io import RESULTS, ROOT, load_config, save_results
from src.common.metrics import auroc, share_tested_at_sensitivity
from src.survey.build_dataset import load
from src.survey.train_baselines import feature_frame, make_model, rule_scores, survey_metrics

FIGURE = ROOT / "reports" / "figures" / "within_country.png"
MODELS = ["logistic_regression", "hist_gradient_boosting", "huge_model"]
RULES = ["random", "pregnant_first"]
TRAIN_COLOR, TEST_COLOR = "#eb6834", "#2a78d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def build(kind, cfg, seed):
    if kind == "huge_model":
        return make_model("hist_gradient_boosting", {**cfg, "hist_gradient_boosting": cfg["huge_model"]}, seed)
    return make_model(kind, cfg, seed)


def train_and_test_scores(kind, X, y, folds, cfg, seed):
    """Out-of-fold P(anemic), shape (n_repeats, n), and the mean AUROC on each fold's own training rows."""
    scores = np.full((max(r for r, _, _ in folds) + 1, len(y)), np.nan)
    train_aurocs = []
    for r, train, test in folds:
        model = build(kind, cfg, seed + r).fit(X.iloc[train], y[train])
        scores[r, test] = model.predict_proba(X.iloc[test])[:, 1]
        train_aurocs.append(auroc(y[train], model.predict_proba(X.iloc[train])[:, 1]))
    assert not np.isnan(scores).any()
    return scores, float(np.mean(train_aurocs))


def ceiling(X, y, folds):
    """Mean over folds of the in-sample AUROC of each answer combination's anemia rate: what perfect memorization of
    the answers reaches on the same training rows the models see. Also returns the number of distinct combinations."""
    key = X.map(str).agg("|".join, axis=1).to_numpy()  # map(str) turns a missing wealth into "nan", its own answer

    def memorized(rows):
        return auroc(y[rows], pd.Series(y[rows]).groupby(key[rows]).transform("mean").to_numpy())

    return float(np.mean([memorized(train) for _, train, _ in folds])), len(set(key))


def plot(df, path):
    countries = list(dict.fromkeys(df.country))
    names = [f"rule: {r}" for r in RULES] + MODELS
    fig, axes = plt.subplots(1, len(countries), figsize=(11, 3.6), sharey=True, sharex=True)
    for ax, country in zip(axes, countries):
        sub = df[df.country == country].set_index("model").loc[names]
        yy = np.arange(len(names))[::-1]
        for yi, (_, r) in zip(yy, sub.iterrows()):
            if not np.isnan(r.train_auroc):
                ax.plot([r.test_auroc, r.train_auroc], [yi, yi], color=GRID, lw=2, zorder=1)
                ax.scatter(r.train_auroc, yi, s=40, color=TRAIN_COLOR, edgecolor="white", lw=1.5, zorder=3)
            ax.plot([r.test_lo, r.test_hi], [yi, yi], color=TEST_COLOR, lw=2, zorder=2)
            ax.scatter(r.test_auroc, yi, s=40, color=TEST_COLOR, edgecolor="white", lw=1.5, zorder=3)
        ax.axvline(0.5, color=INK_2, lw=0.8, ls=":")
        ax.axvline(sub.ceiling.iloc[0], color=INK_2, lw=0.8, ls="--")
        ax.text(sub.ceiling.iloc[0], len(names) - 0.4, " ceiling", color=INK_2, fontsize=8, va="bottom")
        ax.set_title(country.capitalize(), color=INK, loc="left", fontsize=10)
        ax.set_yticks(yy, [n.replace("_", " ") for n in names], color=INK)
        ax.set_xlim(0.45, 1.02)
        ax.set_ylim(-0.6, len(names))
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.tick_params(colors=INK_2, length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.set_xlabel("AUROC (0.5 = coin flip)", color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for c, l in [(TEST_COLOR, "test: new villages (95% CI)"), (TRAIN_COLOR, "train: women it studied")]]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=2, labelcolor=INK_2, fontsize=9)
    fig.suptitle("Within each country: what models learn vs what carries to new villages", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    base, cfg = load_config("default"), load_config("survey")
    seed, boot = base["seed"], base["bootstrap"]
    metrics = {**survey_metrics(cfg), "share_tested": partial(share_tested_at_sensitivity, sens=cfg["triage_sensitivity"])}
    rule_metrics = {k: v for k, v in metrics.items() if k != "brier"}
    rows = []
    for country in cfg["countries"]:
        df = load(country)
        X, y, groups = feature_frame(df, cfg), df.anemic.to_numpy(), df.cluster.to_numpy()
        folds = cluster_folds(y, groups, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        top, n_combos = ceiling(X, y, folds)
        print(f"\n{country}: {len(y)} women, {y.mean():.1%} anemic, {len(set(groups))} clusters; "
              f"{n_combos} distinct answer combinations; ceiling AUROC {top:.3f}", flush=True)
        candidates = [(f"rule: {r}", rule_scores(r, df, cfg["cv"]["n_repeats"], seed), np.nan, rule_metrics)
                      for r in RULES]
        for kind in MODELS:
            s, train_auc = train_and_test_scores(kind, X, y, folds, cfg, seed)
            candidates.append((kind, s, train_auc, metrics))
        for name, s, train_auc, m in candidates:
            summary = summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, m)
            rows.append({"country": country, "model": name, "n": len(y), "prevalence": float(y.mean()),
                         "ceiling": top, "train_auroc": train_auc, "summary": summary})
            print(f"  {name:24s} train {train_auc:.3f}  test {summary['auroc']['estimate']:.3f} "
                  f"[{summary['auroc']['lo']:.3f}, {summary['auroc']['hi']:.3f}]  "
                  f"test {summary['share_tested']['estimate']:.0%} to catch 90%", flush=True)

    flat = pd.DataFrame([{
        **{k: r[k] for k in ("country", "model", "n", "prevalence", "ceiling", "train_auroc")},
        "test_auroc": r["summary"]["auroc"]["estimate"],
        "test_lo": r["summary"]["auroc"]["lo"], "test_hi": r["summary"]["auroc"]["hi"],
        **{k: r["summary"][k]["estimate"] for k in ("pr_auc", "recall_top", "share_tested")},
        "brier": r["summary"]["brier"]["estimate"] if "brier" in r["summary"] else np.nan,
    } for r in rows])
    flat.to_csv(RESULTS / "within_country.csv", index=False)
    plot(flat, FIGURE)
    save_results("within_country", {"rows": rows}, {"default": base, "survey": cfg})
    print(f"-> {(RESULTS / 'within_country.csv').relative_to(ROOT)}, {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
