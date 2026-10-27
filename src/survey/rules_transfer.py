"""Rules vs models across countries: do simple testing rules travel as well as models trained on Zimbabwe?

Rules need no training and are scored on each country as they are. The one-question rules (age, wealth) take their
direction from the training country only. Models are fit on all of the training country and applied unchanged; on the
training country itself they are scored out-of-fold by cluster. Writes results/rules_transfer.csv, results/rules_transfer.json
and reports/figures/rules_transfer.png.

Run from the repo root: python -m src.survey.rules_transfer
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
from src.survey.train_baselines import feature_frame, make_model, oof_pipeline_scores, rule_scores, survey_metrics

FIGURE = ROOT / "reports" / "figures" / "rules_transfer.png"
SIMPLE_RULES = ["random", "pregnant_first"]
ONE_QUESTION_RULES = ["age", "wealth"]
MODELS = ["logistic_regression", "hist_gradient_boosting"]
RULE_COLOR, MODEL_COLOR = "#8a8984", "#2a78d6"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def one_question_direction(train, feature):
    """+1 if higher values are more often anemic in the training country, else -1."""
    return 1 if auroc(train.anemic.to_numpy(), train[feature].to_numpy()) >= 0.5 else -1


def one_question_scores(df, feature, direction, fill, n_repeats, seed):
    """Rank by one answer in the training country's direction; ties broken at random. Missing answers get `fill`."""
    values = df[feature].fillna(fill).to_numpy(dtype=float)
    return np.array([direction * values + 0.5 * np.random.default_rng(seed + r).random(len(df)) for r in range(n_repeats)])


def plot(df, path):
    countries = list(dict.fromkeys(df.country))
    names = list(dict.fromkeys(df.contestant))
    fig, axes = plt.subplots(1, len(countries), figsize=(11, 3.8), sharey=True, sharex=True)
    for ax, country in zip(axes, countries):
        sub = df[df.country == country].set_index("contestant").loc[names]
        yy = np.arange(len(names))[::-1]
        for yi, (_, r) in zip(yy, sub.iterrows()):
            color = MODEL_COLOR if r.kind == "model" else RULE_COLOR
            ax.plot([r.auroc_lo, r.auroc_hi], [yi, yi], color=color, lw=2, zorder=2)
            ax.scatter(r.auroc, yi, s=40, color=color, edgecolor="white", lw=1.5, zorder=3)
        ax.axvline(0.5, color=INK_2, lw=0.8, ls=":")
        ax.set_title(country, color=INK, loc="left", fontsize=10)
        ax.set_yticks(yy, [n.replace("_", " ") for n in names], color=INK)
        ax.set_xlim(0.44, 0.64)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.tick_params(colors=INK_2, length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.set_xlabel("test AUROC (0.5 = coin flip)", color=INK_2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l)
               for c, l in [(MODEL_COLOR, "model trained on Zimbabwe"), (RULE_COLOR, "rule (no training)")]]
    fig.legend(handles=handles, frameon=False, loc="upper right", ncol=2, labelcolor=INK_2, fontsize=9)
    fig.suptitle("Rules vs models, scored in each country (95% CI)", x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    base, cfg = load_config("default"), load_config("survey")
    seed, boot, n_rep = base["seed"], base["bootstrap"], cfg["cv"]["n_repeats"]
    metrics = {k: v for k, v in survey_metrics(cfg).items() if k != "brier"}
    metrics["share_tested"] = partial(share_tested_at_sensitivity, sens=cfg["triage_sensitivity"])
    train_name = cfg["train_country"]
    train = load(train_name)
    Xtr, ytr = feature_frame(train, cfg), train.anemic.to_numpy()
    directions = {f: one_question_direction(train, f) for f in ONE_QUESTION_RULES}
    fills = {f: float(train[f].median()) for f in ONE_QUESTION_RULES}
    print(f"one-question rule directions from {train_name}: "
          + ", ".join(f"{f} {'higher' if d > 0 else 'lower'} first" for f, d in directions.items()), flush=True)
    fitted = {kind: make_model(kind, cfg, seed).fit(Xtr, ytr) for kind in MODELS}
    folds = cluster_folds(ytr, train.cluster.to_numpy(), cfg["cv"]["n_splits"], n_rep, seed)

    rows = []
    for country in cfg["countries"]:
        df = load(country)
        y, groups = df.anemic.to_numpy(), df.cluster.to_numpy()
        label = f"{country.capitalize()} ({'cross-validated' if country == train_name else 'untouched'})"
        candidates = [(f"rule: {r}", "rule", rule_scores(r, df, n_rep, seed)) for r in SIMPLE_RULES]
        candidates += [(f"rule: {f} ({'higher' if directions[f] > 0 else 'lower'} first)", "rule",
                        one_question_scores(df, f, directions[f], fills[f], n_rep, seed)) for f in ONE_QUESTION_RULES]
        for kind in MODELS:
            s = (oof_pipeline_scores(kind, Xtr, ytr, None, folds, cfg, seed) if country == train_name
                 else fitted[kind].predict_proba(feature_frame(df, cfg))[:, 1])
            candidates.append((kind, "model", s))
        for name, kind, s in candidates:
            m = summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, metrics)
            rows.append({"country": label, "contestant": name, "kind": kind, "n": len(y), "summary": m})
            print(f"{label:28s} {name:32s} AUROC {m['auroc']['estimate']:.3f} [{m['auroc']['lo']:.3f}, "
                  f"{m['auroc']['hi']:.3f}]  test {m['share_tested']['estimate']:.0%} to catch 90%", flush=True)

    flat = pd.DataFrame([{
        **{k: r[k] for k in ("country", "contestant", "kind", "n")},
        "auroc": r["summary"]["auroc"]["estimate"], "auroc_lo": r["summary"]["auroc"]["lo"],
        "auroc_hi": r["summary"]["auroc"]["hi"],
        **{k: r["summary"][k]["estimate"] for k in ("pr_auc", "recall_top", "share_tested")},
    } for r in rows])
    RESULTS.mkdir(exist_ok=True)
    flat.to_csv(RESULTS / "rules_transfer.csv", index=False)
    plot(flat, FIGURE)
    save_results("rules_transfer", {"directions": directions, "rows": rows}, {"default": base, "survey": cfg})
    print(f"-> results/rules_transfer.csv, {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
