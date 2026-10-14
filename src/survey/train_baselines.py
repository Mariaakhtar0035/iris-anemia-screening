"""Stage 7: survey baselines on Zimbabwe, cross-validated by cluster.

Run from the repo root: python -m src.survey.train_baselines
"""
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.evaluate import IMAGE_METRICS, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import brier, recall_at_top_fraction
from src.common.splits import repeated_stratified_grouped
from src.eyelid.report import IMAGE_COLUMNS, markdown_table
from src.survey.build_dataset import load

FIGURES = ROOT / "reports" / "figures"
SURVEY_COLUMNS = {**IMAGE_COLUMNS, "recall_top": "Recall, top 30% tested", "brier": "Brier"}


def survey_metrics(cfg, probabilities=True):
    """CLAUDE.md's survey metrics; Brier only for scores that are probabilities (not the ranking rules)."""
    m = {**IMAGE_METRICS, "recall_top": partial(recall_at_top_fraction, frac=cfg["top_fraction"])}
    return {**m, "brier": brier} if probabilities else m


def make_model(kind, cfg, seed):
    """Preprocessing and classifier in one pipeline, so imputation, scaling and encoding are fit on training folds only."""
    f = cfg["features"]
    prep = ColumnTransformer([
        ("num", StandardScaler(), f["numeric"]),
        ("ord", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), f["ordinal"]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=cfg["onehot_min_frequency"]),
         f["categorical"]),
    ])
    if kind == "dummy":
        clf = DummyClassifier(strategy="prior")
    elif kind == "logistic_regression":
        clf = LogisticRegression(**cfg["logistic_regression"], random_state=seed)
    elif kind == "hist_gradient_boosting":
        clf = HistGradientBoostingClassifier(**cfg["hist_gradient_boosting"], random_state=seed)
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("clf", clf)])


def feature_frame(df, cfg):
    f = cfg["features"]
    return df[f["numeric"] + f["ordinal"] + f["categorical"]]


def oof_pipeline_scores(kind, X, y, w, folds, cfg, seed):
    """Out-of-fold P(anemic), shape (n_repeats, n); `w` = survey weights used in fitting, or None."""
    scores = np.full((max(r for r, _, _ in folds) + 1, len(y)), np.nan)
    for r, train, test in folds:
        fit_kw = {} if w is None else {"clf__sample_weight": w[train]}
        model = make_model(kind, cfg, seed + r).fit(X.iloc[train], y[train], **fit_kw)
        scores[r, test] = model.predict_proba(X.iloc[test])[:, 1]
    assert not np.isnan(scores).any()
    return scores


def rule_scores(rule, df, n_repeats, seed):
    """Testing rules as scores: random order, or pregnant women first and then random order."""
    out = []
    for r in range(n_repeats):
        noise = np.random.default_rng(seed + r).random(len(df))
        out.append(noise if rule == "random" else df.pregnant.to_numpy() + 0.5 * noise)
    return np.array(out)


def plots(y, lr, scores, cfg):
    FIGURES.mkdir(parents=True, exist_ok=True)
    frac_pos, mean_pred = calibration_curve(y, lr, n_bins=cfg["calibration_bins"], strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label="logistic regression")
    ax.set_xlabel("predicted chance of anemia")
    ax.set_ylabel("share actually anemic")
    ax.set_title("Calibration, Zimbabwe (out-of-fold)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "survey_calibration.png", dpi=150)

    budgets = cfg["budget_curve"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, s in scores.items():
        ax.plot(budgets, [recall_at_top_fraction(y, s, b) for b in budgets], "o-", ms=3, label=name)
    ax.axvline(cfg["top_fraction"], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("share of women tested (highest risk first)")
    ax.set_ylabel("share of anemic women found")
    ax.set_title("Recall vs testing budget, Zimbabwe (out-of-fold)")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "survey_budget.png", dpi=150)


def main():
    base, cfg = load_config("default"), load_config("survey")
    seed, boot = base["seed"], base["bootstrap"]
    df = load(cfg["train_country"])
    X, y, w, groups = feature_frame(df, cfg), df.anemic.to_numpy(), df.weight.to_numpy(), df.cluster.to_numpy()
    folds = repeated_stratified_grouped(y, groups, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
    print(f"{cfg['train_country']}: {len(y)} women, {y.mean():.1%} anemic, {len(set(groups))} clusters; "
          f"{len(folds)} folds", flush=True)

    rows, mean_scores = [], {}
    for rule in ("random", "pregnant_first"):
        s = rule_scores(rule, df, cfg["cv"]["n_repeats"], seed)
        rows.append({"model": f"rule: {rule}", "weights": "-",
                     **summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, survey_metrics(cfg, False))})
        mean_scores[f"rule: {rule}"] = s.mean(0)
    for kind in ("dummy", "logistic_regression", "hist_gradient_boosting"):
        for weighted in (False, True):
            s = oof_pipeline_scores(kind, X, y, w if weighted else None, folds, cfg, seed)
            m = summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, survey_metrics(cfg))
            rows.append({"model": kind, "weights": "survey" if weighted else "none", **m})
            if not weighted:
                mean_scores[kind] = s.mean(0)
            print(f"{kind} weights={weighted}: AUROC {m['auroc']['estimate']:.3f}", flush=True)

    plots(y, mean_scores["logistic_regression"], {k: v for k, v in mean_scores.items() if k != "dummy"}, cfg)
    table = markdown_table(rows, ["model", "weights"], SURVEY_COLUMNS)
    text = (
        "# Stage 7: survey baselines on Zimbabwe\n\n"
        f"Generated by `python -m src.survey.train_baselines`. {len(y)} women 15-49 from the Zimbabwe 2015 DHS with an Hb "
        "result. Anemic = DHS `v457` at any severity (WHO: Hb adjusted for altitude and smoking < 12.0 g/dL, < 11.0 if "
        f"pregnant); {y.mean():.1%} anemic. StratifiedGroupKFold by cluster, {cfg['cv']['n_splits']} folds x "
        f"{cfg['cv']['n_repeats']} seeds; scores averaged over seeds. Bootstrap 95% CIs ({boot['n_resamples']} resamples) "
        "resample whole clusters. `weights = survey` fits with the DHS sample weights; metrics are unweighted. Rules rank "
        "women and have no probability, so no Brier score.\n\n"
        f"{table}\n\n"
        "![Calibration](figures/survey_calibration.png) ![Recall vs testing budget](figures/survey_budget.png)\n"
    )
    (ROOT / "reports" / "stage7_survey.md").write_text(text)
    print(text)
    save_results("stage7_survey_baselines", {"n": len(y), "anemic_share": float(y.mean()), "rows": rows},
                 {"default": base, "survey": cfg})


if __name__ == "__main__":
    main()
