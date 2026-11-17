"""Stage 29 Step 5: do the questions together beat the 6-question baseline, and which way of choosing them works best?

Building villages only (the locked final test is still untouched), 5-fold cross-validation by village. Compared:
the dummy and the "pregnant first" rule; the 6 base questions; all questions left after Step 4; and three ways of
choosing a smaller set, each fit inside the training folds only:
  (a) L1 logistic regression, which gives most questions a zero weight;
  (b) the top k questions by gradient boosting permutation importance;
  (c) forward selection by DHS section, starting from the 6 base questions.
Extra target: the Hb number for the base 6 and all questions (Ridge and gradient boosting regressors).

Run from the repo root: python -m src.survey.select_models   (after select_features and screen_features)
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified_grouped
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics



def columns_for(questions, data):
    """A question's model columns: itself, plus its <code>_special side column when it has one."""
    return [c for q in questions for c in (q, f"{q}_special") if c in data]


def split_kinds(X):
    num = [c for c in X if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X if c not in num]


def prepare(X):
    """Blank answers become their own category; numbers keep their blanks for the imputer or the trees."""
    num, cat = split_kinds(X)
    X = X.copy()
    X[cat] = X[cat].astype(object).where(X[cat].notna(), "blank").astype(str)
    return X


def make_model(kind, X, mc, seed, C=None):
    """Preprocessing and model in one pipeline, so imputing, scaling and encoding are learned on training rows only."""
    num, cat = split_kinds(X)
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind in ("hist_gradient_boosting", "hgb_regressor"):
        prep = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, max_categories=250), cat),
        ])
        cat_idx = list(range(len(num), len(num) + len(cat)))
        params = {**mc["hist_gradient_boosting"], "categorical_features": cat_idx or None, "random_state": seed}
        model = (HistGradientBoostingClassifier if kind == "hist_gradient_boosting" else HistGradientBoostingRegressor)
        return Pipeline([("prep", prep), ("model", model(**params))])
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=mc["onehot_min_frequency"]), cat),
    ])
    if kind == "logistic_regression":
        model = LogisticRegression(**mc["logistic_regression"], random_state=seed)
    elif kind == "l1_logistic_regression":
        model = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, max_iter=mc["logistic_regression"]["max_iter"],
                                   random_state=seed)
    elif kind == "ridge":
        model = Ridge(**mc["ridge"])
    else:
        raise ValueError(kind)
    return Pipeline([("prep", prep), ("model", model)])


def l1_questions(fitted, X):
    """Questions with at least one non-zero weight in a fitted L1 pipeline."""
    names = fitted.named_steps["prep"].get_feature_names_out()
    used = names[fitted.named_steps["model"].coef_[0] != 0]
    cols = list(X.columns)
    # Output names look like "num__v012", "num__missingindicator_v012" or "cat__v312_pill"; map back to the column.
    hits = {c for n in used for c in cols if re.match(rf"^(num|cat)__(missingindicator_)?{re.escape(c)}(_|$)", n)}
    return sorted({c.removesuffix("_special") for c in hits})


def inner_holdout(y, groups, seed):
    """One 80/20 split of the given rows by village."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(y)), y, groups))


def importance_ranking(X, y, groups, questions, mc, seed):
    """Questions from most to least important: AUROC drop when shuffled, on 20% of these villages held out."""
    fit_idx, val_idx = inner_holdout(y, groups, seed)
    model = make_model("hist_gradient_boosting", X, mc, seed).fit(X.iloc[fit_idx], y[fit_idx])
    imp = permutation_importance(model, X.iloc[val_idx], y[val_idx], scoring="roc_auc",
                                 n_repeats=mc["importance_repeats"], random_state=seed)
    per_column = pd.Series(imp.importances_mean, index=X.columns)
    per_question = per_column.groupby(lambda c: c.removesuffix("_special")).sum()
    return per_question.reindex(questions).sort_values(ascending=False).index.tolist()


def section_of(question, sections):
    for name, pattern in sections.items():
        if re.match(pattern, question):
            return name
    return None


def forward_sections(data, y, groups, base, sections_q, mc, seed):
    """Start from the base questions; add the DHS section that raises inner-CV AUROC most, until none adds min_gain."""
    folds = repeated_stratified_grouped(y, groups, mc["forward_inner_splits"], 1, seed)

    def score(questions):
        X = prepare(data[columns_for(questions, data)])
        s = np.zeros(len(y))
        for _, tr, te in folds:
            s[te] = make_model("logistic_regression", X, mc, seed).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        return auroc(y, s)

    chosen, current, path = [], score(base), []
    remaining = dict(sections_q)
    while remaining:
        trials = {name: score(base + [q for s in chosen + [name] for q in sections_q[s]]) for name in remaining}
        best = max(trials, key=trials.get)
        if trials[best] - current < mc["forward_min_gain"]:
            break
        chosen.append(best)
        path.append((best, trials[best] - current))
        current = trials[best]
        remaining.pop(best)
    return chosen, path


def oof_method(method, data, y, groups, folds, base, questions, sections_q, mc, seed):
    """Out-of-fold P(anemic) for one method, with the questions it used in each fold."""
    scores, used = np.full(len(y), np.nan), []
    for _, tr, te in folds:
        kind, arg = method
        if kind in ("base", "all"):
            q = base if kind == "base" else questions
            model_kind = arg
        elif kind == "l1":
            q, model_kind = questions, "l1_logistic_regression"
        elif kind == "top_k":
            ranking = importance_ranking(prepare(data[columns_for(questions, data)]).iloc[tr], y[tr], groups[tr],
                                         questions, mc, seed)
            q, model_kind = ranking[:arg[0]], arg[1]
        elif kind == "forward":
            chosen, _ = forward_sections(data.iloc[tr], y[tr], groups[tr], base, sections_q, mc, seed)
            q, model_kind = base + [x for s in chosen for x in sections_q[s]], "logistic_regression"
        X = prepare(data[columns_for(q, data)])
        model = make_model(model_kind, X, mc, seed, C=arg if kind == "l1" else None).fit(X.iloc[tr], y[tr])
        scores[te] = model.predict_proba(X.iloc[te])[:, 1]
        used.append(l1_questions(model, X) if kind == "l1" else q)
    return scores, used


def hb_rows(data, hb, cutoff, y, groups, folds, question_sets, mc, seed, boot):
    """Extra target: predict Hb, then rank women by how far the estimate falls below their own cutoff."""
    rows = []
    for name, q in question_sets.items():
        X = prepare(data[columns_for(q, data)])
        for kind in ("ridge", "hgb_regressor"):
            pred, mean_guess = np.full(len(y), np.nan), np.full(len(y), np.nan)
            for _, tr, te in folds:
                pred[te] = make_model(kind, X, mc, seed).fit(X.iloc[tr], hb[tr]).predict(X.iloc[te])
                mean_guess[te] = hb[tr].mean()
            ci = lambda fn, a, b: bootstrap_ci(fn, a, b, boot["n_resamples"], boot["ci"], groups, seed)
            mae = lambda a, b: float(np.mean(np.abs(a - b)))
            flagged = (pred < cutoff).astype(float)
            rows.append({"questions": name, "model": kind, "mae": ci(mae, hb, pred),
                         "mae_mean_guess": ci(mae, hb, mean_guess),
                         "sens_at_cutoff": ci(lambda a, f: _share(f[a == 1]), y, flagged),
                         "spec_at_cutoff": ci(lambda a, f: _share(1 - f[a == 0]), y, flagged),
                         **summarize(y, -(pred - cutoff), boot["n_resamples"], boot["ci"], seed, groups,
                                     survey_metrics_rank())})
            print(f"Hb {name} {kind}: MAE {rows[-1]['mae']['estimate']:.3f} "
                  f"(mean guess {rows[-1]['mae_mean_guess']['estimate']:.3f}), "
                  f"AUROC {rows[-1]['auroc']['estimate']:.3f}", flush=True)
    return rows


def _share(v):
    if not len(v):
        raise ValueError("no women in this group")
    return float(v.mean())


def survey_metrics_rank():
    return survey_metrics(load_config("survey"), probabilities=False)


def main():
    base_cfg, cfg = load_config("default"), load_config("feature_selection")
    mc, seed, boot = cfg["models"], base_cfg["seed"], base_cfg["bootstrap"]
    data = pd.read_parquet(OUT)
    data = data[~data.locked_test].reset_index(drop=True)          # building villages only
    uni = pd.read_csv(ROOT / "reports" / "stage29_univariate.csv")
    questions = uni.code[uni.step4 == "keep"].tolist()
    base = mc["base_questions"]
    sections_q = {}
    for q in questions:
        if q not in base:
            sections_q.setdefault(section_of(q, mc["sections"]), []).append(q)
    assert None not in sections_q, f"questions outside every section: {sections_q.get(None)}"
    y, hb, cutoff, groups = (data.anemic.to_numpy(), data.hb.to_numpy(), data.hb_cutoff.to_numpy(),
                             data.cluster.to_numpy())
    feats = data.drop(columns=TARGETS)
    folds = repeated_stratified_grouped(y, groups, mc["cv_splits"], mc["cv_repeats"], seed)
    print(f"building villages: {len(y)} women, {len(set(groups))} villages; {len(questions)} questions in "
          f"{len(sections_q)} sections", flush=True)

    metrics, rank_metrics = survey_metrics(load_config("survey")), survey_metrics_rank()
    stats = lambda s, m=metrics: summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, m)
    rows = []
    rng = np.random.default_rng(seed)
    rows.append({"method": "rule: pregnant first, then random", "model": "-", "questions": 1,
                 **stats(feats.v213.eq("yes").to_numpy() + 0.5 * rng.random(len(y)), rank_metrics)})
    methods = [("dummy (everyone the same risk)", ("base", "dummy")),
               ("base 6 questions", ("base", "logistic_regression")),
               ("base 6 questions", ("base", "hist_gradient_boosting")),
               (f"all {len(questions)} questions", ("all", "logistic_regression")),
               (f"all {len(questions)} questions", ("all", "hist_gradient_boosting")),
               *[(f"(a) L1, C = {C}", ("l1", C)) for C in mc["l1_C"]],
               *[(f"(b) top {k} by importance", ("top_k", (k, m))) for k in mc["importance_top_k"]
                 for m in ("logistic_regression", "hist_gradient_boosting")],
               ("(c) forward by DHS section", ("forward", None))]
    selections = {}
    for name, method in methods:
        s, used = oof_method(method, feats, y, groups, folds, base, questions, sections_q, mc, seed)
        model = {"l1": "l1_logistic_regression", "forward": "logistic_regression"}.get(method[0])
        model = model or (method[1][1] if method[0] == "top_k" else method[1])
        n_q = float(np.mean([len(u) for u in used])) if method[1] != "dummy" else 0
        rows.append({"method": name, "model": model, "questions": round(n_q), **stats(s)})
        selections[f"{name} | {model}"] = used
        print(f"{name} | {model}: AUROC {rows[-1]['auroc']['estimate']:.3f} with {n_q:.0f} questions", flush=True)

    candidates = [r for r in rows if r["method"].startswith("(")]
    best = max(r["auroc"]["estimate"] for r in candidates)
    near = [r for r in candidates if r["auroc"]["estimate"] >= best - mc["winner_tolerance"]]
    winner = min(near, key=lambda r: (r["questions"], -r["auroc"]["estimate"]))
    print(f"winner: {winner['method']} | {winner['model']}", flush=True)

    hb = hb_rows(feats, hb, cutoff, y, groups, folds, {"base 6": base, f"all {len(questions)}": questions},
                 mc, seed, boot)
    path = save_results("stage29_models", {"n_women": len(y), "rows": rows, "hb_rows": hb,
                                           "selections": selections,
                                           "winner": {"method": winner["method"], "model": winner["model"]},
                                           "sections": sections_q},
                        {"default": base_cfg, "feature_selection": cfg})

    cols = {**SURVEY_COLUMNS}
    hb_cols = {k: v for k, v in SURVEY_COLUMNS.items() if k != "brier"}
    forward_picks = pd.Series([s for fold in selections["(c) forward by DHS section | logistic_regression"]
                               for s in sections_q if set(sections_q[s]) <= set(fold)]).value_counts()
    text = "\n".join([
        "# Stage 29 Step 5: models and ways of choosing questions (cross-validation)",
        "",
        f"Generated by `python -m src.survey.select_models` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`models`). **Building villages only:** "
        f"{len(y)} women in {len(set(groups))} villages, {mc['cv_splits']}-fold by village; the locked final test is "
        "still untouched. Anemic = DHS `v457` (WHO for women on adjusted Hb: < 12.0 g/dL, < 11.0 if pregnant).",
        "",
        "- Every selection method chooses its questions inside the training folds only, so its score is honest.",
        "- `questions` = how many questions the model used (average over folds).",
        f"- Bootstrap 95% CIs, {boot['n_resamples']} resamples of whole villages. The rule ranks women and has no "
        "probability, so no Brier score.",
        "",
        markdown_table(rows, ["method", "model", "questions"], cols),
        "",
        f"**Winner on cross-validation: {winner['method']} ({winner['model']}).** Rule: the highest AUROC among the "
        f"selection methods, except that one with fewer questions wins if it is within {mc['winner_tolerance']} of "
        "the best. It is scored once on the locked final test in Step 6.",
        "",
        "## (c) Which DHS sections forward selection added (folds out of "
        f"{len(selections['(c) forward by DHS section | logistic_regression'])})",
        "",
        "| section | folds |",
        "|---|---|",
        *[f"| {s} | {n} |" for s, n in forward_picks.items()],
        "",
        "## Extra target: the Hb number",
        "",
        "Regressors predict adjusted Hb (`v456`); women are ranked by how far the estimate falls below their own cutoff "
        "(12.0 g/dL, 11.0 if pregnant). MAE = average error in g/dL; the mean guess predicts everyone at the training "
        "mean. Sens/spec at cutoff apply each woman's cutoff to the estimate itself.",
        "",
        markdown_table(hb, ["questions", "model"], {"mae": "MAE (g/dL)", "mae_mean_guess": "MAE, mean guess",
                                                    "sens_at_cutoff": "Sens at cutoff", "spec_at_cutoff": "Spec at cutoff"}),
        "",
        markdown_table(hb, ["questions", "model"], hb_cols),
        "",
    ])
    (ROOT / "reports" / "stage29_models.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
