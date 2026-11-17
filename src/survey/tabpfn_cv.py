"""Stage 29 Step 7: does TabPFN beat the Step 5 models on the same folds?

TabPFN is a transformer pretrained on millions of made-up tables; it predicts a new table's rows without being trained
on it in the usual way. Same building villages and the same 5 village folds as Step 5, on three question sets: the 6
base questions, the top 10 by importance (chosen inside each training fold exactly as Step 5's winner), and all
questions left after Step 4. Each is compared, on the same resampled villages, with logistic regression on the same
questions.

The locked test was already opened once in Step 6. It is scored again only if TabPFN's best CV AUROC beats Step 5's
best by at least `tabpfn.locked_test_min_gain` (set before any TabPFN result was seen), and then marked a second look.

Needs the TabPFN licence accepted once at ux.priorlabs.ai and TABPFN_TOKEN in .env (git-ignored).
Run from the repo root: python -m src.survey.tabpfn_cv   (after Steps 1-6)
"""
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import OrdinalEncoder

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified_grouped
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, importance_ranking, make_model, prepare, split_kinds
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics


def tabpfn_matrix(X, train):
    """Numbers as they are (blanks stay NaN, TabPFN handles them); categories as integer codes learned on the training
    rows, unseen answers = -1. Returns the matrix and the indices of the category columns."""
    num, cat = split_kinds(X)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X.loc[train, cat])
    A = np.column_stack([X[num].to_numpy(float), enc.transform(X[cat])]) if cat else X[num].to_numpy(float)
    return A, list(range(len(num), len(num) + len(cat)))


def fit_tabpfn(A, cat_idx, train, test, y, cfg, seed):
    from tabpfn import TabPFNClassifier
    model = TabPFNClassifier(random_state=seed, categorical_features_indices=cat_idx,
                             ignore_pretraining_limits=cfg["ignore_pretraining_limits"])
    return model.fit(A[train], y[train]).predict_proba(A[test])[:, 1]


def questions_for(name, base, questions, X_all, y, groups, tr, mc, seed):
    if name == "base":
        return base
    if name == "all":
        return questions
    k = int(name.split("_")[1])
    return importance_ranking(X_all.iloc[tr], y[tr], groups[tr], questions, mc, seed)[:k]


def main():
    load_dotenv(ROOT / ".env")
    base_cfg, cfg = load_config("default"), load_config("feature_selection")
    mc, tc, seed, boot = cfg["models"], cfg["tabpfn"], base_cfg["seed"], base_cfg["bootstrap"]
    data = pd.read_parquet(OUT)
    full = data
    data = data[~data.locked_test].reset_index(drop=True)          # building villages only
    uni = pd.read_csv(ROOT / "reports" / "stage29_univariate.csv")
    questions = uni.code[uni.step4 == "keep"].tolist()
    base = mc["base_questions"]
    feats = data.drop(columns=TARGETS)
    y, groups = data.anemic.to_numpy(), data.cluster.to_numpy()
    folds = repeated_stratified_grouped(y, groups, mc["cv_splits"], mc["cv_repeats"], seed)   # Step 5's folds
    X_all = prepare(feats[columns_for(questions, feats)])
    names = {"base": "base 6 questions", "all": f"all {len(questions)} questions"}
    label = lambda s: names[s] if s in names else f"top {s.split('_')[1]} by importance"
    print(f"building villages: {len(y)} women, {len(set(groups))} villages", flush=True)

    metrics = survey_metrics(load_config("survey"))
    stats = lambda s, m=metrics: summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, m)
    rng = np.random.default_rng(seed)
    dummy = np.full(len(y), np.nan)
    for _, tr, te in folds:
        dummy[te] = y[tr].mean()                                   # the training share anemic, as in Step 5
    rows = [{"questions": "rule: pregnant first, then random", "model": "-",
             **stats(feats.v213.eq("yes").to_numpy() + 0.5 * rng.random(len(y)),
                     survey_metrics(load_config("survey"), probabilities=False))},
            {"questions": "dummy (everyone the same risk)", "model": "dummy", **stats(dummy)}]
    changes = {}
    for s in tc["question_sets"]:
        tab, lr = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for i, (_, tr, te) in enumerate(folds):
            q = questions_for(s, base, questions, X_all, y, groups, tr, mc, seed)
            X = prepare(feats[columns_for(q, feats)])
            A, cat_idx = tabpfn_matrix(X, X.index[tr])
            tab[te] = fit_tabpfn(A, cat_idx, tr, te, y, tc, seed)
            lr[te] = make_model("logistic_regression", X, mc, seed).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            print(f"{label(s)} fold {i + 1}: TabPFN {auroc(y[te], tab[te]):.3f}, LR {auroc(y[te], lr[te]):.3f}", flush=True)
        for model, sc in (("logistic_regression", lr), ("TabPFN 3.5", tab)):
            rows.append({"questions": label(s), "model": model, **stats(sc)})
        packed = np.column_stack([lr, tab])
        changes[label(s)] = bootstrap_ci(lambda yy, ss: auroc(yy, ss[:, 1]) - auroc(yy, ss[:, 0]), y, packed,
                                         boot["n_resamples"], boot["ci"], groups, seed)
        print(f"{label(s)}: TabPFN {rows[-1]['auroc']['estimate']:.3f} vs LR {rows[-2]['auroc']['estimate']:.3f}",
              flush=True)

    step5 = json.loads((ROOT / "results" / "stage29_models.json").read_text())["results"]["rows"]
    step5_best = max(step5, key=lambda r: r["auroc"]["estimate"])
    tab_rows = [r for r in rows if r["model"] == "TabPFN 3.5"]
    tab_best = max(tab_rows, key=lambda r: r["auroc"]["estimate"])
    gain = tab_best["auroc"]["estimate"] - step5_best["auroc"]["estimate"]
    second_look = gain >= tc["locked_test_min_gain"]
    print(f"TabPFN best {tab_best['auroc']['estimate']:.3f} ({tab_best['questions']}) vs Step 5 best "
          f"{step5_best['auroc']['estimate']:.3f}: gain {gain:+.3f}; locked test {'scored' if second_look else 'not opened'}",
          flush=True)

    locked = None
    if second_look:
        test = full.locked_test.to_numpy()
        ffeats, fy = full.drop(columns=TARGETS), full.anemic.to_numpy()
        tr_idx, te_idx = np.flatnonzero(~test), np.flatnonzero(test)
        s = next(k for k in tc["question_sets"] if label(k) == tab_best["questions"])
        Xf_all = prepare(ffeats[columns_for(questions, ffeats)])
        q = questions_for(s, base, questions, Xf_all, fy, full.cluster.to_numpy(), tr_idx, mc, seed)
        X = prepare(ffeats[columns_for(q, ffeats)])
        A, cat_idx = tabpfn_matrix(X, X.index[tr_idx])
        p = fit_tabpfn(A, cat_idx, tr_idx, te_idx, fy, tc, seed)
        locked = {"questions": tab_best["questions"], "model": "TabPFN 3.5 (second look)",
                  **summarize(fy[test], p, boot["n_resamples"], boot["ci"], seed, full.cluster.to_numpy()[test], metrics)}

    path = save_results("stage29_tabpfn", {
        "n_women": len(y), "rows": rows, "auroc_change_vs_lr": changes, "step5_best": {
            "method": step5_best["method"], "model": step5_best["model"], "auroc": step5_best["auroc"]},
        "gain_over_step5_best": gain, "locked_test_scored": second_look, "locked_test": locked},
        {"default": base_cfg, "feature_selection": cfg})

    f = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    text = "\n".join([
        "# Stage 29 Step 7: TabPFN on the same folds",
        "",
        f"Generated by `python -m src.survey.tabpfn_cv` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`tabpfn`). **Building villages only:** "
        f"{len(y)} women in {len(set(groups))} villages, the same {mc['cv_splits']} village folds as Step 5. Anemic = "
        "DHS `v457` (WHO for women on adjusted Hb: < 12.0 g/dL, < 11.0 if pregnant).",
        "",
        "- TabPFN 3.5 is a transformer pretrained on made-up tables (PyTorch). Categories are given to it as codes; "
        "blanks stay blank. The top 10 questions are chosen inside each training fold, as for Step 5's winner.",
        f"- Bootstrap 95% CIs, {boot['n_resamples']} resamples of whole villages.",
        "",
        markdown_table(rows, ["questions", "model"], SURVEY_COLUMNS),
        "",
        "**AUROC change, TabPFN minus logistic regression on the same questions** (paired: same resampled villages):",
        "",
        *[f"- {k}: {f(v)}" for k, v in changes.items()],
        "",
        f"**Against Step 5's best** ({step5_best['method']}, {step5_best['model']}, CV AUROC "
        f"{step5_best['auroc']['estimate']:.3f}): TabPFN's best is {tab_best['auroc']['estimate']:.3f} "
        f"({tab_best['questions']}), a difference of {gain:+.3f}. The rule set before running was to score the locked "
        f"test again only for a gain of at least {tc['locked_test_min_gain']}; "
        + ("it was met, so it was scored once more (below, a second look after Step 6)." if second_look
           else "it was not met, so the locked test was not opened again and Step 6 stands."),
        "",
        *(["## Locked test, second look", "", markdown_table([locked], ["questions", "model"], SURVEY_COLUMNS), ""]
          if locked else []),
    ])
    (ROOT / "reports" / "stage29_tabpfn.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
