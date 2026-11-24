"""Stage 29 Step 8a: the key models again with more power, plus region baselines (review follow-up).

Role: main result, Stage 29 (the headline 5 x 5 CV numbers and region baselines).

5 x 5 repeated village CV on the building villages only (the locked test is not used, except by the amended TabPFN
rule below). Each woman's score is the average of her 5 out-of-fold scores. CIs and paired AUROC differences come from
a village bootstrap of those averaged scores, not from the spread across folds, because repeated-CV folds overlap.

Models: dummy; "pregnant first" rule; two no-model baselines (the training villages' anemia share in her region, and
in her region x urban/rural); base 6 logistic regression; the Step 5 winner (top 10 by importance, chosen inside each
training fold); the Hb ranking (gradient boosting regressor on all questions, score = her cutoff minus predicted Hb);
TabPFN 3.5 on base 6, top 10 and all.

Amended TabPFN rule (configs/feature_selection.yaml `review`, written before Step 7's output was opened): the locked
test is scored again only if a TabPFN set beats base 6 logistic regression by >= tabpfn_min_gain with a 95% CI above 0.

Each model's out-of-fold scores are cached in data/processed/stage29_oof/ so a rerun skips finished models.
Run from the repo root: python -m src.survey.repeated_cv   (after Steps 1-7)
"""
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.common.evaluate import paired_gain, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified_grouped
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, importance_ranking, make_model, prepare
from src.survey.tabpfn_cv import fit_tabpfn, tabpfn_matrix
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics

CACHE = ROOT / "data" / "processed" / "stage29_oof"


def group_rate(codes, y, train, test, smoothing):
    """No-model baseline: each test woman gets her group's anemia share among the training women, pulled toward the
    overall training share as if the group had `smoothing` more average women. Unseen groups get the overall share."""
    prior = y[train].mean()
    k = codes.max() + 1
    s = np.bincount(codes[train], y[train], minlength=k)
    n = np.bincount(codes[train], minlength=k)
    return ((s + smoothing * prior) / (n + smoothing))[codes[test]]


def area_codes(feats, with_residence):
    key = feats.v024.astype(str)
    if with_residence:
        key = key + " | " + feats.v025.astype(str)
    return pd.factorize(key)[0]


class Setup:
    """Data, folds and the per-fold top-10 questions, shared by every model."""

    def __init__(self, seed):
        self.base_cfg, self.cfg = load_config("default"), load_config("feature_selection")
        self.mc, self.rc, self.seed = self.cfg["models"], self.cfg["review"], seed
        data = pd.read_parquet(OUT)
        data = data[~data.locked_test].reset_index(drop=True)
        uni = pd.read_csv(ROOT / "reports" / "stage29_univariate.csv")
        self.questions = uni.code[uni.step4 == "keep"].tolist()
        self.base = self.mc["base_questions"]
        self.feats = data.drop(columns=TARGETS)
        self.y, self.hb = data.anemic.to_numpy(), data.hb.to_numpy()
        self.cutoff, self.groups = data.hb_cutoff.to_numpy(), data.cluster.to_numpy()
        self.folds = repeated_stratified_grouped(self.y, self.groups, self.rc["cv_splits"], self.rc["cv_repeats"], seed)
        self.X_all = prepare(self.feats[columns_for(self.questions, self.feats)])

    def top_k(self, i, tr, k):
        path = CACHE / f"ranking_fold{i}.json"
        if not path.exists():
            ranking = importance_ranking(self.X_all.iloc[tr], self.y[tr], self.groups[tr], self.questions, self.mc,
                                         self.seed)
            path.write_text(json.dumps(ranking))
        return json.loads(path.read_text())[:k]


def model_scores(name, S):
    """(n_repeats, n) out-of-fold scores for one model, cached."""
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    n_rep = S.rc["cv_repeats"]
    out = np.full((n_rep, len(S.y)), np.nan)
    sm = S.cfg["screening"]["smoothing"]
    for i, (r, tr, te) in enumerate(S.folds):
        if name == "dummy":
            out[r, te] = S.y[tr].mean()
        elif name == "rule_pregnant_first":
            out[r, te] = S.feats.v213.eq("yes").to_numpy()[te] + 0.5 * np.random.default_rng(S.seed + i).random(len(te))
        elif name in ("region", "region_residence"):
            out[r, te] = group_rate(area_codes(S.feats, name == "region_residence"), S.y, tr, te, sm)
        else:
            kind, qset = name.split("__")
            q = S.base if qset == "base" else S.questions if qset == "all" else S.top_k(i, tr, int(qset.split("_")[1]))
            X = prepare(S.feats[columns_for(q, S.feats)])
            if kind == "lr":
                out[r, te] = make_model("logistic_regression", X, S.mc, S.seed).fit(X.iloc[tr], S.y[tr]).predict_proba(
                    X.iloc[te])[:, 1]
            elif kind == "hb":
                pred = make_model("hgb_regressor", X, S.mc, S.seed).fit(X.iloc[tr], S.hb[tr]).predict(X.iloc[te])
                out[r, te] = S.cutoff[te] - pred
            elif kind == "tabpfn":
                A, cat_idx = tabpfn_matrix(X, X.index[tr])
                out[r, te] = fit_tabpfn(A, cat_idx, tr, te, S.y, S.cfg["tabpfn"], S.seed)
        print(f"{name}: fold {i + 1}/{len(S.folds)}", flush=True)
    assert not np.isnan(out).any()
    np.save(path, out)
    return out


MODELS = {  # cache name: (row label, gives a probability, uses folds' own selection)
    "dummy": ("dummy (everyone the same risk)", True),
    "rule_pregnant_first": ("rule: pregnant first, then random", False),
    "region": ("baseline: anemia share in her region (no model)", True),
    "region_residence": ("baseline: share in her region x urban/rural (no model)", True),
    "lr__base": ("base 6 questions, logistic regression", True),
    "lr__top_10": ("winner: top 10 by importance, logistic regression", True),
    "lr__all": ("all questions, logistic regression", True),
    "hb__all": ("Hb ranking: all questions, gradient boosting regressor", False),
    "tabpfn__base": ("TabPFN 3.5, base 6 questions", True),
    "tabpfn__top_10": ("TabPFN 3.5, top 10 by importance", True),
    "tabpfn__all": ("TabPFN 3.5, all questions", True),
}


def main():
    load_dotenv(ROOT / ".env")
    seed = load_config("default")["seed"]
    CACHE.mkdir(parents=True, exist_ok=True)
    S = Setup(seed)
    boot = S.base_cfg["bootstrap"]
    survey = load_config("survey")
    print(f"building villages: {len(S.y)} women, {len(set(S.groups))} villages, {len(S.folds)} folds", flush=True)

    scores = {k: model_scores(k, S) for k in MODELS}
    mean = {k: v.mean(0) for k, v in scores.items()}
    rows = []
    for k, (label, proba) in MODELS.items():
        m = summarize(S.y, scores[k], boot["n_resamples"], boot["ci"], seed, S.groups, survey_metrics(survey, proba))
        rows.append({"model": label, **m,
                     "gain_vs_base6": paired_gain(S.y, mean[k], mean["lr__base"], S.groups, boot, seed),
                     "gain_vs_region": paired_gain(S.y, mean[k], mean["region"], S.groups, boot, seed)})
        print(f"{label}: AUROC {m['auroc']['estimate']:.3f}", flush=True)

    # Amended TabPFN rule
    rc = S.rc
    tab = {k: next(r for r in rows if r["model"] == MODELS[k][0]) for k in MODELS if k.startswith("tabpfn")}
    passing = {k: r for k, r in tab.items()
               if r["gain_vs_base6"]["estimate"] >= rc["tabpfn_min_gain"] and r["gain_vs_base6"]["lo"] > 0}
    same_q = {k: paired_gain(S.y, mean[k], mean[k.replace("tabpfn", "lr")], S.groups, boot, seed)
              for k in tab if k.replace("tabpfn", "lr") in mean}
    locked, reused = None, False
    if passing:
        best = max(passing, key=lambda k: passing[k]["gain_vs_base6"]["estimate"])
        # Step 7 (old rule) already scored TabPFN on all questions on the locked test; never look a third time.
        step7 = json.loads((ROOT / "results" / "stage29_tabpfn.json").read_text())["results"]["locked_test"]
        if best == "tabpfn__all" and step7 and step7["questions"].startswith("all"):
            locked, reused = {**step7, "model": f"{MODELS[best][0]} (Step 7's second look, reused)"}, True
        else:
            locked = score_tabpfn_locked(best, S, boot, survey)
    print(f"TabPFN sets passing the amended rule: {list(passing) or 'none'}", flush=True)

    path = save_results("stage29_repeated_cv", {
        "n_women": len(S.y), "n_folds": len(S.folds), "rows": rows,
        "tabpfn_vs_same_questions_lr": same_q, "tabpfn_rule_passed": list(passing), "tabpfn_locked_test": locked,
        "tabpfn_locked_reused_from_step7": reused},
        {"default": S.base_cfg, "feature_selection": S.cfg})

    g = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    gain_rows = [{"model": r["model"], "gain_vs_base6": r["gain_vs_base6"], "gain_vs_region": r["gain_vs_region"]}
                 for r in rows]
    text = "\n".join([
        "# Stage 29 Step 8a: repeated cross-validation with region baselines",
        "",
        f"Generated by `python -m src.survey.repeated_cv` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`review`). **Building villages only:** "
        f"{len(S.y)} women in {len(set(S.groups))} villages, {rc['cv_splits']}-fold by village repeated "
        f"{rc['cv_repeats']} times. Anemic = DHS `v457` (WHO for women on Hb adjusted for altitude and smoking: "
        "< 12.0 g/dL, < 11.0 if pregnant).",
        "",
        "- Each woman's score is the average of her out-of-fold scores over the repeats. CIs and paired differences: "
        f"village bootstrap of those averaged scores, {boot['n_resamples']} resamples (not the fold-to-fold spread, "
        "because repeated-CV folds overlap).",
        "- **Region baselines use no model:** the anemia share among training women in her region (10 regions), or "
        "her region x urban/rural cell, pulled toward the overall share as if each group had "
        f"{S.cfg['screening']['smoothing']} more average women.",
        "- **Hb ranking:** a gradient boosting regressor predicts her adjusted Hb from all questions; her score is her "
        "own cutoff (12.0 g/dL, 11.0 if pregnant) minus that prediction, so women predicted furthest below their "
        "cutoff rank first. It is a ranking, not a probability, so no Brier score.",
        "- **The winner's numbers are optimistic:** top 10 by importance was picked as Step 5's winner using these "
        "same building villages. Its questions are chosen inside each training fold, but the decision to use this "
        "method was not. Base 6 and the region baselines have no such selection.",
        "",
        markdown_table(rows, ["model"], SURVEY_COLUMNS),
        "",
        "**Paired AUROC differences** (same women, village bootstrap):",
        "",
        markdown_table(gain_rows, ["model"], {"gain_vs_base6": "minus base 6", "gain_vs_region": "minus region baseline"}),
        "",
        "## TabPFN",
        "",
        "TabPFN minus logistic regression on the same questions (reported, decides nothing):",
        "",
        *[f"- {MODELS[k][0]}: {g(v)}" for k, v in same_q.items()],
        "",
        "Amended rule (written 2026-11-16 before Step 7's output was opened): score the locked test again only if a "
        f"TabPFN set beats base 6 logistic regression by at least {rc['tabpfn_min_gain']} with its 95% CI above 0. "
        + (f"Passed by: {', '.join(MODELS[k][0] for k in passing)}. "
           + ("The best is the set Step 7 already scored on the locked test under the old rule, so that score is "
              "reused below and the locked test was not opened a third time." if reused else
              "The best was scored once on the locked test (below, a second look after Step 6).") if passing else "**No TabPFN set passed; the locked test was not "
                                                                "opened again.**"),
        "",
        *(["## Locked test, second look (TabPFN)", "", markdown_table([locked], ["model"], SURVEY_COLUMNS), ""]
          if locked else []),
    ])
    (ROOT / "reports" / "stage29_repeated_cv.md").write_text(text)
    print(text)


def score_tabpfn_locked(key, S, boot, survey):
    """Fit the passing TabPFN set on all building women and score it once on the locked test."""
    full = pd.read_parquet(OUT)
    feats, y, groups = full.drop(columns=TARGETS), full.anemic.to_numpy(), full.cluster.to_numpy()
    test = full.locked_test.to_numpy()
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    qset = key.split("__")[1]
    if qset == "base":
        q = S.base
    elif qset == "all":
        q = S.questions
    else:
        X_all = prepare(feats[columns_for(S.questions, feats)])
        q = importance_ranking(X_all.iloc[tr], y[tr], groups[tr], S.questions, S.mc, S.seed)[:int(qset.split("_")[1])]
    X = prepare(feats[columns_for(q, feats)])
    A, cat_idx = tabpfn_matrix(X, X.index[tr])
    p = fit_tabpfn(A, cat_idx, tr, te, y, S.cfg["tabpfn"], S.seed)
    return {"model": f"{MODELS[key][0]} (second look)",
            **summarize(y[te], p, boot["n_resamples"], boot["ci"], S.seed, groups[te], survey_metrics(survey))}


if __name__ == "__main__":
    main()
