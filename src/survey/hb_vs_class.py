"""Stage 29 Step 8e: is the gain from predicting the Hb number, or from many questions with a flexible model?
(review follow-up)

Role: supporting check, Stage 29 (Hb number vs. yes/no classifier; budget curves).

1. Same 482 questions, same 5 x 5 village folds, same gradient boosting settings: a yes/no classifier vs. the Hb
   regressor ranked by (her cutoff - predicted Hb). Paired AUROC differences for any anemia (primary) and moderate or
   severe. TabPFN classification on all questions (Step 8a) is shown for reference.
2. Grouped permutation importance for both models: each group's columns are shuffled together on the held-out folds of
   the first repeat. Groups are the DHS sections fixed in the config, with region as its own group. It says which
   blocks of questions the model relies on, not why.
3. Budget curve: share of cases found when testing 10-70% of women, highest score first. Cross-validation for every
   model; on the locked test only for models already scored there (descriptive).

Run from the repo root: python -m src.survey.hb_vs_class   (after Steps 8a, 8d and 9)
"""
import json
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, recall_at_top_fraction
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.common.evaluate import paired_gain
from src.survey.repeated_cv import CACHE, Setup, area_codes, group_rate
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, make_model, prepare, section_of
from src.survey.severity import severity_label

FIGURES = ROOT / "reports" / "figures"
COLORS = {"region": "#2a78d6", "lr_base": "#eb6834", "gbc_all": "#1baf7a", "hb_all": "#eda100"}  # validated palette
MARKERS = {"region": "s", "lr_base": "o", "gbc_all": "^", "hb_all": "D"}
NAMES = {"random": "random testing", "region": "region share (no model)", "lr_base": "base 6, logistic regression",
         "gbc_all": "all questions, gradient boosting classifier", "hb_all": "all questions, Hb regressor ranking",
         "tabpfn_all": "all questions, TabPFN classifier (Step 8a)"}


def question_groups(questions, sections, region_codes):
    """{group: questions}: region codes on their own, everything else by DHS section."""
    groups = {"region": [q for q in questions if q in region_codes]}
    for q in questions:
        if q not in region_codes:
            groups.setdefault(section_of(q, sections), []).append(q)
    return groups


def grouped_importance(model, X, y_eval, score_fn, groups, repeats, seed):
    """Drop in score_fn(y_eval, prediction) when each group's columns are shuffled together (same row order)."""
    rng = np.random.default_rng(seed)
    base = score_fn(y_eval, model, X)
    out = {}
    for g, qs in groups.items():
        cols = columns_for(qs, X)
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            for c in cols:                       # column by column, so each keeps its own type
                Xp[c] = X[c].to_numpy()[perm]
            drops.append(base - score_fn(y_eval, model, Xp))
        out[g] = float(np.mean(drops))
    return out


def oof_classifier(name, X, y, S):
    path = CACHE / f"{name}.npy"
    if path.exists():
        return np.load(path)
    out = np.full((S.rc["cv_repeats"], len(y)), np.nan)
    for r, tr, te in S.folds:
        out[r, te] = make_model("hist_gradient_boosting", X, S.mc, S.seed).fit(X.iloc[tr], y[tr]).predict_proba(
            X.iloc[te])[:, 1]
    np.save(path, out)
    return out


def main():
    seed = load_config("default")["seed"]
    S = Setup(seed)
    cfg, boot, hv, st = S.cfg, S.base_cfg["bootstrap"], S.cfg["hb_vs_class"], S.cfg["severity_target"]
    survey = load_config("survey")
    full = pd.read_parquet(OUT)
    locked = full.locked_test.to_numpy()
    sev_full = severity_label(full, st, survey)
    sev = sev_full[~locked]
    X_all, sm = S.X_all, cfg["screening"]["smoothing"]
    hb_pred = S.cutoff - np.load(CACHE / "hb__all.npy")
    budgets = hv["budgets"]
    budget_metrics = {f"top_{int(b * 100)}": partial(recall_at_top_fraction, frac=b) for b in budgets}

    # Scores on the building villages, both targets.
    X_base = prepare(S.feats[columns_for(S.base, S.feats)])
    sev_region, sev_lr = (np.full((S.rc["cv_repeats"], len(sev)), np.nan) for _ in range(2))
    for r, tr, te in S.folds:
        sev_region[r, te] = group_rate(area_codes(S.feats, False), sev, tr, te, sm)
        sev_lr[r, te] = make_model("logistic_regression", X_base, S.mc, seed).fit(
            X_base.iloc[tr], sev[tr]).predict_proba(X_base.iloc[te])[:, 1]
    targets = {
        "any anemia": (S.y, {"region": np.load(CACHE / "region.npy"), "lr_base": np.load(CACHE / "lr__base.npy"),
                             "gbc_all": oof_classifier("gbc__all", X_all, S.y, S), "hb_all": S.cutoff - hb_pred,
                             "tabpfn_all": np.load(CACHE / "tabpfn__all.npy")}),
        "moderate or severe": (sev, {"region": sev_region, "lr_base": sev_lr,
                                     "gbc_all": oof_classifier("gbc_sev__all", X_all, sev, S),
                                     "hb_all": st["hb_cutoff_g_dl"] - hb_pred}),
    }
    print("scores ready", flush=True)

    # 1. Classifier vs. Hb regressor, and 3. budget curve (CV).
    compare, budget_rows = [], []
    for target, (y, scores) in targets.items():
        mean = {k: v.mean(0) for k, v in scores.items()}
        for k in ("gbc_all", "hb_all", *(["tabpfn_all"] if "tabpfn_all" in scores else [])):
            compare.append({"target": target, "model": NAMES[k],
                            **summarize(y, scores[k], boot["n_resamples"], boot["ci"], seed, S.groups,
                                        {"auroc": auroc})})
        compare.append({"target": target, "model": "Hb regressor minus classifier (paired)",
                        "auroc": paired_gain(y, mean["hb_all"], mean["gbc_all"], S.groups, boot, seed)})
        if "tabpfn_all" in mean:
            compare.append({"target": target, "model": "Hb regressor minus TabPFN classifier (paired)",
                            "auroc": paired_gain(y, mean["hb_all"], mean["tabpfn_all"], S.groups, boot, seed)})
        for k in ("region", "lr_base", "gbc_all", "hb_all"):
            budget_rows.append({"target": target, "model": NAMES[k],
                                **summarize(y, scores[k], boot["n_resamples"], boot["ci"], seed, S.groups,
                                            budget_metrics)})
    print("comparison and budget done", flush=True)

    # 2. Grouped permutation importance, first repeat's held-out folds.
    groups = question_groups(S.questions, S.mc["sections"], set(hv["region_group"]))
    imp = {"Hb regressor (AUROC drop, any anemia)": [], "Hb regressor (MAE rise, g/dL)": [],
           "classifier (AUROC drop, any anemia)": []}
    rank_auc = lambda yy, m, X, cut: auroc(yy, cut - m.predict(X))
    for r, tr, te in S.folds:
        if r != 0:
            continue
        Xtr, Xte = X_all.iloc[tr], X_all.iloc[te].reset_index(drop=True)
        reg = make_model("hgb_regressor", X_all, S.mc, seed).fit(Xtr, S.hb[tr])
        clf = make_model("hist_gradient_boosting", X_all, S.mc, seed).fit(Xtr, S.y[tr])
        cut = S.cutoff[te]
        imp["Hb regressor (AUROC drop, any anemia)"].append(grouped_importance(
            reg, Xte, S.y[te], lambda yy, m, X: rank_auc(yy, m, X, cut), groups, hv["importance_repeats"], seed))
        imp["Hb regressor (MAE rise, g/dL)"].append(grouped_importance(
            reg, Xte, S.hb[te], lambda yy, m, X: -float(np.mean(np.abs(yy - m.predict(X)))), groups,
            hv["importance_repeats"], seed))
        imp["classifier (AUROC drop, any anemia)"].append(grouped_importance(
            clf, Xte, S.y[te], lambda yy, m, X: auroc(yy, m.predict_proba(X)[:, 1]), groups,
            hv["importance_repeats"], seed))
        print(f"importance fold {len(imp['classifier (AUROC drop, any anemia)'])}", flush=True)
    imp_tables = {k: pd.DataFrame(v) for k, v in imp.items()}

    # 3. Budget curve on the locked test, models already scored there (descriptive).
    tr, te = np.flatnonzero(~locked), np.flatnonzero(locked)
    ffeats, fg = full.drop(columns=TARGETS), full.cluster.to_numpy()[te]
    Xb, Xa = prepare(ffeats[columns_for(S.base, ffeats)]), prepare(ffeats[columns_for(S.questions, ffeats)])
    hb_hat = make_model("hgb_regressor", Xa, S.mc, seed).fit(Xa.iloc[tr], full.hb.to_numpy()[tr]).predict(Xa.iloc[te])
    step6 = json.loads((ROOT / "results" / "stage29_final_test.json").read_text())["results"]["rows"]
    step9 = json.loads((ROOT / "results" / "stage29_severity.json").read_text())["results"]["locked_test"]
    locked_rows = []
    for target, yy, cut, done in (("any anemia", full.anemic.to_numpy(), full.hb_cutoff.to_numpy()[te], step6),
                                  ("moderate or severe", sev_full, st["hb_cutoff_g_dl"], step9)):
        s = {"region": group_rate(area_codes(ffeats, False), yy, tr, te, sm),
             "lr_base": make_model("logistic_regression", Xb, S.mc, seed).fit(Xb.iloc[tr], yy[tr]).predict_proba(
                 Xb.iloc[te])[:, 1],
             "hb_all": cut - hb_hat}
        if target == "moderate or severe":         # Step 9 scored the classifier on the locked test for this target
            s["gbc_all"] = make_model("hist_gradient_boosting", Xa, S.mc, seed).fit(
                Xa.iloc[tr], yy[tr]).predict_proba(Xa.iloc[te])[:, 1]
        for k, v in s.items():
            locked_rows.append({"target": target, "model": NAMES[k], "auroc_check": float(auroc(yy[te], v)),
                                **summarize(yy[te], v, boot["n_resamples"], boot["ci"], seed, fg, budget_metrics)})
    # Frozen models: the refits reproduce the AUROCs already reported.
    want = {("any anemia", "base 6, logistic regression"): next(r for r in step6 if r["model"] == "base 6 questions"),
            ("any anemia", NAMES["hb_all"]): next(r for r in step6 if r["model"].startswith("Hb ranking")),
            ("moderate or severe", "base 6, logistic regression"): next(r for r in step9 if r["model"].startswith("base 6")),
            ("moderate or severe", NAMES["hb_all"]): next(r for r in step9 if r["model"].startswith("Hb ranking")),
            ("moderate or severe", NAMES["gbc_all"]): next(r for r in step9 if r["model"].startswith("all questions")),
            ("moderate or severe", NAMES["region"]): next(
                r for r in step9 if r["model"].startswith("baseline: share in her region (no"))}
    for r in locked_rows:
        if (r["target"], r["model"]) in want:
            assert abs(r["auroc_check"] - want[(r["target"], r["model"])]["auroc"]["estimate"]) < 1e-9, r["model"]

    # Figures: CV budget curves, one per target.
    FIGURES.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0.05, 0.95, 19)
    for target, (y, scores) in targets.items():
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot([0, 1], [0, 1], color="#8a8a85", lw=1.2, ls="--")
        ax.text(0.80, 0.70, "random testing", color="#5b5b56", fontsize=8, rotation=33, ha="center", va="top")
        for k in ("region", "lr_base", "gbc_all", "hb_all"):
            m = scores[k].mean(0)
            found = [recall_at_top_fraction(y, m, f) for f in grid]
            ax.plot(grid, found, color=COLORS[k], lw=2, marker=MARKERS[k], ms=4, label=NAMES[k])
        for b in budgets:
            ax.axvline(b, color="#e4e3df", lw=0.8, zorder=0)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        ax.set_xlabel("share of women tested, highest score first")
        ax.set_ylabel(f"share of {target} cases found")
        ax.set_title(f"Testing budget, {target} (Zimbabwe, cross-validation)", fontsize=10)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(FIGURES / f"stage29_budget_{'any' if target == 'any anemia' else 'modsev'}.png", dpi=150)
        plt.close(fig)

    path = save_results("stage29_hb_vs_class", {
        "compare": compare, "budget_cv": budget_rows, "budget_locked": locked_rows,
        "groups": groups, "importance": {k: v.to_dict() for k, v in imp_tables.items()}},
        {"default": S.base_cfg, "feature_selection": cfg, "survey": survey})

    bcols = {f"top_{int(b * 100)}": f"test {b:.0%}" for b in budgets}
    random_rows = [{"target": t, "model": NAMES["random"],
                    **{f"top_{int(b * 100)}": {"estimate": b, "lo": b, "hi": b} for b in budgets}} for t in targets]
    imp_lines = []
    for k, t in imp_tables.items():
        order = t.mean().sort_values(ascending=False)
        imp_lines += [f"**{k}**", "", "| group | questions | mean | SD over folds |", "|---|---|---|---|",
                      *[f"| {g} | {len(groups[g])} | {order[g]:+.4f} | {t[g].std():.4f} |" for g in order.index], ""]
    text = "\n".join([
        "# Stage 29 Step 8e: Hb regression vs. classification, grouped importance, and the budget curve",
        "",
        f"Generated by `python -m src.survey.hb_vs_class` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`hb_vs_class`). Building villages, the same 5 x 5 repeated village folds as "
        f"Step 8a; village-bootstrap 95% CIs and paired differences, {boot['n_resamples']} resamples.",
        "",
        "- **Any anemia (primary):** DHS `v457` 1-3, adjusted Hb < 12.0 g/dL, or < 11.0 if pregnant (WHO). "
        "**Moderate or severe:** DHS `v457` 1-2, adjusted Hb < 10.0 g/dL for every woman, pregnant or not.",
        f"- Classifier and Hb regressor: the same {len(S.questions)} questions, the same gradient boosting settings "
        "(`models.hist_gradient_boosting`), the same folds. The regressor ranks women by her cutoff minus predicted Hb.",
        "",
        "## 1. Does predicting the Hb number help, with the questions and the learner held fixed?",
        "",
        markdown_table(compare, ["target", "model"], {"auroc": "AUROC"}),
        "",
        "## 2. Grouped permutation importance",
        "",
        "Each group's columns are shuffled together (the same row order for all of them) on the held-out folds of the "
        f"first repeat, {hv['importance_repeats']} shuffles each; the table gives the mean drop in AUROC (or rise in Hb "
        "error) over the 5 folds. Groups are the DHS sections fixed in the config, with region (`"
        + "`, `".join(groups["region"]) + "`) as its own group. This shows which blocks the model relies on, not "
        "why: a block can stand in for a cause, and region may also carry differences between survey teams and their "
        "Hb measurement, not only biology.",
        "",
        *imp_lines,
        "## 3. Budget curve: share of cases found when testing a share of women, highest score first",
        "",
        "Cross-validation (building villages):",
        "",
        markdown_table(random_rows + budget_rows, ["target", "model"], bcols),
        "",
        f"Locked test, descriptive ({len(set(fg))} villages): only models already scored there (Steps 6 and 9), refit "
        "unchanged; the refits reproduce their earlier AUROCs exactly.",
        "",
        markdown_table(locked_rows, ["target", "model"], bcols),
        "",
        "![Budget curve, any anemia](figures/stage29_budget_any.png) "
        "![Budget curve, moderate or severe](figures/stage29_budget_modsev.png)",
        "",
        "Each figure plots the cross-validation scores; the dashed line is random testing. The numbers behind the "
        "curves are in the tables above.",
        "",
    ])
    (ROOT / "reports" / "stage29_hb_vs_class.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
