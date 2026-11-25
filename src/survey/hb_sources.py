"""Stage 29 Step 8d: where does the Hb ranking's gain come from, and what would a clinic see? (review follow-up)

Role: supporting check, Stage 29 (where the Hb ranking's gain comes from; clinic numbers).

1. The same gradient boosting Hb regressor on three question sets: the 6 base questions, base 6 + region (v024), and
   all questions (Step 8a). Women are ranked by her cutoff minus predicted Hb. Compared with paired village-bootstrap
   CIs on the same 5 x 5 folds, for any anemia (the primary target), moderate or severe (DHS v457 1-2: Hb < 10.0 g/dL
   for every woman), and village ranking (Spearman and top-20% overlap; within-region is left out because of the
   cross-validation artifact described in Step 8b).
2. Which questions the all-questions Hb regressor leans on: permutation importance (rise in Hb error when a question
   is shuffled) on each held-out fold of the first repeat.
3. The clinic number: the share of women tested, highest score first, to catch 90% of cases, and the share of those
   tested who have the outcome. Random testing needs 90%. Cross-validation for every model; on the locked test only for
   models already scored there (Steps 6 and 9), so base 6 + region is not taken to the locked test.

Run from the repo root: python -m src.survey.hb_sources   (after Steps 8a, 8b and 9)
"""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, precision_at_sensitivity, share_tested_at_sensitivity
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.survey.area_check import boot_area
from src.common.evaluate import paired_gain
from src.survey.repeated_cv import CACHE, Setup, area_codes, group_rate
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, make_model, prepare
from src.survey.severity import severity_label
from src.survey.train_baselines import survey_metrics


def question_set(spec, S):
    if spec == "all":
        return S.questions
    out = []
    for item in ([spec] if isinstance(spec, str) else spec):
        out += S.base if item == "base_questions" else [item]
    return out


def hb_predictions(name, q, S):
    """(n_repeats, n) out-of-fold predicted Hb from the gradient boosting regressor on questions q, cached."""
    if name == "all":
        return S.cutoff - np.load(CACHE / "hb__all.npy")              # Step 8a stored cutoff - prediction
    path = CACHE / f"hbreg__{name}.npy"
    if path.exists():
        return np.load(path)
    X = prepare(S.feats[columns_for(q, S.feats)])
    out = np.full((S.rc["cv_repeats"], len(S.y)), np.nan)
    for r, tr, te in S.folds:
        out[r, te] = make_model("hgb_regressor", X, S.mc, S.seed).fit(X.iloc[tr], S.hb[tr]).predict(X.iloc[te])
    np.save(path, out)
    return out


def importance(S, hc):
    """Rise in Hb MAE when each question is shuffled, per held-out fold of the first repeat; summed over a question's
    columns (its _special side column included)."""
    X = prepare(S.feats[columns_for(S.questions, S.feats)])
    per_fold = []
    for r, tr, te in S.folds:
        if r != 0:
            continue
        model = make_model("hgb_regressor", X, S.mc, S.seed).fit(X.iloc[tr], S.hb[tr])
        imp = permutation_importance(model, X.iloc[te], S.hb[te], scoring="neg_mean_absolute_error",
                                     n_repeats=hc["importance_repeats"], random_state=S.seed)
        per_fold.append(pd.Series(imp.importances_mean, index=X.columns).groupby(
            lambda c: c.removesuffix("_special")).sum())
        print(f"importance fold {len(per_fold)}", flush=True)
    return pd.DataFrame(per_fold)


def main():
    seed = load_config("default")["seed"]
    S = Setup(seed)
    cfg, boot, hc, st = S.cfg, S.base_cfg["bootstrap"], S.cfg["hb_sources"], S.cfg["severity_target"]
    survey = load_config("survey")
    full = pd.read_parquet(OUT)
    sev_full = severity_label(full, st, survey)
    locked = full.locked_test.to_numpy()
    sev = sev_full[~locked]
    sm, sens = cfg["screening"]["smoothing"], hc["clinic_sensitivity"]
    clinic = {"tested_for_90": partial(share_tested_at_sensitivity, sens=sens),
              "precision_at_90": partial(precision_at_sensitivity, sens=sens)}
    rank_metrics = {**survey_metrics(survey, probabilities=False), **clinic}
    ci = lambda yy, s, g=S.groups: summarize(yy, s, boot["n_resamples"], boot["ci"], seed, g, rank_metrics)

    # 1. Hb regressor on three question sets.
    sets = {k: question_set(v, S) for k, v in hc["question_sets"].items()}
    pred = {k: hb_predictions(k, q, S) for k, q in sets.items()}
    print("Hb predictions ready: " + ", ".join(f"{k} ({len(q)} questions)" for k, q in sets.items()), flush=True)

    # Comparators, any anemia: Step 8a's cache. Moderate or severe: refit on the same folds (as in Step 9).
    any_scores = {"region": np.load(CACHE / "region.npy"), "lr_base": np.load(CACHE / "lr__base.npy"),
                  **{f"hb_{k}": S.cutoff - p for k, p in pred.items()}}
    X_base = prepare(S.feats[columns_for(S.base, S.feats)])
    sev_region, sev_lr = (np.full((S.rc["cv_repeats"], len(sev)), np.nan) for _ in range(2))
    for r, tr, te in S.folds:
        sev_region[r, te] = group_rate(area_codes(S.feats, False), sev, tr, te, sm)
        sev_lr[r, te] = make_model("logistic_regression", X_base, S.mc, seed).fit(
            X_base.iloc[tr], sev[tr]).predict_proba(X_base.iloc[te])[:, 1]
    sev_scores = {"region": sev_region, "lr_base": sev_lr,
                  **{f"hb_{k}": st["hb_cutoff_g_dl"] - p for k, p in pred.items()}}
    names = {"region": "baseline: share in her region (no model)", "lr_base": "base 6, logistic regression",
             "hb_base": "Hb ranking: base 6", "hb_base_region": "Hb ranking: base 6 + region",
             "hb_all": "Hb ranking: all questions"}
    pairs = [("hb_base_region", "hb_base"), ("hb_all", "hb_base_region"), ("hb_all", "hb_base"),
             ("hb_base", "lr_base"), ("hb_base_region", "region")]

    def block(y, scores):
        mean = {k: v.mean(0) for k, v in scores.items()}
        rows = [{"model": names[k], **ci(y, v)} for k, v in scores.items()]
        gains = [{"comparison": f"{names[a]} minus {names[b]}",
                  "auroc": paired_gain(y, mean[a], mean[b], S.groups, boot, seed)} for a, b in pairs]
        return rows, gains

    any_rows, any_gains = block(S.y, any_scores)
    sev_rows, sev_gains = block(sev, sev_scores)
    print("individual metrics done", flush=True)
    village_keys = ["region", "lr_base", "hb_base", "hb_base_region", "hb_all"]
    village = boot_area({k: any_scores[k].mean(0) for k in village_keys}, S.y, S.groups,
                        S.feats.v024.astype(str).to_numpy(), cfg["review"]["area_top_share"],
                        boot["n_resamples"], boot["ci"], seed, reference="hb_base_region")
    print("village metrics done", flush=True)

    # 2. Which questions.
    imp = importance(S, hc)
    inv = pd.read_csv(ROOT / "reports" / "zimbabwe_inventory.csv").set_index("code").label
    order = imp.mean().sort_values(ascending=False)
    top = [{"code": c, "label": inv.get(c, ""), "mae_rise": float(order[c]), "sd": float(imp[c].std()),
            "in_base": c in S.base, "folds_in_top": int((imp.rank(axis=1, ascending=False)[c] <= hc["importance_top"]).sum())}
           for c in order.index[:hc["importance_top"]]]

    # 3. Clinic number on the locked test, models already scored there (Steps 6 and 9).
    tr, te = np.flatnonzero(~locked), np.flatnonzero(locked)
    ffeats, fg = full.drop(columns=TARGETS), full.cluster.to_numpy()[te]
    Xb, Xa = prepare(ffeats[columns_for(S.base, ffeats)]), prepare(ffeats[columns_for(S.questions, ffeats)])
    hb_hat = make_model("hgb_regressor", Xa, S.mc, seed).fit(Xa.iloc[tr], full.hb.to_numpy()[tr]).predict(Xa.iloc[te])
    y_any = full.anemic.to_numpy()
    locked_rows = []
    for target, yy, cut in (("any anemia", y_any, full.hb_cutoff.to_numpy()[te]),
                            ("moderate or severe", sev_full, st["hb_cutoff_g_dl"])):
        s = {"region": group_rate(area_codes(ffeats, False), yy, tr, te, sm),
             "lr_base": make_model("logistic_regression", Xb, S.mc, seed).fit(Xb.iloc[tr], yy[tr]).predict_proba(
                 Xb.iloc[te])[:, 1],
             "hb_all": cut - hb_hat}
        for k, v in s.items():
            m = summarize(yy[te], v, boot["n_resamples"], boot["ci"], seed, fg, clinic)
            locked_rows.append({"target": target, "model": names[k], **m,
                                "auroc_check": float(auroc(yy[te], v))})

    # The locked refits must reproduce the AUROCs already reported in Steps 6 and 9 (frozen models).
    import json
    step6 = json.loads((ROOT / "results" / "zimbabwe_final_test.json").read_text())["results"]["rows"]
    step9 = json.loads((ROOT / "results" / "zimbabwe_severity.json").read_text())["results"]["locked_test"]
    known = {("any anemia", names["lr_base"]): next(r for r in step6 if r["model"] == "base 6 questions"),
             ("any anemia", names["hb_all"]): next(r for r in step6 if r["model"].startswith("Hb ranking")),
             ("moderate or severe", names["lr_base"]): next(r for r in step9 if r["model"].startswith("base 6")),
             ("moderate or severe", names["hb_all"]): next(r for r in step9 if r["model"].startswith("Hb ranking")),
             ("moderate or severe", names["region"]): next(r for r in step9 if r["model"].startswith("baseline: share in her region (no"))}
    for r in locked_rows:
        if (r["target"], r["model"]) in known:
            assert abs(r["auroc_check"] - known[(r["target"], r["model"])]["auroc"]["estimate"]) < 1e-9, r["model"]

    path = save_results("zimbabwe_hb_sources", {
        "question_sets": {k: len(q) for k, q in sets.items()}, "any_anemia": any_rows, "any_gains": any_gains,
        "moderate_or_severe": sev_rows, "severe_gains": sev_gains, "village": village, "importance_top": top,
        "importance_folds": imp.to_dict(), "locked_clinic": locked_rows},
        {"default": S.base_cfg, "feature_selection": cfg, "survey": survey})

    cols = {"auroc": "AUROC", "recall_top": "Recall, top 30% tested", "tested_for_90": "Share tested to catch 90%",
            "precision_at_90": "Share of tested who have it"}
    g = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    c = lambda d: f"{d['estimate']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    vrow = lambda k: (f"| {names[k]} | {c(village[k]['spearman'])} | "
                      + (g(village[k]['spearman_minus_hb_base_region']) if k != "hb_base_region" else "-")
                      + f" | {c(village[k]['top_overlap'])} | "
                      + (g(village[k]['top_overlap_minus_hb_base_region']) if k != "hb_base_region" else "-") + " |")
    text = "\n".join([
        "# Stage 29 Step 8d: where the Hb ranking's gain comes from, and the clinic number",
        "",
        f"Generated by `python -m src.survey.hb_sources` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`hb_sources`). Building villages, the same 5 x 5 repeated village folds as "
        f"Step 8a; village-bootstrap 95% CIs and paired differences, {boot['n_resamples']} resamples.",
        "",
        "- **Primary target: any anemia**, DHS `v457` 1-3: Hb adjusted for altitude and smoking < 12.0 g/dL, or "
        "< 11.0 g/dL if pregnant (WHO).",
        "- **Second target: moderate or severe**, DHS `v457` 1-2: adjusted Hb **< 10.0 g/dL for every woman, pregnant "
        "or not** (DHS: moderate 7.0-9.9, severe < 7.0). For pregnant women this matches WHO's moderate cutoff; for "
        "non-pregnant women it is stricter than WHO's < 11.0. Pre-registered in Step 9.",
        "- **Hb ranking:** a gradient boosting regressor predicts adjusted Hb; women are ranked by her cutoff minus the "
        "prediction (12.0/11.0 for any anemia, 10.0 for moderate or severe). Same model settings for every question "
        f"set: base 6 ({len(sets['base'])} questions), base 6 + region `v024` ({len(sets['base_region'])}), all "
        f"({len(sets['all'])}).",
        f"- **Clinic number:** testing women from the highest score down, the share of all women who must be tested to "
        f"catch {sens:.0%} of cases, and the share of those tested who have the outcome. Random testing needs "
        f"{sens:.0%}.",
        "",
        "## 1. Any anemia (primary)",
        "",
        markdown_table(any_rows, ["model"], cols),
        "",
        markdown_table(any_gains, ["comparison"], {"auroc": "AUROC difference (paired)"}),
        "",
        "## 2. Moderate or severe",
        "",
        markdown_table(sev_rows, ["model"], cols),
        "",
        markdown_table(sev_gains, ["comparison"], {"auroc": "AUROC difference (paired)"}),
        "",
        "## 3. Ranking villages (any anemia)",
        "",
        "Village score = mean score of its women, against its observed anemia share. Within-region is not shown: in "
        "cross-validation it is pulled down for anything that uses region (Step 8b).",
        "",
        "| model | Spearman, all villages | minus base 6 + region | top-20% overlap | minus base 6 + region |",
        "|---|---|---|---|---|",
        *[vrow(k) for k in village_keys],
        "",
        "## 4. Which questions the all-questions Hb regressor leans on",
        "",
        f"Rise in Hb error (MAE, g/dL) when a question is shuffled, on the {len(imp)} held-out folds of the first "
        f"repeat (mean and SD over folds). Questions that share information (region and its proxies, two age "
        "variables) split the credit, so each looks less important than the information they carry together.",
        "",
        f"| # | code | question | MAE rise (g/dL) | SD | in base 6 | folds in top {hc['importance_top']} |",
        "|---|---|---|---|---|---|---|",
        *[f"| {i + 1} | `{r['code']}` | {r['label']} | {r['mae_rise']:.4f} | {r['sd']:.4f} | "
          f"{'yes' if r['in_base'] else ''} | {r['folds_in_top']} |" for i, r in enumerate(top)],
        "",
        "## 5. Clinic number on the locked test (descriptive)",
        "",
        f"{len(set(fg))} villages. Only models already scored on the locked test (Steps 6 and 9); refit unchanged "
        "on the building villages. New metrics on frozen models; no choice depends on them.",
        "",
        markdown_table(locked_rows, ["target", "model"], {"tested_for_90": "Share tested to catch 90%",
                                                          "precision_at_90": "Share of tested who have it"}),
        "",
    ])
    (ROOT / "reports" / "zimbabwe_hb_sources.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
