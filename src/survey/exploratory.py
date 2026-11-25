"""Stage 29 Step 10: four exploratory variants, pre-registered in configs/feature_selection.yaml (`exploratory`).

Role: exploratory, Stage 29. Pre-registered variants; none is a headline result.

Cross-validation only, on the building villages, Step 8a's 5 x 5 village folds. The locked test is not used. Every
variant is reported, whether it helps or not; "success" = paired AUROC gain >= min_gain over its reference with the 95%
CI above 0.
  A. Hb ranking and TabPFN classification combined (ranks averaged), any anemia.
  B. Hb with its uncertainty: P(Hb below her own cutoff) from the regressor's mean and a modelled spread.
  C. Short model for moderate or severe: base 6 + region, logistic and ordinal (proportional odds).
  D. Village averages (wealth, education, modern contraception) added to the all-questions Hb regressor.

Run from the repo root: python -m src.survey.exploratory   (after Steps 8a, 8d and 9)
"""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import brier, calibration_intercept, calibration_slope
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.survey.area_check import boot_area
from src.common.evaluate import paired_gain
from src.survey.repeated_cv import CACHE, Setup
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, make_model, prepare
from src.survey.severity import severity_label
from src.survey.train_baselines import survey_metrics


def severity_index(hb, cutoff):
    """DHS levels as 0 not anemic, 1 mild, 2 moderate, 3 severe (severe < 7.0, moderate < 10.0, mild < her cutoff)."""
    return np.select([hb < 7.0, hb < 10.0, hb < cutoff], [3, 2, 1], 0)


def ordinal_prob(X, level, train, test, mc, seed, at_least):
    """Proportional-odds logistic regression fitted by replicating each training row once per threshold k (target:
    level >= k), with shared slopes and one intercept per threshold. Returns P(level >= at_least) for the test rows."""
    prep = make_model("logistic_regression", X, mc, seed).named_steps["prep"]
    A_tr, A_te = prep.fit_transform(X.iloc[train]), prep.transform(X.iloc[test])
    A_tr, A_te = sparse.csr_matrix(A_tr), sparse.csr_matrix(A_te)
    ks = range(1, int(level.max()) + 1)
    rows, targets = [], []
    for j, k in enumerate(ks):
        onehot = sparse.csr_matrix((np.ones(A_tr.shape[0]), (np.arange(A_tr.shape[0]), np.full(A_tr.shape[0], j))),
                                   shape=(A_tr.shape[0], len(ks)))
        rows.append(sparse.hstack([A_tr, onehot]))
        targets.append((level[train] >= k).astype(int))
    model = LogisticRegression(C=mc["logistic_regression"]["C"], max_iter=mc["logistic_regression"]["max_iter"],
                               fit_intercept=False).fit(sparse.vstack(rows), np.concatenate(targets))
    j = list(ks).index(at_least)
    onehot = sparse.csr_matrix((np.ones(A_te.shape[0]), (np.arange(A_te.shape[0]), np.full(A_te.shape[0], j))),
                               shape=(A_te.shape[0], len(ks)))
    return model.predict_proba(sparse.hstack([A_te, onehot]))[:, 1]


def village_means(feats, groups, spec):
    """Per-village averages of answers (never Hb), one value per woman from her own village's women."""
    out = {}
    for code, how in spec.items():
        x = feats[code].astype(float) if how == "mean" else feats[code].isin(how).astype(float).where(feats[code].notna())
        out[f"village_{code}"] = pd.Series(x.to_numpy()).groupby(groups).transform("mean").to_numpy()
    return pd.DataFrame(out, index=feats.index)


def spread_probabilities(S, X, mean_pred, cutoffs, inner_splits):
    """P(Hb < cutoff) with a modelled spread and with a constant spread, out of fold. cutoffs: {target: array}."""
    out = {f"{t}_{kind}": np.full((S.rc["cv_repeats"], len(S.y)), np.nan) for t in cutoffs for kind in ("model", "const")}
    for i, (r, tr, te) in enumerate(S.folds):
        resid = np.full(len(tr), np.nan)
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=S.seed)
        for itr, iva in inner.split(np.zeros(len(tr)), S.y[tr], S.groups[tr]):
            m = make_model("hgb_regressor", X, S.mc, S.seed).fit(X.iloc[tr[itr]], S.hb[tr[itr]])
            resid[iva] = S.hb[tr[iva]] - m.predict(X.iloc[tr[iva]])
        scale = make_model("hgb_regressor", X, S.mc, S.seed).fit(X.iloc[tr], np.abs(resid))
        sigma = np.clip(np.sqrt(np.pi / 2) * scale.predict(X.iloc[te]), 0.1, None)
        const = float(np.std(resid))
        for t, cut in cutoffs.items():
            c = cut[te] if np.ndim(cut) else cut
            out[f"{t}_model"][r, te] = norm.cdf((c - mean_pred[r, te]) / sigma)
            out[f"{t}_const"][r, te] = norm.cdf((c - mean_pred[r, te]) / const)
        print(f"spread fold {i + 1}/{len(S.folds)}", flush=True)
    return out


def tabpfn_hb(S):
    """E: TabPFN regressor's out-of-fold predicted Hb on all questions, (n_repeats, n), cached."""
    path = CACHE / "tabpfn_hb__all.npy"
    if path.exists():
        return np.load(path)
    from tabpfn import TabPFNRegressor
    from src.survey.tabpfn_cv import tabpfn_matrix
    folds_dir = CACHE / "tabpfn_hb_folds"                 # one file per fold, so a stopped run resumes
    folds_dir.mkdir(exist_ok=True)
    out = np.full((S.rc["cv_repeats"], len(S.y)), np.nan)
    for i, (r, tr, te) in enumerate(S.folds):
        f = folds_dir / f"fold{i}.npy"
        if not f.exists():
            A, cat_idx = tabpfn_matrix(S.X_all, S.X_all.index[tr])
            model = TabPFNRegressor(random_state=S.seed, categorical_features_indices=cat_idx,
                                    ignore_pretraining_limits=S.cfg["tabpfn"]["ignore_pretraining_limits"])
            np.save(f, model.fit(A[tr], S.hb[tr]).predict(A[te]))
        out[r, te] = np.load(f)
        print(f"TabPFN Hb fold {i + 1}/{len(S.folds)}", flush=True)
    np.save(path, out)
    return out


def main():
    seed = load_config("default")["seed"]
    S = Setup(seed)
    cfg, boot, ex, st = S.cfg, S.base_cfg["bootstrap"], S.cfg["exploratory"], S.cfg["severity_target"]
    survey = load_config("survey")
    full = pd.read_parquet(OUT)
    sev = severity_label(full, st, survey)[~full.locked_test.to_numpy()]
    level = severity_index(S.hb, S.cutoff)
    assert ((level >= 2) == sev).all() and ((level >= 1) == S.y).all()
    n_rep = S.rc["cv_repeats"]
    hb_mean = S.cutoff - np.load(CACHE / "hb__all.npy")
    prob_metrics = {**survey_metrics(survey), "cal_slope": calibration_slope, "cal_intercept": calibration_intercept}
    rank_metrics = survey_metrics(survey, probabilities=False)
    stat = lambda y, s, m: summarize(y, s, boot["n_resamples"], boot["ci"], seed, S.groups, m)
    gain = lambda y, a, b: paired_gain(y, a.mean(0) if a.ndim > 1 else a, b.mean(0) if b.ndim > 1 else b,
                                       S.groups, boot, seed)
    ok = lambda g: bool(g["estimate"] >= ex["min_gain"] and g["lo"] > 0)
    refs = {"any anemia": {"hb": S.cutoff - hb_mean, "lr_base": np.load(CACHE / "lr__base.npy")},
            "moderate or severe": {"hb": st["hb_cutoff_g_dl"] - hb_mean}}
    X_base = prepare(S.feats[columns_for(S.base, S.feats)])
    lr_sev = np.full((n_rep, len(S.y)), np.nan)
    for r, tr, te in S.folds:
        lr_sev[r, te] = make_model("logistic_regression", X_base, S.mc, seed).fit(
            X_base.iloc[tr], sev[tr]).predict_proba(X_base.iloc[te])[:, 1]
    refs["moderate or severe"]["lr_base"] = lr_sev
    ys = {"any anemia": S.y, "moderate or severe": sev}
    rows, verdicts = [], []

    def add(target, name, score, metrics, ref_key, ref_name):
        y = ys[target]
        g = gain(y, score, refs[target][ref_key])
        rows.append({"target": target, "variant": name, **stat(y, score, metrics), "gain": g, "reference": ref_name})
        verdicts.append({"target": target, "variant": name, "reference": ref_name, "gain": g, "success": ok(g)})
        print(f"{target} | {name}: AUROC {rows[-1]['auroc']['estimate']:.3f}, vs {ref_name} "
              f"{g['estimate']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]", flush=True)

    # References, for the table.
    for t in ys:
        rows.append({"target": t, "variant": "reference: base 6, logistic regression",
                     **stat(ys[t], refs[t]["lr_base"], prob_metrics)})
        rows.append({"target": t, "variant": "reference: Hb ranking, all questions",
                     **stat(ys[t], refs[t]["hb"], rank_metrics)})

    # A. Combine.
    tab = np.load(CACHE / "tabpfn__all.npy").mean(0)
    hb_rank = refs["any anemia"]["hb"].mean(0)
    combined = (rankdata(hb_rank) + rankdata(tab)) / (2 * len(tab))
    add("any anemia", "A. Hb ranking + TabPFN, ranks averaged", combined, rank_metrics, "hb", "Hb ranking")

    # B. Hb with its uncertainty.
    path_b = CACHE / "spread.npz"
    if path_b.exists():
        spread = dict(np.load(path_b))
    else:
        spread = spread_probabilities(S, S.X_all, hb_mean, {"any anemia": S.cutoff,
                                                            "moderate or severe": st["hb_cutoff_g_dl"]},
                                      ex["spread_inner_splits"])
        np.savez(path_b, **spread)
    for t in ys:
        add(t, "B. P(Hb below her cutoff), modelled spread", spread[f"{t}_model"], prob_metrics, "hb", "Hb ranking")
        add(t, "B. P(Hb below her cutoff), constant spread", spread[f"{t}_const"], prob_metrics, "hb", "Hb ranking")

    # C. Short model: base 6 + region.
    short_q = S.base + ex["short_questions_extra"]
    X_short = prepare(S.feats[columns_for(short_q, S.feats)])
    lr_short = {t: np.full((n_rep, len(S.y)), np.nan) for t in ys}
    ordinal = np.full((n_rep, len(S.y)), np.nan)
    for r, tr, te in S.folds:
        for t, y in ys.items():
            lr_short[t][r, te] = make_model("logistic_regression", X_short, S.mc, seed).fit(
                X_short.iloc[tr], y[tr]).predict_proba(X_short.iloc[te])[:, 1]
        ordinal[r, te] = ordinal_prob(X_short, level, tr, te, S.mc, seed, at_least=2)
    for t in ys:
        add(t, "C. base 6 + region, logistic regression", lr_short[t], prob_metrics, "lr_base", "base 6 LR")
    add("moderate or severe", "C. base 6 + region, ordinal (4 DHS levels)", ordinal, prob_metrics, "lr_base",
        "base 6 LR")
    for name, s in (("C. base 6 + region, logistic regression", lr_short["moderate or severe"]),
                    ("C. base 6 + region, ordinal (4 DHS levels)", ordinal)):
        verdicts.append({"target": "moderate or severe", "variant": name, "reference": "Hb ranking, all questions",
                         "gain": gain(sev, s, refs["moderate or severe"]["hb"]), "success": None})

    # D. Village averages added to the all-questions Hb regressor.
    path_d = CACHE / "hb_village_means.npy"
    if path_d.exists():
        hb_vm = np.load(path_d)
    else:
        X_vm = pd.concat([S.X_all, village_means(S.feats, S.groups, ex["village_means"])], axis=1)
        hb_vm = np.full((n_rep, len(S.y)), np.nan)
        for i, (r, tr, te) in enumerate(S.folds):
            hb_vm[r, te] = make_model("hgb_regressor", X_vm, S.mc, seed).fit(X_vm.iloc[tr], S.hb[tr]).predict(
                X_vm.iloc[te])
            print(f"village means fold {i + 1}/{len(S.folds)}", flush=True)
        np.save(path_d, hb_vm)
    add("any anemia", "D. Hb ranking + village averages", S.cutoff - hb_vm, rank_metrics, "hb", "Hb ranking")
    add("moderate or severe", "D. Hb ranking + village averages", st["hb_cutoff_g_dl"] - hb_vm, rank_metrics, "hb",
        "Hb ranking")
    village = boot_area({"hb": refs["any anemia"]["hb"].mean(0), "hb_vm": (S.cutoff - hb_vm).mean(0)}, S.y,
                        S.groups, S.feats.v024.astype(str).to_numpy(), cfg["review"]["area_top_share"],
                        boot["n_resamples"], boot["ci"], seed, reference="hb")

    # E. TabPFN predicting Hb (only when its cache exists; run tabpfn_hb first, it takes about an hour on a GPU).
    if ex.get("tabpfn_hb") and (CACHE / "tabpfn_hb__all.npy").exists():
        tp = tabpfn_hb(S)
        add("any anemia", "E. TabPFN Hb ranking", S.cutoff - tp, rank_metrics, "hb", "Hb ranking")
        add("moderate or severe", "E. TabPFN Hb ranking", st["hb_cutoff_g_dl"] - tp, rank_metrics, "hb", "Hb ranking")

    path = save_results("zimbabwe_exploratory", {"rows": rows, "verdicts": verdicts, "village_D": village},
                        {"default": S.base_cfg, "feature_selection": cfg, "survey": survey})
    cols = {"auroc": "AUROC", "pr_auc": "PR-AUC", "recall_top": "Recall, top 30%", "brier": "Brier",
            "cal_slope": "Calibration slope", "cal_intercept": "Calibration intercept"}
    g = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    c = lambda d: f"{d['estimate']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    verdict = lambda v: "-" if v["success"] is None else ("**yes**" if v["success"] else "no")
    text = "\n".join([
        "# Stage 29 Step 10: four exploratory variants (pre-registered, cross-validation only)",
        "",
        f"Generated by `python -m src.survey.exploratory` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`exploratory`), written before running. **Exploratory, after the main "
        "analysis: cross-validation on the building villages only (Step 8a's 5 x 5 folds); the locked test is not "
        "used.** Confirming any of these needs a new survey. Village-bootstrap 95% CIs, "
        f"{boot['n_resamples']} resamples.",
        "",
        "- **Any anemia (primary):** DHS `v457` 1-3, adjusted Hb < 12.0 g/dL, or < 11.0 if pregnant (WHO). "
        "**Moderate or severe:** DHS `v457` 1-2, adjusted Hb < 10.0 g/dL for every woman, pregnant or not.",
        f"- **Success** = paired AUROC gain >= {ex['min_gain']} over the stated reference with the 95% CI above 0.",
        "",
        "## Verdicts",
        "",
        "| target | variant | reference | AUROC gain (paired) | success |",
        "|---|---|---|---|---|",
        *[f"| {v['target']} | {v['variant']} | {v['reference']} | {g(v['gain'])} | {verdict(v)} |" for v in verdicts],
        "",
        "## All metrics",
        "",
        "Rankings (the Hb ranking, A, D) have no probability, so no Brier or calibration.",
        "",
        markdown_table(rows, ["target", "variant"], cols),
        "",
        "## D. Village ranking (any anemia)",
        "",
        f"- Hb ranking: Spearman {c(village['hb']['spearman'])}, top-20% overlap {c(village['hb']['top_overlap'])}.",
        f"- With village averages: Spearman {c(village['hb_vm']['spearman'])} "
        f"({g(village['hb_vm']['spearman_minus_hb'])}), top-20% overlap {c(village['hb_vm']['top_overlap'])} "
        f"({g(village['hb_vm']['top_overlap_minus_hb'])}).",
        "",
        "## What each variant is",
        "",
        "- **A.** Each woman's rank under the Hb ranking and under TabPFN classification (all questions), averaged "
        "50/50.",
        "- **B.** The Hb regressor's mean plus a second regressor predicting the size of its error (fitted on "
        "out-of-fold errors inside each training fold); P(anemic) = P(Hb below her own cutoff) under a normal error. "
        "The constant-spread version uses one error size for everyone.",
        "- **C.** Base 6 + region (`v024`): logistic regression, and an ordinal logistic regression on the four DHS "
        "levels (proportional odds) scored as P(moderate or severe).",
        "- **D.** The all-questions Hb regressor plus three village averages of answers: mean wealth score (`v191`), "
        "share with secondary or higher education (`v106`), share using a modern method (`v313`).",
        "- **E.** TabPFN 3.5 predicting the Hb number from all questions, ranked by her cutoff minus the prediction. "
        "Run because Step 8e showed the Hb target beats yes/no for gradient boosting (the condition stated before "
        "Step 8e's results).",
        "",
    ])
    (ROOT / "reports" / "zimbabwe_exploratory.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
