"""Stage 29 Step 9: the pre-registered second target, moderate or severe anemia (DHS v457 codes 1-2).

Role: main result, Stage 29 (the second target, moderate or severe anemia).

Fixed in configs/feature_selection.yaml (`severity_target`) on 2026-11-16 before any model was run on it. DHS
convention: Hb adjusted for altitude and smoking < 10.0 g/dL for every woman, pregnant or not (moderate 7.0-9.9,
severe < 7.0). A second analysis next to the main any-anemia result, not a replacement.

Models (fixed in advance, no new feature selection): dummy; "pregnant first" rule; region and region x urban/rural
baselines (no model); base 6 logistic regression; all-questions gradient boosting; Hb ranking (Step 8a's Hb
regressor, score = 10.0 - predicted Hb). 5 x 5 repeated village CV on the building villages (the same folds as Step
8a), then every model scored once on the locked test; no choice depends on the locked test.

Run from the repo root: python -m src.survey.severity   (after python -m src.survey.repeated_cv)
"""
import numpy as np
import pandas as pd
import pyreadstat

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.common.evaluate import paired_gain
from src.survey.repeated_cv import CACHE, Setup, area_codes, group_rate
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, make_model, prepare
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics

LABELS = {
    "dummy": "dummy (everyone the same risk)",
    "rule": "rule: pregnant first, then random",
    "region": "baseline: share in her region (no model)",
    "region_residence": "baseline: share in her region x urban/rural (no model)",
    "lr_base": "base 6 questions, logistic regression",
    "gb_all": "all questions, gradient boosting",
    "hb_rank": "Hb ranking: 10.0 minus predicted Hb, all questions",
}
PROBA = {"dummy", "region", "region_residence", "lr_base", "gb_all"}


def severity_label(full, st, survey):
    """Moderate-or-severe from the saved adjusted Hb, checked against DHS's own v457 for every woman."""
    y = (full.hb < st["hb_cutoff_g_dl"]).astype(int).to_numpy()
    raw, _ = pyreadstat.read_dta(ROOT / survey["countries"]["zimbabwe"], usecols=[st["label"], "v001"])
    raw = raw[raw[st["label"]].notna()].reset_index(drop=True)
    assert (raw.v001.astype(int).to_numpy() == full.cluster.to_numpy()).all()
    assert (raw[st["label"]].isin(st["levels"]).astype(int).to_numpy() == y).all(), "Hb < 10.0 does not match v457"
    return y


def main():
    seed = load_config("default")["seed"]
    S = Setup(seed)
    cfg, boot, st = S.cfg, S.base_cfg["bootstrap"], S.cfg["severity_target"]
    survey = load_config("survey")
    full = pd.read_parquet(OUT)
    y_full = severity_label(full, st, survey)
    y = y_full[~full.locked_test.to_numpy()]
    groups, sm = S.groups, cfg["screening"]["smoothing"]
    print(f"moderate or severe: {y_full.sum()} women ({y_full.mean():.1%}); building {y.sum()}, locked "
          f"{y_full[full.locked_test].sum()}", flush=True)

    # Cross-validation on the building villages, Step 8a's folds.
    hb_pred = S.cutoff - np.load(CACHE / "hb__all.npy")             # Step 8a's out-of-fold Hb predictions
    X_base = prepare(S.feats[columns_for(S.base, S.feats)])
    X_all = S.X_all
    n_rep = S.rc["cv_repeats"]
    cv = {k: np.full((n_rep, len(y)), np.nan) for k in LABELS}
    for i, (r, tr, te) in enumerate(S.folds):
        cv["dummy"][r, te] = y[tr].mean()
        cv["rule"][r, te] = S.feats.v213.eq("yes").to_numpy()[te] + 0.5 * np.random.default_rng(seed + i).random(len(te))
        cv["region"][r, te] = group_rate(area_codes(S.feats, False), y, tr, te, sm)
        cv["region_residence"][r, te] = group_rate(area_codes(S.feats, True), y, tr, te, sm)
        cv["lr_base"][r, te] = make_model("logistic_regression", X_base, S.mc, seed).fit(
            X_base.iloc[tr], y[tr]).predict_proba(X_base.iloc[te])[:, 1]
        cv["gb_all"][r, te] = make_model("hist_gradient_boosting", X_all, S.mc, seed).fit(
            X_all.iloc[tr], y[tr]).predict_proba(X_all.iloc[te])[:, 1]
        cv["hb_rank"][r, te] = st["hb_cutoff_g_dl"] - hb_pred[r, te]
        print(f"fold {i + 1}/{len(S.folds)}", flush=True)
    rows = []
    for k, label in LABELS.items():
        m = summarize(y, cv[k], boot["n_resamples"], boot["ci"], seed, groups, survey_metrics(survey, k in PROBA))
        rows.append({"model": label, **m,
                     "gain_vs_base6": paired_gain(y, cv[k].mean(0), cv["lr_base"].mean(0), groups, boot, seed),
                     "gain_vs_region": paired_gain(y, cv[k].mean(0), cv["region"].mean(0), groups, boot, seed)})
        print(f"CV {label}: AUROC {m['auroc']['estimate']:.3f}", flush=True)

    # Locked test: every model above, fit on all building women, scored once.
    test = full.locked_test.to_numpy()
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    ffeats, fgroups = full.drop(columns=TARGETS), full.cluster.to_numpy()
    Xb, Xa = prepare(ffeats[columns_for(S.base, ffeats)]), prepare(ffeats[columns_for(S.questions, ffeats)])
    hb_locked = make_model("hgb_regressor", Xa, S.mc, seed).fit(Xa.iloc[tr], full.hb.to_numpy()[tr]).predict(Xa.iloc[te])
    locked = {
        "dummy": np.full(len(te), y_full[tr].mean()),
        "rule": ffeats.v213.eq("yes").to_numpy()[te] + 0.5 * np.random.default_rng(seed).random(len(te)),
        "region": group_rate(area_codes(ffeats, False), y_full, tr, te, sm),
        "region_residence": group_rate(area_codes(ffeats, True), y_full, tr, te, sm),
        "lr_base": make_model("logistic_regression", Xb, S.mc, seed).fit(Xb.iloc[tr], y_full[tr]).predict_proba(
            Xb.iloc[te])[:, 1],
        "gb_all": make_model("hist_gradient_boosting", Xa, S.mc, seed).fit(Xa.iloc[tr], y_full[tr]).predict_proba(
            Xa.iloc[te])[:, 1],
        "hb_rank": st["hb_cutoff_g_dl"] - hb_locked,
    }
    ly, lg = y_full[te], fgroups[te]
    lrows = [{"model": LABELS[k], **summarize(ly, s, boot["n_resamples"], boot["ci"], seed, lg,
                                              survey_metrics(survey, k in PROBA)),
              "gain_vs_base6": paired_gain(ly, s, locked["lr_base"], lg, boot, seed),
              "gain_vs_region": paired_gain(ly, s, locked["region"], lg, boot, seed)} for k, s in locked.items()]

    path = save_results("stage29_severity", {"n_cases": int(y_full.sum()), "share": float(y_full.mean()),
                                             "cv": rows, "locked_test": lrows},
                        {"default": S.base_cfg, "feature_selection": cfg, "survey": survey})
    gains = lambda rs: markdown_table([{"model": r["model"], "a": r["gain_vs_base6"], "b": r["gain_vs_region"]}
                                       for r in rs], ["model"], {"a": "AUROC minus base 6", "b": "AUROC minus region"})
    text = "\n".join([
        "# Stage 29 Step 9: moderate or severe anemia (pre-registered second target)",
        "",
        f"Generated by `python -m src.survey.severity` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`severity_target`), fixed on 2026-11-16 before any model was run on it.",
        "",
        "- **Target: DHS `v457` codes 1-2**, moderate or severe: Hb adjusted for altitude and smoking **< 10.0 g/dL "
        "for every woman, pregnant or not** (DHS convention: moderate 7.0-9.9, severe < 7.0; checked to reproduce "
        "`v457` for all women). Not the WHO 2011 cutoffs. The main analysis (any anemia, < 12.0 g/dL or < 11.0 if "
        "pregnant) is unchanged; this is a second analysis next to it.",
        f"- **{int(y_full.sum())} women ({y_full.mean():.1%})**: {int(y.sum())} in the building villages, "
        f"{int(ly.sum())} in the locked test. At this rate a PR-AUC of {y_full.mean():.2f} is chance (the dummy), and "
        "testing a random 30% finds 30% of cases.",
        "- Models fixed in advance, no new feature selection. The Hb ranking scores women by how far their predicted "
        "Hb falls below 10.0 g/dL (Step 8a's Hb regressor). Rankings have no probability, so no Brier score.",
        f"- 5 x 5 repeated village CV on the building villages (Step 8a's folds); each woman's score is the average "
        f"over repeats. Village-bootstrap 95% CIs, {boot['n_resamples']} resamples.",
        "",
        "## Cross-validation (building villages)",
        "",
        markdown_table(rows, ["model"], SURVEY_COLUMNS),
        "",
        gains(rows),
        "",
        f"## Locked test ({len(set(lg))} villages, {int(ly.sum())} cases), every model scored once",
        "",
        "The locked villages were already used for the any-anemia target (Steps 6 and 7); this is the first time they "
        "are scored for this target.",
        "",
        markdown_table(lrows, ["model"], SURVEY_COLUMNS),
        "",
        gains(lrows),
        "",
    ])
    (ROOT / "reports" / "stage29_severity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
