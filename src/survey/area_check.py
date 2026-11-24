"""Stage 29 Step 8b: does the model rank villages and regions, or only repeat what region already says?

Role: supporting check, Stage 29 (village and region ranking).

Each village gets the mean score of its women; that is compared with the village's observed anemia share.
  - Spearman correlation over all villages, and within regions (average of the per-region correlations, weighted by
    villages), because the region baseline captures all the between-region differences by construction.
  - Top-20% overlap: of the 20% of villages with the highest observed share, the share the model's top 20% catches.
  - Noise ceiling: each village is split into random halves many times; the two halves' shares are correlated and the
    Spearman-Brown correction r = 2h / (1 + h) gives the reliability of a full village's observed share. A perfect
    model's correlation with the observed share can be at most sqrt(reliability).
  - Regions (10): one Spearman over regions, reported but weak.

Main analysis: the 5 x 5 repeated-CV out-of-fold scores on the building villages (Step 8a cache). Descriptive only: the
locked test (79 villages) with the Step 6 models refit unchanged on the building villages; no choice depends on it.
Run from the repo root: python -m src.survey.area_check   (after python -m src.survey.repeated_cv)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.survey.repeated_cv import CACHE, MODELS, area_codes, group_rate
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import columns_for, importance_ranking, make_model, prepare
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics

AREA_MODELS = ["region", "region_residence", "lr__base", "lr__top_10", "lr__all", "hb__all",
               "tabpfn__base", "tabpfn__top_10", "tabpfn__all"]


def village_table(score, y, village, region):
    df = pd.DataFrame({"score": score, "y": y, "village": village, "region": region})
    return df.groupby("village").agg(score=("score", "mean"), share=("y", "mean"), n=("y", "size"),
                                     region=("region", "first")).reset_index()


def _spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0                                     # a constant ranking orders nothing
    return float(spearmanr(a, b).statistic)


def within_region(v, min_villages=3):
    """Per-region Spearman of village score vs. observed share, averaged with weights = villages in the region."""
    rs, ws = [], []
    for _, g in v.groupby("region"):
        if len(g) >= min_villages:
            rs.append(_spearman(g.score.to_numpy(), g.share.to_numpy()))
            ws.append(len(g))
    return float(np.average(rs, weights=ws))


def top_overlap(v, share):
    """Of the top `share` of villages by observed anemia share, the fraction also in the model's top `share`.
    Ties are broken by village id so the result is deterministic."""
    k = max(1, int(round(share * len(v))))
    by_obs = set(v.sort_values(["share", "village"], ascending=[False, True]).village[:k])
    by_model = set(v.sort_values(["score", "village"], ascending=[False, True]).village[:k])
    return len(by_obs & by_model) / k


def area_stats(v, top_share):
    return {"spearman": _spearman(v.score.to_numpy(), v.share.to_numpy()), "within_region": within_region(v),
            "top_overlap": top_overlap(v, top_share)}


def spearman_brown(h):
    return 2 * h / (1 + h)


def noise_ceiling(y, village, region, draws, seed):
    """Reliability of a full village's observed share (split-half, Spearman-Brown corrected), overall and within
    regions, and the most a perfect model could correlate with it (sqrt of the reliability)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "village": village, "region": region})
    overall, within = [], []
    for _ in range(draws):
        half = df.groupby("village").y.transform(lambda s: rng.permutation(len(s)) % 2)
        a = df[half == 0].groupby("village").agg(share=("y", "mean"), region=("region", "first"))
        b = df[half == 1].groupby("village").y.mean().reindex(a.index)
        v = pd.DataFrame({"village": a.index, "score": b.to_numpy(), "share": a.share.to_numpy(),
                          "region": a.region.to_numpy()}).dropna()
        overall.append(_spearman(v.score.to_numpy(), v.share.to_numpy()))
        within.append(within_region(v))
    out = {}
    for name, h in (("overall", np.mean(overall)), ("within_region", np.mean(within))):
        rel = spearman_brown(h)
        out[name] = {"half_village": float(h), "reliability": float(rel), "max_correlation": float(np.sqrt(max(rel, 0)))}
    return out


def boot_area(scores, y, village, region, top_share, n_resamples, ci, seed, reference=None):
    """Village bootstrap of the area statistics (and their difference from a reference model's, same resamples)."""
    tables = {k: village_table(s, y, village, region) for k, s in scores.items()}
    ids = tables[next(iter(tables))].village.to_numpy()
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in tables}
    for _ in range(n_resamples):
        pick = rng.choice(len(ids), len(ids))
        for k, t in tables.items():
            bt = t.iloc[pick].reset_index(drop=True)
            bt["village"] = np.arange(len(bt))         # a village drawn twice counts as two villages
            draws[k].append(area_stats(bt, top_share))
    a = (1 - ci) / 2
    out = {}
    for k, t in tables.items():
        est = area_stats(t, top_share)
        out[k] = {}
        for stat in est:
            vals = np.array([d[stat] for d in draws[k]])
            out[k][stat] = {"estimate": est[stat], "lo": float(np.quantile(vals, a)), "hi": float(np.quantile(vals, 1 - a))}
            if reference and k != reference:
                diff = vals - np.array([d[stat] for d in draws[reference]])
                out[k][f"{stat}_minus_{reference}"] = {
                    "estimate": est[stat] - area_stats(tables[reference], top_share)[stat],
                    "lo": float(np.quantile(diff, a)), "hi": float(np.quantile(diff, 1 - a))}
    return out


def region_level(scores, y, region):
    out = {}
    for k, s in scores.items():
        g = pd.DataFrame({"s": s, "y": y, "r": region}).groupby("r").mean()
        out[k] = _spearman(g.s.to_numpy(), g.y.to_numpy())
    return out


def locked_scores(seed):
    """The Step 6 models refit unchanged on all building women, plus the no-model region baselines, scored on the
    locked villages. Returns (scores by model, y, village, region) for the locked women."""
    cfg = load_config("feature_selection")
    mc = cfg["models"]
    full = pd.read_parquet(OUT)
    feats, y, groups = full.drop(columns=TARGETS), full.anemic.to_numpy(), full.cluster.to_numpy()
    hb, cutoff = full.hb.to_numpy(), full.hb_cutoff.to_numpy()
    test = full.locked_test.to_numpy()
    tr, te = np.flatnonzero(~test), np.flatnonzero(test)
    uni = pd.read_csv(ROOT / "reports" / "stage29_univariate.csv")
    questions = uni.code[uni.step4 == "keep"].tolist()
    base = mc["base_questions"]
    X_all = prepare(feats[columns_for(questions, feats)])
    top10 = importance_ranking(X_all.iloc[tr], y[tr], groups[tr], questions, mc, seed)[:cfg["final_test"]["winner"]["k"]]

    def fit(q, kind, target):
        X = prepare(feats[columns_for(q, feats)])
        m = make_model(kind, X, mc, seed).fit(X.iloc[tr], target[tr])
        return m.predict_proba(X.iloc[te])[:, 1] if kind == "logistic_regression" else m.predict(X.iloc[te])

    sm = cfg["screening"]["smoothing"]
    scores = {"region": group_rate(area_codes(feats, False), y, tr, te, sm),
              "region_residence": group_rate(area_codes(feats, True), y, tr, te, sm),
              "lr__base": fit(base, "logistic_regression", y),
              "lr__top_10": fit(top10, "logistic_regression", y),
              "hb__all": cutoff[te] - fit(questions, "hgb_regressor", hb)}
    return scores, y[te], groups[te], feats.v024.astype(str).to_numpy()[te]


def main():
    base_cfg, cfg, survey = load_config("default"), load_config("feature_selection"), load_config("survey")
    seed, boot, rc = base_cfg["seed"], base_cfg["bootstrap"], cfg["review"]
    data = pd.read_parquet(OUT)
    data = data[~data.locked_test].reset_index(drop=True)
    y, village, region = data.anemic.to_numpy(), data.cluster.to_numpy(), data.v024.astype(str).to_numpy()
    cv = {k: np.load(CACHE / f"{k}.npy").mean(0) for k in AREA_MODELS}
    print(f"building: {len(set(village))} villages; locked models refit next", flush=True)

    ceiling = noise_ceiling(y, village, region, rc["split_half_draws"], seed)
    area = boot_area(cv, y, village, region, rc["area_top_share"], boot["n_resamples"], boot["ci"], seed,
                     reference="region")
    regions = region_level(cv, y, region)

    # Locked test, descriptive: confirm the refit reproduces Step 6, then the same statistics.
    lscores, ly, lvillage, lregion = locked_scores(seed)
    step6 = {r["model"]: r["auroc"]["estimate"] for r in
             json.loads((ROOT / "results" / "stage29_final_test.json").read_text())["results"]["rows"]}
    ind = {k: summarize(ly, s, boot["n_resamples"], boot["ci"], seed, lvillage,
                        survey_metrics(survey, not k.startswith("hb"))) for k, s in lscores.items()}
    for k, label in (("lr__base", "base 6 questions"), ("lr__top_10", "winner: top 10 by importance")):
        assert abs(ind[k]["auroc"]["estimate"] - step6[label]) < 1e-9, f"{k} does not reproduce Step 6"
    assert abs(ind["hb__all"]["auroc"]["estimate"] - next(v for k, v in step6.items() if k.startswith("Hb"))) < 1e-9
    l_area = boot_area(lscores, ly, lvillage, lregion, rc["area_top_share"], boot["n_resamples"], boot["ci"], seed,
                       reference="region")
    l_ceiling = noise_ceiling(ly, lvillage, lregion, rc["split_half_draws"], seed)

    path = save_results("stage29_area_check", {"cv": {"area": area, "regions": regions, "ceiling": ceiling},
                                               "locked_descriptive": {"area": l_area, "ceiling": l_ceiling,
                                                                      "individual": ind}},
                        {"default": base_cfg, "feature_selection": cfg, "survey": survey})

    label = lambda k: MODELS[k][0]
    c = lambda d: f"{d['estimate']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"
    d = lambda d: f"{d['estimate']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"

    def table(a, keys, artifact=True):
        # Within regions the region baseline is 0 by construction (one score per region), so the within-region
        # column is read against 0, not against the baseline's computed value (see ARTIFACT).
        lines = ["| model | Spearman, all villages | minus region | within regions | "
                 f"top-{rc['area_top_share']:.0%} overlap | minus region |", "|---|---|---|---|---|---|"]
        for k in keys:
            r = a[k]
            m = lambda s: d(r[f"{s}_minus_region"]) if f"{s}_minus_region" in r else "-"
            note = ", artifact" if artifact else ""
            w = (f"0 by construction (computed {r['within_region']['estimate']:+.2f}{note})" if k == "region"
                 else c(r["within_region"]))
            lines.append(f"| {label(k)} | {c(r['spearman'])} | {m('spearman')} | {w} | {c(r['top_overlap'])} | "
                         f"{m('top_overlap')} |")
        return lines

    def ceil_lines(cl):
        return [f"- All villages: half-village correlation {cl['overall']['half_village']:.2f}, full-village "
                f"reliability {cl['overall']['reliability']:.2f}, so a perfect model's correlation is at most "
                f"**{cl['overall']['max_correlation']:.2f}**.",
                f"- Within regions: half-village {cl['within_region']['half_village']:.2f}, reliability "
                f"{cl['within_region']['reliability']:.2f}, at most **{cl['within_region']['max_correlation']:.2f}**."]

    n_v = len(set(village))
    text = "\n".join([
        "# Stage 29 Step 8b: does the model rank villages, or only repeat region?",
        "",
        f"Generated by `python -m src.survey.area_check` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`review`). Anemic = DHS `v457` (WHO for women on adjusted Hb: < 12.0 g/dL, "
        "< 11.0 if pregnant).",
        "",
        "Each village's score is the mean score of its women; it is compared with the village's observed anemia share. "
        "**Within regions** averages the correlation inside each region (weighted by villages), because the region "
        "baseline captures all differences between regions by construction; it is the real test of whether a model "
        f"adds anything about places. **Top-{rc['area_top_share']:.0%} overlap:** of the villages with the highest "
        "observed share, the fraction the model also puts in its top group. 95% CIs and differences from the region "
        f"baseline: village bootstrap, {boot['n_resamples']} resamples.",
        "",
        f"## Building villages ({n_v} villages, 5 x 5 repeated-CV out-of-fold scores)",
        "",
        f"About {len(y) / n_v:.0f} women per village, so a village's observed share is noisy. **Noise ceiling** "
        f"(split-half over {rc['split_half_draws']} random splits, Spearman-Brown corrected):",
        "",
        *ceil_lines(ceiling),
        "",
        *table(area, AREA_MODELS),
        "",
        "- **Cross-validation artifact within regions (found after the first run; the analysis is unchanged, only how "
        "it is read).** A village's out-of-fold score is learned without that village. When a high-anemia village is "
        "in the test fold, its region's training average is a little lower, so its own score is a little lower. Inside "
        "one region the region baseline's scores differ only by this (0.221-0.239 in one region of 36 villages), which "
        "gives it a strongly negative within-region correlation. The baseline's true value is 0 (it cannot rank "
        "villages inside a region), so the within-region column is read against 0. Models that use region (top 10, "
        "all questions, Hb ranking, TabPFN top 10 and all) carry a smaller share of the same downward pull within "
        "regions; base 6 has no region question and is least affected. The locked test has no such artifact (all its "
        "villages are left out of training equally), so its within-region numbers are the cleaner ones, though "
        "descriptive.",
        "- **The winner's numbers are optimistic:** it was picked as Step 5's winner using these same building "
        "villages. Base 6 and the region baselines have no such selection.",
        "",
        f"**Regions (10), Spearman of mean score vs. observed share:** "
        + ", ".join(f"{label(k)} {v:.2f}" for k, v in regions.items()) + ". With 10 points this is weak evidence.",
        "",
        f"## Locked test, descriptive only ({len(set(lvillage))} villages)",
        "",
        "These reuse the Step 6 models, refit unchanged on the building villages (the refit reproduces Step 6's "
        "AUROCs exactly), to answer a new question. With 79 villages they cannot confirm anything; no choice depends "
        "on them.",
        "",
        *ceil_lines(l_ceiling),
        "",
        *table(l_area, list(lscores), artifact=False),
        "",
        "**Individual level on the locked test, with the region baselines added:**",
        "",
        markdown_table([{"model": label(k), **v} for k, v in ind.items()], ["model"], SURVEY_COLUMNS),
        "",
    ])
    (ROOT / "reports" / "stage29_area_check.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
