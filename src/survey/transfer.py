"""Stage 8: fit on all of Zimbabwe, score Malawi and Zambia untouched, then test Platt recalibration on 200 local women.

Run from the repo root: python -m src.survey.transfer
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import brier, calibration_intercept, calibration_slope
from src.common.splits import repeated_stratified_grouped
from src.eyelid.report import markdown_table
from src.survey.build_dataset import load
from src.survey.train_baselines import (FIGURES, SURVEY_COLUMNS, feature_frame, make_model, oof_pipeline_scores,
                                        survey_metrics)

CALIBRATION = {"cal_slope": calibration_slope, "cal_intercept": calibration_intercept}
COLUMNS = {**SURVEY_COLUMNS, "cal_slope": "Cal. slope", "cal_intercept": "Cal. intercept"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


def recalibrate(y, p, cfg, seed):
    """Platt scaling fit on n random target women, scored on the rest; repeated with different draws."""
    rc = cfg["recalibration"]
    runs = []
    for i in range(rc["n_repeats"]):
        rng = np.random.default_rng(seed + i)
        fit_idx = rng.choice(len(y), rc["n_women"], replace=False)
        rest = np.setdiff1d(np.arange(len(y)), fit_idx)
        platt = LogisticRegression(C=np.inf).fit(logit(p[fit_idx]), y[fit_idx])
        q = platt.predict_proba(logit(p[rest]))[:, 1]
        runs.append({"before": {"brier": brier(y[rest], p[rest]), "cal_slope": calibration_slope(y[rest], p[rest]),
                                "cal_intercept": calibration_intercept(y[rest], p[rest]), "mean_risk": float(p[rest].mean())},
                     "after": {"brier": brier(y[rest], q), "cal_slope": calibration_slope(y[rest], q),
                               "cal_intercept": calibration_intercept(y[rest], q), "mean_risk": float(q.mean())},
                     "prevalence": float(y[rest].mean())})
    agg = {}
    for when in ("before", "after"):
        for k in runs[0][when]:
            v = np.array([r[when][k] for r in runs])
            agg[f"{k}_{when}"] = {"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))}
    agg["prevalence"] = float(np.mean([r["prevalence"] for r in runs]))
    return agg


def main():
    base, cfg = load_config("default"), load_config("survey")
    seed, boot = base["seed"], base["bootstrap"]
    metrics = {**survey_metrics(cfg), **CALIBRATION}
    train_name = cfg["train_country"]
    train = load(train_name)
    Xtr, ytr = feature_frame(train, cfg), train.anemic.to_numpy()
    targets = [c for c in cfg["countries"] if c != train_name]

    rows, recal, curves = [], {}, {}
    folds = repeated_stratified_grouped(ytr, train.cluster.to_numpy(), cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
    for kind in cfg["transfer_models"]:
        # Reference: the training country's own out-of-fold scores (Stage 7 setup), now with calibration metrics too.
        s = oof_pipeline_scores(kind, Xtr, ytr, None, folds, cfg, seed).mean(0)
        rows.append({"model": kind, "country": f"{train_name} (cross-validated)", "n": len(ytr),
                     "prevalence": f"{ytr.mean():.1%}",
                     **summarize(ytr, s, boot["n_resamples"], boot["ci"], seed, train.cluster.to_numpy(), metrics)})
        curves[(kind, train_name)] = (ytr, s)
        model = make_model(kind, cfg, seed).fit(Xtr, ytr)  # all of the training country; preprocessing fit here only
        for country in targets:
            df = load(country)
            y, p = df.anemic.to_numpy(), model.predict_proba(feature_frame(df, cfg))[:, 1]
            rows.append({"model": kind, "country": f"{country} (untouched)", "n": len(y), "prevalence": f"{y.mean():.1%}",
                         **summarize(y, p, boot["n_resamples"], boot["ci"], seed, df.cluster.to_numpy(), metrics)})
            recal[f"{kind}/{country}"] = recalibrate(y, p, cfg, seed)
            curves[(kind, country)] = (y, p)
            print(f"{kind} -> {country}: AUROC {rows[-1]['auroc']['estimate']:.3f}, mean risk {p.mean():.3f} "
                  f"vs prevalence {y.mean():.3f}", flush=True)

    fig, axes = plt.subplots(1, len(cfg["transfer_models"]), figsize=(9, 4.2), sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), cfg["transfer_models"]):
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for (k, country), (y, p) in curves.items():
            if k == kind:
                fp, mp = calibration_curve(y, p, n_bins=cfg["calibration_bins"], strategy="quantile")
                ax.plot(mp, fp, "o-", ms=3, label=country)
        ax.set_title(kind.replace("_", " "))
        ax.set_xlabel("predicted chance of anemia")
        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 0.6)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("share actually anemic")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(f"Calibration: trained on {train_name}, scored on each country")
    fig.tight_layout()
    fig.savefig(FIGURES / "survey_transfer_calibration.png", dpi=150)

    rc = cfg["recalibration"]
    recal_lines = [f"| {key} | {r['prevalence']:.1%} | {r['mean_risk_before']['mean']:.1%} | {r['mean_risk_after']['mean']:.1%} | "
                   f"{r['brier_before']['mean']:.4f} | {r['brier_after']['mean']:.4f} [{r['brier_after']['lo']:.4f}, {r['brier_after']['hi']:.4f}] | "
                   f"{r['cal_intercept_before']['mean']:.2f} -> {r['cal_intercept_after']['mean']:.2f} | "
                   f"{r['cal_slope_before']['mean']:.2f} -> {r['cal_slope_after']['mean']:.2f} |" for key, r in recal.items()]
    text = (
        "# Stage 8: transfer from Zimbabwe to Malawi and Zambia\n\n"
        f"Generated by `python -m src.survey.transfer`. Models fit on all {len(ytr)} tested women from Zimbabwe 2015 "
        "(preprocessing fit on Zimbabwe only) and applied unchanged to Malawi 2024 and Zambia 2024. Anemic = DHS `v457` at "
        "any severity (WHO cutoffs on altitude- and smoking-adjusted Hb) in every country. Bootstrap 95% CIs "
        f"({boot['n_resamples']} resamples) resample whole clusters. Calibration slope 1 and intercept 0 are ideal.\n\n"
        "The DHS wealth index is built separately for each country and survey, so the \"richest\" quintile in Zambia is "
        "not the same wealth as in Zimbabwe, and the Zimbabwe survey is 9 years older than the other two.\n\n"
        + markdown_table(rows, ["model", "country", "n", "prevalence"], COLUMNS)
        + f"\n\n## Recalibration with {rc['n_women']} local women\n\n"
        f"Platt scaling fit on {rc['n_women']} random women from the target country, evaluated on all the others; "
        f"{rc['n_repeats']} repeats, mean with the 2.5-97.5% range over repeats. Ranking metrics (AUROC, recall) do not "
        "change, because recalibration keeps the order.\n\n"
        "| model / country | prevalence | mean predicted risk before | after | Brier before | Brier after | intercept before -> after | slope before -> after |\n"
        "|---|---|---|---|---|---|---|---|\n" + "\n".join(recal_lines)
        + "\n\n![Transfer calibration](figures/survey_transfer_calibration.png)\n"
    )
    (ROOT / "reports" / "stage8_transfer.md").write_text(text)
    print(text)
    save_results("stage8_survey_transfer", {"rows": rows, "recalibration": recal}, {"default": base, "survey": cfg})


if __name__ == "__main__":
    main()
