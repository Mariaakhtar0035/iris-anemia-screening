"""Stage 29 Step 6: the one-time score on the locked final test (Zimbabwe 2015, 20% of villages).

Role: main result, Stage 29 (the one-time locked-test score).

Every model is fit on all building villages, with its question choice made there too, then scored once on the locked
villages. Candidates were fixed in configs/feature_selection.yaml (`final_test`) before this ran: the Step 5 winner
(top 10 questions by importance, logistic regression) and the best Hb ranking (gradient boosting regressor on all
questions). Next to them: the dummy, the "pregnant first" rule and the 6-question baseline.

Run from the repo root: python -m src.survey.final_test   (after select_features, screen_features, select_models)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci, calibration_intercept, calibration_slope
from src.common.report import markdown_table
from src.common.report import short_commit as _short_commit
from src.survey.screen_features import TARGETS
from src.survey.select_features import OUT
from src.survey.select_models import _share, columns_for, importance_ranking, make_model, prepare
from src.survey.train_baselines import SURVEY_COLUMNS, survey_metrics

FIGURES = ROOT / "reports" / "figures"


def fit_predict(kind, questions, feats, train, test, target, mc, seed, proba=True):
    """Fit on the building rows, predict the locked rows. Columns and categories come from the building rows only."""
    X = prepare(feats[columns_for(questions, feats)])
    model = make_model(kind, X, mc, seed).fit(X[train], target[train])
    return model.predict_proba(X[test])[:, 1] if proba else model.predict(X[test])


def main():
    base_cfg, cfg, survey = load_config("default"), load_config("feature_selection"), load_config("survey")
    mc, ft, seed, boot = cfg["models"], cfg["final_test"], base_cfg["seed"], base_cfg["bootstrap"]
    data = pd.read_parquet(OUT)
    uni = pd.read_csv(ROOT / "reports" / "stage29_univariate.csv")
    questions = uni.code[uni.step4 == "keep"].tolist()
    base = mc["base_questions"]
    feats = data.drop(columns=TARGETS)
    test = data.locked_test.to_numpy()
    train = ~test
    y_all, hb_all = data.anemic.to_numpy(), data.hb.to_numpy()
    y, groups, cutoff = y_all[test], data.cluster.to_numpy()[test], data.hb_cutoff.to_numpy()[test]
    assert not set(data.cluster[train]) & set(groups), "a village is on both sides"
    print(f"building: {train.sum()} women; locked test: {test.sum()} women in {len(set(groups))} villages, "
          f"{y.mean():.1%} anemic", flush=True)

    # The winner's questions: importance ranking on the building villages only, then the top k.
    w = ft["winner"]
    Xb = prepare(feats[columns_for(questions, feats)])
    ranking = importance_ranking(Xb[train], y_all[train], data.cluster.to_numpy()[train], questions, mc, seed)
    winner_q = ranking[:w["k"]]
    print(f"winner's {w['k']} questions: {winner_q}", flush=True)

    scores = {
        "dummy (everyone the same risk)": fit_predict("dummy", base, feats, train, test, y_all, mc, seed),
        "base 6 questions": fit_predict(ft["reference"]["model"], base, feats, train, test, y_all, mc, seed),
        f"winner: top {w['k']} by importance": fit_predict(w["model"], winner_q, feats, train, test, y_all, mc, seed),
    }
    hb_pred = fit_predict(ft["hb_candidate"]["model"], questions, feats, train, test, hb_all, mc, seed, proba=False)
    rank_scores = {
        "rule: pregnant first, then random": feats.v213.eq("yes").to_numpy()[test]
        + 0.5 * np.random.default_rng(seed).random(test.sum()),
        f"Hb ranking: all {len(questions)} questions": -(hb_pred - cutoff),
    }

    metrics = {**survey_metrics(survey), "cal_slope": calibration_slope, "cal_intercept": calibration_intercept}
    rank_metrics = survey_metrics(survey, probabilities=False)
    summ = lambda s, m: summarize(y, s, boot["n_resamples"], boot["ci"], seed, groups, m)
    n_q = {"dummy (everyone the same risk)": 0, "base 6 questions": len(base), f"winner: top {w['k']} by importance": w["k"],
           "rule: pregnant first, then random": 1, f"Hb ranking: all {len(questions)} questions": len(questions)}
    rows = []
    for name, s in {**rank_scores, **scores}.items():
        m = summ(s, metrics if name in scores and not name.startswith("dummy") else rank_metrics)
        if name.startswith("dummy"):
            m["brier"] = bootstrap_ci(metrics["brier"], y, s, boot["n_resamples"], boot["ci"], groups, seed)
        rows.append({"model": name, "questions": n_q[name], **m})
        print(f"{name}: AUROC {m['auroc']['estimate']:.3f} [{m['auroc']['lo']:.3f}, {m['auroc']['hi']:.3f}]", flush=True)
    order = ["dummy (everyone the same risk)", "rule: pregnant first, then random", "base 6 questions",
             f"winner: top {w['k']} by importance", f"Hb ranking: all {len(questions)} questions"]
    rows = sorted(rows, key=lambda r: order.index(r["model"]))

    # Paired change over the 6-question baseline, same resampled villages for both.
    ref = scores["base 6 questions"]
    changes = {}
    for name, s in [(order[3], scores[order[3]]), (order[4], rank_scores[order[4]])]:
        packed = np.column_stack([ref, s])
        changes[name] = bootstrap_ci(lambda yy, ss: auroc(yy, ss[:, 1]) - auroc(yy, ss[:, 0]), y, packed,
                                     boot["n_resamples"], boot["ci"], groups, seed)

    # The Hb number itself.
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    ci = lambda fn, a, b: bootstrap_ci(fn, a, b, boot["n_resamples"], boot["ci"], groups, seed)
    flagged = (hb_pred < cutoff).astype(float)
    hb_test = hb_all[test]
    hb_row = {"model": f"Hb ranking: all {len(questions)} questions", "mae": ci(mae, hb_test, hb_pred),
              "mae_mean_guess": ci(mae, hb_test, np.full(len(y), hb_all[train].mean())),
              "sens_at_cutoff": ci(lambda a, f: _share(f[a == 1]), y, flagged),
              "spec_at_cutoff": ci(lambda a, f: _share(1 - f[a == 0]), y, flagged)}

    # Calibration plot for the models that give a probability.
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
    for name in ("base 6 questions", f"winner: top {w['k']} by importance"):
        frac, mean_pred = calibration_curve(y, scores[name], n_bins=survey["calibration_bins"], strategy="quantile")
        ax.plot(mean_pred, frac, "o-", ms=4, label=name)
    ax.axhline(y.mean(), color="grey", lw=0.8, ls=":", label=f"share anemic ({y.mean():.0%})")
    ax.set_xlim(0, 0.6), ax.set_ylim(0, 0.6)
    ax.set_xlabel("predicted chance of anemia")
    ax.set_ylabel("share actually anemic")
    ax.set_title("Calibration, Zimbabwe locked test")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stage29_calibration.png", dpi=150)

    path = save_results("stage29_final_test", {
        "n_women": int(test.sum()), "n_villages": len(set(groups)), "anemic_share": float(y.mean()),
        "winner_questions": winner_q, "rows": rows, "auroc_change_vs_base6": changes, "hb": hb_row},
        {"default": base_cfg, "feature_selection": cfg, "survey": survey})

    cols = {**SURVEY_COLUMNS, "cal_slope": "Calibration slope", "cal_intercept": "Calibration intercept"}
    inv = pd.read_csv(ROOT / "reports" / "stage29_inventory.csv").set_index("code").label
    f = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    g = lambda d: f"{d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    text = "\n".join([
        "# Stage 29 Step 6: the one-time score on the locked final test (Zimbabwe 2015)",
        "",
        f"Generated by `python -m src.survey.final_test` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`final_test`). Every model was fit on all "
        f"{int(train.sum())} building women, with its questions chosen there, and scored once on the **locked test: "
        f"{int(test.sum())} women in {len(set(groups))} villages, {y.mean():.1%} anemic**. Anemic = DHS `v457` (WHO for "
        "women on Hb adjusted for altitude and smoking: < 12.0 g/dL, < 11.0 if pregnant).",
        "",
        "- The two candidates were named in the config after Step 5 and before this test was opened.",
        f"- Bootstrap 95% CIs, {boot['n_resamples']} resamples of whole villages. Rankings (the rule, the Hb ranking) "
        "have no probability, so no Brier score or calibration.",
        "- Calibration slope 1 and intercept 0 = perfectly calibrated.",
        "",
        markdown_table(rows, ["model", "questions"], cols),
        "",
        "**AUROC change over the 6-question baseline** (paired: same resampled villages):",
        "",
        *[f"- {k}: {f(v)}" for k, v in changes.items()],
        "",
        f"## The winner's {w['k']} questions (chosen on the building villages)",
        "",
        *[f"{i + 1}. `{q}` {inv.get(q, '')}" for i, q in enumerate(winner_q)],
        "",
        "## Extra target: the Hb number",
        "",
        f"- Average error: {g(hb_row['mae'])} g/dL; guessing everyone at the building mean: "
        f"{g(hb_row['mae_mean_guess'])} g/dL.",
        f"- Applying each woman's cutoff to the estimate: sensitivity {g(hb_row['sens_at_cutoff'])}, specificity "
        f"{g(hb_row['spec_at_cutoff'])}.",
        "",
        "![Calibration](figures/stage29_calibration.png)",
        "",
    ])
    (ROOT / "reports" / "stage29_final_test.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
