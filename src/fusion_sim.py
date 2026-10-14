"""Fusion SIMULATION: how many more anemic women a fixed testing budget would find with an eyelid score on top of the
survey score. See configs/fusion.yaml for why this is simulated and what it assumes.

Run from the repo root: python -m src.fusion_sim
"""
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.common.evaluate import IMAGE_METRICS, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import recall_at_top_fraction
from src.eyelid.color_features import feature_table
from src.eyelid.cross_dataset import load_eyes_defy
from src.eyelid.report import markdown_table
from src.survey.build_dataset import load
from src.survey.train_baselines import FIGURES, SURVEY_COLUMNS, feature_frame, make_model


def draw_image_scores(y, pool_scores, pool_y, rng):
    """For each woman, a random eyelid score from pool photos with her anemia status."""
    out = np.empty(len(y))
    for label in (0, 1):
        out[y == label] = rng.choice(pool_scores[pool_y == label], (y == label).sum(), replace=True)
    return out


def main():
    base, cfg, survey_cfg = load_config("default"), load_config("fusion"), load_config("survey")
    color_cfg, cross_cfg = load_config("color"), load_config("cross_dataset")
    seed, boot = base["seed"], base["bootstrap"]
    metrics = {**IMAGE_METRICS, "recall_top": partial(recall_at_top_fraction, frac=survey_cfg["top_fraction"])}
    columns = {k: v for k, v in SURVEY_COLUMNS.items() if k != "brier"}

    ed = load_eyes_defy(cross_cfg)
    colors = feature_table([ROOT / p for p in ed.image_path], ed.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    rule = -colors[color_cfg["rule_feature"]].to_numpy()
    pools = {}
    for name, country in cfg["image_scenarios"].items():
        m = (ed.country == country).to_numpy()
        s, yy = rule[m], ed.anemic.to_numpy()[m]
        slope = float(LogisticRegression(C=np.inf).fit(s[:, None], yy).coef_[0, 0])  # image log-odds per unit score
        pools[name] = (s, yy, slope)
        print(f"{name}: {m.sum()} photos, {yy.sum()} anemic, image log-odds slope {slope:.2f}", flush=True)

    train = load(survey_cfg["train_country"])
    model = make_model(cfg["survey_model"], survey_cfg, seed).fit(feature_frame(train, survey_cfg), train.anemic.to_numpy())
    rows, spread, curves = [], {}, {}
    for country in cfg["targets"]:
        df = load(country)
        y, groups = df.anemic.to_numpy(), df.cluster.to_numpy()
        p = np.clip(model.predict_proba(feature_frame(df, survey_cfg))[:, 1], 1e-6, 1 - 1e-6)
        survey_logit = np.log(p / (1 - p))
        rows.append({"country": country, "scores": "survey only", **summarize(y, p, boot["n_resamples"], boot["ci"], seed, groups, metrics)})
        curves[(country, "survey only")] = (y, p)
        for name, (s, yy, slope) in pools.items():
            draws = []
            for d in range(cfg["n_draws"]):
                img = draw_image_scores(y, s, yy, np.random.default_rng(seed + d))
                combined = survey_logit + slope * img
                if d == 0:
                    rows.append({"country": country, "scores": f"image only, {name}", **summarize(y, img, boot["n_resamples"], boot["ci"], seed, groups, metrics)})
                    rows.append({"country": country, "scores": f"survey + image, {name}", **summarize(y, combined, boot["n_resamples"], boot["ci"], seed, groups, metrics)})
                    curves[(country, f"survey + image, {name}")] = (y, combined)
                draws.append({k: fn(y, combined) for k, fn in metrics.items()})
            spread[f"{country} / survey + image, {name}"] = {
                k: {"min": float(min(d[k] for d in draws)), "max": float(max(d[k] for d in draws))} for k in metrics}
        print(f"{country}: done", flush=True)

    budgets = survey_cfg["budget_curve"]
    fig, axes = plt.subplots(1, len(cfg["targets"]), figsize=(10, 4), sharey=True)
    for ax, country in zip(np.atleast_1d(axes), cfg["targets"]):
        ax.plot(budgets, budgets, "k:", lw=1, label="random testing")
        for (c, name), (y, s) in curves.items():
            if c == country:
                ax.plot(budgets, [recall_at_top_fraction(y, s, b) for b in budgets], "o-", ms=3, label=name)
        ax.axvline(survey_cfg["top_fraction"], color="grey", lw=0.8, ls=":")
        ax.set_title(f"{country} (simulated image scores)")
        ax.set_xlabel("share of women tested (highest risk first)")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("share of anemic women found")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("SIMULATION: survey score alone vs with an eyelid score")
    fig.tight_layout()
    fig.savefig(FIGURES / "fusion_simulation.png", dpi=150)

    spread_lines = [f"| {k} | {v['auroc']['min']:.3f}-{v['auroc']['max']:.3f} | {v['recall_top']['min']:.3f}-{v['recall_top']['max']:.3f} |"
                    for k, v in spread.items()]
    text = (
        "# Fusion simulation: survey score plus an eyelid score\n\n"
        "**This is a simulation, not a trained model.** No dataset has an eyelid photo and survey answers for the same "
        "woman. Each Malawi and Zambia woman keeps her real survey risk score (logistic regression fit on all of Zimbabwe, "
        "Stage 8) and gets an eyelid score drawn at random from Eyes-Defy photos with her anemia status. The eyelid score "
        "is the untrained redness rule (minus mean a*), the image signal that held up best across datasets (Stage 5). "
        "Combined score = survey logit + slope x eyelid score, with the slope fit on the Eyes-Defy subset; this assumes "
        "the two signals are independent given anemia status, which is unproven. Two scenarios: eyelid scores from Indian "
        "photos (a weak signal) and from Italian photos (a strong one, but Italy's anemic adults are few and fairly "
        "severe). Eyes-Defy labels use WHO adult cutoffs by sex; the survey labels use DHS `v457` (WHO for women, "
        f"altitude- and smoking-adjusted). Bootstrap 95% CIs ({boot['n_resamples']} resamples) resample whole clusters, "
        "for one random image draw.\n\n"
        + markdown_table(rows, ["country", "scores"], columns)
        + f"\n\nSpread of the point estimate over {cfg['n_draws']} different random image draws:\n\n"
        "| scores | AUROC range | recall, top 30% range |\n|---|---|---|\n" + "\n".join(spread_lines)
        + "\n\n![Fusion simulation](figures/fusion_simulation.png)\n"
    )
    (ROOT / "reports" / "fusion_simulation.md").write_text(text)
    print(text)
    save_results("fusion_simulation", {"rows": rows, "draw_spread": spread,
                                       "image_slopes": {k: v[2] for k, v in pools.items()}},
                 {"default": base, "fusion": cfg, "survey": survey_cfg, "cross_dataset": cross_cfg})


if __name__ == "__main__":
    main()
