"""Stage 37: TabPFN or the 29 color numbers? Age and sex next to them, and numbers that shouldn't carry paleness.

The three asks from Jaineel's part of the 2026-11-17 mentor meeting, on the Ghana eyelids (Hb < 11.0 g/dL):
1. logistic regression and an MLP on the same 29 numbers as TabPFN (how much is the model, how much the features);
2. age and sex (0/1) next to the 29 numbers;
3. the numbers that shouldn't carry paleness, and combinations of redness, brightness and saturation.
Every score goes through Stage 35's checks (jump test, step at the cutoff, seven photo edits). Settings and the
answers, fixed before running, in configs/model_vs_features.yaml.

Run from the repo root: python -m src.eyelid.model_vs_features
"""
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.common.io import ROOT, load_config
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.metadata import age_sex
from src.eyelid.redness_check import CV, LABELS, LOHO, checks_text, every_score, run
from src.eyelid.report import fmt
from src.eyelid.train_color import _short_commit

EXTRA = ["age_months", "sex_male"]
STAGE37 = {"result": "stage37_model_vs_features", "module": "src.eyelid.model_vs_features",
           "config": "model_vs_features",
           "title": "Stage 37: TabPFN or the 29 color numbers? Age and sex, and numbers that shouldn't carry paleness "
                    "(Ghana eyelids)"}


def mlp_factory(cfg, seed):
    """Standardized columns into an MLP whose size and L2 penalty are picked by an inner CV on the training rows only."""
    m = cfg["mlp"]
    grid = {"mlpclassifier__hidden_layer_sizes": [tuple(h) for h in m["hidden_layer_sizes"]],
            "mlpclassifier__alpha": list(m["alpha"])}

    def make():
        net = make_pipeline(StandardScaler(), MLPClassifier(solver="lbfgs", max_iter=m["max_iter"], random_state=seed))
        inner = StratifiedKFold(m["inner_folds"], shuffle=True, random_state=seed)
        return GridSearchCV(net, grid, scoring="roc_auc", cv=inner, n_jobs=m["n_jobs"])
    return make


def age_and_sex(name, d):
    """Age in months and sex (1 = male) for the Ghana eyelid rows, in the same order as the color numbers."""
    assert name == "ghana_eyelid", f"{name}: age and sex are only wired up for the Ghana eyelids"
    cp = load_cp_anemic(load_config("ceiling")["cp_anemic_subset"])
    assert np.allclose(cp.hb.to_numpy(float), d["hb"]), "Ghana eyelid rows out of order"
    return EXTRA, age_sex(cp)


def answers(d, cfg):
    """The three answers fixed in the config, from one dataset's results (5x5 CV)."""
    s = d["splits"][CV]
    au = lambda k, m: s["sets"][k][m]["scores"]["auroc"]["estimate"]
    others = [m for m in cfg["models"] if m != "tabpfn"]
    best = max(others, key=lambda m: au("all_29", m))
    pairs = s["set_pairs"]
    return {"best_other": best,
            "tabpfn_adds": s["sets"]["all_29"]["tabpfn"]["vs"][best]["lo"] > 0,
            "age_sex_helps": {m: pairs["all_29_age_sex - all_29"][m]["lo"] > 0 for m in cfg["models"]},
            "photo_adds": {m: pairs["all_29_age_sex - age_sex_only"][m]["lo"] > 0 for m in cfg["models"]},
            "weird": {m: au("not_useful", m) >= au("paleness", m) - cfg["weird_margin"]
                      and s["sets"]["not_useful"][m]["vs_rule"]["lo"] > 0 for m in cfg["models"]}}


def write_report(path, cfg, jump_cfg, out, stage):
    d = out["datasets"]["ghana_eyelid"]
    a = answers(d, cfg)
    cv, loho = d["splits"][CV], d["splits"].get(LOHO)
    models = cfg["models"]
    sc = lambda k, m, key="auroc": fmt(cv["sets"][k][m]["scores"][key])
    lo = lambda k, m: fmt(loho["sets"][k][m]["scores"]["auroc"]) if loho else "-"
    mark = lambda r: "" if r.get("no_trace") else " (trace)"
    yes = lambda v: "**yes**" if v else "no"

    rule, dummy = d["rule"]["scores"], cv["dummy"]
    part1 = [f"| dummy | {fmt(dummy['auroc'])} | {fmt(dummy['pr_auc'])} | {fmt(dummy['sens_at_90_spec'])} | "
             f"{fmt(loho['dummy']['auroc']) if loho else '-'} | - |",
             f"| rule (minus mean a*), untrained | {fmt(rule['auroc'])} | {fmt(rule['pr_auc'])} | "
             f"{fmt(rule['sens_at_90_spec'])} | same (untrained) | - |"]
    for m in models:
        vs = cv["sets"]["all_29"]["tabpfn"]["vs"].get(m)
        part1.append(f"| {LABELS[m]}{mark(cv['sets']['all_29'][m])} | {sc('all_29', m)} | {sc('all_29', m, 'pr_auc')} | "
                     f"{sc('all_29', m, 'sens_at_90_spec')} | {lo('all_29', m)} | {fmt(vs) if vs else '-'} |")

    p = cv["set_pairs"]
    part2 = [f"| {LABELS[m]} | {sc('all_29', m)} | {sc('all_29_age_sex', m)} | {fmt(p['all_29_age_sex - all_29'][m])} "
             f"| {yes(a['age_sex_helps'][m])} | {sc('age_sex_only', m)} | {fmt(p['all_29_age_sex - age_sex_only'][m])} |"
             for m in models]

    color_sets = [k for k in cfg["input_sets"] if k not in ("all_29_age_sex", "age_sex_only")]
    part3 = [f"| {k} | {len(d['input_sets'][k])} | "
             + " | ".join(sc(k, m) + mark(cv["sets"][k][m]) for m in models) + " |" for k in color_sets]
    weird = [f"| {LABELS[m]} | {sc('not_useful', m)} | {sc('paleness', m)} | {fmt(p['not_useful - paleness'][m])} | "
             f"{fmt(cv['sets']['not_useful'][m]['vs_rule'])} | {yes(a['weird'][m])} |" for m in models]
    sets = "\n".join(f"| {k} | {', '.join(v)} |" for k, v in d["input_sets"].items())

    best = LABELS[a["best_other"]]
    report = (
        f"# {stage['title']}\n\n"
        f"Generated by `python -m {stage['module']}` at commit `{_short_commit(path)}`; settings and the answers "
        f"(fixed before running) in `configs/{stage['config']}.yaml`.\n\n"
        f"- **Cutoff:** Ghana eyelids (CP-AnemiC, the {d['n']} photos with no copy, {d['anemic']} anemic): children "
        "6-59 months, **Hb < 11.0 g/dL**. Ghana only.\n"
        "- **Models:** logistic regression (standardized, C = 1.0), an MLP (standardized; size and L2 penalty picked "
        "by an inner CV inside each training fold), gradient boosting and TabPFN v2 (raw numbers), on the same inputs "
        "and folds; the dummy and the untrained rule (minus mean a*) next to them. 5-fold CV x 5 repeats by photo, "
        "scores averaged over repeats; also one hospital held out at a time. Bootstrap 95% CIs, 2000 resamples. "
        "Changes are paired (same children, same resamples).\n"
        + checks_text(cfg, jump_cfg, out)
        + "- \"(trace)\" after a score = it fails at least one check. Every Ghana model carried the folder trace in "
        "Stages 30-35, so these numbers compare models and inputs, not how much of a score is anemia.\n\n"
        "## 1. TabPFN or the 29 numbers?\n\n"
        "Mentor (2026-11-17): \"you can train a logistic regression or like an MLP on these extracted 29 features and "
        "see how the model does there as well.\"\n\n"
        f"**Answer:** the best other model on the 29 numbers is {best}; TabPFN minus {best}: "
        f"{fmt(cv['sets']['all_29']['tabpfn']['vs'][a['best_other']])}. TabPFN adds something beyond the feature "
        f"extraction: **{yes(a['tabpfn_adds'])}**.\n\n"
        "| model, all 29 numbers | AUROC (5x5 CV) | PR-AUC | sens @ 90% spec | one hospital held out | "
        "TabPFN minus this model |\n|---|---|---|---|---|---|\n" + "\n".join(part1) + "\n\n"
        "## 2. Age and sex next to the 29 numbers\n\n"
        "Mentor: \"you can add the age and sex and BMI\"; sex as 0/1, scaled like the other columns. The data has no "
        "BMI or other demographics (CP-AnemiC columns: id, image_path, hb, anemic, site, age_months, sex, image_group, "
        "photo_group), so only age (months) and sex (1 = male) are added.\n\n"
        "**Answer:** age and sex help " + (", ".join(LABELS[m] for m in models if a["age_sex_helps"][m]) or "no model")
        + "; the photo adds to age and sex for " + (", ".join(LABELS[m] for m in models if a["photo_adds"][m])
                                                     or "no model") + ".\n\n"
        "| model | 29 numbers | 29 + age + sex | change | helps? | age + sex alone | 29 + age + sex minus age + sex "
        "alone |\n|---|---|---|---|---|---|---|\n" + "\n".join(part2) + "\n\n"
        "## 3. Numbers that shouldn't carry paleness, and combinations of redness, brightness and saturation\n\n"
        "Jaineel (2026-11-17): \"if I gave it everything that was not supposed to be useful ... green, blue, anything "
        "that's not brightness and redness, how does it perform?\" not_useful = hue, green, blue and yellow-blue "
        "(the red channel is left out of both sides, since it is partly redness); paleness = redness + brightness + "
        f"saturation. \"Weird\" (fixed before running): not_useful scores at least paleness minus "
        f"{cfg['weird_margin']} and beats the rule.\n\n"
        "**Answer:** weird for " + (", ".join(LABELS[m] for m in models if a["weird"][m]) or "no model") + ".\n\n"
        "| model | not_useful | paleness | not_useful minus paleness | not_useful minus rule | weird? |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(weird) + "\n\n"
        "AUROC (5x5 CV) on each set:\n\n"
        "| input set | numbers | " + " | ".join(LABELS[m] for m in models) + " |\n|" + "---|" * (2 + len(models))
        + "\n" + "\n".join(part3) + "\n\n"
        "## Input sets\n\n| set | columns |\n|---|---|\n" + sets + "\n\n" + every_score(out))
    (ROOT / "reports" / f"{stage['result']}.md").write_text(report)
    print(report)


def main():
    cfg, seed = load_config("model_vs_features"), load_config("default")["seed"]
    make_mlp = mlp_factory(cfg, seed)
    run(STAGE37 | {"extra": age_and_sex, "write_report": write_report},
        lambda ladder, cols: {m: make_mlp if m == "mlp" else ladder[m][1] for m in cfg["models"]})


if __name__ == "__main__":
    main()
