"""Stage 22: TabPFN for the Hb number, per location.

TabPFN is the best yes/no model in Ghana (0.92). Stage 16 estimated the Hb number with scikit-learn regressors only.
Here TabPFN's regressor estimates Hb from the same 29 color numbers (and age + sex for the eyelid sets), inside each
location, next to Stage 16's regressors and a mean-Hb dummy. The cutoff is applied to the estimate afterwards, one
cutoff per dataset.

Run from the repo root: python -m src.eyelid.hb_tabpfn
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import leave_one_group_out, repeated_stratified
from src.eyelid.ceiling import population_frames, population_inputs
from src.eyelid.hb_regression import load_moscow, mae, oof_regression, regressors, sens_at_cutoff, spec_at_cutoff, within_1
from src.eyelid.metadata import BOTH, NAMES, PHOTO, age_sex, mae_change
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import _short_commit

DUMMY = "dummy (mean Hb)"
COLUMNS = {"mae": "Hb error (MAE, g/dL)", "within_1": "Within 1 g/dL", "auroc": "Ranking AUROC",
           "caught": "Anemic caught at the cutoff", "cleared": "Healthy cleared at the cutoff",
           "vs_best": "MAE change vs best other (see the table above)"}


def locations(cfg, base, hb_cfg, seed):
    """name -> (hb, each person's cutoff, anemic, {input: X}, {split: folds})."""
    color_cfg, probe_cfg, ceiling_cfg = (load_config(n) for n in ("color", "probe", "ceiling"))
    data = population_inputs(population_frames(ceiling_cfg), color_cfg, probe_cfg, ceiling_cfg)
    ed_cut = load_config("cross_dataset")["eyes_defy_cutoff_g_dl"]
    out = {}
    for pop in (p for p in cfg["locations"] if p in NAMES):
        df, inp = data[pop]
        hb, y = df.hb.to_numpy(float), df.anemic.to_numpy()
        cut = (np.full(len(df), base["anemia_cutoff_g_dl"]["child_6_59_months"]) if pop == "cp_anemic"
               else df.sex.map(ed_cut).to_numpy(float))
        assert ((hb < cut).astype(int) == y).all(), f"{pop}: label disagrees with Hb < cutoff"
        splits = {"5x5 CV by photo": repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)}
        if pop == "cp_anemic":
            splits["leave one hospital out"] = leave_one_group_out(df.site.to_numpy())
        out[f"{NAMES[pop]} eyelids"] = (hb, cut, y, {PHOTO: inp["color"], BOTH: np.hstack([inp["color"], age_sex(df)])},
                                        splits)
    if "moscow_nails" in cfg["locations"]:
        ds = load_moscow(hb_cfg | {"cv": cfg["cv"]}, color_cfg, base, seed)
        hb = ds["hb"]
        out["Moscow fingernails"] = (hb, np.full(len(hb), ds["cutoff"]), (hb < ds["cutoff"]).astype(int),
                                     {PHOTO: ds["color"]}, ds["splits"])
    return out


def main():
    base, cfg, hb_cfg = load_config("default"), load_config("hb_tabpfn"), load_config("hb_regression")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    tab = cfg["tabpfn"]
    stage16 = regressors(hb_cfg, seed)
    regs = {DUMMY: DummyRegressor} | stage16 | {"tabpfn": lambda: TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=ROOT / tab["model_path"], device=tab["device"], random_state=seed)}

    def interval(metric, truth, score):
        return bootstrap_ci(metric, truth, score, n_boot, ci, seed=seed)

    rows = []
    for name, (hb, cut, y, X, splits) in locations(cfg, base, hb_cfg, seed).items():
        for split, folds in splits.items():
            preds = {(r, how): oof_regression(make, Xi, hb, folds).mean(0) for r, make in regs.items()
                     for how, Xi in X.items() if not (r == DUMMY and how != PHOTO)}
            print(f"done: {name}, {split}", flush=True)
            for (r, how), p in preds.items():
                row = {"location": name, "split": split, "n": len(hb), "anemic": f"{y.mean():.1%}", "model": r,
                       "input": "-" if r == DUMMY else how,
                       "mae": interval(mae, hb, p), "within_1": interval(within_1, hb, p),
                       "auroc": interval(auroc, y, -p),   # lower estimated Hb = more likely anemic
                       # each person's own cutoff: estimate minus cutoff below 0 = flagged anemic
                       "caught": interval(lambda a, s: sens_at_cutoff(a, s, 0.0), y, p - cut),
                       "cleared": interval(lambda a, s: spec_at_cutoff(a, s, 0.0), y, p - cut)}
                if r == "tabpfn":
                    best = min(stage16, key=lambda o: mae(hb, preds[(o, how)]))
                    row |= {"best_other": best, "vs_best": mae_change(hb, p, preds[(best, how)], n_boot, ci, seed)}
                rows.append(row)

    path = save_results("stage22_hb_tabpfn", {"rows": rows},
                        {"default": base, "hb_tabpfn": cfg, "hb_regression": hb_cfg})

    groups = list(dict.fromkeys((r["location"], r["split"]) for r in rows))
    summary, answers = [], []
    for loc, split in groups:
        g = {(r["model"], r["input"]): r for r in rows if r["location"] == loc and r["split"] == split}
        dummy = g[(DUMMY, "-")]
        for how in (PHOTO, BOTH):
            if ("tabpfn", how) not in g:
                continue
            t = g[("tabpfn", how)]
            b = g[(t["best_other"], how)]
            summary.append(f"| {loc} | {split} | {how} | {fmt(dummy['mae'])} | {fmt(b['mae'])} ({t['best_other']}) | "
                           f"{fmt(t['mae'])} | {fmt(t['vs_best'])} | {fmt(b['auroc'])} | {fmt(t['auroc'])} |")
            if t["vs_best"]["hi"] < 0:
                answers.append(f"{loc}, {split}, {how}: {fmt(t['vs_best'])}")

    cut = load_config("cross_dataset")["eyes_defy_cutoff_g_dl"]
    report = (
        "# Stage 22: TabPFN for the Hb number, per location\n\n"
        f"Generated by `python -m src.eyelid.hb_tabpfn` at commit `{_short_commit(path)}`; settings in "
        "`configs/hb_tabpfn.yaml`. Extends Stage 16 (scikit-learn regressors) and Stage 20 (age and sex). Mentor advice "
        "(2026-10-31): measure both the yes/no and the Hb estimate; (2026-11-03) per location only.\n\n"
        "- **Locations and cutoffs (one per dataset, used only to score the estimate):** Ghana eyelids, the 388 "
        f"CP-AnemiC photos with no copy (children, **Hb < {base['anemia_cutoff_g_dl']['child_6_59_months']} g/dL**); "
        f"India and Italy eyelids (Eyes-Defy adults, **WHO by sex: women < {cut['F']}, men < {cut['M']} g/dL**, "
        "pregnancy not recorded); Moscow fingernails (250 adults, **Hb < 12.0 g/dL for everyone**, no sex column).\n"
        "- **Inputs:** the 29 color numbers (`photo`); `photo + age + sex` for the eyelid sets. The hospital is never an "
        "input.\n"
        "- **Regressors:** a dummy that gives everyone the mean Hb, Stage 16's ridge, random forest and gradient "
        "boosting, and the TabPFN v2 regressor.\n"
        "- **Splits:** 5x5 CV by photo; Ghana also leaving one hospital out; Moscow also leaving one photo session out. "
        f"Bootstrap 95% CIs, {n_boot} resamples; the MAE change is paired (same resampled people).\n"
        "- **Scores:** MAE against the lab Hb (lower is better); share within 1 g/dL; ranking AUROC (people sorted by "
        "estimated Hb); anemic caught and healthy cleared when each person's own cutoff is applied to the estimate.\n"
        "- **Answer, fixed before running:** TabPFN helps at a location if its MAE minus the best of Stage 16's three "
        "regressors (same input and split) has its CI below 0.\n\n"
        "## Answer\n\n"
        + ("TabPFN estimates Hb closer than the best Stage 16 regressor in: " + "; ".join(answers) + ".\n\n" if answers
           else "TabPFN is not clearly closer than the best Stage 16 regressor anywhere.\n\n")
        + "## TabPFN against the best other regressor\n\n"
        "| location | split | input | dummy MAE | best other MAE | TabPFN MAE | change (TabPFN - best other) | "
        "best other ranking AUROC | TabPFN ranking AUROC |\n|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(summary) + "\n\n"
        "## All results\n\n"
        + markdown_table(rows, ["location", "split", "n", "anemic", "model", "input"], COLUMNS) + "\n"
    )
    (ROOT / "reports" / "stage22_hb_tabpfn.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
