"""Stage 27: does the eyelid model catch the sickest children, or does it miss at random?

One AUROC treats a child at Hb 10.9 and a child at Hb 6 the same. Here the Ghana children are split into WHO
severity bands and, at one threshold set on all of them, we count the share of each band that gets flagged.
Settings in configs/severity.yaml.

Run from the repo root: python -m src.eyelid.severity
"""
import numpy as np
from sklearn.metrics import roc_curve

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames, population_inputs
from src.eyelid.report import fmt
from src.eyelid.train_color import _short_commit
from src.ghana_checks import shared_threshold


def band_index(hb, bands):
    """Position in `bands` (name -> [low, high), g/dL) of each Hb value; every value must fall in one band."""
    edges = list(bands.values())
    idx = np.full(len(hb), -1)
    for k, (lo, hi) in enumerate(edges):
        idx[(hb >= lo) & (hb < hi)] = k
    assert (idx >= 0).all(), "an Hb value falls outside every band"
    return idx


def threshold(y, s, kind, level):
    """Flag when score >= threshold. 'specificity': clears at least `level` of the healthy; 'sensitivity': the
    highest threshold that catches at least `level` of the anemic."""
    if kind == "specificity":
        return shared_threshold(y, s, level)
    _, tpr, thr = roc_curve(y, s)
    return thr[np.flatnonzero(tpr >= level - 1e-12)[0]]


def flagged_share(kind, level, members):
    """Metric for bootstrap_ci: share of the rows with band in `members` flagged at the threshold set on all rows.
    ss columns: score, band index."""
    def metric(yy, ss):
        m = np.isin(ss[:, 1], members)
        if not m.any():
            raise ValueError("no row of this band in the resample")
        return float((ss[m, 0] >= threshold(yy, ss[:, 0], kind, level)).mean())
    return metric


def main():
    base, cfg, ceiling = load_config("default"), load_config("severity"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    df, x = population_inputs(population_frames(ceiling), color_cfg, probe_cfg, ceiling)["cp_anemic"]
    y, hb = df.anemic.to_numpy(), df.hb.to_numpy()
    names = list(cfg["bands"])
    band = band_index(hb, cfg["bands"])
    assert ((hb < base["anemia_cutoff_g_dl"]["child_6_59_months"]) == (y == 1)).all()
    anemic_bands = [k for k in range(len(names)) if y[band == k].all()]
    mild, worse = names.index("mild"), [k for k in anemic_bands if names[k] != "mild"]

    folds = repeated_stratified(y, ceiling["cv"]["n_splits"], ceiling["cv"]["n_repeats"], seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling, seed)
    scores = {"rule_low_redness": x["rule"]}
    for m in cfg["models"]:
        kind, make = ladder[m]
        scores[m] = fit_and_score(make, x[kind], y, folds)[0].mean(0)
        print(f"done: {m} {auroc(y, scores[m]):.3f}", flush=True)

    points = {"specificity": cfg["specificity"], "sensitivity": cfg["sensitivity"]}
    rows, gaps = [], []
    for model, s in scores.items():
        packed = np.column_stack([s, band])
        for k, name in enumerate(names):
            row = {"model": model, "band": name, "hb": f"{cfg['bands'][name][0]:g}-{cfg['bands'][name][1]:g}",
                   "n": int((band == k).sum())}
            for kind, level in points.items():
                row[kind] = bootstrap_ci(flagged_share(kind, level, [k]), y, packed, n_boot, ci, seed=seed)
            if k in anemic_bands:  # this band against all the healthy children
                m = (band == k) | (y == 0)
                row["auroc"] = bootstrap_ci(auroc, y[m], s[m], n_boot, ci, seed=seed)
            rows.append(row)
        worse_fn, mild_fn = (flagged_share("specificity", cfg["specificity"], b) for b in (worse, [mild]))
        gaps.append({"model": model,
                     "worse": bootstrap_ci(worse_fn, y, packed, n_boot, ci, seed=seed),
                     "gap": bootstrap_ci(lambda yy, ss: worse_fn(yy, ss) - mild_fn(yy, ss), y, packed, n_boot, ci,
                                         seed=seed)})

    # missed anemic children at the specificity threshold, best model: how low is their Hb?
    best = cfg["models"][-1]
    missed = (y == 1) & (scores[best] < threshold(y, scores[best], "specificity", cfg["specificity"]))
    missed_hb = {"model": best, "n_missed": int(missed.sum()), "n_anemic": int(y.sum()),
                 "median_hb_missed": float(np.median(hb[missed])), "median_hb_caught": float(np.median(hb[(y == 1) & ~missed])),
                 "missed_by_band": {names[k]: int((missed & (band == k)).sum()) for k in anemic_bands}}

    path = save_results("stage27_severity", {"rows": rows, "gaps": gaps, "missed": missed_hb},
                        {"default": base, "severity": cfg, "ceiling": ceiling, "color": color_cfg})
    g = next(r for r in gaps if r["model"] == best)["gap"]
    verdict = "Yes" if g["lo"] > 0 else "Not shown"
    spec, sens = cfg["specificity"], cfg["sensitivity"]
    table = ["| model | band | Hb (g/dL) | n | flagged, 90% of healthy cleared | flagged, 90% of anemic caught | AUROC vs healthy |",
             "|---|---|---|---|---|---|---|"]
    table += [f"| {r['model']} | {r['band']} | {r['hb']} | {r['n']} | {fmt(r['specificity'])} | {fmt(r['sensitivity'])} | "
              f"{fmt(r['auroc']) if 'auroc' in r else '-'} |" for r in rows]
    gap_lines = [f"| {r['model']} | {fmt(r['worse'])} | {fmt(r['gap'])} |" for r in gaps]
    report = (
        "# Stage 27: does the eyelid model catch the sickest children?\n\n"
        f"Generated by `python -m src.eyelid.severity` at commit `{_short_commit(path)}`; settings in "
        "`configs/severity.yaml`.\n\n"
        "- **Children:** Ghana, CP-AnemiC, the 388 photos with no copy. **Anemic: Hb < 11.0 g/dL** (WHO, 6-59 months).\n"
        "- **Bands:** WHO severity grades for this age: severe < 7.0, moderate 7.0-9.9, mild 10.0-10.9 g/dL; healthy "
        "children split at 12.0.\n"
        "- **Scores:** held-out, 5-fold CV x 5 repeats by photo (Stage 9's folds), averaged over repeats. Models on the "
        "29 color numbers, plus the untrained redness rule.\n"
        f"- **Flagged** = sent for a blood test. Two thresholds, each set on all 388 children: one clears {spec:.0%} of "
        f"healthy children, one catches {sens:.0%} of anemic children. For anemic bands, flagged = caught; for healthy "
        f"bands, flagged = false alarm. **Baseline:** a score with no signal flags {1 - spec:.0%} of every band at the "
        f"first threshold and {sens:.0%} at the second.\n"
        f"- Bootstrap 95% CIs, {n_boot} resamples; the threshold is set again in every resample.\n"
        "- **Answer, fixed before running:** the misses are mostly mild cases if, for the best model, caught among "
        "moderate + severe minus caught among mild has its CI above 0 (first threshold).\n\n"
        f"## Answer\n\nAre the misses mostly mild cases ({best})? **{verdict}**: moderate + severe minus mild = {fmt(g)}.\n\n"
        f"At the first threshold {best} misses {missed_hb['n_missed']} of {missed_hb['n_anemic']} anemic children: "
        + ", ".join(f"{v} {k}" for k, v in missed_hb["missed_by_band"].items())
        + f". Median Hb of the missed children {missed_hb['median_hb_missed']:.1f} g/dL, of the caught "
        f"{missed_hb['median_hb_caught']:.1f} g/dL.\n\n"
        "## Share flagged in each Hb band\n\n" + "\n".join(table)
        + "\n\n## Moderate + severe against mild (first threshold)\n\n"
        "| model | caught, moderate + severe | minus caught, mild |\n|---|---|---|\n" + "\n".join(gap_lines) + "\n"
    )
    (ROOT / "reports" / "stage27_severity.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
