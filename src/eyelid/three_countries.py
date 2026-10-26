"""Stage 14: one eyelid model for Ghana, India and Italy together.

Train on a mix of all three (5x5 CV stratified by country x anemia) and score inside each country, where knowing the
country can't help. Shortcut checks: a country-only model, a no-color model, the Italy early batch, and how far the
scores of the three countries sit apart among people with the same anemia status. Fixes for the country and lighting
shortcuts from Stage 11: per-country standardization, reweighing, both, brightness scaling, lighting jitter. One model
per country on the same folds is the reference a mixed model has to match. Then train on two countries, test on the
third.

Run from the repo root: python -m src.eyelid.three_countries
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import repeated_stratified
from src.common.metrics import bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features, image_properties
from src.eyelid.cross_dataset import load_eyes_defy
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.debias import color_variants, fit_predict, group_standardize, reweigh
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

COUNTRIES = ("Ghana", "India", "Italy")
NO_TABPFN = {"tabpfn"}  # TabPFN's fit takes no sample weights, and jitter needs > 1000 training rows (refused on CPU)


def rank_spread(label):
    """Metric on s = [score, country code]: largest minus smallest mean percentile rank across countries, among people
    with this anemia status. 0 = once anemia status is fixed, the score doesn't depend on the country."""
    def f(y, s):
        r = rankdata(s[:, 0]) / len(s)
        means = [r[(y == label) & (s[:, 1] == c)].mean() for c in np.unique(s[:, 1])]
        return float(max(means) - min(means))
    return f


def load_photos(ceiling_cfg, debias_cfg):
    """Ghana (CP-AnemiC photos with no copy) plus Eyes-Defy India and Italy, one photo per person; returns the table,
    labels, country, country code and the Italy early batch flag."""
    cp = load_cp_anemic(ceiling_cfg["cp_anemic_subset"]).assign(country="Ghana")
    ed = load_eyes_defy(load_config("cross_dataset"))
    df = pd.concat([cp, ed])[["id", "image_path", "country", "anemic"]].reset_index(drop=True)
    y, country = df.anemic.to_numpy(), df.country.to_numpy()
    code = np.array([COUNTRIES.index(c) for c in country])
    italy = country == "Italy"
    early = italy.copy()
    early[italy] = (df.id[italy].str.split("_").str[1].astype(int) <= debias_cfg["italy_early_batch_last_folder"]).to_numpy()
    return df, y, country, code, early


def main():
    base, cfg, debias_cfg = load_config("default"), load_config("three_countries"), load_config("debias")
    color_cfg, probe_cfg, ceiling_cfg = load_config("color"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]

    df, y, country, code, early = load_photos(ceiling_cfg, debias_cfg)

    paths = [ROOT / p for p in df.image_path]
    feats = color_variants(paths, debias_cfg, color_cfg["roi"]["alpha_threshold"], seed)
    X, no_color, k = feats["plain"], image_properties(paths, df.id).to_numpy(), debias_cfg["jitter"]["copies"]
    rule = -X[:, list(color_features(np.full((2, 3), 0.5)).keys()).index(color_cfg["rule_feature"])]
    models = {n: make for n, (kind, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if kind == "color" and n != "dummy"}

    def mixed(Xv, weighted=False, standardized=False):
        def f(make, tr, te):
            A, B = Xv[tr], Xv[te]
            if standardized:  # each country with its own training-fold mean and SD, no labels
                A, B = group_standardize(Xv[tr], code[tr], A, code[tr]), group_standardize(Xv[tr], code[tr], B, code[te])
            return fit_predict(make, A, y[tr], B, reweigh(y[tr], code[tr]) if weighted else None)
        return f

    def own_country(make, tr, te):
        out = np.empty(len(te))
        for c in np.unique(code):
            a, b = tr[code[tr] == c], te[code[te] == c]
            out[code[te] == c] = fit_predict(make, X[a], y[a], X[b])
        return out

    def jittered(make, tr, te):
        X_tr = np.vstack([X[tr], feats["jittered"][tr].reshape(-1, X.shape[1])])
        return fit_predict(make, X_tr, np.concatenate([y[tr], np.repeat(y[tr], k)]), X[te])

    methods = [("own country only (reference)", own_country, set()),
               ("none: one mixed model", mixed(X), set()),
               ("per-country standardized", mixed(X, standardized=True), set()),
               ("reweighing", mixed(X, weighted=True), NO_TABPFN),
               ("per-country standardized + reweighing", mixed(X, True, True), NO_TABPFN),
               ("brightness scaled", mixed(feats["brightness"]), set()),
               ("lighting jitter", jittered, NO_TABPFN)]

    folds = repeated_stratified(2 * code + y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)

    def out_of_fold(score_fold):
        s = np.full((cfg["cv"]["n_repeats"], len(y)), np.nan)
        for r, tr, te in folds:
            s[r, te] = score_fold(tr, te)
        assert not np.isnan(s).any()
        return s.mean(0)

    C = country.reshape(-1, 1)
    country_only = lambda: make_pipeline(OneHotEncoder(), LogisticRegression(max_iter=5000, random_state=seed))
    scores = {("control", "dummy (prior)"): np.full(len(y), y.mean()),
              ("control", "country only"): out_of_fold(lambda tr, te: fit_predict(country_only, C[tr], y[tr], C[te])),
              ("control", "no color: size and shape (tabpfn)"):
                  out_of_fold(lambda tr, te: fit_predict(models["tabpfn"], no_color[tr], y[tr], no_color[te])),
              ("control", "rule_low_redness (no training)"): rule}
    for method, fn, skip in methods:
        for name, make in models.items():
            if name not in skip:
                scores[(method, name)] = out_of_fold(lambda tr, te: fn(make, tr, te))
                print(f"done: {method}, {name}", flush=True)

    subsets = {"all": np.ones(len(y), bool), **{c: country == c for c in COUNTRIES}, "Italy early batch": early}
    rows, summary = [], []
    for (method, model), s in scores.items():
        per = {}
        for subset, m in subsets.items():
            res = summarize(y[m], s[m], n_boot, ci, seed, metrics=METRICS)
            rows.append({"method": method, "model": model, "test set": subset, "n": int(m.sum()),
                         "anemic": f"{y[m].mean():.1%}", **res})
            per[subset] = res["auroc"]
        packed = np.column_stack([s, code])
        summary.append({"method": method, "model": model, "auroc": per,
                        "worst_country": min(per[c]["estimate"] for c in COUNTRIES),
                        "rank_spread_healthy": bootstrap_ci(rank_spread(0), y, packed, n_boot, ci, seed=seed),
                        "rank_spread_anemic": bootstrap_ci(rank_spread(1), y, packed, n_boot, ci, seed=seed)})

    transfer = []
    for target in COUNTRIES:
        tr, te = np.flatnonzero(country != target), np.flatnonzero(country == target)
        A_std = group_standardize(X[tr], code[tr], X[tr], code[tr])
        B_std = group_standardize(X[te], code[te], X[te], code[te])  # the new country's own unlabelled statistics
        cand = {("control", "rule_low_redness (no training)"): rule[te]}
        for name, make in models.items():
            cand[("none", name)] = fit_predict(make, X[tr], y[tr], X[te])
            cand[("each country standardized", name)] = fit_predict(make, A_std, y[tr], B_std)
        test_sets = {"all": np.ones(len(te), bool)} | ({"Italy early batch": early[te]} if target == "Italy" else {})
        for subset, m in test_sets.items():
            for (method, model), s in cand.items():
                transfer.append({"train -> test": " + ".join(c for c in COUNTRIES if c != target) + f" -> {target}",
                                 "test set": subset, "n": int(m.sum()), "anemic": f"{y[te][m].mean():.1%}",
                                 "method": method, "model": model,
                                 **summarize(y[te][m], s[m], n_boot, ci, seed, metrics=METRICS)})
        print(f"done: train on two -> {target}", flush=True)

    path = save_results("stage14_three_countries", {"n": len(y), "summary": summary, "rows": rows, "transfer": transfer},
                        {"default": base, "three_countries": cfg, "debias": debias_cfg, "color": color_cfg,
                         "ceiling": ceiling_cfg})
    head = ["method", "model", "pooled AUROC", *COUNTRIES, "Italy early batch", "worst country",
            "rank spread, healthy", "rank spread, anemic"]
    lines = [f"| {r['method']} | {r['model']} | " + " | ".join(fmt(r["auroc"][s]) for s in subsets)
             + f" | {r['worst_country']:.3f} | {fmt(r['rank_spread_healthy'])} | {fmt(r['rank_spread_anemic'])} |"
             for r in summary]
    counts = ", ".join(f"{c} {int((country == c).sum())} ({y[country == c].mean():.1%} anemic)" for c in COUNTRIES)
    report = (
        "# Stage 14: one eyelid model for Ghana, India and Italy\n\n"
        f"Generated by `python -m src.eyelid.three_countries` at commit `{_short_commit(path)}`; settings in "
        "`configs/three_countries.yaml`. Can one model trained on all three countries find anemia inside each of them, "
        "without guessing the country, the lighting or the photo batch?\n\n"
        f"- **Photos:** {len(y)}: {counts}. One photo per person.\n"
        "- **Labels, mixed by design:** Ghana children Hb < 11.0 g/dL; India and Italy adults WHO by sex (women < 12.0, "
        "men < 13.0 g/dL; pregnancy not recorded).\n"
        "- **Input:** the 29 color numbers of the hand-segmented palpebral conjunctiva only (no skin, no background).\n"
        f"- **Split:** {cfg['cv']['n_splits']}-fold CV x {cfg['cv']['n_repeats']} repeats by photo, stratified by "
        "country x anemia, scores averaged over repeats. Bootstrap 95% CIs, "
        f"{n_boot} resamples. AUROC inside a country can't be helped by knowing the country.\n"
        f"- **Italy early batch:** folders 1-{debias_cfg['italy_early_batch_last_folder']} ({int(early.sum())} photos, "
        f"{int(y[early].sum())} anemic), the Italy test that recognizing the batch can't pass (Stage 10).\n"
        "- **Controls:** `country only` = what knowing the country gives (constant inside a country, so 0.5 there); "
        "`no color` = TabPFN on image size and cut-out shape (near 0.5 inside a country = no leak there).\n"
        "- **Fixes:** per-country standardized = each country's numbers with its own training-fold mean and SD, no "
        "labels; reweighing = country and anemia made independent in training (Kamiran and Calders); brightness scaled "
        "= every photo scaled to the same mean brightness; lighting jitter = 5 brightness/contrast-jittered copies of "
        "each training photo. TabPFN skips reweighing and jitter (no sample weights; > 1000 training rows on CPU).\n"
        "- **Rank spread:** largest minus smallest mean percentile rank across the three countries, among people with "
        "the same anemia status. 0 = the score doesn't depend on the country once anemia status is fixed.\n"
        "- **Pass, fixed before running:** inside every country, beat the rule and match the country's own model (same "
        "folds), with a smaller healthy rank spread than the plain mixed model.\n\n"
        "## Inside each country (AUROC)\n\n"
        "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join(lines) + "\n\n"
        "## Train on two countries, test on the third\n\n"
        "Fit on all photos of two countries, applied unchanged to the third. `each country standardized` uses each "
        "country's own unlabelled means and SDs, the new country's included.\n\n"
        + markdown_table(transfer, ["train -> test", "test set", "n", "anemic", "method", "model"], COLUMNS) + "\n\n"
        "## All metrics, mixed training\n\n"
        + markdown_table(rows, ["method", "model", "test set", "n", "anemic"], COLUMNS) + "\n"
    )
    (ROOT / "reports" / "stage14_three_countries.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
