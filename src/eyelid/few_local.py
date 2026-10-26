"""Stage 15: how many labelled local photos does a new country need?

Stage 14: a model trained on two countries doesn't beat the redness rule on the third. Here a clinic in the new country
blood-tests a few people. For each target country its photos are split 5x5 by photo; k labelled photos are drawn from
the training part, and a model is trained (a) on the other two countries plus those k photos, each country standardized,
or (b) on the k photos alone. Both are scored on the held-out photos of the target country, next to the rule, with the
paired AUROC change against the rule and between (a) and (b).

Run from the repo root: python -m src.eyelid.few_local
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import feature_table
from src.eyelid.debias import fit_predict, group_standardize
from src.eyelid.report import fmt, markdown_table
from src.eyelid.three_countries import COUNTRIES, load_photos
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

MIXED, LOCAL = "other two + k local", "k local only"


def local_sample(y, pool, k, seed):
    """k rows of pool drawn at random, stratified by label; the whole pool for k = 'all'."""
    if k == "all":
        return pool
    return pool[:0] if k == 0 else train_test_split(pool, train_size=k, stratify=y[pool], random_state=seed)[0]


def main():
    base, cfg, debias_cfg = load_config("default"), load_config("few_local"), load_config("debias")
    color_cfg, probe_cfg, ceiling_cfg = load_config("color"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    n_splits, n_repeats, ks = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], cfg["local_photos"]

    df, y, country, code, early = load_photos(ceiling_cfg, debias_cfg)
    feats = feature_table([ROOT / p for p in df.image_path], df.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    X, rule = feats.to_numpy(), -feats[color_cfg["rule_feature"]].to_numpy()
    models = {n: make for n, (kind, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if kind == "color" and n != "dummy"}

    rows = []
    for target in COUNTRIES:
        t, others = np.flatnonzero(country == target), np.flatnonzero(country != target)
        s = {}
        for i, (r, tr, te) in enumerate(repeated_stratified(y[t], n_splits, n_repeats, seed)):
            pool, test = t[tr], t[te]
            stats = np.concatenate([others, pool])  # unlabelled: each country's own mean and SD, never the test photos
            Z = lambda rows_: group_standardize(X[stats], code[stats], X[rows_], code[rows_])
            for k in ks:
                local = local_sample(y, pool, k, seed + i)
                train = np.concatenate([others, local])
                for name, make in models.items():
                    s.setdefault((MIXED, k, name), np.full((n_repeats, len(t)), np.nan))[r, te] = \
                        fit_predict(make, Z(train), y[train], Z(test))
                    if k != 0:
                        s.setdefault((LOCAL, k, name), np.full((n_repeats, len(t)), np.nan))[r, te] = \
                            fit_predict(make, X[local], y[local], X[test])
            print(f"done: {target}, fold {i + 1}/{n_splits * n_repeats}", flush=True)
        assert not any(np.isnan(v).any() for v in s.values())

        test_sets = {target: np.ones(len(t), bool)} | ({"Italy early batch": early[t]} if target == "Italy" else {})
        for subset, m in test_sets.items():
            yy, rr = y[t][m], rule[t][m]
            head = {"target": target, "test set": subset, "n": int(m.sum()), "anemic": f"{yy.mean():.1%}"}
            rows.append(head | {"local photos": "-", "trained on": "no training", "model": "dummy (prior)",
                                **summarize(yy, np.full(len(yy), yy.mean()), n_boot, ci, seed, metrics=METRICS)})
            rows.append(head | {"local photos": "-", "trained on": "no training", "model": "rule_low_redness",
                                **summarize(yy, rr, n_boot, ci, seed, metrics=METRICS)})
            for (how, k, name), sc in s.items():
                row = head | {"local photos": k, "trained on": how, "model": name,
                              **summarize(yy, sc[:, m], n_boot, ci, seed, metrics=METRICS),
                              "vs_rule": auroc_change(yy, sc[:, m], rr, n_boot, ci, seed)}
                if how == MIXED and k != 0:
                    row["vs_local"] = auroc_change(yy, sc[:, m], s[(LOCAL, k, name)][:, m], n_boot, ci, seed)
                rows.append(row)

    path = save_results("stage15_few_local", {"rows": rows}, {"default": base, "few_local": cfg, "debias": debias_cfg,
                                                              "color": color_cfg, "ceiling": ceiling_cfg})
    subsets = list(dict.fromkeys(r["test set"] for r in rows))

    def best(subset, how, k):
        return max((r for r in rows if r["test set"] == subset and r["trained on"] == how and r["local photos"] == k),
                   key=lambda r: r["auroc"]["estimate"], default=None)

    answer, first = [], {}
    for subset in subsets:
        rule_row = next(r for r in rows if r["test set"] == subset and r["model"] == "rule_low_redness")
        for k in ks:
            a, b = best(subset, MIXED, k), best(subset, LOCAL, k)
            beats = [r["model"] + f" ({r['trained on']})" for r in rows if r["test set"] == subset
                     and r["local photos"] == k and r["vs_rule"]["lo"] > 0]
            if beats and subset not in first:
                first[subset] = k
            answer.append(f"| {subset} | {k} | {fmt(a['auroc'])} ({a['model']}) | "
                          + (f"{fmt(b['auroc'])} ({b['model']})" if b else "-")
                          + f" | {fmt(a['vs_local']) if 'vs_local' in a else '-'} | {fmt(rule_row['auroc'])} | "
                          + (", ".join(beats) or "none") + " |")
    found = "; ".join(f"{sub}: {first[sub] if sub in first else 'no k'}" for sub in subsets)
    report = (
        "# Stage 15: how many local photos does a new country need?\n\n"
        f"Generated by `python -m src.eyelid.few_local` at commit `{_short_commit(path)}`; settings in "
        "`configs/few_local.yaml`. A model trained on two countries doesn't beat the redness rule on the third (Stage "
        "14). If a clinic in the new country blood-tests k people and labels their photos, how large must k be?\n\n"
        f"- **Photos:** the {len(y)} of Stage 14, one per person. Labels mixed by design: Ghana children Hb < 11.0 g/dL; "
        "India and Italy adults WHO by sex (women < 12.0, men < 13.0 g/dL; pregnancy not recorded).\n"
        "- **Input:** the 29 color numbers of the hand-segmented palpebral conjunctiva only.\n"
        f"- **Split:** for each target country, {n_splits}-fold CV x {n_repeats} repeats by photo, stratified by anemia; "
        f"k = {', '.join(map(str, ks))} labelled photos drawn at random (stratified) from the training part; scores "
        f"averaged over repeats. Bootstrap 95% CIs, {n_boot} resamples.\n"
        f"- **{MIXED}:** the other two countries plus the k photos, each country standardized with its own mean and SD "
        "(the new country's from its unlabelled training photos, never the test photos). k = 0 is Stage 14's "
        "train-on-two, with stricter standardization.\n"
        f"- **{LOCAL}:** the k photos alone, no other country.\n"
        "- **AUROC change:** paired bootstrap (same resampled photos) against the rule, and between the two ways of "
        "training for the same model.\n"
        f"- **Italy early batch:** folders 1-{debias_cfg['italy_early_batch_last_folder']}; local Italy photos can teach "
        "a model the batch, so this is the Italy test the batch can't pass.\n"
        "- **Answer, fixed before running:** the smallest k at which a model beats the rule (CI of the change above 0), "
        "and whether the other two countries help at the same k.\n\n"
        f"**Smallest k that beats the rule:** {found}.\n\n"
        "## Best model at each k (AUROC)\n\n"
        f"| test set | k | best, {MIXED} | best, {LOCAL} | change from adding the other two (same model) | rule | "
        "beats the rule (CI above 0) |\n|---|---|---|---|---|---|---|\n" + "\n".join(answer) + "\n\n"
        "## All results\n\n"
        + markdown_table(rows, ["test set", "n", "anemic", "local photos", "trained on", "model"],
                         COLUMNS | {"vs_rule": "AUROC change vs rule", "vs_local": f"AUROC change vs {LOCAL}"}) + "\n"
    )
    (ROOT / "reports" / "stage15_few_local.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
