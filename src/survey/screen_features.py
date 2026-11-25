"""Stage 29 Steps 3-4: how strong is each kept question on its own, and which questions repeat each other.

Role: main pipeline, Stages 29-30. Single-question screen and the repeat rule; run second.

Reads what src/survey/select_features.py saved and uses only the building villages, never the locked final test.

Step 3, one question at a time: each question splits women into groups (its answers; a number is cut into equal-count
groups, with blank and each special code as groups of their own). A woman's score is her group's anemia rate learned
on the other villages (5-fold by village), so the AUROC is honest and catches U-shapes a straight line would miss. The
extra target gets the same treatment: her group's mean Hb, scored as the error in g/dL against guessing the overall mean.
Nothing is dropped in Step 3.

Step 4: a question repeats one already kept when the kept answer nearly fixes it (Goodman-Kruskal lambda among women
who answered both). One-way on purpose, so a question is only dropped when it adds almost nothing new. Questions are
visited from fewest blanks to most, then strongest in Step 3 first.

Run from the repo root: python -m src.survey.screen_features   (after python -m src.survey.select_features)
"""
import numpy as np
import pandas as pd

from src.common.io import load_config, save_results
from src.common.report import short_commit as _short_commit
from src.common.splits import repeated_stratified_grouped
from src.survey.select_features import paths

TARGETS = ["anemic", "hb", "hb_cutoff", "cluster", "locked_test"]


def group_codes(feats, code, n_bins):
    """Integer group per woman for one question (blank is a group of its own), and which women left it blank."""
    x = feats[code]
    if pd.api.types.is_numeric_dtype(x):
        key = pd.qcut(x, n_bins, labels=False, duplicates="drop").astype("Int64").astype(str).where(x.notna())
        if f"{code}_special" in feats:
            special = feats[f"{code}_special"]
            key = key.where(special.isna() | (special == "value"), "special: " + special.astype(str))
    else:
        key = x
    blank = key.isna().to_numpy()
    return pd.factorize(key.fillna("blank"))[0], blank


def oof_group_mean(codes, target, folds, smoothing):
    """Each woman gets her group's mean target from the other folds, pulled toward the overall mean."""
    out = np.full(len(target), np.nan)
    for _, train, test in folds:
        prior = target[train].mean()
        s = np.bincount(codes[train], target[train], minlength=codes.max() + 1)
        n = np.bincount(codes[train], minlength=codes.max() + 1)
        out[test] = ((s + smoothing * prior) / (n + smoothing))[codes[test]]
    return out


def cluster_weights(groups, n_resamples, seed):
    """Bootstrap by village: row weight = how many times its village was drawn. Shape (n_resamples, n)."""
    ids, idx = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)), size=n_resamples)
    return counts[:, idx].astype(float)


def weighted_auroc(y, score, weights):
    """AUROC under each row of `weights` (ties count half). weights: (n_resamples, n) or (n,). Exact for all-ones."""
    w = np.atleast_2d(weights)
    _, level = np.unique(score, return_inverse=True)
    onehot = np.zeros((len(score), level.max() + 1))
    onehot[np.arange(len(score)), level] = 1
    pos, neg = (w * y) @ onehot, (w * (1 - y)) @ onehot           # (n_resamples, n_levels), levels low to high
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def interval(values, ci):
    a = (1 - ci) / 2
    return float(np.quantile(values, a)), float(np.quantile(values, 1 - a))


def fixed_by(other, given, min_overlap):
    """Goodman-Kruskal lambda: share of the mistakes in guessing `other` that knowing `given` removes, among rows where
    both are answered (code >= 0). 1 = `given` fixes `other` completely. None if fewer than `min_overlap` rows."""
    both = (other >= 0) & (given >= 0)
    if both.sum() < min_overlap:
        return None
    o, g = pd.factorize(other[both])[0], pd.factorize(given[both])[0]
    ko, kg = o.max() + 1, g.max() + 1
    table = np.bincount(g * ko + o, minlength=kg * ko).reshape(kg, ko)
    base_mistakes = both.sum() - table.sum(0).max()               # always guess the most common answer
    if base_mistakes == 0:                                         # `other` has one answer here: nothing to guess
        return 1.0
    mistakes = both.sum() - table.max(1).sum()                     # guess the most common answer within each group
    return float(1 - mistakes / base_mistakes)


def redundant_drops(codes, order, threshold, min_overlap):
    """Walk the questions in `order`; drop one if a question already kept fixes it. Returns {dropped: (kept, lambda)}."""
    kept, drops = [], {}
    for c in order:
        for k in kept:
            v = fixed_by(codes[c], codes[k], min_overlap)
            if v is not None and v >= threshold:
                drops[c] = (k, v)
                break
        else:
            kept.append(c)
    return drops


def main(country=None):
    base, cfg = load_config("default"), load_config("feature_selection")
    country = country or cfg["country"]
    p = paths(country)
    sc, seed, boot = cfg["screening"], base["seed"], base["bootstrap"]
    data = pd.read_parquet(p["out"])
    data = data[~data.locked_test].reset_index(drop=True)          # building villages only
    inv = pd.read_csv(p["inventory"]).set_index("code")
    feats = data.drop(columns=TARGETS)
    questions = list(inv.index[inv.decision == "keep"])
    y, hb, groups = data.anemic.to_numpy(), data.hb.to_numpy(), data.cluster.to_numpy()
    folds = repeated_stratified_grouped(y, groups, sc["cv_splits"], 1, seed)
    w = cluster_weights(groups, boot["n_resamples"], seed)
    print(f"building villages: {len(y)} women, {len(set(groups))} villages, {len(questions)} questions", flush=True)

    # Step 3
    rows, codes = [], {}
    for c in questions:
        code, blank = group_codes(feats, c, sc["number_bins"])
        risk = oof_group_mean(code, y.astype(float), folds, sc["smoothing"])
        aucs = weighted_auroc(y, risk, w)
        hb_hat, hb_mean = oof_group_mean(code, hb, folds, sc["smoothing"]), oof_group_mean(np.zeros_like(code), hb, folds, 0)
        gain = np.abs(hb - hb_mean) - np.abs(hb - hb_hat)          # g/dL of error removed per woman
        gains = w @ gain / w.sum(1)
        lo, hi = interval(aucs, boot["ci"])
        glo, ghi = interval(gains, boot["ci"])
        rows.append({"code": c, "label": inv.label[c], "share_empty": inv.share_empty[c], "kind": inv.kind[c],
                     "auroc": weighted_auroc(y, risk, np.ones(len(y)))[0], "auroc_lo": lo, "auroc_hi": hi,
                     "hb_error_cut": float(gain.mean()), "hb_error_cut_lo": glo, "hb_error_cut_hi": ghi})
        codes[c] = np.where(blank, -1, code)                       # Step 4 compares answered women only
    uni = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    print(uni.head(15)[["code", "label", "auroc", "auroc_lo", "auroc_hi", "hb_error_cut"]].to_string(), flush=True)

    # Step 4
    order = uni.sort_values(["share_empty", "auroc"], ascending=[True, False]).code.tolist()
    drops = redundant_drops(codes, order, sc["redundancy_lambda"], sc["redundancy_min_overlap"])
    uni["step4"] = ["drop: repeats " + drops[c][0] if c in drops else "keep" for c in uni.code]
    uni.to_csv(p["univariate"], index=False)
    survivors = uni[uni.step4 == "keep"]
    print(f"step 4: {len(drops)} repeats dropped, {len(survivors)} questions left", flush=True)

    path = save_results(p["screen_results"], {
        "n_women": len(y), "n_villages": len(set(groups)), "n_questions": len(questions),
        "univariate": uni.to_dict("records"), "redundant": {c: {"repeats": k, "lambda": v} for c, (k, v) in drops.items()},
        "n_left": len(survivors)}, {"default": base, "feature_selection": cfg})

    fmt = lambda r: f"{r.auroc:.3f} [{r.auroc_lo:.3f}, {r.auroc_hi:.3f}]"
    gfmt = lambda r: f"{r.hb_error_cut:+.3f} [{r.hb_error_cut_lo:+.3f}, {r.hb_error_cut_hi:+.3f}]"
    above = uni[uni.auroc_lo > 0.5]
    bands = [(0.60, 1.0), (0.55, 0.60), (0.52, 0.55), (0.0, 0.52)]
    text = "\n".join([
        ("# Stage 29 Steps 3-4" if country == "zimbabwe" else f"# Stage 30, {country.title()}: Stage 29 Steps 3-4")
        + ": each question on its own, and questions that repeat each other",
        "",
        f"Generated by `python -m src.survey.screen_features` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`screening`). **Building villages only:** "
        f"{len(y)} women in {len(set(groups))} villages; the locked final test is not touched. Anemic = DHS `v457` "
        "(WHO for women on adjusted Hb: < 12.0 g/dL, < 11.0 if pregnant). Extra target: adjusted Hb `v456`.",
        "",
        "## Step 3: how strong is each question on its own?",
        "",
        "- **Score:** a woman's group's anemia rate (her answer; numbers cut into "
        f"{sc['number_bins']} equal-count groups, blank and special codes as their own groups), learned on the other "
        f"villages ({sc['cv_splits']}-fold by village), pulled toward the overall rate as if each group had "
        f"{sc['smoothing']} more average women. 0.5 = no better than chance.",
        "- **Hb error cut:** the same groups' mean Hb as an estimate; how much it lowers the average error (g/dL) "
        "compared with guessing everyone's Hb as the overall mean. Positive = helps.",
        f"- Bootstrap 95% CIs, {boot['n_resamples']} resamples of whole villages.",
        "",
        f"**{len(above)} of {len(uni)} questions** beat chance on their own (CI above 0.5). How strong they are:",
        "",
        "| AUROC on its own | questions |",
        "|---|---|",
        *[f"| {lo:.2f}-{hi:.2f} | {int(((uni.auroc >= lo) & (uni.auroc < hi)).sum())} |" for lo, hi in bands],
        "",
        "**The 30 strongest:**",
        "",
        "| # | code | question | share blank | AUROC alone | Hb error cut (g/dL) |",
        "|---|---|---|---|---|---|",
        *[f"| {i + 1} | `{r.code}` | {r.label} | {r.share_empty:.0%} | {fmt(r)} | {gfmt(r)} |"
          for i, r in enumerate(uni.head(30).itertuples())],
        "",
        f"Every question's numbers are in `reports/{p['univariate'].name}`.",
        "",
        "## Step 4: questions that repeat each other",
        "",
        "A question is dropped when a question already kept nearly fixes its answer: among women who answered both "
        f"(at least {sc['redundancy_min_overlap']}), knowing the kept answer removes at least "
        f"{sc['redundancy_lambda']:.0%} of the mistakes in guessing it (Goodman-Kruskal lambda). It is one-way, so a "
        "dropped question adds almost nothing the kept one doesn't already say. Questions are visited from fewest "
        "blanks to most, then strongest in Step 3 first.",
        "",
        f"**{len(drops)} dropped, {len(survivors)} left** for Step 5.",
        "",
        "| dropped | fixed by (kept) | mistakes removed |",
        "|---|---|---|",
        *[f"| `{c}` {inv.label[c]} | `{k}` {inv.label[k]} | {v:.0%} |" for c, (k, v) in drops.items()],
        "",
    ])
    p["screen_report"].write_text(text)


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
