"""Stage 24: one Ghanaian child's eyelid, nail and palm photos together (mentor suggestion, 2026-11-03 meeting).

Stage 13 scored the Ghana palm and nail sets side by side with the eyelids, because nothing linked a child across them.
A third set from the same group (Mendeley 10.17632/nt7r8hv2pz.1) holds CP-AnemiC eyelid photos pixel for pixel, but
names them like the nail and palm sets: label, batch and number (Anemic-105, Anemic-FN-105, AnemicP-105). So:
1. Test the link: photos with the same number should agree in color more than shuffled partners, and more than
   neighbouring numbers (which share a photo session).
2. Take Hb and hospital from the CP-AnemiC sheet through the pixel match.
3. Score each body site alone and combined (early: one model on all the color numbers; late: the mean of the per-site
   scores), and compare the real children with pretend ones (a random nail and palm with the same anemia label).

Run from the repo root: python -m src.ghana_link
"""
import hashlib
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata
from sklearn.dummy import DummyRegressor

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import bootstrap_ci
from src.common.splits import leave_one_group_out, repeated_stratified_grouped
from src.eyelid.audit import photo_groups
from src.eyelid.ceiling import fit_and_score, ladder_models
from src.eyelid.color_features import feature_table
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.hb_regression import mae, oof_regression, regressors
from src.eyelid.metadata import mae_change
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import _short_commit
from src.nails import COLUMNS, METRICS

SITES = ("eyelid", "nail", "palm")
KEY = ["anemic_name", "batch", "num"]
ALL3 = "eyelid + nail + palm"
COMBOS = {"eyelid + nail": ("eyelid", "nail"), "eyelid + palm": ("eyelid", "palm"), ALL3: SITES}


def pixel_key(path):
    """The same key for a photo and its exact copies rotated by 90, 180 or 270 degrees or mirrored."""
    a = np.asarray(Image.open(path).convert("RGBA"))
    views = (np.ascontiguousarray(np.rot90(b, k)) for b in (a, a[:, ::-1]) for k in range(4))
    return min(hashlib.sha256(v.tobytes() + str(v.shape).encode()).hexdigest() for v in views)


def parse(stem, patterns, copy_suffix):
    """'Anemic-FN-105 (3)' -> (1, 'main', 105) with the nail patterns; None when no pattern fits the whole name."""
    name = re.sub(copy_suffix, "", stem).strip()
    for p in patterns:
        m = re.fullmatch(p["pattern"], name)
        if m:
            return p["anemic"], p["batch"], int(m.group(1))
    return None


def numbered(folder, patterns, copy_suffix, near):
    """One row per (label, batch, number) holding exactly one distinct photo that no other number holds; and counts."""
    files = sorted((ROOT / folder).glob("*.png"))
    df = pd.DataFrame({"path": files, "parsed": [parse(f.stem, patterns, copy_suffix) for f in files]})
    df["image_group"] = df.path.map(pixel_key)
    # shortcut: near-copies matched in stored orientation only (as Stage 13); they only merge CV groups here
    df["photo_group"], _ = photo_groups(df, near["dhash_max_bits"], near["max_size_diff"])
    named = df[df.parsed.notna()].copy()
    named[KEY] = pd.DataFrame(named.parsed.tolist(), index=named.index)
    one_photo = named.groupby(KEY).image_group.transform("nunique") == 1
    one_number = named.groupby("image_group").parsed.transform("nunique") == 1
    kept = named[one_photo & one_number].drop_duplicates(KEY).set_index(KEY).sort_index()
    counts = {"files": len(df), "distinct photos": int(df.image_group.nunique()),
              "files left out by name": int(df.parsed.isna().sum()), "numbers kept": len(kept)}
    return kept[["path", "image_group", "photo_group"]], counts


def cp_lookup(keys):
    """CP-AnemiC by pixel key: Hb, age and sex when every copy of the photo agrees on them; the hospital only for a photo with no copy
    (the only rows whose hospital we can trust, as in Stage 7); the photo group for the CV folds."""
    cp = load_cp_anemic("all")
    cp["key"] = [pixel_key(ROOT / p) for p in cp.image_path]
    out = {}
    for key, groups in cp[cp.key.isin(set(keys))].groupby("key").photo_group:
        rows = cp[cp.photo_group.isin(set(groups))]
        agreed = {c: rows[c].iloc[0] if rows[c].nunique() == 1 else np.nan for c in ("hb", "age_months", "sex")}
        out[key] = agreed | {"hospital": rows.site.iloc[0] if len(rows) == 1 else None,
                             "cp_group": rows.photo_group.min()}
    return pd.DataFrame.from_dict(out, orient="index")


def within_ranks(values, groups):
    """Each column ranked inside each group and scaled to 0-1, so labels and batches can't drive the agreement."""
    out = np.empty(values.shape, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] = (rankdata(values[m], axis=0) - 0.5) / m.sum()
    return out


def agreement(xr, yr):
    """Mean over the color numbers of the correlation between two body sites' within-group ranks."""
    return float(np.mean([np.corrcoef(xr[:, j], yr[:, j])[0, 1] for j in range(xr.shape[1])]))


def partner_order(groups, numbers, rng=None, block=None, offset=0):
    """Rows of the second body site to pair with each row: shuffled inside each group (block None), shuffled only among
    `block` neighbouring numbers, or, without rng, shifted `offset` places along the numbers."""
    order = np.arange(len(groups))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(numbers[idx], kind="stable")]
        if rng is None:
            order[idx] = np.roll(idx, offset)
        elif block is None:
            order[idx] = rng.permutation(idx)
        else:
            start = rng.integers(block)  # random block edges each time
            for lo in range(-start, len(idx), block):
                part = idx[max(lo, 0):lo + block]
                order[part] = rng.permutation(part)
    return order


def link_test(x, y, groups, numbers, cfg, rng):
    """Same-number color agreement against shuffled partners; p = share of shuffles agreeing at least as much."""
    xr, yr = within_ranks(x, groups), within_ranks(y, groups)
    real = agreement(xr, yr)
    out = {"n": len(groups), "same number": real}
    for name, block in [("shuffled in batch", None)] + [(f"shuffled among {b} neighbours", b) for b in cfg["block_sizes"]]:
        null = np.array([agreement(xr, yr[partner_order(groups, numbers, rng, block)]) for _ in range(cfg["n_shuffles"])])
        out[name] = {"mean": float(null.mean()), "q95": float(np.quantile(null, 0.95)),
                     "p": float((1 + (null >= real).sum()) / (1 + len(null)))}
    out["offsets"] = {str(d): agreement(xr, yr[partner_order(groups, numbers, offset=d)]) for d in cfg["offsets"]}
    out["links"] = all(v["p"] < cfg["p_max"] for k, v in out.items() if k.startswith("shuffled"))
    return out


def shared_groups(*columns):
    """One id per child; children sharing a photo (or a near-copy) at any body site get the same id, so one fold."""
    parent = list(range(len(columns[0])))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in columns:
        first = {}
        for i, v in enumerate(col):
            parent[find(i)] = find(first.setdefault(v, i))
    return np.array([find(i) for i in range(len(parent))])


def same_label_shuffle(y, rng):
    """A row order that moves each child's partner only to another child with the same anemia label."""
    order = np.arange(len(y))
    for v in np.unique(y):
        idx = np.flatnonzero(y == v)
        order[idx] = rng.permutation(idx)
    return order


def rank_mean(columns):
    """No-training combination: the mean of each score's rank (0-1)."""
    return np.mean([(rankdata(c) - 0.5) / len(c) for c in columns], axis=0)


def load_sites(cfg, base, alpha):
    """Each body site's numbered photos, their counts, and their 29 color numbers (rows: label, batch, number)."""
    tabs, counts, feats = {}, {}, {}
    for site in SITES:
        tabs[site], counts[site] = numbered(cfg["data"][site], cfg["names"][site], cfg["copy_suffix"],
                                            base["cp_anemic_near_duplicate"])
        feats[site] = feature_table(tabs[site].path, range(len(tabs[site])), alpha_threshold=alpha).set_axis(
            tabs[site].index)
        print(f"loaded {site}: {counts[site]}", flush=True)
    return tabs, counts, feats


def children(tabs):
    """Eyelid rows joined to CP-AnemiC; the children (numbers with an agreed Hb and a nail and palm photo); and one
    CV group per child, shared by children that share any photo."""
    eye = tabs["eyelid"].join(cp_lookup(tabs["eyelid"].image_group), on="image_group")
    ids = eye.index[eye.hb.notna()].intersection(tabs["nail"].index).intersection(tabs["palm"].index)
    groups = shared_groups(eye.loc[ids, "cp_group"].to_numpy(), tabs["nail"].loc[ids, "photo_group"].to_numpy(),
                           tabs["palm"].loc[ids, "photo_group"].to_numpy())
    return eye, ids, groups


def main():
    base, cfg, color_cfg = load_config("default"), load_config("ghana_link"), load_config("color")
    probe_cfg, ceiling_cfg, hb_cfg = load_config("probe"), load_config("ceiling"), load_config("hb_regression")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    cutoff = base["anemia_cutoff_g_dl"]["child_6_59_months"]
    rng = np.random.default_rng(seed)
    alpha, rule_feature = color_cfg["roi"]["alpha_threshold"], color_cfg["rule_feature"]
    models = {n: make for n, (kind, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if kind == "color" and n != "dummy"}

    # 1. the link
    tabs, counts, feats = load_sites(cfg, base, alpha)
    links = {}
    for a, b in [("eyelid", "nail"), ("eyelid", "palm"), ("nail", "palm")]:
        common = feats[a].index.intersection(feats[b].index)
        groups = np.array([f"{lab}-{batch}" for lab, batch, _ in common])
        numbers = common.get_level_values("num").to_numpy()
        f = cfg["link_test"]["features"]
        links[f"{a} - {b}"] = link_test(feats[a].loc[common, f].to_numpy(), feats[b].loc[common, f].to_numpy(),
                                        groups, numbers, cfg["link_test"], rng)
        print(f"link {a}-{b}: {links[f'{a} - {b}']}", flush=True)

    # 2. the children: an eyelid with an Hb from CP-AnemiC, plus a nail and a palm under the same number
    eye, ids, groups = children(tabs)
    with_hb = eye.index[eye.hb.notna()]
    hb = eye.loc[ids, "hb"].to_numpy(float)
    y = (hb < cutoff).astype(int)
    name_label = ids.get_level_values("anemic_name").to_numpy()
    hospital = eye.loc[ids, "hospital"].to_numpy(object)
    X = {s: feats[s].loc[ids].to_numpy() for s in SITES}
    X |= {c: np.hstack([X[s] for s in parts]) for c, parts in COMBOS.items()}
    rule = {s: -feats[s].loc[ids, rule_feature].to_numpy() for s in SITES}
    rule |= {c: rank_mean([rule[s] for s in parts]) for c, parts in COMBOS.items()}
    who = {"numbers kept": {s: counts[s]["numbers kept"] for s in SITES},
           "eyelid numbers matched to CP-AnemiC": int(eye.cp_group.notna().sum()),
           "eyelid numbers with one agreed Hb": len(with_hb), "children with all three": len(ids),
           "anemic": float(y.mean()), "file-name label agrees with Hb < cutoff": float((name_label == y).mean()),
           "children with a trusted hospital": int(pd.notna(hospital).sum()), "CV groups": int(len(set(groups)))}
    print(f"children: {who}", flush=True)

    # 3. models
    trusted = pd.notna(hospital)
    splits = {"5x5 CV by child": (np.ones(len(y), bool), repeated_stratified_grouped(
                  y, groups, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)),
              "leave one hospital out": (trusted, leave_one_group_out(hospital[trusted].astype(str)))}
    rows, scores = [], {}
    for split, (m, folds) in splits.items():
        yy = y[m]
        head = {"split": split, "n": int(m.sum()), "anemic": f"{yy.mean():.1%}"}
        s = {(name, inp, "alone" if inp in SITES else "early"): fit_and_score(make, X[inp][m], yy, folds)[0]
             for name, make in models.items() for inp in X}
        s |= {(name, c, "late"): np.mean([s[(name, p, "alone")] for p in parts], axis=0)
              for name in models for c, parts in COMBOS.items()}
        scores[split] = s
        print(f"done: {split}", flush=True)
        rows.append(head | {"model": "dummy (prior)", "input": "-", "how": "-",
                            **summarize(yy, np.full(len(yy), yy.mean()), n_boot, ci, seed, metrics=METRICS)})
        for inp in X:
            rows.append(head | {"model": "rule_low_redness", "input": inp, "how": "alone" if inp in SITES else "rank mean",
                                **summarize(yy, rule[inp][m], n_boot, ci, seed, metrics=METRICS)})
        for (name, inp, how), sc in s.items():
            row = head | {"model": name, "input": inp, "how": how, **summarize(yy, sc, n_boot, ci, seed, metrics=METRICS)}
            if inp in COMBOS:
                row["vs_eyelid"] = auroc_change(yy, sc, s[(name, "eyelid", "alone")], n_boot, ci, seed)
            rows.append(row)

    # pretend children: the same eyelids, a random nail and palm with the same label, late fusion
    folds = splits["5x5 CV by child"][1]
    real = scores["5x5 CV by child"]
    pretend = []
    for d in range(cfg["pretend_draws"]):
        order = {s: same_label_shuffle(y, rng) for s in ("nail", "palm")}
        alone = {(name, s): fit_and_score(make, X[s][order[s]], y, folds)[0]
                 for name, make in models.items() for s in order}
        for name in models:
            for c, parts in COMBOS.items():
                late = np.mean([real[(name, "eyelid", "alone")]] + [alone[(name, p)] for p in parts[1:]], axis=0)
                pretend.append({"draw": d, "model": name, "input": c, "how": "late, pretend partner",
                                **summarize(y, late, n_boot, ci, seed, metrics=METRICS),
                                "vs_eyelid": auroc_change(y, late, real[(name, "eyelid", "alone")], n_boot, ci, seed),
                                "real_minus_pretend": auroc_change(y, real[(name, c, "late")], late, n_boot, ci, seed)})
        print(f"done: pretend draw {d}", flush=True)

    # the Hb number
    regs = {"dummy: everyone gets the mean Hb": DummyRegressor} | {
        n: make for n, make in regressors(hb_cfg, seed).items() if n in cfg["hb_regressors"]}
    preds = {(name, inp): oof_regression(make, X[inp], hb, folds).mean(0) for name, make in regs.items()
             for inp in (*SITES, ALL3) if not (name.startswith("dummy") and inp != "eyelid")}
    preds |= {(name, f"{ALL3} (late)"): np.mean([preds[(name, s)] for s in SITES], axis=0)
              for name in regs if not name.startswith("dummy")}
    hb_rows = []
    for (name, inp), p in preds.items():
        row = {"model": name, "input": "-" if name.startswith("dummy") else inp,
               "mae": bootstrap_ci(mae, hb, p, n_boot, ci, seed=seed)}
        if inp.startswith(ALL3):
            row["vs_eyelid"] = mae_change(hb, p, preds[(name, "eyelid")], n_boot, ci, seed)
        hb_rows.append(row)

    path = save_results("stage24_ghana_link", {"link": links, "who": who, "rows": rows, "pretend": pretend,
                                               "hb_rows": hb_rows},
                        {"default": base, "ghana_link": cfg, "color": color_cfg, "probe": probe_cfg,
                         "ceiling": ceiling_cfg, "hb_regression": hb_cfg})
    write_report(path, cfg, base, links, who, counts, rows, pretend, hb_rows, list(models), cutoff, n_boot)


def write_report(path, cfg, base, links, who, counts, rows, pretend, hb_rows, names, cutoff, n_boot):
    lt = cfg["link_test"]
    shuffles = ["shuffled in batch"] + [f"shuffled among {b} neighbours" for b in lt["block_sizes"]]
    link_lines = [f"| {pair} | {r['n']} | {r['same number']:.3f} | "
                  + " | ".join(f"{r[k]['mean']:.3f} (p = {r[k]['p']:.4f})" for k in shuffles) + " | "
                  + ", ".join(f"{d}: {v:.3f}" for d, v in r["offsets"].items()) + f" | {'yes' if r['links'] else 'no'} |"
                  for pair, r in links.items()]
    count_lines = [f"| {s} | " + " | ".join(str(counts[s][k]) for k in counts["eyelid"]) + " |" for s in SITES]

    best, gains, helps = [], [], []
    for split in dict.fromkeys(r["split"] for r in rows):
        g = [r for r in rows if r["split"] == split]
        rule_row = {r["input"]: r for r in g if r["model"] == "rule_low_redness"}
        for inp, how in [("eyelid", "alone"), ("nail", "alone"), ("palm", "alone")] + [
                (c, h) for c in COMBOS for h in ("early", "late")]:
            top = max((r for r in g if r["input"] == inp and r["how"] == how), key=lambda r: r["auroc"]["estimate"])
            best.append(f"| {split} | {inp} | {how} | {fmt(top['auroc'])} ({top['model']}) | "
                        f"{fmt(rule_row[inp]['auroc'])} |")
            if inp in COMBOS:
                gains.append(f"| {split} | {inp} | {how} | " + " | ".join(
                    fmt(next(r["vs_eyelid"] for r in g if r["model"] == n and r["input"] == inp and r["how"] == how))
                    for n in names) + " |")
        helps.append((split, [f"{r['model']} ({r['how']})" for r in g
                              if r["input"] == ALL3 and r.get("vs_eyelid", {"lo": 0})["lo"] > 0]))

    pre = pd.DataFrame([{"model": p["model"], "input": p["input"], "auroc": p["auroc"]["estimate"],
                         "diff": p["real_minus_pretend"]["estimate"], "diff_lo": p["real_minus_pretend"]["lo"],
                         "diff_hi": p["real_minus_pretend"]["hi"]} for p in pretend])
    real_cv = {(r["model"], r["input"]): r for r in rows if r["split"] == "5x5 CV by child" and r["how"] == "late"}
    pretend_lines = []
    for (model, inp), g in pre.groupby(["model", "input"], sort=False):
        pretend_lines.append(f"| {model} | {inp} | {fmt(real_cv[(model, inp)]['auroc'])} | {g.auroc.mean():.3f} "
                             f"({g.auroc.min():.3f}-{g.auroc.max():.3f}) | {g['diff'].mean():+.3f} "
                             f"(draws' CIs from {g.diff_lo.min():+.3f} to {g.diff_hi.max():+.3f}) |")

    hb_lines = [f"| {r['model']} | {r['input']} | {fmt(r['mae'])} | {fmt(r['vs_eyelid']) if 'vs_eyelid' in r else '-'} |"
                for r in hb_rows]
    linked = [p for p, r in links.items() if r["links"]]
    report = (
        "# Stage 24: one Ghanaian child's eyelid, nail and palm photos together\n\n"
        f"Generated by `python -m src.ghana_link` at commit `{_short_commit(path)}`; settings in "
        "`configs/ghana_link.yaml`. Mentor suggestion (2026-11-03): put each child's nail and eyelid together, as if from "
        "the same person. Per location only (Ghana children).\n\n"
        "- **Data:** eyelids from Appiahene, Asare, Donkoh 2022 (Mendeley 10.17632/nt7r8hv2pz.1, CC BY 4.0, downloaded "
        "2026-11-05): CP-AnemiC eyelid photos, pixel for pixel, but named like the nail and palm sets (Stage 13). "
        "**The link** is the label, batch and number in the file name (Anemic-105 / Anemic-FN-105 / AnemicP-105); the "
        "authors don't document it, so step 1 tests it.\n"
        f"- **Anemia:** Hb < {cutoff} g/dL from the CP-AnemiC sheet (children 6-59 months), one label per child for all "
        "three body sites. The nail and palm file-name labels are not the target.\n"
        "- **Inputs:** the 29 color numbers of each body site. *Early* = one model on the numbers of all the sites "
        "together; *late* = the mean of the single-site scores from the same model and folds. The rule ranks low "
        "redness (a*) and, for several sites, averages those ranks (no training).\n"
        f"- **Split:** {cfg['cv']['n_splits']}-fold CV x {cfg['cv']['n_repeats']} repeats by child, stratified on "
        "anemia; children that share any photo or near-copy stay in one fold. Also leaving one hospital out, for the "
        "children whose eyelid photo has no copy in CP-AnemiC (the only ones whose hospital we can trust). Bootstrap "
        f"95% CIs, {n_boot} resamples; AUROC and MAE changes are paired (same resampled children, same model).\n"
        "- **Answers, fixed before running:** (a) the number links two sites if same-number photos agree in color more "
        f"than in {lt['p_max']} of {lt['n_shuffles']} shuffles, both inside the whole batch and only among neighbouring "
        "numbers; (b) combining helps if eyelid + nail + palm beats the eyelid alone (CI above 0, same model); (c) "
        "pretend children = the same eyelids with a random nail and palm of the same label.\n\n"
        "## Answers\n\n"
        f"- **(a) Does the number link the photos?** Linked: {', '.join(linked) or 'none'}.\n"
        + "".join(f"- **(b) {split}:** eyelid + nail + palm beats the eyelid alone for: {', '.join(h) or 'none'}.\n"
                  for split, h in helps)
        + "\n## 1. Does the file number link the body sites?\n\n"
        "Mean L*, a* and b*, ranked inside each label and batch; the agreement is the mean correlation of those ranks "
        "between the two sites. Neighbours share a photo session, so the stricter test shuffles only among nearby "
        "numbers. The offsets pair each number with the one 1 or 2 places away.\n\n"
        "| sites | n | same number | " + " | ".join(shuffles) + " | offsets | linked |\n|" + "---|" * (5 + len(shuffles))
        + "\n" + "\n".join(link_lines) + "\n\n"
        "| site | " + " | ".join(counts["eyelid"]) + " |\n|" + "---|" * (1 + len(counts["eyelid"])) + "\n"
        + "\n".join(count_lines) + "\n\n"
        "## 2. The children\n\n" + "\n".join(f"- {k}: {v if not isinstance(v, float) else f'{v:.1%}'}"
                                             for k, v in who.items()) + "\n\n"
        "## 3. Best AUROC for each input\n\n| split | input | how | best model | rule |\n|---|---|---|---|---|\n"
        + "\n".join(best) + "\n\n"
        "## 4. What combining changes (AUROC minus the eyelid alone, same model)\n\n"
        "| split | input | how | " + " | ".join(names) + " |\n|" + "---|" * (3 + len(names)) + "\n" + "\n".join(gains)
        + "\n\n## 5. Real children against pretend children (late fusion, 5x5 CV)\n\n"
        f"Pretend = the same eyelids with a random nail and palm of the same anemia label, {cfg['pretend_draws']} draws. "
        "The last column is real minus pretend (paired, same children).\n\n"
        "| model | input | real link | pretend, mean (range over draws) | real minus pretend |\n|---|---|---|---|---|\n"
        + "\n".join(pretend_lines) + "\n\n"
        "## 6. The Hb number (MAE against the lab Hb, g/dL; lower is better; 5x5 CV by child)\n\n"
        "| regressor | input | MAE | change vs eyelid alone |\n|---|---|---|---|\n" + "\n".join(hb_lines) + "\n\n"
        "## All results\n\n"
        + markdown_table(rows, ["split", "n", "anemic", "model", "input", "how"],
                         COLUMNS | {"vs_eyelid": "AUROC change vs eyelid alone"}) + "\n"
    )
    (ROOT / "reports" / "stage24_ghana_link.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
