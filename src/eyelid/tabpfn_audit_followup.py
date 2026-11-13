"""Stage 29 follow-up: why checks 2 (hidden copies) and 7 (never-used photos) failed.

Added after the main audit ran, so nothing here was fixed beforehand; it is reported, not pass/fail. Settings in
configs/tabpfn_audit.yaml (followup_bits, control_draws).

Run from the repo root: python -m src.eyelid.tabpfn_audit_followup
"""
import numpy as np
import pandas as pd
from PIL import Image

from src.common.evaluate import auroc_change, oof_scores, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc
from src.common.splits import repeated_stratified, repeated_stratified_grouped
from src.eyelid.ceiling import ladder_models, population_frames, population_inputs
from src.eyelid.color_features import feature_table
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.report import fmt
from src.eyelid.tabpfn_audit import components, dhash_views, feature_names, min_bits, unseen_photos
from src.eyelid.train_color import _short_commit
from src.ghana_link import shared_groups
from src.ghana_sites import parse_name

AUC = {"auroc": auroc}


def random_groups(groups, rng):
    """The same group sizes, with photos assigned to the groups at random."""
    return rng.permutation(groups)


def main():
    base, cfg, ceiling = load_config("default"), load_config("tabpfn_audit"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    df, x = population_inputs(population_frames(ceiling), color_cfg, probe_cfg, ceiling)["cp_anemic"]
    y, X, names = df.anemic.to_numpy(), x["color"], feature_names()
    tab = ladder_models(color_cfg, probe_cfg, ceiling, seed)["tabpfn"][1]
    cv = ceiling["cv"]
    folds = repeated_stratified(y, cv["n_splits"], cv["n_repeats"], seed)
    ref = oof_scores(tab, X, y, folds)
    out = {"reference": summarize(y, ref, n_boot, ci, seed, metrics=AUC)}

    # 2. how close the possible copies are, and what grouping them costs
    views = np.array([dhash_views(ROOT / p) for p in df.image_path])
    d = min_bits(views, views[:, 0])
    d = np.minimum(d, d.T)
    pair = d[np.triu_indices(len(d), 1)]
    i, j = np.triu_indices(len(d), 1)
    same_label_random = float(np.mean(y[i] == y[j]))
    out["pairs_by_bits"] = {b: int((pair <= b).sum()) for b in (5, 10, 15, 20, 25, cfg["near_bits"])}
    child = df.groupby(["hb", "age_months", "sex", "site"]).ngroup().to_numpy()
    np.fill_diagonal(d, 10 ** 6)

    def grouped_score(groups, s):
        held = oof_scores(tab, X, y, repeated_stratified_grouped(y, groups, cv["n_splits"], cv["n_repeats"], s))
        return {**summarize(y, held, n_boot, ci, seed, metrics=AUC), "vs_reference": auroc_change(y, held, ref, n_boot, ci, seed)}

    out["tight"] = []
    for b in cfg["followup_bits"]:
        close = pair <= b
        groups = shared_groups(components(d <= b), child)
        out["tight"].append({"bits": b, "pairs": int(close.sum()), "same_label": float(np.mean(y[i][close] == y[j][close])),
                             "same_label_random_pairs": same_label_random,
                             "same_hospital": float(np.mean(df.site.to_numpy()[i][close] == df.site.to_numpy()[j][close])),
                             **grouped_score(groups, seed)})
        print(f"done: grouped at {b} bits {out['tight'][-1]['auroc']['estimate']:.3f}", flush=True)
    loose = shared_groups(components(d <= cfg["near_bits"]), child)
    sizes = np.bincount(loose)
    out["loose_groups"] = {"largest": int(sizes.max()), "groups_of_2_plus": int((sizes > 1).sum()),
                           "photos_in_them": int(sizes[sizes > 1].sum())}
    rng = np.random.default_rng(seed)
    out["control"] = []
    for k in range(cfg["control_draws"]):
        out["control"].append(grouped_score(random_groups(loose, rng), seed + k))
        print(f"done: random groups #{k + 1} {out['control'][-1]['auroc']['estimate']:.3f}", flush=True)
    save_results("stage29_followup", out, {"default": base, "tabpfn_audit": cfg, "ceiling": ceiling})

    # 7. what the never-used photos are
    new, counts, near = unseen_photos(cfg["conj_dir"], load_cp_anemic("all"), cfg["near_bits"])
    model = tab().fit(X, y)
    sets = {"never-used": new, "near-copies of CP-AnemiC photos (left out of check 7)": near}
    out["unseen_counts"] = counts
    out["sets"] = []
    for label, s in sets.items():
        xs = feature_table(s.path, range(len(s)), alpha_threshold=alpha)[names]
        ys = s.anemic.to_numpy()
        styles = pd.Series([parse_name(p.stem)[2] for p in s.path])
        size = np.array([Image.open(p).size for p in s.path])
        row = {"set": label, "n": len(s), "anemic": int(ys.sum()),
               "naming_styles": {f"{k}, anemic={a}": int(v) for (k, a), v in
                                 pd.Series(1, index=pd.MultiIndex.from_arrays([styles, ys])).groupby(level=[0, 1]).sum().items()},
               "a_mean_by_label": {int(a): float(xs.a_mean[ys == a].mean()) for a in (0, 1) if (ys == a).any()},
               "L_mean_by_label": {int(a): float(xs.L_mean[ys == a].mean()) for a in (0, 1) if (ys == a).any()},
               "median_size": [int(v) for v in np.median(size, 0)]}
        if len(set(ys)) == 2:
            row["tabpfn"] = summarize(ys, model.predict_proba(xs.to_numpy())[:, 1], n_boot, ci, seed, metrics=AUC)
            row["rule"] = summarize(ys, -xs.a_mean.to_numpy(), n_boot, ci, seed, metrics=AUC)
        out["sets"].append(row)
    cp_size = np.array([Image.open(ROOT / p).size for p in df.image_path])
    out["cp_anemic"] = {"a_mean_by_label": {int(a): float(X[y == a, names.index("a_mean")].mean()) for a in (0, 1)},
                        "L_mean_by_label": {int(a): float(X[y == a, names.index("L_mean")].mean()) for a in (0, 1)},
                        "median_size": [int(v) for v in np.median(cp_size, 0)]}
    path = save_results("stage29_followup", out, {"default": base, "tabpfn_audit": cfg, "ceiling": ceiling})
    write_report(path, cfg, out)


def write_report(path, cfg, out):
    ref = out["reference"]["auroc"]["estimate"]
    tight = "\n".join(f"| <= {r['bits']} bits | {r['pairs']} | {r['same_label']:.2f} (random pairs "
                      f"{r['same_label_random_pairs']:.2f}) | {r['same_hospital']:.2f} | {fmt(r['auroc'])} | "
                      f"{fmt(r['vs_reference'])} |" for r in out["tight"])
    control = "\n".join(f"| random groups, draw {k + 1} | {fmt(r['auroc'])} | {fmt(r['vs_reference'])} |"
                        for k, r in enumerate(out["control"]))
    lg = out["loose_groups"]
    cp = out["cp_anemic"]
    sets = "\n".join(
        f"| {r['set']} | {r['n']} | {r['anemic']} | {r['a_mean_by_label']} | {r['L_mean_by_label']} | "
        f"{r['median_size']} | {fmt(r['tabpfn']['auroc']) if 'tabpfn' in r else '-'} | "
        f"{fmt(r['rule']['auroc']) if 'rule' in r else '-'} | {r['naming_styles']} |" for r in out["sets"])
    report = (
        "# Stage 29 follow-up: why the hidden-copy and never-used-photo checks failed\n\n"
        f"Generated by `python -m src.eyelid.tabpfn_audit_followup` at commit `{_short_commit(path)}`. Added after the "
        "main audit (`reports/stage29_tabpfn_audit.md`) ran, so nothing here was fixed beforehand. Same 388 Ghana "
        f"children, **Hb < 11.0 g/dL**, TabPFN on the 29 color numbers, 5x5 CV: {fmt(out['reference']['auroc'])}.\n\n"
        "## 2. Hidden copies\n\n"
        f"Possible-copy pairs by dHash distance (of 256 bits, over 8 rotations and mirror images): {out['pairs_by_bits']}. "
        f"At the pre-set {cfg['near_bits']} bits the pairs chain into groups: the largest holds {lg['largest']} photos, "
        f"and {lg['photos_in_them']} photos sit in {lg['groups_of_2_plus']} groups.\n\n"
        "| kept together | pairs | same anemia label | same hospital | TabPFN AUROC | change vs plain CV |\n"
        "|---|---|---|---|---|---|\n" + tight + "\n\n"
        f"**Control:** the same group sizes as at {cfg['near_bits']} bits, photos assigned at random. If the control "
        "drops as much as the real grouping, the drop is the cost of forcing one big group into one fold, not a leak.\n\n"
        "| grouping | TabPFN AUROC | change vs plain CV |\n|---|---|---|\n" + control + "\n\n"
        "## 7. The never-used photos\n\n"
        f"Counts: {out['unseen_counts']}. CP-AnemiC (388) for comparison: mean a* by label {cp['a_mean_by_label']}, "
        f"mean L* {cp['L_mean_by_label']}, median size {cp['median_size']} px (0 = healthy, 1 = anemic).\n\n"
        "| set | n | anemic | mean a* by label | mean L* by label | median size (px) | TabPFN AUROC | rule AUROC | "
        "naming styles |\n|---|---|---|---|---|---|---|---|---|\n" + sets + "\n"
    )
    (ROOT / "reports" / "stage29_followup.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
