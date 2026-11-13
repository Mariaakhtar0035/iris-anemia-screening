"""Stage 29: a full audit of the TabPFN eyelid result (Ghana children, 388 photos with no copy).

TabPFN on 29 color numbers (0.92) beats every vision model (0.67-0.79). Seven checks, answers fixed in
configs/tabpfn_audit.yaml before running: what it reads, the right direction, file details only, hidden copies,
never-used photos, model choice without the test children, and shuffled labels. Results are saved after each check,
so a long run can be read part way.

Run from the repo root: python -m src.eyelid.tabpfn_audit
"""
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.common.evaluate import auroc_change, oof_scores, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified, repeated_stratified_grouped
from src.eyelid.ceiling import ladder_models, population_frames, population_inputs
from src.eyelid.color_features import CHANNELS, feature_table
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.report import fmt
from src.eyelid.severity import rank_corr
from src.eyelid.train_color import _short_commit
from src.ghana_link import pixel_key, shared_groups
from src.ghana_sites import parse_name

AUC = {"auroc": auroc}


def feature_names():
    """The 29 color numbers in the order color_features makes them."""
    return [f"{c}_{s}" for c in CHANNELS for s in ("mean", "median", "std")] + ["R_over_G", "a_minus_b"]


def dhash_views(path, n=16):
    """dHash of the photo on mid-grey (as the Stage 2 audit) in all 8 rotations and mirror images: (8, n*n) bool.
    View 0 is the photo as stored."""
    im = Image.open(path).convert("RGBA")
    out = []
    for b in (im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)):
        for k in range(4):
            r = b.rotate(90 * k, expand=True)
            flat = Image.alpha_composite(Image.new("RGBA", r.size, (128, 128, 128, 255)), r).convert("L")
            g = np.asarray(flat.resize((n + 1, n)), dtype=float)
            out.append((g[:, 1:] > g[:, :-1]).flatten())
    return np.array(out)


def min_bits(views_a, base_b):
    """Smallest dHash distance from each photo of a, in any of its 8 views (na, 8, bits), to each photo of b as
    stored (nb, bits): (na, nb)."""
    return np.min([(v[:, None, :] != base_b[None, :, :]).sum(-1) for v in views_a.transpose(1, 0, 2)], axis=0)


def components(linked):
    """Connected-component id of each row of a square boolean adjacency matrix."""
    return connected_components(csr_matrix(linked), directed=False)[1]


def file_details(path):
    """What the file says without its pixels: PNG metadata and file size."""
    im = Image.open(path)
    w, h = im.size
    solid = int((np.asarray(im.convert("RGBA"))[..., 3] == 255).sum())
    size = path.stat().st_size
    dpi = im.info.get("dpi", (-1.0, -1.0))
    return {"dpi_x": float(dpi[0]), "dpi_y": float(dpi[1]), "gamma": float(im.info.get("gamma", -1.0)),
            "file_bytes": float(size), "bytes_per_pixel": size / (w * h), "bytes_per_eyelid_pixel": size / max(solid, 1)}


def nested_choice(makers, X, y, folds, inner_splits, seed):
    """Held-out scores (n_repeats, n) when each outer training split picks its model by inner-CV AUROC on its own
    children only; and the model picked for each outer fold."""
    held = np.full((max(r for r, _, _ in folds) + 1, len(y)), np.nan)
    picks = []
    for k, (r, train, test) in enumerate(folds):
        inner = repeated_stratified(y[train], inner_splits, 1, seed + k)
        inner_auc = {name: auroc(y[train], oof_scores(make, X[train], y[train], inner)[0])
                     for name, make in makers.items()}
        best = max(inner_auc, key=inner_auc.get)
        picks.append(best)
        held[r, test] = makers[best]().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return held, picks


def unseen_photos(conj_dir, cp, near_bits):
    """Ghana conjunctiva photos that are not CP-AnemiC photos: one path per distinct photo with its file-name label,
    after dropping near-copies of CP-AnemiC photos and of each other; and the counts."""
    files = sorted((ROOT / conj_dir).glob("*.png"))
    df = pd.DataFrame({"path": files, "key": [pixel_key(f) for f in files]})
    cp_keys = {pixel_key(ROOT / p) for p in cp.image_path}
    new = df[~df.key.isin(cp_keys)].copy()
    new["anemic"] = [parse_name(f.stem)[1] for f in new.path]
    counts = {"files": len(df), "distinct photos": int(df.key.nunique()),
              "not a CP-AnemiC photo": int(new.key.nunique())}
    new = new[new.groupby("key").anemic.transform("nunique") == 1].drop_duplicates("key").reset_index(drop=True)
    views = np.array([dhash_views(p) for p in new.path])
    cp_base = np.array([dhash_views(ROOT / p)[0] for p in cp.drop_duplicates("image_group").image_path])
    far = min_bits(views, cp_base).min(1) > near_bits
    counts["near-copy of a CP-AnemiC photo"] = int((~far).sum())
    new, views = new[far].reset_index(drop=True), views[far]
    d = min_bits(views, views[:, 0])
    comp = components(np.minimum(d, d.T) <= near_bits)
    agree = pd.Series(new.anemic.to_numpy()).groupby(comp).transform("nunique") == 1
    keep = agree.to_numpy() & ~pd.Series(comp).duplicated().to_numpy()
    counts["dropped as near-copies of each other"] = int(len(new) - keep.sum())
    new = new[keep].reset_index(drop=True)
    counts |= {"kept": len(new), "anemic": int(new.anemic.sum())}
    return new, counts


def main():
    base, cfg, ceiling = load_config("default"), load_config("tabpfn_audit"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    df, x = population_inputs(population_frames(ceiling), color_cfg, probe_cfg, ceiling)["cp_anemic"]
    y, X = df.anemic.to_numpy(), x["color"]
    names = feature_names()
    assert feature_table([ROOT / df.image_path[0]], [0], alpha_threshold=alpha).columns.tolist() == names
    assert sorted(sum(cfg["groups"].values(), [])) == sorted(names), "the groups must split the 29 numbers exactly"
    assert ((df.hb < base["anemia_cutoff_g_dl"]["child_6_59_months"]) == (y == 1)).all()
    ladder = ladder_models(color_cfg, probe_cfg, ceiling, seed)
    tab = ladder["tabpfn"][1]
    cv = ceiling["cv"]
    folds = repeated_stratified(y, cv["n_splits"], cv["n_repeats"], seed)
    configs = {"default": base, "tabpfn_audit": cfg, "ceiling": ceiling, "color": color_cfg}
    out = {"n": len(y), "anemic": int(y.sum())}

    def save():
        return save_results("stage29_tabpfn_audit", out, configs)

    ref = oof_scores(tab, X, y, folds)
    out["reference"] = summarize(y, ref, n_boot, ci, seed)
    ref_auc = out["reference"]["auroc"]["estimate"]
    print(f"done: reference {ref_auc:.3f}", flush=True)

    # 4. what it reads
    out["groups"] = []
    for g, cols in cfg["groups"].items():
        held = oof_scores(tab, X[:, [names.index(c) for c in cols]], y, folds)
        out["groups"].append({"group": g, "n_numbers": len(cols), **summarize(y, held, n_boot, ci, seed, metrics=AUC),
                              "vs_all": auroc_change(y, held, ref, n_boot, ci, seed)})
        print(f"done: group {g} {out['groups'][-1]['auroc']['estimate']:.3f}", flush=True)
    save()

    # 5. direction
    score = ref.mean(0)
    out["direction"] = {c: bootstrap_ci(rank_corr, X[:, names.index(c)], score, n_boot, ci, seed=seed)
                        for c in ("a_mean", "L_mean")}

    # 3. file details only
    files = pd.DataFrame([file_details(ROOT / p) for p in df.image_path])
    out["file_values"] = {c: {str(k): int(v) for k, v in files[c].value_counts().head(5).items()}
                          for c in ("dpi_x", "dpi_y", "gamma")}
    out["file_details"] = []
    for part, cols in {"metadata (dpi, gamma)": ["dpi_x", "dpi_y", "gamma"],
                       "file size": ["file_bytes", "bytes_per_pixel", "bytes_per_eyelid_pixel"]}.items():
        held = oof_scores(tab, files[cols].to_numpy(), y, folds)
        out["file_details"].append({"input": part, **summarize(y, held, n_boot, ci, seed, metrics=AUC)})
    print("done: file details", flush=True)
    save()

    # 2. hidden copies and possible same-child pairs
    views = np.array([dhash_views(ROOT / p) for p in df.image_path])
    d = min_bits(views, views[:, 0])
    d = np.minimum(d, d.T)
    np.fill_diagonal(d, 10 ** 6)
    copy_comp = components(d <= cfg["near_bits"])
    child = df.groupby(["hb", "age_months", "sex", "site"]).ngroup().to_numpy()
    groups = shared_groups(copy_comp, child)
    gfolds = repeated_stratified_grouped(y, groups, cv["n_splits"], cv["n_repeats"], seed)
    held = oof_scores(tab, X, y, gfolds)
    sizes = np.bincount(groups)
    out["copies"] = {"near_copy_pairs": int((np.triu(d, 1) <= cfg["near_bits"]).sum()),
                     "closest_pair_bits": int(d.min()), "median_nearest_bits": int(np.median(d.min(1))),
                     "same_child_pairs": int((np.bincount(child) > 1).sum()),
                     "photos_in_shared_groups": int(sizes[sizes > 1].sum()),
                     **summarize(y, held, n_boot, ci, seed, metrics=AUC)}
    print(f"done: copies {out['copies']['auroc']['estimate']:.3f}", flush=True)
    save()

    # 7. never-used photos
    new, counts = unseen_photos(cfg["conj_dir"], load_cp_anemic("all"), cfg["near_bits"])
    yn = new.anemic.to_numpy()
    out["unseen"] = {"counts": counts}
    if len(set(yn)) == 2:
        xn = feature_table(new.path, range(len(new)), alpha_threshold=alpha)[names].to_numpy()
        hgb = ladder["hist_gradient_boosting"][1]
        for name, s in {"tabpfn": tab().fit(X, y).predict_proba(xn)[:, 1],
                        "hist_gradient_boosting": hgb().fit(X, y).predict_proba(xn)[:, 1],
                        "rule_low_redness": -xn[:, names.index(color_cfg["rule_feature"])]}.items():
            out["unseen"][name] = summarize(yn, s, n_boot, ci, seed, metrics=AUC)
    print(f"done: unseen photos {counts}", flush=True)
    save()

    # 6. model choice without the test children
    makers = {m: ladder[m][1] for m in cfg["models"]}
    held, picks = nested_choice(makers, X, y, folds, cfg["inner_splits"], seed)
    out["nested"] = {"picks": {m: picks.count(m) for m in cfg["models"]}, **summarize(y, held, n_boot, ci, seed)}
    print(f"done: nested {out['nested']['auroc']['estimate']:.3f} {out['nested']['picks']}", flush=True)
    save()

    # 1. shuffled labels
    one = repeated_stratified(y, cv["n_splits"], 1, seed)
    real = auroc(y, oof_scores(tab, X, y, one)[0])
    rng, shuffled = np.random.default_rng(seed), []
    for k in range(cfg["n_shuffles"]):
        ys = rng.permutation(y)
        shuffled.append(auroc(ys, oof_scores(tab, X, ys, repeated_stratified(ys, cv["n_splits"], 1, seed))[0]))
        if (k + 1) % 10 == 0:
            print(f"shuffles: {k + 1}", flush=True)
    s = np.array(shuffled)
    out["shuffled"] = {"real_one_run": real, "n": len(s), "mean": float(s.mean()), "sd": float(s.std()),
                       "max": float(s.max()), "n_at_or_above_real": int((s >= real).sum()),
                       "p": float((1 + (s >= real).sum()) / (1 + len(s))), "all": shuffled}
    path = save()
    write_report(path, cfg, out, ref_auc, n_boot)


def write_report(path, cfg, out, ref_auc, n_boot):
    g = {r["group"]: r for r in out["groups"]}
    red = g["redness"]["auroc"]["estimate"]
    others = max(r["auroc"]["estimate"] for k, r in g.items() if k != "redness")
    reads = ("reads redness" if red >= cfg["reads_redness"] else
             "mostly something else" if red < 0.80 and others >= cfg["reads_redness"] else "mixed")
    direction = out["direction"]["a_mean"]
    meta = out["file_details"][0]["auroc"]
    cop, nest, sh, un = out["copies"], out["nested"], out["shuffled"], out["unseen"]
    rows = [
        ("1. Shuffled labels", f"real {sh['real_one_run']:.3f} vs {sh['n']} shuffles: mean {sh['mean']:.3f}, max "
         f"{sh['max']:.3f} (p = {sh['p']:.3f})", "real beats every shuffle", sh["n_at_or_above_real"] == 0),
        ("2. Hidden copies + same-child pairs kept together", f"{fmt(cop['auroc'])} vs {ref_auc:.3f}; "
         f"{cop['near_copy_pairs']} near-copy pairs, {cop['same_child_pairs']} same-child pairs",
         f"drop < {cfg['max_drop']}", ref_auc - cop["auroc"]["estimate"] < cfg["max_drop"]),
        ("3. File metadata only (dpi, gamma)", fmt(meta), "CI includes 0.5 or below 0.6",
         meta["lo"] <= 0.5 or meta["estimate"] < 0.6),
        ("4. What it reads", f"redness alone {red:.3f}; best other group {others:.3f}: {reads}",
         "reported (redness >= 0.85 = reads redness)", None),
        ("5. Right direction (score vs a_mean)", fmt(direction), "CI below 0 (paler = riskier)", direction["hi"] < 0),
        ("6. Model chosen without the test children", f"{fmt(nest['auroc'])}; picks {nest['picks']}",
         f">= {ref_auc - cfg['max_drop']:.3f}", nest["auroc"]["estimate"] >= ref_auc - cfg["max_drop"]),
    ]
    if "tabpfn" in un:
        t = un["tabpfn"]["auroc"]
        rows.append(("7. Never-used photos", f"{fmt(t)} on {un['counts']['kept']} photos ({un['counts']['anemic']} "
                     "anemic)", f">= {cfg['hold_min']}", t["estimate"] >= cfg["hold_min"]))
    else:
        rows.append(("7. Never-used photos", f"not run: {un['counts']}", "-", None))
    verdict = {True: "**pass**", False: "**fail**", None: "-"}
    table = "| check | result | pass rule | verdict |\n|---|---|---|---|\n" + "\n".join(
        f"| {a} | {b} | {c} | {verdict[v]} |" for a, b, c, v in rows)
    groups = "| group | numbers | AUROC | change vs all 29 |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['group']} | {r['n_numbers']} | {fmt(r['auroc'])} | {fmt(r['vs_all'])} |" for r in out["groups"])
    files = "| input | AUROC |\n|---|---|\n" + "\n".join(f"| {r['input']} | {fmt(r['auroc'])} |"
                                                      for r in out["file_details"])
    unseen = "\n".join(f"| {k} | {fmt(v['auroc'])} |" for k, v in un.items() if k != "counts")
    report = (
        "# Stage 29: a full audit of the TabPFN eyelid result\n\n"
        f"Generated by `python -m src.eyelid.tabpfn_audit` at commit `{_short_commit(path)}`; settings and every pass "
        "rule in `configs/tabpfn_audit.yaml`, fixed before running.\n\n"
        f"- **Children:** Ghana, CP-AnemiC, the {out['n']} photos with no copy ({out['anemic']} anemic). **Anemic: Hb < "
        "11.0 g/dL** (WHO, 6-59 months).\n"
        "- **Model:** TabPFN v2 on the 29 color numbers, 5-fold CV x 5 repeats by photo (Stage 9's folds), held-out "
        f"scores averaged over repeats: {fmt(out['reference']['auroc'])}.\n"
        f"- Bootstrap 95% CIs, {n_boot} resamples; changes are paired (same children).\n\n"
        "## Answers\n\n" + table + "\n\n## 4. What it reads (TabPFN on each group alone)\n\n" + groups
        + f"\n\nDirection: rank correlation of the score with a_mean {fmt(out['direction']['a_mean'])}, with L_mean "
        f"{fmt(out['direction']['L_mean'])} (negative = lower value, higher risk).\n\n"
        "## 3. File details only\n\n" + files + f"\n\nValues seen: {out['file_values']}. File size also depends on the "
        "picture itself, so it is reported, not pass/fail.\n\n"
        "## 2. Hidden copies\n\n"
        f"dHash over 8 rotations and mirror images; possible copies at <= {cfg['near_bits']} of 256 bits. Closest pair "
        f"{cop['closest_pair_bits']} bits; median distance to the nearest other photo {cop['median_nearest_bits']} "
        f"bits. {cop['photos_in_shared_groups']} photos are in a shared fold group.\n\n"
        "## 7. Never-used photos\n\n"
        f"Counts: {un['counts']}. Label from the file name (cutoff not stated). Same research group, so this guards "
        "against our choices fitting the 388; it is not a new site.\n\n| model | AUROC |\n|---|---|\n" + unseen + "\n\n"
        f"## 1. Shuffled labels\n\nOne 5-fold run each: real {sh['real_one_run']:.3f}; shuffles mean {sh['mean']:.3f} "
        f"(sd {sh['sd']:.3f}), max {sh['max']:.3f}, {sh['n_at_or_above_real']} of {sh['n']} at or above the real "
        "score.\n"
    )
    (ROOT / "reports" / "stage29_tabpfn_audit.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
