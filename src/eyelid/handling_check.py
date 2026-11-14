"""Stage 30: is the score reading anemia (Hb), or how the photos were taken, cut out or saved?

Pallor changes gradually with Hb, so a score that reads it can't jump at the anemia cutoff (a convention) and should
follow Hb inside each group. A trace of how anemic and healthy photos were handled (they are stored in separate folders
in the Ghana sets) would act like a switch at the cutoff. Four checks, on every dataset with Hb, same 29 color numbers:
A1 jump at the cutoff, A2 handling numbers only (no color), A3 Hb inside each group, B edited copies of the Ghana
eyelid test photos. Settings and the answers, fixed before running, in configs/handling_check.yaml.

Run from the repo root: python -m src.eyelid.handling_check
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified, repeated_stratified_grouped
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features, feature_table, image_properties
from src.eyelid.cross_dataset import load_eyes_defy
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.hb_regression import load_moscow, oof_regression
from src.eyelid.quality import measure
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.severity import rank_corr
from src.eyelid.tabpfn_audit import feature_names, unseen_photos
from src.eyelid.train_color import _short_commit
from src.ghana_link import children, load_sites

NAMES = {"ghana_eyelid": "Ghana eyelids (CP-AnemiC)", "ghana_nail": "Ghana nails (linked children)",
         "ghana_palm": "Ghana palms (linked children)", "india_eyelid": "India eyelids (Eyes-Defy)",
         "italy_eyelid": "Italy eyelids (Eyes-Defy)", "moscow_nail": "Moscow nails (Yakimov)"}
RULE = "rule_low_redness"


# ---------- statistics ----------

def near_cutoff(z, window):
    """People within `window` g/dL of their own cutoff, on either side (z = Hb minus the cutoff)."""
    return (z >= -window) & (z < window)


def auroc_rows(y, s):
    """AUROC of every row of s (n_draws, n) against y, from ranks."""
    r = rankdata(s, axis=1)
    n1 = int(y.sum())
    return (r[:, y == 1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def spearman_rows(s, h):
    """Rank correlation of every row of s (n_draws, n) with h."""
    rs, rh = rankdata(s, axis=1), rankdata(h)
    rs, rh = rs - rs.mean(1, keepdims=True), rh - rh.mean()
    return (rs @ rh) / np.sqrt((rs ** 2).sum(1) * (rh ** 2).sum())


def steady_reference(z, target_auroc, window, n_sims, rng, grid=np.geomspace(0.02, 50, 120), n_fit=200):
    """Draws of a score that falls steadily with Hb (-z + noise), with the noise set so its AUROC on everyone matches
    target_auroc: each draw's AUROC on everyone, near the cutoff, and its rank correlation with z inside each group."""
    y = (z < 0).astype(int)
    noise = rng.standard_normal((n_fit, len(z)))
    sd = grid[np.argmin([abs(auroc_rows(y, -z + g * noise).mean() - target_auroc) for g in grid])]
    s = -z + sd * rng.standard_normal((n_sims, len(z)))
    near = near_cutoff(z, window)
    return {"noise_sd": float(sd), "auroc_all": auroc_rows(y, s), "auroc_near": auroc_rows(y[near], s[:, near]),
            "rho_anemic": spearman_rows(s[:, y == 1], z[y == 1]), "rho_healthy": spearman_rows(s[:, y == 0], z[y == 0])}


def jump_test(z, score, cfg, rng, n_boot, ci, seed):
    """A1 for one score: its AUROC near the cutoff against the steady reference, and its rank correlation with z
    inside each group."""
    y = (z < 0).astype(int)
    near = near_cutoff(z, cfg["window_g_dl"])
    row = {"near_anemic": int(y[near].sum()), "near_healthy": int((1 - y[near]).sum())}
    for g, m in (("anemic", y == 1), ("healthy", y == 0)):
        row[f"rho_{g}"] = bootstrap_ci(lambda a, b: rank_corr(b, a), z[m], score[m], n_boot, ci, seed=seed)
    if min(row["near_anemic"], row["near_healthy"]) < cfg["min_side"]:
        return row | {"answer": "too few people near the cutoff"}
    ref = steady_reference(z, auroc(y, score), cfg["window_g_dl"], cfg["n_sims"], rng)
    near_auc = bootstrap_ci(auroc, y[near], score[near], n_boot, ci, seed=seed)
    p = float((ref["auroc_near"] >= near_auc["estimate"]).mean())
    return row | {"auroc_near": near_auc, "p": p, "jumps": p < cfg["p_max"], "reference": {
        "noise_sd": ref["noise_sd"], "auroc_all_median": float(np.median(ref["auroc_all"])),
        "auroc_near_median": float(np.median(ref["auroc_near"])),
        "auroc_near_97_5": float(np.percentile(ref["auroc_near"], 97.5)),
        "rho_anemic_median": float(np.median(ref["rho_anemic"])),
        "rho_healthy_median": float(np.median(ref["rho_healthy"]))},
        "answer": "jumps at the cutoff" if p < cfg["p_max"] else "no jump beyond a steady signal"}


# ---------- photo edits (B) ----------

def edit(rgb, kind, value):
    """Edited copy of eyelid pixels (n, 3) in 0-1."""
    if kind == "brightness":
        out = rgb * value
    elif kind == "gamma":
        out = np.clip(rgb, 0, 1) ** value
    elif kind == "saturation":
        gray = (rgb @ np.array([0.299, 0.587, 0.114]))[:, None]
        out = gray + value * (rgb - gray)
    elif kind in ("white_balance", "gains"):
        g = (value[0], 1.0, value[1]) if kind == "white_balance" else value
        out = rgb * np.asarray(g, dtype=float)
    else:
        raise ValueError(f"unknown edit {kind!r}")
    return np.clip(out, 0.0, 1.0)


def edit_list(cfg, gains61):
    """(label, kind, value) for every edit in the config."""
    out = [(f"{k} x{v}" if k == "brightness" else f"{k} {v}", k, v)
           for k in ("brightness", "gamma", "saturation") for v in cfg["edits"][k]]
    out += [(f"white balance R x{r}, B x{b}", "white_balance", (r, b)) for r, b in cfg["edits"]["white_balance"]]
    if cfg["edits"]["batch61"]:
        out.append((f"batch61 gains R x{gains61[0]:.2f}, G x{gains61[1]:.2f}, B x{gains61[2]:.2f}", "gains", gains61))
    return out


# ---------- data ----------

def datasets(cfg, base, color_cfg, seed):
    """name -> hb, each person's cutoff, anemic, the 29 numbers, photo paths (None for Moscow) and CV groups."""
    alpha, child = color_cfg["roi"]["alpha_threshold"], base["anemia_cutoff_g_dl"]["child_6_59_months"]
    out = {}

    def add(name, hb, cut, X, paths, groups=None, anemic=None):
        hb, cut = np.asarray(hb, float), np.broadcast_to(np.asarray(cut, float), (len(hb),)).copy()
        y = (hb < cut).astype(int)
        assert anemic is None or (np.asarray(anemic) == y).all(), f"{name}: label disagrees with Hb < cutoff"
        out[name] = {"hb": hb, "cut": cut, "y": y, "X": np.asarray(X, float), "paths": paths, "groups": groups}
        print(f"loaded {name}: {len(y)} people, {y.sum()} anemic", flush=True)

    want = set(cfg["datasets"])
    if "ghana_eyelid" in want:
        cp = load_cp_anemic(load_config("ceiling")["cp_anemic_subset"])
        paths = [ROOT / p for p in cp.image_path]
        add("ghana_eyelid", cp.hb, child, feature_table(paths, cp.id, alpha_threshold=alpha), paths, anemic=cp.anemic)
    if want & {"ghana_nail", "ghana_palm"}:
        tabs, _, feats = load_sites(load_config("ghana_link"), base, alpha)
        eye, ids, groups = children(tabs)
        for site in ("nail", "palm"):
            if f"ghana_{site}" in want:
                add(f"ghana_{site}", eye.loc[ids, "hb"], child, feats[site].loc[ids], list(tabs[site].loc[ids, "path"]),
                    groups=groups)
    if want & {"india_eyelid", "italy_eyelid"}:
        cdc = load_config("cross_dataset")
        ed = load_eyes_defy(cdc)
        for country in ("India", "Italy"):
            if f"{country.lower()}_eyelid" in want:
                d = ed[ed.country == country].reset_index(drop=True)
                paths = [ROOT / p for p in d.image_path]
                add(f"{country.lower()}_eyelid", d.hb, d.sex.map(cdc["eyes_defy_cutoff_g_dl"]),
                    feature_table(paths, d.id, alpha_threshold=alpha), paths, anemic=d.anemic)
    if "moscow_nail" in want:
        ds = load_moscow({"cv": cfg["cv"]}, color_cfg, base, seed)
        add("moscow_nail", ds["hb"], ds["cutoff"], ds["color"], None)
    return {k: out[k] for k in cfg["datasets"]}


def folds_for(y, groups, cv, seed):
    if groups is None:
        return repeated_stratified(y, cv["n_splits"], cv["n_repeats"], seed)
    return repeated_stratified_grouped(y, groups, cv["n_splits"], cv["n_repeats"], seed)


def held_out(makers, X, y, folds, variants=None):
    """Held-out P(anemic) (n_repeats, n) per model on X and on each edited version of X (same rows). Models are always
    trained on X as it is; the edited test rows go through the same fitted model in one call."""
    variants = variants or {}
    reps = max(r for r, _, _ in folds) + 1
    out = {v: {m: np.full((reps, len(y)), np.nan) for m in makers} for v in ["as is", *variants]}
    for r, train, test in folds:
        for m, make in makers.items():
            model = make().fit(X[train], y[train])
            p = model.predict_proba(np.vstack([X[test], *(Xv[test] for Xv in variants.values())]))[:, 1]
            for k, v in enumerate(out):
                out[v][m][r, test] = p[k * len(test):(k + 1) * len(test)]
    assert all(not np.isnan(s).any() for v in out.values() for s in v.values())
    return out


def handling_numbers(paths, quality_cfg, alpha):
    """No color: size and shape of the cut-out, soft-edge share, sharpness, region size, file bytes. Columns that are
    the same in every photo are dropped (TabPFN refuses them)."""
    t = image_properties(paths, range(len(paths))).reset_index(drop=True)
    q = [measure(p, quality_cfg, alpha)[0] for p in paths]
    t["sharpness"] = [r["blur"] for r in q]
    t["region_px"] = [r["size"] for r in q]
    t["file_bytes"] = [p.stat().st_size for p in paths]
    t["bytes_per_pixel"] = t.file_bytes / (t.width * t.height)
    t["bytes_per_region_px"] = t.file_bytes / t.region_px
    return t.loc[:, t.nunique() > 1]


def within_group_hb(make, X, hb, y, groups, cv, seed, n_boot, ci):
    """A3: TabPFN regressor trained and tested inside one group at a time; rank correlation of its held-out estimate
    with the real Hb. Folds stratified on Hb tertiles inside the group."""
    out = {}
    for g, m in (("anemic", y == 1), ("healthy", y == 0)):
        idx = np.flatnonzero(m)
        bins = np.digitize(hb[idx], np.quantile(hb[idx], [1 / 3, 2 / 3]))
        f = folds_for(bins, None if groups is None else np.asarray(groups)[idx], cv, seed)
        pred = oof_regression(make, X[idx], hb[idx], f).mean(0)
        out[g] = {"n": len(idx), "rho": bootstrap_ci(lambda a, b: rank_corr(b, a), hb[idx], pred, n_boot, ci, seed=seed)}
        out[g]["answer"] = "carries Hb" if out[g]["rho"]["lo"] > 0 else "no Hb signal shown"
    return out


# ---------- run ----------

def main():
    base, cfg, color_cfg = load_config("default"), load_config("handling_check"), load_config("color")
    probe_cfg, ceiling_cfg, quality_cfg = load_config("probe"), load_config("ceiling"), load_config("quality")
    audit_cfg, reg_cfg = load_config("tabpfn_audit"), load_config("hb_tabpfn")["tabpfn"]
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "handling_check": cfg, "color": color_cfg, "ceiling": ceiling_cfg, "quality": quality_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    makers = {m: ladder[m][1] for m in ("dummy", "hist_gradient_boosting", "tabpfn")}
    reg = lambda: TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path=ROOT / reg_cfg["model_path"],
                                                             device=reg_cfg["device"], random_state=seed)
    a_col = feature_names().index(color_cfg["rule_feature"])
    data = datasets(cfg, base, color_cfg, seed)
    out = {"datasets": {}, "edits": None}

    # B setup: edited Ghana eyelid photos (the color numbers recomputed on edited pixels)
    variants, edits = {}, []
    if "ghana_eyelid" in data:
        px = [roi_pixels(p, alpha) for p in data["ghana_eyelid"]["paths"]]
        new, _, _ = unseen_photos(audit_cfg["conj_dir"], load_cp_anemic("all"), audit_cfg["near_bits"])
        gains61 = tuple(float(v) for v in np.mean([roi_pixels(p, alpha).mean(0) for p in new.path], 0)
                        / np.mean([p.mean(0) for p in px], 0))
        edits = edit_list(cfg, gains61)
        variants = {lab: np.array([list(color_features(edit(p, k, v)).values()) for p in px]) for lab, k, v in edits}
        print(f"edits ready: {[e[0] for e in edits]}", flush=True)

    # A1 (and B on the Ghana eyelids)
    for name, d in data.items():
        y, X, z = d["y"], d["X"], d["hb"] - d["cut"]
        folds = folds_for(y, d["groups"], cfg["cv"], seed)
        var = variants if name == "ghana_eyelid" else {}
        held = held_out(makers, X, y, folds, var)
        scores = {m: s.mean(0) for m, s in held["as is"].items()} | {RULE: -X[:, a_col]}
        res = {"n": len(y), "anemic": int(y.sum()), "cutoff": sorted(set(np.round(d["cut"], 2).tolist())),
               "scores": {m: summarize(y, held["as is"][m] if m in held["as is"] else s, n_boot, ci, seed)
                          for m, s in scores.items()},
               "jump": {m: jump_test(z, s, cfg, rng, n_boot, ci, seed) for m, s in scores.items() if m != "dummy"}}
        if var:
            res["edits"] = []
            for lab, Xv in var.items():
                for m in ("hist_gradient_boosting", "tabpfn", RULE):
                    s = held[lab][m] if m != RULE else -Xv[:, a_col]
                    ref = held["as is"][m] if m != RULE else scores[RULE]
                    res["edits"].append({"edit": lab, "model": m, "auroc": summarize(y, s, n_boot, ci, seed)["auroc"],
                                         "change": auroc_change(y, s, ref, n_boot, ci, seed)})
        out["datasets"][name] = res
        print(f"done A1: {name} tabpfn {res['scores']['tabpfn']['auroc']['estimate']:.3f} "
              f"{res['jump']['tabpfn'].get('answer')}", flush=True)
        save_results("stage30_handling_check", out, configs)

    # A3: Hb inside each group
    for name, d in data.items():
        out["datasets"][name]["within_group_hb"] = within_group_hb(reg, d["X"], d["hb"], d["y"], d["groups"],
                                                                   cfg["cv_light"], seed, n_boot, ci)
        print(f"done A3: {name} {out['datasets'][name]['within_group_hb']}", flush=True)
        save_results("stage30_handling_check", out, configs)

    # A2: handling numbers only
    for name, d in data.items():
        if d["paths"] is None:
            continue
        t = handling_numbers(d["paths"], quality_cfg, alpha)
        held = held_out({"tabpfn": makers["tabpfn"]}, t.to_numpy(float), d["y"],
                        folds_for(d["y"], d["groups"], cfg["cv_light"], seed))["as is"]["tabpfn"]
        s = summarize(d["y"], held, n_boot, ci, seed)
        out["datasets"][name]["handling_only"] = {"columns": list(t.columns), "tabpfn": s,
                                                 "answer": "label outside the color" if s["auroc"]["lo"] > 0.5
                                                 else "no label outside the color shown"}
        print(f"done A2: {name} {s['auroc']}", flush=True)
        save_results("stage30_handling_check", out, configs)

    out["fragile_edits"] = fragile(out["datasets"].get("ghana_eyelid", {}).get("edits", []), cfg)
    path = save_results("stage30_handling_check", out, configs)
    write_report(path, cfg, out)


def fragile(rows, cfg):
    """B answer: edits where TabPFN's AUROC drops by at least fragile_drop (CI below 0) while the rule's barely moves."""
    by = {(r["edit"], r["model"]): r["change"] for r in rows}
    return [e for e, m in by if m == "tabpfn" and by[(e, m)]["hi"] < 0 and by[(e, m)]["estimate"] <= -cfg["fragile_drop"]
            and by[(e, RULE)]["estimate"] > -cfg["rule_stable_drop"]]


def write_report(path, cfg, out):
    D = out["datasets"]
    w = cfg["window_g_dl"]

    def jump_row(name, m, j):
        if "auroc_near" not in j:
            return (f"| {NAMES[name]} | {m} | {j['near_anemic']} / {j['near_healthy']} | - | - | - | "
                    f"{fmt(j['rho_anemic'])} | - | {j['answer']} |")
        r = j["reference"]
        return (f"| {NAMES[name]} | {m} | {j['near_anemic']} / {j['near_healthy']} | {fmt(j['auroc_near'])} | "
                f"{r['auroc_near_median']:.3f} (97.5%: {r['auroc_near_97_5']:.3f}) | {j['p']:.4f} | "
                f"{fmt(j['rho_anemic'])} | {r['rho_anemic_median']:+.2f} | **{j['answer']}** |")

    a1 = "\n".join(jump_row(n, m, j) for n, d in D.items() for m, j in d["jump"].items())
    overall = "\n".join(f"| {NAMES[n]} | {d['n']} | {d['anemic']} | {d['cutoff']} | " + " | ".join(
        fmt(d["scores"][m]["auroc"]) for m in ("dummy", RULE, "hist_gradient_boosting", "tabpfn")) +
        f" | {fmt(d['scores']['tabpfn']['pr_auc'])} | {fmt(d['scores']['tabpfn']['sens_at_90_spec'])} |"
        for n, d in D.items())
    a3 = "\n".join(f"| {NAMES[n]} | {g} | {r['n']} | {fmt(r['rho'])} | {r['answer']} |"
                   for n, d in D.items() for g, r in d.get("within_group_hb", {}).items())
    a2 = "\n".join(f"| {NAMES[n]} | {fmt(d['handling_only']['tabpfn']['auroc'])} | "
                   f"{fmt(d['handling_only']['tabpfn']['pr_auc'])} | {d['handling_only']['answer']} |"
                   for n, d in D.items() if "handling_only" in d)
    b = "\n".join(f"| {r['edit']} | {r['model']} | {fmt(r['auroc'])} | {fmt(r['change'])} |"
                  for r in D.get("ghana_eyelid", {}).get("edits", []))
    report = (
        "# Stage 30: is the score reading anemia, or how the photos were handled?\n\n"
        f"Generated by `python -m src.eyelid.handling_check` at commit `{_short_commit(path)}`; settings and the "
        "answers (fixed before running) in `configs/handling_check.yaml`.\n\n"
        "- **Cutoffs, one per dataset:** Ghana eyelids, nails, palms: children 6-59 months, **Hb < 11.0 g/dL**. India "
        "and Italy eyelids (Eyes-Defy adults): **WHO by sex, women < 12.0, men < 13.0 g/dL**, pregnancy not recorded. "
        "Moscow nails (adults, no sex column): **Hb < 12.0 g/dL** for everyone.\n"
        "- **Input:** the same 29 color numbers as every stage. 5-fold CV x 5 repeats by photo / patient / child "
        "(linked Ghana children sharing a photo stay in one fold), scores averaged over repeats. Bootstrap 95% CIs, "
        "2000 resamples.\n"
        "- **Why:** pallor changes gradually with Hb, so a score that reads it can't jump at a cutoff that is a "
        "convention, and should follow Hb inside each group.\n\n"
        "## The models on each dataset\n\n"
        "| dataset | n | anemic | cutoff (g/dL) | dummy | rule | gradient boosting | TabPFN | TabPFN PR-AUC | "
        "TabPFN sens @ 90% spec |\n|---|---|---|---|---|---|---|---|---|---|\n" + overall + "\n\n"
        "## A1. Does the score jump at the cutoff?\n\n"
        f"Near the cutoff = within {w} g/dL of the person's own cutoff on either side. **Steady reference** = a score "
        f"that falls steadily with Hb, with its noise set so its AUROC on everyone equals the model's own; "
        f"{cfg['n_sims']} draws on the same people, no Hb measurement error (the steepest a steady signal can be near "
        f"the cutoff). p = share of reference draws at or above the model's near-cutoff AUROC; **jumps** if p < "
        f"{cfg['p_max']}. Rank correlation is with Hb minus the person's cutoff (negative = lower Hb, higher score).\n\n"
        "| dataset | model | near: anemic / healthy | near-cutoff AUROC | steady reference (median) | p | "
        "rank corr. with Hb, anemic only | reference | answer |\n|---|---|---|---|---|---|---|---|---|\n" + a1 + "\n\n"
        "## A3. Do the photos carry Hb inside each group?\n\n"
        "TabPFN regressor on the 29 numbers, trained and tested inside one group at a time (5-fold CV x 1 repeat). "
        "Rank correlation of the held-out estimate with the real Hb; **carries Hb** if the CI is above 0.\n\n"
        "| dataset | group | n | rank correlation | answer |\n|---|---|---|---|---|\n" + a3 + "\n\n"
        "## A2. Handling numbers only (no color)\n\n"
        "Width, height, cut-out shape, soft-edge share, sharpness (Laplacian variance, Stage 21), region size, file "
        "bytes per pixel and per region pixel. TabPFN, 5-fold CV x 1 repeat. Not run on Moscow (nail boxes inside one "
        "photo from a fixed light box).\n\n"
        "| dataset | TabPFN AUROC | PR-AUC | answer |\n|---|---|---|---|\n" + a2 + "\n\n"
        "## B. Edited copies of the Ghana eyelid test photos\n\n"
        "Models trained on the photos as they are; each edit applied to the test photos' eyelid pixels, and the 29 "
        "numbers recomputed. batch61 moves the mean color of the 388 photos to that of the 61 never-used photos (Stage "
        "29; that batch is 84% anemic, so part of its color gap is anemia). **Fragile** if TabPFN drops by at least "
        f"{cfg['fragile_drop']} (CI below 0) while the rule moves less than {cfg['rule_stable_drop']}: "
        f"{out['fragile_edits'] or 'none'}.\n\n"
        "| edit | model | AUROC | change vs unedited |\n|---|---|---|---|\n" + b + "\n"
    )
    (ROOT / "reports" / "stage30_handling_check.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
