"""Stage 31: how much of the Ghana score is real anemia signal once the folder trace is taken out?

Stage 30 found the Ghana scores jump at the Hb cutoff: part of them reads how each label folder was handled. A folder
trace is shared by everyone in the folder, so it drops out of differences inside a folder. The within-folder ridge (a
fixed-effects estimator) learns its weights only from how color follows Hb inside each label group, then scores raw
test photos. Four parts, settings and answers fixed before running in configs/trace_free.yaml:
A the within-folder ridge on every dataset (checked on the clean ones), B transfer to another dataset, C do the Hb-number
models jump too, D which single color numbers jump.

Run from the repo root: python -m src.eyelid.trace_free
"""
import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import leave_one_group_out
from src.eyelid.ceiling import ladder_models
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.handling_check import NAMES, RULE, datasets, folds_for, held_out, jump_test
from src.eyelid.hb_regression import mae, regressors
from src.eyelid.report import fmt
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.train_color import _short_commit

WITHIN = "within_folder_ridge"
HB_MODELS = ("ridge_on_hb", "tabpfn_regressor_on_hb", WITHIN)
ORDER = ("dummy", RULE, "ridge_on_hb", "tabpfn_regressor_on_hb", "tabpfn", WITHIN)
LABELS = {"dummy": "dummy", RULE: "rule (minus mean a*)", "ridge_on_hb": "ridge on Hb",
          "tabpfn_regressor_on_hb": "TabPFN regressor on Hb", "tabpfn": "TabPFN classifier",
          WITHIN: "**within-folder ridge**"}
CV = "5x5 CV"


def centre_by_group(X, t, y):
    """Copies of X and t with each label group's own mean taken out."""
    Xc, tc = np.array(X, float), np.array(t, float)
    for g in np.unique(y):
        m = y == g
        Xc[m] -= Xc[m].mean(0)
        tc[m] -= tc[m].mean()
    return Xc, tc


def fit_within(make, X, t, y):
    """Fit on rows centred inside each label group: only differences inside a group can shape the weights."""
    return make().fit(*centre_by_group(X, t, y))


def held_out_hb(makers, X, t, y, folds):
    """Held-out estimates of t, (n_repeats, n) per model. WITHIN is fit on group-centred training rows; the test
    rows are scored as they are, so their labels are never used."""
    reps = max(r for r, _, _ in folds) + 1
    out = {m: np.full((reps, len(t)), np.nan) for m in makers}
    for r, tr, te in folds:
        for m, make in makers.items():
            model = fit_within(make, X[tr], t[tr], y[tr]) if m == WITHIN else make().fit(X[tr], t[tr])
            out[m][r, te] = model.predict(X[te])
    assert all(not np.isnan(p).any() for p in out.values())
    return out


def feature_jumps(z, X, names, cfg, rng, n_boot, ci, seed):
    """D: the jump test on each single color number, turned so higher = more anemic on this dataset."""
    y = (z < 0).astype(int)
    rows = []
    for k, name in enumerate(names):
        sign = 1.0 if auroc(y, X[:, k]) >= 0.5 else -1.0
        j = jump_test(z, sign * X[:, k], cfg, rng, n_boot, ci, seed)
        rows.append({"feature": name, "higher_means_anemic": sign > 0, "auroc": auroc(y, sign * X[:, k]),
                     "auroc_near": j.get("auroc_near"), "reference_near_median": j.get("reference", {}).get(
                         "auroc_near_median"), "p": j.get("p"), "jumps": j.get("p", 1.0) < cfg["p_max"] / len(names)})
    return rows


def keeps_signal(change, cfg):
    return change["estimate"] > -cfg["keep_drop"] and change["hi"] > 0


def answers(out, cfg, p_max):
    D = out["datasets"]
    kept = {n: keeps_signal(D[n]["splits"][CV]["within_vs_ridge"], cfg) for n in cfg["clean"] if n in D}
    trusted = sum(kept.values()) >= cfg["min_clean_kept"]
    free = {n: d["splits"][CV]["jump"][WITHIN].get("p", -1.0) >= p_max for n, d in D.items()}  # no p: too few to test
    return {"clean_kept": kept, "method_trusted": trusted, "free_of_jump": free,
            "trace_free_auroc": {n: D[n]["splits"][CV]["scores"][WITHIN]["auroc"] for n in D
                                 if n.startswith("ghana_") and free[n] and trusted},
            "hb_models_carry_trace": {f"{n} / {m}": bool(d["splits"][CV]["jump"][m].get("jumps"))
                                      for n, d in D.items() for m in HB_MODELS[:2]}}


def main():
    base, cfg, color_cfg = load_config("default"), load_config("trace_free"), load_config("color")
    jump_cfg, hb_cfg, reg_cfg = load_config("handling_check"), load_config("hb_regression"), load_config("hb_tabpfn")["tabpfn"]
    probe_cfg, ceiling_cfg = load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    configs = {"default": base, "trace_free": cfg, "handling_check": jump_cfg, "color": color_cfg, "ceiling": ceiling_cfg,
               "hb_regression": hb_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    clf = {m: ladder[m][1] for m in ("dummy", "tabpfn")}
    ridge = regressors(hb_cfg, seed)["ridge"]
    reg = lambda: TabPFNRegressor.create_default_for_version(ModelVersion.V2, model_path=ROOT / reg_cfg["model_path"],
                                                             device=reg_cfg["device"], random_state=seed)
    hb_makers = {"ridge_on_hb": ridge, "tabpfn_regressor_on_hb": reg, WITHIN: ridge}
    names = feature_names()
    a_col = names.index(color_cfg["rule_feature"])
    data = datasets(cfg, base, color_cfg, seed)
    out = {"datasets": {}, "transfer": []}

    for name, d in data.items():
        y, X, z = d["y"], d["X"], d["hb"] - d["cut"]
        splits = {CV: folds_for(y, d["groups"], cfg["cv"], seed)}
        if name == "ghana_eyelid":
            cp = load_cp_anemic(ceiling_cfg["cp_anemic_subset"])
            assert np.allclose(cp.hb.to_numpy(float), d["hb"]), "Ghana eyelid rows out of order"
            splits["leave one hospital out"] = leave_one_group_out(cp.site.to_numpy())
        res = {"n": len(y), "anemic": int(y.sum()), "cutoff": sorted(set(np.round(d["cut"], 2).tolist())), "splits": {}}
        for split, folds in splits.items():
            prob = held_out(clf, X, y, folds)["as is"]
            est = held_out_hb(hb_makers, X, z, y, folds)
            s = {"dummy": prob["dummy"], RULE: -X[:, a_col], "tabpfn": prob["tabpfn"]} | {m: -p for m, p in est.items()}
            res["splits"][split] = {
                "scores": {m: summarize(y, s[m], n_boot, ci, seed) for m in ORDER},
                "jump": {m: jump_test(z, np.atleast_2d(s[m]).mean(0), jump_cfg, rng, n_boot, ci, seed)
                         for m in ORDER if m != "dummy"},
                "mae": {m: bootstrap_ci(mae, z, est[m].mean(0), n_boot, ci, seed=seed) for m in HB_MODELS[:2]},
                "within_vs_ridge": auroc_change(y, s[WITHIN], s["ridge_on_hb"], n_boot, ci, seed),
                "within_vs_tabpfn": auroc_change(y, s[WITHIN], s["tabpfn"], n_boot, ci, seed)}
            print(f"done A/C: {name} {split} within {res['splits'][split]['scores'][WITHIN]['auroc']['estimate']:.3f} "
                  f"{res['splits'][split]['jump'][WITHIN].get('answer')}", flush=True)
        res["features"] = feature_jumps(z, X, names, jump_cfg, rng, n_boot, ci, seed)
        out["datasets"][name] = res
        save_results("stage31_trace_free", out, configs)

    for src, dst in cfg["transfer"]:
        a, b = data[src], data[dst]
        za = a["hb"] - a["cut"]
        s = {"dummy": clf["dummy"]().fit(a["X"], a["y"]).predict_proba(b["X"])[:, 1], RULE: -b["X"][:, a_col],
             "ridge_on_hb": -ridge().fit(a["X"], za).predict(b["X"]),
             "tabpfn_regressor_on_hb": -reg().fit(a["X"], za).predict(b["X"]),
             "tabpfn": clf["tabpfn"]().fit(a["X"], a["y"]).predict_proba(b["X"])[:, 1],
             WITHIN: -fit_within(ridge, a["X"], za, a["y"]).predict(b["X"])}
        row = {"train": src, "test": dst, "n": len(b["y"]), "anemic": int(b["y"].sum()),
               "scores": {m: summarize(b["y"], s[m], n_boot, ci, seed) for m in ORDER},
               "within_vs_tabpfn": auroc_change(b["y"], s[WITHIN], s["tabpfn"], n_boot, ci, seed),
               "within_vs_rule": auroc_change(b["y"], s[WITHIN], s[RULE], n_boot, ci, seed)}
        row["answer"] = "transfers better" if row["within_vs_tabpfn"]["lo"] > 0 else "not shown to transfer better"
        out["transfer"].append(row)
        print(f"done B: {src} -> {dst}: {row['answer']}", flush=True)
        save_results("stage31_trace_free", out, configs)

    out["answers"] = answers(out, cfg, jump_cfg["p_max"])
    path = save_results("stage31_trace_free", out, configs)
    write_report(path, cfg, jump_cfg, out)


def write_report(path, cfg, jump_cfg, out):
    D, ans = out["datasets"], out["answers"]

    def jump_cell(j):
        if "p" not in j:
            return f"- | - | {j['answer']}"
        return f"{fmt(j['auroc_near'])} | {j['reference']['auroc_near_median']:.3f} | {j['p']:.4f} | **{j['answer']}**"

    scores = "\n".join(
        f"| {NAMES[n]} | {split} | {LABELS[m]} | " + " | ".join(fmt(r['scores'][m][k]) for k in
                                                              ("auroc", "pr_auc", "sens_at_90_spec")) + " |"
        for n, d in D.items() for split, r in d["splits"].items() for m in ORDER)
    jumps = "\n".join(f"| {NAMES[n]} | {split} | {LABELS[m]} | {jump_cell(r['jump'][m])} | {fmt(r['jump'][m]['rho_anemic'])} |"
                      for n, d in D.items() for split, r in d["splits"].items() for m in ORDER if m != "dummy")
    change = "\n".join(f"| {NAMES[n]} | {split} | {fmt(r['within_vs_ridge'])} | {fmt(r['within_vs_tabpfn'])} |"
                       for n, d in D.items() for split, r in d["splits"].items())
    clean = ", ".join(f"{NAMES[n]}: {'yes' if v else 'no'}" for n, v in ans["clean_kept"].items())
    tf = "\n".join(f"- **{NAMES[n]}: {fmt(v)}**" for n, v in ans["trace_free_auroc"].items()) or "- none (see the rule above)"
    maes = "\n".join(f"| {NAMES[n]} | {LABELS[m]} | {fmt(d['splits'][CV]['mae'][m])} | "
                     f"{d['splits'][CV]['jump'][m]['answer']} | {fmt(d['splits'][CV]['jump'][m]['rho_anemic'])} |"
                     for n, d in D.items() for m in HB_MODELS[:2])
    transfer = "\n".join(f"| {NAMES[r['train']]} -> {NAMES[r['test']]} | {r['n']} | {r['anemic']} | {LABELS[m]} | " +
                         " | ".join(fmt(r['scores'][m][k]) for k in ("auroc", "pr_auc", "sens_at_90_spec")) + " |"
                         for r in out["transfer"] for m in ORDER)
    tchange = "\n".join(f"| {NAMES[r['train']]} -> {NAMES[r['test']]} | {fmt(r['within_vs_tabpfn'])} | "
                        f"{fmt(r['within_vs_rule'])} | **{r['answer']}** |" for r in out["transfer"])
    feats = "\n".join(
        f"| {NAMES[n]} | {sum(f['jumps'] for f in d['features'])} of {len(d['features'])} | " +
        (", ".join(f"{f['feature']} ({f['auroc']:.2f} / {f['auroc_near']['estimate']:.2f} vs "
                   f"{f['reference_near_median']:.2f})" for f in sorted(d["features"], key=lambda f: -f["auroc"])
                   if f["jumps"]) or "-") + " |" for n, d in D.items())
    report = (
        "# Stage 31: how much of the Ghana score is real anemia signal once the folder trace is taken out?\n\n"
        f"Generated by `python -m src.eyelid.trace_free` at commit `{_short_commit(path)}`; settings and the answers "
        "(fixed before running) in `configs/trace_free.yaml`.\n\n"
        "- **Cutoffs, one per dataset:** Ghana eyelids, nails, palms: children 6-59 months, **Hb < 11.0 g/dL**. India "
        "and Italy eyelids (Eyes-Defy adults): **WHO by sex, women < 12.0, men < 13.0 g/dL**, pregnancy not recorded. "
        "Moscow nails (adults, no sex column): **Hb < 12.0 g/dL** for everyone.\n"
        "- **Input:** the same 29 color numbers. 5-fold CV x 5 repeats by photo / patient / child, scores averaged over "
        "repeats; Ghana eyelids also leave one hospital out. Bootstrap 95% CIs, 2000 resamples. Hb models estimate Hb "
        "minus the person's own cutoff; the score is minus the estimate.\n"
        "- **Within-folder ridge:** in each training fold, each label group's mean is taken out of the 29 numbers and "
        "of Hb, and ridge is fit on what is left, so only differences inside a group shape the weights. Test photos "
        "are scored as they are (their label is never used). A trace shared by a whole folder can't be learned this "
        "way.\n\n"
        "## A. The models on each dataset\n\n"
        "| dataset | split | model | AUROC | PR-AUC | sens @ 90% spec |\n|---|---|---|---|---|---|\n" + scores + "\n\n"
        f"### Does each score jump at the cutoff? (Stage 30's test: within {jump_cfg['window_g_dl']} g/dL, "
        f"{jump_cfg['n_sims']} steady-reference draws, jumps if p < {jump_cfg['p_max']})\n\n"
        "| dataset | split | model | near-cutoff AUROC | steady reference (median) | p | answer | rank corr. with "
        "Hb, anemic only |\n|---|---|---|---|---|---|---|---|\n" + jumps + "\n\n"
        "### Within-folder ridge minus the others (paired)\n\n"
        "| dataset | split | minus ridge on Hb | minus TabPFN classifier |\n|---|---|---|---|\n" + change + "\n\n"
        f"- **Keeps the signal on the clean datasets** (change vs ridge on Hb above -{cfg['keep_drop']} and CI upper "
        f"bound above 0): {clean}. Method trusted (at least {cfg['min_clean_kept']} of 3): "
        f"**{'yes' if ans['method_trusted'] else 'no'}**.\n"
        "- **Free of the jump:** " + ", ".join(f"{NAMES[n]}: {'yes' if v else 'no'}"
                                               for n, v in ans["free_of_jump"].items()) + ".\n"
        "- **Ghana's trace-free number** (within-folder ridge AUROC, stated only if free of the jump there and the "
        "method is trusted):\n" + tf + "\n\n"
        "## B. Transfer: fit on all of one dataset, applied unchanged to another\n\n"
        "| train -> test | n | anemic | model | AUROC | PR-AUC | sens @ 90% spec |\n|---|---|---|---|---|---|---|\n" +
        transfer + "\n\n| train -> test | within-folder ridge minus TabPFN classifier | minus rule | answer |\n"
        "|---|---|---|---|\n" + tchange + "\n\n"
        "## C. Do the Hb-number models jump too? (5x5 CV)\n\n"
        "| dataset | model | Hb error (MAE, g/dL) | jump answer | rank corr. with Hb, anemic only |\n"
        "|---|---|---|---|---|\n" + maes + "\n\n"
        "## D. Which single color numbers jump at the cutoff?\n\n"
        f"Each number on its own, turned so higher = more anemic; jumps if p < {jump_cfg['p_max']} / 29 (Bonferroni). "
        "In brackets: its AUROC on everyone / near the cutoff vs the steady reference's median near the cutoff (a "
        "steady signal is always weaker near the cutoff than on everyone).\n\n"
        "| dataset | numbers that jump | which |\n|---|---|---|\n" + feats + "\n")
    (ROOT / "reports" / "stage31_trace_free.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
