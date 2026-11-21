"""Stage 40: do the two newest trace checks pass clean data, and do they catch a trace we plant?

The folder-only line (Stage 39) and the "not useful" numbers (Stage 37) were run on Ghana only. Here both go on the clean
sets (India and Italy eyelids, Moscow nails), as they are and with Stage 33's planted folder traces, next to Stage 30's
jump test on the same plants. Audit controls only. Settings and the answer, fixed before running, in
configs/check_power.yaml.

Run from the repo root: python -m src.eyelid.check_power
"""
import numpy as np

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.edit_robustness import moscow_pixels
from src.eyelid.handling_check import NAMES, RULE, datasets, edit_list, folds_for, held_out, jump_test
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.severity_reference import beyond_folder
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.trace_split import LABELS, plant_numbers, planted
from src.eyelid.train_color import _short_commit


def folder_check(z, s, depth, n_boot, ci, seed):
    """AUROC more than `depth` g/dL below each person's own cutoff minus the folder-only line (z = Hb minus cutoff).
    Flags the score unless the CI is above 0 (Stage 39's rule)."""
    r = bootstrap_ci(beyond_folder(-depth, 0.0), (z < 0).astype(int), np.column_stack([s, z]), n_boot, ci, seed=seed)
    return r | {"flags": r["lo"] <= 0}


def not_useful_check(y, nu, pale, rule, margin, n_boot, ci, seed):
    """Stage 37's rule for one model: on the numbers that shouldn't carry paleness it scores at least its paleness
    AUROC minus `margin` and beats the rule (paired CI above 0)."""
    a_nu, a_pale = (bootstrap_ci(auroc, y, s, n_boot, ci, seed=seed) for s in (nu, pale))
    vs_rule = auroc_change(y, nu, rule, n_boot, ci, seed)
    return {"not_useful": a_nu, "paleness": a_pale, "vs_rule": vs_rule,
            "flags": bool(a_nu["estimate"] >= a_pale["estimate"] - margin and vs_rule["lo"] > 0)}


def checks(X, y, z, folds, makers, trained, cols, a_col, cfg, n_boot, ci, seed):
    """Both checks on one version of the data (as it is, or with a planted trace). Returns the results and the scores
    on all 29 numbers (trained models and the rule)."""
    fit = lambda Xs: {m: s.mean(0) for m, s in held_out(makers, Xs, y, folds)["as is"].items()}
    full = fit(X) | {RULE: -X[:, a_col]}
    sub = {k: fit(X[:, c]) for k, c in cols.items()}
    return {"metrics": {m: summarize(y, s, n_boot, ci, seed) for m, s in full.items()},
            "folder": {m: folder_check(z, s, cfg["depth_g_dl"], n_boot, ci, seed) for m, s in full.items()},
            "not_useful": {m: not_useful_check(y, sub["not_useful"][m], sub["paleness"][m], full[RULE],
                                               cfg["weird_margin"], n_boot, ci, seed) for m in trained}}, full


def answers(out, cfg, models, min_gain):
    """False flags on the clean scores (trained models), and plants caught among those that count. A plant on the rule
    has no not-useful check, so it counts as missed there."""
    counting = [r for r in out["planted"] if r["gain"] >= min_gain]
    res = {"counting": len(counting), "jump_test": {"caught": sum(bool(r["jumps"]) for r in counting)}}
    for check in ("folder", "not_useful"):
        false = sum(bool(d[check][m]["flags"]) for d in out["clean"].values() for m in models)
        caught = sum(bool(r[check] and r[check]["flags"]) for r in counting)
        res[check] = {"false_flags": false, "of_clean": len(out["clean"]) * len(models), "caught": caught,
                      "useful": false <= cfg["max_false_flags"] and caught >= cfg["min_caught"]}
    return res


def main():
    base, cfg, color_cfg = load_config("default"), load_config("check_power"), load_config("color")
    ts_cfg, mvf_cfg, jump_cfg = load_config("trace_split"), load_config("model_vs_features"), load_config("handling_check")
    probe_cfg, ceiling_cfg = load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "check_power": cfg, "trace_split": ts_cfg, "model_vs_features": mvf_cfg,
               "handling_check": jump_cfg, "color": color_cfg, "ceiling": ceiling_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    makers = {m: ladder[m][1] for m in cfg["models"]}
    trained = [m for m in cfg["models"] if m != "dummy"]
    names = feature_names()
    cols = {k: [names.index(c) for c in mvf_cfg["input_sets"][k]] for k in ("not_useful", "paleness")}
    a_col = names.index(color_cfg["rule_feature"])
    plants = edit_list({"edits": ts_cfg["plants"]}, None)

    out = {"clean": {}, "planted": [], "people_below_depth": {}}
    for name, d in datasets(cfg, base, color_cfg, seed).items():
        y, X, z = d["y"], d["X"], d["hb"] - d["cut"]
        out["people_below_depth"][name] = int((z < -cfg["depth_g_dl"]).sum())
        folds = folds_for(y, d["groups"], cfg["cv"], seed)
        args = (y, z, folds, makers, trained, cols, a_col, cfg, n_boot, ci, seed)
        out["clean"][name], _ = checks(X, *args)
        pixels = moscow_pixels() if d["paths"] is None else (roi_pixels(p, alpha) for p in d["paths"])
        X0, edited = plant_numbers(pixels, plants)
        assert np.allclose(X0, X, rtol=1e-4, atol=1e-6), f"{name}: pixels don't give the stored numbers"
        for lab, Xe in edited.items():
            res, scores = checks(planted(X, Xe, y), *args)
            for m, s in scores.items():
                gain = res["metrics"][m]["auroc"]["estimate"] - out["clean"][name]["metrics"][m]["auroc"]["estimate"]
                jumps = (jump_test(z, s, jump_cfg, rng, n_boot, ci, seed).get("jumps", False)
                         if gain >= ts_cfg["plant_min_gain"] else None)
                out["planted"].append({"dataset": name, "plant": lab, "model": m, "auroc": res["metrics"][m]["auroc"],
                                       "gain": gain, "jumps": jumps, "folder": res["folder"][m],
                                       "not_useful": res["not_useful"].get(m)})
            print(f"done {name} {lab}", flush=True)
        save_results("stage40_check_power", out, configs)
    out["answers"] = answers(out, cfg, trained, ts_cfg["plant_min_gain"])
    path = save_results("stage40_check_power", out, configs)
    write_report(path, cfg, ts_cfg, out)


def write_report(path, cfg, ts_cfg, out):
    a, depth, min_gain = out["answers"], cfg["depth_g_dl"], ts_cfg["plant_min_gain"]
    yes = lambda b: "**yes**" if b else "no"
    clean = []
    for name, d in out["clean"].items():
        for m in sorted(d["metrics"], key=list(LABELS).index):
            r, nu = d["metrics"][m], d["not_useful"].get(m)
            clean.append(f"| {NAMES[name]} | {LABELS[m]} | {fmt(r['auroc'])} | {fmt(r['pr_auc'])} | "
                         f"{fmt(r['sens_at_90_spec'])} | {fmt(d['folder'][m])} | {yes(d['folder'][m]['flags'])} | "
                         + (f"{nu['not_useful']['estimate']:.3f} vs {nu['paleness']['estimate']:.3f} | "
                            f"{yes(nu['flags'])} |" if nu else "- | - |"))
    plants = [f"| {NAMES[r['dataset']]} | {r['plant']} | {LABELS[r['model']]} | {fmt(r['auroc'])} | {r['gain']:+.3f} | "
              f"{yes(r['jumps'])} | {fmt(r['folder'])} | {yes(r['folder']['flags'])} | "
              + (f"{r['not_useful']['not_useful']['estimate']:.3f} vs {r['not_useful']['paleness']['estimate']:.3f} | "
                 f"{yes(r['not_useful']['flags'])} |" if r["not_useful"] else "- | - |")
              for r in out["planted"] if r["gain"] >= min_gain]
    n = a["counting"]
    summary = [f"| jump test (Stage 30) | - | {a['jump_test']['caught']} of {n} | - |"] + [
        f"| {label} | {a[k]['false_flags']} of {a[k]['of_clean']} | {a[k]['caught']} of {n} | {yes(a[k]['useful'])} |"
        for k, label in (("folder", "folder-only line (Stage 39)"), ("not_useful", "not-useful numbers (Stage 37)"))]
    below = ", ".join(f"{NAMES[k]} {v}" for k, v in out["people_below_depth"].items())
    report = (
        "# Stage 40: do the two newest trace checks pass clean data, and do they catch a trace we plant?\n\n"
        f"Generated by `python -m src.eyelid.check_power` at commit `{_short_commit(path)}`; settings and the answer "
        "(fixed before running) in `configs/check_power.yaml`.\n\n"
        "- **Audit controls only, no new Ghana claims.** Cutoffs: India and Italy eyelids (Eyes-Defy adults), WHO by "
        "sex, **women < 12.0, men < 13.0 g/dL** (pregnancy not recorded); Moscow nails, **Hb < 12.0 g/dL** for everyone "
        "(no sex column).\n"
        f"- **Scores:** {cfg['cv']['n_splits']}-fold CV x {cfg['cv']['n_repeats']} repeats, held-out scores averaged "
        "over repeats; logistic regression, gradient boosting and TabPFN on the 29 color numbers, the untrained rule "
        "(minus mean a*) and the dummy (not counted in the answer). Bootstrap 95% CIs, 2000 resamples.\n"
        f"- **Folder-only line** (Stage 39): AUROC more than {depth:g} g/dL below each person's own cutoff minus "
        "(H x A + 0.5 x M) / (H + M); flags the score unless the CI is above 0. People that far below: " + below + ".\n"
        "- **Not-useful numbers** (Stage 37): a model on hue, green, blue and yellow-blue scores at least its "
        f"paleness-numbers AUROC minus {cfg['weird_margin']} and beats the rule (CI above 0). Trained models only.\n"
        "- **Planted traces** (Stage 33): one mild edit of the anemic photos only (gamma 0.9, 10% less saturation, or a "
        f"4% white-balance shift), models retrained. A plant counts if it lifts the AUROC by at least {min_gain}.\n"
        f"- **Answer, fixed before running:** a check is useful if it flags at most {cfg['max_false_flags']} of the 9 "
        f"clean scores (trained models) and catches at least {cfg['min_caught']} of the counting plants.\n\n"
        "## Answer\n\n| check | flags on clean data | plants caught | useful? |\n|---|---|---|---|\n"
        + "\n".join(summary) + "\n\n"
        "## Clean data, nothing planted\n\n"
        f"| dataset | score | AUROC | PR-AUC | sens @ 90% spec | more than {depth:g} below minus the folder line | "
        "folder line flags? | not useful vs paleness | not-useful flags? |\n|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(clean) + "\n\n"
        "## Planted traces that count\n\n"
        "| dataset | plant | score | AUROC planted | gain | jump test flags? | minus the folder line | folder line flags? "
        "| not useful vs paleness | not-useful flags? |\n|---|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(plants) + "\n")
    (ROOT / "reports" / "stage40_check_power.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
