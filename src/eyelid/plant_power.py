"""Stage 41: do the photo-edit test, the step at the cutoff and Hb inside the anemic group catch a trace we plant?

Stage 40 put three of the paper draft's checks (jump test, folder-only line, not-useful numbers) on Stage 33's planted
folder traces; none caught the gamma and saturation plants on India. Here the other three color checks go on the same
plants and clean scores, and all six are combined. Audit controls only (India, Italy, Moscow). Settings and the answers,
fixed before running, in configs/plant_power.yaml (data, models and thresholds from configs/check_power.yaml).

Run from the repo root: python -m src.eyelid.plant_power
"""
import json

import numpy as np

from src.common.evaluate import auroc_change
from src.common.io import RESULTS, ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features
from src.eyelid.edit_robustness import moscow_pixels
from src.eyelid.handling_check import NAMES, RULE, datasets, edit, edit_list, folds_for, held_out, jump_test
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.severity import rank_corr
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.trace_split import LABELS, normal_scores, planted, step_at_cutoff
from src.eyelid.train_color import _short_commit

CHECKS = {"jump": "jump test (Stage 30)", "folder": "folder-only line (Stage 39)",
          "not_useful": "not-useful numbers (Stage 37)", "edit": "photo-edit test (Stage 32)",
          "step": "step at the cutoff (Stage 33)", "hb_inside": "Hb inside the anemic group (Stage 30)"}
NEW = ("edit", "step", "hb_inside")


def plant_edit_numbers(pixels, plants, edits):
    """The 29 numbers of every photo with no plant (key None) or each plant, each as it is (None) and under each test
    edit: {plant: {edit: (n, 29)}}. The plant is part of the photo, so the test edit goes on top of it."""
    nums = lambda px: list(color_features(px).values())
    keys = [(None, None, None), *plants]
    out = {p: {e: [] for e in [None, *(lab for lab, _, _ in edits)]} for p, _, _ in keys}
    for px in pixels:
        for plab, kind, value in keys:
            base = px if kind is None else edit(px, kind, value)
            out[plab][None].append(nums(base))
            for elab, ek, ev in edits:
                out[plab][elab].append(nums(edit(base, ek, ev)))
    return {p: {e: np.array(v) for e, v in d.items()} for p, d in out.items()}


def breaks(change, rule_change, cfg):
    """Stage 32's rule: the edit costs at least fragile_drop (CI below 0) while the rule moves less than
    rule_stable_drop."""
    return bool(change["hi"] < 0 and change["estimate"] <= -cfg["fragile_drop"]
                and rule_change["estimate"] > -cfg["rule_stable_drop"])


def checks(y, z, held, models, edit_cfg, min_broken, window, n_boot, ci, seed):
    """The three checks for each model (and the rule, without the edit test) on one version of the data.
    held: {"as is" or edit label: {model: score (n,) or (n_repeats, n)}}."""
    edits = [e for e in held if e != "as is"]
    rule_change = {e: auroc_change(y, held[e][RULE], held["as is"][RULE], n_boot, ci, seed) for e in edits}
    an, out = z < 0, {}
    for m in [*models, RULE]:
        s = np.atleast_2d(held["as is"][m]).mean(0)
        step = bootstrap_ci(lambda zz, qq: step_at_cutoff(zz, qq, window), z, normal_scores(s), n_boot, ci, seed=seed)
        rho = bootstrap_ci(lambda a, b: rank_corr(b, a), z[an], s[an], n_boot, ci, seed=seed)
        r = {"auroc": bootstrap_ci(auroc, y, s, n_boot, ci, seed=seed), "step": step | {"flags": step["lo"] > 0},
             "hb_inside": rho | {"flags": rho["hi"] >= 0}}
        if m != RULE:
            ch = {e: auroc_change(y, held[e][m], held["as is"][m], n_boot, ci, seed) for e in edits}
            broken = [e for e in edits if breaks(ch[e], rule_change[e], edit_cfg)]
            r["edit"] = {"changes": ch, "broken_by": broken, "flags": len(broken) >= min_broken}
        out[m] = r
    return out


def answers(clean, plants, cfg, trained, min_gain, india):
    """Per check: clean scores flagged (trained models) and counting plants caught (a check that can't test a score
    counts as a miss); the union of the checks with at most max_false_flags clean flags; India's missed plants."""
    counting = [r for r in plants if r["gain"] >= min_gain]
    flag = lambda r, k: bool(r.get(k) and r[k]["flags"])
    res = {"counting": len(counting), "checks": {}}
    for k in CHECKS:
        false = sum(flag(clean[d][m], k) for d in clean for m in trained)
        caught = sum(flag(r, k) for r in counting)
        res["checks"][k] = {"false_flags": false, "caught": caught,
                            "india_caught": sum(flag(r, k) for r in counting if india(r)),
                            "useful": false <= cfg["max_false_flags"] and caught >= cfg["min_caught"]}
    ok = [k for k in CHECKS if res["checks"][k]["false_flags"] <= cfg["max_false_flags"]]
    res["combined"] = {"checks": ok,
                       "false_flags": sum(any(flag(clean[d][m], k) for k in ok) for d in clean for m in trained),
                       "caught": sum(any(flag(r, k) for k in ok) for r in counting)}
    res["all_six"] = {"false_flags": sum(any(flag(clean[d][m], k) for k in CHECKS) for d in clean for m in trained),
                      "caught": sum(any(flag(r, k) for k in CHECKS) for r in counting)}
    res["india_missed"] = {"n": sum(map(india, counting)),
                           "caught_by_any": sum(any(flag(r, k) for k in CHECKS) for r in counting if india(r))}
    res["of_clean"] = len(clean) * len(trained)
    return res


def main():
    base, cfg, cp_cfg, color_cfg = (load_config(n) for n in ("default", "plant_power", "check_power", "color"))
    ts_cfg, jump_cfg, edit_cfg = load_config("trace_split"), load_config("handling_check"), load_config("edit_robustness")
    probe_cfg, ceiling_cfg = load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "plant_power": cfg, "check_power": cp_cfg, "trace_split": ts_cfg,
               "handling_check": jump_cfg, "edit_robustness": edit_cfg, "color": color_cfg, "ceiling": ceiling_cfg}
    prior = json.loads((RESULTS / "stage40_check_power.json").read_text())
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    makers = {m: ladder[m][1] for m in cp_cfg["models"]}
    trained = [m for m in cp_cfg["models"] if m != "dummy"]
    a_col = feature_names().index(color_cfg["rule_feature"])
    plants = edit_list({"edits": ts_cfg["plants"]}, None)
    edits = [e for e in edit_list(jump_cfg, (1.0, 1.0, 1.0)) if e[1] != "gains"]  # batch61 is Ghana-only
    window, min_gain = ts_cfg["window_g_dl"], ts_cfg["plant_min_gain"]
    args = (edit_cfg, cfg["edit_min_broken"], window, n_boot, ci, seed)
    p40 = {(r["dataset"], r["plant"], r["model"]): r for r in prior["results"]["planted"]}

    out = {"clean": {}, "planted": [], "edits": [e[0] for e in edits]}
    for name, d in datasets(cp_cfg, base, color_cfg, seed).items():
        y, z = d["y"], d["hb"] - d["cut"]
        folds = folds_for(y, d["groups"], cp_cfg["cv"], seed)
        pixels = moscow_pixels() if d["paths"] is None else (roi_pixels(p, alpha) for p in d["paths"])
        nums = plant_edit_numbers(pixels, plants, edits)
        assert np.allclose(nums[None][None], d["X"], rtol=1e-4, atol=1e-6), f"{name}: pixels don't give the numbers"

        def held_for(plab):
            Xs = {e: nums[None][e] if plab is None else planted(nums[None][e], nums[plab][e], y) for e in nums[None]}
            h = held_out(makers, Xs[None], y, folds, {e: Xs[e] for e in Xs if e is not None})
            return {v: h[v] | {RULE: -Xs[None if v == "as is" else v][:, a_col]} for v in h}

        held0 = held_for(None)
        clean = checks(y, z, held0, trained, *args)
        prior_clean = prior["results"]["clean"][name]
        for m in trained:
            clean[m] |= {"jump": {"flags": bool(jump_test(z, held0["as is"][m].mean(0), jump_cfg, rng, n_boot, ci,
                                                          seed).get("jumps", False))},
                         "folder": prior_clean["folder"][m], "not_useful": prior_clean["not_useful"][m]}
        out["clean"][name] = clean
        for plab, _, _ in plants:
            res = checks(y, z, held_for(plab), trained, *args)
            for m, r in res.items():
                p = p40[(name, plab, m)]  # counting plants by Stage 40's gain keeps the same plant set
                gain = r["auroc"]["estimate"] - clean[m]["auroc"]["estimate"]
                assert abs(p["gain"] - gain) < 0.01, f"{name} {plab} {m}: gain {gain:.3f}, Stage 40 {p['gain']:.3f}"
                out["planted"].append({"dataset": name, "plant": plab, "model": m, "gain": p["gain"],
                                       "gain_here": gain, **r,
                                       "jump": {"flags": bool(p["jumps"])} if p["jumps"] is not None else None,
                                       "folder": p["folder"], "not_useful": p["not_useful"]})
            print(f"done {name} {plab}", flush=True)
        save_results("stage41_plant_power", out, configs)
    india = lambda r: r["dataset"] == "india_eyelid" and not r["plant"].startswith("white balance")
    out["answers"] = answers(out["clean"], out["planted"], cp_cfg, trained, min_gain, india)
    path = save_results("stage41_plant_power", out, configs)
    write_report(path, cp_cfg, ts_cfg, out)


def write_report(path, cp_cfg, ts_cfg, out):
    a, min_gain = out["answers"], ts_cfg["plant_min_gain"]
    yes = lambda b: "**yes**" if b else "no"
    n, nc = a["counting"], a["of_clean"]
    summary = [f"| {CHECKS[k]} | {c['false_flags']} of {nc} | {c['caught']} of {n} | {c['india_caught']} of "
               f"{a['india_missed']['n']} | {yes(c['useful'])} |" for k, c in a["checks"].items()]
    comb = a["combined"]
    mark = lambda r, k: ("-" if not r.get(k) else yes(r[k]["flags"]))
    clean = [f"| {NAMES[d]} | {LABELS[m]} | {fmt(r['auroc'])} | "
             + (f"{', '.join(r['edit']['broken_by']) or 'none'} | " if "edit" in r else "- | ")
             + f"{fmt(r['step'])} | {yes(r['step']['flags'])} | {fmt(r['hb_inside'])} | {yes(r['hb_inside']['flags'])} | "
             + f"{mark(r, 'jump')} |"
             for d, rs in out["clean"].items() for m, r in rs.items()]
    plants = [f"| {NAMES[r['dataset']]} | {r['plant']} | {LABELS[r['model']]} | {r['gain']:+.3f} | "
              + " | ".join(mark(r, k) for k in CHECKS) + " |" for r in out["planted"] if r["gain"] >= min_gain]
    report = (
        "# Stage 41: do the photo-edit test, the step at the cutoff and Hb inside the anemic group catch a trace we "
        "plant?\n\n"
        f"Generated by `python -m src.eyelid.plant_power` at commit `{_short_commit(path)}`; settings and the answers "
        "(fixed before running) in `configs/plant_power.yaml`, data and thresholds in `configs/check_power.yaml`.\n\n"
        "- **Audit controls only, no new Ghana claims.** India and Italy eyelids (Eyes-Defy adults), WHO by sex, "
        "**women < 12.0, men < 13.0 g/dL** (pregnancy not recorded); Moscow nails, **Hb < 12.0 g/dL** for everyone (no "
        "sex column).\n"
        f"- **Scores:** Stage 40's: {cp_cfg['cv']['n_splits']}-fold CV x {cp_cfg['cv']['n_repeats']} repeats; logistic "
        "regression, gradient boosting and TabPFN on the 29 color numbers, the untrained rule (minus mean a*); the same "
        "planted traces (one mild edit of the anemic photos only). Bootstrap 95% CIs, 2000 resamples.\n"
        f"- **Checks:** photo-edit test = at least 1 of the 7 Stage 32 edits ({', '.join(out['edits'])}) costs 0.05 "
        "or more (CI below 0) while the rule moves less than 0.02 (trained models only); step = the step at the cutoff "
        "has its CI above 0; Hb inside the anemic group = the score's rank correlation with Hb there doesn't have its "
        "CI below 0. The jump test on clean scores is run here; on the plants, and the folder line and the not-useful "
        "numbers, come from Stage 40.\n"
        f"- **Answers, fixed before running:** a check is useful if it flags at most {cp_cfg['max_false_flags']} of "
        f"the {nc} clean scores and catches at least {cp_cfg['min_caught']} of the {n} counting plants (gain >= "
        f"{min_gain}); combined = the union of the checks with at most {cp_cfg['max_false_flags']} clean flag.\n\n"
        "## Answer\n\n| check | clean scores flagged | plants caught | India gamma / saturation plants caught | "
        "useful? |\n|---|---|---|---|---|\n" + "\n".join(summary) + "\n\n"
        f"- **Combined** ({', '.join(CHECKS[k] for k in comb['checks']) or 'no check qualifies'}): "
        f"{comb['false_flags']} of {nc} clean scores flagged, {comb['caught']} of {n} plants caught.\n"
        f"- **All six together:** {a['all_six']['false_flags']} of {nc} clean scores flagged, {a['all_six']['caught']} "
        f"of {n} plants caught.\n"
        f"- **India's gamma and saturation plants:** {a['india_missed']['caught_by_any']} of {a['india_missed']['n']} "
        "caught by any check.\n\n"
        "## Clean data, nothing planted\n\n| dataset | score | AUROC | edits that break it | step at the cutoff "
        "(score SDs) | step flags? | rank corr. with Hb, anemic only | flags? | jump test flags? |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
        + "\n".join(clean) + "\n\n"
        "## Planted traces that count: which check flags each\n\n| dataset | plant | score | gain | "
        + " | ".join(CHECKS.values()) + " |\n|" + "---|" * (4 + len(CHECKS)) + "\n" + "\n".join(plants) + "\n")
    (ROOT / "reports" / "stage41_plant_power.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
