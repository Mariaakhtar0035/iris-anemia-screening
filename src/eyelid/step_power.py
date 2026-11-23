"""Stage 42: is the step at the cutoff solid, at other windows and on fewer people?

Stage 41 found the step at the cutoff (Stage 33) is the one useful trace check: no clean score flagged, 17 of 24 planted
traces caught. It was run at one window (2 g/dL) and on each set's full size only. Here the same scores (clean and with
Stage 33's planted traces) go through the step at four windows, and through the step on random subsets of 95 and 150
people. Audit controls only (India, Italy, Moscow). Settings and the answers, fixed before running, in
configs/step_power.yaml (data, models and thresholds from configs/check_power.yaml).

Run from the repo root: python -m src.eyelid.step_power
"""
import json

import numpy as np

from src.common.io import RESULTS, ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.edit_robustness import moscow_pixels
from src.eyelid.handling_check import NAMES, RULE, datasets, edit_list, folds_for, held_out
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.trace_split import LABELS, normal_scores, plant_numbers, planted, step_at_cutoff
from src.eyelid.train_color import _short_commit

wkey = lambda w: f"{w:g}"


def step_check(z, s, window, n_boot, ci, seed):
    """Stage 33's step at the cutoff on one score (in normal scores); flags if the CI is above 0."""
    r = bootstrap_ci(lambda zz, qq: step_at_cutoff(zz, qq, window), z, normal_scores(s), n_boot, ci, seed=seed)
    return r | {"flags": bool(r["lo"] > 0)}


def subsets(y, size, draws, rng):
    """`draws` random sets of `size` people, each keeping the set's anemic share."""
    an, he = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    k = round(size * len(an) / len(y))
    return [np.concatenate([rng.choice(an, k, replace=False), rng.choice(he, size - k, replace=False)])
            for _ in range(draws)]


def rate(flags, n_boot, ci, seed):
    """Share flagged: flags (draws, scores) averaged over scores in each draw, then over draws; CI over draws."""
    per_draw = np.asarray(flags, float).mean(1)
    return bootstrap_ci(lambda a, b: float(b.mean()), per_draw, per_draw, n_boot, ci, seed=seed)


def answers(rows, cfg, cp_cfg, window, india, n_boot, ci, seed):
    """Per window: clean trained scores flagged and counting plants caught. Per size: catch and false-flag rates on
    the sets with more people than the size, next to the same plants at full size."""
    clean = [r for r in rows if r["plant"] is None]
    plants = [r for r in rows if r["plant"] is not None]
    res = {"of_clean": len(clean), "counting": len(plants), "windows": {}, "sizes": {}}
    for w in cfg["windows_g_dl"]:
        f = lambda r: r["windows"][wkey(w)]["flags"]
        false, caught = sum(map(f, clean)), sum(map(f, plants))
        res["windows"][wkey(w)] = {"false_flags": false, "caught": caught,
                                   "india_caught": sum(f(r) for r in plants if india(r)),
                                   "passes": false <= cp_cfg["max_false_flags"] and caught >= cp_cfg["min_caught"]}
    res["robust"] = all(v["passes"] for v in res["windows"].values())
    for size in cfg["sizes"]:
        pl, cl = ([r for r in g if size in r["sizes"]] for g in (plants, clean))
        if not pl:
            continue
        flags = lambda g: np.array([r["sizes"][size] for r in g]).T  # (draws, scores)
        res["sizes"][size] = {
            "datasets": sorted({r["dataset"] for r in pl}), "n_plants": len(pl), "n_clean": len(cl),
            "catch_rate": rate(flags(pl), n_boot, ci, seed), "false_flag_rate": rate(flags(cl), n_boot, ci, seed),
            "full_size_caught": sum(r["windows"][wkey(window)]["flags"] for r in pl),
            "by_dataset": {d: rate(flags([r for r in pl if r["dataset"] == d]), n_boot, ci, seed)
                           for d in sorted({r["dataset"] for r in pl})}}
    halves = [s for s in sorted(res["sizes"]) if res["sizes"][s]["catch_rate"]["estimate"] >= cfg["half"]]
    res["smallest_half"] = halves[0] if halves else None
    return res


def main():
    base, cfg, cp_cfg, color_cfg = (load_config(n) for n in ("default", "step_power", "check_power", "color"))
    ts_cfg, probe_cfg, ceiling_cfg = load_config("trace_split"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "step_power": cfg, "check_power": cp_cfg, "trace_split": ts_cfg, "color": color_cfg,
               "ceiling": ceiling_cfg}
    p40 = {(r["dataset"], r["plant"], r["model"]): r
           for r in json.loads((RESULTS / "stage40_check_power.json").read_text())["results"]["planted"]}
    p41 = json.loads((RESULTS / "stage41_plant_power.json").read_text())["results"]
    s41 = {(r["dataset"], r["plant"], r["model"]): r["step"]["flags"] for r in p41["planted"]}
    s41 |= {(d, None, m): r["step"]["flags"] for d, rs in p41["clean"].items() for m, r in rs.items()}
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    trained = [m for m in cp_cfg["models"] if m != "dummy"]
    makers = {m: ladder[m][1] for m in trained}
    a_col = feature_names().index(color_cfg["rule_feature"])
    plants = edit_list({"edits": ts_cfg["plants"]}, None)
    window, min_gain = ts_cfg["window_g_dl"], ts_cfg["plant_min_gain"]
    rng = np.random.default_rng(seed)

    out = {"people_within": {}, "rows": []}
    for name, d in datasets(cp_cfg, base, color_cfg, seed).items():
        y, X, z = d["y"], d["X"], d["hb"] - d["cut"]
        out["people_within"][name] = {wkey(w): int((np.abs(z) < w).sum()) for w in cfg["windows_g_dl"]}
        folds = folds_for(y, d["groups"], cp_cfg["cv"], seed)
        fit = lambda Xs: ({m: s.mean(0) for m, s in held_out(makers, Xs, y, folds)["as is"].items()}
                          | {RULE: -Xs[:, a_col]})
        pixels = moscow_pixels() if d["paths"] is None else (roi_pixels(p, alpha) for p in d["paths"])
        X0, edited = plant_numbers(pixels, plants)
        assert np.allclose(X0, X, rtol=1e-4, atol=1e-6), f"{name}: pixels don't give the stored numbers"
        sets = {size: subsets(y, size, cfg["draws"], rng) for size in cfg["sizes"] if size < len(y)}
        clean = fit(X)
        for plab, Xp in [(None, X), *((lab, planted(X, Xe, y)) for lab, Xe in edited.items())]:
            scores = clean if plab is None else fit(Xp)
            for m, s in scores.items():
                if plab is None and m == RULE:
                    continue  # the clean rule isn't one of the 9 clean scores
                gain = None
                if plab is not None:
                    gain, here = p40[(name, plab, m)]["gain"], auroc(y, s) - auroc(y, clean[m])
                    assert abs(gain - here) < 0.01, f"{name} {plab} {m}: gain {here:.3f}, Stage 40 {gain:.3f}"
                    if gain < min_gain:
                        continue
                out["rows"].append({
                    "dataset": name, "plant": plab, "model": m, "gain": gain, "stage41_flags": s41[(name, plab, m)],
                    "windows": {wkey(w): step_check(z, s, w, n_boot, ci, seed) for w in cfg["windows_g_dl"]},
                    "sizes": {size: [step_check(z[i], s[i], window, n_boot, ci, seed)["flags"] for i in idx]
                              for size, idx in sets.items()}})
            print(f"done {name} {plab}", flush=True)
        save_results("stage42_step_power", out, configs)
    india = lambda r: r["dataset"] == "india_eyelid" and not r["plant"].startswith("white balance")
    out["answers"] = answers(out["rows"], cfg, cp_cfg, window, india, n_boot, ci, seed)
    path = save_results("stage42_step_power", out, configs)
    write_report(path, cfg, cp_cfg, ts_cfg, out)


def write_report(path, cfg, cp_cfg, ts_cfg, out):
    a, window, yes = out["answers"], wkey(ts_cfg["window_g_dl"]), (lambda b: "**yes**" if b else "no")
    ws = [wkey(w) for w in cfg["windows_g_dl"]]
    n, nc, ni = a["counting"], a["of_clean"], sum(1 for r in out["rows"] if r["dataset"] == "india_eyelid"
                                                  and r["plant"] and not r["plant"].startswith("white balance"))
    within = lambda w: ", ".join(f"{v[w]}" for v in out["people_within"].values())
    agree = sum(r["windows"][window]["flags"] == r["stage41_flags"] for r in out["rows"])
    win = [f"| {w} g/dL | {within(w)} | {v['false_flags']} of {nc} | {v['caught']} of {n} | {v['india_caught']} of {ni} "
           f"| {yes(v['passes'])} |" for w, v in a["windows"].items()]
    pct = lambda r: f"{r['estimate']:.0%} [{r['lo']:.0%}, {r['hi']:.0%}]"
    size = [f"| {s} | {', '.join(NAMES[d] for d in v['datasets'])} | {v['n_plants']} | {pct(v['catch_rate'])} | "
            f"{v['full_size_caught']} of {v['n_plants']} | {pct(v['false_flag_rate'])} ({v['n_clean']} scores) | "
            + "; ".join(f"{NAMES[d].split(' (')[0]} {pct(r)}" for d, r in v["by_dataset"].items()) + " |"
            for s, v in a["sizes"].items()]
    cell = lambda r: f"**{fmt(r)}**" if r["flags"] else fmt(r)
    rows = [f"| {NAMES[r['dataset']]} | {r['plant'] or 'none (clean)'} | {LABELS[r['model']]} | "
            + (f"{r['gain']:+.3f}" if r["gain"] is not None else "-") + " | "
            + " | ".join(cell(r["windows"][w]) for w in ws) + " | "
            + " | ".join(f"{np.mean(r['sizes'][s]):.0%}" if s in r["sizes"] else "-" for s in cfg["sizes"]) + " |"
            for r in out["rows"]]
    half = a["smallest_half"]
    report = (
        "# Stage 42: is the step at the cutoff solid, at other windows and on fewer people?\n\n"
        f"Generated by `python -m src.eyelid.step_power` at commit `{_short_commit(path)}`; settings and the answers "
        "(fixed before running) in `configs/step_power.yaml`, data and thresholds in `configs/check_power.yaml`.\n\n"
        "- **Audit controls only, no new Ghana claims.** India and Italy eyelids (Eyes-Defy adults), WHO by sex, "
        "**women < 12.0, men < 13.0 g/dL** (pregnancy not recorded); Moscow nails, **Hb < 12.0 g/dL** for everyone (no "
        "sex column).\n"
        f"- **Scores:** as Stages 40-41: {cp_cfg['cv']['n_splits']}-fold CV x {cp_cfg['cv']['n_repeats']} repeats; "
        "logistic regression, gradient boosting and TabPFN on the 29 color numbers, the untrained rule (minus mean a*); "
        "Stage 33's planted traces (one mild edit of the anemic photos only), counted if Stage 40's gain is at least "
        f"{ts_cfg['plant_min_gain']}. Bootstrap 95% CIs, 2000 resamples.\n"
        "- **Step at the cutoff** (Stage 33): the score's normal scores = a + b x (Hb minus cutoff) + step x anemic, fit "
        "on everyone within the window of their own cutoff; flags the score if the step's CI is above 0.\n"
        f"- **Fewer people:** {cfg['draws']} random sets of each size, each keeping the set's anemic share, step at "
        f"{window} g/dL. The models are not refit, so this is how many people the check needs to see a trace of this "
        "size. Rate CIs are over draws.\n"
        f"- **Answers, fixed before running:** robust to the window if every window flags at most "
        f"{cp_cfg['max_false_flags']} of the {nc} clean scores and catches at least {cp_cfg['min_caught']} of the {n} "
        f"counting plants; the smallest size whose catch rate is at least {cfg['half']:.0%}.\n\n"
        "## Answer\n\n"
        f"- **Window:** robust? {yes(a['robust'])}. At {window} g/dL the flags agree with Stage 41 on {agree} of "
        f"{len(out['rows'])} scores.\n"
        f"- **Fewer people:** catches at least {cfg['half']:.0%} of the plants from "
        + (f"**{half} people**" if half else f"**more than {max(cfg['sizes'])} people**") + ".\n\n"
        "## Windows (each set at full size)\n\n| window | people within (India, Italy, Moscow) | clean scores "
        "flagged | plants caught | India gamma / saturation plants | passes? |\n|---|---|---|---|---|---|\n"
        + "\n".join(win) + "\n\n"
        f"## Fewer people (window {window} g/dL)\n\n| people | sets | counting plants | catch rate | the same plants "
        "at full size | false-flag rate | catch rate by set |\n|---|---|---|---|---|---|---|\n" + "\n".join(size) + "\n\n"
        "## Every score (step in score SDs; bold = flags)\n\n| dataset | plant | score | gain | "
        + " | ".join(f"step at {w} g/dL" for w in ws) + " | "
        + " | ".join(f"flagged at {s} people" for s in cfg["sizes"]) + " |\n|"
        + "---|" * (4 + len(ws) + len(cfg["sizes"])) + "\n" + "\n".join(rows) + "\n")
    (ROOT / "reports" / "stage42_step_power.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
