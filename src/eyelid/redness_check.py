"""Stage 35: TabPFN on Ghana with redness only (and larger input sets), through every trace check.

Each input set (mean a* alone, the redness numbers, all 29 minus the numbers that jump on their own, all 29) goes into
logistic regression and TabPFN on Ghana eyelids, nails and palms. Every score gets Stage 30's jump test, Stage 33's step
at the cutoff and Stage 32's seven test-photo edits. Settings and the answers, fixed before running, in
configs/redness_check.yaml.

Run from the repo root: python -m src.eyelid.redness_check
"""
import numpy as np

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import leave_one_group_out
from src.eyelid.ceiling import ladder_models
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.edit_robustness import fragile_edits
from src.eyelid.handling_check import NAMES, RULE, datasets, edit_list, folds_for, held_out, jump_test
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.trace_free import feature_jumps
from src.eyelid.trace_split import plant_numbers, split
from src.eyelid.train_color import _short_commit

CV, LOHO = "5x5 CV", "leave one hospital out"
LABELS = {"dummy": "dummy", RULE: "rule (minus mean a*)", "logistic_regression": "logistic regression",
          "tabpfn": "TabPFN", "hist_gradient_boosting": "gradient boosting", "mlp": "MLP",
          "forced_gradient_boosting": "gradient boosting, less red can only mean more anemic"}
STAGE35 = {"result": "stage35_redness_check", "module": "src.eyelid.redness_check", "config": "redness_check",
           "title": "Stage 35: TabPFN on Ghana with redness only, through every trace check",
           "models": "Logistic regression and TabPFN v2 on each input set"}


def input_sets(cfg, names, jumpers, extra=()):
    """{set: feature names} in the config's order. without_jumpers is skipped when nothing jumps (it equals all 29).
    A set may also use the extra (non-color) columns, or be {"all_plus": [extra columns]}: all 29 plus those."""
    out = {}
    for k, spec in cfg["input_sets"].items():
        if spec == "all":
            out[k] = list(names)
        elif spec == "all_but_jumpers":
            if jumpers:
                out[k] = [n for n in names if n not in jumpers]
        elif isinstance(spec, dict):
            assert set(spec["all_plus"]) <= set(extra), f"{k}: not extra columns: {sorted(set(spec['all_plus']) - set(extra))}"
            out[k] = list(names) + list(spec["all_plus"])
        else:
            known = set(names) | set(extra)
            assert set(spec) <= known, f"{k}: not color numbers or extra columns: {sorted(set(spec) - known)}"
            out[k] = list(spec)
    return out


def check(y, z, s, cfg, jump_cfg, rng, n_boot, ci, seed):
    """Scores, Stage 30's jump test and Stage 33's step for one score ((n_repeats, n) or (n,))."""
    m = np.atleast_2d(s).mean(0)
    return {"scores": summarize(y, s, n_boot, ci, seed),
            "jump": jump_test(z, m, jump_cfg, rng, n_boot, ci, seed),
            "split": split(z, m, cfg["window_g_dl"], n_boot, ci, seed)}


def shows_no_trace(r, p_max):
    """Fixed before running: jump p >= p_max, step CI includes 0, and no edit breaks it (models only)."""
    step = r["split"].get("step")
    if "p" not in r["jump"] or step is None:
        return False
    return r["jump"]["p"] >= p_max and step["lo"] < 0 < step["hi"] and not r.get("fragile")


def reported(sets, model):
    """The largest input set (last in the config's order first) that shows no trace and beats the rule, or None."""
    for k in reversed(list(sets)):
        r = sets[k][model]
        if r["no_trace"] and r["vs_rule"]["lo"] > 0:
            return k
    return None


def run(stage, makers_for):
    """One stage: stage = its names and report header, makers_for(ladder, columns) = {model: factory} per input set.
    Optional: stage["extra"](name, d) -> (column names, array) appended after the 29 numbers (photo edits leave them
    as they are); stage["write_report"] replaces the report; cfg set_pairs / model_pairs add paired AUROC changes
    (first minus second) between input sets per model and between models per input set."""
    base, cfg, color_cfg = load_config("default"), load_config(stage["config"]), load_config("color")
    jump_cfg, probe_cfg, ceiling_cfg = load_config("handling_check"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha, p_max = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"], jump_cfg["p_max"]
    configs = {"default": base, stage["config"]: cfg, "handling_check": jump_cfg, "color": color_cfg,
               "ceiling": ceiling_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    names = feature_names()
    a_col = names.index(color_cfg["rule_feature"])
    edits = [e for e in edit_list(jump_cfg, (1.0, 1.0, 1.0)) if e[1] != "gains"]  # Stage 32's seven
    run = lambda s: check(y, z, s, cfg, jump_cfg, rng, n_boot, ci, seed)
    out = {"edits": [e[0] for e in edits], "datasets": {}}

    for name, d in datasets(cfg, base, color_cfg, seed).items():
        y, z = d["y"], d["hb"] - d["cut"]
        X, fixed = plant_numbers((roi_pixels(p, alpha) for p in d["paths"]), edits)
        assert np.allclose(X, d["X"], rtol=1e-4, atol=1e-6), f"{name}: pixels don't give the stored numbers"
        jumpers = [r["feature"] for r in feature_jumps(z, X, names, jump_cfg, rng, n_boot, ci, seed) if r["jumps"]]
        extra, E = stage["extra"](name, d) if "extra" in stage else ([], np.empty((len(y), 0)))
        X, fixed = np.hstack([X, E]), {e: np.hstack([Xe, E]) for e, Xe in fixed.items()}
        columns = names + list(extra)
        sets = input_sets(cfg, names, jumpers, extra)
        splits = {CV: folds_for(y, d["groups"], cfg["cv"], seed)}
        if name == "ghana_eyelid":
            cp = load_cp_anemic(ceiling_cfg["cp_anemic_subset"])
            assert np.allclose(cp.hb.to_numpy(float), d["hb"]), "Ghana eyelid rows out of order"
            splits[LOHO] = leave_one_group_out(cp.site.to_numpy())
        rule = {v: -(X if v == "as is" else fixed[v])[:, a_col] for v in ["as is", *fixed]}
        rule_rows = [{"edit": e, "model": RULE, "change": auroc_change(y, rule[e], rule["as is"], n_boot, ci, seed)}
                     for e in fixed]
        res = {"n": len(y), "anemic": int(y.sum()), "cutoff": sorted(set(np.round(d["cut"], 2).tolist())),
               "jumpers": jumpers, "input_sets": sets, "rule": run(rule["as is"]), "splits": {}}
        res["rule"]["no_trace"] = shows_no_trace(res["rule"], p_max)
        for sp, folds in splits.items():
            dummy = held_out({"dummy": ladder["dummy"][1]}, X, y, folds)["as is"]["dummy"]
            r_sp = {"dummy": summarize(y, dummy, n_boot, ci, seed), "sets": {}}
            kept = {}
            for k, cols in sets.items():
                idx = [columns.index(c) for c in cols]
                variants = {e: Xe[:, idx] for e, Xe in fixed.items()} if sp == CV else None
                held = held_out(makers_for(ladder, cols), X[:, idx], y, folds, variants)
                kept[k] = held["as is"]
                r_sp["sets"][k] = {}
                for m in cfg["models"]:
                    r = run(held["as is"][m]) | {
                        "vs_rule": auroc_change(y, held["as is"][m], rule["as is"], n_boot, ci, seed)}
                    if sp == CV:
                        rows = rule_rows + [{"edit": e, "model": m, "change": auroc_change(
                            y, held[e][m], held["as is"][m], n_boot, ci, seed)} for e in fixed]
                        r["edits"] = {row["edit"]: row["change"] for row in rows if row["model"] == m}
                        r["fragile"] = fragile_edits(rows, m, cfg)
                    r["no_trace"] = shows_no_trace(r, p_max) if sp == CV else None
                    r_sp["sets"][k][m] = r
                show = lambda r: (f"{r['scores']['auroc']['estimate']:.3f} (jump p {r['jump'].get('p', float('nan')):.4f}"
                                  + (f", fragile {len(r['fragile'])})" if "fragile" in r else ")"))
                print(f"done {name} {sp} {k}: " + ", ".join(f"{m} {show(r_sp['sets'][k][m])}" for m in cfg["models"]),
                      flush=True)
            change = lambda a, b: auroc_change(y, a, b, n_boot, ci, seed)
            for k in sets:
                for m1, m2 in cfg.get("model_pairs", []):
                    r_sp["sets"][k][m1].setdefault("vs", {})[m2] = change(kept[k][m1], kept[k][m2])
            if cfg.get("set_pairs"):
                r_sp["set_pairs"] = {f"{a} - {b}": {m: change(kept[a][m], kept[b][m]) for m in cfg["models"]}
                                     for a, b in cfg["set_pairs"]}
            res["splits"][sp] = r_sp
        res["reported"] = {m: reported(res["splits"][CV]["sets"], m) for m in cfg["models"]}
        out["datasets"][name] = res
        save_results(stage["result"], out, configs)

    path = save_results(stage["result"], out, configs)
    stage.get("write_report", write_report)(path, cfg, jump_cfg, out, stage)


def main():
    cfg = load_config("redness_check")
    run(STAGE35, lambda ladder, cols: {m: ladder[m][1] for m in cfg["models"]})


def yn(v):
    return "-" if v is None else ("**yes**" if v else "no")


def every_score(out):
    """Markdown: every score with its checks, then the dummy."""
    D = out["datasets"]

    def near(j):
        return (f"{fmt(j['auroc_near'])} vs {j['reference']['auroc_near_median']:.3f} (p {j['p']:.4f})"
                if "p" in j else j.get("answer", "-"))

    def row(n, sp, k, m, r):
        step = fmt(r["split"]["step"]) if "step" in r["split"] else "-"
        edits = (f"{len(r['fragile'])} of {len(out['edits'])} "
                 f"(avg {np.mean([c['estimate'] for c in r['edits'].values()]):+.3f})" if "edits" in r else "-")
        return (f"| {NAMES[n]} | {sp} | {k} | {LABELS[m]} | {fmt(r['scores']['auroc'])} | {fmt(r['scores']['pr_auc'])} | "
                f"{fmt(r['scores']['sens_at_90_spec'])} | {fmt(r['vs_rule']) if 'vs_rule' in r else '-'} | "
                f"{near(r['jump'])} | {step} | {fmt(r['jump']['rho_anemic'])} | {fmt(r['jump']['rho_healthy'])} | "
                f"{edits} | {yn(r.get('no_trace'))} |")

    rows = []
    for n, d in D.items():
        rows.append(row(n, "-", "-", RULE, d["rule"]))
        for sp, s in d["splits"].items():
            rows += [row(n, sp, k, m, r) for k, ms in s["sets"].items() for m, r in ms.items()]
    dummy = "\n".join(f"| {NAMES[n]} | {sp} | {fmt(s['dummy']['auroc'])} | {fmt(s['dummy']['pr_auc'])} | "
                      f"{fmt(s['dummy']['sens_at_90_spec'])} |" for n, d in D.items() for sp, s in d["splits"].items())
    return (
        "## Every score\n\n"
        "Near-cutoff AUROC vs the steady reference's median (p); step in score SDs; rank correlation of the score with "
        "Hb minus the cutoff inside each group (negative = follows Hb); edits that break it (average AUROC change over "
        "the seven).\n\n"
        "| dataset | split | input set | model | AUROC | PR-AUC | sens @ 90% spec | change vs rule | near-cutoff AUROC | "
        "step | rho anemic | rho healthy | edits that break it | no trace? |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n\n"
        "## Dummy\n\n| dataset | split | AUROC | PR-AUC | sens @ 90% spec |\n|---|---|---|---|---|\n" + dummy + "\n")


def checks_text(cfg, jump_cfg, out):
    """Markdown bullets: how each score is checked, and what "no trace" means."""
    return (
        f"- **Checks:** Stage 30's jump test (within {jump_cfg['window_g_dl']} g/dL, {jump_cfg['n_sims']} steady draws, "
        f"jumps if p < {jump_cfg['p_max']}); Stage 33's step at the cutoff (within {cfg['window_g_dl']} g/dL, score "
        "SDs); Stage 32's seven test-photo edits (" + ", ".join(out["edits"]) + f"), an edit breaks a model if AUROC "
        f"drops >= {cfg['fragile_drop']} (CI below 0) while the rule moves < {cfg['rule_stable_drop']}.\n"
        "- **No trace we can find** = jump p >= 0.025, step CI includes 0, no edit breaks it (rule: the first two). "
        "Not proof: the jump test caught 10 of 24 planted traces in Stage 33.\n")


def write_report(path, cfg, jump_cfg, out, stage):
    D = out["datasets"]
    answers = "\n".join(
        f"| {NAMES[n]} | {LABELS[m]} | " + (f"{k} | {fmt(d['splits'][CV]['sets'][k][m]['scores']['auroc'])} | "
                                           f"{fmt(d['splits'][CV]['sets'][k][m]['vs_rule'])}" if k else
                                           "no input set passes on this dataset | - | -") + " |"
        for n, d in D.items() for m, k in d["reported"].items())
    passes = "\n".join(
        f"| {NAMES[n]} | {yn(d['rule']['no_trace'])} | " + " | ".join(
            "/".join(("yes" if d["splits"][CV]["sets"][k][m]["no_trace"] else "no") for m in cfg["models"])
            if k in d["splits"][CV]["sets"] else "(same as all_29)" for k in cfg["input_sets"]) + " |"
        for n, d in D.items())
    sets = "\n".join(f"| {k} | {', '.join(v) if isinstance(v, list) else v} |" for k, v in cfg["input_sets"].items())
    jumpers = "\n".join(f"| {NAMES[n]} | {', '.join(d['jumpers']) or 'none'} |" for n, d in D.items())
    report = (
        f"# {stage['title']}\n\n"
        f"Generated by `python -m {stage['module']}` at commit `{_short_commit(path)}`; settings and the answers "
        f"(fixed before running) in `configs/{stage['config']}.yaml`.\n\n"
        "- **Cutoff:** Ghana eyelids (CP-AnemiC, 388 photos), nails and palms (302 linked children): children 6-59 "
        "months, **Hb < 11.0 g/dL** on every set. Ghana only (India and Italy are kept for audits; not used here).\n"
        f"- **Inputs:** only color numbers (no hospital, file name, child ID or photo size). {stage['models']}"
        "; dummy and the untrained rule (minus mean a*) next to them. 5-fold CV x 5 repeats "
        "by photo / child, scores averaged over repeats; the eyelids also leave one hospital out. Bootstrap 95% CIs, "
        "2000 resamples.\n" + checks_text(cfg, jump_cfg, out) + "\n"
        "## The number we would report (largest input set with no trace that beats the rule)\n\n"
        "| dataset | model | input set | AUROC | change vs rule |\n|---|---|---|---|---|\n" + answers + "\n\n"
        "## No trace we can find? (5x5 CV; " + " / ".join(LABELS[m] for m in cfg["models"]) + ")\n\n"
        "| dataset | rule | " + " | ".join(cfg["input_sets"]) + " |\n|" + "---|" * (2 + len(cfg["input_sets"])) + "\n"
        + passes + "\n\n"
        "## Input sets\n\n| set | numbers |\n|---|---|\n" + sets + "\n\n"
        "Numbers that jump on their own (Bonferroni p < 0.025 / 29), left out of without_jumpers:\n\n"
        "| dataset | numbers |\n|---|---|\n" + jumpers + "\n\n" + every_score(out))
    (ROOT / "reports" / f"{stage['result']}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
