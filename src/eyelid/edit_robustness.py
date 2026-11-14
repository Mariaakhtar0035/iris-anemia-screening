"""Stage 32: is TabPFN fragile to small photo edits on every dataset, or only on Ghana? Do color-edited training copies
fix it, and do they take the Ghana folder trace away?

Stage 30 B edited the Ghana eyelid test photos only. Here the same seven edits go to every dataset with Hb, and a second
TabPFN is fit on the training photos plus randomly edited copies of them (the copies stay in their photo's fold).
Settings and the answers, fixed before running, in configs/edit_robustness.yaml.

Run from the repo root: python -m src.eyelid.edit_robustness
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features
from src.eyelid.handling_check import NAMES, RULE, datasets, edit, edit_list, folds_for, jump_test
from src.eyelid.hb_regression import box_pixels
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.train_color import _short_commit

COPIES = "tabpfn_edited_copies"
MODELS = ("dummy", RULE, "logistic_regression", "tabpfn", COPIES)
LABELS = {"dummy": "dummy", RULE: "rule", "logistic_regression": "logistic regression", "tabpfn": "TabPFN",
          COPIES: "TabPFN + edited copies"}


def moscow_pixels():
    """Each Moscow photo's three nail boxes, one photo at a time, in load_moscow's row order."""
    root = ROOT / load_config("nails")["data_dir"]
    d = pd.read_csv(root / "metadata.csv")
    for i, boxes in zip(d.PATIENT_ID, d.NAIL_BOUNDING_BOXES):
        rgb = np.asarray(Image.open(root / "photo" / f"{i}.jpg").convert("RGB"), dtype=np.float32) / 255.0
        yield box_pixels(rgb, json.loads(boxes))


def random_edit(rgb, ranges, rng):
    """One random white balance, brightness, gamma and saturation edit, in that order."""
    log_uniform = lambda k: float(np.exp(rng.uniform(*np.log(ranges[k]))))
    out = edit(rgb, "white_balance", (log_uniform("red_gain"), log_uniform("blue_gain")))
    out = edit(out, "brightness", log_uniform("brightness"))
    out = edit(out, "gamma", log_uniform("gamma"))
    return edit(out, "saturation", float(rng.uniform(*ranges["saturation"])))


def edited_numbers(pixels, edits, cfg, seed):
    """The 29 numbers of each photo as it is (n, 29), under each fixed edit {label: (n, 29)}, and on n_copies randomly
    edited copies (n_copies, n, 29). One photo's pixels in memory at a time."""
    nums = lambda px: list(color_features(px).values())
    base, fixed, copies = [], {lab: [] for lab, _, _ in edits}, []
    for i, px in enumerate(pixels):
        rng = np.random.default_rng([seed, i])
        base.append(nums(px))
        for lab, kind, value in edits:
            fixed[lab].append(nums(edit(px, kind, value)))
        copies.append([nums(random_edit(px, cfg["ranges"], rng)) for _ in range(cfg["n_copies"])])
    return np.array(base), {k: np.array(v) for k, v in fixed.items()}, np.array(copies).transpose(1, 0, 2)


def held_out_edits(makers, X, copies, y, folds, variants):
    """Held-out P(anemic) (n_repeats, n) per model on X and on each edited version of X (same rows). COPIES is fit on
    the training photos plus their edited copies, the others on the training photos as they are."""
    reps = max(r for r, _, _ in folds) + 1
    out = {v: {m: np.full((reps, len(y)), np.nan) for m in makers} for v in ["as is", *variants]}
    for r, tr, te in folds:
        test = np.vstack([X[te], *(Xv[te] for Xv in variants.values())])
        for m, make in makers.items():
            Xtr, ytr = ((np.vstack([X[tr], *copies[:, tr]]), np.tile(y[tr], 1 + len(copies))) if m == COPIES
                        else (X[tr], y[tr]))
            p = make().fit(Xtr, ytr).predict_proba(test)[:, 1]
            for k, v in enumerate(out):
                out[v][m][r, te] = p[k * len(te):(k + 1) * len(te)]
    assert all(not np.isnan(s).any() for v in out.values() for s in v.values())
    return out


def fragile_edits(rows, model, cfg):
    """Edits where `model` drops by at least fragile_drop (CI below 0) while the rule barely moves (Stage 30)."""
    by = {(r["edit"], r["model"]): r["change"] for r in rows}
    return [e for e, m in by if m == model and by[(e, m)]["hi"] < 0 and by[(e, m)]["estimate"] <= -cfg["fragile_drop"]
            and by[(e, RULE)]["estimate"] > -cfg["rule_stable_drop"]]


def main():
    base, cfg, color_cfg = load_config("default"), load_config("edit_robustness"), load_config("color")
    jump_cfg, probe_cfg, ceiling_cfg = load_config("handling_check"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "edit_robustness": cfg, "handling_check": jump_cfg, "color": color_cfg,
               "ceiling": ceiling_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    tab = ceiling_cfg["tabpfn"]
    makers = {m: ladder[m][1] for m in ("dummy", "logistic_regression", "tabpfn")} | {
        COPIES: lambda: TabPFNClassifier.create_default_for_version(
            ModelVersion.V2, model_path=ROOT / tab["model_path"], device=tab["device"], random_state=seed,
            **cfg["copies_tabpfn"])}
    a_col = feature_names().index(color_cfg["rule_feature"])
    edits = [e for e in edit_list(jump_cfg, (1.0, 1.0, 1.0)) if e[1] != "gains"]  # batch61 is Ghana-only
    data = datasets(cfg, base, color_cfg, seed)
    out = {"edits": [e[0] for e in edits], "datasets": {}}

    for name, d in data.items():
        y, z = d["y"], d["hb"] - d["cut"]
        px = moscow_pixels() if d["paths"] is None else (roi_pixels(p, alpha) for p in d["paths"])
        X, fixed, copies = edited_numbers(px, edits, cfg, seed)
        assert np.allclose(X, d["X"], rtol=1e-4, atol=1e-6), f"{name}: recomputed numbers differ from the stored ones"
        held = held_out_edits(makers, X, copies, y, folds_for(y, d["groups"], cfg["cv"], seed), fixed)
        s = {v: held[v] | {RULE: -(X if v == "as is" else fixed[v])[:, a_col]} for v in held}
        res = {"n": len(y), "anemic": int(y.sum()), "cutoff": sorted(set(np.round(d["cut"], 2).tolist())),
               "scores": {m: summarize(y, s["as is"][m], n_boot, ci, seed) for m in MODELS},
               "copies_vs_tabpfn": auroc_change(y, s["as is"][COPIES], s["as is"]["tabpfn"], n_boot, ci, seed),
               "jump": {m: jump_test(z, np.atleast_2d(s["as is"][m]).mean(0), jump_cfg, rng, n_boot, ci, seed)
                        for m in ("tabpfn", COPIES)},
               "edits": [{"edit": lab, "model": m,
                          "auroc": bootstrap_ci(auroc, y, np.atleast_2d(s[lab][m]).mean(0), n_boot, ci, seed=seed),
                          "change": auroc_change(y, s[lab][m], s["as is"][m], n_boot, ci, seed)}
                         for lab in fixed for m in MODELS if m != "dummy"]}
        res["fragile"] = {m: fragile_edits(res["edits"], m, cfg) for m in ("logistic_regression", "tabpfn", COPIES)}
        res["answers"] = {
            "tabpfn_fragile": len(res["fragile"]["tabpfn"]) >= cfg["min_fragile"],
            "copies_robust": not res["fragile"][COPIES],
            "copies_keep_accuracy": res["copies_vs_tabpfn"]["estimate"] > -cfg["keep_drop"],
            "copies_remove_jump": (res["jump"][COPIES].get("p", -1.0) >= jump_cfg["p_max"]) if name.startswith("ghana_")
            else None}
        out["datasets"][name] = res
        print(f"done {name}: TabPFN fragile on {len(res['fragile']['tabpfn'])}, copies on "
              f"{len(res['fragile'][COPIES])}; {res['answers']}", flush=True)
        save_results("stage32_edit_robustness", out, configs)

    path = save_results("stage32_edit_robustness", out, configs)
    write_report(path, cfg, jump_cfg, out)


def write_report(path, cfg, jump_cfg, out):
    D = out["datasets"]
    yn = lambda v: "-" if v is None else ("**yes**" if v else "no")
    scores = "\n".join(f"| {NAMES[n]} | {d['n']} | {d['anemic']} | {d['cutoff']} | {LABELS[m]} | " +
                       " | ".join(fmt(d["scores"][m][k]) for k in ("auroc", "pr_auc", "sens_at_90_spec")) + " |"
                       for n, d in D.items() for m in MODELS)
    by = lambda d: {(r["edit"], r["model"]): r for r in d["edits"]}
    edits = "\n".join(f"| {NAMES[n]} | {e} | " + " | ".join(
        f"{by(d)[(e, m)]['auroc']['estimate']:.3f} ({fmt(by(d)[(e, m)]['change'])})" for m in MODELS if m != "dummy") + " |"
        for n, d in D.items() for e in out["edits"])
    jp = lambda j: f"{j['p']:.4f}" if "p" in j else j["answer"]
    summary = "\n".join(
        f"| {NAMES[n]} | {len(d['fragile']['tabpfn'])} of {len(out['edits'])} | {yn(d['answers']['tabpfn_fragile'])} | "
        f"{len(d['fragile']['logistic_regression'])} | {len(d['fragile'][COPIES])} | {yn(d['answers']['copies_robust'])} | "
        f"{fmt(d['copies_vs_tabpfn'])} | {yn(d['answers']['copies_keep_accuracy'])} | {jp(d['jump']['tabpfn'])} | "
        f"{jp(d['jump'][COPIES])} | {yn(d['answers']['copies_remove_jump'])} |" for n, d in D.items())
    r = cfg["ranges"]
    report = (
        "# Stage 32: is TabPFN fragile to small photo edits on every dataset, and do edited training copies fix it?\n\n"
        f"Generated by `python -m src.eyelid.edit_robustness` at commit `{_short_commit(path)}`; settings and the "
        "answers (fixed before running) in `configs/edit_robustness.yaml`.\n\n"
        "- **Cutoffs, one per dataset:** Ghana eyelids, nails, palms: children 6-59 months, **Hb < 11.0 g/dL**. India "
        "and Italy eyelids (Eyes-Defy adults): **WHO by sex, women < 12.0, men < 13.0 g/dL**, pregnancy not recorded. "
        "Moscow nails (adults, no sex column): **Hb < 12.0 g/dL** for everyone.\n"
        "- **Input:** the same 29 color numbers, recomputed on edited pixels. 5-fold CV x 5 repeats by photo / patient "
        "/ child, scores averaged over repeats. Bootstrap 95% CIs, 2000 resamples.\n"
        "- **Test edits** (Stage 30's, test photos only, models trained on photos as they are): "
        + ", ".join(out["edits"]) + ".\n"
        f"- **Edited training copies:** each training photo plus {cfg['n_copies']} copies, each with a random white "
        f"balance (R x{r['red_gain'][0]}-{r['red_gain'][1]}, B x{r['blue_gain'][0]}-{r['blue_gain'][1]}), brightness "
        f"(x{r['brightness'][0]}-{r['brightness'][1]}), gamma ({r['gamma'][0]}-{r['gamma'][1]}) and saturation "
        f"({r['saturation'][0]}-{r['saturation'][1]}) edit. The test edits lie inside these ranges (the in-range "
        "case).\n"
        f"- **Fragile edit:** the model's AUROC drops by at least {cfg['fragile_drop']} (paired CI below 0) while the "
        f"rule moves less than {cfg['rule_stable_drop']}. TabPFN **is fragile on a dataset** if at least "
        f"{cfg['min_fragile']} of the {len(out['edits'])} edits are fragile.\n\n"
        "## Answers\n\n"
        "| dataset | TabPFN fragile edits | TabPFN fragile? | logistic regression fragile edits | copies fragile "
        "edits | copies robust? | copies minus TabPFN (unedited) | keeps accuracy? | TabPFN jump p | copies jump p | "
        "copies remove the jump? |\n|---|---|---|---|---|---|---|---|---|---|---|\n" + summary + "\n\n"
        f"Jump p: Stage 30's jump test (within {jump_cfg['window_g_dl']} g/dL, {jump_cfg['n_sims']} draws), jumps if "
        f"p < {jump_cfg['p_max']}; \"remove the jump\" is asked on the Ghana sets only.\n\n"
        "## The models on unedited photos\n\n"
        "| dataset | n | anemic | cutoff (g/dL) | model | AUROC | PR-AUC | sens @ 90% spec |\n"
        "|---|---|---|---|---|---|---|---|\n" + scores + "\n\n"
        "## Each edit: AUROC on edited test photos (change vs unedited, paired)\n\n"
        "| dataset | edit | " + " | ".join(LABELS[m] for m in MODELS if m != "dummy") + " |\n|---|---|---|---|---|---|\n"
        + edits + "\n")
    (ROOT / "reports" / "stage32_edit_robustness.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
