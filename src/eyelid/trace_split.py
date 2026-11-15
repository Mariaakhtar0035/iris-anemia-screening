"""Stage 33: how much of each score is real? Split it into the part that follows Hb smoothly and a step at the cutoff.

Stage 31 could not build a model free of the Ghana folder trace. Here the trace is measured instead of avoided: a pallor
reader's average score changes smoothly with Hb, so a step in the score exactly at the anemia cutoff is the trace (a
regression discontinuity). Taking the step out of every anemic person's score leaves the part that follows Hb, and its
AUROC is the trace-free estimate. A: the split on every dataset. B: a planted folder trace on the clean datasets (the
anemic photos only get a mild edit, models retrained), to check the split removes a trace we know is there and how often
Stage 30's jump test flags it. Settings and the answers, fixed before running, in configs/trace_split.yaml.

Run from the repo root: python -m src.eyelid.trace_split
"""
import numpy as np
from scipy.stats import norm, rankdata

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features
from src.eyelid.edit_robustness import moscow_pixels
from src.eyelid.handling_check import NAMES, RULE, datasets, edit, edit_list, folds_for, held_out, jump_test
from src.eyelid.report import fmt
from src.eyelid.roi import roi_pixels
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.train_color import _short_commit

LABELS = {"dummy": "dummy", RULE: "rule (minus mean a*)", "logistic_regression": "logistic regression",
          "hist_gradient_boosting": "gradient boosting", "tabpfn": "TabPFN"}


# ---------- the split ----------

def normal_scores(s):
    """Ranks mapped to a standard normal (ties share a rank): same order and AUROC, one scale for every model."""
    return norm.ppf((rankdata(s) - 0.5) / len(s))


def step_at_cutoff(z, q, window):
    """Step in q at z = 0 (anemic side minus healthy side): q = a + b z + step * anemic, one slope for both sides, fit
    on everyone within `window` g/dL of the cutoff (one shared slope: only 14-40 people sit on one side in India and
    Italy, too few to extrapolate a line of their own to the cutoff)."""
    m = np.abs(z) < window
    design = np.column_stack([np.ones(m.sum()), z[m], (z[m] < 0).astype(float)])
    return float(np.linalg.lstsq(design, q[m], rcond=None)[0][2])


def mostly_tied(s):
    """Scores with fewer distinct values than half the people (the dummy): moving the anemic ones by any step reorders
    their ties by label, so they are not split."""
    return len(np.unique(s)) < len(s) / 2


def trace_free(z, q, window):
    """Every anemic person's score with the step taken out."""
    return q - step_at_cutoff(z, q, window) * (z < 0)


def tf_auroc(z, q, window):
    return auroc((z < 0).astype(int), trace_free(z, q, window))


def split(z, score, window, n_boot, ci, seed):
    """A for one score: step, AUROC, trace-free AUROC, their change and the real-signal share, people resampled."""
    q, y = normal_scores(score), lambda zz: (zz < 0).astype(int)
    boot = lambda f: bootstrap_ci(f, z, q, n_boot, ci, seed=seed)
    out = {"auroc": boot(lambda zz, qq: auroc(y(zz), qq))}
    if mostly_tied(score):
        return out
    out |= {"step": boot(lambda zz, qq: step_at_cutoff(zz, qq, window)),
            "trace_free_auroc": boot(lambda zz, qq: tf_auroc(zz, qq, window)),
            "change": boot(lambda zz, qq: tf_auroc(zz, qq, window) - auroc(y(zz), qq))}
    if out["auroc"]["lo"] > 0.5:
        out["share"] = boot(lambda zz, qq: (tf_auroc(zz, qq, window) - 0.5) / (auroc(y(zz), qq) - 0.5))
    return out


def recovery(z, planted, unplanted, window, n_boot, ci, seed):
    """B: plant gain (planted minus unplanted AUROC) and recovery (trace-free AUROC of the planted score minus the
    unplanted AUROC), both paired on the same resampled people."""
    packed = np.column_stack([normal_scores(unplanted), normal_scores(planted)])
    y = lambda zz: (zz < 0).astype(int)
    boot = lambda f: bootstrap_ci(f, z, packed, n_boot, ci, seed=seed)
    out = {"gain": boot(lambda zz, pp: auroc(y(zz), pp[:, 1]) - auroc(y(zz), pp[:, 0]))}
    if not mostly_tied(planted):
        out["recovery"] = boot(lambda zz, pp: tf_auroc(zz, pp[:, 1], window) - auroc(y(zz), pp[:, 0]))
    return out


# ---------- planted trace ----------

def plant_numbers(pixels, plants):
    """The 29 numbers of each photo as it is (n, 29) and under each plant edit {label: (n, 29)}."""
    nums = lambda px: list(color_features(px).values())
    base, out = [], {lab: [] for lab, _, _ in plants}
    for px in pixels:
        base.append(nums(px))
        for lab, kind, value in plants:
            out[lab].append(nums(edit(px, kind, value)))
    return np.array(base), {k: np.array(v) for k, v in out.items()}


def planted(X, X_edited, y):
    """X with the anemic rows swapped for their edited version: a trace shared by everyone in the anemic folder."""
    return np.where(y[:, None] == 1, X_edited, X)


def keeps_clean(change, cfg):
    """Two-sided: a step estimated below 0 by chance adds label information when taken out, so a rise fails too."""
    return abs(change["estimate"]) <= cfg["keep_drop"] and change["lo"] < 0 < change["hi"]


def answers(out, cfg, models):
    A, B = out["split"], out["planted"]
    kept = {m: {n: keeps_clean(A[n][m]["change"], cfg) for n in cfg["clean"] if n in A} for m in models}
    counting = {m: [r for r in B if r["model"] == m and "recovery" in r and r["gain"]["estimate"] >= cfg["plant_min_gain"]]
                for m in models}
    removed = {m: sum(abs(r["recovery"]["estimate"]) <= cfg["recover_tol"] for r in rs) for m, rs in counting.items()}
    trusted = {m: sum(kept[m].values()) >= cfg["min_clean_kept"] and len(counting[m]) > 0
               and removed[m] >= cfg["min_removed_share"] * len(counting[m]) for m in models}
    all_counting = [r for rs in counting.values() for r in rs]
    return {"clean_kept": kept, "plants_counting": {m: len(rs) for m, rs in counting.items()}, "plants_removed": removed,
            "trusted": trusted,
            "jump_test_flags_plants": [sum(bool(r["jump_test"].get("jumps")) for r in all_counting), len(all_counting)],
            "step_above_zero": {n: {m: A[n][m]["step"]["lo"] > 0 for m in models} for n in A},
            "trace_free_auroc": {n: {m: A[n][m]["trace_free_auroc"] for m in models if trusted[m]}
                                 for n in A if n.startswith("ghana_")}}


# ---------- run ----------

def main():
    base, cfg, color_cfg = load_config("default"), load_config("trace_split"), load_config("color")
    jump_cfg, probe_cfg, ceiling_cfg = load_config("handling_check"), load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, w, alpha = boot["n_resamples"], boot["ci"], cfg["window_g_dl"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "trace_split": cfg, "handling_check": jump_cfg, "color": color_cfg, "ceiling": ceiling_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    makers = {m: ladder[m][1] for m in cfg["models"]}
    models = [m for m in (*cfg["models"], RULE) if m != "dummy"]
    a_col = feature_names().index(color_cfg["rule_feature"])
    plants = edit_list({"edits": cfg["plants"]}, None)

    def scores_for(X, y, folds):
        return {m: s.mean(0) for m, s in held_out(makers, X, y, folds)["as is"].items()} | {RULE: -X[:, a_col]}

    out = {"split": {}, "planted": []}
    for name, d in datasets(cfg, base, color_cfg, seed).items():
        y, X, z = d["y"], d["X"], d["hb"] - d["cut"]
        folds = folds_for(y, d["groups"], cfg["cv"], seed)
        scores = scores_for(X, y, folds)
        out["split"][name] = {m: split(z, s, w, n_boot, ci, seed) for m, s in scores.items()}
        print(f"done A {name}: " + ", ".join(f"{m} {r['auroc']['estimate']:.3f}->"
                                             f"{r.get('trace_free_auroc', {'estimate': np.nan})['estimate']:.3f}"
                                             for m, r in out["split"][name].items()), flush=True)
        if name in cfg["clean"]:
            pixels = moscow_pixels() if d["paths"] is None else (roi_pixels(p, alpha) for p in d["paths"])
            X0, edited = plant_numbers(pixels, plants)
            assert np.allclose(X0, X, rtol=1e-4, atol=1e-6), f"{name}: pixels don't give the stored numbers"
            for lab, Xe in edited.items():
                for m, s in scores_for(planted(X, Xe, y), y, folds).items():
                    out["planted"].append({"dataset": name, "plant": lab, "model": m,
                                           "auroc": bootstrap_ci(auroc, y, s, n_boot, ci, seed=seed),
                                           "jump_test": jump_test(z, s, jump_cfg, rng, n_boot, ci, seed),
                                           **recovery(z, s, scores[m], w, n_boot, ci, seed)})
                print(f"done B {name} {lab}", flush=True)
        save_results("stage33_trace_split", out, configs)
    out["answers"] = answers(out, cfg, models)
    path = save_results("stage33_trace_split", out, configs)
    write_report(path, cfg, out, models)


def write_report(path, cfg, out, models):
    A, B, ans = out["split"], out["planted"], out["answers"]
    order = ["dummy", *models]
    f = lambda r, k: fmt(r[k]) if k in r else "-"
    a_rows = "\n".join(f"| {NAMES[n]} | {LABELS[m]} | " + " | ".join(f(d[m], k) for k in (
        "auroc", "step", "trace_free_auroc", "change", "share")) + " |" for n, d in A.items() for m in order)
    b_rows = "\n".join(
        f"| {NAMES[r['dataset']]} | {r['plant']} | {LABELS[r['model']]} | {fmt(A[r['dataset']][r['model']]['auroc'])} | "
        f"{fmt(r['auroc'])} | {fmt(r['gain'])} | {r['jump_test'].get('answer', '-')} | {f(r, 'recovery')} |" for r in B)
    kept = "\n".join(f"| {LABELS[m]} | " + " | ".join("yes" if v else "no" for v in ans["clean_kept"][m].values()) +
                     f" | {ans['plants_removed'][m]} of {ans['plants_counting'][m]} | **{'yes' if ans['trusted'][m] else 'no'}** |"
                     for m in models)
    ghana = "\n".join(f"| {NAMES[n]} | {LABELS[m]} | {fmt(A[n][m]['auroc'])} | {fmt(v)} | {f(A[n][m], 'share')} |"
                      for n, d in ans["trace_free_auroc"].items() for m, v in d.items()) or "| none | - | - | - | - |"
    flagged, total = ans["jump_test_flags_plants"]
    report = (
        "# Stage 33: how much of each score is real? The score split into a smooth Hb part and a step at the cutoff\n\n"
        f"Generated by `python -m src.eyelid.trace_split` at commit `{_short_commit(path)}`; settings and the answers "
        "(fixed before running) in `configs/trace_split.yaml`.\n\n"
        "- **Cutoffs, one per dataset:** Ghana eyelids, nails, palms: children 6-59 months, **Hb < 11.0 g/dL**. India "
        "and Italy eyelids (Eyes-Defy adults): **WHO by sex, women < 12.0, men < 13.0 g/dL**, pregnancy not recorded. "
        "Moscow nails (no sex column): **Hb < 12.0 g/dL for everyone**.\n"
        "- **Idea:** pallor changes smoothly with Hb, so a pallor reader's average score has no step exactly at the "
        "cutoff. The step (anemic side minus healthy side at the cutoff, from a straight line on each side within "
        f"{cfg['window_g_dl']} g/dL) is the folder trace. Taking it out of every anemic person's score leaves the part "
        "that follows Hb. Scores are ranks mapped to a standard normal, so the step is in score SDs.\n"
        "- 5-fold CV x 5 repeats by photo / patient / child; bootstrap 95% CIs, 2000 resamples (people resampled, the "
        "step refit each time). Real-signal share = (trace-free AUROC - 0.5) / (AUROC - 0.5).\n\n"
        "## A. The split on every dataset\n\n"
        "| dataset | model | AUROC | step at the cutoff (SDs) | trace-free AUROC | change | real-signal share |\n"
        "|---|---|---|---|---|---|---|\n" + a_rows + "\n\n"
        "## B. Planted folder trace on the clean datasets (anemic photos only edited, models retrained)\n\n"
        "| dataset | plant | model | AUROC unplanted | AUROC planted | gain | Stage 30 jump test | recovery (trace-free "
        "planted minus unplanted) |\n|---|---|---|---|---|---|---|---|\n" + b_rows + "\n\n"
        "## Answers (rules fixed before running)\n\n"
        f"Keeps the clean signal (|change| <= {cfg['keep_drop']}, CI includes 0) on "
        + " / ".join(NAMES[n] for n in cfg["clean"]) +
        f"; removes counting plants (gain >= {cfg['plant_min_gain']}; |recovery| <= {cfg['recover_tol']}); trusted if "
        f"kept on >= {cfg['min_clean_kept']} clean sets and removes >= {cfg['min_removed_share']:.0%} of counting plants.\n\n"
        "| model | " + " | ".join(f"kept: {NAMES[n]}" for n in cfg["clean"]) + " | plants removed | trusted |\n"
        "|---|" + "---|" * len(cfg["clean"]) + "---|---|\n" + kept + "\n\n"
        f"Stage 30's jump test flags **{flagged} of {total}** counting plants.\n\n"
        "**Ghana trace-free AUROC (trusted models only):**\n\n"
        "| dataset | model | AUROC | trace-free AUROC | real-signal share |\n|---|---|---|---|---|\n" + ghana + "\n")
    (ROOT / "reports" / "stage33_trace_split.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
