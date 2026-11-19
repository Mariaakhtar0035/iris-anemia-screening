"""Stage 38: numbers for the group talk, from the mentors' advice to the whole group on 2026-11-17.

On the Ghana eyelids (models trained on Hb < 11.0 g/dL):
4. severity on the x axis, AUROC on the y axis: how well the same score picks out children below 11, 10, 9, 8, 7 g/dL;
5. 100 children come in and only 20 (or 30) can get a blood test: how many of the anemic ones does each score catch?
Settings and the answers, fixed before running, in configs/presentation_numbers.yaml.

Run from the repo root: python -m src.eyelid.presentation_numbers
"""
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci, recall_at_top_fraction
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames, population_inputs
from src.eyelid.report import fmt
from src.eyelid.train_color import _short_commit

LABELS = {"dummy": "dummy", "rule": "redness rule (untrained)", "logistic_regression": "logistic regression",
          "tabpfn": "TabPFN"}
COLORS = {"rule": "#2a78d6", "logistic_regression": "#eb6834", "tabpfn": "#1baf7a"}  # checked with the dataviz validator


def severity_change(low, high):
    """Metric for bootstrap_ci: one score's AUROC for Hb < low minus its AUROC for Hb < high. ss columns: score, Hb."""
    return lambda yy, ss: auroc(ss[:, 1] < low, ss[:, 0]) - auroc(ss[:, 1] < high, ss[:, 0])


def catch_change(frac):
    """Metric for bootstrap_ci: catch rate in the top `frac` minus the rule's. ss columns: rule, model."""
    return lambda yy, ss: recall_at_top_fraction(yy, ss[:, 1], frac) - recall_at_top_fraction(yy, ss[:, 0], frac)


def figure(severity, cutoffs, counts, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(cutoffs))
    ax.axhline(0.5, color="#8a8a85", lw=1, ls="--")
    ax.text(x[-1] + 0.1, 0.505, "no signal", va="bottom", fontsize=9, color="#55554f")
    for m, color in COLORS.items():
        r = [severity[m][f"{t:g}"]["auroc"] for t in cutoffs]
        est = [v["estimate"] for v in r]
        ax.fill_between(x, [v["lo"] for v in r], [v["hi"] for v in r], color=color, alpha=0.12, lw=0)
        ax.plot(x, est, color=color, lw=2, marker="o", ms=8, mec="white", mew=2, label=LABELS[m])
        ax.text(x[-1] + 0.1, est[-1], f"{LABELS[m]} {est[-1]:.2f}", va="center", fontsize=9, color="#2b2b28")
    ax.set_xticks(x, [f"Hb < {t:g}\n({n} children)" for t, n in zip(cutoffs, counts)], fontsize=9)
    ax.set_xlim(-0.3, x[-1] + 1.6)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUROC (score trained on Hb < 11)")
    ax.set_title("Ghana eyelids: does the score pick out the sicker children better?", fontsize=11, loc="left")
    ax.grid(axis="y", color="#e6e6e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    base, cfg, ceiling = load_config("default"), load_config("presentation_numbers"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    df, x = population_inputs(population_frames(ceiling), color_cfg, probe_cfg, ceiling)["cp_anemic"]
    y, hb = df.anemic.to_numpy(), df.hb.to_numpy()
    cut = base["anemia_cutoff_g_dl"]["child_6_59_months"]
    assert ((hb < cut) == (y == 1)).all() and cfg["severity_cutoffs"][0] == cut

    folds = repeated_stratified(y, ceiling["cv"]["n_splits"], ceiling["cv"]["n_repeats"], seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling, seed)
    held = {"rule": x["rule"][None, :]}
    for m in cfg["models"]:
        kind, make = ladder[m]
        held[m] = fit_and_score(make, x[kind], y, folds)[0]
        print(f"done: {m} {auroc(y, held[m].mean(0)):.3f}", flush=True)
    score = {m: h.mean(0) for m, h in held.items()}

    cutoffs, sick = cfg["severity_cutoffs"], cfg["sickest_cutoff"]
    counts = [int((hb < t).sum()) for t in cutoffs]
    severity = {m: {f"{t:g}": summarize(hb < t, held[m], n_boot, ci, seed) for t in cutoffs} for m in held}
    rise = {m: bootstrap_ci(severity_change(sick, cut), y, np.column_stack([s, hb]), n_boot, ci, seed=seed)
            for m, s in score.items()}
    catch = {m: {f"{f:g}": {"all anemic": bootstrap_ci(partial(recall_at_top_fraction, frac=f), y, s, n_boot, ci,
                                                       seed=seed),
                            f"Hb < {sick:g}": bootstrap_ci(partial(recall_at_top_fraction, frac=f), hb < sick, s,
                                                           n_boot, ci, seed=seed),
                            "minus rule": None if m == "rule" else bootstrap_ci(
                                catch_change(f), y, np.column_stack([score["rule"], s]), n_boot, ci, seed=seed)}
                 for f in cfg["capacities"]} for m, s in score.items()}

    out = {"n": len(y), "anemic": int(y.sum()), "counts": dict(zip([f"{t:g}" for t in cutoffs], counts)),
           "severity": severity, "sickest_minus_all": rise, "catch": catch}
    path = save_results("stage38_presentation_numbers", out,
                        {"default": base, "presentation_numbers": cfg, "ceiling": ceiling, "color": color_cfg})
    fig_path = ROOT / "reports" / "figures" / "stage38_severity_auroc.png"
    figure(severity, cutoffs, counts, fig_path)
    write_report(path, cfg, out, cut)


def write_report(path, cfg, out, cut):
    models = sorted(out["severity"], key=list(LABELS).index)
    cutoffs, sick, prev = cfg["severity_cutoffs"], cfg["sickest_cutoff"], out["anemic"] / out["n"]
    yes = lambda c: "**yes**" if c["lo"] > 0 else "no"
    sev = lambda m, t, k="auroc": fmt(out["severity"][m][f"{t:g}"][k])
    rows4 = [f"| {LABELS[m]} | " + " | ".join(sev(m, t) for t in cutoffs)
             + f" | {fmt(out['sickest_minus_all'][m])} | {yes(out['sickest_minus_all'][m])} |" for m in models]
    full4 = [f"| {LABELS[m]} | Hb < {t:g} | {out['counts'][f'{t:g}']} | {sev(m, t)} | {sev(m, t, 'pr_auc')} | "
             f"{sev(m, t, 'sens_at_90_spec')} |" for m in models for t in cutoffs]
    rows5 = []
    for f in cfg["capacities"]:
        rows5.append(f"| {f:.0%} tested | random pick | {f:.0%} | {f:.0%} | - | - |")
        for m in models:
            c = out["catch"][m][f"{f:g}"]
            rows5.append(f"| {f:.0%} tested | {LABELS[m]} | {fmt(c['all anemic'])} | {fmt(c[f'Hb < {sick:g}'])} | "
                         + (f"{fmt(c['minus rule'])} | {yes(c['minus rule'])} |" if c["minus rule"] else "- | - |"))
        rows5.append(f"| {f:.0%} tested | perfect score | {min(1, f / prev):.0%} | - | - | - |")
    up = [LABELS[m] for m in models if out["sickest_minus_all"][m]["lo"] > 0]
    worth = {f: [LABELS[m] for m in models if m != "rule" and out["catch"][m][f"{f:g}"]["minus rule"]["lo"] > 0]
             for f in cfg["capacities"]}
    report = (
        "# Stage 38: numbers for the group talk (Ghana eyelids): severity and a fixed number of blood tests\n\n"
        f"Generated by `python -m src.eyelid.presentation_numbers` at commit `{_short_commit(path)}`; settings and the "
        "answers (fixed before running) in `configs/presentation_numbers.yaml`.\n\n"
        f"- **Cutoff:** Ghana eyelids (CP-AnemiC, the {out['n']} photos with no copy, {out['anemic']} anemic), children "
        f"6-59 months. Every model is trained on **Hb < {cut:g} g/dL**; the same held-out score is then checked against "
        "lower cutoffs. Ghana only.\n"
        "- **Scores:** 5-fold CV x 5 repeats by photo (Stage 9's folds), held-out scores averaged over repeats; models "
        "on the 29 color numbers, the untrained redness rule, and the dummy. Bootstrap 95% CIs, 2000 resamples; "
        "changes are paired (same children, same resamples).\n"
        "- **Caveat:** every trained Ghana score carries the folder trace (Stages 30-37), so these numbers show what the "
        "scores do, not how much of it is anemia.\n\n"
        "## 4. Severity on the x axis, AUROC on the y axis\n\n"
        "Mentor (2026-11-17): \"On the X axis severity, on the Y axis AUROC ... maybe top 10% we get something "
        "surprisingly high.\" Each column: the children below that Hb against everyone else.\n\n"
        f"**Answer:** the score picks out the sickest children (Hb < {sick:g}, about the sickest 9%) better than all "
        f"anemic children for: {', '.join(up) or 'no score'}.\n\n"
        "![AUROC by Hb cutoff](figures/stage38_severity_auroc.png)\n\n"
        "| score | " + " | ".join(f"Hb < {t:g} ({out['counts'][f'{t:g}']})" for t in cutoffs)
        + f" | Hb < {sick:g} minus Hb < {cut:g} | better for the sickest? |\n|" + "---|" * (len(cutoffs) + 3) + "\n"
        + "\n".join(rows4) + "\n\n"
        "## 5. 100 children, blood tests for only 20 (or 30)\n\n"
        "Mentor: \"100 people come in. You only have capacity to test 20. If you randomly pick the 20 ...\" The children "
        f"with the highest scores get the test. {prev:.1%} of these children are anemic, so even a perfect score can "
        f"catch at most {min(1, cfg['capacities'][0] / prev):.0%} of them with 20 tests.\n\n"
        + "".join(f"**Answer, {f:.0%} tested:** catches more anemic children than the rule: "
                  f"{', '.join(worth[f]) or 'no model'}.\n\n" for f in cfg["capacities"])
        + f"| tests | score | anemic children caught | of those with Hb < {sick:g}, caught | minus the rule | "
        "better than the rule? |\n|---|---|---|---|---|---|\n" + "\n".join(rows5) + "\n\n"
        "## Every severity score\n\n| score | children | n below | AUROC | PR-AUC | sens @ 90% spec |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(full4) + "\n")
    (ROOT / "reports" / "stage38_presentation_numbers.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
