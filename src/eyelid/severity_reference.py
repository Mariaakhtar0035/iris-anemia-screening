"""Stage 39: does any Ghana body site's score read how sick a child is, beyond the anemic / healthy folder?

Stage 38's severity curve on the Ghana eyelids, nails and palms, with two reference lines for every score, each matched
to its own AUROC at Hb < 11.0 g/dL: a score that knows only the folder (it falls at lower cutoffs) and a score that
follows Hb (it rises where fewer children sit near the cutoff). Settings and the answer, fixed before running, in
configs/severity_reference.yaml.

Run from the repo root: python -m src.eyelid.severity_reference
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames, population_inputs
from src.eyelid.handling_check import auroc_rows, folds_for, steady_reference
from src.eyelid.report import fmt
from src.eyelid.train_color import _short_commit
from src.ghana_link import children, load_sites

NAMES = {"ghana_eyelid": "Eyelids (388)", "linked_eyelid": "Eyelids, linked (302)", "linked_nail": "Nails, linked (302)",
         "linked_palm": "Palms, linked (302)"}
LABELS = {"dummy": "dummy", "rule": "redness rule (untrained)", "logistic_regression": "logistic regression",
          "tabpfn": "TabPFN"}


def folder_line(hb, s, t, cut):
    """AUROC at Hb < t of a score that knows only anemic vs healthy, with the same AUROC as s at Hb < cut."""
    a = auroc(hb < cut, s)
    h, m = (hb >= cut).sum(), ((hb >= t) & (hb < cut)).sum()
    return float((h * a + 0.5 * m) / (h + m))


def beyond_folder(t, cut):
    """Metric for bootstrap_ci: AUROC at Hb < t minus the folder-only line, both from the resample. ss: score, Hb."""
    return lambda yy, ss: auroc(ss[:, 1] < t, ss[:, 0]) - folder_line(ss[:, 1], ss[:, 0], t, cut)


def hb_line(hb, s, cut, cutoffs, n_sims, rng):
    """Draws of a score that follows Hb, matched to s's AUROC at Hb < cut: median and 2.5-97.5% AUROC at each cutoff."""
    z = hb - cut
    sd = steady_reference(z, auroc(hb < cut, s), 1.0, 1, rng)["noise_sd"]  # only its noise level is used
    draws = -z + sd * rng.standard_normal((n_sims, len(z)))
    out = {}
    for t in cutoffs:
        a = auroc_rows((hb < t).astype(int), draws)
        out[f"{t:g}"] = {"median": float(np.median(a)), "lo": float(np.percentile(a, 2.5)),
                         "hi": float(np.percentile(a, 97.5))}
    return out


def sets(cfg, base, color_cfg, probe_cfg, ceiling, seed):
    """name -> (Hb, the 29 numbers, the rule's score, CV folds)."""
    cut, cv, out = base["anemia_cutoff_g_dl"]["child_6_59_months"], ceiling["cv"], {}
    if "ghana_eyelid" in cfg["datasets"]:
        df, x = population_inputs(population_frames(ceiling), color_cfg, probe_cfg, ceiling)["cp_anemic"]
        hb = df.hb.to_numpy(float)
        assert ((hb < cut) == (df.anemic.to_numpy() == 1)).all()
        out["ghana_eyelid"] = (hb, x["color"], x["rule"],
                               repeated_stratified((hb < cut).astype(int), cv["n_splits"], cv["n_repeats"], seed))
    tabs, _, feats = load_sites(load_config("ghana_link"), base, color_cfg["roi"]["alpha_threshold"])
    eye, ids, groups = children(tabs)
    hb = eye.loc[ids, "hb"].to_numpy(float)
    folds = folds_for((hb < cut).astype(int), groups, cv, seed)
    for site in ("eyelid", "nail", "palm"):
        if f"linked_{site}" in cfg["datasets"]:
            X = feats[site].loc[ids]
            out[f"linked_{site}"] = (hb, X.to_numpy(float), -X[color_cfg["rule_feature"]].to_numpy(float), folds)
    return {k: out[k] for k in cfg["datasets"]}


def figure(out, cutoffs, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharey=True)
    x = np.arange(len(cutoffs))
    for ax, (name, d) in zip(axes.flat, out["datasets"].items()):
        r = d["scores"]["tabpfn"]
        hbl = [r["follows_hb"][f"{t:g}"] for t in cutoffs]
        ax.fill_between(x, [v["lo"] for v in hbl], [v["hi"] for v in hbl], color="#8a8a85", alpha=0.15, lw=0)
        ax.plot(x, [v["median"] for v in hbl], color="#55554f", lw=1.5, ls=":", label="follows Hb (same AUROC at 11)")
        ax.plot(x, [r["folder_line"][f"{t:g}"] for t in cutoffs], color="#55554f", lw=1.5, ls="--",
                label="knows only the folder (same AUROC at 11)")
        est = [r["severity"][f"{t:g}"]["auroc"]["estimate"] for t in cutoffs]
        ax.plot(x, est, color="#1baf7a", lw=2, marker="o", ms=8, mec="white", mew=2, label="TabPFN")
        ax.set_title(NAMES[name], fontsize=10, loc="left")
        ax.set_xticks(x, [f"< {t:g}\n({d['counts'][f'{t:g}']})" for t in cutoffs], fontsize=8)
        ax.set_ylim(0.4, 1.0)
        ax.grid(axis="y", color="#e6e6e1", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Ghana: does TabPFN's score read how sick a child is? (Hb cutoff, children below it)", fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    base, cfg, ceiling = load_config("default"), load_config("severity_reference"), load_config("ceiling")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    cut, t8, cutoffs = base["anemia_cutoff_g_dl"]["child_6_59_months"], cfg["answer_cutoff"], cfg["severity_cutoffs"]
    assert cutoffs[0] == cut
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling, seed)
    out = {"datasets": {}}
    for name, (hb, X, rule, folds) in sets(cfg, base, color_cfg, probe_cfg, ceiling, seed).items():
        y = (hb < cut).astype(int)
        held = {"rule": rule[None, :]}
        for m in cfg["models"]:
            held[m] = fit_and_score(ladder[m][1], X, y, folds)[0]
            print(f"done: {name} {m} {auroc(y, held[m].mean(0)):.3f}", flush=True)
        d = {"n": len(y), "anemic": int(y.sum()), "counts": {f"{t:g}": int((hb < t).sum()) for t in cutoffs},
             "scores": {}}
        for m, h in held.items():
            s = h.mean(0)
            d["scores"][m] = {
                "severity": {f"{t:g}": summarize(hb < t, h, n_boot, ci, seed) for t in cutoffs},
                "folder_line": {f"{t:g}": folder_line(hb, s, t, cut) for t in cutoffs},
                "follows_hb": hb_line(hb, s, cut, cutoffs, cfg["n_sims"], rng),
                "beyond_folder": bootstrap_ci(beyond_folder(t8, cut), y, np.column_stack([s, hb]), n_boot, ci,
                                              seed=seed)}
        out["datasets"][name] = d
    path = save_results("stage39_severity_reference", out, {"default": base, "severity_reference": cfg,
                                                            "ceiling": ceiling, "color": color_cfg})
    figure(out, cutoffs, ROOT / "reports" / "figures" / "stage39_severity_reference.png")
    write_report(path, cfg, out, cut)


def write_report(path, cfg, out, cut):
    cutoffs, t8 = cfg["severity_cutoffs"], cfg["answer_cutoff"]
    yes = lambda c: "**yes**" if c["lo"] > 0 else "no"
    answer, rows, full = [], [], []
    for name, d in out["datasets"].items():
        models = sorted(d["scores"], key=list(LABELS).index)
        for m in models:
            r = d["scores"][m]
            cells = [f"{r['severity'][f'{t:g}']['auroc']['estimate']:.3f} / {r['folder_line'][f'{t:g}']:.3f} / "
                     f"{r['follows_hb'][f'{t:g}']['median']:.3f}" for t in cutoffs[1:]]
            rows.append(f"| {NAMES[name]} | {LABELS[m]} | {fmt(r['severity'][f'{cut:g}']['auroc'])} | "
                        + " | ".join(cells) + f" | {fmt(r['beyond_folder'])} | {yes(r['beyond_folder'])} |")
            full += [f"| {NAMES[name]} | {LABELS[m]} | Hb < {t:g} | {d['counts'][f'{t:g}']} | "
                     f"{fmt(r['severity'][f'{t:g}']['auroc'])} | {fmt(r['severity'][f'{t:g}']['pr_auc'])} | "
                     f"{fmt(r['severity'][f'{t:g}']['sens_at_90_spec'])} |" for t in cutoffs]
        passing = [LABELS[m] for m in models if d["scores"][m]["beyond_folder"]["lo"] > 0]
        answer.append(f"- **{NAMES[name]}** ({d['anemic']} anemic, {d['counts'][f'{t8:g}']} below {t8:g}): "
                      + (", ".join(passing) if passing else "no score"))
    report = (
        "# Stage 39: does any Ghana body site's score read how sick a child is, beyond the anemic / healthy folder?\n\n"
        f"Generated by `python -m src.eyelid.severity_reference` at commit `{_short_commit(path)}`; settings and the "
        "answer (fixed before running) in `configs/severity_reference.yaml`.\n\n"
        f"- **Cutoff:** Ghana children 6-59 months; every model is trained on **Hb < {cut:g} g/dL** and the same held-out "
        "score is checked against lower cutoffs (children below each vs everyone else). Ghana only.\n"
        "- **Sets:** the 388 eyelid photos with no copy (folds by photo, as Stage 38), and the 302 children with an eyelid, "
        "nail and palm photo linked by file number (Stage 24; folds by child group; the link is probable, not certain).\n"
        "- **Scores:** 5-fold CV x 5 repeats, held-out scores averaged over repeats; the untrained redness rule, the dummy, "
        "and logistic regression and TabPFN on the 29 color numbers. Bootstrap 95% CIs, 2000 resamples.\n"
        "- **Two reference lines per score,** matched to its own AUROC at Hb < 11: *knows only the folder* (every "
        "anemic-healthy pair ordered right with the same chance, two anemic children a coin flip; it falls at lower "
        "cutoffs) and *follows Hb* (minus Hb plus noise, the jump test's steady reference; median of 2000 draws).\n"
        f"- **Answer, fixed before running:** a score reads severity beyond the folder if its AUROC at Hb < {t8:g} minus "
        "the folder-only line has its CI above 0.\n\n"
        f"## Answer: which scores read severity beyond the folder?\n\n" + "\n".join(answer) + "\n\n"
        "![TabPFN against the two reference lines](figures/stage39_severity_reference.png)\n\n"
        "## Each score against its two reference lines\n\n"
        "Each cell: the score's AUROC / knows only the folder / follows Hb (median).\n\n"
        f"| set | score | Hb < {cut:g} | " + " | ".join(f"Hb < {t:g}" for t in cutoffs[1:])
        + f" | Hb < {t8:g} minus the folder line | beyond the folder? |\n|" + "---|" * (len(cutoffs) + 4) + "\n"
        + "\n".join(rows) + "\n\n"
        "## Every severity score\n\n| set | score | children | n below | AUROC | PR-AUC | sens @ 90% spec |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(full) + "\n")
    (ROOT / "reports" / "stage39_severity_reference.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
