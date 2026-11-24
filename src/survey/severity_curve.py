"""Severity curve: does the survey model pick out women better as their anemia gets more severe? (mentor request)

Role: supporting figure for README 3.4. Uses saved predictions only; no model is fit.

Settings in configs/countries.yaml (`severity_curve`), written before running. For each country, the Hb ranking on all
questions (Stage 30, village split, out-of-fold on the building villages, averaged over the 5 repeats) gives each woman a
predicted Hb; lower = higher risk. For each Hb cutoff (one cutoff for every woman, on adjusted Hb):
  - main: AUROC for women below the cutoff vs everyone else (mild cases included), the screening question;
  - second: women below the cutoff vs women not anemic at all (Hb >= control), dropping those in between. Easier.
Village-bootstrap 95% CIs (2000 resamples). A point needs at least `min_cases` cases.

Run from the repo root, after python -m src.survey.countries:
  python -m src.survey.severity_curve
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.report import short_commit
from src.survey.countries import CACHE, Country

FIGURE = ROOT / "reports" / "figures" / "severity_curve.png"
COLORS = {"zimbabwe": "#1f77b4", "malawi": "#ff7f0e", "zambia": "#2ca02c"}
MARKERS = {"zimbabwe": "o", "malawi": "s", "zambia": "^"}


def curve_points(hb, score, groups, cutoffs, control, min_cases, boot, seed):
    """One row per cutoff: case count, share, and AUROC with CI for both comparisons (None if too few cases)."""
    rows = []
    for c in cutoffs:
        case = hb < c
        row = {"cutoff": c, "cases": int(case.sum()), "share": float(case.mean()), "vs_rest": None, "vs_not_anemic": None}
        if case.sum() >= min_cases:
            row["vs_rest"] = bootstrap_ci(auroc, case.astype(int), score, boot["n_resamples"], boot["ci"], groups, seed)
            keep = case | (hb >= control)
            row["vs_not_anemic"] = bootstrap_ci(auroc, case[keep].astype(int), score[keep], boot["n_resamples"],
                                                boot["ci"], groups[keep], seed)
        rows.append(row)
    return rows


def plot(curves, sc, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    titles = {"vs_rest": "Below the cutoff vs everyone else\n(the screening question)",
              "vs_not_anemic": f"Below the cutoff vs not anemic (Hb ≥ {sc['control_g_dl']:g})\n(easier: mild cases left out)"}
    cutoffs = sc["cutoffs_g_dl"]
    x = np.arange(len(cutoffs))
    for ax, key in zip(axes, titles):
        for name, rows in curves.items():
            pts = [(i, r[key]) for i, r in enumerate(rows) if r[key]]
            xi = np.array([i for i, _ in pts]) + {"zimbabwe": -0.08, "malawi": 0.0, "zambia": 0.08}[name]
            est = np.array([m["estimate"] for _, m in pts])
            lo, hi = np.array([m["lo"] for _, m in pts]), np.array([m["hi"] for _, m in pts])
            ax.errorbar(xi, est, yerr=[est - lo, hi - est], color=COLORS[name], marker=MARKERS[name], ms=6, lw=2,
                        capsize=0, label=name.title())
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        shares = [np.mean([rows[i]["share"] for rows in curves.values()]) for i in range(len(cutoffs))]
        ax.set_xticks(x, [f"< {c:g}\n({s:.0%})" if s >= 0.1 else f"< {c:g}\n({s:.1%})" for c, s in zip(cutoffs, shares)])
        ax.set_xlabel("Hb cutoff, g/dL (share of women below it)  →  more severe")
        ax.set_title(titles[key], fontsize=10, loc="left")
        ax.grid(axis="y", color="#e4e3df")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUROC (0.5 = coin flip), 95% CI")
    axes[0].set_ylim(0.45, 0.95)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Survey model (Hb ranking, all questions): cross-validated, building villages, village split",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    base, fs, cc = load_config("default"), load_config("feature_selection"), load_config("countries")
    sc, seed, boot = cc["severity_curve"], base["seed"], base["bootstrap"]
    curves = {}
    for name in cc["countries"]:
        C = Country(name, cc, fs, seed)
        pred = np.load(CACHE / f"{name}__{sc['split']}__{sc['model']}__both.npy").mean(0)   # predicted Hb
        assert len(pred) == len(C.hb), f"{name}: cached predictions don't match the building women"
        curves[name] = curve_points(C.hb, -pred, C.groups, sc["cutoffs_g_dl"], sc["control_g_dl"], sc["min_cases"],
                                    boot, seed)
        print(name, [(r["cutoff"], r["cases"], r["vs_rest"] and round(r["vs_rest"]["estimate"], 3)) for r in curves[name]],
              flush=True)
    plot(curves, sc, FIGURE)
    path = save_results("severity_curve", {"curves": curves}, {"default": base, "countries": cc})
    write_report(path, curves, sc)


def write_report(path, curves, sc):
    f = lambda m: "-" if m is None else f"{m['estimate']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"
    lines = [
        "# Severity curve: does the survey model pick out the sickest women better?",
        "",
        f"Generated by `python -m src.survey.severity_curve` at commit `{short_commit(path)}`; settings in "
        "`configs/countries.yaml` (`severity_curve`), written before running.",
        "",
        "- **Score:** each woman's predicted Hb from the Hb ranking on all questions (Stage 30, village split, "
        "out-of-fold on the building villages, averaged over 5 repeats). Lower predicted Hb = higher risk. No model was "
        "refit and no locked test was used.",
        "- **Cutoffs:** one cutoff for every woman, pregnant or not, on Hb adjusted for altitude and smoking. So the "
        "< 12 point differs slightly from the 'any anemia' results elsewhere, which use < 11 for pregnant women.",
        "- **Vs everyone else:** women below the cutoff against all other women, mild cases included. This is the "
        "screening question.",
        f"- **Vs not anemic:** women below the cutoff against women with Hb >= {sc['control_g_dl']:g} only. Easier, "
        "because the women in between are left out.",
        f"- Village-bootstrap 95% CIs, 2000 resamples. A point needs at least {sc['min_cases']} cases.",
        "",
        "![Severity curve](figures/severity_curve.png)",
        "",
    ]
    for name, rows in curves.items():
        lines += [f"**{name.title()}**", "",
                  "| Hb cutoff (g/dL) | cases | share of women | AUROC vs everyone else | AUROC vs not anemic |",
                  "|---|---|---|---|---|",
                  *[f"| < {r['cutoff']:g} | {r['cases']} | {r['share']:.1%} | {f(r['vs_rest'])} | {f(r['vs_not_anemic'])} |"
                    for r in rows], ""]
    (ROOT / "reports" / "severity_curve.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
