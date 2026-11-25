"""Stage 31: do village-level geospatial covariates (malaria, bed nets, rainfall, travel time, night lights) add to the
survey questions?

Role: main result, Stage 31 (Zimbabwe, Malawi, Zambia; cross-validation on the building villages only).

Settings in configs/covariates.yaml, written before running. Each woman gets her village's six covariate values from the
DHS Geospatial Covariates file (joined on DHSCLUST = v001; -9999 = missing). The same Stage 30 folds (village split,
5 x 5) score the Stage 30 models (read from their saved predictions) next to four new ones: covariates alone, base 6 +
covariates, all questions + covariates (gradient boosting), and the Hb ranking + covariates. Pre-registered changes in
AUROC with village-bootstrap CIs. The locked test is spent and is not touched.

Run from the repo root, after python -m src.survey.countries:
  python -m src.survey.covariates
"""
import numpy as np
import pandas as pd

from src.common.evaluate import paired_gain, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.report import short_commit
from src.survey.countries import CACHE, Country, oof
from src.survey.train_baselines import survey_metrics

PROBA = {"dummy", "place", "lr_base", "lr_base_place", "gbc_all", "cov_lr", "lr_base_cov", "gbc_all_cov"}
NAMES = {"dummy": "dummy", "place[v024]": "place share, region (no model)", "lr_base": "base 6, LR",
         "lr_base_place[v024]": "base 6 + region, LR", "gbc_all": "all questions, gradient boosting",
         "hb_all": "all questions, Hb ranking", "cov_lr": "6 covariates alone, LR",
         "lr_base_cov": "base 6 + covariates, LR", "gbc_all_cov": "all questions + covariates, gradient boosting",
         "hb_all_cov": "all questions + covariates, Hb ranking"}


def load_covariates(cfg):
    """One row per village (index = village number), the pre-registered columns only, missing code -> NaN."""
    g = pd.read_csv(ROOT / cfg["file"])
    key = g[cfg["join_column"]]
    assert (key % 1 == 0).all() and key.is_unique, "village numbers must be unique whole numbers"
    cov = g[cfg["covariates"]].replace(cfg["missing_code"], np.nan)
    cov.index = key.astype(int).to_numpy()
    return cov


def attach(feats, groups, cov):
    """Each woman's village covariates added as columns; every village must be in the covariate file."""
    missing = sorted(set(groups) - set(cov.index))
    assert not missing, f"villages without covariates: {missing[:10]}"
    vals = cov.loc[groups].reset_index(drop=True)
    return pd.concat([feats.reset_index(drop=True), vals], axis=1)


def run_country(cv, base_cfg, fs, cc):
    """All models, gains and single-covariate AUROCs for one country (cv = that country's settings)."""
    seed, boot, sm = base_cfg["seed"], base_cfg["bootstrap"], fs["screening"]["smoothing"]
    m_prob, m_rank = survey_metrics(load_config("survey")), survey_metrics(load_config("survey"), probabilities=False)
    C = Country(cv["country"], cc, fs, seed)
    cov = load_covariates(cv)
    C.feats = attach(C.feats, C.groups, cov)
    names, ep = cv["covariates"], cv.get("extra_place")
    models = [("dummy", None, "dummy"), ("place[v024]", "v024", "place"), ("lr_base", C.base, "lr"),
              ("lr_base_place[v024]", C.base + ["v024"], "lr"), ("gbc_all", C.questions, "gbc"),
              ("hb_all", C.questions, "hb"), ("cov_lr", names, "lr"), ("lr_base_cov", C.base + names, "lr"),
              ("gbc_all_cov", C.questions + names, "gbc"), ("hb_all_cov", C.questions + names, "hb")]
    if ep:
        models += [(f"place[{ep}]", ep, "place"), (f"lr_base_place[{ep}]", C.base + [ep], "lr")]
    comparisons = cv["comparisons"] + cv.get("extra_comparisons", [])
    info = {"country": cv["country"], "women": int(len(C.groups)), "villages": int(len(set(C.groups))),
            "women_with_missing_covariates": int(C.feats[names].isna().any(axis=1).sum()),
            "villages_with_missing_covariates": int(len(set(C.groups[C.feats[names].isna().any(axis=1).to_numpy()])))}
    print(info, flush=True)

    rows, gains, single = [], [], []
    for target, y in C.y.items():
        s = {mid: oof(C, cv["split"], mid, q, kind, target, sm) for mid, q, kind in models}
        for mid, _, _ in models:
            m = summarize(y, s[mid], boot["n_resamples"], boot["ci"], seed, C.groups,
                          m_prob if mid.split("[")[0] in PROBA else m_rank)
            rows.append({"target": target, "model": mid, **m})
            print(f"  {target} | {mid}: AUROC {m['auroc']['estimate']:.3f}", flush=True)
        for new, ref in comparisons:
            gains.append({"target": target, "model": new, "reference": ref,
                          "auroc": paired_gain(y, s[new].mean(0), s[ref].mean(0), C.groups, boot, seed)})
        for c in names:                                   # one covariate on its own, higher value = higher score
            x = C.feats[c].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            single.append({"target": target, "covariate": c,
                           "auroc": bootstrap_ci(auroc, y[ok], x[ok], boot["n_resamples"], boot["ci"], C.groups[ok],
                                                 seed)})
    summary = cov.describe().T[["min", "50%", "max"]].round(4).to_dict(orient="index")
    return {"info": info, "covariate_ranges": summary, "cv": rows, "gains": gains, "single": single, "cfg": cv}


def main():
    base_cfg, fs, cc, cv = (load_config("default"), load_config("feature_selection"), load_config("countries"),
                            load_config("covariates"))
    shared = {k: cv[k] for k in ("join_column", "missing_code", "split", "comparisons")}
    settings = [cv] + [{"country": k, **shared, **v} for k, v in cv["more_countries"].items()]
    out = {s["country"]: run_country(s, base_cfg, fs, cc) for s in settings}
    for r in out.values():
        r.pop("cfg")
    path = save_results("covariates", {"countries": out},
                        {"default": base_cfg, "feature_selection": fs, "countries": cc, "covariates": cv})
    write_report(path, out, settings)


FILES = {"zimbabwe": ("Zimbabwe 2015", "ZWGC72FL"), "malawi": ("Malawi 2024", "MWGC81FL"),
         "zambia": ("Zambia 2024", "ZMGC81FL")}


def write_report(path, out, settings):
    c = lambda d: f"{d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    g = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    name = lambda m: NAMES.get(m) or (f"place share, {m[6:-1]} (no model)" if m.startswith("place[")
                                      else f"base 6 + {m[14:-1]}, LR")
    lines = [
        "# Stage 31: do village covariates (malaria, bed nets, rainfall, travel time, night lights) add anything?",
        "",
        f"Generated by `python -m src.survey.covariates` at commit `{short_commit(path)}`; settings in "
        "`configs/covariates.yaml`, written before running (Zimbabwe first, Malawi and Zambia added later, same design).",
        "",
        "- **Data:** the DHS Geospatial Covariates file of each country, joined on village number. Values are averages "
        "over a 2 km (town) or 10 km (rural) circle around the village's displaced GPS point. Building women only (the "
        "Stage 29 locked tests are spent and not touched).",
        "- **Targets:** any anemia (adjusted Hb < 12.0 g/dL, or < 11.0 if pregnant) and moderate or severe "
        "(adjusted Hb < 10.0 g/dL for every woman).",
        "- **Evaluation:** village split, 5 x 5 cross-validation (the Stage 30 folds); the first six models of each "
        "country are Stage 30's saved predictions. Village-bootstrap 95% CIs, 2000 resamples. Blank covariates are left "
        "blank; the models handle them.",
        "",
        "## Summary: change in AUROC from adding the covariates (pre-registered)",
        "",
        "| country | any anemia: base 6 | any anemia: all questions | moderate or severe: base 6 | "
        "moderate or severe: all questions |", "|---|---|---|---|---|",
    ]
    for k, r in out.items():
        pick = lambda t, m: next(g(x["auroc"]) for x in r["gains"] if x["target"] == t and x["model"] == m)
        ts = list(dict.fromkeys(x["target"] for x in r["cv"]))
        lines.append(f"| {FILES[k][0]} | " + " | ".join(pick(t, m) for t in ts for m in ("lr_base_cov", "gbc_all_cov"))
                     + " |")
    lines.append("")
    for k, r in out.items():
        info, cv = r["info"], next(s for s in settings if s["country"] == k)
        lines += [f"# {FILES[k][0]} ({FILES[k][1]})", "",
                  f"{info['women']:,} building women in {info['villages']} villages; {info['women_with_missing_covariates']} "
                  f"women in {info['villages_with_missing_covariates']} village(s) have a blank covariate.", "",
                  "Covariates (chosen before running): " + ", ".join(f"`{n}`" for n in cv["covariates"]) + ".", "",
                  "| covariate | min | median | max |", "|---|---|---|---|",
                  *[f"| `{n}` | {v['min']:g} | {v['50%']:g} | {v['max']:g} |" for n, v in r["covariate_ranges"].items()],
                  ""]
        for target in dict.fromkeys(x["target"] for x in r["cv"]):
            lines += [f"## {FILES[k][0]}: {target}", "",
                      "| model | AUROC | PR-AUC | Sens @ 90% spec | recall, top 30% | Brier |", "|---|---|---|---|---|---|"]
            for x in (x for x in r["cv"] if x["target"] == target):
                b = c(x["brier"]) if "brier" in x else "-"
                lines.append(f"| {name(x['model'])} | {c(x['auroc'])} | {c(x['pr_auc'])} | "
                             f"{c(x['sens_at_90_spec'])} | {c(x['recall_top'])} | {b} |")
            lines += ["", "**Pre-registered changes in AUROC**", "", "| model | minus | change |", "|---|---|---|",
                      *[f"| {name(x['model'])} | {name(x['reference'])} | {g(x['auroc'])} |"
                        for x in r["gains"] if x["target"] == target],
                      "", "**Each covariate on its own** (AUROC with the raw value as the score; below 0.5 means higher "
                      "values go with less anemia)", "", "| covariate | AUROC |", "|---|---|",
                      *[f"| `{x['covariate']}` | {c(x['auroc'])} |" for x in r["single"] if x["target"] == target], ""]
    (ROOT / "reports" / "covariates.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
