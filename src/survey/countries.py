"""Stage 30: random vs village split, and each country trained and tested on its own (mentor follow-up).

Settings in configs/countries.yaml, written before running. For each country (Zimbabwe 2015, Malawi 2024, Zambia 2024):
  - questions: that country's Stage 29-rule cleaning and repeat rule (src/survey/select_features.py and
    screen_features.py with the country name); v861 (self-reported HIV test result, 2024 surveys only) is left out of
    the main models and put back in one extra row;
  - targets: any anemia (DHS v457 1-3 = adjusted Hb < 12.0 g/dL, or < 11.0 if pregnant, in all three files) and
    moderate or severe = adjusted Hb < 10.0 g/dL for every woman (the same rule in all three countries);
  - fixed models (no feature selection) scored by 5 x 5 cross-validation on the building villages with two splits:
    by village (new villages) and at random over women (new women from villages already surveyed);
  - Malawi and Zambia: every model fit on all building villages and scored once on that country's locked 20% of
    villages. Zimbabwe's locked test is not touched (already used in Stage 29);
  - transfer: base 6 logistic regression with the Stage 7/8 coding, trained on Zimbabwe's building women, scored on
    each country's locked test, next to the same model trained at home.

Run from the repo root, after the per-country cleaning:
  python -m src.survey.select_features malawi && python -m src.survey.screen_features malawi   (same for zambia)
  python -m src.survey.countries
"""
import numpy as np
import pandas as pd

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.splits import repeated_stratified, repeated_stratified_grouped
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit
from src.survey.build_dataset import clean, read_raw
from src.survey.repeated_cv import group_rate, paired_gain
from src.survey.screen_features import TARGETS
from src.survey.select_features import paths
from src.survey.select_models import columns_for, make_model, prepare
from src.survey.train_baselines import SURVEY_COLUMNS, feature_frame, survey_metrics
from src.survey.train_baselines import make_model as stage7_model

CACHE = ROOT / "data" / "processed" / "stage30_oof"
PROBA = {"dummy", "place", "lr_base", "lr_base_place", "gbc_all", "gbc_all_extra"}


class Country:
    """One country's building and locked women, questions, targets and folds."""

    def __init__(self, name, cc, fs, seed):
        p = paths(name)
        data = pd.read_parquet(p["out"])
        uni = pd.read_csv(p["univariate"])
        self.name, self.seed, self.mc = name, seed, fs["models"]
        self.locked = data.locked_test.to_numpy()
        self.all_feats = data.drop(columns=TARGETS)
        self.all_y = {"any anemia": data.anemic.to_numpy(),
                      "moderate or severe": (data.hb < cc["severity_cutoff_g_dl"]).astype(int).to_numpy()}
        self.all_hb, self.all_cutoff = data.hb.to_numpy(), data.hb_cutoff.to_numpy()
        self.all_groups = data.cluster.to_numpy()
        kept = uni.code[uni.step4 == "keep"].tolist()
        self.extra = [c for c in cc["extra_with"] if c in kept]
        self.questions = [c for c in kept if c not in cc["excluded_from_main"]]
        self.base = self.mc["base_questions"]
        missing = [q for q in self.base if q not in self.all_feats]
        assert not missing, f"{name}: base questions missing after cleaning: {missing}"
        self.places = [v for v in cc["place"][name] if v in self.all_feats]
        b = ~self.locked
        self.feats = self.all_feats[b].reset_index(drop=True)
        self.y = {k: v[b] for k, v in self.all_y.items()}
        self.hb, self.cutoff, self.groups = self.all_hb[b], self.all_cutoff[b], self.all_groups[b]
        n, r = cc["cv_splits"], cc["cv_repeats"]
        self.folds = {"village": repeated_stratified_grouped(self.y["any anemia"], self.groups, n, r, seed),
                      "random": repeated_stratified(self.y["any anemia"], n, r, seed)}
        self.n_rep = r


def model_list(C):
    """(model id, questions or place variable, kind) for this country; ids include the place variable."""
    out = [("dummy", None, "dummy"), ("lr_base", C.base, "lr"), ("gbc_all", C.questions, "gbc"),
           ("hb_all", C.questions, "hb")]
    for v in C.places:
        out += [(f"place[{v}]", v, "place"), (f"lr_base_place[{v}]", C.base + [v], "lr")]
    if C.extra:
        out += [("gbc_all_extra", C.questions + C.extra, "gbc"), ("hb_all_extra", C.questions + C.extra, "hb")]
    return out


def oof(C, split, model_id, q, kind, target, sm):
    """(n_repeats, n) out-of-fold scores on the building villages, cached per country, split, model and target."""
    tag = "both" if kind == "hb" else ("any" if target == "any anemia" else "sev")
    path = CACHE / f"{C.name}__{split}__{model_id}__{tag}.npy"
    if path.exists():
        raw = np.load(path)
    else:
        y = C.y[target]
        raw = np.full((C.n_rep, len(y)), np.nan)
        X = prepare(C.feats[columns_for(q, C.feats)]) if kind in ("lr", "gbc", "hb") else None
        codes = pd.factorize(C.feats[q].astype(str))[0] if kind == "place" else None
        for r, tr, te in C.folds[split]:
            if kind == "dummy":
                raw[r, te] = y[tr].mean()
            elif kind == "place":
                raw[r, te] = group_rate(codes, y, tr, te, sm)
            elif kind == "lr":
                raw[r, te] = make_model("logistic_regression", X, C.mc, C.seed).fit(
                    X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            elif kind == "gbc":
                raw[r, te] = make_model("hist_gradient_boosting", X, C.mc, C.seed).fit(
                    X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
            elif kind == "hb":
                raw[r, te] = make_model("hgb_regressor", X, C.mc, C.seed).fit(X.iloc[tr], C.hb[tr]).predict(X.iloc[te])
        np.save(path, raw)
    if kind == "hb":                                   # predicted Hb -> score: her cutoff minus the prediction
        cut = C.cutoff if target == "any anemia" else load_config("countries")["severity_cutoff_g_dl"]
        return cut - raw
    return raw


def locked_scores(C, models, target, sm):
    """Every model fit on all building women of the country, scored once on its locked villages."""
    tr, te = np.flatnonzero(~C.locked), np.flatnonzero(C.locked)
    y = C.all_y[target]
    out = {}
    for model_id, q, kind in models:
        if kind == "dummy":
            out[model_id] = np.full(len(te), y[tr].mean())
            continue
        if kind == "place":
            out[model_id] = group_rate(pd.factorize(C.all_feats[q].astype(str))[0], y, tr, te, sm)
            continue
        X = prepare(C.all_feats[columns_for(q, C.all_feats)])
        if kind == "lr":
            out[model_id] = make_model("logistic_regression", X, C.mc, C.seed).fit(
                X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        elif kind == "gbc":
            out[model_id] = make_model("hist_gradient_boosting", X, C.mc, C.seed).fit(
                X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        elif kind == "hb":
            pred = make_model("hgb_regressor", X, C.mc, C.seed).fit(X.iloc[tr], C.all_hb[tr]).predict(X.iloc[te])
            cut = C.all_cutoff[te] if target == "any anemia" else load_config("countries")["severity_cutoff_g_dl"]
            out[model_id] = cut - pred
    return out


def stage7_table(country, survey):
    """Base 6 with the Stage 7/8 coding, the same women in the same order as the Stage 29/30 table."""
    raw, labels = read_raw(country, survey)
    df = clean(raw, labels, survey, country)
    saved = pd.read_parquet(paths(country)["out"], columns=["cluster", "locked_test", "hb"])
    assert (df.cluster.to_numpy() == saved.cluster.to_numpy()).all(), f"{country}: row order differs"
    return feature_frame(df, survey), df, saved.locked_test.to_numpy()


def transfer(countries, survey, cc, seed, boot):
    """Base 6 (Stage 7 coding) trained on Zimbabwe's building women vs. at home, on each country's locked test."""
    Xz, dz, lz = stage7_table("zimbabwe", survey)
    rows = []
    for c in countries:
        if c == "zimbabwe":
            continue
        Xc, dc, lc = stage7_table(c, survey)
        groups = dc.cluster.to_numpy()[lc]
        for target in ("any anemia", "moderate or severe"):
            lab = lambda d: (d.anemic if target == "any anemia" else (d.hb_adjusted < cc["severity_cutoff_g_dl"])
                             ).astype(int).to_numpy()
            yz, yc = lab(dz), lab(dc)
            fit = lambda X, y, mask: stage7_model("logistic_regression", survey, seed).fit(X[mask], y[mask])
            s_home = fit(Xc, yc, ~lc).predict_proba(Xc[lc])[:, 1]
            s_tr = fit(Xz, yz, ~lz).predict_proba(Xc[lc])[:, 1]
            m = survey_metrics(survey)
            for name, s in (("trained at home", s_home), ("trained on Zimbabwe", s_tr)):
                rows.append({"country": c, "target": target, "model": f"base 6, {name}",
                             **summarize(yc[lc], s, boot["n_resamples"], boot["ci"], seed, groups, m)})
            rows.append({"country": c, "target": target, "model": "home minus Zimbabwe-trained (paired AUROC)",
                         "auroc": paired_gain(yc[lc], s_home, s_tr, groups, boot, seed)})
    return rows


def main():
    base_cfg, fs, cc, survey = (load_config("default"), load_config("feature_selection"), load_config("countries"),
                                load_config("survey"))
    seed, boot, sm = base_cfg["seed"], base_cfg["bootstrap"], fs["screening"]["smoothing"]
    CACHE.mkdir(parents=True, exist_ok=True)
    m_prob, m_rank = survey_metrics(survey), survey_metrics(survey, probabilities=False)
    cv_rows, split_gaps, locked_rows, info = [], [], [], {}

    for name in cc["countries"]:
        C = Country(name, cc, fs, seed)
        models = model_list(C)
        info[name] = {"women_building": int(len(C.groups)), "villages_building": int(len(set(C.groups))),
                      "women_locked": int(C.locked.sum()), "villages_locked": int(len(set(C.all_groups[C.locked]))),
                      "questions": len(C.questions), "extra": C.extra, "places": C.places,
                      "share": {t: float(v.mean()) for t, v in C.all_y.items()},
                      "cases": {t: int(v.sum()) for t, v in C.all_y.items()}}
        print(f"{name}: {info[name]}", flush=True)
        for target in C.y:
            y = C.y[target]
            scores = {}
            for split in cc["splits"]:
                for model_id, q, kind in models:                 # fit everything first, then score
                    scores[(split, model_id)] = oof(C, split, model_id, q, kind, target, sm)
                for model_id, q, kind in models:
                    s = scores[(split, model_id)]
                    kind_key = model_id.split("[")[0]
                    m = summarize(y, s, boot["n_resamples"], boot["ci"], seed, C.groups,
                                  m_prob if kind_key in PROBA else m_rank)
                    gain = (paired_gain(y, s.mean(0), scores[(split, "lr_base")].mean(0), C.groups, boot, seed)
                            if model_id != "lr_base" else None)
                    cv_rows.append({"country": name, "target": target, "split": split, "model": model_id, **m,
                                    "minus_base6": gain})
                    print(f"  {target} | {split} | {model_id}: AUROC {m['auroc']['estimate']:.3f}", flush=True)
            for model_id, _, _ in models:
                split_gaps.append({"country": name, "target": target, "model": model_id,
                                   "auroc": paired_gain(y, scores[("random", model_id)].mean(0),
                                                        scores[("village", model_id)].mean(0), C.groups, boot, seed)})
            if name != "zimbabwe":
                ls = locked_scores(C, models, target, sm)
                ly, lg = C.all_y[target][C.locked], C.all_groups[C.locked]
                for model_id, s in ls.items():
                    kind_key = model_id.split("[")[0]
                    locked_rows.append({"country": name, "target": target, "model": model_id,
                                        **summarize(ly, s, boot["n_resamples"], boot["ci"], seed, lg,
                                                    m_prob if kind_key in PROBA else m_rank),
                                        "minus_base6": (paired_gain(ly, s, ls["lr_base"], lg, boot, seed)
                                                        if model_id != "lr_base" else None)})
                print(f"  {target}: locked test scored", flush=True)

    tr_rows = transfer(cc["countries"], survey, cc, seed, boot)
    path = save_results("stage30_countries", {"info": info, "cv": cv_rows, "random_minus_village": split_gaps,
                                              "locked": locked_rows, "transfer": tr_rows},
                        {"default": base_cfg, "feature_selection": fs, "countries": cc, "survey": survey})
    write_report(path, info, cv_rows, split_gaps, locked_rows, tr_rows, cc, boot)


NAMES = {"dummy": "dummy", "lr_base": "base 6, logistic regression", "gbc_all": "all questions, gradient boosting",
         "hb_all": "all questions, Hb ranking", "gbc_all_extra": "all + self-reported HIV result, gradient boosting",
         "hb_all_extra": "all + self-reported HIV result, Hb ranking"}


def label(model_id):
    if model_id.startswith("place["):
        return f"place share, {model_id[6:-1]} (no model)"
    if model_id.startswith("lr_base_place["):
        return f"base 6 + {model_id[14:-1]}, logistic regression"
    return NAMES[model_id]


def write_report(path, info, cv_rows, split_gaps, locked_rows, tr_rows, cc, boot):
    c = lambda d: f"{d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    g = lambda d: "-" if d is None else f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    lines = [
        "# Stage 30: random vs village split, and each country on its own",
        "",
        f"Generated by `python -m src.survey.countries` at commit `{_short_commit(path)}`; settings in "
        "`configs/countries.yaml`, written before running. Village-bootstrap 95% CIs, "
        f"{boot['n_resamples']} resamples; paired differences on the same women.",
        "",
        "- **Any anemia:** DHS `v457` 1-3 = Hb adjusted for altitude and smoking < 12.0 g/dL, or < 11.0 if pregnant "
        "(checked to match in all three files).",
        f"- **Moderate or severe:** adjusted Hb **< {cc['severity_cutoff_g_dl']} g/dL for every woman, in all three "
        "countries.** This equals Zimbabwe 2015's `v457` 1-2, but **not** Malawi/Zambia 2024's `v457` 1-2, which use "
        "WHO 2011 (< 11.0, or < 10.0 if pregnant).",
        "- **Questions:** each country's own cleaning with the Stage 29 rules. `v861` (self-reported result of her last "
        "HIV test; asked in 2024 only) is left out of the main models and added back in one extra row.",
        "- **Fixed models, no feature selection.** Splits: **village** = whole villages held out (new villages); "
        "**random** = women held out at random, so neighbours of a test woman can be in training (new women from "
        "villages already surveyed).",
        "",
        "## The data",
        "",
        "| country | building women / villages | locked women / villages | questions | any anemia | moderate or severe |",
        "|---|---|---|---|---|---|",
        *[f"| {n} | {i['women_building']} / {i['villages_building']} | {i['women_locked']} / {i['villages_locked']} "
          f"| {i['questions']} | {i['share']['any anemia']:.1%} ({i['cases']['any anemia']}) | "
          f"{i['share']['moderate or severe']:.1%} ({i['cases']['moderate or severe']}) |" for n, i in info.items()],
        "",
        "Zimbabwe's locked test is not used here (it was used in Stage 29). Malawi's and Zambia's were locked before "
        "anything else and are scored once below.",
        "",
    ]
    for target in ("any anemia", "moderate or severe"):
        lines += [f"## {target.capitalize()}: cross-validation on the building villages (5 x 5)", ""]
        for name in info:
            rows = [r for r in cv_rows if r["country"] == name and r["target"] == target]
            tab = []
            for model_id in dict.fromkeys(r["model"] for r in rows):
                v = next(r for r in rows if r["model"] == model_id and r["split"] == "village")
                rd = next(r for r in rows if r["model"] == model_id and r["split"] == "random")
                gap = next(s for s in split_gaps if s["country"] == name and s["target"] == target
                           and s["model"] == model_id)
                tab.append(f"| {label(model_id)} | {c(v['auroc'])} | {c(rd['auroc'])} | {g(gap['auroc'])} | "
                           f"{g(v['minus_base6'])} |")
            lines += [f"**{name.title()}**", "",
                      "| model | AUROC, village split | AUROC, random split | random minus village | "
                      "minus base 6 (village split) |", "|---|---|---|---|---|", *tab, ""]
        lr = [r for r in locked_rows if r["target"] == target]
        if lr:
            lines += [f"### {target.capitalize()}: locked test, scored once (Malawi, Zambia)", "",
                      markdown_table([{**r, "model": label(r["model"])} for r in lr], ["country", "model"],
                                     SURVEY_COLUMNS), "",
                      "| country | model | AUROC minus base 6 (paired) |", "|---|---|---|",
                      *[f"| {r['country']} | {label(r['model'])} | {g(r['minus_base6'])} |" for r in lr
                        if r["minus_base6"]], ""]
    lines += ["## Transfer: base 6 trained on Zimbabwe vs. at home, on each country's locked test", "",
              "Base 6 logistic regression with the Stage 7/8 coding, trained on Zimbabwe's building women or on the "
              "country's own building women; both scored on the country's locked villages.", "",
              markdown_table([r for r in tr_rows if "minus" not in r["model"]], ["country", "target", "model"],
                             SURVEY_COLUMNS), "",
              "| country | target | home minus Zimbabwe-trained AUROC (paired) |", "|---|---|---|",
              *[f"| {r['country']} | {r['target']} | {g(r['auroc'])} |" for r in tr_rows if "minus" in r["model"]],
              ""]
    (ROOT / "reports" / "stage30_countries.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
