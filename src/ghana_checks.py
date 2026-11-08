"""Stage 25: cheating and fairness checks on the Stage 24 children (one Ghanaian child's eyelid, nail and palm).

Stage 24 split by child, left hospitals out and compared real children with pretend ones. Four more checks on the same
children, folds and label (configs/ghana_checks.yaml):
1. Fairness inside Ghana: girls vs boys, younger vs older. Does adding nail and palm widen a gap?
2. No color: can a model tell anemia from the size and shape of the three photos alone?
3. Same child, same mistakes: do one child's eyelid and palm scores err together (why pretend children score higher)?
4. Wrong matches: how far would a share of wrong matches (a random same-label nail and palm) push the score up?
Per-child held-out scores are saved too.

Run from the repo root: python -m src.ghana_checks
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_curve

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified_grouped
from src.eyelid.ceiling import fit_and_score, ladder_models, train_text
from src.eyelid.color_features import image_properties
from src.eyelid.debias import AUROC_ONLY
from src.eyelid.report import fmt, markdown_table
from src.eyelid.train_color import _short_commit
from src.ghana_link import ALL3, SITES, children, load_sites, partner_order, rank_mean, same_label_shuffle

LATE = f"{ALL3} (late)"
SHAPE = "size and shape, all three"


def shared_threshold(y, s, spec):
    """The threshold with the highest sensitivity that keeps specificity at or above `spec` (as sens @ 90% spec)."""
    fpr, _, thr = roc_curve(y, s)
    return thr[np.flatnonzero(fpr <= 1 - spec + 1e-12)[-1]]


def auroc_gap(yy, ss, col):
    """AUROC in group 1 minus group 0; ss holds score columns and the 0/1 group last."""
    g = ss[:, -1] == 1
    return auroc(yy[g], ss[g, col]) - auroc(yy[~g], ss[~g, col])


def rate_gap(label, spec):
    """Share flagged at one threshold shared by everyone, group 1 minus group 0, among children with `label`:
    label 1 gives the sensitivity gap, label 0 the false-alarm gap."""
    def f(yy, ss, col):
        flag = ss[:, col] >= shared_threshold(yy, ss[:, col], spec)
        g, m = ss[:, -1] == 1, yy == label
        if not (m & g).any() or not (m & ~g).any():
            raise ValueError("a group has no child with this label")
        return float(flag[m & g].mean() - flag[m & ~g].mean())
    return f


def fairness(y, eye, all3, group, spec, n, ci, seed):
    """Gaps with the eyelid alone and with all three, and the paired change in the size of each gap."""
    packed = np.column_stack([eye, all3, group])
    out = {}
    for name, f in {"auroc": auroc_gap, "sensitivity": rate_gap(1, spec), "false alarms": rate_gap(0, spec)}.items():
        out[name] = {"eyelid": bootstrap_ci(lambda yy, ss: f(yy, ss, 0), y, packed, n, ci, seed=seed),
                     ALL3: bootstrap_ci(lambda yy, ss: f(yy, ss, 1), y, packed, n, ci, seed=seed),
                     "change": bootstrap_ci(lambda yy, ss: abs(f(yy, ss, 1)) - abs(f(yy, ss, 0)), y, packed, n, ci,
                                            seed=seed)}
    return out


def within_label_corr(y, a, b):
    """Rank correlation of two scores inside each anemia label, averaged over the two labels."""
    return float(np.mean([spearmanr(a[y == v], b[y == v])[0] for v in (0, 1)]))


def wrong_matches(y, eye, nail, palm, share, draws, rng):
    """All-three late AUROC per draw when `share` of the children get a random same-label nail and palm."""
    out = []
    for _ in range(draws):
        wrong = rng.permutation(len(y)) < round(share * len(y))
        n2, p2 = nail.copy(), palm.copy()
        n2[wrong] = nail[same_label_shuffle(y, rng)][wrong]
        p2[wrong] = palm[same_label_shuffle(y, rng)][wrong]
        out.append(auroc(y, (eye + n2 + p2) / 3))
    return np.array(out)


def main():
    base, cfg, link_cfg, color_cfg = (load_config(n) for n in ("default", "ghana_checks", "ghana_link", "color"))
    probe_cfg, ceiling_cfg = load_config("probe"), load_config("ceiling")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    cutoff = base["anemia_cutoff_g_dl"]["child_6_59_months"]
    rng = np.random.default_rng(seed)
    models = {n: make for n, (_, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if n in cfg["models"]}

    tabs, _, feats = load_sites(link_cfg, base, color_cfg["roi"]["alpha_threshold"])
    eye, ids, groups = children(tabs)
    kids = eye.loc[ids]
    y = (kids.hb.to_numpy(float) < cutoff).astype(int)
    folds = repeated_stratified_grouped(y, groups, link_cfg["cv"]["n_splits"], link_cfg["cv"]["n_repeats"], seed)
    X = {s: feats[s].loc[ids].to_numpy() for s in SITES}
    X[f"{ALL3} (early)"] = np.hstack([X[s] for s in SITES])
    X[SHAPE] = np.hstack([image_properties(list(tabs[s].loc[ids, "path"]), range(len(ids))).to_numpy()
                          for s in SITES])
    print(f"children: {len(y)}, anemic {y.mean():.1%}", flush=True)

    # baselines, then held-out scores (mean over the 5 repeats) and training AUROC
    head = {"n": len(y), "anemic": f"{y.mean():.1%}"}
    rule = {s: -feats[s].loc[ids, color_cfg["rule_feature"]].to_numpy() for s in SITES}
    rows = [head | {"model": "dummy (prior)", "input": "-", "train": "-",
                    **summarize(y, np.full(len(y), y.mean()), n_boot, ci, seed)},
            head | {"model": "rule_low_redness", "input": "eyelid", "train": "-",
                    **summarize(y, rule["eyelid"], n_boot, ci, seed)},
            head | {"model": "rule_low_redness", "input": f"{ALL3} (rank mean)", "train": "-",
                    **summarize(y, rank_mean(list(rule.values())), n_boot, ci, seed)}]
    scores = {}
    for name, make in models.items():
        for inp, x in X.items():
            held, train = fit_and_score(make, x, y, folds)
            scores[(name, inp)] = held.mean(0)
            rows.append(head | {"model": name, "input": inp, "train": train_text(train),
                                **summarize(y, held, n_boot, ci, seed)})
            print(f"done: {name} {inp}", flush=True)
        scores[(name, LATE)] = np.mean([scores[(name, s)] for s in SITES], axis=0)
        rows.append(head | {"model": name, "input": LATE, "train": "-",
                            **summarize(y, scores[(name, LATE)], n_boot, ci, seed)})

    # 1. fairness
    fc = cfg["fairness"]
    sex, age = kids.sex.to_numpy(object), kids.age_months.to_numpy(float)
    a = fc["age_split_months"]
    groupings = {"girls vs boys": (np.array([s in ("Female", "Male") for s in sex]), sex == "Female",
                                   ("girls", "boys")),
                 f"under {a} months vs older": (~np.isnan(age), age < a, (f"6-{a - 1} months", f"{a}-59 months"))}
    fair, per_group = [], []
    for gname, (known, g, labels) in groupings.items():
        for name in models:
            yy, e, l3 = y[known], scores[(name, "eyelid")][known], scores[(name, LATE)][known]
            fair.append({"grouping": gname, "model": name, "n": int(known.sum()),
                         **fairness(yy, e, l3, g[known].astype(float), fc["specificity"], n_boot, ci, seed)})
            for val, label in ((True, labels[0]), (False, labels[1])):
                m = g[known] == val
                per_group.append({"grouping": gname, "group": label, "model": name, "n": int(m.sum()),
                                  "anemic": int(yy[m].sum()),
                                  "eyelid": summarize(yy[m], e[m], n_boot, ci, seed, metrics=AUROC_ONLY)["auroc"],
                                  ALL3: summarize(yy[m], l3[m], n_boot, ci, seed, metrics=AUROC_ONLY)["auroc"]})
    print("done: fairness", flush=True)

    # 3. same child, same mistakes
    label_batch = np.array([f"{lab}-{b}" for lab, b, _ in ids])
    neighbour = partner_order(label_batch, ids.get_level_values("num").to_numpy(), offset=1)
    corr = []
    for name in models:
        e = scores[(name, "eyelid")]
        for site in ("nail", "palm"):
            p = scores[(name, site)]
            same = lambda yy, ss: within_label_corr(yy, ss[:, 0], ss[:, 1])
            null = np.array([within_label_corr(y, e, p[same_label_shuffle(y, rng)]) for _ in range(cfg["n_shuffles"])])
            corr.append({"model": name, "partner": site,
                         "same number": bootstrap_ci(same, y, np.column_stack([e, p]), n_boot, ci, seed=seed),
                         "neighbouring number": bootstrap_ci(same, y, np.column_stack([e, p[neighbour]]), n_boot, ci,
                                                             seed=seed),
                         "same minus neighbour": bootstrap_ci(
                             lambda yy, ss: within_label_corr(yy, ss[:, 0], ss[:, 1])
                             - within_label_corr(yy, ss[:, 0], ss[:, 2]),
                             y, np.column_stack([e, p, p[neighbour]]), n_boot, ci, seed=seed),
                         "random same-label": {"mean": float(null.mean()), "q975": float(np.quantile(null, 0.975))}})
    print("done: same mistakes", flush=True)

    # 4. wrong matches
    wrong = []
    for name in models:
        e, nl, pl = (scores[(name, s)] for s in SITES)
        for share in cfg["wrong_share"]:
            au = wrong_matches(y, e, nl, pl, share, cfg["wrong_draws"], rng)
            wrong.append({"model": name, "share wrong": share, "auroc mean": float(au.mean()),
                          "lo": float(np.quantile(au, 0.025)), "hi": float(np.quantile(au, 0.975)),
                          "vs eyelid": float(au.mean() - auroc(y, e))})

    per_child = [{"id": f"{lab}-{b}-{n}", "group": int(gr), "anemic": int(v), "hb": float(h),
                  "age_months": None if np.isnan(ag) else float(ag), "sex": s if isinstance(s, str) else None,
                  "hospital": hosp if isinstance(hosp, str) else None,
                  "scores": {f"{m} | {inp}": float(sc[i]) for (m, inp), sc in scores.items()}}
                 for i, ((lab, b, n), gr, v, h, ag, s, hosp) in enumerate(
                     zip(ids, groups, y, kids.hb, age, sex, kids.hospital))]
    path = save_results("stage25_ghana_checks", {"rows": rows, "fairness": fair, "per_group": per_group,
                                                 "same_mistakes": corr, "wrong_matches": wrong,
                                                 "children": per_child},
                        {"default": base, "ghana_checks": cfg, "ghana_link": link_cfg, "color": color_cfg,
                         "probe": probe_cfg, "ceiling": ceiling_cfg})
    write_report(path, cfg, rows, fair, per_group, corr, wrong, list(models), cutoff, n_boot, len(y))


def write_report(path, cfg, rows, fair, per_group, corr, wrong, names, cutoff, n_boot, n):
    fc = cfg["fairness"]
    by = {(r["model"], r["input"]): r for r in rows}
    shape = {m: by[(m, SHAPE)]["auroc"] for m in names}
    no_shortcut = [m for m, a in shape.items() if a["lo"] <= 0.5 or a["estimate"] < 0.6]
    widened = [f"{f['model']}, {f['grouping']}, {k}" for f in fair for k in ("auroc", "sensitivity", "false alarms")
               if f[k]["change"]["lo"] > 0]
    holds = [f"{c['model']} ({c['partner']})" for c in corr
             if c["same number"]["lo"] > 0 and c["same number"]["lo"] > c["random same-label"]["q975"]]
    child = [f"{c['model']} ({c['partner']})" for c in corr if c["same minus neighbour"]["lo"] > 0]

    fair_lines = [f"| {f['grouping']} | {f['model']} | {f['n']} | {k} | {fmt(f[k]['eyelid'])} | {fmt(f[k][ALL3])} | "
                  f"{fmt(f[k]['change'])} |" for f in fair for k in ("auroc", "sensitivity", "false alarms")]
    group_lines = [f"| {g['grouping']} | {g['group']} | {g['model']} | {g['n']} ({g['anemic']} anemic) | "
                   f"{fmt(g['eyelid'])} | {fmt(g[ALL3])} |" for g in per_group]
    corr_lines = [f"| {c['model']} | eyelid - {c['partner']} | {fmt(c['same number'])} | {fmt(c['neighbouring number'])} "
                  f"| {fmt(c['same minus neighbour'])} | {c['random same-label']['mean']:+.3f} "
                  f"(97.5%: {c['random same-label']['q975']:.3f}) |" for c in corr]
    wrong_lines = [f"| {w['model']} | {w['share wrong']:.0%} | {w['auroc mean']:.3f} [{w['lo']:.3f}, {w['hi']:.3f}] | "
                   f"{w['vs eyelid']:+.3f} |" for w in wrong]
    report = (
        "# Stage 25: cheating and fairness checks on one Ghanaian child's eyelid, nail and palm\n\n"
        f"Generated by `python -m src.ghana_checks` at commit `{_short_commit(path)}`; settings in "
        "`configs/ghana_checks.yaml`. The same children, folds and label as Stage 24 "
        "([`stage24_ghana_link.md`](stage24_ghana_link.md)).\n\n"
        f"- **Children:** {n}, each with an eyelid, nail and palm photo under the same file number. **Anemia: Hb < "
        f"{cutoff} g/dL** from the CP-AnemiC sheet (children 6-59 months), one label per child. Age and sex from the "
        "same sheet, when every copy of the photo agrees.\n"
        "- **Scores:** held-out P(anemic) from 5-fold CV x 5 repeats by child (children sharing a photo in one fold), "
        f"averaged over the repeats. *Late* = the mean of the three single-site scores. Bootstrap 95% CIs, {n_boot} "
        "resamples, paired where two scores are compared.\n"
        "- **Answers, fixed before running:** (1) adding nail and palm is fair if it doesn't widen a gap (paired CI of "
        "the change in the gap's size not entirely above 0); (2) no shortcut if size and shape alone give an AUROC "
        "whose CI includes 0.5 or below 0.6; (3) same mistakes if the same-number score correlation's CI is above 0 "
        "and above random partners', and the child (not just the session) if same number minus neighbour is above "
        "0; (4) what-if only.\n\n"
        "## Answers\n\n"
        f"- **(1) Fairness:** adding nail and palm widens a gap for: {', '.join(widened) or 'none'}.\n"
        f"- **(2) No color:** no shortcut in size and shape for: {', '.join(no_shortcut) or 'none'} ("
        + ", ".join(f"{m} {fmt(a)}" for m, a in shape.items()) + ").\n"
        f"- **(3) Same mistakes:** holds for: {', '.join(holds) or 'none'}. The child, not just the session, for: "
        f"{', '.join(child) or 'none'}.\n"
        "- **(4) Wrong matches:** see section 4.\n\n"
        "## 0. Scores\n\n"
        + markdown_table(rows, ["model", "input", "train"],
                         {"auroc": "AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec"}) + "\n\n"
        "**train** = AUROC of each fold's model on its own training children, mean (min-max).\n\n"
        "## 1. Fairness inside Ghana\n\n"
        f"Gap = first group minus second. Sensitivity and false alarms at one threshold for everyone (the one giving "
        f"{fc['specificity']:.0%} specificity on all children). *Change* = |gap with all three| - |gap with the eyelid "
        "alone|, paired: above 0 means adding nail and palm widens the gap.\n\n"
        "| grouping | model | n | measure | gap, eyelid alone | gap, all three (late) | change in gap size |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(fair_lines) + "\n\n"
        "| grouping | group | model | children | AUROC, eyelid alone | AUROC, all three (late) |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(group_lines) + "\n\n"
        "## 2. No color: size and shape of the three photos\n\n"
        "See the *size and shape, all three* rows in section 0: image width, height, aspect and the opaque and edge "
        "shares of each of the three photos (15 numbers).\n\n"
        "## 3. Same child, same mistakes\n\n"
        "Inside each anemia label, the rank correlation between the eyelid's score and the partner's, averaged over "
        "the two labels. Neighbouring number = same label and batch, so the same photo session, but another child. "
        f"Random = {cfg['n_shuffles']} random same-label partners (what Stage 24's pretend children had).\n\n"
        "| model | sites | same number | neighbouring number | same minus neighbour | random same-label, mean |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(corr_lines) + "\n\n"
        "## 4. What if some matches were wrong?\n\n"
        "The photos are matched inside the Anemic / Non-anemic folders, so a wrong match still gives the model a "
        "photo of a child with the same answer. Here a share of the children get a random same-label nail and palm "
        f"(held-out scores reused, no refit), {cfg['wrong_draws']} draws. This shows how much wrong matches could "
        "inflate the score; it can't say how many are actually wrong.\n\n"
        "| model | share of matches wrong | all-three AUROC, mean [2.5%, 97.5% over draws] | vs eyelid alone |\n"
        "|---|---|---|---|\n" + "\n".join(wrong_lines) + "\n"
    )
    (ROOT / "reports" / "stage25_ghana_checks.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
