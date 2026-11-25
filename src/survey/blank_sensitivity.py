"""Stage 29 Step 8c: does the 70%-blank rule throw away signal? (review follow-up)

Role: supporting check, Stage 29 (the 70%-blank rule).

Steps 1-4 are redone with no emptiness limit (`review.sensitivity_max_share_empty`), so the questions dropped only for
being mostly blank come back; blanks stay an answer of their own, as everywhere else. Step 4's repeat rule is rerun on
the larger set. Then gradient boosting (anemic yes/no) and the gradient boosting Hb regressor are compared with and
without them, on the same 5 x 5 repeated village folds as Step 8a. Building villages only; main outputs untouched.

Run from the repo root: python -m src.survey.blank_sensitivity   (after Steps 1-4 and 8a)
"""
import numpy as np
import pandas as pd

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.report import short_commit as _short_commit
from src.survey.repeated_cv import CACHE, Setup
from src.survey.screen_features import group_codes, oof_group_mean, redundant_drops, weighted_auroc
from src.survey.select_features import feature_table, inventory, read_tested
from src.survey.select_models import columns_for, make_model, prepare


def oof(kind, X, S, target):
    out = np.full((S.rc["cv_repeats"], len(S.y)), np.nan)
    for r, tr, te in S.folds:
        m = make_model(kind, X, S.mc, S.seed).fit(X.iloc[tr], target[tr])
        out[r, te] = m.predict_proba(X.iloc[te])[:, 1] if kind == "hist_gradient_boosting" else m.predict(X.iloc[te])
    return out.mean(0)


def main():
    seed = load_config("default")["seed"]
    S = Setup(seed)
    boot, cfg, sc = S.base_cfg["bootstrap"], S.cfg, S.cfg["screening"]
    survey = load_config("survey")

    # Steps 1-2 with no emptiness limit, on the same women in the same order as the saved table.
    raw, meta, text = read_tested(ROOT / survey["countries"][cfg["country"]], cfg)
    saved = pd.read_parquet(ROOT / "data" / "processed" / "zimbabwe_clean.parquet")
    assert (raw[cfg["cluster"]].astype(int).to_numpy() == saved.cluster.to_numpy()).all()
    loose = inventory(raw, meta, text, {**cfg, "max_share_empty": cfg["review"]["sensitivity_max_share_empty"]})
    strict = pd.read_csv(ROOT / "reports" / "zimbabwe_inventory.csv").set_index("code")
    added = [c for c in loose.code[loose.decision == "keep"] if strict.reason.get(c) == "mostly empty"]
    feats_all = feature_table(raw, loose[loose.code.isin(set(added) | set(strict.index[strict.decision == "keep"]))],
                              meta)
    feats = feats_all[~saved.locked_test.to_numpy()].reset_index(drop=True)
    print(f"{len(added)} mostly-blank questions come back", flush=True)

    # Steps 3-4 on the larger set: order by blanks then strength alone, drop repeats.
    folds5 = [f for f in S.folds if f[0] == 0]
    share = loose.set_index("code").share_empty
    pool = [c for c in loose.code[loose.decision == "keep"]]
    codes, strength = {}, {}
    for c in pool:
        code, blank = group_codes(feats, c, sc["number_bins"])
        risk = oof_group_mean(code, S.y.astype(float), folds5, sc["smoothing"])
        strength[c] = weighted_auroc(S.y, risk, np.ones(len(S.y)))[0]
        codes[c] = np.where(blank, -1, code)
    order = sorted(pool, key=lambda c: (share[c], -strength[c]))
    drops = redundant_drops(codes, order, sc["redundancy_lambda"], sc["redundancy_min_overlap"])
    loose_q = [c for c in pool if c not in drops]
    new_q = [c for c in loose_q if c in added]
    print(f"after the repeat rule: {len(loose_q)} questions ({len(new_q)} of them came back)", flush=True)

    X_strict = prepare(S.feats[columns_for(S.questions, S.feats)])
    X_loose = prepare(feats[columns_for(loose_q, feats)])
    cls = {"strict": oof("hist_gradient_boosting", X_strict, S, S.y),
           "loose": oof("hist_gradient_boosting", X_loose, S, S.y)}
    hb_strict = S.cutoff - np.load(CACHE / "hb__all.npy").mean(0)  # Step 8a's Hb predictions on the strict set
    hb = {"strict": hb_strict, "loose": oof("hgb_regressor", X_loose, S, S.hb)}

    ci = lambda fn, a, b: bootstrap_ci(fn, a, b, boot["n_resamples"], boot["ci"], S.groups, seed)
    pair = lambda a, b: np.column_stack([a, b])
    res = {
        "n_added": len(added), "n_questions_strict": len(S.questions), "n_questions_loose": len(loose_q),
        "added_after_repeats": [{"code": c, "label": loose.set_index("code").label[c], "share_empty": float(share[c]),
                                 "auroc_alone": float(strength[c])} for c in new_q],
        "gb_auroc": {k: ci(auroc, S.y, v) for k, v in cls.items()},
        "gb_auroc_gain": ci(lambda y, s: auroc(y, s[:, 1]) - auroc(y, s[:, 0]), S.y, pair(cls["strict"], cls["loose"])),
        "hb_rank_auroc": {k: ci(auroc, S.y, S.cutoff - v) for k, v in hb.items()},
        "hb_rank_auroc_gain": ci(lambda y, s: auroc(y, s[:, 1]) - auroc(y, s[:, 0]), S.y,
                                 pair(S.cutoff - hb["strict"], S.cutoff - hb["loose"])),
        "hb_mae": {k: ci(lambda a, b: float(np.mean(np.abs(a - b))), S.hb, v) for k, v in hb.items()},
        "hb_mae_change": ci(lambda a, b: float(np.mean(np.abs(a - b[:, 1])) - np.mean(np.abs(a - b[:, 0]))), S.hb,
                            pair(hb["strict"], hb["loose"])),
    }
    path = save_results("zimbabwe_blank_sensitivity", res, {"default": S.base_cfg, "feature_selection": cfg})

    c = lambda d: f"{d['estimate']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
    g = lambda d: f"{d['estimate']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"
    top = sorted(res["added_after_repeats"], key=lambda r: -r["auroc_alone"])[:15]
    text = "\n".join([
        "# Stage 29 Step 8c: does the 70%-blank rule throw away signal?",
        "",
        f"Generated by `python -m src.survey.blank_sensitivity` at commit `{_short_commit(path)}`; settings in "
        "`configs/feature_selection.yaml` (`review`). Building villages only, the same 5 x 5 repeated village folds as "
        "Step 8a; village-bootstrap 95% CIs, "
        f"{boot['n_resamples']} resamples. Anemic = DHS `v457` (WHO for women on adjusted Hb: < 12.0 g/dL, < 11.0 if "
        "pregnant).",
        "",
        f"With no emptiness limit, **{len(added)} questions come back**. Blanks stay an answer of their own (a "
        "category, or a was-blank indicator for numbers). After the repeat rule is rerun on the larger set there are "
        f"**{len(loose_q)} questions** ({len(new_q)} of them new) instead of {len(S.questions)}.",
        "",
        "| | 70%-blank rule (main) | no limit | difference |",
        "|---|---|---|---|",
        f"| gradient boosting, AUROC | {c(res['gb_auroc']['strict'])} | {c(res['gb_auroc']['loose'])} | "
        f"{g(res['gb_auroc_gain'])} |",
        f"| Hb regressor, ranking AUROC | {c(res['hb_rank_auroc']['strict'])} | {c(res['hb_rank_auroc']['loose'])} | "
        f"{g(res['hb_rank_auroc_gain'])} |",
        f"| Hb regressor, MAE (g/dL) | {c(res['hb_mae']['strict'])} | {c(res['hb_mae']['loose'])} | "
        f"{g(res['hb_mae_change'])} |",
        "",
        "The strongest returning questions on their own (Step 3 score, point estimate):",
        "",
        "| code | question | share blank | AUROC alone |",
        "|---|---|---|---|",
        *[f"| `{r['code']}` | {r['label']} | {r['share_empty']:.0%} | {r['auroc_alone']:.3f} |" for r in top],
        "",
    ])
    (ROOT / "reports" / "zimbabwe_blank_sensitivity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
