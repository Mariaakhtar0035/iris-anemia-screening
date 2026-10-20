"""Stage 11: can the India/Italy shortcut be taken out of the eyelid model? (mentor advice M5-M8)

The methods, run on Eyes-Defy pooled with the same folds:
- the plain models;
- making images alike (gray-world, brightness scaling, per-country feature standardization);
- brightness and contrast jitter on the training photos;
- reweighing so country and label are independent in training (Kamiran and Calders, 2012);
- a fairness-penalized logistic regression;
- separate models per country.

Every method is scored on accuracy within each country and on how differently it treats the two countries. Then
leave-one-country-out, and transfer to CP-AnemiC.

Run from the repo root: python -m src.eyelid.debias
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler

from src.common.evaluate import summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci, precision_at_sensitivity, share_tested_at_sensitivity
from src.common.splits import repeated_stratified
from src.eyelid.color_features import color_features
from src.eyelid.cross_dataset import eyes_defy_embeddings, load_eyes_defy
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.report import markdown_table
from src.eyelid.roi import roi_pixels
from src.eyelid.train_color import model_factories
from src.eyelid.train_probe import probe_factory

AUROC_ONLY = {"auroc": auroc}


class FairLogisticRegression:
    """L2 logistic regression on standardized features, plus lam * sum over labels of the squared gap in mean
    predicted risk between two groups. With lam = 0 it is sklearn's StandardScaler + LogisticRegression(C)."""

    def __init__(self, lam, C=1.0):
        self.lam, self.C = lam, C

    def fit(self, X, y, groups):
        self.scaler = StandardScaler().fit(X)
        A = np.hstack([self.scaler.transform(X), np.ones((len(X), 1))])
        y, g = np.asarray(y, float), np.asarray(groups)
        g0, g1 = np.unique(g)
        cells = [((y == k) & (g == g0), (y == k) & (g == g1)) for k in (0, 1)]
        n = len(y)

        def loss_and_grad(w):
            z = A @ w
            p = expit(z)
            loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * w[:-1] @ w[:-1] / (self.C * n)
            grad = A.T @ (p - y) / n
            grad[:-1] += w[:-1] / (self.C * n)
            dp = p * (1 - p)
            for a, b in cells:
                gap = p[a].mean() - p[b].mean()
                loss += self.lam * gap ** 2
                grad += self.lam * 2 * gap * (A[a].T @ dp[a] / a.sum() - A[b].T @ dp[b] / b.sum())
            return loss, grad

        self.w = minimize(loss_and_grad, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x
        return self

    def predict_proba(self, X):
        p = expit(np.hstack([self.scaler.transform(X), np.ones((len(X), 1))]) @ self.w)
        return np.column_stack([1 - p, p])


def reweigh(y, groups):
    """Kamiran-Calders weights P(group) P(label) / P(group, label): country and label become independent."""
    w = np.empty(len(y))
    for g in np.unique(groups):
        for k in np.unique(y):
            m = (groups == g) & (y == k)
            w[m] = (groups == g).mean() * (y == k).mean() / m.mean()
    return w


def group_standardize(X_fit, g_fit, X_apply, g_apply):
    """Standardize each group's features with that group's own mean and SD from X_fit (no labels used)."""
    out = np.empty_like(X_apply, dtype=float)
    for g in np.unique(g_apply):
        ref = X_fit[g_fit == g]
        out[g_apply == g] = (X_apply[g_apply == g] - ref.mean(0)) / (ref.std(0) + 1e-9)
    return out


def jitter(rgb, rng, cfg):
    b = rng.uniform(1 - cfg["brightness"], 1 + cfg["brightness"])
    c = rng.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
    m = rgb.mean(0)
    return np.clip(((rgb - m) * c + m) * b, 0, 1)


def color_variants(paths, cfg, alpha_threshold, seed):
    """Color features per photo: as is, gray-world, brightness-scaled, and `copies` jittered versions."""
    out = {"plain": [], "gray_world": [], "brightness": [], "jittered": []}
    rng = np.random.default_rng(seed)
    for p in paths:
        rgb = roi_pixels(p, alpha_threshold)
        out["plain"].append(list(color_features(rgb).values()))
        out["gray_world"].append(list(color_features(rgb, normalize=True).values()))
        out["brightness"].append(list(color_features(np.clip(rgb * cfg["brightness_target"] / rgb.mean(), 0, 1)).values()))
        out["jittered"].append([list(color_features(jitter(rgb, rng, cfg["jitter"])).values())
                                for _ in range(cfg["jitter"]["copies"])])
    return {k: np.array(v) for k, v in out.items()}


def fit_predict(make, X_tr, y_tr, X_te, weight=None, groups=None):
    model = make()
    if isinstance(model, FairLogisticRegression):
        model.fit(X_tr, y_tr, groups)
    elif weight is not None:  # pipelines name their last step; plain estimators take sample_weight directly
        key = f"{model.steps[-1][0]}__sample_weight" if hasattr(model, "steps") else "sample_weight"
        model.fit(X_tr, y_tr, **{key: weight})
    else:
        model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def fairness(y, s, india, early, cfg, boot, seed):
    """The Stage 11 fairness metrics for one set of scores (averaged over repeats)."""
    n, ci = boot["n_resamples"], boot["ci"]
    packed = np.column_stack([s, india])

    def rank_gap(label):
        def f(yy, ss):
            r = rankdata(ss[:, 0]) / len(ss)
            m = yy == label
            return float(r[m & (ss[:, 1] == 1)].mean() - r[m & (ss[:, 1] == 0)].mean())
        return f

    def rate_gap(label):
        def f(yy, ss):
            fpr, tpr, thr = roc_curve(yy, ss[:, 0])
            t = thr[np.flatnonzero(tpr >= cfg["referral_sensitivity"] - 1e-12)[0]]
            flag, m = ss[:, 0] >= t, yy == label
            return float(flag[m & (ss[:, 1] == 1)].mean() - flag[m & (ss[:, 1] == 0)].mean())
        return f

    it = ~india
    out = {"india": summarize(y[india], s[india], n, ci, seed, metrics=AUROC_ONLY)["auroc"],
           "italy": summarize(y[it], s[it], n, ci, seed, metrics=AUROC_ONLY)["auroc"],
           "italy_early": summarize(y[it & early], s[it & early], n, ci, seed, metrics=AUROC_ONLY)["auroc"],
           "rank_gap_non_anemic": bootstrap_ci(rank_gap(0), y, packed, n, ci, seed=seed),
           "rank_gap_anemic": bootstrap_ci(rank_gap(1), y, packed, n, ci, seed=seed),
           "tpr_gap": bootstrap_ci(rate_gap(1), y, packed, n, ci, seed=seed),
           "fpr_gap": bootstrap_ci(rate_gap(0), y, packed, n, ci, seed=seed),
           "share_tested": bootstrap_ci(lambda yy, ss: share_tested_at_sensitivity(yy, ss, cfg["referral_sensitivity"]),
                                        y, s, n, ci, seed=seed),
           "precision": bootstrap_ci(lambda yy, ss: precision_at_sensitivity(yy, ss, cfg["referral_sensitivity"]),
                                     y, s, n, ci, seed=seed)}
    out["worst"] = {"estimate": min(out["india"]["estimate"], out["italy"]["estimate"]), "lo": np.nan, "hi": np.nan}
    return out


def main():
    base, cfg = load_config("default"), load_config("debias")
    color_cfg, probe_cfg = load_config("color"), load_config("probe")
    seed, boot = base["seed"], base["bootstrap"]
    alpha = color_cfg["roi"]["alpha_threshold"]
    ed = load_eyes_defy(load_config("cross_dataset"))
    y, country = ed.anemic.to_numpy(), ed.country.to_numpy()
    india = country == "India"
    early = ~india & (ed.id.str.split("_").str[1].astype(int) <= cfg["italy_early_batch_last_folder"]).to_numpy()
    feats = color_variants([ROOT / p for p in ed.image_path], cfg, alpha, seed)
    emb = eyes_defy_embeddings(ed, probe_cfg)
    a_col = list(color_features(np.full((2, 3), 0.5)).keys()).index(color_cfg["rule_feature"])
    rule = -feats["plain"][:, a_col]
    color = model_factories(color_cfg, seed)
    lr, rf, probe = color["logistic_regression"], color["random_forest"], probe_factory(probe_cfg, seed)
    k = cfg["jitter"]["copies"]

    def plain(make, X):
        return lambda tr, te: fit_predict(make, X[tr], y[tr], X[te])

    def standardized(make):
        X = feats["plain"]
        return lambda tr, te: fit_predict(make, group_standardize(X[tr], country[tr], X[tr], country[tr]), y[tr],
                                          group_standardize(X[tr], country[tr], X[te], country[te]))

    def jittered(make):
        def f(tr, te):
            X_tr = np.vstack([feats["plain"][tr], feats["jittered"][tr].reshape(-1, feats["plain"].shape[1])])
            return fit_predict(make, X_tr, np.concatenate([y[tr], np.repeat(y[tr], k)]), feats["plain"][te])
        return f

    def reweighed(make):
        return lambda tr, te: fit_predict(make, feats["plain"][tr], y[tr], feats["plain"][te], reweigh(y[tr], country[tr]))

    def fair(lam):
        return lambda tr, te: fit_predict(lambda: FairLogisticRegression(lam, cfg["fair_lr"]["C"]), feats["plain"][tr],
                                          y[tr], feats["plain"][te], groups=country[tr])

    def per_country(make):
        def f(tr, te):
            out = np.empty(len(te))
            for c in np.unique(country):
                a, b = tr[country[tr] == c], te[country[te] == c]
                out[country[te] == c] = fit_predict(make, feats["plain"][a], y[a], feats["plain"][b])
            return out
        return f

    methods = {
        ("none", "logistic_regression"): plain(lr, feats["plain"]),
        ("none", "random_forest"): plain(rf, feats["plain"]),
        ("none", "dinov2_probe"): plain(probe, emb),
        ("M6 gray-world", "logistic_regression"): plain(lr, feats["gray_world"]),
        ("M6 gray-world", "random_forest"): plain(rf, feats["gray_world"]),
        ("M6 brightness scaled", "logistic_regression"): plain(lr, feats["brightness"]),
        ("M6 brightness scaled", "random_forest"): plain(rf, feats["brightness"]),
        ("M6 per-country standardized", "logistic_regression"): standardized(lr),
        ("M6 per-country standardized", "random_forest"): standardized(rf),
        ("M6 brightness/contrast jitter", "logistic_regression"): jittered(lr),
        ("M6 brightness/contrast jitter", "random_forest"): jittered(rf),
        ("M5 reweighing", "logistic_regression"): reweighed(lr),
        ("M5 reweighing", "random_forest"): reweighed(rf),
        **{(f"M5 fairness penalty, lambda={lam:g}", "logistic_regression"): fair(lam) for lam in cfg["fair_lr"]["lambdas"]},
        ("M8 one model per country", "logistic_regression"): per_country(lr),
        ("M8 one model per country", "random_forest"): per_country(rf),
    }
    strata = 2 * india + y
    folds = repeated_stratified(strata, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
    n_rep = cfg["cv"]["n_repeats"]
    scores = {("baseline", "dummy (prior)"): np.full(len(y), y.mean()),
              ("baseline", "country rule (India = anemic)"): india.astype(float),
              ("baseline", "rule_low_redness (no training)"): rule}
    for key, score_fold in methods.items():
        s = np.full((n_rep, len(y)), np.nan)
        for r, tr, te in folds:
            s[r, te] = score_fold(tr, te)
        assert not np.isnan(s).any()
        scores[key] = s.mean(0)
        print(f"{key}: India {auroc(y[india], scores[key][india]):.3f}, Italy {auroc(y[~india], scores[key][~india]):.3f}",
              flush=True)

    rows = []
    for (method, model), s in scores.items():
        info = {"method": method, "model": model}
        pooled = summarize(y, s, boot["n_resamples"], boot["ci"], seed) if np.unique(s).size > 1 else {}
        rows.append({**info, **pooled, **fairness(y, s, india, early, cfg, boot, seed)})

    # Leave one country out, and Eyes-Defy -> CP-AnemiC. Target-country standardization uses the target's own
    # unlabelled feature statistics.
    cp = load_cp_anemic("singletons")
    cp_feats = color_variants([ROOT / p for p in cp.image_path], cfg, alpha, seed)
    transfer = []
    targets = {"India -> Italy": (india, ~india, feats, y, country),
               "Italy -> India": (~india, india, feats, y, country),
               "Eyes-Defy -> CP-AnemiC": (np.ones(len(y), bool), None, cp_feats, cp.anemic.to_numpy(),
                                          np.full(len(cp), "CP"))}
    for name, (tr, te, tf, ty, tg) in targets.items():
        Xt = {k2: (v[te] if te is not None else v) for k2, v in tf.items()}
        yt = ty[te] if te is not None else ty
        gt = tg[te] if te is not None else tg
        Xs, ys = feats["plain"][tr], y[tr]
        # For transfer, "per-country standardized" = each side standardized with its own statistics.
        Xs_std = group_standardize(Xs, country[tr], Xs, country[tr])
        Xt_std = group_standardize(Xt["plain"], gt, Xt["plain"], gt)
        Xs_jit = np.vstack([Xs, feats["jittered"][tr].reshape(-1, Xs.shape[1])])
        ys_jit = np.concatenate([ys, np.repeat(ys, k)])
        cand = {"rule_low_redness (no training)": -Xt["plain"][:, a_col],
                "logistic_regression": fit_predict(lr, Xs, ys, Xt["plain"]),
                "random_forest": fit_predict(rf, Xs, ys, Xt["plain"]),
                "logistic_regression, gray-world": fit_predict(lr, feats["gray_world"][tr], ys, Xt["gray_world"]),
                "logistic_regression, brightness scaled": fit_predict(lr, feats["brightness"][tr], ys, Xt["brightness"]),
                "logistic_regression, each side standardized": fit_predict(lr, Xs_std, ys, Xt_std),
                "random_forest, each side standardized": fit_predict(rf, Xs_std, ys, Xt_std),
                "logistic_regression, jitter": fit_predict(lr, Xs_jit, ys_jit, Xt["plain"]),
                "random_forest, jitter": fit_predict(rf, Xs_jit, ys_jit, Xt["plain"])}
        if name.startswith("Eyes-Defy"):
            cand["logistic_regression, reweighing"] = fit_predict(lr, Xs, ys, Xt["plain"], reweigh(ys, country[tr]))
            for lam in cfg["fair_lr"]["lambdas"]:
                cand[f"fairness penalty, lambda={lam:g}"] = fit_predict(
                    lambda: FairLogisticRegression(lam, cfg["fair_lr"]["C"]), Xs, ys, Xt["plain"], groups=country[tr])
        subsets = {"all": np.ones(len(yt), bool)}
        if name == "India -> Italy":
            subsets["Italy early batch"] = early[te]  # the Italy test the batch alone can't pass
        for subset, m in subsets.items():
            for model, s in cand.items():
                transfer.append({"train -> test": name, "test set": subset, "n": int(m.sum()), "anemic": f"{yt[m].mean():.1%}",
                                 "model": model, **summarize(yt[m], s[m], boot["n_resamples"], boot["ci"], seed)})
        print(f"transfer {name} done", flush=True)

    def f3(m):
        return f"{m['estimate']:.3f}" if np.isnan(m["lo"]) else f"{m['estimate']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"

    acc_head = ["method", "model", "pooled AUROC", "India AUROC", "Italy AUROC", "Italy early batch AUROC", "worst country"]
    fair_head = ["method", "model", "rank gap, non-anemic", "rank gap, anemic", "TPR gap", "FPR gap",
                 f"share tested for {cfg['referral_sensitivity']:.0%} sens.", "precision there"]
    acc_lines = [f"| {r['method']} | {r['model']} | {f3(r['auroc']) if 'auroc' in r else '-'} | {f3(r['india'])} | "
                 f"{f3(r['italy'])} | {f3(r['italy_early'])} | {r['worst']['estimate']:.3f} |" for r in rows]
    fair_lines = [f"| {r['method']} | {r['model']} | {f3(r['rank_gap_non_anemic'])} | {f3(r['rank_gap_anemic'])} | "
                  f"{f3(r['tpr_gap'])} | {f3(r['fpr_gap'])} | {f3(r['share_tested'])} | {f3(r['precision'])} |" for r in rows]

    def table(head, lines):
        return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join(lines)

    n_it_early = int(early.sum())
    text = (
        "# Stage 11: taking the India/Italy shortcut out\n\n"
        "Generated by `python -m src.eyelid.debias`; settings in `configs/debias.yaml`. Mentor advice M5-M8 "
        "(`docs/eyelid_followup.md`). Eyes-Defy: India 95 and Italy 122 photos, anemic = WHO adult cutoffs by sex "
        f"(women < 12.0, men < 13.0 g/dL); 71.6% of India and 18.9% of Italy anemic. Color features from the "
        f"hand-segmented palpebral conjunctiva. {cfg['cv']['n_splits']}-fold CV x {n_rep} repeats stratified by country x "
        f"anemia, scores averaged over repeats; bootstrap 95% CIs ({boot['n_resamples']} resamples).\n\n"
        f"- **Italy early batch**: folders 1-31 only ({n_it_early} photos, {int(y[early].sum())} anemic). Almost all "
        "Italian anemia is in this batch (Stage 10), so this is the Italy check that recognizing the batch can't pass.\n"
        "- **Rank gap**: mean percentile rank of Indian minus Italian photos, among people with the same anemia status. "
        "0 = once anemia is fixed, the model doesn't care about the country.\n"
        f"- **TPR / FPR gap**: India minus Italy, at the one threshold that catches {cfg['referral_sensitivity']:.0%} of "
        "anemic people overall.\n\n"
        "## Accuracy, overall and inside each country\n\n" + table(acc_head, acc_lines)
        + "\n\n## Fairness gaps and testing load\n\n" + table(fair_head, fair_lines)
        + "\n\n## Across countries and datasets\n\nFit on one side, applied unchanged to the other. \"Each side "
        "standardized\" uses each side's own unlabelled feature means and SDs. CP-AnemiC: the 388 photos with no copy, "
        "anemic = Hb < 11.0 g/dL (children), so the cutoffs differ by design.\n\n"
        + markdown_table(transfer, ["train -> test", "test set", "n", "anemic", "model"]) + "\n"
    )
    (ROOT / "reports" / "stage11_debias.md").write_text(text)
    print(text)
    save_results("stage11_debias", {"rows": rows, "transfer": transfer},
                 {"default": base, "debias": cfg, "color": color_cfg, "probe": probe_cfg})


if __name__ == "__main__":
    main()
