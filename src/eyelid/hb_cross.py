"""Stage 17: an Hb model trained on one eyelid dataset, applied unchanged to the other, each with its own cutoff.

Stage 16 showed eyelid color estimates Hb within a dataset. The mentors' idea is that an Hb estimate, unlike a yes/no
label, can cross datasets whose anemia cutoffs differ: estimate Hb, then apply the local cutoff. Here a model is fit on
all of CP-AnemiC (Ghana children) and scored on Eyes-Defy (India and Italy adults), and the reverse, next to yes/no
classifiers trained the same way and the untrained redness rule. A second version standardizes each dataset's color
numbers on its own photos (no labels), the per-country scaling of Stage 14. scikit-learn only.

Each person's risk score is their estimated Hb minus their own cutoff (lower = more likely anemic), so a man and a
woman with the same estimate are judged against 13.0 and 12.0 g/dL respectively. When the estimates barely vary, that
score is driven by the cutoff (that is, by sex), so the AUROC of the estimate alone is reported too.

Run from the repo root: python -m src.eyelid.hb_cross
"""
from functools import partial

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.eyelid.color_features import feature_table
from src.eyelid.dataset import load_cp_anemic
from src.eyelid.hb_regression import (CLASS_METRICS, classifiers, mae, regressors, sens_at_cutoff, spec_at_cutoff,
                                      within_1)
from src.eyelid.report import markdown_table
from src.eyelid.train_color import _short_commit

COLUMNS = {"mae": "Hb error (MAE, g/dL)", "bias": "Mean estimate - lab Hb", "within_1": "Within 1 g/dL",
           "auroc_estimate": "AUROC, ranked by estimate alone", "auroc": "AUROC", "pr_auc": "PR-AUC", "sens_at_90_spec": "Sens @ 90% spec",
           "share_tested_90": "Share tested to catch 90%", "sens_at_cutoff": "Sens at own cutoff",
           "spec_at_cutoff": "Spec at own cutoff"}


def bias(hb, pred):
    """Mean of (estimate - lab Hb): positive = the model reads people as healthier than they are."""
    return float(np.mean(np.asarray(pred) - np.asarray(hb)))


def standardize(X):
    """Scale each color number to mean 0, sd 1 within one dataset, from that dataset's own photos (no labels)."""
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def load_ghana(cfg, base, color_cfg):
    cp = load_cp_anemic(cfg["cp_anemic_subset"])
    feats = feature_table([ROOT / p for p in cp.image_path], cp.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    cutoff = base["anemia_cutoff_g_dl"]["child_6_59_months"]
    return {"hb": cp.hb.to_numpy(dtype=float), "cutoff": np.full(len(cp), cutoff), "feats": feats,
            "country": np.full(len(cp), "Ghana")}


def load_eyes_defy(cfg, color_cfg):
    """Photos with Hb, their palpebral mask, WHO cutoff by sex. Same as src.eyelid.cross_dataset.load_eyes_defy, which
    can't be imported here because that module loads PyTorch."""
    xcfg = load_config("cross_dataset")
    df = pd.read_csv(ROOT / "data" / "processed" / "eyes_defy_labels.csv").dropna(subset=["hb"]).reset_index(drop=True)
    masks = []
    for p in df.image_path:
        found = [m for m in (ROOT / p).parent.glob(xcfg["eyes_defy_mask_glob"]) if "forniceal" not in m.name]
        assert len(found) == 1, f"{p}: expected one palpebral mask, found {found}"
        masks.append(found[0])
    cutoff = df.sex.map(xcfg["eyes_defy_cutoff_g_dl"])
    assert cutoff.notna().all(), "a sex value with no cutoff"
    feats = feature_table(masks, df.id, alpha_threshold=color_cfg["roi"]["alpha_threshold"])
    return {"hb": df.hb.to_numpy(dtype=float), "cutoff": cutoff.to_numpy(dtype=float), "feats": feats,
            "country": df.country.to_numpy()}


def main():
    base, cfg, color_cfg = load_config("default"), load_config("hb_cross"), load_config("color")
    reg_cfg = load_config("hb_regression")
    seed, boot = base["seed"], base["bootstrap"]
    data = {"cp_anemic": load_ghana(cfg, base, color_cfg), "eyes_defy": load_eyes_defy(cfg, color_cfg)}
    for name, d in data.items():
        y = d["hb"] < d["cutoff"]
        print(f"{name}: {len(y)} photos, Hb mean {d['hb'].mean():.2f} g/dL, {y.mean():.1%} anemic at its own cutoff",
              flush=True)

    def ci(metric, truth, score):
        return bootstrap_ci(metric, truth, score, boot["n_resamples"], boot["ci"], None, seed)

    rows = []
    for src, dst in cfg["directions"]:
        s, t = data[src], data[dst]
        ys, yt = (s["hb"] < s["cutoff"]).astype(int), (t["hb"] < t["cutoff"]).astype(int)
        subsets = {"all": np.ones(len(yt), bool)}
        if dst == "eyes_defy":
            subsets |= {c: t["country"] == c for c in ("India", "Italy")}
        for version in cfg["feature_versions"]:
            Xs, Xt = s["feats"].to_numpy(), t["feats"].to_numpy()
            if version == "standardized per dataset":
                Xs, Xt = standardize(Xs), standardize(Xt)
            runs = {}
            for name, make in {"dummy: source mean Hb": DummyRegressor, **regressors(reg_cfg, seed)}.items():
                runs[("Hb value", name)] = make().fit(Xs, s["hb"]).predict(Xt)
            for name, make in {"dummy": lambda: DummyClassifier(strategy="prior"), **classifiers(reg_cfg, seed)}.items():
                runs[("anemic yes/no", name)] = make().fit(Xs, ys).predict_proba(Xt)[:, 1]
            if version == "as photographed":  # the rule needs no training, so it doesn't change with the version
                runs[("anemic yes/no", "rule: less red = anemic (no training)")] = -t["feats"][color_cfg["rule_feature"]].to_numpy()
            for (target, name), out in runs.items():
                for subset, m in subsets.items():
                    hb, cut, y = t["hb"][m], t["cutoff"][m], yt[m]
                    info = {"direction": f"{src} -> {dst}", "test set": subset, "n": int(m.sum()),
                            "anemic": f"{y.mean():.1%}", "features": version, "target": target, "model": name}
                    if target == "Hb value":
                        pred = out[m]
                        margin = pred - cut  # below 0 = the estimate falls under this person's own cutoff
                        stats = {"mae": ci(mae, hb, pred), "bias": ci(bias, hb, pred), "within_1": ci(within_1, hb, pred),
                                 "auroc_estimate": ci(auroc, y, -pred),
                                 **{k: ci(f, y, -margin) for k, f in CLASS_METRICS.items()},
                                 "sens_at_cutoff": ci(partial(sens_at_cutoff, cutoff=0.0), y, margin),
                                 "spec_at_cutoff": ci(partial(spec_at_cutoff, cutoff=0.0), y, margin)}
                    else:
                        stats = {k: ci(f, y, out[m]) for k, f in CLASS_METRICS.items()}
                    rows.append({**info, **stats})
            print(f"{src} -> {dst} | {version}: " + ", ".join(
                f"{r['model'].split(':')[0]} ({r['target'][:2]}) {r['auroc']['estimate']:.3f}" for r in rows
                if r["direction"] == f"{src} -> {dst}" and r["features"] == version and r["test set"] == "all"), flush=True)

    path = save_results("stage17_hb_cross", {"rows": rows}, {"default": base, "hb_cross": cfg, "hb_regression": reg_cfg,
                                                             "color": color_cfg, "cross_dataset": load_config("cross_dataset")})
    table = markdown_table(rows, ["direction", "test set", "n", "anemic", "features", "target", "model"], COLUMNS)
    text = (
        "# Stage 17: an Hb model across eyelid datasets, each with its own cutoff\n\n"
        f"Generated by `python -m src.eyelid.hb_cross` at commit `{_short_commit(path)}`; settings in "
        "`configs/hb_cross.yaml`. Mentor advice (2026-10-31): estimate Hb, then apply the local cutoff, so datasets "
        "with different cutoffs can share one model.\n\n"
        "- **Data:** CP-AnemiC, the 388 photos with no copy (children, **anemic = Hb < 11.0 g/dL**); Eyes-Defy, 217 "
        "photos with Hb (adults, **WHO by sex: women < 12.0, men < 13.0 g/dL**, pregnancy not recorded). The cutoffs "
        "differ by design; each is applied only to its own dataset.\n"
        "- **Input:** the 29 color numbers of the palpebral conjunctiva. `as photographed` uses them unchanged; "
        "`standardized per dataset` scales each dataset's numbers to mean 0, sd 1 on its own photos (no labels).\n"
        "- **Hb value rows:** a regressor fit on all of one dataset predicts Hb on the other. Hb error and bias are "
        "against the lab Hb. The risk score is the estimate minus the person's own cutoff, so `Sens/Spec at own cutoff` "
        "apply each dataset's cutoff to the estimate. `AUROC, ranked by estimate alone` ignores the cutoff (and so sex): it "
        "asks only whether a lower estimate goes with anemia. When estimates barely vary, the other AUROC is driven by "
        "the cutoff, i.e. by sex (in Eyes-Defy women are more often anemic but have the lower cutoff).\n"
        "- **anemic yes/no rows:** classifiers fit on the source labels (its own cutoff) and applied unchanged; the "
        "untrained rule (minus mean a*) for reference.\n"
        "- **Italy can't be trusted on its own:** 21 of its 23 anemic patients are in one early photo batch (Stage 10), "
        "and that batch alone predicts anemia at AUROC 0.91.\n"
        "- Bootstrap 95% CIs, 2000 resamples, one photo per person.\n\n"
        f"{table}\n"
    )
    (ROOT / "reports" / "stage17_hb_cross.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
