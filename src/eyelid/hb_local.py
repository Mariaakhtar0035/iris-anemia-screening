"""Stage 18: correct a Ghana-trained Hb model with a few local Hb measurements at the new site.

Stage 17 found the Ghana model's Hb estimates are shifted at Eyes-Defy (almost 3 g/dL too low in Italy), so the local
cutoff flags nearly everyone. If a new site measures Hb on k people, a simple correction fit on them may fix the shift.
For each site (India, Italy) and k, k people are drawn at random; the correction is fit on them and everything is scored
on the rest of the site. Compared with no correction, with a regressor trained on the k people alone (as in Stage 15),
and with the untrained redness rule. A shift keeps everyone's order, so it can't change the ranking (AUROC); it can only
fix the cutoff decisions. Numbers are means over draws with the 2.5-97.5% range over draws (as in Stage 8).

Run from the repo root: python -m src.eyelid.hb_local
"""
import numpy as np

from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc
from src.eyelid.hb_cross import load_eyes_defy, load_ghana
from src.eyelid.hb_regression import regressors
from src.eyelid.train_color import _short_commit


def shift(pred_k, hb_k):
    """Return f(pred): add the mean local error, so the k people are right on average."""
    offset = float(np.mean(hb_k - pred_k))
    return lambda pred: pred + offset


def shift_and_scale(pred_k, hb_k):
    """Return f(pred): least-squares line from estimate to lab Hb over the k people; falls back to a shift if the
    estimates of the k people are all equal."""
    if np.ptp(pred_k) == 0:
        return shift(pred_k, hb_k)
    slope, intercept = np.polyfit(pred_k, hb_k, 1)
    return lambda pred: intercept + slope * pred


CORRECTIONS = {"shift": shift, "shift and scale": shift_and_scale}


def cutoff_stats(hb, pred, cutoff):
    """Hb error, bias, and the cutoff applied to the estimate (each person against their own cutoff)."""
    anemic, flagged = hb < cutoff, pred < cutoff
    return {"mae": float(np.mean(np.abs(pred - hb))), "bias": float(np.mean(pred - hb)),
            "sens_at_cutoff": float(flagged[anemic].mean()) if anemic.any() else np.nan,
            "spec_at_cutoff": float((~flagged[~anemic]).mean()) if (~anemic).any() else np.nan,
            "auroc_estimate": safe_auroc(anemic, -pred)}


def safe_auroc(y, s):
    try:
        return auroc(np.asarray(y).astype(int), s)
    except ValueError:  # a draw that left only one class in the rest of the site
        return np.nan


def summarize_draws(draws):
    """Mean and 2.5-97.5% range over draws for every metric (draws with an undefined metric are skipped for it)."""
    out = {}
    for key in draws[0]:
        v = np.array([d[key] for d in draws], dtype=float)
        v = v[~np.isnan(v)]
        out[key] = {"estimate": float(v.mean()), "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                    "n_draws": int(len(v))}
    return out


def main():
    base, cfg, color_cfg = load_config("default"), load_config("hb_local"), load_config("color")
    cross_cfg, reg_cfg = load_config("hb_cross"), load_config("hb_regression")
    seed = base["seed"]
    ghana, ed = load_ghana(cross_cfg, base, color_cfg), load_eyes_defy(cross_cfg, color_cfg)
    Xg, Xe = ghana["feats"].to_numpy(), ed["feats"].to_numpy()
    rule = -ed["feats"][color_cfg["rule_feature"]].to_numpy()
    reg = regressors(reg_cfg, seed)
    preds = {m: reg[m]().fit(Xg, ghana["hb"]).predict(Xe) for m in cfg["models"]}  # trained once, on all of Ghana

    rows = []
    for site in cfg["sites"]:
        idx = np.flatnonzero(ed["country"] == site)
        hb, cut, X = ed["hb"][idx], ed["cutoff"][idx], Xe[idx]
        print(f"{site}: {len(idx)} photos, Hb mean {hb.mean():.2f}, {(hb < cut).mean():.1%} anemic", flush=True)
        for k in cfg["k_local"]:
            rng = np.random.default_rng(seed + k)
            draws = {}
            for _ in range(cfg["n_draws"]):
                pick = rng.choice(len(idx), k, replace=False)
                rest = np.setdiff1d(np.arange(len(idx)), pick)

                def add(name, pred_rest, extra=None):
                    draws.setdefault(name, []).append({**cutoff_stats(hb[rest], pred_rest, cut[rest]), **(extra or {})})

                for m, p in preds.items():
                    p = p[idx]
                    add(f"Ghana {m}, no correction", p[rest])
                    for cname, fit in CORRECTIONS.items():
                        add(f"Ghana {m}, {cname} on {k} local", fit(p[pick], hb[pick])(p[rest]))
                local = reg[cfg["local_only_model"]]().fit(X[pick], hb[pick]).predict(X[rest])
                add(f"{cfg['local_only_model']} trained on the {k} local only", local)
                draws.setdefault("rule: less red = anemic (no training)", []).append(
                    {"auroc_estimate": safe_auroc(hb[rest] < cut[rest], rule[idx][rest])})
            for name, d in draws.items():
                rows.append({"site": site, "k_local": k, "model": name, **summarize_draws(d)})
            print(f"  k={k}: " + "; ".join(f"{r['model']}: MAE {r['mae']['estimate']:.2f}, sens {r['sens_at_cutoff']['estimate']:.2f}, "
                                            f"spec {r['spec_at_cutoff']['estimate']:.2f}" for r in rows
                                            if r["site"] == site and r["k_local"] == k and "mae" in r and "random_forest" in r["model"]),
                  flush=True)

    path = save_results("stage18_hb_local", {"rows": rows}, {"default": base, "hb_local": cfg, "hb_cross": cross_cfg,
                                                             "hb_regression": reg_cfg, "color": color_cfg})
    cols = {"mae": "Hb error (MAE)", "bias": "Mean estimate - lab Hb", "sens_at_cutoff": "Sens at own cutoff",
            "spec_at_cutoff": "Spec at own cutoff", "auroc_estimate": "AUROC (ranked by estimate)"}
    fmt = lambda m: f"{m['estimate']:.2f} [{m['lo']:.2f}, {m['hi']:.2f}]"
    lines = ["| site | k local | model | " + " | ".join(cols.values()) + " |", "|" + "---|" * (3 + len(cols))]
    for r in rows:
        lines.append(f"| {r['site']} | {r['k_local']} | {r['model']} | " + " | ".join(fmt(r[c]) if c in r else "-" for c in cols) + " |")
    text = (
        "# Stage 18: correct a Ghana-trained Hb model with a few local Hb measurements\n\n"
        f"Generated by `python -m src.eyelid.hb_local` at commit `{_short_commit(path)}`; settings in "
        "`configs/hb_local.yaml`.\n\n"
        "- **Model:** regressors trained once on all 388 CP-AnemiC photos (Ghana children, 29 color numbers), applied "
        "to Eyes-Defy (Stage 17). **Anemic at the new site = WHO by sex: women < 12.0, men < 13.0 g/dL**; each person "
        "is judged against their own cutoff.\n"
        f"- **Each draw:** k people at the site (India or Italy) are picked at random and have a lab Hb. `shift` adds "
        "their mean error to every estimate; `shift and scale` fits a straight line from estimate to lab Hb on them. "
        "Everything is scored on the **rest** of the site, never on the k people used to fit. "
        f"{cfg['n_draws']} draws per k; numbers are the mean with the 2.5-97.5% range over draws (not a bootstrap).\n"
        "- **Comparisons:** no correction; a regressor trained on the k local people alone; the untrained redness rule.\n"
        "- A shift keeps everyone's order, so it cannot change the AUROC; only `shift and scale` can (it reverses the "
        "order when the fitted slope is negative).\n"
        "- **Italy can't be trusted on its own:** 21 of its 23 anemic patients are in one early photo batch (Stage 10).\n\n"
        + "\n".join(lines) + "\n"
    )
    (ROOT / "reports" / "stage18_hb_local.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
