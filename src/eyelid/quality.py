"""Stage 21: photo quality and glare, per location.

Mentor advice (2026-10-31): put a step in the data pipeline that screens out bad-quality photos. Every photo here
already comes as a hand cut-out, so quality is measured on the pixels inside it: glare (light reflected off the wet
surface, bright and colorless), dark and blown-out pixels, blur, and region size. Three questions, per location:
(a) does quality differ between anemic and healthy photos (a possible shortcut)? (b) does removing glare pixels before
the 29 color numbers help? (c) does a retake gate, dropping the worst photos at test time, help on the photos kept?

Run from the repo root: python -m src.eyelid.quality
"""
import cv2
import numpy as np
from scipy.stats import rankdata

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.common.metrics import auroc, bootstrap_ci
from src.common.splits import repeated_stratified
from src.eyelid.ceiling import fit_and_score, ladder_models, population_frames
from src.eyelid.color_features import color_features
from src.eyelid.report import fmt, markdown_table
from src.eyelid.roi import read_rgba
from src.eyelid.train_color import _short_commit
from src.ghana_sites import load_site
from src.nails import COLUMNS, METRICS

NAMES = {"cp_anemic": "Ghana eyelids", "eyes_defy_india": "India eyelids", "eyes_defy_italy": "Italy eyelids",
         "palm": "Ghana palms", "fingernails": "Ghana fingernails"}
MEASURES = ["glare", "dark", "blown_out", "blur", "size"]


def measure(path, cfg, alpha_threshold):
    """Quality numbers of one cut-out, its 29 color numbers, and the same 29 numbers with glare pixels left out."""
    rgba = np.asarray(read_rgba(path), dtype=np.float32) / 255.0
    inside = rgba[..., 3] >= alpha_threshold / 255.0
    rgb = rgba[..., :3]
    px = rgb[inside]
    v = px.max(1)
    s = np.where(v > 0, (v - px.min(1)) / np.maximum(v, 1e-6), 0.0)
    glare = (v >= cfg["glare"]["min_value"]) & (s <= cfg["glare"]["max_saturation"])
    k = cfg["blur_erode_px"]
    core = cv2.erode(inside.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    if core.sum() < cfg["min_pixels"]:  # thin region: measure blur on the whole region
        core = inside
    lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    q = {"glare": float(glare.mean()), "dark": float((v < cfg["dark_max_value"]).mean()),
         "blown_out": float((px >= cfg["blown_min"]).any(1).mean()), "blur": float(lap[core].var()),
         "size": float(inside.sum())}
    keep = ~glare if (~glare).sum() >= cfg["min_pixels"] else np.ones(len(px), bool)
    return q, color_features(px), color_features(px[keep])


def badness(q):
    """Within-location percentile rank of how bad each photo is on its worst measure (1 = worst photo)."""
    n = len(q["glare"])
    ranks = [rankdata(q[m]) / n for m in ("glare", "dark", "blown_out")] + [rankdata(-q["blur"]) / n]
    return np.max(ranks, axis=0)


def kept_minus_all(y, score, keep, n_boot, ci, seed):
    """AUROC on the kept photos minus AUROC on all photos, both inside the same bootstrap resample."""
    packed = np.column_stack([score, keep])

    def change(yy, ss):
        k = ss[:, 1] == 1
        return auroc(yy[k], ss[k, 0]) - auroc(yy, ss[:, 0])
    return bootstrap_ci(change, y, packed, n_boot, ci, seed=seed)


def main():
    base, cfg, color_cfg, probe_cfg = (load_config(n) for n in ("default", "quality", "color", "probe"))
    ceiling_cfg, debias_cfg, sites_cfg = load_config("ceiling"), load_config("debias"), load_config("ghana_sites")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci = boot["n_resamples"], boot["ci"]
    alpha, rule_col = color_cfg["roi"]["alpha_threshold"], color_cfg["rule_feature"]
    models = {n: make for n, (kind, make) in ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed).items()
              if kind == "color" and n != "dummy"}

    frames = population_frames(ceiling_cfg)
    sets = {k: (frames[k], [ROOT / p for p in frames[k].image_path]) for k in cfg["locations"] if k in frames}
    for site, folder in sites_cfg["sites"].items():
        if site in cfg["locations"]:
            photos, _ = load_site(folder, base["cp_anemic_near_duplicate"])
            sets[site] = (photos, list(photos.path))

    measure_rows, glare_rows, gate_rows = [], [], []
    for key, (df, paths) in sets.items():
        y = df.anemic.to_numpy()
        out = [measure(p, cfg, alpha) for p in paths]
        q = {m: np.array([o[0][m] for o in out]) for m in MEASURES}
        X = {"as cut out": np.array([list(o[1].values()) for o in out]),
             "glare removed": np.array([list(o[2].values()) for o in out])}
        col = list(out[0][1]).index(rule_col)
        rule = {how: -x[:, col] for how, x in X.items()}
        folds = repeated_stratified(y, cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], seed)
        s = {(name, how): fit_and_score(make, x, y, folds)[0] for name, make in models.items() for how, x in X.items()}
        print(f"done: {key}", flush=True)

        test_sets = {NAMES[key]: np.ones(len(y), bool)}
        if key == "eyes_defy_italy":
            folder = df.id.str.split("_").str[1].astype(int).to_numpy()
            test_sets["Italy early batch"] = folder <= debias_cfg["italy_early_batch_last_folder"]
        bad = badness(q)
        for subset, m in test_sets.items():
            yy = y[m]
            head = {"test set": subset, "n": int(m.sum()), "anemic": f"{yy.mean():.1%}"}
            for name in MEASURES:
                measure_rows.append(head | {"measure": name, "median": float(np.median(q[name][m])),
                                            **summarize(yy, q[name][m], n_boot, ci, seed, metrics=METRICS)})
            for how in X:
                glare_rows.append(head | {"model": "rule_low_redness", "features": how,
                                          **summarize(yy, rule[how][m], n_boot, ci, seed, metrics=METRICS)}
                                  | ({"vs_cut_out": auroc_change(yy, rule[how][m], rule["as cut out"][m], n_boot, ci,
                                                                 seed)} if how != "as cut out" else {}))
                for name in models:
                    sc = s[(name, how)][:, m]
                    glare_rows.append(head | {"model": name, "features": how,
                                              **summarize(yy, sc, n_boot, ci, seed, metrics=METRICS)}
                                      | ({"vs_cut_out": auroc_change(yy, sc, s[(name, "as cut out")][:, m], n_boot,
                                                                     ci, seed)} if how != "as cut out" else {}))
            for share in cfg["reject_share"]:
                keep = bad[m] <= np.quantile(bad[m], 1 - share)
                for name, score in {"rule_low_redness": rule["as cut out"][m],
                                    **{n: s[(n, "as cut out")][:, m].mean(0) for n in models}}.items():
                    gate_rows.append(head | {"reject": f"{share:.0%}", "rejected": int((~keep).sum()),
                                             "anemic among rejected": f"{yy[~keep].mean():.1%}", "model": name,
                                             "auroc_all": bootstrap_ci(auroc, yy, score, n_boot, ci, seed=seed),
                                             "auroc_kept": bootstrap_ci(auroc, yy[keep], score[keep], n_boot, ci,
                                                                        seed=seed),
                                             "change": kept_minus_all(yy, score, keep, n_boot, ci, seed)})

    path = save_results("stage21_quality", {"measures": measure_rows, "glare": glare_rows, "gate": gate_rows},
                        {"default": base, "quality": cfg, "color": color_cfg, "ceiling": ceiling_cfg,
                         "debias": debias_cfg, "ghana_sites": sites_cfg})

    warn = [f"{r['test set']}: {r['measure']} {fmt(r['auroc'])}" for r in measure_rows
            if r["auroc"]["lo"] > 0.5 or r["auroc"]["hi"] < 0.5]
    glare_help = [f"{r['test set']}: {r['model']} {fmt(r['vs_cut_out'])}" for r in glare_rows
                  if "vs_cut_out" in r and r["vs_cut_out"]["lo"] > 0]
    gate_help = [f"{r['test set']}, reject {r['reject']}: {r['model']} {fmt(r['change'])}" for r in gate_rows
                 if r["change"]["lo"] > 0]
    med = {r["test set"]: r["median"] for r in measure_rows if r["measure"] == "glare"}
    glare_lines = "\n".join(
        f"| {t} | {med[t]:.1%} | " + " | ".join(
            fmt(next(r["vs_cut_out"] for r in glare_rows if r["test set"] == t and r["model"] == n
                     and "vs_cut_out" in r)) for n in ["rule_low_redness", *models]) + " |" for t in med)
    gate_lines = "\n".join(
        f"| {r['test set']} | {r['reject']} ({r['rejected']}) | {r['anemic']} | {r['anemic among rejected']} | "
        f"{r['model']} | {fmt(r['auroc_all'])} | {fmt(r['auroc_kept'])} | {fmt(r['change'])} |"
        for r in gate_rows if r["model"] in ("rule_low_redness", "tabpfn"))
    g = cfg["glare"]
    report = (
        "# Stage 21: photo quality and glare, per location\n\n"
        f"Generated by `python -m src.eyelid.quality` at commit `{_short_commit(path)}`; settings in "
        "`configs/quality.yaml`. Mentor advice (2026-10-31): screen out bad-quality photos in the data pipeline.\n\n"
        "- **Photos and labels (one per dataset, never mixed):** Ghana eyelids, the 388 CP-AnemiC photos with no copy "
        f"(children, **Hb < {base['anemia_cutoff_g_dl']['child_6_59_months']} g/dL**); India and Italy eyelids "
        "(Eyes-Defy adults, **WHO by sex: women < 12.0, men < 13.0 g/dL**, pregnancy not recorded); Ghana palms and "
        "fingernails (Asare et al. 2022, **the dataset's own label**, no Hb, cutoff not stated).\n"
        f"- **Quality numbers** from the pixels inside each cut-out: glare = share of pixels with HSV value >= "
        f"{g['min_value']} and saturation <= {g['max_saturation']} (light reflected off the surface); dark = value < "
        f"{cfg['dark_max_value']}; blown-out = any channel >= {cfg['blown_min']}; blur = variance of the Laplacian "
        f"inside the region (eroded {cfg['blur_erode_px']} px; low = blurry); size = pixels. Thresholds fixed before "
        "running. On palms and fingernails near-white can also be the nail tip or pale skin, so there (b) means "
        "near-white pixels removed, not only glare.\n"
        f"- **Split:** {cfg['cv']['n_splits']}-fold CV x {cfg['cv']['n_repeats']} repeats by photo, stratified by "
        f"anemia. Bootstrap 95% CIs, {n_boot} resamples; changes are paired (same resampled photos).\n"
        "- **Retake gate:** each photo's worst percentile rank on glare, dark, blown-out (high = bad) or blur (low = "
        "bad) inside its location, no labels used; the worst share is dropped at test time. Models are trained on "
        "every photo.\n\n"
        "## Answers (fixed before running)\n\n"
        "- **(a) Quality numbers that differ between anemic and healthy photos (AUROC CI excludes 0.5):** "
        + ("; ".join(warn) or "none") + ".\n"
        "- **(b) Removing glare helps (AUROC change CI above 0):** " + ("; ".join(glare_help) or "nowhere") + ".\n"
        "- **(c) The retake gate helps (kept minus all, CI above 0):** " + ("; ".join(gate_help) or "nowhere") + ".\n\n"
        "## (b) Glare removed: AUROC change against the plain cut-out, same model\n\n"
        "| test set | median glare share | " + " | ".join(["rule_low_redness", *models]) + " |\n|"
        + "---|" * (3 + len(models)) + "\n" + glare_lines + "\n\n"
        "## (c) Retake gate (rule and TabPFN; every model in the JSON)\n\n"
        "| test set | rejected | anemic, all | anemic, rejected | model | AUROC, all | AUROC, kept | change |\n"
        "|---|---|---|---|---|---|---|---|\n" + gate_lines + "\n\n"
        "## (a) Each quality number on its own as a score (0.5 = no difference between anemic and healthy)\n\n"
        + markdown_table([r | {"median": f"{r['median']:.4g}"} for r in measure_rows],
                         ["test set", "n", "anemic", "measure", "median"], COLUMNS) + "\n\n"
        "## All glare results\n\n"
        + markdown_table(glare_rows, ["test set", "n", "anemic", "model", "features"],
                         COLUMNS | {"vs_cut_out": "AUROC change vs plain cut-out"}) + "\n"
    )
    (ROOT / "reports" / "stage21_quality.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
