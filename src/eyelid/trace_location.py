"""Stage 34: where in the photo is the Ghana folder trace: at the edge of the cut-out, or all through it?

A: the 29 color numbers on the inner half of each cut-out (core) and on the outer ring, models retrained on each, and
Stage 30's jump test on every score. B: Stage 30 A2's handling numbers (no color), each through the jump test. Settings
and the answers, fixed before running, in configs/trace_location.yaml.

Run from the repo root: python -m src.eyelid.trace_location
"""
import cv2
import numpy as np

from src.common.evaluate import auroc_change, summarize
from src.common.io import ROOT, load_config, save_results
from src.eyelid.ceiling import ladder_models
from src.eyelid.color_features import color_features
from src.eyelid.handling_check import NAMES, RULE, datasets, folds_for, handling_numbers, held_out, jump_test
from src.eyelid.report import fmt
from src.eyelid.roi import read_rgba
from src.eyelid.tabpfn_audit import feature_names
from src.eyelid.trace_free import feature_jumps, keeps_signal
from src.eyelid.train_color import _short_commit

PARTS = ("whole", "core", "ring")
LABELS = {"dummy": "dummy", RULE: "rule (minus mean a*)", "logistic_regression": "logistic regression",
          "tabpfn": "TabPFN"}


def core_and_ring(rgba, alpha, core_share):
    """RGB (0-1) of the core_share of cut-out pixels farthest from the cut-out's edge, and of the rest (the ring)."""
    inside = rgba[..., 3] >= alpha / 255.0
    # padded so a cut-out touching the image border still counts the border as its edge
    dist = cv2.distanceTransform(np.pad(inside, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1][inside]
    px = rgba[..., :3][inside]
    order = np.argsort(dist, kind="stable")
    k = len(px) - int(round(core_share * len(px)))
    return px[order[k:]], px[order[:k]]


def part_numbers(paths, alpha, core_share):
    """The 29 numbers on the whole cut-out, its core and its ring: {part: (n, 29)}. One photo in memory at a time."""
    out = {p: [] for p in PARTS}
    for path in paths:
        core, ring = core_and_ring(np.asarray(read_rgba(path), dtype=np.float32) / 255.0, alpha, core_share)
        for p, px in zip(PARTS, (np.vstack([core, ring]), core, ring)):
            out[p].append(list(color_features(px).values()))
    return {p: np.array(v) for p, v in out.items()}


def answers(out, cfg, models, p_max):
    D = out["datasets"]
    jumps = lambda n, p: bool(D[n]["parts"][p]["tabpfn"]["jump"].get("jumps"))
    where = {}
    for n in (n for n in D if n.startswith("ghana_")):
        ring, core = jumps(n, "ring"), jumps(n, "core")
        where[n] = ("all through the cut-out" if ring and core else "at the cut-out edge" if ring else
                    "in the core, not the edge" if core else "no jump in either part")
    return {"where": where,
            "core_keeps_clean": {n: {m: keeps_signal(D[n]["parts"]["core"][m]["change"], cfg) for m in models}
                                 for n in cfg["clean"] if n in D},
            "handling_numbers_that_jump": {n: [r["feature"] for r in d["handling_jumps"] if r["jumps"]]
                                           for n, d in D.items()}}


def main():
    base, cfg, color_cfg = load_config("default"), load_config("trace_location"), load_config("color")
    jump_cfg, probe_cfg, ceiling_cfg = load_config("handling_check"), load_config("probe"), load_config("ceiling")
    quality_cfg = load_config("quality")
    seed, boot = base["seed"], base["bootstrap"]
    n_boot, ci, alpha = boot["n_resamples"], boot["ci"], color_cfg["roi"]["alpha_threshold"]
    configs = {"default": base, "trace_location": cfg, "handling_check": jump_cfg, "color": color_cfg,
               "ceiling": ceiling_cfg, "quality": quality_cfg}
    rng = np.random.default_rng(seed)
    ladder = ladder_models(color_cfg, probe_cfg, ceiling_cfg, seed)
    makers = {m: ladder[m][1] for m in cfg["models"]}
    models = [m for m in (*cfg["models"], RULE) if m != "dummy"]
    a_col = feature_names().index(color_cfg["rule_feature"])
    out = {"datasets": {}}
    for name, d in datasets(cfg, base, color_cfg, seed).items():
        y, z = d["y"], d["hb"] - d["cut"]
        parts = part_numbers(d["paths"], alpha, cfg["core_share"])
        assert np.allclose(parts["whole"], d["X"], rtol=1e-4, atol=1e-6), f"{name}: parts don't give the stored numbers"
        folds = folds_for(y, d["groups"], cfg["cv"], seed)
        scores = {p: held_out(makers, X, y, folds)["as is"] | {RULE: -X[:, a_col]} for p, X in parts.items()}
        res = {"n": len(y), "anemic": int(y.sum()), "parts": {}}
        for p in PARTS:
            res["parts"][p] = {}
            for m, s in scores[p].items():
                r = {"scores": summarize(y, s, n_boot, ci, seed)}
                if m != "dummy":
                    r["jump"] = jump_test(z, np.atleast_2d(s).mean(0), jump_cfg, rng, n_boot, ci, seed)
                if p != "whole":
                    r["change"] = auroc_change(y, s, scores["whole"][m], n_boot, ci, seed)
                res["parts"][p][m] = r
        H = handling_numbers(d["paths"], quality_cfg, alpha)
        # the jump test needs each number to vary inside both label groups (rank correlation inside each)
        keep = [c for c in H.columns if all(H.loc[y == g, c].nunique() > 1 for g in (0, 1))]
        res["handling_constant_in_a_group"] = [c for c in H.columns if c not in keep]
        res["handling_jumps"] = feature_jumps(z, H[keep].to_numpy(float), keep, jump_cfg, rng, n_boot, ci, seed)
        out["datasets"][name] = res
        print(f"done {name}: TabPFN " + ", ".join(
            f"{p} {res['parts'][p]['tabpfn']['scores']['auroc']['estimate']:.3f} "
            f"({res['parts'][p]['tabpfn']['jump'].get('answer')})" for p in PARTS), flush=True)
        save_results("stage34_trace_location", out, configs)
    out["answers"] = answers(out, cfg, models, jump_cfg["p_max"])
    path = save_results("stage34_trace_location", out, configs)
    write_report(path, cfg, out, jump_cfg["p_max"])


def write_report(path, cfg, out, p_max):
    D, ans = out["datasets"], out["answers"]
    f = lambda r, k: fmt(r[k]) if k in r else "-"

    def row(n, p, m, r):
        j = r.get("jump", {})
        near = (f"{fmt(j['auroc_near'])} | {j['reference']['auroc_near_median']:.3f} | {j['p']:.4f}"
                if "auroc_near" in j else "- | - | -")
        return (f"| {NAMES[n]} | {p} | {LABELS[m]} | {fmt(r['scores']['auroc'])} | {fmt(r['scores']['pr_auc'])} | "
                f"{fmt(r['scores']['sens_at_90_spec'])} | {f(r, 'change')} | {near} | {j.get('answer', '-')} |")

    a_rows = "\n".join(row(n, p, m, r) for n, d in D.items() for p in PARTS for m, r in d["parts"][p].items())
    b_rows = "\n".join(
        f"| {NAMES[n]} | {r['feature']} | {'higher' if r['higher_means_anemic'] else 'lower'} | {r['auroc']:.3f} | "
        + (f"{fmt(r['auroc_near'])} | {r['reference_near_median']:.3f} | {r['p']:.4f}" if r["p"] is not None
           else "- | - | -") + f" | {'**yes**' if r['jumps'] else 'no'} |" for n, d in D.items() for r in d["handling_jumps"])
    where = "\n".join(f"| {NAMES[n]} | **{w}** |" for n, w in ans["where"].items())
    constant = "; ".join(f"{NAMES[n]}: {', '.join(d['handling_constant_in_a_group'])}" for n, d in D.items()
                         if d["handling_constant_in_a_group"]) or "none"
    kept = "\n".join(f"| {NAMES[n]} | {LABELS[m]} | {'yes' if v else 'no'} |"
                     for n, d in ans["core_keeps_clean"].items() for m, v in d.items())
    report = (
        "# Stage 34: where in the photo is the Ghana folder trace?\n\n"
        f"Generated by `python -m src.eyelid.trace_location` at commit `{_short_commit(path)}`; settings and the answers "
        "(fixed before running) in `configs/trace_location.yaml`.\n\n"
        "- **Cutoffs, one per dataset:** Ghana eyelids, nails, palms: children 6-59 months, **Hb < 11.0 g/dL**. India "
        "and Italy eyelids (Eyes-Defy adults): **WHO by sex, women < 12.0, men < 13.0 g/dL**, pregnancy not recorded. "
        "Moscow nails are left out (boxes in a full photo, no cut-out edge).\n"
        f"- **Core** = the {cfg['core_share']:.0%} of each cut-out's pixels farthest from its edge; **ring** = the rest. "
        "The same 29 color numbers on each part; models retrained on each; 5-fold CV x 5 repeats by photo / child; "
        "bootstrap 95% CIs, 2000 resamples. Jump test as Stage 30 (near the cutoff = within 1 g/dL).\n\n"
        "## Where is the trace? (TabPFN on the core and on the ring)\n\n| dataset | answer |\n|---|---|\n" + where + "\n\n"
        "## Does the core keep the clean signal? (AUROC change vs the whole cut-out > "
        f"-{cfg['keep_drop']}, CI upper end > 0)\n\n| dataset | model | kept |\n|---|---|---|\n" + kept + "\n\n"
        "## A. Whole cut-out, core and ring\n\n"
        "| dataset | part | model | AUROC | PR-AUC | sens @ 90% spec | change vs whole | near-cutoff AUROC | steady "
        "reference (median) | p | jump answer |\n|---|---|---|---|---|---|---|---|---|---|---|\n" + a_rows + "\n\n"
        f"## B. Handling numbers (no color), each alone; jumps if p < {p_max} / number of handling numbers\n\n"
        "| dataset | number | anemic is | AUROC | near-cutoff AUROC | steady reference (median) | p | jumps |\n"
        "|---|---|---|---|---|---|---|---|\n" + b_rows + "\n\n"
        f"Not tested (the same value for everyone in one label group): {constant}.\n")
    (ROOT / "reports" / "stage34_trace_location.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
