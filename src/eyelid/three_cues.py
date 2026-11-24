"""Stage 43: hue and yellow-blue only, and gradient boosting forced to read redness, saturation and brightness the pallor
way (less red, less saturated, brighter = more anemic), on the Ghana eyelids, nails and palms, through Stage 35's
trace checks. Settings and the answers, fixed before running, in configs/three_cues.yaml.

Run from the repo root: python -m src.eyelid.three_cues
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.common.io import load_config
from src.eyelid.forced_redness import constraints
from src.eyelid.handling_check import NAMES
from src.eyelid.redness_check import CV, LABELS, run, yn
from src.eyelid.report import fmt

FORCED = "forced_three_cues"
STAGE43 = {"result": "stage43_three_cues", "module": "src.eyelid.three_cues", "config": "three_cues",
           "title": "Stage 43: hue and yellow-blue only, and a model forced to read all three paleness cues (Ghana)",
           "models": "Logistic regression, gradient boosting (free, and forced: less red, less saturated or brighter "
                     "can only mean more anemic) and TabPFN v2 on each input set"}


def answers(cfg, out):
    """Markdown for the two answers fixed in the config."""
    trained = [m for m in cfg["models"] if m != FORCED]  # on hue_yellow_blue the forced model is the free one
    weird, forced = [], []
    for n, d in out["datasets"].items():
        s = d["splits"][CV]["sets"]
        au = lambda k, m: s[k][m]["scores"]["auroc"]
        for m in trained:
            h, p = au("hue_yellow_blue", m), au("paleness", m)
            is_weird = h["estimate"] >= p["estimate"] - cfg["weird_margin"] and s["hue_yellow_blue"][m]["vs_rule"]["lo"] > 0
            weird.append(f"| {NAMES[n]} | {LABELS[m]} | {fmt(h)} | {fmt(p)} | "
                         f"{fmt(d['splits'][CV]['set_pairs']['hue_yellow_blue - paleness'][m])} | "
                         f"{fmt(s['hue_yellow_blue'][m]['vs_rule'])} | {yn(not s['hue_yellow_blue'][m]['no_trace'])} | "
                         f"{yn(is_weird)} |")
        cell = lambda k: f"{fmt(au(k, FORCED))} ({'no trace' if s[k][FORCED]['no_trace'] else 'trace'})"
        forced.append(f"| {NAMES[n]} | " + " | ".join(cell(k) for k in ("three_cue_means", "three_cues", "paleness"))
                      + f" | {fmt(s['three_cues'][FORCED]['vs_rule'])} | "
                      f"{fmt(s['three_cues']['hist_gradient_boosting']['vs'][FORCED])} | {d['reported'][FORCED] or 'none'} |")
    return (
        "## Answers\n\n"
        f"**1. Hue and yellow-blue only.** Weird = scores at least the paleness numbers' AUROC minus "
        f"{cfg['weird_margin']} and beats the rule (Stage 37's rule).\n\n"
        "| dataset | model | hue + yellow-blue (6) | paleness (14) | difference | hue + yellow-blue minus rule | trace? | "
        "weird? |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(weird) + "\n\n"
        "**2. Forced three-cue model** (less red, less saturated or brighter can only mean more anemic). Reportable = "
        "some input set shows no trace and beats the rule (Stage 36's rule).\n\n"
        "| dataset | three cue means (3) | three cues (9) | paleness (9 forced, 5 free) | three cues minus rule | "
        "free minus forced, three cues | reportable set |\n|---|---|---|---|---|---|---|\n" + "\n".join(forced) + "\n\n")


def main():
    cfg, hgb, seed = load_config("three_cues"), load_config("ceiling")["hist_gradient_boosting"], \
        load_config("default")["seed"]

    def makers_for(ladder, cols):
        cst = constraints(cols, cfg["falls_with"], cfg["rises_with"])
        forced = lambda: HistGradientBoostingClassifier(**hgb, monotonic_cst=cst, random_state=seed)
        return {m: forced if m == FORCED else ladder[m][1] for m in cfg["models"]}

    run(STAGE43 | {"answers": answers}, makers_for)


if __name__ == "__main__":
    main()
