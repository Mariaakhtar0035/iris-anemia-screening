"""Stage 36: gradient boosting where less red can only mean more anemic (Ghana), through Stage 35's trace checks.

Settings and the answers, fixed before running, in configs/forced_redness.yaml.

Run from the repo root: python -m src.eyelid.forced_redness
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from src.common.io import load_config
from src.eyelid.redness_check import run

FORCED = "forced_gradient_boosting"
STAGE36 = {"result": "stage36_forced_redness", "module": "src.eyelid.forced_redness", "config": "forced_redness",
           "title": "Stage 36: a model where less red can only mean more anemic (Ghana), through every trace check",
           "models": "Gradient boosting, free and with P(anemic) only allowed to fall as a redness number rises, on "
                     "each input set"}


def constraints(cols, forced, rises=()):
    """-1 (P(anemic) can't rise with the number) for the forced numbers, +1 (can't fall) for `rises`, 0 (free) for
    the rest."""
    return [-1 if c in forced else 1 if c in rises else 0 for c in cols]


def main():
    cfg, hgb, seed = load_config("forced_redness"), load_config("ceiling")["hist_gradient_boosting"], \
        load_config("default")["seed"]

    def makers_for(ladder, cols):
        cst = constraints(cols, cfg["less_red_more_anemic"])
        return {"hist_gradient_boosting": ladder["hist_gradient_boosting"][1],
                FORCED: lambda: HistGradientBoostingClassifier(**hgb, monotonic_cst=cst, random_state=seed)}

    run(STAGE36, makers_for)


if __name__ == "__main__":
    main()
