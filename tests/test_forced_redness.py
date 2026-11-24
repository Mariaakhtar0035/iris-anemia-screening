import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.eyelid.forced_redness import constraints


def test_forced_model_never_raises_risk_with_more_redness():
    assert constraints(["S_mean", "a_mean", "R_over_G"], ["a_mean", "R_over_G"]) == [0, -1, -1]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (np.sin(3 * X[:, 1]) + rng.normal(0, 0.3, 400) > 0).astype(int)  # a wiggly, not monotone, pattern
    m = HistGradientBoostingClassifier(monotonic_cst=constraints(["free", "a_mean"], ["a_mean"]), random_state=0).fit(X, y)
    grid = np.column_stack([np.zeros(50), np.linspace(-3, 3, 50)])
    assert (np.diff(m.predict_proba(grid)[:, 1]) <= 1e-12).all()


def test_three_cue_constraints_point_the_pallor_way():
    cols = ["a_mean", "S_mean", "L_mean", "H_mean"]
    assert constraints(cols, ["a_mean", "S_mean"], ["L_mean"]) == [-1, -1, 1, 0]  # brighter can only mean more anemic
