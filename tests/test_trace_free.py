import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eyelid.handling_check import jump_test
from src.eyelid.trace_free import WITHIN, centre_by_group, feature_jumps, fit_within, held_out_hb, keeps_signal

CFG = {"window_g_dl": 1.0, "n_sims": 500, "p_max": 0.025, "min_side": 8}
ridge = lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def toy(seed=0, n=400):
    """x0 follows Hb steadily (pallor); x1 is a folder switch (trace) with no Hb in it."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-4, 3, n)
    y = (z < 0).astype(int)
    return z, y, np.column_stack([-z + rng.normal(0, 3.0, n), 1.0 * y + rng.normal(0, 0.5, n)])


def test_centre_by_group_leaves_no_group_mean_and_copies():
    z, y, X = toy()
    Xc, tc = centre_by_group(X, z, y)
    for g in (0, 1):
        assert np.allclose(Xc[y == g].mean(0), 0) and abs(tc[y == g].mean()) < 1e-9
    assert not np.allclose(X, Xc)  # the input is untouched


def test_within_folder_ridge_ignores_the_folder_switch_and_does_not_jump():
    z, y, X = toy()
    usual = ridge().fit(X, z)[-1].coef_
    within = fit_within(ridge, X, z, y)[-1].coef_
    assert abs(usual[1]) > 0.5 and abs(within[1]) < 0.15 and within[0] < 0
    rng = np.random.default_rng(1)
    s_usual, s_within = -ridge().fit(X, z).predict(X), -fit_within(ridge, X, z, y).predict(X)
    assert jump_test(z, s_usual, CFG, rng, 100, 0.95, 0)["answer"] == "jumps at the cutoff"
    assert jump_test(z, s_within, CFG, rng, 100, 0.95, 0)["answer"] == "no jump beyond a steady signal"


def test_test_labels_never_reach_the_estimate():
    z, y, X = toy()
    tr, te = np.arange(300), np.arange(300, 400)
    flipped = y.copy()
    flipped[te] = 1 - flipped[te]
    full = [(0, tr, te), (0, te, tr)]  # every row gets a test fold, as held_out_hb requires
    a, b = held_out_hb({WITHIN: ridge}, X, z, y, full), held_out_hb({WITHIN: ridge}, X, z, flipped, full)
    assert np.allclose(a[WITHIN][0, te], b[WITHIN][0, te])  # fit on the same rows; only the test rows' labels changed


def test_feature_jumps_turns_each_number_and_flags_the_switch():
    z, y, X = toy()
    rows = feature_jumps(z, X, ["pallor", "switch"], CFG, np.random.default_rng(2), 100, 0.95, 0)
    assert [r["higher_means_anemic"] for r in rows] == [True, True] and all(r["auroc"] >= 0.5 for r in rows)
    assert [r["jumps"] for r in rows] == [False, True]
    flipped = feature_jumps(z, -X[:, :1], ["pale"], CFG, np.random.default_rng(2), 100, 0.95, 0)
    assert flipped[0]["higher_means_anemic"] is False and flipped[0]["auroc"] >= 0.5


def test_keeps_signal():
    cfg = {"keep_drop": 0.05}
    assert keeps_signal({"estimate": -0.02, "lo": -0.08, "hi": 0.03}, cfg)
    assert not keeps_signal({"estimate": -0.07, "lo": -0.12, "hi": 0.01}, cfg)
    assert not keeps_signal({"estimate": -0.03, "lo": -0.06, "hi": -0.01}, cfg)
