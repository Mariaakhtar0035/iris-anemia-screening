import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.common.io import load_config
from src.eyelid.handling_check import auroc_rows, edit, edit_list, fragile, jump_test, near_cutoff, spearman_rows

CFG = {"window_g_dl": 1.0, "n_sims": 500, "p_max": 0.025, "min_side": 8}


def test_rank_helpers_match_sklearn_and_scipy():
    rng = np.random.default_rng(0)
    y, s, h = rng.integers(0, 2, 50), rng.normal(size=(3, 50)), rng.normal(size=50)
    assert np.allclose(auroc_rows(y, s), [roc_auc_score(y, r) for r in s])
    assert np.allclose(spearman_rows(s, h), [spearmanr(r, h)[0] for r in s])
    assert near_cutoff(np.array([-1.5, -1.0, -0.2, 0.0, 0.9, 1.0]), 1.0).tolist() == [False, True, True, True, True, False]


def test_a_steady_score_passes_and_a_switch_jumps():
    rng = np.random.default_rng(1)
    z = rng.uniform(-4, 3, 400)
    steady = jump_test(z, -z + rng.normal(0, 1.0, 400), CFG, rng, 200, 0.95, 0)
    switch = jump_test(z, (z < 0) + rng.normal(0, 0.3, 400), CFG, rng, 200, 0.95, 0)
    assert steady["answer"] == "no jump beyond a steady signal" and steady["rho_anemic"]["hi"] < 0
    assert switch["answer"] == "jumps at the cutoff" and abs(switch["rho_anemic"]["estimate"]) < 0.2
    few = jump_test(np.array([-0.5] * 3 + [2.0] * 50 + [0.5] * 3 + [-3.0] * 50), rng.normal(size=106), CFG, rng, 50, 0.95, 0)
    assert few["answer"] == "too few people near the cutoff"


def test_edits():
    rgb = np.array([[0.6, 0.3, 0.2], [0.9, 0.8, 0.7]])
    assert np.allclose(edit(rgb, "brightness", 1.0), rgb) and np.allclose(edit(rgb, "gamma", 1.0), rgb)
    assert np.allclose(edit(rgb, "saturation", 1.0), rgb) and np.ptp(edit(rgb, "saturation", 0.0), axis=1).max() < 1e-9
    warm = edit(rgb, "white_balance", (1.1, 0.9))
    assert (warm[:, 0] > rgb[:, 0]).all() and (warm[:, 2] < rgb[:, 2]).all() and np.allclose(warm[:, 1], rgb[:, 1])
    assert edit(rgb, "brightness", 2.0).max() <= 1.0 and (edit(rgb, "gamma", 0.8) > rgb).all()
    assert len(edit_list(load_config("handling_check"), (0.9, 0.8, 0.8))) == 8


def test_fragile_needs_tabpfn_to_drop_while_the_rule_holds():
    c = lambda e, lo, hi: {"estimate": e, "lo": lo, "hi": hi}
    rows = [{"edit": "a", "model": "tabpfn", "change": c(-0.10, -0.15, -0.05)},
            {"edit": "a", "model": "rule_low_redness", "change": c(-0.01, -0.02, 0.0)},
            {"edit": "b", "model": "tabpfn", "change": c(-0.10, -0.15, -0.05)},
            {"edit": "b", "model": "rule_low_redness", "change": c(-0.05, -0.08, -0.02)}]
    assert fragile(rows, {"fragile_drop": 0.05, "rule_stable_drop": 0.02}) == ["a"]
