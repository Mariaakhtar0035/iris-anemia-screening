import numpy as np

from src.common.metrics import auroc
from src.eyelid.handling_check import RULE
from src.eyelid.trace_split import answers, normal_scores, planted, recovery, split, step_at_cutoff, trace_free


def toy(n=4000, step=0.0, seed=0):
    """z = Hb minus cutoff; a smooth pallor score (-z + noise) plus `step` for every anemic person."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-4, 3, n)
    smooth = -0.4 * z + rng.normal(0, 1, n)
    return z, smooth, smooth + step * (z < 0)


def test_normal_scores_keep_order_and_auroc():
    s = np.array([0.1, 0.9, 0.5, 0.5, 0.3])
    q = normal_scores(s)
    assert np.allclose(np.argsort(q, kind="stable"), np.argsort(s, kind="stable")) and q[2] == q[3]
    y = np.array([0, 1, 1, 0, 0])
    assert auroc(y, q) == auroc(y, s)


def test_step_is_found_when_planted_and_near_zero_when_smooth():
    z, smooth, stepped = toy(step=1.5)
    assert abs(step_at_cutoff(z, smooth, 2.0)) < 0.15
    assert abs(step_at_cutoff(z, stepped, 2.0) - 1.5) < 0.15


def test_trace_free_auroc_matches_the_smooth_score():
    z, smooth, stepped = toy(step=1.5)
    y = (z < 0).astype(int)
    assert auroc(y, stepped) > auroc(y, smooth) + 0.1
    assert abs(auroc(y, trace_free(z, stepped, 2.0)) - auroc(y, smooth)) < 0.02


def test_split_on_a_smooth_score_keeps_its_auroc_and_reports_share():
    z, smooth, _ = toy(n=600)
    r = split(z, smooth, 2.0, n_boot=200, ci=0.95, seed=0)
    assert r["step"]["lo"] < 0 < r["step"]["hi"] and abs(r["change"]["estimate"]) < 0.03 and "share" in r
    assert set(split(z, np.round(smooth), 2.0, n_boot=50, ci=0.95, seed=0)) == {"auroc"}  # mostly tied: not split


def test_recovery_after_a_plant_on_the_normal_scale():
    # re-ranking after the plant bends the step a little: the toy's average bias is within 0.025 (5 seeds, n = 1500)
    z, smooth, _ = toy(seed=1)
    r = recovery(z, normal_scores(smooth) + 1.0 * (z < 0), smooth, 2.0, n_boot=100, ci=0.95, seed=0)
    assert r["gain"]["estimate"] > 0.1 and abs(r["recovery"]["estimate"]) < 0.04


def test_planted_edits_only_anemic_rows():
    X, Xe, y = np.zeros((4, 2)), np.ones((4, 2)), np.array([1, 0, 1, 0])
    assert (planted(X, Xe, y) == y[:, None]).all()


def test_answers_trust_rule():
    c = lambda e, lo, hi: {"estimate": e, "lo": lo, "hi": hi}
    cfg = {"clean": ["a", "b", "c"], "keep_drop": 0.05, "plant_min_gain": 0.05, "recover_tol": 0.05,
           "min_clean_kept": 2, "min_removed_share": 0.67}
    ok, bad = {"change": c(-0.01, -0.05, 0.03)}, {"change": c(-0.20, -0.30, -0.10)}
    common = {"step": c(0.0, -0.1, 0.1), "trace_free_auroc": c(0.6, 0.55, 0.65)}
    A = {"a": {"m1": ok | common, RULE: bad | common}, "b": {"m1": ok | common, RULE: bad | common},
         "c": {"m1": bad | common, RULE: ok | common}, "ghana_x": {"m1": ok | common, RULE: ok | common}}
    B = [{"model": "m1", "gain": c(0.10, 0.05, 0.15), "recovery": c(0.02, -0.03, 0.06), "jump_test": {"jumps": True}},
         {"model": "m1", "gain": c(0.01, -0.02, 0.04), "recovery": c(0.30, 0.2, 0.4), "jump_test": {"jumps": False}},
         {"model": RULE, "gain": c(0.10, 0.05, 0.15), "recovery": c(0.02, -0.03, 0.06), "jump_test": {"jumps": False}}]
    ans = answers({"split": A, "planted": B}, cfg, ["m1", RULE])
    assert ans["trusted"] == {"m1": True, RULE: False}  # the rule loses the clean signal on 2 of 3
    assert ans["plants_counting"] == {"m1": 1, RULE: 1} and ans["jump_test_flags_plants"] == [1, 2]
    assert list(ans["trace_free_auroc"]["ghana_x"]) == ["m1"]
