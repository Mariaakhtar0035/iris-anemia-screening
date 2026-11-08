import numpy as np

from src.common.metrics import auroc
from src.ghana_checks import auroc_gap, rate_gap, shared_threshold, within_label_corr, wrong_matches


def test_shared_threshold_keeps_specificity_and_gaps_compare_groups():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 100)
    s = y + rng.normal(0, 1, 200)
    t = shared_threshold(y, s, 0.9)
    assert (s[y == 0] >= t).mean() <= 0.1 + 1e-9
    g = np.r_[np.zeros(100), np.ones(100)]
    packed = np.column_stack([s, g])
    assert np.isclose(auroc_gap(y, packed, 0), auroc(y[100:], s[100:]) - auroc(y[:100], s[:100]))
    flag = s >= t
    assert np.isclose(rate_gap(1, 0.9)(y, packed, 0), flag[100:][y[100:] == 1].mean() - flag[:100][y[:100] == 1].mean())


def test_within_label_corr_is_one_for_the_same_score_and_near_zero_for_noise():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 200)
    a = rng.normal(size=400)
    assert np.isclose(within_label_corr(y, a, a), 1.0)
    assert abs(within_label_corr(y, a, rng.normal(size=400))) < 0.15


def test_wrong_matches_changes_nothing_at_share_zero():
    rng = np.random.default_rng(2)
    y = np.array([0, 1] * 50)
    e, n, p = (y + rng.normal(0, 1, 100) for _ in range(3))
    real = auroc(y, (e + n + p) / 3)
    assert np.allclose(wrong_matches(y, e, n, p, 0.0, 5, rng), real)
    assert not np.allclose(wrong_matches(y, e, n, p, 1.0, 5, rng), real)
