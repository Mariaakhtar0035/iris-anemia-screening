import numpy as np

from src.common.evaluate import auroc_change


def test_late_fusion_of_two_noisy_views_gains_over_one():
    # the comparison Stage 28 reports: the mean of two scores against the first, paired on the same people
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 150)
    a, b = y + rng.normal(0, 1.5, 300), y + rng.normal(0, 1.5, 300)
    gain = auroc_change(y, (a + b) / 2, a, 500, 0.95, 0)
    assert gain["estimate"] > 0 and gain["lo"] > 0
    same = auroc_change(y, a, a, 200, 0.95, 0)
    assert same["estimate"] == 0 and same["lo"] == same["hi"] == 0
