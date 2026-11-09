import numpy as np
import pytest

from src.eyelid.severity import band_index, flagged_share, threshold


def test_bands_thresholds_and_flagged_share():
    bands = {"severe": [0, 7], "mild": [7, 11], "healthy": [11, 99]}
    hb = np.array([5.0, 6.9, 7.0, 10.9, 11.0, 14.0])
    assert band_index(hb, bands).tolist() == [0, 0, 1, 1, 2, 2]   # lower edge included
    with pytest.raises(AssertionError):
        band_index(np.array([120.0]), bands)

    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 200)
    s = y + rng.normal(0, 1, 400)
    assert (s[y == 0] >= threshold(y, s, "specificity", 0.9)).mean() <= 0.1 + 1e-9
    t = threshold(y, s, "sensitivity", 0.9)
    assert (s[y == 1] >= t).mean() >= 0.9
    assert (s[y == 1] > t).mean() < 0.9                           # the highest threshold that still catches 90%

    # perfect score: every anemic band fully caught, healthy never flagged
    band = np.where(y == 0, 2, np.arange(400) // 2 % 2)
    packed = np.column_stack([y.astype(float), band])
    assert flagged_share("specificity", 0.9, [0])(y, packed) == 1.0
    assert flagged_share("specificity", 0.9, [0, 1])(y, packed) == 1.0
    assert flagged_share("specificity", 0.9, [2])(y, packed) == 0.0
    with pytest.raises(ValueError):
        flagged_share("specificity", 0.9, [7])(y, packed)
