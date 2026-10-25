import numpy as np

from src.common.splits import repeated_stratified
from src.eyelid.debias import group_standardize
from src.eyelid.three_countries import rank_spread


def test_rank_spread_is_zero_when_country_does_not_matter():
    y = np.array([0, 0, 0, 0, 1, 1])
    code = np.array([0, 1, 0, 1, 0, 1])
    same = np.column_stack([[0.1, 0.1, 0.2, 0.2, 0.9, 0.9], code])
    assert rank_spread(0)(y, same) == 0
    by_country = np.column_stack([[0.1, 0.5, 0.2, 0.6, 0.9, 0.95], code])
    assert np.isclose(rank_spread(0)(y, by_country), 2 / 6)  # healthy ranks: country 0 at 1 and 2, country 1 at 3 and 4


def test_every_fold_holds_every_country_and_label():
    code = np.repeat([0, 1, 2], [388, 95, 122])  # Ghana, India, Italy
    y = np.concatenate([np.arange(388) < 165, np.arange(95) < 68, np.arange(122) < 23]).astype(int)
    for _, _, te in repeated_stratified(2 * code + y, 5, 5, 42):
        assert len(np.unique(2 * code[te] + y[te])) == 6


def test_standardization_uses_only_the_training_rows():
    X_tr, g_tr = np.array([[0.0], [2.0], [10.0], [14.0]]), np.array([0, 0, 1, 1])
    out = group_standardize(X_tr, g_tr, np.array([[4.0], [12.0]]), np.array([0, 1]))
    assert np.allclose(out.ravel(), [3.0, 0.0])  # (4 - 1) / 1 and (12 - 12) / 2
