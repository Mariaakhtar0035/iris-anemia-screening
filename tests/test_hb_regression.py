import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor

from src.common.splits import repeated_stratified
from src.eyelid.hb_regression import mae, oof_regression, sens_at_cutoff, spec_at_cutoff, within_1


def test_hb_error_metrics():
    hb, pred = np.array([10.0, 12.0, 9.0]), np.array([10.5, 10.0, 9.0])
    assert mae(hb, pred) == pytest.approx(2.5 / 3)
    assert within_1(hb, pred) == pytest.approx(2 / 3)


def test_cutoff_is_applied_to_the_estimate():
    anemic = np.array([1, 1, 0, 0])
    pred = np.array([10.0, 11.5, 10.9, 12.0])  # cutoff 11.0: first anemic caught, second missed; one healthy flagged
    assert sens_at_cutoff(anemic, pred, 11.0) == 0.5
    assert spec_at_cutoff(anemic, pred, 11.0) == 0.5
    with pytest.raises(ValueError):
        sens_at_cutoff(np.zeros(3), pred[:3], 11.0)


def test_oof_regression_never_predicts_a_row_it_trained_on():
    # 1-nearest-neighbour copies the training value exactly, so any leak would reproduce the true Hb.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    hb = rng.uniform(7, 14, 60)
    folds = repeated_stratified((hb < 11).astype(int), 5, 2, 0)
    preds = oof_regression(lambda: KNeighborsRegressor(n_neighbors=1), X, hb, folds)
    assert preds.shape == (2, 60)
    assert not np.isclose(preds, hb).any()


def test_cross_dataset_helpers():
    from src.eyelid.hb_cross import bias, standardize

    assert bias(np.array([12.0, 10.0]), np.array([11.0, 10.0])) == -0.5  # negative = estimates too low
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = standardize(X)
    assert np.allclose(Z[:, 0].mean(), 0) and np.allclose(Z[:, 0].std(), 1)
    assert np.allclose(Z[:, 1], 0)  # a constant column stays finite


def test_local_corrections():
    from src.eyelid.hb_local import cutoff_stats, shift, shift_and_scale

    pred_k, hb_k = np.array([10.0, 11.0, 12.0]), np.array([13.0, 14.0, 15.0])
    assert np.allclose(shift(pred_k, hb_k)(np.array([9.0])), [12.0])            # fixes a constant offset
    assert np.allclose(shift_and_scale(pred_k, hb_k)(np.array([9.0])), [12.0])  # here the line is the same shift
    assert np.allclose(shift_and_scale(np.array([11.0, 11.0]), np.array([12.0, 14.0]))(np.array([11.0])), [13.0])
    s = cutoff_stats(np.array([11.0, 13.0]), np.array([11.5, 12.5]), np.array([12.0, 12.0]))
    assert s["sens_at_cutoff"] == 1.0 and s["spec_at_cutoff"] == 1.0 and s["bias"] == 0.0
