from functools import partial

import numpy as np
import pytest

from src.common.metrics import (
    auroc,
    bootstrap_ci,
    brier,
    pr_auc,
    recall_at_top_fraction,
    sensitivity_at_specificity,
)

# Small case worked by hand: negatives score 0.1, 0.4; positives 0.35, 0.8.
Y = [0, 0, 1, 1]
S = [0.1, 0.4, 0.35, 0.8]

PERFECT_Y = [0, 0, 0, 1, 1, 1]
PERFECT_S = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]


def test_auroc_hand_case():
    # 3 of 4 positive-negative pairs ranked correctly
    assert auroc(Y, S) == pytest.approx(0.75)


def test_pr_auc_hand_case():
    # precision 1 at recall 0.5, then 2/3 at recall 1.0
    assert pr_auc(Y, S) == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_sensitivity_at_specificity_hand_case():
    # 90% specificity with 2 negatives means no false positives, so only the 0.8 positive is caught
    assert sensitivity_at_specificity(Y, S, spec=0.90) == pytest.approx(0.5)


def test_perfect_and_inverted_scores():
    assert auroc(PERFECT_Y, PERFECT_S) == 1.0
    assert pr_auc(PERFECT_Y, PERFECT_S) == 1.0
    assert sensitivity_at_specificity(PERFECT_Y, PERFECT_S) == 1.0
    inverted = [-x for x in PERFECT_S]
    assert auroc(PERFECT_Y, inverted) == 0.0
    assert sensitivity_at_specificity(PERFECT_Y, inverted) == 0.0


def test_brier():
    assert brier([0, 1], [0.0, 1.0]) == 0.0
    assert brier([0, 1], [0.5, 0.5]) == pytest.approx(0.25)


def test_recall_at_top_fraction():
    y = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]  # 3 positives
    s = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    assert recall_at_top_fraction(y, s, frac=0.30) == pytest.approx(1 / 3)  # top 3 rows hold 1 positive
    assert recall_at_top_fraction(y, s, frac=0.40) == pytest.approx(2 / 3)  # top 4 rows hold 2
    assert recall_at_top_fraction(y, s, frac=0.25) == pytest.approx(1 / 3)  # ceil(2.5) = 3 rows
    assert recall_at_top_fraction(y, s, frac=1.0) == 1.0


def test_recall_at_top_fraction_ties_follow_input_order():
    y = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert recall_at_top_fraction(y, [0.5] * 10, frac=0.30) == pytest.approx(1 / 3)


def test_undefined_metrics_raise():
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        recall_at_top_fraction([0, 0, 0], [0.1, 0.2, 0.3])


def _noisy_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = y + rng.normal(0, 1, n)
    return y, s


def test_bootstrap_is_seeded_and_brackets_estimate():
    y, s = _noisy_data()
    first = bootstrap_ci(auroc, y, s, n_resamples=500, seed=7)
    assert first == bootstrap_ci(auroc, y, s, n_resamples=500, seed=7)
    assert first["lo"] <= first["estimate"] <= first["hi"]
    assert first["n_resamples"] == 500


def test_bootstrap_on_perfect_scores_is_a_point():
    y = [0] * 25 + [1] * 25
    s = list(range(50))
    result = bootstrap_ci(auroc, y, s, n_resamples=300)
    assert result["estimate"] == 1.0
    assert result["lo"] == pytest.approx(1.0) and result["hi"] == pytest.approx(1.0)


def test_bootstrap_counts_degenerate_resamples():
    # with 2 rows, about half of resamples hold a single class and AUROC is undefined there
    result = bootstrap_ci(auroc, [0, 1], [0.2, 0.8], n_resamples=400)
    assert 0 < result["n_skipped"] < 400


def test_bootstrap_works_with_partial_metric():
    y, s = _noisy_data()
    result = bootstrap_ci(partial(sensitivity_at_specificity, spec=0.9), y, s, n_resamples=200)
    assert 0.0 <= result["lo"] <= result["estimate"] <= result["hi"] <= 1.0


def test_grouped_bootstrap_keeps_groups_together():
    # group "a" has 1 row, group "b" has 5. Resampling 2 groups gives 2, 6 or 10 rows,
    # while resampling rows would always give 6.
    y = np.array([0, 1, 0, 1, 0, 1])
    s = np.zeros(6)
    groups = ["a", "b", "b", "b", "b", "b"]
    n_rows = lambda y_, s_: len(y_)
    grouped = bootstrap_ci(n_rows, y, s, n_resamples=500, groups=groups)
    assert (grouped["lo"], grouped["hi"]) == (2.0, 10.0)
    by_row = bootstrap_ci(n_rows, y, s, n_resamples=500)
    assert (by_row["lo"], by_row["hi"]) == (6.0, 6.0)
