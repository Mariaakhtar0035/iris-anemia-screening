"""Evaluation metrics, plus a bootstrap wrapper that puts a confidence interval on any of them.

Rank metrics (everything except brier) raise ValueError when they're undefined, e.g. a sample with
only one class. bootstrap_ci relies on that to skip and count degenerate resamples.
"""
import math

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def _check_both_classes(y):
    if len(np.unique(y)) < 2:
        raise ValueError("metric needs both classes present")


def auroc(y, s):
    _check_both_classes(y)
    return float(roc_auc_score(y, s))


def pr_auc(y, s):
    """Average precision: the step-wise area under the precision-recall curve, no interpolation."""
    _check_both_classes(y)
    return float(average_precision_score(y, s))


def sensitivity_at_specificity(y, s, spec=0.90):
    """Highest sensitivity reachable while keeping specificity at or above `spec`."""
    _check_both_classes(y)
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= 1 - spec + 1e-12].max())


def recall_at_top_fraction(y, s, frac=0.30):
    """Share of all positives found when only the top `frac` of people by score get tested."""
    y, s = np.asarray(y), np.asarray(s)
    if y.sum() == 0:
        raise ValueError("recall needs at least one positive")
    k = math.ceil(frac * len(s))
    # Ties at the cut are shared out at random, so take their expected number of positives (a constant score gives frac).
    cut = np.sort(s)[::-1][k - 1]
    above, tied = s > cut, s == cut
    expected = y[above].sum() + (k - above.sum()) * y[tied].mean()
    return float(expected / y.sum())


def brier(y, p):
    return float(brier_score_loss(y, p))


def _recalibration_fit(y, p):
    """Logistic regression of the outcome on logit(p): slope 1 and intercept 0 mean perfectly calibrated."""
    from sklearn.linear_model import LogisticRegression
    _check_both_classes(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    fit = LogisticRegression(C=np.inf).fit(np.log(p / (1 - p))[:, None], y)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def calibration_slope(y, p):
    """Below 1: predictions too spread out (overconfident); above 1: too timid."""
    return _recalibration_fit(y, p)[0]


def calibration_intercept(y, p):
    """Intercept of the same fit; with the slope near 1, above 0 means risk is underestimated overall."""
    return _recalibration_fit(y, p)[1]


def bootstrap_ci(metric, y, s, n_resamples=2000, ci=0.95, groups=None, seed=42):
    """Percentile bootstrap interval for `metric(y, s)`.

    Pass `groups` (patient or cluster ids) to resample whole groups instead of rows.
    For metrics with extra arguments use functools.partial, e.g. partial(sensitivity_at_specificity, spec=0.9).
    Resamples where the metric is undefined are skipped and reported in `n_skipped`.
    """
    y, s = np.asarray(y), np.asarray(s)
    estimate = metric(y, s)
    rng = np.random.default_rng(seed)

    if groups is not None:
        _, inverse = np.unique(np.asarray(groups), return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        rows_by_group = np.split(order, np.cumsum(np.bincount(inverse))[:-1])

    stats = []
    for _ in range(n_resamples):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
            idx = np.concatenate([rows_by_group[g] for g in picked])
        try:
            stats.append(metric(y[idx], s[idx]))
        except ValueError:
            pass

    if not stats:
        raise ValueError("every bootstrap resample was degenerate; the metric is undefined on this data")
    lo, hi = np.percentile(stats, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {
        "estimate": float(estimate),
        "lo": float(lo),
        "hi": float(hi),
        "n_resamples": n_resamples,
        "n_skipped": n_resamples - len(stats),
    }
