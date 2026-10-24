"""Out-of-fold scoring and the standard metric summary with bootstrap intervals."""
from functools import partial

import numpy as np

from src.common.metrics import auroc, bootstrap_ci, pr_auc, sensitivity_at_specificity

IMAGE_METRICS = {
    "auroc": auroc,
    "pr_auc": pr_auc,
    "sens_at_90_spec": partial(sensitivity_at_specificity, spec=0.90),
}


def oof_scores(make_model, X, y, folds):
    """Out-of-fold P(positive) for every row, shape (n_repeats, n_rows)."""
    n_repeats = max(r for r, _, _ in folds) + 1
    scores = np.full((n_repeats, len(y)), np.nan)
    for r, train, test in folds:
        model = make_model().fit(X[train], y[train])
        scores[r, test] = model.predict_proba(X[test])[:, 1]
    assert not np.isnan(scores).any(), "some rows were never in a test fold"
    return scores


def summarize(y, scores, n_resamples, ci, seed, groups=None, metrics=IMAGE_METRICS):
    """Metrics on the per-row score averaged over repeats, each with a bootstrap CI.

    Every row's score only ever comes from models that never saw that row, so the average stays out-of-fold.
    """
    scores = np.atleast_2d(scores)
    mean_score = scores.mean(axis=0)
    out = {name: bootstrap_ci(fn, y, mean_score, n_resamples, ci, groups, seed) for name, fn in metrics.items()}
    if scores.shape[0] > 1:
        per_repeat = [auroc(y, s) for s in scores]
        out["auroc_per_repeat"] = {"mean": float(np.mean(per_repeat)), "sd": float(np.std(per_repeat))}
    return out


def auroc_change(y, scores, reference, n_resamples, ci, seed):
    """AUROC of scores minus AUROC of reference, with a paired bootstrap CI: both scored on the same resampled rows.
    Each may be (n,) or (n_repeats, n); repeats are averaged as in summarize."""
    packed = np.column_stack([np.atleast_2d(reference).mean(0), np.atleast_2d(scores).mean(0)])
    return bootstrap_ci(lambda yy, ss: auroc(yy, ss[:, 1]) - auroc(yy, ss[:, 0]), y, packed, n_resamples, ci, seed=seed)
