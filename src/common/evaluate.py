"""The standard metric summary with bootstrap intervals, and paired AUROC differences."""
from functools import partial

import numpy as np

from src.common.metrics import auroc, bootstrap_ci, pr_auc, sensitivity_at_specificity

BASE_METRICS = {
    "auroc": auroc,
    "pr_auc": pr_auc,
    "sens_at_90_spec": partial(sensitivity_at_specificity, spec=0.90),
}


def summarize(y, scores, n_resamples, ci, seed, groups=None, metrics=BASE_METRICS):
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


def paired_gain(y, score, reference, groups, boot, seed):
    """AUROC(score) minus AUROC(reference) on the same women, with a CI from a bootstrap that resamples whole villages
    (groups) and scores both models on each resample. boot: {"n_resamples": ..., "ci": ...}."""
    packed = np.column_stack([reference, score])
    return bootstrap_ci(lambda yy, ss: auroc(yy, ss[:, 1]) - auroc(yy, ss[:, 0]), y, packed,
                        boot["n_resamples"], boot["ci"], groups, seed)
