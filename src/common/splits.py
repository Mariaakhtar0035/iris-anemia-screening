"""Cross-validation folds. Every function returns a list of (repeat, train_idx, test_idx)."""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedGroupKFold


def repeated_stratified(y, n_splits, n_repeats, seed):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(i // n_splits, tr, te) for i, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y))]


def repeated_stratified_grouped(y, groups, n_splits, n_repeats, seed):
    """Like repeated_stratified, but every member of a group lands in the same fold."""
    folds = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        folds += [(r, tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups)]
    return folds


def leave_one_group_out(groups):
    return [(0, tr, te) for tr, te in LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups)]
