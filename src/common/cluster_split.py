"""Cluster-based folds, with a check that no cluster lands on both sides of a split.

Women from the same DHS cluster (village) are alike, so a row-level split lets the model peek at the test set.
This wraps the shared fold maker and asserts the rule on every fold, so a leak fails loudly instead of quietly.
"""
import numpy as np

from src.common.splits import repeated_stratified_grouped


def assert_no_cluster_overlap(train_clusters, test_clusters):
    overlap = np.intersect1d(train_clusters, test_clusters)
    assert len(overlap) == 0, f"{len(overlap)} clusters in both train and test, e.g. {overlap[:5]}"


def cluster_folds(y, clusters, n_splits, n_repeats, seed):
    """Same folds as repeated_stratified_grouped, each one checked for cluster overlap."""
    clusters = np.asarray(clusters)
    folds = repeated_stratified_grouped(np.asarray(y), clusters, n_splits, n_repeats, seed)
    for _, train, test in folds:
        assert_no_cluster_overlap(clusters[train], clusters[test])
    return folds
