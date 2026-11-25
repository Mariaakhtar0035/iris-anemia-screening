"""Smoke tests for the cluster split check. They use made-up data, so they run without the DHS files."""
import numpy as np
import pytest

from src.common.cluster_split import assert_no_cluster_overlap, cluster_folds


def fake_survey(seed=42, n_clusters=60, per_cluster=20):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    y = rng.integers(0, 2, len(clusters))
    return y, clusters


def test_cluster_folds_never_share_a_cluster():
    y, clusters = fake_survey()
    folds = cluster_folds(y, clusters, n_splits=5, n_repeats=2, seed=42)
    assert len(folds) == 10
    for _, train, test in folds:
        assert not set(clusters[train]) & set(clusters[test])


def test_check_catches_a_row_level_split():
    y, clusters = fake_survey()
    rng = np.random.default_rng(0)
    rows = rng.permutation(len(y))
    train, test = rows[: len(y) // 2], rows[len(y) // 2:]
    with pytest.raises(AssertionError):
        assert_no_cluster_overlap(clusters[train], clusters[test])


def test_same_folds_as_the_shared_maker():
    from src.common.splits import repeated_stratified_grouped
    y, clusters = fake_survey()
    ours = cluster_folds(y, clusters, n_splits=5, n_repeats=2, seed=42)
    theirs = repeated_stratified_grouped(y, clusters, n_splits=5, n_repeats=2, seed=42)
    for (r1, tr1, te1), (r2, tr2, te2) in zip(ours, theirs):
        assert r1 == r2 and np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
