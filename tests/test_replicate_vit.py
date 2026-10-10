import numpy as np

from src.eyelid.replicate_vit import make_split


def test_paper_split_lets_photos_cross_and_by_photo_does_not():
    n = 217
    paper_train, paper_test = make_split("paper", n, seed=0, test_size=0.2)
    photo_train, photo_test = make_split("by_photo", n, seed=0, test_size=0.2)

    for train, test in ((paper_train, paper_test), (photo_train, photo_test)):
        assert len(train) + len(test) == 3 * n  # every version of every photo used exactly once
        assert len(np.unique(np.vstack([train, test]), axis=0)) == 3 * n

    assert len(set(paper_train[:, 0]) & set(paper_test[:, 0])) > 0.5 * len(set(paper_test[:, 0]))
    assert not set(photo_train[:, 0]) & set(photo_test[:, 0])
    # by_photo keeps all three versions of each test photo in test
    _, counts = np.unique(photo_test[:, 0], return_counts=True)
    assert (counts == 3).all()


def test_splits_are_seeded():
    a, b = make_split("paper", 50, 3, 0.2), make_split("paper", 50, 3, 0.2)
    assert (a[1] == b[1]).all()
