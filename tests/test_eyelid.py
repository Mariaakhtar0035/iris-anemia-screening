import numpy as np
import pytest
from PIL import Image

from src.common.splits import repeated_stratified, repeated_stratified_grouped
from src.eyelid.color_features import color_features, gray_world
from src.eyelid.roi import roi_pixels


def test_roi_keeps_only_opaque_pixels(tmp_path):
    img = np.zeros((4, 4, 4), dtype=np.uint8)
    img[:2, :, :] = [255, 0, 0, 255]  # 8 opaque red pixels
    img[2, :, :] = [0, 255, 0, 128]   # 4 half-transparent green edge pixels
    path = tmp_path / "x.png"
    Image.fromarray(img, "RGBA").save(path)
    rgb = roi_pixels(path)
    assert rgb.shape == (8, 3)
    assert np.allclose(rgb, [1, 0, 0])


def test_color_features_on_pure_red():
    f = color_features(np.tile([1.0, 0.0, 0.0], (10, 1)))
    assert f["R_mean"] == 1.0 and f["G_mean"] == 0.0
    assert f["H_mean"] == pytest.approx(180.0)  # red rotated to 180 degrees
    assert f["a_mean"] > 50  # strongly red in CIELAB
    assert f["R_std"] == 0.0
    assert len(f) == 29


def test_hue_rotation_keeps_reds_together():
    # reds just either side of 0/360 degrees would average to ~180 (cyan) without the rotation
    rgb = np.array([[1.0, 0.0, 0.05], [1.0, 0.05, 0.0]])
    f = color_features(rgb)
    assert abs(f["H_mean"] - 180.0) < 5


def test_gray_world_equalizes_channel_means():
    rgb = np.random.default_rng(0).uniform(0.2, 0.6, (100, 3)) * [1.0, 0.5, 0.8]
    assert np.allclose(gray_world(rgb).mean(axis=0), gray_world(rgb).mean(), atol=1e-6)


def test_splits_are_seeded_and_cover_every_row_once_per_repeat():
    y = np.array([0, 1] * 20)
    a, b = repeated_stratified(y, 5, 3, seed=1), repeated_stratified(y, 5, 3, seed=1)
    assert all((ta == tb).all() for (_, ta, _), (_, tb, _) in zip(a, b))
    for r in range(3):
        tested = np.concatenate([te for rep, _, te in a if rep == r])
        assert sorted(tested) == list(range(len(y)))


def test_grouped_splits_never_split_a_group():
    y = np.array([0, 1] * 30)
    groups = np.repeat(np.arange(20), 3)
    for _, train, test in repeated_stratified_grouped(y, groups, 5, 2, seed=0):
        assert not set(groups[train]) & set(groups[test])


def test_training_rows_keep_copies_out_of_test():
    from src.eyelid.ceiling import training_rows
    groups = np.array(["a", "a", "b", "c", "c", "c"])
    assert list(training_rows(groups, ["a", "c"], every_copy=True)) == [0, 1, 3, 4, 5]
    assert list(training_rows(groups, ["a", "c"], every_copy=False)) == [0, 3]
    assert "b" not in groups[training_rows(groups, ["a", "c"], every_copy=True)]


def test_white_background_mask_becomes_transparent(tmp_path):
    img = np.full((4, 4, 4), 255, dtype=np.uint8)  # fully opaque, pure white, like Eyes-Defy Italy 1-31 masks
    img[0, :2, :3] = [200, 100, 110]                # two conjunctiva pixels
    path = tmp_path / "white.png"
    Image.fromarray(img, "RGBA").save(path)
    rgb = roi_pixels(path)
    assert rgb.shape == (2, 3)
    assert np.allclose(rgb, np.array([200, 100, 110]) / 255)


def test_reweighing_makes_group_and_label_independent():
    from src.eyelid.debias import reweigh
    rng = np.random.default_rng(0)
    g = rng.choice(["India", "Italy"], 300)
    y = (rng.random(300) < np.where(g == "India", 0.7, 0.2)).astype(int)
    w = reweigh(y, g)
    for grp in ("India", "Italy"):
        m = g == grp
        assert np.isclose((w[m] * y[m]).sum() / w[m].sum(), (w * y).sum() / w.sum())


def test_fair_lr_without_penalty_matches_sklearn():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from src.eyelid.debias import FairLogisticRegression
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    g = rng.choice(["a", "b"], 200)
    ours = FairLogisticRegression(0.0, C=1.0).fit(X, y, g).predict_proba(X)[:, 1]
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(ours, ref, atol=1e-3)  # same objective; the optimizers stop at slightly different points


def test_fair_lr_penalty_shrinks_the_group_gap():
    from src.eyelid.debias import FairLogisticRegression
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    g = np.where(X[:, 1] > 0, "a", "b")              # feature 1 reveals the group
    y = (X[:, 0] + X[:, 1] + rng.normal(size=400) > 0).astype(int)

    def gap(lam):
        p = FairLogisticRegression(lam).fit(X, y, g).predict_proba(X)[:, 1]
        return sum((p[(y == k) & (g == "a")].mean() - p[(y == k) & (g == "b")].mean()) ** 2 for k in (0, 1))
    assert gap(100.0) < 0.25 * gap(0.0)
