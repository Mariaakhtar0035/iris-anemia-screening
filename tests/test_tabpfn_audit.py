import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.common.io import load_config
from src.common.metrics import auroc
from src.common.splits import repeated_stratified
from src.eyelid.tabpfn_audit import components, dhash_views, feature_names, file_details, min_bits, nested_choice


def test_groups_split_the_29_numbers():
    groups = load_config("tabpfn_audit")["groups"]
    assert sorted(sum(groups.values(), [])) == sorted(feature_names()) and len(feature_names()) == 29
    assert [len(v) for v in groups.values()] == [5, 6, 3, 15]


def test_dhash_finds_a_rotated_mirrored_copy_and_file_details(tmp_path):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (40, 90, 4), dtype=np.uint8)
    a[..., 3] = 255
    Image.fromarray(a).save(tmp_path / "a.png", dpi=(72, 72))
    Image.fromarray(np.ascontiguousarray(np.rot90(a[:, ::-1]))).save(tmp_path / "b.png")
    Image.fromarray(rng.integers(0, 255, (40, 90, 4), dtype=np.uint8)).save(tmp_path / "c.png")
    views = np.array([dhash_views(tmp_path / f) for f in ("b.png", "c.png")])
    d = min_bits(views, dhash_views(tmp_path / "a.png")[None, 0])
    assert d[0, 0] == 0 and d[1, 0] > 30          # the copy matches in some view; an unrelated photo doesn't
    assert components(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], bool)).tolist()[:2] == [0, 0]
    f = file_details(tmp_path / "a.png")
    assert round(f["dpi_x"]) == 72 and f["gamma"] == -1.0
    assert np.isclose(f["bytes_per_pixel"], (tmp_path / "a.png").stat().st_size / (40 * 90))


def test_nested_choice_picks_the_informative_model():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60)
    X = np.column_stack([y + rng.normal(0, 0.7, 120), rng.normal(size=120)])
    makers = {"dummy": lambda: DummyClassifier(strategy="prior"), "lr": lambda: LogisticRegression()}
    held, picks = nested_choice(makers, X, y, repeated_stratified(y, 5, 1, 0), 5, 0)
    assert picks == ["lr"] * 5 and auroc(y, held[0]) > 0.8
