import numpy as np
from PIL import Image

from src.common.io import load_config
from src.ghana_link import agreement, parse, partner_order, pixel_key, same_label_shuffle, shared_groups, within_ranks


def test_parse_reads_label_batch_and_number_and_leaves_out_odd_names():
    cfg = load_config("ghana_link")
    nail, eye, palm, cut = cfg["names"]["nail"], cfg["names"]["eyelid"], cfg["names"]["palm"], cfg["copy_suffix"]
    assert parse("Anemic-FN-105 (3)", nail, cut) == (1, "main", 105)
    assert parse("Non-anemic-Fin-007", nail, cut) == (0, "small", 7)
    assert parse("Anemic-005 (2) - Copy", eye, cut) == (1, "main", 5)
    assert parse("Anemic-Conj-012-i10vHz", eye, cut) == (1, "small", 12)
    assert parse("AnemicP-126", palm, cut) == parse("Anemic-126", palm, cut) == (1, "main", 126)
    assert parse("Anemic-001FV", eye, cut) is None          # letters after the number
    assert parse("Anemic-Conj-N-00141", eye, cut) is None   # batch we leave out
    assert parse("Anmeic-fn-004 1", nail, cut) is None


def test_pixel_key_ignores_rotation_and_mirroring(tmp_path):
    a = np.random.default_rng(0).integers(0, 255, (5, 7, 4), dtype=np.uint8)
    for name, img in {"a": a, "rot": np.rot90(a), "flip": a[:, ::-1]}.items():
        Image.fromarray(np.ascontiguousarray(img), "RGBA").save(tmp_path / f"{name}.png")
    keys = {pixel_key(tmp_path / f"{n}.png") for n in ("a", "rot", "flip")}
    assert len(keys) == 1


def test_shuffles_stay_inside_groups_and_neighbours():
    rng = np.random.default_rng(0)
    groups = np.array(["a"] * 20 + ["b"] * 10)
    numbers = np.r_[np.arange(20), np.arange(10)]
    for block in (None, 5):
        order = partner_order(groups, numbers, rng, block)
        assert (groups[order] == groups).all() and sorted(order) == list(range(30))
        if block:
            assert np.abs(numbers[order] - numbers).max() < 2 * block
    shifted = partner_order(groups, numbers, offset=1)
    assert numbers[shifted][1] == 0 and numbers[shifted][21] == 0
    y = np.array([0, 1] * 15)
    assert (y[same_label_shuffle(y, rng)] == y).all()


def test_agreement_is_one_for_identical_photos_and_groups_join_shared_photos():
    x = np.random.default_rng(1).normal(size=(30, 3))
    g = np.array(["a"] * 15 + ["b"] * 15)
    assert np.isclose(agreement(within_ranks(x, g), within_ranks(x, g)), 1.0)
    ids = shared_groups(np.array([1, 2, 3, 4]), np.array(["n1", "n2", "n2", "n3"]), np.array(["p1", "p2", "p3", "p1"]))
    assert ids[1] == ids[2] and ids[0] == ids[3] and ids[0] != ids[1]
