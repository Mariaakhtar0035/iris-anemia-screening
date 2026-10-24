import numpy as np
from PIL import Image

from src.ghana_sites import parse_name, rotation_key


def test_parse_name_reads_label_and_style_despite_typos():
    assert parse_name("Non-AnemicP-010 (3)") == ("Non-AnemicP-010", 0, "p")
    assert parse_name("Anemic-Pa-018") == ("Anemic-Pa-018", 1, "pa")
    assert parse_name("Non-Anrmic-FN-010 (12)") == ("Non-Anrmic-FN-010", 0, "fn")
    assert parse_name("Anmeic-fn-004 1 (2)") == ("Anmeic-fn-004 1", 1, "fn")
    assert parse_name("Anemic-262") == ("Anemic-262", 1, "none")


def test_rotation_key_matches_rotated_copies_only(tmp_path):
    a = np.random.default_rng(0).integers(0, 256, (5, 7, 4), dtype=np.uint8)
    for name, img in [("a", a), ("a_rot", np.rot90(a)), ("a_rot180", np.rot90(a, 2)), ("other", a[::-1])]:
        Image.fromarray(np.ascontiguousarray(img), "RGBA").save(tmp_path / f"{name}.png")
    key = {p.stem: rotation_key(p) for p in tmp_path.glob("*.png")}
    assert key["a"] == key["a_rot"] == key["a_rot180"]
    assert key["other"] != key["a"]  # a mirror image is a different photo
