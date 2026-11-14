import numpy as np

from src.eyelid.edit_robustness import COPIES, edited_numbers, fragile_edits, held_out_edits, random_edit
from src.eyelid.handling_check import RULE

RANGES = {"red_gain": [0.9, 1.1], "blue_gain": [0.9, 1.1], "brightness": [0.8, 1.25], "gamma": [0.8, 1.25],
          "saturation": [0.8, 1.0]}


def test_random_edit_is_seeded_bounded_and_identity_on_unit_ranges():
    rgb = np.random.default_rng(0).uniform(0, 1, (50, 3))
    a, b = (random_edit(rgb, RANGES, np.random.default_rng(3)) for _ in range(2))
    assert np.allclose(a, b) and a.min() >= 0 and a.max() <= 1 and not np.allclose(a, rgb)
    unit = {k: [1.0, 1.0] for k in RANGES}
    assert np.allclose(random_edit(rgb, unit, np.random.default_rng(3)), rgb)


def test_edited_numbers_shapes_and_copies_differ():
    rng = np.random.default_rng(1)
    px = [rng.uniform(0.2, 0.9, (40, 3)), rng.uniform(0.2, 0.9, (60, 3))]
    edits = [("brightness x0.85", "brightness", 0.85), ("gamma 0.8", "gamma", 0.8)]
    X, fixed, copies = edited_numbers(iter(px), edits, {"ranges": RANGES, "n_copies": 3}, seed=0)
    assert X.shape == (2, 29) and set(fixed) == {e[0] for e in edits} and fixed["gamma 0.8"].shape == (2, 29)
    assert copies.shape == (3, 2, 29) and not np.allclose(copies[0], X)


class Stub:
    """Scores each row by its first number and remembers how many rows it was fit on."""
    seen = []

    def fit(self, X, y):
        Stub.seen.append(len(X))
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_copies_are_only_in_the_copies_models_training_rows_and_rows_map_back():
    rng = np.random.default_rng(2)
    X, y = rng.uniform(size=(10, 29)), np.array([0, 1] * 5)
    copies, variants = rng.uniform(size=(3, 10, 29)), {"e": rng.uniform(size=(10, 29))}
    tr, te = np.arange(6), np.arange(6, 10)
    Stub.seen = []
    out = held_out_edits({"tabpfn": Stub, COPIES: Stub}, X, copies, y, [(0, tr, te), (0, te, tr)], variants)
    assert Stub.seen == [6, 24, 4, 16]  # plain: the photos; copies model: photos + 3 copies each
    assert np.allclose(out["as is"]["tabpfn"][0], X[:, 0]) and np.allclose(out["e"][COPIES][0], variants["e"][:, 0])


def test_fragile_edits_per_model():
    c = lambda e, hi: {"estimate": e, "lo": e - 0.05, "hi": hi}
    rows = [{"edit": "a", "model": "tabpfn", "change": c(-0.10, -0.05)},
            {"edit": "a", "model": COPIES, "change": c(-0.01, 0.02)},
            {"edit": "a", "model": RULE, "change": c(-0.01, 0.0)},
            {"edit": "b", "model": "tabpfn", "change": c(-0.10, -0.05)},
            {"edit": "b", "model": COPIES, "change": c(-0.08, -0.02)},
            {"edit": "b", "model": RULE, "change": c(-0.05, -0.02)}]
    cfg = {"fragile_drop": 0.05, "rule_stable_drop": 0.02}
    assert fragile_edits(rows, "tabpfn", cfg) == ["a"] and fragile_edits(rows, COPIES, cfg) == []
