import numpy as np

from src.eyelid.handling_check import RULE
from src.eyelid.trace_location import answers, core_and_ring


def disk(n=41, r=15):
    """RGBA disk: red where the pixel is within r/2 of the centre, blue towards the edge, transparent outside."""
    yy, xx = np.mgrid[:n, :n]
    d = np.hypot(yy - n // 2, xx - n // 2)
    rgba = np.zeros((n, n, 4), np.float32)
    rgba[..., 3] = d <= r
    rgba[..., 0] = d < r / 2
    rgba[..., 2] = d >= r / 2
    return rgba


def test_core_is_the_inner_share_and_ring_the_rest():
    rgba = disk()
    core, ring = core_and_ring(rgba, 255, 0.5)
    inside = int(rgba[..., 3].sum())
    assert len(core) + len(ring) == inside and abs(len(core) - inside / 2) <= 1
    assert core[:, 0].mean() > 0.4 and ring[:, 0].mean() < 0.1  # core red, ring blue


def test_cut_out_touching_the_border_still_has_an_edge_there():
    rgba = np.ones((5, 5, 4), np.float32)
    rgba[..., 0] = 0.0
    rgba[2, 2, 0] = 1.0  # only the centre pixel is red
    core, _ = core_and_ring(rgba, 255, 1 / 25)
    assert len(core) == 1 and core[0, 0] == 1.0


def test_where_answers_cover_all_four_outcomes():
    j = lambda b: {"jump": {"jumps": b}}
    D = {f"ghana_{k}": {"parts": {"core": {"tabpfn": j(c)}, "ring": {"tabpfn": j(r)}}, "handling_jumps": []}
         for k, (c, r) in {"a": (True, True), "b": (False, True), "c": (True, False), "d": (False, False)}.items()}
    D["india_eyelid"] = {"parts": {"core": {m: {"change": {"estimate": -0.01, "lo": -0.04, "hi": 0.02}}
                                            for m in ("tabpfn", RULE)}}, "handling_jumps": [
        {"feature": "sharpness", "jumps": True}, {"feature": "width", "jumps": False}]}
    ans = answers({"datasets": D}, {"clean": ["india_eyelid"], "keep_drop": 0.05}, ["tabpfn", RULE], 0.025)
    assert ans["where"] == {"ghana_a": "all through the cut-out", "ghana_b": "at the cut-out edge",
                            "ghana_c": "in the core, not the edge", "ghana_d": "no jump in either part"}
    assert ans["core_keeps_clean"] == {"india_eyelid": {"tabpfn": True, RULE: True}}
    assert ans["handling_numbers_that_jump"]["india_eyelid"] == ["sharpness"]
