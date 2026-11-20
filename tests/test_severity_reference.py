import numpy as np

from src.common.metrics import auroc
from src.eyelid.severity_reference import beyond_folder, folder_line, hb_line


def children(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng, rng.normal(10.8, 1.8, n)  # bunched near the cutoff, like the Ghana children


def test_folder_line_matches_a_score_that_knows_only_the_folder():
    rng, hb = children(40000)  # many children below 8, so the check is on the formula, not on sampling noise
    folder = (hb < 11) + rng.normal(0, 0.8, len(hb))  # blurred to an AUROC well below 1
    for t in (10.0, 8.0, 7.0):
        assert abs(auroc(hb < t, folder) - folder_line(hb, folder, t, 11.0)) < 0.02
    assert abs(beyond_folder(8.0, 11.0)(None, np.column_stack([folder, hb]))) < 0.02


def test_a_score_that_follows_hb_beats_the_folder_line():
    rng, hb = children()
    pallor = -hb + rng.normal(0, 1.5, len(hb))
    assert beyond_folder(8.0, 11.0)(None, np.column_stack([pallor, hb])) > 0.1


def test_hb_line_is_matched_at_the_cutoff_and_rises_for_the_sickest():
    rng, hb = children(1000)
    pallor = -hb + rng.normal(0, 1.5, len(hb))
    line = hb_line(hb, pallor, 11.0, [11.0, 8.0], 300, rng)
    assert abs(line["11"]["median"] - auroc(hb < 11, pallor)) < 0.02
    assert line["8"]["median"] > line["11"]["median"]
