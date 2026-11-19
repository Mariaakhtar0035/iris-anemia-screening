import numpy as np

from src.eyelid.presentation_numbers import catch_change, severity_change


def test_only_a_pallor_score_can_rise_for_the_sickest_children():
    rng = np.random.default_rng(0)
    hb = rng.normal(10.8, 1.8, 2000)  # bunched near the cutoff like the Ghana children, few near 8 (with an even
    # spread a pallor score doesn't rise: fewer children near a cutoff, not severity itself, makes it easier)
    pallor = -hb + rng.normal(0, 1, len(hb))                  # follows Hb all the way down
    folder = (hb < 11) + rng.normal(0, 0.1, len(hb))          # only knows anemic vs healthy
    change = severity_change(8.0, 11.0)
    assert change(None, np.column_stack([pallor, hb])) > 0
    # Hb < 8 against the rest, where the Hb 8-11 children score as high as the sickest: about 0.75 instead of 1.0
    assert change(None, np.column_stack([folder, hb])) < -0.2


def test_catch_change_is_zero_against_itself_and_positive_for_a_better_score():
    y = np.array([1] * 20 + [0] * 80)
    perfect, flat = -np.arange(100.0), np.zeros(100)
    assert catch_change(0.2)(y, np.column_stack([perfect, perfect])) == 0
    assert np.isclose(catch_change(0.2)(y, np.column_stack([flat, perfect])), 1 - 0.2)
