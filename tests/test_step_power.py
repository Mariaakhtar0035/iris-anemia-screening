import numpy as np

from src.eyelid.step_power import answers, step_check, subsets

CFG = {"windows_g_dl": [1.0, 2.0], "sizes": [50, 150], "draws": 20, "half": 0.5}
CP_CFG = {"max_false_flags": 1, "min_caught": 1}


def test_subsets_keep_size_and_anemic_share():
    y = np.array([1] * 30 + [0] * 70)
    for i in subsets(y, 50, 5, np.random.default_rng(0)):
        assert len(i) == len(set(i)) == 50 and y[i].sum() == 15


def test_step_flags_a_planted_step_not_a_smooth_score():
    rng = np.random.default_rng(0)
    z = rng.uniform(-4, 3, 400)
    smooth = -z + rng.normal(0, 1.5, len(z))
    assert not step_check(z, smooth, 2.0, 200, 0.95, 0)["flags"]
    assert step_check(z, smooth + 3 * (z < 0), 2.0, 200, 0.95, 0)["flags"]


def test_answers_count_windows_and_rates_on_bigger_sets_only():
    w = lambda *f: {"1": {"flags": f[0]}, "2": {"flags": f[1]}}
    rows = [{"dataset": "moscow_nail", "plant": None, "model": "m", "windows": w(True, False),
             "sizes": {50: [False] * 20}},
            {"dataset": "moscow_nail", "plant": "gamma 0.9", "model": "m", "windows": w(False, True),
             "sizes": {50: [True] * 10 + [False] * 10}},
            {"dataset": "india_eyelid", "plant": "gamma 0.9", "model": "m", "windows": w(True, True), "sizes": {}}]
    india = lambda r: r["dataset"] == "india_eyelid"
    a = answers(rows, CFG, CP_CFG, 2.0, india, 200, 0.95, 0)
    assert a["windows"]["1"] == {"false_flags": 1, "caught": 1, "india_caught": 1, "passes": True}
    assert a["windows"]["2"]["caught"] == 2 and a["robust"]
    s = a["sizes"][50]
    assert s["datasets"] == ["moscow_nail"] and s["n_plants"] == 1 and s["full_size_caught"] == 1
    assert s["catch_rate"]["estimate"] == 0.5 and s["false_flag_rate"]["estimate"] == 0.0
    assert 150 not in a["sizes"] and a["smallest_half"] == 50
