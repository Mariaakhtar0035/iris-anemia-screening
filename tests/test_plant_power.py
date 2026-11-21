import numpy as np

from src.eyelid.handling_check import RULE
from src.eyelid.plant_power import CHECKS, answers, breaks, checks

EDIT_CFG = {"fragile_drop": 0.05, "rule_stable_drop": 0.02}


def test_breaks_needs_a_real_drop_while_the_rule_holds():
    drop, small, rule_ok, rule_moves = ({"estimate": -0.1, "hi": -0.04}, {"estimate": -0.03, "hi": -0.01},
                                        {"estimate": -0.01}, {"estimate": -0.05})
    assert breaks(drop, rule_ok, EDIT_CFG)
    assert not breaks(small, rule_ok, EDIT_CFG)
    assert not breaks(drop, rule_moves, EDIT_CFG)


def test_checks_pass_a_pallor_score_and_flag_a_folder_trace():
    rng = np.random.default_rng(0)
    z = rng.uniform(-4, 3, 2000)  # Hb minus the cutoff
    y = (z < 0).astype(int)
    pallor, rule = -z + rng.normal(0, 1.5, len(z)), -z + rng.normal(0, 3, len(z))
    trace = pallor + 3 * y  # the edit wipes the folder trace out, as a test edit does to a delicate one
    held = {"as is": {"pallor": pallor, "trace": trace, RULE: rule},
            "gamma 0.8": {"pallor": pallor, "trace": pallor, RULE: rule}}
    r = checks(y, z, held, ["pallor", "trace"], EDIT_CFG, 1, 2.0, 200, 0.95, 0)
    assert not any(r["pallor"][k]["flags"] for k in ("edit", "step", "hb_inside"))
    assert r["trace"]["edit"]["broken_by"] == ["gamma 0.8"] and r["trace"]["step"]["flags"]
    assert "edit" not in r[RULE]


def test_answers_combine_only_checks_that_pass_clean_data():
    f = lambda *on: {k: {"flags": k in on} for k in CHECKS}
    clean = {"a": {"m": f("folder")}, "b": {"m": f("folder", "hb_inside")}}
    plants = [{"dataset": "india_eyelid", "plant": "gamma 0.9", "gain": 0.2, **f("folder", "step")},
              {"dataset": "india_eyelid", "plant": "white balance R x1.04, B x0.96", "gain": 0.2, **f("jump")},
              {"dataset": "moscow_nail", "plant": "gamma 0.9", "gain": 0.1, **f("edit"), "not_useful": None},
              {"dataset": "moscow_nail", "plant": "saturation 0.9", "gain": 0.01, **f("edit")}]  # doesn't count
    india = lambda r: r["dataset"] == "india_eyelid" and not r["plant"].startswith("white balance")
    a = answers(clean, plants, {"max_false_flags": 1, "min_caught": 2}, ["m"], 0.05, india)
    assert a["counting"] == 3 and a["of_clean"] == 2
    assert a["checks"]["folder"] == {"false_flags": 2, "caught": 1, "india_caught": 1, "useful": False}
    assert a["checks"]["hb_inside"]["false_flags"] == 1 and a["checks"]["not_useful"]["caught"] == 0
    assert "folder" not in a["combined"]["checks"] and a["combined"]["false_flags"] == 1
    assert a["combined"]["caught"] == 3 and a["all_six"] == {"false_flags": 2, "caught": 3}
    assert a["india_missed"] == {"n": 1, "caught_by_any": 1}
