import numpy as np

from src.eyelid.check_power import answers, folder_check, not_useful_check


def people(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return rng, rng.normal(-0.2, 1.8, n)  # z = Hb minus each person's own cutoff


def test_folder_check_passes_pallor_and_flags_a_folder_only_score():
    rng, z = people()
    pallor = -z + rng.normal(0, 1.5, len(z))
    folder = (z < 0) + rng.normal(0, 0.8, len(z))
    assert not folder_check(z, pallor, 2.0, 200, 0.95, 0)["flags"]
    assert folder_check(z, folder, 2.0, 200, 0.95, 0)["flags"]


def test_not_useful_check_flags_only_when_the_odd_numbers_match_paleness_and_beat_the_rule():
    rng, z = people(1000)
    y = (z < 0).astype(int)
    pale = -z + rng.normal(0, 1.0, len(z))
    rule = -z + rng.normal(0, 4.0, len(z))
    weak = -z + rng.normal(0, 6.0, len(z))
    assert not_useful_check(y, pale + rng.normal(0, 0.1, len(z)), pale, rule, 0.05, 200, 0.95, 0)["flags"]
    assert not not_useful_check(y, weak, pale, rule, 0.05, 200, 0.95, 0)["flags"]


def test_answers_count_flags_and_treat_rule_plants_as_missed_by_the_not_useful_check():
    flag = lambda b: {"flags": b}
    out = {"clean": {"a": {"folder": {"lr": flag(True)}, "not_useful": {"lr": flag(False)}},
                     "b": {"folder": {"lr": flag(False)}, "not_useful": {"lr": flag(False)}}},
           "planted": [{"gain": 0.2, "jumps": True, "folder": flag(True), "not_useful": flag(True)},
                       {"gain": 0.1, "jumps": False, "folder": flag(True), "not_useful": None},   # a plant on the rule
                       {"gain": 0.01, "jumps": None, "folder": flag(True), "not_useful": flag(True)}]}  # doesn't count
    a = answers(out, {"max_false_flags": 1, "min_caught": 2}, ["lr"], 0.05)
    assert a["counting"] == 2 and a["jump_test"]["caught"] == 1
    assert a["folder"] == {"false_flags": 1, "of_clean": 2, "caught": 2, "useful": True}
    assert a["not_useful"] == {"false_flags": 0, "of_clean": 2, "caught": 1, "useful": False}
