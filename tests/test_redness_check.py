import pytest

from src.eyelid.redness_check import input_sets, reported, shows_no_trace

NAMES = ["R_mean", "S_mean", "a_mean", "a_std", "R_over_G"]
CFG = {"input_sets": {"mean_redness": ["a_mean"], "without_jumpers": "all_but_jumpers", "all_29": "all"}}


def test_input_sets_drop_jumpers_and_skip_when_nothing_jumps():
    assert input_sets(CFG, NAMES, ["S_mean", "R_over_G"]) == {
        "mean_redness": ["a_mean"], "without_jumpers": ["R_mean", "a_mean", "a_std"], "all_29": NAMES}
    assert list(input_sets(CFG, NAMES, [])) == ["mean_redness", "all_29"]
    with pytest.raises(AssertionError):
        input_sets({"input_sets": {"x": ["a_mean", "made_up"]}}, NAMES, [])


def result(p=0.3, step=(-0.2, 0.4), fragile=(), rule_lo=0.02):
    return {"jump": {"p": p}, "split": {"step": {"lo": step[0], "hi": step[1]}}, "fragile": list(fragile),
            "vs_rule": {"lo": rule_lo}}


def test_no_trace_needs_all_three_checks():
    assert shows_no_trace(result(), 0.025)
    assert not shows_no_trace(result(p=0.01), 0.025)
    assert not shows_no_trace(result(step=(0.1, 0.9)), 0.025)
    assert not shows_no_trace(result(fragile=["gamma 0.8"]), 0.025)
    assert not shows_no_trace({"jump": {"answer": "too few"}, "split": {}}, 0.025)
    rule = result()
    del rule["fragile"]
    assert shows_no_trace(rule, 0.025)


def test_reported_takes_the_largest_passing_set_that_beats_the_rule():
    m = "tabpfn"
    sets = {"small": {m: result() | {"no_trace": True}},
            "middle": {m: result(rule_lo=-0.01) | {"no_trace": True}},
            "large": {m: result() | {"no_trace": False}}}
    assert reported(sets, m) == "small"
    sets["middle"][m]["vs_rule"]["lo"] = 0.01
    assert reported(sets, m) == "middle"
    sets = {"small": {m: result(rule_lo=-0.1) | {"no_trace": True}}}
    assert reported(sets, m) is None
