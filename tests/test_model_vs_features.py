import numpy as np
import pytest

from src.eyelid.model_vs_features import age_and_sex, answers, mlp_factory
from src.eyelid.redness_check import CV

MODELS = ["logistic_regression", "mlp", "tabpfn"]
CFG = {"models": MODELS, "weird_margin": 0.05,
       "mlp": {"hidden_layer_sizes": [[4], [8]], "alpha": [0.1, 10.0], "inner_folds": 3, "max_iter": 500, "n_jobs": 1}}


def test_mlp_picks_its_settings_from_the_grid_and_learns_a_clear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + rng.normal(0, 0.5, 200) > 0).astype(int)
    m = mlp_factory(CFG, seed=0)().fit(X[:150], y[:150])
    assert m.best_params_["mlpclassifier__hidden_layer_sizes"] in [(4,), (8,)]
    assert m.best_params_["mlpclassifier__alpha"] in [0.1, 10.0]
    p = m.predict_proba(X[150:])[:, 1]
    assert p.shape == (50,) and np.corrcoef(p, X[150:, 0])[0, 1] > 0.5


def test_age_and_sex_only_for_the_ghana_eyelids():
    with pytest.raises(AssertionError):
        age_and_sex("ghana_nail", {"hb": np.zeros(3)})


def ci(est, lo):
    return {"estimate": est, "lo": lo, "hi": est + (est - lo)}


def fake(au, vs_tabpfn, pairs, rule_lo):
    """au: {set: {model: AUROC}}; vs_tabpfn: {model: CI lo of TabPFN minus it}; pairs: {pair: {model: CI lo}}."""
    sets = {k: {m: {"scores": {"auroc": ci(a, a - 0.05)}, "vs_rule": ci(0.1, rule_lo.get((k, m), 0.01))}
                for m, a in ms.items()} for k, ms in au.items()}
    sets["all_29"]["tabpfn"]["vs"] = {m: ci(0.05, lo) for m, lo in vs_tabpfn.items()}
    return {"splits": {CV: {"sets": sets, "set_pairs": {p: {m: ci(0.0, lo) for m, lo in ms.items()}
                                                         for p, ms in pairs.items()}}}}


def test_answers_follow_the_rules_fixed_in_the_config():
    au = {"all_29": {"logistic_regression": 0.70, "mlp": 0.80, "tabpfn": 0.92},
          "not_useful": {"logistic_regression": 0.60, "mlp": 0.76, "tabpfn": 0.90},
          "paleness": {"logistic_regression": 0.70, "mlp": 0.80, "tabpfn": 0.88}}
    pairs = {"all_29_age_sex - all_29": {"logistic_regression": 0.01, "mlp": -0.02, "tabpfn": -0.01},
             "all_29_age_sex - age_sex_only": {"logistic_regression": 0.1, "mlp": 0.1, "tabpfn": 0.2}}
    a = answers(fake(au, {"logistic_regression": 0.1, "mlp": 0.03}, pairs, {("not_useful", "mlp"): -0.02}), CFG)
    assert a["best_other"] == "mlp" and a["tabpfn_adds"]  # compared with the best other model, the MLP here
    assert a["age_sex_helps"] == {"logistic_regression": True, "mlp": False, "tabpfn": False}
    assert all(a["photo_adds"].values())
    # LR: 0.60 < 0.70 - 0.05; MLP: close enough but doesn't beat the rule; TabPFN: above paleness and beats the rule
    assert a["weird"] == {"logistic_regression": False, "mlp": False, "tabpfn": True}
    a = answers(fake(au, {"logistic_regression": 0.1, "mlp": -0.01}, pairs, {}), CFG)
    assert not a["tabpfn_adds"]
