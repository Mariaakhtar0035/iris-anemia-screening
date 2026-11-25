"""Stage 31 (src/survey/covariates.py): joining village covariates to women, on made-up data."""
import numpy as np
import pandas as pd
import pytest


def _cfg(tmp_path):
    pd.DataFrame({"DHSCLUST": [1.0, 2.0, 3.0], "Malaria": [0.1, -9999, 0.3], "Rain": [500, 600, 700],
                  "Unused": [9, 9, 9]}).to_csv(tmp_path / "gc.csv", index=False)
    return {"file": str(tmp_path / "gc.csv"), "join_column": "DHSCLUST", "missing_code": -9999,
            "covariates": ["Malaria", "Rain"]}


def test_missing_code_becomes_blank_and_only_chosen_columns_are_kept(tmp_path):
    from src.survey.covariates import load_covariates

    cov = load_covariates(_cfg(tmp_path))
    assert list(cov.columns) == ["Malaria", "Rain"]
    assert list(cov.index) == [1, 2, 3]
    assert np.isnan(cov.loc[2, "Malaria"]) and cov.loc[3, "Malaria"] == 0.3


def test_each_woman_gets_her_own_villages_values(tmp_path):
    from src.survey.covariates import attach, load_covariates

    cov = load_covariates(_cfg(tmp_path))
    feats = pd.DataFrame({"age": [20, 30, 40, 50]}, index=[7, 8, 9, 10])   # index need not line up
    out = attach(feats, np.array([3, 1, 3, 2]), cov)
    assert out.age.tolist() == [20, 30, 40, 50]
    assert out.Rain.tolist() == [700, 500, 700, 600]


def test_a_village_missing_from_the_file_stops_the_run(tmp_path):
    from src.survey.covariates import attach, load_covariates

    cov = load_covariates(_cfg(tmp_path))
    with pytest.raises(AssertionError, match="villages without covariates"):
        attach(pd.DataFrame({"age": [20]}), np.array([4]), cov)


def test_blank_cells_stay_blank_and_other_values_untouched(tmp_path):
    """Malawi and Zambia mark missing with an empty cell instead of -9999."""
    from src.survey.covariates import load_covariates

    (tmp_path / "gc.csv").write_text("DHSCLUST,Malaria,Rain\n1,0.1,500\n2,,600\n3,0.3,\n")
    cfg = {"file": str(tmp_path / "gc.csv"), "join_column": "DHSCLUST", "missing_code": -9999,
           "covariates": ["Malaria", "Rain"]}
    cov = load_covariates(cfg)
    assert np.isnan(cov.loc[2, "Malaria"]) and np.isnan(cov.loc[3, "Rain"])
    assert cov.loc[1, "Malaria"] == 0.1 and cov.loc[2, "Rain"] == 600
