import numpy as np
import pandas as pd

from src.common.io import load_config
from src.survey.build_dataset import clean


def test_label_is_any_anemia_and_untested_women_are_dropped():
    cfg = load_config("survey")
    raw = pd.DataFrame({
        "age": [20, 30, 40, 25, 35], "pregnant": [0, 1, 0, 0, 1], "wealth": [1, 2, np.nan, 4, 5],
        "contraception": [0, 3, 0, 0, 0], "residence": [1, 2, 2, 1, 2], "education": [0, 1, 2, 3, 1],
        "cluster": [1, 1, 2, 2, 3], "weight": [1_500_000, 500_000, 1_000_000, 1_000_000, 2_000_000], "region": [1] * 5,
        "hb_adjusted": [75, 105, 118, 130, np.nan], "anemia_level": [1, 2, 3, 4, np.nan],  # severe, moderate, mild, not, untested
    })
    labels = {"v312": {0: "not using", 3: "injections"}, "v025": {1: "urban", 2: "rural"}, "v024": {1: "a"}}
    df = clean(raw, labels, cfg, "test")
    assert len(df) == 4
    assert df.anemic.tolist() == [1, 1, 1, 0]
    assert df.weight.tolist() == [1.5, 0.5, 1.0, 1.0]
    assert df.hb_adjusted.tolist() == [7.5, 10.5, 11.8, 13.0]
    assert df.wealth.isna().sum() == 1  # left for the model pipeline to impute


def test_pipeline_fits_on_training_rows_and_handles_unseen_values():
    from src.survey.train_baselines import make_model

    cfg = load_config("survey")
    rng = np.random.default_rng(0)
    n = 300
    train = pd.DataFrame({
        "age": rng.integers(15, 50, n), "pregnant": rng.integers(0, 2, n), "wealth": rng.integers(1, 6, n).astype(float),
        "education": rng.integers(0, 4, n), "contraception": rng.choice(["not using", "pill", "injections"], n),
        "residence": rng.choice(["urban", "rural"], n),
    })
    y = rng.integers(0, 2, n)
    model = make_model("logistic_regression", cfg, seed=0).fit(train, y)
    # A 2024-only contraception code and a missing wealth index, as in Malawi and Zambia.
    target = train.head(3).assign(contraception=["standard days method (sdm)", "pill", "not using"],
                                  wealth=[np.nan, 2.0, 3.0])
    p = model.predict_proba(target)[:, 1]
    assert np.isfinite(p).all()
    imputed = model.named_steps["prep"].named_transformers_["ord"].named_steps["simpleimputer"].statistics_[0]
    assert imputed == np.median(train.wealth)  # imputation learned from training rows only
