"""Stage 30 (src/survey/countries.py) and the small shared report helpers, on made-up data."""
import json

import numpy as np
import pandas as pd

from src.common.io import load_config


def _women(n, seed):
    """A made-up Stage 7-coded table: the 6 base questions, labels, village and locked flag."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(15, 50, n), "pregnant": rng.integers(0, 2, n),
        "wealth": rng.integers(1, 6, n).astype(float), "education": rng.integers(0, 4, n),
        "contraception": rng.choice(["not using", "pill", "injections"], n),
        "residence": rng.choice(["urban", "rural"], n),
        "hb_adjusted": rng.uniform(7.0, 15.0, n), "cluster": np.repeat(np.arange(n // 10), 10),
    })
    df["anemic"] = (df.hb_adjusted < np.where(df.pregnant == 1, 11.0, 12.0)).astype(int)
    locked = np.isin(df.cluster, [0, 1, 2])
    return df, locked


def _table(df, locked, survey):
    from src.survey.train_baselines import feature_frame
    return feature_frame(df, survey), df, locked


def test_transfer_scores_use_the_common_severity_cutoff_and_score_only_locked_women():
    from src.survey.countries import transfer_scores

    survey, cc = load_config("survey"), load_config("countries")
    zim, home = _table(*_women(200, 0), survey), _table(*_women(150, 1), survey)
    df, locked = home[1], home[2]

    y, s_home, s_zim = transfer_scores(zim, home, "moderate or severe", cc, survey, seed=0)
    expected = (df.hb_adjusted < cc["severity_cutoff_g_dl"]).astype(int).to_numpy()[locked]  # same rule, pregnant or not
    assert cc["severity_cutoff_g_dl"] == 10.0
    assert (y == expected).all()
    assert len(s_home) == len(s_zim) == locked.sum()

    y_any, _, _ = transfer_scores(zim, home, "any anemia", cc, survey, seed=0)
    assert (y_any == df.anemic.to_numpy()[locked]).all()


def test_home_model_never_sees_the_locked_labels():
    from src.survey.countries import transfer_scores

    survey, cc = load_config("survey"), load_config("countries")
    zim = _table(*_women(200, 0), survey)
    df, locked = _women(150, 1)
    flipped = df.copy()
    flipped.loc[locked, "anemic"] = 1 - flipped.loc[locked, "anemic"]  # change only the locked women's answers

    _, s_a, z_a = transfer_scores(zim, _table(df, locked, survey), "any anemia", cc, survey, seed=0)
    _, s_b, z_b = transfer_scores(zim, _table(flipped, locked, survey), "any anemia", cc, survey, seed=0)
    assert np.allclose(s_a, s_b) and np.allclose(z_a, z_b)


def test_pregnant_first_rule_ranks_every_pregnant_woman_first():
    from src.survey.train_baselines import rule_scores

    df = pd.DataFrame({"pregnant": [0, 1, 0, 1, 0, 0]})
    scores = rule_scores("pregnant_first", df, n_repeats=3, seed=0)
    for s in scores:
        assert s[df.pregnant == 1].min() > s[df.pregnant == 0].max()


def test_markdown_table_marks_missing_metrics():
    from src.common.report import markdown_table

    row = {"model": "base 6", "auroc": {"estimate": 0.58, "lo": 0.566, "hi": 0.594}}
    lines = markdown_table([row], ["model"]).split("\n")
    assert lines[0] == "| model | AUROC | PR-AUC | Sens @ 90% spec |"
    assert lines[2] == "| base 6 | 0.580 [0.566, 0.594] | - | - |"


def test_short_commit_flags_uncommitted_changes(tmp_path):
    from src.common.report import short_commit

    clean, dirty = tmp_path / "clean.json", tmp_path / "dirty.json"
    clean.write_text(json.dumps({"git": {"commit": "abcdef1234", "dirty": False}}))
    dirty.write_text(json.dumps({"git": {"commit": "abcdef1234", "dirty": True}}))
    assert short_commit(clean) == "abcdef1"
    assert short_commit(dirty) == "abcdef1 + uncommitted changes"


def test_severity_curve_perfect_score_and_too_few_cases():
    from src.survey.severity_curve import curve_points

    rng = np.random.default_rng(0)
    hb = rng.uniform(6.0, 15.0, 600)
    villages = np.repeat(np.arange(60), 10)
    boot = {"n_resamples": 100, "ci": 0.95}
    rows = curve_points(hb, -hb, villages, [12.0, 10.0, 6.5], control=12.0, min_cases=30, boot=boot, seed=0)
    assert [r["cases"] for r in rows] == [int((hb < c).sum()) for c in (12.0, 10.0, 6.5)]
    for r in rows[:2]:                      # lower true Hb = higher score: perfect separation at every cutoff
        assert r["vs_rest"]["estimate"] == r["vs_not_anemic"]["estimate"] == 1.0
    assert rows[2]["cases"] < 30 and rows[2]["vs_rest"] is None   # too few cases: no point drawn
