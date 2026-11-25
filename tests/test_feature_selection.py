"""Stage 29 Steps 1-4 on a made-up survey: column rules, special codes, the locked village split, the one-question
scores and the repeat rule."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.common.io import load_config
from src.survey.screen_features import fixed_by, group_codes, oof_group_mean, redundant_drops, weighted_auroc
from src.survey.select_features import feature_table, inventory, lock_test


def fake_survey():
    n = 40
    df = pd.DataFrame({
        "v457": [1, 4] * 20,                                   # the label
        "v456": np.linspace(90, 140, n),                       # adjusted Hb: leakage
        "hw53_1": np.linspace(90, 140, n),                     # a child's Hb: leakage
        "v001": np.repeat(np.arange(10), 4),                   # cluster: bookkeeping
        "v012": np.arange(15, 15 + n),                         # age: a number, kept
        "v106": [0, 1, 2, 3] * 10,                             # education: a category, kept
        "v115": [10, 20, 30, 40, 50, 60, 996, 998] * 5,        # minutes to water, a quarter labelled codes
        "b4_01": [1, 2] * 20,                                  # most recent birth: kept
        "b4_02": [1, 2] * 20,                                  # older birth: repeat
        "v999": [1] * n,                                       # constant
        "v998": [np.nan] * 35 + [1, 2, 1, 2, 1],               # mostly empty
        "v997": [np.nan] * n,                                  # not asked
    })
    meta = SimpleNamespace(
        column_names_to_labels={"v457": "anemia level", "v456": "hemoglobin level adjusted", "hw53_1": "hemoglobin level",
                                "v001": "cluster number", "v012": "respondent's current age",
                                "v106": "highest educational level", "v115": "time to get to water source",
                                "b4_01": "sex of child", "b4_02": "sex of child", "v999": "something constant",
                                "v998": "asked of few", "v997": "na - not in this country"},
        variable_value_labels={"v106": {0: "no education", 1: "primary", 2: "secondary", 3: "higher"},
                               "v115": {996: "on premises", 998: "don't know"},
                               "b4_01": {1: "male", 2: "female"}, "b4_02": {1: "male", 2: "female"}},
    )
    return df, meta


def test_inventory_reasons():
    df, meta = fake_survey()
    reasons = inventory(df, meta, [], load_config("feature_selection")).set_index("code").reason
    assert reasons["v457"] == "the anemia label"
    assert reasons["v456"] == reasons["hw53_1"] == "leakage: blood test"
    assert reasons["v001"] == "bookkeeping"
    assert reasons["b4_02"] == "repeat: older birth or child"
    assert reasons["v999"] == "constant"
    assert reasons["v998"] == "mostly empty"
    assert reasons["v997"] == "not asked in this survey"
    assert (reasons[["v012", "v106", "v115", "b4_01"]] == "").all()


def test_no_blood_test_column_reaches_the_models():
    df, meta = fake_survey()
    inv = inventory(df, meta, [], load_config("feature_selection"))
    feats = feature_table(df, inv, meta)
    assert not {"v457", "v456", "hw53_1", "v001"} & set(feats.columns)


def test_special_codes_move_to_a_side_column():
    df, meta = fake_survey()
    inv = inventory(df, meta, [], load_config("feature_selection"))
    feats = feature_table(df, inv, meta)
    assert inv.set_index("code").kind["v115"] == "number"
    assert feats.v115.max() == 60 and feats.v115.isna().sum() == 10      # 996 and 998 are not minutes
    assert set(feats.v115_special) == {"value", "on premises", "don't know"}
    assert set(feats.v106) == {"no education", "primary", "secondary", "higher"}


def test_locked_split_keeps_villages_whole():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 20)
    y = rng.integers(0, 2, len(groups))
    locked = lock_test(y, groups, 5, 42)
    assert not set(groups[locked]) & set(groups[~locked])
    assert 0.15 < locked.mean() < 0.25
    assert (locked == lock_test(y, groups, 5, 42)).all()                  # same seed, same villages


# Steps 3-4 helpers


def test_weighted_auroc_matches_sklearn_and_counts_weights():
    rng = np.random.default_rng(1)
    y, s = rng.integers(0, 2, 300), rng.integers(0, 5, 300).astype(float)  # many ties
    assert np.isclose(weighted_auroc(y, s, np.ones(300))[0], roc_auc_score(y, s))
    w = rng.integers(0, 3, 300).astype(float)
    rep = np.repeat(np.arange(300), w.astype(int))                          # weight 2 = the row twice
    assert np.isclose(weighted_auroc(y, s, w)[0], roc_auc_score(y[rep], s[rep]))


def test_group_rate_comes_from_other_folds_only():
    codes = np.array([0, 0, 1, 1])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    folds = [(0, np.array([1, 3]), np.array([0, 2])), (0, np.array([0, 2]), np.array([1, 3]))]
    assert np.allclose(oof_group_mean(codes, y, folds, 0), [1, 1, 0, 0])
    assert np.allclose(oof_group_mean(codes, y, folds, 1e9), [0.5] * 4)    # huge smoothing = the overall rate


def test_numbers_get_groups_with_blank_and_special_apart():
    feats = pd.DataFrame({"v115": [5.0, 10, 20, 40, np.nan, np.nan],
                          "v115_special": ["value", "value", "value", "value", "on premises", None]})
    codes, blank = group_codes(feats, "v115", 2)
    assert list(blank) == [False, False, False, False, False, True]
    assert codes[0] == codes[1] != codes[2] == codes[3]
    assert len({codes[4], codes[5], codes[0], codes[2]}) == 4


def test_repeat_rule_is_one_way():
    exposure = np.array([0, 1, 2, 3] * 100)            # 3 = pregnant
    pregnant = (exposure == 3).astype(int)
    assert fixed_by(pregnant, exposure, 10) == 1.0     # exposure fixes pregnant
    assert fixed_by(exposure, pregnant, 10) < 0.5      # not the other way round
    drops = redundant_drops({"exposure": exposure, "pregnant": pregnant}, ["exposure", "pregnant"], 0.95, 10)
    assert drops == {"pregnant": ("exposure", 1.0)}
    assert redundant_drops({"pregnant": pregnant, "exposure": exposure}, ["pregnant", "exposure"], 0.95, 10) == {}
    blanks = np.where(np.arange(400) < 395, -1, pregnant)
    assert fixed_by(blanks, exposure, 10) is None      # too few women answered both


# Step 5 helpers
def test_sections_follow_dhs_numbering():
    from src.survey.select_models import section_of
    sections = load_config("feature_selection")["models"]["sections"]
    assert section_of("hw2_1", sections).startswith("child body measures")
    assert section_of("h4_1", sections).startswith("child health")
    assert section_of("ml101", sections).startswith("bed nets")
    assert section_of("mmc1", sections).startswith("siblings")
    assert section_of("m45_1", sections).startswith("pregnancy and delivery")
    assert section_of("v3a05", sections).startswith("contraception")
    assert section_of("v754cp", sections).startswith("HIV")
    assert section_of("v714", sections).startswith("partner, work")


def test_l1_reports_the_questions_it_used():
    from src.survey.select_models import l1_questions, make_model, prepare
    rng = np.random.default_rng(0)
    n = 600
    signal = rng.integers(0, 2, n)
    X = prepare(pd.DataFrame({"v012": signal * 3 + rng.normal(0, 0.1, n), "v115": rng.normal(0, 1, n),
                              "v115_special": rng.choice(["value", "on premises"], n),
                              "v312": rng.choice(["pill", "none"], n)}))
    mc = load_config("feature_selection")["models"]
    model = make_model("l1_logistic_regression", X, mc, 0, C=0.01).fit(X, signal)
    assert l1_questions(model, X) == ["v012"]                       # the noise questions get zero weight


def test_blank_answers_become_a_category():
    from src.survey.select_models import prepare
    X = prepare(pd.DataFrame({"v312": ["pill", None], "v012": [20.0, np.nan]}))
    assert list(X.v312) == ["pill", "blank"] and np.isnan(X.v012[1])  # numbers keep blanks for the imputer


# Step 7 helper
def test_tabpfn_matrix_learns_categories_on_training_rows_only():
    from src.survey.select_models import prepare
    from src.survey.tabpfn_cv import tabpfn_matrix
    X = prepare(pd.DataFrame({"v012": [20.0, np.nan, 30.0, 40.0], "v312": ["pill", "none", None, "iud"]}))
    A, cat_idx = tabpfn_matrix(X, X.index[:3])
    assert cat_idx == [1]
    assert np.isnan(A[1, 0])                                  # blank number stays blank for TabPFN
    assert A[3, 1] == -1                                      # "iud" only appears in the test row: unseen
    assert len({A[0, 1], A[1, 1], A[2, 1]}) == 3              # pill, none, blank: three codes


# Step 8 helpers
def test_region_baseline_uses_training_women_only():
    from src.survey.repeated_cv import group_rate
    codes = np.array([0, 0, 1, 1, 0, 1])
    y = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])     # the test women (4, 5) disagree with their groups
    out = group_rate(codes, y, np.arange(4), np.array([4, 5]), 0)
    assert list(out) == [1.0, 0.0]


def test_area_statistics():
    from src.survey.area_check import spearman_brown, top_overlap, within_region
    v = pd.DataFrame({"village": range(8), "region": ["a"] * 4 + ["b"] * 4,
                      "share": [.1, .2, .3, .4, .5, .6, .7, .8], "score": [1, 2, 3, 4, 4, 3, 2, 1]})
    assert np.isclose(within_region(v), 0.0)                       # +1 in region a, -1 in region b
    assert top_overlap(v.assign(score=v.share), 0.25) == 1.0
    assert top_overlap(v.assign(score=-v.share), 0.25) == 0.0
    assert np.isclose(spearman_brown(0.5), 2 / 3)


def test_noise_ceiling_is_high_when_villages_differ_a_lot():
    from src.survey.area_check import noise_ceiling
    rng = np.random.default_rng(0)
    village = np.repeat(np.arange(40), 50)
    rate = np.repeat(np.linspace(0.02, 0.98, 40), 50)
    y = (rng.random(len(rate)) < rate).astype(int)
    c = noise_ceiling(y, village, np.where(village < 20, "a", "b"), 5, 0)
    assert c["overall"]["reliability"] > 0.9 and c["overall"]["max_correlation"] > 0.95


def test_hb_sources_question_sets():
    from types import SimpleNamespace

    from src.survey.hb_sources import question_set
    S = SimpleNamespace(base=["v012", "v213"], questions=["v012", "v024", "v190"])
    assert question_set("base_questions", S) == ["v012", "v213"]
    assert question_set(["base_questions", "v024"], S) == ["v012", "v213", "v024"]
    assert question_set("all", S) == ["v012", "v024", "v190"]


def test_grouped_importance_shuffles_a_group_together():
    from src.survey.hb_vs_class import grouped_importance, question_groups
    sections = load_config("feature_selection")["models"]["sections"]
    g = question_groups(["v024", "v139", "v190", "v312"], sections, {"v024", "v139", "v101"})
    assert g["region"] == ["v024", "v139"] and "v190" in g["background and household (v0xx-v1xx)"]

    class Sum:                                              # predicts a + b: only the pair carries the signal
        def predict(self, X):
            return X["a"].to_numpy() + X["b"].to_numpy()
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500), "c": rng.normal(size=500)})
    y = X.a + X.b
    score = lambda yy, m, XX: -float(np.mean(np.abs(yy - m.predict(XX))))
    imp = grouped_importance(Sum(), X, y.to_numpy(), score, {"ab": ["a", "b"], "c": ["c"]}, 2, 0)
    assert imp["ab"] > 0.5 and imp["c"] == 0


# Step 10 helpers
def test_severity_levels_follow_dhs():
    from src.survey.exploratory import severity_index
    hb = np.array([6.5, 9.9, 10.5, 11.5, 12.5, 10.5])
    cut = np.array([12.0, 12.0, 12.0, 12.0, 12.0, 11.0])     # the last woman is pregnant
    assert list(severity_index(hb, cut)) == [3, 2, 1, 1, 0, 1]


def test_village_means_use_answers_of_her_own_village():
    from src.survey.exploratory import village_means
    feats = pd.DataFrame({"v191": [1.0, 3.0, 10.0, 20.0], "v313": ["modern method", "no method", None, "modern method"]})
    vm = village_means(feats, np.array([1, 1, 2, 2]), {"v191": "mean", "v313": ["modern method"]})
    assert list(vm.village_v191) == [2.0, 2.0, 15.0, 15.0]
    assert list(vm.village_v313) == [0.5, 0.5, 1.0, 1.0]            # a blank answer is left out, not counted as "no"


def test_ordinal_model_ranks_by_severity():
    from src.survey.exploratory import ordinal_prob
    from src.survey.select_models import prepare
    rng = np.random.default_rng(0)
    x = rng.normal(size=800)
    level = np.digitize(x + rng.normal(0, 0.5, 800), [-0.5, 0.5, 1.5])  # 0..3, higher x = more severe
    X = prepare(pd.DataFrame({"v012": x}))
    p = ordinal_prob(X, level, np.arange(600), np.arange(600, 800), load_config("feature_selection")["models"], 0, 2)
    assert roc_auc_score(level[600:] >= 2, p) > 0.85


# Stage 30 helpers
def test_paths_keep_zimbabwe_names_and_separate_other_countries():
    from src.survey.select_features import OUT, paths
    assert paths("zimbabwe")["out"] == OUT and paths("zimbabwe")["results"] == "zimbabwe_inventory"
    m = paths("malawi")
    assert m["out"].name == "malawi_clean.parquet" and "zimbabwe" not in str(m["inventory"])


def test_country_model_list_and_labels():
    from types import SimpleNamespace

    from src.survey.countries import label, model_list
    C = SimpleNamespace(base=["v012"], questions=["v012", "v024"], places=["v024", "sdistrict"], extra=["v861"])
    ids = [m[0] for m in model_list(C)]
    assert {"place[v024]", "place[sdistrict]", "lr_base_place[sdistrict]", "gbc_all_extra", "hb_all_extra"} <= set(ids)
    assert "v861" not in model_list(C)[2][1]                 # the main all-questions model leaves the extra out
    assert label("place[sdistrict]") == "place share, sdistrict (no model)"
    assert label("lr_base_place[v024]") == "base 6 + v024, logistic regression"
