import numpy as np
import pandas as pd
import pytest

from heart_audit.baseline import GRIDS, design, loso_splits, nested_fit, pipeline, repeated_splits, youden_threshold
from heart_audit.data import TARGET, load_uci920, to_kaggle_schema


@pytest.fixture(scope="module")
def uci(raw_dir):
    return to_kaggle_schema(load_uci920(raw_dir))


def test_arms(uci):
    a, b = design(uci, "A"), design(uci, "B")
    assert "Cholesterol" in a and "Cholesterol" not in b
    assert not any(c.startswith("missing_") for c in b)
    assert a["missing_Cholesterol"].sum() == 0 + 23 + 123 + 56


def test_loso_is_four_disjoint_source_folds(uci):
    splits = loso_splits(uci)
    assert len(splits) == 4
    for train, test in splits:
        assert len(set(uci["source"].iloc[test])) == 1
        assert not set(uci["source"].iloc[test]) & set(uci["source"].iloc[train])


def test_repeated_splits_shape(uci):
    splits = repeated_splits(uci, seed=0)
    assert len(splits) == 50
    for r in range(5):
        tests = np.concatenate([t for _, t in splits[r * 10:(r + 1) * 10]])
        assert sorted(tests) == list(range(len(uci)))


def test_held_out_site_with_a_fully_missing_feature(uci):
    """Switzerland has no cholesterol at all; the pipeline must still fit and predict."""
    train = uci["source"] != "switzerland"
    for arm in ("A", "B"):
        X = design(uci, arm)
        pipe = pipeline(GRIDS["logistic_regression"][0](0), list(X.columns))
        pipe.fit(X[train], uci.loc[train, TARGET])
        p = pipe.predict_proba(X[~train])[:, 1]
        assert np.isfinite(p).all() and len(p) == 123


def test_feature_missing_in_every_training_row_and_unseen_level(uci):
    X = design(uci, "A").copy()
    X.loc[:, "MaxHR"] = np.nan                         # constant-missing in training
    X.loc[X.index[:5], "RestingECG"] = "NEW"           # unseen category at test time
    pipe = pipeline(GRIDS["logistic_regression"][0](0), list(X.columns))
    pipe.fit(X.iloc[5:], uci[TARGET].iloc[5:])
    assert np.isfinite(pipe.predict_proba(X.iloc[:5])[:, 1]).all()


def test_imputation_never_sees_held_out_rows(uci):
    """Sentinel: a huge value in the held-out site must not move the fitted median."""
    X = design(uci, "B")
    test = (uci["source"] == "va").to_numpy()
    planted = X.copy()
    planted.loc[test, "MaxHR"] = 1e6
    fitted = []
    for frame in (X, planted):
        pipe = pipeline(GRIDS["logistic_regression"][0](0), list(frame.columns))
        pipe.fit(frame[~test], uci.loc[~test, TARGET])
        fitted.append(pipe.named_steps["prep"].named_transformers_["num"][0].statistics_)
    assert np.array_equal(fitted[0], fitted[1])


def test_nested_fit_loso_runs(uci):
    out = nested_fit(uci, "B", "logistic_regression", loso_splits(uci), seed=0)
    assert np.isfinite(out.oof).all() and len(out.best_params) == 4
    assert set(out.fold) == {0, 1, 2, 3}


def test_youden_hand_case():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert youden_threshold(y, p) == 0.35         # J = 0.5 at 0.35 and 0.8; first wins
    assert youden_threshold(np.array([0, 1]), np.array([0.2, 0.9])) == 0.9
