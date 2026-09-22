import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from heart_audit.conventional import (
    HEADLINE, MODELS, accuracy_table, impute, reproduction_gate, run_conventional, run_seeds, split_and_scale,
)
from heart_audit.data import PROJECT_ROOT, TARGET, load_kaggle918


@pytest.fixture(scope="module")
def kaggle(raw_dir):
    return load_kaggle918(raw_dir)


def test_same_seed_same_results(kaggle):
    a, b = run_conventional(kaggle, seed=7), run_conventional(kaggle, seed=7)
    assert a.accuracy == b.accuracy
    for m in MODELS:
        assert np.array_equal(a.predictions[m], b.predictions[m])


def test_split_is_the_unstratified_80_20_split(kaggle):
    r = run_conventional(kaggle, seed=11)
    _, expected = train_test_split(np.arange(len(kaggle)), test_size=0.2, random_state=11)
    assert np.array_equal(r.test_index, expected)
    assert len(r.test_index) == 184
    assert np.array_equal(r.y_test, kaggle[TARGET].to_numpy()[expected])


def test_all_models_scored(kaggle):
    r = run_conventional(kaggle, seed=3)
    assert set(r.accuracy) == set(MODELS) == {"decision_tree", "logistic_regression", "random_forest", "svm"}
    for m in MODELS:
        assert r.predictions[m].shape == (184,)
        assert r.accuracy[m] == pytest.approx((r.predictions[m] == r.y_test).mean())
    assert r.headline == r.accuracy[HEADLINE] and HEADLINE == "random_forest"


def test_train_scope_scaler_never_sees_test_rows(kaggle):
    seed = 5
    _, test_idx = train_test_split(np.arange(len(kaggle)), test_size=0.2, random_state=seed)
    planted = kaggle.copy()
    planted.loc[planted.index[test_idx[0]], "MaxHR"] = 1e9
    col = "MaxHR"
    clean_train = split_and_scale(kaggle, seed, "train").scaler
    planted_train = split_and_scale(planted, seed, "train").scaler
    planted_full = split_and_scale(planted, seed, "full").scaler
    i = list(clean_train.feature_names_in_).index(col)
    assert planted_train.mean_[i] == clean_train.mean_[i]
    assert planted_full.mean_[i] > 1e6


def test_reproduction_gate():
    accs = np.arange(1, 1001) / 1000          # 0.001 .. 1.000
    lo, hi, passed = reproduction_gate(accs, 0.5)
    assert lo == pytest.approx(np.percentile(accs, 2.5)) and hi == pytest.approx(np.percentile(accs, 97.5))
    assert passed
    assert not reproduction_gate(accs, 0.99)[2]


def test_default_run_matches_committed_control(kaggle):
    committed = pd.read_csv(PROJECT_ROOT / "results" / "control_seeds.csv").head(3)
    for row in committed.to_dict("records"):
        acc = run_conventional(kaggle, int(row["seed"])).accuracy
        assert all(acc[m] == pytest.approx(row[m]) for m in MODELS)


def test_train_scope_imputation_never_sees_test_rows(kaggle):
    seed = 9
    _, test_idx = train_test_split(np.arange(len(kaggle)), test_size=0.2, random_state=seed)
    train_idx = np.setdiff1d(np.arange(len(kaggle)), test_idx)
    planted = kaggle.copy()
    planted.loc[planted.index[test_idx[:40]], "Cholesterol"] = 9000
    zero = planted["Cholesterol"].eq(0).to_numpy()
    imputed_train = impute(planted, train_idx)
    imputed_full = impute(planted, np.arange(len(planted)))
    expected = planted["Cholesterol"].iloc[train_idx][~zero[train_idx]].median()
    assert (imputed_train["Cholesterol"][zero] == expected).all()
    assert (imputed_full["Cholesterol"][zero] > expected).all()


def test_missing_values_require_an_impute_scope(kaggle):
    holed = kaggle.copy()
    holed.loc[holed.index[0], "MaxHR"] = np.nan
    with pytest.raises(ValueError):
        run_conventional(holed, 1)
    r = run_conventional(holed, 1, impute_scope="train", models=("random_forest",))
    assert set(r.accuracy) == {"random_forest"}


def test_grouped_split_keeps_groups_together(kaggle):
    groups = np.arange(len(kaggle)) // 2
    r = run_conventional(kaggle, 4, groups=groups, models=("logistic_regression",))
    test_groups = set(groups[r.test_index])
    train_groups = set(groups[np.setdiff1d(np.arange(len(kaggle)), r.test_index)])
    assert not test_groups & train_groups


def test_run_seeds_preserves_order(kaggle):
    out = run_seeds(kaggle, [5, 3], n_jobs=2, models=("decision_tree",))
    assert [r.seed for r in out] == [5, 3]
    assert list(accuracy_table(out).columns) == ["seed", "decision_tree"]
