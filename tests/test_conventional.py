import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from heart_audit.conventional import HEADLINE, MODELS, reproduction_gate, run_conventional, split_and_scale
from heart_audit.data import TARGET, load_kaggle918


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
