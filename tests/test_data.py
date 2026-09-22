import pytest

from heart_audit.data import (
    FEATURES,
    TARGET,
    load_kaggle918,
    load_uci920,
    missing_mask,
    to_kaggle_schema,
)


@pytest.fixture(scope="module")
def uci(raw_dir):
    return to_kaggle_schema(load_uci920(raw_dir))


@pytest.fixture(scope="module")
def kaggle(raw_dir):
    return load_kaggle918(raw_dir)


def _by_source(series, uci):
    return series.groupby(uci["source"]).sum().to_dict()


def test_source_row_counts(uci):
    assert uci["source"].value_counts().to_dict() == {
        "cleveland": 303, "hungary": 294, "va": 200, "switzerland": 123,
    }


def test_eleven_features_plus_target_and_source(uci):
    assert FEATURES == [
        "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
        "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
    ]
    assert list(uci.columns) == FEATURES + [TARGET, "source"]


def test_cholesterol_missing_counts_under_canonical_encoding(uci):
    # Switzerland records missing cholesterol as 0, not '?': both encodings must count.
    assert _by_source(missing_mask(uci)["Cholesterol"], uci) == {
        "cleveland": 0, "hungary": 23, "switzerland": 123, "va": 56,
    }


def test_cholesterol_missing_rates_match_spec(uci):
    rates = (missing_mask(uci)["Cholesterol"].groupby(uci["source"]).mean() * 100).round(1)
    assert rates.to_dict() == {"cleveland": 0.0, "hungary": 7.8, "switzerland": 100.0, "va": 28.0}


def test_slope_missing_counts(uci):
    m = missing_mask(uci)["ST_Slope"]
    assert _by_source(m, uci) == {"cleveland": 0, "hungary": 190, "switzerland": 17, "va": 102}
    assert m.sum() == 309


def test_target_is_num_greater_than_zero(uci):
    assert set(uci[TARGET].unique()) == {0, 1}
    assert uci[TARGET].sum() == 509


def test_categorical_codes_mapped(uci):
    assert set(uci["ChestPainType"].dropna()) == {"TA", "ATA", "NAP", "ASY"}
    assert set(uci["ST_Slope"].dropna()) == {"Up", "Flat", "Down"}
    assert set(uci["Sex"]) == {"M", "F"}


def test_kaggle_shape_and_columns(kaggle):
    assert kaggle.shape == (918, 12)
    assert list(kaggle.columns) == FEATURES + [TARGET]


def test_kaggle_has_no_missing_slope(kaggle):
    assert kaggle["ST_Slope"].isna().sum() == 0
    assert set(kaggle["ST_Slope"]) == {"Up", "Flat", "Down"}


def test_kaggle_zero_cholesterol_is_a_disease_marker(kaggle):
    zero = missing_mask(kaggle)["Cholesterol"]
    assert zero.sum() == 172
    assert round(kaggle.loc[zero, TARGET].mean() * 100, 1) == 88.4
    assert round(kaggle.loc[~zero, TARGET].mean() * 100, 1) == 47.7
