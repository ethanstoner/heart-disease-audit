import numpy as np
import pandas as pd
import pytest

from heart_audit.data import FEATURES, TARGET, load_kaggle918, load_uci920, to_kaggle_schema
from heart_audit.provenance import filled_slopes, match_sources, slope_disagreements


def _row(age, chol, slope, source=None):
    r = {"Age": age, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 130.0, "Cholesterol": chol,
         "FastingBS": 0.0, "RestingECG": "Normal", "MaxHR": 150.0, "ExerciseAngina": "N",
         "Oldpeak": 1.0, "ST_Slope": slope, TARGET: 1}
    if source:
        r["source"] = source
    return r


def test_matcher_cases():
    uci = pd.DataFrame([
        _row(40, 200.0, "Flat", "cleveland"),   # unique, slope present
        _row(50, np.nan, np.nan, "va"),         # chol '?' -> Kaggle 0; slope was '?'
        _row(60, 250.0, "Up", "hungary"),       # two candidates, same source
        _row(60, 250.0, "Up", "hungary"),
    ])
    kaggle = pd.DataFrame([
        _row(40, 200, "Flat"),
        _row(50, 0, "Up"),
        _row(60, 250, "Up"),
        _row(70, 300, "Down"),                  # no candidate
    ])[FEATURES + [TARGET]]
    m = match_sources(kaggle, uci)
    assert m["source"].tolist() == ["cleveland", "va", "hungary", "unknown"]
    assert m["n_candidates"].tolist() == [1, 1, 2, 0]
    assert m["uci_row"].tolist()[:2] == [0, 1]
    assert m["uci_row"].isna().tolist()[2:] == [True, True]
    assert slope_disagreements(m) == 0
    filled = filled_slopes(m)
    assert filled.index.tolist() == [1]
    assert filled["kaggle_slope"].tolist() == ["Up"]


def test_disagreement_is_counted():
    uci = pd.DataFrame([_row(40, 200.0, "Flat", "cleveland")])
    kaggle = pd.DataFrame([_row(40, 200, "Up")])[FEATURES + [TARGET]]
    assert slope_disagreements(match_sources(kaggle, uci)) == 1


def test_candidates_from_two_sources_are_unknown():
    uci = pd.DataFrame([_row(40, 200.0, "Flat", "cleveland"), _row(40, 200.0, "Flat", "va")])
    kaggle = pd.DataFrame([_row(40, 200, "Flat")])[FEATURES + [TARGET]]
    assert match_sources(kaggle, uci)["source"].tolist() == ["unknown"]


@pytest.fixture(scope="module")
def real(raw_dir):
    return match_sources(load_kaggle918(raw_dir), to_kaggle_schema(load_uci920(raw_dir)))


def test_real_match_counts(real):
    assert len(real) == 918
    assert (real["n_candidates"] > 0).sum() == 739
    assert (real["n_candidates"] == 1).sum() == 738
    assert (real["source"] == "unknown").sum() == 179


def test_matcher_never_disagrees_where_uci_has_slope(real):
    agreed = real["uci_row"].notna() & real["uci_slope"].notna()
    assert agreed.sum() == 531
    assert slope_disagreements(real) == 0


def test_recovered_fill_in(real):
    filled = filled_slopes(real)
    assert len(filled) == 207
    assert filled["kaggle_slope"].value_counts().to_dict() == {"Up": 150, "Flat": 57}
    assert filled["source"].value_counts().to_dict() == {"hungary": 166, "va": 41}
