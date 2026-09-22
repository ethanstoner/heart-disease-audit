import numpy as np
import pandas as pd
import pytest

from heart_audit.data import FEATURES, TARGET, load_kaggle918, load_uci920, to_kaggle_schema
from heart_audit.provenance import MATCH_KEY, filled_cells, filled_slopes, match_sources, slope_disagreements


def _row(age, chol, slope, source=None, fbs=0.0):
    r = {"Age": age, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 130.0, "Cholesterol": chol,
         "FastingBS": fbs, "RestingECG": "Normal", "MaxHR": 150.0, "ExerciseAngina": "N",
         "Oldpeak": 1.0, "ST_Slope": slope, TARGET: 1}
    if source:
        r["source"] = source
    return r


def _kaggle(rows):
    return pd.DataFrame(rows)[FEATURES + [TARGET]]


def test_matcher_cases():
    uci = pd.DataFrame([
        _row(40, 200.0, "Flat", "cleveland"),   # unique, slope present
        _row(50, np.nan, np.nan, "va"),         # chol and slope '?': matched on the recorded fields only
        _row(60, 250.0, "Up", "hungary"),       # two identical UCI rows: not one-to-one
        _row(60, 250.0, "Up", "hungary"),
    ])
    kaggle = _kaggle([
        _row(40, 200, "Flat"),
        _row(50, 231, "Up"),                    # filled with an ordinary-looking value
        _row(60, 250, "Up"),
        _row(70, 300, "Down"),                  # no candidate
    ])
    m = match_sources(kaggle, uci)
    assert m["source"].tolist() == ["cleveland", "va", "hungary", "unknown"]
    assert m["n_candidates"].tolist() == [1, 1, 2, 0]
    assert m["uci_row"].tolist()[:2] == [0, 1]
    assert m["uci_row"].isna().tolist()[2:] == [True, True]
    assert slope_disagreements(m) == 0
    assert filled_slopes(m).index.tolist() == [1]
    cells = filled_cells(kaggle, uci, m)
    assert sorted(zip(cells["kaggle_row"], cells["column"], cells["kaggle_value"])) == [
        (1, "Cholesterol", 231), (1, "ST_Slope", "Up"),
    ]


def test_uci_row_compatible_with_two_kaggle_rows_is_not_paired():
    uci = pd.DataFrame([_row(40, np.nan, "Flat", "va")])
    kaggle = _kaggle([_row(40, 200, "Flat"), _row(40, 210, "Flat")])
    m = match_sources(kaggle, uci)
    assert m["uci_row"].isna().all()
    assert m["source"].tolist() == ["va", "va"]


def test_disagreement_is_counted():
    uci = pd.DataFrame([_row(40, 200.0, "Flat", "cleveland")])
    assert slope_disagreements(match_sources(_kaggle([_row(40, 200, "Up")]), uci)) == 1


def test_candidates_from_two_sources_are_unknown():
    uci = pd.DataFrame([_row(40, 200.0, "Flat", "cleveland"), _row(40, 200.0, "Flat", "va")])
    assert match_sources(_kaggle([_row(40, 200, "Flat")]), uci)["source"].tolist() == ["unknown"]


@pytest.fixture(scope="module")
def frames(raw_dir):
    return load_kaggle918(raw_dir), to_kaggle_schema(load_uci920(raw_dir))


@pytest.fixture(scope="module")
def real(frames):
    return match_sources(*frames)


def test_real_match_counts(real):
    assert len(real) == 918
    assert real["uci_row"].notna().sum() == 912
    assert real["uci_row"].dropna().is_unique


def test_matcher_never_disagrees_where_uci_has_slope(real):
    agreed = real["uci_row"].notna() & real["uci_slope"].notna()
    assert agreed.sum() == 610
    assert slope_disagreements(real) == 0


def test_every_recovered_missing_value_was_filled(frames, real):
    kaggle, uci = frames
    counts = filled_cells(kaggle, uci, real)["column"].value_counts().to_dict()
    assert counts == {
        "ST_Slope": 302, "FastingBS": 89, "Oldpeak": 60, "RestingBP": 57,
        "MaxHR": 53, "ExerciseAngina": 53, "Cholesterol": 28, "RestingECG": 2,
    }
    assert kaggle.isna().sum().sum() == 0


def test_label_rule_survives_leaving_the_label_out_of_the_match(frames):
    """Matching without HeartDisease: the pairing cannot manufacture a link between filled values and the label."""
    kaggle, uci = frames
    no_label = [c for c in MATCH_KEY if c != TARGET]
    m = match_sources(kaggle, uci, key=no_label)
    paired = m["uci_row"].notna().to_numpy()
    uci_y = uci[TARGET].to_numpy()[m.loc[paired, "uci_row"].astype(int)]
    assert (kaggle[TARGET].to_numpy()[paired] == uci_y).all()          # labels agree, unprompted
    slope = filled_cells(kaggle, uci, m).query("column == 'ST_Slope'")
    y = kaggle[TARGET].to_numpy()[slope["kaggle_row"]]
    assert len(slope) == 296 and paired.sum() == 904
    assert ((slope["kaggle_value"] == "Flat") == (y == 1)).all()
