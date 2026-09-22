import numpy as np
import pandas as pd
import pytest

from heart_audit.data import FEATURES, TARGET, load_kaggle918, load_statlog270, load_uci920, to_kaggle_schema
from heart_audit.leakage import (
    contamination, duplicate_groups, fill_gap, matched_removal, oof_scores, reverted_frame, T7_MODELS,
    with_canonical_missing,
)
from heart_audit.provenance import filled_cells, match_sources


@pytest.fixture(scope="module")
def frames(raw_dir):
    kaggle, uci = load_kaggle918(raw_dir), to_kaggle_schema(load_uci920(raw_dir))
    matches = match_sources(kaggle, uci)
    return kaggle, uci, matches, filled_cells(kaggle, uci, matches)


def test_reverted_frame_sets_exactly_the_filled_cells_missing(frames):
    kaggle, _, _, cells = frames
    rev = reverted_frame(kaggle, cells)
    assert rev[FEATURES].isna().sum().sum() == len(cells)
    for column, group in cells.groupby("column"):
        assert rev[column].iloc[group["kaggle_row"]].isna().all()
    assert kaggle[FEATURES].isna().sum().sum() == 0


def test_matched_removal_preserves_outcome_counts(frames):
    kaggle, _, matches, cells = frames
    target = cells.loc[cells["column"] == "ST_Slope", "kaggle_row"].to_numpy()
    pool = np.setdiff1d(np.flatnonzero(matches["uci_row"].notna()), target)
    drawn = matched_removal(kaggle, target, pool, np.random.default_rng(0))
    y = kaggle[TARGET].to_numpy()
    assert len(drawn) == len(target) == 302
    assert y[drawn].sum() == y[target].sum() == 111
    assert np.isin(drawn, pool).all() and len(np.unique(drawn)) == len(drawn)


def test_fill_gap_on_synthetic_frame():
    kaggle = pd.DataFrame({"ST_Slope": ["Flat", "Up"] * 50, TARGET: [1, 0] * 50})
    matches = pd.DataFrame({"uci_row": np.arange(100), "source": ["va"] * 100})
    cells = pd.DataFrame({"kaggle_row": np.arange(50), "column": "ST_Slope"})
    out = fill_gap(kaggle, matches, cells, "ST_Slope", "Flat", "Up", n_boot=50, seed=0)
    assert out["gap_filled"] == 1.0 and out["gap_observed"] == 1.0 and out["difference"] == 0.0


def test_statlog_merge_duplicates(raw_dir, frames):
    _, uci, _, _ = frames
    merged = pd.concat([uci, load_statlog270(raw_dir)], ignore_index=True)
    groups = duplicate_groups(merged)
    assert len(merged) == 1190
    assert len(merged) - len(np.unique(groups)) == 272


def test_contamination_counts_test_rows_with_a_training_twin():
    groups = np.array([0, 0, 1, 2, 2, 3])
    assert contamination(groups, np.array([0, 2])) == 0.5      # row 0 has twin 1 in train; row 2 none
    assert contamination(groups, np.array([3, 4])) == 0.0      # twins both in test
    assert contamination(groups, np.array([3])) == 1.0


def test_canonical_missing_turns_zeros_into_nan(frames):
    kaggle = frames[0]
    X = with_canonical_missing(kaggle)
    assert X["Cholesterol"].isna().sum() == 172 and X["RestingBP"].isna().sum() == 1


def test_oof_scores_cover_every_row(frames):
    _, uci, _, _ = frames
    for scheme in ("loso", "kfold"):
        s = oof_scores(uci, scheme, seed=0, make_model=T7_MODELS["logistic_regression"])
        assert s.shape == (920,) and np.isfinite(s).all() and ((0 <= s) & (s <= 1)).all()
