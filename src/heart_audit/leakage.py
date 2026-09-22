"""Teardown experiments T1–T7. Pre-registered in analysis/PREDICTIONS.md.

Functions here are pure and fast; the multi-seed runs live in scripts/run_teardown.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_audit.conventional import CATEGORICAL, NUMERIC
from heart_audit.data import FEATURES, TARGET, missing_mask

FILLED_SOURCES = ("hungary", "va")


# --- T1: undisclosed imputation -------------------------------------------------------

def filled_mask(kaggle: pd.DataFrame, cells: pd.DataFrame, column: str) -> np.ndarray:
    mask = np.zeros(len(kaggle), dtype=bool)
    mask[cells.loc[cells["column"] == column, "kaggle_row"].to_numpy()] = True
    return mask


def fill_gap(kaggle: pd.DataFrame, matches: pd.DataFrame, cells: pd.DataFrame, column: str,
             a, b, n_boot: int, seed: int, sources=FILLED_SOURCES) -> dict:
    """P1.1: disease-rate gap between values a and b, filled vs observed, within `sources`."""
    y = kaggle[TARGET].to_numpy()
    values = kaggle[column].to_numpy()
    in_scope = matches["uci_row"].notna().to_numpy() & matches["source"].isin(sources).to_numpy()
    filled = filled_mask(kaggle, cells, column) & in_scope
    observed = ~filled_mask(kaggle, cells, column) & in_scope

    def gap(rows):
        ya, yb = y[rows & (values == a)], y[rows & (values == b)]
        return ya.mean() - yb.mean(), len(ya), len(yb)

    gf, nfa, nfb = gap(filled)
    go, noa, nob = gap(observed)
    rng = np.random.default_rng(seed)
    f_idx, o_idx = np.flatnonzero(filled), np.flatnonzero(observed)
    boots = []
    for _ in range(n_boot):
        fb = np.zeros(len(y), dtype=int)
        ob = np.zeros(len(y), dtype=int)
        np.add.at(fb, rng.choice(f_idx, len(f_idx)), 1)
        np.add.at(ob, rng.choice(o_idx, len(o_idx)), 1)

        def wgap(w):
            wa, wb = w * (values == a), w * (values == b)
            return (wa @ y) / wa.sum() - (wb @ y) / wb.sum()

        boots.append(wgap(fb) - wgap(ob))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "column": column, "a": a, "b": b,
        "gap_filled": gf, "gap_observed": go, "difference": gf - go, "ci95": [lo, hi],
        "n_filled": [nfa, nfb], "n_observed": [noa, nob],
        "small_group": min(nfa, nfb, noa, nob) < 20,
    }


def fill_table(kaggle: pd.DataFrame, matches: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Per column and source: filled count, disease rate among filled rows vs observed rows."""
    y = kaggle[TARGET].to_numpy()
    paired = matches["uci_row"].notna().to_numpy()
    rows = []
    for column in cells["column"].unique():
        filled = filled_mask(kaggle, cells, column)
        for source in sorted(matches.loc[filled, "source"].unique()):
            here = paired & (matches["source"] == source).to_numpy()
            rows.append({
                "column": column, "source": source,
                "n_filled": int((filled & here).sum()),
                "disease_rate_filled": float(y[filled & here].mean()),
                "n_observed": int((~filled & here).sum()),
                "disease_rate_observed": float(y[~filled & here].mean()) if (~filled & here).any() else np.nan,
            })
    return pd.DataFrame(rows)


def reverted_frame(kaggle: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """The published CSV with every recovered filled cell set back to missing."""
    out = kaggle.copy()
    for column, group in cells.groupby("column"):
        out[column] = out[column].astype("object" if column in CATEGORICAL else float)
        out.loc[out.index[group["kaggle_row"].to_numpy()], column] = np.nan
    return out


def matched_removal(kaggle: pd.DataFrame, target_rows: np.ndarray, pool: np.ndarray,
                    rng: np.random.Generator) -> np.ndarray:
    """Rows drawn from `pool` with the same HeartDisease counts as `target_rows` (deviation D5)."""
    key = kaggle[TARGET]
    chosen = []
    for k, n in key.iloc[target_rows].value_counts().items():
        candidates = pool[key.iloc[pool].to_numpy() == k]
        chosen.append(rng.choice(candidates, n, replace=False))
    return np.sort(np.concatenate(chosen))


# --- T2: provenance leak ---------------------------------------------------------------

def cv_oof_proba(X, y, seed: int, make_model=lambda: LogisticRegression(max_iter=1000)) -> np.ndarray:
    """Pooled out-of-fold probabilities from stratified 10-fold CV (shuffled)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    oof = np.empty(len(y))
    for train, test in StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y):
        oof[test] = make_model().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return oof


def source_design(uci: pd.DataFrame) -> np.ndarray:
    return pd.get_dummies(uci["source"]).astype(float).to_numpy()


def missingness_design(frame: pd.DataFrame) -> np.ndarray:
    return missing_mask(frame).astype(float).to_numpy()


# --- T3: duplicates --------------------------------------------------------------------

def duplicate_groups(frame: pd.DataFrame) -> np.ndarray:
    """Group id per row; rows equal on all features and the target (missing equal missing) share one."""
    cols = frame[FEATURES + [TARGET]].astype(object)
    key = cols.where(cols.notna(), "NA").astype(str).agg("|".join, axis=1)
    return pd.factorize(key)[0]


def contamination(groups: np.ndarray, test_index: np.ndarray) -> float:
    """Share of test rows whose duplicate group also appears among the training rows."""
    train = np.setdiff1d(np.arange(len(groups)), test_index)
    return float(np.isin(groups[test_index], groups[train]).mean())


# --- T7: fold structure ----------------------------------------------------------------

def honest_pipeline(model):
    """Train-fold imputation, one-hot and scaling around `model`, for the 11 raw features."""
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    return make_pipeline(prep, model)


def with_canonical_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Features with every canonically missing value as NaN (zeros in Cholesterol / RestingBP)."""
    X = frame[FEATURES].copy()
    for col in ("Cholesterol", "RestingBP"):
        X[col] = X[col].mask(X[col].eq(0))
    for col in CATEGORICAL:
        X[col] = X[col].astype(object)
    return X


def oof_scores(frame: pd.DataFrame, scheme: str, seed: int, make_model) -> np.ndarray:
    """Out-of-fold P(disease) under 'loso' (leave one source out) or 'kfold' (stratified 10-fold)."""
    X, y = with_canonical_missing(frame), frame[TARGET].to_numpy()
    if scheme == "loso":
        splits = LeaveOneGroupOut().split(X, y, groups=frame["source"])
    else:
        splits = StratifiedKFold(10, shuffle=True, random_state=seed).split(X, y)
    oof = np.empty(len(y))
    for train, test in splits:
        model = honest_pipeline(make_model()).fit(X.iloc[train], y[train])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
    return oof


T7_MODELS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(random_state=0),
}
