"""The conventional pipeline: the survey's modal recipe (survey/modal_pipeline.json) as a callable.

Single unstratified 80/20 split, one-hot encoding, StandardScaler, default-hyperparameter
DT / LR / RF / SVM, random forest as the headline. `scale_scope="full"` fits the scaler on all
rows before the split, as most surveyed notebooks do; `"train"` fits it on the training rows.

Options used only by the teardown experiments: `impute_scope` imputes every canonically
missing value (median / mode, from all rows or training rows) before encoding; `groups`
switches to a split that keeps each group on one side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart_audit.data import FEATURES, TARGET, missing_mask

CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
TEST_SIZE = 0.2
HEADLINE = "random_forest"
# Default hyperparameters; random_state is set only so that runs are reproducible.
MODELS = {
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
    "svm": lambda seed: SVC(random_state=seed),
}

Scope = Literal["full", "train"]


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    test_index: np.ndarray
    scaler: StandardScaler


@dataclass
class Results:
    seed: int
    scale_scope: Scope
    accuracy: dict[str, float]
    predictions: dict[str, np.ndarray]
    test_index: np.ndarray
    y_test: np.ndarray

    @property
    def headline(self) -> float:
        return self.accuracy[HEADLINE]


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.get_dummies(df.drop(columns=[TARGET]), columns=CATEGORICAL).astype(float)
    return X, df[TARGET].to_numpy()


def split_indices(n: int, seed: int, groups: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if groups is None:
        return train_test_split(np.arange(n), test_size=TEST_SIZE, random_state=seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train, test = next(splitter.split(np.arange(n), groups=groups))
    return train, test


def impute(df: pd.DataFrame, stat_rows: np.ndarray) -> pd.DataFrame:
    """Replace canonically missing values with the median (numeric) or mode (categorical)
    of the non-missing values in `stat_rows`."""
    df = df.copy()
    missing = missing_mask(df)
    for col in FEATURES:
        observed = df[col].iloc[stat_rows][~missing[col].iloc[stat_rows].to_numpy()]
        fill = observed.median() if col in NUMERIC else observed.mode().iloc[0]
        df[col] = df[col].mask(missing[col], fill)
    return df


def split_and_scale(df: pd.DataFrame, seed: int, scale_scope: Scope = "full",
                    impute_scope: Scope | None = None, groups: np.ndarray | None = None) -> Split:
    df = df[FEATURES + [TARGET]].reset_index(drop=True)
    train_idx, test_idx = split_indices(len(df), seed, groups)
    if impute_scope is not None:
        df = impute(df, np.arange(len(df)) if impute_scope == "full" else train_idx)
    elif df[FEATURES].isna().any().any():
        raise ValueError("frame has missing values: pass impute_scope")
    X, y = encode(df)
    fit_rows = X if scale_scope == "full" else X.iloc[train_idx]
    scaler = StandardScaler().fit(fit_rows)
    Xs = scaler.transform(X)
    return Split(Xs[train_idx], Xs[test_idx], y[train_idx], y[test_idx], test_idx, scaler)


def run_conventional(df: pd.DataFrame, seed: int, scale_scope: Scope = "full",
                     impute_scope: Scope | None = None, groups: np.ndarray | None = None,
                     models: tuple[str, ...] = tuple(MODELS)) -> Results:
    s = split_and_scale(df, seed, scale_scope, impute_scope, groups)
    predictions = {name: MODELS[name](seed).fit(s.X_train, s.y_train).predict(s.X_test) for name in models}
    accuracy = {name: float((p == s.y_test).mean()) for name, p in predictions.items()}
    return Results(seed, scale_scope, accuracy, predictions, s.test_index, s.y_test)


def run_seeds(df: pd.DataFrame, seeds, n_jobs: int = -1, **kwargs) -> list[Results]:
    """run_conventional over many split seeds, in parallel, in seed order."""
    return Parallel(n_jobs=n_jobs)(delayed(run_conventional)(df, s, **kwargs) for s in seeds)


def accuracy_table(results: list[Results]) -> pd.DataFrame:
    return pd.DataFrame([{"seed": r.seed, **r.accuracy} for r in results])


def reproduction_gate(accuracies, target: float) -> tuple[float, float, bool]:
    """Central 95% of the control's accuracy distribution, and whether `target` lies inside it."""
    lo, hi = np.percentile(accuracies, [2.5, 97.5])
    return float(lo), float(hi), bool(lo <= target <= hi)
