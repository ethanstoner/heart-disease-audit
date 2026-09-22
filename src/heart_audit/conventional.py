"""The conventional pipeline: the survey's modal recipe (survey/modal_pipeline.json) as a callable.

Single unstratified 80/20 split, one-hot encoding, StandardScaler, default-hyperparameter
DT / LR / RF / SVM, random forest as the headline. `scale_scope="full"` fits the scaler on all
rows before the split, as most surveyed notebooks do; `"train"` fits it on the training rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from heart_audit.data import TARGET

CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
TEST_SIZE = 0.2
HEADLINE = "random_forest"
# Default hyperparameters; random_state is set only so that runs are reproducible.
MODELS = {
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "logistic_regression": lambda seed: LogisticRegression(),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
    "svm": lambda seed: SVC(random_state=seed),
}

ScaleScope = Literal["full", "train"]


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
    scale_scope: ScaleScope
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


def split_and_scale(df: pd.DataFrame, seed: int, scale_scope: ScaleScope = "full") -> Split:
    X, y = encode(df)
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=TEST_SIZE, random_state=seed)
    fit_rows = X if scale_scope == "full" else X.iloc[train_idx]
    scaler = StandardScaler().fit(fit_rows)
    Xs = scaler.transform(X)
    return Split(Xs[train_idx], Xs[test_idx], y[train_idx], y[test_idx], test_idx, scaler)


def run_conventional(df: pd.DataFrame, seed: int, scale_scope: ScaleScope = "full") -> Results:
    s = split_and_scale(df, seed, scale_scope)
    predictions = {name: make(seed).fit(s.X_train, s.y_train).predict(s.X_test) for name, make in MODELS.items()}
    accuracy = {name: float((p == s.y_test).mean()) for name, p in predictions.items()}
    return Results(seed, scale_scope, accuracy, predictions, s.test_index, s.y_test)


def reproduction_gate(accuracies, target: float) -> tuple[float, float, bool]:
    """Central 95% of the control's accuracy distribution, and whether `target` lies inside it."""
    lo, hi = np.percentile(accuracies, [2.5, 97.5])
    return float(lo), float(hi), bool(lo <= target <= hi)
