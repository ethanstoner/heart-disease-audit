"""The honest baseline: nested cross-validation on the 920-row UCI frame with provenance.

Primary: leave-one-source-out outer loop. Control: 10-fold x 5 repeated stratified CV.
Inner loop in both: leave-one-site-out over the training sites, scoring ROC AUC.
Arm A keeps provenance proxies (missingness indicators). Arm B drops them and drops
Cholesterol entirely, because a median-imputed zero still marks the hospital.
Missing values are imputed inside the pipeline (median / mode) from training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from heart_audit.conventional import CATEGORICAL, NUMERIC
from heart_audit.data import FEATURES, TARGET, missing_mask
from heart_audit.leakage import with_canonical_missing

Arm = Literal["A", "B"]

# Grids are fixed here, before any baseline result exists.
GRIDS = {
    "logistic_regression": (lambda s: LogisticRegression(max_iter=2000),
                            {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
    "knn": (lambda s: KNeighborsClassifier(),
            {"clf__n_neighbors": [5, 15, 31, 51], "clf__weights": ["uniform", "distance"]}),
    "xgboost": (lambda s: XGBClassifier(random_state=s, n_jobs=1),
                {"clf__max_depth": [2, 3, 4], "clf__n_estimators": [100, 300],
                 "clf__learning_rate": [0.05, 0.1]}),
    "mlp": (lambda s: MLPClassifier(max_iter=2000, early_stopping=True, random_state=s),
            {"clf__hidden_layer_sizes": [(16,), (32,), (16, 16)], "clf__alpha": [1e-4, 1e-2]}),
}
MODEL_NAMES = list(GRIDS)
REFERENCE = "logistic_regression"


def design(frame: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    X = with_canonical_missing(frame)
    if arm == "A":
        ind = missing_mask(frame).astype(float).add_prefix("missing_")
        return pd.concat([X, ind], axis=1)
    return X.drop(columns=["Cholesterol"])


def pipeline(model, columns: list[str]) -> Pipeline:
    numeric = [c for c in NUMERIC if c in columns]
    indicators = [c for c in columns if c.startswith("missing_")]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()), numeric),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ("ind", "passthrough", indicators),
    ])
    return Pipeline([("prep", prep), ("clf", model)])


@dataclass
class OuterFit:
    oof: np.ndarray                                   # P(disease), one per patient
    fold: np.ndarray                                  # outer fold id per patient
    train_threshold: dict[int, float] = field(default_factory=dict)   # Youden on training rows
    best_params: list[dict] = field(default_factory=list)
    fold_auc: list[float] = field(default_factory=list)


def youden_threshold(y, p) -> float:
    order = np.unique(p)
    best, thr = -1.0, 0.5
    for t in order:
        pred = p >= t
        j = pred[y == 1].mean() + (~pred[y == 0]).mean() - 1
        if j > best:
            best, thr = j, float(t)
    return thr


def nested_fit(frame: pd.DataFrame, arm: Arm, model: str, splits, seed: int, n_jobs: int = 1) -> OuterFit:
    """Outer loop over `splits` (train, test index pairs); inner leave-one-site-out grid search."""
    from sklearn.metrics import roc_auc_score

    X, y, groups = design(frame, arm), frame[TARGET].to_numpy(), frame["source"].to_numpy()
    make, grid = GRIDS[model]
    oof, fold = np.full(len(y), np.nan), np.full(len(y), -1)
    out = OuterFit(oof, fold)
    for k, (train, test) in enumerate(splits):
        search = GridSearchCV(pipeline(make(seed), list(X.columns)), grid, scoring="roc_auc",
                              cv=LeaveOneGroupOut(), n_jobs=n_jobs, error_score="raise")
        search.fit(X.iloc[train], y[train], groups=groups[train])
        p = search.predict_proba(X.iloc[test])[:, 1]
        oof[test], fold[test] = p, k
        out.best_params.append(search.best_params_)
        out.train_threshold[k] = youden_threshold(y[train], search.predict_proba(X.iloc[train])[:, 1])
        out.fold_auc.append(float(roc_auc_score(y[test], p)) if len(np.unique(y[test])) == 2 else np.nan)
    return out


def loso_splits(frame: pd.DataFrame):
    return list(LeaveOneGroupOut().split(frame, groups=frame["source"]))


def repeated_splits(frame: pd.DataFrame, seed: int, n_splits: int = 10, n_repeats: int = 5):
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
                .split(frame, frame[TARGET]))
