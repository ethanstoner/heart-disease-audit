"""Pre-registered notebook survey: screening order and modal pipeline. See survey/PROTOCOL.md."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

SEED = 20260922


def screening_order(items: list[dict], seed: int = SEED) -> list[tuple[str, list[dict]]]:
    """Non-fork results grouped by repository, repositories in seeded-shuffle order,
    each repository's notebooks in path order with exact duplicate results removed."""
    by_repo: dict[str, dict[str, dict]] = {}
    for item in items:
        if not item["fork"]:
            by_repo.setdefault(item["repo"], {})[item["path"]] = item
    repos = sorted(by_repo)
    perm = np.random.default_rng(seed).permutation(len(repos))
    return [(repos[k], [by_repo[repos[k]][p] for p in sorted(by_repo[repos[k]])]) for k in perm]


def modal_pipeline(coding: pd.DataFrame) -> tuple[dict, list[str]]:
    """Per-field mode over included notebooks (rows in screening order), and the fields that tied.

    Ties go to the tied value screened first. `models`: types in at least half the notebooks;
    if fewer than 2 qualify, the 4 most frequent including any tied for 4th.
    """
    modal, ties = {}, []
    for col in coding.columns:
        if col == "models":
            counts = Counter(m for cell in coding[col] for m in set(cell.split(";")))
            chosen = [m for m, c in counts.items() if c >= len(coding) / 2]
            if len(chosen) < 2:
                cutoff = sorted(counts.values(), reverse=True)[:4][-1]
                chosen = [m for m, c in counts.items() if c >= cutoff]
            modal[col] = sorted(chosen)
            continue
        values = coding[col].dropna()
        counts = values.value_counts()
        top = counts[counts == counts.max()].index
        if len(top) > 1:
            ties.append(col)
        modal[col] = next(v for v in values if v in top)
    return modal, ties


MODEL_TYPES = {
    "logistic_regression", "knn", "svm", "decision_tree", "random_forest", "gradient_boosting",
    "xgboost", "lightgbm", "catboost", "naive_bayes", "mlp", "other",
}
YES_NO = {"yes", "no"}
YES_NO_NA = {"yes", "no", "na"}

# field -> allowed values; "unit" = float in [0, 1]; "fraction" = float in (0, 1) or blank;
# "count" = positive int; "models" = ;-separated MODEL_TYPES; "free" = any string or blank
RUBRIC = {
    "metric_name": {"accuracy", "auc", "f1", "other"},
    "metric_value": "unit",
    "accuracy_value": "unit_or_blank",
    "value_source": {"output", "text"},
    "eval_design": {"single_split", "cv", "both"},
    "test_size": "fraction",
    "stratify": YES_NO_NA,
    "seed_fixed": YES_NO,
    "models": "models",
    "headline_model": MODEL_TYPES,
    "n_models": "count",
    "best_of_n": YES_NO,
    "tuning": {"none", "grid", "random", "other"},
    "hyperparams": {"default", "set"},
    "fit_before_split": YES_NO_NA,
    "scaling": {"standard", "minmax", "none", "other"},
    "encoding": {"onehot", "label", "mixed", "none"},
    "chol_zero": {"ignored", "imputed", "dropped", "other"},
    "chol_zero_before_split": YES_NO_NA,
    "outlier_removal": YES_NO,
    "outlier_before_split": YES_NO_NA,
    "resampling": {"none", "smote", "other"},
    "features_dropped": "free",
    "dedup": {"dropped", "checked", "none"},
    "source_imputation_mentioned": YES_NO,
}


def _blank(v) -> bool:
    return v is None or v == "" or (isinstance(v, float) and np.isnan(v))


def rubric_violations(coding: dict) -> list[str]:
    """Every field whose value is outside the protocol's allowed set, as 'field: value'."""
    out = []
    for field, allowed in RUBRIC.items():
        v = coding.get(field)
        if allowed == "free":
            ok = True
        elif allowed == "unit":
            ok = isinstance(v, (int, float)) and not _blank(v) and 0 <= v <= 1
        elif allowed == "unit_or_blank":
            ok = _blank(v) or (isinstance(v, (int, float)) and 0 <= v <= 1)
        elif allowed == "fraction":
            ok = _blank(v) or (isinstance(v, (int, float)) and 0 < v < 1)
        elif allowed == "count":
            ok = isinstance(v, (int, np.integer)) and v >= 1
        elif allowed == "models":
            ok = isinstance(v, str) and v != "" and set(v.split(";")) <= MODEL_TYPES
        else:
            ok = v in allowed
        if not ok:
            out.append(f"{field}: {v!r}")
    return out
