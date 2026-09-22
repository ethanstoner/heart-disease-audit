"""Behaviour of the two LOSO AUC estimators under 50 label shuffles (deviation D6).

Same shuffles, pipeline and seeds as tests/test_methodology.py::test_shuffled_label_gate.
Writes results/shuffled_label_gate.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from heart_audit.baseline import loso_splits, nested_fit
from heart_audit.data import TARGET, load_uci920, to_kaggle_schema
from heart_audit.evaluation import bootstrap_auc_ci, site_weighted_auc_ci

OUT = Path(__file__).resolve().parents[1] / "results" / "shuffled_label_gate.json"


def main() -> None:
    uci = to_kaggle_schema(load_uci920())
    rng = np.random.default_rng(50)
    pooled, within, pooled_contains, within_contains = [], [], 0, 0
    for i in range(50):
        shuffled = uci.assign(**{TARGET: rng.permutation(uci[TARGET].to_numpy())})
        fit = nested_fit(shuffled, "A", "logistic_regression", loso_splits(shuffled), seed=i)
        a, lo, hi = bootstrap_auc_ci(shuffled[TARGET], fit.oof, 1000, seed=i)
        pooled.append(a)
        pooled_contains += lo <= 0.5 <= hi
        w, wlo, whi = site_weighted_auc_ci(shuffled[TARGET], fit.oof, shuffled["source"], 1000, seed=i)
        within.append(w)
        within_contains += wlo <= 0.5 <= whi
    required = int(stats.binom.ppf(0.025, 50, 0.95))
    out = {
        "shuffles": 50, "required_intervals_containing_0.5": required,
        "pooled_oof_auc": {"mean": float(np.mean(pooled)), "min": float(np.min(pooled)), "max": float(np.max(pooled)),
                           "intervals_containing_0.5": int(pooled_contains), "passes": bool(pooled_contains >= required)},
        "within_site_auc": {"mean": float(np.mean(within)), "min": float(np.min(within)), "max": float(np.max(within)),
                            "intervals_containing_0.5": int(within_contains), "passes": bool(within_contains >= required)},
    }
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
