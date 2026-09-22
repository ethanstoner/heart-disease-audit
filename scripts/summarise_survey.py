"""Modal pipeline and headline-accuracy distribution, under the primary and sensitivity readings.

Writes survey/modal_pipeline.json. See survey/PROTOCOL.md "Modal pipeline" and DEVIATIONS.md.
"""
import json
from pathlib import Path

import pandas as pd

from heart_audit.survey import RUBRIC, modal_pipeline

SURVEY = Path(__file__).resolve().parents[1] / "survey"
VALUE_FIELDS = ["metric_value", "accuracy_value"]

# D2 sensitivity: the headline model's best shown held-out accuracy where it differs from the
# pre-registered "last-executed configuration" value (taken from the coders' notes).
D2_BEST_SHOWN = {
    "003-1": 0.9444, "006-1": 0.8674, "022-1": 0.8785, "028-1": 0.9130,
    "048-1": 0.8611, "054-1": 0.8453, "055-1": 0.86,
}


def load(name: str) -> pd.DataFrame:
    coding = pd.read_csv(SURVEY / name, keep_default_na=False, na_values=[""])
    # D4: a blank features_dropped is the coded value "nothing dropped", not a missing value.
    coding["features_dropped"] = coding["features_dropped"].fillna("none")
    return coding


def accuracy_summary(acc: pd.Series) -> dict:
    q = acc.dropna().quantile([0.25, 0.5, 0.75])
    return {"n": int(acc.notna().sum()), "median": round(q[0.5], 4),
            "iqr": [round(q[0.25], 4), round(q[0.75], 4)],
            "min": round(acc.min(), 4), "max": round(acc.max(), 4)}


def summarise(coding: pd.DataFrame) -> dict:
    fields = [f for f in RUBRIC if f not in VALUE_FIELDS]
    modal, ties = modal_pipeline(coding[fields])
    shares = {f: f"{(coding[f] == modal[f]).sum()}/{coding[f].notna().sum()}" for f in fields if f != "models"}
    model_counts = pd.Series([m for s in coding["models"] for m in set(s.split(";"))]).value_counts()
    return {
        "modal": {k: (v.item() if hasattr(v, "item") else v) for k, v in modal.items()},
        "shares": shares,
        "ties": ties,
        "model_counts": model_counts.to_dict(),
        "accuracy": accuracy_summary(coding["accuracy_value"]),
    }


def main() -> None:
    primary = load("coding.csv")
    d2 = primary["accuracy_value"].copy()
    d2[primary["file"].isin(D2_BEST_SHOWN)] = primary["file"].map(D2_BEST_SHOWN)
    out = {
        "primary": summarise(primary),
        "sensitivity_literal_e4": summarise(load("coding_literal.csv")),
        "sensitivity_d2_best_shown_accuracy": accuracy_summary(d2),
    }
    (SURVEY / "modal_pipeline.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
