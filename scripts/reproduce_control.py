"""Run the control over 1000 derived split seeds and test the reproduction gate.

Writes results/control_seeds.csv (one row per seed, one accuracy column per model).
"""
import json
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from heart_audit.conventional import run_conventional, reproduction_gate
from heart_audit.data import load_kaggle918
from heart_audit.seeds import derive

ROOT = Path(__file__).resolve().parents[1]
N_SEEDS = 1000


def one(df, seed):
    return {"seed": seed, **run_conventional(df, seed).accuracy}


def main() -> None:
    df = load_kaggle918()
    start = time.perf_counter()
    rows = Parallel(n_jobs=-1)(delayed(one)(df, s) for s in derive("control", N_SEEDS))
    elapsed = time.perf_counter() - start
    out = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "control_seeds.csv", index=False)

    survey = json.loads((ROOT / "survey" / "modal_pipeline.json").read_text(encoding="utf-8"))
    targets = {
        "primary": survey["primary"]["accuracy"]["median"],
        "literal_e4": survey["sensitivity_literal_e4"]["accuracy"]["median"],
        "d2_best_shown": survey["sensitivity_d2_best_shown_accuracy"]["median"],
    }
    print(f"{N_SEEDS} seeds in {elapsed:.0f}s")
    print(out.drop(columns="seed").describe().round(4).to_string())
    for name, target in targets.items():
        lo, hi, passed = reproduction_gate(out["random_forest"], target)
        print(f"gate[{name}] target={target} central95=[{lo:.4f}, {hi:.4f}] passed={passed}")


if __name__ == "__main__":
    main()
