"""Run the teardown experiments T1–T7 exactly as pre-registered in analysis/PREDICTIONS.md
(with deviation D5 from analysis/DEVIATIONS.md).

Writes results/teardown.json (every statistic and prediction verdict), per-seed tables in
results/teardown/, and results/control_predictions.npz.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from xgboost import XGBClassifier

from heart_audit.conventional import accuracy_table, encode, impute, run_seeds, split_indices, MODELS
from heart_audit.data import FEATURES, TARGET, load_kaggle918, load_statlog270, load_uci920, to_kaggle_schema
from heart_audit.evaluation import auc, mcnemar_exact, paired_auc_diff, winners_curse
from heart_audit.leakage import (
    T7_MODELS, contamination, cv_oof_proba, duplicate_groups, fill_gap, fill_table, matched_removal,
    missingness_design, oof_scores, reverted_frame, source_design,
)
from heart_audit.provenance import filled_cells, match_sources
from heart_audit.seeds import derive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "teardown"
N_SEEDS, N_BOOT, N_NULL = 1000, 2000, 50
RF = ("random_forest",)


def excess_over_majority(results) -> np.ndarray:
    return np.array([r.accuracy["random_forest"] - max(r.y_test.mean(), 1 - r.y_test.mean()) for r in results])


def rf_median(frame, seeds, **kw) -> float:
    return float(np.median([r.accuracy["random_forest"] for r in run_seeds(frame, seeds, models=RF, **kw)]))


def xgb_run(frame: pd.DataFrame, seed: int, mode: str) -> float:
    """XGBoost inside the control's split and encoding. mode: 'native' (zeros -> NaN), 'full', 'train'."""
    df = frame[FEATURES + [TARGET]].reset_index(drop=True)
    train, test = split_indices(len(df), seed)
    if mode == "native":
        for col in ("Cholesterol", "RestingBP"):
            df[col] = df[col].mask(df[col].eq(0))
    else:
        df = impute(df, np.arange(len(df)) if mode == "full" else train)
    X, y = encode(df)
    X = X.to_numpy()
    model = XGBClassifier(random_state=seed, n_jobs=1).fit(X[train], y[train])
    return float((model.predict(X[test]) == y[test]).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    kaggle, uci = load_kaggle918(), to_kaggle_schema(load_uci920())
    matches = match_sources(kaggle, uci)
    cells = filled_cells(kaggle, uci, matches)
    seeds = derive("control", N_SEEDS)
    res: dict = {"n_seeds": N_SEEDS, "n_boot": N_BOOT}

    # ---- control with predictions (T4, T5, T6 baseline) --------------------------------
    control = run_seeds(kaggle, seeds)
    committed = pd.read_csv(ROOT / "results" / "control_seeds.csv")
    assert np.allclose(accuracy_table(control)[list(MODELS)].to_numpy(), committed[list(MODELS)].to_numpy())
    np.savez_compressed(
        ROOT / "results" / "control_predictions.npz",
        seeds=np.array(seeds), test_index=np.stack([r.test_index for r in control]),
        y_test=np.stack([r.y_test for r in control]).astype(np.int8),
        **{m: np.stack([r.predictions[m] for r in control]).astype(np.int8) for m in MODELS},
    )
    control_rf = np.array([r.accuracy["random_forest"] for r in control])
    print(f"control done {time.perf_counter() - start:.0f}s")

    # ---- T1 ------------------------------------------------------------------------------
    fill_table(kaggle, matches, cells).to_csv(OUT / "t1_fill_table.csv", index=False)
    gap_seed = derive("t1_gap", 1)[0]
    gaps = [fill_gap(kaggle, matches, cells, "ST_Slope", "Flat", "Up", N_BOOT, gap_seed),
            fill_gap(kaggle, matches, cells, "FastingBS", 1, 0, N_BOOT, gap_seed)]
    for g in gaps:
        g["verdict"] = "supported" if g["difference"] > 0 else "falsified"
        g["strong"] = bool(g["ci95"][0] > 0)
    res["P1.1"] = gaps

    slope_rows = cells.loc[cells["column"] == "ST_Slope", "kaggle_row"].to_numpy()
    arm_f = run_seeds(kaggle.drop(index=kaggle.index[slope_rows]), seeds, models=RF)
    f_acc = np.array([r.accuracy["random_forest"] for r in arm_f])
    pool = np.setdiff1d(np.flatnonzero(matches["uci_row"].notna().to_numpy()), slope_rows)
    null_medians, null_excess = [], []
    for i, draw_seed in enumerate(derive("t1_removal", N_NULL)):
        drop = matched_removal(kaggle, slope_rows, pool, np.random.default_rng(draw_seed))
        arm = run_seeds(kaggle.drop(index=kaggle.index[drop]), seeds, models=RF)
        null_medians.append(float(np.median([r.accuracy["random_forest"] for r in arm])))
        null_excess.append(float(np.median(excess_over_majority(arm))))
    null_lo = float(np.percentile(null_medians, 2.5))
    res["P1.2"] = {
        "rows_removed": int(len(slope_rows)),
        "median_full": float(np.median(control_rf)),
        "median_filled_removed": float(np.median(f_acc)),
        "null_medians_central95": [null_lo, float(np.percentile(null_medians, 97.5))],
        "excess_over_majority_full": float(np.median(excess_over_majority(control))),
        "excess_over_majority_filled_removed": float(np.median(excess_over_majority(arm_f))),
        "excess_over_majority_null_median": float(np.median(null_excess)),
    }
    res["P1.2"]["verdict"] = ("supported" if res["P1.2"]["median_filled_removed"] < res["P1.2"]["median_full"]
                              and res["P1.2"]["median_filled_removed"] < null_lo else "falsified")
    pd.DataFrame({"null_median": null_medians, "null_excess": null_excess}).to_csv(OUT / "t1_null.csv", index=False)
    print(f"T1.2 done {time.perf_counter() - start:.0f}s")

    arm_p = run_seeds(kaggle, seeds, impute_scope="train", models=RF)
    arm_r = run_seeds(reverted_frame(kaggle, cells), seeds, impute_scope="train", models=RF)
    d13 = np.array([r.accuracy["random_forest"] - p.accuracy["random_forest"] for r, p in zip(arm_r, arm_p)])
    res["P1.3"] = {"median_paired_diff": float(np.median(d13)),
                   "central95": np.percentile(d13, [2.5, 97.5]).tolist(),
                   "mean_paired_diff": float(d13.mean()),
                   "cells_reverted": int(len(cells))}
    res["P1.3"]["verdict"] = "supported" if res["P1.3"]["median_paired_diff"] < 0 else "falsified"
    pd.DataFrame({"seed": seeds, "published": [p.accuracy["random_forest"] for p in arm_p],
                  "reverted": [r.accuracy["random_forest"] for r in arm_r],
                  "filled_removed": f_acc, "control": control_rf}).to_csv(OUT / "t1_seeds.csv", index=False)

    # ---- T2 ------------------------------------------------------------------------------
    y_u = uci[TARGET].to_numpy()
    prevalence = uci.groupby("source")[TARGET].transform("mean").to_numpy()
    t2_seed = derive("t2_cv", 1)[0]
    src_cv = auc(y_u, cv_oof_proba(source_design(uci), y_u, t2_seed))
    miss_cv = auc(y_u, cv_oof_proba(missingness_design(uci), y_u, t2_seed))
    chol0 = kaggle["Cholesterol"].eq(0).astype(float)
    res["P2.1"] = {"in_sample_auc": auc(y_u, prevalence), "cv_auc": src_cv,
                   "site_majority_accuracy": float(uci.groupby("source")[TARGET].agg(
                       lambda s: max(s.sum(), len(s) - s.sum())).sum() / len(uci))}
    res["P2.1"]["verdict"] = "supported" if abs(src_cv - res["P2.1"]["in_sample_auc"]) <= 0.02 else "falsified"
    res["P2.2"] = {"missingness_only_cv_auc": miss_cv, "kaggle_chol_zero_auc": auc(kaggle[TARGET], chol0),
                   "verdict": "supported" if miss_cv >= 0.65 else "falsified"}

    # ---- T3 ------------------------------------------------------------------------------
    merged = pd.concat([uci, load_statlog270()], ignore_index=True)
    groups = duplicate_groups(merged)
    rand = run_seeds(merged, seeds, impute_scope="train", models=RF)
    grp = run_seeds(merged, seeds, impute_scope="train", groups=groups, models=RF)
    cont = np.array([contamination(groups, r.test_index) for r in rand])
    res["P3.1"] = {"rows": len(merged), "duplicates": int(len(merged) - len(np.unique(groups))),
                   "median_random": float(np.median([r.accuracy["random_forest"] for r in rand])),
                   "median_grouped": float(np.median([r.accuracy["random_forest"] for r in grp])),
                   "contamination_median": float(np.median(cont))}
    res["P3.1"]["verdict"] = "supported" if res["P3.1"]["median_random"] > res["P3.1"]["median_grouped"] else "falsified"
    pd.DataFrame({"seed": seeds, "random": [r.accuracy["random_forest"] for r in rand],
                  "grouped": [r.accuracy["random_forest"] for r in grp], "contamination": cont}).to_csv(
        OUT / "t3_seeds.csv", index=False)
    print(f"T3 done {time.perf_counter() - start:.0f}s")

    # ---- T4 ------------------------------------------------------------------------------
    acc = accuracy_table(control)
    names = list(MODELS)
    best = acc[names].to_numpy()
    winners = best == best.max(axis=1, keepdims=True)
    share = (winners / winners.sum(axis=1, keepdims=True)).mean(axis=0)
    d_rf_lr = acc["random_forest"] - acc["logistic_regression"]
    mcn = np.array([mcnemar_exact(r.y_test, r.predictions["random_forest"], r.predictions["logistic_regression"])
                    for r in control])
    res["P4.1"] = {"win_share": dict(zip(names, share.round(4).tolist())),
                   "verdict": "supported" if share[names.index("random_forest")] < 0.5 else "falsified"}
    res["P4.2"] = {"sd_rf_minus_lr": float(d_rf_lr.std(ddof=1)), "sd_rf": float(acc["random_forest"].std(ddof=1)),
                   "mean_rf_minus_lr": float(d_rf_lr.mean()),
                   "share_rf_ahead": float((d_rf_lr > 0).mean()), "share_lr_ahead": float((d_rf_lr < 0).mean())}
    res["P4.2"]["verdict"] = "supported" if res["P4.2"]["sd_rf_minus_lr"] < res["P4.2"]["sd_rf"] else "falsified"
    res["P4.3"] = {"share_significant": float((mcn < 0.05).mean()),
                   "verdict": "supported" if (mcn < 0.05).mean() < 0.2 else "falsified"}
    pd.DataFrame({"seed": seeds, "mcnemar_rf_lr_p": mcn}).to_csv(OUT / "t4_mcnemar.csv", index=False)

    # ---- T5 ------------------------------------------------------------------------------
    t5_seeds = derive("t5", N_SEEDS)
    bias = np.array(Parallel(n_jobs=-1)(
        delayed(winners_curse)(np.stack([r.predictions[m] == r.y_test for m in names]), N_BOOT, s)
        for r, s in zip(control, t5_seeds)))
    res["P5.1"] = {"mean_bias_pp": float(bias.mean() * 100), "central95_pp": (np.percentile(bias, [2.5, 97.5]) * 100).tolist(),
                   "naive_best_of_4_median": float(np.median(best.max(axis=1)))}
    res["P5.1"]["verdict"] = "supported" if 0.5 <= res["P5.1"]["mean_bias_pp"] <= 3.0 else "falsified"
    pd.DataFrame({"seed": seeds, "bias": bias}).to_csv(OUT / "t5_bias.csv", index=False)
    print(f"T5 done {time.perf_counter() - start:.0f}s")

    # ---- T6 ------------------------------------------------------------------------------
    scale_train = run_seeds(kaggle, seeds, scale_scope="train")
    ident = {m: float(np.mean([(a.predictions[m] == b.predictions[m]).mean() for a, b in zip(control, scale_train)]))
             for m in names}
    scale_diff = {m: np.array([a.accuracy[m] - b.accuracy[m] for a, b in zip(control, scale_train)]) for m in names}
    imp_full = run_seeds(kaggle, seeds, impute_scope="full", models=RF)
    imp_train = run_seeds(kaggle, seeds, impute_scope="train", models=RF)
    d_imp = np.array([a.accuracy["random_forest"] - b.accuracy["random_forest"] for a, b in zip(imp_full, imp_train)])
    xgb = {mode: np.array(Parallel(n_jobs=-1)(delayed(xgb_run)(kaggle, s, mode) for s in seeds))
           for mode in ("native", "full", "train")}
    res["T6_sanity"] = {"prediction_identity": ident, "passed": all(ident[m] >= 0.999 for m in ("random_forest", "decision_tree"))}
    res["P6.1"] = {m: {"median_paired_diff": float(np.median(scale_diff[m])), "mean_abs_diff": float(np.abs(scale_diff[m]).mean())}
                   for m in names}
    res["P6.1"]["verdict"] = ("supported" if all(abs(res["P6.1"][m]["median_paired_diff"]) < 0.005
                                                 for m in ("logistic_regression", "svm")) else "falsified")
    res["P6.2"] = {"median_paired_diff": float(np.median(d_imp)), "central95": np.percentile(d_imp, [2.5, 97.5]).tolist(),
                   "verdict": "supported" if abs(float(np.median(d_imp))) < 0.005 else "falsified"}
    res["T6_xgboost"] = {mode: float(np.median(v)) for mode, v in xgb.items()} | {
        "median_full_minus_train": float(np.median(xgb["full"] - xgb["train"])),
        "median_native_minus_train": float(np.median(xgb["native"] - xgb["train"]))}
    pd.DataFrame({"seed": seeds, **{f"scale_{m}": scale_diff[m] for m in names}, "impute_rf": d_imp,
                  **{f"xgb_{k}": v for k, v in xgb.items()}}).to_csv(OUT / "t6_seeds.csv", index=False)
    print(f"T6 done {time.perf_counter() - start:.0f}s")

    # ---- T7 ------------------------------------------------------------------------------
    t7_cv, t7_boot = derive("t7_cv", 1)[0], derive("t7_boot", 1)[0]
    res["P7.1"], oof = {}, {"source": uci["source"], "y": y_u}
    for name, make in T7_MODELS.items():
        loso, kfold = oof_scores(uci, "loso", t7_cv, make), oof_scores(uci, "kfold", t7_cv, make)
        diff, lo, hi = paired_auc_diff(y_u, loso, kfold, N_BOOT, t7_boot)
        per_site = {s: {"loso": auc(y_u[uci.source == s], loso[uci.source == s]),
                        "kfold": auc(y_u[uci.source == s], kfold[uci.source == s]),
                        "n_neg": int((y_u[uci.source == s] == 0).sum()),
                        "n_pos": int((y_u[uci.source == s] == 1).sum())}
                    for s in uci["source"].unique()}
        res["P7.1"][name] = {"auc_loso": auc(y_u, loso), "auc_kfold": auc(y_u, kfold),
                             "diff": diff, "ci95": [lo, hi], "per_site": per_site}
        oof[f"{name}_loso"], oof[f"{name}_kfold"] = loso, kfold
    res["P7.1"]["verdict"] = "supported" if res["P7.1"]["logistic_regression"]["ci95"][1] < 0 else "falsified"
    pd.DataFrame(oof).to_csv(OUT / "t7_oof.csv", index=False)

    res["runtime_seconds"] = round(time.perf_counter() - start)
    (ROOT / "results" / "teardown.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v.get("verdict") if isinstance(v, dict) else [g["verdict"] for g in v]
                      for k, v in res.items() if k.startswith("P")}, indent=1))


if __name__ == "__main__":
    main()
