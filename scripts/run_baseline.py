"""The honest baseline (spec: notebook 05). Writes results/baseline.json and results/baseline/*.csv.

Primary: leave-one-source-out, n-weighted within-site AUC (deviation D6). Control: 10x5
repeated stratified CV, pooled out-of-fold AUC per repeat. Both nested, inner loop
leave-one-site-out, both arms, four models.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from heart_audit.baseline import MODEL_NAMES, REFERENCE, loso_splits, nested_fit, repeated_splits
from heart_audit.data import TARGET, load_uci920, to_kaggle_schema
from heart_audit.evaluation import (
    auc, bootstrap_auc_ci, calibration, delong_auc_ci, holm, nadeau_bengio, site_weighted_auc_ci,
)
from heart_audit.seeds import derive

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "baseline"
N_BOOT = 2000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    uci = to_kaggle_schema(load_uci920())
    y, sites = uci[TARGET].to_numpy(), uci["source"].to_numpy()
    model_seed, boot_seed, cv_seed = derive("baseline", 3)
    loso, rep = loso_splits(uci), repeated_splits(uci, cv_seed)
    n_train, n_test = len(rep[0][0]), len(rep[0][1])
    res = {"n_boot": N_BOOT, "seeds": {"model": model_seed, "bootstrap": boot_seed, "cv": cv_seed}, "arms": {}}
    oof_rows = {"source": sites, "y": y}
    fold_aucs = {}

    for arm in ("A", "B"):
        res["arms"][arm] = {}
        for model in MODEL_NAMES:
            fit = nested_fit(uci, arm, model, loso, model_seed, n_jobs=-1)
            sw, lo, hi = site_weighted_auc_ci(y, fit.oof, sites, N_BOOT, boot_seed)
            per_site = {}
            for k, (_, test) in enumerate(loso):
                s = sites[test][0]
                a, se, dlo, dhi = delong_auc_ci(y[test], fit.oof[test])
                icpt, slope = calibration(y[test], fit.oof[test])
                per_site[s] = {
                    "auc": a, "delong_ci95": [dlo, dhi], "n_pos": int(y[test].sum()), "n_neg": int((1 - y[test]).sum()),
                    "underpowered": bool((1 - y[test]).sum() < 20),
                    "calibration_intercept": icpt, "calibration_slope": slope,
                    "balanced_accuracy_0.5": float(balanced_accuracy_score(y[test], fit.oof[test] >= 0.5)),
                    "balanced_accuracy_train_youden": float(balanced_accuracy_score(
                        y[test], fit.oof[test] >= fit.train_threshold[k])),
                    "best_params": {p: str(v) for p, v in fit.best_params[k].items()},
                }
            oof_rows[f"{arm}_{model}_loso"] = fit.oof

            ctrl = nested_fit(uci, arm, model, rep, model_seed, n_jobs=-1)
            per_repeat = []
            for r in range(5):
                # each patient appears once per repeat; rebuild that repeat's pooled OOF vector
                p = np.empty(len(y))
                for k in range(r * 10, (r + 1) * 10):
                    p[rep[k][1]] = ctrl.fold_oof[k]
                per_repeat.append(auc(y, p))
                if r == 0:
                    first = p
            _, blo, bhi = bootstrap_auc_ci(y, first, N_BOOT, boot_seed)
            fold_aucs[(arm, model)] = np.array(ctrl.fold_auc)
            oof_rows[f"{arm}_{model}_cv_repeat1"] = first
            res["arms"][arm][model] = {
                "loso": {"site_weighted_auc": sw, "ci95": [lo, hi],
                         "pooled_auc_biased_under_null": auc(y, fit.oof), "per_site": per_site},
                "control_10x5": {"pooled_auc_mean_over_repeats": float(np.mean(per_repeat)),
                                 "repeat_sd": float(np.std(per_repeat, ddof=1)),
                                 "repeat1_bootstrap_ci95": [blo, bhi],
                                 "within_site_auc_repeat1": site_weighted_auc_ci(y, first, sites, 200, boot_seed)[0]},
            }
            print(f"{arm} {model}: LOSO within-site {sw:.3f} [{lo:.3f}, {hi:.3f}] | "
                  f"10x5 pooled {np.mean(per_repeat):.3f}  ({time.perf_counter() - start:.0f}s)", flush=True)

        # Significance vs the reference model on the 10x5 control, Holm across the three comparisons
        others = [m for m in MODEL_NAMES if m != REFERENCE]
        tests = [nadeau_bengio(fold_aucs[(arm, m)] - fold_aucs[(arm, REFERENCE)], n_train, n_test) for m in others]
        adjusted = holm([p for _, p in tests])
        res["arms"][arm]["significance_vs_lr"] = {
            m: {"mean_fold_auc_diff": float((fold_aucs[(arm, m)] - fold_aucs[(arm, REFERENCE)]).mean()),
                "t": t, "p": p, "p_holm": float(ph)}
            for m, (t, p), ph in zip(others, tests, adjusted)}

    xgb = res["arms"]["A"]["significance_vs_lr"]["xgboost"], res["arms"]["B"]["significance_vs_lr"]["xgboost"]
    res["P8.1"] = {"arm_A_p_holm": xgb[0]["p_holm"], "arm_B_p_holm": xgb[1]["p_holm"],
                   "verdict": "falsified" if any(x["p_holm"] < 0.05 and x["mean_fold_auc_diff"] > 0 for x in xgb)
                   else "supported"}
    res["runtime_seconds"] = round(time.perf_counter() - start)
    pd.DataFrame(oof_rows).to_csv(OUT / "oof.csv", index=False)
    (ROOT / "results" / "baseline.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res["P8.1"]), res["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
