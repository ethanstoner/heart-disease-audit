"""Tests of the statistics themselves: do the intervals cover, and is the significance test calibrated?

These are the tests that protect the project's claims. Each uses a data-generating process
with a known answer. Tolerances are three binomial standard errors.
"""
import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from heart_audit.baseline import loso_splits, nested_fit
from heart_audit.data import TARGET, load_uci920, to_kaggle_schema
from heart_audit.evaluation import bootstrap_auc_ci, delong_auc_ci, nadeau_bengio, site_weighted_auc_ci

N_SIM = 200
TOL = 3 * np.sqrt(0.95 * 0.05 / N_SIM)


def _binormal(rng, n_pos, n_neg, d):
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    return y, np.r_[rng.normal(d, 1, n_pos), rng.normal(0, 1, n_neg)]


@pytest.mark.parametrize("method", ["bootstrap", "delong"])
def test_auc_interval_coverage(method):
    """95% intervals must cover the true AUC in about 95% of datasets (binormal, AUC = Phi(d/sqrt2))."""
    d = 1.0
    true_auc = stats.norm.cdf(d / np.sqrt(2))
    rng = np.random.default_rng(20260922)
    covered = 0
    for i in range(N_SIM):
        y, s = _binormal(rng, 100, 100, d)
        if method == "bootstrap":
            _, lo, hi = bootstrap_auc_ci(y, s, 1000, seed=i)
        else:
            _, _, lo, hi = delong_auc_ci(y, s)
        covered += lo <= true_auc <= hi
    assert abs(covered / N_SIM - 0.95) <= TOL, covered / N_SIM


def _rejection_rates():
    """Two logistic regressions with equal true skill (each sees one of two equally informative
    features), compared by 10x5 repeated CV. Returns (corrected, uncorrected) rejection rates."""
    rng = np.random.default_rng(1)
    corrected = naive = 0
    for i in range(N_SIM):
        X = rng.normal(size=(300, 2))
        y = (rng.random(300) < 1 / (1 + np.exp(-(X[:, 0] + X[:, 1])))).astype(int)
        diffs = []
        for tr, te in RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=i).split(X, y):
            pa = LogisticRegression().fit(X[tr, :1], y[tr]).predict_proba(X[te, :1])[:, 1]
            pb = LogisticRegression().fit(X[tr, 1:], y[tr]).predict_proba(X[te, 1:])[:, 1]
            diffs.append(roc_auc_score(y[te], pa) - roc_auc_score(y[te], pb))
        corrected += nadeau_bengio(diffs, n_train=270, n_test=30)[1] < 0.05
        naive += stats.ttest_1samp(diffs, 0).pvalue < 0.05
    return corrected / N_SIM, naive / N_SIM


def test_corrected_t_test_is_not_anti_conservative():
    corrected, naive = _rejection_rates()
    assert corrected <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / N_SIM), corrected
    assert naive > corrected           # the uncorrected test over-rejects; the correction matters


def test_shuffled_label_gate(raw_dir):
    """Over 50 label shuffles, the honest pipeline's LOSO AUC interval contains 0.5 in >= 44/50 runs.
    44 is the 2.5th percentile of Binomial(50, 0.95). The estimate is the n-weighted within-site
    AUC (deviation D6): pooled out-of-fold AUC failed this gate (42/50, mean 0.489)."""
    uci = to_kaggle_schema(load_uci920(raw_dir))
    rng = np.random.default_rng(50)
    contains = 0
    for i in range(50):
        shuffled = uci.assign(**{TARGET: rng.permutation(uci[TARGET].to_numpy())})
        fit = nested_fit(shuffled, "A", "logistic_regression", loso_splits(shuffled), seed=i)
        _, lo, hi = site_weighted_auc_ci(shuffled[TARGET], fit.oof, shuffled["source"], 1000, seed=i)
        contains += lo <= 0.5 <= hi
    assert contains >= stats.binom.ppf(0.025, 50, 0.95), contains
