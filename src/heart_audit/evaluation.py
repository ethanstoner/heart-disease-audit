"""Resampling, significance tests and calibration used by the teardown and the honest baseline."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score


def stratified_bootstrap(y, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) row indices, resampled with replacement within each class."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(y)), dtype=int)
    for cls in np.unique(y):
        rows = np.flatnonzero(y == cls)
        out[:, rows] = rows[rng.integers(0, len(rows), size=(n_boot, len(rows)))]
    return out


def auc(y, score) -> float:
    return float(roc_auc_score(y, score))


def _auc_rows(y_rows: np.ndarray, s_rows: np.ndarray) -> np.ndarray:
    """AUC of every row of a (B, n) resample, by the Mann-Whitney rank formula (ties averaged)."""
    ranks = stats.rankdata(s_rows, axis=1)
    pos = y_rows == 1
    n1 = pos.sum(axis=1)
    n0 = y_rows.shape[1] - n1
    return ((ranks * pos).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_auc_ci(y, score, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float, float]:
    """AUC with a stratified percentile bootstrap interval (patients are the resampling unit)."""
    y, score = np.asarray(y), np.asarray(score)
    idx = stratified_bootstrap(y, n_boot, seed)
    boots = _auc_rows(y[idx], score[idx])
    a = (1 - level) / 2 * 100
    return auc(y, score), *np.percentile(boots, [a, 100 - a])


def paired_auc_diff(y, score_a, score_b, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) on the same patients, with a paired stratified percentile bootstrap."""
    y, score_a, score_b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    idx = stratified_bootstrap(y, n_boot, seed)
    boots = _auc_rows(y[idx], score_a[idx]) - _auc_rows(y[idx], score_b[idx])
    a = (1 - level) / 2 * 100
    return auc(y, score_a) - auc(y, score_b), *np.percentile(boots, [a, 100 - a])


def site_weighted_auc(y, score, sites) -> float:
    """n-weighted mean of within-site AUCs (the LOSO discrimination estimate; deviation D6)."""
    y, score, sites = np.asarray(y), np.asarray(score), np.asarray(sites)
    total = 0.0
    for s in np.unique(sites):
        m = sites == s
        total += auc(y[m], score[m]) * m.sum()
    return total / len(y)


def site_weighted_auc_ci(y, score, sites, n_boot: int, seed: int, level: float = 0.95) -> tuple[float, float, float]:
    """site_weighted_auc with a percentile bootstrap stratified by (site, class)."""
    y, score, sites = np.asarray(y), np.asarray(score), np.asarray(sites)
    boots = np.zeros(n_boot)
    for k, s in enumerate(np.unique(sites)):
        rows = np.flatnonzero(sites == s)
        idx = rows[stratified_bootstrap(y[rows], n_boot, seed + k)]
        boots += _auc_rows(y[idx], score[idx]) * len(rows)
    a = (1 - level) / 2 * 100
    return site_weighted_auc(y, score, sites), *np.percentile(boots / len(y), [a, 100 - a])


def mcnemar_exact(y, pred_a, pred_b) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs."""
    ca, cb = np.asarray(pred_a) == np.asarray(y), np.asarray(pred_b) == np.asarray(y)
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5)))


def winners_curse(correct: np.ndarray, n_boot: int, seed: int) -> float:
    """Mean over resamples of (in-bag accuracy − out-of-bag accuracy) of the in-bag winner.

    `correct` is (models, patients) booleans from one shared test set. Ties for the winner
    contribute the mean over the tied models.
    """
    correct = np.asarray(correct, dtype=float)
    n = correct.shape[1]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))
    weights = np.zeros((n_boot, n))
    np.add.at(weights, (np.repeat(np.arange(n_boot), n), draws.ravel()), 1)
    in_bag = weights @ correct.T / n
    oob = (weights == 0).astype(float)
    keep = oob.sum(axis=1) > 0
    in_bag, oob = in_bag[keep], oob[keep]
    out_bag = oob @ correct.T / oob.sum(axis=1, keepdims=True)
    winners = in_bag == in_bag.max(axis=1, keepdims=True)
    gap = np.where(winners, in_bag - out_bag, 0).sum(axis=1) / winners.sum(axis=1)
    return float(gap.mean())


def delong_auc_ci(y, score, level: float = 0.95) -> tuple[float, float, float, float]:
    """AUC, its DeLong standard error, and a normal-approximation interval clipped to [0, 1]."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    pos, neg = score[y == 1], score[y == 0]
    # psi(x, z) = 1 if x > z, 0.5 if tied, 0 otherwise
    psi = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    a = psi.mean()
    v10, v01 = psi.mean(axis=1), psi.mean(axis=0)
    se = np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg))
    z = stats.norm.ppf(0.5 + level / 2)
    return float(a), float(se), float(max(0.0, a - z * se)), float(min(1.0, a + z * se))


def nadeau_bengio(diffs, n_train: int, n_test: int) -> tuple[float, float]:
    """Corrected resampled t-test on J per-resample score differences (Nadeau & Bengio 2003).

    Variance is inflated by (1/J + n_test/n_train); df = J − 1. Returns (t, two-sided p).
    """
    d = np.asarray(diffs, dtype=float)
    j = len(d)
    var = d.var(ddof=1) * (1 / j + n_test / n_train)
    if var == 0:
        return (np.inf if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0)
    t = d.mean() / np.sqrt(var)
    return float(t), float(2 * stats.t.sf(abs(t), j - 1))


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _nll(y, eta):
    return float(np.sum(np.logaddexp(0, eta) - y * eta))


def calibration(y, prob) -> tuple[float, float]:
    """(calibration-in-the-large intercept, calibration slope) by logistic recalibration."""
    y, lp = np.asarray(y, dtype=float), _logit(prob)
    intercept = optimize.minimize_scalar(lambda a: _nll(y, a + lp)).x
    slope = optimize.minimize(lambda w: _nll(y, w[0] + w[1] * lp), x0=[0.0, 1.0]).x[1]
    return float(intercept), float(slope)
