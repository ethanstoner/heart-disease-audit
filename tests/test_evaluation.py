import numpy as np
import pytest

from heart_audit.evaluation import (
    auc, bootstrap_auc_ci, calibration, delong_auc_ci, holm, mcnemar_exact, nadeau_bengio,
    paired_auc_diff, stratified_bootstrap, winners_curse,
)


def test_stratified_bootstrap_keeps_class_counts():
    y = np.array([0] * 7 + [1] * 3)
    idx = stratified_bootstrap(y, 50, seed=1)
    assert idx.shape == (50, 10)
    assert (y[idx].sum(axis=1) == 3).all()


def test_mcnemar_hand_cases():
    y = np.ones(10, dtype=int)
    a = np.array([1] * 5 + [0] * 5)          # a right on 5, b right on none -> b=5, c=0
    b = np.zeros(10, dtype=int)
    assert mcnemar_exact(y, a, b) == pytest.approx(2 * 0.5**5)
    assert mcnemar_exact(y, a, a) == 1.0
    c = np.array([0] * 3 + [1] * 3 + [0] * 4)  # discordant 3 vs 3
    d = np.array([1] * 3 + [0] * 3 + [0] * 4)
    assert mcnemar_exact(y, c, d) == 1.0


def test_paired_auc_diff_of_identical_scores_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.random(200)
    assert paired_auc_diff(y, s, s, 100, seed=0) == (0.0, 0.0, 0.0)


def test_bootstrap_auc_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    a, lo, hi = bootstrap_auc_ci(y, s, 500, seed=2)
    assert a == pytest.approx(auc(y, s)) and lo < a < hi


def test_winners_curse_is_zero_for_one_model_and_positive_for_many():
    rng = np.random.default_rng(3)
    one = rng.random((1, 184)) < 0.85
    assert abs(winners_curse(one, 4000, seed=4)) < 0.003
    many = rng.random((20, 184)) < 0.85            # 20 equally good, independent models
    assert winners_curse(many, 2000, seed=5) > 0.03


def test_delong_matches_bruteforce_structural_components():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 120)
    s = np.round(y * 0.8 + rng.normal(0, 1, 120), 1)    # rounding creates ties
    a, se, lo, hi = delong_auc_ci(y, s)
    pos, neg = s[y == 1], s[y == 0]
    v10 = np.array([np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for q in neg]) for p in pos])
    v01 = np.array([np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for p in pos]) for q in neg])
    assert a == pytest.approx(auc(y, s))
    assert se == pytest.approx(np.sqrt(v10.var(ddof=1) / len(pos) + v01.var(ddof=1) / len(neg)))
    assert lo < a < hi


def test_nadeau_bengio_hand_case():
    d = np.array([0.01, 0.03, 0.02, 0.00, 0.04])
    t, p = nadeau_bengio(d, n_train=90, n_test=10)
    expected_t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / 5 + 10 / 90))
    assert t == pytest.approx(expected_t)
    assert 0 < p < 1


def test_holm_hand_case():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_calibration_recovers_known_miscalibration():
    rng = np.random.default_rng(7)
    true_logit = rng.normal(0, 1.5, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-true_logit))
    ok = calibration(y, 1 / (1 + np.exp(-true_logit)))
    assert ok[0] == pytest.approx(0, abs=0.05) and ok[1] == pytest.approx(1, abs=0.05)
    overconfident = calibration(y, 1 / (1 + np.exp(-2 * true_logit)))
    assert overconfident[1] == pytest.approx(0.5, abs=0.03)
