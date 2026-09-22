import numpy as np

from heart_audit.plots import control_distribution, survey_practices


def test_figures_render(tmp_path):
    acc = np.random.default_rng(0).integers(150, 170, 200) / 184
    p = control_distribution(acc, acc[:30], 0.86, (0.82, 0.91), 184, tmp_path / "a.png")
    q = survey_practices(["a", "b"], [3, 5], 10, tmp_path / "b.png")
    assert p.stat().st_size > 10_000 and q.stat().st_size > 5_000


def test_teardown_figures_render(tmp_path):
    from heart_audit.plots import effect_summary, fill_rule, paired_difference, site_dumbbell
    panel = {"title": "x", "values": ["a", "b"], "filled": [1.0, 0.0], "observed": [0.7, 0.3],
             "n_filled": [5, 6], "n_observed": [7, 8]}
    assert fill_rule([panel, panel], tmp_path / "f.png").exists()
    assert effect_summary([{"label": "a", "median": 1, "lo": 0, "hi": 2}], tmp_path / "e.png").exists()
    assert paired_difference(np.array([0.01, -0.01, 0.0, 0.02]), "d", "t", tmp_path / "p.png").exists()
    assert site_dumbbell(["s1", "s2"], [0.8, 0.7], [0.9, 0.75], "L", "R", "t", tmp_path / "d.png").exists()


def test_honest_estimates_renders(tmp_path):
    from heart_audit.plots import honest_estimates
    rows = [{"label": "a", "loso": 0.8, "lo": 0.75, "hi": 0.85, "cv": 0.88}]
    assert honest_estimates(rows, tmp_path / "h.png").exists()
