import numpy as np

from heart_audit.plots import control_distribution, survey_practices


def test_figures_render(tmp_path):
    acc = np.random.default_rng(0).integers(150, 170, 200) / 184
    p = control_distribution(acc, acc[:30], 0.86, (0.82, 0.91), 184, tmp_path / "a.png")
    q = survey_practices(["a", "b"], [3, 5], 10, tmp_path / "b.png")
    assert p.stat().st_size > 10_000 and q.stat().st_size > 5_000
